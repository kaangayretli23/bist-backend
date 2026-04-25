"""
BIST Pro - BES Fund Analysis Module
"""
import time, threading, traceback, json, os, re
from datetime import datetime, timedelta
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pass
from config import sf, si, safe_dict, app, BASE_DIR
from flask import jsonify, request

# =====================================================================
# BES (Bireysel Emeklilik Sistemi) FON ANALIZ MODULU
# TEFAS API uzerinden fon verisi cekilir, analiz & oneri yapilir
# =====================================================================
_bes_cache = {}
_bes_cache_lock = threading.Lock()
BES_CACHE_TTL = 1800  # 30 dakika
_tefas_semaphore = threading.Semaphore(1)  # TEFAS API rate limiter - tek seferde 1 istek

# BES background analiz thread state
_bes_bg_loading = False
_bes_bg_error = ''

TEFAS_API_URL = "https://www.tefas.gov.tr/api/DB/BindHistoryInfo"
TEFAS_ALLOC_URL = "https://www.tefas.gov.tr/api/DB/BindHistoryAllocation"
TEFAS_COMPARE_URL = "https://www.tefas.gov.tr/api/DB/BindComparisonFundReturns"
TEFAS_HEADERS = {
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://www.tefas.gov.tr/TarihselVeriler.aspx',
    'Origin': 'https://www.tefas.gov.tr',
}

# BES Fon Gruplari
BES_FUND_GROUPS = {
    'hisse': 'Hisse Senedi',
    'borclanma': 'Borçlanma Araçları',
    'katilim': 'Katılım',
    'karma': 'Karma / Dengeli',
    'doviz': 'Döviz',
    'altin': 'Altın / Kıymetli Maden',
    'endeks': 'Endeks',
    'para_piyasasi': 'Para Piyasası / Likit',
    'standart': 'Standart',
    'diger': 'Diğer',
}

def _bes_cache_get(key):
    with _bes_cache_lock:
        item = _bes_cache.get(key)
        if item and time.time() - item['ts'] < BES_CACHE_TTL:
            return item['data']
    return None

def _bes_cache_set(key, data):
    with _bes_cache_lock:
        _bes_cache[key] = {'data': data, 'ts': time.time()}

def _bes_bg_analyze_top():
    from bes_analysis import _analyze_fund_performance, _bes_optimize, _simulate_bes
    """Arka planda BES fon analizi yap ve cache'e kaydet (Render 30s timeout bypass)
    Strateji:
    1. TEFAS Compare API ile resmi getirileri cek (en guvenilir)
    2. Basarisizsa broad fetch + manual hesaplama
    3. Son care: fallback veri"""
    global _bes_bg_loading, _bes_bg_error
    _bes_bg_loading = True
    _bes_bg_error = ''
    try:
        today = datetime.now()
        pool = []

        # ===== YONTEM 1: TEFAS Compare API (resmi getiriler) =====
        print("[BES-BG] ADIM 1: TEFAS Compare API deneniyor...")
        compare_data = _fetch_tefas_compare()

        if compare_data and isinstance(compare_data, list) and len(compare_data) > 0:
            print(f"[BES-BG] Compare API'den {len(compare_data)} fon alindi")

            # Compare verisini parse et
            parsed_funds = []
            for row in compare_data:
                parsed = _parse_compare_row(row)
                if parsed and parsed['code'] and (parsed['price'] > 0 or parsed['total_value'] > 0):
                    parsed_funds.append(parsed)

            if parsed_funds:
                # Buyukluge gore sirala
                parsed_funds.sort(key=lambda x: x['total_value'], reverse=True)
                print(f"[BES-BG] {len(parsed_funds)} fon parse edildi (Compare API)")

                for f in parsed_funds[:30]:
                    rets = f['returns']
                    # Volatilite tahmini: gunluk getiriden
                    vol_est = abs(f['daily_return']) * (252 ** 0.5) * 100 if f['daily_return'] else 0
                    if vol_est < 1 and rets.get('3a'):
                        vol_est = abs(rets['3a']) / 3 * 2  # Kaba tahmin

                    # Sharpe tahmini
                    ret_6m = rets.get('6a') or rets.get('3a') or rets.get('1a') or 0
                    sharpe_est = sf((ret_6m / max(vol_est, 1)) * 0.5) if vol_est > 0 else 0

                    pool.append({
                        'code': f['code'],
                        'name': f['name'],
                        'category': _classify_fund(f['name']),
                        'currentPrice': f['price'],
                        'firstPrice': f['price'],
                        'totalReturn': rets.get('1y') or rets.get('6a') or 0,
                        'totalDays': 252,
                        'returns': {
                            '1h': rets.get('1h'),
                            '1a': rets.get('1a'),
                            '3a': rets.get('3a'),
                            '6a': rets.get('6a'),
                            '1y': rets.get('1y'),
                        },
                        'volatility': sf(vol_est),
                        'maxDrawdown': 0,
                        'sharpe': sharpe_est,
                        'dailyReturns': [],
                        'priceHistory': [],
                    })

                if pool:
                    _bes_cache_set('bes_analysis_pool', pool)
                    print(f"[BES-BG] Compare API basarili: {len(pool)} fon cache'e yazildi")
                    # Ornek getiriler logla
                    for p in pool[:3]:
                        print(f"[BES-BG]   {p['code']}: 1a={p['returns'].get('1a')}% 3a={p['returns'].get('3a')}% 6a={p['returns'].get('6a')}% 1y={p['returns'].get('1y')}%")
                    return

        print("[BES-BG] Compare API basarisiz veya bos, ADIM 2'ye geciliyor...")

        # ===== YONTEM 2: Broad fetch + manual hesaplama =====
        print("[BES-BG] ADIM 2: TEFAS broad fetch deneniyor...")
        raw = None
        for days_back in [90, 60, 30, 14, 7]:
            start = (today - timedelta(days=days_back)).strftime('%d.%m.%Y')
            end = today.strftime('%d.%m.%Y')
            raw = _fetch_tefas_funds(start, end)
            if raw and isinstance(raw, list) and len(raw) > 0:
                print(f"[BES-BG] TEFAS broad fetch basarili: {len(raw)} satir ({days_back} gun)")
                break

        if raw:
            # Fon koduna gore grupla
            fund_map = {}
            parse_ok = 0
            parse_fail = 0
            for row in (raw if isinstance(raw, list) else []):
                parsed = _parse_fund_row(row)
                if parsed and parsed['code']:
                    code = parsed['code']
                    if code not in fund_map:
                        fund_map[code] = {'rows': [], 'meta': parsed}
                    fund_map[code]['rows'].append(row)
                    if parsed.get('total_value', 0) >= fund_map[code]['meta'].get('total_value', 0):
                        fund_map[code]['meta'] = parsed
                    if parsed.get('price', 0) > 0:
                        parse_ok += 1
                    else:
                        parse_fail += 1

            print(f"[BES-BG] {len(fund_map)} unique fon, price>0: {parse_ok}, price=0: {parse_fail}")

            if fund_map:
                sorted_funds = sorted(fund_map.values(), key=lambda x: x['meta'].get('total_value', 0), reverse=True)

                # Debug: ilk 3 fonun verisini logla
                for fd in sorted_funds[:3]:
                    m = fd['meta']
                    print(f"[BES-BG] DEBUG fon: {m['code']} name={m['name'][:30]} price={m['price']} rows={len(fd['rows'])}")

                for fund_data in sorted_funds[:20]:
                    code = fund_data['meta']['code']
                    rows = fund_data['rows']
                    if len(rows) >= 2:
                        try:
                            perf = _analyze_fund_performance(rows, code)
                            if perf:
                                pool.append(perf)
                        except Exception as fe:
                            print(f"[BES-BG] Analiz hatasi {code}: {fe}")

                print(f"[BES-BG] Broad fetch'ten {len(pool)} fon analiz edildi")

                # Yetersizse sequential fetch
                if len(pool) < 5:
                    remaining = [fd for fd in sorted_funds[:20] if fd['meta']['code'] not in [p['code'] for p in pool]]
                    for fund_data in remaining[:5]:
                        code = fund_data['meta']['code']
                        try:
                            print(f"[BES-BG] Sequential fetch: {code}")
                            history = _fetch_tefas_history_chunked(code, days=200)
                            perf = _analyze_fund_performance(history, code)
                            if perf:
                                pool.append(perf)
                            time.sleep(1.5)
                        except Exception as fe:
                            print(f"[BES-BG] Sequential fetch hatasi {code}: {fe}")
                            time.sleep(2)

        # ===== YONTEM 3: Fallback =====
        if not pool:
            print("[BES-BG] Tum yontemler basarisiz, fallback moda geciliyor...")
            if raw and isinstance(raw, list):
                fund_map_fb = {}
                for row in raw:
                    parsed = _parse_fund_row(row)
                    if parsed and parsed['code']:
                        code = parsed['code']
                        if code not in fund_map_fb or parsed.get('total_value', 0) > fund_map_fb[code].get('total_value', 0):
                            fund_map_fb[code] = parsed
                for f in sorted(fund_map_fb.values(), key=lambda x: x.get('total_value', 0), reverse=True)[:15]:
                    pool.append({
                        'code': f['code'],
                        'name': f.get('name', ''),
                        'category': _classify_fund(f.get('name', '')),
                        'currentPrice': f.get('price', 0),
                        'firstPrice': f.get('price', 0),
                        'totalReturn': 0, 'totalDays': 1,
                        'returns': {'1h': None, '1a': None, '3a': None, '6a': None, '1y': None},
                        'volatility': 0, 'maxDrawdown': 0, 'sharpe': 0,
                        'dailyReturns': [], 'priceHistory': [],
                    })

        if pool:
            _bes_cache_set('bes_analysis_pool', pool)
            print(f"[BES-BG] Analiz tamamlandi: {len(pool)} fon cache'e yazildi")
        else:
            _bes_bg_error = 'Fon analizi yapilamadi - TEFAS API yanit vermiyor'
            print("[BES-BG] Hicbir fon analiz edilemedi")
    except Exception as e:
        _bes_bg_error = str(e)
        print(f"[BES-BG] HATA: {e}")
        traceback.print_exc()
    finally:
        _bes_bg_loading = False

def _classify_fund(fund_name):
    """Fon adina gore kategori tahmini"""
    name_upper = fund_name.upper() if fund_name else ''
    if any(k in name_upper for k in ['HİSSE', 'HISSE', 'EQUITY', 'PAY']): return 'hisse'
    if any(k in name_upper for k in ['BORÇLANMA', 'BORCLANMA', 'TAHVİL', 'TAHVIL', 'BONO', 'BOND']): return 'borclanma'
    if any(k in name_upper for k in ['KATILIM', 'KATKIM', 'SUKUK']): return 'katilim'
    if any(k in name_upper for k in ['KARMA', 'DENGELİ', 'DENGELI', 'MIX', 'BALANCED']): return 'karma'
    if any(k in name_upper for k in ['DÖVİZ', 'DOVIZ', 'EURO', 'DOLAR', 'USD', 'EUR', 'FX']): return 'doviz'
    if any(k in name_upper for k in ['ALTIN', 'KIYMETLI', 'GOLD', 'PRECIOUS', 'GÜMÜŞ', 'GUMUS']): return 'altin'
    if any(k in name_upper for k in ['ENDEKS', 'INDEX']): return 'endeks'
    if any(k in name_upper for k in ['LİKİT', 'LIKIT', 'PARA PİYASASI', 'PARA PIYASASI', 'MONEY']): return 'para_piyasasi'
    if any(k in name_upper for k in ['STANDART', 'STANDARD']): return 'standart'
    return 'diger'

def _fetch_tefas_funds(start_date, end_date, fund_code=''):
    """TEFAS API'den BES fon verisi cek (max 90 gun) - retry destekli, semaphore ile rate limited"""
    _tefas_semaphore.acquire()
    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = {
                    'fontip': 'EMK',
                    'sfontur': '',
                    'fonkod': fund_code,
                    'fongrup': '',
                    'bastarih': start_date,
                    'bittarih': end_date,
                    'fonturkod': '',
                    'fonunvantip': '',
                }
                resp = req_lib.post(TEFAS_API_URL, headers=TEFAS_HEADERS, data=data, timeout=30)
                if resp.status_code in (403, 429, 503):
                    wait = (attempt + 1) * 3
                    print(f"[BES] TEFAS rate limit ({resp.status_code}), {wait}s bekleniyor... (deneme {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                result = resp.json()
                # Debug: ilk cagrilarda API yapisini logla
                if fund_code and isinstance(result, dict):
                    keys = list(result.keys())[:5]
                    print(f"[BES] TEFAS response keys ({fund_code}): {keys}")
                if isinstance(result, dict) and 'data' in result:
                    rows = result['data']
                    if rows and isinstance(rows, list) and len(rows) > 0:
                        print(f"[BES] TEFAS {fund_code or 'ALL'}: {len(rows)} satir, ornek keys: {list(rows[0].keys())[:8] if isinstance(rows[0], dict) else 'not-dict'}")
                    return rows
                if isinstance(result, list):
                    if result and isinstance(result[0], dict):
                        print(f"[BES] TEFAS {fund_code or 'ALL'}: {len(result)} satir (list), ornek keys: {list(result[0].keys())[:8]}")
                    return result
                return result
            except Exception as e:
                print(f"[BES] TEFAS fetch hata (deneme {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
        return []
    finally:
        _tefas_semaphore.release()
        time.sleep(0.8)  # Her TEFAS cagrisi arasinda 0.8s bekleme


def _fetch_tefas_compare():
    """TEFAS Compare API'den tum EMK fonlarinin donemsel getirilerini cek.
    Bu endpoint resmi hesaplanmis getirileri dogrudan dondurur."""
    _tefas_semaphore.acquire()
    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = {
                    'fontip': 'EMK',
                    'sfontur': '',
                    'fonkod': '',
                    'fongrup': '',
                    'fonturkod': '',
                    'fonunvantip': '',
                }
                resp = req_lib.post(TEFAS_COMPARE_URL, headers=TEFAS_HEADERS, data=data, timeout=30)
                if resp.status_code in (403, 429, 503):
                    wait = (attempt + 1) * 3
                    print(f"[BES-CMP] TEFAS rate limit ({resp.status_code}), {wait}s bekleniyor...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                result = resp.json()
                rows = result.get('data', result) if isinstance(result, dict) else result
                if rows and isinstance(rows, list) and len(rows) > 0:
                    sample = rows[0] if isinstance(rows[0], dict) else {}
                    print(f"[BES-CMP] TEFAS Compare basarili: {len(rows)} fon, ornek keys: {list(sample.keys())[:12]}")
                    # Ilk satirdan tum key'leri logla (debug)
                    if sample:
                        print(f"[BES-CMP] Ornek fon: {dict(list(sample.items())[:10])}")
                    return rows
                print(f"[BES-CMP] TEFAS Compare bos yanit: {type(result)}")
                return []
            except Exception as e:
                print(f"[BES-CMP] TEFAS Compare hata (deneme {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
        return []
    finally:
        _tefas_semaphore.release()
        time.sleep(0.8)


def _parse_compare_row(row):
    """TEFAS Compare API satirindan getiri bilgilerini cek"""
    if not isinstance(row, dict):
        return None

    code = _get_tefas_field(row, 'FonKodu', 'fonkodu', 'FONKODU', 'FonKod', default='')
    name = _get_tefas_field(row, 'FonUnvani', 'fonunvani', 'FONUNVANI', 'FonAdi', 'Fon', default='')
    price = sf(_get_tefas_field(row, 'BirimPayDegeri', 'birimpay', 'BirimPayDeger', 'FonFiyat', default=0))
    total_value = sf(_get_tefas_field(row, 'ToplamDeger', 'toplamdeger', 'TOPLAMDEGER', default=0))

    # Getiri alanlari - TEFAS farkli isimler kullanabilir
    ret_1h = _get_tefas_field(row, 'HaftalikGetiri', 'haftalikgetiri', 'BirHaftalikGetiri',
                               '1HaftalikGetiri', 'Haftalik', default=None)
    ret_1a = _get_tefas_field(row, 'AylikGetiri', 'aylikgetiri', 'BirAylikGetiri',
                               '1AylikGetiri', 'Aylik', default=None)
    ret_3a = _get_tefas_field(row, 'UcAylikGetiri', 'ucaylikgetiri', '3AylikGetiri',
                               'UcAylik', default=None)
    ret_6a = _get_tefas_field(row, 'AltiAylikGetiri', 'altiaylikgetiri', '6AylikGetiri',
                               'AltiAylik', default=None)
    ret_1y = _get_tefas_field(row, 'YillikGetiri', 'yillikgetiri', 'BirYillikGetiri',
                               '1YillikGetiri', 'Yillik', 'YilBasindanGetiri', default=None)
    ret_daily = _get_tefas_field(row, 'GunlukGetiri', 'gunlukgetiri', 'Gunluk', default=None)

    # Sayisal deger alanlari da dene (bazi TEFAS versiyonlari farkli isimlendirme kullaniyor)
    for key, val_ref in row.items():
        kl = key.lower().replace('İ', 'i').replace('ı', 'i')
        if val_ref is not None and isinstance(val_ref, (int, float)):
            if ret_1h is None and ('hafta' in kl or '1h' in kl or '1w' in kl): ret_1h = val_ref
            elif ret_1a is None and ('1ay' in kl or '1a' in kl or '1m' in kl) and 'aylik' not in kl: ret_1a = val_ref
            elif ret_3a is None and ('3ay' in kl or 'ucay' in kl or '3a' in kl or '3m' in kl): ret_3a = val_ref
            elif ret_6a is None and ('6ay' in kl or 'altiay' in kl or '6a' in kl or '6m' in kl): ret_6a = val_ref
            elif ret_1y is None and ('yil' in kl or '1y' in kl or '12' in kl) and 'basindan' not in kl: ret_1y = val_ref

    returns = {}
    if ret_1h is not None: returns['1h'] = sf(float(ret_1h))
    if ret_1a is not None: returns['1a'] = sf(float(ret_1a))
    if ret_3a is not None: returns['3a'] = sf(float(ret_3a))
    if ret_6a is not None: returns['6a'] = sf(float(ret_6a))
    if ret_1y is not None: returns['1y'] = sf(float(ret_1y))

    return {
        'code': code or '',
        'name': name or '',
        'price': price or 0,
        'total_value': total_value or 0,
        'returns': returns,
        'daily_return': sf(float(ret_daily)) if ret_daily is not None else 0,
    }

def _fetch_tefas_allocation(fund_code, start_date, end_date):
    """TEFAS API'den fon portfoy dagilimini cek"""
    try:
        data = {
            'fontip': 'EMK',
            'fonkod': fund_code,
            'bastarih': start_date,
            'bittarih': end_date,
        }
        resp = req_lib.post(TEFAS_ALLOC_URL, headers=TEFAS_HEADERS, data=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, dict) and 'data' in result:
            return result['data']
        if isinstance(result, list):
            return result
        return result
    except Exception as e:
        print(f"[BES] TEFAS allocation hata: {e}")
        return []

def _fetch_tefas_history_chunked(fund_code, days=365):
    """90 gunluk chunk'larla uzun sureli fon gecmisi cek - semaphore ile rate limited"""
    all_data = []
    end = datetime.now()
    start = end - timedelta(days=days)
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=89), end)
        sd = chunk_start.strftime('%d.%m.%Y')
        ed = chunk_end.strftime('%d.%m.%Y')
        chunk_data = _fetch_tefas_funds(sd, ed, fund_code)
        if chunk_data and isinstance(chunk_data, list):
            all_data.extend(chunk_data)
        chunk_start = chunk_end + timedelta(days=1)
        # Rate limiting artik _fetch_tefas_funds icinde semaphore ile yapiliyor
    return all_data

def _get_tefas_field(row, *keys, default=None):
    """TEFAS API field'ini case-insensitive olarak bul"""
    for key in keys:
        if key in row:
            return row[key]
    # Case-insensitive fallback
    row_lower = {k.lower(): v for k, v in row.items()}
    for key in keys:
        kl = key.lower()
        if kl in row_lower:
            return row_lower[kl]
    return default

def _parse_fund_row(row):
    """TEFAS API'den donen bir fund row'unu parse et - genis field name destegi"""
    if isinstance(row, dict):
        code = _get_tefas_field(row, 'FonKodu', 'fonkodu', 'FONKODU', 'FonKod', default='')
        name = _get_tefas_field(row, 'FonUnvani', 'fonunvani', 'FONUNVANI', 'FonAdi', default='')
        date_raw = _get_tefas_field(row, 'Tarih', 'tarih', 'TARIH', default='')
        price = sf(_get_tefas_field(row, 'BirimPayDegeri', 'birimpay', 'BIRIMPAY', 'BirimPayDeger', default=0))
        total_value = sf(_get_tefas_field(row, 'ToplamDeger', 'toplamdeger', 'TOPLAMDEGER', default=0))
        investors = si(_get_tefas_field(row, 'YatirimciSayisi', 'yatirimcisayisi', 'YATIRIMCISAYISI', default=0))
        shares = sf(_get_tefas_field(row, 'PaySayisi', 'paysayisi', 'PAYSAYISI', default=0))

        # Eger price 0 ama total_value var ise, total_value/shares'den hesapla
        if (not price or price <= 0) and total_value and total_value > 0 and shares and shares > 0:
            price = sf(total_value / shares)

        # Tarih WCF formatinda olabilir
        if date_raw and isinstance(date_raw, str) and '/Date(' in date_raw:
            dt = _parse_tefas_date(date_raw)
            if dt:
                date_raw = dt.strftime('%d.%m.%Y')

        return {
            'code': code or '',
            'name': name or '',
            'date': date_raw or '',
            'price': price or 0,
            'total_value': total_value or 0,
            'investors': investors or 0,
            'shares': shares or 0,
        }
    return None

def _parse_tefas_date(date_val):
    """TEFAS tarih formatlarini parse et: 'dd.MM.yyyy', 'yyyy-MM-dd', '/Date(timestamp)/' """
    if not date_val:
        return None
    if isinstance(date_val, str):
        # WCF date format: /Date(1645488000000)/
        wcf_match = re.match(r'/Date\((\-?\d+)\)/', date_val)
        if wcf_match:
            ts = int(wcf_match.group(1)) / 1000
            return datetime.fromtimestamp(ts)
        for fmt in ['%d.%m.%Y', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S']:
            try:
                return datetime.strptime(date_val, fmt)
            except Exception:
                continue
    return None

