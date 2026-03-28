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

def _analyze_fund_performance(history_data, fund_code):
    """Fon gecmis verisinden performans metrikleri hesapla"""
    if not history_data or len(history_data) < 2:
        return None

    # Fiyatlari cikar ve sırala
    prices = []
    for row in history_data:
        parsed = _parse_fund_row(row)
        if parsed and parsed['price'] and parsed['price'] > 0:
            dt = _parse_tefas_date(parsed['date'])
            prices.append({
                'date': parsed['date'],
                'date_parsed': dt,
                'price': parsed['price'],
                'code': parsed['code'],
                'name': parsed['name'],
            })

    if len(prices) < 2:
        return None

    # Tarihe gore sirala (en eski en basta) - datetime objeleriyle dogru siralama
    try:
        prices.sort(key=lambda x: x['date_parsed'] or datetime.min)
    except Exception:
        # Fallback: string siralama
        prices.sort(key=lambda x: x['date'])

    current_price = prices[-1]['price']
    first_price = prices[0]['price']
    total_return = ((current_price - first_price) / first_price) * 100 if first_price > 0 else 0
    total_days = len(prices)

    # Donemsel getiriler - tarih bazli lookback (takvim gunu -> islem gunu donusumu)
    returns = {}
    period_map = {'1h': 7, '1a': 30, '3a': 90, '6a': 180, '1y': 365}
    for label, cal_days in period_map.items():
        # Tarih bazli lookback: en son tarihten cal_days gun oncesine en yakin veri noktasini bul
        target_date = prices[-1]['date_parsed'] - timedelta(days=cal_days) if prices[-1].get('date_parsed') else None
        found_price = None

        if target_date:
            # Target tarihine en yakin (ve oncesindeki) veri noktasini bul
            for p in prices:
                if p.get('date_parsed') and p['date_parsed'] <= target_date:
                    found_price = p['price']
                # Once erisince devam et (sirali oldugu icin son eslesme en yakin olacak)

        # Tarih bazli bulunamadiysa, index bazli fallback
        if not found_price:
            # Tahmini islem gunu: takvim gunu * 5/7 (hafta ici orani)
            approx_trading_days = max(1, int(cal_days * 5 / 7))
            if len(prices) >= approx_trading_days:
                found_price = prices[-min(approx_trading_days, len(prices))]['price']
            elif len(prices) >= max(cal_days // 4, 2):
                # En az ceyrek kadar veri varsa mevcut verinin basindan hesapla
                found_price = prices[0]['price']

        if found_price and found_price > 0:
            returns[label] = sf(((current_price - found_price) / found_price) * 100)
        else:
            returns[label] = None

    # Volatilite (gunluk fiyat degisim std sapma)
    daily_returns = []
    for i in range(1, len(prices)):
        if prices[i-1]['price'] > 0:
            dr = (prices[i]['price'] - prices[i-1]['price']) / prices[i-1]['price']
            daily_returns.append(dr)

    volatility = 0
    if daily_returns:
        mean_r = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_r)**2 for r in daily_returns) / len(daily_returns)
        volatility = sf(variance ** 0.5 * (252 ** 0.5) * 100)  # Yillik volatilite %

    # Max drawdown
    max_dd = 0
    peak = prices[0]['price']
    for p in prices:
        if p['price'] > peak:
            peak = p['price']
        dd = ((peak - p['price']) / peak) * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe benzeri metrik (risksiz oran ~%45 TRY)
    risk_free_daily = 0.45 / 252
    avg_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
    std_daily = (sum((r - avg_daily_return)**2 for r in daily_returns) / len(daily_returns)) ** 0.5 if daily_returns else 1
    sharpe = sf(((avg_daily_return - risk_free_daily) / std_daily * (252 ** 0.5)) if std_daily > 0 else 0)

    return {
        'code': fund_code,
        'name': prices[-1].get('name', ''),
        'category': _classify_fund(prices[-1].get('name', '')),
        'currentPrice': current_price,
        'firstPrice': first_price,
        'totalReturn': sf(total_return),
        'totalDays': total_days,
        'returns': returns,
        'volatility': volatility,
        'maxDrawdown': sf(max_dd),
        'sharpe': sharpe,
        'dailyReturns': daily_returns[-30:] if daily_returns else [],
        'priceHistory': [{'date': p['date'], 'price': p['price']} for p in prices[-90:]],
    }

def _bes_optimize(funds_perf, risk_profile='moderate', horizon_months=12):
    """
    BES fon dagilim optimizasyonu.
    Risk profiline gore ideal fon agirliklarini hesaplar.
    """
    if not funds_perf:
        return []

    risk_weights = {
        'conservative': {'return_w': 0.2, 'risk_w': 0.6, 'sharpe_w': 0.2, 'max_equity': 20},
        'moderate':     {'return_w': 0.35, 'risk_w': 0.35, 'sharpe_w': 0.3, 'max_equity': 50},
        'aggressive':   {'return_w': 0.5, 'risk_w': 0.15, 'sharpe_w': 0.35, 'max_equity': 80},
    }
    weights = risk_weights.get(risk_profile, risk_weights['moderate'])

    scored_funds = []
    for f in funds_perf:
        if not f:
            continue
        rets = f.get('returns', {})
        ret_6m = rets.get('6a') or rets.get('3a') or rets.get('1a') or 0
        vol = f.get('volatility', 50)
        sharpe = f.get('sharpe', 0)
        category = f.get('category', 'diger')

        # Normalizing: return pozitif iyi, vol dusuk iyi, sharpe yuksek iyi
        ret_score = min(max(ret_6m, -50), 100) / 100
        vol_score = 1 - min(vol, 100) / 100
        sharpe_score = min(max(sharpe, -3), 3) / 3

        total_score = (
            ret_score * weights['return_w'] +
            vol_score * weights['risk_w'] +
            sharpe_score * weights['sharpe_w']
        )

        scored_funds.append({
            'code': f['code'],
            'name': f['name'],
            'category': category,
            'categoryLabel': BES_FUND_GROUPS.get(category, 'Diğer'),
            'score': sf(total_score * 100),
            'return6m': sf(ret_6m),
            'volatility': f['volatility'],
            'sharpe': f['sharpe'],
            'maxDrawdown': f['maxDrawdown'],
            'currentPrice': f['currentPrice'],
        })

    scored_funds.sort(key=lambda x: x['score'], reverse=True)

    # Kategori bazli dagilim onerisi
    category_targets = {
        'conservative': {'borclanma': 40, 'para_piyasasi': 25, 'altin': 15, 'katilim': 10, 'hisse': 5, 'diger': 5},
        'moderate':     {'hisse': 25, 'borclanma': 25, 'altin': 15, 'katilim': 15, 'para_piyasasi': 10, 'diger': 10},
        'aggressive':   {'hisse': 40, 'endeks': 15, 'altin': 15, 'doviz': 10, 'borclanma': 10, 'karma': 10},
    }
    targets = category_targets.get(risk_profile, category_targets['moderate'])

    # Her kategori icin en iyi fonu sec
    recommendations = []
    used_categories = set()
    for cat, target_pct in sorted(targets.items(), key=lambda x: x[1], reverse=True):
        best_in_cat = [f for f in scored_funds if f['category'] == cat]
        if best_in_cat:
            pick = best_in_cat[0]
            pick['recommendedPct'] = target_pct
            pick['reasoning'] = _fund_reasoning(pick, cat, target_pct, risk_profile)
            recommendations.append(pick)
            used_categories.add(cat)

    # Hedef kategoride fon bulunamazsa en iyi genel fonlardan tamamla
    total_allocated = sum(r['recommendedPct'] for r in recommendations)
    if total_allocated < 100:
        remaining = 100 - total_allocated
        remaining_funds = [f for f in scored_funds if f['code'] not in [r['code'] for r in recommendations]]
        if remaining_funds:
            pick = remaining_funds[0]
            pick['recommendedPct'] = remaining
            pick['reasoning'] = f"Kalan %{remaining} oran portföy dengeleme amacıyla önerildi."
            recommendations.append(pick)

    return recommendations

def _fund_reasoning(fund, category, pct, risk_profile):
    """Fon onerisi icin aciklama metni uret"""
    cat_label = BES_FUND_GROUPS.get(category, category)
    sharpe = fund.get('sharpe', 0)
    vol = fund.get('volatility', 0)
    ret = fund.get('return6m', 0)

    reasons = []
    if ret > 10: reasons.append(f"6 aylık getirisi %{ret} ile güçlü")
    elif ret > 0: reasons.append(f"6 aylık %{ret} pozitif getiri")
    else: reasons.append(f"6 aylık getiri %{ret}")

    if vol < 10: reasons.append("düşük volatilite")
    elif vol < 20: reasons.append("orta seviye volatilite")
    else: reasons.append(f"%{vol} volatilite")

    if sharpe > 1: reasons.append("yüksek risk-getiri oranı")
    elif sharpe > 0: reasons.append("pozitif risk-getiri dengesi")

    profile_labels = {'conservative': 'muhafazakar', 'moderate': 'dengeli', 'aggressive': 'agresif'}
    profile_label = profile_labels.get(risk_profile, 'dengeli')

    return f"{cat_label} kategorisinde {profile_label} profil için %{pct} ağırlık. " + ", ".join(reasons) + "."

def _simulate_bes(recommendations, monthly_contribution, horizon_months):
    """BES portfoy simulasyonu: aylık katkı ile birikim projeksiyonu"""
    if not recommendations or monthly_contribution <= 0 or horizon_months <= 0:
        return {'error': 'Geçersiz parametreler'}

    # Devlet katkisi (%30, yillik max limit - 2024 icin ~27.000 TL civarı)
    yearly_contribution = monthly_contribution * 12
    devlet_katkisi_rate = 0.30
    devlet_katkisi_yearly_max = 30000  # Yaklasik yillik limit
    devlet_katkisi_yearly = min(yearly_contribution * devlet_katkisi_rate, devlet_katkisi_yearly_max)
    devlet_katkisi_monthly = devlet_katkisi_yearly / 12

    # Her fon icin aylik getiri tahmini (gecmis veriden)
    total_monthly = monthly_contribution + devlet_katkisi_monthly
    monthly_results = []
    fund_balances = {}

    for rec in recommendations:
        pct = rec.get('recommendedPct', 0) / 100
        ret_6m = rec.get('return6m', 0) or 0
        # 6 aylik getiriyi aylik getiriye cevir
        monthly_return = ((1 + ret_6m / 100) ** (1/6) - 1)
        fund_balances[rec['code']] = {
            'name': rec['name'],
            'code': rec['code'],
            'pct': pct,
            'monthlyReturn': monthly_return,
            'balance': 0,
            'totalContribution': 0,
        }

    total_balance = 0
    total_contributed = 0
    total_devlet = 0
    total_gain = 0

    for month in range(1, horizon_months + 1):
        month_contribution = monthly_contribution
        month_devlet = devlet_katkisi_monthly
        total_contributed += month_contribution
        total_devlet += month_devlet

        for code, fb in fund_balances.items():
            contrib = (month_contribution + month_devlet) * fb['pct']
            fb['totalContribution'] += contrib
            # Birikim: onceki bakiye * (1 + aylik getiri) + yeni katki
            fb['balance'] = fb['balance'] * (1 + fb['monthlyReturn']) + contrib

        total_balance = sum(fb['balance'] for fb in fund_balances.values())
        total_gain = total_balance - total_contributed - total_devlet

        if month % 3 == 0 or month == horizon_months or month <= 3:
            monthly_results.append({
                'month': month,
                'totalBalance': sf(total_balance),
                'totalContributed': sf(total_contributed),
                'devletKatkisi': sf(total_devlet),
                'totalGain': sf(total_gain),
                'gainPct': sf((total_gain / (total_contributed + total_devlet)) * 100) if (total_contributed + total_devlet) > 0 else 0,
            })

    fund_details = []
    for code, fb in fund_balances.items():
        gain = fb['balance'] - fb['totalContribution']
        fund_details.append({
            'code': fb['code'],
            'name': fb['name'],
            'pct': sf(fb['pct'] * 100),
            'balance': sf(fb['balance']),
            'totalContribution': sf(fb['totalContribution']),
            'gain': sf(gain),
            'gainPct': sf((gain / fb['totalContribution']) * 100) if fb['totalContribution'] > 0 else 0,
            'monthlyReturn': sf(fb['monthlyReturn'] * 100, 3),
        })

    return {
        'totalBalance': sf(total_balance),
        'totalContributed': sf(total_contributed),
        'devletKatkisi': sf(total_devlet),
        'totalGain': sf(total_gain),
        'totalGainPct': sf((total_gain / (total_contributed + total_devlet)) * 100) if (total_contributed + total_devlet) > 0 else 0,
        'horizonMonths': horizon_months,
        'monthlyContribution': monthly_contribution,
        'monthlyDevlet': sf(devlet_katkisi_monthly),
        'timeline': monthly_results,
        'fundDetails': fund_details,
    }
