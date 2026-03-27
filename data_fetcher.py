"""
BIST Pro - Data Fetcher & Background Loader Module
"""
import os, time, threading, traceback, json, re
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pass
from config import (
    sf, si, _lock, _stock_cache, _index_cache, _hist_cache,
    _cget, _cset, _ctouch, _cget_hist, _get_stocks,
    _loader_started, _status, CACHE_TTL, CACHE_STALE_TTL, HIST_CACHE_TTL,
    BIST100_STOCKS, BIST30, INDEX_TICKERS, PARALLEL_WORKERS, app, BASE_DIR
)

# Lazy imports to avoid circular dependencies
def _get_db_save_snapshot():
    from database import _db_save_market_snapshot
    return _db_save_market_snapshot

def _get_auto_engine():
    try:
        from auto_trader import _auto_engine_cycle
        return _auto_engine_cycle
    except Exception:
        return None

import urllib.request
import urllib.error
import requests as req_lib
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

IS_YATIRIM_BASE = "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil"
IS_YATIRIM_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://www.isyatirim.com.tr/',
    'Accept': 'application/json',
}

# ---- IS YATIRIM API ----
def _fetch_isyatirim_df(symbol, days=365):
    """Is Yatirim'dan OHLCV DataFrame cek - BIRINCIL KAYNAK"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        sd = start_date.strftime('%d-%m-%Y')
        ed = end_date.strftime('%d-%m-%Y')
        url = f"{IS_YATIRIM_BASE}?hisse={symbol}&startdate={sd}&enddate={ed}"

        print(f"  [ISYATIRIM] {symbol} {days}d cekiliyor...")
        resp = None
        last_err = None
        for http_attempt in range(2):
            try:
                resp = req_lib.get(url, headers=IS_YATIRIM_HEADERS, timeout=10, verify=False)
                resp.raise_for_status()
                break
            except Exception as http_e:
                last_err = http_e
                if http_attempt < 1:
                    print(f"  [ISYATIRIM] {symbol} HTTP hata ({http_e}), 0.5s sonra tekrar...")
                    time.sleep(0.5)
        if resp is None:
            print(f"  [ISYATIRIM] {symbol} HTTP denemesi basarisiz: {last_err}")
            return None

        try:
            data = resp.json()
        except Exception as json_e:
            print(f"  [ISYATIRIM] {symbol}: JSON parse hatasi: {json_e}, ilk 200 karakter: {resp.text[:200]}")
            return None

        # API yanit formatini kontrol et (value, d, veya dogrudan liste)
        rows = data.get('value', [])
        if not rows:
            rows = data.get('d', [])
        if not rows and isinstance(data, list):
            rows = data

        if not rows or len(rows) < 2:
            print(f"  [ISYATIRIM] {symbol}: bos veri ({len(rows) if rows else 0} satir), response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            return None

        # Ilk satirdaki TUM kolonlari logla
        if len(rows) > 0:
            cols = list(rows[0].keys())
            print(f"  [ISYATIRIM] {symbol}: {len(rows)} satir, kolonlar: {cols}")

        # DataFrame olustur - kolon isimlerini otomatik kesfet
        df_raw = pd.DataFrame(rows)

        # Tarih kolonu - esnek arama
        date_col = None
        for c in df_raw.columns:
            cu = c.upper()
            if 'TARIH' in cu or 'DATE' in cu or 'TARH' in cu:
                date_col = c; break
        if not date_col:
            print(f"  [ISYATIRIM] {symbol}: tarih kolonu bulunamadi, kolonlar: {list(df_raw.columns)}")
            return None

        # OHLCV kolonlari - esnek mapping (API degisikliklerine karsi guclu)
        col_map = {}
        for c in df_raw.columns:
            cu = c.upper()
            # Close: KAPANIS_FIYATI, KAPANIS, HISSE_KAPANIS, vb.
            if 'Close' not in col_map and 'KAPANIS' in cu and 'DUZELTILMIS' not in cu:
                col_map['Close'] = c
            # Open: ACILIS_FIYATI, ACILIS, vb.
            elif 'Open' not in col_map and 'ACILIS' in cu:
                col_map['Open'] = c
            # High: EN_YUKSEK, YUKSEK, YUKSEK_FIYAT, vb.
            elif 'High' not in col_map and ('YUKSEK' in cu or 'EN_YUKSEK' in cu or 'HIGH' in cu):
                col_map['High'] = c
            # Low: EN_DUSUK, DUSUK, DUSUK_FIYAT, vb.
            elif 'Low' not in col_map and ('DUSUK' in cu or 'EN_DUSUK' in cu or 'LOW' in cu):
                col_map['Low'] = c
            # Volume: HACIM_LOT, HACIM (LOT), ISLEM_HACMI, vb.
            elif 'Volume' not in col_map and 'HACIM' in cu and 'TL' not in cu:
                col_map['Volume'] = c

        # Volume bulunamadiysa HACIM iceren herhangi bir kolonu dene
        if 'Volume' not in col_map:
            for c in df_raw.columns:
                cu = c.upper()
                if 'HACIM' in cu or 'VOLUME' in cu or 'ADET' in cu:
                    col_map['Volume'] = c; break

        # Close bulunamadiysa DUZELTILMIS KAPANIS veya herhangi kapanis
        if 'Close' not in col_map:
            for c in df_raw.columns:
                if 'KAPANIS' in c.upper():
                    col_map['Close'] = c; break
            if 'Close' not in col_map:
                # Son care: numerik kolonlari dene (CLOSE, PRICE, FIYAT)
                for c in df_raw.columns:
                    cu = c.upper()
                    if 'CLOSE' in cu or 'FIYAT' in cu or 'PRICE' in cu:
                        col_map['Close'] = c; break
            if 'Close' not in col_map:
                print(f"  [ISYATIRIM] {symbol}: Close kolonu bulunamadi, kolonlar: {list(df_raw.columns)}")
                return None

        print(f"  [ISYATIRIM] {symbol} mapping: {col_map}")

        # DataFrame build
        df = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(df_raw[date_col])))
        df['Close'] = pd.to_numeric(df_raw[col_map['Close']].values, errors='coerce')
        df['Open'] = pd.to_numeric(df_raw[col_map.get('Open', col_map['Close'])].values, errors='coerce')
        df['High'] = pd.to_numeric(df_raw[col_map.get('High', col_map['Close'])].values, errors='coerce')
        df['Low'] = pd.to_numeric(df_raw[col_map.get('Low', col_map['Close'])].values, errors='coerce')

        if 'Volume' in col_map:
            df['Volume'] = pd.to_numeric(df_raw[col_map['Volume']].values, errors='coerce').fillna(0).astype(int)
        else:
            df['Volume'] = 0

        # NaN degerlerini doldur: Open/High/Low icin Close kullan (kritik fix)
        df['Open'] = df['Open'].fillna(df['Close'])
        df['High'] = df['High'].fillna(df['Close'])
        df['Low'] = df['Low'].fillna(df['Close'])

        # High en az Close kadar, Low en fazla Close kadar olmali
        df['High'] = df[['High', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Close']].min(axis=1)

        df = df.dropna(subset=['Close']).sort_index()

        if len(df) < 2:
            return None

        # Veri kalitesi kontrolu
        nan_counts = df[['Open','High','Low','Close']].isna().sum()
        total_nan = nan_counts.sum()
        if total_nan > 0:
            print(f"  [ISYATIRIM] {symbol} UYARI: {total_nan} NaN deger doldurulamadi: {nan_counts.to_dict()}")

        # Son fiyat kontrolu
        last_close = float(df['Close'].iloc[-1])
        if last_close <= 0:
            print(f"  [ISYATIRIM] {symbol} UYARI: Son kapanis fiyati 0 veya negatif: {last_close}")

        print(f"  [ISYATIRIM] {symbol} OK: {len(df)} bar, {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}, son fiyat: {last_close}")
        return df

    except Exception as e:
        print(f"  [ISYATIRIM] {symbol} HATA: {e}")
        import traceback
        traceback.print_exc()
        return None


def _fetch_isyatirim_quick(symbol):
    """Is Yatirim'dan son fiyat bilgisi (quick - dashboard icin)"""
    try:
        df = _fetch_isyatirim_df(symbol, days=5)
        if df is None or len(df) < 2:
            return None
        cur = float(df['Close'].iloc[-1])
        prev = float(df['Close'].iloc[-2])
        return {
            'close': cur, 'prev': prev,
            'open': float(df['Open'].iloc[-1]),
            'high': float(df['High'].iloc[-1]),
            'low': float(df['Low'].iloc[-1]),
            'volume': int(df['Volume'].iloc[-1]),
        }
    except Exception as e:
        print(f"  [ISYATIRIM-Q] {symbol}: {e}")
        return None


# ---- YAHOO HTTP API (YEDEK) ----
def _fetch_yahoo_http(symbol, period1_days=14):
    """Yahoo Finance v8 chart API - yedek kaynak"""
    try:
        now = int(time.time())
        p1 = now - (period1_days * 86400)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={now}&interval=1d"
        r = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(r, timeout=8) as resp:
            raw = json.loads(resp.read().decode())

        result = raw.get('chart', {}).get('result', [])
        if not result: return None

        data = result[0]
        quote = data.get('indicators', {}).get('quote', [{}])[0]
        closes = quote.get('close', [])
        if not closes or len(closes) < 2: return None

        valid = [(i, c) for i, c in enumerate(closes) if c is not None]
        if len(valid) < 2: return None

        last_i, cur = valid[-1]; prev_i, prev = valid[-2]
        opens = quote.get('open', []); highs = quote.get('high', [])
        lows = quote.get('low', []); volumes = quote.get('volume', [])

        o_val = float(opens[last_i]) if last_i < len(opens) and opens[last_i] is not None else float(cur)
        h_val = float(highs[last_i]) if last_i < len(highs) and highs[last_i] is not None else float(cur)
        l_val = float(lows[last_i]) if last_i < len(lows) and lows[last_i] is not None else float(cur)
        v_val = int(volumes[last_i]) if last_i < len(volumes) and volumes[last_i] is not None else 0
        return {
            'close': float(cur), 'prev': float(prev),
            'open': o_val, 'high': max(h_val, float(cur)), 'low': min(l_val, float(cur)),
            'volume': v_val,
        }
    except Exception as e:
        print(f"  [YAHOO-HTTP] {symbol}: {e}")
        return None


def _fetch_yahoo_http_df(symbol, period1_days=365):
    """Yahoo Finance v8 - tam DataFrame (yedek)"""
    try:
        now = int(time.time())
        p1 = now - (period1_days * 86400)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={now}&interval=1d"
        r = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(r, timeout=12) as resp:
            raw = json.loads(resp.read().decode())

        result = raw.get('chart', {}).get('result', [])
        if not result: return None

        data = result[0]
        timestamps = data.get('timestamp', [])
        quote = data.get('indicators', {}).get('quote', [{}])[0]
        if not timestamps or not quote.get('close'): return None

        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        df = pd.DataFrame({
            'Open': [float(v) if v else np.nan for v in quote.get('open', [])],
            'High': [float(v) if v else np.nan for v in quote.get('high', [])],
            'Low': [float(v) if v else np.nan for v in quote.get('low', [])],
            'Close': [float(v) if v else np.nan for v in quote.get('close', [])],
            'Volume': [int(v) if v else 0 for v in quote.get('volume', [])],
        }, index=pd.DatetimeIndex(dates))
        df = df.dropna(subset=['Close'])

        # NaN degerlerini doldur: Open/High/Low icin Close kullan
        df['Open'] = df['Open'].fillna(df['Close'])
        df['High'] = df['High'].fillna(df['Close'])
        df['Low'] = df['Low'].fillna(df['Close'])
        df['High'] = df[['High', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Close']].min(axis=1)

        if len(df) < 10: return None
        print(f"  [YAHOO-DF] {symbol}: {len(df)} bar")
        return df
    except Exception as e:
        print(f"  [YAHOO-DF] {symbol}: {e}")
        return None


# ---- BIRLESIK FETCHER (3 katmanli fallback) ----
def _fetch_stock_data(sym, retry_count=2):
    """Hisse verisi cek: 1.IsYatirim -> 2.Yahoo HTTP -> 3.yfinance (retry destekli)"""
    for attempt in range(1, retry_count + 1):
        # 1. Is Yatirim
        data = _fetch_isyatirim_quick(sym)
        if data:
            return data

        # 2. Yahoo HTTP
        for ticker in [f"{sym}.IS", sym]:
            data = _fetch_yahoo_http(ticker)
            if data:
                print(f"  [YAHOO] {sym} OK")
                return data

        # 3. yfinance (son care)
        if YF_OK:
            try:
                h = yf.Ticker(f"{sym}.IS").history(period="5d", timeout=6)
                if h is not None and not h.empty and len(h) >= 2:
                    cur, prev = float(h['Close'].iloc[-1]), float(h['Close'].iloc[-2])
                    if prev > 0:
                        print(f"  [YF] {sym} OK: {cur}")
                        return {'close': cur, 'prev': prev, 'open': float(h['Open'].iloc[-1]),
                                'high': float(h['High'].iloc[-1]), 'low': float(h['Low'].iloc[-1]),
                                'volume': int(h['Volume'].iloc[-1])}
            except Exception as e:
                print(f"  [YF] {sym}: {e}")

        if attempt < retry_count:
            wait = attempt * 0.5
            print(f"  [RETRY] {sym} deneme {attempt}/{retry_count} basarisiz, {wait}s bekleniyor...")
            time.sleep(wait)

    print(f"  [ALL] {sym} BASARISIZ ({retry_count} deneme)")
    return None


def _fetch_hist_df(sym, period='1y'):
    """Tam DataFrame cek (indicators icin): 1.IsYatirim -> 2.Yahoo HTTP DF -> 3.yfinance"""
    period_days = {'1mo':30,'3mo':90,'6mo':180,'1y':365,'2y':730,'5y':1825}.get(period, 365)

    # 1. Is Yatirim
    df = _fetch_isyatirim_df(sym, period_days)
    if df is not None and len(df) >= 10:
        return df

    # 2. Yahoo HTTP DataFrame
    for ticker in [f"{sym}.IS", sym]:
        df = _fetch_yahoo_http_df(ticker, period_days)
        if df is not None and len(df) >= 10:
            return df

    # 3. yfinance
    if YF_OK:
        try:
            h = yf.Ticker(f"{sym}.IS").history(period=period, timeout=10)
            if h is not None and not h.empty and len(h) >= 10:
                # NaN temizligi: yfinance verisi de NaN icerebilir
                if 'Close' in h.columns:
                    h = h.dropna(subset=['Close'])
                    if 'Open' in h.columns: h['Open'] = h['Open'].fillna(h['Close'])
                    if 'High' in h.columns: h['High'] = h['High'].fillna(h['Close'])
                    if 'Low' in h.columns: h['Low'] = h['Low'].fillna(h['Close'])
                    if 'Volume' in h.columns: h['Volume'] = h['Volume'].fillna(0)
                print(f"  [YF-HIST] {sym} OK: {len(h)} bar")
                return h
        except Exception as e:
            print(f"  [YF-HIST] {sym}: {e}")

    return None


def _resample_to_tf(hist_daily, tf):
    """
    Gunluk OHLCV verisini haftalik ('weekly') veya aylik ('monthly') bara donustur.
    Portabl yaklasim: pandas groupby + Period kullanır (resample versiyonuna bagli degil).
    """
    try:
        df = hist_daily.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        period_code = 'W' if tf == 'weekly' else 'M'
        df['_p'] = df.index.to_period(period_code)

        agg = {'Close': 'last', 'High': 'max', 'Low': 'min', 'Volume': 'sum'}
        if 'Open' in df.columns:
            agg['Open'] = 'first'

        resampled = df.groupby('_p').agg(agg)
        resampled.index = resampled.index.to_timestamp()
        resampled = resampled.dropna(subset=['Close'])

        # Son bar tamamlanmamis olabilir (suanki hafta/ay) — cikar
        if len(resampled) > 2:
            resampled = resampled.iloc[:-1]

        if 'High' not in resampled.columns:
            resampled['High'] = resampled['Close']
        if 'Low' not in resampled.columns:
            resampled['Low'] = resampled['Close']

        return resampled if len(resampled) >= 5 else None
    except Exception as e:
        print(f"  [MTF-RESAMPLE] {tf}: {e}")
        return None


def _process_stock(sym, retry_count=2):
    """Tek hisseyi cek ve cache formatinda dondur (thread-safe)"""
    try:
        data = _fetch_stock_data(sym, retry_count=retry_count)
        if data:
            cur, prev = sf(data['close']), sf(data['prev'])
            if prev > 0:
                ch = sf(cur - prev); o = sf(data.get('open', cur))
                return sym, {
                    'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                    'price': cur, 'prevClose': prev,
                    'change': ch, 'changePct': sf(ch / prev * 100),
                    'volume': si(data.get('volume', 0)),
                    'open': o, 'high': sf(data.get('high', cur)),
                    'low': sf(data.get('low', cur)),
                    'gap': sf(o - prev), 'gapPct': sf((o - prev) / prev * 100),
                }
    except Exception as e:
        print(f"  [WORKER] {sym} hata: {e}")
    return sym, None

def _fetch_stocks_parallel(symbols, label="STOCKS", retry_count=2):
    """Hisse listesini paralel cek, basarisizlari dondur"""
    ok_count = 0
    fail_list = []
    total = len(symbols)

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(_process_stock, sym, retry_count): sym for sym in symbols}
        done_count = 0
        for future in as_completed(futures):
            sym, result = future.result()
            done_count += 1
            if result:
                _cset(_stock_cache, sym, result)
                ok_count += 1
            else:
                fail_list.append(sym)
                _ctouch(_stock_cache, sym)  # Keep stale data alive
            _status['loaded'] = _status.get('loaded', 0) + 1
            if done_count % 20 == 0 or done_count == total:
                print(f"  [{label}] {done_count}/{total}, cache={len(_stock_cache)}, fail={len(fail_list)}")

    print(f"[LOADER] {label}: {ok_count} OK, {len(fail_list)} basarisiz")
    if fail_list:
        print(f"[LOADER] {label} basarisiz: {fail_list}")
    return fail_list

def _background_loop():
    print(f"[LOADER] Thread basliyor, YF={YF_OK}, workers={PARALLEL_WORKERS}")

    while True:
        t0 = time.time()
        try:
            # === FAZE 1: Endeksler (paralel) ===
            _status['phase'] = 'indices'
            _status['error'] = ''
            print(f"\n[LOADER] ====== FAZE 1: Endeksler ======")

            def _fetch_one_index(item):
                key, (tsym, name) = item
                data = _fetch_index_data(key, tsym, name)
                return key, tsym, name, data

            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                for key, tsym, name, data in executor.map(_fetch_one_index, INDEX_TICKERS.items()):
                    if data:
                        cur, prev = sf(data['close'], 4), sf(data['prev'], 4)
                        _cset(_index_cache, key, {
                            'name': name, 'value': cur,
                            'change': sf(cur - prev, 4),
                            'changePct': sf((cur - prev) / prev * 100 if prev else 0),
                            'volume': si(data.get('volume', 0)),
                        })

            print(f"[LOADER] Endeksler: {len(_index_cache)}/{len(INDEX_TICKERS)}")

            # === FAZE 2: BIST30 hisseleri (paralel) ===
            _status['phase'] = 'stocks'
            _status['total'] = len(BIST100_STOCKS)
            _status['loaded'] = 0
            print(f"\n[LOADER] ====== FAZE 2: BIST30 ({len(BIST30)} hisse) [paralel x{PARALLEL_WORKERS}] ======")

            bist30_fail = _fetch_stocks_parallel(BIST30, label="BIST30")

            # === FAZE 3: Kalan BIST100 hisseleri (paralel) ===
            with _lock:
                cached_and_valid = {s for s in BIST100_STOCKS.keys()
                                    if s in _stock_cache and time.time() - _stock_cache[s]['ts'] < CACHE_TTL}
            remaining = [s for s in BIST100_STOCKS.keys() if s not in cached_and_valid]
            phase3_fail = []
            if remaining:
                print(f"\n[LOADER] ====== FAZE 3: {len(remaining)} kalan hisse [paralel x{PARALLEL_WORKERS}] ======")
                phase3_fail = _fetch_stocks_parallel(remaining, label="BIST100")

            # === FAZE 4: Retry failed stocks (paralel) ===
            all_failed = list(set(bist30_fail + phase3_fail))
            if all_failed:
                print(f"\n[LOADER] ====== FAZE 4: {len(all_failed)} basarisiz hisse yeniden deneniyor ======")
                retry_fail = _fetch_stocks_parallel(all_failed, label="RETRY", retry_count=2)
                print(f"[LOADER] FAZE 4: {len(all_failed) - len(retry_fail)}/{len(all_failed)} kurtarildi")

            elapsed = round(time.time() - t0, 1)
            _status['phase'] = 'done'
            _status['lastRun'] = datetime.now().isoformat()
            print(f"\n[LOADER] ====== SONUC: {len(_stock_cache)} hisse, {len(_index_cache)} endeks ({elapsed}s) ======\n")

            # === FAZE 5: Tarihsel veri on-yukleme (ayri thread - ana donguyu BLOKLAMAZ) ===
            threading.Thread(target=_preload_hist_data, daemon=True).start()

            # === FAZE 5b: DB snapshot kaydet (restart sonrası hızlı preload için) ===
            _save_fn = _get_db_save_snapshot()
            if _save_fn:
                threading.Thread(target=_save_fn, daemon=True).start()

            # === FAZE 6: Otomatik alert kontrolu (cooldown destekli) ===
            _auto_check_all_alerts()

            # === FAZE 7: Otomatik alim-satim motoru ===
            try:
                _engine_fn = _get_auto_engine()
                if _engine_fn:
                    _engine_fn()
            except Exception as ate:
                print(f"[AUTO-TRADE] Engine hatasi: {ate}")

        except Exception as e:
            print(f"[LOADER] FATAL: {e}")
            traceback.print_exc()
            _status['phase'] = 'error'
            _status['error'] = str(e)

        time.sleep(300)


def _auto_check_all_alerts():
    """Background loop'ta tum aktif alert'leri kontrol et (cooldown destekli)"""
    try:
        db = get_db()
        rows = db.execute("SELECT * FROM alerts WHERE active=1 AND triggered=0").fetchall()
        if not rows:
            db.close()
            return

        now = datetime.now()
        triggered_count = 0
        for r in rows:
            # Cooldown kontrolu (ayni alert 30dk icinde tekrar tetiklenmez)
            cooldown_until = r['cooldown_until'] if 'cooldown_until' in r.keys() else None
            if cooldown_until:
                try:
                    cd_time = datetime.fromisoformat(cooldown_until)
                    if now < cd_time:
                        continue
                except:
                    pass

            stock = _cget(_stock_cache, r['symbol'])
            if not stock:
                continue

            price = stock['price']
            fire = False
            if r['condition'] == 'price_above' and price >= r['target_value']:
                fire = True
            elif r['condition'] == 'price_below' and price <= r['target_value']:
                fire = True
            elif r['condition'] == 'change_above' and stock.get('changePct', 0) >= r['target_value']:
                fire = True
            elif r['condition'] == 'change_below' and stock.get('changePct', 0) <= r['target_value']:
                fire = True

            if fire:
                cooldown_end = (now + timedelta(minutes=30)).isoformat()
                db.execute("UPDATE alerts SET triggered=1, triggered_at=?, cooldown_until=? WHERE id=?",
                           (now.isoformat(), cooldown_end, r['id']))
                triggered_count += 1

                # Telegram bildirim
                _send_telegram_alerts(r['user_id'], [{
                    'symbol': r['symbol'], 'condition': r['condition'],
                    'targetValue': r['target_value'], 'currentPrice': price,
                    'message': f"{r['symbol']} uyarisi tetiklendi: {r['condition']} {r['target_value']} (Guncel: {price})"
                }])

        if triggered_count > 0:
            db.commit()
            print(f"[ALERT-AUTO] {triggered_count} uyari tetiklendi")
        db.close()
    except Exception as e:
        print(f"[ALERT-AUTO] Hata: {e}")

def _preload_one_hist(sym):
    """Tek hisse icin tarihsel veriyi cek ve cache'le"""
    try:
        cached = _cget_hist(f"{sym}_1y")
        if cached is not None:
            return sym, True
        df = _fetch_hist_df(sym, '1y')
        if df is not None and len(df) >= 30:
            _cset(_hist_cache, f"{sym}_1y", df)
            return sym, True
    except Exception as e:
        print(f"  [HIST] {sym}: {e}")
    return sym, False

def _preload_hist_data():
    """Tum hisselerin tarihsel verisini paralel on-yukle"""
    # XU100 endeks verisini de yukle (market regime icin)
    if _cget_hist("XU100_1y") is None:
        try:
            xu_df = _fetch_isyatirim_df("XU100", 365)
            if xu_df is not None and len(xu_df) >= 30:
                _cset(_hist_cache, "XU100_1y", xu_df)
                print("[HIST-PRELOAD] XU100 endeks verisi yuklendi")
        except Exception as e:
            print(f"[HIST-PRELOAD] XU100 hata: {e}")

    symbols = list(BIST100_STOCKS.keys())
    # Sadece cache'de olmayanlari cek
    to_fetch = [s for s in symbols if _cget_hist(f"{s}_1y") is None]
    if not to_fetch:
        print(f"[HIST-PRELOAD] Tum {len(symbols)} hisse zaten cache'de")
        return

    print(f"\n[HIST-PRELOAD] ====== {len(to_fetch)} hisse icin tarihsel veri cekilecek [paralel x{PARALLEL_WORKERS}] ======")
    t0 = time.time()
    ok = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(_preload_one_hist, sym): sym for sym in to_fetch}
        done = 0
        for future in as_completed(futures):
            sym, success = future.result()
            done += 1
            if success:
                ok += 1
            if done % 30 == 0 or done == len(to_fetch):
                print(f"  [HIST-PRELOAD] {done}/{len(to_fetch)}, OK={ok}")
    elapsed = round(time.time() - t0, 1)
    print(f"[HIST-PRELOAD] Tamamlandi: {ok}/{len(to_fetch)} ({elapsed}s)\n")

def _ensure_loader():
    global _loader_started
    if _loader_started: return
    _loader_started = True
    # Cold-start: DB snapshot'ından cache'i önceden doldur (kullanıcı anında veri görür)
    _db_load_market_snapshot()
    print("[LOADER] Thread baslatiliyor (before_request)")
    t = threading.Thread(target=_background_loop, daemon=True)
    t.start()

@app.before_request
def before_req():
    """Her request'te loader'in calistigini garanti et"""
    _ensure_loader()
    _start_telegram_thread()


def _fetch_index_data(key, tsym, name):
    """Endeks verisi cek"""
    # Endeksler icin Is Yatirim: XU100 = BIST100 endeksi
    isyatirim_map = {'XU100':'XU100','XU030':'XU030','XBANK':'XBANK'}
    is_sym = isyatirim_map.get(key)

    if is_sym:
        data = _fetch_isyatirim_quick(is_sym)
        if data:
            print(f"  [ISYATIRIM-IDX] {key} OK: {data['close']}")
            return data

    # Yahoo HTTP fallback
    data = _fetch_yahoo_http(tsym)
    if data:
        print(f"  [YAHOO-IDX] {key} OK: {data['close']}")
        return data

    # yfinance fallback
    if YF_OK:
        try:
            h = yf.Ticker(tsym).history(period="5d", timeout=10)
            if h is not None and not h.empty and len(h) >= 2:
                cur, prev = float(h['Close'].iloc[-1]), float(h['Close'].iloc[-2])
                print(f"  [YF-IDX] {key} OK: {cur}")
                return {'close': cur, 'prev': prev,
                        'volume': int(h['Volume'].iloc[-1]) if 'Volume' in h.columns else 0}
        except Exception as e:
            print(f"  [YF-IDX] {key}: {e}")

    return None


# =====================================================================
# BACKGROUND LOADER (paralel - ThreadPoolExecutor)
# =====================================================================
PARALLEL_WORKERS = 12  # Paralel hisse cekme sayisi (performans iyilestirmesi)


