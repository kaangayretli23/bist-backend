"""
Ham Veri Çekme Fonksiyonları
İş Yatırım, Yahoo Finance HTTP, borsapy kaynaklarından OHLCV çekme.
data_fetcher.py'dan ayrıştırıldı (700 satır kuralı).
"""
import time
import json
import urllib.request
from datetime import datetime, timedelta

try:
    import numpy as np
except ImportError:
    np = None
try:
    import requests
    req_lib = requests
except ImportError:
    requests = None
    req_lib = None
try:
    import pandas as pd
except ImportError:
    pd = None

def _normalize_ohlcv_df(df):
    """DataFrame OHLCV normalizasyonu: NaN doldur, High>=Close, Low<=Close"""
    df['Open'] = df['Open'].fillna(df['Close'])
    df['High'] = df['High'].fillna(df['Close'])
    df['Low'] = df['Low'].fillna(df['Close'])
    df['High'] = df[['High', 'Close']].max(axis=1)
    df['Low'] = df[['Low', 'Close']].min(axis=1)
    return df

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
                # verify=False: isyatirim.com.tr cert chain bazı işletim sistemlerinde eksik; sadece fiyat verisi, credential yok
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
            df['Volume'] = pd.to_numeric(df_raw[col_map['Volume']], errors='coerce').fillna(0).astype(int).values
        else:
            df['Volume'] = 0

        df = _normalize_ohlcv_df(df)
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

        df = _normalize_ohlcv_df(df)

        if len(df) < 10: return None
        print(f"  [YAHOO-DF] {symbol}: {len(df)} bar")
        return df
    except Exception as e:
        print(f"  [YAHOO-DF] {symbol}: {e}")
        return None


# ---- BORSAPY (BIRINCIL KAYNAK) ----
try:
    import borsapy as _bp
    BP_OK = True
except ImportError:
    BP_OK = False


def _fetch_borsapy_quick(sym):
    """borsapy ile anlık fiyat verisi"""
    if not BP_OK:
        return None
    try:
        t = _bp.Ticker(sym)
        fi = t.fast_info
        cur = fi.last_price
        prev = fi.previous_close
        if cur is None or cur <= 0:
            return None
        cur = float(cur)
        prev = float(prev) if prev is not None else cur
        # Piyasa kapali oldugunda intraday degerler None olabilir
        open_ = float(fi.open) if fi.open is not None else cur
        high_ = float(fi.day_high) if fi.day_high is not None else cur
        low_  = float(fi.day_low)  if fi.day_low  is not None else cur
        vol_  = int(fi.volume)     if fi.volume   is not None else 0
        print(f"  [BORSAPY] {sym} OK: {cur}")
        return {
            'close': cur, 'prev': prev,
            'open': open_, 'high': high_, 'low': low_, 'volume': vol_,
        }
    except Exception as e:
        print(f"  [BORSAPY] {sym}: {e}")
        return None


def _fetch_borsapy_hist(sym, period='1y'):
    """borsapy ile tarihsel DataFrame"""
    if not BP_OK:
        return None
    period_map = {30: '1mo', 90: '3mo', 180: '6mo', 365: '1y', 730: '2y', 1825: '5y'}
    try:
        t = _bp.Ticker(sym)
        bp_period = period if period in ('1mo','3mo','6mo','1y','2y','5y') else '1y'
        h = t.history(period=bp_period)
        if h is None or h.empty or len(h) < 10:
            return None
        h.index = h.index.tz_localize(None) if h.index.tzinfo is None else h.index.tz_convert(None)
        h = h[['Open','High','Low','Close','Volume']].dropna(subset=['Close'])
        h = _normalize_ohlcv_df(h)
        print(f"  [BORSAPY-HIST] {sym} OK: {len(h)} bar")
        return h
    except Exception as e:
        print(f"  [BORSAPY-HIST] {sym}: {e}")
        return None


