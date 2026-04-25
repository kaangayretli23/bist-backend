"""
Signals Temel Veri ve 52 Haftalık Hesaplamaları (SİNYAL PİPELINE İÇİN)

Bu modül: İş Yatırım API birincil kaynak, yfinance fallback.
Döndürdüğü alanlar: pe, pb, marketCap, dividendYield, debtToEquity, roe, roa, profitMargin.
Sinyal skoru hesaplama için calc_fundamentals / calc_52w yanında kullanılır.

⚠ Bkz. fundamental_data.py: AYNI İSİMDE AMA FARKLI MODÜL!
fundamental_data.py → Sadece Yahoo Finance, Türkçe alan adları (fk, pd_dd, piyasa_degeri),
    haberler/telegram raporlama için. Bu iki modülü KARIŞTIRMA.
"""
import numpy as np
import time, threading
try:
    import requests as req_lib
except ImportError:
    req_lib = None
try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    yf = None
    YF_OK = False
from config import sf, si, _lock, _stock_cache, _cget, BIST100_STOCKS
from data_fetcher_raw import IS_YATIRIM_HEADERS

# Fundamental data cache (signals_core'dan taşındı)
_fundamental_cache = {}
_fundamental_cache_lock = threading.Lock()

def calc_fundamentals(hist, symbol):
    """Temel verileri mevcut fiyat/hacim verisinden hesapla"""
    try:
        c = hist['Close'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        # NaN temizligi
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        v = np.where(np.isnan(v), 0, v)
        n = len(c)
        cur = float(c[-1])

        # Ortalama gunluk hacim (son 20 gun)
        avg_vol_20 = float(np.mean(v[-20:])) if n >= 20 else float(np.mean(v))
        avg_vol_60 = float(np.mean(v[-60:])) if n >= 60 else avg_vol_20

        # Volatilite (yillik)
        if n >= 20:
            daily_returns = np.diff(c[-60:]) / c[-60:-1] if n >= 60 else np.diff(c) / c[:-1]
            volatility = sf(float(np.std(daily_returns)) * (252 ** 0.5) * 100)
        else:
            volatility = 0

        # Beta hesapla (BIST100'e gore) - basit yaklasim: volatilite bazli
        beta = sf(volatility / 25) if volatility else 1.0  # BIST100 avg vol ~25%

        # Ortalama islem hacmi (TL)
        avg_turnover = sf(cur * avg_vol_20)

        # 1 aylik, 3 aylik, 6 aylik, 1 yillik getiri
        returns = {}
        for label, days in [('1ay', 22), ('3ay', 66), ('6ay', 132), ('1yil', 252)]:
            if n > days:
                ret = sf(((cur - float(c[-days])) / float(c[-days])) * 100)
                returns[label] = ret

        # Gunluk ortalama aralik (ATR benzeri) - NaN-safe
        if n >= 14:
            daily_range = [(float(h[i]) - float(l[i])) for i in range(-14, 0) if h[i] == h[i] and l[i] == l[i]]
            avg_daily_range = sf(np.mean(daily_range)) if daily_range else 0
            avg_daily_range_pct = sf(avg_daily_range / cur * 100) if cur > 0 else 0
        else:
            avg_daily_range = 0
            avg_daily_range_pct = 0

        # 52 haftalik high/low'dan uzaklik (NaN-safe)
        if n >= 252:
            hi52 = float(np.nanmax(h[-252:]))
            lo52 = float(np.nanmin(l[-252:]))
        else:
            hi52 = float(np.nanmax(h))
            lo52 = float(np.nanmin(l))
        # NaN kontrolu
        if hi52 != hi52: hi52 = cur  # NaN ise cur kullan
        if lo52 != lo52: lo52 = cur
        dist_from_high = sf(((cur - hi52) / hi52) * 100) if hi52 else 0
        dist_from_low = sf(((cur - lo52) / lo52) * 100) if lo52 else 0

        return {
            'avgVolume20': si(avg_vol_20),
            'avgVolume60': si(avg_vol_60),
            'avgTurnover': avg_turnover,
            'volatility': volatility,
            'beta': beta,
            'returns': returns,
            'avgDailyRange': avg_daily_range,
            'avgDailyRangePct': avg_daily_range_pct,
            'distFromHigh52w': dist_from_high,
            'distFromLow52w': dist_from_low,
        }
    except Exception as e:
        print(f"  [FUND] {symbol} hata: {e}")
        return {}

def calc_52w(hist):
    """52 hafta (veya mevcut veri) high/low hesapla - NaN-safe"""
    try:
        h=hist['High'].values.astype(float)
        l=hist['Low'].values.astype(float)
        c=float(hist['Close'].iloc[-1])
        hi52=sf(float(np.nanmax(h))); lo52=sf(float(np.nanmin(l)))
        # NaN fallback
        if hi52 == 0 and c > 0: hi52 = sf(c)
        if lo52 == 0 and c > 0: lo52 = sf(c)
        rng=hi52-lo52
        pos=sf((c-lo52)/rng*100 if rng>0 else 50)
        return {'high52w':hi52,'low52w':lo52,'currentPct':pos,'range':sf(rng)}
    except Exception: return {'high52w':0,'low52w':0,'currentPct':50,'range':0}


def fetch_fundamental_data(symbol):
    """Is Yatirim'dan temel analiz verilerini cek (F/K, PD/DD vs.)"""
    try:
        # Cache kontrol (1 saat)
        with _fundamental_cache_lock:
            cached = _fundamental_cache.get(symbol)
        if cached and time.time() - cached['ts'] < 3600:
            return cached['data']

        url = f"https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/MaliTablo?hession={symbol}&doession=2024&dession=4"
        headers = IS_YATIRIM_HEADERS.copy()

        try:
            # verify=False: isyatirim cert chain uyumsuzluğu; public mali tablo endpoint'i, secret yok
            resp = req_lib.get(url, headers=headers, timeout=10, verify=False)
            if resp.status_code == 200:
                data = resp.json()
                rows = data.get('value', [])
                if rows:
                    result = _parse_fundamental_data(rows, symbol)
                    with _fundamental_cache_lock:
                        _fundamental_cache[symbol] = {'data': result, 'ts': time.time()}
                    return result
        except Exception as e:
            print(f"  [FUNDAMENTAL] {symbol} Is Yatirim hata: {e}")

        # Fallback: yfinance info
        if YF_OK:
            try:
                tkr = yf.Ticker(f"{symbol}.IS")
                info = tkr.info or {}
                result = {
                    'pe': sf(info.get('trailingPE', 0)),
                    'forwardPE': sf(info.get('forwardPE', 0)),
                    'pb': sf(info.get('priceToBook', 0)),
                    'marketCap': info.get('marketCap', 0),
                    'dividendYield': sf(info.get('dividendYield', 0) * 100) if info.get('dividendYield') else 0,
                    'debtToEquity': sf(info.get('debtToEquity', 0)),
                    'roe': sf(info.get('returnOnEquity', 0) * 100) if info.get('returnOnEquity') else 0,
                    'roa': sf(info.get('returnOnAssets', 0) * 100) if info.get('returnOnAssets') else 0,
                    'profitMargin': sf(info.get('profitMargins', 0) * 100) if info.get('profitMargins') else 0,
                    'source': 'yfinance',
                }
                with _fundamental_cache_lock:
                    _fundamental_cache[symbol] = {'data': result, 'ts': time.time()}
                return result
            except Exception as e:
                print(f"  [FUNDAMENTAL] {symbol} yfinance hata: {e}")

        return {}
    except Exception as e:
        print(f"  [FUNDAMENTAL] {symbol}: {e}")
        return {}

def _parse_fundamental_data(rows, symbol):
    """Is Yatirim mali tablo verisini parse et"""
    try:
        result = {'source': 'isyatirim'}
        for row in rows:
            key = str(row.get('KALEM', '')).upper()
            val = row.get('DEGER', 0)
            if 'NET KAR' in key or 'NET DONEM' in key:
                result['netProfit'] = sf(float(val)) if val else 0
            elif 'SATIS' in key or 'GELIR' in key:
                result['revenue'] = sf(float(val)) if val else 0
            elif 'OZKAYN' in key:
                result['equity'] = sf(float(val)) if val else 0
        return result
    except Exception:
        return {'source': 'isyatirim'}


# =====================================================================
# FEATURE 7: ENHANCED TELEGRAM/EMAIL ALERT SYSTEM
# =====================================================================

