"""
BIST Pro v3.0 - Kapsamlı Borsa Analiz Platformu Backend
=========================================================
Özellikler:
  - Dashboard: Endeks kartları, movers, piyasa genişliği
  - Hisse Detay: Teknik analiz, Fibonacci, destek/direnç, olaylar
  - Screener/Tarayıcı: Koşul tabanlı filtreleme
  - Portföy: Maliyet, PnL, risk metrikleri
  - Backtest: MA cross, breakout, mean reversion
  - Uyarılar: Cooldown, teyit kuralı
  - KAP Akışı (placeholder)

JSON Serialization: Tüm numpy/pandas tipleri native Python'a dönüştürülür.
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import yfinance as yf
import pandas as pd
import numpy as np
import traceback
from datetime import datetime, timedelta
from functools import lru_cache
import hashlib
import json
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)
app.config['TRAP_HTTP_EXCEPTIONS'] = False
app.config['PROPAGATE_EXCEPTIONS'] = False


# Global error handlers - HER ZAMAN JSON dönsün, asla HTML dönmesin
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint bulunamadı'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Sunucu hatası', 'details': str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.exception("Unhandled exception")
    return jsonify({'error': 'Unhandled error', 'details': str(e)}), 500


# =============================================================================
# CONFIGURATION
# =============================================================================

CACHE_TTL = 300  # 5 dakika cache
MAX_BATCH_SIZE = 20  # Paralel istek limiti
YF_TIMEOUT = 10  # yfinance timeout (sn)
YF_TIMEOUT = 15

# =============================================================================
# SAFE JSON SERIALIZATION HELPERS
# =============================================================================

def sf(val, decimals=2):
    """Safe float - numpy/pandas float → Python float"""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return 0.0

def si(val):
    """Safe int - numpy/pandas int → Python int"""
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0

def ss(val, default=""):
    """Safe string"""
    if val is None:
        return default
    return str(val)

def safe_dict(d):
    """Recursively convert all values in dict to JSON-safe types"""
    if isinstance(d, dict):
        return {str(k): safe_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [safe_dict(item) for item in d]
    elif isinstance(d, (np.integer,)):
        return int(d)
    elif isinstance(d, (np.floating,)):
        return round(float(d), 4)
    elif isinstance(d, np.ndarray):
        return [safe_dict(x) for x in d.tolist()]
    elif isinstance(d, pd.Timestamp):
        return d.strftime('%Y-%m-%d')
    elif isinstance(d, (np.bool_,)):
        return bool(d)
    elif isinstance(d, float) and np.isnan(d):
        return 0.0
    return d

# =============================================================================
# SIMPLE IN-MEMORY CACHE
# =============================================================================

_cache = {}
_cache_lock = threading.Lock()

def cache_get(key):
    with _cache_lock:
        item = _cache.get(key)
        if item and time.time() - item['ts'] < CACHE_TTL:
            return item['data']
    return None

def cache_set(key, data):
    with _cache_lock:
        _cache[key] = {'data': data, 'ts': time.time()}

def cache_key(*args):
    raw = '|'.join(str(a) for a in args)
    return hashlib.md5(raw.encode()).hexdigest()

# =============================================================================
# STOCK LISTS
# =============================================================================

BIST100_STOCKS = {
    'THYAO': 'Türk Hava Yolları', 'GARAN': 'Garanti BBVA',
    'ISCTR': 'İş Bankası (C)', 'AKBNK': 'Akbank',
    'TUPRS': 'Tüpraş', 'BIMAS': 'BİM',
    'SAHOL': 'Sabancı Holding', 'KCHOL': 'Koç Holding',
    'EREGL': 'Ereğli Demir Çelik', 'SISE': 'Şişe Cam',
    'PETKM': 'Petkim', 'ASELS': 'Aselsan',
    'TOASO': 'Tofaş', 'TCELL': 'Turkcell',
    'ENKAI': 'Enka İnşaat', 'KOZAL': 'Koza Altın',
    'KRDMD': 'Kardemir (D)', 'TTKOM': 'Türk Telekom',
    'ARCLK': 'Arçelik', 'SOKM': 'Şok Marketler',
    'HEKTS': 'Hektaş', 'PGSUS': 'Pegasus',
    'TAVHL': 'TAV Havalimanları', 'DOHOL': 'Doğan Holding',
    'VESTL': 'Vestel', 'MGROS': 'Migros',
    'FROTO': 'Ford Otosan', 'SASA': 'SASA Polyester',
    'EKGYO': 'Emlak Konut GYO', 'TTRAK': 'Türk Traktör',
    'AEFES': 'Anadolu Efes', 'YKBNK': 'Yapı Kredi',
    'VAKBN': 'Vakıfbank', 'HALKB': 'Halkbank',
    'TSKB': 'TSKB', 'ALARK': 'Alarko Holding',
    'GESAN': 'Giresun San.', 'OTKAR': 'Otokar',
    'CEMTS': 'Çemtaş', 'CIMSA': 'Çimsa',
    'OYAKC': 'Oyak Çimento', 'GUBRF': 'Gübre Fabrikaları',
    'KONTR': 'Kontrolmatik', 'KOZAA': 'Koza Anadolu',
    'SKBNK': 'Şekerbank', 'ISGYO': 'İş GYO',
    'ULKER': 'Ülker', 'CCOLA': 'Coca-Cola İçecek',
    'TRGYO': 'Torunlar GYO', 'ENJSA': 'Enerjisa',
    'AKSEN': 'Aksa Enerji', 'ODAS': 'Odaş Elektrik',
    'AGHOL': 'AG Anadolu Grubu', 'BRYAT': 'Borusan Yatırım',
    'DOAS': 'Doğuş Otomotiv', 'BERA': 'Bera Holding',
    'MPARK': 'MLP Sağlık', 'GENIL': 'Gen İlaç',
    'ANHYT': 'Anadolu Hayat', 'AGROT': 'Agrotech',
    'BTCIM': 'Batıçim', 'BUCIM': 'Bursa Çimento',
    'EGEEN': 'Ege Endüstri', 'ISMEN': 'İş Yatırım Menkul',
    'KLRHO': 'Kiler Holding', 'LOGO': 'Logo Yazılım',
    'MAVI': 'Mavi Giyim', 'NETAS': 'Netaş Telekom',
    'PRKME': 'Park Elektrik', 'QUAGR': 'QUA Granite',
    'SMRTG': 'Smartiks', 'TKFEN': 'Tekfen Holding',
    'TURSG': 'Türkiye Sigorta', 'VERUS': 'Verusaturk',
    'YGYO': 'Yeşil GYO', 'ZOREN': 'Zorlu Enerji'
}

BIST30_STOCKS = [
    'THYAO', 'GARAN', 'ISCTR', 'AKBNK', 'TUPRS', 'BIMAS',
    'SAHOL', 'KCHOL', 'EREGL', 'SISE', 'ASELS', 'TCELL',
    'ENKAI', 'FROTO', 'PGSUS', 'TAVHL', 'TOASO', 'ARCLK',
    'PETKM', 'TTKOM', 'KOZAL', 'YKBNK', 'VAKBN', 'HALKB',
    'EKGYO', 'SASA', 'MGROS', 'SOKM', 'DOHOL', 'VESTL'
]

BANKING_STOCKS = [
    'GARAN', 'ISCTR', 'AKBNK', 'YKBNK', 'VAKBN', 'HALKB',
    'TSKB', 'SKBNK', 'ALBRK', 'QNBFB', 'DENIZ', 'ICBCT'
]

SECTOR_MAP = {
    'bankacılık': BANKING_STOCKS,
    'havacılık': ['THYAO', 'PGSUS', 'TAVHL'],
    'otomotiv': ['TOASO', 'FROTO', 'DOAS', 'TTRAK', 'OTKAR'],
    'enerji': ['TUPRS', 'PETKM', 'AKSEN', 'ENJSA', 'ODAS', 'ZOREN'],
    'holding': ['SAHOL', 'KCHOL', 'DOHOL', 'AGHOL', 'ALARK'],
    'perakende': ['BIMAS', 'SOKM', 'MGROS', 'MAVI'],
    'teknoloji': ['ASELS', 'LOGO', 'NETAS'],
    'telekomünikasyon': ['TCELL', 'TTKOM'],
    'demir_celik': ['EREGL', 'KRDMD', 'CEMTS'],
    'cam_seramik': ['SISE', 'TRKCM'],
    'gıda': ['ULKER', 'CCOLA', 'AEFES'],
    'inşaat': ['ENKAI', 'TKFEN'],
    'gayrimenkul': ['EKGYO', 'TRGYO', 'ISGYO', 'YGYO'],
    'kimya': ['SASA', 'GUBRF', 'HEKTS'],
    'sağlık': ['MPARK', 'GENIL'],
    'sigorta': ['ANHYT', 'TURSG'],
}

# =============================================================================
# CORE DATA FETCHING (with cache)
# =============================================================================

def fetch_stock_hist(symbol, period='1y'):
    """Hisse verisi çek (cached)"""
    ck = cache_key('hist', symbol, period)
    cached = cache_get(ck)
    if cached is not None:
        return cached

    ticker_symbol = f"{symbol.upper()}.IS"
    stock = yf.Ticker(ticker_symbol)
    hist = stock.history(period=period, timeout=YF_TIMEOUT)
if hist.empty:
    time.sleep(0.4)
    hist = stock.history(period=period, timeout=YF_TIMEOUT)

    if not hist.empty:
        cache_set(ck, hist)
    return hist

def fetch_stock_info(symbol):
    """Hisse info çek (cached)"""
    ck = cache_key('info', symbol)
    cached = cache_get(ck)
    if cached is not None:
        return cached

    ticker_symbol = f"{symbol.upper()}.IS"
    stock = yf.Ticker(ticker_symbol)
    try:
        info = stock.info
        cache_set(ck, info)
        return info
    except Exception:
        return {}

def fetch_stock_quick(symbol):
    """Hızlı fiyat verisi (2 günlük) - dashboard/listing için"""
    ck = cache_key('quick', symbol)
    cached = cache_get(ck)
    if cached is not None:
        return cached
def fetch_quick_many(symbols):
    """Birden çok hisseyi sınırlı paralellik ile hızlı çek."""
    symbols = list(symbols)
    if not symbols:
        return []
    max_workers = min(MAX_BATCH_SIZE, len(symbols))
    out = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_stock_quick, s): s for s in symbols}
        for fut in as_completed(futs):
            try:
                d = fut.result()
                if d:
                    out.append(d)
            except Exception as err:
                app.logger.warning("quick fetch failed for %s: %s", futs[fut], err)
    return out

    try:
        ticker = yf.Ticker(f"{symbol.upper()}.IS")
        hist = ticker.history(period="5d", timeout=YF_TIMEOUT)
if hist.empty:
    time.sleep(0.4)
    hist = ticker.history(period="5d", timeout=YF_TIMEOUT)
def fetch_quick_batch(symbols, period="5d"):
    """
    Çoklu hisseyi tek seferde yf.download ile çekip dashboard/list için özet üretir.
    100 hisseyi tek tek çağırmak yerine 1-2 çağrı ile işi bitirir → Render'da en kritik fark.
    """
    symbols = [s.upper() for s in symbols]
    ck = cache_key('quick_batch', period, ','.join(sorted(symbols)))
    cached = cache_get(ck)
    if cached is not None:
        return cached

    tickers = [f"{s}.IS" for s in symbols]
    df = yf.download(
        " ".join(tickers),
        period=period,
        group_by="ticker",
        threads=True,
        progress=False
    )

    if df is None or df.empty:
        return []

    results = []
    multi = isinstance(df.columns, pd.MultiIndex)
    lvl0 = set(df.columns.get_level_values(0)) if multi else set()
    lvl1 = set(df.columns.get_level_values(1)) if multi else set()

    for s in symbols:
        t = f"{s}.IS"
        try:
            if multi:
                if t in lvl0:
                    h = df[t].dropna(how="all")
                elif t in lvl1:
                    h = df.xs(t, axis=1, level=1, drop_level=True).dropna(how="all")
                else:
                    continue
            else:
                # tek ticker gelmişse
                if len(symbols) == 1:
                    h = df.dropna(how="all")
                else:
                    continue

            if h.empty or len(h) < 2 or "Close" not in h.columns:
                continue

            current = sf(h["Close"].iloc[-1])
            prev = sf(h["Close"].iloc[-2])
            change = sf(current - prev)
            change_pct = sf((change / prev) * 100 if prev else 0)

            volume = si(h["Volume"].iloc[-1]) if "Volume" in h.columns else 0
            day_open = sf(h["Open"].iloc[-1]) if "Open" in h.columns else 0
            day_high = sf(h["High"].iloc[-1]) if "High" in h.columns else 0
            day_low  = sf(h["Low"].iloc[-1]) if "Low" in h.columns else 0

            gap = sf(day_open - prev)
            gap_pct = sf((gap / prev) * 100 if prev else 0)

            results.append({
                "code": s,
                "name": BIST100_STOCKS.get(s, s),
                "price": current,
                "prevClose": prev,
                "change": change,
                "changePct": change_pct,
                "volume": volume,
                "open": day_open,
                "high": day_high,
                "low": day_low,
                "gap": gap,
                "gapPct": gap_pct,
            })
        except Exception:
            continue

    cache_set(ck, results)
    return results



        if hist.empty or len(hist) < 2:
            return None

        current = sf(hist['Close'].iloc[-1])
        prev = sf(hist['Close'].iloc[-2])
        change = sf(current - prev)
        change_pct = sf((change / prev) * 100 if prev != 0 else 0)
        volume = si(hist['Volume'].iloc[-1])
        day_open = sf(hist['Open'].iloc[-1])
        day_high = sf(hist['High'].iloc[-1])
        day_low = sf(hist['Low'].iloc[-1])
        prev_close = prev

        # Gap hesaplama (bugün açılış vs dün kapanış)
        gap = sf(day_open - prev)
        gap_pct = sf((gap / prev) * 100 if prev != 0 else 0)

        result = {
            'code': ss(symbol.upper()),
            'name': ss(BIST100_STOCKS.get(symbol.upper(), symbol.upper())),
            'price': current,
            'change': change,
            'changePct': change_pct,
            'volume': volume,
            'open': day_open,
            'high': day_high,
            'low': day_low,
            'prevClose': prev_close,
            'gap': gap,
            'gapPct': gap_pct,
        }
        cache_set(ck, result)
        return result
    except Exception as e:
        print(f"Quick fetch error {symbol}: {e}")
        return None

# =============================================================================
# TECHNICAL INDICATORS (bullet-proof serialization)
# =============================================================================

def calc_rsi(closes, period=14):
    """RSI hesapla"""
    if len(closes) < period + 1:
        return {'name': 'RSI', 'value': 50.0, 'signal': 'neutral', 'explanation': 'Yetersiz veri'}

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))

    if avg_loss == 0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = sf(rsi)
    signal = 'buy' if rsi < 30 else ('sell' if rsi > 70 else 'neutral')
    zone = 'Aşırı satım' if rsi < 30 else ('Aşırı alım' if rsi > 70 else 'Nötr')

    return {
        'name': 'RSI',
        'value': rsi,
        'signal': signal,
        'explanation': f"RSI {rsi} - {zone} bölgesi"
    }

def calc_rsi_single(closes, period=14):
    """Tek RSI değeri (history için)"""
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = float(np.mean(gains[-period:]))
    avg_loss = float(np.mean(losses[-period:]))
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return sf(100.0 - (100.0 / (1.0 + rs)))

def calc_macd(closes):
    """MACD"""
    if len(closes) < 26:
        return {'name': 'MACD', 'macd': 0.0, 'signal': 0.0, 'histogram': 0.0,
                'signalType': 'neutral', 'explanation': 'Yetersiz veri'}

    series = pd.Series(closes.copy() if isinstance(closes, np.ndarray) else closes, dtype=float)
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line

    macd_val = sf(macd_line.iloc[-1])
    signal_val = sf(signal_line.iloc[-1])
    hist_val = sf(histogram.iloc[-1])

    sig = 'buy' if macd_val > signal_val else ('sell' if macd_val < signal_val else 'neutral')

    return {
        'name': 'MACD',
        'macd': macd_val,
        'signal': signal_val,
        'histogram': hist_val,
        'signalType': sig,
        'explanation': f"MACD {'yükseliş' if sig == 'buy' else 'düşüş' if sig == 'sell' else 'nötr'} sinyali"
    }

def calc_macd_history(closes):
    """MACD history (grafik için)"""
    if len(closes) < 26:
        return []
    series = pd.Series(closes.copy() if isinstance(closes, np.ndarray) else closes, dtype=float)
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return [
        {'macd': sf(macd_line.iloc[i]), 'signal': sf(signal_line.iloc[i]), 'histogram': sf(histogram.iloc[i])}
        for i in range(26, len(closes))
    ]

def calc_bollinger(closes, current_price, period=20):
    """Bollinger Bands"""
    if len(closes) < period:
        return {'name': 'Bollinger Bands', 'upper': 0.0, 'middle': 0.0, 'lower': 0.0,
                'signal': 'neutral', 'explanation': 'Yetersiz veri', 'bandwidth': 0.0}

    recent = closes[-period:]
    sma = float(np.mean(recent))
    std = float(np.std(recent))

    upper = sf(sma + 2 * std)
    middle = sf(sma)
    lower = sf(sma - 2 * std)
    bandwidth = sf((upper - lower) / middle * 100 if middle != 0 else 0)

    cp = float(current_price)
    signal = 'buy' if cp < lower else ('sell' if cp > upper else 'neutral')
    zone = 'alt bant altı (aşırı satım)' if cp < lower else ('üst bant üstü (aşırı alım)' if cp > upper else 'bantlar arası')

    return {
        'name': 'Bollinger Bands',
        'upper': upper, 'middle': middle, 'lower': lower,
        'bandwidth': bandwidth,
        'signal': signal,
        'explanation': f"Fiyat {zone}"
    }

def calc_bollinger_history(closes, period=20):
    """Bollinger history (grafik için)"""
    result = []
    for i in range(period, len(closes)):
        window = closes[i - period:i]
        sma = float(np.mean(window))
        std = float(np.std(window))
        result.append({
            'upper': sf(sma + 2 * std),
            'middle': sf(sma),
            'lower': sf(sma - 2 * std)
        })
    return result

def calc_ema(closes, current_price):
    """EMA 20/50/200"""
    result = {'name': 'EMA', 'signal': 'neutral', 'explanation': 'Yetersiz veri'}
    series = pd.Series(closes.copy() if isinstance(closes, np.ndarray) else closes, dtype=float)

    if len(closes) >= 20:
        result['ema20'] = sf(series.ewm(span=20, adjust=False).mean().iloc[-1])
    if len(closes) >= 50:
        result['ema50'] = sf(series.ewm(span=50, adjust=False).mean().iloc[-1])
    if len(closes) >= 200:
        result['ema200'] = sf(series.ewm(span=200, adjust=False).mean().iloc[-1])

    cp = float(current_price)
    ema20 = result.get('ema20', cp)
    ema50 = result.get('ema50', cp)

    if cp > ema20 > ema50:
        result['signal'] = 'buy'
        result['explanation'] = 'Güçlü yükseliş trendi (Fiyat > EMA20 > EMA50)'
    elif cp < ema20 < ema50:
        result['signal'] = 'sell'
        result['explanation'] = 'Güçlü düşüş trendi (Fiyat < EMA20 < EMA50)'
    else:
        result['signal'] = 'neutral'
        result['explanation'] = 'Kararsız trend'

    return result

def calc_ema_history(closes):
    """EMA history (grafik overlay için)"""
    series = pd.Series(closes.copy() if isinstance(closes, np.ndarray) else closes, dtype=float)
    ema20 = series.ewm(span=20, adjust=False).mean() if len(closes) >= 20 else pd.Series([])
    ema50 = series.ewm(span=50, adjust=False).mean() if len(closes) >= 50 else pd.Series([])
    ema200 = series.ewm(span=200, adjust=False).mean() if len(closes) >= 200 else pd.Series([])

    result = []
    for i in range(len(closes)):
        point = {}
        if i < len(ema20):
            point['ema20'] = sf(ema20.iloc[i])
        if i < len(ema50):
            point['ema50'] = sf(ema50.iloc[i])
        if i < len(ema200):
            point['ema200'] = sf(ema200.iloc[i])
        result.append(point)
    return result

def calc_stochastic(closes, highs, lows, period=14):
    """Stochastic Oscillator"""
    if len(closes) < period:
        return {'name': 'Stochastic', 'k': 50.0, 'd': 50.0, 'signal': 'neutral', 'explanation': 'Yetersiz veri'}

    rc = closes[-period:]
    rh = highs[-period:]
    rl = lows[-period:]

    highest = float(np.max(rh))
    lowest = float(np.min(rl))
    current = float(rc[-1])

    k = sf(((current - lowest) / (highest - lowest)) * 100 if highest != lowest else 50)

    # %D = 3-period SMA of %K (simplified: use single value)
    d = k  # Basitleştirilmiş

    signal = 'buy' if k < 20 else ('sell' if k > 80 else 'neutral')
    zone = 'aşırı satım' if k < 20 else ('aşırı alım' if k > 80 else 'nötr')

    return {
        'name': 'Stochastic',
        'k': k, 'd': d,
        'signal': signal,
        'explanation': f"Stochastic %K={k} - {zone}"
    }

def calc_atr(highs, lows, closes, period=14):
    """Average True Range"""
    if len(closes) < period + 1:
        return {'name': 'ATR', 'value': 0.0, 'signal': 'neutral', 'explanation': 'Yetersiz veri', 'pct': 0.0}

    tr_list = []
    for i in range(1, len(closes)):
        tr = max(
            float(highs[i]) - float(lows[i]),
            abs(float(highs[i]) - float(closes[i - 1])),
            abs(float(lows[i]) - float(closes[i - 1]))
        )
        tr_list.append(tr)

    atr = sf(np.mean(tr_list[-period:]))
    current = float(closes[-1])
    atr_pct = sf((atr / current) * 100 if current != 0 else 0)

    volatility = 'yüksek' if atr_pct > 3 else ('düşük' if atr_pct < 1 else 'orta')

    return {
        'name': 'ATR',
        'value': atr,
        'pct': atr_pct,
        'signal': 'neutral',
        'explanation': f"Volatilite: {volatility} (ATR%: {atr_pct})"
    }

def calc_adx(highs, lows, closes, period=14):
    """ADX - Average Directional Index (tam hesaplama)"""
    n = len(closes)
    if n < period + 1:
        return {'name': 'ADX', 'value': 25.0, 'plusDI': 0.0, 'minusDI': 0.0,
                'signal': 'neutral', 'explanation': 'Yetersiz veri'}

    # True Range
    tr_list = []
    plus_dm = []
    minus_dm = []
    for i in range(1, n):
        h = float(highs[i])
        l = float(lows[i])
        ph = float(highs[i - 1])
        pl = float(lows[i - 1])
        pc = float(closes[i - 1])

        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))

        up_move = h - ph
        down_move = pl - l

        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)

    # Smoothed
    if len(tr_list) < period:
        return {'name': 'ADX', 'value': 25.0, 'plusDI': 0.0, 'minusDI': 0.0,
                'signal': 'neutral', 'explanation': 'Yetersiz veri'}

    atr_s = float(np.mean(tr_list[:period]))
    plus_dm_s = float(np.mean(plus_dm[:period]))
    minus_dm_s = float(np.mean(minus_dm[:period]))

    for i in range(period, len(tr_list)):
        atr_s = (atr_s * (period - 1) + tr_list[i]) / period
        plus_dm_s = (plus_dm_s * (period - 1) + plus_dm[i]) / period
        minus_dm_s = (minus_dm_s * (period - 1) + minus_dm[i]) / period

    plus_di = sf((plus_dm_s / atr_s) * 100 if atr_s != 0 else 0)
    minus_di = sf((minus_dm_s / atr_s) * 100 if atr_s != 0 else 0)

    di_sum = plus_di + minus_di
    dx = abs(plus_di - minus_di) / di_sum * 100 if di_sum != 0 else 0
    adx_val = sf(dx)

    trend = 'güçlü' if adx_val > 25 else 'zayıf'
    direction = 'yükseliş' if plus_di > minus_di else 'düşüş'
    signal = 'buy' if plus_di > minus_di and adx_val > 25 else \
        ('sell' if minus_di > plus_di and adx_val > 25 else 'neutral')

    return {
        'name': 'ADX',
        'value': adx_val,
        'plusDI': plus_di,
        'minusDI': minus_di,
        'signal': signal,
        'explanation': f"Trend gücü: {trend}, Yön: {direction} (ADX={adx_val})"
    }

def calc_cci(highs, lows, closes, current_price, period=20):
    """Commodity Channel Index"""
    if len(closes) < period:
        return {'name': 'CCI', 'value': 0.0, 'signal': 'neutral', 'explanation': 'Yetersiz veri'}

    tp = (np.array(highs, dtype=float) + np.array(lows, dtype=float) + np.array(closes, dtype=float)) / 3
    recent = tp[-period:]
    sma = float(np.mean(recent))
    mean_dev = float(np.mean(np.abs(recent - sma)))

    cci = sf((float(tp[-1]) - sma) / (0.015 * mean_dev) if mean_dev != 0 else 0)
    signal = 'buy' if cci < -100 else ('sell' if cci > 100 else 'neutral')
    zone = 'aşırı satım' if cci < -100 else ('aşırı alım' if cci > 100 else 'nötr')

    return {
        'name': 'CCI',
        'value': cci,
        'signal': signal,
        'explanation': f"CCI={cci} - {zone}"
    }

def calc_williams_r(highs, lows, closes, current_price, period=14):
    """Williams %R"""
    if len(closes) < period:
        return {'name': 'Williams %R', 'value': -50.0, 'signal': 'neutral', 'explanation': 'Yetersiz veri'}

    rh = highs[-period:]
    rl = lows[-period:]
    highest = float(np.max(rh))
    lowest = float(np.min(rl))
    cp = float(current_price)

    wr = sf(((highest - cp) / (highest - lowest)) * -100 if highest != lowest else -50)
    signal = 'buy' if wr < -80 else ('sell' if wr > -20 else 'neutral')

    return {
        'name': 'Williams %R',
        'value': wr,
        'signal': signal,
        'explanation': f"W%R={wr} - {'aşırı satım' if wr < -80 else 'aşırı alım' if wr > -20 else 'nötr'}"
    }

def calc_obv(closes, volumes):
    """On Balance Volume"""
    if len(closes) < 10:
        return {'name': 'OBV', 'value': 0, 'trend': 'neutral', 'signal': 'neutral', 'explanation': 'Yetersiz veri'}

    obv = 0
    obv_values = [0]
    for i in range(1, len(closes)):
        if float(closes[i]) > float(closes[i - 1]):
            obv += int(volumes[i])
        elif float(closes[i]) < float(closes[i - 1]):
            obv -= int(volumes[i])
        obv_values.append(obv)

    lookback = min(10, len(obv_values) - 1)
    trend = 'up' if obv_values[-1] > obv_values[-lookback] else 'down'
    signal = 'buy' if trend == 'up' else 'sell'

    return {
        'name': 'OBV',
        'value': si(abs(obv_values[-1])),
        'trend': trend,
        'signal': signal,
        'explanation': f"Hacim trendi: {'yükseliş' if trend == 'up' else 'düşüş'}"
    }

def calc_volume_profile(hist, bins=20):
    """Volume Profile - fiyat seviyelerine göre hacim dağılımı"""
    try:
        closes = hist['Close'].values.astype(float)
        volumes = hist['Volume'].values.astype(float)

        price_min = float(np.min(closes))
        price_max = float(np.max(closes))

        if price_min == price_max:
            return {'levels': [], 'poc': sf(price_min)}

        edges = np.linspace(price_min, price_max, bins + 1)
        levels = []
        max_vol = 0
        poc_price = price_min

        for i in range(bins):
            mask = (closes >= edges[i]) & (closes < edges[i + 1])
            vol = float(np.sum(volumes[mask]))
            mid = float((edges[i] + edges[i + 1]) / 2)
            levels.append({'price': sf(mid), 'volume': si(vol)})
            if vol > max_vol:
                max_vol = vol
                poc_price = mid

        return {
            'levels': levels,
            'poc': sf(poc_price),  # Point of Control
            'explanation': f"En yoğun işlem hacmi: {sf(poc_price)} TL seviyesi"
        }
    except Exception as e:
        print(f"Volume profile error: {e}")
        return {'levels': [], 'poc': 0.0}

# =============================================================================
# SUPPORT/RESISTANCE & FIBONACCI
# =============================================================================

def calc_support_resistance(hist, atr_filter=True):
    """Destek/Direnç - swing high/low + ATR filtresi"""
    try:
        closes = hist['Close'].values.astype(float)
        highs = hist['High'].values.astype(float)
        lows = hist['Low'].values.astype(float)

        n = min(90, len(closes))
        rc = closes[-n:]
        rh = highs[-n:]
        rl = lows[-n:]

        # ATR hesapla (filtre için)
        atr_val = 0
        if atr_filter and len(closes) > 14:
            tr_list = []
            for i in range(1, len(closes)):
                tr_list.append(max(
                    float(highs[i] - lows[i]),
                    abs(float(highs[i] - closes[i - 1])),
                    abs(float(lows[i] - closes[i - 1]))
                ))
            atr_val = float(np.mean(tr_list[-14:])) * 0.5  # Yarım ATR filtre

        supports = []
        resistances = []

        # Swing high/low tespit (5-bar pivot)
        for i in range(2, n - 2):
            # Direnç: swing high
            if rh[i] > rh[i - 1] and rh[i] > rh[i - 2] and rh[i] > rh[i + 1] and rh[i] > rh[i + 2]:
                resistances.append(float(rh[i]))
            # Destek: swing low
            if rl[i] < rl[i - 1] and rl[i] < rl[i - 2] and rl[i] < rl[i + 1] and rl[i] < rl[i + 2]:
                supports.append(float(rl[i]))

        # ATR filtresi: birbirine çok yakın seviyeleri birleştir
        def cluster_levels(levels, min_distance):
            if not levels:
                return []
            levels = sorted(levels)
            clustered = [levels[0]]
            for lv in levels[1:]:
                if lv - clustered[-1] > min_distance:
                    clustered.append(lv)
                else:
                    clustered[-1] = (clustered[-1] + lv) / 2
            return clustered

        if atr_val > 0:
            supports = cluster_levels(supports, atr_val)
            resistances = cluster_levels(resistances, atr_val)

        current = float(closes[-1])
        nearest_sup = sorted([s for s in supports if s < current], reverse=True)[:3]
        nearest_res = sorted([r for r in resistances if r > current])[:3]

        return {
            'supports': [sf(s) for s in nearest_sup],
            'resistances': [sf(r) for r in nearest_res],
            'allSupports': [sf(s) for s in sorted(supports, reverse=True)],
            'allResistances': [sf(r) for r in sorted(resistances)],
            'current': sf(current),
            'atrFilter': sf(atr_val),
            'explanation': f"En yakın destek: {sf(nearest_sup[0]) if nearest_sup else 'Yok'}, "
                           f"En yakın direnç: {sf(nearest_res[0]) if nearest_res else 'Yok'}"
        }
    except Exception as e:
        print(f"S/R error: {e}")
        return {'supports': [], 'resistances': [], 'current': 0.0, 'explanation': 'Hesaplanamadı'}

def calc_fibonacci(hist):
    """Fibonacci Retracement"""
    try:
        closes = hist['Close'].values.astype(float)
        n = min(90, len(closes))
        recent = closes[-n:]

        high = float(np.max(recent))
        low = float(np.min(recent))
        diff = high - low

        levels = {
            '0.0': sf(high),
            '23.6': sf(high - diff * 0.236),
            '38.2': sf(high - diff * 0.382),
            '50.0': sf(high - diff * 0.5),
            '61.8': sf(high - diff * 0.618),
            '78.6': sf(high - diff * 0.786),
            '100.0': sf(low)
        }

        current = float(closes[-1])
        zone = "Belirsiz"
        level_keys = list(levels.keys())
        level_vals = list(levels.values())

        for i in range(len(level_vals) - 1):
            if current <= level_vals[i] and current >= level_vals[i + 1]:
                zone = f"{level_keys[i]}% - {level_keys[i + 1]}% arası"
                break

        return {
            'levels': levels,
            'high': sf(high),
            'low': sf(low),
            'currentZone': zone,
            'explanation': f"Fiyat {zone} Fibonacci bölgesinde"
        }
    except Exception as e:
        print(f"Fibonacci error: {e}")
        return {'levels': {}, 'explanation': 'Hesaplanamadı'}

# =============================================================================
# AGGREGATED INDICATORS
# =============================================================================

def calc_all_indicators(hist, current_price):
    """Tüm teknik indikatörleri hesapla"""
    closes = hist['Close'].values.astype(float)
    highs = hist['High'].values.astype(float)
    lows = hist['Low'].values.astype(float)
    volumes = hist['Volume'].values.astype(float)
    cp = float(current_price)

    # RSI history
    rsi_history = []
    for i in range(len(closes)):
        if i >= 14:
            rsi_val = calc_rsi_single(closes[:i + 1])
            if rsi_val is not None:
                rsi_history.append({
                    'date': hist.index[i].strftime('%Y-%m-%d'),
                    'value': rsi_val
                })

    indicators = {
        'rsi': calc_rsi(closes),
        'rsiHistory': rsi_history,
        'macd': calc_macd(closes),
        'macdHistory': calc_macd_history(closes),
        'bollinger': calc_bollinger(closes, cp),
        'bollingerHistory': calc_bollinger_history(closes),
        'stochastic': calc_stochastic(closes, highs, lows),
        'ema': calc_ema(closes, cp),
        'emaHistory': calc_ema_history(closes),
        'atr': calc_atr(highs, lows, closes),
        'adx': calc_adx(highs, lows, closes),
        'cci': calc_cci(highs, lows, closes, cp),
        'williamsr': calc_williams_r(highs, lows, closes, cp),
        'obv': calc_obv(closes, volumes),
        'volumeProfile': calc_volume_profile(hist),
    }

    # Genel sinyal özeti
    signals = [v.get('signal', 'neutral') for v in indicators.values() if isinstance(v, dict) and 'signal' in v]
    buy_count = signals.count('buy')
    sell_count = signals.count('sell')
    total = len(signals)

    if buy_count > sell_count and buy_count >= total * 0.4:
        overall = 'buy'
        strength = sf(buy_count / total * 100)
    elif sell_count > buy_count and sell_count >= total * 0.4:
        overall = 'sell'
        strength = sf(sell_count / total * 100)
    else:
        overall = 'neutral'
        strength = sf(50)

    indicators['summary'] = {
        'overall': overall,
        'strength': strength,
        'buySignals': buy_count,
        'sellSignals': sell_count,
        'neutralSignals': total - buy_count - sell_count,
        'explanation': f"Toplam {total} indikatörden {buy_count} AL, {sell_count} SAT, {total - buy_count - sell_count} NÖTR sinyal"
    }

    return indicators

# =============================================================================
# CHART DATA
# =============================================================================

def prepare_chart_data(hist, max_points=None):
    """Candlestick + hacim grafik verisi"""
    try:
        data = hist if max_points is None else hist.tail(max_points)

        candlestick = []
        for date, row in data.iterrows():
            candlestick.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': sf(row['Open']),
                'high': sf(row['High']),
                'low': sf(row['Low']),
                'close': sf(row['Close']),
                'volume': si(row['Volume']),
            })

        return {
            'candlestick': candlestick,
            'dates': [c['date'] for c in candlestick],
            'prices': [c['close'] for c in candlestick],
            'volumes': [c['volume'] for c in candlestick],
            'dataPoints': len(candlestick),
        }
    except Exception as e:
        print(f"Chart data error: {e}")
        return {'candlestick': [], 'dates': [], 'prices': [], 'volumes': [], 'dataPoints': 0}

# =============================================================================
# EVENTS (Temettü, Bölünme, Bedelli/Bedelsiz)
# =============================================================================

def fetch_stock_events(symbol):
    """Hisse olayları: temettü, bölünme vb."""
    try:
        ticker = yf.Ticker(f"{symbol.upper()}.IS")

        # Temettüler
        dividends = []
        try:
            div_data = ticker.dividends
            if div_data is not None and len(div_data) > 0:
                for date, val in div_data.items():
                    dividends.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'amount': sf(val),
                        'type': 'dividend'
                    })
        except Exception:
            pass

        # Bölünmeler
        splits = []
        try:
            split_data = ticker.splits
            if split_data is not None and len(split_data) > 0:
                for date, val in split_data.items():
                    splits.append({
                        'date': date.strftime('%Y-%m-%d'),
                        'ratio': sf(val),
                        'type': 'split'
                    })
        except Exception:
            pass

        # Takvim (gelecek olaylar)
        calendar = {}
        try:
            cal = ticker.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    calendar = {str(k): str(v) for k, v in cal.items()}
                elif isinstance(cal, pd.DataFrame):
                    calendar = {str(col): str(cal[col].iloc[0]) for col in cal.columns}
        except Exception:
            pass

        return {
            'dividends': dividends[-20:],  # Son 20
            'splits': splits,
            'calendar': calendar,
            'totalDividends': len(dividends),
            'totalSplits': len(splits),
        }
    except Exception as e:
        print(f"Events error {symbol}: {e}")
        return {'dividends': [], 'splits': [], 'calendar': {}}

# =============================================================================
# IN-MEMORY STORAGE (Portföy, Uyarılar, Watchlist)
# =============================================================================

portfolio_store = {}  # {user_id: [{symbol, quantity, avgCost, date}]}
alert_store = []      # [{id, symbol, condition, threshold, email, cooldown, lastTriggered, active}]
watchlist_store = {}   # {user_id: [symbols]}

# =============================================================================
# API ROUTES
# =============================================================================

# ─────────────────────────────── DASHBOARD ───────────────────────────────

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    """Ana sayfa dashboard - endeksler, movers, piyasa genişliği"""
    try:
        ck = cache_key('dashboard_v1')
        cached = cache_get(ck)
        if cached is not None:
            return jsonify(cached)

all_stocks = fetch_quick_batch(BIST100_STOCKS.keys(), period="5d")

        if not all_stocks:
            payload = {
                'success': True,
                'stockCount': 0,
                'movers': {'topGainers': [], 'topLosers': [], 'volumeLeaders': [], 'gapStocks': []},
                'marketBreadth': {'advancing': 0, 'declining': 0, 'unchanged': 0, 'advDecRatio': 0},
                'allStocks': [],
                'message': 'Piyasa verisi henüz alınamadı. Lütfen birkaç saniye bekleyip sayfayı yenileyin.'
            }
            cache_set(ck, payload)
            return jsonify(payload)

        sorted_by_change = sorted(all_stocks, key=lambda x: x['changePct'], reverse=True)
        top_gainers = sorted_by_change[:5]
        top_losers = sorted_by_change[-5:][::-1]
        volume_leaders = sorted(all_stocks, key=lambda x: x['volume'], reverse=True)[:5]
        gap_stocks = sorted(all_stocks, key=lambda x: abs(x['gapPct']), reverse=True)[:5]

        advancing = sum(1 for s in all_stocks if s['changePct'] > 0)
        declining = sum(1 for s in all_stocks if s['changePct'] < 0)
        unchanged = len(all_stocks) - advancing - declining

        payload = safe_dict({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'stockCount': len(all_stocks),
            'movers': {
                'topGainers': top_gainers,
                'topLosers': top_losers,
                'volumeLeaders': volume_leaders,
                'gapStocks': gap_stocks,
            },
            'marketBreadth': {
                'advancing': advancing,
                'declining': declining,
                'unchanged': unchanged,
                'advDecRatio': sf(advancing / declining if declining > 0 else advancing),
            },
            'allStocks': sorted_by_change,
        })
        cache_set(ck, payload)
        return jsonify(payload)

    except Exception as e:
        app.logger.exception("Dashboard error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/indices', methods=['GET'])
def get_indices():
    """Endeks kartları - BIST100, BIST30, Bankacılık"""
    try:
        indices = {}

        # XU100 (BIST100 endeksi)
        try:
            xu100 = yf.Ticker("XU100.IS")
            h = xu100.history(period="5d")
            if not h.empty and len(h) >= 2:
                indices['XU100'] = {
                    'name': 'BIST 100',
                    'value': sf(h['Close'].iloc[-1]),
                    'change': sf(h['Close'].iloc[-1] - h['Close'].iloc[-2]),
                    'changePct': sf((h['Close'].iloc[-1] - h['Close'].iloc[-2]) / h['Close'].iloc[-2] * 100),
                    'high': sf(h['High'].iloc[-1]),
                    'low': sf(h['Low'].iloc[-1]),
                    'volume': si(h['Volume'].iloc[-1]),
                }
        except Exception as e:
            print(f"XU100 error: {e}")

        # XU030 (BIST30 endeksi)
        try:
            xu030 = yf.Ticker("XU030.IS")
            h = xu030.history(period="5d")
            if not h.empty and len(h) >= 2:
                indices['XU030'] = {
                    'name': 'BIST 30',
                    'value': sf(h['Close'].iloc[-1]),
                    'change': sf(h['Close'].iloc[-1] - h['Close'].iloc[-2]),
                    'changePct': sf((h['Close'].iloc[-1] - h['Close'].iloc[-2]) / h['Close'].iloc[-2] * 100),
                    'high': sf(h['High'].iloc[-1]),
                    'low': sf(h['Low'].iloc[-1]),
                    'volume': si(h['Volume'].iloc[-1]),
                }
        except Exception as e:
            print(f"XU030 error: {e}")

        # XBANK (Bankacılık endeksi)
        try:
            xbank = yf.Ticker("XBANK.IS")
            h = xbank.history(period="5d")
            if not h.empty and len(h) >= 2:
                indices['XBANK'] = {
                    'name': 'BIST Bankacılık',
                    'value': sf(h['Close'].iloc[-1]),
                    'change': sf(h['Close'].iloc[-1] - h['Close'].iloc[-2]),
                    'changePct': sf((h['Close'].iloc[-1] - h['Close'].iloc[-2]) / h['Close'].iloc[-2] * 100),
                    'high': sf(h['High'].iloc[-1]),
                    'low': sf(h['Low'].iloc[-1]),
                    'volume': si(h['Volume'].iloc[-1]),
                }
        except Exception as e:
            print(f"XBANK error: {e}")

        # USD/TRY
        try:
            usdtry = yf.Ticker("USDTRY=X")
            h = usdtry.history(period="5d")
            if not h.empty and len(h) >= 2:
                indices['USDTRY'] = {
                    'name': 'USD/TRY',
                    'value': sf(h['Close'].iloc[-1], 4),
                    'change': sf(h['Close'].iloc[-1] - h['Close'].iloc[-2], 4),
                    'changePct': sf((h['Close'].iloc[-1] - h['Close'].iloc[-2]) / h['Close'].iloc[-2] * 100),
                }
        except Exception as e:
            print(f"USDTRY error: {e}")

        # Altın (ONS)
        try:
            gold = yf.Ticker("GC=F")
            h = gold.history(period="5d")
            if not h.empty and len(h) >= 2:
                indices['GOLD'] = {
                    'name': 'Altın (ONS)',
                    'value': sf(h['Close'].iloc[-1]),
                    'change': sf(h['Close'].iloc[-1] - h['Close'].iloc[-2]),
                    'changePct': sf((h['Close'].iloc[-1] - h['Close'].iloc[-2]) / h['Close'].iloc[-2] * 100),
                }
        except Exception as e:
            print(f"Gold error: {e}")

        return jsonify(safe_dict({'success': True, 'indices': indices}))
    except Exception as e:
        print(f"Indices error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/market-breadth', methods=['GET'])
def get_market_breadth():
    """Piyasa genişliği: yükselen/düşen, yeni zirve/dip"""
    try:
        all_data = []
        for code in BIST100_STOCKS:
            try:
                ticker = yf.Ticker(f"{code}.IS")
                hist = ticker.history(period="3mo")
                if hist.empty or len(hist) < 2:
                    continue

                current = float(hist['Close'].iloc[-1])
                prev = float(hist['Close'].iloc[-2])
                high_52w = float(hist['High'].max())
                low_52w = float(hist['Low'].min())
                change_pct = (current - prev) / prev * 100

                all_data.append({
                    'code': code,
                    'changePct': sf(change_pct),
                    'nearHigh': current >= high_52w * 0.95,  # %5 yakınında
                    'nearLow': current <= low_52w * 1.05,
                    'atHigh': current >= high_52w * 0.99,
                    'atLow': current <= low_52w * 1.01,
                })
            except Exception:
                continue

        advancing = sum(1 for d in all_data if d['changePct'] > 0)
        declining = sum(1 for d in all_data if d['changePct'] < 0)
        new_highs = [d['code'] for d in all_data if d['atHigh']]
        new_lows = [d['code'] for d in all_data if d['atLow']]
        near_highs = [d['code'] for d in all_data if d['nearHigh']]
        near_lows = [d['code'] for d in all_data if d['nearLow']]

        return jsonify(safe_dict({
            'success': True,
            'advancing': advancing,
            'declining': declining,
            'unchanged': len(all_data) - advancing - declining,
            'newHighs': new_highs,
            'newLows': new_lows,
            'nearHighs': near_highs,
            'nearLows': near_lows,
            'advDecRatio': sf(advancing / declining if declining > 0 else float(advancing)),
            'total': len(all_data),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────── BIST100 LIST ───────────────────────────

@app.route('/api/bist100', methods=['GET'])
def get_bist100_list():
    """BIST100 hisse listesi"""
    try:
        sector_filter = request.args.get('sector', None)
        sort_by = request.args.get('sort', 'code')
        order = request.args.get('order', 'asc')

        ck = cache_key('bist100_v1', sector_filter or '', sort_by, order)
        cached = cache_get(ck)
        if cached is not None:
            return jsonify(cached)

        stocks_to_fetch = BIST100_STOCKS.keys()
        if sector_filter and sector_filter in SECTOR_MAP:
            stocks_to_fetch = [s for s in SECTOR_MAP[sector_filter] if s in BIST100_STOCKS]

stocks = fetch_quick_batch(BIST30_STOCKS, period="5d")

        reverse = (order == 'desc')
        if sort_by == 'change':
            stocks.sort(key=lambda x: x.get('changePct', 0), reverse=reverse)
        elif sort_by == 'volume':
            stocks.sort(key=lambda x: x.get('volume', 0), reverse=reverse)
        elif sort_by == 'price':
            stocks.sort(key=lambda x: x.get('price', 0), reverse=reverse)
        else:
            stocks.sort(key=lambda x: x.get('code', ''), reverse=reverse)

        payload = safe_dict({
            'success': True,
            'stocks': stocks,
            'count': len(stocks),
            'sectors': list(SECTOR_MAP.keys()),
        })
        cache_set(ck, payload)
        return jsonify(payload)

    except Exception as e:
        app.logger.exception("BIST100 list error")
        return jsonify({'error': str(e)}), 500

# ─────────────────────────────── STOCK DETAIL ────────────────────────────

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    """Hisse detay - fiyat, teknik analiz, grafik, fibonacci, S/R"""
    try:
        period = request.args.get('period', '1y')
        symbol = symbol.upper()

        print(f"\n📊 Hisse analizi: {symbol} (Period: {period})")

        hist = fetch_stock_hist(symbol, period)
        if hist.empty:
            return jsonify({'error': f'Hisse bulunamadı: {symbol}'}), 404

        print(f"✅ {len(hist)} günlük veri alındı")

        # Info
        info = fetch_stock_info(symbol)
        stock_name = info.get('longName', info.get('shortName', BIST100_STOCKS.get(symbol, symbol)))
        currency = info.get('currency', 'TRY')
        market_cap = info.get('marketCap', 0)
        pe_ratio = info.get('trailingPE', None)
        pb_ratio = info.get('priceToBook', None)
        div_yield = info.get('dividendYield', None)
        beta = info.get('beta', None)
        avg_volume = info.get('averageVolume', 0)
        week52_high = info.get('fiftyTwoWeekHigh', None)
        week52_low = info.get('fiftyTwoWeekLow', None)

        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0

        # Tüm hesaplamalar
        indicators = calc_all_indicators(hist, current_price)
        chart_data = prepare_chart_data(hist)
        fibonacci = calc_fibonacci(hist)
        support_resistance = calc_support_resistance(hist)

        result = {
            'success': True,
            'code': ss(symbol),
            'name': ss(stock_name),
            'price': sf(current_price),
            'change': sf(change),
            'changePercent': sf(change_pct),
            'volume': si(hist['Volume'].iloc[-1]),
            'avgVolume': si(avg_volume),
            'dayHigh': sf(hist['High'].iloc[-1]),
            'dayLow': sf(hist['Low'].iloc[-1]),
            'dayOpen': sf(hist['Open'].iloc[-1]),
            'prevClose': sf(prev_close),
            'currency': ss(currency),
            'period': ss(period),
            'dataPoints': len(hist),
            # Fundamentals
            'marketCap': si(market_cap),
            'peRatio': sf(pe_ratio) if pe_ratio else None,
            'pbRatio': sf(pb_ratio) if pb_ratio else None,
            'dividendYield': sf(div_yield * 100) if div_yield else None,
            'beta': sf(beta) if beta else None,
            'week52High': sf(week52_high) if week52_high else None,
            'week52Low': sf(week52_low) if week52_low else None,
            # Teknik
            'indicators': indicators,
            'chartData': chart_data,
            'fibonacci': fibonacci,
            'supportResistance': support_resistance,
        }

        return jsonify(safe_dict(result))

    except Exception as e:
        error_details = traceback.format_exc()
        print(f"\n❌ HATA:\n{error_details}")
        return jsonify({'error': str(e), 'details': error_details}), 500


@app.route('/api/stock/<symbol>/events', methods=['GET'])
def get_stock_events(symbol):
    """Hisse olayları: temettü, bölünme, bilanço tarihleri"""
    try:
        events = fetch_stock_events(symbol.upper())
        return jsonify(safe_dict({'success': True, 'symbol': symbol.upper(), 'events': events}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/<symbol>/kap', methods=['GET'])
def get_stock_kap(symbol):
    """KAP bildirimleri (placeholder - gerçek KAP API entegrasyonu yapılabilir)"""
    try:
        # KAP'ın resmi API'si yok, scraping veya 3. parti servis gerekir
        # Placeholder olarak bilgi dönüyoruz
        return jsonify(safe_dict({
            'success': True,
            'symbol': symbol.upper(),
            'message': 'KAP entegrasyonu aktif değil. KAP verisi için https://www.kap.org.tr adresini ziyaret edin.',
            'filters': ['finansal_rapor', 'ozel_durum', 'pay_alim_satim', 'genel_kurul', 'temettü'],
            'notifications': [],
            'note': 'KAP API entegrasyonu için ayrı bir scraping servisi gereklidir.'
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────── COMPARE ─────────────────────────────────

@app.route('/api/compare', methods=['POST'])
def compare_stocks():
    """İki hisse karşılaştır"""
    try:
        data = request.json
        symbols = data.get('symbols', [])

        if len(symbols) < 2:
            return jsonify({'error': 'En az 2 hisse kodu gerekli'}), 400

        period = data.get('period', '6mo')
        results = []

        for sym in symbols[:5]:  # Max 5 hisse
            hist = fetch_stock_hist(sym, period)
            if hist.empty:
                continue

            current = float(hist['Close'].iloc[-1])
            first = float(hist['Close'].iloc[0])
            perf = (current - first) / first * 100

            closes = hist['Close'].values.astype(float)
            volatility = float(np.std(np.diff(closes) / closes[:-1]) * np.sqrt(252) * 100) if len(closes) > 1 else 0

            results.append({
                'code': ss(sym.upper()),
                'name': ss(BIST100_STOCKS.get(sym.upper(), sym.upper())),
                'price': sf(current),
                'performance': sf(perf),
                'volatility': sf(volatility),
                'rsi': calc_rsi(closes)['value'],
                'volume': si(hist['Volume'].iloc[-1]),
            })

        return jsonify(safe_dict({'success': True, 'comparison': results, 'period': period}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────── SCREENER ────────────────────────────────

@app.route('/api/screener', methods=['POST'])
def stock_screener():
    """
    Tarayıcı - koşul tabanlı filtreleme
    Body:
    {
        "conditions": [
            {"indicator": "rsi", "operator": "<", "value": 30},
            {"indicator": "ema_cross", "operator": "==", "value": "bullish"},
            {"indicator": "volume_ratio", "operator": ">", "value": 2},
            {"indicator": "price_above_ema20", "operator": "==", "value": true},
            {"indicator": "breakout", "operator": "==", "value": true}
        ],
        "sector": "bankacılık",  // optional
        "sort": "rsi",
        "limit": 20
    }
    """
    try:
        data = request.json or {}
        conditions = data.get('conditions', [])
        sector = data.get('sector', None)
        sort_by = data.get('sort', 'code')
        limit = min(data.get('limit', 50), 100)

        # Hangi hisseleri tara
        symbols = list(BIST100_STOCKS.keys())
        if sector and sector in SECTOR_MAP:
            symbols = [s for s in SECTOR_MAP[sector] if s in BIST100_STOCKS]

        matches = []

        for sym in symbols:
            try:
                hist = fetch_stock_hist(sym, '6mo')
                if hist.empty or len(hist) < 30:
                    continue

                closes = hist['Close'].values.astype(float)
                highs = hist['High'].values.astype(float)
                lows = hist['Low'].values.astype(float)
                volumes = hist['Volume'].values.astype(float)
                current = float(closes[-1])

                # Metrikleri hesapla
                rsi_data = calc_rsi(closes)
                rsi_val = rsi_data['value']

                ema_data = calc_ema(closes, current)
                ema20 = ema_data.get('ema20', current)
                ema50 = ema_data.get('ema50', current)

                macd_data = calc_macd(closes)

                # Hacim oranı (son gün / 20 gün ort)
                avg_vol_20 = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
                vol_ratio = float(volumes[-1]) / avg_vol_20 if avg_vol_20 > 0 else 1.0

                # Breakout: fiyat son 20 günlük yüksek üstünde
                high_20 = float(np.max(highs[-21:-1])) if len(highs) > 21 else float(np.max(highs[:-1]))
                is_breakout = current > high_20

                # Bollinger
                boll_data = calc_bollinger(closes, current)

                # ATR
                atr_data = calc_atr(highs, lows, closes)

                # Tüm koşulları kontrol et
                stock_metrics = {
                    'rsi': rsi_val,
                    'ema20': ema20,
                    'ema50': ema50,
                    'ema_cross': 'bullish' if ema20 > ema50 else 'bearish',
                    'price_above_ema20': current > ema20,
                    'price_above_ema50': current > ema50,
                    'volume_ratio': sf(vol_ratio),
                    'breakout': is_breakout,
                    'macd_signal': macd_data['signalType'],
                    'bollinger_signal': boll_data['signal'],
                    'atr_pct': atr_data['pct'],
                }

                passed = True
                for cond in conditions:
                    indicator = cond.get('indicator', '')
                    operator = cond.get('operator', '==')
                    target = cond.get('value')

                    metric_val = stock_metrics.get(indicator)
                    if metric_val is None:
                        continue

                    if operator == '<' and not (float(metric_val) < float(target)):
                        passed = False; break
                    elif operator == '>' and not (float(metric_val) > float(target)):
                        passed = False; break
                    elif operator == '<=' and not (float(metric_val) <= float(target)):
                        passed = False; break
                    elif operator == '>=' and not (float(metric_val) >= float(target)):
                        passed = False; break
                    elif operator == '==' and str(metric_val) != str(target):
                        passed = False; break
                    elif operator == '!=' and str(metric_val) == str(target):
                        passed = False; break

                if passed:
                    matches.append({
                        'code': ss(sym),
                        'name': ss(BIST100_STOCKS.get(sym, sym)),
                        'price': sf(current),
                        'changePct': sf((current - float(closes[-2])) / float(closes[-2]) * 100 if len(closes) > 1 else 0),
                        'metrics': {k: (sf(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                                    for k, v in stock_metrics.items()},
                    })
            except Exception as e:
                print(f"Screener error for {sym}: {e}")
                continue

        # Sıralama
        if sort_by in ('rsi', 'volume_ratio', 'atr_pct'):
            matches.sort(key=lambda x: x['metrics'].get(sort_by, 0))
        elif sort_by == 'change':
            matches.sort(key=lambda x: x['changePct'], reverse=True)

        return jsonify(safe_dict({
            'success': True,
            'matches': matches[:limit],
            'totalMatches': len(matches),
            'scannedStocks': len(symbols),
            'conditions': conditions,
            'availableIndicators': [
                'rsi', 'ema_cross', 'price_above_ema20', 'price_above_ema50',
                'volume_ratio', 'breakout', 'macd_signal', 'bollinger_signal', 'atr_pct'
            ],
            'availableSectors': list(SECTOR_MAP.keys()),
        }))
    except Exception as e:
        print(f"Screener error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────── PORTFOLIO ───────────────────────────────

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Portföy getir"""
    user_id = request.args.get('user', 'default')
    portfolio = portfolio_store.get(user_id, [])

    positions = []
    total_value = 0
    total_cost = 0
    daily_pnl = 0

    for pos in portfolio:
        try:
            data = fetch_stock_quick(pos['symbol'])
            if not data:
                continue

            qty = pos['quantity']
            avg_cost = pos['avgCost']
            current_price = data['price']

            market_value = current_price * qty
            cost_basis = avg_cost * qty
            unrealized_pnl = market_value - cost_basis
            unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100 if cost_basis != 0 else 0
            daily_change = data['change'] * qty

            total_value += market_value
            total_cost += cost_basis
            daily_pnl += daily_change

            positions.append({
                'symbol': ss(pos['symbol']),
                'name': data['name'],
                'quantity': si(qty),
                'avgCost': sf(avg_cost),
                'currentPrice': sf(current_price),
                'marketValue': sf(market_value),
                'costBasis': sf(cost_basis),
                'unrealizedPnL': sf(unrealized_pnl),
                'unrealizedPnLPct': sf(unrealized_pnl_pct),
                'dailyChange': sf(daily_change),
                'weight': 0,  # aşağıda hesaplanacak
            })
        except Exception as e:
            print(f"Portfolio calc error {pos.get('symbol')}: {e}")

    # Ağırlık hesapla
    for p in positions:
        p['weight'] = sf(p['marketValue'] / total_value * 100 if total_value > 0 else 0)

    total_pnl = total_value - total_cost
    total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0

    return jsonify(safe_dict({
        'success': True,
        'userId': user_id,
        'positions': positions,
        'summary': {
            'totalValue': sf(total_value),
            'totalCost': sf(total_cost),
            'totalPnL': sf(total_pnl),
            'totalPnLPct': sf(total_pnl_pct),
            'dailyPnL': sf(daily_pnl),
            'positionCount': len(positions),
        }
    }))


@app.route('/api/portfolio', methods=['POST'])
def add_to_portfolio():
    """Portföye pozisyon ekle"""
    try:
        data = request.json
        user_id = data.get('user', 'default')
        symbol = data.get('symbol', '').upper()
        quantity = float(data.get('quantity', 0))
        avg_cost = float(data.get('avgCost', 0))

        if not symbol or quantity <= 0 or avg_cost <= 0:
            return jsonify({'error': 'Geçersiz parametreler'}), 400

        if user_id not in portfolio_store:
            portfolio_store[user_id] = []

        # Aynı hisse varsa birleştir (ortalama maliyet)
        existing = next((p for p in portfolio_store[user_id] if p['symbol'] == symbol), None)
        if existing:
            total_qty = existing['quantity'] + quantity
            existing['avgCost'] = (existing['avgCost'] * existing['quantity'] + avg_cost * quantity) / total_qty
            existing['quantity'] = total_qty
        else:
            portfolio_store[user_id].append({
                'symbol': symbol,
                'quantity': quantity,
                'avgCost': avg_cost,
                'addedAt': datetime.now().isoformat(),
            })

        return jsonify({'success': True, 'message': f'{symbol} portföye eklendi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio', methods=['DELETE'])
def remove_from_portfolio():
    """Portföyden pozisyon çıkar"""
    try:
        data = request.json
        user_id = data.get('user', 'default')
        symbol = data.get('symbol', '').upper()

        if user_id in portfolio_store:
            portfolio_store[user_id] = [p for p in portfolio_store[user_id] if p['symbol'] != symbol]

        return jsonify({'success': True, 'message': f'{symbol} portföyden çıkarıldı'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/portfolio/risk', methods=['GET'])
def portfolio_risk():
    """Portföy risk analizi: max drawdown, volatilite, korelasyon"""
    try:
        user_id = request.args.get('user', 'default')
        portfolio = portfolio_store.get(user_id, [])

        if not portfolio:
            return jsonify({'success': True, 'message': 'Portföy boş', 'risk': {}})

        symbols = [p['symbol'] for p in portfolio]
        weights = []
        returns_matrix = []

        for pos in portfolio:
            hist = fetch_stock_hist(pos['symbol'], '1y')
            if hist.empty or len(hist) < 30:
                continue

            closes = hist['Close'].values.astype(float)
            daily_returns = np.diff(closes) / closes[:-1]
            returns_matrix.append(daily_returns[-252:])  # Son 1 yıl
            weights.append(pos['quantity'] * pos['avgCost'])

        if not returns_matrix:
            return jsonify({'success': True, 'risk': {'message': 'Yetersiz veri'}})

        # Eşit uzunluğa getir
        min_len = min(len(r) for r in returns_matrix)
        returns_matrix = [r[-min_len:] for r in returns_matrix]
        returns_array = np.array(returns_matrix)

        # Ağırlıkları normalize et
        total_weight = sum(weights)
        norm_weights = np.array([w / total_weight for w in weights]) if total_weight > 0 else np.ones(len(weights)) / len(weights)

        # Portföy getirisi
        portfolio_returns = np.dot(norm_weights, returns_array)

        # Volatilite (yıllıklaştırılmış)
        volatility = sf(float(np.std(portfolio_returns) * np.sqrt(252) * 100))

        # Max Drawdown
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = sf(float(np.min(drawdowns) * 100))

        # Sharpe (risk-free rate = %40 TRY faiz)
        annual_return = sf(float((cumulative[-1] - 1) * 100)) if len(cumulative) > 0 else 0.0
        risk_free = 40.0
        sharpe = sf((annual_return - risk_free) / volatility if volatility > 0 else 0)

        # Korelasyon matrisi
        correlation = {}
        if len(returns_array) > 1:
            corr_matrix = np.corrcoef(returns_array)
            for i, sym_i in enumerate(symbols[:len(returns_array)]):
                correlation[sym_i] = {}
                for j, sym_j in enumerate(symbols[:len(returns_array)]):
                    correlation[sym_i][sym_j] = sf(corr_matrix[i][j])

        return jsonify(safe_dict({
            'success': True,
            'risk': {
                'volatility': volatility,
                'maxDrawdown': max_dd,
                'annualReturn': annual_return,
                'sharpeRatio': sharpe,
                'correlation': correlation,
                'dataPoints': min_len,
                'explanation': f"Yıllık volatilite: %{volatility}, Max DD: %{max_dd}, Sharpe: {sharpe}"
            }
        }))
    except Exception as e:
        print(f"Portfolio risk error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────── BACKTEST ────────────────────────────────

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """
    Backtest motoru
    Stratejiler: ma_cross, breakout, mean_reversion
    Body:
    {
        "symbol": "THYAO",
        "strategy": "ma_cross",
        "params": {"fast": 20, "slow": 50},
        "period": "2y",
        "commission": 0.001,
        "slippage": 0.001,
        "initialCapital": 100000
    }
    """
    try:
        data = request.json
        symbol = data.get('symbol', '').upper()
        strategy = data.get('strategy', 'ma_cross')
        params = data.get('params', {})
        period = data.get('period', '2y')
        commission = float(data.get('commission', 0.001))  # %0.1
        slippage = float(data.get('slippage', 0.001))      # %0.1
        initial_capital = float(data.get('initialCapital', 100000))

        hist = fetch_stock_hist(symbol, period)
        if hist.empty or len(hist) < 60:
            return jsonify({'error': 'Yetersiz veri'}), 400

        closes = hist['Close'].values.astype(float)
        highs = hist['High'].values.astype(float)
        lows = hist['Low'].values.astype(float)
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]

        # Sinyal üret
        signals = generate_signals(strategy, closes, highs, lows, params)

        # Backtest simülasyonu
        capital = initial_capital
        position = 0
        shares = 0
        trades = []
        equity_curve = []
        entry_price = 0

        for i in range(len(closes)):
            price = float(closes[i])

            if signals[i] == 1 and position == 0:  # AL
                cost = price * (1 + commission + slippage)
                shares = int(capital / cost)
                if shares > 0:
                    capital -= shares * cost
                    position = 1
                    entry_price = cost
                    trades.append({
                        'date': dates[i], 'type': 'BUY', 'price': sf(price),
                        'shares': shares, 'cost': sf(shares * cost)
                    })

            elif signals[i] == -1 and position == 1:  # SAT
                revenue = shares * price * (1 - commission - slippage)
                pnl = revenue - shares * entry_price
                capital += revenue
                trades.append({
                    'date': dates[i], 'type': 'SELL', 'price': sf(price),
                    'shares': shares, 'revenue': sf(revenue), 'pnl': sf(pnl)
                })
                position = 0
                shares = 0

            # Equity curve
            equity = capital + (shares * price if position == 1 else 0)
            equity_curve.append({
                'date': dates[i],
                'equity': sf(equity),
                'price': sf(price),
                'position': position,
            })

        # Son pozisyon açıksa kapat
        if position == 1:
            final_price = float(closes[-1])
            revenue = shares * final_price * (1 - commission - slippage)
            capital += revenue
            trades.append({
                'date': dates[-1], 'type': 'SELL (Final)', 'price': sf(final_price),
                'shares': shares, 'revenue': sf(revenue)
            })

        final_equity = capital

        # Performans metrikleri
        total_return = (final_equity - initial_capital) / initial_capital * 100
        n_days = len(closes)
        n_years = n_days / 252
        cagr = ((final_equity / initial_capital) ** (1 / n_years) - 1) * 100 if n_years > 0 else 0

        # Max Drawdown
        equities = [e['equity'] for e in equity_curve]
        eq_arr = np.array(equities)
        running_max = np.maximum.accumulate(eq_arr)
        drawdowns = (eq_arr - running_max) / running_max
        max_dd = float(np.min(drawdowns)) * 100

        # Win rate
        winning = sum(1 for t in trades if t.get('pnl', 0) > 0 and t['type'] == 'SELL')
        losing = sum(1 for t in trades if t.get('pnl', 0) <= 0 and t['type'] == 'SELL')
        total_trades = winning + losing
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0

        # Sharpe
        daily_returns = np.diff(eq_arr) / eq_arr[:-1]
        sharpe = float(np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)) if np.std(daily_returns) > 0 else 0

        # Buy & Hold karşılaştırma
        bh_return = (float(closes[-1]) / float(closes[0]) - 1) * 100

        return jsonify(safe_dict({
            'success': True,
            'symbol': symbol,
            'strategy': strategy,
            'params': params,
            'results': {
                'initialCapital': sf(initial_capital),
                'finalEquity': sf(final_equity),
                'totalReturn': sf(total_return),
                'cagr': sf(cagr),
                'sharpeRatio': sf(sharpe),
                'maxDrawdown': sf(max_dd),
                'winRate': sf(win_rate),
                'totalTrades': total_trades,
                'winningTrades': winning,
                'losingTrades': losing,
                'buyAndHoldReturn': sf(bh_return),
                'alpha': sf(total_return - bh_return),
            },
            'trades': trades[-50:],  # Son 50 işlem
            'equityCurve': equity_curve,
            'availableStrategies': {
                'ma_cross': {'description': 'Moving Average Crossover', 'params': ['fast', 'slow']},
                'breakout': {'description': 'Breakout (N-day high)', 'params': ['lookback']},
                'mean_reversion': {'description': 'Mean Reversion (RSI)', 'params': ['rsi_low', 'rsi_high']},
            }
        }))
    except Exception as e:
        print(f"Backtest error: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


def generate_signals(strategy, closes, highs, lows, params):
    """Strateji sinyalleri üret. 1=AL, -1=SAT, 0=bekle"""
    n = len(closes)
    signals = [0] * n

    if strategy == 'ma_cross':
        fast = int(params.get('fast', 20))
        slow = int(params.get('slow', 50))
        series = pd.Series(closes.copy(), dtype=float)
        ema_fast = series.ewm(span=fast, adjust=False).mean().values
        ema_slow = series.ewm(span=slow, adjust=False).mean().values

        for i in range(slow + 1, n):
            if ema_fast[i] > ema_slow[i] and ema_fast[i - 1] <= ema_slow[i - 1]:
                signals[i] = 1  # Golden cross
            elif ema_fast[i] < ema_slow[i] and ema_fast[i - 1] >= ema_slow[i - 1]:
                signals[i] = -1  # Death cross

    elif strategy == 'breakout':
        lookback = int(params.get('lookback', 20))
        for i in range(lookback + 1, n):
            high_n = float(np.max(highs[i - lookback:i]))
            low_n = float(np.min(lows[i - lookback:i]))
            if float(closes[i]) > high_n:
                signals[i] = 1
            elif float(closes[i]) < low_n:
                signals[i] = -1

    elif strategy == 'mean_reversion':
        rsi_low = float(params.get('rsi_low', 30))
        rsi_high = float(params.get('rsi_high', 70))
        for i in range(15, n):
            rsi_val = calc_rsi_single(closes[:i + 1])
            if rsi_val is not None:
                if rsi_val < rsi_low:
                    signals[i] = 1
                elif rsi_val > rsi_high:
                    signals[i] = -1

    return signals


# ─────────────────────────────── ALERTS ──────────────────────────────────

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Tüm uyarıları listele"""
    user_email = request.args.get('email', None)
    if user_email:
        filtered = [a for a in alert_store if a.get('email') == user_email]
    else:
        filtered = alert_store

    return jsonify(safe_dict({
        'success': True,
        'alerts': filtered,
        'totalAlerts': len(filtered),
    }))


@app.route('/api/alerts', methods=['POST'])
def create_alert():
    """
    Uyarı oluştur
    Body:
    {
        "email": "user@example.com",
        "symbol": "THYAO",
        "condition": "price_above",  // price_above, price_below, rsi_above, rsi_below, volume_spike, breakout
        "threshold": 300,
        "cooldownMinutes": 60,
        "confirmBars": 2  // Teyit: kaç bar üst üste koşul sağlanmalı
    }
    """
    try:
        data = request.json
        email = data.get('email')
        symbol = data.get('symbol', '').upper()
        condition = data.get('condition')
        threshold = data.get('threshold')
        cooldown = int(data.get('cooldownMinutes', 60))
        confirm_bars = int(data.get('confirmBars', 1))

        if not email or not symbol or not condition:
            return jsonify({'error': 'email, symbol, condition zorunlu'}), 400

        alert_id = f"alert_{len(alert_store) + 1}_{int(time.time())}"

        alert = {
            'id': alert_id,
            'email': email,
            'symbol': symbol,
            'condition': condition,
            'threshold': sf(threshold) if threshold else None,
            'cooldownMinutes': cooldown,
            'confirmBars': confirm_bars,
            'active': True,
            'triggered': False,
            'triggerCount': 0,
            'consecutiveHits': 0,
            'lastTriggered': None,
            'createdAt': datetime.now().isoformat(),
        }

        alert_store.append(alert)

        return jsonify(safe_dict({
            'success': True,
            'message': f'Uyarı oluşturuldu: {symbol} {condition} {threshold or ""}',
            'alert': alert,
            'availableConditions': [
                'price_above', 'price_below',
                'rsi_above', 'rsi_below',
                'volume_spike', 'breakout',
                'macd_cross_up', 'macd_cross_down',
                'bollinger_upper', 'bollinger_lower'
            ]
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts/<alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    """Uyarı sil"""
    global alert_store
    alert_store = [a for a in alert_store if a.get('id') != alert_id]
    return jsonify({'success': True, 'message': 'Uyarı silindi'})


@app.route('/api/alerts/check', methods=['POST'])
def check_alerts():
    """
    Uyarıları kontrol et (cron job veya frontend polling ile çağrılır)
    Cooldown ve teyit kuralı uygular
    """
    try:
        triggered = []
        now = datetime.now()

        for alert in alert_store:
            if not alert.get('active', True):
                continue

            # Cooldown kontrolü
            if alert.get('lastTriggered'):
                last = datetime.fromisoformat(alert['lastTriggered'])
                cooldown_td = timedelta(minutes=alert.get('cooldownMinutes', 60))
                if now - last < cooldown_td:
                    continue

            symbol = alert['symbol']
            condition = alert['condition']
            threshold = alert.get('threshold', 0)

            try:
                hist = fetch_stock_hist(symbol, '3mo')
                if hist.empty:
                    continue

                closes = hist['Close'].values.astype(float)
                highs = hist['High'].values.astype(float)
                lows = hist['Low'].values.astype(float)
                volumes = hist['Volume'].values.astype(float)
                current = float(closes[-1])

                met = False

                if condition == 'price_above' and current > float(threshold):
                    met = True
                elif condition == 'price_below' and current < float(threshold):
                    met = True
                elif condition == 'rsi_above':
                    rsi = calc_rsi(closes)['value']
                    if rsi > float(threshold):
                        met = True
                elif condition == 'rsi_below':
                    rsi = calc_rsi(closes)['value']
                    if rsi < float(threshold):
                        met = True
                elif condition == 'volume_spike':
                    avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
                    ratio = float(volumes[-1]) / avg_vol if avg_vol > 0 else 1
                    if ratio > float(threshold or 2):
                        met = True
                elif condition == 'breakout':
                    high_20 = float(np.max(highs[-21:-1])) if len(highs) > 21 else float(np.max(highs[:-1]))
                    if current > high_20:
                        met = True
                elif condition == 'macd_cross_up':
                    macd = calc_macd(closes)
                    if macd['signalType'] == 'buy':
                        met = True
                elif condition == 'macd_cross_down':
                    macd = calc_macd(closes)
                    if macd['signalType'] == 'sell':
                        met = True

                # Teyit kuralı
                if met:
                    alert['consecutiveHits'] = alert.get('consecutiveHits', 0) + 1
                else:
                    alert['consecutiveHits'] = 0

                confirm_needed = alert.get('confirmBars', 1)
                if alert['consecutiveHits'] >= confirm_needed:
                    alert['triggered'] = True
                    alert['triggerCount'] = alert.get('triggerCount', 0) + 1
                    alert['lastTriggered'] = now.isoformat()
                    alert['consecutiveHits'] = 0

                    triggered.append({
                        'alertId': alert['id'],
                        'symbol': symbol,
                        'condition': condition,
                        'threshold': threshold,
                        'currentValue': sf(current),
                        'email': alert['email'],
                        'message': f"⚠️ {symbol}: {condition} koşulu sağlandı (Değer: {sf(current)})",
                    })

            except Exception as e:
                print(f"Alert check error {symbol}: {e}")

        return jsonify(safe_dict({
            'success': True,
            'triggered': triggered,
            'checkedAlerts': len(alert_store),
            'triggeredCount': len(triggered),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────── WATCHLIST ───────────────────────────────

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    """Takip listesi"""
    user_id = request.args.get('user', 'default')
    symbols = watchlist_store.get(user_id, [])

    stocks = []
    for sym in symbols:
        data = fetch_stock_quick(sym)
        if data:
            stocks.append(data)

    return jsonify(safe_dict({'success': True, 'watchlist': stocks}))


@app.route('/api/watchlist', methods=['POST'])
def update_watchlist():
    """Takip listesine ekle/çıkar"""
    try:
        data = request.json
        user_id = data.get('user', 'default')
        symbol = data.get('symbol', '').upper()
        action = data.get('action', 'add')  # add / remove

        if user_id not in watchlist_store:
            watchlist_store[user_id] = []

        if action == 'add' and symbol not in watchlist_store[user_id]:
            watchlist_store[user_id].append(symbol)
        elif action == 'remove':
            watchlist_store[user_id] = [s for s in watchlist_store[user_id] if s != symbol]

        return jsonify({'success': True, 'watchlist': watchlist_store[user_id]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────── SECTORS ─────────────────────────────────

@app.route('/api/sectors', methods=['GET'])
def get_sectors():
    """Sektör listesi ve performansları"""
    try:
        sector_data = []
        for sector_name, symbols in SECTOR_MAP.items():
            changes = []
            for sym in symbols:
                data = fetch_stock_quick(sym)
                if data:
                    changes.append(data['changePct'])

            avg_change = sf(np.mean(changes)) if changes else 0.0

            sector_data.append({
                'name': sector_name,
                'stockCount': len(symbols),
                'avgChange': avg_change,
                'symbols': symbols,
            })

        sector_data.sort(key=lambda x: x['avgChange'], reverse=True)

        return jsonify(safe_dict({'success': True, 'sectors': sector_data}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ─────────────────────────────── SEARCH ──────────────────────────────────

@app.route('/api/search', methods=['GET'])
def search_stock():
    """Hisse arama"""
    query = request.args.get('q', '').upper()
    if not query:
        return jsonify({'results': []})

    results = []
    for code, name in BIST100_STOCKS.items():
        if query in code or query in name.upper():
            results.append({'code': code, 'name': name})

    return jsonify({'success': True, 'results': results[:10]})


# ─────────────────────────────── HEALTH ──────────────────────────────────

@app.route('/api/health', methods=['GET'])
def health_check():
    """Sağlık kontrolü"""
    return jsonify({
        'status': 'ok',
        'version': '3.0.0',
        'timestamp': datetime.now().isoformat(),
        'features': [
            'dashboard', 'indices', 'market_breadth', 'stock_detail',
            'events', 'screener', 'portfolio', 'portfolio_risk',
            'backtest', 'alerts', 'watchlist', 'sectors', 'search',
            'compare', 'kap_placeholder'
        ],
        'stockCount': len(BIST100_STOCKS),
        'cacheEntries': len(_cache),
    })


@app.route('/', methods=['GET'])
def index():
    """Ana sayfa - HTML frontend serve et"""
    return send_from_directory(BASE_DIR, 'index.html')


@app.route('/api/docs', methods=['GET'])
def api_docs():
    """API dökümantasyonu"""
    return jsonify({
        'name': 'BIST Pro v3.0 API',
        'docs': {
            'dashboard': 'GET /api/dashboard',
            'indices': 'GET /api/indices',
            'marketBreadth': 'GET /api/market-breadth',
            'bist100': 'GET /api/bist100?sector=&sort=&order=',
            'bist30': 'GET /api/bist30',
            'stockDetail': 'GET /api/stock/<symbol>?period=1y',
            'stockEvents': 'GET /api/stock/<symbol>/events',
            'stockKap': 'GET /api/stock/<symbol>/kap',
            'compare': 'POST /api/compare {symbols: [], period: "6mo"}',
            'screener': 'POST /api/screener {conditions: [...], sector: ""}',
            'portfolio': 'GET/POST/DELETE /api/portfolio',
            'portfolioRisk': 'GET /api/portfolio/risk',
            'backtest': 'POST /api/backtest {symbol, strategy, params, period}',
            'alerts': 'GET/POST /api/alerts',
            'alertsCheck': 'POST /api/alerts/check',
            'watchlist': 'GET/POST /api/watchlist',
            'sectors': 'GET /api/sectors',
            'search': 'GET /api/search?q=',
            'health': 'GET /api/health',
        }
    })


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("🚀 BIST Pro v3.0 Backend Başlatılıyor...")
    print(f"📊 http://localhost:{port}")
    print("=" * 60)
    print("✅ Modüller:")
    print("   📈 Dashboard (endeksler, movers, piyasa genişliği)")
    print("   📊 Hisse Detay (15+ teknik indikatör, Fibonacci, S/R)")
    print("   🔍 Screener (koşul builder, sektör filtreleri)")
    print("   💼 Portföy (PnL, risk, korelasyon, max drawdown)")
    print("   🔄 Backtest (MA cross, breakout, mean reversion)")
    print("   🔔 Uyarılar (cooldown, teyit kuralı)")
    print("   📋 Watchlist, Sektörler, Arama, Karşılaştırma")
    print("   📰 KAP Akışı (placeholder)")
    print(f"   📦 {len(BIST100_STOCKS)} BIST hissesi yüklü")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port)
