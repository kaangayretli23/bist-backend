"""
BIST Pro v3.0 - Kapsamli Borsa Analiz Platformu Backend
=========================================================
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
app.config['TRAP_HTTP_EXCEPTIONS'] = True
app.config['PROPAGATE_EXCEPTIONS'] = False

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint bulunamadi'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Sunucu hatasi', 'details': str(e)}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.exception("Unhandled exception")
    return jsonify({'error': 'Unhandled error', 'details': str(e)}), 500

CACHE_TTL = 300
MAX_BATCH_SIZE = 20
YF_TIMEOUT = 15

def sf(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0.0
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return 0.0

def si(val):
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0

def ss(val, default=""):
    if val is None:
        return default
    return str(val)

def safe_dict(d):
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

BIST100_STOCKS = {
    'THYAO': 'Turk Hava Yollari', 'GARAN': 'Garanti BBVA',
    'ISCTR': 'Is Bankasi (C)', 'AKBNK': 'Akbank',
    'TUPRS': 'Tupras', 'BIMAS': 'BIM',
    'SAHOL': 'Sabanci Holding', 'KCHOL': 'Koc Holding',
    'EREGL': 'Eregli Demir Celik', 'SISE': 'Sise Cam',
    'PETKM': 'Petkim', 'ASELS': 'Aselsan',
    'TOASO': 'Tofas', 'TCELL': 'Turkcell',
    'ENKAI': 'Enka Insaat', 'KOZAL': 'Koza Altin',
    'KRDMD': 'Kardemir (D)', 'TTKOM': 'Turk Telekom',
    'ARCLK': 'Arcelik', 'SOKM': 'Sok Marketler',
    'HEKTS': 'Hektas', 'PGSUS': 'Pegasus',
    'TAVHL': 'TAV Havalimanlari', 'DOHOL': 'Dogan Holding',
    'VESTL': 'Vestel', 'MGROS': 'Migros',
    'FROTO': 'Ford Otosan', 'SASA': 'SASA Polyester',
    'EKGYO': 'Emlak Konut GYO', 'TTRAK': 'Turk Traktor',
    'AEFES': 'Anadolu Efes', 'YKBNK': 'Yapi Kredi',
    'VAKBN': 'Vakifbank', 'HALKB': 'Halkbank',
    'TSKB': 'TSKB', 'ALARK': 'Alarko Holding',
    'GESAN': 'Giresun San.', 'OTKAR': 'Otokar',
    'CEMTS': 'Cemtas', 'CIMSA': 'Cimsa',
    'OYAKC': 'Oyak Cimento', 'GUBRF': 'Gubre Fabrikalari',
    'KONTR': 'Kontrolmatik', 'KOZAA': 'Koza Anadolu',
    'SKBNK': 'Sekerbank', 'ISGYO': 'Is GYO',
    'ULKER': 'Ulker', 'CCOLA': 'Coca-Cola Icecek',
    'TRGYO': 'Torunlar GYO', 'ENJSA': 'Enerjisa',
    'AKSEN': 'Aksa Enerji', 'ODAS': 'Odas Elektrik',
    'AGHOL': 'AG Anadolu Grubu', 'BRYAT': 'Borusan Yatirim',
    'DOAS': 'Dogus Otomotiv', 'BERA': 'Bera Holding',
    'MPARK': 'MLP Saglik', 'GENIL': 'Gen Ilac',
    'ANHYT': 'Anadolu Hayat', 'AGROT': 'Agrotech',
    'BTCIM': 'Baticim', 'BUCIM': 'Bursa Cimento',
    'EGEEN': 'Ege Endustri', 'ISMEN': 'Is Yatirim Menkul',
    'KLRHO': 'Kiler Holding', 'LOGO': 'Logo Yazilim',
    'MAVI': 'Mavi Giyim', 'NETAS': 'Netas Telekom',
    'PRKME': 'Park Elektrik', 'QUAGR': 'QUA Granite',
    'SMRTG': 'Smartiks', 'TKFEN': 'Tekfen Holding',
    'TURSG': 'Turkiye Sigorta', 'VERUS': 'Verusaturk',
    'YGYO': 'Yesil GYO', 'ZOREN': 'Zorlu Enerji'
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
    'bankacilik': BANKING_STOCKS,
    'havacilik': ['THYAO', 'PGSUS', 'TAVHL'],
    'otomotiv': ['TOASO', 'FROTO', 'DOAS', 'TTRAK', 'OTKAR'],
    'enerji': ['TUPRS', 'PETKM', 'AKSEN', 'ENJSA', 'ODAS', 'ZOREN'],
    'holding': ['SAHOL', 'KCHOL', 'DOHOL', 'AGHOL', 'ALARK'],
    'perakende': ['BIMAS', 'SOKM', 'MGROS', 'MAVI'],
    'teknoloji': ['ASELS', 'LOGO', 'NETAS'],
    'telekomunikasyon': ['TCELL', 'TTKOM'],
    'demir_celik': ['EREGL', 'KRDMD', 'CEMTS'],
    'cam_seramik': ['SISE', 'TRKCM'],
    'gida': ['ULKER', 'CCOLA', 'AEFES'],
    'insaat': ['ENKAI', 'TKFEN'],
    'gayrimenkul': ['EKGYO', 'TRGYO', 'ISGYO', 'YGYO'],
    'kimya': ['SASA', 'GUBRF', 'HEKTS'],
    'saglik': ['MPARK', 'GENIL'],
    'sigorta': ['ANHYT', 'TURSG'],
}

# =============================================================================
# CORE DATA FETCHING
# =============================================================================

def fetch_stock_hist(symbol, period='1y'):
    ck = cache_key('hist', symbol, period)
    cached = cache_get(ck)
    if cached is not None:
        return cached
    try:
        ticker_symbol = f"{symbol.upper()}.IS"
        stock = yf.Ticker(ticker_symbol)
        hist = stock.history(period=period, timeout=YF_TIMEOUT)
        if hist.empty:
            time.sleep(0.4)
            hist = stock.history(period=period, timeout=YF_TIMEOUT)
        if not hist.empty:
            cache_set(ck, hist)
        return hist
    except Exception as e:
        print(f"fetch_stock_hist error {symbol}: {e}")
        return pd.DataFrame()


def fetch_stock_info(symbol):
    ck = cache_key('info', symbol)
    cached = cache_get(ck)
    if cached is not None:
        return cached
    try:
        ticker_symbol = f"{symbol.upper()}.IS"
        stock = yf.Ticker(ticker_symbol)
        info = stock.info
        cache_set(ck, info)
        return info
    except Exception:
        return {}


def fetch_stock_quick(symbol):
    ck = cache_key('quick', symbol)
    cached = cache_get(ck)
    if cached is not None:
        return cached
    try:
        ticker = yf.Ticker(f"{symbol.upper()}.IS")
        hist = ticker.history(period="5d", timeout=YF_TIMEOUT)
        if hist.empty:
            time.sleep(0.4)
            hist = ticker.history(period="5d", timeout=YF_TIMEOUT)
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
        gap = sf(day_open - prev)
        gap_pct = sf((gap / prev) * 100 if prev != 0 else 0)
        result = {
            'code': ss(symbol.upper()),
            'name': ss(BIST100_STOCKS.get(symbol.upper(), symbol.upper())),
            'price': current, 'change': change, 'changePct': change_pct,
            'volume': volume, 'open': day_open, 'high': day_high, 'low': day_low,
            'prevClose': prev, 'gap': gap, 'gapPct': gap_pct,
        }
        cache_set(ck, result)
        return result
    except Exception as e:
        print(f"Quick fetch error {symbol}: {e}")
        return None


def fetch_quick_batch(symbols, period="5d"):
    symbols = [s.upper() for s in symbols]
    ck = cache_key('quick_batch', period, ','.join(sorted(symbols)))
    cached = cache_get(ck)
    if cached is not None:
        return cached
    try:
        tickers = [f"{s}.IS" for s in symbols]
        df = yf.download(" ".join(tickers), period=period, group_by="ticker", threads=True, progress=False)
        if df is None or df.empty:
            return []
        results = []
        multi = isinstance(df.columns, pd.MultiIndex)
        for s in symbols:
            t = f"{s}.IS"
            try:
                if multi:
                    lvl0 = set(df.columns.get_level_values(0))
                    lvl1 = set(df.columns.get_level_values(1))
                    if t in lvl0:
                        h = df[t].dropna(how="all")
                    elif t in lvl1:
                        h = df.xs(t, axis=1, level=1, drop_level=True).dropna(how="all")
                    else:
                        continue
                else:
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
                day_low = sf(h["Low"].iloc[-1]) if "Low" in h.columns else 0
                gap = sf(day_open - prev)
                gap_pct = sf((gap / prev) * 100 if prev else 0)
                results.append({
                    "code": s, "name": BIST100_STOCKS.get(s, s),
                    "price": current, "prevClose": prev,
                    "change": change, "changePct": change_pct,
                    "volume": volume, "open": day_open,
                    "high": day_high, "low": day_low,
                    "gap": gap, "gapPct": gap_pct,
                })
            except Exception:
                continue
        cache_set(ck, results)
        return results
    except Exception as e:
        print(f"fetch_quick_batch error: {e}")
        return []


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calc_rsi(closes, period=14):
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
    zone = 'Asiri satim' if rsi < 30 else ('Asiri alim' if rsi > 70 else 'Notr')
    return {'name': 'RSI', 'value': rsi, 'signal': signal, 'explanation': f"RSI {rsi} - {zone}"}

def calc_rsi_single(closes, period=14):
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
    if len(closes) < 26:
        return {'name': 'MACD', 'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'signalType': 'neutral', 'explanation': 'Yetersiz veri'}
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
    return {'name': 'MACD', 'macd': macd_val, 'signal': signal_val, 'histogram': hist_val, 'signalType': sig, 'explanation': f"MACD {'yukselis' if sig == 'buy' else 'dusus' if sig == 'sell' else 'notr'} sinyali"}

def calc_macd_history(closes):
    if len(closes) < 26:
        return []
    series = pd.Series(closes.copy() if isinstance(closes, np.ndarray) else closes, dtype=float)
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - signal_line
    return [{'macd': sf(macd_line.iloc[i]), 'signal': sf(signal_line.iloc[i]), 'histogram': sf(histogram.iloc[i])} for i in range(26, len(closes))]

def calc_bollinger(closes, current_price, period=20):
    if len(closes) < period:
        return {'name': 'Bollinger Bands', 'upper': 0.0, 'middle': 0.0, 'lower': 0.0, 'signal': 'neutral', 'explanation': 'Yetersiz veri', 'bandwidth': 0.0}
    recent = closes[-period:]
    sma = float(np.mean(recent))
    std = float(np.std(recent))
    upper = sf(sma + 2 * std)
    middle = sf(sma)
    lower = sf(sma - 2 * std)
    bandwidth = sf((upper - lower) / middle * 100 if middle != 0 else 0)
    cp = float(current_price)
    signal = 'buy' if cp < lower else ('sell' if cp > upper else 'neutral')
    return {'name': 'Bollinger Bands', 'upper': upper, 'middle': middle, 'lower': lower, 'bandwidth': bandwidth, 'signal': signal, 'explanation': f"Fiyat {'alt bant alti' if cp < lower else 'ust bant ustu' if cp > upper else 'bantlar arasi'}"}

def calc_bollinger_history(closes, period=20):
    result = []
    for i in range(period, len(closes)):
        window = closes[i - period:i]
        sma = float(np.mean(window))
        std = float(np.std(window))
        result.append({'upper': sf(sma + 2 * std), 'middle': sf(sma), 'lower': sf(sma - 2 * std)})
    return result

def calc_ema(closes, current_price):
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
        result['explanation'] = 'Guclu yukselis trendi (Fiyat > EMA20 > EMA50)'
    elif cp < ema20 < ema50:
        result['signal'] = 'sell'
        result['explanation'] = 'Guclu dusus trendi (Fiyat < EMA20 < EMA50)'
    else:
        result['signal'] = 'neutral'
        result['explanation'] = 'Kararsiz trend'
    return result

def calc_ema_history(closes):
    series = pd.Series(closes.copy() if isinstance(closes, np.ndarray) else closes, dtype=float)
    ema20 = series.ewm(span=20, adjust=False).mean() if len(closes) >= 20 else pd.Series([])
    ema50 = series.ewm(span=50, adjust=False).mean() if len(closes) >= 50 else pd.Series([])
    ema200 = series.ewm(span=200, adjust=False).mean() if len(closes) >= 200 else pd.Series([])
    result = []
    for i in range(len(closes)):
        point = {}
        if i < len(ema20): point['ema20'] = sf(ema20.iloc[i])
        if i < len(ema50): point['ema50'] = sf(ema50.iloc[i])
        if i < len(ema200): point['ema200'] = sf(ema200.iloc[i])
        result.append(point)
    return result

def calc_stochastic(closes, highs, lows, period=14):
    if len(closes) < period:
        return {'name': 'Stochastic', 'k': 50.0, 'd': 50.0, 'signal': 'neutral', 'explanation': 'Yetersiz veri'}
    rc, rh, rl = closes[-period:], highs[-period:], lows[-period:]
    highest = float(np.max(rh))
    lowest = float(np.min(rl))
    current = float(rc[-1])
    k = sf(((current - lowest) / (highest - lowest)) * 100 if highest != lowest else 50)
    signal = 'buy' if k < 20 else ('sell' if k > 80 else 'neutral')
    return {'name': 'Stochastic', 'k': k, 'd': k, 'signal': signal, 'explanation': f"Stochastic %K={k}"}

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return {'name': 'ATR', 'value': 0.0, 'signal': 'neutral', 'explanation': 'Yetersiz veri', 'pct': 0.0}
    tr_list = []
    for i in range(1, len(closes)):
        tr = max(float(highs[i]) - float(lows[i]), abs(float(highs[i]) - float(closes[i-1])), abs(float(lows[i]) - float(closes[i-1])))
        tr_list.append(tr)
    atr = sf(np.mean(tr_list[-period:]))
    current = float(closes[-1])
    atr_pct = sf((atr / current) * 100 if current != 0 else 0)
    volatility = 'yuksek' if atr_pct > 3 else ('dusuk' if atr_pct < 1 else 'orta')
    return {'name': 'ATR', 'value': atr, 'pct': atr_pct, 'signal': 'neutral', 'explanation': f"Volatilite: {volatility} (ATR%: {atr_pct})"}

def calc_adx(highs, lows, closes, period=14):
    n = len(closes)
    if n < period + 1:
        return {'name': 'ADX', 'value': 25.0, 'plusDI': 0.0, 'minusDI': 0.0, 'signal': 'neutral', 'explanation': 'Yetersiz veri'}
    tr_list, plus_dm, minus_dm = [], [], []
    for i in range(1, n):
        h, l, ph, pl, pc = float(highs[i]), float(lows[i]), float(highs[i-1]), float(lows[i-1]), float(closes[i-1])
        tr_list.append(max(h - l, abs(h - pc), abs(l - pc)))
        up_move, down_move = h - ph, pl - l
        plus_dm.append(up_move if up_move > down_move and up_move > 0 else 0)
        minus_dm.append(down_move if down_move > up_move and down_move > 0 else 0)
    if len(tr_list) < period:
        return {'name': 'ADX', 'value': 25.0, 'plusDI': 0.0, 'minusDI': 0.0, 'signal': 'neutral', 'explanation': 'Yetersiz veri'}
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
    signal = 'buy' if plus_di > minus_di and adx_val > 25 else ('sell' if minus_di > plus_di and adx_val > 25 else 'neutral')
    return {'name': 'ADX', 'value': adx_val, 'plusDI': plus_di, 'minusDI': minus_di, 'signal': signal, 'explanation': f"ADX={adx_val}"}

def calc_cci(highs, lows, closes, current_price, period=20):
    if len(closes) < period:
        return {'name': 'CCI', 'value': 0.0, 'signal': 'neutral', 'explanation': 'Yetersiz veri'}
    tp = (np.array(highs, dtype=float) + np.array(lows, dtype=float) + np.array(closes, dtype=float)) / 3
    recent = tp[-period:]
    sma = float(np.mean(recent))
    mean_dev = float(np.mean(np.abs(recent - sma)))
    cci = sf((float(tp[-1]) - sma) / (0.015 * mean_dev) if mean_dev != 0 else 0)
    signal = 'buy' if cci < -100 else ('sell' if cci > 100 else 'neutral')
    return {'name': 'CCI', 'value': cci, 'signal': signal, 'explanation': f"CCI={cci}"}

def calc_williams_r(highs, lows, closes, current_price, period=14):
    if len(closes) < period:
        return {'name': 'Williams %R', 'value': -50.0, 'signal': 'neutral', 'explanation': 'Yetersiz veri'}
    highest = float(np.max(highs[-period:]))
    lowest = float(np.min(lows[-period:]))
    cp = float(current_price)
    wr = sf(((highest - cp) / (highest - lowest)) * -100 if highest != lowest else -50)
    signal = 'buy' if wr < -80 else ('sell' if wr > -20 else 'neutral')
    return {'name': 'Williams %R', 'value': wr, 'signal': signal, 'explanation': f"W%R={wr}"}

def calc_obv(closes, volumes):
    if len(closes) < 10:
        return {'name': 'OBV', 'value': 0, 'trend': 'neutral', 'signal': 'neutral', 'explanation': 'Yetersiz veri'}
    obv = 0
    obv_values = [0]
    for i in range(1, len(closes)):
        if float(closes[i]) > float(closes[i-1]):
            obv += int(volumes[i])
        elif float(closes[i]) < float(closes[i-1]):
            obv -= int(volumes[i])
        obv_values.append(obv)
    lookback = min(10, len(obv_values) - 1)
    trend = 'up' if obv_values[-1] > obv_values[-lookback] else 'down'
    signal = 'buy' if trend == 'up' else 'sell'
    return {'name': 'OBV', 'value': si(abs(obv_values[-1])), 'trend': trend, 'signal': signal, 'explanation': f"Hacim trendi: {'yukselis' if trend == 'up' else 'dusus'}"}

def calc_volume_profile(hist, bins=20):
    try:
        closes = hist['Close'].values.astype(float)
        volumes = hist['Volume'].values.astype(float)
        price_min, price_max = float(np.min(closes)), float(np.max(closes))
        if price_min == price_max:
            return {'levels': [], 'poc': sf(price_min)}
        edges = np.linspace(price_min, price_max, bins + 1)
        levels = []
        max_vol, poc_price = 0, price_min
        for i in range(bins):
            mask = (closes >= edges[i]) & (closes < edges[i + 1])
            vol = float(np.sum(volumes[mask]))
            mid = float((edges[i] + edges[i + 1]) / 2)
            levels.append({'price': sf(mid), 'volume': si(vol)})
            if vol > max_vol:
                max_vol = vol
                poc_price = mid
        return {'levels': levels, 'poc': sf(poc_price), 'explanation': f"POC: {sf(poc_price)} TL"}
    except Exception as e:
        return {'levels': [], 'poc': 0.0}

def calc_support_resistance(hist, atr_filter=True):
    try:
        closes = hist['Close'].values.astype(float)
        highs = hist['High'].values.astype(float)
        lows = hist['Low'].values.astype(float)
        n = min(90, len(closes))
        rh, rl = highs[-n:], lows[-n:]
        atr_val = 0
        if atr_filter and len(closes) > 14:
            tr_list = []
            for i in range(1, len(closes)):
                tr_list.append(max(float(highs[i] - lows[i]), abs(float(highs[i] - closes[i-1])), abs(float(lows[i] - closes[i-1]))))
            atr_val = float(np.mean(tr_list[-14:])) * 0.5
        supports, resistances = [], []
        for i in range(2, n - 2):
            if rh[i] > rh[i-1] and rh[i] > rh[i-2] and rh[i] > rh[i+1] and rh[i] > rh[i+2]:
                resistances.append(float(rh[i]))
            if rl[i] < rl[i-1] and rl[i] < rl[i-2] and rl[i] < rl[i+1] and rl[i] < rl[i+2]:
                supports.append(float(rl[i]))
        def cluster_levels(levels, min_dist):
            if not levels: return []
            levels = sorted(levels)
            clustered = [levels[0]]
            for lv in levels[1:]:
                if lv - clustered[-1] > min_dist:
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
            'supports': [sf(s) for s in nearest_sup], 'resistances': [sf(r) for r in nearest_res],
            'allSupports': [sf(s) for s in sorted(supports, reverse=True)],
            'allResistances': [sf(r) for r in sorted(resistances)],
            'current': sf(current), 'atrFilter': sf(atr_val),
            'explanation': f"Destek: {sf(nearest_sup[0]) if nearest_sup else 'Yok'}, Direnc: {sf(nearest_res[0]) if nearest_res else 'Yok'}"
        }
    except Exception as e:
        return {'supports': [], 'resistances': [], 'current': 0.0, 'explanation': 'Hesaplanamadi'}

def calc_fibonacci(hist):
    try:
        closes = hist['Close'].values.astype(float)
        n = min(90, len(closes))
        recent = closes[-n:]
        high, low = float(np.max(recent)), float(np.min(recent))
        diff = high - low
        levels = {
            '0.0': sf(high), '23.6': sf(high - diff * 0.236), '38.2': sf(high - diff * 0.382),
            '50.0': sf(high - diff * 0.5), '61.8': sf(high - diff * 0.618),
            '78.6': sf(high - diff * 0.786), '100.0': sf(low)
        }
        current = float(closes[-1])
        zone = "Belirsiz"
        lk, lv = list(levels.keys()), list(levels.values())
        for i in range(len(lv) - 1):
            if current <= lv[i] and current >= lv[i + 1]:
                zone = f"{lk[i]}% - {lk[i+1]}% arasi"
                break
        return {'levels': levels, 'high': sf(high), 'low': sf(low), 'currentZone': zone, 'explanation': f"Fiyat {zone} Fibonacci bolgesinde"}
    except Exception as e:
        return {'levels': {}, 'explanation': 'Hesaplanamadi'}

def calc_all_indicators(hist, current_price):
    closes = hist['Close'].values.astype(float)
    highs = hist['High'].values.astype(float)
    lows = hist['Low'].values.astype(float)
    volumes = hist['Volume'].values.astype(float)
    cp = float(current_price)
    rsi_history = []
    for i in range(len(closes)):
        if i >= 14:
            rv = calc_rsi_single(closes[:i+1])
            if rv is not None:
                rsi_history.append({'date': hist.index[i].strftime('%Y-%m-%d'), 'value': rv})
    indicators = {
        'rsi': calc_rsi(closes), 'rsiHistory': rsi_history,
        'macd': calc_macd(closes), 'macdHistory': calc_macd_history(closes),
        'bollinger': calc_bollinger(closes, cp), 'bollingerHistory': calc_bollinger_history(closes),
        'stochastic': calc_stochastic(closes, highs, lows),
        'ema': calc_ema(closes, cp), 'emaHistory': calc_ema_history(closes),
        'atr': calc_atr(highs, lows, closes), 'adx': calc_adx(highs, lows, closes),
        'cci': calc_cci(highs, lows, closes, cp), 'williamsr': calc_williams_r(highs, lows, closes, cp),
        'obv': calc_obv(closes, volumes), 'volumeProfile': calc_volume_profile(hist),
    }
    signals = [v.get('signal', 'neutral') for v in indicators.values() if isinstance(v, dict) and 'signal' in v]
    buy_c, sell_c, total = signals.count('buy'), signals.count('sell'), len(signals)
    if buy_c > sell_c and buy_c >= total * 0.4:
        overall, strength = 'buy', sf(buy_c / total * 100)
    elif sell_c > buy_c and sell_c >= total * 0.4:
        overall, strength = 'sell', sf(sell_c / total * 100)
    else:
        overall, strength = 'neutral', sf(50)
    indicators['summary'] = {'overall': overall, 'strength': strength, 'buySignals': buy_c, 'sellSignals': sell_c, 'neutralSignals': total - buy_c - sell_c, 'explanation': f"{total} indikator: {buy_c} AL, {sell_c} SAT, {total - buy_c - sell_c} NOTR"}
    return indicators

def prepare_chart_data(hist, max_points=None):
    try:
        data = hist if max_points is None else hist.tail(max_points)
        candlestick = [{'date': date.strftime('%Y-%m-%d'), 'open': sf(row['Open']), 'high': sf(row['High']), 'low': sf(row['Low']), 'close': sf(row['Close']), 'volume': si(row['Volume'])} for date, row in data.iterrows()]
        return {'candlestick': candlestick, 'dates': [c['date'] for c in candlestick], 'prices': [c['close'] for c in candlestick], 'volumes': [c['volume'] for c in candlestick], 'dataPoints': len(candlestick)}
    except Exception:
        return {'candlestick': [], 'dates': [], 'prices': [], 'volumes': [], 'dataPoints': 0}

def fetch_stock_events(symbol):
    try:
        ticker = yf.Ticker(f"{symbol.upper()}.IS")
        dividends = []
        try:
            dd = ticker.dividends
            if dd is not None and len(dd) > 0:
                for date, val in dd.items():
                    dividends.append({'date': date.strftime('%Y-%m-%d'), 'amount': sf(val), 'type': 'dividend'})
        except Exception: pass
        splits = []
        try:
            sd = ticker.splits
            if sd is not None and len(sd) > 0:
                for date, val in sd.items():
                    splits.append({'date': date.strftime('%Y-%m-%d'), 'ratio': sf(val), 'type': 'split'})
        except Exception: pass
        return {'dividends': dividends[-20:], 'splits': splits, 'calendar': {}, 'totalDividends': len(dividends), 'totalSplits': len(splits)}
    except Exception as e:
        return {'dividends': [], 'splits': [], 'calendar': {}}

# =============================================================================
# IN-MEMORY STORAGE & API ROUTES
# =============================================================================

portfolio_store = {}
alert_store = []
watchlist_store = {}

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    try:
        ck = cache_key('dashboard_v1')
        cached = cache_get(ck)
        if cached is not None:
            return jsonify(cached)
        all_stocks = fetch_quick_batch(BIST100_STOCKS.keys(), period="5d")
        if not all_stocks:
            payload = {'success': True, 'stockCount': 0, 'movers': {'topGainers': [], 'topLosers': [], 'volumeLeaders': [], 'gapStocks': []}, 'marketBreadth': {'advancing': 0, 'declining': 0, 'unchanged': 0, 'advDecRatio': 0}, 'allStocks': [], 'message': 'Piyasa verisi alinamadi.'}
            cache_set(ck, payload)
            return jsonify(payload)
        sbc = sorted(all_stocks, key=lambda x: x.get('changePct', 0), reverse=True)
        adv = sum(1 for s in all_stocks if s.get('changePct', 0) > 0)
        dec = sum(1 for s in all_stocks if s.get('changePct', 0) < 0)
        unc = len(all_stocks) - adv - dec
        payload = safe_dict({'success': True, 'timestamp': datetime.now().isoformat(), 'stockCount': len(all_stocks),
            'movers': {'topGainers': sbc[:5], 'topLosers': sbc[-5:][::-1], 'volumeLeaders': sorted(all_stocks, key=lambda x: x.get('volume', 0), reverse=True)[:5], 'gapStocks': sorted(all_stocks, key=lambda x: abs(x.get('gapPct', 0)), reverse=True)[:5]},
            'marketBreadth': {'advancing': adv, 'declining': dec, 'unchanged': unc, 'advDecRatio': sf(adv / dec if dec > 0 else adv)}, 'allStocks': sbc})
        cache_set(ck, payload)
        return jsonify(payload)
    except Exception as e:
        app.logger.exception("Dashboard error")
        return jsonify({'error': str(e)}), 500

@app.route('/api/indices', methods=['GET'])
def get_indices():
    try:
        indices = {}
        for key, (tsym, name) in {'XU100': ('XU100.IS', 'BIST 100'), 'XU030': ('XU030.IS', 'BIST 30'), 'XBANK': ('XBANK.IS', 'BIST Bankacilik')}.items():
            try:
                t = yf.Ticker(tsym)
                h = t.history(period="5d", timeout=YF_TIMEOUT)
                if not h.empty and len(h) >= 2:
                    indices[key] = {'name': name, 'value': sf(h['Close'].iloc[-1]), 'change': sf(h['Close'].iloc[-1] - h['Close'].iloc[-2]), 'changePct': sf((h['Close'].iloc[-1] - h['Close'].iloc[-2]) / h['Close'].iloc[-2] * 100), 'high': sf(h['High'].iloc[-1]), 'low': sf(h['Low'].iloc[-1]), 'volume': si(h['Volume'].iloc[-1])}
            except Exception as e:
                print(f"{key} error: {e}")
        try:
            h = yf.Ticker("USDTRY=X").history(period="5d", timeout=YF_TIMEOUT)
            if not h.empty and len(h) >= 2:
                indices['USDTRY'] = {'name': 'USD/TRY', 'value': sf(h['Close'].iloc[-1], 4), 'change': sf(h['Close'].iloc[-1] - h['Close'].iloc[-2], 4), 'changePct': sf((h['Close'].iloc[-1] - h['Close'].iloc[-2]) / h['Close'].iloc[-2] * 100)}
        except Exception: pass
        try:
            h = yf.Ticker("GC=F").history(period="5d", timeout=YF_TIMEOUT)
            if not h.empty and len(h) >= 2:
                indices['GOLD'] = {'name': 'Altin (ONS)', 'value': sf(h['Close'].iloc[-1]), 'change': sf(h['Close'].iloc[-1] - h['Close'].iloc[-2]), 'changePct': sf((h['Close'].iloc[-1] - h['Close'].iloc[-2]) / h['Close'].iloc[-2] * 100)}
        except Exception: pass
        return jsonify(safe_dict({'success': True, 'indices': indices}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bist100', methods=['GET'])
def get_bist100_list():
    try:
        sector_filter = request.args.get('sector', None)
        sort_by = request.args.get('sort', 'code')
        order = request.args.get('order', 'asc')
        ck = cache_key('bist100_v1', sector_filter or '', sort_by, order)
        cached = cache_get(ck)
        if cached is not None:
            return jsonify(cached)
        stocks_to_fetch = list(BIST100_STOCKS.keys())
        if sector_filter and sector_filter in SECTOR_MAP:
            stocks_to_fetch = [s for s in SECTOR_MAP[sector_filter] if s in BIST100_STOCKS]
        stocks = fetch_quick_batch(stocks_to_fetch, period="5d")
        reverse = (order == 'desc')
        if sort_by == 'change':
            stocks.sort(key=lambda x: x.get('changePct', 0), reverse=reverse)
        elif sort_by == 'volume':
            stocks.sort(key=lambda x: x.get('volume', 0), reverse=reverse)
        elif sort_by == 'price':
            stocks.sort(key=lambda x: x.get('price', 0), reverse=reverse)
        else:
            stocks.sort(key=lambda x: x.get('code', ''), reverse=reverse)
        payload = safe_dict({'success': True, 'stocks': stocks, 'count': len(stocks), 'sectors': list(SECTOR_MAP.keys())})
        cache_set(ck, payload)
        return jsonify(payload)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bist30', methods=['GET'])
def get_bist30_list():
    try:
        stocks = fetch_quick_batch(BIST30_STOCKS, period="5d")
        stocks.sort(key=lambda x: x.get('code', ''))
        return jsonify(safe_dict({'success': True, 'stocks': stocks, 'count': len(stocks)}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>', methods=['GET'])
def get_stock_data(symbol):
    try:
        period = request.args.get('period', '1y')
        symbol = symbol.upper()
        hist = fetch_stock_hist(symbol, period)
        if hist.empty:
            return jsonify({'error': f'Hisse bulunamadi: {symbol}'}), 404
        info = fetch_stock_info(symbol)
        stock_name = info.get('longName', info.get('shortName', BIST100_STOCKS.get(symbol, symbol)))
        current_price = float(hist['Close'].iloc[-1])
        prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100 if prev_close != 0 else 0
        indicators = calc_all_indicators(hist, current_price)
        chart_data = prepare_chart_data(hist)
        fibonacci = calc_fibonacci(hist)
        sr = calc_support_resistance(hist)
        result = {
            'success': True, 'code': ss(symbol), 'name': ss(stock_name),
            'price': sf(current_price), 'change': sf(change), 'changePercent': sf(change_pct),
            'volume': si(hist['Volume'].iloc[-1]), 'avgVolume': si(info.get('averageVolume', 0)),
            'dayHigh': sf(hist['High'].iloc[-1]), 'dayLow': sf(hist['Low'].iloc[-1]),
            'dayOpen': sf(hist['Open'].iloc[-1]), 'prevClose': sf(prev_close),
            'currency': ss(info.get('currency', 'TRY')), 'period': ss(period), 'dataPoints': len(hist),
            'marketCap': si(info.get('marketCap', 0)),
            'peRatio': sf(info.get('trailingPE')) if info.get('trailingPE') else None,
            'pbRatio': sf(info.get('priceToBook')) if info.get('priceToBook') else None,
            'dividendYield': sf(info.get('dividendYield', 0) * 100) if info.get('dividendYield') else None,
            'beta': sf(info.get('beta')) if info.get('beta') else None,
            'week52High': sf(info.get('fiftyTwoWeekHigh')) if info.get('fiftyTwoWeekHigh') else None,
            'week52Low': sf(info.get('fiftyTwoWeekLow')) if info.get('fiftyTwoWeekLow') else None,
            'indicators': indicators, 'chartData': chart_data, 'fibonacci': fibonacci, 'supportResistance': sr,
        }
        return jsonify(safe_dict(result))
    except Exception as e:
        return jsonify({'error': str(e), 'details': traceback.format_exc()}), 500

@app.route('/api/stock/<symbol>/events', methods=['GET'])
def get_stock_events(symbol):
    try:
        events = fetch_stock_events(symbol.upper())
        return jsonify(safe_dict({'success': True, 'symbol': symbol.upper(), 'events': events}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/kap', methods=['GET'])
def get_stock_kap(symbol):
    return jsonify({'success': True, 'symbol': symbol.upper(), 'message': 'KAP entegrasyonu aktif degil.', 'notifications': []})

@app.route('/api/compare', methods=['POST'])
def compare_stocks():
    try:
        data = request.json
        symbols = data.get('symbols', [])
        if len(symbols) < 2:
            return jsonify({'error': 'En az 2 hisse kodu gerekli'}), 400
        period = data.get('period', '6mo')
        results = []
        for sym in symbols[:5]:
            hist = fetch_stock_hist(sym, period)
            if hist.empty: continue
            current = float(hist['Close'].iloc[-1])
            first = float(hist['Close'].iloc[0])
            perf = (current - first) / first * 100
            closes = hist['Close'].values.astype(float)
            vol = float(np.std(np.diff(closes) / closes[:-1]) * np.sqrt(252) * 100) if len(closes) > 1 else 0
            results.append({'code': ss(sym.upper()), 'name': ss(BIST100_STOCKS.get(sym.upper(), sym.upper())), 'price': sf(current), 'performance': sf(perf), 'volatility': sf(vol), 'rsi': calc_rsi(closes)['value'], 'volume': si(hist['Volume'].iloc[-1])})
        return jsonify(safe_dict({'success': True, 'comparison': results, 'period': period}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/screener', methods=['POST'])
def stock_screener():
    try:
        data = request.json or {}
        conditions = data.get('conditions', [])
        sector = data.get('sector', None)
        sort_by = data.get('sort', 'code')
        limit = min(data.get('limit', 50), 100)
        symbols = list(BIST100_STOCKS.keys())
        if sector and sector in SECTOR_MAP:
            symbols = [s for s in SECTOR_MAP[sector] if s in BIST100_STOCKS]
        matches = []
        for sym in symbols:
            try:
                hist = fetch_stock_hist(sym, '6mo')
                if hist.empty or len(hist) < 30: continue
                closes = hist['Close'].values.astype(float)
                highs = hist['High'].values.astype(float)
                lows = hist['Low'].values.astype(float)
                volumes = hist['Volume'].values.astype(float)
                current = float(closes[-1])
                rsi_val = calc_rsi(closes)['value']
                ema_data = calc_ema(closes, current)
                ema20, ema50 = ema_data.get('ema20', current), ema_data.get('ema50', current)
                macd_data = calc_macd(closes)
                avg_vol_20 = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
                vol_ratio = float(volumes[-1]) / avg_vol_20 if avg_vol_20 > 0 else 1.0
                high_20 = float(np.max(highs[-21:-1])) if len(highs) > 21 else float(np.max(highs[:-1]))
                is_breakout = current > high_20
                boll_data = calc_bollinger(closes, current)
                atr_data = calc_atr(highs, lows, closes)
                metrics = {'rsi': rsi_val, 'ema20': ema20, 'ema50': ema50, 'ema_cross': 'bullish' if ema20 > ema50 else 'bearish', 'price_above_ema20': current > ema20, 'price_above_ema50': current > ema50, 'volume_ratio': sf(vol_ratio), 'breakout': is_breakout, 'macd_signal': macd_data['signalType'], 'bollinger_signal': boll_data['signal'], 'atr_pct': atr_data['pct']}
                passed = True
                for cond in conditions:
                    ind, op, tgt = cond.get('indicator', ''), cond.get('operator', '=='), cond.get('value')
                    mv = metrics.get(ind)
                    if mv is None: continue
                    if op == '<' and not (float(mv) < float(tgt)): passed = False; break
                    elif op == '>' and not (float(mv) > float(tgt)): passed = False; break
                    elif op == '<=' and not (float(mv) <= float(tgt)): passed = False; break
                    elif op == '>=' and not (float(mv) >= float(tgt)): passed = False; break
                    elif op == '==' and str(mv) != str(tgt): passed = False; break
                if passed:
                    matches.append({'code': ss(sym), 'name': ss(BIST100_STOCKS.get(sym, sym)), 'price': sf(current), 'changePct': sf((current - float(closes[-2])) / float(closes[-2]) * 100 if len(closes) > 1 else 0), 'metrics': {k: (sf(v) if isinstance(v, (int, float, np.integer, np.floating)) else v) for k, v in metrics.items()}})
            except Exception: continue
        return jsonify(safe_dict({'success': True, 'matches': matches[:limit], 'totalMatches': len(matches), 'scannedStocks': len(symbols), 'availableIndicators': ['rsi', 'ema_cross', 'price_above_ema20', 'price_above_ema50', 'volume_ratio', 'breakout', 'macd_signal', 'bollinger_signal', 'atr_pct'], 'availableSectors': list(SECTOR_MAP.keys())}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── PORTFOLIO ───

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    user_id = request.args.get('user', 'default')
    portfolio = portfolio_store.get(user_id, [])
    positions, total_value, total_cost, daily_pnl = [], 0, 0, 0
    for pos in portfolio:
        try:
            data = fetch_stock_quick(pos['symbol'])
            if not data: continue
            qty, avg_cost, cp = pos['quantity'], pos['avgCost'], data['price']
            mv, cb = cp * qty, avg_cost * qty
            upnl = mv - cb
            upnl_pct = (upnl / cb) * 100 if cb != 0 else 0
            dc = data['change'] * qty
            total_value += mv; total_cost += cb; daily_pnl += dc
            positions.append({'symbol': ss(pos['symbol']), 'name': data['name'], 'quantity': si(qty), 'avgCost': sf(avg_cost), 'currentPrice': sf(cp), 'marketValue': sf(mv), 'costBasis': sf(cb), 'unrealizedPnL': sf(upnl), 'unrealizedPnLPct': sf(upnl_pct), 'dailyChange': sf(dc), 'weight': 0})
        except Exception: pass
    for p in positions:
        p['weight'] = sf(p['marketValue'] / total_value * 100 if total_value > 0 else 0)
    tp = total_value - total_cost
    return jsonify(safe_dict({'success': True, 'userId': user_id, 'positions': positions, 'summary': {'totalValue': sf(total_value), 'totalCost': sf(total_cost), 'totalPnL': sf(tp), 'totalPnLPct': sf((tp / total_cost) * 100 if total_cost > 0 else 0), 'dailyPnL': sf(daily_pnl), 'positionCount': len(positions)}}))

@app.route('/api/portfolio', methods=['POST'])
def add_to_portfolio():
    try:
        data = request.json
        user_id = data.get('user', 'default')
        symbol = data.get('symbol', '').upper()
        quantity = float(data.get('quantity', 0))
        avg_cost = float(data.get('avgCost', 0))
        if not symbol or quantity <= 0 or avg_cost <= 0:
            return jsonify({'error': 'Gecersiz parametreler'}), 400
        if user_id not in portfolio_store:
            portfolio_store[user_id] = []
        existing = next((p for p in portfolio_store[user_id] if p['symbol'] == symbol), None)
        if existing:
            tq = existing['quantity'] + quantity
            existing['avgCost'] = (existing['avgCost'] * existing['quantity'] + avg_cost * quantity) / tq
            existing['quantity'] = tq
        else:
            portfolio_store[user_id].append({'symbol': symbol, 'quantity': quantity, 'avgCost': avg_cost, 'addedAt': datetime.now().isoformat()})
        return jsonify({'success': True, 'message': f'{symbol} portfoye eklendi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['DELETE'])
def remove_from_portfolio():
    try:
        data = request.json
        user_id = data.get('user', 'default')
        symbol = data.get('symbol', '').upper()
        if user_id in portfolio_store:
            portfolio_store[user_id] = [p for p in portfolio_store[user_id] if p['symbol'] != symbol]
        return jsonify({'success': True, 'message': f'{symbol} portfoyden cikarildi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/risk', methods=['GET'])
def portfolio_risk():
    try:
        user_id = request.args.get('user', 'default')
        portfolio = portfolio_store.get(user_id, [])
        if not portfolio:
            return jsonify({'success': True, 'message': 'Portfoy bos', 'risk': {}})
        weights, returns_matrix = [], []
        for pos in portfolio:
            hist = fetch_stock_hist(pos['symbol'], '1y')
            if hist.empty or len(hist) < 30: continue
            closes = hist['Close'].values.astype(float)
            returns_matrix.append((np.diff(closes) / closes[:-1])[-252:])
            weights.append(pos['quantity'] * pos['avgCost'])
        if not returns_matrix:
            return jsonify({'success': True, 'risk': {'message': 'Yetersiz veri'}})
        min_len = min(len(r) for r in returns_matrix)
        returns_matrix = [r[-min_len:] for r in returns_matrix]
        ra = np.array(returns_matrix)
        tw = sum(weights)
        nw = np.array([w / tw for w in weights]) if tw > 0 else np.ones(len(weights)) / len(weights)
        pr = np.dot(nw, ra)
        vol = sf(float(np.std(pr) * np.sqrt(252) * 100))
        cum = np.cumprod(1 + pr)
        rm = np.maximum.accumulate(cum)
        dd = (cum - rm) / rm
        max_dd = sf(float(np.min(dd) * 100))
        ar = sf(float((cum[-1] - 1) * 100)) if len(cum) > 0 else 0.0
        sharpe = sf((ar - 40.0) / vol if vol > 0 else 0)
        return jsonify(safe_dict({'success': True, 'risk': {'volatility': vol, 'maxDrawdown': max_dd, 'annualReturn': ar, 'sharpeRatio': sharpe, 'dataPoints': min_len}}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── BACKTEST ───

def generate_signals(strategy, closes, highs, lows, params):
    n = len(closes)
    signals = [0] * n
    if strategy == 'ma_cross':
        fast, slow = int(params.get('fast', 20)), int(params.get('slow', 50))
        series = pd.Series(closes.copy(), dtype=float)
        ef = series.ewm(span=fast, adjust=False).mean().values
        es = series.ewm(span=slow, adjust=False).mean().values
        for i in range(slow + 1, n):
            if ef[i] > es[i] and ef[i-1] <= es[i-1]: signals[i] = 1
            elif ef[i] < es[i] and ef[i-1] >= es[i-1]: signals[i] = -1
    elif strategy == 'breakout':
        lb = int(params.get('lookback', 20))
        for i in range(lb + 1, n):
            if float(closes[i]) > float(np.max(highs[i-lb:i])): signals[i] = 1
            elif float(closes[i]) < float(np.min(lows[i-lb:i])): signals[i] = -1
    elif strategy == 'mean_reversion':
        rl, rh = float(params.get('rsi_low', 30)), float(params.get('rsi_high', 70))
        for i in range(15, n):
            rv = calc_rsi_single(closes[:i+1])
            if rv is not None:
                if rv < rl: signals[i] = 1
                elif rv > rh: signals[i] = -1
    return signals

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    try:
        data = request.json
        symbol = data.get('symbol', '').upper()
        strategy = data.get('strategy', 'ma_cross')
        params = data.get('params', {})
        period = data.get('period', '2y')
        comm = float(data.get('commission', 0.001))
        slip = float(data.get('slippage', 0.001))
        ic = float(data.get('initialCapital', 100000))
        hist = fetch_stock_hist(symbol, period)
        if hist.empty or len(hist) < 60:
            return jsonify({'error': 'Yetersiz veri'}), 400
        closes = hist['Close'].values.astype(float)
        highs = hist['High'].values.astype(float)
        lows = hist['Low'].values.astype(float)
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]
        signals = generate_signals(strategy, closes, highs, lows, params)
        capital, position, shares, trades, eq_curve, entry_price = ic, 0, 0, [], [], 0
        for i in range(len(closes)):
            price = float(closes[i])
            if signals[i] == 1 and position == 0:
                cost = price * (1 + comm + slip)
                shares = int(capital / cost)
                if shares > 0:
                    capital -= shares * cost; position = 1; entry_price = cost
                    trades.append({'date': dates[i], 'type': 'BUY', 'price': sf(price), 'shares': shares})
            elif signals[i] == -1 and position == 1:
                rev = shares * price * (1 - comm - slip)
                pnl = rev - shares * entry_price
                capital += rev
                trades.append({'date': dates[i], 'type': 'SELL', 'price': sf(price), 'shares': shares, 'pnl': sf(pnl)})
                position = 0; shares = 0
            eq_curve.append({'date': dates[i], 'equity': sf(capital + (shares * price if position == 1 else 0)), 'price': sf(price), 'position': position})
        if position == 1:
            fp = float(closes[-1])
            capital += shares * fp * (1 - comm - slip)
            trades.append({'date': dates[-1], 'type': 'SELL (Final)', 'price': sf(fp), 'shares': shares})
        fe = capital
        tr = (fe - ic) / ic * 100
        ny = len(closes) / 252
        cagr = ((fe / ic) ** (1 / ny) - 1) * 100 if ny > 0 else 0
        ea = np.array([e['equity'] for e in eq_curve])
        rm = np.maximum.accumulate(ea)
        dd = (ea - rm) / rm
        max_dd = float(np.min(dd)) * 100
        winning = sum(1 for t in trades if t.get('pnl', 0) > 0 and t['type'] == 'SELL')
        losing = sum(1 for t in trades if t.get('pnl', 0) <= 0 and t['type'] == 'SELL')
        tt = winning + losing
        wr = (winning / tt * 100) if tt > 0 else 0
        dr = np.diff(ea) / ea[:-1]
        sharpe = float(np.mean(dr) / np.std(dr) * np.sqrt(252)) if np.std(dr) > 0 else 0
        bhr = (float(closes[-1]) / float(closes[0]) - 1) * 100
        return jsonify(safe_dict({'success': True, 'symbol': symbol, 'strategy': strategy, 'params': params, 'results': {'initialCapital': sf(ic), 'finalEquity': sf(fe), 'totalReturn': sf(tr), 'cagr': sf(cagr), 'sharpeRatio': sf(sharpe), 'maxDrawdown': sf(max_dd), 'winRate': sf(wr), 'totalTrades': tt, 'winningTrades': winning, 'losingTrades': losing, 'buyAndHoldReturn': sf(bhr), 'alpha': sf(tr - bhr)}, 'trades': trades[-50:], 'equityCurve': eq_curve, 'availableStrategies': {'ma_cross': {'description': 'MA Crossover', 'params': ['fast', 'slow']}, 'breakout': {'description': 'Breakout', 'params': ['lookback']}, 'mean_reversion': {'description': 'Mean Reversion', 'params': ['rsi_low', 'rsi_high']}}}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── ALERTS ───

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    email = request.args.get('email', None)
    filtered = [a for a in alert_store if a.get('email') == email] if email else alert_store
    return jsonify(safe_dict({'success': True, 'alerts': filtered, 'totalAlerts': len(filtered)}))

@app.route('/api/alerts', methods=['POST'])
def create_alert():
    try:
        data = request.json
        email, symbol, condition = data.get('email'), data.get('symbol', '').upper(), data.get('condition')
        threshold = data.get('threshold')
        if not email or not symbol or not condition:
            return jsonify({'error': 'email, symbol, condition zorunlu'}), 400
        aid = f"alert_{len(alert_store)+1}_{int(time.time())}"
        alert = {'id': aid, 'email': email, 'symbol': symbol, 'condition': condition, 'threshold': sf(threshold) if threshold else None, 'cooldownMinutes': int(data.get('cooldownMinutes', 60)), 'confirmBars': int(data.get('confirmBars', 1)), 'active': True, 'triggered': False, 'triggerCount': 0, 'consecutiveHits': 0, 'lastTriggered': None, 'createdAt': datetime.now().isoformat()}
        alert_store.append(alert)
        return jsonify(safe_dict({'success': True, 'message': f'Uyari olusturuldu: {symbol} {condition}', 'alert': alert}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/<alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    global alert_store
    alert_store = [a for a in alert_store if a.get('id') != alert_id]
    return jsonify({'success': True, 'message': 'Uyari silindi'})

@app.route('/api/alerts/check', methods=['POST'])
def check_alerts():
    try:
        triggered = []
        now = datetime.now()
        for alert in alert_store:
            if not alert.get('active', True): continue
            if alert.get('lastTriggered'):
                if now - datetime.fromisoformat(alert['lastTriggered']) < timedelta(minutes=alert.get('cooldownMinutes', 60)): continue
            symbol, condition, threshold = alert['symbol'], alert['condition'], alert.get('threshold', 0)
            try:
                hist = fetch_stock_hist(symbol, '3mo')
                if hist.empty: continue
                closes = hist['Close'].values.astype(float)
                volumes = hist['Volume'].values.astype(float)
                highs = hist['High'].values.astype(float)
                current = float(closes[-1])
                met = False
                if condition == 'price_above' and current > float(threshold): met = True
                elif condition == 'price_below' and current < float(threshold): met = True
                elif condition == 'rsi_above' and calc_rsi(closes)['value'] > float(threshold): met = True
                elif condition == 'rsi_below' and calc_rsi(closes)['value'] < float(threshold): met = True
                elif condition == 'volume_spike':
                    avg_vol = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
                    if float(volumes[-1]) / avg_vol > float(threshold or 2) if avg_vol > 0 else False: met = True
                elif condition == 'breakout':
                    high_20 = float(np.max(highs[-21:-1])) if len(highs) > 21 else float(np.max(highs[:-1]))
                    if current > high_20: met = True
                elif condition == 'macd_cross_up' and calc_macd(closes)['signalType'] == 'buy': met = True
                elif condition == 'macd_cross_down' and calc_macd(closes)['signalType'] == 'sell': met = True
                if met:
                    alert['consecutiveHits'] = alert.get('consecutiveHits', 0) + 1
                else:
                    alert['consecutiveHits'] = 0
                if alert['consecutiveHits'] >= alert.get('confirmBars', 1):
                    alert['triggered'] = True
                    alert['triggerCount'] = alert.get('triggerCount', 0) + 1
                    alert['lastTriggered'] = now.isoformat()
                    alert['consecutiveHits'] = 0
                    triggered.append({'alertId': alert['id'], 'symbol': symbol, 'condition': condition, 'currentValue': sf(current), 'email': alert['email']})
            except Exception: pass
        return jsonify(safe_dict({'success': True, 'triggered': triggered, 'checkedAlerts': len(alert_store), 'triggeredCount': len(triggered)}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ─── WATCHLIST, SECTORS, SEARCH, HEALTH, STATIC ───

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    user_id = request.args.get('user', 'default')
    stocks = [d for sym in watchlist_store.get(user_id, []) if (d := fetch_stock_quick(sym))]
    return jsonify(safe_dict({'success': True, 'watchlist': stocks}))

@app.route('/api/watchlist', methods=['POST'])
def update_watchlist():
    try:
        data = request.json
        uid, sym, act = data.get('user', 'default'), data.get('symbol', '').upper(), data.get('action', 'add')
        if uid not in watchlist_store: watchlist_store[uid] = []
        if act == 'add' and sym not in watchlist_store[uid]: watchlist_store[uid].append(sym)
        elif act == 'remove': watchlist_store[uid] = [s for s in watchlist_store[uid] if s != sym]
        return jsonify({'success': True, 'watchlist': watchlist_store[uid]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sectors', methods=['GET'])
def get_sectors():
    try:
        sd = []
        for sn, syms in SECTOR_MAP.items():
            changes = [d['changePct'] for sym in syms if (d := fetch_stock_quick(sym))]
            sd.append({'name': sn, 'stockCount': len(syms), 'avgChange': sf(np.mean(changes)) if changes else 0.0, 'symbols': syms})
        sd.sort(key=lambda x: x['avgChange'], reverse=True)
        return jsonify(safe_dict({'success': True, 'sectors': sd}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_stock():
    query = request.args.get('q', '').upper()
    if not query: return jsonify({'results': []})
    results = [{'code': c, 'name': n} for c, n in BIST100_STOCKS.items() if query in c or query in n.upper()]
    return jsonify({'success': True, 'results': results[:10]})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'version': '3.0.0', 'timestamp': datetime.now().isoformat(), 'stockCount': len(BIST100_STOCKS), 'cacheEntries': len(_cache)})

@app.route('/', methods=['GET'])
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/api/docs', methods=['GET'])
def api_docs():
    return jsonify({'name': 'BIST Pro v3.0 API', 'endpoints': ['dashboard', 'indices', 'bist100', 'bist30', 'stock/<symbol>', 'screener', 'portfolio', 'backtest', 'alerts', 'watchlist', 'sectors', 'search', 'health']})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"BIST Pro v3.0 starting on port {port}")
    app.run(host='0.0.0.0', port=port)
