"""
BIST Pro v3.0 - Kapsamli Borsa Analiz Platformu
"""
from flask import Flask, jsonify, request, send_from_directory, make_response
from flask_cors import CORS
import traceback
import os
import time
import threading
import hashlib
import json
from datetime import datetime, timedelta

# yfinance lazy import - hata durumunda bile uygulama ayakta kalsin
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
CORS(app)

# =============================================================================
# CRITICAL: Force JSON error responses - gunicorn+flask HTML error fix
# =============================================================================

@app.after_request
def force_json_errors(response):
    """500 hatalarinda bile JSON donsun, asla HTML donmesin"""
    if response.status_code >= 400 and response.content_type and 'text/html' in response.content_type:
        try:
            data = json.loads(response.get_data(as_text=True))
        except (json.JSONDecodeError, Exception):
            error_msg = f"HTTP {response.status_code}"
            response = make_response(
                json.dumps({"error": error_msg, "status": response.status_code}),
                response.status_code
            )
            response.headers['Content-Type'] = 'application/json'
    return response

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint bulunamadi', 'status': 404}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Sunucu hatasi', 'details': str(e), 'status': 500}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    tb = traceback.format_exc()
    print(f"UNHANDLED: {tb}")
    return jsonify({'error': str(e), 'traceback': tb[:500], 'status': 500}), 500

# =============================================================================
# CONFIG
# =============================================================================
CACHE_TTL = 300
YF_TIMEOUT = 15

def sf(val, decimals=2):
    if val is None:
        return 0.0
    try:
        v = float(val)
        if v != v:  # NaN check
            return 0.0
        return round(v, decimals)
    except (TypeError, ValueError):
        return 0.0

def si(val):
    try:
        return int(val) if val is not None else 0
    except (TypeError, ValueError):
        return 0

def ss(val, default=""):
    return str(val) if val is not None else default

def safe_dict(d):
    if isinstance(d, dict):
        return {str(k): safe_dict(v) for k, v in d.items()}
    elif isinstance(d, list):
        return [safe_dict(item) for item in d]
    elif hasattr(d, 'item'):  # numpy scalar
        return d.item()
    elif isinstance(d, float):
        if d != d:  # NaN
            return 0.0
        return round(d, 4)
    return d

# =============================================================================
# CACHE
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
    return hashlib.md5('|'.join(str(a) for a in args).encode()).hexdigest()

# =============================================================================
# STOCK DATA
# =============================================================================

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

SECTOR_MAP = {
    'bankacilik': ['GARAN', 'ISCTR', 'AKBNK', 'YKBNK', 'VAKBN', 'HALKB', 'TSKB', 'SKBNK'],
    'havacilik': ['THYAO', 'PGSUS', 'TAVHL'],
    'otomotiv': ['TOASO', 'FROTO', 'DOAS', 'TTRAK', 'OTKAR'],
    'enerji': ['TUPRS', 'PETKM', 'AKSEN', 'ENJSA', 'ODAS', 'ZOREN'],
    'holding': ['SAHOL', 'KCHOL', 'DOHOL', 'AGHOL', 'ALARK'],
    'perakende': ['BIMAS', 'SOKM', 'MGROS', 'MAVI'],
    'teknoloji': ['ASELS', 'LOGO', 'NETAS'],
    'telekomunikasyon': ['TCELL', 'TTKOM'],
    'demir_celik': ['EREGL', 'KRDMD', 'CEMTS'],
    'gida': ['ULKER', 'CCOLA', 'AEFES'],
    'insaat': ['ENKAI', 'TKFEN'],
    'gayrimenkul': ['EKGYO', 'TRGYO', 'ISGYO', 'YGYO'],
    'kimya': ['SASA', 'GUBRF', 'HEKTS'],
}

# =============================================================================
# DATA FETCH FUNCTIONS
# =============================================================================

def fetch_stock_hist(symbol, period='1y'):
    if not YF_AVAILABLE:
        return pd.DataFrame()
    ck = cache_key('hist', symbol, period)
    cached = cache_get(ck)
    if cached is not None:
        return cached
    try:
        stock = yf.Ticker(f"{symbol.upper()}.IS")
        hist = stock.history(period=period, timeout=YF_TIMEOUT)
        if hist is not None and not hist.empty:
            cache_set(ck, hist)
            return hist
        return pd.DataFrame()
    except Exception as e:
        print(f"hist error {symbol}: {e}")
        return pd.DataFrame()


def fetch_stock_info(symbol):
    if not YF_AVAILABLE:
        return {}
    ck = cache_key('info', symbol)
    cached = cache_get(ck)
    if cached is not None:
        return cached
    try:
        info = yf.Ticker(f"{symbol.upper()}.IS").info
        if info:
            cache_set(ck, info)
        return info or {}
    except Exception:
        return {}


def fetch_stock_quick(symbol):
    ck = cache_key('quick', symbol)
    cached = cache_get(ck)
    if cached is not None:
        return cached
    if not YF_AVAILABLE:
        return None
    try:
        ticker = yf.Ticker(f"{symbol.upper()}.IS")
        hist = ticker.history(period="5d", timeout=YF_TIMEOUT)
        if hist is None or hist.empty or len(hist) < 2:
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
            'code': symbol.upper(), 'name': BIST100_STOCKS.get(symbol.upper(), symbol.upper()),
            'price': current, 'change': change, 'changePct': change_pct,
            'volume': volume, 'open': day_open, 'high': day_high, 'low': day_low,
            'prevClose': prev, 'gap': gap, 'gapPct': gap_pct,
        }
        cache_set(ck, result)
        return result
    except Exception as e:
        print(f"quick error {symbol}: {e}")
        return None


def fetch_quick_batch(symbols, period="5d"):
    """yf.download ile toplu veri cek - Render icin optimize"""
    if not YF_AVAILABLE:
        return []
    symbols = [s.upper() for s in symbols]
    ck = cache_key('batch', period, len(symbols))
    cached = cache_get(ck)
    if cached is not None:
        return cached
    try:
        tickers_str = " ".join(f"{s}.IS" for s in symbols)
        print(f"[BATCH] Downloading {len(symbols)} tickers...")
        df = yf.download(tickers_str, period=period, group_by="ticker", threads=True, progress=False)
        if df is None or df.empty:
            print("[BATCH] Empty result")
            return []
        results = []
        multi = isinstance(df.columns, pd.MultiIndex)
        print(f"[BATCH] Got data, multi={multi}, shape={df.shape}")
        for s in symbols:
            t = f"{s}.IS"
            try:
                h = None
                if multi:
                    cols_l0 = list(set(df.columns.get_level_values(0)))
                    cols_l1 = list(set(df.columns.get_level_values(1)))
                    if t in cols_l0:
                        h = df[t].dropna(how="all")
                    elif s in cols_l0:
                        h = df[s].dropna(how="all")
                    elif t in cols_l1:
                        h = df.xs(t, axis=1, level=1, drop_level=True).dropna(how="all")
                else:
                    if len(symbols) == 1:
                        h = df.dropna(how="all")
                if h is None or h.empty or len(h) < 2:
                    continue
                if "Close" not in h.columns:
                    continue
                current = sf(h["Close"].iloc[-1])
                prev = sf(h["Close"].iloc[-2])
                if prev == 0:
                    continue
                change = sf(current - prev)
                change_pct = sf((change / prev) * 100)
                volume = si(h["Volume"].iloc[-1]) if "Volume" in h.columns else 0
                day_open = sf(h["Open"].iloc[-1]) if "Open" in h.columns else current
                day_high = sf(h["High"].iloc[-1]) if "High" in h.columns else current
                day_low = sf(h["Low"].iloc[-1]) if "Low" in h.columns else current
                gap = sf(day_open - prev)
                gap_pct = sf((gap / prev) * 100)
                results.append({
                    "code": s, "name": BIST100_STOCKS.get(s, s),
                    "price": current, "prevClose": prev,
                    "change": change, "changePct": change_pct,
                    "volume": volume, "open": day_open,
                    "high": day_high, "low": day_low,
                    "gap": gap, "gapPct": gap_pct,
                })
            except Exception as ex:
                print(f"[BATCH] Skip {s}: {ex}")
                continue
        print(f"[BATCH] Parsed {len(results)}/{len(symbols)} stocks")
        if results:
            cache_set(ck, results)
        return results
    except Exception as e:
        print(f"[BATCH] FATAL: {e}")
        print(traceback.format_exc())
        return []

# =============================================================================
# INDICATORS
# =============================================================================

def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return {'name': 'RSI', 'value': 50.0, 'signal': 'neutral', 'explanation': 'Yetersiz veri'}
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    ag = float(np.mean(gains[-period:]))
    al = float(np.mean(losses[-period:]))
    rsi = 100.0 if al == 0 else sf(100.0 - (100.0 / (1.0 + ag / al)))
    sig = 'buy' if rsi < 30 else ('sell' if rsi > 70 else 'neutral')
    return {'name': 'RSI', 'value': rsi, 'signal': sig, 'explanation': f"RSI {rsi}"}

def calc_rsi_single(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    ag = float(np.mean(np.where(deltas > 0, deltas, 0)[-period:]))
    al = float(np.mean(np.where(deltas < 0, -deltas, 0)[-period:]))
    return 100.0 if al == 0 else sf(100.0 - (100.0 / (1.0 + ag / al)))

def calc_macd(closes):
    if len(closes) < 26:
        return {'name': 'MACD', 'macd': 0.0, 'signal': 0.0, 'histogram': 0.0, 'signalType': 'neutral', 'explanation': 'Yetersiz veri'}
    s = pd.Series(list(closes), dtype=float)
    ml = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    sl = ml.ewm(span=9).mean()
    mv, sv, hv = sf(ml.iloc[-1]), sf(sl.iloc[-1]), sf((ml - sl).iloc[-1])
    sig = 'buy' if mv > sv else ('sell' if mv < sv else 'neutral')
    return {'name': 'MACD', 'macd': mv, 'signal': sv, 'histogram': hv, 'signalType': sig, 'explanation': f"MACD {sig}"}

def calc_macd_history(closes):
    if len(closes) < 26:
        return []
    s = pd.Series(list(closes), dtype=float)
    ml = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    sl = ml.ewm(span=9).mean()
    h = ml - sl
    return [{'macd': sf(ml.iloc[i]), 'signal': sf(sl.iloc[i]), 'histogram': sf(h.iloc[i])} for i in range(26, len(closes))]

def calc_bollinger(closes, cp, period=20):
    if len(closes) < period:
        return {'name': 'Bollinger', 'upper': 0, 'middle': 0, 'lower': 0, 'signal': 'neutral', 'explanation': 'Yetersiz', 'bandwidth': 0}
    r = closes[-period:]
    sma, std = float(np.mean(r)), float(np.std(r))
    u, m, l = sf(sma + 2*std), sf(sma), sf(sma - 2*std)
    bw = sf((u - l) / m * 100 if m else 0)
    sig = 'buy' if float(cp) < l else ('sell' if float(cp) > u else 'neutral')
    return {'name': 'Bollinger', 'upper': u, 'middle': m, 'lower': l, 'bandwidth': bw, 'signal': sig, 'explanation': f"BB {sig}"}

def calc_bollinger_history(closes, period=20):
    r = []
    for i in range(period, len(closes)):
        w = closes[i-period:i]
        sma, std = float(np.mean(w)), float(np.std(w))
        r.append({'upper': sf(sma+2*std), 'middle': sf(sma), 'lower': sf(sma-2*std)})
    return r

def calc_ema(closes, cp):
    result = {'name': 'EMA', 'signal': 'neutral', 'explanation': 'Yetersiz veri'}
    s = pd.Series(list(closes), dtype=float)
    if len(closes) >= 20: result['ema20'] = sf(s.ewm(span=20).mean().iloc[-1])
    if len(closes) >= 50: result['ema50'] = sf(s.ewm(span=50).mean().iloc[-1])
    if len(closes) >= 200: result['ema200'] = sf(s.ewm(span=200).mean().iloc[-1])
    e20, e50 = result.get('ema20', cp), result.get('ema50', cp)
    if float(cp) > e20 > e50: result['signal'], result['explanation'] = 'buy', 'Yukselis trendi'
    elif float(cp) < e20 < e50: result['signal'], result['explanation'] = 'sell', 'Dusus trendi'
    else: result['explanation'] = 'Kararsiz'
    return result

def calc_ema_history(closes):
    s = pd.Series(list(closes), dtype=float)
    e20 = s.ewm(span=20).mean() if len(closes) >= 20 else pd.Series([])
    e50 = s.ewm(span=50).mean() if len(closes) >= 50 else pd.Series([])
    r = []
    for i in range(len(closes)):
        p = {}
        if i < len(e20): p['ema20'] = sf(e20.iloc[i])
        if i < len(e50): p['ema50'] = sf(e50.iloc[i])
        r.append(p)
    return r

def calc_stochastic(closes, highs, lows, period=14):
    if len(closes) < period:
        return {'name': 'Stochastic', 'k': 50, 'd': 50, 'signal': 'neutral', 'explanation': 'Yetersiz'}
    hi, lo, cur = float(np.max(highs[-period:])), float(np.min(lows[-period:])), float(closes[-1])
    k = sf(((cur - lo) / (hi - lo)) * 100 if hi != lo else 50)
    sig = 'buy' if k < 20 else ('sell' if k > 80 else 'neutral')
    return {'name': 'Stochastic', 'k': k, 'd': k, 'signal': sig, 'explanation': f"K={k}"}

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return {'name': 'ATR', 'value': 0, 'pct': 0, 'signal': 'neutral', 'explanation': 'Yetersiz'}
    tr = [max(float(highs[i])-float(lows[i]), abs(float(highs[i])-float(closes[i-1])), abs(float(lows[i])-float(closes[i-1]))) for i in range(1, len(closes))]
    atr = sf(np.mean(tr[-period:]))
    pct = sf(atr / float(closes[-1]) * 100 if closes[-1] else 0)
    return {'name': 'ATR', 'value': atr, 'pct': pct, 'signal': 'neutral', 'explanation': f"ATR%={pct}"}

def calc_adx(highs, lows, closes, period=14):
    n = len(closes)
    if n < period + 1:
        return {'name': 'ADX', 'value': 25, 'plusDI': 0, 'minusDI': 0, 'signal': 'neutral', 'explanation': 'Yetersiz'}
    tr, pdm, mdm = [], [], []
    for i in range(1, n):
        h, l, ph, pl, pc = float(highs[i]), float(lows[i]), float(highs[i-1]), float(lows[i-1]), float(closes[i-1])
        tr.append(max(h-l, abs(h-pc), abs(l-pc)))
        um, dm = h-ph, pl-l
        pdm.append(um if um > dm and um > 0 else 0)
        mdm.append(dm if dm > um and dm > 0 else 0)
    if len(tr) < period:
        return {'name': 'ADX', 'value': 25, 'plusDI': 0, 'minusDI': 0, 'signal': 'neutral', 'explanation': 'Yetersiz'}
    atr_s, pdm_s, mdm_s = float(np.mean(tr[:period])), float(np.mean(pdm[:period])), float(np.mean(mdm[:period]))
    for i in range(period, len(tr)):
        atr_s = (atr_s*(period-1) + tr[i]) / period
        pdm_s = (pdm_s*(period-1) + pdm[i]) / period
        mdm_s = (mdm_s*(period-1) + mdm[i]) / period
    pdi = sf((pdm_s/atr_s)*100 if atr_s else 0)
    mdi = sf((mdm_s/atr_s)*100 if atr_s else 0)
    ds = pdi + mdi
    adx = sf(abs(pdi-mdi)/ds*100 if ds else 0)
    sig = 'buy' if pdi > mdi and adx > 25 else ('sell' if mdi > pdi and adx > 25 else 'neutral')
    return {'name': 'ADX', 'value': adx, 'plusDI': pdi, 'minusDI': mdi, 'signal': sig, 'explanation': f"ADX={adx}"}

def calc_cci(highs, lows, closes, cp, period=20):
    if len(closes) < period:
        return {'name': 'CCI', 'value': 0, 'signal': 'neutral', 'explanation': 'Yetersiz'}
    tp = (np.array(highs,dtype=float) + np.array(lows,dtype=float) + np.array(closes,dtype=float)) / 3
    r = tp[-period:]
    sma = float(np.mean(r))
    md = float(np.mean(np.abs(r - sma)))
    cci = sf((float(tp[-1]) - sma) / (0.015 * md) if md else 0)
    sig = 'buy' if cci < -100 else ('sell' if cci > 100 else 'neutral')
    return {'name': 'CCI', 'value': cci, 'signal': sig, 'explanation': f"CCI={cci}"}

def calc_williams_r(highs, lows, closes, cp, period=14):
    if len(closes) < period:
        return {'name': 'Williams %R', 'value': -50, 'signal': 'neutral', 'explanation': 'Yetersiz'}
    hi, lo = float(np.max(highs[-period:])), float(np.min(lows[-period:]))
    wr = sf(((hi - float(cp)) / (hi - lo)) * -100 if hi != lo else -50)
    sig = 'buy' if wr < -80 else ('sell' if wr > -20 else 'neutral')
    return {'name': 'Williams %R', 'value': wr, 'signal': sig, 'explanation': f"WR={wr}"}

def calc_obv(closes, volumes):
    if len(closes) < 10:
        return {'name': 'OBV', 'value': 0, 'trend': 'neutral', 'signal': 'neutral', 'explanation': 'Yetersiz'}
    obv, vals = 0, [0]
    for i in range(1, len(closes)):
        if float(closes[i]) > float(closes[i-1]): obv += int(volumes[i])
        elif float(closes[i]) < float(closes[i-1]): obv -= int(volumes[i])
        vals.append(obv)
    lb = min(10, len(vals)-1)
    trend = 'up' if vals[-1] > vals[-lb] else 'down'
    return {'name': 'OBV', 'value': si(abs(vals[-1])), 'trend': trend, 'signal': 'buy' if trend == 'up' else 'sell', 'explanation': f"OBV {trend}"}

def calc_volume_profile(hist, bins=20):
    try:
        c = hist['Close'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        mn, mx = float(np.min(c)), float(np.max(c))
        if mn == mx: return {'levels': [], 'poc': sf(mn)}
        edges = np.linspace(mn, mx, bins+1)
        levels, max_vol, poc = [], 0, mn
        for i in range(bins):
            mask = (c >= edges[i]) & (c < edges[i+1])
            vol = float(np.sum(v[mask]))
            mid = float((edges[i]+edges[i+1])/2)
            levels.append({'price': sf(mid), 'volume': si(vol)})
            if vol > max_vol: max_vol, poc = vol, mid
        return {'levels': levels, 'poc': sf(poc)}
    except Exception:
        return {'levels': [], 'poc': 0}

def calc_support_resistance(hist):
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        n = min(90, len(c))
        rh, rl = h[-n:], l[-n:]
        sups, ress = [], []
        for i in range(2, n-2):
            if rh[i] > rh[i-1] and rh[i] > rh[i-2] and rh[i] > rh[i+1] and rh[i] > rh[i+2]: ress.append(float(rh[i]))
            if rl[i] < rl[i-1] and rl[i] < rl[i-2] and rl[i] < rl[i+1] and rl[i] < rl[i+2]: sups.append(float(rl[i]))
        cur = float(c[-1])
        ns = sorted([s for s in sups if s < cur], reverse=True)[:3]
        nr = sorted([r for r in ress if r > cur])[:3]
        return {'supports': [sf(s) for s in ns], 'resistances': [sf(r) for r in nr], 'current': sf(cur), 'explanation': f"S:{sf(ns[0]) if ns else '-'} R:{sf(nr[0]) if nr else '-'}"}
    except Exception:
        return {'supports': [], 'resistances': [], 'current': 0, 'explanation': '-'}

def calc_fibonacci(hist):
    try:
        c = hist['Close'].values.astype(float)
        n = min(90, len(c))
        r = c[-n:]
        hi, lo = float(np.max(r)), float(np.min(r))
        d = hi - lo
        levels = {'0.0': sf(hi), '23.6': sf(hi-d*0.236), '38.2': sf(hi-d*0.382), '50.0': sf(hi-d*0.5), '61.8': sf(hi-d*0.618), '78.6': sf(hi-d*0.786), '100.0': sf(lo)}
        cur = float(c[-1])
        zone = "Belirsiz"
        lk, lv = list(levels.keys()), list(levels.values())
        for i in range(len(lv)-1):
            if cur <= lv[i] and cur >= lv[i+1]:
                zone = f"{lk[i]}%-{lk[i+1]}%"
                break
        return {'levels': levels, 'high': sf(hi), 'low': sf(lo), 'currentZone': zone, 'explanation': f"Fib zone: {zone}"}
    except Exception:
        return {'levels': {}, 'explanation': '-'}

def calc_all_indicators(hist, cp):
    c = hist['Close'].values.astype(float)
    h = hist['High'].values.astype(float)
    l = hist['Low'].values.astype(float)
    v = hist['Volume'].values.astype(float)
    cp = float(cp)
    rsi_hist = []
    for i in range(14, len(c)):
        rv = calc_rsi_single(c[:i+1])
        if rv is not None:
            rsi_hist.append({'date': hist.index[i].strftime('%Y-%m-%d'), 'value': rv})
    ind = {
        'rsi': calc_rsi(c), 'rsiHistory': rsi_hist,
        'macd': calc_macd(c), 'macdHistory': calc_macd_history(c),
        'bollinger': calc_bollinger(c, cp), 'bollingerHistory': calc_bollinger_history(c),
        'stochastic': calc_stochastic(c, h, l),
        'ema': calc_ema(c, cp), 'emaHistory': calc_ema_history(c),
        'atr': calc_atr(h, l, c), 'adx': calc_adx(h, l, c),
        'cci': calc_cci(h, l, c, cp), 'williamsr': calc_williams_r(h, l, c, cp),
        'obv': calc_obv(c, v), 'volumeProfile': calc_volume_profile(hist),
    }
    sigs = [x.get('signal', 'neutral') for x in ind.values() if isinstance(x, dict) and 'signal' in x]
    bc, sc, t = sigs.count('buy'), sigs.count('sell'), len(sigs)
    if bc > sc and bc >= t*0.4: ov, st = 'buy', sf(bc/t*100)
    elif sc > bc and sc >= t*0.4: ov, st = 'sell', sf(sc/t*100)
    else: ov, st = 'neutral', 50.0
    ind['summary'] = {'overall': ov, 'strength': st, 'buySignals': bc, 'sellSignals': sc, 'neutralSignals': t-bc-sc}
    return ind

def prepare_chart_data(hist):
    try:
        cs = [{'date': d.strftime('%Y-%m-%d'), 'open': sf(r['Open']), 'high': sf(r['High']), 'low': sf(r['Low']), 'close': sf(r['Close']), 'volume': si(r['Volume'])} for d, r in hist.iterrows()]
        return {'candlestick': cs, 'dates': [c['date'] for c in cs], 'prices': [c['close'] for c in cs], 'volumes': [c['volume'] for c in cs], 'dataPoints': len(cs)}
    except Exception:
        return {'candlestick': [], 'dates': [], 'prices': [], 'volumes': [], 'dataPoints': 0}

def fetch_stock_events(symbol):
    if not YF_AVAILABLE:
        return {'dividends': [], 'splits': [], 'calendar': {}}
    try:
        t = yf.Ticker(f"{symbol.upper()}.IS")
        divs = [{'date': d.strftime('%Y-%m-%d'), 'amount': sf(v)} for d, v in (t.dividends or pd.Series()).items()]
        splits = [{'date': d.strftime('%Y-%m-%d'), 'ratio': sf(v)} for d, v in (t.splits or pd.Series()).items()]
        return {'dividends': divs[-20:], 'splits': splits, 'calendar': {}, 'totalDividends': len(divs), 'totalSplits': len(splits)}
    except Exception:
        return {'dividends': [], 'splits': [], 'calendar': {}}

# =============================================================================
# STORAGE
# =============================================================================
portfolio_store = {}
alert_store = []
watchlist_store = {}

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'version': '3.0.0', 'yf': YF_AVAILABLE, 'time': datetime.now().isoformat(), 'stocks': len(BIST100_STOCKS), 'cache': len(_cache)})

@app.route('/', methods=['GET'])
def index():
    try:
        return send_from_directory(BASE_DIR, 'index.html')
    except Exception as e:
        return jsonify({'error': 'index.html bulunamadi', 'base_dir': BASE_DIR, 'details': str(e)}), 500

@app.route('/api/docs', methods=['GET'])
def api_docs():
    return jsonify({'name': 'BIST Pro v3.0', 'endpoints': ['/api/health', '/api/dashboard', '/api/indices', '/api/bist100', '/api/bist30', '/api/stock/<sym>', '/api/screener', '/api/portfolio', '/api/backtest', '/api/alerts', '/api/watchlist', '/api/sectors', '/api/search']})

@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    try:
        ck = cache_key('dash_v2')
        cached = cache_get(ck)
        if cached is not None:
            return jsonify(cached)
        stocks = fetch_quick_batch(list(BIST100_STOCKS.keys()), period="5d")
        if not stocks:
            return jsonify({'success': True, 'stockCount': 0, 'movers': {'topGainers': [], 'topLosers': [], 'volumeLeaders': [], 'gapStocks': []}, 'marketBreadth': {'advancing': 0, 'declining': 0, 'unchanged': 0, 'advDecRatio': 0}, 'allStocks': [], 'message': 'Veri yuklenemedi, lutfen sayfayi yenileyin.'})
        sbc = sorted(stocks, key=lambda x: x.get('changePct', 0), reverse=True)
        adv = sum(1 for s in stocks if s.get('changePct', 0) > 0)
        dec = sum(1 for s in stocks if s.get('changePct', 0) < 0)
        payload = safe_dict({'success': True, 'timestamp': datetime.now().isoformat(), 'stockCount': len(stocks),
            'movers': {'topGainers': sbc[:5], 'topLosers': sbc[-5:][::-1], 'volumeLeaders': sorted(stocks, key=lambda x: x.get('volume',0), reverse=True)[:5], 'gapStocks': sorted(stocks, key=lambda x: abs(x.get('gapPct',0)), reverse=True)[:5]},
            'marketBreadth': {'advancing': adv, 'declining': dec, 'unchanged': len(stocks)-adv-dec, 'advDecRatio': sf(adv/dec if dec > 0 else adv)},
            'allStocks': sbc})
        cache_set(ck, payload)
        return jsonify(payload)
    except Exception as e:
        print(f"DASHBOARD ERROR: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'success': False}), 500

@app.route('/api/indices', methods=['GET'])
def get_indices():
    try:
        indices = {}
        for key, tsym, name in [('XU100','XU100.IS','BIST 100'), ('XU030','XU030.IS','BIST 30'), ('XBANK','XBANK.IS','BIST Bankacilik')]:
            try:
                h = yf.Ticker(tsym).history(period="5d", timeout=YF_TIMEOUT)
                if h is not None and not h.empty and len(h) >= 2:
                    indices[key] = {'name': name, 'value': sf(h['Close'].iloc[-1]), 'change': sf(h['Close'].iloc[-1]-h['Close'].iloc[-2]), 'changePct': sf((h['Close'].iloc[-1]-h['Close'].iloc[-2])/h['Close'].iloc[-2]*100), 'high': sf(h['High'].iloc[-1]), 'low': sf(h['Low'].iloc[-1]), 'volume': si(h['Volume'].iloc[-1])}
            except Exception as e:
                print(f"{key}: {e}")
        for tsym, key, name in [("USDTRY=X", "USDTRY", "USD/TRY"), ("GC=F", "GOLD", "Altin")]:
            try:
                h = yf.Ticker(tsym).history(period="5d", timeout=YF_TIMEOUT)
                if h is not None and not h.empty and len(h) >= 2:
                    indices[key] = {'name': name, 'value': sf(h['Close'].iloc[-1], 4), 'change': sf(h['Close'].iloc[-1]-h['Close'].iloc[-2], 4), 'changePct': sf((h['Close'].iloc[-1]-h['Close'].iloc[-2])/h['Close'].iloc[-2]*100)}
            except Exception: pass
        return jsonify(safe_dict({'success': True, 'indices': indices}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/bist100', methods=['GET'])
def get_bist100_list():
    try:
        sf_param = request.args.get('sector', None)
        sort_by = request.args.get('sort', 'code')
        order = request.args.get('order', 'asc')
        ck = cache_key('b100', sf_param or '', sort_by, order)
        cached = cache_get(ck)
        if cached is not None:
            return jsonify(cached)
        syms = list(BIST100_STOCKS.keys())
        if sf_param and sf_param in SECTOR_MAP:
            syms = [s for s in SECTOR_MAP[sf_param] if s in BIST100_STOCKS]
        stocks = fetch_quick_batch(syms, period="5d")
        rev = (order == 'desc')
        key_map = {'change': 'changePct', 'volume': 'volume', 'price': 'price'}
        sk = key_map.get(sort_by, 'code')
        stocks.sort(key=lambda x: x.get(sk, 0) if sk != 'code' else x.get('code', ''), reverse=rev)
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
        cp = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else cp
        ch = cp - prev
        chp = (ch/prev)*100 if prev else 0
        result = {
            'success': True, 'code': symbol, 'name': info.get('longName', info.get('shortName', BIST100_STOCKS.get(symbol, symbol))),
            'price': sf(cp), 'change': sf(ch), 'changePercent': sf(chp),
            'volume': si(hist['Volume'].iloc[-1]), 'avgVolume': si(info.get('averageVolume', 0)),
            'dayHigh': sf(hist['High'].iloc[-1]), 'dayLow': sf(hist['Low'].iloc[-1]),
            'dayOpen': sf(hist['Open'].iloc[-1]), 'prevClose': sf(prev),
            'currency': info.get('currency', 'TRY'), 'period': period, 'dataPoints': len(hist),
            'marketCap': si(info.get('marketCap', 0)),
            'peRatio': sf(info.get('trailingPE')) if info.get('trailingPE') else None,
            'pbRatio': sf(info.get('priceToBook')) if info.get('priceToBook') else None,
            'dividendYield': sf(info.get('dividendYield',0)*100) if info.get('dividendYield') else None,
            'beta': sf(info.get('beta')) if info.get('beta') else None,
            'week52High': sf(info.get('fiftyTwoWeekHigh')) if info.get('fiftyTwoWeekHigh') else None,
            'week52Low': sf(info.get('fiftyTwoWeekLow')) if info.get('fiftyTwoWeekLow') else None,
            'indicators': calc_all_indicators(hist, cp), 'chartData': prepare_chart_data(hist),
            'fibonacci': calc_fibonacci(hist), 'supportResistance': calc_support_resistance(hist),
        }
        return jsonify(safe_dict(result))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/events', methods=['GET'])
def get_stock_events(symbol):
    try:
        return jsonify(safe_dict({'success': True, 'symbol': symbol.upper(), 'events': fetch_stock_events(symbol.upper())}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/kap', methods=['GET'])
def get_stock_kap(symbol):
    return jsonify({'success': True, 'symbol': symbol.upper(), 'message': 'KAP entegrasyonu aktif degil.', 'notifications': []})

@app.route('/api/compare', methods=['POST'])
def compare_stocks():
    try:
        data = request.json or {}
        symbols = data.get('symbols', [])
        if len(symbols) < 2: return jsonify({'error': 'En az 2 hisse'}), 400
        period = data.get('period', '6mo')
        results = []
        for sym in symbols[:5]:
            hist = fetch_stock_hist(sym, period)
            if hist.empty: continue
            c = hist['Close'].values.astype(float)
            cur, first = float(c[-1]), float(c[0])
            results.append({'code': sym.upper(), 'name': BIST100_STOCKS.get(sym.upper(), sym), 'price': sf(cur), 'performance': sf((cur-first)/first*100), 'volatility': sf(float(np.std(np.diff(c)/c[:-1])*np.sqrt(252)*100) if len(c)>1 else 0), 'rsi': calc_rsi(c)['value'], 'volume': si(hist['Volume'].iloc[-1])})
        return jsonify(safe_dict({'success': True, 'comparison': results, 'period': period}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/screener', methods=['POST'])
def stock_screener():
    try:
        data = request.json or {}
        conditions = data.get('conditions', [])
        sector = data.get('sector', None)
        limit = min(data.get('limit', 50), 100)
        syms = list(BIST100_STOCKS.keys())
        if sector and sector in SECTOR_MAP:
            syms = [s for s in SECTOR_MAP[sector] if s in BIST100_STOCKS]
        matches = []
        for sym in syms:
            try:
                hist = fetch_stock_hist(sym, '6mo')
                if hist.empty or len(hist) < 30: continue
                c, h, l, v = hist['Close'].values.astype(float), hist['High'].values.astype(float), hist['Low'].values.astype(float), hist['Volume'].values.astype(float)
                cur = float(c[-1])
                ed = calc_ema(c, cur)
                e20, e50 = ed.get('ema20', cur), ed.get('ema50', cur)
                avg20 = float(np.mean(v[-20:])) if len(v) >= 20 else float(np.mean(v))
                vr = float(v[-1])/avg20 if avg20 > 0 else 1
                h20 = float(np.max(h[-21:-1])) if len(h) > 21 else float(np.max(h[:-1]))
                m = {'rsi': calc_rsi(c)['value'], 'ema_cross': 'bullish' if e20>e50 else 'bearish', 'price_above_ema20': cur>e20, 'price_above_ema50': cur>e50, 'volume_ratio': sf(vr), 'breakout': cur>h20, 'macd_signal': calc_macd(c)['signalType'], 'atr_pct': calc_atr(h,l,c)['pct']}
                ok = True
                for cd in conditions:
                    ind, op, tgt = cd.get('indicator',''), cd.get('operator','=='), cd.get('value')
                    mv = m.get(ind)
                    if mv is None: continue
                    if op == '<' and not (float(mv) < float(tgt)): ok=False; break
                    elif op == '>' and not (float(mv) > float(tgt)): ok=False; break
                    elif op == '==' and str(mv) != str(tgt): ok=False; break
                if ok:
                    matches.append({'code': sym, 'name': BIST100_STOCKS.get(sym, sym), 'price': sf(cur), 'changePct': sf((cur-float(c[-2]))/float(c[-2])*100 if len(c)>1 else 0), 'metrics': {k: (sf(v2) if isinstance(v2,(int,float)) else v2) for k,v2 in m.items()}})
            except Exception: continue
        return jsonify(safe_dict({'success': True, 'matches': matches[:limit], 'totalMatches': len(matches), 'scannedStocks': len(syms), 'availableIndicators': ['rsi','ema_cross','price_above_ema20','price_above_ema50','volume_ratio','breakout','macd_signal','atr_pct'], 'availableSectors': list(SECTOR_MAP.keys())}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    uid = request.args.get('user', 'default')
    pf = portfolio_store.get(uid, [])
    pos, tv, tc, dp = [], 0, 0, 0
    for p in pf:
        try:
            d = fetch_stock_quick(p['symbol'])
            if not d: continue
            q, ac, cp = p['quantity'], p['avgCost'], d['price']
            mv, cb = cp*q, ac*q
            upnl = mv - cb
            dc = d['change'] * q
            tv += mv; tc += cb; dp += dc
            pos.append({'symbol': p['symbol'], 'name': d['name'], 'quantity': si(q), 'avgCost': sf(ac), 'currentPrice': sf(cp), 'marketValue': sf(mv), 'costBasis': sf(cb), 'unrealizedPnL': sf(upnl), 'unrealizedPnLPct': sf(upnl/cb*100 if cb else 0), 'dailyChange': sf(dc), 'weight': 0})
        except Exception: pass
    for p in pos: p['weight'] = sf(p['marketValue']/tv*100 if tv > 0 else 0)
    tp = tv - tc
    return jsonify(safe_dict({'success': True, 'userId': uid, 'positions': pos, 'summary': {'totalValue': sf(tv), 'totalCost': sf(tc), 'totalPnL': sf(tp), 'totalPnLPct': sf(tp/tc*100 if tc else 0), 'dailyPnL': sf(dp), 'positionCount': len(pos)}}))

@app.route('/api/portfolio', methods=['POST'])
def add_to_portfolio():
    try:
        d = request.json
        uid, sym, qty, ac = d.get('user','default'), d.get('symbol','').upper(), float(d.get('quantity',0)), float(d.get('avgCost',0))
        if not sym or qty <= 0 or ac <= 0: return jsonify({'error': 'Gecersiz'}), 400
        if uid not in portfolio_store: portfolio_store[uid] = []
        ex = next((p for p in portfolio_store[uid] if p['symbol']==sym), None)
        if ex:
            tq = ex['quantity']+qty
            ex['avgCost'] = (ex['avgCost']*ex['quantity']+ac*qty)/tq
            ex['quantity'] = tq
        else:
            portfolio_store[uid].append({'symbol': sym, 'quantity': qty, 'avgCost': ac, 'addedAt': datetime.now().isoformat()})
        return jsonify({'success': True, 'message': f'{sym} eklendi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['DELETE'])
def remove_from_portfolio():
    try:
        d = request.json
        uid, sym = d.get('user','default'), d.get('symbol','').upper()
        if uid in portfolio_store:
            portfolio_store[uid] = [p for p in portfolio_store[uid] if p['symbol'] != sym]
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio/risk', methods=['GET'])
def portfolio_risk():
    try:
        uid = request.args.get('user', 'default')
        pf = portfolio_store.get(uid, [])
        if not pf: return jsonify({'success': True, 'risk': {}})
        wts, rm = [], []
        for p in pf:
            hist = fetch_stock_hist(p['symbol'], '1y')
            if hist.empty or len(hist) < 30: continue
            c = hist['Close'].values.astype(float)
            rm.append((np.diff(c)/c[:-1])[-252:])
            wts.append(p['quantity']*p['avgCost'])
        if not rm: return jsonify({'success': True, 'risk': {'message': 'Yetersiz'}})
        ml = min(len(r) for r in rm)
        rm = [r[-ml:] for r in rm]
        ra = np.array(rm)
        tw = sum(wts)
        nw = np.array([w/tw for w in wts]) if tw else np.ones(len(wts))/len(wts)
        pr = np.dot(nw, ra)
        vol = sf(float(np.std(pr)*np.sqrt(252)*100))
        cum = np.cumprod(1+pr)
        mx = np.maximum.accumulate(cum)
        mdd = sf(float(np.min((cum-mx)/mx)*100))
        ar = sf(float((cum[-1]-1)*100)) if len(cum) else 0
        sh = sf((ar-40)/vol if vol else 0)
        return jsonify(safe_dict({'success': True, 'risk': {'volatility': vol, 'maxDrawdown': mdd, 'annualReturn': ar, 'sharpeRatio': sh, 'dataPoints': ml}}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def generate_signals(strategy, closes, highs, lows, params):
    n = len(closes)
    sigs = [0]*n
    if strategy == 'ma_cross':
        f, s = int(params.get('fast',20)), int(params.get('slow',50))
        se = pd.Series(list(closes), dtype=float)
        ef, es = se.ewm(span=f).mean().values, se.ewm(span=s).mean().values
        for i in range(s+1, n):
            if ef[i]>es[i] and ef[i-1]<=es[i-1]: sigs[i]=1
            elif ef[i]<es[i] and ef[i-1]>=es[i-1]: sigs[i]=-1
    elif strategy == 'breakout':
        lb = int(params.get('lookback',20))
        for i in range(lb+1, n):
            if float(closes[i])>float(np.max(highs[i-lb:i])): sigs[i]=1
            elif float(closes[i])<float(np.min(lows[i-lb:i])): sigs[i]=-1
    elif strategy == 'mean_reversion':
        rl, rh = float(params.get('rsi_low',30)), float(params.get('rsi_high',70))
        for i in range(15, n):
            rv = calc_rsi_single(closes[:i+1])
            if rv is not None:
                if rv < rl: sigs[i]=1
                elif rv > rh: sigs[i]=-1
    return sigs

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    try:
        d = request.json
        sym, strat, params, period = d.get('symbol','').upper(), d.get('strategy','ma_cross'), d.get('params',{}), d.get('period','2y')
        comm, slip, ic = float(d.get('commission',0.001)), float(d.get('slippage',0.001)), float(d.get('initialCapital',100000))
        hist = fetch_stock_hist(sym, period)
        if hist.empty or len(hist)<60: return jsonify({'error': 'Yetersiz veri'}), 400
        c, h, l = hist['Close'].values.astype(float), hist['High'].values.astype(float), hist['Low'].values.astype(float)
        dates = [x.strftime('%Y-%m-%d') for x in hist.index]
        sigs = generate_signals(strat, c, h, l, params)
        cap, pos, sh, trades, eq, ep = ic, 0, 0, [], [], 0
        for i in range(len(c)):
            p = float(c[i])
            if sigs[i]==1 and pos==0:
                cost = p*(1+comm+slip)
                sh = int(cap/cost)
                if sh > 0: cap -= sh*cost; pos=1; ep=cost; trades.append({'date':dates[i],'type':'BUY','price':sf(p),'shares':sh})
            elif sigs[i]==-1 and pos==1:
                rev = sh*p*(1-comm-slip)
                pnl = rev - sh*ep
                cap += rev; trades.append({'date':dates[i],'type':'SELL','price':sf(p),'shares':sh,'pnl':sf(pnl)}); pos=0; sh=0
            eq.append({'date':dates[i],'equity':sf(cap+(sh*p if pos==1 else 0)),'price':sf(p),'position':pos})
        if pos==1: cap += sh*float(c[-1])*(1-comm-slip); trades.append({'date':dates[-1],'type':'SELL','price':sf(float(c[-1])),'shares':sh})
        fe = cap
        tr = (fe-ic)/ic*100
        ny = len(c)/252
        cagr = ((fe/ic)**(1/ny)-1)*100 if ny>0 else 0
        ea = np.array([e['equity'] for e in eq])
        rm = np.maximum.accumulate(ea)
        mdd = float(np.min((ea-rm)/rm))*100
        w = sum(1 for t in trades if t.get('pnl',0)>0 and t['type']=='SELL')
        lo = sum(1 for t in trades if t.get('pnl',0)<=0 and t['type']=='SELL')
        tt = w+lo
        wr = w/tt*100 if tt else 0
        dr = np.diff(ea)/ea[:-1]
        sharpe = float(np.mean(dr)/np.std(dr)*np.sqrt(252)) if np.std(dr)>0 else 0
        bhr = (float(c[-1])/float(c[0])-1)*100
        return jsonify(safe_dict({'success': True, 'symbol': sym, 'strategy': strat, 'params': params, 'results': {'initialCapital':sf(ic),'finalEquity':sf(fe),'totalReturn':sf(tr),'cagr':sf(cagr),'sharpeRatio':sf(sharpe),'maxDrawdown':sf(mdd),'winRate':sf(wr),'totalTrades':tt,'winningTrades':w,'losingTrades':lo,'buyAndHoldReturn':sf(bhr),'alpha':sf(tr-bhr)}, 'trades': trades[-50:], 'equityCurve': eq, 'availableStrategies': {'ma_cross':{'description':'MA Cross','params':['fast','slow']},'breakout':{'description':'Breakout','params':['lookback']},'mean_reversion':{'description':'Mean Rev','params':['rsi_low','rsi_high']}}}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    email = request.args.get('email')
    f = [a for a in alert_store if a.get('email')==email] if email else alert_store
    return jsonify(safe_dict({'success': True, 'alerts': f, 'totalAlerts': len(f)}))

@app.route('/api/alerts', methods=['POST'])
def create_alert():
    try:
        d = request.json
        email, sym, cond = d.get('email'), d.get('symbol','').upper(), d.get('condition')
        if not email or not sym or not cond: return jsonify({'error': 'email,symbol,condition zorunlu'}), 400
        a = {'id': f"a{len(alert_store)+1}_{int(time.time())}", 'email': email, 'symbol': sym, 'condition': cond, 'threshold': sf(d.get('threshold')) if d.get('threshold') else None, 'cooldownMinutes': int(d.get('cooldownMinutes',60)), 'confirmBars': int(d.get('confirmBars',1)), 'active': True, 'triggered': False, 'triggerCount': 0, 'consecutiveHits': 0, 'lastTriggered': None, 'createdAt': datetime.now().isoformat()}
        alert_store.append(a)
        return jsonify(safe_dict({'success': True, 'alert': a}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/<aid>', methods=['DELETE'])
def delete_alert(aid):
    global alert_store
    alert_store = [a for a in alert_store if a.get('id') != aid]
    return jsonify({'success': True})

@app.route('/api/alerts/check', methods=['POST'])
def check_alerts():
    return jsonify({'success': True, 'triggered': [], 'checkedAlerts': len(alert_store), 'triggeredCount': 0})

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    uid = request.args.get('user', 'default')
    stocks = [d for sym in watchlist_store.get(uid, []) if (d := fetch_stock_quick(sym))]
    return jsonify(safe_dict({'success': True, 'watchlist': stocks}))

@app.route('/api/watchlist', methods=['POST'])
def update_watchlist():
    try:
        d = request.json
        uid, sym, act = d.get('user','default'), d.get('symbol','').upper(), d.get('action','add')
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
            changes = []
            for sym in syms:
                d = fetch_stock_quick(sym)
                if d: changes.append(d['changePct'])
            sd.append({'name': sn, 'stockCount': len(syms), 'avgChange': sf(np.mean(changes)) if changes else 0, 'symbols': syms})
        sd.sort(key=lambda x: x['avgChange'], reverse=True)
        return jsonify(safe_dict({'success': True, 'sectors': sd}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_stock():
    q = request.args.get('q', '').upper()
    if not q: return jsonify({'results': []})
    return jsonify({'success': True, 'results': [{'code': c, 'name': n} for c, n in BIST100_STOCKS.items() if q in c or q in n.upper()][:10]})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"BIST Pro v3.0 on port {port}")
    app.run(host='0.0.0.0', port=port)
