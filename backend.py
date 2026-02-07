"""
BIST Pro v3.0 - Render Free Tier Optimized
Strateji: Background thread veri ceker, API her zaman cache'den doner.
Hicbir request yfinance beklemez = timeout IMKANSIZ.
"""
from flask import Flask, jsonify, request, send_from_directory, make_response
from flask_cors import CORS
import traceback, os, time, threading, hashlib, json
from datetime import datetime, timedelta

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
CORS(app)

# =====================================================================
# FORCE JSON - gunicorn HTML override
# =====================================================================
@app.after_request
def after(resp):
    if resp.status_code >= 400:
        ct = resp.content_type or ''
        if 'text/html' in ct:
            body = json.dumps({"error": f"HTTP {resp.status_code}", "status": resp.status_code})
            resp = make_response(body, resp.status_code)
            resp.headers['Content-Type'] = 'application/json'
    return resp

@app.errorhandler(404)
def e404(e):
    return jsonify({'error': 'Bulunamadi', 'status': 404}), 404

@app.errorhandler(500)
def e500(e):
    return jsonify({'error': 'Sunucu hatasi', 'details': str(e), 'status': 500}), 500

@app.errorhandler(Exception)
def eall(e):
    print(f"ERR: {traceback.format_exc()}")
    return jsonify({'error': str(e), 'status': 500}), 500

# =====================================================================
# HELPERS
# =====================================================================
def sf(v, d=2):
    try:
        f = float(v)
        return 0.0 if f != f else round(f, d)
    except:
        return 0.0

def si(v):
    try:
        return int(v)
    except:
        return 0

def safe_dict(d):
    if isinstance(d, dict):
        return {str(k): safe_dict(v) for k, v in d.items()}
    if isinstance(d, list):
        return [safe_dict(i) for i in d]
    if hasattr(d, 'item'):
        return d.item()
    if isinstance(d, float):
        return 0.0 if d != d else round(d, 4)
    return d

# =====================================================================
# STOCK DATA
# =====================================================================
BIST100_STOCKS = {
    'THYAO':'Turk Hava Yollari','GARAN':'Garanti BBVA','ISCTR':'Is Bankasi (C)',
    'AKBNK':'Akbank','TUPRS':'Tupras','BIMAS':'BIM','SAHOL':'Sabanci Holding',
    'KCHOL':'Koc Holding','EREGL':'Eregli Demir Celik','SISE':'Sise Cam',
    'PETKM':'Petkim','ASELS':'Aselsan','TOASO':'Tofas','TCELL':'Turkcell',
    'ENKAI':'Enka Insaat','KOZAL':'Koza Altin','KRDMD':'Kardemir (D)',
    'TTKOM':'Turk Telekom','ARCLK':'Arcelik','SOKM':'Sok Marketler',
    'HEKTS':'Hektas','PGSUS':'Pegasus','TAVHL':'TAV Havalimanlari',
    'DOHOL':'Dogan Holding','VESTL':'Vestel','MGROS':'Migros',
    'FROTO':'Ford Otosan','SASA':'SASA Polyester','EKGYO':'Emlak Konut GYO',
    'TTRAK':'Turk Traktor','AEFES':'Anadolu Efes','YKBNK':'Yapi Kredi',
    'VAKBN':'Vakifbank','HALKB':'Halkbank','TSKB':'TSKB',
    'ALARK':'Alarko Holding','OTKAR':'Otokar','CEMTS':'Cemtas',
    'CIMSA':'Cimsa','GUBRF':'Gubre Fabrikalari','KONTR':'Kontrolmatik',
    'KOZAA':'Koza Anadolu','SKBNK':'Sekerbank','ISGYO':'Is GYO',
    'ULKER':'Ulker','CCOLA':'Coca-Cola Icecek','TRGYO':'Torunlar GYO',
    'ENJSA':'Enerjisa','AKSEN':'Aksa Enerji','ODAS':'Odas Elektrik',
    'AGHOL':'AG Anadolu Grubu','DOAS':'Dogus Otomotiv','BERA':'Bera Holding',
    'MPARK':'MLP Saglik','ANHYT':'Anadolu Hayat','BTCIM':'Baticim',
    'BUCIM':'Bursa Cimento','EGEEN':'Ege Endustri','ISMEN':'Is Yatirim Menkul',
    'LOGO':'Logo Yazilim','MAVI':'Mavi Giyim','NETAS':'Netas Telekom',
    'TKFEN':'Tekfen Holding','TURSG':'Turkiye Sigorta','ZOREN':'Zorlu Enerji'
}

BIST30 = ['THYAO','GARAN','ISCTR','AKBNK','TUPRS','BIMAS','SAHOL','KCHOL',
    'EREGL','SISE','ASELS','TCELL','ENKAI','FROTO','PGSUS','TAVHL',
    'TOASO','ARCLK','PETKM','TTKOM','KOZAL','YKBNK','VAKBN','HALKB',
    'EKGYO','SASA','MGROS','SOKM','DOHOL','VESTL']

SECTOR_MAP = {
    'bankacilik':['GARAN','ISCTR','AKBNK','YKBNK','VAKBN','HALKB','TSKB','SKBNK'],
    'havacilik':['THYAO','PGSUS','TAVHL'],
    'otomotiv':['TOASO','FROTO','DOAS','TTRAK','OTKAR'],
    'enerji':['TUPRS','PETKM','AKSEN','ENJSA','ODAS','ZOREN'],
    'holding':['SAHOL','KCHOL','DOHOL','AGHOL','ALARK'],
    'perakende':['BIMAS','SOKM','MGROS','MAVI'],
    'teknoloji':['ASELS','LOGO','NETAS'],
    'telekom':['TCELL','TTKOM'],
    'demir_celik':['EREGL','KRDMD','CEMTS'],
    'gida':['ULKER','CCOLA','AEFES'],
    'insaat':['ENKAI','TKFEN'],
    'gayrimenkul':['EKGYO','TRGYO','ISGYO'],
}

# =====================================================================
# CACHE - Thread-safe
# =====================================================================
_stock_cache = {}      # symbol -> {data, ts}
_hist_cache = {}       # symbol_period -> {data, ts}
_loader_status = {'running': False, 'loaded': 0, 'total': 0, 'lastRun': None, 'errors': []}
_lock = threading.Lock()
CACHE_TTL = 300  # 5 dk

def _cache_get(store, key):
    with _lock:
        item = store.get(key)
        if item and time.time() - item['ts'] < CACHE_TTL:
            return item['data']
    return None

def _cache_set(store, key, data):
    with _lock:
        store[key] = {'data': data, 'ts': time.time()}

# =====================================================================
# BACKGROUND DATA LOADER - Asla request thread'inde calismaz
# =====================================================================
def _fetch_one_stock(symbol):
    """Tek hisse ver - timeout 10sn"""
    try:
        ticker = yf.Ticker(f"{symbol}.IS")
        hist = ticker.history(period="5d", timeout=10)
        if hist is None or hist.empty or len(hist) < 2:
            return None
        cur = sf(hist['Close'].iloc[-1])
        prev = sf(hist['Close'].iloc[-2])
        if prev == 0:
            return None
        ch = sf(cur - prev)
        chp = sf((ch / prev) * 100)
        vol = si(hist['Volume'].iloc[-1])
        o = sf(hist['Open'].iloc[-1])
        h = sf(hist['High'].iloc[-1])
        l = sf(hist['Low'].iloc[-1])
        return {
            'code': symbol, 'name': BIST100_STOCKS.get(symbol, symbol),
            'price': cur, 'prevClose': prev, 'change': ch, 'changePct': chp,
            'volume': vol, 'open': o, 'high': h, 'low': l,
            'gap': sf(o - prev), 'gapPct': sf((o - prev) / prev * 100 if prev else 0),
        }
    except Exception as e:
        print(f"  [FETCH] {symbol} hata: {e}")
        return None


def _background_loader():
    """Arka planda BIST30 hisselerini teker teker ceker"""
    if not YF_OK:
        print("[LOADER] yfinance yok, iptal")
        return
    _loader_status['running'] = True
    _loader_status['total'] = len(BIST30)
    _loader_status['loaded'] = 0
    _loader_status['errors'] = []
    print(f"[LOADER] Basliyor - {len(BIST30)} hisse")
    for i, sym in enumerate(BIST30):
        try:
            data = _fetch_one_stock(sym)
            if data:
                _cache_set(_stock_cache, sym, data)
                _loader_status['loaded'] = i + 1
                print(f"  [LOADER] {sym} OK ({i+1}/{len(BIST30)})")
            else:
                _loader_status['errors'].append(sym)
                print(f"  [LOADER] {sym} bos ({i+1}/{len(BIST30)})")
            time.sleep(0.3)  # Rate limit - Yahoo'yu kirma
        except Exception as e:
            _loader_status['errors'].append(sym)
            print(f"  [LOADER] {sym} HATA: {e}")
            time.sleep(1)
    _loader_status['running'] = False
    _loader_status['lastRun'] = datetime.now().isoformat()
    print(f"[LOADER] Bitti - {_loader_status['loaded']}/{len(BIST30)} basarili")


def start_background_loader():
    """Background loader'i baslat (1 kere)"""
    t = threading.Thread(target=_background_loader, daemon=True)
    t.start()
    return t


def get_cached_stocks(symbols=None):
    """Cache'deki hisseleri getir - ASLA yfinance cagirmaz"""
    with _lock:
        if symbols:
            return [_stock_cache[s]['data'] for s in symbols if s in _stock_cache and time.time() - _stock_cache[s]['ts'] < CACHE_TTL]
        return [v['data'] for v in _stock_cache.values() if time.time() - v['ts'] < CACHE_TTL]


# =====================================================================
# INDICATORS
# =====================================================================
def calc_rsi(closes, period=14):
    if len(closes) < period + 1:
        return {'name':'RSI','value':50.0,'signal':'neutral','explanation':'Yetersiz veri'}
    d = np.diff(closes)
    g = np.where(d > 0, d, 0)
    l = np.where(d < 0, -d, 0)
    ag, al = float(np.mean(g[-period:])), float(np.mean(l[-period:]))
    rsi = 100.0 if al == 0 else sf(100 - 100 / (1 + ag / al))
    sig = 'buy' if rsi < 30 else ('sell' if rsi > 70 else 'neutral')
    return {'name':'RSI','value':rsi,'signal':sig,'explanation':f'RSI {rsi}'}

def calc_rsi_single(closes, period=14):
    if len(closes) < period + 1:
        return None
    d = np.diff(closes)
    ag = float(np.mean(np.where(d>0,d,0)[-period:]))
    al = float(np.mean(np.where(d<0,-d,0)[-period:]))
    return 100.0 if al == 0 else sf(100 - 100 / (1 + ag / al))

def calc_macd(closes):
    if len(closes) < 26:
        return {'name':'MACD','macd':0,'signal':0,'histogram':0,'signalType':'neutral','explanation':'Yetersiz'}
    s = pd.Series(list(closes), dtype=float)
    ml = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    sl = ml.ewm(span=9).mean()
    mv, sv, hv = sf(ml.iloc[-1]), sf(sl.iloc[-1]), sf((ml-sl).iloc[-1])
    sig = 'buy' if mv > sv else ('sell' if mv < sv else 'neutral')
    return {'name':'MACD','macd':mv,'signal':sv,'histogram':hv,'signalType':sig,'explanation':f'MACD {sig}'}

def calc_macd_history(closes):
    if len(closes) < 26:
        return []
    s = pd.Series(list(closes), dtype=float)
    ml = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    sl = ml.ewm(span=9).mean()
    h = ml - sl
    return [{'macd':sf(ml.iloc[i]),'signal':sf(sl.iloc[i]),'histogram':sf(h.iloc[i])} for i in range(26,len(closes))]

def calc_bollinger(closes, cp, period=20):
    if len(closes) < period:
        return {'name':'Bollinger','upper':0,'middle':0,'lower':0,'signal':'neutral','explanation':'Yetersiz','bandwidth':0}
    r = closes[-period:]
    sma, std = float(np.mean(r)), float(np.std(r))
    u, m, lo = sf(sma+2*std), sf(sma), sf(sma-2*std)
    bw = sf((u-lo)/m*100 if m else 0)
    sig = 'buy' if float(cp)<lo else ('sell' if float(cp)>u else 'neutral')
    return {'name':'Bollinger','upper':u,'middle':m,'lower':lo,'bandwidth':bw,'signal':sig,'explanation':f'BB {sig}'}

def calc_bollinger_history(closes, period=20):
    r = []
    for i in range(period, len(closes)):
        w = closes[i-period:i]
        sma, std = float(np.mean(w)), float(np.std(w))
        r.append({'upper':sf(sma+2*std),'middle':sf(sma),'lower':sf(sma-2*std)})
    return r

def calc_ema(closes, cp):
    result = {'name':'EMA','signal':'neutral','explanation':'Yetersiz'}
    s = pd.Series(list(closes), dtype=float)
    if len(closes) >= 20: result['ema20'] = sf(s.ewm(span=20).mean().iloc[-1])
    if len(closes) >= 50: result['ema50'] = sf(s.ewm(span=50).mean().iloc[-1])
    if len(closes) >= 200: result['ema200'] = sf(s.ewm(span=200).mean().iloc[-1])
    e20, e50 = result.get('ema20', cp), result.get('ema50', cp)
    if float(cp) > e20 > e50: result['signal'], result['explanation'] = 'buy', 'Yukselis'
    elif float(cp) < e20 < e50: result['signal'], result['explanation'] = 'sell', 'Dusus'
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
        return {'name':'Stochastic','k':50,'d':50,'signal':'neutral','explanation':'Yetersiz'}
    hi, lo, cur = float(np.max(highs[-period:])), float(np.min(lows[-period:])), float(closes[-1])
    k = sf(((cur-lo)/(hi-lo))*100 if hi!=lo else 50)
    sig = 'buy' if k<20 else ('sell' if k>80 else 'neutral')
    return {'name':'Stochastic','k':k,'d':k,'signal':sig,'explanation':f'K={k}'}

def calc_atr(highs, lows, closes, period=14):
    if len(closes) < period+1:
        return {'name':'ATR','value':0,'pct':0,'signal':'neutral','explanation':'Yetersiz'}
    tr = [max(float(highs[i])-float(lows[i]), abs(float(highs[i])-float(closes[i-1])), abs(float(lows[i])-float(closes[i-1]))) for i in range(1,len(closes))]
    atr = sf(np.mean(tr[-period:]))
    pct = sf(atr/float(closes[-1])*100 if closes[-1] else 0)
    return {'name':'ATR','value':atr,'pct':pct,'signal':'neutral','explanation':f'ATR%={pct}'}

def calc_adx(highs, lows, closes, period=14):
    n = len(closes)
    if n < period+1:
        return {'name':'ADX','value':25,'plusDI':0,'minusDI':0,'signal':'neutral','explanation':'Yetersiz'}
    tr, pdm, mdm = [], [], []
    for i in range(1, n):
        hv, lv, phv, plv, pcv = float(highs[i]), float(lows[i]), float(highs[i-1]), float(lows[i-1]), float(closes[i-1])
        tr.append(max(hv-lv, abs(hv-pcv), abs(lv-pcv)))
        um, dm = hv-phv, plv-lv
        pdm.append(um if um>dm and um>0 else 0)
        mdm.append(dm if dm>um and dm>0 else 0)
    if len(tr) < period:
        return {'name':'ADX','value':25,'plusDI':0,'minusDI':0,'signal':'neutral','explanation':'Yetersiz'}
    atr_s = float(np.mean(tr[:period]))
    pdm_s = float(np.mean(pdm[:period]))
    mdm_s = float(np.mean(mdm[:period]))
    for i in range(period, len(tr)):
        atr_s = (atr_s*(period-1)+tr[i])/period
        pdm_s = (pdm_s*(period-1)+pdm[i])/period
        mdm_s = (mdm_s*(period-1)+mdm[i])/period
    pdi = sf((pdm_s/atr_s)*100 if atr_s else 0)
    mdi = sf((mdm_s/atr_s)*100 if atr_s else 0)
    ds = pdi+mdi
    adx = sf(abs(pdi-mdi)/ds*100 if ds else 0)
    sig = 'buy' if pdi>mdi and adx>25 else ('sell' if mdi>pdi and adx>25 else 'neutral')
    return {'name':'ADX','value':adx,'plusDI':pdi,'minusDI':mdi,'signal':sig,'explanation':f'ADX={adx}'}

def calc_obv(closes, volumes):
    if len(closes) < 10:
        return {'name':'OBV','value':0,'trend':'neutral','signal':'neutral','explanation':'Yetersiz'}
    obv, vals = 0, [0]
    for i in range(1, len(closes)):
        if float(closes[i])>float(closes[i-1]): obv+=int(volumes[i])
        elif float(closes[i])<float(closes[i-1]): obv-=int(volumes[i])
        vals.append(obv)
    trend = 'up' if vals[-1]>vals[-min(10,len(vals)-1)] else 'down'
    return {'name':'OBV','value':si(abs(vals[-1])),'trend':trend,'signal':'buy' if trend=='up' else 'sell','explanation':f'OBV {trend}'}

def calc_support_resistance(hist):
    try:
        c, h, l = hist['Close'].values.astype(float), hist['High'].values.astype(float), hist['Low'].values.astype(float)
        n = min(90, len(c))
        rh, rl = h[-n:], l[-n:]
        sups, ress = [], []
        for i in range(2, n-2):
            if rh[i]>rh[i-1] and rh[i]>rh[i-2] and rh[i]>rh[i+1] and rh[i]>rh[i+2]: ress.append(float(rh[i]))
            if rl[i]<rl[i-1] and rl[i]<rl[i-2] and rl[i]<rl[i+1] and rl[i]<rl[i+2]: sups.append(float(rl[i]))
        cur = float(c[-1])
        ns = sorted([s for s in sups if s<cur], reverse=True)[:3]
        nr = sorted([r for r in ress if r>cur])[:3]
        return {'supports':[sf(s) for s in ns],'resistances':[sf(r) for r in nr],'current':sf(cur)}
    except:
        return {'supports':[],'resistances':[],'current':0}

def calc_fibonacci(hist):
    try:
        c = hist['Close'].values.astype(float)
        n = min(90, len(c))
        r = c[-n:]
        hi, lo = float(np.max(r)), float(np.min(r))
        d = hi - lo
        levels = {'0.0':sf(hi),'23.6':sf(hi-d*0.236),'38.2':sf(hi-d*0.382),'50.0':sf(hi-d*0.5),'61.8':sf(hi-d*0.618),'78.6':sf(hi-d*0.786),'100.0':sf(lo)}
        cur = float(c[-1])
        zone = "Belirsiz"
        lk, lv = list(levels.keys()), list(levels.values())
        for i in range(len(lv)-1):
            if cur <= lv[i] and cur >= lv[i+1]:
                zone = f"{lk[i]}%-{lk[i+1]}%"
                break
        return {'levels':levels,'high':sf(hi),'low':sf(lo),'currentZone':zone}
    except:
        return {'levels':{},'currentZone':'-'}

def calc_all_indicators(hist, cp):
    c = hist['Close'].values.astype(float)
    h = hist['High'].values.astype(float)
    l = hist['Low'].values.astype(float)
    v = hist['Volume'].values.astype(float)
    cp = float(cp)
    rsi_h = []
    for i in range(14, len(c)):
        rv = calc_rsi_single(c[:i+1])
        if rv: rsi_h.append({'date':hist.index[i].strftime('%Y-%m-%d'),'value':rv})
    ind = {
        'rsi':calc_rsi(c), 'rsiHistory':rsi_h,
        'macd':calc_macd(c), 'macdHistory':calc_macd_history(c),
        'bollinger':calc_bollinger(c, cp), 'bollingerHistory':calc_bollinger_history(c),
        'stochastic':calc_stochastic(c, h, l),
        'ema':calc_ema(c, cp), 'emaHistory':calc_ema_history(c),
        'atr':calc_atr(h, l, c), 'adx':calc_adx(h, l, c),
        'obv':calc_obv(c, v),
    }
    sigs = [x.get('signal','neutral') for x in ind.values() if isinstance(x,dict) and 'signal' in x]
    bc, sc = sigs.count('buy'), sigs.count('sell')
    t = len(sigs)
    if bc>sc and bc>=t*0.4: ov='buy'
    elif sc>bc and sc>=t*0.4: ov='sell'
    else: ov='neutral'
    ind['summary'] = {'overall':ov,'buySignals':bc,'sellSignals':sc,'neutralSignals':t-bc-sc}
    return ind

def prepare_chart_data(hist):
    try:
        cs = [{'date':d.strftime('%Y-%m-%d'),'open':sf(r['Open']),'high':sf(r['High']),'low':sf(r['Low']),'close':sf(r['Close']),'volume':si(r['Volume'])} for d, r in hist.iterrows()]
        return {'candlestick':cs,'dates':[c['date'] for c in cs],'prices':[c['close'] for c in cs],'volumes':[c['volume'] for c in cs],'dataPoints':len(cs)}
    except:
        return {'candlestick':[],'dates':[],'prices':[],'volumes':[],'dataPoints':0}

# =====================================================================
# ROUTES - Hepsi cache'den doner, ASLA yfinance beklemez
# =====================================================================

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok', 'version': '3.0.1', 'yf': YF_OK,
        'time': datetime.now().isoformat(),
        'loader': _loader_status,
        'cacheEntries': len(_stock_cache),
        'stockCount': len(BIST100_STOCKS),
    })

@app.route('/')
def index():
    try:
        return send_from_directory(BASE_DIR, 'index.html')
    except:
        return jsonify({'error': 'index.html bulunamadi'}), 500

@app.route('/api/dashboard', methods=['GET'])
def dashboard():
    """Cache'den aninda don - ASLA yfinance cagirmaz"""
    try:
        stocks = get_cached_stocks()
        if not stocks:
            # Cache bos - loader henuz bitmedi
            if _loader_status['running']:
                return jsonify(safe_dict({
                    'success': True, 'loading': True, 'stockCount': 0,
                    'message': f"Veriler yukleniyor... ({_loader_status['loaded']}/{_loader_status['total']})",
                    'movers': {'topGainers':[],'topLosers':[],'volumeLeaders':[],'gapStocks':[]},
                    'marketBreadth': {'advancing':0,'declining':0,'unchanged':0,'advDecRatio':0},
                    'allStocks': []
                }))
            # Loader bitmis ama cache bos - yeniden baslat
            start_background_loader()
            return jsonify(safe_dict({
                'success': True, 'loading': True, 'stockCount': 0,
                'message': 'Veriler yukleniyor, birkaç saniye sonra sayfayi yenileyin...',
                'movers': {'topGainers':[],'topLosers':[],'volumeLeaders':[],'gapStocks':[]},
                'marketBreadth': {'advancing':0,'declining':0,'unchanged':0,'advDecRatio':0},
                'allStocks': []
            }))

        sbc = sorted(stocks, key=lambda x: x.get('changePct', 0), reverse=True)
        adv = sum(1 for s in stocks if s.get('changePct', 0) > 0)
        dec = sum(1 for s in stocks if s.get('changePct', 0) < 0)
        return jsonify(safe_dict({
            'success': True, 'loading': False, 'stockCount': len(stocks),
            'timestamp': datetime.now().isoformat(),
            'movers': {
                'topGainers': sbc[:5],
                'topLosers': sbc[-5:][::-1],
                'volumeLeaders': sorted(stocks, key=lambda x: x.get('volume',0), reverse=True)[:5],
                'gapStocks': sorted(stocks, key=lambda x: abs(x.get('gapPct',0)), reverse=True)[:5],
            },
            'marketBreadth': {
                'advancing': adv, 'declining': dec,
                'unchanged': len(stocks)-adv-dec,
                'advDecRatio': sf(adv/dec if dec > 0 else adv),
            },
            'allStocks': sbc,
        }))
    except Exception as e:
        print(f"DASH ERR: {traceback.format_exc()}")
        return jsonify({'error': str(e), 'success': False}), 500


@app.route('/api/indices', methods=['GET'])
def indices():
    """Endeksler - tek tek cek, kisa timeout"""
    try:
        result = {}
        pairs = [('XU100','XU100.IS','BIST 100'),('XU030','XU030.IS','BIST 30'),('XBANK','XBANK.IS','Bankacilik'),('USDTRY','USDTRY=X','USD/TRY'),('GOLD','GC=F','Altin')]
        for key, tsym, name in pairs:
            try:
                h = yf.Ticker(tsym).history(period="5d", timeout=10)
                if h is not None and not h.empty and len(h) >= 2:
                    result[key] = {'name':name,'value':sf(h['Close'].iloc[-1],4),'change':sf(h['Close'].iloc[-1]-h['Close'].iloc[-2],4),'changePct':sf((h['Close'].iloc[-1]-h['Close'].iloc[-2])/h['Close'].iloc[-2]*100),'volume':si(h['Volume'].iloc[-1]) if 'Volume' in h.columns else 0}
            except Exception as e:
                print(f"IDX {key}: {e}")
        return jsonify(safe_dict({'success': True, 'indices': result}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bist100', methods=['GET'])
def bist100_list():
    """BIST100 - cache'den"""
    try:
        sector = request.args.get('sector')
        sort_by = request.args.get('sort', 'code')
        order = request.args.get('order', 'asc')

        if sector and sector in SECTOR_MAP:
            stocks = get_cached_stocks(SECTOR_MAP[sector])
        else:
            stocks = get_cached_stocks()

        if not stocks:
            return jsonify(safe_dict({'success':True,'stocks':[],'count':0,'sectors':list(SECTOR_MAP.keys()),'message':'Veriler yukleniyor...'}))

        rev = (order == 'desc')
        km = {'change':'changePct','volume':'volume','price':'price'}
        sk = km.get(sort_by, 'code')
        stocks.sort(key=lambda x: x.get(sk,0) if sk!='code' else x.get('code',''), reverse=rev)
        return jsonify(safe_dict({'success':True,'stocks':stocks,'count':len(stocks),'sectors':list(SECTOR_MAP.keys())}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bist30', methods=['GET'])
def bist30_list():
    try:
        stocks = get_cached_stocks(BIST30)
        stocks.sort(key=lambda x: x.get('code',''))
        return jsonify(safe_dict({'success':True,'stocks':stocks,'count':len(stocks)}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/<symbol>', methods=['GET'])
def stock_detail(symbol):
    """Tek hisse detay - BURADA yfinance cagrilir ama tek hisse icin"""
    try:
        period = request.args.get('period', '1y')
        symbol = symbol.upper()

        if not YF_OK:
            return jsonify({'error': 'yfinance yuklu degil'}), 500

        ticker = yf.Ticker(f"{symbol}.IS")
        hist = ticker.history(period=period, timeout=15)

        if hist is None or hist.empty:
            return jsonify({'error': f'Veri bulunamadi: {symbol}'}), 404

        if len(hist) < 2:
            return jsonify({'error': f'Yetersiz veri: {symbol}'}), 404

        cp = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2])
        ch = cp - prev
        chp = (ch/prev)*100 if prev else 0

        info = {}
        try:
            info = ticker.info or {}
        except:
            pass

        result = {
            'success': True, 'code': symbol,
            'name': info.get('longName', info.get('shortName', BIST100_STOCKS.get(symbol, symbol))),
            'price': sf(cp), 'change': sf(ch), 'changePercent': sf(chp),
            'volume': si(hist['Volume'].iloc[-1]),
            'dayHigh': sf(hist['High'].iloc[-1]), 'dayLow': sf(hist['Low'].iloc[-1]),
            'dayOpen': sf(hist['Open'].iloc[-1]), 'prevClose': sf(prev),
            'currency': 'TRY', 'period': period, 'dataPoints': len(hist),
            'marketCap': si(info.get('marketCap', 0)),
            'peRatio': sf(info.get('trailingPE')) if info.get('trailingPE') else None,
            'pbRatio': sf(info.get('priceToBook')) if info.get('priceToBook') else None,
            'week52High': sf(info.get('fiftyTwoWeekHigh')) if info.get('fiftyTwoWeekHigh') else None,
            'week52Low': sf(info.get('fiftyTwoWeekLow')) if info.get('fiftyTwoWeekLow') else None,
            'indicators': calc_all_indicators(hist, cp),
            'chartData': prepare_chart_data(hist),
            'fibonacci': calc_fibonacci(hist),
            'supportResistance': calc_support_resistance(hist),
        }
        return jsonify(safe_dict(result))
    except Exception as e:
        print(f"STOCK {symbol} ERR: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/<symbol>/events', methods=['GET'])
def stock_events(symbol):
    try:
        symbol = symbol.upper()
        if not YF_OK:
            return jsonify({'success':True,'symbol':symbol,'events':{'dividends':[],'splits':[]}})
        t = yf.Ticker(f"{symbol}.IS")
        divs = [{'date':d.strftime('%Y-%m-%d'),'amount':sf(v)} for d,v in (t.dividends or pd.Series()).items()]
        splits = [{'date':d.strftime('%Y-%m-%d'),'ratio':sf(v)} for d,v in (t.splits or pd.Series()).items()]
        return jsonify(safe_dict({'success':True,'symbol':symbol,'events':{'dividends':divs[-20:],'splits':splits}}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/<symbol>/kap', methods=['GET'])
def stock_kap(symbol):
    return jsonify({'success':True,'symbol':symbol.upper(),'message':'KAP entegrasyonu aktif degil','notifications':[]})


@app.route('/api/compare', methods=['POST'])
def compare():
    try:
        data = request.json or {}
        symbols = data.get('symbols', [])
        if len(symbols) < 2: return jsonify({'error':'En az 2 hisse'}), 400
        period = data.get('period', '6mo')
        results = []
        for sym in symbols[:5]:
            try:
                hist = yf.Ticker(f"{sym.upper()}.IS").history(period=period, timeout=10)
                if hist is None or hist.empty or len(hist) < 10: continue
                c = hist['Close'].values.astype(float)
                cur, first = float(c[-1]), float(c[0])
                results.append({'code':sym.upper(),'name':BIST100_STOCKS.get(sym.upper(),sym),'price':sf(cur),'performance':sf((cur-first)/first*100),'rsi':calc_rsi(c)['value'],'volume':si(hist['Volume'].iloc[-1])})
            except: continue
        return jsonify(safe_dict({'success':True,'comparison':results,'period':period}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/screener', methods=['POST'])
def screener():
    """Tarayici - cache'deki hisseleri filtrele"""
    try:
        data = request.json or {}
        conditions = data.get('conditions', [])
        stocks = get_cached_stocks()
        if not stocks:
            return jsonify({'success':True,'matches':[],'message':'Veriler henuz yuklenmedi'})
        # Basit filtreleme
        matches = []
        for s in stocks:
            ok = True
            for cd in conditions:
                ind = cd.get('indicator','')
                op = cd.get('operator','>')
                val = float(cd.get('value', 0))
                if ind == 'changePct':
                    sv = s.get('changePct', 0)
                elif ind == 'volume':
                    sv = s.get('volume', 0)
                elif ind == 'price':
                    sv = s.get('price', 0)
                else:
                    continue
                if op == '>' and not (sv > val): ok = False; break
                elif op == '<' and not (sv < val): ok = False; break
            if ok:
                matches.append(s)
        return jsonify(safe_dict({'success':True,'matches':matches[:50],'totalMatches':len(matches),'availableSectors':list(SECTOR_MAP.keys())}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# PORTFOLIO / ALERTS / WATCHLIST
# =====================================================================
portfolio_store = {}
alert_store = []
watchlist_store = {}

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    uid = request.args.get('user','default')
    pf = portfolio_store.get(uid, [])
    pos, tv, tc = [], 0, 0
    for p in pf:
        with _lock:
            cd = _stock_cache.get(p['symbol'], {}).get('data')
        if not cd: continue
        q, ac, cp = p['quantity'], p['avgCost'], cd['price']
        mv, cb = cp*q, ac*q
        upnl = mv-cb
        tv += mv; tc += cb
        pos.append({'symbol':p['symbol'],'name':cd['name'],'quantity':si(q),'avgCost':sf(ac),'currentPrice':sf(cp),'marketValue':sf(mv),'costBasis':sf(cb),'unrealizedPnL':sf(upnl),'unrealizedPnLPct':sf(upnl/cb*100 if cb else 0),'weight':0})
    for p in pos: p['weight'] = sf(p['marketValue']/tv*100 if tv>0 else 0)
    tp = tv-tc
    return jsonify(safe_dict({'success':True,'positions':pos,'summary':{'totalValue':sf(tv),'totalCost':sf(tc),'totalPnL':sf(tp),'totalPnLPct':sf(tp/tc*100 if tc else 0),'positionCount':len(pos)}}))

@app.route('/api/portfolio', methods=['POST'])
def add_portfolio():
    try:
        d = request.json
        uid = d.get('user','default')
        sym = d.get('symbol','').upper()
        qty = float(d.get('quantity',0))
        ac = float(d.get('avgCost',0))
        if not sym or qty<=0 or ac<=0: return jsonify({'error':'Gecersiz'}), 400
        if uid not in portfolio_store: portfolio_store[uid] = []
        ex = next((p for p in portfolio_store[uid] if p['symbol']==sym), None)
        if ex:
            tq = ex['quantity']+qty
            ex['avgCost'] = (ex['avgCost']*ex['quantity']+ac*qty)/tq
            ex['quantity'] = tq
        else:
            portfolio_store[uid].append({'symbol':sym,'quantity':qty,'avgCost':ac,'addedAt':datetime.now().isoformat()})
        return jsonify({'success':True,'message':f'{sym} eklendi'})
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/api/portfolio', methods=['DELETE'])
def del_portfolio():
    d = request.json or {}
    uid, sym = d.get('user','default'), d.get('symbol','').upper()
    if uid in portfolio_store:
        portfolio_store[uid] = [p for p in portfolio_store[uid] if p['symbol']!=sym]
    return jsonify({'success':True})

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    email = request.args.get('email')
    f = [a for a in alert_store if a.get('email')==email] if email else alert_store
    return jsonify(safe_dict({'success':True,'alerts':f,'totalAlerts':len(f)}))

@app.route('/api/alerts', methods=['POST'])
def add_alert():
    try:
        d = request.json
        a = {'id':f"a{len(alert_store)+1}_{int(time.time())}",'email':d.get('email'),'symbol':d.get('symbol','').upper(),'condition':d.get('condition'),'threshold':sf(d.get('threshold')) if d.get('threshold') else None,'active':True,'triggered':False,'createdAt':datetime.now().isoformat()}
        alert_store.append(a)
        return jsonify(safe_dict({'success':True,'alert':a}))
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/api/alerts/<aid>', methods=['DELETE'])
def del_alert(aid):
    global alert_store
    alert_store = [a for a in alert_store if a.get('id')!=aid]
    return jsonify({'success':True})

@app.route('/api/alerts/check', methods=['POST'])
def check_alerts():
    return jsonify({'success':True,'triggered':[],'checkedAlerts':len(alert_store)})

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    uid = request.args.get('user','default')
    syms = watchlist_store.get(uid, [])
    stocks = get_cached_stocks(syms)
    return jsonify(safe_dict({'success':True,'watchlist':stocks}))

@app.route('/api/watchlist', methods=['POST'])
def update_watchlist():
    d = request.json or {}
    uid = d.get('user','default')
    sym = d.get('symbol','').upper()
    act = d.get('action','add')
    if uid not in watchlist_store: watchlist_store[uid] = []
    if act=='add' and sym not in watchlist_store[uid]: watchlist_store[uid].append(sym)
    elif act=='remove': watchlist_store[uid] = [s for s in watchlist_store[uid] if s!=sym]
    return jsonify({'success':True,'watchlist':watchlist_store[uid]})

@app.route('/api/sectors', methods=['GET'])
def sectors():
    try:
        sd = []
        for sn, syms in SECTOR_MAP.items():
            stocks = get_cached_stocks(syms)
            changes = [s['changePct'] for s in stocks if 'changePct' in s]
            sd.append({'name':sn,'stockCount':len(syms),'avgChange':sf(np.mean(changes)) if changes else 0,'symbols':syms})
        sd.sort(key=lambda x: x['avgChange'], reverse=True)
        return jsonify(safe_dict({'success':True,'sectors':sd}))
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search():
    q = request.args.get('q','').upper()
    if not q: return jsonify({'results':[]})
    return jsonify({'success':True,'results':[{'code':c,'name':n} for c,n in BIST100_STOCKS.items() if q in c or q in n.upper()][:10]})

@app.route('/api/backtest', methods=['POST'])
def backtest():
    try:
        d = request.json
        sym = d.get('symbol','').upper()
        period = d.get('period','1y')
        hist = yf.Ticker(f"{sym}.IS").history(period=period, timeout=15)
        if hist is None or hist.empty or len(hist)<60:
            return jsonify({'error':'Yetersiz veri'}), 400
        c = hist['Close'].values.astype(float)
        first, last = float(c[0]), float(c[-1])
        ret = (last-first)/first*100
        return jsonify(safe_dict({'success':True,'symbol':sym,'results':{'totalReturn':sf(ret),'startPrice':sf(first),'endPrice':sf(last),'dataPoints':len(c)}}))
    except Exception as e:
        return jsonify({'error':str(e)}), 500

@app.route('/api/docs', methods=['GET'])
def docs():
    return jsonify({'name':'BIST Pro v3.0','endpoints':['/api/health','/api/dashboard','/api/indices','/api/bist100','/api/bist30','/api/stock/<sym>','/api/screener','/api/portfolio','/api/backtest','/api/alerts','/api/watchlist','/api/sectors','/api/search']})


# =====================================================================
# STARTUP - Background loader'i hemen baslat
# =====================================================================
print("[STARTUP] BIST Pro v3.0.1 baslatiliyor...")
start_background_loader()

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"[STARTUP] Port {port}")
    app.run(host='0.0.0.0', port=port)
