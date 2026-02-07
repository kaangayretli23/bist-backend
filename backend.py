"""
BIST Pro v3.0.3 - RENDER BULLETPROOF FINAL
Thread guvenli: before_request ile lazy start.
Hicbir route yfinance CAGIRMAZ.
"""
from flask import Flask, jsonify, request, send_from_directory, make_response
from flask_cors import CORS
import traceback, os, time, threading, json
from datetime import datetime

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

@app.after_request
def force_json(resp):
    if resp.status_code >= 400 and 'text/html' in (resp.content_type or ''):
        resp = make_response(json.dumps({"error": f"HTTP {resp.status_code}"}), resp.status_code)
        resp.headers['Content-Type'] = 'application/json'
    return resp

@app.errorhandler(404)
def e404(e): return jsonify({'error':'Not found'}), 404
@app.errorhandler(500)
def e500(e): return jsonify({'error':str(e)}), 500
@app.errorhandler(Exception)
def eall(e):
    print(f"ERR: {traceback.format_exc()}")
    return jsonify({'error':str(e)}), 500

def sf(v, d=2):
    try:
        f = float(v)
        return 0.0 if f != f else round(f, d)
    except: return 0.0

def si(v):
    try: return int(v)
    except: return 0

def safe_dict(d):
    if isinstance(d, dict): return {str(k): safe_dict(v) for k, v in d.items()}
    if isinstance(d, list): return [safe_dict(i) for i in d]
    if hasattr(d, 'item'): return d.item()
    if isinstance(d, float): return 0.0 if d != d else round(d, 4)
    return d

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
INDEX_TICKERS = {
    'XU100':('XU100.IS','BIST 100'),'XU030':('XU030.IS','BIST 30'),
    'XBANK':('XBANK.IS','Bankacilik'),'USDTRY':('USDTRY=X','USD/TRY'),
    'GOLD':('GC=F','Altin'),
}

# =====================================================================
# CACHE
# =====================================================================
_lock = threading.Lock()
_stock_cache = {}
_index_cache = {}
_hist_cache = {}
_loader_started = False
_status = {'phase':'idle','loaded':0,'total':0,'lastRun':None,'error':''}
CACHE_TTL = 600

def _cget(store, key):
    with _lock:
        item = store.get(key)
        if item and time.time() - item['ts'] < CACHE_TTL:
            return item['data']
    return None

def _cset(store, key, data):
    with _lock:
        store[key] = {'data': data, 'ts': time.time()}

def _get_stocks(symbols=None):
    with _lock:
        if symbols:
            return [_stock_cache[s]['data'] for s in symbols
                    if s in _stock_cache and time.time()-_stock_cache[s]['ts']<CACHE_TTL]
        return [v['data'] for v in _stock_cache.values()
                if time.time()-v['ts']<CACHE_TTL]

def _get_indices():
    with _lock:
        return {k:v['data'] for k,v in _index_cache.items()
                if time.time()-v['ts']<CACHE_TTL}

# =====================================================================
# BACKGROUND LOADER - yf.download BATCH mode (tek HTTP cagrisi)
# =====================================================================
def _parse_batch_result(df, symbols):
    """yf.download sonucunu parse et"""
    results = []
    if df is None or df.empty:
        return results

    multi = isinstance(df.columns, pd.MultiIndex)
    print(f"[PARSE] shape={df.shape}, multi={multi}, cols={list(df.columns[:5])}")

    for sym in symbols:
        t = f"{sym}.IS"
        try:
            h = None
            if multi:
                # MultiIndex: (Ticker, Field) veya (Field, Ticker)
                level0 = list(set(df.columns.get_level_values(0)))
                level1 = list(set(df.columns.get_level_values(1)))

                if t in level0:
                    h = df[t].dropna(how='all')
                elif sym in level0:
                    h = df[sym].dropna(how='all')
                elif t in level1:
                    h = df.xs(t, axis=1, level=1, drop_level=True).dropna(how='all')
                elif sym in level1:
                    h = df.xs(sym, axis=1, level=1, drop_level=True).dropna(how='all')
                else:
                    # Ticker ismiyle eşleşme bulamadık - tüm level değerlerini logla
                    print(f"  [PARSE] {sym}: Ticker bulunamadi. L0 ornekleri: {level0[:3]}, L1 ornekleri: {level1[:3]}")
                    continue
            else:
                if len(symbols) == 1:
                    h = df.dropna(how='all')

            if h is None or h.empty or len(h) < 2:
                continue

            # Sütun isimlerini normalize et
            cols = [str(c).lower() for c in h.columns]
            col_map = {}
            for orig, low in zip(h.columns, cols):
                if 'close' in low: col_map['Close'] = orig
                elif 'open' in low: col_map['Open'] = orig
                elif 'high' in low: col_map['High'] = orig
                elif 'low' in low: col_map['Low'] = orig
                elif 'volume' in low: col_map['Volume'] = orig

            if 'Close' not in col_map:
                continue

            cur = sf(h[col_map['Close']].iloc[-1])
            prev = sf(h[col_map['Close']].iloc[-2])
            if prev == 0:
                continue

            ch = sf(cur - prev)
            o = sf(h[col_map.get('Open', col_map['Close'])].iloc[-1])
            results.append({
                'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                'price': cur, 'prevClose': prev,
                'change': ch, 'changePct': sf(ch / prev * 100),
                'volume': si(h[col_map.get('Volume', col_map['Close'])].iloc[-1]) if 'Volume' in col_map else 0,
                'open': o,
                'high': sf(h[col_map.get('High', col_map['Close'])].iloc[-1]),
                'low': sf(h[col_map.get('Low', col_map['Close'])].iloc[-1]),
                'gap': sf(o - prev),
                'gapPct': sf((o - prev) / prev * 100),
            })
        except Exception as e:
            print(f"  [PARSE] {sym} hata: {e}")
            continue

    return results


def _background_loop():
    if not YF_OK:
        print("[LOADER] yfinance yok, cikiliyor")
        return

    while True:
        try:
            # === FAZE 1: Endeksler (tek tek - sadece 5 tane) ===
            _status['phase'] = 'indices'
            _status['error'] = ''
            print(f"[LOADER] === FAZE 1: Endeksler ===")
            for key, (tsym, name) in INDEX_TICKERS.items():
                try:
                    print(f"  [IDX] {key} ({tsym}) indiriliyor...")
                    h = yf.Ticker(tsym).history(period="5d", timeout=15)
                    if h is not None and not h.empty and len(h) >= 2:
                        cur, prev = sf(h['Close'].iloc[-1], 4), sf(h['Close'].iloc[-2], 4)
                        _cset(_index_cache, key, {
                            'name': name, 'value': cur,
                            'change': sf(cur - prev, 4),
                            'changePct': sf((cur - prev) / prev * 100 if prev else 0),
                            'volume': si(h['Volume'].iloc[-1]) if 'Volume' in h.columns else 0,
                        })
                        print(f"  [IDX] {key} OK: {cur}")
                    else:
                        print(f"  [IDX] {key} BOS VERI")
                except Exception as e:
                    print(f"  [IDX] {key} HATA: {e}")
                time.sleep(0.3)

            print(f"[LOADER] Endeks cache: {len(_index_cache)} / {len(INDEX_TICKERS)}")

            # === FAZE 2: BIST30 hisseleri (BATCH download) ===
            _status['phase'] = 'stocks'
            _status['total'] = len(BIST30)
            _status['loaded'] = 0
            print(f"[LOADER] === FAZE 2: BIST30 batch download ===")

            tickers_str = " ".join(f"{s}.IS" for s in BIST30)
            print(f"[LOADER] yf.download basliyor: {len(BIST30)} hisse...")
            t_start = time.time()

            try:
                df = yf.download(tickers_str, period="5d", group_by="ticker",
                                 threads=True, progress=False, timeout=30)
                elapsed = round(time.time() - t_start, 1)
                print(f"[LOADER] yf.download tamamlandi: {elapsed}s, shape={df.shape if df is not None else 'None'}")

                if df is not None and not df.empty:
                    stocks = _parse_batch_result(df, BIST30)
                    for s in stocks:
                        _cset(_stock_cache, s['code'], s)
                    _status['loaded'] = len(stocks)
                    print(f"[LOADER] {len(stocks)}/{len(BIST30)} hisse parse edildi")
                else:
                    print(f"[LOADER] yf.download BOS DONDU!")
                    _status['error'] = 'yf.download bos'

                    # Fallback: teker teker dene (ilk 10 hisse)
                    print(f"[LOADER] Fallback: tek tek deneniyor...")
                    for i, sym in enumerate(BIST30[:10]):
                        try:
                            h = yf.Ticker(f"{sym}.IS").history(period="5d", timeout=15)
                            if h is not None and not h.empty and len(h) >= 2:
                                cur, prev = sf(h['Close'].iloc[-1]), sf(h['Close'].iloc[-2])
                                if prev > 0:
                                    ch = sf(cur - prev); o = sf(h['Open'].iloc[-1])
                                    _cset(_stock_cache, sym, {
                                        'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                                        'price': cur, 'prevClose': prev,
                                        'change': ch, 'changePct': sf(ch / prev * 100),
                                        'volume': si(h['Volume'].iloc[-1]),
                                        'open': o, 'high': sf(h['High'].iloc[-1]), 'low': sf(h['Low'].iloc[-1]),
                                        'gap': sf(o - prev), 'gapPct': sf((o - prev) / prev * 100),
                                    })
                                    _status['loaded'] = i + 1
                                    print(f"  [FALLBACK] {sym} OK: {cur}")
                        except Exception as e:
                            print(f"  [FALLBACK] {sym} HATA: {e}")
                        time.sleep(1)

            except Exception as e:
                print(f"[LOADER] yf.download HATA: {e}")
                _status['error'] = str(e)

            _status['phase'] = 'done'
            _status['lastRun'] = datetime.now().isoformat()
            print(f"[LOADER] === TAMAMLANDI: {len(_stock_cache)} hisse, {len(_index_cache)} endeks ===")

        except Exception as e:
            print(f"[LOADER] FATAL: {e}")
            traceback.print_exc()
            _status['phase'] = 'error'
            _status['error'] = str(e)

        # 5 dk bekle
        print(f"[LOADER] 5dk sonra tekrar yuklenecek...")
        time.sleep(300)


def _ensure_loader():
    """Thread-safe: loader'i sadece 1 kere baslat"""
    global _loader_started
    with _lock:
        if _loader_started:
            return
        _loader_started = True
    print("[LOADER] Thread baslatiliyor (before_request)")
    t = threading.Thread(target=_background_loop, daemon=True)
    t.start()

@app.before_request
def before_req():
    """Her request'te loader'in calistigini garanti et"""
    _ensure_loader()

# =====================================================================
# INDICATORS
# =====================================================================
def calc_rsi(closes, period=14):
    if len(closes)<period+1: return {'name':'RSI','value':50.0,'signal':'neutral'}
    d=np.diff(closes)
    ag=float(np.mean(np.where(d>0,d,0)[-period:]))
    al=float(np.mean(np.where(d<0,-d,0)[-period:]))
    rsi=100.0 if al==0 else sf(100-100/(1+ag/al))
    return {'name':'RSI','value':rsi,'signal':'buy' if rsi<30 else ('sell' if rsi>70 else 'neutral')}

def calc_rsi_single(closes, period=14):
    if len(closes)<period+1: return None
    d=np.diff(closes)
    ag=float(np.mean(np.where(d>0,d,0)[-period:]))
    al=float(np.mean(np.where(d<0,-d,0)[-period:]))
    return 100.0 if al==0 else sf(100-100/(1+ag/al))

def calc_macd(closes):
    if len(closes)<26: return {'name':'MACD','macd':0,'signal':0,'histogram':0,'signalType':'neutral'}
    s=pd.Series(list(closes),dtype=float); ml=s.ewm(span=12).mean()-s.ewm(span=26).mean(); sl=ml.ewm(span=9).mean()
    mv,sv=sf(ml.iloc[-1]),sf(sl.iloc[-1])
    return {'name':'MACD','macd':mv,'signal':sv,'histogram':sf(mv-sv),'signalType':'buy' if mv>sv else ('sell' if mv<sv else 'neutral')}

def calc_macd_history(closes):
    if len(closes)<26: return []
    s=pd.Series(list(closes),dtype=float); ml=s.ewm(span=12).mean()-s.ewm(span=26).mean(); sl=ml.ewm(span=9).mean()
    return [{'macd':sf(ml.iloc[i]),'signal':sf(sl.iloc[i]),'histogram':sf((ml-sl).iloc[i])} for i in range(26,len(closes))]

def calc_bollinger(closes, cp, period=20):
    if len(closes)<period: return {'name':'Bollinger','upper':0,'middle':0,'lower':0,'signal':'neutral','bandwidth':0}
    r=closes[-period:]; sma,std=float(np.mean(r)),float(np.std(r))
    u,m,lo=sf(sma+2*std),sf(sma),sf(sma-2*std)
    return {'name':'Bollinger','upper':u,'middle':m,'lower':lo,'bandwidth':sf((u-lo)/m*100 if m else 0),'signal':'buy' if float(cp)<lo else ('sell' if float(cp)>u else 'neutral')}

def calc_bollinger_history(closes, period=20):
    r=[]
    for i in range(period,len(closes)):
        w=closes[i-period:i]; sma,std=float(np.mean(w)),float(np.std(w))
        r.append({'upper':sf(sma+2*std),'middle':sf(sma),'lower':sf(sma-2*std)})
    return r

def calc_ema(closes, cp):
    result={'name':'EMA','signal':'neutral'}
    s=pd.Series(list(closes),dtype=float)
    if len(closes)>=20: result['ema20']=sf(s.ewm(span=20).mean().iloc[-1])
    if len(closes)>=50: result['ema50']=sf(s.ewm(span=50).mean().iloc[-1])
    if len(closes)>=200: result['ema200']=sf(s.ewm(span=200).mean().iloc[-1])
    e20,e50=result.get('ema20',cp),result.get('ema50',cp)
    if float(cp)>e20>e50: result['signal']='buy'
    elif float(cp)<e20<e50: result['signal']='sell'
    return result

def calc_ema_history(closes):
    s=pd.Series(list(closes),dtype=float)
    e20=s.ewm(span=20).mean() if len(closes)>=20 else pd.Series([])
    e50=s.ewm(span=50).mean() if len(closes)>=50 else pd.Series([])
    r=[]
    for i in range(len(closes)):
        p={}
        if i<len(e20): p['ema20']=sf(e20.iloc[i])
        if i<len(e50): p['ema50']=sf(e50.iloc[i])
        r.append(p)
    return r

def calc_stochastic(closes, highs, lows, period=14):
    if len(closes)<period: return {'name':'Stochastic','k':50,'d':50,'signal':'neutral'}
    hi,lo,cur=float(np.max(highs[-period:])),float(np.min(lows[-period:])),float(closes[-1])
    k=sf(((cur-lo)/(hi-lo))*100 if hi!=lo else 50)
    return {'name':'Stochastic','k':k,'d':k,'signal':'buy' if k<20 else ('sell' if k>80 else 'neutral')}

def calc_atr(highs, lows, closes, period=14):
    if len(closes)<period+1: return {'name':'ATR','value':0,'pct':0,'signal':'neutral'}
    tr=[max(float(highs[i])-float(lows[i]),abs(float(highs[i])-float(closes[i-1])),abs(float(lows[i])-float(closes[i-1]))) for i in range(1,len(closes))]
    atr=sf(np.mean(tr[-period:]))
    return {'name':'ATR','value':atr,'pct':sf(atr/float(closes[-1])*100 if closes[-1] else 0),'signal':'neutral'}

def calc_adx(highs, lows, closes, period=14):
    n=len(closes)
    if n<period+1: return {'name':'ADX','value':25,'plusDI':0,'minusDI':0,'signal':'neutral'}
    tr,pdm,mdm=[],[],[]
    for i in range(1,n):
        hv,lv,phv,plv,pcv=float(highs[i]),float(lows[i]),float(highs[i-1]),float(lows[i-1]),float(closes[i-1])
        tr.append(max(hv-lv,abs(hv-pcv),abs(lv-pcv)))
        um,dm=hv-phv,plv-lv
        pdm.append(um if um>dm and um>0 else 0); mdm.append(dm if dm>um and dm>0 else 0)
    if len(tr)<period: return {'name':'ADX','value':25,'plusDI':0,'minusDI':0,'signal':'neutral'}
    atr_s,pdm_s,mdm_s=float(np.mean(tr[:period])),float(np.mean(pdm[:period])),float(np.mean(mdm[:period]))
    for i in range(period,len(tr)):
        atr_s=(atr_s*(period-1)+tr[i])/period; pdm_s=(pdm_s*(period-1)+pdm[i])/period; mdm_s=(mdm_s*(period-1)+mdm[i])/period
    pdi=sf((pdm_s/atr_s)*100 if atr_s else 0); mdi=sf((mdm_s/atr_s)*100 if atr_s else 0)
    ds=pdi+mdi; adx=sf(abs(pdi-mdi)/ds*100 if ds else 0)
    return {'name':'ADX','value':adx,'plusDI':pdi,'minusDI':mdi,'signal':'buy' if pdi>mdi and adx>25 else ('sell' if mdi>pdi and adx>25 else 'neutral')}

def calc_obv(closes, volumes):
    if len(closes)<10: return {'name':'OBV','value':0,'trend':'neutral','signal':'neutral'}
    obv,vals=0,[0]
    for i in range(1,len(closes)):
        if float(closes[i])>float(closes[i-1]): obv+=int(volumes[i])
        elif float(closes[i])<float(closes[i-1]): obv-=int(volumes[i])
        vals.append(obv)
    trend='up' if vals[-1]>vals[-min(10,len(vals)-1)] else 'down'
    return {'name':'OBV','value':si(abs(vals[-1])),'trend':trend,'signal':'buy' if trend=='up' else 'sell'}

def calc_support_resistance(hist):
    try:
        c,h,l=hist['Close'].values.astype(float),hist['High'].values.astype(float),hist['Low'].values.astype(float)
        n=min(90,len(c)); rh,rl=h[-n:],l[-n:]; sups,ress=[],[]
        for i in range(2,n-2):
            if rh[i]>rh[i-1] and rh[i]>rh[i-2] and rh[i]>rh[i+1] and rh[i]>rh[i+2]: ress.append(float(rh[i]))
            if rl[i]<rl[i-1] and rl[i]<rl[i-2] and rl[i]<rl[i+1] and rl[i]<rl[i+2]: sups.append(float(rl[i]))
        cur=float(c[-1])
        return {'supports':[sf(s) for s in sorted([s for s in sups if s<cur],reverse=True)[:3]],'resistances':[sf(r) for r in sorted([r for r in ress if r>cur])[:3]],'current':sf(cur)}
    except: return {'supports':[],'resistances':[],'current':0}

def calc_fibonacci(hist):
    try:
        c=hist['Close'].values.astype(float); n=min(90,len(c)); r=c[-n:]
        hi,lo=float(np.max(r)),float(np.min(r)); d=hi-lo
        levels={'0.0':sf(hi),'23.6':sf(hi-d*0.236),'38.2':sf(hi-d*0.382),'50.0':sf(hi-d*0.5),'61.8':sf(hi-d*0.618),'78.6':sf(hi-d*0.786),'100.0':sf(lo)}
        cur=float(c[-1]); zone="Belirsiz"
        lk,lv=list(levels.keys()),list(levels.values())
        for i in range(len(lv)-1):
            if cur<=lv[i] and cur>=lv[i+1]: zone=f"{lk[i]}%-{lk[i+1]}%"; break
        return {'levels':levels,'high':sf(hi),'low':sf(lo),'currentZone':zone}
    except: return {'levels':{},'currentZone':'-'}

def calc_all_indicators(hist, cp):
    c,h,l,v=hist['Close'].values.astype(float),hist['High'].values.astype(float),hist['Low'].values.astype(float),hist['Volume'].values.astype(float)
    cp=float(cp)
    rsi_h=[{'date':hist.index[i].strftime('%Y-%m-%d'),'value':rv} for i in range(14,len(c)) if (rv:=calc_rsi_single(c[:i+1])) is not None]
    ind={'rsi':calc_rsi(c),'rsiHistory':rsi_h,'macd':calc_macd(c),'macdHistory':calc_macd_history(c),'bollinger':calc_bollinger(c,cp),'bollingerHistory':calc_bollinger_history(c),'stochastic':calc_stochastic(c,h,l),'ema':calc_ema(c,cp),'emaHistory':calc_ema_history(c),'atr':calc_atr(h,l,c),'adx':calc_adx(h,l,c),'obv':calc_obv(c,v)}
    sigs=[x.get('signal','neutral') for x in ind.values() if isinstance(x,dict) and 'signal' in x]
    bc,sc=sigs.count('buy'),sigs.count('sell'); t=len(sigs)
    ind['summary']={'overall':'buy' if bc>sc and bc>=t*0.4 else ('sell' if sc>bc and sc>=t*0.4 else 'neutral'),'buySignals':bc,'sellSignals':sc,'neutralSignals':t-bc-sc}
    return ind

def prepare_chart_data(hist):
    try:
        cs=[{'date':d.strftime('%Y-%m-%d'),'open':sf(r['Open']),'high':sf(r['High']),'low':sf(r['Low']),'close':sf(r['Close']),'volume':si(r['Volume'])} for d,r in hist.iterrows()]
        return {'candlestick':cs,'dates':[c['date'] for c in cs],'prices':[c['close'] for c in cs],'volumes':[c['volume'] for c in cs],'dataPoints':len(cs)}
    except: return {'candlestick':[],'dates':[],'prices':[],'volumes':[],'dataPoints':0}

# =====================================================================
# ROUTES - SIFIR YF CAGRISI
# =====================================================================
@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok', 'version': '3.0.4', 'yf': YF_OK,
        'time': datetime.now().isoformat(),
        'loader': _status,
        'loaderStarted': _loader_started,
        'stockCache': len(_stock_cache),
        'indexCache': len(_index_cache),
        'cachedStocks': list(_stock_cache.keys()),
        'cachedIndices': list(_index_cache.keys()),
    })

@app.route('/api/debug')
def debug():
    """Detayli debug - Render loglarinda ne olduğunu goster"""
    stock_details = {}
    with _lock:
        for k, v in _stock_cache.items():
            age = round(time.time() - v['ts'], 1)
            stock_details[k] = {'price': v['data']['price'], 'age_sec': age}
    index_details = {}
    with _lock:
        for k, v in _index_cache.items():
            age = round(time.time() - v['ts'], 1)
            index_details[k] = {'value': v['data']['value'], 'age_sec': age}
    return jsonify({
        'loaderStarted': _loader_started,
        'status': _status,
        'stockCache': stock_details,
        'indexCache': index_details,
        'totalStocks': len(stock_details),
        'totalIndices': len(index_details),
        'yfinance': YF_OK,
        'time': datetime.now().isoformat(),
    })

@app.route('/')
def index():
    try: return send_from_directory(BASE_DIR, 'index.html')
    except: return jsonify({'error':'index.html bulunamadi'}),500

@app.route('/api/dashboard')
def dashboard():
    try:
        stocks=_get_stocks()
        if not stocks:
            return jsonify(safe_dict({'success':True,'loading':True,'stockCount':0,'message':f"Veriler yukleniyor ({_status['loaded']}/{_status['total']})...",'movers':{'topGainers':[],'topLosers':[],'volumeLeaders':[],'gapStocks':[]},'marketBreadth':{'advancing':0,'declining':0,'unchanged':0,'advDecRatio':0},'allStocks':[]}))
        sbc=sorted(stocks,key=lambda x:x.get('changePct',0),reverse=True)
        adv=sum(1 for s in stocks if s.get('changePct',0)>0)
        dec=sum(1 for s in stocks if s.get('changePct',0)<0)
        return jsonify(safe_dict({'success':True,'loading':False,'stockCount':len(stocks),'timestamp':datetime.now().isoformat(),'movers':{'topGainers':sbc[:5],'topLosers':sbc[-5:][::-1],'volumeLeaders':sorted(stocks,key=lambda x:x.get('volume',0),reverse=True)[:5],'gapStocks':sorted(stocks,key=lambda x:abs(x.get('gapPct',0)),reverse=True)[:5]},'marketBreadth':{'advancing':adv,'declining':dec,'unchanged':len(stocks)-adv-dec,'advDecRatio':sf(adv/dec if dec>0 else adv)},'allStocks':sbc}))
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/indices')
def indices():
    try:
        data=_get_indices()
        if not data:
            return jsonify(safe_dict({'success':True,'loading':True,'indices':{},'message':'Endeksler yukleniyor...'}))
        return jsonify(safe_dict({'success':True,'indices':data}))
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/bist100')
def bist100():
    try:
        sector=request.args.get('sector'); sort_by=request.args.get('sort','code'); order=request.args.get('order','asc')
        stocks=_get_stocks(SECTOR_MAP[sector]) if sector and sector in SECTOR_MAP else _get_stocks()
        if not stocks:
            return jsonify(safe_dict({'success':True,'stocks':[],'count':0,'sectors':list(SECTOR_MAP.keys()),'loading':True,'message':f"Hisse verileri yukleniyor ({_status['loaded']}/{_status['total']})..."}))
        rev=(order=='desc'); km={'change':'changePct','volume':'volume','price':'price'}; sk=km.get(sort_by,'code')
        stocks.sort(key=lambda x:x.get(sk,0) if sk!='code' else x.get('code',''),reverse=rev)
        return jsonify(safe_dict({'success':True,'stocks':stocks,'count':len(stocks),'sectors':list(SECTOR_MAP.keys())}))
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/bist30')
def bist30():
    try:
        stocks=_get_stocks(BIST30)
        if not stocks: return jsonify(safe_dict({'success':True,'stocks':[],'count':0,'loading':True}))
        stocks.sort(key=lambda x:x.get('code','')); return jsonify(safe_dict({'success':True,'stocks':stocks,'count':len(stocks)}))
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/stock/<symbol>')
def stock_detail(symbol):
    """Hisse detay - cache'den al, yoksa background'da yukle"""
    try:
        symbol=symbol.upper(); period=request.args.get('period','1y')
        hist=_cget(_hist_cache, f"{symbol}_{period}")
        if hist is None:
            # Background'da yukle
            def _bg():
                try:
                    h=yf.Ticker(f"{symbol}.IS").history(period=period,timeout=12)
                    if h is not None and not h.empty and len(h)>=2:
                        _cset(_hist_cache,f"{symbol}_{period}",h)
                        print(f"  [BG] {symbol} {period} yuklendi: {len(h)} bar")
                except Exception as e:
                    print(f"  [BG] {symbol} hata: {e}")
            threading.Thread(target=_bg, daemon=True).start()
            # Quick data varsa onu don
            quick=_cget(_stock_cache, symbol)
            if quick:
                return jsonify(safe_dict({'success':True,'code':symbol,'name':quick['name'],'price':quick['price'],'change':quick['change'],'changePercent':quick['changePct'],'volume':quick['volume'],'dayOpen':quick['open'],'dayHigh':quick['high'],'dayLow':quick['low'],'prevClose':quick['prevClose'],'currency':'TRY','period':period,'loading':True,'message':'Detayli veri yukleniyor, 5sn sonra yenileyin','indicators':{},'chartData':{'candlestick':[],'dates':[],'prices':[],'volumes':[],'dataPoints':0},'fibonacci':{'levels':{}},'supportResistance':{'supports':[],'resistances':[]}}))
            return jsonify({'success':True,'loading':True,'message':f'{symbol} verileri yukleniyor...','code':symbol}),202

        cp=float(hist['Close'].iloc[-1])
        prev=float(hist['Close'].iloc[-2]) if len(hist)>1 else cp
        return jsonify(safe_dict({'success':True,'code':symbol,'name':BIST100_STOCKS.get(symbol,symbol),'price':sf(cp),'change':sf(cp-prev),'changePercent':sf((cp-prev)/prev*100 if prev else 0),'volume':si(hist['Volume'].iloc[-1]),'dayHigh':sf(hist['High'].iloc[-1]),'dayLow':sf(hist['Low'].iloc[-1]),'dayOpen':sf(hist['Open'].iloc[-1]),'prevClose':sf(prev),'currency':'TRY','period':period,'dataPoints':len(hist),'indicators':calc_all_indicators(hist,cp),'chartData':prepare_chart_data(hist),'fibonacci':calc_fibonacci(hist),'supportResistance':calc_support_resistance(hist)}))
    except Exception as e:
        print(f"STOCK {symbol}: {traceback.format_exc()}")
        return jsonify({'error':str(e)}),500

@app.route('/api/stock/<symbol>/events')
def stock_events(symbol):
    return jsonify({'success':True,'symbol':symbol.upper(),'events':{'dividends':[],'splits':[]}})

@app.route('/api/stock/<symbol>/kap')
def stock_kap(symbol):
    return jsonify({'success':True,'symbol':symbol.upper(),'message':'KAP aktif degil','notifications':[]})

@app.route('/api/compare', methods=['POST'])
def compare():
    try:
        data=request.json or {}; symbols=data.get('symbols',[])
        if len(symbols)<2: return jsonify({'error':'En az 2 hisse'}),400
        results=[{'code':s['code'],'name':s['name'],'price':s['price'],'changePct':s['changePct'],'volume':s['volume']} for sym in symbols[:5] if (s:=_cget(_stock_cache,sym.upper()))]
        return jsonify(safe_dict({'success':True,'comparison':results}))
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/screener', methods=['POST'])
def screener():
    try:
        stocks=_get_stocks()
        if not stocks: return jsonify({'success':True,'matches':[],'message':'Veriler yukleniyor'})
        conditions=request.json.get('conditions',[]) if request.json else []
        matches=[]
        for s in stocks:
            ok=True
            for cd in conditions:
                ind,op,val=cd.get('indicator',''),cd.get('operator','>'),float(cd.get('value',0))
                sv=s.get(ind,s.get('changePct',0))
                try:
                    if op=='>' and not(float(sv)>val): ok=False; break
                    elif op=='<' and not(float(sv)<val): ok=False; break
                except: ok=False; break
            if ok: matches.append(s)
        return jsonify(safe_dict({'success':True,'matches':matches[:50],'totalMatches':len(matches)}))
    except Exception as e:
        return jsonify({'error':str(e)}),500

# ---- PORTFOLIO / ALERTS / WATCHLIST ----
portfolio_store={}; alert_store=[]; watchlist_store={}

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    uid=request.args.get('user','default'); pf=portfolio_store.get(uid,[]); pos=[]; tv=tc=0
    for p in pf:
        cd=_cget(_stock_cache,p['symbol'])
        if not cd: continue
        q,ac,cp=p['quantity'],p['avgCost'],cd['price']; mv=cp*q; cb=ac*q; upnl=mv-cb; tv+=mv; tc+=cb
        pos.append({'symbol':p['symbol'],'name':cd['name'],'quantity':si(q),'avgCost':sf(ac),'currentPrice':sf(cp),'marketValue':sf(mv),'costBasis':sf(cb),'unrealizedPnL':sf(upnl),'unrealizedPnLPct':sf(upnl/cb*100 if cb else 0),'weight':0})
    for p in pos: p['weight']=sf(p['marketValue']/tv*100 if tv>0 else 0)
    tp=tv-tc
    return jsonify(safe_dict({'success':True,'positions':pos,'summary':{'totalValue':sf(tv),'totalCost':sf(tc),'totalPnL':sf(tp),'totalPnLPct':sf(tp/tc*100 if tc else 0),'positionCount':len(pos)}}))

@app.route('/api/portfolio', methods=['POST'])
def add_portfolio():
    try:
        d=request.json; uid=d.get('user','default'); sym=d.get('symbol','').upper(); qty=float(d.get('quantity',0)); ac=float(d.get('avgCost',0))
        if not sym or qty<=0 or ac<=0: return jsonify({'error':'Gecersiz'}),400
        if uid not in portfolio_store: portfolio_store[uid]=[]
        ex=next((p for p in portfolio_store[uid] if p['symbol']==sym),None)
        if ex: tq=ex['quantity']+qty; ex['avgCost']=(ex['avgCost']*ex['quantity']+ac*qty)/tq; ex['quantity']=tq
        else: portfolio_store[uid].append({'symbol':sym,'quantity':qty,'avgCost':ac,'addedAt':datetime.now().isoformat()})
        return jsonify({'success':True,'message':f'{sym} eklendi'})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/portfolio', methods=['DELETE'])
def del_portfolio():
    d=request.json or {}; uid,sym=d.get('user','default'),d.get('symbol','').upper()
    if uid in portfolio_store: portfolio_store[uid]=[p for p in portfolio_store[uid] if p['symbol']!=sym]
    return jsonify({'success':True})

@app.route('/api/portfolio/risk')
def portfolio_risk():
    return jsonify({'success':True,'risk':{'message':'Risk analizi yukleniyor'}})

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    email=request.args.get('email')
    return jsonify(safe_dict({'success':True,'alerts':[a for a in alert_store if a.get('email')==email] if email else alert_store}))

@app.route('/api/alerts', methods=['POST'])
def add_alert():
    try:
        d=request.json; a={'id':f"a{len(alert_store)+1}_{int(time.time())}",'email':d.get('email'),'symbol':d.get('symbol','').upper(),'condition':d.get('condition'),'active':True,'createdAt':datetime.now().isoformat()}
        alert_store.append(a); return jsonify(safe_dict({'success':True,'alert':a}))
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/alerts/<aid>', methods=['DELETE'])
def del_alert(aid):
    global alert_store; alert_store=[a for a in alert_store if a.get('id')!=aid]; return jsonify({'success':True})

@app.route('/api/alerts/check', methods=['POST'])
def check_alerts():
    return jsonify({'success':True,'triggered':[]})

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    return jsonify(safe_dict({'success':True,'watchlist':_get_stocks(watchlist_store.get(request.args.get('user','default'),[]))}))

@app.route('/api/watchlist', methods=['POST'])
def update_watchlist():
    d=request.json or {}; uid=d.get('user','default'); sym=d.get('symbol','').upper(); act=d.get('action','add')
    if uid not in watchlist_store: watchlist_store[uid]=[]
    if act=='add' and sym not in watchlist_store[uid]: watchlist_store[uid].append(sym)
    elif act=='remove': watchlist_store[uid]=[s for s in watchlist_store[uid] if s!=sym]
    return jsonify({'success':True,'watchlist':watchlist_store[uid]})

@app.route('/api/sectors')
def sectors():
    try:
        sd=[]
        for sn,syms in SECTOR_MAP.items():
            stocks=_get_stocks(syms); changes=[s['changePct'] for s in stocks if 'changePct' in s]
            sd.append({'name':sn,'stockCount':len(syms),'avgChange':sf(np.mean(changes)) if changes else 0,'symbols':syms})
        sd.sort(key=lambda x:x['avgChange'],reverse=True)
        return jsonify(safe_dict({'success':True,'sectors':sd}))
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/search')
def search():
    q=request.args.get('q','').upper()
    return jsonify({'success':True,'results':[{'code':c,'name':n} for c,n in BIST100_STOCKS.items() if q in c or q in n.upper()][:10]}) if q else jsonify({'results':[]})

@app.route('/api/backtest', methods=['POST'])
def backtest():
    return jsonify({'success':True,'message':'Backtest hazirlaniyor'})

@app.route('/api/docs')
def docs():
    return jsonify({'name':'BIST Pro v3.0.4','endpoints':['/api/health','/api/debug','/api/dashboard','/api/indices','/api/bist100','/api/bist30','/api/stock/<sym>','/api/screener','/api/portfolio','/api/backtest','/api/alerts','/api/watchlist','/api/sectors','/api/search']})

# NO MODULE-LEVEL THREAD START - before_request handles it
print("[STARTUP] BIST Pro v3.0.4 ready - batch loader")

if __name__ == '__main__':
    port=int(os.environ.get("PORT",5000))
    print(f"[STARTUP] Port {port}")
    app.run(host='0.0.0.0',port=port)
