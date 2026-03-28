"""
BIST Pro v7.1.0 - Main Entry Point
Modular architecture: config, database, indicators, signals, data_fetcher,
trade_plans, auto_trader, bes_system
"""
# Foundation
from config import *
# Python'da import * underscore ile baslayan isimleri atlar, bu yuzden acikca import ediyoruz
from config import (
    _lock, _stock_cache, _index_cache, _hist_cache,
    _loader_started, _status, _plan_lock_cache, _plan_lock_cache_lock,
    _html_page_cache, _html_page_cache_lock,
    _cget, _cset, _ctouch, _cget_hist, _get_stocks, _get_indices
)

# Database init & cache
try:
    from database import (
        init_db, _db_upsert_cache, _db_get_cache,
        _db_save_market_snapshot, _db_load_market_snapshot, _db_save_plan_locks
    )
except ImportError as e:
    print(f"[HATA] database.py import hatasi: {e}")
    print("[BILGI] 'git pull origin main' yaparak dosyalari guncelle")
    # Dummy fonksiyonlar — crash onleme
    def init_db(): pass
    def _db_load_market_snapshot(): pass
    def _db_save_market_snapshot(): pass
    def _db_save_plan_locks(): pass
    def _db_upsert_cache(*a): pass
    def _db_get_cache(*a): return None

# Technical indicators
try:
    from indicators import (
        calc_rsi, calc_rsi_single, calc_macd,
        calc_bollinger, calc_ema,
        calc_stochastic, calc_atr, calc_adx, calc_obv,
        calc_support_resistance, calc_fibonacci, calc_williams_r, calc_cci,
        calc_mfi, calc_vwap, calc_ichimoku, calc_psar, calc_pivot_points,
        calc_roc, calc_aroon, calc_trix, calc_dmi,
        calc_all_indicators, calc_mtf_signal, calc_divergence,
        calc_volume_profile, calc_smc, calc_chart_patterns,
        calc_fibonacci_adv, calc_pivot_points_adv, calc_advanced_indicators,
        calc_dynamic_thresholds, calc_candlestick_patterns,
        prepare_chart_data, _resample_to_tf,
    )
except ImportError as e:
    print(f"[HATA] indicators.py import hatasi: {e}")

# Signal & recommendation system
try:
    from signals import (
        calc_recommendation, calc_fundamentals, calc_52w,
        calc_signal_backtest, calc_market_regime, calc_sector_relative_strength,
        fetch_fundamental_data, check_signal_alerts, calc_ml_confidence,
    )
except ImportError as e:
    print(f"[HATA] signals.py import hatasi: {e}")

# Data fetcher & background loader
try:
    from data_fetcher import (
        _fetch_stock_data, _fetch_hist_df, _fetch_index_data,
        _process_stock, _fetch_stocks_parallel, _background_loop,
        _auto_check_all_alerts, _preload_hist_data, _ensure_loader,
        _fetch_isyatirim_df, _fetch_isyatirim_quick,
        _fetch_yahoo_http, _fetch_yahoo_http_df,
        IS_YATIRIM_BASE, IS_YATIRIM_HEADERS,
    )
except ImportError as e:
    print(f"[HATA] data_fetcher.py import hatasi: {e}")

# Trade plans
try:
    from trade_plans import (
        calc_trade_plan,
    )
except ImportError as e:
    print(f"[HATA] trade_plans.py import hatasi: {e}")

# Auto-trading engine
try:
    import auto_trader
except ImportError as e:
    print(f"[HATA] auto_trader.py import hatasi: {e}")

# BES fund system
try:
    import bes_system
except ImportError as e:
    print(f"[HATA] bes_system.py import hatasi: {e}")

# Initialize database
init_db()

# Load market snapshot from DB (cold-start acceleration)
_db_load_market_snapshot()

@app.after_request
def after_req(resp):
    # 4xx/5xx HTML -> JSON
    if resp.status_code >= 400 and 'text/html' in (resp.content_type or ''):
        resp = make_response(json.dumps({"error": f"HTTP {resp.status_code}"}), resp.status_code)
        resp.headers['Content-Type'] = 'application/json'
    # gzip sikistirma (>1KB ve henuz sikistirilmamis)
    if (resp.status_code == 200
        and 'Content-Encoding' not in resp.headers
        and resp.content_length and resp.content_length > 1024
        and 'gzip' in request.headers.get('Accept-Encoding', '')):
        ct = resp.content_type or ''
        if 'json' in ct or 'javascript' in ct or 'text/' in ct:
            data = resp.get_data()
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as gz:
                gz.write(data)
            compressed = buf.getvalue()
            if len(compressed) < len(data):
                resp.set_data(compressed)
                resp.headers['Content-Encoding'] = 'gzip'
                resp.headers['Content-Length'] = len(compressed)
                resp.headers['Vary'] = 'Accept-Encoding'
    return resp

@app.errorhandler(404)
def e404(e): return jsonify({'error':'Not found'}), 404
@app.errorhandler(500)
def e500(e): return jsonify({'error':str(e)}), 500
@app.errorhandler(Exception)
def eall(e):
    print(f"ERR: {traceback.format_exc()}")
    return jsonify({'error':str(e)}), 500


def _cache_freshness():
    """Cache durumunu dondur: fresh/stale/loading"""
    now = time.time()
    with _lock:
        if not _stock_cache:
            return 'loading', None
        ages = [now - v['ts'] for v in _stock_cache.values()]
    avg_age = sum(ages) / len(ages) if ages else 9999
    if avg_age < CACHE_TTL:
        return 'fresh', datetime.fromtimestamp(now - avg_age).isoformat()
    elif avg_age < CACHE_STALE_TTL:
        return 'stale', datetime.fromtimestamp(now - avg_age).isoformat()
    else:
        return 'expired', None

def _api_meta(data_quality=None, extra=None):
    """Tum API response'larina eklenecek meta bilgisi"""
    freshness, last_updated = _cache_freshness()
    meta = {
        'lastUpdated': last_updated or _status.get('lastRun'),
        'snapshotTimestamp': datetime.now().isoformat(),
        'dataQuality': data_quality or freshness,
        'loaderPhase': _status.get('phase', 'idle'),
    }
    if extra:
        meta.update(extra)
    return meta


@app.route('/api/health')
def health():
    import shutil
    now = time.time()
    with _lock:
        fresh = [k for k, v in _stock_cache.items() if now - v['ts'] < CACHE_TTL]
        stale = [k for k, v in _stock_cache.items() if CACHE_TTL <= now - v['ts'] < CACHE_STALE_TTL]
    hist_ready = sum(1 for s in BIST100_STOCKS if _cget_hist(f"{s}_1y") is not None)
    try:
        usage = shutil.disk_usage('/')
        disk_info = {
            'totalMB': round(usage.total / 1024**2),
            'usedMB': round(usage.used / 1024**2),
            'freeMB': round(usage.free / 1024**2),
            'usedPercent': round(usage.used / usage.total * 100, 1),
        }
    except Exception as e:
        disk_info = {'error': str(e)}
    return jsonify({
        'status': 'ok', 'version': '7.0.0', 'yf': YF_OK,
        'time': datetime.now().isoformat(),
        'loader': _status,
        'loaderStarted': _loader_started,
        'stockCache': len(_stock_cache),
        'stockCacheFresh': len(fresh),
        'stockCacheStale': len(stale),
        'indexCache': len(_index_cache),
        'histCache': hist_ready,
        'histCacheTotal': len(_hist_cache),
        'cachedStocks': list(_stock_cache.keys()),
        'cachedIndices': list(_index_cache.keys()),
        'totalDefined': len(BIST100_STOCKS),
        'missingStocks': [s for s in BIST100_STOCKS.keys() if s not in _stock_cache],
        'disk': disk_info,
    })

@app.route('/api/debug')
def debug():
    """Detayli debug - Render loglarinda ne olduğunu goster"""
    now = time.time()
    stock_details = {}
    with _lock:
        for k, v in _stock_cache.items():
            age = round(now - v['ts'], 1)
            status = 'fresh' if age < CACHE_TTL else ('stale' if age < CACHE_STALE_TTL else 'expired')
            stock_details[k] = {'price': v['data']['price'], 'age_sec': age, 'status': status}
    index_details = {}
    with _lock:
        for k, v in _index_cache.items():
            age = round(now - v['ts'], 1)
            index_details[k] = {'value': v['data']['value'], 'age_sec': age}
    missing = [s for s in BIST100_STOCKS.keys() if s not in _stock_cache]
    return jsonify({
        'loaderStarted': _loader_started,
        'status': _status,
        'stockCache': stock_details,
        'indexCache': index_details,
        'totalStocks': len(stock_details),
        'totalDefined': len(BIST100_STOCKS),
        'missingStocks': missing,
        'totalMissing': len(missing),
        'totalIndices': len(index_details),
        'yfinance': YF_OK,
        'time': datetime.now().isoformat(),
    })

@app.route('/api/test-fetch/<symbol>')
def test_fetch(symbol):
    """Veri cekme pipeline'ini test et - debug icin"""
    symbol = symbol.upper()
    results = {}

    # 1. Is Yatirim quick
    try:
        data = _fetch_isyatirim_quick(symbol)
        results['isyatirim_quick'] = {'success': data is not None, 'data': data}
    except Exception as e:
        results['isyatirim_quick'] = {'success': False, 'error': str(e)}

    # 2. Is Yatirim DF (30 gun)
    try:
        df = _fetch_isyatirim_df(symbol, days=30)
        if df is not None:
            nan_counts = df[['Open','High','Low','Close','Volume']].isna().sum().to_dict()
            results['isyatirim_df'] = {
                'success': True, 'rows': len(df),
                'last_close': float(df['Close'].iloc[-1]),
                'last_high': float(df['High'].iloc[-1]),
                'last_low': float(df['Low'].iloc[-1]),
                'nan_counts': nan_counts,
                'date_range': f"{df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}",
            }
        else:
            results['isyatirim_df'] = {'success': False, 'data': None}
    except Exception as e:
        results['isyatirim_df'] = {'success': False, 'error': str(e)}

    # 3. Yahoo HTTP
    try:
        data = _fetch_yahoo_http(f"{symbol}.IS")
        results['yahoo_http'] = {'success': data is not None, 'data': data}
    except Exception as e:
        results['yahoo_http'] = {'success': False, 'error': str(e)}

    # 4. Yahoo DF
    try:
        df = _fetch_yahoo_http_df(f"{symbol}.IS", period1_days=30)
        if df is not None:
            results['yahoo_df'] = {'success': True, 'rows': len(df), 'last_close': float(df['Close'].iloc[-1])}
        else:
            results['yahoo_df'] = {'success': False, 'data': None}
    except Exception as e:
        results['yahoo_df'] = {'success': False, 'error': str(e)}

    # 5. Cache durumu
    cached = _cget(_stock_cache, symbol)
    results['cache'] = {'has_cache': cached is not None}
    if cached:
        results['cache']['data'] = cached

    return jsonify(safe_dict(results))

# --- index.html bellekte gzipli cache ---
# Not: _index_cache adı endeks verisi cache'i ile çakışıyor — farklı isim kullan
_html_page_cache = {'raw': None, 'gz': None, 'mtime': 0}
_html_page_cache_lock = threading.Lock()

def _load_index_html():
    """index.html'i diskten oku, gziple, bellekte tut"""
    fpath = os.path.join(BASE_DIR, 'index.html')
    try:
        mt = os.path.getmtime(fpath)
        with _html_page_cache_lock:
            if _html_page_cache['raw'] and _html_page_cache['mtime'] == mt:
                return True
        with open(fpath, 'rb') as f:
            raw = f.read()
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as gz:
            gz.write(raw)
        with _html_page_cache_lock:
            _html_page_cache['raw'] = raw
            _html_page_cache['gz'] = buf.getvalue()
            _html_page_cache['mtime'] = mt
        print(f"[INDEX] Cached: {len(raw)} bytes -> gzip {len(buf.getvalue())} bytes")
        return True
    except Exception as e:
        print(f"[INDEX] Load error: {e}")
        return False

_load_index_html()

@app.route('/ping')
def ping():
    return 'ok', 200

@app.route('/api/network-info')
def network_info():
    """Yerel ag bilgisi — telefondan erisim icin IP goster"""
    import socket
    local_ip = '127.0.0.1'
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        pass
    port = int(os.environ.get("PORT", 5000))
    return jsonify({
        'localIP': local_ip,
        'port': port,
        'localURL': f'http://{local_ip}:{port}',
        'info': 'Ayni WiFi agindaki telefondan bu URL ile erisebilirsin'
    })

@app.route('/')
def index():
    with _html_page_cache_lock:
        raw = _html_page_cache['raw']
        gz = _html_page_cache['gz']
    if not raw:
        if not _load_index_html():
            return jsonify({'error':'index.html bulunamadi'}), 500
        with _html_page_cache_lock:
            raw = _html_page_cache['raw']
            gz = _html_page_cache['gz']
    # gzip destegi varsa sikistirilmis gonder
    ae = request.headers.get('Accept-Encoding', '')
    if 'gzip' in ae and gz:
        resp = make_response(gz)
        resp.headers['Content-Encoding'] = 'gzip'
    else:
        resp = make_response(raw)
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    resp.headers['Cache-Control'] = 'public, max-age=300'
    resp.headers['Vary'] = 'Accept-Encoding'
    return resp

@app.route('/api/dashboard')
def dashboard():
    try:
        stocks=_get_stocks()
        if not stocks:
            return jsonify(safe_dict({'success':True,'loading':True,'stockCount':0,'message':f"Veriler yukleniyor ({_status['loaded']}/{_status['total']})...",'movers':{'topGainers':[],'topLosers':[],'volumeLeaders':[],'gapStocks':[]},'marketBreadth':{'advancing':0,'declining':0,'unchanged':0,'advDecRatio':0},'allStocks':[],'meta':_api_meta('loading')}))
        sbc=sorted(stocks,key=lambda x:x.get('changePct',0),reverse=True)
        adv=sum(1 for s in stocks if s.get('changePct',0)>0)
        dec=sum(1 for s in stocks if s.get('changePct',0)<0)
        return jsonify(safe_dict({'success':True,'loading':False,'stockCount':len(stocks),'timestamp':datetime.now().isoformat(),'movers':{'topGainers':sbc[:5],'topLosers':sbc[-5:][::-1],'volumeLeaders':sorted(stocks,key=lambda x:x.get('volume',0),reverse=True)[:5],'gapStocks':sorted(stocks,key=lambda x:abs(x.get('gapPct',0)),reverse=True)[:5]},'marketBreadth':{'advancing':adv,'declining':dec,'unchanged':len(stocks)-adv-dec,'advDecRatio':sf(adv/dec if dec>0 else adv)},'allStocks':sbc,'meta':_api_meta()}))
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/indices')
def indices():
    try:
        data=_get_indices()
        if not data:
            return jsonify(safe_dict({'success':True,'loading':True,'indices':{},'message':'Endeksler yukleniyor...'}))

        # Gold/TL ve Silver/TL hesapla
        usdtry_data = data.get('USDTRY')
        gold_data = data.get('GOLD')
        silver_data = data.get('SILVER')
        if usdtry_data and gold_data:
            usd_rate = usdtry_data.get('value', 0)
            if usd_rate > 0:
                gold_usd = gold_data.get('value', 0)
                gold_tl = sf(gold_usd * usd_rate / 31.1035, 2)  # ons -> gram (1 ons = 31.1035 gram)
                data['GOLDTL'] = {
                    'name': 'Altin/TL (gram)',
                    'value': gold_tl,
                    'change': 0,
                    'changePct': gold_data.get('changePct', 0),
                    'volume': 0,
                }
        if usdtry_data and silver_data:
            usd_rate = usdtry_data.get('value', 0)
            if usd_rate > 0:
                silver_usd = silver_data.get('value', 0)
                silver_tl = sf(silver_usd * usd_rate / 31.1035, 2)  # ons -> gram
                data['SILVERTL'] = {
                    'name': 'Gumus/TL (gram)',
                    'value': silver_tl,
                    'change': 0,
                    'changePct': silver_data.get('changePct', 0),
                    'volume': 0,
                }

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
        return jsonify(safe_dict({'success':True,'stocks':stocks,'count':len(stocks),'sectors':list(SECTOR_MAP.keys()),'meta':_api_meta()}))
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
    """Tek hisse detay - SENKRON, 3 katmanli fallback (IsYatirim -> Yahoo -> yfinance)"""
    try:
        symbol=symbol.upper(); period=request.args.get('period','1y')

        # 1. Once hist cache'e bak
        hist=_cget_hist(f"{symbol}_{period}")

        # 2. Cache'de yoksa SENKRON cek (birlesik fetcher)
        if hist is None:
            print(f"[DETAIL] {symbol} {period} cekiliyor...")
            hist = _fetch_hist_df(symbol, period)
            if hist is not None and len(hist) >= 2:
                _cset(_hist_cache, f"{symbol}_{period}", hist)
                print(f"[DETAIL] {symbol} OK: {len(hist)} bar")
            else:
                hist = None

        # 3. Hist yoksa quick cache'den don
        if hist is None:
            quick=_cget(_stock_cache, symbol)
            if quick:
                return jsonify(safe_dict({
                    'success':True,'code':symbol,'name':quick['name'],
                    'price':quick['price'],'change':quick['change'],
                    'changePercent':quick['changePct'],
                    'volume':quick['volume'],'dayOpen':quick['open'],
                    'dayHigh':quick['high'],'dayLow':quick['low'],
                    'prevClose':quick['prevClose'],'currency':'TRY',
                    'period':period,'dataPoints':0,
                    'indicators':{},'chartData':{'candlestick':[],'dates':[],'prices':[],'volumes':[],'dataPoints':0},
                    'fibonacci':{'levels':{}},'supportResistance':{'supports':[],'resistances':[]},
                    'pivotPoints':{'classic':{},'camarilla':{},'woodie':{},'current':0},
                    'recommendation':{'weekly':{'action':'neutral','confidence':0,'reasons':[]},'monthly':{'action':'neutral','confidence':0,'reasons':[]},'yearly':{'action':'neutral','confidence':0,'reasons':[]}}
                }))
            return jsonify({'error':f'{symbol} verisi bulunamadi'}),404

        # 4. Hist var - TAM ANALIZ yap
        # Son NaN temizligi (guvenlik katmani)
        if hist[['Open','High','Low']].isna().any().any():
            hist['Open'] = hist['Open'].fillna(hist['Close'])
            hist['High'] = hist['High'].fillna(hist['Close'])
            hist['Low'] = hist['Low'].fillna(hist['Close'])
            hist['Volume'] = hist['Volume'].fillna(0)

        cp=float(hist['Close'].iloc[-1])
        prev=float(hist['Close'].iloc[-2]) if len(hist)>1 else cp
        w52=calc_52w(hist)
        ind=calc_all_indicators(hist,cp)
        rec=calc_recommendation(hist, ind)

        # ML Confidence (sadece aktif sinyal varsa)
        ml_conf = {}
        for tf_label in ['weekly', 'monthly', 'yearly']:
            tf_rec = rec.get(tf_label, {})
            tf_action = tf_rec.get('action', 'NOTR')
            if tf_action in ('AL', 'TUTUN/AL'):
                ml_conf[tf_label] = calc_ml_confidence(hist, ind, float(tf_rec.get('score', 0)), 'buy')
            elif tf_action in ('SAT', 'TUTUN/SAT'):
                ml_conf[tf_label] = calc_ml_confidence(hist, ind, float(tf_rec.get('score', 0)), 'sell')
            else:
                ml_conf[tf_label] = {'confidence': 50, 'grade': 'C', 'factors': []}

        return jsonify(safe_dict({
            'success':True,'code':symbol,
            'name':BIST100_STOCKS.get(symbol,symbol),
            'price':sf(cp),'change':sf(cp-prev),
            'changePercent':sf((cp-prev)/prev*100 if prev else 0),
            'volume':si(hist['Volume'].iloc[-1]),
            'dayHigh':sf(hist['High'].iloc[-1]),
            'dayLow':sf(hist['Low'].iloc[-1]),
            'dayOpen':sf(hist['Open'].iloc[-1]),
            'prevClose':sf(prev),'currency':'TRY',
            'period':period,'dataPoints':len(hist),
            'week52':w52,
            'marketValue':sf(cp * si(hist['Volume'].iloc[-1])),
            'indicators':ind,
            'chartData':prepare_chart_data(hist),
            'fibonacci':calc_fibonacci(hist),
            'supportResistance':calc_support_resistance(hist),
            'pivotPoints':calc_pivot_points(hist),
            'recommendation':rec,
            'mlConfidence':ml_conf,
            'signalBacktest':calc_signal_backtest(hist),
            'tradePlan':calc_trade_plan(hist, ind, symbol=symbol),
            'marketRegime':calc_market_regime(),
            'fundamentals':calc_fundamentals(hist, symbol),
            'meta':_api_meta(),
        }))
    except Exception as e:
        print(f"STOCK {symbol}: {traceback.format_exc()}")
        return jsonify({'error':str(e)}),500

@app.route('/api/commodity/<symbol>')
def commodity_detail(symbol):
    """Emtia detay - Altin, Gumus, GOLDTL, SILVERTL icin hisse gibi tam analiz"""
    try:
        symbol = symbol.upper()
        period = request.args.get('period', '1y')

        COMMODITY_MAP = {
            'GOLD': ('GC=F', 'Altin (USD/ons)', 'USD'),
            'SILVER': ('SI=F', 'Gumus (USD/ons)', 'USD'),
            'GOLDTL': ('GC=F', 'Altin/TL (gram)', 'TRY'),
            'SILVERTL': ('SI=F', 'Gumus/TL (gram)', 'TRY'),
            'USDTRY': ('USDTRY=X', 'Dolar/TL', 'TRY'),
            'EURTRY': ('EURTRY=X', 'Euro/TL', 'TRY'),
        }

        if symbol not in COMMODITY_MAP:
            return jsonify({'error': f'{symbol} desteklenmiyor. Desteklenen: {list(COMMODITY_MAP.keys())}'}), 400

        yahoo_sym, name, currency = COMMODITY_MAP[symbol]
        period_days = {'1mo':30,'3mo':90,'6mo':180,'1y':365,'2y':730,'5y':1825}.get(period, 365)

        # Hist cache key
        cache_key = f"COMMODITY_{symbol}_{period}"
        hist = _cget_hist(cache_key)

        if hist is None:
            print(f"[COMMODITY] {symbol} ({yahoo_sym}) {period} cekiliyor...")
            # Yahoo HTTP
            hist = _fetch_yahoo_http_df(yahoo_sym, period_days)
            # yfinance fallback
            if (hist is None or len(hist) < 10) and YF_OK:
                try:
                    h = yf.Ticker(yahoo_sym).history(period=period, timeout=15)
                    if h is not None and not h.empty and len(h) >= 10:
                        hist = h
                        print(f"  [YF-COMMODITY] {symbol} OK: {len(hist)} bar")
                except Exception as e:
                    print(f"  [YF-COMMODITY] {symbol}: {e}")

            if hist is not None and len(hist) >= 2:
                # GOLDTL / SILVERTL icin: USD fiyat * USDTRY / 31.1035 (ons->gram)
                if symbol in ('GOLDTL', 'SILVERTL'):
                    usd_hist = _fetch_yahoo_http_df('USDTRY=X', period_days)
                    if usd_hist is None and YF_OK:
                        try:
                            usd_hist = yf.Ticker('USDTRY=X').history(period=period, timeout=15)
                        except Exception:
                            pass
                    if usd_hist is not None and len(usd_hist) >= 2:
                        # Normalize dates (remove time component for matching)
                        hist.index = hist.index.normalize()
                        usd_hist.index = usd_hist.index.normalize()
                        # Remove duplicate dates (keep last)
                        hist = hist[~hist.index.duplicated(keep='last')]
                        usd_hist = usd_hist[~usd_hist.index.duplicated(keep='last')]
                        common_dates = hist.index.intersection(usd_hist.index)
                        print(f"  [COMMODITY] {symbol} tarih eslestirme: hist={len(hist)}, usd={len(usd_hist)}, ortak={len(common_dates)}")
                        if len(common_dates) >= 10:
                            hist = hist.loc[common_dates].copy()
                            usd_rates = usd_hist.loc[common_dates, 'Close']
                            ons_to_gram = 31.1035
                            for col in ['Open', 'High', 'Low', 'Close']:
                                hist[col] = hist[col] * usd_rates.values / ons_to_gram
                            print(f"  [COMMODITY] {symbol} TL donusumu OK: {len(hist)} bar, son fiyat: {hist['Close'].iloc[-1]:.2f}")
                        else:
                            # Fallback: tek kur ile tum seriyi donustur
                            print(f"  [COMMODITY] {symbol} TL donusumu: ortak tarih az, son kur ile donusturuluyor")
                            last_usd_rate = float(usd_hist['Close'].iloc[-1])
                            ons_to_gram = 31.1035
                            for col in ['Open', 'High', 'Low', 'Close']:
                                hist[col] = hist[col] * last_usd_rate / ons_to_gram
                    else:
                        # USDTRY verisi yok, indeks cache'den kur al
                        print(f"  [COMMODITY] {symbol}: USDTRY hist yok, cache'den kur aliniyor")
                        usd_idx = _cget(_index_cache, 'USDTRY')
                        if usd_idx:
                            last_rate = usd_idx.get('value', 0)
                            if last_rate > 0:
                                ons_to_gram = 31.1035
                                for col in ['Open', 'High', 'Low', 'Close']:
                                    hist[col] = hist[col] * last_rate / ons_to_gram

                _cset(_hist_cache, cache_key, hist)
                print(f"[COMMODITY] {symbol} OK: {len(hist)} bar")
            else:
                hist = None

        if hist is None:
            # Fallback: indeks cache'den basit veri don
            idx = _cget(_index_cache, symbol) or _cget(_index_cache, symbol.replace('TL',''))
            if idx:
                return jsonify(safe_dict({
                    'success': True, 'code': symbol, 'name': name,
                    'price': idx.get('value', 0), 'change': idx.get('change', 0),
                    'changePercent': idx.get('changePct', 0), 'volume': 0,
                    'currency': currency, 'period': period, 'dataPoints': 0,
                    'indicators': {}, 'chartData': {'dates':[],'prices':[],'volumes':[],'dataPoints':0},
                    'fibonacci': {'levels':{}}, 'supportResistance': {'supports':[],'resistances':[]},
                    'pivotPoints': {'classic':{},'camarilla':{},'woodie':{},'current':0},
                    'recommendation': {'weekly':{'action':'neutral','confidence':0,'reasons':[]},'monthly':{'action':'neutral','confidence':0,'reasons':[]},'yearly':{'action':'neutral','confidence':0,'reasons':[]}}
                }))
            return jsonify({'error': f'{symbol} verisi bulunamadi'}), 404

        # Tam analiz (hisse gibi)
        cp = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else cp
        w52 = calc_52w(hist)

        return jsonify(safe_dict({
            'success': True, 'code': symbol, 'name': name,
            'price': sf(cp, 4 if currency == 'USD' else 2),
            'change': sf(cp - prev, 4 if currency == 'USD' else 2),
            'changePercent': sf((cp - prev) / prev * 100 if prev else 0),
            'volume': si(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
            'dayHigh': sf(hist['High'].iloc[-1], 4 if currency == 'USD' else 2),
            'dayLow': sf(hist['Low'].iloc[-1], 4 if currency == 'USD' else 2),
            'dayOpen': sf(hist['Open'].iloc[-1], 4 if currency == 'USD' else 2),
            'prevClose': sf(prev, 4 if currency == 'USD' else 2),
            'currency': currency,
            'period': period, 'dataPoints': len(hist),
            'week52': w52,
            'indicators': calc_all_indicators(hist, cp),
            'chartData': prepare_chart_data(hist),
            'fibonacci': calc_fibonacci(hist),
            'supportResistance': calc_support_resistance(hist),
            'pivotPoints': calc_pivot_points(hist),
            'recommendation': calc_recommendation(hist, None),
            'fundamentals': calc_fundamentals(hist, symbol),
        }))
    except Exception as e:
        print(f"COMMODITY {symbol}: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/<symbol>/events')
def stock_events(symbol):
    return jsonify({'success':True,'symbol':symbol.upper(),'events':{'dividends':[],'splits':[]}})

@app.route('/api/stock/<symbol>/kap')
def stock_kap(symbol):
    """KAP bildirimlerini scrape et"""
    symbol = symbol.upper()
    try:
        url = f"https://www.kap.org.tr/tr/api/disclosures?company={symbol}&type=FR&lang=tr"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://www.kap.org.tr/',
        }
        try:
            resp = req_lib.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                notifications = []
                items = data if isinstance(data, list) else data.get('disclosures', data.get('data', []))
                for item in items[:20]:
                    notifications.append({
                        'title': item.get('title', item.get('subject', '')),
                        'date': item.get('publishDate', item.get('date', '')),
                        'type': item.get('type', item.get('disclosureType', '')),
                        'summary': item.get('summary', '')[:200],
                    })
                if notifications:
                    return jsonify({'success': True, 'symbol': symbol, 'notifications': notifications})
        except Exception as e:
            print(f"[KAP-API] {symbol}: {e}")

        # Fallback: Is Yatirim haberler
        try:
            url2 = f"https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HaberlerHisseTekil?hession={symbol}&startdate={datetime.now().strftime('%d-%m-%Y')}&enddate={datetime.now().strftime('%d-%m-%Y')}"
            resp2 = req_lib.get(url2, headers=IS_YATIRIM_HEADERS, timeout=10, verify=False)
            if resp2.status_code == 200:
                data2 = resp2.json()
                news = data2.get('value', [])
                notifications = [{'title': n.get('BASLIK', ''), 'date': n.get('TARIH', ''), 'type': 'haber', 'summary': ''} for n in news[:10]]
                return jsonify({'success': True, 'symbol': symbol, 'notifications': notifications})
        except Exception as e:
            print(f"[KAP-ISYATIRIM] {symbol}: {e}")

        return jsonify({'success': True, 'symbol': symbol, 'notifications': [], 'message': 'KAP verisi alinamadi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare():
    """Hisseleri detayli karsilastir"""
    try:
        data = request.json or {}
        symbols = data.get('symbols', [])
        if len(symbols) < 2:
            return jsonify({'error': 'En az 2 hisse gerekli'}), 400

        results = []
        for sym in symbols[:5]:
            sym = sym.upper()
            s = _cget(_stock_cache, sym)
            if not s:
                continue

            # Indicator hesapla
            hist = _cget_hist(f"{sym}_1y")
            indicators = {}
            if hist is not None and len(hist) >= 14:
                c = hist['Close'].values.astype(float)
                h = hist['High'].values.astype(float)
                l = hist['Low'].values.astype(float)
                rsi = calc_rsi(c)
                macd = calc_macd(c)
                ema = calc_ema(c, float(c[-1]))
                w52 = calc_52w(hist)
                indicators = {
                    'rsi': rsi.get('value', 0),
                    'rsiSignal': rsi.get('signal', 'neutral'),
                    'macdSignal': macd.get('signalType', 'neutral'),
                    'ema20': ema.get('ema20', 0),
                    'ema50': ema.get('ema50', 0),
                    'high52w': w52.get('high52w', 0),
                    'low52w': w52.get('low52w', 0),
                    'pos52w': w52.get('currentPct', 50),
                }

            results.append({
                'code': sym, 'name': s['name'], 'price': s['price'],
                'change': s['change'], 'changePct': s['changePct'],
                'volume': s['volume'], 'open': s.get('open', 0),
                'high': s.get('high', 0), 'low': s.get('low', 0),
                'prevClose': s.get('prevClose', 0),
                'gap': s.get('gap', 0), 'gapPct': s.get('gapPct', 0),
                **indicators
            })

        return jsonify(safe_dict({'success': True, 'comparison': results}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
                except Exception: ok=False; break
            if ok: matches.append(s)
        return jsonify(safe_dict({'success':True,'matches':matches[:50],'totalMatches':len(matches)}))
    except Exception as e:
        return jsonify({'error':str(e)}),500

# ---- PORTFOLIO / ALERTS / WATCHLIST (DB-backed, user-aware) ----
@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    try:
        uid = request.args.get('userId', request.args.get('user', ''))
        if not uid:
            return jsonify(safe_dict({'success':True,'positions':[],'summary':{'totalValue':0,'totalCost':0,'totalPnL':0,'totalPnLPct':0,'positionCount':0},'needsLogin':True}))
        db = get_db()
        rows = db.execute("SELECT * FROM portfolios WHERE user_id=?", (uid,)).fetchall()
        db.close()
        pos=[]; tv=tc=0
        for r in rows:
            cd=_cget(_stock_cache, r['symbol'])
            if not cd: continue
            q,ac,cp=r['quantity'],r['avg_cost'],float(cd.get('price') or 0)
            if cp<=0: continue
            mv=cp*q; cb=ac*q; upnl=mv-cb; tv+=mv; tc+=cb
            pos.append({'id':r['id'],'symbol':r['symbol'],'name':cd.get('name',r['symbol']),'quantity':q,'avgCost':sf(ac),'currentPrice':sf(cp),'marketValue':sf(mv),'costBasis':sf(cb),'unrealizedPnL':sf(upnl),'unrealizedPnLPct':sf(upnl/cb*100 if cb else 0),'changePct':float(cd.get('changePct') or 0),'weight':0})
        for p in pos: p['weight']=sf(float(p['marketValue'])/tv*100 if tv>0 else 0)
        tp=tv-tc
        dp=sum(float(p['marketValue'])*p['changePct']/100 for p in pos)
        return jsonify(safe_dict({'success':True,'positions':pos,'summary':{'totalValue':sf(tv),'totalCost':sf(tc),'totalPnL':sf(tp),'totalPnLPct':sf(tp/tc*100 if tc else 0),'dailyPnL':sf(dp),'positionCount':len(pos)}}))
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/portfolio', methods=['POST'])
def add_portfolio():
    try:
        d=request.json or {}
        uid=d.get('userId',d.get('user',''))
        sym=d.get('symbol','').upper(); qty=float(d.get('quantity',0)); ac=float(d.get('avgCost',0))
        if not uid: return jsonify({'error':'Giris yapmaniz gerekli'}),401
        if not sym or qty<=0 or ac<=0: return jsonify({'error':'Gecersiz veri'}),400
        db = get_db()
        existing = db.execute("SELECT * FROM portfolios WHERE user_id=? AND symbol=?", (uid, sym)).fetchone()
        if existing:
            new_qty = existing['quantity'] + qty
            new_avg = (existing['avg_cost'] * existing['quantity'] + ac * qty) / new_qty
            db.execute("UPDATE portfolios SET quantity=?, avg_cost=? WHERE id=?", (new_qty, new_avg, existing['id']))
        else:
            db.execute("INSERT INTO portfolios (user_id, symbol, quantity, avg_cost) VALUES (?, ?, ?, ?)", (uid, sym, qty, ac))
        db.commit(); db.close()
        return jsonify({'success':True,'message':f'{sym} portfoye eklendi'})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/portfolio', methods=['DELETE'])
def del_portfolio():
    try:
        d=request.json or {}
        uid=d.get('userId',d.get('user',''))
        sym=d.get('symbol','').upper()
        pid=d.get('id')
        db = get_db()
        if pid:
            db.execute("DELETE FROM portfolios WHERE id=?", (pid,))
        elif uid and sym:
            db.execute("DELETE FROM portfolios WHERE user_id=? AND symbol=?", (uid, sym))
        db.commit(); db.close()
        return jsonify({'success':True})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/portfolio/risk')
def portfolio_risk():
    return jsonify({'success':True,'risk':{'message':'Risk analizi yukleniyor'}})

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify(safe_dict({'success':True,'watchlist':[],'symbols':[],'needsLogin':True}))
        db = get_db()
        rows = db.execute("SELECT symbol FROM watchlists WHERE user_id=?", (uid,)).fetchall()
        db.close()
        symbols = [r['symbol'] for r in rows]
        stocks = _get_stocks(symbols) if symbols else []
        return jsonify(safe_dict({'success':True,'watchlist':stocks,'symbols':symbols}))
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/watchlist', methods=['POST'])
def update_watchlist():
    try:
        d=request.json or {}
        uid=d.get('userId','')
        sym=d.get('symbol','').upper()
        action=d.get('action','add')
        if not uid: return jsonify({'error':'Giris yapmaniz gerekli'}),401
        if not sym: return jsonify({'error':'Hisse kodu gerekli'}),400
        db = get_db()
        if action == 'add':
            try:
                db.execute("INSERT INTO watchlists (user_id, symbol) VALUES (?, ?)", (uid, sym))
            except sqlite3.IntegrityError:
                pass
        elif action == 'remove':
            db.execute("DELETE FROM watchlists WHERE user_id=? AND symbol=?", (uid, sym))
        elif action == 'toggle':
            existing = db.execute("SELECT id FROM watchlists WHERE user_id=? AND symbol=?", (uid, sym)).fetchone()
            if existing:
                db.execute("DELETE FROM watchlists WHERE id=?", (existing['id'],))
                action = 'removed'
            else:
                db.execute("INSERT INTO watchlists (user_id, symbol) VALUES (?, ?)", (uid, sym))
                action = 'added'
        db.commit(); db.close()
        return jsonify({'success':True,'action':action,'symbol':sym})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify(safe_dict({'success':True,'alerts':[],'needsLogin':True}))
        db = get_db()
        rows = db.execute("SELECT * FROM alerts WHERE user_id=? ORDER BY created_at DESC", (uid,)).fetchall()
        db.close()
        alerts = [{
            'id':r['id'],'symbol':r['symbol'],'condition':r['condition'],
            'targetValue':r['target_value'],'active':bool(r['active']),
            'triggered':bool(r['triggered']),'triggeredAt':r['triggered_at'],
            'createdAt':r['created_at'],
        } for r in rows]
        return jsonify(safe_dict({'success':True,'alerts':alerts}))
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/alerts', methods=['POST'])
def add_alert():
    try:
        d=request.json or {}
        uid=d.get('userId','')
        sym=d.get('symbol','').upper()
        condition=d.get('condition','price_above')
        target=float(d.get('targetValue',d.get('threshold',0)))
        if not uid: return jsonify({'error':'Giris yapmaniz gerekli'}),401
        if not sym or target<=0: return jsonify({'error':'Gecersiz veri'}),400
        db = get_db()
        db.execute("INSERT INTO alerts (user_id, symbol, condition, target_value) VALUES (?, ?, ?, ?)",
                   (uid, sym, condition, target))
        db.commit(); db.close()
        return jsonify({'success':True,'message':f'{sym} icin uyari eklendi'})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/alerts/<int:aid>', methods=['DELETE'])
def del_alert(aid):
    try:
        db = get_db()
        db.execute("DELETE FROM alerts WHERE id=?", (aid,))
        db.commit(); db.close()
        return jsonify({'success':True})
    except Exception as e: return jsonify({'error':str(e)}),500

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
    try:
        d = request.json or {}
        sym = d.get('symbol', '').upper()
        strategy = d.get('strategy', 'ma_cross')
        params = d.get('params', {})
        period = d.get('period', '1y')
        commission = float(d.get('commission', 0.001))
        initial_capital = float(d.get('initialCapital', 100000))

        if not sym:
            return jsonify({'error': 'Hisse kodu gerekli'}), 400

        hist = _fetch_hist_df(sym, period)
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{sym} icin yeterli veri yok'}), 400

        closes = hist['Close'].values.astype(float)
        highs = hist['High'].values.astype(float)
        lows = hist['Low'].values.astype(float)
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]
        n = len(closes)

        # Sinyal uret
        signals = [0] * n  # 1=al, -1=sat, 0=bekle

        if strategy == 'ma_cross':
            fast_p = int(params.get('fast', 20))
            slow_p = int(params.get('slow', 50))
            s = pd.Series(closes)
            fast_ma = s.rolling(fast_p).mean().values
            slow_ma = s.rolling(slow_p).mean().values
            for i in range(slow_p + 1, n):
                if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
                    signals[i] = 1
                elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
                    signals[i] = -1

        elif strategy == 'breakout':
            lookback = int(params.get('lookback', 20))
            for i in range(lookback, n):
                high_n = max(highs[i-lookback:i])
                low_n = min(lows[i-lookback:i])
                if closes[i] > high_n:
                    signals[i] = 1
                elif closes[i] < low_n:
                    signals[i] = -1

        elif strategy == 'mean_reversion':
            rsi_low = float(params.get('rsi_low', 30))
            rsi_high = float(params.get('rsi_high', 70))
            for i in range(15, n):
                rsi = calc_rsi_single(closes[:i+1])
                if rsi is not None:
                    if rsi < rsi_low: signals[i] = 1
                    elif rsi > rsi_high: signals[i] = -1

        # Backtest calistir
        cash = initial_capital
        shares = 0
        equity_curve = []
        trades = []
        peak_equity = initial_capital
        max_dd = 0
        wins = 0
        losses = 0

        for i in range(n):
            price = closes[i]
            if signals[i] == 1 and shares == 0:
                shares = int(cash * (1 - commission) / price)
                if shares > 0:
                    cost = shares * price * (1 + commission)
                    cash -= cost
                    trades.append({'date': dates[i], 'action': 'AL', 'price': sf(price), 'shares': shares, 'pnl': 0})
            elif signals[i] == -1 and shares > 0:
                revenue = shares * price * (1 - commission)
                pnl = revenue - (trades[-1]['shares'] * trades[-1]['price'] * (1 + commission)) if trades else 0
                cash += revenue
                trades.append({'date': dates[i], 'action': 'SAT', 'price': sf(price), 'shares': shares, 'pnl': sf(pnl)})
                if pnl > 0: wins += 1
                else: losses += 1
                shares = 0

            equity = cash + shares * price
            equity_curve.append({'date': dates[i], 'equity': sf(equity)})
            if equity > peak_equity: peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100
            if dd > max_dd: max_dd = dd

        final_equity = cash + shares * closes[-1]
        total_return = sf(((final_equity - initial_capital) / initial_capital) * 100)
        bh_return = sf(((closes[-1] - closes[0]) / closes[0]) * 100)
        years = n / 252
        cagr = sf(((final_equity / initial_capital) ** (1 / years) - 1) * 100) if years > 0 else 0

        daily_returns = np.diff([e['equity'] for e in equity_curve]) / np.array([e['equity'] for e in equity_curve[:-1]])
        sharpe = sf(float(np.mean(daily_returns) / np.std(daily_returns) * (252**0.5))) if len(daily_returns) > 0 and np.std(daily_returns) > 0 else 0

        total_trades = wins + losses
        win_rate = sf(wins / total_trades * 100) if total_trades > 0 else 0
        alpha = sf(float(total_return) - float(bh_return))

        # Exposure (piyasada kalma orani)
        in_market_days = sum(1 for i in range(n) if (any(t['action'] == 'AL' and dates.index(t['date']) <= i for t in trades if t['date'] in dates)))
        exposure_pct = sf(in_market_days / n * 100) if n > 0 else 0

        # Ortalama trade getirisi
        trade_pnls = [float(t['pnl']) for t in trades if t['action'] == 'SAT' and t['pnl'] != 0]
        avg_trade = sf(np.mean(trade_pnls)) if trade_pnls else 0
        avg_win = sf(np.mean([p for p in trade_pnls if p > 0])) if [p for p in trade_pnls if p > 0] else 0
        avg_loss = sf(np.mean([p for p in trade_pnls if p < 0])) if [p for p in trade_pnls if p < 0] else 0
        profit_factor = sf(abs(sum(p for p in trade_pnls if p > 0) / sum(p for p in trade_pnls if p < 0))) if any(p < 0 for p in trade_pnls) else 999

        return jsonify(safe_dict({
            'success': True,
            'results': {
                'totalReturn': total_return, 'cagr': cagr, 'sharpeRatio': sharpe,
                'maxDrawdown': sf(-max_dd), 'winRate': win_rate, 'totalTrades': total_trades,
                'buyAndHoldReturn': bh_return, 'alpha': alpha,
                'finalEquity': sf(final_equity), 'initialCapital': sf(initial_capital),
                'exposure': exposure_pct,
                'avgTrade': avg_trade,
                'avgWin': avg_win,
                'avgLoss': avg_loss,
                'profitFactor': profit_factor,
                'commission': sf(commission * 100),
            },
            'equityCurve': equity_curve[::max(1, len(equity_curve)//200)],
            'trades': trades[-50:],
            'warnings': [
                'Hayatta kalma yanliligi (Survivorship Bias): Bu backtest sadece bugun BIST100 endeksinde bulunan hisseleri kapsamaktadir. Gecmiste endeksten cikarilmis (iflas, birlesme, borsadan cikarma) hisseler dahil degildir. Gercek performans bu sonuclardan daha dusuk olabilir.',
                'Kayma (Slippage): Gercek islemlerde emir fiyati ile gerceklesen fiyat arasinda fark olabilir. Ozellikle dusuk hacimli hisselerde bu etki belirgindir.',
                'Komisyon: Backtest %' + str(sf(commission * 100)) + ' komisyon varsayimi kullanmaktadir.',
            ],
        }))
    except Exception as e:
        print(f"[BACKTEST] Hata: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# =====================================================================
# ISI HARITASI (HEATMAP)
# =====================================================================
@app.route('/api/heatmap')
def heatmap():
    """Sektor bazli isi haritasi verisi"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'sectors': [], 'loading': True})

        stock_map = {s['code']: s for s in stocks}
        sectors = []

        for sector_name, symbols in SECTOR_MAP.items():
            sector_stocks = []
            total_change = 0
            count = 0
            for sym in symbols:
                if sym in stock_map:
                    s = stock_map[sym]
                    sector_stocks.append({
                        'code': s['code'], 'name': s['name'],
                        'price': s['price'], 'changePct': s['changePct'],
                        'volume': s['volume'],
                    })
                    total_change += s['changePct']
                    count += 1

            avg_change = sf(total_change / count) if count > 0 else 0
            sectors.append({
                'name': sector_name,
                'displayName': {
                    'bankacilik': 'Bankacilik', 'havacilik': 'Havacilik',
                    'otomotiv': 'Otomotiv', 'enerji': 'Enerji',
                    'holding': 'Holding', 'perakende': 'Perakende',
                    'teknoloji': 'Teknoloji', 'telekom': 'Telekom',
                    'demir_celik': 'Demir Celik', 'gida': 'Gida',
                    'insaat': 'Insaat', 'gayrimenkul': 'Gayrimenkul',
                }.get(sector_name, sector_name),
                'avgChange': avg_change,
                'stockCount': count,
                'stocks': sorted(sector_stocks, key=lambda x: x['changePct'], reverse=True),
            })

        sectors.sort(key=lambda x: x['avgChange'], reverse=True)
        return jsonify(safe_dict({'success': True, 'sectors': sectors}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# GUNLUK RAPOR
# =====================================================================
@app.route('/api/report')
def daily_report():
    """Gunluk piyasa raporu"""
    try:
        stocks = _get_stocks()
        indices = _get_indices()

        if not stocks:
            return jsonify({'success': True, 'loading': True, 'message': 'Veriler yukleniyor'})

        # Piyasa genisligi
        up = [s for s in stocks if s.get('changePct', 0) > 0]
        down = [s for s in stocks if s.get('changePct', 0) < 0]
        unchanged = [s for s in stocks if s.get('changePct', 0) == 0]

        # En cok artanlar / azalanlar
        sorted_up = sorted(stocks, key=lambda x: x.get('changePct', 0), reverse=True)
        sorted_down = sorted(stocks, key=lambda x: x.get('changePct', 0))

        # Hacim liderleri
        sorted_vol = sorted(stocks, key=lambda x: x.get('volume', 0), reverse=True)

        # Gap analizi
        gap_up = sorted([s for s in stocks if s.get('gapPct', 0) > 1], key=lambda x: x.get('gapPct', 0), reverse=True)
        gap_down = sorted([s for s in stocks if s.get('gapPct', 0) < -1], key=lambda x: x.get('gapPct', 0))

        # Ortalama degisim
        all_changes = [s.get('changePct', 0) for s in stocks]
        avg_change = sf(np.mean(all_changes)) if all_changes else 0

        # Sektor performansi
        sector_perf = []
        stock_map = {s['code']: s for s in stocks}
        for sname, syms in SECTOR_MAP.items():
            changes = [stock_map[s]['changePct'] for s in syms if s in stock_map]
            if changes:
                sector_perf.append({
                    'name': sname, 'avgChange': sf(np.mean(changes)),
                    'bestStock': max([(s, stock_map[s]['changePct']) for s in syms if s in stock_map], key=lambda x: x[1])[0] if changes else '',
                })
        sector_perf.sort(key=lambda x: x['avgChange'], reverse=True)

        # Rapor metni olustur
        report_lines = []
        bist100 = indices.get('XU100', {})
        if bist100:
            direction = 'yukselis' if bist100.get('changePct', 0) > 0 else 'dusus'
            report_lines.append(f"BIST 100 endeksi %{bist100.get('changePct', 0)} {direction} gosteriyor.")

        report_lines.append(f"Toplam {len(stocks)} hisseden {len(up)} yukselen, {len(down)} dusen, {len(unchanged)} degismez.")
        report_lines.append(f"Piyasa ortalama degisimi: %{avg_change}")

        if sorted_up:
            report_lines.append(f"Gunun yildizi: {sorted_up[0]['code']} (%{sorted_up[0]['changePct']})")
        if sorted_down:
            report_lines.append(f"Gunun kaybi: {sorted_down[0]['code']} (%{sorted_down[0]['changePct']})")

        if sector_perf:
            report_lines.append(f"En iyi sektor: {sector_perf[0]['name']} (%{sector_perf[0]['avgChange']})")

        return jsonify(safe_dict({
            'success': True,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'summary': ' '.join(report_lines),
            'reportLines': report_lines,
            'marketBreadth': {
                'advancing': len(up), 'declining': len(down),
                'unchanged': len(unchanged), 'total': len(stocks),
                'avgChange': avg_change,
            },
            'topGainers': [{'code': s['code'], 'name': s['name'], 'changePct': s['changePct'], 'price': s['price']} for s in sorted_up[:5]],
            'topLosers': [{'code': s['code'], 'name': s['name'], 'changePct': s['changePct'], 'price': s['price']} for s in sorted_down[:5]],
            'volumeLeaders': [{'code': s['code'], 'name': s['name'], 'volume': s['volume'], 'changePct': s['changePct']} for s in sorted_vol[:5]],
            'gapUp': [{'code': s['code'], 'gapPct': s.get('gapPct', 0)} for s in gap_up[:5]],
            'gapDown': [{'code': s['code'], 'gapPct': s.get('gapPct', 0)} for s in gap_down[:5]],
            'sectorPerformance': sector_perf,
            'indices': indices,
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# SINYAL TARAMA (Tum hisseleri tarar, guclu AL/SAT sinyallerini bulur)
# =====================================================================
_signals_cache = {'data': None, 'ts': 0}
_signals_cache_lock = threading.Lock()
_opps_cache = {'data': None, 'ts': 0}
_opps_cache_lock = threading.Lock()
COMPUTED_CACHE_TTL = 120  # 2 dakika - agir hesaplamalar icin

def _compute_signal_for_stock(stock, timeframe):
    """Tek hisse icin sinyal hesapla (thread-safe, paralel calisir)"""
    sym = stock['code']
    try:
        hist = _cget_hist(f"{sym}_1y")
        if hist is None:
            return None

        c = hist['Close'].values.astype(float)
        cp = float(c[-1])

        ind = calc_all_indicators(hist, cp)
        summary = ind.get('summary', {})
        buy_count = summary.get('buySignals', 0)
        sell_count = summary.get('sellSignals', 0)
        total_ind = summary.get('totalIndicators', 1)

        rec = calc_recommendation(hist, ind)
        tf_rec = rec.get(timeframe, {})
        action = tf_rec.get('action', 'NOTR')
        score = float(tf_rec.get('score', 0))
        confidence = float(tf_rec.get('confidence', 0))

        consensus_pct = (buy_count / total_ind * 100) if total_ind > 0 else 50
        if score < 0:
            consensus_pct = 100 - consensus_pct
        composite = min(100, (abs(score) / 12 * 60) + (consensus_pct * 0.4))

        rsi_val = ind.get('rsi', {}).get('value', 50)
        macd_type = ind.get('macd', {}).get('signalType', 'neutral')
        macd_hist = ind.get('macd', {}).get('histogram', 0)

        sr = calc_support_resistance(hist)
        supports = sr.get('supports', [])[:2]
        resistances = sr.get('resistances', [])[:2]
        stop_loss = supports[0] if supports else sf(cp * 0.95)
        target = resistances[0] if resistances else sf(cp * 1.10)

        sig_type = 'buy' if score > 0 else 'sell'
        ml_conf = calc_ml_confidence(hist, ind, score, sig_type)
        candle_patterns = ind.get('candlestick', {}).get('patterns', [])

        # MTF: gercek coklu zaman dilimi analizi
        try:
            mtf = calc_mtf_signal(hist)
        except Exception:
            mtf = {'mtfScore': 0, 'mtfAlignment': '0/3',
                   'mtfDirection': 'neutral', 'mtfStrength': 'Uyumsuz'}

        # MTF filtresi: sinyal ile MTF yonu uyumsuzsa composite skoru zayiflat
        mtf_direction = mtf.get('mtfDirection', 'neutral')
        if mtf_direction != 'neutral' and mtf_direction != sig_type:
            composite = composite * 0.6   # Uyumsuz MTF → skoru %40 dusur
        elif mtf_direction == sig_type and mtf.get('mtfScore', 0) == 3:
            composite = min(100, composite * 1.2)  # 3/3 uyum → %20 artir

        # Faz 3: Divergence analizi
        try:
            div = calc_divergence(hist)
        except Exception:
            div = {'summary': {'signal': 'neutral', 'bullish': 0, 'bearish': 0, 'hasRecent': False}}
        div_summary = div.get('summary', {})
        div_signal  = div_summary.get('signal', 'neutral')

        # Divergence + composite uyum bonusu / penaltısı
        if div_summary.get('hasRecent', False):
            if div_signal == sig_type:
                composite = min(100, composite * 1.10)  # Uyumlu recent divergence → +%10
            elif div_signal != 'neutral':
                composite = composite * 0.85            # Ters divergence → -%15

        # Faz 4: Hacim Profili & VWAP
        try:
            vp = calc_volume_profile(hist)
        except Exception:
            vp = {'vwap': 0, 'poc': 0, 'vah': 0, 'val': 0, 'volumeAnomaly': False, 'vwapSignal': 'neutral'}

        # VWAP + composite uyum
        vwap_sig = vp.get('vwapSignal', 'neutral')
        if vwap_sig == sig_type:
            composite = min(100, composite * 1.05)  # Uyumlu VWAP → +%5
        elif vwap_sig != 'neutral':
            composite = composite * 0.95            # Ters VWAP → -%5

        # Hacim anomalisi → belirsizlik: composite'i hafif düşür
        if vp.get('volumeAnomaly', False) and sig_type == 'sell':
            composite = min(100, composite * 1.08)  # Yüksek hacim + sat → güçlü satış

        # Faz 5: SMC analizi
        try:
            smc = calc_smc(hist)
        except Exception:
            smc = {'signal': 'neutral', 'structureTrend': 'neutral', 'bullScore': 0, 'bearScore': 0,
                   'bosEvents': [], 'chochEvents': [], 'fvgs': [], 'orderBlocks': [],
                   'summary': {'hasBOS': False, 'hasCHoCH': False, 'activeFvgCount': 0, 'activeObCount': 0}}
        smc_signal  = smc.get('signal', 'neutral')
        smc_summary = smc.get('summary', {})

        # SMC CHoCH ters işaret → güçlü uyarı
        if smc_summary.get('hasCHoCH', False):
            choch_types = [cc.get('type', '') for cc in smc.get('chochEvents', [])]
            if any('bullish' in t for t in choch_types) and sig_type == 'buy':
                composite = min(100, composite * 1.15)  # Bullish CHoCH + al → +%15
            elif any('bearish' in t for t in choch_types) and sig_type == 'sell':
                composite = min(100, composite * 1.15)  # Bearish CHoCH + sat → +%15
        # BOS uyumu
        if smc_summary.get('hasBOS', False) and smc_signal == sig_type:
            composite = min(100, composite * 1.08)      # Uyumlu BOS → +%8

        # Faz 6: Grafik formasyonları
        try:
            patt = calc_chart_patterns(hist)
        except Exception:
            patt = {'signal': 'neutral', 'patterns': [], 'completedPatterns': [],
                    'summary': {'total': 0, 'bullish': 0, 'bearish': 0, 'completed': 0}}
        patt_signal  = patt.get('signal', 'neutral')
        patt_summary = patt.get('summary', {})

        # Tamamlanmış formasyon + uyumlu sinyal → güçlü bonus
        if patt_summary.get('completed', 0) > 0:
            if patt_signal == sig_type:
                composite = min(100, composite * 1.20)  # Tamamlanmış uyumlu formasyon → +%20
            elif patt_signal != 'neutral':
                composite = composite * 0.75            # Ters tamamlanmış formasyon → -%25
        elif patt_signal == sig_type:
            composite = min(100, composite * 1.05)      # Bekleyen uyumlu formasyon → +%5

        # Faz 7: Fibonacci & Pivot Noktaları
        try:
            fib    = calc_fibonacci_adv(hist)
            pivots = calc_pivot_points_adv(hist)
        except Exception:
            fib    = {'trend': 'neutral', 'goldenPocket': {'inZone': False}}
            pivots = {'bias': 'neutral', 'classic': {}}

        piv_bias = pivots.get('bias', 'neutral')
        if piv_bias == sig_type:
            composite = min(100, composite * 1.05)   # Pivot bias uyumu → +%5
        elif piv_bias != 'neutral':
            composite = composite * 0.97             # Ters pivot bias → -%3

        # Golden Pocket bölgesinde mi? (0.618-0.65) → kritik al/sat bölgesi
        in_golden = fib.get('goldenPocket', {}).get('inZone', False)
        fib_trend = fib.get('trend', 'neutral')
        if in_golden:
            if fib_trend == 'uptrend' and sig_type == 'buy':
                composite = min(100, composite * 1.12)  # Uptrend golden pocket → güçlü al
            elif fib_trend == 'downtrend' and sig_type == 'sell':
                composite = min(100, composite * 1.08)  # Downtrend golden pocket → sat

        # Faz 9: İleri Teknik İndikatörler
        try:
            adv = calc_advanced_indicators(hist)
        except Exception:
            adv = {'summary': {'signal': 'neutral', 'buyCount': 0, 'sellCount': 0}}

        adv_summary = adv.get('summary', {})
        adv_signal  = adv_summary.get('signal', 'neutral')
        adv_buy     = adv_summary.get('buyCount', 0)
        adv_sell    = adv_summary.get('sellCount', 0)

        # 3/3 ileri indikatör uyumu → güçlü sinyal
        if adv_signal == sig_type:
            if adv_buy == 3 or adv_sell == 3:
                composite = min(100, composite * 1.15)  # Tam uyum → +%15
            else:
                composite = min(100, composite * 1.07)  # Kısmi uyum → +%7
        elif adv_signal != 'neutral':
            composite = composite * 0.90                # Ters adv signal → -%10

        # Ichimoku cloud'un altında ve satış → ekstra baskı
        ich = adv.get('ichimoku', {})
        if ich.get('belowCloud', False) and sig_type == 'sell':
            composite = min(100, composite * 1.08)
        elif ich.get('aboveCloud', False) and sig_type == 'buy':
            composite = min(100, composite * 1.08)

        return {
            'code': sym,
            'name': BIST100_STOCKS.get(sym, sym),
            'price': sf(cp),
            'changePct': stock.get('changePct', 0),
            'volume': stock.get('volume', 0),
            'action': action,
            'score': sf(score),
            'confidence': sf(confidence),
            'composite': sf(composite),
            'mlConfidence': ml_conf.get('confidence', 50),
            'mlGrade': ml_conf.get('grade', 'C'),
            'buySignals': buy_count,
            'sellSignals': sell_count,
            'totalIndicators': total_ind,
            'rsi': sf(rsi_val),
            'macdSignal': macd_type,
            'macdHistogram': macd_hist,
            'stopLoss': stop_loss,
            'target': target,
            'supports': supports,
            'resistances': resistances,
            'reasons': tf_rec.get('reasons', [])[:5],
            'reason': tf_rec.get('reason', ''),
            'indicatorBreakdown': tf_rec.get('indicatorBreakdown', {}),
            'strategy': tf_rec.get('strategy', ''),
            'candlestickPatterns': candle_patterns[:3],
            'dynamicThresholds': ind.get('dynamicThresholds', {}),
            'tradePlan': calc_trade_plan(hist, ind, symbol=sym),
            # MTF alanları
            'mtfScore':       mtf.get('mtfScore', 0),
            'mtfAlignment':   mtf.get('mtfAlignment', '0/3'),
            'mtfDirection':   mtf_direction,
            'mtfStrength':    mtf.get('mtfStrength', 'Uyumsuz'),
            'mtfDescription': mtf.get('description', ''),
            # Faz 3: Divergence
            'divergenceSignal':  div_signal,
            'divergenceCount':   div_summary.get('count', 0),
            'divergenceBullish': div_summary.get('bullish', 0),
            'divergenceBearish': div_summary.get('bearish', 0),
            'hasRecentDivergence': div_summary.get('hasRecent', False),
            'divergences':       div.get('recentDivergences', [])[:3],
            # Faz 4: Hacim Profili & VWAP
            'vwap':           vp.get('vwap', 0),
            'vwapSignal':     vwap_sig,
            'vwapPct':        vp.get('vwapPct', 0),
            'poc':            vp.get('poc', 0),
            'vah':            vp.get('vah', 0),
            'val':            vp.get('val', 0),
            'volumeAnomaly':  vp.get('volumeAnomaly', False),
            'volumeRatio':    vp.get('volumeRatio', 0),
            'volumeTrend':    vp.get('volumeTrend', ''),
            # Faz 5: SMC
            'smcSignal':         smc_signal,
            'smcStructure':      smc.get('structureTrend', 'neutral'),
            'smcBullScore':      smc.get('bullScore', 0),
            'smcBearScore':      smc.get('bearScore', 0),
            'hasBOS':            smc_summary.get('hasBOS', False),
            'hasCHoCH':          smc_summary.get('hasCHoCH', False),
            'activeFvgCount':    smc_summary.get('activeFvgCount', 0),
            'activeObCount':     smc_summary.get('activeObCount', 0),
            'smcEntryZones':     smc.get('entryZones', [])[:3],
            # Faz 6: Grafik Formasyonları
            'patternSignal':     patt_signal,
            'patternCount':      patt_summary.get('total', 0),
            'completedPatterns': patt_summary.get('completed', 0),
            'bullishPatterns':   patt_summary.get('bullish', 0),
            'bearishPatterns':   patt_summary.get('bearish', 0),
            'patterns':          patt.get('completedPatterns', [])[:2] + patt.get('pendingPatterns', [])[:2],
            # Faz 7: Fibonacci & Pivot
            'fibTrend':          fib.get('trend', 'neutral'),
            'fibSwingHigh':      fib.get('swingHigh', 0),
            'fibSwingLow':       fib.get('swingLow', 0),
            'inGoldenPocket':    fib.get('goldenPocket', {}).get('inZone', False),
            'fibNearestSupport': (fib.get('nearestSupports', [{}]) or [{}])[0].get('level', 0),
            'fibNearestResist':  (fib.get('nearestResistances', [{}]) or [{}])[0].get('level', 0),
            'pivotBias':         piv_bias,
            'pivotPP':           pivots.get('classic', {}).get('pp', 0),
            'pivotR1':           pivots.get('classic', {}).get('r1', 0),
            'pivotS1':           pivots.get('classic', {}).get('s1', 0),
            # Faz 9: İleri İndikatörler
            'advSignal':         adv_signal,
            'advBuyCount':       adv_buy,
            'advSellCount':      adv_sell,
            'ichimokuSignal':    adv.get('ichimoku', {}).get('signal', 'neutral'),
            'aboveCloud':        adv.get('ichimoku', {}).get('aboveCloud', False),
            'belowCloud':        adv.get('ichimoku', {}).get('belowCloud', False),
            'stochasticK':       adv.get('stochastic', {}).get('k', 50),
            'stochasticSignal':  adv.get('stochastic', {}).get('signal', 'neutral'),
            'williamsR':         adv.get('williamsR', {}).get('value', -50),
            'williamsSignal':    adv.get('williamsR', {}).get('signal', 'neutral'),
        }
    except Exception:
        return None

@app.route('/api/signals')
def signal_scanner():
    """Tum hisselerin sinyal taramasi - composite score ile sirali (PARALEL)"""
    try:
        timeframe = request.args.get('timeframe', 'weekly')
        min_score = float(request.args.get('minScore', 0))
        signal_type = request.args.get('type', 'all')

        # Computed cache kontrol (2 dk)
        with _signals_cache_lock:
            sc = _signals_cache
            if sc['data'] and (time.time() - sc['ts']) < COMPUTED_CACHE_TTL:
                cached = sc['data']
                # Client-side filtreleme uygula
                filtered = cached['all_results']
                if signal_type == 'buy':
                    filtered = [r for r in filtered if float(r['score']) > 0]
                elif signal_type == 'sell':
                    filtered = [r for r in filtered if float(r['score']) < 0]
                if min_score > 0:
                    filtered = [r for r in filtered if abs(float(r['score'])) >= min_score]
                return jsonify(safe_dict({
                    'success': True, 'timeframe': timeframe,
                    'totalScanned': cached['totalScanned'],
                    'signalCount': len(filtered), 'signals': filtered,
                    'marketRegime': calc_market_regime(),
                    'timestamp': cached['timestamp'], 'meta': _api_meta(),
                }))

        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True, 'signals': [], 'message': 'Veriler yukleniyor...'})

        hist_ready = sum(1 for s in stocks if _cget_hist(f"{s['code']}_1y") is not None)
        if hist_ready < 10:
            return jsonify({'success': True, 'loading': True, 'signals': [], 'message': f'Tarihsel veriler hazirlaniyor ({hist_ready}/{len(stocks)})...'})

        # PARALEL sinyal hesaplama - ThreadPoolExecutor ile ~5x hiz
        results = []
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_compute_signal_for_stock, s, timeframe): s for s in stocks}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        results.sort(key=lambda x: float(x['score']), reverse=True)

        # Cache'e kaydet (filtresiz tam liste)
        with _signals_cache_lock:
            _signals_cache['data'] = {
                'all_results': results,
                'totalScanned': len(stocks),
                'timestamp': datetime.now().isoformat(),
            }
            _signals_cache['ts'] = time.time()

        # Filtrele
        filtered = results
        if signal_type == 'buy':
            filtered = [r for r in results if float(r['score']) > 0]
        elif signal_type == 'sell':
            filtered = [r for r in results if float(r['score']) < 0]
        if min_score > 0:
            filtered = [r for r in filtered if abs(float(r['score'])) >= min_score]

        return jsonify(safe_dict({
            'success': True, 'timeframe': timeframe,
            'totalScanned': len(stocks),
            'signalCount': len(filtered), 'signals': filtered,
            'marketRegime': calc_market_regime(),
            'timestamp': datetime.now().isoformat(), 'meta': _api_meta(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# FIRSAT OZETI (Gunluk/Haftalik/Aylik/Yillik)
# =====================================================================
def _compute_opportunity_for_stock(stock):
    """Tek hisse icin firsat analizi - gelismis versiyon:
    ADX trend gucu, RSI divergence, Stochastic, trend alignment, confluence filtresi"""
    sym = stock['code']
    try:
        hist = _cget_hist(f"{sym}_1y")
        if hist is None:
            return None

        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        cp = float(c[-1])
        n = len(c)

        if n < 30:
            return None

        events = []
        event_score = 0
        buy_count = 0
        sell_count = 0

        # === 1. RSI Oversold/Overbought ===
        rsi_val = calc_rsi(c).get('value', 50)
        if rsi_val < 30:
            events.append({'type': 'rsi_oversold', 'text': f'RSI {sf(rsi_val)} - Asiri satim bolgesinde, toparlanma bekleniyor', 'impact': 'positive'})
            event_score += 3
            buy_count += 1
        elif rsi_val > 70:
            events.append({'type': 'rsi_overbought', 'text': f'RSI {sf(rsi_val)} - Asiri alim bolgesinde, duzeltme gelebilir', 'impact': 'negative'})
            event_score -= 3
            sell_count += 1

        # === 2. RSI Divergence (en guvenilir sinyallerden biri) ===
        # Onceki 30 bar ile son 30 bar arasinda fiyat vs RSI uyumsuzlugu
        if n >= 60:
            recent_start = n - 30
            prior_start = n - 60
            # Bullish divergence: fiyat daha dusuk dip, RSI daha yuksek dip
            recent_min_abs = recent_start + int(np.argmin(c[recent_start:]))
            prior_min_abs = prior_start + int(np.argmin(c[prior_start:recent_start]))
            if recent_min_abs > 14 and prior_min_abs > 14:
                rsi_at_recent_low = calc_rsi_single(c[:recent_min_abs+1]) or 50
                rsi_at_prior_low = calc_rsi_single(c[:prior_min_abs+1]) or 50
                price_lower_low = float(c[recent_min_abs]) < float(c[prior_min_abs]) * 0.99
                rsi_higher_low = rsi_at_recent_low > rsi_at_prior_low + 3
                if price_lower_low and rsi_higher_low and rsi_val < 50:
                    events.append({'type': 'bullish_divergence', 'text': f'RSI Yukselis Uyumsuzlugu: Fiyat dip yaparken RSI yukseldi ({sf(rsi_at_prior_low)}->{sf(rsi_at_recent_low)}) - Guclu alis sinyali', 'impact': 'very_positive'})
                    event_score += 4
                    buy_count += 1
            # Bearish divergence: fiyat daha yuksek tepe, RSI daha dusuk tepe
            recent_max_abs = recent_start + int(np.argmax(c[recent_start:]))
            prior_max_abs = prior_start + int(np.argmax(c[prior_start:recent_start]))
            if recent_max_abs > 14 and prior_max_abs > 14:
                rsi_at_recent_high = calc_rsi_single(c[:recent_max_abs+1]) or 50
                rsi_at_prior_high = calc_rsi_single(c[:prior_max_abs+1]) or 50
                price_higher_high = float(c[recent_max_abs]) > float(c[prior_max_abs]) * 1.01
                rsi_lower_high = rsi_at_recent_high < rsi_at_prior_high - 3
                if price_higher_high and rsi_lower_high and rsi_val > 50:
                    events.append({'type': 'bearish_divergence', 'text': f'RSI Dusus Uyumsuzlugu: Fiyat zirve yaparken RSI dustü ({sf(rsi_at_prior_high)}->{sf(rsi_at_recent_high)}) - Guclu satis sinyali', 'impact': 'very_negative'})
                    event_score -= 4
                    sell_count += 1

        # === 3. MACD Crossover ===
        macd = calc_macd(c)
        if macd.get('signalType') == 'buy':
            events.append({'type': 'macd_cross', 'text': 'MACD alis kesisimi - Yukari momentum basladi', 'impact': 'positive'})
            event_score += 2
            buy_count += 1
        elif macd.get('signalType') == 'sell':
            events.append({'type': 'macd_cross', 'text': 'MACD satis kesisimi - Asagi momentum basladi', 'impact': 'negative'})
            event_score -= 2
            sell_count += 1

        # === 4. Golden/Death Cross ===
        if n >= 200:
            ema50 = pd.Series(c).ewm(span=50).mean().values
            ema200 = pd.Series(c).ewm(span=200).mean().values
            if ema50[-1] > ema200[-1] and ema50[-2] <= ema200[-2]:
                events.append({'type': 'golden_cross', 'text': 'ALTIN KESISIM! EMA50 > EMA200 - Guclu uzun vadeli alis sinyali', 'impact': 'very_positive'})
                event_score += 5
                buy_count += 1
            elif ema50[-1] < ema200[-1] and ema50[-2] >= ema200[-2]:
                events.append({'type': 'death_cross', 'text': 'OLUM KESISIMI! EMA50 < EMA200 - Guclu uzun vadeli satis sinyali', 'impact': 'very_negative'})
                event_score -= 5
                sell_count += 1

        # === 5. ADX - Trend Gucu ve Yonu (yeni!) ===
        adx_data = calc_adx(h, l, c)
        adx_val = float(adx_data.get('value', 25))
        plus_di = float(adx_data.get('plusDI', 0))
        minus_di = float(adx_data.get('minusDI', 0))
        sideways_market = adx_val < 15  # Yatay piyasa - sinyaller daha az guvenilir
        if adx_val > 30:
            if plus_di > minus_di:
                events.append({'type': 'adx_strong_bull', 'text': f'ADX={sf(adx_val)} - Guclu yukselis trendi (+DI={sf(plus_di)} > -DI={sf(minus_di)})', 'impact': 'positive'})
                event_score += 2
                buy_count += 1
            else:
                events.append({'type': 'adx_strong_bear', 'text': f'ADX={sf(adx_val)} - Guclu dusus trendi (-DI={sf(minus_di)} > +DI={sf(plus_di)})', 'impact': 'negative'})
                event_score -= 2
                sell_count += 1
        elif sideways_market:
            # Yatay piyasada tum skor %30 azalt (sinyaller daha az guvenilir)
            event_score = int(event_score * 0.7)

        # === 6. Stochastic Oversold/Overbought (yeni!) ===
        stoch = calc_stochastic(c, h, l)
        stoch_k = float(stoch.get('k', 50))
        if stoch_k < 20:
            events.append({'type': 'stoch_oversold', 'text': f'Stochastic %K={sf(stoch_k)} - Asiri satim bolgesinde, donus bekleniyor', 'impact': 'positive'})
            event_score += 2
            buy_count += 1
        elif stoch_k > 80:
            events.append({'type': 'stoch_overbought', 'text': f'Stochastic %K={sf(stoch_k)} - Asiri alim bolgesinde', 'impact': 'negative'})
            event_score -= 2
            sell_count += 1

        # === 7. Volume Spike ===
        if n >= 20:
            vol_avg = np.mean(v[-20:])
            vol_today = v[-1]
            if vol_avg > 0 and vol_today > vol_avg * 2:
                ratio = sf(vol_today / vol_avg)
                direction = 'yukselis' if c[-1] > c[-2] else 'dusus'
                impact = 'positive' if c[-1] > c[-2] else 'negative'
                events.append({'type': 'volume_spike', 'text': f'Hacim patlamasi ({ratio}x ortalama) + {direction} hareketi', 'impact': impact})
                if c[-1] > c[-2]:
                    event_score += 2; buy_count += 1
                else:
                    event_score -= 2; sell_count += 1

        # === 8. Bollinger Band ===
        bb = calc_bollinger(c, cp)
        if bb.get('lower', 0) > 0 and cp < bb['lower']:
            events.append({'type': 'bb_break_lower', 'text': f'Fiyat alt Bollinger bandinin altinda ({sf(bb["lower"])}) - Toparlanma bekleniyor', 'impact': 'positive'})
            event_score += 2
            buy_count += 1
        elif bb.get('upper', 0) > 0 and cp > bb['upper']:
            events.append({'type': 'bb_break_upper', 'text': f'Fiyat ust Bollinger bandini asti ({sf(bb["upper"])}) - Asiri alim', 'impact': 'negative'})
            event_score -= 1
            sell_count += 1

        # === 9. 52-Week High/Low (duzeltildi: 'position' -> 'currentPct') ===
        w52 = calc_52w(hist)
        w52_pos = w52.get('currentPct', 50)
        if w52_pos < 10:
            events.append({'type': '52w_low', 'text': f'52 haftalik dibin %{sf(w52_pos)} uzerinde - Tarihi dip bolgesi', 'impact': 'positive'})
            event_score += 2
            buy_count += 1
        elif w52_pos > 90:
            events.append({'type': '52w_high', 'text': f'52 haftalik zirveye %{sf(100-w52_pos)} mesafede', 'impact': 'neutral'})

        # === 10. Support/Resistance Breakout ===
        sr = calc_support_resistance(hist)
        if sr.get('resistances'):
            nearest_res = sr['resistances'][0]
            if cp > nearest_res * 0.99 and c[-2] < nearest_res:
                events.append({'type': 'resistance_break', 'text': f'Direnc kirdi ({sf(nearest_res)} TL) - Yukari kirilim', 'impact': 'positive'})
                event_score += 3
                buy_count += 1
        if sr.get('supports'):
            nearest_sup = sr['supports'][0]
            if cp < nearest_sup * 1.01 and c[-2] > nearest_sup:
                events.append({'type': 'support_break', 'text': f'Destek kirildi ({sf(nearest_sup)} TL) - Asagi kirilim', 'impact': 'negative'})
                event_score -= 3
                sell_count += 1

        # === 11. Trend Alignment - sinyal trend ile uyumlu mu? (yeni!) ===
        if n >= 50:
            s_pd = pd.Series(c)
            ema20_val = float(s_pd.ewm(span=20).mean().iloc[-1])
            ema50_val = float(s_pd.ewm(span=50).mean().iloc[-1])
            uptrend = cp > ema20_val > ema50_val
            downtrend = cp < ema20_val < ema50_val
            if event_score > 0 and uptrend:
                events.append({'type': 'trend_aligned_bull', 'text': f'Trend teyidi: Yukselis trendinde alis sinyali (EMA20 > EMA50) - Guclu uyum', 'impact': 'positive'})
                event_score += 2
            elif event_score > 0 and downtrend:
                events.append({'type': 'trend_counter_bull', 'text': f'Trend uyarisi: Dusus trendinde alis denemesi (EMA20 < EMA50) - Dikkat!', 'impact': 'neutral'})
                event_score -= 1
            elif event_score < 0 and downtrend:
                events.append({'type': 'trend_aligned_bear', 'text': f'Trend teyidi: Dusus trendinde satis sinyali (EMA20 < EMA50) - Guclu uyum', 'impact': 'negative'})
                event_score -= 2
            elif event_score < 0 and uptrend:
                events.append({'type': 'trend_counter_bear', 'text': f'Trend uyarisi: Yukselis trendinde satis denemesi (EMA20 > EMA50) - Dikkat!', 'impact': 'neutral'})
                event_score += 1

        # === 12. Candlestick Patterns ===
        o_arr = hist['Open'].values.astype(float) if 'Open' in hist.columns else c.copy()
        candles = calc_candlestick_patterns(o_arr, h, l, c)
        for p in candles.get('patterns', []):
            impact = 'positive' if p['type'] == 'bullish' else ('negative' if p['type'] == 'bearish' else 'neutral')
            events.append({'type': 'candlestick', 'text': f"Mum Formasyonu: {p['name']} - {p['description']}", 'impact': impact})
            if p['type'] == 'bullish':
                event_score += p['strength']; buy_count += 1
            elif p['type'] == 'bearish':
                event_score -= p['strength']; sell_count += 1

        # === 13. Confluence Filtresi - minimum 2 bagimsiz sinyal gerekli ===
        dominant_count = max(buy_count, sell_count)
        if dominant_count < 2:
            return None

        # === 14. Minimum Score Filtresi - zayif firsatlari elemsek ===
        MIN_SCORE = 4
        if abs(event_score) < MIN_SCORE:
            return None

        # === 15. Opportunity Score (0-100 normalize) ===
        # Maksimum teorik skor ~28 (tum sinyaller + divergence + golden cross + trend)
        MAX_POSSIBLE = 28
        opp_score = min(100, int(abs(event_score) / MAX_POSSIBLE * 100))
        opp_direction = 'buy' if event_score > 0 else 'sell'

        returns = {}
        for label, days in [('daily', 1), ('weekly', 5), ('monthly', 22), ('yearly', 252)]:
            actual_days = min(days, n - 1)
            if actual_days > 0:
                ret = ((c[-1] - c[-1-actual_days]) / c[-1-actual_days]) * 100
                returns[label] = sf(ret)
            else:
                returns[label] = 0

        dyn = calc_dynamic_thresholds(c, h, l, v)

        return {
            'code': sym,
            'name': BIST100_STOCKS.get(sym, sym),
            'price': sf(cp),
            'changePct': stock.get('changePct', 0),
            'eventScore': event_score,
            'opportunityScore': opp_score,
            'direction': opp_direction,
            'events': events,
            'eventCount': len(events),
            'buySignals': buy_count,
            'sellSignals': sell_count,
            'returns': returns,
            'rsi': sf(rsi_val),
            'adx': sf(adx_val),
            'stochastic': sf(stoch_k),
            'macdSignal': macd.get('signalType', 'neutral'),
            'sidewaysMarket': sideways_market,
            'dynamicThresholds': dyn,
            'candlestickPatterns': candles.get('patterns', []),
            'tradePlan': calc_trade_plan(hist, symbol=sym),
        }
    except Exception:
        return None

@app.route('/api/opportunities')
def opportunities():
    """Coklu zaman dilimli firsat raporu - PARALEL hesaplama"""
    try:
        # Computed cache kontrol (2 dk)
        with _opps_cache_lock:
            oc = _opps_cache
            if oc['data'] and (time.time() - oc['ts']) < COMPUTED_CACHE_TTL:
                return jsonify(safe_dict(oc['data']))

        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True, 'message': 'Veriler yukleniyor...'})

        hist_ready = sum(1 for s in stocks if _cget_hist(f"{s['code']}_1y") is not None)
        if hist_ready < 10:
            return jsonify({'success': True, 'loading': True, 'message': f'Tarihsel veriler hazirlaniyor ({hist_ready}/{len(stocks)})...'})

        # PARALEL firsat hesaplama
        opportunities_list = []
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_compute_opportunity_for_stock, s): s for s in stocks}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    opportunities_list.append(result)

        opportunities_list.sort(key=lambda x: x.get('opportunityScore', abs(x['eventScore'])), reverse=True)
        buy_opps = [o for o in opportunities_list if o['eventScore'] > 0]
        sell_opps = [o for o in opportunities_list if o['eventScore'] < 0]

        result_data = {
            'success': True,
            'totalScanned': len(stocks),
            'buyOpportunities': buy_opps[:20],
            'sellOpportunities': sell_opps[:20],
            'marketRegime': calc_market_regime(),
            'timestamp': datetime.now().isoformat(),
            'meta': _api_meta(),
        }

        with _opps_cache_lock:
            _opps_cache['data'] = result_data
            _opps_cache['ts'] = time.time()

        return jsonify(safe_dict(result_data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/locked-plans')
def locked_plans():
    """Kilitli yatirım planlarını döner (plan kilitleme sistemi)"""
    try:
        now = time.time()
        result = []
        with _plan_lock_cache_lock:
            items = list(_plan_lock_cache.items())
        for key, entry in items:
            parts = key.rsplit('_', 1)
            sym = parts[0]
            tf  = parts[1] if len(parts) == 2 else 'daily'
            cfg = PLAN_LOCK_CONFIG.get(tf, PLAN_LOCK_CONFIG['daily'])
            age = now - entry['locked_at']
            if age > cfg['max_lock']:
                continue
            remaining = max(0, cfg['max_lock'] - age)
            result.append({
                'symbol':       sym,
                'name':         BIST100_STOCKS.get(sym, sym),
                'timeframe':    tf,
                'lockedAt':     entry['locked_at'],
                'lockedPrice':  entry['locked_price'],
                'signal':       entry['signal'],
                'ageSeconds':   int(age),
                'remainingSec': int(remaining),
                'maxLockSec':   cfg['max_lock'],
                'tfPlan':       entry.get('tf_plan', {}),
            })
        # Skora gore sirala (fırsat olarak daha iyi olanlar once)
        result.sort(key=lambda x: x['lockedAt'], reverse=True)
        return jsonify({'success': True, 'plans': result, 'count': len(result)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# CANLI STRATEJI SIMULASYONU
# =====================================================================
_strat_cache = {'data': None, 'ts': 0}
_strat_cache_lock = threading.Lock()

def _compute_strategy_for_stock(stock):
    """Tek hisse icin 3 strateji hesapla (thread-safe)"""
    sym = stock['code']
    try:
        hist = _cget_hist(f"{sym}_1y")
        if hist is None:
            return None

        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        cp = float(c[-1])
        n = len(c)
        result = {'ma_cross': None, 'breakout': None, 'mean_reversion': None}

        if n >= 50:
            s = pd.Series(c)
            ema20 = s.ewm(span=20).mean().values
            ema50 = s.ewm(span=50).mean().values
            signal = None
            if ema20[-1] > ema50[-1] and ema20[-2] <= ema50[-2]: signal = 'AL'
            elif ema20[-1] < ema50[-1] and ema20[-2] >= ema50[-2]: signal = 'SAT'
            elif ema20[-1] > ema50[-1]: signal = 'ALIS POZISYONUNDA'
            elif ema20[-1] < ema50[-1]: signal = 'SATIS POZISYONUNDA'
            distance = sf(((ema20[-1] - ema50[-1]) / ema50[-1]) * 100)
            result['ma_cross'] = {
                'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                'price': sf(cp), 'changePct': stock.get('changePct', 0),
                'signal': signal, 'ema20': sf(ema20[-1]), 'ema50': sf(ema50[-1]),
                'distance': distance, 'freshSignal': signal in ('AL', 'SAT'),
            }

        if n >= 20:
            high_20 = float(np.max(h[-20:]))
            low_20 = float(np.min(l[-20:]))
            signal = None
            if cp >= high_20 * 0.99: signal = 'YUKARI KIRILIM'
            elif cp <= low_20 * 1.01: signal = 'ASAGI KIRILIM'
            else:
                pos = ((cp - low_20) / (high_20 - low_20) * 100) if high_20 != low_20 else 50
                signal = f'BANT ICINDE (%{sf(pos)})'
            result['breakout'] = {
                'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                'price': sf(cp), 'changePct': stock.get('changePct', 0),
                'signal': signal, 'high20': sf(high_20), 'low20': sf(low_20),
                'freshSignal': 'KIRILIM' in (signal or ''),
            }

        if n >= 15:
            rsi = calc_rsi(c).get('value', 50)
            signal = None
            if rsi < 30: signal = 'ASIRI SATIM → AL'
            elif rsi < 40: signal = 'SATIM BOLGESI → ALIS FIRSATI'
            elif rsi > 70: signal = 'ASIRI ALIM → SAT'
            elif rsi > 60: signal = 'ALIM BOLGESI → DIKKAT'
            else: signal = 'NOTR BOLGE'
            result['mean_reversion'] = {
                'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                'price': sf(cp), 'changePct': stock.get('changePct', 0),
                'signal': signal, 'rsi': sf(rsi), 'freshSignal': rsi < 30 or rsi > 70,
            }

        return result
    except Exception:
        return None

@app.route('/api/strategies/live')
def live_strategies():
    """3 stratejiyi tum hisselere canli uygular (PARALEL)"""
    try:
        # Computed cache kontrol (2 dk)
        with _strat_cache_lock:
            sc = _strat_cache
            if sc['data'] and (time.time() - sc['ts']) < COMPUTED_CACHE_TTL:
                return jsonify(safe_dict(sc['data']))

        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True, 'message': 'Veriler yukleniyor...'})

        results = {'ma_cross': [], 'breakout': [], 'mean_reversion': []}
        strategy_names = {
            'ma_cross': 'Hareketli Ortalama Kesisimi',
            'breakout': 'Kirilim Stratejisi',
            'mean_reversion': 'Ortalamaya Donus (RSI)',
        }

        # PARALEL strateji hesaplama
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_compute_strategy_for_stock, s): s for s in stocks}
            for future in as_completed(futures):
                r = future.result()
                if r:
                    for key in ('ma_cross', 'breakout', 'mean_reversion'):
                        if r[key]:
                            results[key].append(r[key])

        for key in results:
            results[key].sort(key=lambda x: (not x.get('freshSignal', False), -abs(x.get('changePct', 0))))

        result_data = {
            'success': True,
            'strategies': results,
            'strategyNames': strategy_names,
            'totalStocks': len(stocks),
            'timestamp': datetime.now().isoformat(),
        }

        with _strat_cache_lock:
            _strat_cache['data'] = result_data
            _strat_cache['ts'] = time.time()

        return jsonify(safe_dict(result_data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# TEMETTU TAKVIMLERI
# =====================================================================
@app.route('/api/dividends')
def dividend_calendar():
    """BIST hisselerinin temettu bilgileri (yfinance dividends)"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True, 'dividends': [], 'message': 'Veriler yukleniyor...'})

        # Eksik tarihsel verileri paralel cek
        # Sadece cache'deki veriyi kullan - HTTP fetch YAPMA
        dividends_list = []
        for stock in stocks:
            sym = stock['code']
            try:
                ticker_sym = sym + '.IS'
                hist = _cget_hist(f"{sym}_1y")

                # yfinance Ticker ile temettu bilgisi
                if YF_OK:
                    tkr = yf.Ticker(ticker_sym)
                    divs = tkr.dividends
                    if divs is not None and len(divs) > 0:
                        cp = stock.get('price', 0)
                        # Son 3 yilin temettuleri
                        recent_divs = []
                        total_div_1y = 0
                        one_year_ago = datetime.now() - timedelta(days=365)
                        for dt, amt in divs.items():
                            div_date = dt.to_pydatetime().replace(tzinfo=None) if hasattr(dt, 'to_pydatetime') else dt
                            recent_divs.append({
                                'date': div_date.strftime('%Y-%m-%d'),
                                'amount': sf(float(amt)),
                                'year': div_date.year,
                            })
                            if div_date >= one_year_ago:
                                total_div_1y += float(amt)

                        if recent_divs:
                            # Temettu verimi
                            div_yield = sf((total_div_1y / cp * 100) if cp > 0 else 0)
                            # Son temettu
                            last_div = recent_divs[-1]

                            # Yillik temettu ozeti
                            yearly_divs = {}
                            for d in recent_divs:
                                yr = d['year']
                                yearly_divs[yr] = yearly_divs.get(yr, 0) + float(d['amount'])

                            dividends_list.append({
                                'code': sym,
                                'name': BIST100_STOCKS.get(sym, sym),
                                'price': sf(cp),
                                'lastDividend': last_div,
                                'dividendYield': div_yield,
                                'totalDiv1Y': sf(total_div_1y),
                                'history': recent_divs[-10:],  # Son 10 temettu
                                'yearlyTotals': {str(k): sf(v) for k, v in sorted(yearly_divs.items())},
                                'changePct': stock.get('changePct', 0),
                            })
            except Exception:
                continue

        # Temettu verimine gore sirala
        dividends_list.sort(key=lambda x: float(x.get('dividendYield', 0)), reverse=True)

        return jsonify(safe_dict({
            'success': True,
            'count': len(dividends_list),
            'dividends': dividends_list,
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# TELEGRAM OTOMATIK BILDIRIM
# =====================================================================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

def send_telegram(message):
    """Telegram mesaji gonder"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        import requests as req
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        req.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}, timeout=10)
        return True
    except Exception:
        return False

def _auto_signal_check():
    """Arka planda guclu sinyalleri tespit edip Telegram bildirim gonder - Enhanced v7"""
    while True:
        try:
            time.sleep(600)  # 10dk arayla kontrol
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                continue

            stocks = _get_stocks()
            if not stocks:
                continue

            # Enhanced: check_signal_alerts kullan (mum formasyonu, dinamik esikler dahil)
            signal_alerts = check_signal_alerts()
            if not signal_alerts:
                continue

            # Piyasa rejimi
            regime = calc_market_regime()
            regime_emoji = {'strong_bull': '🐂🐂', 'bull': '🐂', 'strong_bear': '🐻🐻', 'bear': '🐻', 'sideways': '↔️'}.get(regime.get('regime', ''), '❓')

            alerts_text = []
            for alert in signal_alerts[:15]:
                emoji = '🟢' if alert.get('signal') == 'bullish' else ('🔴' if alert.get('signal') == 'bearish' else '⚪')
                alerts_text.append(f"{emoji} {alert['message']}")

            if alerts_text:
                header = f"📊 <b>BIST Sinyal Raporu v7</b> ({datetime.now().strftime('%H:%M')})\n"
                header += f"{regime_emoji} Piyasa: {regime.get('description', 'Bilinmiyor')}\n\n"
                msg = header + '\n'.join(alerts_text)
                send_telegram(msg)

        except Exception:
            continue

# Telegram bildirim thread'ini baslat
_telegram_thread_started = False
_telegram_thread_lock = threading.Lock()
def _start_telegram_thread():
    global _telegram_thread_started
    with _telegram_thread_lock:
        if _telegram_thread_started:
            return
        _telegram_thread_started = True
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        t = threading.Thread(target=_auto_signal_check, daemon=True)
        t.start()
        print("[TELEGRAM] Otomatik sinyal bildirimi aktif")

@app.route('/api/telegram/test')
def test_telegram():
    """Telegram baglantisini test et"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({'success': False, 'error': 'TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID env degiskenleri gerekli',
                        'setup': 'Render Dashboard > Environment Variables:\n1. TELEGRAM_BOT_TOKEN = @BotFather\'dan alinan token\n2. TELEGRAM_CHAT_ID = @userinfobot\'tan alinan chat ID'})
    ok = send_telegram("✅ BIST Pro Telegram bildirimi calisiyor!")
    return jsonify({'success': ok, 'message': 'Test mesaji gonderildi' if ok else 'Gonderilemedi'})

@app.route('/api/telegram/send-report', methods=['POST'])
def send_telegram_report():
    """Manuel sinyal raporu gonder"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'error': 'Veriler henuz yuklenmedi'}), 400

        strong_buys = []
        strong_sells = []
        for stock in stocks:
            sym = stock['code']
            try:
                hist = _cget_hist(f"{sym}_1y")
                if hist is None:
                    continue
                c = hist['Close'].values.astype(float)
                cp = float(c[-1])
                ind = calc_all_indicators(hist, cp)
                summary = ind.get('summary', {})
                bc = summary.get('buySignals', 0)
                sc = summary.get('sellSignals', 0)
                total = summary.get('totalIndicators', 1)

                if bc >= total * 0.6:
                    strong_buys.append((sym, sf(cp), bc, total, stock.get('changePct', 0)))
                elif sc >= total * 0.6:
                    strong_sells.append((sym, sf(cp), sc, total, stock.get('changePct', 0)))
            except Exception:
                continue

        msg = f"📊 <b>BIST Gunluk Sinyal Raporu</b>\n📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n"

        if strong_buys:
            msg += "🟢 <b>GUCLU ALIS SINYALLERI:</b>\n"
            for sym, price, bc, total, chg in sorted(strong_buys, key=lambda x: -x[2]):
                msg += f"  • {sym}: {price} TL (%{chg}) - {bc}/{total} AL\n"

        if strong_sells:
            msg += "\n🔴 <b>GUCLU SATIS SINYALLERI:</b>\n"
            for sym, price, sc, total, chg in sorted(strong_sells, key=lambda x: -x[2]):
                msg += f"  • {sym}: {price} TL (%{chg}) - {sc}/{total} SAT\n"

        if not strong_buys and not strong_sells:
            msg += "Guclu sinyal bulunamadi. Piyasa notr gorunuyor."

        ok = send_telegram(msg)
        return jsonify({'success': ok, 'message': msg})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# KULLANICI SISTEMI (AUTH)
# =====================================================================
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        d = request.json or {}
        username = d.get('username', '').strip()
        password = d.get('password', '')
        email = d.get('email', '').strip()

        if not username or len(username) < 3:
            return jsonify({'error': 'Kullanici adi en az 3 karakter olmali'}), 400
        if not password or len(password) < 4:
            return jsonify({'error': 'Sifre en az 4 karakter olmali'}), 400

        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if existing:
            db.close()
            return jsonify({'error': 'Bu kullanici adi zaten alinmis'}), 400

        user_id = str(uuid.uuid4())[:8]
        db.execute("INSERT INTO users (id, username, password_hash, email) VALUES (?, ?, ?, ?)",
                   (user_id, username, hash_password(password), email))
        db.commit()
        db.close()

        return jsonify({'success': True, 'userId': user_id, 'username': username})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        d = request.json or {}
        username = d.get('username', '').strip()
        password = d.get('password', '')

        if not username or not password:
            return jsonify({'error': 'Kullanici adi ve sifre gerekli'}), 400

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

        if not user:
            db.close()
            return jsonify({'error': 'Kullanici bulunamadi. Kayit olmaniz gerekebilir.'}), 401

        db.close()
        if user['password_hash'] != hash_password(password):
            return jsonify({'error': 'Sifre hatali'}), 401

        return jsonify({'success': True, 'userId': user['id'], 'username': user['username'], 'email': user['email'] or ''})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/profile', methods=['POST'])
def update_profile():
    try:
        d = request.json or {}
        user_id = d.get('userId', '')
        if not user_id:
            return jsonify({'error': 'Giris yapmaniz gerekli'}), 401

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if not user:
            db.close()
            return jsonify({'error': 'Kullanici bulunamadi'}), 404

        email = d.get('email', user['email'] or '')
        telegram = d.get('telegramChatId', user['telegram_chat_id'] or '')
        db.execute("UPDATE users SET email=?, telegram_chat_id=? WHERE id=?", (email, telegram, user_id))
        db.commit()
        db.close()
        return jsonify({'success': True, 'message': 'Profil guncellendi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/check')
def check_session():
    """localStorage oturumunun hala gecerli olup olmadigini kontrol et"""
    uid = request.args.get('userId', '')
    if not uid:
        return jsonify({'valid': False})
    try:
        db = get_db()
        user = db.execute("SELECT id FROM users WHERE id=?", (uid,)).fetchone()
        db.close()
        return jsonify({'valid': user is not None})
    except Exception:
        return jsonify({'valid': False})


# =====================================================================
# ALERT CHECK (tetikleme kontrolu)
# =====================================================================
@app.route('/api/alerts/check', methods=['POST'])
def check_alerts():
    """Tetiklenen uyarilari kontrol et"""
    try:
        d = request.json or {}
        uid = d.get('userId', '')
        if not uid:
            return jsonify({'success': True, 'triggered': []})
        db = get_db()
        rows = db.execute("SELECT * FROM alerts WHERE user_id=? AND active=1 AND triggered=0", (uid,)).fetchall()
        triggered = []
        for r in rows:
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
                db.execute("UPDATE alerts SET triggered=1, triggered_at=? WHERE id=?",
                           (datetime.now().isoformat(), r['id']))
                triggered.append({
                    'id': r['id'], 'symbol': r['symbol'], 'condition': r['condition'],
                    'targetValue': r['target_value'], 'currentPrice': price,
                    'message': f"{r['symbol']} uyarisi tetiklendi: {r['condition']} {r['target_value']} (Guncel: {price})"
                })
        db.commit()
        db.close()

        # Telegram bildirim gonder
        if triggered:
            _send_telegram_alerts(uid, triggered)

        return jsonify(safe_dict({'success': True, 'triggered': triggered}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# TELEGRAM BILDIRIM (per-user alerts)
# =====================================================================
def _send_telegram_alerts(user_id, triggered_alerts):
    """Tetiklenen uyarilari Telegram'a gonder"""
    if not TELEGRAM_BOT_TOKEN:
        return
    try:
        db = get_db()
        user = db.execute("SELECT telegram_chat_id FROM users WHERE id=?", (user_id,)).fetchone()
        db.close()
        if not user or not user['telegram_chat_id']:
            return

        chat_id = user['telegram_chat_id']
        for alert in triggered_alerts:
            text = f"🔔 *BIST Pro Uyari*\n\n{alert['message']}"
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            try:
                req_lib.post(url, json={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'}, timeout=5)
            except Exception:
                pass
    except Exception:
        pass


# =====================================================================
# YENI ENDPOINTLER - Sinyal Performans, Sektor RS, Market Regime, Alerts
# =====================================================================
@app.route('/api/market/regime')
def market_regime_endpoint():
    """Piyasa rejimi (boga/ayi/yatay)"""
    try:
        regime = calc_market_regime()
        return jsonify(safe_dict({'success': True, **regime}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sectors/analysis')
def sectors_analysis():
    """Sektor bazli goreceli guc analizi"""
    try:
        result = calc_sector_relative_strength()
        return jsonify(safe_dict({'success': True, **result}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/backtest-signals')
def stock_signal_backtest(symbol):
    """Hisse bazli sinyal backtest sonuclari"""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 50:
            return jsonify({'error': f'{symbol} icin yeterli veri yok'}), 400
        result = calc_signal_backtest(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/mtf')
def stock_mtf(symbol):
    """
    Hisse icin gercek coklu zaman dilimi (MTF) analizi.
    Gunluk OHLCV verisini haftalik ve aylik bara resample ederek
    her zaman diliminde RSI/MACD/EMA/Bollinger indikatörlerini hesaplar.
    Kac zaman diliminin ayni yonde oldugunu MTF skoru olarak doner (0-3).
    """
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 30 bar gerekli)'}), 400

        mtf = calc_mtf_signal(hist)
        return jsonify(safe_dict({
            'success': True,
            'symbol': symbol,
            'mtf': mtf,
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/divergence')
def stock_divergence(symbol):
    """
    Hisse icin RSI ve MACD uyumsuzluk (divergence) analizi.
    Klasik Boğa/Ayı + Gizli Boğa/Ayı + MACD divergence tespit eder.
    Son 90 barda tarama yapar; son 20 barda tespit edilenleri 'recentDivergences' olarak isaretler.
    """
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 50:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 50 bar)'}), 400
        result = calc_divergence(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/volume-profile')
def stock_volume_profile(symbol):
    """
    Hisse icin Hacim Profili ve VWAP analizi.
    POC (Point of Control), VAH/VAL (Value Area), VWAP, hacim anomalisi döner.
    """
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 20:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 20 bar)'}), 400
        result = calc_volume_profile(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/smc')
def stock_smc(symbol):
    """
    Hisse icin Smart Money Concepts (SMC) analizi.
    Fair Value Gap, Order Block, Break of Structure (BOS), Change of Character (CHoCH) döner.
    Kurumsal işlem izleri ve potansiyel giriş bölgeleri tespit eder.
    """
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 30 bar)'}), 400
        result = calc_smc(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/patterns')
def stock_chart_patterns(symbol):
    """
    Hisse icin grafik formasyon analizi.
    Çift Tepe/Dip, OBO/Ters OBO, Üçgen, Bayrak formasyonlarini tespit eder.
    Tamamlanmis formasyonlar ve hedef fiyat seviyeleri döner.
    """
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 30 bar)'}), 400
        result = calc_chart_patterns(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/fibonacci')
def stock_fibonacci(symbol):
    """
    Fibonacci retracement ve extension seviyeleri.
    Son 60 barin swing high/low noktasindan hesaplanir.
    Golden Pocket (0.618-0.65) bolgesi de isaretlenir.
    """
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 20:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 20 bar)'}), 400
        result = calc_fibonacci_adv(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/pivots')
def stock_pivot_points(symbol):
    """
    Klasik, Camarilla ve Woodie Pivot Noktalari.
    Son kapanan gunun OHLC'sinden hesaplanir.
    Destek/direnc seviyeleri + fiyat bias (bullish/bearish).
    """
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 3:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 3 bar)'}), 400
        result = calc_pivot_points_adv(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/advanced-indicators')
def stock_advanced_indicators(symbol):
    """
    Ileri teknik indikatörler:
    Ichimoku Cloud (bulut analizi), Stochastic (14,3,3), Williams %R (14).
    Her indikatörün sinyal + asiri alim/satim durumu döner.
    """
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 14:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 14 bar)'}), 400
        result = calc_advanced_indicators(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/full-analysis')
def stock_full_analysis(symbol):
    """
    Kapsamli tam analiz: Tum fazlari tek endpoint'te birlestirir.
    MTF + Divergence + Volume Profile + SMC + Patterns +
    Fibonacci + Pivots + Advanced Indicators + Temel Gostergeler.
    Frontend dashboard icin tek sorgu ile tam analiz.
    """
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 30 bar)'}), 400

        # Temel fiyat bilgisi
        stocks = _get_stocks()
        stock_info = next((s for s in stocks if s.get('code') == symbol), {})
        cp = float(hist['Close'].iloc[-1])

        # Tum analizleri paralel olmayan ama ozlü sekilde calistir
        def _safe(fn, *args, **kwargs):
            try: return fn(*args, **kwargs)
            except Exception as ex: return {'error': str(ex)}

        ind          = _safe(calc_indicators, hist)
        mtf          = _safe(calc_mtf_signal,       hist)
        div          = _safe(calc_divergence,        hist)
        vp           = _safe(calc_volume_profile,    hist)
        smc          = _safe(calc_smc,               hist)
        patterns     = _safe(calc_chart_patterns,    hist)
        fib          = _safe(calc_fibonacci_adv,         hist)
        pivots       = _safe(calc_pivot_points_adv,      hist)
        adv          = _safe(calc_advanced_indicators, hist)
        sr           = _safe(calc_support_resistance, hist)
        candles      = _safe(calc_candlestick_patterns, hist)
        backtest     = _safe(calc_signal_backtest,   hist)

        # Composite sinyal sayimi (tüm faz sinyalleri)
        all_signals = [
            mtf.get('mtfDirection', 'neutral'),
            div.get('summary', {}).get('signal', 'neutral'),
            vp.get('vwapSignal', 'neutral'),
            smc.get('signal', 'neutral'),
            patterns.get('signal', 'neutral'),
            adv.get('summary', {}).get('signal', 'neutral'),
            pivots.get('bias', 'neutral') if 'bias' in pivots else 'neutral',
        ]
        buy_votes  = all_signals.count('buy')
        sell_votes = all_signals.count('sell')
        consensus  = ('buy'  if buy_votes > sell_votes
                      else ('sell' if sell_votes > buy_votes else 'neutral'))
        confidence = round(max(buy_votes, sell_votes) / len(all_signals) * 100)

        return jsonify(safe_dict({
            'success':     True,
            'symbol':      symbol,
            'name':        BIST100_STOCKS.get(symbol, symbol),
            'price':       sf(cp),
            'changePct':   stock_info.get('changePct', 0),
            'consensus':   consensus,
            'buyVotes':    buy_votes,
            'sellVotes':   sell_votes,
            'neutralVotes': len(all_signals) - buy_votes - sell_votes,
            'confidence':  confidence,
            # Faz verileri
            'indicators':         ind,
            'mtf':                mtf,
            'divergence':         div,
            'volumeProfile':      vp,
            'smc':                smc,
            'chartPatterns':      patterns,
            'fibonacci':          fib,
            'pivots':             pivots,
            'advancedIndicators': adv,
            'supportResistance':  sr,
            'candlestickPatterns': candles,
            'backtest':           backtest,
            'timestamp':          datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/fundamentals')
def stock_fundamentals_endpoint(symbol):
    """Hisse temel analiz verileri (F/K, PD/DD)"""
    try:
        symbol = symbol.upper()
        result = fetch_fundamental_data(symbol)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, 'fundamentals': result}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/signals')
def signal_alerts_endpoint():
    """Sinyal bazli otomatik uyarilar (mum formasyonu, altin kesisim vb.)"""
    try:
        alerts = check_signal_alerts()
        return jsonify(safe_dict({
            'success': True,
            'alerts': alerts[:30],
            'totalAlerts': len(alerts),
            'marketRegime': calc_market_regime(),
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals/performance')
def signals_performance():
    """Tum hisselerin sinyal performans ozeti"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True})

        results = []
        for stock in stocks[:50]:  # Performans icin ilk 50
            sym = stock['code']
            try:
                hist = _cget_hist(f"{sym}_1y")
                if hist is None or len(hist) < 60:
                    continue
                bt = calc_signal_backtest(hist)
                if bt.get('totalSignals', 0) > 3:
                    overall = bt.get('overall', {})
                    results.append({
                        'code': sym,
                        'name': BIST100_STOCKS.get(sym, sym),
                        'totalSignals': bt['totalSignals'],
                        'winRate5d': overall.get('winRate5d', 0),
                        'winRate10d': overall.get('winRate10d', 0),
                        'winRate20d': overall.get('winRate20d', 0),
                        'avgReturn10d': overall.get('avgRet10d', 0),
                    })
            except Exception:
                continue

        results.sort(key=lambda x: float(x.get('winRate10d', 0)), reverse=True)
        return jsonify(safe_dict({
            'success': True,
            'performance': results,
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/signals/calibration')
def signals_calibration():
    """
    BIST indikatör kalibrasyonu:
    - Tekli hisse: ?symbol=THYAO
    - Toplu (ilk 20 hisse): parametre yok
    Döndürür: RSI en iyi eşik, en iyi performans gösteren indikatörler (Profit Factor bazli)
    """
    try:
        symbol = request.args.get('symbol', '').upper().strip()

        # Tek hisse modu
        if symbol:
            hist = _cget_hist(f"{symbol}_1y")
            if hist is None:
                hist = _fetch_hist_df(symbol, '1y')
            if hist is None or len(hist) < 60:
                return jsonify({'error': f'{symbol} icin yeterli veri yok'}), 400
            bt = calc_signal_backtest(hist)
            return jsonify(safe_dict({
                'success': True,
                'symbol': symbol,
                'rsiCalibration': bt.get('rsiCalibration', {}),
                'bestRsiThreshold': bt.get('bestRsiThreshold', '30/70'),
                'rankedIndicators': bt.get('rankedIndicators', [])[:5],
                'benchmark': bt.get('benchmark', {}),
                'totalSignals': bt.get('totalSignals', 0),
                'timestamp': datetime.now().isoformat(),
            }))

        # Toplu mod: ilk 20 hisse üzerinden RSI kalibrasyon ortalaması
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True})

        agg_rsi = {}   # { '30/70': [profitFactor, ...], ... }
        agg_ind = {}   # { 'RSI < 30': [profitFactor, ...], ... }
        processed = 0

        for stock in stocks[:20]:
            sym = stock['code']
            try:
                hist = _cget_hist(f"{sym}_1y")
                if hist is None or len(hist) < 60:
                    continue
                bt = calc_signal_backtest(hist)
                if bt.get('totalSignals', 0) < 5:
                    continue

                # RSI kalibrasyon topla
                for thresh, stats in bt.get('rsiCalibration', {}).items():
                    agg_rsi.setdefault(thresh, []).append(float(stats.get('profitFactor10d', 0)))

                # Indikatör sıralaması topla
                for ind in bt.get('rankedIndicators', []):
                    name = ind.get('reason', '')
                    if name:
                        agg_ind.setdefault(name, []).append(float(ind.get('profitFactor10d', 0)))

                processed += 1
            except Exception:
                continue

        if processed == 0:
            return jsonify({'success': True, 'loading': True, 'message': 'Veri hazırlanıyor'})

        # RSI eşik özeti
        rsi_summary = {}
        for thresh, pfs in agg_rsi.items():
            avg_pf = float(np.mean(pfs)) if pfs else 0
            rsi_summary[thresh] = {
                'avgProfitFactor10d': sf(avg_pf),
                'stockCount': len(pfs),
            }
        best_rsi_bulk = max(rsi_summary, key=lambda k: float(rsi_summary[k]['avgProfitFactor10d'])) if rsi_summary else '30/70'

        # Indikatör özeti
        ind_summary = []
        for name, pfs in agg_ind.items():
            avg_pf = float(np.mean(pfs)) if pfs else 0
            ind_summary.append({
                'reason': name,
                'avgProfitFactor10d': sf(avg_pf),
                'stockCount': len(pfs),
            })
        ind_summary.sort(key=lambda x: float(x['avgProfitFactor10d']), reverse=True)

        return jsonify(safe_dict({
            'success': True,
            'processedStocks': processed,
            'rsiCalibrationSummary': rsi_summary,
            'bestRsiThreshold': best_rsi_bulk,
            'topIndicators': ind_summary[:5],
            'allIndicators': ind_summary,
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# BES routes are in bes_system.py (auto-registered via import)

@app.route('/api/docs')
def docs():
    return jsonify({'name': 'BIST Pro v7.1.0', 'endpoints': [
        '/api/health', '/api/debug', '/api/dashboard', '/api/indices',
        '/api/bist100', '/api/bist30', '/api/stock/<sym>', '/api/stock/<sym>/kap',
        '/api/stock/<sym>/backtest-signals', '/api/stock/<sym>/fundamentals',
        '/api/commodity/<sym>', '/api/compare', '/api/screener', '/api/heatmap', '/api/report',
        '/api/backtest', '/api/sectors', '/api/sectors/analysis', '/api/search',
        '/api/signals', '/api/signals/performance', '/api/opportunities',
        '/api/market/regime', '/api/alerts/signals',
        '/api/auth/register', '/api/auth/login', '/api/auth/profile',
        '/api/portfolio', '/api/watchlist', '/api/alerts', '/api/alerts/check',
        '/api/bes/funds', '/api/bes/fund/<code>', '/api/bes/analyze',
        '/api/bes/simulate', '/api/bes/top',
        '/api/auto-trade/config', '/api/auto-trade/toggle',
        '/api/auto-trade/status', '/api/auto-trade/trades', '/api/auto-trade/close',
    ]})

# NO MODULE-LEVEL THREAD START - before_request handles it
print("[STARTUP] BIST Pro v7.1.0 ready - batch loader + SQLite + uyelik + advanced analytics + BES")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"[STARTUP] Port {port}")
    app.run(host='0.0.0.0', port=port)

