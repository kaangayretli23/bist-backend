"""
System & diagnostic routes: health, debug, ping, network-info, index page, docs.
"""
import os, gzip, io, shutil, socket, time
from datetime import datetime
from flask import Blueprint, jsonify, request, make_response
from config import (
    BASE_DIR, YF_OK,
    BIST100_STOCKS, CACHE_TTL, CACHE_STALE_TTL,
    _lock, _stock_cache, _index_cache, _hist_cache, _status, _loader_started,
    _cget, _cget_hist, _html_page_cache, _html_page_cache_lock,
    safe_dict,
)
try:
    from data_fetcher import (
        _fetch_isyatirim_quick, _fetch_isyatirim_df,
        _fetch_yahoo_http, _fetch_yahoo_http_df,
    )
except ImportError as e:
    print(f"[HATA] routes_system data_fetcher import: {e}")

system_bp = Blueprint('system', __name__)


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


@system_bp.route('/api/health')
def health():
    now = time.time()
    with _lock:
        fresh = [k for k, v in _stock_cache.items() if now - v['ts'] < CACHE_TTL]
        stale = [k for k, v in _stock_cache.items() if CACHE_TTL <= now - v['ts'] < CACHE_STALE_TTL]
        stock_count = len(_stock_cache)
        index_count = len(_index_cache)
        hist_count = len(_hist_cache)
        cached_stocks = list(_stock_cache.keys())
        cached_indices = list(_index_cache.keys())
        missing = [s for s in BIST100_STOCKS.keys() if s not in _stock_cache]
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
        'status': 'ok', 'version': '7.1.0', 'yf': YF_OK,
        'time': datetime.now().isoformat(),
        'loader': _status,
        'loaderStarted': _loader_started,
        'stockCache': stock_count,
        'stockCacheFresh': len(fresh),
        'stockCacheStale': len(stale),
        'indexCache': index_count,
        'histCache': hist_ready,
        'histCacheTotal': hist_count,
        'cachedStocks': cached_stocks,
        'cachedIndices': cached_indices,
        'totalDefined': len(BIST100_STOCKS),
        'missingStocks': missing,
        'disk': disk_info,
    })


@system_bp.route('/api/debug')
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


@system_bp.route('/api/test-fetch/<symbol>')
def test_fetch(symbol):
    """Veri cekme pipeline'ini test et - debug icin"""
    symbol = symbol.upper()
    results = {}

    try:
        data = _fetch_isyatirim_quick(symbol)
        results['isyatirim_quick'] = {'success': data is not None, 'data': data}
    except Exception as e:
        results['isyatirim_quick'] = {'success': False, 'error': str(e)}

    try:
        df = _fetch_isyatirim_df(symbol, days=30)
        if df is not None:
            nan_counts = df[['Open', 'High', 'Low', 'Close', 'Volume']].isna().sum().to_dict()
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

    try:
        data = _fetch_yahoo_http(f"{symbol}.IS")
        results['yahoo_http'] = {'success': data is not None, 'data': data}
    except Exception as e:
        results['yahoo_http'] = {'success': False, 'error': str(e)}

    try:
        df = _fetch_yahoo_http_df(f"{symbol}.IS", period1_days=30)
        if df is not None:
            results['yahoo_df'] = {'success': True, 'rows': len(df), 'last_close': float(df['Close'].iloc[-1])}
        else:
            results['yahoo_df'] = {'success': False, 'data': None}
    except Exception as e:
        results['yahoo_df'] = {'success': False, 'error': str(e)}

    cached = _cget(_stock_cache, symbol)
    results['cache'] = {'has_cache': cached is not None}
    if cached:
        results['cache']['data'] = cached

    return jsonify(safe_dict(results))


@system_bp.route('/ping')
def ping():
    return 'ok', 200


@system_bp.route('/api/network-info')
def network_info():
    """Yerel ag bilgisi — telefondan erisim icin IP goster"""
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


@system_bp.route('/')
def index():
    _load_index_html()  # mtime degistiyse otomatik yeniden yukler
    with _html_page_cache_lock:
        raw = _html_page_cache['raw']
        gz = _html_page_cache['gz']
    if not raw:
        return jsonify({'error': 'index.html bulunamadi'}), 500
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


@system_bp.route('/api/docs')
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
