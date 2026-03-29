"""
BIST Pro v7.1.0 - Main Entry Point
Modular architecture: config, database, indicators, signals, data_fetcher,
trade_plans, auto_trader, bes_system, routes_*
"""
import gzip, io, json, traceback
from flask import request, make_response

# Foundation
from config import *
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
    def init_db(): pass
    def _db_load_market_snapshot(): pass
    def _db_save_market_snapshot(): pass
    def _db_save_plan_locks(): pass
    def _db_upsert_cache(*a): pass
    def _db_get_cache(*a): return None

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

# Route blueprints
from routes_system   import system_bp
from routes_market   import market_bp
from routes_portfolio import portfolio_bp
from routes_auth     import auth_bp
from routes_telegram import telegram_bp, _start_telegram_thread
from routes_stock    import stock_bp
from routes_analysis import analysis_bp

app.register_blueprint(system_bp)
app.register_blueprint(market_bp)
app.register_blueprint(portfolio_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(telegram_bp)
app.register_blueprint(stock_bp)
app.register_blueprint(analysis_bp)

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
def e404(e): return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def e500(e): return jsonify({'error': str(e)}), 500

@app.errorhandler(Exception)
def eall(e):
    print(f"ERR: {traceback.format_exc()}")
    return jsonify({'error': str(e)}), 500


print("[STARTUP] BIST Pro v7.1.0 ready - batch loader + SQLite + uyelik + advanced analytics + BES")

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"[STARTUP] Port {port}")
    app.run(host='0.0.0.0', port=port)
