"""
BIST Pro v7.1.0 - Main Entry Point
Modular architecture: config, database, indicators, signals, data_fetcher,
trade_plans, auto_trader, bes_system, routes_*
"""
import gzip, io, json, traceback, threading, time, logging

# Socket.IO / engineio v2 heartbeat paketleri (~h~N) harmless ama log gürültüsü yaratır
logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('socketio').setLevel(logging.ERROR)

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
    # Routes (endpoint decoratorları) ayrı modüllerde — import edilince @app.route'lar kaydolur
    import auto_trader_routes  # noqa: F401
    import auto_trader_routes_positions  # noqa: F401
    import auto_trader_routes_analytics  # noqa: F401
except ImportError as e:
    print(f"[HATA] auto_trader.py import hatasi: {e}")

# BES fund system
try:
    import bes_system
except ImportError as e:
    print(f"[HATA] bes_system.py import hatasi: {e}")

# Route blueprints
from routes_system    import system_bp
from routes_market    import market_bp
from routes_portfolio import portfolio_bp
from routes_auth      import auth_bp
from routes_telegram  import telegram_bp, _start_telegram_thread
from routes_stock     import stock_bp
from routes_analysis         import analysis_bp
from routes_analysis_reports import analysis_reports_bp
from routes_news             import news_bp

app.register_blueprint(system_bp)
app.register_blueprint(market_bp)
app.register_blueprint(portfolio_bp)
app.register_blueprint(auth_bp)
app.register_blueprint(telegram_bp)
app.register_blueprint(stock_bp)
app.register_blueprint(analysis_bp)
app.register_blueprint(analysis_reports_bp)
app.register_blueprint(news_bp)

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


# Gerçek zamanlı fiyat monitörünü başlat
try:
    from realtime_prices import start_realtime_monitor
    start_realtime_monitor()
except Exception as e:
    print(f"[UYARI] Gerçek zamanlı fiyat monitörü başlatılamadı: {e}")

# KAP duyuru monitörünü başlat
try:
    from kap_scraper import start_kap_monitor
    start_kap_monitor()
except Exception as e:
    print(f"[UYARI] KAP monitör başlatılamadı: {e}")

# Telegram sinyal + rapor thread'lerini startup'ta hemen başlat
# (eskiden ilk HTTP request'e kadar bekliyordu — 10:25 açılış brifingi kaçabiliyordu)
try:
    _start_telegram_thread()
except Exception as e:
    print(f"[UYARI] Telegram thread başlatılamadı: {e}")

# Sinyal sonuç takip thread'i (saatlik)
def _outcome_checker_loop():
    time.sleep(300)  # İlk çalıştırmayı 5 dakika geciktir (DB hazır olsun)
    while True:
        try:
            from signal_tracker import check_pending_outcomes
            check_pending_outcomes()
        except Exception as e:
            print(f"[OUTCOME-CHECKER] Hata: {e}")
        time.sleep(3600)  # Saatte bir kontrol

_oc_thread = threading.Thread(target=_outcome_checker_loop, daemon=True)
_oc_thread.start()

# Pre-market watchlist (09:55-10:04 TR — hafta ici, gunde 1 kez)
try:
    from auto_trader_premarket import start_premarket_thread
    start_premarket_thread()
except Exception as e:
    print(f"[UYARI] Pre-market thread baslatilamadi: {e}")

# Gun sonu raporu (18:30-18:39 TR — hafta ici, gunde 1 kez)
try:
    from auto_trader_eod import start_eod_thread
    start_eod_thread()
except Exception as e:
    print(f"[UYARI] EOD thread baslatilamadi: {e}")

# ── STARTUP SELF-CHECK ──
# Kritik fonksiyonların import zinciri sağlam mı? NameError gibi sessiz hataları yakala.
def _startup_selfcheck():
    checks = [
        ('signals.calc_market_regime',    lambda: __import__('signals', fromlist=['calc_market_regime']).calc_market_regime),
        ('signals.check_signal_alerts',   lambda: __import__('signals', fromlist=['check_signal_alerts']).check_signal_alerts),
        ('signals.calc_recommendation',   lambda: __import__('signals', fromlist=['calc_recommendation']).calc_recommendation),
        ('indicators.calc_all_indicators',lambda: __import__('indicators', fromlist=['calc_all_indicators']).calc_all_indicators),
    ]
    ok, fail = 0, []
    for name, getter in checks:
        try:
            fn = getter()
            if not callable(fn):
                raise TypeError(f"{name} callable degil")
            ok += 1
        except Exception as e:
            fail.append(f"{name}: {e}")
    if fail:
        for msg in fail:
            print(f"[SELFCHECK] HATA — {msg}")
        print(f"[SELFCHECK] {ok}/{ok+len(fail)} gecti — yukardaki hataları düzelt")
    else:
        print(f"[SELFCHECK] Tüm {ok} kritik fonksiyon OK")

_startup_selfcheck()

print("[STARTUP] BIST Pro v7.1.0 ready - batch loader + SQLite + uyelik + advanced analytics + BES + KAP/Haber/Temel")

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    print(f"[STARTUP] Port {port}")
    app.run(host='0.0.0.0', port=port)
