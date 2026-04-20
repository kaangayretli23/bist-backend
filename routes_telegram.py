"""
BIST Pro - Telegram HTTP Endpoints
4 adet @telegram_bp.route('/api/telegram/...') endpointi.

Tüm Telegram mantığı 3 alt modüle bölündü (600 satır kuralı):
  - telegram_state         → shared globals (locks, dicts, env constants)
  - telegram_notifications → send_telegram + sinyal/SL/TP/trailing bildirimleri
  - telegram_callbacks     → buton handler'ları + polling/cleanup thread'leri

Geriye dönük uyumluluk için tüm public surface bu modülden re-export edilir
(kap_scraper, auto_trader, telegram_reports, vb. eski import yollarını kullanır).
"""
from datetime import datetime
from flask import Blueprint, jsonify
from config import safe_dict, sf, _get_stocks, _cget_hist
from auth_middleware import require_user

try:
    from indicators import calc_all_indicators
except ImportError as e:
    print(f"[HATA] routes_telegram indicators import: {e}")

# ---- Re-export public surface (bakward compat) ----
from telegram_state import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    TELEGRAM_NEWS_BOT_TOKEN, TELEGRAM_NEWS_CHAT_ID,
    _pending_signals, _pending_lock,
    _pending_trailing, _pending_trailing_lock,
    _warning_cooldown, _warning_lock,
    _last_trailing_notified,
    SL_WARNING_COOLDOWN, TP_WARNING_COOLDOWN,
    TRAILING_UPDATE_COOLDOWN, TRAILING_MIN_MOVE_PCT,
)
from telegram_notifications import (
    send_telegram, send_news_telegram,
    send_telegram_with_keyboard, edit_telegram_message,
    send_trade_signal, send_position_closed_notification,
    _can_send_warning, clear_warning_cooldown,
    send_sl_warning, send_tp_approaching, send_trailing_update,
)
from telegram_callbacks import (
    _answer_callback, _handle_approve, _handle_reject,
    _handle_trailing_approve, _handle_trailing_reject,
    _process_update, _telegram_polling, _cleanup_expired_signals,
    _start_telegram_thread,
)


telegram_bp = Blueprint('telegram', __name__)


@telegram_bp.route('/api/telegram/test')
def test_telegram():
    """Telegram baglantisini test et"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({
            'success': False,
            'error': 'TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID env degiskenleri gerekli',
        })
    ok = send_telegram("✅ BIST Pro Telegram bildirimi calisiyor!")
    return jsonify({'success': ok, 'message': 'Test mesaji gonderildi' if ok else 'Gonderilemedi'})


@telegram_bp.route('/api/telegram/pending-signals')
def get_pending_signals():
    """Bekleyen sinyalleri listele"""
    with _pending_lock:
        signals = [
            {
                'id': sid,
                'symbol': s['symbol'],
                'price': s['price'],
                'score': s['score'],
                'confidence': s['confidence'],
                'expiresAt': s['expires_at'].isoformat(),
            }
            for sid, s in _pending_signals.items()
        ]
    with _pending_trailing_lock:
        trailing_updates = [
            {
                'id': tid,
                'symbol': t['symbol'],
                'newTrailing': t['new_trailing'],
                'newHighest': t['new_highest'],
                'expiresAt': t['expires_at'].isoformat(),
            }
            for tid, t in _pending_trailing.items()
        ]
    return jsonify(safe_dict({
        'success': True, 'signals': signals, 'count': len(signals),
        'trailingUpdates': trailing_updates, 'trailingCount': len(trailing_updates),
    }))


@telegram_bp.route('/api/telegram/send-report', methods=['POST'])
@require_user
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
                sc_count = summary.get('sellSignals', 0)
                total = summary.get('totalIndicators', 1)

                if bc >= total * 0.6:
                    strong_buys.append((sym, sf(cp), bc, total, stock.get('changePct', 0)))
                elif sc_count >= total * 0.6:
                    strong_sells.append((sym, sf(cp), sc_count, total, stock.get('changePct', 0)))
            except Exception:
                continue

        msg = f"📊 <b>BIST Gunluk Sinyal Raporu</b>\n📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n"

        if strong_buys:
            msg += "🟢 <b>GUCLU ALIS SINYALLERI:</b>\n"
            for sym, price, bc, total, chg in sorted(strong_buys, key=lambda x: -x[2]):
                msg += f"  • {sym}: {price} TL (%{chg}) - {bc}/{total} AL\n"

        if strong_sells:
            msg += "\n🔴 <b>GUCLU SATIS SINYALLERI:</b>\n"
            for sym, price, sc_count, total, chg in sorted(strong_sells, key=lambda x: -x[2]):
                msg += f"  • {sym}: {price} TL (%{chg}) - {sc_count}/{total} SAT\n"

        if not strong_buys and not strong_sells:
            msg += "Guclu sinyal bulunamadi. Piyasa notr gorunuyor."

        ok = send_telegram(msg)
        return jsonify({'success': ok, 'message': msg})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@telegram_bp.route('/api/telegram/report/<period>')
def send_performance_report(period):
    """Aninda rapor gonder:
       /api/telegram/report/daily     — gunluk performans
       /api/telegram/report/weekly    — haftalik performans
       /api/telegram/report/market    — bugunun piyasa kapanis raporu
       /api/telegram/report/open      — acilis brifing
    """
    from telegram_reports import _build_performance_report, _build_market_report
    if period == 'daily':
        ok = send_telegram(_build_performance_report(days=1))
    elif period == 'weekly':
        ok = send_telegram(_build_performance_report(days=7))
    elif period == 'market':
        ok = send_telegram(_build_market_report(period='close'))
    elif period == 'open':
        ok = send_telegram(_build_market_report(period='open'))
    else:
        return jsonify({'error': 'period: daily | weekly | market | open'}), 400
    return jsonify({'success': ok, 'period': period})
