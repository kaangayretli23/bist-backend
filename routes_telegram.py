"""
Telegram notification routes and background signal alerting.
"""
import os, time, threading
from datetime import datetime
from flask import Blueprint, jsonify, request
from config import safe_dict, sf, _get_stocks, _cget_hist, BIST100_STOCKS

try:
    from indicators import calc_all_indicators
except ImportError as e:
    print(f"[HATA] routes_telegram indicators import: {e}")
try:
    from signals import calc_market_regime, check_signal_alerts
except ImportError as e:
    print(f"[HATA] routes_telegram signals import: {e}")

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

telegram_bp = Blueprint('telegram', __name__)

_telegram_thread_started = False
_telegram_thread_lock = threading.Lock()


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
            time.sleep(600)
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                continue

            stocks = _get_stocks()
            if not stocks:
                continue

            signal_alerts = check_signal_alerts()
            if not signal_alerts:
                continue

            regime = calc_market_regime()
            regime_emoji = {
                'strong_bull': '🐂🐂', 'bull': '🐂',
                'strong_bear': '🐻🐻', 'bear': '🐻',
                'sideways': '↔️',
            }.get(regime.get('regime', ''), '❓')

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


@telegram_bp.route('/api/telegram/test')
def test_telegram():
    """Telegram baglantisini test et"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({
            'success': False,
            'error': 'TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID env degiskenleri gerekli',
            'setup': 'Render Dashboard > Environment Variables:\n1. TELEGRAM_BOT_TOKEN = @BotFather\'dan alinan token\n2. TELEGRAM_CHAT_ID = @userinfobot\'tan alinan chat ID',
        })
    ok = send_telegram("✅ BIST Pro Telegram bildirimi calisiyor!")
    return jsonify({'success': ok, 'message': 'Test mesaji gonderildi' if ok else 'Gonderilemedi'})


@telegram_bp.route('/api/telegram/send-report', methods=['POST'])
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
