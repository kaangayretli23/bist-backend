"""Auth, Telegram bildirim ve kullanici yonetimi route'lari"""
import uuid
import threading
import time
from datetime import datetime
from flask import request, jsonify

from config import app, safe_dict, sf, BIST100_STOCKS, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from database import get_db, hash_password
from data_fetcher import _get_stocks, _cget_hist, _cget, _stock_cache
from indicators import calc_all_indicators
from signals import calc_market_regime, check_signal_alerts

try:
    import requests as req_lib
except ImportError:
    req_lib = None


# =====================================================================
# TELEGRAM FONKSIYONLARI
# =====================================================================
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

            signal_alerts = check_signal_alerts()
            if not signal_alerts:
                continue

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
# TELEGRAM ROUTE'LARI
# =====================================================================
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
