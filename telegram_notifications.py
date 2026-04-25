"""
BIST Pro - Telegram Bildirim Gönderme
Temel mesaj gönderme + sinyal/SL/TP/trailing bildirimleri.
routes_telegram.py'dan ayrıştırıldı (600 satır kuralı).
"""
import time
import uuid
from datetime import datetime, timedelta

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


# =====================================================================
# TEMEL GONDERME FONKSIYONlARI
# =====================================================================

def send_telegram(message):
    """Trade bot — kritik AL/SAT bildirimleri (TELEGRAM_CHAT_ID)"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        import requests as req
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        req.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }, timeout=10)
        return True
    except Exception:
        return False


def _news_quiet_hours() -> bool:
    """BIST news botu sessiz saatleri: 18:00 sonrasi ve 07:00 oncesi (TR saati).
    Piyasa kapandiktan sonra kullaniciyi rahatsiz etmemek icin."""
    try:
        from datetime import datetime, timezone, timedelta
        _tr = datetime.now(timezone(timedelta(hours=3)))
        h = _tr.hour
        return h >= 18 or h < 7
    except Exception:
        return False


def send_news_telegram(message):
    """Haber/rapor botu — KAP haberleri, günlük raporlar (TELEGRAM_NEWS_BOT_TOKEN + TELEGRAM_NEWS_CHAT_ID).
    18:00–07:00 TR arasi sessiz (piyasa kapali)."""
    if not TELEGRAM_NEWS_BOT_TOKEN or not TELEGRAM_NEWS_CHAT_ID:
        return False
    if _news_quiet_hours():
        return False
    try:
        import requests as req
        url = f"https://api.telegram.org/bot{TELEGRAM_NEWS_BOT_TOKEN}/sendMessage"
        req.post(url, json={
            'chat_id': TELEGRAM_NEWS_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }, timeout=10)
        return True
    except Exception:
        return False


def send_telegram_with_keyboard(message, keyboard):
    """Inline butonlu Telegram mesaji gonder"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        import requests as req
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        req.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML',
            'reply_markup': {'inline_keyboard': keyboard}
        }, timeout=10)
        return True
    except Exception as e:
        print(f"[TELEGRAM] Keyboard mesaj hatasi: {e}")
        return False


def edit_telegram_message(message_id, new_text):
    """Mevcut Telegram mesajini duzenle (butonlari kaldir)"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        import requests as req
        req.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/editMessageText",
            json={
                'chat_id': TELEGRAM_CHAT_ID,
                'message_id': message_id,
                'text': new_text,
                'parse_mode': 'HTML'
            },
            timeout=5
        )
    except Exception:
        pass


# =====================================================================
# SINYAL BILDIRIMlERI
# =====================================================================

def send_trade_signal(uid, symbol, price, quantity, score, confidence, sl, tp1, tp2, tp3, trailing_sl):
    """Al sinyali bildirimi — Midas uyumlu format, onay/red butonlu"""
    with _pending_lock:
        for sig in _pending_signals.values():
            if sig['symbol'] == symbol and sig['uid'] == uid:
                return False

    signal_id = str(uuid.uuid4())[:8]

    # Hesaplamalar
    risk_pct   = round((price - sl) / price * 100, 1) if price > sl else 0
    tp1_pct    = round((tp1 - price) / price * 100, 1) if tp1 > price else 0
    tp2_pct    = round((tp2 - price) / price * 100, 1) if tp2 > price else 0
    tp3_pct    = round((tp3 - price) / price * 100, 1) if tp3 > price else 0
    rr         = round(tp1_pct / risk_pct, 1) if risk_pct > 0 else 0

    # Tam lot (Midas için integer)
    lot        = max(1, int(quantity))
    toplam_tl  = round(lot * price, 2)
    risk_tl    = round(lot * (price - sl), 2) if price > sl else 0
    kar_tl1    = round(lot * (tp1 - price), 2) if tp1 > price else 0

    # Limit emir önerisi: fiyatın %0.3 altı (daha iyi dolum)
    limit_oneri = round(price * 0.997, 2)

    with _pending_lock:
        _pending_signals[signal_id] = {
            'uid': uid, 'symbol': symbol, 'price': price,
            'quantity': lot, 'score': score, 'confidence': confidence,
            'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
            'trailing_sl': trailing_sl,
            'expires_at': datetime.now() + timedelta(minutes=15),
        }

    msg = (
        f"🟢 <b>AL SİNYALİ — {symbol}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📊 Skor: <b>{score:.1f}</b>  |  Güven: <b>%{confidence:.0f}</b>  |  R/R: <b>1:{rr}</b>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"📱 <b>MİDAS'A GİR:</b>\n"
        f"  1️⃣ Hisse: <b>{symbol}</b>\n"
        f"  2️⃣ Limit emir: <b>{limit_oneri:.2f} TL</b>  <i>(şu an: {price:.2f})</i>\n"
        f"  3️⃣ Adet: <b>{lot} adet</b>  <i>(toplam ~{toplam_tl:.0f} TL)</i>\n"
        f"  4️⃣ Stop-Loss: <b>{sl:.2f} TL</b>  <i>(-%{risk_pct} / -{risk_tl:.0f} TL)</i>\n"
        f"  5️⃣ Kâr al: <b>{tp1:.2f} TL</b>  <i>(+%{tp1_pct} / +{kar_tl1:.0f} TL)</i>\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"🎯 TP2: {tp2:.2f} TL  (+%{tp2_pct})\n"
        f"🎯 TP3: {tp3:.2f} TL  (+%{tp3_pct})\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"✅ Onaylarsan sisteme kaydedilir  |  ⏰ 15 dk"
    )

    keyboard = [[
        {'text': '✅ Aldım / Takibe Al', 'callback_data': f'approve_{signal_id}'},
        {'text': '❌ Geç', 'callback_data': f'reject_{signal_id}'}
    ]]

    return send_telegram_with_keyboard(msg, keyboard)


def send_position_closed_notification(symbol, close_price, pnl, pnl_pct, reason):
    """Pozisyon kapanma bildirimi"""
    clear_warning_cooldown(symbol)
    if pnl >= 0:
        emoji = "💚"
        label = "KÂR"
        pnl_str = f"+{pnl:.2f} TL  (+%{pnl_pct:.1f})"
    else:
        emoji = "🔴"
        label = "ZARAR"
        pnl_str = f"{pnl:.2f} TL  (%{pnl_pct:.1f})"

    msg = (
        f"{emoji} <b>{label}: {symbol} KAPATILDI</b>\n"
        f"💰 Çıkış fiyatı: {close_price:.2f} TL\n"
        f"📊 PnL: {pnl_str}\n"
        f"📝 Sebep: {reason}"
    )
    send_telegram(msg)


def _can_send_warning(key: str, cooldown: int) -> bool:
    """Cooldown süresi geçmişse True döner ve timestamp günceller"""
    now = time.time()
    with _warning_lock:
        last = _warning_cooldown.get(key, 0)
        if now - last < cooldown:
            return False
        _warning_cooldown[key] = now
        return True


def clear_warning_cooldown(symbol: str):
    """Pozisyon kapanınca veya trailing güncellenince cooldown sıfırla"""
    with _warning_lock:
        for k in list(_warning_cooldown.keys()):
            if k.endswith(f'_{symbol}'):
                del _warning_cooldown[k]


def send_sl_warning(symbol, cur_price, sl_price):
    """Stop-loss yaklasma uyarisi — saatte en fazla 1 kez"""
    if not _can_send_warning(f'SL_{symbol}', SL_WARNING_COOLDOWN):
        return
    pct = round((cur_price - sl_price) / sl_price * 100, 1)
    msg = (
        f"⚠️ <b>STOP-LOSS YAKLAŞIYOR: {symbol}</b>\n"
        f"💰 Güncel: {cur_price:.2f} TL\n"
        f"🛡 Stop-Loss: {sl_price:.2f} TL  (%{pct} uzakta)"
    )
    send_telegram(msg)


def send_tp_approaching(symbol, cur_price, tp_price, tp_label):
    """Hedef yaklasma bildirimi — saatte en fazla 1 kez"""
    if not _can_send_warning(f'TP_{symbol}_{tp_label}', TP_WARNING_COOLDOWN):
        return
    pct = round((tp_price - cur_price) / cur_price * 100, 1)
    msg = (
        f"🎯 <b>HEDEF YAKLAŞIYOR: {symbol}</b>\n"
        f"💰 Güncel: {cur_price:.2f} TL\n"
        f"🎯 {tp_label}: {tp_price:.2f} TL  (%{pct} kaldı)"
    )
    send_telegram(msg)


def send_trailing_update(symbol, new_trailing, highest_price, entry_price=None,
                         tp1=None, tp2=None, tp3=None, position_id=None):
    """Trailing stop yukarı kaydığında onay butonlu bildirim.
    Kullanıcı onaylarsa DB güncellenir, reddederse eski trailing kalır.
    """
    with _warning_lock:
        last_notif = _last_trailing_notified.get(symbol, 0)
        if last_notif > 0:
            move_pct = (new_trailing - last_notif) / last_notif * 100
            if move_pct < TRAILING_MIN_MOVE_PCT:
                return
    if not _can_send_warning(f'TRAIL_{symbol}', TRAILING_UPDATE_COOLDOWN):
        return
    with _warning_lock:
        _last_trailing_notified[symbol] = new_trailing

    profit_line = ""
    if entry_price and entry_price > 0:
        lock_pct = (new_trailing - entry_price) / entry_price * 100
        profit_line = f"\n🔒 Kilitlenen kâr: %{lock_pct:+.1f}"

    tp_line = ""
    cur = highest_price
    tp_parts = []
    if tp1 and tp1 > 0 and cur < tp1:
        tp_parts.append(f"TP1: {tp1:.2f}")
    if tp2 and tp2 > 0 and cur < tp2:
        tp_parts.append(f"TP2: {tp2:.2f}")
    if tp3 and tp3 > 0 and cur < tp3:
        tp_parts.append(f"TP3: {tp3:.2f}")
    if tp_parts:
        tp_line = "\n🎯 Hedefler: " + " → ".join(tp_parts)

    trail_id = str(uuid.uuid4())[:8]
    with _pending_trailing_lock:
        _pending_trailing[trail_id] = {
            'position_id': position_id,
            'symbol': symbol,
            'new_trailing': new_trailing,
            'new_highest': highest_price,
            'expires_at': datetime.now() + timedelta(minutes=30),
        }

    msg = (
        f"📈 <b>TRAILING STOP ÖNERİSİ: {symbol}</b>\n"
        f"🔝 Zirve: {highest_price:.2f} TL\n"
        f"🛡 Yeni Stop: {new_trailing:.2f} TL{profit_line}{tp_line}\n"
        f"⚡ Onaylarsan Midas'ta stop emrini bu seviyeye çek."
    )
    keyboard = [[
        {'text': '✅ Onayla', 'callback_data': f'trail_approve_{trail_id}'},
        {'text': '❌ Geç', 'callback_data': f'trail_reject_{trail_id}'}
    ]]
    send_telegram_with_keyboard(msg, keyboard)
