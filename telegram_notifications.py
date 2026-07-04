"""
BIST Pro - Telegram Bildirim Gönderme
Temel mesaj gönderme + sinyal/SL/TP/trailing bildirimleri.
routes_telegram.py'dan ayrıştırıldı (600 satır kuralı).
"""
import time
import uuid
from datetime import datetime, timedelta, timezone

_TZ_TR = timezone(timedelta(hours=3))

from telegram_state import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    TELEGRAM_NEWS_BOT_TOKEN, TELEGRAM_NEWS_CHAT_ID,
    _pending_signals, _pending_lock,
    _pending_trailing, _pending_trailing_lock,
    _pending_sl_change, _pending_sl_change_lock,
    _pending_tp_exec, _tp_exec_asked, _pending_tp_exec_lock,
    _warning_cooldown, _warning_lock,
    _last_trailing_notified,
    SL_WARNING_COOLDOWN, TP_WARNING_COOLDOWN,
    TRAILING_UPDATE_COOLDOWN, TRAILING_MIN_MOVE_PCT,
)


# =====================================================================
# TEMEL GONDERME FONKSIYONlARI
# =====================================================================

def send_telegram(message):
    """Trade bot — kritik AL/SAT bildirimleri (TELEGRAM_CHAT_ID).
    HTTP cevabini dogrular: 200 + body['ok']==True olmadigi surece False doner.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        import requests as req
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = req.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }, timeout=10)
        if not resp.ok:
            print(f"[TELEGRAM] send_telegram HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        try:
            data = resp.json()
        except Exception:
            return False
        if not data.get('ok'):
            print(f"[TELEGRAM] send_telegram API hata: {data.get('description', '?')}")
            return False
        return True
    except Exception as e:
        print(f"[TELEGRAM] send_telegram exception: {e}")
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
    18:00–07:00 TR arasi sessiz (piyasa kapali).
    HTTP cevabini dogrular."""
    if not TELEGRAM_NEWS_BOT_TOKEN or not TELEGRAM_NEWS_CHAT_ID:
        return False
    if _news_quiet_hours():
        return False
    try:
        import requests as req
        url = f"https://api.telegram.org/bot{TELEGRAM_NEWS_BOT_TOKEN}/sendMessage"
        resp = req.post(url, json={
            'chat_id': TELEGRAM_NEWS_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }, timeout=10)
        if not resp.ok:
            print(f"[TELEGRAM-NEWS] HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        try:
            data = resp.json()
        except Exception:
            return False
        if not data.get('ok'):
            print(f"[TELEGRAM-NEWS] API hata: {data.get('description', '?')}")
            return False
        return True
    except Exception as e:
        print(f"[TELEGRAM-NEWS] exception: {e}")
        return False


def send_telegram_with_keyboard(message, keyboard):
    """Inline butonlu Telegram mesaji gonder. HTTP cevabini dogrular —
    bu fonksiyonun True donmesi sinyalin gercekten kullanicıya ulastigini ifade
    eder; trade onay akisinin guvenligi buna bagli."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        import requests as req
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = req.post(url, json={
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML',
            'reply_markup': {'inline_keyboard': keyboard}
        }, timeout=10)
        if not resp.ok:
            print(f"[TELEGRAM] keyboard HTTP {resp.status_code}: {resp.text[:200]}")
            return False
        try:
            data = resp.json()
        except Exception:
            return False
        if not data.get('ok'):
            print(f"[TELEGRAM] keyboard API hata: {data.get('description', '?')}")
            return False
        return True
    except Exception as e:
        print(f"[TELEGRAM] keyboard exception: {e}")
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
    """Al sinyali bildirimi — Midas uyumlu format, onay/red butonlu.

    Y1: Duplicate check + signal_id reserve + insert atomik blok icinde.
    Eski kodda iki ayri 'with _pending_lock' arasinda race vardi:
    iki scanner thread ayni sembol icin paralel sinyal ekleyebilirdi.
    """
    # Tam lot (Midas için integer)
    lot        = max(1, int(quantity))

    # ─── Y1: ATOMIK CHECK + RESERVE ─────────────────────────────────
    # Tek lock altinda: (1) duplicate kontrol, (2) UUID uretip _pending_signals'a
    # placeholder INSERT. Sonraki dis IO (HTTP send) lock disinda — sinyal
    # mesaj ID'si zaten reserved.
    expires_at = datetime.now() + timedelta(minutes=15)
    with _pending_lock:
        for sig in _pending_signals.values():
            if sig['symbol'] == symbol and sig['uid'] == uid:
                return False
        # Collision-safe uuid (cok dusuk ihtimal ama defensive)
        signal_id = str(uuid.uuid4())[:8]
        while signal_id in _pending_signals:
            signal_id = str(uuid.uuid4())[:8]
        _pending_signals[signal_id] = {
            'uid': uid, 'symbol': symbol, 'price': price,
            'quantity': lot, 'score': score, 'confidence': confidence,
            'sl': sl, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
            'trailing_sl': trailing_sl,
            'expires_at': expires_at,
        }
    # ────────────────────────────────────────────────────────────────

    # Persist (restart-resilient onay sinyali)
    try:
        from database import _db_save_pending_signal
        with _pending_lock:
            _signal_snapshot = _pending_signals.get(signal_id)
        if _signal_snapshot:
            _db_save_pending_signal(signal_id, _signal_snapshot)
    except Exception:
        pass

    # Hesaplamalar (lock disinda — local degiskenler)
    risk_pct   = round((price - sl) / price * 100, 1) if price > sl else 0
    tp1_pct    = round((tp1 - price) / price * 100, 1) if tp1 > price else 0
    tp2_pct    = round((tp2 - price) / price * 100, 1) if tp2 > price else 0
    tp3_pct    = round((tp3 - price) / price * 100, 1) if tp3 > price else 0
    rr         = round(tp1_pct / risk_pct, 1) if risk_pct > 0 else 0

    toplam_tl  = round(lot * price, 2)
    risk_tl    = round(lot * (price - sl), 2) if price > sl else 0
    kar_tl1    = round(lot * (tp1 - price), 2) if tp1 > price else 0

    # Limit emir önerisi: fiyatın %0.3 altı (daha iyi dolum)
    limit_oneri = round(price * 0.997, 2)

    # Iptal koşulları:
    #   chase_above: sinyalden %1.5 yukari kacarsa, alis cazibesini yitirdi (R/R bozulur)
    #   abandon_below: SL altina inmise zaten zarar bolgesinde, hic alma
    chase_above = round(price * 1.015, 2)
    abandon_below = round(sl, 2)
    sig_time = datetime.now(_TZ_TR)
    sig_time_str = sig_time.strftime('%H:%M')
    expire_time_str = (sig_time + timedelta(minutes=15)).strftime('%H:%M')

    msg = (
        f"🟢 <b>AL SİNYALİ — {symbol}</b>  <i>({sig_time_str})</i>\n"
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
        f"⛔ <b>ALMA:</b> fiyat <b>{chase_above:.2f}</b> üstüne çıkarsa "
        f"veya <b>{abandon_below:.2f}</b> altına inerse\n"
        f"⏰ Geçerlilik: <b>{expire_time_str}</b>'e kadar (15 dk)\n"
        f"✅ Onaylarsan sisteme kaydedilir"
    )

    keyboard = [
        [
            {'text': '✅ Aldım / Takibe Al', 'callback_data': f'approve_{signal_id}'},
            {'text': '❌ Geç', 'callback_data': f'reject_{signal_id}'}
        ],
        [
            {'text': '⏸ Ertele 30dk', 'callback_data': f'snooze_{signal_id}'},
            {'text': '📊 Detay', 'callback_data': f'detail_{signal_id}'}
        ]
    ]

    ok = send_telegram_with_keyboard(msg, keyboard)
    # Y1 (cleanup): HTTP send basarisizsa reserved signal_id'yi temizle —
    # cleanup loop'unu 15 dk beklemeden serbest birakir.
    if not ok:
        with _pending_lock:
            _pending_signals.pop(signal_id, None)
        try:
            from database import _db_delete_pending_signal
            _db_delete_pending_signal(signal_id)
        except Exception:
            pass
        print(f"[TELEGRAM] send_trade_signal HTTP basarisiz, signal_id={signal_id} serbest")
    return ok


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
        # B#3 dedup: aynı pozisyon için eski bekleyen trailing önerilerini KALDIR — sadece
        # EN GÜNCEL kalsın. Aksi halde uptrend'de eskiler birikir, kullanıcı eski (daha düşük)
        # bir onaya basınca stop geriye çekilebilirdi. (Ayrıca _auto_update_trailing only-up guard'lı.)
        for _old in [tid for tid, t in _pending_trailing.items()
                     if t.get('position_id') == position_id]:
            _pending_trailing.pop(_old, None)
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


def send_sl_change_request(position_id, symbol, field, old_val, new_val, reason,
                           cur_price=None, expires_min=60):
    """SL/TP seviye degisikligi onerildiginde Telegram onay mesaji gonder.
    Onay verilirse DB guncellenir; reddedilir veya suresi dolarsa eski seviye korunur.

    field: 'stop_loss' | 'take_profit1' | 'take_profit2' | 'take_profit3'
    reason: kisa aciklama (orn. 'TP1 hit -> SL break-even', 'manuel ayar')
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False

    sl_id = str(uuid.uuid4())[:8]
    with _pending_sl_change_lock:
        _pending_sl_change[sl_id] = {
            'position_id': position_id,
            'symbol': symbol,
            'field': field,
            'old_val': float(old_val or 0),
            'new_val': float(new_val or 0),
            'reason': reason or '',
            'expires_at': datetime.now() + timedelta(minutes=expires_min),
        }

    field_label = {
        'stop_loss':    '🛡 Stop-Loss',
        'take_profit1': '🎯 TP1',
        'take_profit2': '🎯 TP2',
        'take_profit3': '🎯 TP3',
    }.get(field, field)

    cur_line = f"\n💰 Guncel: {cur_price:.2f} TL" if cur_price else ""
    move_line = ""
    try:
        if old_val and old_val > 0:
            diff_pct = (float(new_val) - float(old_val)) / float(old_val) * 100
            arrow = "↑" if diff_pct > 0 else "↓"
            move_line = f"  ({arrow}%{abs(diff_pct):.2f})"
    except Exception:
        pass

    msg = (
        f"⚙️ <b>SEVIYE DEGISIM ONERISI: {symbol}</b>\n"
        f"{field_label}: {float(old_val or 0):.2f} → <b>{float(new_val or 0):.2f}</b>{move_line}"
        f"{cur_line}\n"
        f"📌 Sebep: {reason}\n"
        f"⏱ Onay yoksa {expires_min} dk sonra iptal — eski seviye kalir."
    )
    keyboard = [[
        {'text': '✅ Onayla', 'callback_data': f'slchg_approve_{sl_id}'},
        {'text': '❌ Geç',    'callback_data': f'slchg_reject_{sl_id}'}
    ]]
    return send_telegram_with_keyboard(msg, keyboard)


def send_tp_exec_request(position_id, uid, symbol, kind, tp_field, sell_qty,
                         price, tp_target, reason, expires_min=180):
    """TP hedefine ulasinca OTOMATIK icra yerine Telegram onayi iste.
    Onaylaninca execute_tp_exec() kismi/tam satisi yapar; reddedilir/suresi dolarsa
    pozisyon ve TP dokunulmaz. Ayni pozisyon+tp_field icin tekrar sormaz (dedup).
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    ask_key = f"{position_id}_{tp_field}"
    tp_id = str(uuid.uuid4())[:8]
    with _pending_tp_exec_lock:
        if ask_key in _tp_exec_asked:
            return True  # zaten soruldu — tekrar gonderme (spam onleme)
        _tp_exec_asked.add(ask_key)
        _pending_tp_exec[tp_id] = {
            'position_id': position_id, 'uid': uid, 'symbol': symbol,
            'kind': kind, 'tp_field': tp_field, 'sell_qty': float(sell_qty or 0),
            'price': float(price or 0), 'tp_target': float(tp_target or 0),
            'reason': reason or '',
            'expires_at': datetime.now() + timedelta(minutes=expires_min),
        }

    _is_stop = tp_field == 'trailing'
    label = {'take_profit1': '🎯 TP1', 'take_profit2': '🎯 TP2', 'take_profit3': '🎯 TP3',
             'trailing': '🛑 Trailing-Stop'}.get(tp_field, tp_field)
    short = label.split()[-1]
    if price and tp_target:
        _op = '≤' if _is_stop else '≥'
        _lbl2 = 'stop' if _is_stop else short
        price_line = f"\n💰 Fiyat: {float(price):.2f} {_op} {_lbl2}: {float(tp_target):.2f}"
    else:
        price_line = ""
    if kind == 'full':
        action_line = f"Öneri: <b>tamamını sat</b> ({float(sell_qty or 0):.0f} lot) + pozisyonu kapat"
    else:
        action_line = f"Öneri: <b>%50 sat</b> ({float(sell_qty or 0):.0f} lot) + {short} kapat"
    _suffix = '' if _is_stop else ' HEDEFİ'
    msg = (
        f"<b>{label}{_suffix} — {symbol}</b>{price_line}\n"
        f"{action_line}\n"
        f"⚠️ <b>Onayın olmadan işlem YAPILMAZ.</b>\n"
        f"⏱ {expires_min} dk içinde yanıt yoksa iptal — pozisyon aynen kalır."
    )
    keyboard = [[
        {'text': '✅ Onayla & Uygula', 'callback_data': f'tpexec_approve_{tp_id}'},
        {'text': '❌ Geç',             'callback_data': f'tpexec_reject_{tp_id}'}
    ]]
    ok = send_telegram_with_keyboard(msg, keyboard)
    if not ok:
        # Gonderim basarisiz — kaydi geri al ki sonraki cycle tekrar denesin
        with _pending_tp_exec_lock:
            _pending_tp_exec.pop(tp_id, None)
            _tp_exec_asked.discard(ask_key)
    return True
