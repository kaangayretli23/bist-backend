"""
BIST Pro - Telegram Callback Handlers + Arka Plan Thread'leri
Buton basımlarını işler, polling ve cleanup loop'larını çalıştırır.
routes_telegram.py'dan ayrıştırıldı (600 satır kuralı).
"""
import time
import threading
from datetime import datetime

import telegram_state as _state
from telegram_state import (
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    _pending_signals, _pending_lock,
    _pending_trailing, _pending_trailing_lock,
    _telegram_thread_lock,
)
from telegram_notifications import send_telegram, edit_telegram_message


# =====================================================================
# CALLBACK HANDLER (TELEGRAM'DAN GELEN BUTON BASILMALARI)
# =====================================================================

def _answer_callback(callback_id):
    """Callback query'yi yanıtla (yükleniyor ikonunu kaldır)"""
    try:
        import requests as req
        req.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/answerCallbackQuery",
            json={'callback_query_id': callback_id},
            timeout=5
        )
    except Exception:
        pass


def _handle_approve(signal_id, message_id):
    """Kullanici ONAYLA'ya basti"""
    with _pending_lock:
        signal = _pending_signals.pop(signal_id, None)

    if not signal:
        edit_telegram_message(message_id, "⚠️ Sinyal bulunamadı veya süresi dolmuş.")
        return

    if datetime.now() > signal['expires_at']:
        edit_telegram_message(message_id, f"⏰ <b>{signal['symbol']}</b> sinyalinin süresi dolmuş.")
        return

    try:
        from auto_trader import _auto_open_position, _auto_log_trade
        from config import _cget, _stock_cache

        # Onay anındaki güncel fiyatı al — sinyal fiyatından sapma kontrolü
        stock = _cget(_stock_cache, signal['symbol'])
        exec_price = float(stock.get('price', 0)) if stock else 0
        if exec_price <= 0:
            exec_price = signal['price']  # Cache boşsa sinyal fiyatını kullan

        # Fiyat %3'ten fazla değiştiyse uyar ama yine de aç
        price_drift = abs(exec_price - signal['price']) / signal['price'] * 100
        drift_warn = f"\n⚠️ Fiyat %{price_drift:.1f} değişti ({signal['price']:.2f}→{exec_price:.2f})" if price_drift > 1 else ""

        # SL/TP'yi exec_price'a göre yeniden hesapla (oranları koru)
        sig_price = signal['price']
        if sig_price > 0 and abs(exec_price - sig_price) / sig_price > 0.001:
            ratio = exec_price / sig_price
            sl  = round(signal['sl'] * ratio, 2)
            tp1 = round(signal['tp1'] * ratio, 2)
            tp2 = round(signal['tp2'] * ratio, 2) if signal.get('tp2') else signal.get('tp2')
            tp3 = round(signal['tp3'] * ratio, 2) if signal.get('tp3') else signal.get('tp3')
            trailing = round(signal['trailing_sl'] * ratio, 2) if signal.get('trailing_sl') else signal.get('trailing_sl')
        else:
            sl, tp1, tp2, tp3, trailing = signal['sl'], signal['tp1'], signal.get('tp2'), signal.get('tp3'), signal.get('trailing_sl')

        pos_id = _auto_open_position(
            signal['uid'], signal['symbol'], exec_price,
            signal['quantity'], sl, tp1, tp2, tp3, trailing
        )
        if pos_id:
            _auto_log_trade(
                signal['uid'], signal['symbol'], 'BUY',
                exec_price, signal['quantity'],
                f"Telegram onaylı | Skor={signal['score']:.1f}, Güven=%{signal['confidence']:.0f}",
                signal['score'], signal['confidence'], pos_id
            )
            edit_telegram_message(
                message_id,
                f"✅ <b>{signal['symbol']} ONAYLANDI</b>\n"
                f"💰 {exec_price:.2f} TL × {signal['quantity']:.2f} adet\n"
                f"🛡 SL: {sl:.2f}  |  🎯 TP1: {tp1:.2f}\n"
                f"📂 Portföye eklendi.{drift_warn}"
            )
        else:
            edit_telegram_message(message_id, f"❌ {signal['symbol']} pozisyon açılamadı.")
    except Exception as e:
        edit_telegram_message(message_id, f"❌ Hata oluştu: {e}")
        print(f"[TELEGRAM] Approve hatasi: {e}")


def _handle_reject(signal_id, message_id):
    """Kullanici REDDET'e basti"""
    with _pending_lock:
        signal = _pending_signals.pop(signal_id, None)

    if signal:
        # Hard reject: 12 saat boyunca bu hisse bir daha onerilmesin
        try:
            from auto_trader_risk import _reject_cooldown_block
            _reject_cooldown_block(signal.get('uid', ''), signal['symbol'], reason='hard')
        except Exception as _rc_err:
            print(f"[TELEGRAM] Reject cooldown yazma hatasi: {_rc_err}")
        edit_telegram_message(
            message_id,
            f"❌ <b>{signal['symbol']}</b> sinyali reddedildi.\n"
            f"<i>Bu hisse 12 saat tekrar önerilmeyecek.</i>"
        )


def _handle_trailing_approve(trail_id, message_id):
    """Trailing stop onaylandı — DB'yi güncelle"""
    with _pending_trailing_lock:
        trail = _pending_trailing.pop(trail_id, None)

    if not trail:
        edit_telegram_message(message_id, "⚠️ Trailing güncelleme bulunamadı veya süresi dolmuş.")
        return

    if datetime.now() > trail['expires_at']:
        edit_telegram_message(message_id, f"⏰ <b>{trail['symbol']}</b> trailing güncellemesinin süresi dolmuş.")
        return

    try:
        from auto_trader import _auto_update_trailing
        _auto_update_trailing(trail['position_id'], trail['new_trailing'], trail['new_highest'])
        edit_telegram_message(
            message_id,
            f"✅ <b>{trail['symbol']} TRAILING GÜNCELLENDİ</b>\n"
            f"🛡 Yeni Stop: {trail['new_trailing']:.2f} TL\n"
            f"🔝 Zirve: {trail['new_highest']:.2f} TL\n"
            f"⚡ Midas'ta stop emrini bu seviyeye çek."
        )
    except Exception as e:
        edit_telegram_message(message_id, f"❌ Trailing güncelleme hatası: {e}")
        print(f"[TELEGRAM] Trailing approve hatasi: {e}")


def _handle_trailing_reject(trail_id, message_id):
    """Trailing stop reddedildi — eski değer korunur"""
    with _pending_trailing_lock:
        trail = _pending_trailing.pop(trail_id, None)

    if trail:
        edit_telegram_message(
            message_id,
            f"❌ <b>{trail['symbol']}</b> trailing güncellemesi reddedildi.\n"
            f"🛡 Mevcut stop seviyesi korunuyor."
        )


def _process_update(update):
    """Tek bir Telegram update'i islemi"""
    if 'callback_query' not in update:
        return

    cq = update['callback_query']
    callback_id = cq.get('id')
    data = cq.get('data', '')
    message_id = cq.get('message', {}).get('message_id')

    _answer_callback(callback_id)

    if data.startswith('trail_approve_'):
        _handle_trailing_approve(data.replace('trail_approve_', ''), message_id)
    elif data.startswith('trail_reject_'):
        _handle_trailing_reject(data.replace('trail_reject_', ''), message_id)
    elif data.startswith('approve_'):
        _handle_approve(data.replace('approve_', ''), message_id)
    elif data.startswith('reject_'):
        _handle_reject(data.replace('reject_', ''), message_id)


# =====================================================================
# ARKA PLAN THREADLERI
# =====================================================================

def _telegram_polling():
    """Telegram long-polling — buton basimlarini dinle.
    _last_update_id telegram_state modülünde tutuluyor; attribute erişimiyle güncelliyoruz."""
    import requests as req

    print("[TELEGRAM] Polling thread basladi")
    while True:
        try:
            if not TELEGRAM_BOT_TOKEN:
                time.sleep(60)
                continue

            resp = req.get(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates",
                params={
                    'offset': _state._last_update_id + 1,
                    'timeout': 30,
                    'allowed_updates': ['callback_query']
                },
                timeout=35
            )

            if resp.status_code != 200:
                time.sleep(5)
                continue

            updates = resp.json().get('result', [])
            for update in updates:
                _state._last_update_id = update['update_id']
                try:
                    _process_update(update)
                except Exception as e:
                    print(f"[TELEGRAM] Update isleme hatasi: {e}")

        except Exception as e:
            print(f"[TELEGRAM] Polling hatasi: {e}")
            time.sleep(5)


def _cleanup_expired_signals():
    """Suresi dolan bekleyen sinyalleri ve trailing güncellemeleri temizle"""
    while True:
        time.sleep(60)
        try:
            now = datetime.now()
            expired_list = []
            with _pending_lock:
                expired = [sid for sid, s in _pending_signals.items() if now > s['expires_at']]
                for sid in expired:
                    expired_list.append(_pending_signals[sid].copy())
                    del _pending_signals[sid]
            for sig in expired_list:
                print(f"[TELEGRAM] Suresi dolan sinyal temizlendi: {sig['symbol']} ({sig.get('uid', '')})")
                # Soft reject: 2 saat sessiz kalsin (yanit yok, ama belki gun icinde tekrar degerlendirilir)
                try:
                    from auto_trader_risk import _reject_cooldown_block
                    _reject_cooldown_block(sig.get('uid', ''), sig['symbol'], reason='soft')
                except Exception as _rc_err:
                    print(f"[TELEGRAM] Expired reject cooldown hatasi: {_rc_err}")
                send_telegram(f"⏰ <b>{sig['symbol']}</b> sinyali yanıtlanmadı, iptal edildi.\n<i>Bu hisse 2 saat tekrar önerilmeyecek.</i>")

            # Trailing güncellemelerini temizle (süresi dolanlar otomatik onaylanır)
            expired_trails = []
            with _pending_trailing_lock:
                exp_ids = [tid for tid, t in _pending_trailing.items() if now > t['expires_at']]
                for tid in exp_ids:
                    expired_trails.append(_pending_trailing.pop(tid))
            for trail in expired_trails:
                try:
                    from auto_trader import _auto_update_trailing
                    _auto_update_trailing(trail['position_id'], trail['new_trailing'], trail['new_highest'])
                    print(f"[TELEGRAM] Trailing süresi doldu, otomatik onaylandı: {trail['symbol']} → {trail['new_trailing']:.2f}")
                except Exception:
                    pass
        except Exception:
            pass


def _start_telegram_thread():
    """Telegram bildirim + polling + cleanup + performans raporu thread'lerini başlat.
    Idempotent: birden fazla çağrıldığında sadece bir kez başlar."""
    with _telegram_thread_lock:
        if _state._telegram_thread_started:
            return
        _state._telegram_thread_started = True

    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    from telegram_reports import _auto_signal_check, _performance_reporter
    threading.Thread(target=_auto_signal_check, daemon=True).start()
    threading.Thread(target=_telegram_polling, daemon=True).start()
    threading.Thread(target=_cleanup_expired_signals, daemon=True).start()
    threading.Thread(target=_performance_reporter, daemon=True).start()
    print("[TELEGRAM] Sinyal bildirimi + polling + cleanup + performans raporu aktif")
