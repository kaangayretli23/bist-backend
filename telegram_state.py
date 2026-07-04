"""
BIST Pro - Telegram Shared State
Tüm Telegram alt modüllerinin paylaştığı mutable global state ve sabitler.

Neden ayrı modül: send fn, callback handler ve routes'un ortak
locks/dicts'e erişmesi gerekir. Tek modülde tutmak bölünmüş modüllerde
circular/partial-load sorununu önler.
"""
import os
import threading

# =====================================================================
# ENV CONFIG — Telegram bot credentials
# =====================================================================

TELEGRAM_BOT_TOKEN      = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID        = os.environ.get('TELEGRAM_CHAT_ID', '')
# Haber/rapor botu — ayrı bot token varsa onu kullan, yoksa ana bota fallback
TELEGRAM_NEWS_BOT_TOKEN = os.environ.get('TELEGRAM_NEWS_BOT_TOKEN', '') or TELEGRAM_BOT_TOKEN
TELEGRAM_NEWS_CHAT_ID   = os.environ.get('TELEGRAM_NEWS_CHAT_ID', '') or TELEGRAM_CHAT_ID

# =====================================================================
# THREAD LIFECYCLE
# =====================================================================

_telegram_thread_started = False
_telegram_thread_lock = threading.Lock()

# =====================================================================
# PENDING SIGNAL/TRAILING MAPS
# =====================================================================

# Bekleyen sinyaller:
#   {signal_id: {uid, symbol, price, quantity, sl, tp1, tp2, tp3, trailing_sl, score, confidence, expires_at}}
_pending_signals = {}
_pending_lock = threading.Lock()

def load_pending_from_db():
    """Restart sonrasi DB'den onay bekleyen sinyalleri yukle. init_db()'den sonra
    cagrilmali — _start_telegram_thread() icinde tetiklenir."""
    try:
        from database import _db_load_pending_signals
        loaded = _db_load_pending_signals()
        with _pending_lock:
            _pending_signals.update(loaded)
        if loaded:
            print(f"[TELEGRAM-STATE] {len(loaded)} bekleyen sinyal DB'den yuklendi")
    except Exception as e:
        print(f"[TELEGRAM-STATE] Pending yukleme hatasi: {e}")


def has_pending_signal(uid: str, symbol: str) -> bool:
    """O4: Belirli kullanici+sembol icin pending sinyal var mi (thread-safe).
    Scanner aday loop'unda _pending_signals dict scan'i encapsulate eder.
    """
    if not uid or not symbol:
        return False
    with _pending_lock:
        for ps in _pending_signals.values():
            if ps.get('uid') == uid and ps.get('symbol') == symbol:
                return True
    return False

# Bekleyen trailing güncellemeleri:
#   {trail_id: {position_id, symbol, new_trailing, new_highest, expires_at}}
_pending_trailing = {}
_pending_trailing_lock = threading.Lock()

# Bekleyen SL/TP değişim önerileri (TP1 hit sonrası BE move vb. — sistem
# otomatik degistirmeden önce kullanici onayini bekler):
#   {sl_id: {position_id, symbol, field('stop_loss'|'take_profit1'|...), old_val, new_val, reason, expires_at}}
_pending_sl_change = {}
_pending_sl_change_lock = threading.Lock()

# Bekleyen TP icra onaylari — TP hedefine ulasinca OTOMATIK icra YOK; kullanici
# Telegram'dan onaylayinca kismi sat / tam kapat yapilir. Onaysiz pozisyon+TP dokunulmaz.
#   {tp_id: {position_id, uid, symbol, kind('partial'|'full'), tp_field, sell_qty,
#            price, tp_target, reason, expires_at}}
_pending_tp_exec = {}
# Ayni pozisyon+TP seviyesi icin tekrar tekrar sormayi onler: {f"{position_id}_{tp_field}"}
_tp_exec_asked: set = set()
_pending_tp_exec_lock = threading.Lock()

# =====================================================================
# WARNING COOLDOWNS (aynı hisse için tekrar spam önleme)
# =====================================================================

_warning_cooldown: dict[str, float] = {}  # "SL_SYM" veya "TP_SYM_label" → son gönderim ts
_warning_lock = threading.Lock()

SL_WARNING_COOLDOWN      = 3600   # 1 saat — SL yaklaşma uyarısı
TP_WARNING_COOLDOWN      = 3600   # 1 saat — TP yaklaşma uyarısı
TRAILING_UPDATE_COOLDOWN = 900    # 15 dk — trailing stop güncelleme bildirimi
TRAILING_MIN_MOVE_PCT    = 0.5    # en az %0.5 hareket olmadan bildirim yok

_last_trailing_notified: dict[str, float] = {}  # symbol → son bildirilen trailing fiyatı

# =====================================================================
# POLLING OFFSET
# =====================================================================

_last_update_id = 0
