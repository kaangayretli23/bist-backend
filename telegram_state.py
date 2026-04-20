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

# Bekleyen trailing güncellemeleri:
#   {trail_id: {position_id, symbol, new_trailing, new_highest, expires_at}}
_pending_trailing = {}
_pending_trailing_lock = threading.Lock()

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
