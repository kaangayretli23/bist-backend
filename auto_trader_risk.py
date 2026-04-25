"""
BIST Pro - Auto Trading Risk Helpers
SL cooldown (DB persisted) + panic-sell ring buffer (deque) + karar log helper.
auto_trader_engine.py'dan ayrıştırıldı (700 satır kuralı).
"""
import time
from collections import deque
from config import get_db


def _log_decision(uid: str, sym: str, decision: str, reason: str = '',
                  detail: str = '', tf: str = '', price: float = 0.0,
                  score: float = 0.0, confidence: float = 0.0) -> None:
    """auto_decisions tablosuna karar yaz. Sessiz hata — log yazimi engel olmasin.
    decision: 'SKIP', 'BUY', 'PENDING' (Telegram onayi bekleniyor)
    reason: 'reject_cooldown', 'sl_cooldown', 'regime', 'volume', 'rsi', 'budget',
            'score', 'gap_down', 'opened', 'pending', 'plan_stale', 'deviation', vb.
    """
    if not uid or not sym:
        return
    try:
        db = get_db()
        try:
            db.execute(
                "INSERT INTO auto_decisions "
                "(user_id, symbol, timeframe, decision, reason, detail, price, score, confidence) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (uid, sym.upper(), tf or '', decision, reason or '',
                 (detail or '')[:500], float(price or 0), float(score or 0), float(confidence or 0))
            )
            db.commit()
        finally:
            db.close()
    except Exception as _le:
        print(f"[DECISION-LOG] Yazilamadi {sym}/{reason}: {_le}")

# SL sonrası yeniden giriş engeli: {uid_sym: (timestamp, trade_style)}
_sl_cooldown: dict = {}

# Reject sonrasi yeniden oneri engeli: {uid_sym: (expires_at_ts, reason)}
# reason='hard' -> kullanici acikca reddetti (12 saat)
# reason='soft' -> 15 dk icinde yanit vermedi (2 saat)
_reject_cooldown: dict = {}

_REJECT_COOLDOWN_SECS = {
    'hard': 12 * 3600,  # yarim gun
    'soft':  2 * 3600,  # 2 saat
}


def _reject_cooldown_block(uid: str, sym: str, reason: str = 'hard') -> None:
    """Telegram reddi veya expire sonrasi bu sembolu gecici olarak bloklar.
    reason='hard' kullanici reddetti, reason='soft' yanit gelmedi.
    """
    if not uid or not sym:
        return
    import time as _t
    secs = _REJECT_COOLDOWN_SECS.get(reason, _REJECT_COOLDOWN_SECS['hard'])
    _reject_cooldown[f"{uid}_{sym}"] = (_t.time() + secs, reason)
    hours = secs / 3600
    print(f"[REJECT-COOLDOWN] {sym} -> {hours:.1f} saat boyunca onerilmeyecek ({reason})")


def _reject_cooldown_check(uid: str, sym: str) -> bool:
    """True donerse bu hisse reject cooldown'da, tekrar oneri yok."""
    if not uid or not sym:
        return False
    import time as _t
    key = f"{uid}_{sym}"
    entry = _reject_cooldown.get(key)
    if not entry:
        return False
    expires_at, _reason = entry
    if _t.time() >= expires_at:
        _reject_cooldown.pop(key, None)
        return False
    return True

# Panic-sell için pozisyon başına fiyat geçmişi (O(1) popleft deque)
# {position_id: deque([(ts, price), ...])}
_panic_price_history: dict = {}
_PANIC_MAX_WINDOW_SECS = 15 * 60  # Ring-buffer'ı 15 dk ile sınırla (config en fazla 15 olsa da)
_PANIC_MAX_SAMPLES = 120          # 30sn aralıkla 60dk tampon (aşırı büyümeyi engeller)


def _panic_track_and_check(pos_id: int, cur_price: float, entry_price: float,
                            drop_pct: float, window_min: int) -> tuple[bool, float, float]:
    """Fiyat geçmişini güncelle ve panic tetiklenmiş mi kontrol et.
    Returns: (tetiklendi, pencere_zirvesi, düşüş_yüzdesi)
    """
    now = time.time()
    window_secs = max(60, window_min * 60)
    hist = _panic_price_history.get(pos_id)
    if hist is None:
        hist = deque(maxlen=_PANIC_MAX_SAMPLES)
        _panic_price_history[pos_id] = hist
    hist.append((now, cur_price))
    # Pencere dışı örnekleri at (O(1) popleft)
    cutoff = now - min(window_secs, _PANIC_MAX_WINDOW_SECS)
    while hist and hist[0][0] < cutoff:
        hist.popleft()

    if len(hist) < 3:  # Anlamlı bir pencere oluşmadı
        return False, 0.0, 0.0

    peak = max(p for _, p in hist)
    if peak <= 0:
        return False, 0.0, 0.0

    drop = (peak - cur_price) / peak * 100
    # Koşul: pencere içinde %drop_pct düşüş (kâr/zarar fark etmez)
    if drop >= drop_pct:
        return True, peak, drop
    return False, peak, drop


def _panic_clear(pos_id: int) -> None:
    _panic_price_history.pop(pos_id, None)


# Trade style'a göre SL cooldown süreleri (saniye)
_SL_COOLDOWN_SECS = {
    'daily':   4 * 3600,   # 4 saat — günlük stil
    'swing':  24 * 3600,   # 24 saat — swing
    'monthly': 72 * 3600,  # 3 gün — pozisyon
}


def _sl_cooldown_block(uid: str, sym: str, trade_style: str) -> None:
    """SL tetiklenince bu hisse için cooldown başlat. DB'ye de yaz (restart dayanıklı)."""
    key = f"{uid}_{sym}"
    now = time.time()
    _sl_cooldown[key] = (now, trade_style)
    try:
        db = get_db()
        try:
            db.execute(
                "INSERT INTO sl_cooldown (uid_sym, hit_at, trade_style) VALUES (?, ?, ?) "
                "ON CONFLICT (uid_sym) DO UPDATE SET hit_at=EXCLUDED.hit_at, trade_style=EXCLUDED.trade_style",
                [key, now, trade_style]
            )
            db.commit()
        finally:
            db.close()
    except Exception as e:
        print(f"[SL-COOLDOWN] DB yazma hatası: {e}")


def _sl_cooldown_check(uid: str, sym: str, trade_style: str) -> bool:
    """True döndürürse bu hisse cooldown'da, yeni pozisyon açma."""
    key = f"{uid}_{sym}"
    entry = _sl_cooldown.get(key)
    if not entry:
        return False
    hit_at, style = entry
    cooldown = _SL_COOLDOWN_SECS.get(style, _SL_COOLDOWN_SECS.get(trade_style, 24 * 3600))
    if time.time() - hit_at < cooldown:
        remaining_h = (cooldown - (time.time() - hit_at)) / 3600
        print(f"[AUTO-TRADE] {sym} SL cooldown aktif — {remaining_h:.1f} saat kaldı")
        return True
    del _sl_cooldown[key]
    # DB'den de sil
    try:
        db = get_db()
        try:
            db.execute("DELETE FROM sl_cooldown WHERE uid_sym=?", [key])
            db.commit()
        finally:
            db.close()
    except Exception:
        pass
    return False


def _sl_cooldown_load_from_db() -> None:
    """Startup'ta DB'deki cooldown kayıtlarını belleğe yükle."""
    try:
        db = get_db()
        try:
            rows = db.execute("SELECT uid_sym, hit_at, trade_style FROM sl_cooldown").fetchall()
        finally:
            db.close()
        now = time.time()
        loaded, expired_keys = 0, []
        for row in rows:
            try:
                key = row['uid_sym'] if hasattr(row, 'keys') else row[0]
                hit_at = float(row['hit_at'] if hasattr(row, 'keys') else row[1])
                style = row['trade_style'] if hasattr(row, 'keys') else row[2]
            except Exception:
                continue
            cooldown = _SL_COOLDOWN_SECS.get(style, 24 * 3600)
            if now - hit_at < cooldown:
                _sl_cooldown[key] = (hit_at, style)
                loaded += 1
            else:
                expired_keys.append(key)
        if expired_keys:
            try:
                db = get_db()
                try:
                    for k in expired_keys:
                        db.execute("DELETE FROM sl_cooldown WHERE uid_sym=?", [k])
                    db.commit()
                finally:
                    db.close()
            except Exception:
                pass
        if loaded:
            print(f"[SL-COOLDOWN] {loaded} kayıt DB'den yüklendi, {len(expired_keys)} süresi dolmuş kayıt temizlendi")
    except Exception as e:
        print(f"[SL-COOLDOWN] DB yükleme hatası: {e}")
