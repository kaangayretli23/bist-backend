"""
BIST Pro - Auto Trading Risk Helpers
SL cooldown (DB persisted) + panic-sell ring buffer (deque) + karar log helper.
auto_trader_engine.py'dan ayrıştırıldı (700 satır kuralı).
"""
import os
import time
from collections import deque
from config import get_db

# Gecikmeli feed (yf fallback) icin varsayilan piyasa gecikmesi (sn).
# WS (TradingView) ~canli kabul (lag 0); yf ~15dk gecikmeli. 0 = kaynak-ayrimi kapali.
_DELAYED_FEED_LAG_SEC = int(os.environ.get('AUTO_DELAYED_FEED_LAG_SEC', '900'))


def _source_lag_sec(source) -> int:
    """Kaynak tipine gore ek piyasa gecikmesi. ws=canli→0, yf=gecikmeli→~900, bilinmiyor→0."""
    return _DELAYED_FEED_LAG_SEC if (source or '').lower() == 'yf' else 0


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

def _data_freshness(sym: str):
    """#1 Data freshness (kaynak-duyarlı): EN TAZE kaynağın ETKİN yaşı.

    Döner: (effective_age_sec, raw_age_sec, source) — kaynak yoksa (None, None, None).

    ETKİN yaş = yazım yaşı + kaynak gecikmesi. Böylece yf (15dk gecikmeli) fiyat 'şimdi'
    yazılsa bile canlı sayılmaz — 'veri yeni mi?' yerine 'piyasa verisi ne kadar geride?'
    ölçülür. İki kaynaktan ETKİN yaşı en küçük olan seçilir:
      • batch _stock_cache[sym] — loader (borsapy WS / yf) son yazımı
      • realtime get_quote(sym) — son tick (source='ws'|'yf')
    """
    now = time.time()
    cands = []  # (effective_age, raw_age, source)
    try:
        from config import _stock_cache, _lock
        with _lock:
            it = _stock_cache.get(sym)
        if it and it.get('ts'):
            raw = now - float(it['ts'])
            data = it.get('data')
            src = data.get('source') if isinstance(data, dict) else None
            cands.append((raw + _source_lag_sec(src), raw, src or 'batch'))
    except Exception:
        pass
    try:
        from realtime_prices import get_quote
        q = get_quote(sym)
        if q and q.get('ts'):
            raw = now - float(q['ts'])
            src = q.get('source')
            cands.append((raw + _source_lag_sec(src), raw, src or 'rt'))
    except Exception:
        pass
    if not cands:
        return (None, None, None)
    return min(cands, key=lambda c: c[0])


def _data_age_sec(sym: str):
    """Geriye-uyum sarmalayıcısı: EN TAZE kaynağın ETKİN yaşı (sn) veya None."""
    return _data_freshness(sym)[0]


# SL sonrası yeniden giriş engeli: {uid_sym: (timestamp, trade_style)}
_sl_cooldown: dict = {}

# Reject sonrasi yeniden oneri engeli: {uid_sym: (expires_at_ts, reason)}
# reason='hard' -> kullanici acikca reddetti (12 saat)
# reason='soft' -> 15 dk icinde yanit vermedi (45 dk)
_reject_cooldown: dict = {}

_REJECT_COOLDOWN_SECS = {
    'hard': 12 * 3600,  # yarim gun
    'soft':  45 * 60,   # 45 dk
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
_PROCESS_START = time.time()      # restart-ısınma tespiti için (Kemal #2b-a)
_panic_warmup_logged: set = set() # pozisyon başına bir kez log (spam önle)


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
        # Kemal #2b(a): panic ısınma süresini ÖLÇ. Restart sonrası ya da yeni pozisyonda
        # pencere dolana kadar panic kör kalır. process uptime kısaysa muhtemelen restart-ısınma.
        if pos_id not in _panic_warmup_logged:
            _panic_warmup_logged.add(pos_id)
            _uptime = time.time() - _PROCESS_START
            _tag = '(muhtemelen restart-isinma)' if _uptime < 180 else '(yeni pozisyon)'
            print(f"[PANIC-WARMUP] pos #{pos_id}: panic penceresi dolmadi ({len(hist)} ornek), "
                  f"uptime={_uptime:.0f}s {_tag} — flash-crash korumasi henuz aktif degil")
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


# Kemal #4a: Ardışık-zarar kill-switch — bugün üst üste M zarar olursa o gün scanner durur.
_MAX_CONSECUTIVE_LOSSES = 3


def _consecutive_loss_freeze(uid: str, max_losses: int = _MAX_CONSECUTIVE_LOSSES) -> tuple[bool, int]:
    """Bugün kapanan pozisyonlarda, en son işlemden geriye doğru ARDIŞIK zarar say.
    streak >= max_losses ise freeze=True. Sadece BUGÜNÜN kapanışlarına bakar → yeni günde
    otomatik sıfırlanır. DD freeze'in tamamlayıcısı (toplam değil, seri bazlı kötü-gün durdurucu).
    Returns: (freeze_mi, streak)
    """
    if not uid or max_losses <= 0:
        return False, 0
    try:
        from datetime import datetime as _dt, timezone as _tz, timedelta as _td
        _today = (_dt.now(_tz.utc) + _td(hours=3)).strftime('%Y-%m-%d')
        db = get_db()
        try:
            rows = db.execute(
                "SELECT pnl, closed_at FROM auto_positions "
                "WHERE user_id=? AND status='closed' ORDER BY closed_at DESC LIMIT ?",
                (uid, max(max_losses * 4, 12)),
            ).fetchall()
        finally:
            db.close()
        streak = 0
        for r in rows:
            if str(r['closed_at'] or '')[:10] != _today:
                break  # bugünden eski kapanış → bugünün serisi bitti
            if float(r['pnl'] or 0) < 0:
                streak += 1
            else:
                break  # kazanç/başabaş → seri kırıldı
        return (streak >= max_losses), streak
    except Exception:
        return False, 0


# Kemal #4c: Açık pozisyon korelasyon ölçümü (limit YOK — sadece gözlem; throttle 30dk/uid).
_corr_log_throttle: dict = {}


def _log_portfolio_correlation(uid: str, open_positions: list, min_interval_sec: int = 1800) -> None:
    """Açık pozisyonların ortalama ikili korelasyonunu ölç + logla. BIST'te asıl risk beta/endeks
    korelasyonu; sektör cap'i bunu yakalamaz. Aksiyon YOK, sadece görünürlük (Kemal #4c).
    Tarih-hizalı (DataFrame index ile); throttle ile log spam'i önlenir."""
    try:
        if not open_positions or len(open_positions) < 2:
            return
        _now = time.time()
        if _now - _corr_log_throttle.get(uid, 0) < min_interval_sec:
            return
        from config import _cget_hist
        import pandas as _pd
        import numpy as _np
        ret = {}
        for p in open_positions:
            h = _cget_hist(f"{p['symbol']}_1y")
            if h is not None and len(h) >= 21:
                ret[p['symbol']] = h['Close'].astype(float).pct_change().iloc[-20:]
        if len(ret) < 2:
            return
        df = _pd.DataFrame(ret).dropna()
        if len(df) < 10:
            return
        cm = df.corr().values
        vals = cm[_np.triu_indices_from(cm, k=1)]
        vals = vals[~_np.isnan(vals)]
        if len(vals) == 0:
            return
        _corr_log_throttle[uid] = _now
        _log_decision(uid, 'PORTFOLIO', 'INFO', 'correlation',
                      detail=f"acik {len(ret)} poz ort.korelasyon={float(_np.mean(vals)):.2f}, "
                             f"max={float(_np.max(vals)):.2f} (>0.7 yuksek konsantrasyon)")
    except Exception:
        pass


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
