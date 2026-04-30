"""
BIST Pro - Auto Trader Piyasa Rejim Filtresi
XU100 endeksinin EMA50/EMA200 + gunluk momentum'una gore 3 mod:
  - risk-on:  trend yukari (close>EMA50>EMA200) + gunluk > +%0.5 → normal islem
  - neutral:  notr trend (kosul disindaki butun durumlar) → daha siki filtre
  - risk-off: gunluk <= -%1.5 VEYA (EMA50<EMA200 ve fiyat<EMA50) → yeni pozisyon YOK
              Death cross olsa bile fiyat EMA50 ustunde ise toparlanma sayilir → neutral.

Sonuc 5 dk cache'lenir; her cycle'da pandas EMA hesabi pahali.
Lazy import: config, pandas modul-icinde fonksiyon ic.
"""
import time

_REGIME_CACHE: dict = {'mode': 'risk-on', 'ts': 0.0, 'detail': 'Henuz degerlendirilmedi'}
_TTL_SEC = 300  # 5 dk


def _classify_regime() -> tuple:
    """XU100 cache'ten oku → (mode, detail). Veri yoksa default risk-on (fail-open).
    EMA50/EMA200 tarihsel hist'ten; day_pct CANLI endeks cache'inden (gun ici taze)."""
    try:
        from config import _cget_hist, _cget, _index_cache, _stock_cache
        hist = _cget_hist("XU100_1y")
        if hist is None or len(hist) < 200:
            return 'risk-on', f'Yetersiz XU100 verisi ({len(hist) if hist is not None else 0} bar)'
        close = hist['Close']
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])

        # Canli endeks: hist'in son bar'i dunden kalmis olabilir, gun ici degisim
        # icin _index_cache'teki live XU100 quote'unu kullan.
        day_pct = None
        last = None
        try:
            xu_live = _cget(_index_cache, 'XU100') or _cget(_stock_cache, 'XU100')
            if xu_live:
                if 'changePct' in xu_live:
                    day_pct = float(xu_live.get('changePct') or 0.0)
                if 'price' in xu_live:
                    last = float(xu_live.get('price') or 0.0) or None
        except Exception:
            pass
        # Fallback: live yoksa hist'ten close-to-close
        if day_pct is None:
            last_h = float(close.iloc[-1])
            prev = float(close.iloc[-2]) if len(close) >= 2 else last_h
            day_pct = ((last_h - prev) / prev * 100.0) if prev > 0 else 0.0
            if last is None:
                last = last_h
        if last is None:
            last = float(close.iloc[-1])

        if last > ema50 > ema200 and day_pct > 0.5:
            return 'risk-on', (f"XU100={last:.0f} > EMA50={ema50:.0f} > EMA200={ema200:.0f}, "
                               f"gunluk +%{day_pct:.2f}")
        # Risk-off: belirgin negatif gun VEYA (death cross + fiyat hala EMA50 altinda).
        # Death cross olsa bile fiyat EMA50 uzerine cikmissa toparlanma sayilir → neutral.
        if day_pct <= -1.5 or (ema50 < ema200 and last < ema50):
            return 'risk-off', (f"XU100={last:.0f}, EMA50={ema50:.0f} vs EMA200={ema200:.0f}, "
                                f"gunluk %{day_pct:.2f}")
        return 'neutral', (f"XU100={last:.0f}, EMA50={ema50:.0f}, EMA200={ema200:.0f}, "
                           f"gunluk %{day_pct:.2f}")
    except Exception as e:
        return 'risk-on', f'Rejim hesap hatasi (default risk-on): {e}'


def get_market_regime(force: bool = False) -> tuple:
    """Cache'li mod (TTL 5dk). force=True bypass."""
    global _REGIME_CACHE
    now = time.time()
    if not force and (now - _REGIME_CACHE['ts']) < _TTL_SEC:
        return _REGIME_CACHE['mode'], _REGIME_CACHE['detail']
    mode, detail = _classify_regime()
    if mode != _REGIME_CACHE.get('mode'):
        print(f"[REGIME] Degisim: {_REGIME_CACHE.get('mode')} → {mode}: {detail}")
        # Telegram'a tek seferlik bildirim (mod degistiginde)
        try:
            from telegram_notifications import send_telegram
            emoji = {'risk-on': '🟢', 'neutral': '🟡', 'risk-off': '🔴'}.get(mode, '⚪')
            send_telegram(
                f"{emoji} <b>Piyasa Rejimi: {mode.upper()}</b>\n"
                f"{detail}\n"
                + ("⛔ Yeni pozisyon acilmayacak (sadece mevcut yonetim)." if mode == 'risk-off'
                   else "⚠️ Daha siki filtre: A+ sinyaller, gunluk limit yarida." if mode == 'neutral'
                   else "✅ Normal islem.")
            )
        except Exception:
            pass
    else:
        print(f"[REGIME] {mode}: {detail}")
    _REGIME_CACHE = {'mode': mode, 'ts': now, 'detail': detail}
    return mode, detail


def regime_blocks_new_position() -> tuple:
    """Risk-off → True (alim engellenir). detail caller log/telegram icin."""
    mode, detail = get_market_regime()
    if mode == 'risk-off':
        return True, f"Risk-off rejim: {detail}"
    return False, ''


def regime_score_threshold_bonus() -> float:
    """Neutral mod'da min_score'a +1.0 ekle (A+ sinyal kuralı). Risk-on/off'ta 0."""
    mode, _ = get_market_regime()
    return 1.0 if mode == 'neutral' else 0.0


def regime_daily_trade_factor() -> float:
    """Neutral mod'da max_daily_trades carpani (0.5 = yariya inder)."""
    mode, _ = get_market_regime()
    return 0.5 if mode == 'neutral' else 1.0
