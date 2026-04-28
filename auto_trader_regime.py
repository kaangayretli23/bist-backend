"""
BIST Pro - Auto Trader Piyasa Rejim Filtresi
XU100 endeksinin EMA50/EMA200 + gunluk momentum'una gore 3 mod:
  - risk-on:  trend yukari (close>EMA50>EMA200) + gunluk > +%0.5 → normal islem
  - neutral:  notr trend (kosul disindaki butun durumlar) → daha siki filtre
  - risk-off: bear trend (EMA50<EMA200) veya gunluk <= -%1.5 → yeni pozisyon YOK

Sonuc 5 dk cache'lenir; her cycle'da pandas EMA hesabi pahali.
Lazy import: config, pandas modul-icinde fonksiyon ic.
"""
import time

_REGIME_CACHE: dict = {'mode': 'risk-on', 'ts': 0.0, 'detail': 'Henuz degerlendirilmedi'}
_TTL_SEC = 300  # 5 dk


def _classify_regime() -> tuple:
    """XU100 cache'ten oku → (mode, detail). Veri yoksa default risk-on (fail-open)."""
    try:
        from config import _cget_hist
        hist = _cget_hist("XU100_1y")
        if hist is None or len(hist) < 200:
            return 'risk-on', f'Yetersiz XU100 verisi ({len(hist) if hist is not None else 0} bar)'
        close = hist['Close']
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema200 = float(close.ewm(span=200, adjust=False).mean().iloc[-1])
        last = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else last
        day_pct = ((last - prev) / prev * 100.0) if prev > 0 else 0.0

        if last > ema50 > ema200 and day_pct > 0.5:
            return 'risk-on', (f"XU100={last:.0f} > EMA50={ema50:.0f} > EMA200={ema200:.0f}, "
                               f"gunluk +%{day_pct:.2f}")
        if ema50 < ema200 or day_pct <= -1.5:
            return 'risk-off', (f"EMA50={ema50:.0f} vs EMA200={ema200:.0f}, "
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
