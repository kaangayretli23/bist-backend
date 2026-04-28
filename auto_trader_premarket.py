"""
BIST Pro - Pre-market Watchlist Thread
Hafta ici sabah 09:55-10:04 araliginda BIR KEZ calisir:
top-N sinyali aktif kullanicilarin Telegram'ina yollar. Pozisyon acmaz.

backend.py startup'ta start_premarket_thread() cagrilir.
"""
import threading
import time
from datetime import datetime, timezone, timedelta

_TZ_TR = timezone(timedelta(hours=3))
_WIN_START_MIN = 9 * 60 + 55     # 09:55
_WIN_END_MIN = 10 * 60 + 5       # 10:05 (exclusive)

# Gunde tek calistirma — date-string seedlenir runtime'da
_last_run_date: str = ''
_lock = threading.Lock()


def _build_watchlist(cfg: dict, top_n: int = 10) -> list:
    """Aktif konfige gore aday listesi. Pozisyon acmaz. Lazy import."""
    candidates = []
    try:
        from config import _cget_hist, _get_stocks
        from indicators import calc_all_indicators
        from signals import calc_recommendation as _calc_rec
    except Exception as e:
        print(f"[PRE-MARKET] Modul import hatasi: {e}")
        return []

    _tf_map = {'daily': 'daily', 'swing': 'weekly', 'monthly': 'monthly'}
    tf_key = _tf_map.get(cfg.get('tradeStyle', 'swing'), 'weekly')
    allowed = set(s.strip() for s in (cfg.get('allowedSymbols') or '').split(',') if s.strip())
    blocked = set(s.strip() for s in (cfg.get('blockedSymbols') or '').split(',') if s.strip())
    min_score = float(cfg.get('minScore', 5.0))
    min_conf = float(cfg.get('minConfidence', 60))

    stocks = _get_stocks() or []
    for s in stocks:
        sym = (s.get('code') or '').strip()
        if not sym:
            continue
        if allowed and sym not in allowed:
            continue
        if sym in blocked:
            continue
        hist = _cget_hist(f"{sym}_1y")
        if hist is None or len(hist) < 30:
            continue
        try:
            last_close = float(hist['Close'].iloc[-1])
            if last_close <= 0:
                continue
            indics = calc_all_indicators(hist, last_close)
            recs = _calc_rec(hist, indics, symbol=sym)
            rec = recs.get(tf_key) or recs.get('weekly', {})
            signal = rec.get('action', '')
            score = float(rec.get('score', 0))
            conf = float(rec.get('confidence', 0))
            if signal in ('AL', 'GÜÇLÜ AL') and score >= min_score and conf >= min_conf:
                candidates.append({
                    'symbol': sym, 'price': last_close, 'name': s.get('name', sym),
                    'score': score, 'confidence': conf, 'signal': signal,
                })
        except Exception:
            continue

    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_n]


def _send_watchlist_telegram(watchlist: list) -> None:
    if not watchlist:
        msg = "🌅 <b>Sabah Tarama (09:55)</b>\n⚠️ Bugün AL sinyali eşiğini geçen hisse yok."
    else:
        lines = [f"🌅 <b>Sabah Tarama — Top {len(watchlist)} Aday</b>"]
        for i, c in enumerate(watchlist, 1):
            sig_emoji = '🚀' if c['signal'] == 'GÜÇLÜ AL' else '✅'
            lines.append(
                f"{i}. {sig_emoji} <b>{c['symbol']}</b> "
                f"sc={c['score']:.1f} guv=%{c['confidence']:.0f} "
                f"({c['price']:.2f} TL)"
            )
        # Rejim bilgisi
        try:
            from auto_trader_regime import get_market_regime
            mode, detail = get_market_regime()
            emoji = {'risk-on': '🟢', 'neutral': '🟡', 'risk-off': '🔴'}.get(mode, '⚪')
            lines.append(f"\n{emoji} Piyasa: <b>{mode.upper()}</b>")
            if mode == 'risk-off':
                lines.append("⛔ Risk-off — motor yeni pozisyon açmayacak.")
            elif mode == 'neutral':
                lines.append("⚠️ Neutral — sıkı filtre, günlük limit yarıda.")
        except Exception:
            pass
        lines.append("\n💡 Açılışta motor bunları tarayıp uygun olanı otomatik açacak.")
        msg = '\n'.join(lines)
    try:
        from telegram_notifications import send_telegram
        send_telegram(msg)
    except Exception as e:
        print(f"[PRE-MARKET] Telegram gonderim hatasi: {e}")


def _run_once_for_all_users() -> None:
    """Aktif tum kullanicilar icin watchlist + telegram gonderim."""
    try:
        from config import get_db
        from auto_trader import _auto_get_config
        db = get_db()
        users = db.execute("SELECT user_id FROM auto_config WHERE enabled=1").fetchall()
        db.close()
    except Exception as e:
        print(f"[PRE-MARKET] DB user fetch hatasi: {e}")
        return

    for u in users:
        uid = u['user_id']
        try:
            cfg = _auto_get_config(uid)
            if not cfg or not cfg.get('enabled'):
                continue
            watchlist = _build_watchlist(cfg, top_n=10)
            _send_watchlist_telegram(watchlist)
            print(f"[PRE-MARKET] {uid}: {len(watchlist)} aday yollandi")
        except Exception as e:
            print(f"[PRE-MARKET] {uid} hata: {e}")


def _premarket_loop() -> None:
    global _last_run_date
    print("[PRE-MARKET] Watchlist thread basladi (09:55-10:04 TR)")
    while True:
        try:
            now = datetime.now(_TZ_TR)
            today = now.strftime('%Y-%m-%d')
            minute_of_day = now.hour * 60 + now.minute
            in_window = (now.weekday() < 5
                         and _WIN_START_MIN <= minute_of_day < _WIN_END_MIN)
            with _lock:
                already_ran = (_last_run_date == today)
                if in_window and not already_ran:
                    _last_run_date = today
                    do_run = True
                else:
                    do_run = False
            if do_run:
                print(f"[PRE-MARKET] {today} {now.strftime('%H:%M')} → tarama basliyor")
                _run_once_for_all_users()
        except Exception as e:
            print(f"[PRE-MARKET] Loop hatasi: {e}")
        time.sleep(60)


def start_premarket_thread():
    """backend.py startup'tan cagrilir."""
    t = threading.Thread(target=_premarket_loop, daemon=True, name='premarket-watchlist')
    t.start()
    return t


def run_for_user_now(uid: str, top_n: int = 10) -> list:
    """Manuel tetik (UI butonu) — verilen kullanicinin watchlist'ini olustur,
    Telegram'a yolla, sonuclari dondur. Auto-trade aktif olmasa bile calisir
    (sadece bilgilendirme; pozisyon ACMAZ)."""
    try:
        from auto_trader import _auto_get_config
        cfg = _auto_get_config(uid)
        if not cfg:
            return []
        watchlist = _build_watchlist(cfg, top_n=top_n)
        _send_watchlist_telegram(watchlist)
        print(f"[PRE-MARKET-MANUAL] {uid}: {len(watchlist)} aday yollandi")
        return watchlist
    except Exception as e:
        print(f"[PRE-MARKET-MANUAL] {uid} hata: {e}")
        return []
