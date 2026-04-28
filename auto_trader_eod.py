"""
BIST Pro - Gun Sonu Raporu Thread
Hafta ici aksam 18:30 (BIST kapanis +30 dk) Telegram'a gunluk ozet yollar.
backend.py startup'ta start_eod_thread() cagrilir. Gunde 1 kez (date dedup).
"""
import threading
import time
from datetime import datetime, timezone, timedelta

_TZ_TR = timezone(timedelta(hours=3))
_WIN_START_MIN = 18 * 60 + 30   # 18:30
_WIN_END_MIN = 18 * 60 + 40     # 18:40 (exclusive)

_last_run_date: str = ''
_lock = threading.Lock()


def _build_eod_summary(uid: str) -> str:
    """Daily-summary mantigini SQL bazli icerik olarak format uretir."""
    try:
        from config import get_db, _cget, _stock_cache
        from auto_trader import _auto_get_open_positions, _calc_trade_costs
    except Exception as e:
        return f"[EOD] Modul import hatasi: {e}"

    today = datetime.now(_TZ_TR).strftime('%Y-%m-%d')
    today_floor = today + ' 00:00:00'
    db = get_db()
    try:
        closed = db.execute(
            "SELECT symbol, entry_price, close_price, quantity, pnl, pnl_pct, close_reason "
            "FROM auto_positions WHERE user_id=? AND status='closed' AND closed_at >= ? "
            "ORDER BY pnl DESC",
            (uid, today_floor)
        ).fetchall()
        trades_today = db.execute(
            "SELECT COUNT(*) AS c, action FROM auto_trades "
            "WHERE user_id=? AND created_at >= ? GROUP BY action",
            (uid, today_floor)
        ).fetchall()
        dec_summary = db.execute(
            "SELECT decision, COUNT(*) AS c FROM auto_decisions "
            "WHERE user_id=? AND created_at >= ? GROUP BY decision",
            (uid, today_floor)
        ).fetchall()
    finally:
        db.close()

    realized = sum(float(r['pnl'] or 0) for r in closed)
    winners = sum(1 for r in closed if float(r['pnl'] or 0) > 0)
    losers = sum(1 for r in closed if float(r['pnl'] or 0) < 0)
    trade_counts = {r['action']: int(r['c']) for r in trades_today}
    dec_counts = {r['decision']: int(r['c']) for r in dec_summary}

    # Acik pozisyon unrealized
    open_positions = _auto_get_open_positions(uid) or []
    unrealized = 0.0
    for p in open_positions:
        stock = _cget(_stock_cache, p['symbol'])
        cur = float(stock.get('price', 0)) if stock else 0
        if cur > 0:
            unrealized += (cur - float(p['entryPrice'])) * float(p['quantity'])

    total = realized + unrealized

    # Komisyon
    commission = 0.0
    for r in closed:
        try:
            entry = float(r['entry_price'] or 0)
            cp = float(r['close_price'] or 0)
            qty = float(r['quantity'] or 0)
            commission += _calc_trade_costs(uid, entry * qty, cp * qty)
        except Exception:
            pass

    # Format
    lines = [f"🌙 <b>Gün Sonu Raporu — {today}</b>", ""]
    sgn = lambda n: f"+{n:,.0f}".replace(',', '.') if n >= 0 else f"{n:,.0f}".replace(',', '.')

    lines += ["💰 <b>P&L:</b>",
              f"  Gerçekleşen: {sgn(realized)} TL ({winners}K/{losers}Z)",
              f"  Açık (unrealized): {sgn(unrealized)} TL",
              f"  <b>NET: {sgn(total)} TL</b>",
              f"  Komisyon: {commission:,.0f}".replace(',', '.') + " TL",
              ""]

    lines += ["📊 <b>İşlemler:</b>",
              f"  Bugün BUY: {trade_counts.get('BUY', 0)}",
              f"  Bugün SELL: " +
              str(sum(v for k, v in trade_counts.items() if k.startswith('SELL'))),
              f"  Açık pozisyon: {len(open_positions)}",
              ""]

    if dec_counts:
        lines += ["⚖️ <b>Karar dağılımı:</b>",
                  f"  ✅ Alım: {dec_counts.get('BUY', 0)}",
                  f"  ⏸ Bekleyen: {dec_counts.get('PENDING', 0)}",
                  f"  🚫 Atlanan: {dec_counts.get('SKIP', 0)}",
                  ""]

    if closed:
        winners_list = [r for r in closed if float(r['pnl'] or 0) > 0][:3]
        losers_list = sorted([r for r in closed if float(r['pnl'] or 0) < 0],
                             key=lambda r: float(r['pnl'] or 0))[:3]
        if winners_list:
            lines.append("🏆 <b>Bugünün kazananları:</b>")
            for r in winners_list:
                p = float(r['pnl'] or 0); pp = float(r['pnl_pct'] or 0)
                lines.append(f"  {r['symbol']} {sgn(p)} TL ({sgn(pp)}%)")
            lines.append("")
        if losers_list:
            lines.append("📉 <b>Bugünün kaybedenleri:</b>")
            for r in losers_list:
                p = float(r['pnl'] or 0); pp = float(r['pnl_pct'] or 0)
                lines.append(f"  {r['symbol']} {sgn(p)} TL ({sgn(pp)}%)")
            lines.append("")

    # Yarın için hatırlatma
    try:
        from auto_trader_regime import get_market_regime
        mode, _detail = get_market_regime()
        emoji = {'risk-on': '🟢', 'neutral': '🟡', 'risk-off': '🔴'}.get(mode, '⚪')
        lines.append(f"{emoji} Piyasa rejimi: <b>{mode.upper()}</b>")
    except Exception:
        pass

    return '\n'.join(lines)


def _run_once_for_all_users() -> None:
    try:
        from config import get_db
        from auto_trader import _auto_get_config
        db = get_db()
        users = db.execute("SELECT user_id FROM auto_config WHERE enabled=1").fetchall()
        db.close()
    except Exception as e:
        print(f"[EOD] DB user fetch hatasi: {e}")
        return

    for u in users:
        uid = u['user_id']
        try:
            cfg = _auto_get_config(uid)
            if not cfg or not cfg.get('enabled'):
                continue
            msg = _build_eod_summary(uid)
            try:
                from telegram_notifications import send_telegram
                send_telegram(msg)
                print(f"[EOD] {uid}: gun sonu raporu yollandi")
            except Exception as e:
                print(f"[EOD] {uid} telegram hatasi: {e}")
        except Exception as e:
            print(f"[EOD] {uid} hata: {e}")


def _eod_loop() -> None:
    global _last_run_date
    print("[EOD] Gun sonu raporu thread basladi (18:30-18:39 TR)")
    while True:
        try:
            now = datetime.now(_TZ_TR)
            today = now.strftime('%Y-%m-%d')
            minute_of_day = now.hour * 60 + now.minute
            in_window = (now.weekday() < 5
                         and _WIN_START_MIN <= minute_of_day < _WIN_END_MIN)
            with _lock:
                already_ran = (_last_run_date == today)
                do_run = in_window and not already_ran
                if do_run:
                    _last_run_date = today
            if do_run:
                print(f"[EOD] {today} {now.strftime('%H:%M')} → rapor olusturuluyor")
                _run_once_for_all_users()
        except Exception as e:
            print(f"[EOD] Loop hatasi: {e}")
        time.sleep(60)


def start_eod_thread():
    """backend.py startup'tan cagrilir."""
    t = threading.Thread(target=_eod_loop, daemon=True, name='eod-report')
    t.start()
    return t
