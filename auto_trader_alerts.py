"""
BIST Pro - Auto Trader Gunluk Risk Uyarilari
Telegram'a: (a) gunluk zarar limiti %70'e ulasinca, (b) max_daily_trades doldu uyarilari.
Gun icinde tekrar etmesin diye dedupe: auto_decisions tablosuna decision='ALERT' yazilir;
restart sonrasi da gunluk dedup korunur.
"""
from datetime import datetime, timezone, timedelta

_TZ_TR = timezone(timedelta(hours=3))

# RAM cache: {date_str: {uid: set(['loss70','loss100','dailymax'])}}
# Sadece hizlandirma — kanonik kaynak DB. Cold start'ta DB'den seedlenir.
_alerts_sent: dict = {}


def _today_str():
    return datetime.now(_TZ_TR).strftime('%Y-%m-%d')


def _already_sent(uid: str, code: str) -> bool:
    today = _today_str()
    # Gun degistiyse eski kayitlari temizle
    for d in list(_alerts_sent.keys()):
        if d != today:
            _alerts_sent.pop(d, None)
    if code in _alerts_sent.get(today, {}).get(uid, set()):
        return True
    # DB tabanlı: bugun bu kod icin ALERT yazilmis mi
    try:
        from config import get_db
        db = get_db()
        try:
            row = db.execute(
                "SELECT 1 FROM auto_decisions "
                "WHERE user_id=? AND decision='ALERT' AND reason=? AND created_at >= ? "
                "LIMIT 1",
                (uid, code, today + ' 00:00:00')
            ).fetchone()
            if row:
                _alerts_sent.setdefault(today, {}).setdefault(uid, set()).add(code)
                return True
        finally:
            db.close()
    except Exception:
        pass
    return False


def _mark_sent(uid: str, code: str) -> None:
    today = _today_str()
    _alerts_sent.setdefault(today, {}).setdefault(uid, set()).add(code)
    # DB'ye de yaz — restart sonrasi tekrar tetiklenmesin
    try:
        from auto_trader_risk import _log_decision
        _log_decision(uid, '_DAILY', 'ALERT', reason=code, detail=f"Gunluk uyari: {code}")
    except Exception:
        pass


def _send(message: str) -> None:
    try:
        from telegram_notifications import send_telegram
        send_telegram(message)
    except Exception as e:
        print(f"[AUTO-ALERT] Telegram gonderim hatasi: {e}")


def check_daily_alerts(uid: str, cfg: dict, today_realized_pnl: float,
                       unrealized_pnl: float, daily_trades: int) -> None:
    """Gunluk risk esiklerinde Telegram uyarisi gonder. Gun icinde ayni uyari 1 kez.

    Thresholds:
      - loss70:   total_pnl <= -0.70 * (capital * 0.05)  → uyar
      - loss100:  total_pnl <= -(capital * 0.05)         → limit asildi, alim durdu
      - dailymax: daily_trades >= max_daily_trades       → gunluk limit
    """
    try:
        capital = float(cfg.get('capital', 0) or 0)
        if capital <= 0:
            return
        daily_limit = capital * 0.05
        total_pnl = float(today_realized_pnl) + float(unrealized_pnl)

        # Loss alerts
        if total_pnl <= -daily_limit:
            if not _already_sent(uid, 'loss100'):
                _send(
                    "🚨 <b>GÜNLÜK ZARAR LİMİTİ AŞILDI</b>\n"
                    f"Gerçekleşen: {today_realized_pnl:.0f} TL\n"
                    f"Açık pozisyon: {unrealized_pnl:.0f} TL\n"
                    f"Toplam: <b>{total_pnl:.0f} TL</b> (limit: -{daily_limit:.0f} TL)\n"
                    "⛔ Bugün yeni alım yapılmayacak."
                )
                _mark_sent(uid, 'loss100')
        elif total_pnl <= -0.70 * daily_limit:
            if not _already_sent(uid, 'loss70'):
                _send(
                    "⚠️ <b>Günlük zarar %70 eşiğinde</b>\n"
                    f"Toplam PnL: {total_pnl:.0f} TL\n"
                    f"Limit: -{daily_limit:.0f} TL (sermayenin %5'i)\n"
                    f"Kalan tolerans: {(daily_limit + total_pnl):.0f} TL\n"
                    "💡 Açık pozisyonları gözden geçirmeyi düşün."
                )
                _mark_sent(uid, 'loss70')

        # Daily trade limit
        max_trades = int(cfg.get('maxDailyTrades', 0) or 0)
        if max_trades > 0 and daily_trades >= max_trades:
            if not _already_sent(uid, 'dailymax'):
                _send(
                    "📊 <b>Günlük işlem limiti doldu</b>\n"
                    f"Bugünkü alımlar: {daily_trades}/{max_trades}\n"
                    "⛔ Yeni pozisyon açılmayacak (yarına kadar)."
                )
                _mark_sent(uid, 'dailymax')
    except Exception as e:
        print(f"[AUTO-ALERT] check_daily_alerts hatasi: {e}")
