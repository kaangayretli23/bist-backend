"""
signal_tracker.py - Sinyal performans takip sistemi
Her AL/SAT sinyalini kaydeder, 5/10/20 günlük sonuçları izler.
"""
import time
from datetime import datetime, timedelta

from config import sf, get_db

_DEDUP_HOURS = 6  # Aynı sinyal 6 saat içinde tekrar kaydedilmez


def _get_db():
    # config.get_db() Postgres kullanıyorsa PgConnection, SQLite kullanıyorsa sqlite3.Connection döner.
    # Her ikisi de .execute/.fetchone/.fetchall/.commit/.close arayüzünü destekler.
    return get_db()


# =====================================================================
# SİNYAL KAYIT
# =====================================================================

def log_signals_batch(results, timeframe):
    """
    Tarama sonucu listesindeki AL/SAT sinyallerini signal_log'a kaydeder.
    Aynı symbol+timeframe+action için 6 saat içinde tekrar kayıt yapılmaz.
    """
    if not results:
        return
    try:
        db = _get_db()
        cutoff_ts = time.time() - _DEDUP_HOURS * 3600
        logged = 0
        for r in results:
            sym = r.get('symbol', '') or r.get('code', '')
            score = float(r.get('score', 0))
            if score == 0:
                continue
            action = 'AL' if score > 0 else 'SAT'
            price = r.get('price') or r.get('currentPrice') or r.get('cur') or 0
            price = float(price) if price else 0.0

            # Indikatörler (hem düz alan hem nested indicators dict)
            ind = r.get('indicators') or {}
            rsi_val  = r.get('rsi')  or ind.get('rsi')
            macd_val = r.get('macd') or ind.get('macd')
            try:
                rsi_val  = float(rsi_val)  if rsi_val  is not None else None
            except Exception:
                rsi_val = None
            try:
                macd_val = float(macd_val) if macd_val is not None else None
            except Exception:
                macd_val = None

            # Dedup kontrolü
            existing = db.execute(
                "SELECT id FROM signal_log WHERE symbol=? AND timeframe=? AND action=? AND logged_at>?",
                (sym, timeframe, action, cutoff_ts)
            ).fetchone()
            if existing:
                continue

            db.execute(
                """INSERT INTO signal_log
                   (symbol, timeframe, action, score, price_at_signal, rsi, macd, logged_at)
                   VALUES (?,?,?,?,?,?,?,?)""",
                (sym, timeframe, action, score, price, rsi_val, macd_val, time.time())
            )
            logged += 1

        db.commit()
        db.close()
        if logged:
            print(f"[SIGNAL-TRACKER] {logged} sinyal kaydedildi ({timeframe})")
    except Exception as e:
        print(f"[SIGNAL-TRACKER] log hatası: {e}")


# =====================================================================
# SONUÇ GÜNCELLEME (arka plan thread)
# =====================================================================

def check_pending_outcomes():
    """
    Sonucu henüz kaydedilmemiş sinyaller için 5/10/20 günlük fiyatları kontrol eder.
    Saatte bir arka plan thread'den çağrılır.
    """
    try:
        from config import _cget_hist
        db = _get_db()
        now_ts = time.time()

        rows = db.execute(
            "SELECT id, symbol, action, price_at_signal, logged_at "
            "FROM signal_log WHERE outcome_checked=0"
        ).fetchall()

        updated = 0
        fully_done = 0
        for row in rows:
            sig_id       = row['id']
            sym          = row['symbol']
            action       = row['action']
            entry_price  = row['price_at_signal']
            logged_at    = row['logged_at']

            df = _cget_hist(f"{sym}_1y")
            if df is None or len(df) < 5:
                continue

            sig_dt   = datetime.fromtimestamp(logged_at)
            all_done = True
            any_new  = False

            for horizon in [1, 3, 5, 10, 20]:
                target_dt = sig_dt + timedelta(days=horizon)
                if datetime.now() < target_dt:
                    all_done = False
                    continue

                # Zaten kaydedilmiş mi?
                exists = db.execute(
                    "SELECT id FROM signal_outcomes WHERE signal_id=? AND horizon_days=?",
                    (sig_id, horizon)
                ).fetchone()
                if exists:
                    continue

                try:
                    idx_obj = df.index
                    if hasattr(idx_obj, 'tz') and idx_obj.tz is not None:
                        idx_obj = idx_obj.tz_convert(None)
                    pos = idx_obj.searchsorted(target_dt)
                    if pos >= len(df):
                        pos = len(df) - 1
                    target_price = float(df.iloc[pos]['Close'])

                    if entry_price and entry_price > 0:
                        pct = (target_price - entry_price) / entry_price * 100.0
                        if action == 'SAT':
                            pct = -pct  # SAT için kazanç → fiyat düşünce pozitif
                        win = 1 if pct > 0 else 0

                        db.execute(
                            """INSERT OR REPLACE INTO signal_outcomes
                               (signal_id, horizon_days, price_at_horizon, return_pct, win)
                               VALUES (?,?,?,?,?)""",
                            (sig_id, horizon, target_price, round(pct, 3), win)
                        )
                        updated += 1
                        any_new = True
                except Exception:
                    pass

            if all_done:
                db.execute("UPDATE signal_log SET outcome_checked=1 WHERE id=?", (sig_id,))
                fully_done += 1

        db.commit()
        db.close()
        if updated:
            print(f"[SIGNAL-TRACKER] {updated} sonuç güncellendi, {fully_done} sinyal tamamlandı")
    except Exception as e:
        print(f"[SIGNAL-TRACKER] outcome check hatası: {e}")


# =====================================================================
# İSTATİSTİK API
# =====================================================================

def get_signal_stats():
    """Timeframe + horizon bazlı kazanma oranı, ortalama getiri istatistikleri döner."""
    try:
        db = _get_db()

        # Zaman dilimine ve horizona göre özet
        rows = db.execute("""
            SELECT sl.timeframe, so.horizon_days,
                   COUNT(*)            AS total,
                   SUM(so.win)         AS wins,
                   AVG(so.return_pct)  AS avg_return,
                   MAX(so.return_pct)  AS best,
                   MIN(so.return_pct)  AS worst
            FROM signal_log sl
            JOIN signal_outcomes so ON sl.id = so.signal_id
            GROUP BY sl.timeframe, so.horizon_days
            ORDER BY sl.timeframe, so.horizon_days
        """).fetchall()

        by_timeframe = []
        for r in rows:
            total = int(r['total'] or 0)
            wins  = int(r['wins']  or 0)
            by_timeframe.append({
                'timeframe':  r['timeframe'],
                'horizon':    r['horizon_days'],
                'total':      total,
                'wins':       wins,
                'losses':     total - wins,
                'winRate':    round(wins / total * 100, 1) if total else 0,
                'avgReturn':  round(float(r['avg_return'] or 0), 2),
                'best':       round(float(r['best']  or 0), 2),
                'worst':      round(float(r['worst'] or 0), 2),
            })

        # Skor aralığına göre
        bucket_rows = db.execute("""
            SELECT
                CASE
                    WHEN ABS(sl.score) < 3 THEN '0-3'
                    WHEN ABS(sl.score) < 6 THEN '3-6'
                    WHEN ABS(sl.score) < 9 THEN '6-9'
                    ELSE '9+'
                END AS score_bucket,
                so.horizon_days,
                COUNT(*)           AS total,
                SUM(so.win)        AS wins,
                AVG(so.return_pct) AS avg_return
            FROM signal_log sl
            JOIN signal_outcomes so ON sl.id = so.signal_id
            GROUP BY score_bucket, so.horizon_days
            ORDER BY score_bucket, so.horizon_days
        """).fetchall()

        by_score = []
        for r in bucket_rows:
            total = int(r['total'] or 0)
            wins  = int(r['wins']  or 0)
            by_score.append({
                'bucket':    r['score_bucket'],
                'horizon':   r['horizon_days'],
                'total':     total,
                'wins':      wins,
                'winRate':   round(wins / total * 100, 1) if total else 0,
                'avgReturn': round(float(r['avg_return'] or 0), 2),
            })

        # Aylık trend (son 6 ay)
        trend_rows = db.execute("""
            SELECT strftime('%Y-%m', datetime(sl.logged_at, 'unixepoch')) AS month,
                   so.horizon_days,
                   COUNT(*)           AS total,
                   SUM(so.win)        AS wins,
                   AVG(so.return_pct) AS avg_return
            FROM signal_log sl
            JOIN signal_outcomes so ON sl.id = so.signal_id
            WHERE sl.logged_at > ?
            GROUP BY month, so.horizon_days
            ORDER BY month, so.horizon_days
        """, (time.time() - 180 * 86400,)).fetchall()

        monthly_trend = []
        for r in trend_rows:
            total = int(r['total'] or 0)
            wins  = int(r['wins']  or 0)
            monthly_trend.append({
                'month':     r['month'],
                'horizon':   r['horizon_days'],
                'total':     total,
                'wins':      wins,
                'winRate':   round(wins / total * 100, 1) if total else 0,
                'avgReturn': round(float(r['avg_return'] or 0), 2),
            })

        # Genel özet
        sumrow = db.execute("""
            SELECT COUNT(DISTINCT sl.id)    AS total_signals,
                   COUNT(so.id)             AS total_outcomes,
                   COALESCE(SUM(so.win), 0) AS total_wins,
                   AVG(so.return_pct)       AS overall_avg_return
            FROM signal_log sl
            LEFT JOIN signal_outcomes so ON sl.id = so.signal_id
        """).fetchone()

        db.close()
        total_sigs = int(sumrow['total_signals'] or 0)
        total_out  = int(sumrow['total_outcomes'] or 0)
        total_wins = int(sumrow['total_wins'] or 0)
        return {
            'summary': {
                'totalSignals':    total_sigs,
                'totalOutcomes':   total_out,
                'totalWins':       total_wins,
                'overallWinRate':  round(total_wins / total_out * 100, 1) if total_out else 0,
                'overallAvgReturn': round(float(sumrow['overall_avg_return'] or 0), 2),
            },
            'byTimeframe':  by_timeframe,
            'byScoreBucket': by_score,
            'monthlyTrend': monthly_trend,
        }
    except Exception as e:
        print(f"[SIGNAL-TRACKER] stats hatası: {e}")
        return {'error': str(e)}


def get_recent_signal_log(limit=100, timeframe=None, action=None, symbol=None):
    """Son sinyal kayıtlarını outcomes ile birlikte döner."""
    try:
        db = _get_db()
        where, params = [], []
        if timeframe:
            where.append("sl.timeframe=?"); params.append(timeframe)
        if action:
            where.append("sl.action=?");    params.append(action)
        if symbol:
            where.append("sl.symbol LIKE ?"); params.append(f"%{symbol.upper()}%")

        where_sql = ("WHERE " + " AND ".join(where)) if where else ""

        rows = db.execute(f"""
            SELECT sl.id, sl.symbol, sl.timeframe, sl.action, sl.score,
                   sl.price_at_signal, sl.rsi, sl.macd,
                   sl.logged_at, sl.outcome_checked,
                   GROUP_CONCAT(
                       so.horizon_days || ':' || so.return_pct || ':' || so.win, '|'
                   ) AS outcomes
            FROM signal_log sl
            LEFT JOIN signal_outcomes so ON sl.id = so.signal_id
            {where_sql}
            GROUP BY sl.id
            ORDER BY sl.logged_at DESC
            LIMIT ?
        """, params + [limit]).fetchall()

        result = []
        for r in rows:
            oc = {}
            if r['outcomes']:
                for part in r['outcomes'].split('|'):
                    try:
                        hd, pct, win = part.split(':')
                        oc[int(hd)] = {'pct': float(pct), 'win': int(win)}
                    except Exception:
                        pass
            result.append({
                'id':             r['id'],
                'symbol':         r['symbol'],
                'timeframe':      r['timeframe'],
                'action':         r['action'],
                'score':          round(float(r['score']), 1),
                'price':          r['price_at_signal'],
                'rsi':            r['rsi'],
                'macd':           r['macd'],
                'loggedAt':       datetime.fromtimestamp(r['logged_at']).strftime('%d.%m.%Y %H:%M'),
                'outcomeChecked': bool(r['outcome_checked']),
                'outcomes':       oc,
            })
        db.close()
        return result
    except Exception as e:
        print(f"[SIGNAL-TRACKER] log query hatası: {e}")
        return []
