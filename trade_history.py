# -*- coding: utf-8 -*-
"""
BIST Pro — Islem gecmisi + DURUST performans aynasi (FAZ 3, adim 2).

TASARIM KARARI: bu modul trading koduna DOKUNMAZ. Salt-okunur istatistik uretir.
Kaynaklar birlesir:
  - trade_history tablosu : disaridan alinan gercek gecmis (Midas ekstresi)
  - auto_positions        : sistemin kapattigi pozisyonlar (status='closed')

NEDEN "DURUST" AYNA:
Bugun sistemde 5 kapanmis islem var ve PF 1.71 cikiyor — iyi gorunuyor ama tek bir
EREGL kazanci tasiyor. Kucuk orneklemde PF, insana olmayan bir edge'i varmis gibi
gosterir. Bu modul bunu ENGELLER:
  1) Once orneklem yeterliligi soylenir, sonra sayi.
  2) PF icin bootstrap guven araligi uretilir; aralik 1.0'i iceriyorsa
     "edge KANITLANAMADI" denir — nokta tahmin tek basina asla sunulmaz.
  3) KIRILGANLIK: en iyi 2 islem cikarilinca ne kaliyor. Kullanicinin gercek
     3 aylik verisinde PF 2.22 idi ama o 2 islem cikinca net -216 TL.
"""
import math
import random


# Orneklem esikleri — literaturdeki "PF icin en az 30 islem" pratigi.
MIN_N_INFORMATIVE = 30    # altinda nokta tahmin GOSTERILMEZ
MIN_N_SUGGESTIVE  = 15    # altinda hicbir sey, arasinda "fikir verir" uyarisiyla


def _table(db):
    db.execute("""CREATE TABLE IF NOT EXISTS trade_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT, symbol TEXT, quantity REAL,
        entry_price REAL, exit_price REAL,
        pnl REAL, pnl_pct REAL,
        period TEXT,            -- 'YYYY-MM' (Midas ekstresinde gun yok, ay seviyesinde)
        source TEXT,            -- 'midas' | 'manual'
        note TEXT,
        imported_at REAL DEFAULT (strftime('%s','now')),
        UNIQUE(user_id, symbol, quantity, entry_price, exit_price, period, source)
    )""")


def record_history(uid, rows, source='midas'):
    """rows: [(symbol, qty, entry, exit, period, note)] -> (eklenen, atlanan).
    UNIQUE kisiti sayesinde idempotent: ayni ekstre iki kez yuklenirse cift kayit olmaz."""
    from config import get_db
    db = get_db()
    _table(db)
    added = skipped = 0
    for sym, qty, entry, exit_, period, note in rows:
        cost = qty * entry
        pnl = qty * exit_ - cost
        pnl_pct = (pnl / cost * 100) if cost else 0.0
        try:
            cur = db.execute(
                """INSERT OR IGNORE INTO trade_history
                   (user_id, symbol, quantity, entry_price, exit_price, pnl, pnl_pct,
                    period, source, note)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (uid, sym, qty, entry, exit_, pnl, pnl_pct, period, source, note))
            if cur.rowcount:
                added += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"[TRADE-HIST] {sym} eklenemedi: {e}")
    db.commit(); db.close()
    return added, skipped


def load_trades(uid):
    """Iki kaynagi birlestir -> [{symbol, pnl, pnl_pct, source, period}] (kronolojik degil)."""
    from config import get_db
    db = get_db()
    _table(db)
    out = []
    for r in db.execute(
            "SELECT symbol, pnl, pnl_pct, period, source FROM trade_history WHERE user_id=?", (uid,)):
        out.append({'symbol': r[0], 'pnl': float(r[1] or 0), 'pnl_pct': float(r[2] or 0),
                    'period': r[3], 'source': r[4]})
    try:
        for r in db.execute(
                """SELECT symbol, pnl, pnl_pct, substr(opened_at,1,7)
                   FROM auto_positions WHERE user_id=? AND status='closed'""", (uid,)):
            out.append({'symbol': r[0], 'pnl': float(r[1] or 0), 'pnl_pct': float(r[2] or 0),
                        'period': r[3], 'source': 'system'})
    except Exception as e:
        print(f"[TRADE-HIST] auto_positions okunamadi: {e}")
    db.close()
    return out


def _pf(pnls):
    """Profit factor. Kayip yoksa None (tanimsiz — 'sonsuz' diye sunmak yaniltir)."""
    win = sum(p for p in pnls if p > 0)
    loss = abs(sum(p for p in pnls if p < 0))
    if loss == 0:
        return None
    return win / loss


def _pf_ci(pnls, iters=4000, alpha=0.10, seed=42):
    """PF icin bootstrap guven araligi (varsayilan %90).

    NEDEN: PF nokta tahmini kucuk orneklemde cok oynak. Aralik 1.0'i iceriyorsa
    'kazandiran sistem' iddiasi veriyle DESTEKLENMIYOR demektir.
    """
    n = len(pnls)
    if n < 5:
        return None, None
    rnd = random.Random(seed)
    vals = []
    for _ in range(iters):
        s = [pnls[rnd.randrange(n)] for _ in range(n)]
        v = _pf(s)
        if v is not None:
            vals.append(v)
    if len(vals) < iters * 0.5:      # cogu ornekte hic kayip yok -> aralik anlamsiz
        return None, None
    vals.sort()
    lo = vals[int(len(vals) * (alpha / 2))]
    hi = vals[int(len(vals) * (1 - alpha / 2)) - 1]
    return lo, hi


def compute_stats(uid):
    """Panelin tek veri kaynagi. Orneklem yetersizse SAYI VERMEZ."""
    trades = load_trades(uid)
    n = len(trades)
    pnls = [t['pnl'] for t in trades]

    res = {
        'n': n,
        'sufficient': n >= MIN_N_INFORMATIVE,
        'suggestive': MIN_N_SUGGESTIVE <= n < MIN_N_INFORMATIVE,
        'min_n': MIN_N_INFORMATIVE,
        'sources': {},
    }
    for t in trades:
        res['sources'][t['source']] = res['sources'].get(t['source'], 0) + 1

    if n == 0:
        res['verdict'] = 'Hic kapanmis islem yok.'
        return res

    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    res['wins'], res['losses'], res['flat'] = len(wins), len(losses), n - len(wins) - len(losses)
    res['net'] = sum(pnls)

    if n < MIN_N_SUGGESTIVE:
        # Bilincli olarak PF/win-rate DONDURMUYORUZ — yaniltici olur.
        res['verdict'] = (f"Orneklem cok kucuk (n={n}). PF/isabet orani bu veriden "
                          f"hesaplanabilir ama ANLAMSIZ olur; gosterilmiyor. "
                          f"En az {MIN_N_SUGGESTIVE} islem gerekli.")
        return res

    res['win_rate'] = len(wins) / n * 100
    res['avg_win'] = (sum(wins) / len(wins)) if wins else 0.0
    res['avg_loss'] = (sum(losses) / len(losses)) if losses else 0.0
    res['expectancy'] = sum(pnls) / n
    res['pf'] = _pf(pnls)
    lo, hi = _pf_ci(pnls)
    res['pf_lo'], res['pf_hi'] = lo, hi
    res['pf_proven'] = bool(lo is not None and lo > 1.0)

    # --- KIRILGANLIK: en iyi 2 islem cikarilinca ne kaliyor ---
    top2 = sorted(pnls, reverse=True)[:2]
    rest = sorted(pnls, reverse=True)[2:]
    res['top2_sum'] = sum(top2)
    res['net_without_top2'] = sum(rest)
    res['pf_without_top2'] = _pf(rest)
    res['top2_share'] = (sum(top2) / sum(wins) * 100) if wins else 0.0
    res['fragile'] = bool(sum(pnls) > 0 and sum(rest) <= 0)

    # --- SCALP CHURN: kucuk islemler net ne katiyor ---
    # 'Kucuk' = mutlak P&L medyanin altinda. Cok sayida kucuk islem net negatifse
    # islem sikligi edge'i sulandiriyor demektir (gercek veride 41 islem net -216 idi).
    mags = sorted(abs(p) for p in pnls)
    med = mags[len(mags) // 2] if mags else 0
    small = [p for p in pnls if abs(p) <= med]
    res['small_n'] = len(small)
    res['small_net'] = sum(small)
    res['churn_drag'] = bool(len(small) >= 10 and sum(small) < 0)

    if res['pf_proven']:
        res['verdict'] = f"PF guven araligi tamamen 1.0 uzerinde — edge veriyle destekleniyor."
    elif lo is not None:
        res['verdict'] = (f"PF nokta tahmini {res['pf']:.2f} ama %90 aralik "
                          f"[{lo:.2f}–{hi:.2f}] 1.0'i iceriyor → edge KANITLANAMADI.")
    else:
        res['verdict'] = "Guven araligi hesaplanamadi (yeterli kayip islemi yok)."
    return res


MIDAS_NOTE = ("Midas ekstresi (Nisan-Haziran 2026), FIFO eslestirme, komisyon 0. "
              "Ekstrede gun yok — tarih AY seviyesinde.")
