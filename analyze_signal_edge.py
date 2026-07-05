"""
analyze_signal_edge.py — Sinyal edge analiz araci (Kemal raund 6 hazirligi)

signal_log + signal_outcomes tablolarini join edip su 3 raporu uretir:
  1. FAKTOR EDGE       — her scoreBreakdown kovasinin (poz/neg/sifir) win% ve
                         EXCESS getirisi (return_pct - index_return_pct, Kemal #3).
  2. ML_CONFIDENCE DECILE — ml_confidence kovalarina gore win% + ort. getiri + excess.
                         Kemal raund 6'nin ana konusu: "conf kalibre mi?" Skor kapisini
                         ml_confidence'a tasimadan ONCE bu tabloya bakilacak.
  3. MTF SHADOW EDGE   — factors JSON'undaki '_mtf' shadow sinyalinin (skora katilmayan)
                         yon-uyumuna gore win% + excess. 3 hafta veri birikince okunur.

Kullanim:
    python analyze_signal_edge.py [--since YYYY-MM-DD] [--min-n 10]

Not: excess = getiri - endeks(XU100) getirisi. index_return_pct NULL olan (eski) satirlar
excess hesabindan dislanir; ham getiri her zaman raporlanir. Yon-duzeltme (SAT sinyalinde
isaret ters) hem return hem index tarafinda zaten signal_tracker'da yapiliyor.
"""
import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict
from datetime import datetime

# Windows cp1254 konsol/redirect'te Unicode ok vb. patlamasin — stdout'u UTF-8'e sabitle.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DB_PATH = "bist.db"


def _load_rows(since_epoch):
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        """
        SELECT s.id, s.action, s.score, s.factors, s.ml_confidence, s.logged_at,
               o.horizon_days, o.return_pct, o.win, o.index_return_pct
        FROM signal_log s
        JOIN signal_outcomes o ON o.signal_id = s.id
        WHERE s.logged_at >= ? AND o.return_pct IS NOT NULL
        """,
        (since_epoch,),
    ).fetchall()
    con.close()
    return rows


def _fmt_bucket(agg, min_n):
    """agg: key -> [n, wins, sum_ret, n_ex, sum_excess] → yazdirilabilir satirlar."""
    out = []
    for key in sorted(agg):
        n, w, sr, nex, sex = agg[key]
        if n < min_n:
            continue
        win = 100 * w / n if n else 0
        ret = sr / n if n else 0
        exc = (sex / nex) if nex else None
        exc_s = f"{exc:>9.2f}" if exc is not None else "     n/a"
        out.append(f"    {str(key):<26}{n:>6}{win:>8.1f}{ret:>10.2f}{exc_s}")
    return out


def report_factor_edge(rows, min_n):
    # faktor -> horizon -> sign -> [n, wins, sum_ret, n_excess, sum_excess]
    stats = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0, 0, 0.0])))
    horizons = set()
    for r in rows:
        try:
            f = json.loads(r["factors"]) if r["factors"] else {}
        except Exception:
            continue
        hd = r["horizon_days"]
        horizons.add(hd)
        ret = r["return_pct"]
        win = r["win"] or 0
        idx = r["index_return_pct"]
        excess = (ret - idx) if idx is not None else None
        for name, contrib in f.items():
            if name.startswith("_"):
                continue  # shadow/meta anahtarlar ( or. _mtf) faktor kovasi degil
            try:
                c = float(contrib)
            except (TypeError, ValueError):
                continue
            sign = "poz" if c > 0 else ("neg" if c < 0 else "sifir")
            cell = stats[name][hd][sign]
            cell[0] += 1
            cell[1] += win
            cell[2] += ret
            if excess is not None:
                cell[3] += 1
                cell[4] += excess

    print("=" * 70)
    print("1. FAKTOR EDGE  (n / win% / ort_ret% / ort_EXCESS%)")
    print("=" * 70)
    for hd in sorted(horizons):
        print(f"\n--- horizon {hd} gun ---")
        print(f"    {'faktor/kova':<26}{'n':>6}{'win%':>8}{'ret%':>10}{'excess%':>9}")
        for name in sorted(stats):
            agg = {f"{name} [{sign}]": stats[name][hd][sign] for sign in ("poz", "neg", "sifir")}
            for line in _fmt_bucket(agg, min_n):
                print(line)


def report_confidence_decile(rows, min_n):
    # conf bucket -> horizon -> [n, wins, sum_ret, n_excess, sum_excess]
    def bucket(conf):
        if conf is None:
            return None
        c = float(conf)
        lo = int(c // 10) * 10
        return f"{lo}-{lo+10}"

    stats = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0, 0, 0.0]))
    horizons = set()
    for r in rows:
        b = bucket(r["ml_confidence"])
        if b is None:
            continue
        hd = r["horizon_days"]
        horizons.add(hd)
        idx = r["index_return_pct"]
        excess = (r["return_pct"] - idx) if idx is not None else None
        cell = stats[hd][b]
        cell[0] += 1
        cell[1] += r["win"] or 0
        cell[2] += r["return_pct"]
        if excess is not None:
            cell[3] += 1
            cell[4] += excess

    print("\n" + "=" * 70)
    print("2. ML_CONFIDENCE DECILE  (n / win% / ort_ret% / ort_EXCESS%)")
    print("   Monotonik artan win%/excess → conf kalibre (skor kapisi tasinabilir).")
    print("=" * 70)
    for hd in sorted(horizons):
        print(f"\n--- horizon {hd} gun ---")
        print(f"    {'conf araligi':<26}{'n':>6}{'win%':>8}{'ret%':>10}{'excess%':>9}")
        for line in _fmt_bucket(stats[hd], min_n):
            print(line)


def report_mtf_shadow(rows, min_n):
    # _mtf shadow: dir uyumu (align/karsit/notr) -> horizon -> agg
    stats = defaultdict(lambda: defaultdict(lambda: [0, 0, 0.0, 0, 0.0]))
    horizons = set()
    seen_any = False
    for r in rows:
        try:
            f = json.loads(r["factors"]) if r["factors"] else {}
        except Exception:
            continue
        mtf = f.get("_mtf")
        if not isinstance(mtf, dict):
            continue
        seen_any = True
        mdir = mtf.get("dir", "neutral")
        action = (r["action"] or "").upper()
        sig_dir = "buy" if action in ("AL", "BUY") else "sell"
        if mdir == "neutral":
            key = "notr"
        elif mdir == sig_dir:
            key = "uyumlu"
        else:
            key = "karsit"
        hd = r["horizon_days"]
        horizons.add(hd)
        idx = r["index_return_pct"]
        excess = (r["return_pct"] - idx) if idx is not None else None
        cell = stats[hd][key]
        cell[0] += 1
        cell[1] += r["win"] or 0
        cell[2] += r["return_pct"]
        if excess is not None:
            cell[3] += 1
            cell[4] += excess

    print("\n" + "=" * 70)
    print("3. MTF SHADOW EDGE  (skora katilmayan sinyal — n / win% / ret% / excess%)")
    print("=" * 70)
    if not seen_any:
        print("    (Henuz '_mtf' shadow verisi yok — shadow loglama sonrasi sinyaller birikmeli.)")
        return
    for hd in sorted(horizons):
        print(f"\n--- horizon {hd} gun ---")
        print(f"    {'mtf yon-uyumu':<26}{'n':>6}{'win%':>8}{'ret%':>10}{'excess%':>9}")
        for line in _fmt_bucket(stats[hd], min_n):
            print(line)


def main():
    ap = argparse.ArgumentParser(description="Sinyal edge analizi (Kemal raund 6)")
    ap.add_argument("--since", default=None, help="YYYY-MM-DD (default: tum veri)")
    ap.add_argument("--min-n", type=int, default=10, help="kova basina min gozlem (default 10)")
    args = ap.parse_args()

    since_epoch = 0.0
    if args.since:
        since_epoch = datetime.strptime(args.since, "%Y-%m-%d").timestamp()

    rows = _load_rows(since_epoch)
    n_ex = sum(1 for r in rows if r["index_return_pct"] is not None)
    print(f"Yuklenen sinyal x horizon satiri: {len(rows)}  (excess hesaplanabilir: {n_ex})")
    if args.since:
        print(f"Filtre: logged_at >= {args.since}")
    if not rows:
        print("Veri yok.")
        return

    report_factor_edge(rows, args.min_n)
    report_confidence_decile(rows, args.min_n)
    report_mtf_shadow(rows, args.min_n)

    print("\n" + "-" * 70)
    print("NOT: excess<0 = sinyal endeksin altinda kaldi (edge yok/negatif).")
    print("Yorumlarken efektif n gun-kumelenmesiyle gorunenden kucuktur (Kemal uyarisi).")


if __name__ == "__main__":
    main()
