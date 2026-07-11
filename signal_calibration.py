# -*- coding: utf-8 -*-
"""Sinyal kalibrasyonu — skoru GERÇEK geçmiş forward-sonuca eşler.

Amaç: 'Güven %70, GÜÇLÜ AL' gibi YANILTICI bir sayı yerine, o skor bandının
GERÇEK tarihsel win-oranını ve ortalama getirisini göstermek. Sistem böylece
yalan söylemez; kullanıcı gerçek olasılıkla karar verir.

Kaynak: 5.267 AL sinyali, 5 gün forward, |ret|<30% (2026-04..07 dönemi, mean-reverting rejim).
UYARI: IN-SAMPLE ve tek-rejim. Periyodik yeniden-fit gerekir (recalibrate() ile).
Bulgu: hiçbir skor bandı %48 win'i geçmiyor; skor YÜKSELDİKÇE genelde KÖTÜLEŞİYOR
(8+ → win %37, ort −1.67%). Yani skor öngörü sağlamıyor — bu tablo onu dürüstçe gösterir.
"""

# (skor_alt, skor_ust, gercek_win_pct, ort_5g_getiri_pct, n)
_SCORE_CALIB = [
    (float('-inf'), 2.0, 47.0, -0.04, 1642),
    (2.0,  4.0, 43.3, -0.24, 1588),
    (4.0,  6.0, 45.7, -0.04, 1408),
    (6.0,  7.0, 47.0,  0.02,  338),
    (7.0,  8.0, 42.1, -0.50,  159),
    (8.0,  float('inf'), 37.1, -1.67, 132),
]
_BASELINE_WIN = 45.1   # tüm AL ortalama gerçek win — 'coin flip' referansı


def signal_reality(score):
    """Skor -> (gercek_win_pct, ort_5g_getiri_pct, n, zayif_mi).
    zayif_mi=True: bu bant baseline'ın altında / negatif EV (aktif zarar bölgesi)."""
    try:
        score = float(score)
    except (TypeError, ValueError):
        return _BASELINE_WIN, -0.15, 0, False
    for lo, hi, win, ev, n in _SCORE_CALIB:
        if lo <= score < hi:
            zayif = (win < _BASELINE_WIN - 3) or (ev < -0.3)
            return win, ev, n, zayif
    return _BASELINE_WIN, -0.15, 0, False


def reality_line(score):
    """Telegram kartı için dürüst gerçeklik satırı (HTML)."""
    win, ev, n, zayif = signal_reality(score)
    flag = " — <b>⚠️ tarihsel zayıf</b>" if zayif else ""
    return (f"📉 <i>Gerçek geçmiş (bu skor bandı): win %{win:.0f}, "
            f"5g ort %{ev:+.1f} (n={n})</i>{flag}")


def calibrated_confidence(score):
    """Skorun GERÇEK tarihsel win-oranı (%) — kartta 'güven' yerine gösterilecek dürüst sayı."""
    win, _ev, _n, _z = signal_reality(score)
    return win


def overbought_penalty(rsi, regime):
    """ADIM 2 — aşırı-alım + mean-reverting rejim CEZASI (skordan düşülecek puan).

    Ana kayıp modu: yüksek momentum = zaten tepede = geri döner. Skor bunu ödüllendiriyordu;
    burada tersine çeviriyoruz. GÜÇLÜ TREND'de (strong_bull) ceza YOK (momentum çalışabilir);
    yatay/ayı/nötr rejimde overbought'u cezalandır → tepeleri 'al' diye bağırma.
    """
    try:
        rsi = float(rsi)
    except (TypeError, ValueError):
        return 0.0
    if regime in ('strong_bull', 'bull'):   # trend — momentuma izin ver
        return 0.0
    # sideways / bear / strong_bear / neutral / bilinmeyen → mean-revert riski
    if rsi >= 72:
        return 2.5
    if rsi >= 65:
        return 1.5
    if rsi >= 58:
        return 0.7
    return 0.0


def recalibrate(db_path=None, horizon_days=5, min_n=30):
    """Kalibrasyonu DB'den YENİDEN fit et (yeni veri geldikçe çağır).
    _SCORE_CALIB'i güncellemek için çıktıyı elle koda yapıştır ya da ileride
    otomatik yükle. Döner: bucket listesi [(lab, n, win, ev)]."""
    import sqlite3
    path = db_path or r'C:\Users\Kaan\bist-backend\bist.db'
    db = sqlite3.connect(path); db.row_factory = sqlite3.Row
    buckets = [("<2", "s.score<2"), ("2-4", "s.score>=2 AND s.score<4"),
               ("4-6", "s.score>=4 AND s.score<6"), ("6-7", "s.score>=6 AND s.score<7"),
               ("7-8", "s.score>=7 AND s.score<8"), ("8+", "s.score>=8")]
    out = []
    for lab, cond in buckets:
        r = db.execute(f"""SELECT COUNT(*) n,
                AVG(CASE WHEN o.return_pct>0 THEN 1.0 ELSE 0 END)*100 win, AVG(o.return_pct) ev
                FROM signal_outcomes o JOIN signal_log s ON o.signal_id=s.id
                WHERE s.action='AL' AND o.horizon_days=? AND ABS(o.return_pct)<30 AND {cond}""",
                (horizon_days,)).fetchone()
        if r['n'] and r['n'] >= min_n:
            out.append((lab, r['n'], round(r['win'], 1), round(r['ev'], 2)))
    db.close()
    return out
