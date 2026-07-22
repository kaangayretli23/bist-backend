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


# setup_quality bandı -> tarihsel 5g MUTLAK hareket (ölçüldü 2026-07-21, 40.064 gözlem).
# Dilim tablosu (gün-nötrleştirilmiş, iki alt-dönemde de monoton, t=18.71):
#   setup_q 16.8→%4.11 | 25.5→%4.41 | 33.3→%4.66 | 43.2→%4.94 | 62.6→%5.74
# 3 banda sadeleştirildi. NOT: büyük kısmı volatilite kümelenmesi (bilinen, güvenilir olgu).
_MOVE_BANDS = [
    (45.0, 'YÜKSEK',  5.5),
    (30.0, 'ORTA',    4.7),
    (0.0,  'DÜŞÜK',   4.2),
]


def movement_expectation(setup_q):
    """setup_quality (0-100) -> (etiket, tarihsel_5g_mutlak_hareket_%) veya None.

    ⚠️ YÖN ÖNGÖRÜSÜ DEĞİL. Yalnızca 'bu hisse ne kadar HAREKET eder' (oynaklık).
    setup_quality'nin yön testi çöktü (korelasyon ~0) ama HAREKET testi güçlü: 40.064
    gözlemde setup_q dilimleri 5g mutlak hareketle monoton artıyor (t=18.71, iki dönemde).
    Kartta kullanıcıya 'burada iş var mı' sinyali; çıkış kararı ve yön kullanıcıda.
    """
    try:
        q = float(setup_q)
    except (TypeError, ValueError):
        return None
    for thr, lab, hist in _MOVE_BANDS:
        if q >= thr:
            return (lab, hist)
    return None


def movement_line(setup_q):
    """Telegram kartı için hareket-beklentisi satırı (HTML). setup_q yoksa boş döner."""
    m = movement_expectation(setup_q)
    if not m:
        return ""
    lab, hist = m
    return (f"⚡ <i>Hareket beklentisi: <b>{lab}</b> — bu bantta 5g ort ±%{hist:.1f} "
            f"salınım (n>8k). <b>Yön değil, oynaklık.</b></i>\n")


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


def setup_features(df):
    """OHLC DataFrame (Close/High/Low, kronolojik; son satır = değerlendirme anı) ->
    kurulum özellikleri. En az 20 bar gerekir, yoksa None."""
    import numpy as np
    if df is None or len(df) < 20:
        return None
    c = df['Close'].values.astype(float)
    h = df['High'].values.astype(float)
    l = df['Low'].values.astype(float)
    vol14 = float(np.mean((h[-14:] - l[-14:]) / c[-14:]) * 100)
    ret5 = float((c[-1] / c[-6] - 1) * 100) if len(c) > 6 else 0.0
    diff = np.diff(c[-15:])
    up = float(np.clip(diff, 0, None).mean())
    dn = float(-np.clip(diff, None, 0).mean())
    rsi = 100 - 100 / (1 + up / dn) if dn > 0 else 100.0
    return {'vol14': vol14, 'ret5': ret5, 'rsi': rsi}


def setup_quality(vol14, ret5, rsi):
    """⛔ OOS DOĞRULAMASI ÇÖKTÜ (2026-07-20) — KARTLARDA ARTIK GÖSTERİLMİYOR.

    5.770 gerçek AL sinyalinde (129 sembol, 2026-04..07) geçmişe dönük test edildi:
        canlı-eşdeğer korelasyon : r1 +0.010 | r5 -0.018 | r20 -0.012
        gün-nötrleştirilmiş      : r1 -0.001 | r5 +0.002 | r20 +0.004
    Bant tablosu monoton değil; en yüksek bant (65-80) EN KÖTÜ sonucu verdi.
    Aşağıdaki in-sample +0.45 korelasyon DÖNGÜSELDİ (aynı 39 işlemden fit edildi).

    ⚠️ İlk testte +0.25 korelasyon çıkmıştı — LOOK-AHEAD BUG'ı: sinyal gününün barı
    dahildi, o barın kapanışı sinyalden SONRA oluşuyor. Doğru kurulum: önceki barlar
    + signal_log.price_at_signal. Bu ders kayıtlı kalsın.

    Fonksiyon referans için duruyor; hiçbir karar yolunda KULLANILMIYOR.

    ── Orijinal açıklama (tarihsel) ──
    HIZLI-TRADE KURULUM KALİTESİ (0-100).

    Kullanıcının GERÇEK kazanan işlemleriyle korele faktörlerden türetildi (Midas 3 ay, 39 işlem):
    volatilite (hareket alanı) + kısa momentum (RSI, önceki 5g getiri). Yüksek = hızlı momentum
    trade'i için iyi ADAY.

    ⚠️ ÖNEMLİ: Bu bir YÖN-ÖNGÖRÜSÜ DEĞİL. Yalnızca HIZLI çıkışla çalışır — multi-day tutulursa
    momentum reverse eder (kanıtlı). 39-işlem/2-outlier küçük örnekleminden → DENEYSEL, 1 ayda
    yeni veriyle doğrulanacak (takvim hatırlatması). Ağırlıklar ENV ile ayarlanabilir.
    """
    def c01(x):
        return 0.0 if x < 0 else (1.0 if x > 1.0 else x)
    vol_n = c01((float(vol14) - 1.5) / (8.0 - 1.5))    # %1.5..8 aralık -> 0..1
    ret_n = c01((float(ret5) + 5.0) / 15.0)            # -5..+10 -> 0..1
    rsi_n = c01((float(rsi) - 40.0) / 40.0)            # 40..80 -> 0..1
    import os
    def w(k, d):
        try: return float(os.environ.get(k, d))
        except (TypeError, ValueError): return float(d)
    wv, wr, ws = w('SETUP_W_VOL', 0.40), w('SETUP_W_RET', 0.35), w('SETUP_W_RSI', 0.25)
    tot = wv + wr + ws
    q = (wv * vol_n + wr * ret_n + ws * rsi_n) / tot if tot else 0.0
    return round(q * 100)


def setup_quality_from_df(df):
    """OHLC df -> (setup_quality 0-100, features) tek çağrı. Kısa yol."""
    f = setup_features(df)
    if not f:
        return None, None
    return setup_quality(f['vol14'], f['ret5'], f['rsi']), f


def log_setup_quality(uid, symbol, price, setup_q, features, score=None, confidence=None):
    """setup_quality'yi KALICI kaydet — 1-ay sonra OOS doğrulaması için (kartta gösterip
    unutmayalım). Forward getiri sonra OHLC'den hesaplanır; burada giriş-anı snapshot'ı tutulur.
    Lazy-create; sinyal seyrek olduğu için kendi bağlantısını açar."""
    try:
        from config import get_db
        db = get_db()
        db.execute("""CREATE TABLE IF NOT EXISTS setup_quality_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, symbol TEXT, price REAL,
            setup_q REAL, score REAL, confidence REAL, vol14 REAL, ret5 REAL, rsi REAL,
            logged_at REAL DEFAULT (strftime('%s','now')))""")
        f = features or {}
        db.execute(
            """INSERT INTO setup_quality_log
               (user_id, symbol, price, setup_q, score, confidence, vol14, ret5, rsi)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (uid, symbol, price, setup_q, score, confidence,
             f.get('vol14'), f.get('ret5'), f.get('rsi')))
        db.commit(); db.close()
    except Exception as _e:
        print(f"[SETUP-LOG] kayit hatasi: {_e}")


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
