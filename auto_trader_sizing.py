# -*- coding: utf-8 -*-
"""
BIST Pro — Pozisyon buyuklugu (sizing) karari. TEK YER.

FAZ 2'de auto_trader_scanner.py'dan ayrildi: scanner 700 satir kuralini asiyordu ve
PARK EDILMIS kalite-pay mantigi sicak tarama dosyasinin icinde duruyordu.

NIHAI ADET = min(risk-bazli adet, sermaye-tavani adedi)
  - risk-bazli adet  : riskPerTrade / SL mesafesi  (scanner'da hesaplanir)
  - sermaye tavani % : BU MODUL karar verir       (position_pct_for)

IKI MOD:
  ALLOC_QUALITY_ENABLED=0  (VARSAYILAN — PARK)
      Duz esit tavan: her hisse AUTO_MAX_POSITION_PCT (~%20). Guven/skora gore BUYUTME YOK.
      NEDEN PARK: olcum ml_confidence'in ANTI-predictive oldugunu, skor 8+ bandinin EN KOTU
      forward getiriyi verdigini gosterdi (sinyal PF 0.75). Guveni odullendiren kalite-pay
      bu durumda zarari BUYUTUR. Silinmedi cunku setup_quality 1-ay OOS dogrulamasini
      gecerse (2026-08-11) yeniden acilabilir.

  ALLOC_QUALITY_ENABLED=1  (deneysel)
      Kalite-bazli standalone pay — asagidaki oncelik hiyerarsisi.
"""
import os as _os


def _f(key, default):
    try:
        return float(_os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _i(key, default):
    try:
        return int(_os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _c01(x):
    return 0.0 if x < 0 else (1.0 if x > 1.0 else x)


def quality_position_pct(rr, score, confidence):
    """KALITE-BAZLI standalone pozisyon yuzdesi (sermayenin %'si) -> (pct, q).

    Kullanici tasarimi — ONCELIK HIYERARSISI (R/R lider DEGIL):
      1) BIRINCIL = conviction: guven/skor'dan hangisi gucluyse o lider.
      2) IKINCIL  = R/R sadece tiebreaker: benzer conviction'da yuksek R/R one gecer
                    (kucuk ALLOC_RR_TILT katkisi → ana sirayi bozmaz, sadece ayirir).
      3) CEZA     = cok dusuk R/R (ALLOC_RR_FLOOR alti) geri plana atilir (×penalty).
      4) AI 2. goz EN SON bakar (scanner'da tilt olarak, bu fonksiyonun disinda).

    Paylar birbirine NORMALIZE EDILMEZ → her hisse bagimsiz % alir, cherry-pick korunur.
    Sonuc [ALLOC_FLOOR_PCT .. ALLOC_CEIL_PCT] bandina eslenir.
    """
    rr_min, rr_max = _f('ALLOC_RR_MIN', 1.0), _f('ALLOC_RR_MAX', 3.0)
    sc_min, sc_max = _f('ALLOC_SCORE_MIN', 5.0), _f('ALLOC_SCORE_MAX', 10.0)
    rr_n   = _c01((float(rr or 0) - rr_min) / (rr_max - rr_min)) if rr_max > rr_min else 0.0
    sc_n   = _c01((float(score or 0) - sc_min) / (sc_max - sc_min)) if sc_max > sc_min else 0.0
    conf_n = _c01(float(confidence or 0) / 100.0)

    # 1) BIRINCIL conviction — varsayilan 'confidence': skoru lider YAPMA
    # (backtest: yuksek skor = en kotu forward bucket).
    mode = _os.environ.get('ALLOC_CONVICTION_MODE', 'confidence').lower()
    if mode == 'confidence':
        conviction = conf_n
    elif mode == 'score':
        conviction = sc_n
    elif mode == 'blend':
        conviction = 0.6 * conf_n + 0.4 * sc_n
    else:  # 'max' — hangisi gucluyse o basta
        conviction = max(conf_n, sc_n)

    # 2) R/R IKINCIL tiebreaker
    rr_tilt = _f('ALLOC_RR_TILT', 0.15)
    q = conviction * (1.0 - rr_tilt) + rr_n * rr_tilt

    # 3) Cok dusuk R/R → geri plana at
    if float(rr or 0) < _f('ALLOC_RR_FLOOR', 1.3):
        q *= _f('ALLOC_RR_PENALTY', 0.6)

    q = _c01(q)
    floor_pct, ceil_pct = _f('ALLOC_FLOOR_PCT', 10.0), _f('ALLOC_CEIL_PCT', 25.0)
    return floor_pct + (ceil_pct - floor_pct) * q, q


def position_pct_for(rr, score, confidence):
    """Scanner'in cagirdigi TEK giris noktasi -> (pos_pct, q, mode_label).

    Mod secimini (park mi, kalite mi) burada kapsuller; scanner'in bilmesi gerekmez.
    """
    if _i('ALLOC_QUALITY_ENABLED', 0) == 1:
        pct, q = quality_position_pct(rr, score, confidence)
        return pct, q, f"kalite-pay q={q:.2f}"
    return _f('AUTO_MAX_POSITION_PCT', 20.0), 0.0, "duz-tavan (kalite PARK)"
