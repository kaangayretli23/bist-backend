# -*- coding: utf-8 -*-
"""
BIST Pro — CIKIS DANISMANI (FAZ 3, adim 1).

Sistemin kanitlanmis tek tarafi CIKIS. Bu modul acik pozisyona bakip
"kes" / "kar al" onerisi uretir. EMIR VERMEZ, yalnizca uyarir — nihai karar kullanicida.

═══ ESIKLER TAHMINLE DEGIL VERIDEN CIKARILDI ═══
Kaynak: signal_outcomes x signal_log, 4.728 AL sinyalinin 1/3/5/10/20 gunluk tam yolu
(2026-04..06). Ayni sinyalin farkli ufuklardaki getirisi yan yana konarak
"su anda X% zarardaysa sonra ne oluyor?" kosullu beklentisi olculdu.

KAYBEDEN (kesme) — olculen toparlama oranlari:
    3g'de <=-3%  -> 20g'de artida olma %16.4  (n=1238), ort getiri -5.98%
    3g'de <=-5%  -> %11.7 (n=685),  ort -8.00%
    3g'de <=-10% -> % 1.8 (n=113),  ort -14.60%
    1g'de <=-5%  -> 10g'de %10.0 (n=170)
  Yani zarar derinlestikce toparlama sansi MONOTON dusuyor; tutmak zarari BUYUTUYOR.

  ⭐ REJIM TESTI (en kritik dogrulama): kural yalnizca dusus rejiminde calismiyor.
     referans POZITIF aylarda: referans +1.21% / kaybeden -3.12% -> fark -4.33%
     referans NEGATIF aylarda: referans -3.40% / kaybeden -7.69% -> fark -4.29%
     Neredeyse AYNI fark. Rejim artefakti degil, gercek devam kalibi.

KAZANAN (kar alma) — erken kar GERI VERILIYOR:
    3g'de +0..2%  -> 3g'den 20g'ye EK getiri +0.48%
    3g'de +2..5%  -> -1.43%
    3g'de +5..10% -> -3.46%
    3g'de +10%+   -> -4.73%
  Erken kazanc ne kadar buyukse geri verme o kadar fazla → "kazanani birak" BURADA
  CALISMIYOR. Bu, kullanicinin gercek edge'iyle (hizli cikis) birebir ortusuyor.

⚠️ DURUSTLUK NOTU: veri 3 ay ve tek rejim ailesi (2026-04..06). Kaybeden kurali
   her iki yonde de tutarli (guclu). Kazanan kurali YON olarak tutarli ama BUYUKLUK
   rejime bagli (-1.87% / -5.11% / -7.87% aylik) → daha zayif, 'tavsiye' seviyesinde.
   Gercek bir BOGA piyasasi bu veride YOK; orada kazanan kurali bozulabilir.

═══ KAPSAM SINIRI — BU MODUL INTRADAY ICIN DEGIL ═══
Esikler GUNLUK bar verisinden cikti; en kisa olcum ufku 1 GUN. Kullanicinin tipik
islemi ise saatlik (gercek pozisyonlar 3-127 saat yasadi). Sonuc:
  • Bu modul COK-GUNLUK tutulan kaybedeni hedefler — asil kayip modu buydu
    (Haziran: KLSER/EBEBK/ARDYZ/LOGO gibi tutulup buyuyen zararlar).
  • GUN-ICI cokus zaten market_alerts.py'nin HIZ ALARMI ile kapsaniyor
    (15 dk'da >=%4 dusus, yalniz portfoydeki hisse). Ikisi BIRBIRINI TAMAMLAR,
    cakismaz.
  • EXIT_CUT_DEEP_DAYS'i 1.0'in ALTINA cekmek VERININ DISINA cikmaktir —
    1 gunden kisa ufuk icin olcumumuz YOK. Dusurursen bu bir TAHMIN olur, olcum degil.
"""
import os


def _f(key, default):
    try:
        return float(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _i(key, default):
    try:
        return int(os.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def exit_advice(pnl_pct, days_held):
    """Acik pozisyon -> cikis onerisi (veya None).

    pnl_pct   : giristen bu yana yuzde getiri (ornek: -3.4)
    days_held : pozisyonun acik oldugu gun sayisi (kesirli olabilir)

    Doner: {kind, severity, title, detail, evidence} veya None.
      kind='cut'  -> kaybeden, kesmeyi dusun
      kind='take' -> kazanan, kar almayi/trail sikmayi dusun
    """
    if pnl_pct is None or days_held is None:
        return None
    try:
        pnl_pct = float(pnl_pct); days_held = float(days_held)
    except (TypeError, ValueError):
        return None

    if not _i('EXIT_ADVISOR_ENABLED', 1):
        return None

    # ── KAYBEDEN kademeleri (esikler yukaridaki olcumden) ──
    # Derin zarar erken bile olsa guclu sinyal; sig zarar icin gun sarti aranir.
    deep = _f('EXIT_CUT_DEEP_PCT', -5.0)     # 1g'de -5% -> toparlama %10
    deep_days = _f('EXIT_CUT_DEEP_DAYS', 1.0)
    mid = _f('EXIT_CUT_MID_PCT', -3.0)      # 3g'de -3% -> toparlama %16
    mid_days = _f('EXIT_CUT_MID_DAYS', 3.0)

    if pnl_pct <= deep and days_held >= deep_days:
        return {
            'kind': 'cut', 'severity': 'high',
            'title': 'Kaybeden pozisyon — kesmeyi düşün',
            'detail': (f"%{pnl_pct:.1f} zararda, {days_held:.0f} gündür açık. "
                       f"Bu derinlikte toparlama geçmişte %10-12."),
            'evidence': 'n=685, 3g≤-5% olanların 20g ort. getirisi -8.0%',
        }
    if pnl_pct <= mid and days_held >= mid_days:
        return {
            'kind': 'cut', 'severity': 'medium',
            'title': 'Kaybeden pozisyon — toparlama olasılığı düşük',
            'detail': (f"%{pnl_pct:.1f} zararda, {days_held:.0f} gündür açık. "
                       f"Benzer durumların yalnızca %16'sı artıya döndü."),
            'evidence': 'n=1238, 3g≤-3% olanların 20g ort. getirisi -6.0%',
        }

    # ── KAZANAN kademesi (daha zayif kanit → 'tavsiye') ──
    take = _f('EXIT_TAKE_PCT', 5.0)
    take_days = _f('EXIT_TAKE_DAYS', 3.0)
    if pnl_pct >= take and days_held >= take_days:
        return {
            'kind': 'take', 'severity': 'low',
            'title': 'Kâr geri verilebilir — almayı/trail sıkmayı düşün',
            'detail': (f"%+{pnl_pct:.1f} kârda, {days_held:.0f} gündür açık. "
                       f"Bu seviyedeki kârlar geçmişte ort. %3.5 geri verildi."),
            'evidence': 'n=390, 3g +5..10% olanların 3g→20g ek getirisi -3.5% '
                        '(⚠ büyüklük rejime bağlı, yön tutarlı)',
        }
    return None


def advice_for_position(pos, current_price, now_ts=None):
    """auto_positions kaydi + guncel fiyat -> exit_advice(). Sure hesabini kapsuller."""
    import time
    from datetime import datetime
    try:
        entry = float(pos.get('entryPrice') or pos.get('entry_price') or 0)
        if entry <= 0 or not current_price:
            return None
        pnl_pct = (float(current_price) - entry) / entry * 100

        opened = pos.get('openedAt') or pos.get('opened_at')
        if not opened:
            return None
        try:
            dtv = datetime.fromisoformat(str(opened).replace('Z', '+00:00'))
            opened_ts = dtv.timestamp()
        except Exception:
            return None
        days = ((now_ts or time.time()) - opened_ts) / 86400.0
        return exit_advice(pnl_pct, days)
    except Exception as e:
        print(f"[EXIT-ADVISOR] hata: {e}")
        return None
