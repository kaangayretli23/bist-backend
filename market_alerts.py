"""
market_alerts.py — Piyasa-geneli erken uyarı katmanı (P0/P1/P2).

Açık pozisyon SL/TP monitöründen (realtime_monitor.py) AYRI ve ONA EK bir katman.
2026-07-08 THYAO olayı sonrası eklendi: hisse sabit SL çizgisini GEÇMEDEN sert düştü,
tek koruma sabit-SL fiyat geçişi olduğu için hiçbir uyarı gitmedi. Bu modül o boşluğu kapatır.

  P0-a) HIZ ALARMI: BIST100+BIST30 evreninde bir hisse son X dk'da ≥%Y sert düşerse Telegram uyar.
  P0-b) ENDEKS DEVRE KESİCİ: XU100/XU030 gün içi -%L1/-%L2 eşiğini geçerse Telegram uyar.
  P1)   BAYAT-FİYAT KORUMASI: price_too_stale() — pozisyon monitörü canlı fiyat çok eskiyse
        SL'i donmuş fiyatla kıyaslamasın, "korumasızsın" uyarısı göndersin.
  P2)   is_volatile_mode() — volatil seansta polling hızlansın (realtime_prices._loop kullanır).

GÜVENLİK: Hepsi READ-ONLY + yalnız Telegram UYARI. Otomatik alım/satım YOK (kullanıcı kuralı).
Fiyat örneklemesi batch _stock_cache'ten okunur → network YOK, sıcak yolu yavaşlatmaz.
"""
import os
import math
import time
import threading
from collections import deque


def _env_float(k, d):
    try:
        return float(os.environ.get(k, d))
    except Exception:
        return d


def _env_int(k, d):
    try:
        return int(os.environ.get(k, d))
    except Exception:
        return d


# ── Konfig (env ile ayarlanır) ──
_ENABLED           = os.environ.get('MARKET_ALERTS_ENABLED', '1') == '1'
_VEL_WINDOW_SEC    = _env_int('MARKET_VELOCITY_WINDOW_MIN', 15) * 60
_VEL_DROP_PCT      = _env_float('MARKET_VELOCITY_DROP_PCT', 4.0)     # pencerede ≥%4 düşüş → uyar
_VEL_COOLDOWN_SEC  = _env_int('MARKET_VELOCITY_COOLDOWN_MIN', 30) * 60
_VEL_MIN_COVER_SEC = _env_int('MARKET_VELOCITY_MIN_COVER_MIN', 5) * 60   # min geçmiş (restart false-fire önle)
# Endeks (XU100/XU030) iki-yönlü nabız: her _IDX_STEP_PCT puan hareket → bildir (düşüş+çıkış).
# L1/L2 sadece düşüşte ŞİDDET işareti (🔴/🔴🔴) + volatil-mod tetiği için kullanılır.
_IDX_STEP_PCT      = _env_float('INDEX_STEP_PCT', 0.5)
_IDX_MOVE_COOLDOWN_SEC = _env_int('INDEX_MOVE_COOLDOWN_SEC', 90)
_IDX_L1_PCT        = _env_float('INDEX_CIRCUIT_L1_PCT', 2.0)
_IDX_L2_PCT        = _env_float('INDEX_CIRCUIT_L2_PCT', 3.0)
_VOLATILE_HOLD_SEC = _env_int('MARKET_VOLATILE_HOLD_MIN', 5) * 60

# P1 bayat-fiyat koruması
MONITOR_MAX_PRICE_AGE_SEC = _env_int('MONITOR_MAX_PRICE_AGE_SEC', 180)
_STALE_WARN_COOLDOWN_SEC  = _env_int('MONITOR_STALE_WARN_COOLDOWN_MIN', 30) * 60

# P0-c portföy adım-alarmı: açık pozisyonda günlük değişim her N puan DÜŞÜNCE bilgi ver
_POS_STEP_ENABLED      = os.environ.get('PORTFOLIO_STEP_ALERTS_ENABLED', '1') == '1'
_POS_STEP_PCT          = _env_float('PORTFOLIO_STEP_PCT', 1.0)          # her %1 (puan) düşüş
_POS_STEP_COOLDOWN_SEC = _env_int('PORTFOLIO_STEP_COOLDOWN_SEC', 20)    # aynı poz tekrar-uyarı min aralık

# ── Durum ──
_hist: dict = {}          # sym → deque[(ts, price)]
_hist_lock = threading.Lock()
_vel_cooldown: dict = {}  # sym → last_alert_ts
_idx_step_ref: dict = {}  # code → {'level': int, 'ts': last_notify_ts}
_pos_step_ref: dict = {}  # pos_key → {'ref': int_level, 'ts': last_notify_ts}
_stale_warn_cooldown: dict = {}  # sym → last_warn_ts
_last_volatile_ts = 0.0
_state_lock = threading.Lock()

# Buffer başına maksimum örnek (30sn tick × 15dk ≈ 30; güvenli üst sınır)
_MAX_SAMPLES = 240


def _open_position_syms() -> set:
    """Açık pozisyon sembolleri — hisse bazlı uyarılar (hız + adım) yalnız bunlar için gider."""
    try:
        from realtime_prices import _get_open_positions
        return {p['symbol'] for p in _get_open_positions()}
    except Exception:
        return set()


# =====================================================================
# ACİLİYET KADEMESİ — bildirimin görsel şiddetini duruma göre tırmandırır
# =====================================================================
def _urgency(sl_dist_pct, move_pct):
    """Durumun ciddiyetine göre (baş_emoji, etiket, alt_satır) döner.

    Birincil ölçü: SL'e uzaklık (mesafe %). İkincil: düşüşün büyüklüğü (move_pct, pozitif).
    Kademe: 📉 BİLGİ (uzak/hafif) → ⚠️ DİKKAT → 🚨🔴 ACİL → 🆘🔴🔴 KRİTİK (SL altı).
    Böylece tek bakışta 'sıradan mı, acil mi' anlaşılır.
    """
    sd = sl_dist_pct
    mv = abs(move_pct or 0)
    if sd is not None and sd <= 0:
        return ('🆘🔴🔴', 'KRİTİK', '‼️ FİYAT SL ALTINDA — DERHAL KARAR VER ‼️')
    if (sd is not None and sd <= 1.5) or mv >= 6:
        return ('🚨🔴', 'ACİL', '‼️ SL çok yakın — HEMEN GÖZDEN GEÇİR')
    if (sd is not None and sd <= 3) or mv >= 3:
        return ('⚠️', 'DİKKAT', '⚠️ Zayıflıyor — yakından izle')
    return ('📉', 'BİLGİ', None)


# =====================================================================
# FİYAT ÖRNEKLEME (rolling buffer, network yok)
# =====================================================================
def record_prices() -> None:
    """Açık pozisyon sembollerinin anlık fiyatını rolling buffer'a yaz (hız alarmı portföy-özel).
    Batch cache → network YOK."""
    now = time.time()
    try:
        from config import _cget, _stock_cache
    except Exception:
        return
    cutoff = now - _VEL_WINDOW_SEC
    syms = _open_position_syms()
    # Kapanan pozisyonların buffer'ını temizle (RAM sızıntısı önle)
    with _hist_lock:
        for k in list(_hist.keys()):
            if k not in syms:
                _hist.pop(k, None)
        for sym in syms:
            st = _cget(_stock_cache, sym)
            if not st:
                continue
            price = st.get('price')
            if not price or price <= 0:
                continue
            dq = _hist.get(sym)
            if dq is None:
                dq = deque(maxlen=_MAX_SAMPLES)
                _hist[sym] = dq
            dq.append((now, float(price)))
            while dq and dq[0][0] < cutoff:
                dq.popleft()


def _velocity(sym):
    """(drop_pct, ref_price, cur_price, cover_sec) — pencere içi en eski→son değişim. Yetersizse None."""
    with _hist_lock:
        dq = _hist.get(sym)
        if not dq or len(dq) < 2:
            return None
        oldest_ts, oldest_p = dq[0]
        newest_ts, newest_p = dq[-1]
    cover = newest_ts - oldest_ts
    if cover < _VEL_MIN_COVER_SEC or oldest_p <= 0:
        return None
    pct = (newest_p - oldest_p) / oldest_p * 100
    return (pct, oldest_p, newest_p, cover)


# =====================================================================
# P0-a: HIZ ALARMI
# =====================================================================
def check_velocity_alerts() -> list:
    """SADECE AÇIK POZİSYONLARDA sert/hızlı düşüşü yakala → uyarı (cooldown'lu, SL mesafeli).
    Portföyde olmayan hisseler için bildirim GÖNDERMEZ (kullanıcı isteği)."""
    msgs = []
    now = time.time()
    try:
        from realtime_prices import _get_open_positions
        positions = {p['symbol']: p for p in _get_open_positions()}
    except Exception:
        return msgs

    for sym, pos in positions.items():
        v = _velocity(sym)
        if not v:
            continue
        pct, ref_p, cur_p, cover = v
        if pct > -_VEL_DROP_PCT:      # yeterince sert düşmedi
            continue
        with _state_lock:
            if now - _vel_cooldown.get(sym, 0) < _VEL_COOLDOWN_SEC:
                continue
            _vel_cooldown[sym] = now
        mins = max(1, int(cover // 60))
        entry = float(pos['entry_price'] or 0)
        sl = float(pos['stop_loss'] or 0)
        sl_dist = ((cur_p - sl) / sl * 100) if sl > 0 else None
        he, hl, tail = _urgency(sl_dist, pct)   # pct negatif → büyüklüğü _urgency içinde alınır
        line = (f"⚡️{he} <b>SERT DÜŞÜŞ · {hl} — {sym}</b> (portföy)\n"
                f"Son ~{mins}dk: <b>%{pct:.1f}</b>  ({ref_p:.2f} → {cur_p:.2f})")
        if entry:
            line += f"\nGiriş {entry:.2f}, P/L %{(cur_p - entry) / entry * 100:.1f}"
        if sl_dist is not None:
            line += f" | SL {sl:.2f} mesafe %{sl_dist:.1f}"
        line += f"\n{tail or '⚠️ <b>GÖZDEN GEÇİR</b>'}"
        msgs.append(line)
    return msgs


# =====================================================================
# P0-b: ENDEKS DEVRE KESİCİ
# =====================================================================
def check_index_moves() -> list:
    """XU100/XU030 iki-yönlü nabız: gün içi değişim her _IDX_STEP_PCT puan hareket edince bildir.

    DÜŞÜŞ ve ÇIKIŞ ikisini de belirtir (kullanıcı isteği). Düşüşte -%L1/-%L2'yi geçince
    ayrıca 🔴/🔴🔴 şiddet işareti + volatil-mod tetiği. Sınır-titremesini önlemek için cooldown.
    """
    msgs = []
    now = time.time()
    try:
        from config import _get_indices
        idx = _get_indices()
    except Exception:
        return msgs
    step = _IDX_STEP_PCT if _IDX_STEP_PCT > 0 else 0.5
    n_open = 0
    try:
        from realtime_prices import _get_open_positions
        n_open = len(_get_open_positions())
    except Exception:
        pass

    for code, label in (('XU100', 'BIST 100'), ('XU030', 'BIST 30')):
        d = idx.get(code)
        if not d:
            continue
        chg = d.get('changePct')
        if chg is None:
            p, pc = d.get('price'), d.get('prevClose')
            chg = (p - pc) / pc * 100 if (p and pc and pc > 0) else None
        if chg is None:
            continue
        chg = float(chg)
        level = math.floor(chg / step)
        rec = _idx_step_ref.get(code)
        if rec is None:                       # ilk görüş → baseline, bildirme
            _idx_step_ref[code] = {'level': level, 'ts': 0.0}
            continue
        if level == rec['level']:             # adım değişmedi
            continue
        if now - rec['ts'] < _IDX_MOVE_COOLDOWN_SEC:   # sınır titremesi koruması
            rec['level'] = level
            continue
        down = level < rec['level']
        arrow = '🔻' if down else '🔺'
        sev = ''
        crit = False
        if down and chg <= -_IDX_L2_PCT:
            sev = ' 🔴🔴 🚨'
            crit = True
        elif down and chg <= -_IDX_L1_PCT:
            sev = ' 🔴'
        head = '🆘🔴🔴 <b>PANİK SATIŞI</b> · ' if crit else ''
        line = f"{head}{arrow} <b>{label}</b> gün içi: <b>%{chg:+.1f}</b>{sev}"
        if sev:   # belirgin düşüş → pozisyon hatırlat + volatil mod
            urgent = '‼️ ' if crit else ''
            line += f"\n{urgent}Piyasa geneli satış — {n_open} açık pozisyonun var, gözden geçir."
            _mark_volatile()
        msgs.append(line)
        rec['level'] = level
        rec['ts'] = now
    return msgs


# =====================================================================
# P0-c: PORTFÖY ADIM-ALARMI (açık pozisyona özel, ince taneli)
# =====================================================================
def check_position_step_alerts() -> list:
    """Açık pozisyonlar: günlük değişim (changePct) her _POS_STEP_PCT puan DÜŞÜNCE bilgi ver.

    Örn +%8 → +%7 (hisse hâlâ kârda olsa bile) → bilgi. Yükselişte baseline'ı sessizce
    yukarı çeker (re-arm) — böylece +%7 → +%9 → +%8 tekrar uyarır. Negatifte de simetrik
    (-%3 → -%4 uyarır). Hız alarmından (sert çöküş) AYRI; bu ince-taneli portföy takibi.
    """
    if not _POS_STEP_ENABLED:
        return []
    msgs = []
    now = time.time()
    try:
        from config import _cget, _stock_cache
        from realtime_prices import _get_open_positions
        positions = _get_open_positions()
    except Exception:
        return []
    step = _POS_STEP_PCT if _POS_STEP_PCT > 0 else 1.0
    live_keys = set()
    for p in positions:
        sym = p['symbol']
        uid = p['user_id']
        key = f"{uid}_{sym}"
        live_keys.add(key)
        st = _cget(_stock_cache, sym)
        if not st:
            continue
        chg = st.get('changePct')
        if chg is None:
            continue
        chg = float(chg)
        cur_level = math.floor(chg / step)   # kaçıncı adım (aşağı yuvarlanmış)
        rec = _pos_step_ref.get(key)
        if rec is None:                       # ilk görüş → baseline kur, uyarma
            _pos_step_ref[key] = {'ref': cur_level, 'ts': 0.0}
            continue
        ref_level = rec['ref']
        if cur_level >= ref_level:            # sabit veya yükseliş → baseline'ı takip et, uyarma
            rec['ref'] = cur_level
            continue
        # ---- DÜŞÜŞ ADIMI ----
        if now - rec['ts'] < _POS_STEP_COOLDOWN_SEC:
            rec['ref'] = cur_level            # cooldown'da: uyarma ama baseline'ı indir
            continue
        prev_chg = ref_level * step
        drop_pts = (ref_level - cur_level) * step
        entry = float(p['entry_price'] or 0)
        cur_price = st.get('price')
        sl = float(p['stop_loss'] or 0)
        sl_dist = ((cur_price - sl) / sl * 100) if (sl > 0 and cur_price) else None
        he, hl, tail = _urgency(sl_dist, (-chg if chg < 0 else 0.0))
        line = (f"{he} <b>{hl} — {sym}</b>\n"
                f"Günlük değişim: %{prev_chg:+.0f} → <b>%{chg:+.1f}</b>  (−{drop_pts:.0f} puan)")
        if entry and cur_price:
            line += f"\nP/L: %{(cur_price - entry) / entry * 100:+.1f}"
        if sl_dist is not None:
            line += f" | SL {sl:.2f} mesafe %{sl_dist:.1f}"
        if tail:
            line += f"\n{tail}"
        msgs.append(line)
        rec['ref'] = cur_level
        rec['ts'] = now
    # Kapanan pozisyonların state'ini temizle (RAM sızıntısı önle)
    for k in list(_pos_step_ref.keys()):
        if k not in live_keys:
            _pos_step_ref.pop(k, None)
    return msgs


# =====================================================================
# ORKESTRASYON — realtime_prices._loop her tick çağırır
# =====================================================================
def check_market_alerts_once() -> None:
    """Fiyat örnekle + hız/endeks uyarılarını üret ve Telegram'a gönder. Piyasa kapalıysa no-op."""
    if not _ENABLED:
        return
    try:
        from auto_trader_engine import _is_market_open
        if not _is_market_open():
            return
    except Exception:
        pass

    record_prices()

    msgs = []
    try:
        vel = check_velocity_alerts()
        if vel:
            _mark_volatile()   # portföyde sert düşüş → volatil mod
        msgs += vel
    except Exception as e:
        print(f"[MARKET-ALERTS] velocity hata: {e}")
    try:
        step = check_position_step_alerts()
        if step:
            _mark_volatile()   # portföyde düşüş adımı → volatil mod
        msgs += step
    except Exception as e:
        print(f"[MARKET-ALERTS] step hata: {e}")
    try:
        msgs += check_index_moves()   # volatil mod'u kendi içinde (yalnız belirgin düşüşte) tetikler
    except Exception as e:
        print(f"[MARKET-ALERTS] index hata: {e}")

    if msgs:
        try:
            from routes_telegram import send_telegram
            for m in msgs:
                try:
                    send_telegram(m)
                except Exception:
                    pass
        except Exception:
            pass


# =====================================================================
# P2: VOLATİL MOD (adaptif polling)
# =====================================================================
def _mark_volatile() -> None:
    global _last_volatile_ts
    with _state_lock:
        _last_volatile_ts = time.time()


def is_volatile_mode() -> bool:
    """Son uyarıdan bu yana _VOLATILE_HOLD_SEC geçmediyse True → polling hızlansın."""
    with _state_lock:
        return (time.time() - _last_volatile_ts) < _VOLATILE_HOLD_SEC


# =====================================================================
# P1: BAYAT-FİYAT KORUMASI (pozisyon monitörü kullanır)
# =====================================================================
def price_too_stale(sym: str):
    """(too_stale_bool, age_sec) — canlı fiyat > eşik yaşındaysa (veya hiç yoksa) True.

    Pozisyon monitörü bunu SL/TP kıyaslamasından ÖNCE çağırır: True ise donmuş/eski fiyatla
    karar verMEZ, kullanıcıyı 'korumasızsın' diye uyarır. Feed çöküş anında donarsa sessiz
    körlüğü sesli uyarıya çevirir (THYAO olayının en kötü-hal versiyonu).
    """
    try:
        from realtime_prices import get_quote
        q = get_quote(sym)
    except Exception:
        q = None
    if not q or not q.get('ts'):
        return (True, None)
    age = time.time() - float(q['ts'])
    return (age > MONITOR_MAX_PRICE_AGE_SEC, age)


def should_warn_stale(sym: str) -> bool:
    """Cooldown'lu: bu sembol için bayat-fiyat uyarısı şimdi gönderilsin mi?"""
    now = time.time()
    with _state_lock:
        if now - _stale_warn_cooldown.get(sym, 0) < _STALE_WARN_COOLDOWN_SEC:
            return False
        _stale_warn_cooldown[sym] = now
    return True
