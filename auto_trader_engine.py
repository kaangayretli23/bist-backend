"""
BIST Pro - Auto Trading Engine Orkestratörü
İnce orkestratör: her kullanıcı için SL/TP kontrolü → Plan Merkezi → Sinyal taraması.

Alt modüller (600 satır kuralı):
  auto_trader_positions._step1_manage_positions — açık pozisyon SL/TP/trailing
  auto_trader_plan._step2a_plan_positions       — Plan Merkezi kilitli planları
  auto_trader_scanner._step2b_scan_signals      — sinyal tabanlı tarama
"""
import traceback
from datetime import datetime, timezone, timedelta
from auto_trader_risk import _sl_cooldown_load_from_db

# Alt adımlar — ayrıştırılmış modüllerden
from auto_trader_positions import _step1_manage_positions
from auto_trader_plan import _step2a_plan_positions
from auto_trader_scanner import _step2b_scan_signals

# BIST seans saatleri (UTC+3)
_TZ_TR = timezone(timedelta(hours=3))
MARKET_OPEN_H  = 10
MARKET_CLOSE_H = 18  # 18:00'de kapanır

# Orkestratör içinde kullanılan SL-cooldown DB yükleme bayrağı
_sl_cooldown_loaded = False


def _is_market_open() -> bool:
    """BIST şu an açık mı? Hafta içi 10:00–18:00 (TR saati)"""
    now = datetime.now(_TZ_TR)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN_H <= now.hour < MARKET_CLOSE_H


def _auto_engine_cycle():
    """Ana oto-trade döngüsü. Scheduler tarafından periyodik olarak çağrılır."""
    global _sl_cooldown_loaded
    if not _is_market_open():
        return

    # Lazy import — Flask startup sırasında circular import riskini önler
    from config import get_db, _cget, _stock_cache
    from auto_trader import (
        _auto_trade_lock, _auto_get_config,
        _auto_get_open_positions, _auto_get_daily_trade_count,
    )

    with _auto_trade_lock:
        # SL cooldown ilk yükleme — lock altında thread-safe
        if not _sl_cooldown_loaded:
            _sl_cooldown_load_from_db()
            _sl_cooldown_loaded = True
        try:
            db = get_db()
            users = db.execute(
                "SELECT user_id FROM auto_config WHERE enabled=1"
            ).fetchall()
            db.close()

            for user_row in users:
                uid = user_row['user_id']
                cfg = _auto_get_config(uid)
                if not cfg or not cfg.get('enabled'):
                    continue

                # ADIM 1: açık pozisyonları SL/TP/trailing açısından yönet
                positions = _auto_get_open_positions(uid)
                _step1_manage_positions(uid, cfg, positions)

                # ADIM 2 ön kontrolleri
                open_positions = _auto_get_open_positions(uid)
                open_symbols = {p['symbol'] for p in open_positions}

                if len(open_positions) >= cfg['maxPositions']:
                    continue

                daily_trades = _auto_get_daily_trade_count(uid)
                if daily_trades >= cfg['maxDailyTrades']:
                    continue

                allowed = set(s.strip() for s in cfg['allowedSymbols'].split(',') if s.strip()) if cfg['allowedSymbols'] else set()
                blocked = set(s.strip() for s in cfg['blockedSymbols'].split(',') if s.strip()) if cfg['blockedSymbols'] else set()

                slots = cfg['maxPositions'] - len(open_positions)
                daily_remaining = cfg['maxDailyTrades'] - daily_trades

                # İlk 15 dk (10:00-10:15) ve son 30 dk (17:30-18:00) yeni pozisyon açma
                _now_tr = datetime.now(_TZ_TR)
                _session_minute = _now_tr.hour * 60 + _now_tr.minute
                _open_cutoff  = MARKET_OPEN_H  * 60 + 15   # 10:15
                _close_cutoff = MARKET_CLOSE_H * 60 - 30   # 17:30
                if _session_minute < _open_cutoff or _session_minute >= _close_cutoff:
                    print(f"[AUTO-TRADE] Seans penceresi dışı ({_now_tr.strftime('%H:%M')} TR) — yeni pozisyon açılmıyor")
                    continue

                # Günlük zarar limiti: gerçekleşmiş + açık pozisyonların mark-to-market zararı
                try:
                    _db_chk = get_db()
                    _today_start = datetime.now(_TZ_TR).strftime('%Y-%m-%d') + ' 00:00:00'
                    _today_pnl_row = _db_chk.execute(
                        "SELECT COALESCE(SUM(pnl), 0) as total_pnl FROM auto_positions "
                        "WHERE user_id=? AND status='closed' AND closed_at>=?",
                        (uid, _today_start)
                    ).fetchone()
                    _db_chk.close()
                    _today_pnl = float(_today_pnl_row['total_pnl'] if _today_pnl_row else 0)
                    # Açık pozisyonların unrealized PnL'ini ekle
                    _unrealized = 0.0
                    for _p in open_positions:
                        _cp = 0.0
                        try:
                            from realtime_prices import get_price as _rt_p
                            _rp = _rt_p(_p['symbol'])
                            if _rp and _rp > 0:
                                _cp = _rp
                        except Exception:
                            pass
                        if _cp <= 0:
                            _stk = _cget(_stock_cache, _p['symbol'])
                            _cp = float(_stk.get('price', 0)) if _stk else 0
                        if _cp > 0:
                            _unrealized += (_cp - _p['entryPrice']) * _p['quantity']
                    _total_pnl = _today_pnl + _unrealized
                    _daily_loss_limit = cfg['capital'] * 0.05
                    if _total_pnl < -_daily_loss_limit:
                        print(f"[AUTO-TRADE] Günlük zarar limiti aşıldı "
                              f"(gerçekleşen={_today_pnl:.0f} + açık={_unrealized:.0f} = {_total_pnl:.0f} TL, "
                              f"limit={_daily_loss_limit:.0f} TL) — bugün yeni alım yok")
                        continue
                except Exception as _dloss_err:
                    print(f"[AUTO-TRADE] Günlük zarar kontrolü hatası: {_dloss_err}")

                # ADIM 2a: Plan Merkezi kilitli planları
                open_positions, open_symbols, slots, daily_remaining = _step2a_plan_positions(
                    uid, cfg, slots, daily_remaining, open_positions, open_symbols, allowed, blocked
                )

                # ADIM 2b: Sinyal taraması
                _step2b_scan_signals(
                    uid, cfg, slots, daily_remaining, open_positions, open_symbols, allowed, blocked
                )

            print(f"[AUTO-TRADE] Dongu tamamlandi: {len(users)} kullanici tarandi")
        except Exception as e:
            print(f"[AUTO-TRADE] Engine hatasi: {e}")
            traceback.print_exc()
