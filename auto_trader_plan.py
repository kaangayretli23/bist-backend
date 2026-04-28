"""
BIST Pro - Auto Trader: Plan Merkezi Yürütücüsü (ADIM 2a)
Kilitli AL planlarını kontrol edip pozisyon açar.
auto_trader_engine.py'dan ayrıştırıldı (600 satır kuralı).
"""
import time
# Not: config, auto_trader, signals, indicators, signals_core, routes_telegram
# fonksiyon içinde lazy import edilir (circular/partial load önlemi).
from auto_trader_risk import _sl_cooldown_check, _reject_cooldown_check, _log_decision
from signals_market import REGIMES_BEARISH


def _step2a_plan_positions(uid, cfg, slots, daily_remaining, open_positions, open_symbols, allowed, blocked):
    """Plan Merkezi'ndeki kilitli AL planlarını kontrol edip pozisyon aç."""
    from config import (
        _stock_cache, _cget, _cget_hist,
        _plan_lock_cache, _plan_lock_cache_lock, PLAN_LOCK_CONFIG,
    )
    from auto_trader import _auto_open_position, _auto_log_trade, _auto_get_open_positions

    _tf_map_plan = {'daily': 'daily', 'swing': 'weekly', 'monthly': 'monthly'}
    plan_tf = _tf_map_plan.get(cfg.get('tradeStyle', 'swing'), 'weekly')

    try:
        with _plan_lock_cache_lock:
            locked_items = list(_plan_lock_cache.items())

        for lock_key, entry in locked_items:
            if slots <= 0 or daily_remaining <= 0:
                break
            # key format: "SYMBOL_timeframe"
            parts = lock_key.rsplit('_', 1)
            if len(parts) != 2:
                continue
            sym, tf = parts[0], parts[1]

            if tf != plan_tf:
                continue
            if entry.get('signal', 'BEKLE') not in ('AL', 'GÜÇLÜ AL'):
                continue
            if sym in open_symbols:
                continue
            if allowed and sym not in allowed:
                continue
            if sym in blocked:
                _log_decision(uid, sym, 'SKIP', 'blocked', tf=tf)
                continue
            if _sl_cooldown_check(uid, sym, cfg.get('tradeStyle', 'swing')):
                _log_decision(uid, sym, 'SKIP', 'sl_cooldown', tf=tf)
                continue
            if _reject_cooldown_check(uid, sym):
                _log_decision(uid, sym, 'SKIP', 'reject_cooldown', tf=tf)
                continue
            # Sektor cap: ayni sektorden max N pozisyon
            try:
                from auto_trader_sectors import sector_full
                _full, _sec, _cur = sector_full(sym, open_positions, int(cfg.get('maxPerSector', 2)))
                if _full:
                    _log_decision(uid, sym, 'SKIP', 'sector_cap',
                                  detail=f"[PLAN] {_sec} dolu ({_cur}/{int(cfg.get('maxPerSector', 2))})",
                                  tf=tf)
                    continue
            except Exception:
                pass

            # Plan suresini kontrol et
            lock_cfg = PLAN_LOCK_CONFIG.get(tf, PLAN_LOCK_CONFIG.get('daily', {}))
            age = time.time() - entry.get('locked_at', 0)
            if age > lock_cfg.get('max_lock', 86400):
                _log_decision(uid, sym, 'SKIP', 'plan_expired',
                              detail=f"age={age/3600:.1f}h > {lock_cfg.get('max_lock', 86400)/3600:.0f}h",
                              tf=tf)
                continue  # Plan suresi dolmus

            # Guncel fiyat al
            stock = _cget(_stock_cache, sym)
            cur_price = float(stock.get('price', 0)) if stock else 0
            if cur_price <= 0:
                locked_price = entry.get('locked_price', 0)
                if locked_price > 0:
                    cur_price = locked_price
                else:
                    continue

            # KORUMA 1: Fiyat sapma kontrolu
            _deviation_limits = {'daily': 0.03, 'weekly': 0.05, 'monthly': 0.08}
            _dev_limit = _deviation_limits.get(tf, 0.05)
            locked_price = entry.get('locked_price', cur_price)
            if locked_price > 0:
                deviation = abs(cur_price - locked_price) / locked_price
                if deviation > _dev_limit:
                    print(f"[AUTO-TRADE] {sym} plan iptal — fiyat sapması %{deviation*100:.1f} "
                          f"(limit %{_dev_limit*100:.0f}, kilitli={locked_price:.2f}, güncel={cur_price:.2f})")
                    _log_decision(uid, sym, 'SKIP', 'deviation',
                                  detail=f"sapma=%{deviation*100:.1f} > limit=%{_dev_limit*100:.0f}",
                                  tf=tf, price=cur_price)
                    with _plan_lock_cache_lock:
                        _plan_lock_cache.pop(lock_key, None)
                    continue

            # Plan'dan SL/TP al
            tf_plan = entry.get('tf_plan', {})
            buy_plan = tf_plan.get('buy', {})

            stop_loss = round(cur_price * (1 - cfg['stopLossPct'] / 100), 2)
            plan_sl = float(buy_plan.get('stopLoss', 0)) if buy_plan else 0
            if 0 < plan_sl < cur_price and plan_sl >= cur_price * 0.85:
                stop_loss = plan_sl

            # KORUMA 2: SL geçerlilik kontrolu
            if stop_loss >= cur_price:
                print(f"[AUTO-TRADE] {sym} plan iptal — güncel fiyat ({cur_price:.2f}) SL'nin ({stop_loss:.2f}) altında")
                continue

            # KORUMA 3: Entry range kontrolü
            _entry_range = buy_plan.get('entryRange', {}) if buy_plan else {}
            _er_min = float(_entry_range.get('min', 0))
            _er_max = float(_entry_range.get('max', 0))
            if _er_min > 0 and _er_max > 0:
                if not (_er_min <= cur_price <= _er_max):
                    print(f"[AUTO-TRADE] {sym} plan iptal — fiyat ({cur_price:.2f}) "
                          f"giriş aralığı dışında [{_er_min:.2f}-{_er_max:.2f}]")
                    continue

            # KORUMA 4: Piyasa rejimi
            try:
                from signals import calc_market_regime
                _regime = calc_market_regime()
                if _regime.get('regime') in REGIMES_BEARISH and float(_regime.get('strength', 0)) > 60:
                    print(f"[AUTO-TRADE] {sym} plan ertelendi — güçlü ayı piyasası "
                          f"(güç={_regime.get('strength', 0):.0f})")
                    _log_decision(uid, sym, 'SKIP', 'regime',
                                  detail=f"{_regime.get('regime')}/{_regime.get('strength', 0):.0f}",
                                  tf=tf, price=cur_price)
                    continue
            except Exception:
                pass

            # KORUMA 5: Hacim filtresi
            try:
                hist_vol = _cget_hist(f"{sym}_1y")
                if hist_vol is not None and len(hist_vol) >= 20:
                    vol_today = float(hist_vol['Volume'].iloc[-1])
                    vol_avg20 = float(hist_vol['Volume'].iloc[-20:].mean())
                    if vol_avg20 > 0 and vol_today < vol_avg20 * 0.5:
                        print(f"[AUTO-TRADE] {sym} plan ertelendi — düşük hacim "
                              f"(bugün={vol_today:.0f}, ort20={vol_avg20:.0f})")
                        _log_decision(uid, sym, 'SKIP', 'volume',
                                      detail=f"bugun={vol_today:.0f} < ort20*0.5={vol_avg20*0.5:.0f}",
                                      tf=tf, price=cur_price)
                        continue
            except Exception:
                pass

            # KORUMA 6: RSI aşırı alım kontrolü
            try:
                from indicators import calc_rsi_single
                hist_chk = _cget_hist(f"{sym}_1y")
                if hist_chk is not None and len(hist_chk) >= 20:
                    closes = list(hist_chk['Close'].astype(float))
                    rsi_now = calc_rsi_single(closes)
                    if rsi_now is not None and rsi_now > 80:
                        print(f"[AUTO-TRADE] {sym} plan ertelendi — RSI={rsi_now:.1f} aşırı alım")
                        _log_decision(uid, sym, 'SKIP', 'rsi_overbought',
                                      detail=f"RSI={rsi_now:.1f} > 80",
                                      tf=tf, price=cur_price)
                        continue
            except Exception:
                pass

            # TP hesapla
            tp1 = round(cur_price * (1 + cfg['takeProfitPct'] / 100), 2)
            tp2 = round(cur_price * (1 + cfg['takeProfitPct'] * 1.5 / 100), 2)
            tp3 = round(cur_price * (1 + cfg['takeProfitPct'] * 2.0 / 100), 2)
            targets = buy_plan.get('targets', []) if buy_plan else []

            def _tp_val(t, default):
                try:
                    return float(t.get('price', default)) if isinstance(t, dict) else float(t)
                except Exception:
                    return default

            if len(targets) >= 1 and _tp_val(targets[0], 0) > cur_price:
                tp1 = _tp_val(targets[0], tp1)
            if len(targets) >= 2 and _tp_val(targets[1], 0) > cur_price:
                tp2 = _tp_val(targets[1], tp2)
            if len(targets) >= 3 and _tp_val(targets[2], 0) > cur_price:
                tp3 = _tp_val(targets[2], tp3)

            # TP monotonik kontrol (kademeli kar alma icin tp1<tp2<tp3 + min %0.5 spread)
            _tp_def_1 = round(cur_price * (1 + cfg['takeProfitPct'] / 100), 2)
            _tp_def_2 = round(cur_price * (1 + cfg['takeProfitPct'] * 1.5 / 100), 2)
            _tp_def_3 = round(cur_price * (1 + cfg['takeProfitPct'] * 2.0 / 100), 2)
            if (not (cur_price < tp1 < tp2 < tp3)
                    or (tp2 - tp1) / cur_price < 0.005
                    or (tp3 - tp2) / cur_price < 0.005):
                print(f"[AUTO-TRADE] [PLAN] {sym} TP siralama bozuk "
                      f"(tp1={tp1:.2f}, tp2={tp2:.2f}, tp3={tp3:.2f}) → default formul")
                tp1, tp2, tp3 = _tp_def_1, _tp_def_2, _tp_def_3

            trailing_sl = round(cur_price * (1 - cfg['trailingPct'] / 100), 2) if cfg['trailingStop'] else 0

            # Pozisyon buyuklugu
            sl_distance = cur_price - stop_loss
            if sl_distance <= 0:
                continue
            risk_amount = cfg['capital'] * (cfg['riskPerTrade'] / 100)
            quantity = int(risk_amount / sl_distance)
            if quantity < 1:
                continue
            position_cost = quantity * cur_price

            # Serbest sermayeye gore quantity'i kirp
            used_capital = sum(p['entryPrice'] * p['quantity'] for p in open_positions)
            free_capital = max(0, cfg['capital'] - used_capital)
            if position_cost > free_capital:
                affordable_qty = int(free_capital / cur_price) if cur_price > 0 else 0
                if affordable_qty < 1:
                    print(f"[AUTO-TRADE-PLAN] {sym} atlandi — serbest sermaye {free_capital:.0f} TL (1 lot {cur_price:.2f} TL)")
                    _log_decision(uid, sym, 'SKIP', 'budget',
                                  detail=f"[PLAN] serbest={free_capital:.0f} TL, 1 lot={cur_price:.2f} TL",
                                  tf=tf, price=cur_price)
                    continue
                print(f"[AUTO-TRADE-PLAN] {sym} adet kirpildi: {quantity} -> {affordable_qty} "
                      f"(serbest={free_capital:.0f} TL)")
                quantity = affordable_qty
                position_cost = quantity * cur_price

            # Gerçek skor/güven hesabı — hardcoded 10/100 yerine calc_recommendation
            plan_score = 0.0
            plan_conf = 0.0
            _score_ok = False
            try:
                from indicators import calc_all_indicators as _caii
                from signals_core import calc_recommendation as _crec, _splice_live_close as _splice
                hist_sig = _cget_hist(f"{sym}_1y")
                if hist_sig is not None and len(hist_sig) >= 30:
                    hist_live = _splice(hist_sig, cur_price)
                    _ind = _caii(hist_live, cur_price)
                    _rec = _crec(hist_live, _ind, symbol=sym)
                    _tf_key = {'daily': 'daily', 'swing': 'weekly', 'monthly': 'monthly'}.get(tf, 'weekly')
                    _tfr = _rec.get(_tf_key) or _rec.get('weekly', {})
                    plan_score = float(_tfr.get('score', 0))
                    plan_conf = float(_tfr.get('confidence', 0))
                    _score_ok = True
            except Exception as _ps_err:
                print(f"[AUTO-TRADE] {sym} plan skor hesabi hatasi: {_ps_err}")

            # Plan bayatlama filtresi: güncel composite zayıfsa planı atla
            # (plan eski bir AL sinyalinden kilitlendi ama piyasa artık onaylamıyor)
            _plan_min_conf = cfg.get('minConfidence', 60) * 0.5  # sinyal taramasının yarısı
            _plan_min_score = max(1.5, cfg.get('minScore', 3.0) * 0.5)
            if _score_ok and (plan_conf < _plan_min_conf or plan_score < _plan_min_score):
                print(f"[AUTO-TRADE] {sym} plan bayatlamis — atlandi "
                      f"(skor={plan_score:.1f} < {_plan_min_score:.1f} veya "
                      f"guven=%{plan_conf:.0f} < %{_plan_min_conf:.0f})")
                _log_decision(uid, sym, 'SKIP', 'plan_stale',
                              detail=f"sc={plan_score:.1f}/{_plan_min_score:.1f}, conf={plan_conf:.0f}/{_plan_min_conf:.0f}",
                              tf=tf, price=cur_price, score=plan_score, confidence=plan_conf)
                continue
            if not _score_ok:
                print(f"[AUTO-TRADE] {sym} plan skor hesaplanamadi — atlandi (hist cache bos olabilir)")
                _log_decision(uid, sym, 'SKIP', 'score_unavailable', tf=tf, price=cur_price)
                continue

            # Telegram bildirimi veya direkt ac (fail-safe: TG aktifken gonderim
            # basarisizsa pozisyon ACMA — kullanici onay bekliyordu).
            _tg_sent = False
            _already_pending = False
            _tg_configured = False
            try:
                from routes_telegram import send_trade_signal, _pending_signals, _pending_lock
                import os as _os
                _tg_configured = bool(_os.environ.get('TELEGRAM_BOT_TOKEN')
                                      and _os.environ.get('TELEGRAM_CHAT_ID'))
                if _tg_configured:
                    with _pending_lock:
                        _already_pending = any(
                            ps['symbol'] == sym and ps['uid'] == uid
                            for ps in _pending_signals.values()
                        )
                    if not _already_pending:
                        _tg_sent = send_trade_signal(
                            uid, sym, cur_price, quantity,
                            plan_score, plan_conf,
                            stop_loss, tp1, tp2, tp3, trailing_sl
                        )
            except Exception:
                pass

            if _already_pending or _tg_sent:
                open_symbols.add(sym)
                _log_decision(uid, sym, 'PENDING', 'telegram_approve',
                              detail=f"[PLAN] qty={quantity}, SL={stop_loss:.2f}, TP1={tp1:.2f}",
                              tf=tf, price=cur_price, score=plan_score, confidence=plan_conf)
            elif _tg_configured:
                _log_decision(uid, sym, 'SKIP', 'telegram_failed',
                              detail='[PLAN] Telegram onay sinyali gonderilemedi; pozisyon acilmadi',
                              tf=tf, price=cur_price, score=plan_score, confidence=plan_conf)
                continue
            else:
                pos_id = _auto_open_position(uid, sym, cur_price, quantity, stop_loss, tp1, tp2, tp3, trailing_sl)
                if pos_id:
                    _auto_log_trade(
                        uid, sym, 'BUY', cur_price, quantity,
                        f"[PLAN] Kilitli plan ({tf}), SL={stop_loss:.2f}, TP1={tp1:.2f}",
                        plan_score, plan_conf, pos_id
                    )
                    open_positions = _auto_get_open_positions(uid)
                    open_symbols.add(sym)
                    slots -= 1
                    daily_remaining -= 1
                    print(f"[AUTO-TRADE] {sym} Plan Merkezi planından pozisyon açıldı (fiyat={cur_price:.2f})")
                    _log_decision(uid, sym, 'BUY', 'opened',
                                  detail=f"[PLAN] qty={quantity}, SL={stop_loss:.2f}, TP1={tp1:.2f}",
                                  tf=tf, price=cur_price, score=plan_score, confidence=plan_conf)
    except Exception as _plan_err:
        print(f"[AUTO-TRADE] Plan Merkezi kontrolü hatası: {_plan_err}")

    return open_positions, open_symbols, slots, daily_remaining
