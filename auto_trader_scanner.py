"""
BIST Pro - Auto Trader: Sinyal Taraması (ADIM 2b)
Sinyal skoru yüksek adayları tarayıp kalan pozisyon slotlarını doldurur.
auto_trader_engine.py'dan ayrıştırıldı (600 satır kuralı).
"""
# Not: config, auto_trader, signals, indicators, signals_core, routes_telegram,
# realtime_prices, data_fetcher, trade_plans fonksiyon içinde lazy import edilir.
from auto_trader_risk import _sl_cooldown_check
from signals_market import REGIMES_BEARISH


def _step2b_scan_signals(uid, cfg, slots, daily_remaining, open_positions, open_symbols, allowed, blocked):
    """Sinyal skoru yüksek adayları tara ve kalan pozisyon slotlarını doldur."""
    from config import _cget_hist, _cset, _get_stocks, _hist_cache
    from auto_trader import _auto_open_position, _auto_log_trade, _auto_get_open_positions, _auto_get_daily_trade_count
    from trade_plans import calc_trade_plan

    if slots <= 0 or daily_remaining <= 0:
        return

    candidates = []
    stocks = _get_stocks()
    _on_demand_fetches = 0

    try:
        from indicators import calc_all_indicators
        from signals import calc_recommendation as _calc_rec
        _signals_ok = True
    except Exception as _sig_err:
        print(f"[AUTO-TRADE] Sinyal modulu yuklenemedi: {_sig_err}")
        _signals_ok = False

    _tf_map = {'daily': 'daily', 'swing': 'weekly', 'monthly': 'monthly'}

    for s in stocks:
        sym = s.get('code', '')
        if not sym or sym in open_symbols:
            continue
        if allowed and sym not in allowed:
            continue
        if sym in blocked:
            continue
        if _sl_cooldown_check(uid, sym, cfg.get('tradeStyle', 'swing')):
            continue
        if not _signals_ok:
            continue

        hist = _cget_hist(f"{sym}_1y")
        if hist is None or len(hist) < 30:
            if _on_demand_fetches >= 30:
                continue
            try:
                from data_fetcher import _fetch_hist_df
                _on_demand_fetches += 1
                hist = _fetch_hist_df(sym, '1y')
                if hist is not None and len(hist) >= 30:
                    _cset(_hist_cache, f"{sym}_1y", hist)
                else:
                    continue
            except Exception:
                continue

        live_price = float(s.get('price', 0))
        if live_price <= 0:
            live_price = float(hist['Close'].iloc[-1])
        if live_price <= 0:
            continue

        try:
            from signals_core import _splice_live_close
            hist_live = _splice_live_close(hist, live_price)
            indics = calc_all_indicators(hist_live, live_price)
            recs = _calc_rec(hist_live, indics, symbol=sym)
            tf_key = _tf_map.get(cfg['tradeStyle'], 'weekly')
            rec = recs.get(tf_key) or recs.get('weekly', {})
            signal = rec.get('action', '')
            score = float(rec.get('score', 0))
            confidence = float(rec.get('confidence', 0))
        except Exception as _rec_err:
            print(f"[AUTO-TRADE] {sym} sinyal hesaplama hatasi: {_rec_err}")
            continue

        if signal not in ('AL', 'GÜÇLÜ AL') or score < cfg['minScore'] or confidence < cfg['minConfidence']:
            continue

        # Piyasa rejimi: güçlü ayı piyasasında yeni alım yapma
        try:
            from signals import calc_market_regime as _cmr
            _reg = _cmr()
            if _reg.get('regime') in REGIMES_BEARISH and float(_reg.get('strength', 0)) > 60:
                print(f"[AUTO-TRADE] {sym} atlandi — ayi piyasasi (guc={_reg.get('strength')})")
                continue
        except Exception:
            pass

        # Hacim filtresi
        if len(hist) >= 20:
            try:
                vol_today = float(hist['Volume'].iloc[-1])
                vol_avg20 = float(hist['Volume'].iloc[-20:].mean())
                if vol_avg20 > 0 and vol_today < vol_avg20 * 0.5:
                    continue
            except Exception:
                pass

        # Gap down koruması
        if len(hist) >= 2:
            try:
                prev_close = float(hist['Close'].iloc[-2])
                today_open = float(hist['Open'].iloc[-1])
                if prev_close > 0 and (prev_close - today_open) / prev_close > 0.03:
                    continue
            except Exception:
                pass

        candidates.append({
            'symbol': sym, 'price': live_price,
            'score': score, 'confidence': confidence,
            'name': s.get('name', sym),
        })

    candidates.sort(key=lambda x: x['score'], reverse=True)
    slots = cfg['maxPositions'] - len(open_positions)
    daily_remaining = cfg['maxDailyTrades'] - _auto_get_daily_trade_count(uid)

    for cand in candidates[:min(slots, daily_remaining)]:
        sym = cand['symbol']
        price = cand['price']
        if price <= 0:
            continue

        stop_loss = round(price * (1 - cfg['stopLossPct'] / 100), 2)
        tp1 = round(price * (1 + cfg['takeProfitPct'] / 100), 2)
        tp2 = round(price * (1 + cfg['takeProfitPct'] * 1.5 / 100), 2)
        tp3 = round(price * (1 + cfg['takeProfitPct'] * 2 / 100), 2)
        trailing_sl = round(price * (1 - cfg['trailingPct'] / 100), 2) if cfg['trailingStop'] else 0

        # Trade planini kontrol et (daha iyi SL/TP varsa kullan)
        hist = _cget_hist(f"{sym}_1y")
        if hist is not None and len(hist) >= 20:
            try:
                plan = calc_trade_plan(hist, symbol=sym)
                tf_key = {'daily': 'daily', 'swing': 'weekly', 'monthly': 'monthly'}.get(cfg['tradeStyle'], 'weekly')
                tf_plan = plan.get(tf_key, {})
                buy_plan = tf_plan.get('buy', {})
                if buy_plan:
                    plan_sl = float(buy_plan.get('stopLoss', 0))
                    if 0 < plan_sl < price and plan_sl >= price * 0.85:
                        stop_loss = plan_sl
                    targets = buy_plan.get('targets', [])

                    def _tp_val(t, default):
                        try:
                            return float(t.get('price', default)) if isinstance(t, dict) else float(t)
                        except Exception:
                            return default

                    if len(targets) >= 1 and _tp_val(targets[0], 0) > price:
                        tp1 = _tp_val(targets[0], tp1)
                    if len(targets) >= 2 and _tp_val(targets[1], 0) > price:
                        tp2 = _tp_val(targets[1], tp2)
                    if len(targets) >= 3 and _tp_val(targets[2], 0) > price:
                        tp3 = _tp_val(targets[2], tp3)
            except Exception:
                pass

        risk_amount = cfg['capital'] * (cfg['riskPerTrade'] / 100)
        sl_distance = price - stop_loss
        if sl_distance <= 0:
            continue
        quantity = int(risk_amount / sl_distance)
        if quantity < 1:
            continue
        position_cost = quantity * price

        used_capital = sum(p['entryPrice'] * p['quantity'] for p in open_positions)
        if used_capital + position_cost > cfg['capital']:
            continue

        _tg_sent = False
        _already_pending = False
        try:
            from routes_telegram import send_trade_signal, _pending_signals, _pending_lock
            import os as _os
            if _os.environ.get('TELEGRAM_BOT_TOKEN') and _os.environ.get('TELEGRAM_CHAT_ID'):
                with _pending_lock:
                    _already_pending = any(
                        ps['symbol'] == sym and ps['uid'] == uid
                        for ps in _pending_signals.values()
                    )
                if not _already_pending:
                    _tg_sent = send_trade_signal(
                        uid, sym, price, quantity,
                        cand['score'], cand['confidence'],
                        stop_loss, tp1, tp2, tp3, trailing_sl
                    )
        except Exception as _tg_err:
            print(f"[AUTO-TRADE] Telegram sinyal hatası: {_tg_err}")

        if _already_pending or _tg_sent:
            try:
                from realtime_prices import subscribe as _rt_sub
                _rt_sub(sym)
            except Exception:
                pass
            open_symbols.add(sym)
        else:
            pos_id = _auto_open_position(uid, sym, price, quantity, stop_loss, tp1, tp2, tp3, trailing_sl)
            if pos_id:
                _auto_log_trade(uid, sym, 'BUY', price, quantity,
                               f"Skor={cand['score']:.1f}, Guven=%{cand['confidence']:.0f}, SL={stop_loss:.2f}, TP1={tp1:.2f}",
                               cand['score'], cand['confidence'], pos_id)
                open_positions = _auto_get_open_positions(uid)
                open_symbols.add(sym)
