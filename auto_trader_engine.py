"""
BIST Pro - Auto Trading Engine Döngüsü
Sinyal tarama, pozisyon yönetimi, SL/TP/trailing kontrolü.
auto_trader.py'dan ayrıştırıldı (700 satır kuralı).

Alt fonksiyonlar:
  _step1_manage_positions  — açık pozisyonların SL/TP/trailing kontrolü
  _step2a_plan_positions   — Plan Merkezi kilitli planlarını yürüt
  _step2b_scan_signals     — sinyal taraması ile kalan slotları doldur
  _auto_engine_cycle       — ince orkestratör (her kullanıcı için yukarıdakileri çağırır)
"""
import traceback, time
from datetime import datetime, timezone, timedelta
from config import (
    get_db, sf, _cget, _cget_hist, _cset, _get_stocks, _stock_cache, _hist_cache,
)
# Risk helpers (SL cooldown + panic-sell ring buffer) — ayrı modülde
from auto_trader_risk import (
    _panic_track_and_check, _panic_clear,
    _sl_cooldown_block, _sl_cooldown_check, _sl_cooldown_load_from_db,
)

from trade_plans import calc_trade_plan

# auto_trader — lazy import (fonksiyon içinde), circular import/partial load önlemi

# BIST seans saatleri (UTC+3)
_TZ_TR = timezone(timedelta(hours=3))
MARKET_OPEN_H  = 10
MARKET_CLOSE_H = 18  # 18:00'de kapanır


def _is_market_open() -> bool:
    """BIST şu an açık mı? Hafta içi 10:00–18:00 (TR saati)"""
    now = datetime.now(_TZ_TR)
    if now.weekday() >= 5:
        return False
    return MARKET_OPEN_H <= now.hour < MARKET_CLOSE_H


# =====================================================================
# ADIM 1 — Açık pozisyonları SL/TP/trailing açısından kontrol et
# =====================================================================

def _step1_manage_positions(uid, cfg, positions):
    """Açık pozisyonları SL/TP/trailing açısından kontrol et."""
    from auto_trader import _auto_close_position, _auto_partial_close, _auto_log_trade, _auto_update_trailing, _auto_update_highest_price
    for pos in positions:
        sym = pos['symbol']
        # Gerçek zamanlı fiyatı önce dene, yoksa cache/fetch
        cur_price = 0.0
        try:
            from realtime_prices import get_price as _rt_get
            rt_price = _rt_get(sym)
            if rt_price and rt_price > 0:
                cur_price = rt_price
        except Exception:
            pass
        if cur_price <= 0:
            stock = _cget(_stock_cache, sym)
            cur_price = float(stock.get('price', 0)) if stock else 0
        if cur_price <= 0:
            try:
                from data_fetcher import _process_stock
                _, fresh = _process_stock(sym, retry_count=1)
                if fresh:
                    _cset(_stock_cache, sym, fresh)
                    cur_price = float(fresh.get('price', 0))
            except Exception:
                pass
        if cur_price <= 0:
            print(f"[AUTO-TRADE] {sym} fiyat alinamadi, SL/TP kontrolu atlandi")
            continue

        # Stop-Loss kontrolu
        sl = pos['stopLoss']
        if sl > 0 and cur_price <= sl:
            _auto_close_position(pos['id'], cur_price, f"Stop-Loss tetiklendi ({sl:.2f})")
            _auto_log_trade(uid, sym, 'SELL_SL', cur_price, pos['quantity'],
                           f"SL tetiklendi: {cur_price:.2f} <= {sl:.2f}", 0, 0, pos['id'])
            _sl_cooldown_block(uid, sym, cfg.get('tradeStyle', 'swing'))
            _panic_clear(pos['id'])
            continue

        # SL yaklasma uyarisi (%2 icinde)
        if sl > 0 and cur_price <= sl * 1.02:
            try:
                from routes_telegram import send_sl_warning
                send_sl_warning(sym, cur_price, sl)
            except Exception:
                pass

        # Take-Profit kontrolu — kademeli kâr alma
        # TP1: %50 sat, TP2: kalanın %50'si, TP3: tamamını kapat
        tp3 = pos['takeProfit3']
        tp2 = pos['takeProfit2']
        tp1 = pos['takeProfit1']
        qty = pos['quantity']
        if tp3 > 0 and cur_price >= tp3:
            _auto_close_position(pos['id'], cur_price, f"TP3 hedef ({tp3:.2f})")
            _auto_log_trade(uid, sym, 'SELL_TP3', cur_price, qty,
                           f"TP3: {cur_price:.2f} >= {tp3:.2f}", 0, 0, pos['id'])
            _panic_clear(pos['id'])
            continue
        elif tp2 > 0 and cur_price >= tp2:
            sell_qty = int(qty * 0.5)
            if sell_qty < 1:
                sell_qty = qty  # çok az kaldıysa hepsini sat
            _auto_partial_close(pos['id'], sell_qty, cur_price,
                               f"TP2 hedef ({tp2:.2f})", clear_tp_field='take_profit2')
            _auto_log_trade(uid, sym, 'SELL_TP2', cur_price, sell_qty,
                           f"TP2 kısmi: {cur_price:.2f} >= {tp2:.2f}", 0, 0, pos['id'])
        elif tp1 > 0 and cur_price >= tp1:
            sell_qty = int(qty * 0.5)
            if sell_qty < 1:
                sell_qty = qty
            _auto_partial_close(pos['id'], sell_qty, cur_price,
                               f"TP1 hedef ({tp1:.2f})", clear_tp_field='take_profit1')
            _auto_log_trade(uid, sym, 'SELL_TP1', cur_price, sell_qty,
                           f"TP1 kısmi: {cur_price:.2f} >= {tp1:.2f}", 0, 0, pos['id'])

        # TP yaklasma bildirimi (%2 icinde, henuz tetiklenmemis)
        try:
            from routes_telegram import send_tp_approaching
            if tp1 > 0 and cur_price >= tp1 * 0.98 and cur_price < tp1:
                send_tp_approaching(sym, cur_price, tp1, "TP1")
            elif tp2 > 0 and cur_price >= tp2 * 0.98 and cur_price < tp2:
                send_tp_approaching(sym, cur_price, tp2, "TP2")
            elif tp3 > 0 and cur_price >= tp3 * 0.98 and cur_price < tp3:
                send_tp_approaching(sym, cur_price, tp3, "TP3")
        except Exception:
            pass

        # Trailing Stop kontrolu
        if cfg['trailingStop']:
            highest = pos['highestPrice']
            if cur_price > highest:
                new_trailing = cur_price * (1 - cfg['trailingPct'] / 100)
                import os as _os_tg
                _tg_configured = bool(_os_tg.environ.get('TELEGRAM_BOT_TOKEN') and _os_tg.environ.get('TELEGRAM_CHAT_ID'))
                if _tg_configured:
                    # Telegram var: highest'i güncelle, trailing onayını bekle
                    _auto_update_highest_price(pos['id'], cur_price)
                    try:
                        from routes_telegram import send_trailing_update
                        send_trailing_update(sym, new_trailing, cur_price, pos.get('entryPrice'),
                                             tp1=tp1, tp2=tp2, tp3=tp3, position_id=pos['id'])
                    except Exception as _tr_err:
                        print(f"[AUTO-TRADER] Trailing bildirim hatası ({sym}): {_tr_err}")
                else:
                    # Telegram yok: trailing'i doğrudan güncelle (onay beklenemiyor)
                    _auto_update_trailing(pos['id'], round(new_trailing, 2), cur_price)
            else:
                trailing_sl = pos['trailingStop']
                if trailing_sl > 0 and cur_price <= trailing_sl:
                    _auto_close_position(pos['id'], cur_price, f"Trailing-Stop ({trailing_sl:.2f})")
                    _auto_log_trade(uid, sym, 'SELL_TRAIL', cur_price, pos['quantity'],
                                   f"Trailing SL: {cur_price:.2f} <= {trailing_sl:.2f}", 0, 0, pos['id'])
                    _sl_cooldown_block(uid, sym, cfg.get('tradeStyle', 'swing'))
                    _panic_clear(pos['id'])
                    continue

        # Panic-Sell kontrolü (ani ters dönüş — SL beklemeden çık)
        if cfg.get('panicSellEnabled'):
            try:
                drop_pct = float(cfg.get('panicDropPct', 2.0))
                window_min = int(cfg.get('panicWindowMin', 5))
                triggered, peak, drop = _panic_track_and_check(
                    pos['id'], cur_price, pos['entryPrice'], drop_pct, window_min
                )
                if triggered:
                    reason = f"Panic-Sell: {window_min}dk içinde zirveden %{drop:.1f} düşüş ({peak:.2f}→{cur_price:.2f})"
                    _auto_close_position(pos['id'], cur_price, reason)
                    _auto_log_trade(uid, sym, 'SELL_PANIC', cur_price, pos['quantity'],
                                   reason, 0, 0, pos['id'])
                    _panic_clear(pos['id'])
                    try:
                        from routes_telegram import send_telegram
                        send_telegram(
                            f"🚨 <b>PANİK SATIŞ: {sym}</b>\n"
                            f"🔝 Zirve: {peak:.2f} TL\n"
                            f"💰 Şu an: {cur_price:.2f} TL  (-%{drop:.1f})\n"
                            f"⏱ Pencere: {window_min} dk\n"
                            f"⚡ Midas'ta hemen sat — ani ters dönüş tespit edildi."
                        )
                    except Exception:
                        pass
                    continue
            except Exception as _panic_err:
                print(f"[AUTO-TRADE] Panic-sell kontrolü hatası ({sym}): {_panic_err}")


# =====================================================================
# ADIM 2a — Plan Merkezi kilitli planlarını yürüt
# =====================================================================

def _step2a_plan_positions(uid, cfg, slots, daily_remaining, open_positions, open_symbols, allowed, blocked):
    """Plan Merkezi'ndeki kilitli AL planlarını kontrol edip pozisyon aç."""
    from auto_trader import _auto_open_position, _auto_log_trade, _auto_get_open_positions
    _tf_map_plan = {'daily': 'daily', 'swing': 'weekly', 'monthly': 'monthly'}
    plan_tf = _tf_map_plan.get(cfg.get('tradeStyle', 'swing'), 'weekly')

    try:
        from config import _plan_lock_cache, _plan_lock_cache_lock, PLAN_LOCK_CONFIG
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
                continue
            if _sl_cooldown_check(uid, sym, cfg.get('tradeStyle', 'swing')):
                continue

            # Plan suresini kontrol et
            lock_cfg = PLAN_LOCK_CONFIG.get(tf, PLAN_LOCK_CONFIG.get('daily', {}))
            age = time.time() - entry.get('locked_at', 0)
            if age > lock_cfg.get('max_lock', 86400):
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
                if _regime.get('regime') in ('bear', 'strong_bear') and float(_regime.get('strength', 0)) > 60:
                    print(f"[AUTO-TRADE] {sym} plan ertelendi — güçlü ayı piyasası "
                          f"(güç={_regime.get('strength', 0):.0f})")
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

            # Sermaye kontrolu
            used_capital = sum(p['entryPrice'] * p['quantity'] for p in open_positions)
            if used_capital + position_cost > cfg['capital']:
                continue

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
                continue
            if not _score_ok:
                print(f"[AUTO-TRADE] {sym} plan skor hesaplanamadi — atlandi (hist cache bos olabilir)")
                continue

            # Telegram bildirimi veya direkt ac
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
                            uid, sym, cur_price, quantity,
                            plan_score, plan_conf,
                            stop_loss, tp1, tp2, tp3, trailing_sl
                        )
            except Exception:
                pass

            if _already_pending or _tg_sent:
                open_symbols.add(sym)
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
    except Exception as _plan_err:
        print(f"[AUTO-TRADE] Plan Merkezi kontrolü hatası: {_plan_err}")

    return open_positions, open_symbols, slots, daily_remaining


# =====================================================================
# ADIM 2b — Sinyal taraması ile kalan slotları doldur
# =====================================================================

def _step2b_scan_signals(uid, cfg, slots, daily_remaining, open_positions, open_symbols, allowed, blocked):
    """Sinyal skoru yüksek adayları tara ve kalan pozisyon slotlarını doldur."""
    from auto_trader import _auto_open_position, _auto_log_trade, _auto_get_open_positions, _auto_get_daily_trade_count
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
            if _reg.get('regime') in ('bear', 'strong_bear') and float(_reg.get('strength', 0)) > 60:
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


# =====================================================================
# Orkestratör — her kullanıcı için tüm adımları sırayla çalıştır
# =====================================================================

_sl_cooldown_loaded = False


def _auto_engine_cycle():
    """Ana oto-trade döngüsü. Scheduler tarafından periyodik olarak çağrılır."""
    global _sl_cooldown_loaded
    if not _is_market_open():
        return

    from auto_trader import _auto_trade_lock, _auto_get_config, _auto_get_open_positions, _auto_get_daily_trade_count

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


# ---- AUTO-TRADE API ENDPOINTS ----
