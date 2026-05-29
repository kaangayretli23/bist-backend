"""
BIST Pro - Auto Trader: Sinyal Taraması (ADIM 2b)
Sinyal skoru yüksek adayları tarayıp kalan pozisyon slotlarını doldurur.
auto_trader_engine.py'dan ayrıştırıldı (600 satır kuralı).
"""
# Not: config, auto_trader, signals, indicators, signals_core, routes_telegram,
# realtime_prices, data_fetcher, trade_plans fonksiyon içinde lazy import edilir.
from auto_trader_risk import _sl_cooldown_check, _reject_cooldown_check, _log_decision
from signals_market import REGIMES_BEARISH


def _step2b_scan_signals(uid, cfg, slots, daily_remaining, open_positions, open_symbols, allowed, blocked):
    """Sinyal skoru yüksek adayları tara ve kalan pozisyon slotlarını doldur."""
    from config import _cget_hist, _cset, _get_stocks, _hist_cache
    from auto_trader import _auto_open_position, _auto_log_trade, _auto_get_open_positions, _auto_get_daily_trade_count
    from trade_plans import calc_trade_plan

    if slots <= 0 or daily_remaining <= 0:
        return

    # A3: Drawdown freeze — son N gun realized PnL + acik pozisyon unrealized PnL
    # toplami sermayenin -X%'inden kotuyse yeni pozisyon acma. cfg.drawdownFreezePct=0 → kapali.
    #
    # K3: FAIL-CLOSED semantik. DD freeze bir koruma katmani — hata olursa scanner'i
    # durdur, sessiz devam etme. Aksi takdirde:
    #   - cap 0 gozukursesi esik 0 olur, hicbir zaman tetiklenmez,
    #   - DB exception sessizce yutulur, koruma yokmus gibi devam eder.
    # Hata varsa decision log + return: "guvende kal, pozisyon acma".
    _dd_pct = float(cfg.get('drawdownFreezePct', 0) or 0)
    if _dd_pct > 0:
        try:
            from config import get_db as _ddb
            _dd_win = int(cfg.get('drawdownFreezeWindowDays', 7) or 7)
            from datetime import datetime as _dt, timedelta as _td
            _cutoff = (_dt.now() - _td(days=_dd_win)).strftime('%Y-%m-%d %H:%M:%S')
            _cap = float(cfg.get('capital', 0) or 0)
            if _cap <= 0:
                # Capital yok/0 → drawdown anlamsiz, ama freeze aktif tutuldugu icin
                # guvenli taraf: scanner'i durdur, kullanici capital ayarini duzeltsin.
                _log_decision(uid, 'PORTFOLIO', 'SKIP', 'drawdown_freeze_no_capital',
                              detail=f"capital=0 ama dd_freeze=%{_dd_pct} aktif — capital ayarini guncelle")
                print(f"[AUTO-TRADE] {uid} DD freeze aktif ama capital=0 — scanner durduruldu (guvenli)")
                return

            _db = _ddb()
            try:
                _row = _db.execute(
                    "SELECT COALESCE(SUM(pnl), 0) AS s FROM auto_positions "
                    "WHERE user_id=? AND status='closed' AND closed_at>=?",
                    (uid, _cutoff),
                ).fetchone()
                _realized = float(_row['s']) if _row else 0.0
            finally:
                _db.close()

            # Unrealized PnL acik pozisyonlardan (cache-stale OK; kotuyse zaten kotu)
            _unrealized = 0.0
            from config import _cget, _stock_cache as _sc
            for _p in open_positions:
                _stk = _cget(_sc, _p['symbol']) or {}
                _cur = float(_stk.get('price', 0) or 0)
                if _cur > 0:
                    _unrealized += (_cur - float(_p['entryPrice'])) * float(_p['quantity'])

            _total_pnl = _realized + _unrealized
            _threshold = -_cap * (_dd_pct / 100.0)
            if _total_pnl <= _threshold:
                _log_decision(uid, 'PORTFOLIO', 'SKIP', 'drawdown_freeze',
                              detail=f"PnL={_total_pnl:.0f} TL ≤ esik={_threshold:.0f} "
                                     f"(cap={_cap:.0f}, win={_dd_win}g, dd={_dd_pct}%)")
                print(f"[AUTO-TRADE] {uid} drawdown freeze aktif: "
                      f"realized={_realized:.0f}, unrealized={_unrealized:.0f}, "
                      f"toplam={_total_pnl:.0f} TL ≤ esik={_threshold:.0f} TL "
                      f"({_dd_win}g, %{_dd_pct})")
                return
        except Exception as _dd_err:
            # FAIL-CLOSED: hatayi yutma, scanner'i durdur.
            _log_decision(uid, 'PORTFOLIO', 'SKIP', 'drawdown_freeze_error',
                          detail=f"DD freeze hesap hatasi (guvenli mod): {_dd_err}")
            print(f"[AUTO-TRADE] {uid} DD freeze HATA — scanner durduruldu (guvenli): {_dd_err}")
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
    _tf_now = _tf_map.get(cfg.get('tradeStyle', 'swing'), 'weekly')

    # Piyasa rejim filtresi: neutral mod'da min_score'a +1 bonus (sadece A+ sinyaller)
    try:
        from auto_trader_regime import regime_score_threshold_bonus
        _min_score_eff = float(cfg['minScore']) + regime_score_threshold_bonus()
    except Exception:
        _min_score_eff = float(cfg['minScore'])

    for s in stocks:
        sym = s.get('code', '')
        if not sym or sym in open_symbols:
            continue
        if allowed and sym not in allowed:
            continue
        if sym in blocked:
            _log_decision(uid, sym, 'SKIP', 'blocked', tf=_tf_now)
            continue
        if _sl_cooldown_check(uid, sym, cfg.get('tradeStyle', 'swing')):
            _log_decision(uid, sym, 'SKIP', 'sl_cooldown', tf=_tf_now)
            continue
        if _reject_cooldown_check(uid, sym):
            _log_decision(uid, sym, 'SKIP', 'reject_cooldown', tf=_tf_now)
            continue
        # Sektor cap: ayni sektorden max N pozisyon (korelasyon riski)
        try:
            from auto_trader_sectors import sector_full
            _full, _sec, _cur = sector_full(sym, open_positions, int(cfg.get('maxPerSector', 2)))
            if _full:
                _log_decision(uid, sym, 'SKIP', 'sector_cap',
                              detail=f"{_sec} dolu ({_cur}/{int(cfg.get('maxPerSector', 2))})",
                              tf=_tf_now)
                continue
        except Exception:
            pass
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

        if signal not in ('AL', 'GÜÇLÜ AL') or score < _min_score_eff or confidence < cfg['minConfidence']:
            _log_decision(uid, sym, 'SKIP', 'score',
                          detail=f"sig={signal}, sc={score:.1f}/{_min_score_eff:.1f}, conf={confidence:.0f}/{cfg['minConfidence']}",
                          tf=_tf_now, price=live_price, score=score, confidence=confidence)
            continue

        # B3+B4: KAP haber/duyuru filtresi — son haberlerde negatif sentiment varsa
        # veya yuksek skorlu 'important' duyuru (temettu/sermaye/SPK) varsa pozisyon acma.
        # Bilgilendirme/temettu aciklama gunu volatilite zıplamasını önler.
        try:
            from kap_scraper import get_stock_sentiment
            _sent = get_stock_sentiment(sym)
            _sent_label = _sent.get('label', 'nötr')
            _sent_score = int(_sent.get('score', 0) or 0)
            _imp_cnt = int(_sent.get('important_count', 0) or 0)
            if _sent_label == 'negatif':
                _log_decision(uid, sym, 'SKIP', 'kap_negative',
                              detail=f"sentiment={_sent_label}, sc={_sent_score}, imp={_imp_cnt}",
                              tf=_tf_now, price=live_price, score=score, confidence=confidence)
                continue
            # Aciklama gunu: 2+ onemli haber + nötr/negatif sentiment → riski azalt
            if _imp_cnt >= 2 and _sent_label != 'pozitif':
                _log_decision(uid, sym, 'SKIP', 'kap_announcement',
                              detail=f"important={_imp_cnt}, sentiment={_sent_label}",
                              tf=_tf_now, price=live_price, score=score, confidence=confidence)
                continue
        except Exception:
            pass

        # B2: Sektor momentum filtresi — hisse AL diyor ama sektor son 1 ay
        # ortalama ≤ -%3 ise (rotasyon dışı, defansif düşüş) reddet.
        try:
            from config import SECTOR_MAP
            from signals_market import calc_sector_relative_strength
            _sym_sector = None
            for _sec_name, _sec_syms in SECTOR_MAP.items():
                if sym in _sec_syms:
                    _sym_sector = _sec_name
                    break
            if _sym_sector:
                _rs = calc_sector_relative_strength()
                _sec_data = next((x for x in _rs.get('sectors', []) if x.get('name') == _sym_sector), None)
                if _sec_data:
                    _avg_1m = float(_sec_data.get('avgChange1m', 0) or 0)
                    if _avg_1m <= -3.0:
                        _log_decision(uid, sym, 'SKIP', 'sector_momentum',
                                      detail=f"sec={_sym_sector} 1m={_avg_1m:.2f}% (≤-3%)",
                                      tf=_tf_now, price=live_price, score=score, confidence=confidence)
                        continue
        except Exception:
            pass

        # B1: Multi-timeframe alignment guard
        # Sinyal tek timeframe AL diyebilir ama digerleri SAT/NOTR ise — yanlis sinyal riski.
        # En az 2 timeframe AL veya AL ailesi (TUTUN/AL) olmali.
        try:
            _al_actions = ('AL', 'GÜÇLÜ AL', 'TUTUN/AL')
            _aligned_count = sum(1 for _tf in ('weekly', 'monthly', 'yearly')
                                 if (recs.get(_tf) or {}).get('action') in _al_actions)
            if _aligned_count < 2:
                _log_decision(uid, sym, 'SKIP', 'mtf_misalign',
                              detail=f"AL/TUTUN={_aligned_count}/3 (W={recs.get('weekly',{}).get('action','?')} "
                                     f"M={recs.get('monthly',{}).get('action','?')} Y={recs.get('yearly',{}).get('action','?')})",
                              tf=_tf_now, price=live_price, score=score, confidence=confidence)
                continue
        except Exception:
            pass

        # A4: Anti-FOMO — bugun ATR'nin 1.5x'i kadar zaten yukselmisse, gec kaldik (tepe alimi riski).
        try:
            from indicators_basic import calc_atr
            _h_arr = hist['High'].values.astype(float)
            _l_arr = hist['Low'].values.astype(float)
            _c_arr = hist['Close'].values.astype(float)
            _atr_data = calc_atr(_h_arr, _l_arr, _c_arr)
            _atr_val = float(_atr_data.get('value', 0))
            if _atr_val > 0 and len(hist) >= 2:
                _today_open = float(hist['Open'].iloc[-1])
                _open_to_now = live_price - _today_open
                # Acılıstan simdiye 1.5 ATR uzeri yuksellis = tepe alimi
                if _open_to_now > _atr_val * 1.5:
                    _log_decision(uid, sym, 'SKIP', 'fomo',
                                  detail=f"open→now={_open_to_now:.2f} > 1.5×ATR={_atr_val*1.5:.2f}",
                                  tf=_tf_now, price=live_price, score=score, confidence=confidence)
                    continue
        except Exception:
            pass

        # A5: Volume confirm (sıkılaştırılmış) — 20-bar ortalama hacmin %80 altıysa zayıf onay.
        # Mevcut %50 hard-skip duruyor, bu ek filtre soft-skip (sinyal güçlü olsa bile).
        if len(hist) >= 20:
            try:
                _vol_today = float(hist['Volume'].iloc[-1])
                _vol_avg20 = float(hist['Volume'].iloc[-20:].mean())
                if _vol_avg20 > 0 and _vol_today < _vol_avg20 * 0.8:
                    _log_decision(uid, sym, 'SKIP', 'volume_weak',
                                  detail=f"bugun={_vol_today:.0f} < ort20*0.8={_vol_avg20*0.8:.0f}",
                                  tf=_tf_now, price=live_price, score=score, confidence=confidence)
                    continue
            except Exception:
                pass

        # Piyasa rejimi: güçlü ayı piyasasında yeni alım yapma
        try:
            from signals import calc_market_regime as _cmr
            _reg = _cmr()
            if _reg.get('regime') in REGIMES_BEARISH and float(_reg.get('strength', 0)) > 60:
                print(f"[AUTO-TRADE] {sym} atlandi — ayi piyasasi (guc={_reg.get('strength')})")
                _log_decision(uid, sym, 'SKIP', 'regime',
                              detail=f"{_reg.get('regime')}/{_reg.get('strength', 0):.0f}",
                              tf=_tf_now, price=live_price)
                continue
        except Exception:
            pass

        # Hacim filtresi (relatif: bugun ort20'nin %50'sinin altindaysa skip)
        if len(hist) >= 20:
            try:
                vol_today = float(hist['Volume'].iloc[-1])
                vol_avg20 = float(hist['Volume'].iloc[-20:].mean())
                if vol_avg20 > 0 and vol_today < vol_avg20 * 0.5:
                    _log_decision(uid, sym, 'SKIP', 'volume',
                                  detail=f"bugun={vol_today:.0f} < ort20*0.5={vol_avg20*0.5:.0f}",
                                  tf=_tf_now, price=live_price)
                    continue
            except Exception:
                pass

        # Likidite filtresi (mutlak TL turnover: cok sig hisseyi filtrele)
        # 20 gunluk ortalama (Close * Volume) cfg.minTurnoverTL altindaysa skip.
        if len(hist) >= 20:
            try:
                _min_turn = float(cfg.get('minTurnoverTL', 1_000_000) or 0)
                if _min_turn > 0:
                    _turn20 = float((hist['Close'].iloc[-20:] * hist['Volume'].iloc[-20:]).mean())
                    if _turn20 < _min_turn:
                        _log_decision(uid, sym, 'SKIP', 'turnover',
                                      detail=f"ort20_TL={_turn20:.0f} < esik={_min_turn:.0f}",
                                      tf=_tf_now, price=live_price)
                        continue
            except Exception:
                pass

        # Gap down koruması
        if len(hist) >= 2:
            try:
                prev_close = float(hist['Close'].iloc[-2])
                today_open = float(hist['Open'].iloc[-1])
                if prev_close > 0 and (prev_close - today_open) / prev_close > 0.03:
                    _log_decision(uid, sym, 'SKIP', 'gap_down',
                                  detail=f"prev={prev_close:.2f} -> open={today_open:.2f}",
                                  tf=_tf_now, price=live_price)
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
    try:
        from auto_trader_regime import regime_daily_trade_factor
        _daily_cap_eff = max(1, int(cfg['maxDailyTrades'] * regime_daily_trade_factor()))
    except Exception:
        _daily_cap_eff = int(cfg['maxDailyTrades'])
    daily_remaining = _daily_cap_eff - _auto_get_daily_trade_count(uid)

    for cand in candidates[:min(slots, daily_remaining)]:
        sym = cand['symbol']
        price = cand['price']
        if price <= 0:
            continue

        # A1: ATR-based dinamik SL — sabit %3 yerine volatiliteye gore.
        # Volatil hisselerde geniş SL (eskiden anında tetikleniyordu),
        # sakin hisselerde dar SL (eskiden gereksiz büyük zarar).
        # Formul: SL = price - 1.5 × ATR. Cap: cfg.stopLossPct*1.5 (asiri genis olmasin),
        #         floor: cfg.stopLossPct*0.5 (cok dar olmasin).
        stop_loss = round(price * (1 - cfg['stopLossPct'] / 100), 2)  # default fallback
        try:
            from indicators_basic import calc_atr as _calc_atr_sl
            _atr_data = _calc_atr_sl(
                hist['High'].values.astype(float),
                hist['Low'].values.astype(float),
                hist['Close'].values.astype(float),
            )
            _atr_v = float(_atr_data.get('value', 0))
            if _atr_v > 0:
                _atr_sl_distance = _atr_v * 1.5
                _max_distance = price * (cfg['stopLossPct'] * 1.5 / 100)  # cap
                _min_distance = price * (cfg['stopLossPct'] * 0.5 / 100)  # floor
                _atr_sl_distance = max(_min_distance, min(_max_distance, _atr_sl_distance))
                stop_loss = round(price - _atr_sl_distance, 2)
                print(f"[AUTO-TRADE] {sym} ATR-SL: ATR={_atr_v:.3f}, SL distance={_atr_sl_distance:.3f} ({_atr_sl_distance/price*100:.2f}%)")
        except Exception as _atr_sl_err:
            print(f"[AUTO-TRADE] {sym} ATR-SL hesap hatasi (sabit %SL kullanildi): {_atr_sl_err}")

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

        # TP monotonik kontrol — kademeli kar alma icin tp1 < tp2 < tp3 sart.
        # Plan override'ında targets esit/geri sirali gelmis olabilir; en az %0.5
        # spread garanti etmek icin default formul fallback'i.
        _tp_default_1 = round(price * (1 + cfg['takeProfitPct'] / 100), 2)
        _tp_default_2 = round(price * (1 + cfg['takeProfitPct'] * 1.5 / 100), 2)
        _tp_default_3 = round(price * (1 + cfg['takeProfitPct'] * 2 / 100), 2)
        if not (price < tp1 < tp2 < tp3) or (tp2 - tp1) / price < 0.005 or (tp3 - tp2) / price < 0.005:
            print(f"[AUTO-TRADE] {sym} TP siralama bozuk "
                  f"(tp1={tp1:.2f}, tp2={tp2:.2f}, tp3={tp3:.2f}) → default formul")
            tp1, tp2, tp3 = _tp_default_1, _tp_default_2, _tp_default_3

        # R/R guard — TP1/SL ratio < 1 ise pozisyon acmiyoruz.
        # Mantik: kazansa bile az kazanir, kaybetse cok kaybeder; matematik tutmaz
        # (TUKAS gibi: SL %4.2, TP1 %3.1 → R/R=0.74 → win-rate %60+ gerekir kar icin).
        sl_distance_for_rr = price - stop_loss
        tp1_distance = tp1 - price
        if sl_distance_for_rr > 0 and tp1_distance > 0:
            rr = tp1_distance / sl_distance_for_rr
            if rr < 1.0:
                _log_decision(uid, sym, 'SKIP', 'poor_rr',
                              detail=f"R/R={rr:.2f} (TP1=+{tp1_distance/price*100:.1f}%, "
                                     f"SL=-{sl_distance_for_rr/price*100:.1f}%)",
                              price=price, score=cand['score'], confidence=cand['confidence'])
                print(f"[AUTO-TRADE] {sym} atlandi — kotu R/R={rr:.2f}")
                continue

        risk_amount = cfg['capital'] * (cfg['riskPerTrade'] / 100)
        sl_distance = price - stop_loss
        if sl_distance <= 0:
            continue
        quantity = int(risk_amount / sl_distance)
        if quantity < 1:
            continue
        position_cost = quantity * price

        # Serbest sermayeye gore quantity'i kirp — "100 TL kaldi, 2000 TL'lik al" demesin
        used_capital = sum(p['entryPrice'] * p['quantity'] for p in open_positions)
        free_capital = max(0, cfg['capital'] - used_capital)
        if position_cost > free_capital:
            affordable_qty = int(free_capital / price) if price > 0 else 0
            if affordable_qty < 1:
                print(f"[AUTO-TRADE] {sym} atlandi — serbest sermaye {free_capital:.0f} TL (1 lot {price:.2f} TL)")
                _log_decision(uid, sym, 'SKIP', 'budget',
                              detail=f"serbest={free_capital:.0f} TL, 1 lot={price:.2f} TL",
                              tf=_tf_now, price=price, score=cand['score'], confidence=cand['confidence'])
                continue
            print(f"[AUTO-TRADE] {sym} adet kirpildi: {quantity} -> {affordable_qty} "
                  f"(serbest={free_capital:.0f} TL)")
            quantity = affordable_qty
            position_cost = quantity * price

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
                        uid, sym, price, quantity,
                        cand['score'], cand['confidence'],
                        stop_loss, tp1, tp2, tp3, trailing_sl
                    )
        except Exception as _tg_err:
            print(f"[AUTO-TRADE] Telegram sinyal hatası: {_tg_err}")

        # Karar:
        #  1) Pending var veya Telegram gonderildi → onay bekle (PENDING)
        #  2) Telegram aktif ama gonderim basarisiz → fail-safe: pozisyon ACMA
        #     (kullanicı Telegram bekliyordu ama uyarı gitmedi; sessiz acmak yanlis olur)
        #  3) Telegram konfigure degil → eskisi gibi otomatik ac
        if _already_pending or _tg_sent:
            try:
                from realtime_prices import subscribe as _rt_sub
                _rt_sub(sym)
            except Exception:
                pass
            open_symbols.add(sym)
            # Y3: Pending de used_capital'i tuketir. Sonraki adaylar bu kosulla
            # serbest sermayeyi hesaplasin (ardisik pending'ler kapasiteyi asmasin).
            open_positions.append({
                'symbol': sym, 'entryPrice': price, 'quantity': quantity,
                '_pending': True,  # debug icin isaret
            })
            _log_decision(uid, sym, 'PENDING', 'telegram_approve',
                          detail=f"qty={quantity}, SL={stop_loss:.2f}, TP1={tp1:.2f}",
                          tf=_tf_now, price=price, score=cand['score'], confidence=cand['confidence'])
        elif _tg_configured:
            _log_decision(uid, sym, 'SKIP', 'telegram_failed',
                          detail='Telegram onay sinyali gonderilemedi; pozisyon acilmadi',
                          tf=_tf_now, price=price, score=cand['score'], confidence=cand['confidence'])
            continue
        else:
            pos_id = _auto_open_position(uid, sym, price, quantity, stop_loss, tp1, tp2, tp3, trailing_sl)
            if pos_id:
                _auto_log_trade(uid, sym, 'BUY', price, quantity,
                               f"Skor={cand['score']:.1f}, Guven=%{cand['confidence']:.0f}, SL={stop_loss:.2f}, TP1={tp1:.2f}",
                               cand['score'], cand['confidence'], pos_id)
                open_positions = _auto_get_open_positions(uid)
                open_symbols.add(sym)
                _log_decision(uid, sym, 'BUY', 'opened',
                              detail=f"qty={quantity}, SL={stop_loss:.2f}, TP1={tp1:.2f}",
                              tf=_tf_now, price=price, score=cand['score'], confidence=cand['confidence'])
