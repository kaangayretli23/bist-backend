"""
BIST Pro - Auto Trader: Sinyal Taraması (ADIM 2b)
Sinyal skoru yüksek adayları tarayıp kalan pozisyon slotlarını doldurur.
auto_trader_engine.py'dan ayrıştırıldı (600 satır kuralı).
"""
# Not: config, auto_trader, signals, indicators, signals_core, routes_telegram,
# realtime_prices, data_fetcher, trade_plans fonksiyon içinde lazy import edilir.
import os as _os_mod
from auto_trader_risk import _sl_cooldown_check, _reject_cooldown_check, _log_decision, _data_freshness
from signals_market import REGIMES_BEARISH


def _int_env(key, default):
    try:
        return int(_os_mod.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _float_env(key, default):
    try:
        return float(_os_mod.environ.get(key, str(default)))
    except (TypeError, ValueError):
        return default


def _quality_position_pct(rr, score, confidence):
    """KALİTE-BAZLI standalone pozisyon yüzdesi (sermayenin %'si).

    Kullanıcı tasarımı — ÖNCELİK HİYERARŞİSİ (R/R lider DEĞİL):
      1) BİRİNCİL = conviction: güven/skor'dan hangisi güçlüyse o lider (mode='max').
      2) İKİNCİL  = R/R sadece tiebreaker: benzer conviction'da yüksek R/R öne geçer
                    (küçük ALLOC_RR_TILT katkısı → ana sırayı bozmaz, sadece ayırır).
      3) CEZA     = çok düşük R/R (ALLOC_RR_FLOOR altı) geri plana atılır (×penalty).
      4) AI 2. göz EN SON bakar (scanner'da tilt olarak, bu fonksiyonun dışında).

    Paylar birbirine NORMALİZE EDİLMEZ → her hisse bağımsız % alır, cherry-pick korunur.
    Sonuç [floor .. ceil] bandına eşlenir. Tüm parametreler ENV ile ayarlanabilir.
    """
    def _c01(x):
        return 0.0 if x < 0 else (1.0 if x > 1.0 else x)
    rr_min, rr_max = _float_env('ALLOC_RR_MIN', 1.0), _float_env('ALLOC_RR_MAX', 3.0)
    sc_min, sc_max = _float_env('ALLOC_SCORE_MIN', 5.0), _float_env('ALLOC_SCORE_MAX', 10.0)
    rr_n   = _c01((float(rr or 0) - rr_min) / (rr_max - rr_min)) if rr_max > rr_min else 0.0
    sc_n   = _c01((float(score or 0) - sc_min) / (sc_max - sc_min)) if sc_max > sc_min else 0.0
    conf_n = _c01(float(confidence or 0) / 100.0)

    # 1) BİRİNCİL conviction — varsayılan 'confidence': skoru lider YAPMA (backtest:
    # yüksek skor = en kötü forward bucket). 'max'/'score'/'blend' ile değiştirilebilir.
    mode = _os_mod.environ.get('ALLOC_CONVICTION_MODE', 'confidence').lower()
    if mode == 'confidence':
        conviction = conf_n
    elif mode == 'score':
        conviction = sc_n
    elif mode == 'blend':
        conviction = 0.6 * conf_n + 0.4 * sc_n
    else:  # 'max' — hangisi güçlüyse o başta
        conviction = max(conf_n, sc_n)

    # 2) R/R İKİNCİL tiebreaker (küçük katkı → benzer conviction'da ayırır)
    rr_tilt = _float_env('ALLOC_RR_TILT', 0.15)
    q = conviction * (1.0 - rr_tilt) + rr_n * rr_tilt

    # 3) Çok düşük R/R → geri plana at
    rr_floor   = _float_env('ALLOC_RR_FLOOR', 1.3)
    rr_penalty = _float_env('ALLOC_RR_PENALTY', 0.6)
    if float(rr or 0) < rr_floor:
        q *= rr_penalty

    q = _c01(q)
    floor_pct = _float_env('ALLOC_FLOOR_PCT', 10.0)
    ceil_pct  = _float_env('ALLOC_CEIL_PCT', 25.0)
    return floor_pct + (ceil_pct - floor_pct) * q, q


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
                      f"toplam={_total_pnl:.0f} TL <= esik={_threshold:.0f} TL "
                      f"({_dd_win}g, %{_dd_pct})")
                return
        except Exception as _dd_err:
            # FAIL-CLOSED: hatayi yutma, scanner'i durdur.
            _log_decision(uid, 'PORTFOLIO', 'SKIP', 'drawdown_freeze_error',
                          detail=f"DD freeze hesap hatasi (guvenli mod): {_dd_err}")
            print(f"[AUTO-TRADE] {uid} DD freeze HATA — scanner durduruldu (guvenli): {_dd_err}")
            return

    # Kemal #4a: Ardışık-zarar kill-switch — bugün üst üste M zarar → o gün yeni pozisyon yok.
    # DD freeze toplam PnL'e bakar; bu seri bazlı (kötü-gün erken durdurucu). Hata olursa FAIL-OPEN
    # (ikincil koruma; DD freeze + diğer kapılar zaten devrede, tüm trading'i kilitlemeyelim).
    try:
        from auto_trader_risk import _consecutive_loss_freeze
        _kill, _streak = _consecutive_loss_freeze(uid, int(cfg.get('maxConsecutiveLosses', 3) or 3))
        if _kill:
            _log_decision(uid, 'PORTFOLIO', 'SKIP', 'consecutive_loss_freeze',
                          detail=f"bugun ardışık {_streak} zarar — scanner gun sonuna kadar donduruldu")
            print(f"[AUTO-TRADE] {uid} ardışık {_streak} zarar — kill-switch aktif, bugun yeni pozisyon yok")
            return
    except Exception as _kl_err:
        print(f"[AUTO-TRADE] {uid} kill-switch kontrol hatasi (devam): {_kl_err}")

    # #4c: Açık pozisyon korelasyonunu ÖLÇ (limit yok, throttle 30dk) — beta/endeks konsantrasyonu.
    try:
        from auto_trader_risk import _log_portfolio_correlation
        _log_portfolio_correlation(uid, open_positions)
    except Exception:
        pass

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

    # #7 Data freshness eşiği (dk→sn). Varsayılan 15 dk; 0 = gate kapalı.
    _max_data_age_sec = _int_env('AUTO_MAX_DATA_AGE_MIN', 15) * 60

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

        # #7 Data freshness gate: fiyat/hacim verisi bayatsa (feed kesintisi vb.) sinyal ÜRETME.
        # AUTO_MAX_DATA_AGE_MIN=0 → kapali. Yaş bilinmiyorsa (kaynak yok) gate karar vermez.
        if _max_data_age_sec > 0:
            _eff_age, _raw_age, _src = _data_freshness(sym)
            if _eff_age is not None and _eff_age > _max_data_age_sec:
                # Kaynak gecikmesi eklendiyse (yf) 'gecikmeli', yoksa gerçekten 'bayat'
                _delayed = _raw_age is not None and (_eff_age - _raw_age) > 1
                _lbl = 'gecikmeli' if _delayed else 'bayat'
                _log_decision(uid, sym, 'SKIP', 'stale_data',
                              detail=f"veri {_lbl} ({_src}) etkin {int(_eff_age // 60)}dk (>{_max_data_age_sec // 60}dk)",
                              tf=_tf_now)
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

        # NOT (minScore vs minConfidence): burada `confidence`, calc_recommendation'da
        # skordan TÜRETİLİR → conf = min(abs(score)/14*100, 100). Yani ikisi monotonik ilişkili;
        # hangisi daha sıkıysa BAĞLAYICI kapı odur (çoğu ayarda minConfidence baskın çıkar).
        # Bu bir çifte-sayım DEĞİL ama iki ayrı bağımsız eşik gibi de DEĞİL — aynı büyüklüğün
        # iki görünümü. Gerçek ayrıştırma (gate'i bağımsız ml_confidence'a taşımak) Kemal raund 6
        # işidir; şimdilik davranış korunuyor, sadece niyet belgeleniyor.
        # ADIM 2 — aşırı-alım/rejim cezası: mean-reverting rejimde overbought isimlerin
        # ETKİN skorunu düşür (tepeleri "al" diye bağırma). Gate bu eff_score'a göre.
        eff_score = score
        _ob_pen = 0.0
        if _int_env('AUTO_OVERBOUGHT_PENALTY', 1) == 1:
            try:
                from indicators_basic import calc_rsi_single
                _rsi_now = calc_rsi_single(hist_live['Close'])
            except Exception:
                _rsi_now = 50.0
            try:
                from signals import calc_market_regime as _cmr_pen
                _regime_now = (_cmr_pen() or {}).get('regime', 'sideways')
            except Exception:
                _regime_now = 'sideways'
            try:
                from signal_calibration import overbought_penalty
                _ob_pen = overbought_penalty(_rsi_now, _regime_now)
            except Exception:
                _ob_pen = 0.0
            eff_score = score - _ob_pen

        if signal not in ('AL', 'GÜÇLÜ AL') or eff_score < _min_score_eff or confidence < cfg['minConfidence']:
            _detail = f"sig={signal}, sc={score:.1f}/{_min_score_eff:.1f}, conf={confidence:.0f}/{cfg['minConfidence']}"
            if _ob_pen > 0:
                _detail += f" [aşırı-alım cezası −{_ob_pen:.1f} → eff={eff_score:.1f}]"
            _log_decision(uid, sym, 'SKIP', 'score', detail=_detail,
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
        # NOT (Kemal #2a): hist son bari BUGÜNE ait degilse hist['Open'].iloc[-1] dunun acilisidir →
        # 'open→now' = live_price - dunku_acilis ANLAMSIZ cikar (ya hep FOMO der ya hic). Bu durumda
        # anti-FOMO'yu UYGULAMA; 'fomo_stale_open' olarak logla ki kac kez stale oldugunu olcebilelim.
        try:
            from indicators_basic import calc_atr
            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            _ist_today = (_dt.now(_tz.utc) + _td(hours=3)).date()
            try:
                _last_bar_date = hist.index[-1].date()
            except Exception:
                _last_bar_date = None
            if _last_bar_date is not None and _last_bar_date != _ist_today:
                _log_decision(uid, sym, 'INFO', 'fomo_stale_open',
                              detail=f"son bar {_last_bar_date} != bugun {_ist_today} — anti-FOMO atlandi",
                              tf=_tf_now, price=live_price, score=score, confidence=confidence)
            else:
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
            'signal': signal,          # AL / GÜÇLÜ AL (AI review + telegram için)
            'hist': hist,              # BUG FIX: bu sembolün DOĞRU hist'i (cached DataFrame ref).
                                       # Eskiden 2. döngüde ATR-SL, döngüden sızan son sembolün
                                       # hist'ini kullanıyordu → yanlış SL. Artık candidate taşıyor.
            'score_breakdown': rec.get('scoreBreakdown'),  # AI bağlamı için faktör kovaları
            'reason': rec.get('reason'),                   # sistemin kendi kısa gerekçesi
        })

    # Güven (confidence) öncelikli sırala; eşitlikte skor ayırıcı.
    # Bu sıra hem sınırlı slotlara hangi hisselerin gireceğini, hem AI-review
    # sırasını, hem de Telegram'a geliş sırasını belirler → en güvenli önce.
    candidates.sort(key=lambda x: (x['confidence'], x['score']), reverse=True)
    slots = cfg['maxPositions'] - len(open_positions)
    try:
        from auto_trader_regime import regime_daily_trade_factor
        _daily_cap_eff = max(1, int(cfg['maxDailyTrades'] * regime_daily_trade_factor()))
    except Exception:
        _daily_cap_eff = int(cfg['maxDailyTrades'])
    daily_remaining = _daily_cap_eff - _auto_get_daily_trade_count(uid)

    # ===== AI ikincil filtre kurulumu (opsiyonel) =====
    # ENABLED=0 veya key yok → _ai_enabled False, eski akış hiç değişmez.
    # AI yalnızca EN GÜÇLÜ top-N adayda ve skor >= eşik olduğunda çalışır (tüm BIST'e DEĞİL).
    import os as _os_ai
    _ai_enabled = False
    try:
        import ai_reviewer as _air
        _ai_enabled = _air.is_ai_review_enabled()
    except Exception as _ai_imp_err:
        print(f"[AUTO-TRADE] ai_reviewer yuklenemedi (AI atlandi): {_ai_imp_err}")
    _ai_top_n = _int_env('AI_REVIEW_TOP_N', 5)
    _ai_min_score = _float_env('AI_MIN_SCORE_FOR_REVIEW', 7.0)
    _ai_fail_mode = _os_ai.environ.get('AI_REVIEW_FAIL_MODE', 'manual_review')
    _tg_ok = bool(_os_ai.environ.get('TELEGRAM_BOT_TOKEN') and _os_ai.environ.get('TELEGRAM_CHAT_ID'))
    _ai_reviews_done = 0
    if _ai_enabled:
        print(f"[AUTO-TRADE] AI ikincil filtre AKTIF (top_n={_ai_top_n}, min_score={_ai_min_score}, fail={_ai_fail_mode})")

    for cand in candidates[:min(slots, daily_remaining)]:
        sym = cand['symbol']
        price = cand['price']
        if price <= 0:
            continue

        # BUG FIX: bu adayın DOĞRU hist'ini candidate'ten al (döngüden sızan stale hist DEĞİL).
        # ATR-SL ve trade-plan hesabı bunu kullanır. Yoksa cache'ten tazele; o da yoksa
        # ATR atlanır, sabit %SL fallback devrede kalır.
        hist = cand.get('hist')
        if hist is None or len(hist) < 20:
            hist = _cget_hist(f"{sym}_1y")

        # A1: ATR-based dinamik SL — sabit %3 yerine volatiliteye gore.
        # Volatil hisselerde geniş SL (eskiden anında tetikleniyordu),
        # sakin hisselerde dar SL (eskiden gereksiz büyük zarar).
        # Formul: SL = price - 1.5 × ATR. Cap: cfg.stopLossPct*1.5 (asiri genis olmasin),
        #         floor: cfg.stopLossPct*0.5 (cok dar olmasin).
        stop_loss = round(price * (1 - cfg['stopLossPct'] / 100), 2)  # default fallback
        _atr_v = 0.0
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

        # Trade planini kontrol et (daha iyi SL/TP varsa kullan) — hist yukarida (döngü başı) yüklendi
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

        # SL EMNİYET FLOOR — SL asla gürültü-seviyesinde dar olmasın.
        # Bug: plan-SL override'ı ATR floor'unu bypass edip destek seviyesini SL yapıyordu
        # → CCOLA %0.23 SL (günlük aralık %3.57'de anında stop). Nihai kural: SL mesafesi
        # >= max(config-floor %0.75, 1×ATR, %1). Volatil hissede otomatik daha geniş.
        _hard_min_sl = max(price * (cfg['stopLossPct'] * 0.5 / 100),
                           _atr_v if _atr_v > 0 else 0.0,
                           price * 0.01)
        if (price - stop_loss) < _hard_min_sl:
            _old_sl = stop_loss
            stop_loss = round(price - _hard_min_sl, 2)
            print(f"[AUTO-TRADE] {sym} SL floor uygulandı: {_old_sl} → {stop_loss} "
                  f"(mesafe >=%{_hard_min_sl/price*100:.2f}, gürültü-koruması)")

        # TP monotonik kontrol — kademeli kar alma icin tp1 < tp2 < tp3 sart.
        # Plan override'ında targets esit/geri sirali gelmis olabilir; en az %0.5
        # spread garanti etmek icin default formul fallback'i.
        _tp_default_1 = round(price * (1 + cfg['takeProfitPct'] / 100), 2)
        _tp_default_2 = round(price * (1 + cfg['takeProfitPct'] * 1.5 / 100), 2)
        _tp_default_3 = round(price * (1 + cfg['takeProfitPct'] * 2 / 100), 2)
        if not (price < tp1 < tp2 < tp3) or (tp2 - tp1) / price < 0.005 or (tp3 - tp2) / price < 0.005:
            print(f"[AUTO-TRADE] {sym} TP siralama bozuk "
                  f"(tp1={tp1:.2f}, tp2={tp2:.2f}, tp3={tp3:.2f}) -> default formul")
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
        risk_qty = int(risk_amount / sl_distance)
        # POZİSYON-BAŞINA notional tavanı → sepet çeşitlendirme. İKİ MOD:
        #   ALLOC_QUALITY_ENABLED=0 (DEFAULT — PARK): düz eşit tavan (AUTO_MAX_POSITION_PCT,
        #     ~%20). Güven/skora göre BÜYÜTME YOK. Sebep: ölçüm ml_confidence'ın ANTI-predictive
        #     olduğunu gösterdi (bkz memory: auto_buy_negative_edge_finding) → güveni ödüllendiren
        #     kalite-pay zararı büyütür. Strateji doğrulanana dek kapalı.
        #   ALLOC_QUALITY_ENABLED=1: kalite-bazlı standalone pay (conviction lider, R/R tiebreaker).
        # risk_qty ile min() → per-trade ZARAR yine riskPerTrade ile sınırlı (ayrı emniyet).
        _rr_now = ((tp1 - price) / sl_distance) if (sl_distance > 0 and tp1 > price) else 0.0
        if _int_env('ALLOC_QUALITY_ENABLED', 0) == 1:
            _pos_pct, _q = _quality_position_pct(_rr_now, cand['score'], cand['confidence'])
            _size_mode = f"kalite-pay q={_q:.2f}"
        else:
            _pos_pct, _q = _float_env('AUTO_MAX_POSITION_PCT', 20.0), 0.0
            _size_mode = "düz-tavan (kalite PARK)"
        _cap_qty = int((cfg['capital'] * _pos_pct / 100.0) / price) if price > 0 else 0
        quantity = min(risk_qty, max(1, _cap_qty))
        if quantity < 1:
            continue
        position_cost = quantity * price
        print(f"[AUTO-TRADE] {sym} {_size_mode}: %{_pos_pct:.1f} (R/R={_rr_now:.2f}, "
              f"skor={cand['score']:.1f}, guven=%{cand['confidence']:.0f}) "
              f"-> {quantity} lot / {position_cost:.0f} TL")

        # Serbest sermayeye gore quantity'i kirp — "100 TL kaldi, 2000 TL'lik al" demesin
        used_capital = sum(p['entryPrice'] * p['quantity'] for p in open_positions)
        free_capital = max(0, cfg['capital'] - used_capital)

        # AUTO_STRICT_CAPITAL=1: kısmi (clip'lenmiş) pozisyon AÇMA. Tam pozisyon serbest
        # sermayeye sığmıyorsa işlemi ENGELLE — sadece uyarı yazıp devam etme. Risk-parity
        # bozulmasın + "para yetmiyor ama yine de bir şeyler al" davranışı olmasın.
        import os as _os_cap
        _strict_capital = _os_cap.environ.get('AUTO_STRICT_CAPITAL', '0') == '1'
        if position_cost > free_capital:
            if _strict_capital:
                print(f"[AUTO-TRADE] {sym} atlandi — STRICT sermaye: maliyet {position_cost:.0f} TL "
                      f"> serbest {free_capital:.0f} TL (clip yok, pozisyon acilmadi)")
                _log_decision(uid, sym, 'SKIP', 'budget_strict',
                              detail=f"maliyet={position_cost:.0f} > serbest={free_capital:.0f} TL "
                                     f"(AUTO_STRICT_CAPITAL=1 → clip yok, engellendi)",
                              tf=_tf_now, price=price, score=cand['score'], confidence=cand['confidence'])
                continue
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
            # #4b: kırpma risk-parity'yi bozar (artık bütçe-bazlı, risk-bazlı değil). Efektif
            # risk'i logla ki hedeften sapma görünsün; çok küçükse trade anlamsız.
            try:
                _eff_risk = (quantity * sl_distance / cfg['capital'] * 100) if cfg.get('capital') else 0
                _log_decision(uid, sym, 'INFO', 'risk_clipped',
                              detail=f"adet={quantity} (kırpıldı), efektif risk=%{_eff_risk:.2f} "
                                     f"vs hedef=%{cfg['riskPerTrade']:.2f} (serbest sermaye sınırı)",
                              tf=_tf_now, price=price, score=cand['score'], confidence=cand['confidence'])
            except Exception:
                pass

        # SON EMNİYET GUARD'ı (strict/non-strict fark etmez): hiçbir koşulda serbest
        # sermayeyi aşan pozisyon açma. Yukarıdaki mantıkta bir kaçak olsa bile burada durur.
        if position_cost > free_capital + 0.01:
            print(f"[AUTO-TRADE] {sym} atlandi — sermaye guard: maliyet {position_cost:.0f} > serbest {free_capital:.0f} TL")
            _log_decision(uid, sym, 'SKIP', 'budget_overflow',
                          detail=f"son guard: maliyet={position_cost:.0f} > serbest={free_capital:.0f} TL",
                          tf=_tf_now, price=price, score=cand['score'], confidence=cand['confidence'])
            continue

        # ===== AI İKİNCİL İNCELEME (opsiyonel, sadece top-N güçlü aday) =====
        # AI pozisyon AÇMAZ; sadece yorumlar. WAIT/REJECT → aday bloklanır (açılmaz).
        # APPROVE → normal akış + Telegram kartına AI özeti eklenir. Hata → fail-mode.
        ai_verdict = None
        if _ai_enabled and _ai_reviews_done < _ai_top_n and cand['score'] >= _ai_min_score:
            _ai_reviews_done += 1
            try:
                _rr_val = round((tp1 - price) / (price - stop_loss), 2) if (price - stop_loss) > 0 else None
                _regime_ctx = None
                try:
                    from signals import calc_market_regime as _cmr2
                    _rg = _cmr2()
                    _regime_ctx = f"{_rg.get('regime')}/{_rg.get('strength')}"
                except Exception:
                    pass
                ai_verdict = _air.review_trade_candidate({
                    'symbol': sym, 'name': cand.get('name'), 'signal': cand.get('signal'),
                    'price': price, 'score': cand['score'], 'confidence': cand['confidence'],
                    'stop_loss': stop_loss, 'tp1': tp1, 'tp2': tp2, 'tp3': tp3,
                    'rr': _rr_val, 'quantity': quantity, 'position_cost': position_cost,
                    'regime': _regime_ctx, 'score_breakdown': cand.get('score_breakdown'),
                })
            except Exception as _ai_err:
                print(f"[AUTO-TRADE] {sym} AI review beklenmeyen hata: {_ai_err}")
                ai_verdict = None

            if ai_verdict is not None:
                _dec = ai_verdict.get('decision', 'WAIT')
                _is_fallback = bool(ai_verdict.get('_fallback'))
                # Devam koşulu: APPROVE ya da (AI çağrılamadı + fail_mode=manual_review → insana bırak)
                _proceed = (_dec == 'APPROVE_CANDIDATE') or (_is_fallback and _ai_fail_mode == 'manual_review')
                if not _proceed:
                    print(f"[AUTO-TRADE] {sym} AI vetosu: {_dec} — pozisyon acilmadi")
                    _log_decision(uid, sym, 'SKIP', 'ai_review',
                                  detail=f"AI={_dec} conf={ai_verdict.get('confidence')} "
                                         f"ozet={(ai_verdict.get('summary') or '')[:120]}",
                                  tf=_tf_now, price=price, score=cand['score'], confidence=cand['confidence'])
                    # Bilgi amaçlı Telegram (BUTON YOK — sadece "AI önermedi", sessiz değil)
                    if _tg_ok:
                        try:
                            from telegram_notifications import send_telegram as _st_info
                            _st_info(f"🤖 <b>{sym}</b> AI ikinci-göz: <b>{_dec}</b>\n"
                                     f"{ai_verdict.get('telegram_text') or ai_verdict.get('summary') or ''}\n"
                                     f"<i>Pozisyon açılmadı — istersen manuel değerlendir.</i>")
                        except Exception:
                            pass
                    continue
                # Limit nedeniyle atlandıysa kullanıcıya not düş (manuel akışa devam ediyoruz)
                if _is_fallback:
                    print(f"[AUTO-TRADE] {sym} AI atlandi ({ai_verdict.get('_reason')}) — manuel incelemeye gonderiliyor")

        # FAZ 2 (YER A) — AI conviction tilt: taban tavanını AI'nın güvenine göre
        # DAR bir bantta KÜÇÜLT (asla büyütme). Sadece APPROVE + gerçek (non-fallback)
        # verdict'te uygulanır; AI kapalı/başarısız → tam tavan korunur. AI emir açmaz,
        # sadece öneri boyutunu ayarlar; nihai onay yine kullanıcıda kalır.
        if (ai_verdict is not None and not ai_verdict.get('_fallback')
                and _int_env('AUTO_AI_SIZE_TILT', 1) == 1):
            _conf = float(ai_verdict.get('confidence', 0) or 0)
            _factor = max(0.6, min(1.0, _conf / 100.0)) if _conf else 0.85
            _tilted = int(quantity * _factor)
            if 1 <= _tilted < quantity:
                print(f"[AUTO-TRADE] {sym} AI tilt: adet {quantity} -> {_tilted} "
                      f"(guven %{_conf:.0f}, faktor {_factor:.2f})")
                _log_decision(uid, sym, 'INFO', 'ai_size_tilt',
                              detail=f"adet {quantity}->{_tilted}, AI guven %{_conf:.0f}, faktor {_factor:.2f}",
                              tf=_tf_now, price=price, score=cand['score'], confidence=cand['confidence'])
                quantity = _tilted
                position_cost = quantity * price

        _tg_sent = False
        _already_pending = False
        _tg_configured = False
        try:
            from routes_telegram import send_trade_signal
            from telegram_state import has_pending_signal
            import os as _os
            _tg_configured = bool(_os.environ.get('TELEGRAM_BOT_TOKEN')
                                  and _os.environ.get('TELEGRAM_CHAT_ID'))
            if _tg_configured:
                # O4: helper ile thread-safe check (eski O(N) inline scan'in
                # yerine encapsulated; pending state telegram_state'te tutuluyor)
                _already_pending = has_pending_signal(uid, sym)
                if not _already_pending:
                    # Hızlı-trade kurulum kalitesi (deneysel) — kullanıcının gerçek edge'i
                    try:
                        from signal_calibration import setup_quality_from_df
                        _sq, _ = setup_quality_from_df(cand.get('hist'))
                    except Exception:
                        _sq = None
                    _tg_sent = send_trade_signal(
                        uid, sym, price, quantity,
                        cand['score'], cand['confidence'],
                        stop_loss, tp1, tp2, tp3, trailing_sl,
                        ai_verdict=ai_verdict,  # APPROVE ise AI özeti karta eklenir (None ise eski kart)
                        setup_q=_sq,
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
