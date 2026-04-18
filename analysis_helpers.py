"""
Analysis Compute Helper Fonksiyonları
_compute_signal_for_stock, _compute_opportunity_for_stock, _compute_strategy_for_stock
routes_analysis.py'dan ayrıştırıldı (700 satır kuralı).
"""
import threading
import numpy as np
import pandas as pd
from config import (
    BIST100_STOCKS, BIST30, SECTOR_MAP, PARALLEL_WORKERS,
    _cget, _cget_hist, _plan_lock_cache, _plan_lock_cache_lock,
    _get_stocks, PLAN_LOCK_CONFIG, CACHE_TTL,
    sf, si, safe_dict,
)
from api_utils import _api_meta

try:
    from indicators import (
        calc_rsi, calc_rsi_single, calc_macd, calc_bollinger,
        calc_ema, calc_stochastic, calc_atr, calc_adx,
        calc_support_resistance, calc_all_indicators,
        calc_mtf_signal, calc_divergence, calc_volume_profile,
        calc_smc, calc_chart_patterns, calc_fibonacci_adv,
        calc_pivot_points_adv, calc_advanced_indicators,
        calc_dynamic_thresholds, calc_candlestick_patterns,
    )
except ImportError as e:
    print(f"[HATA] analysis_helpers indicators import: {e}")
try:
    from signals import (
        calc_recommendation, calc_52w, calc_market_regime,
        check_signal_alerts, calc_ml_confidence,
    )
except ImportError as e:
    print(f"[HATA] analysis_helpers signals import: {e}")
try:
    from trade_plans import calc_trade_plan
except ImportError as e:
    print(f"[HATA] analysis_helpers trade_plans import: {e}")

# Module-level caches for expensive computations
_signals_cache = {'data': None, 'ts': 0}
_signals_cache_lock = threading.Lock()
_opps_cache = {'data': None, 'ts': 0}
_opps_cache_lock = threading.Lock()
_strat_cache = {'data': None, 'ts': 0}
_strat_cache_lock = threading.Lock()
COMPUTED_CACHE_TTL = 120  # 2 dakika


# =====================================================================
# SINYAL TARAMA YARDIMCI FONKSİYONLARI
# =====================================================================

def _compute_signal_for_stock(stock, timeframe):
    """Tek hisse icin sinyal hesapla (thread-safe, paralel calisir)"""
    sym = stock['code']
    try:
        hist = _cget_hist(f"{sym}_1y")
        if hist is None:
            return None

        c = hist['Close'].values.astype(float)
        cp = float(c[-1])

        ind = calc_all_indicators(hist, cp)
        summary = ind.get('summary', {})
        buy_count = summary.get('buySignals', 0)
        sell_count = summary.get('sellSignals', 0)
        total_ind = summary.get('totalIndicators', 1)

        rec = calc_recommendation(hist, ind, symbol=sym)
        tf_rec = rec.get(timeframe, {})
        action = tf_rec.get('action', 'NOTR')
        score = float(tf_rec.get('score', 0))
        confidence = float(tf_rec.get('confidence', 0))

        consensus_pct = (buy_count / total_ind * 100) if total_ind > 0 else 50
        if score < 0:
            consensus_pct = 100 - consensus_pct
        composite = min(100, (abs(score) / 12 * 60) + (consensus_pct * 0.4))

        rsi_val = ind.get('rsi', {}).get('value', 50)
        macd_type = ind.get('macd', {}).get('signalType', 'neutral')
        macd_hist = ind.get('macd', {}).get('histogram', 0)

        sr = calc_support_resistance(hist)
        supports = sr.get('supports', [])[:2]
        resistances = sr.get('resistances', [])[:2]
        stop_loss = supports[0] if supports else sf(cp * 0.95)
        target = resistances[0] if resistances else sf(cp * 1.10)

        sig_type = 'buy' if score > 0 else 'sell'
        ml_conf = calc_ml_confidence(hist, ind, score, sig_type, symbol=sym)
        candle_patterns = ind.get('candlestick', {}).get('patterns', [])

        try:
            mtf = calc_mtf_signal(hist)
        except Exception:
            mtf = {'mtfScore': 0, 'mtfAlignment': '0/3', 'mtfDirection': 'neutral', 'mtfStrength': 'Uyumsuz'}

        mtf_direction = mtf.get('mtfDirection', 'neutral')
        if mtf_direction != 'neutral' and mtf_direction != sig_type:
            composite = composite * 0.6
        elif mtf_direction == sig_type and mtf.get('mtfScore', 0) == 3:
            composite = min(100, composite * 1.2)

        try:
            div = calc_divergence(hist)
        except Exception:
            div = {'summary': {'signal': 'neutral', 'bullish': 0, 'bearish': 0, 'hasRecent': False}}
        div_summary = div.get('summary', {})
        div_signal  = div_summary.get('signal', 'neutral')

        if div_summary.get('hasRecent', False):
            if div_signal == sig_type:
                composite = min(100, composite * 1.10)
            elif div_signal != 'neutral':
                composite = composite * 0.85

        try:
            vp = calc_volume_profile(hist)
        except Exception:
            vp = {'vwap': 0, 'poc': 0, 'vah': 0, 'val': 0, 'volumeAnomaly': False, 'vwapSignal': 'neutral'}

        vwap_sig = vp.get('vwapSignal', 'neutral')
        if vwap_sig == sig_type:
            composite = min(100, composite * 1.05)
        elif vwap_sig != 'neutral':
            composite = composite * 0.95

        if vp.get('volumeAnomaly', False) and sig_type == 'sell':
            composite = min(100, composite * 1.08)

        try:
            smc = calc_smc(hist)
        except Exception:
            smc = {'signal': 'neutral', 'structureTrend': 'neutral', 'bullScore': 0, 'bearScore': 0,
                   'bosEvents': [], 'chochEvents': [], 'fvgs': [], 'orderBlocks': [],
                   'summary': {'hasBOS': False, 'hasCHoCH': False, 'activeFvgCount': 0, 'activeObCount': 0}}
        smc_signal  = smc.get('signal', 'neutral')
        smc_summary = smc.get('summary', {})

        if smc_summary.get('hasCHoCH', False):
            choch_types = [cc.get('type', '') for cc in smc.get('chochEvents', [])]
            if any('bullish' in t for t in choch_types) and sig_type == 'buy':
                composite = min(100, composite * 1.15)
            elif any('bearish' in t for t in choch_types) and sig_type == 'sell':
                composite = min(100, composite * 1.15)
        if smc_summary.get('hasBOS', False) and smc_signal == sig_type:
            composite = min(100, composite * 1.08)

        try:
            patt = calc_chart_patterns(hist)
        except Exception:
            patt = {'signal': 'neutral', 'patterns': [], 'completedPatterns': [],
                    'summary': {'total': 0, 'bullish': 0, 'bearish': 0, 'completed': 0}}
        patt_signal  = patt.get('signal', 'neutral')
        patt_summary = patt.get('summary', {})

        if patt_summary.get('completed', 0) > 0:
            if patt_signal == sig_type:
                composite = min(100, composite * 1.20)
            elif patt_signal != 'neutral':
                composite = composite * 0.75
        elif patt_signal == sig_type:
            composite = min(100, composite * 1.05)

        try:
            fib    = calc_fibonacci_adv(hist)
            pivots = calc_pivot_points_adv(hist)
        except Exception:
            fib    = {'trend': 'neutral', 'goldenPocket': {'inZone': False}}
            pivots = {'bias': 'neutral', 'classic': {}}

        piv_bias = pivots.get('bias', 'neutral')
        if piv_bias == sig_type:
            composite = min(100, composite * 1.05)
        elif piv_bias != 'neutral':
            composite = composite * 0.97

        in_golden = fib.get('goldenPocket', {}).get('inZone', False)
        fib_trend = fib.get('trend', 'neutral')
        if in_golden:
            if fib_trend == 'uptrend' and sig_type == 'buy':
                composite = min(100, composite * 1.12)
            elif fib_trend == 'downtrend' and sig_type == 'sell':
                composite = min(100, composite * 1.08)

        try:
            adv = calc_advanced_indicators(hist)
        except Exception:
            adv = {'summary': {'signal': 'neutral', 'buyCount': 0, 'sellCount': 0}}

        adv_summary = adv.get('summary', {})
        adv_signal  = adv_summary.get('signal', 'neutral')
        adv_buy     = adv_summary.get('buyCount', 0)
        adv_sell    = adv_summary.get('sellCount', 0)

        if adv_signal == sig_type:
            if adv_buy == 3 or adv_sell == 3:
                composite = min(100, composite * 1.15)
            else:
                composite = min(100, composite * 1.07)
        elif adv_signal != 'neutral':
            composite = composite * 0.90

        ich = adv.get('ichimoku', {})
        if ich.get('belowCloud', False) and sig_type == 'sell':
            composite = min(100, composite * 1.08)
        elif ich.get('aboveCloud', False) and sig_type == 'buy':
            composite = min(100, composite * 1.08)

        return {
            'code': sym,
            'name': BIST100_STOCKS.get(sym, sym),
            'price': sf(cp),
            'changePct': stock.get('changePct', 0),
            'volume': stock.get('volume', 0),
            'action': action,
            'score': sf(score),
            'confidence': sf(confidence),
            'composite': sf(composite),
            'mlConfidence': ml_conf.get('confidence', 50),
            'mlGrade': ml_conf.get('grade', 'C'),
            'buySignals': buy_count,
            'sellSignals': sell_count,
            'totalIndicators': total_ind,
            'rsi': sf(rsi_val),
            'macdSignal': macd_type,
            'macdHistogram': macd_hist,
            'stopLoss': stop_loss,
            'target': target,
            'supports': supports,
            'resistances': resistances,
            'reasons': tf_rec.get('reasons', [])[:5],
            'reason': tf_rec.get('reason', ''),
            'indicatorBreakdown': tf_rec.get('indicatorBreakdown', {}),
            'strategy': tf_rec.get('strategy', ''),
            'candlestickPatterns': candle_patterns[:3],
            'dynamicThresholds': ind.get('dynamicThresholds', {}),
            'tradePlan': calc_trade_plan(hist, ind, symbol=sym),
            'mtfScore':       mtf.get('mtfScore', 0),
            'mtfAlignment':   mtf.get('mtfAlignment', '0/3'),
            'mtfDirection':   mtf_direction,
            'mtfStrength':    mtf.get('mtfStrength', 'Uyumsuz'),
            'mtfDescription': mtf.get('description', ''),
            'divergenceSignal':  div_signal,
            'divergenceCount':   div_summary.get('count', 0),
            'divergenceBullish': div_summary.get('bullish', 0),
            'divergenceBearish': div_summary.get('bearish', 0),
            'hasRecentDivergence': div_summary.get('hasRecent', False),
            'divergences':       div.get('recentDivergences', [])[:3],
            'vwap':           vp.get('vwap', 0),
            'vwapSignal':     vwap_sig,
            'vwapPct':        vp.get('vwapPct', 0),
            'poc':            vp.get('poc', 0),
            'vah':            vp.get('vah', 0),
            'val':            vp.get('val', 0),
            'volumeAnomaly':  vp.get('volumeAnomaly', False),
            'volumeRatio':    vp.get('volumeRatio', 0),
            'volumeTrend':    vp.get('volumeTrend', ''),
            'smcSignal':         smc_signal,
            'smcStructure':      smc.get('structureTrend', 'neutral'),
            'smcBullScore':      smc.get('bullScore', 0),
            'smcBearScore':      smc.get('bearScore', 0),
            'hasBOS':            smc_summary.get('hasBOS', False),
            'hasCHoCH':          smc_summary.get('hasCHoCH', False),
            'activeFvgCount':    smc_summary.get('activeFvgCount', 0),
            'activeObCount':     smc_summary.get('activeObCount', 0),
            'smcEntryZones':     smc.get('entryZones', [])[:3],
            'patternSignal':     patt_signal,
            'patternCount':      patt_summary.get('total', 0),
            'completedPatterns': patt_summary.get('completed', 0),
            'bullishPatterns':   patt_summary.get('bullish', 0),
            'bearishPatterns':   patt_summary.get('bearish', 0),
            'patterns':          patt.get('completedPatterns', [])[:2] + patt.get('pendingPatterns', [])[:2],
            'fibTrend':          fib.get('trend', 'neutral'),
            'fibSwingHigh':      fib.get('swingHigh', 0),
            'fibSwingLow':       fib.get('swingLow', 0),
            'inGoldenPocket':    fib.get('goldenPocket', {}).get('inZone', False),
            'fibNearestSupport': (fib.get('nearestSupports', [{}]) or [{}])[0].get('level', 0),
            'fibNearestResist':  (fib.get('nearestResistances', [{}]) or [{}])[0].get('level', 0),
            'pivotBias':         piv_bias,
            'pivotPP':           pivots.get('classic', {}).get('pp', 0),
            'pivotR1':           pivots.get('classic', {}).get('r1', 0),
            'pivotS1':           pivots.get('classic', {}).get('s1', 0),
            'advSignal':         adv_signal,
            'advBuyCount':       adv_buy,
            'advSellCount':      adv_sell,
            'ichimokuSignal':    adv.get('ichimoku', {}).get('signal', 'neutral'),
            'aboveCloud':        adv.get('ichimoku', {}).get('aboveCloud', False),
            'belowCloud':        adv.get('ichimoku', {}).get('belowCloud', False),
            'stochasticK':       adv.get('stochastic', {}).get('k', 50),
            'stochasticSignal':  adv.get('stochastic', {}).get('signal', 'neutral'),
            'williamsR':         adv.get('williamsR', {}).get('value', -50),
            'williamsSignal':    adv.get('williamsR', {}).get('signal', 'neutral'),
        }
    except Exception as e:
        print(f"[SIGNAL-COMPUTE] {sym} hata: {e}")
        return None


def _compute_opportunity_for_stock(stock):
    """Tek hisse icin firsat analizi - ADX, RSI divergence, Stochastic, trend alignment."""
    sym = stock['code']
    try:
        hist = _cget_hist(f"{sym}_1y")
        if hist is None:
            return None

        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        cp = float(c[-1])
        n = len(c)

        if n < 30:
            return None

        events = []
        event_score = 0
        buy_count = 0
        sell_count = 0

        rsi_val = calc_rsi(c).get('value', 50)
        if rsi_val < 30:
            events.append({'type': 'rsi_oversold', 'text': f'RSI {sf(rsi_val)} - Asiri satim bolgesinde, toparlanma bekleniyor', 'impact': 'positive'})
            event_score += 3; buy_count += 1
        elif rsi_val > 70:
            events.append({'type': 'rsi_overbought', 'text': f'RSI {sf(rsi_val)} - Asiri alim bolgesinde, duzeltme gelebilir', 'impact': 'negative'})
            event_score -= 3; sell_count += 1

        if n >= 60:
            recent_start = n - 30
            prior_start = n - 60
            recent_min_abs = recent_start + int(np.argmin(c[recent_start:]))
            prior_min_abs = prior_start + int(np.argmin(c[prior_start:recent_start]))
            if recent_min_abs > 14 and prior_min_abs > 14:
                rsi_at_recent_low = calc_rsi_single(c[:recent_min_abs + 1]) or 50
                rsi_at_prior_low  = calc_rsi_single(c[:prior_min_abs + 1]) or 50
                price_lower_low = float(c[recent_min_abs]) < float(c[prior_min_abs]) * 0.99
                rsi_higher_low  = rsi_at_recent_low > rsi_at_prior_low + 3
                if price_lower_low and rsi_higher_low and rsi_val < 50:
                    events.append({'type': 'bullish_divergence', 'text': f'RSI Yukselis Uyumsuzlugu: Fiyat dip yaparken RSI yukseldi ({sf(rsi_at_prior_low)}->{sf(rsi_at_recent_low)}) - Guclu alis sinyali', 'impact': 'very_positive'})
                    event_score += 4; buy_count += 1
            recent_max_abs = recent_start + int(np.argmax(c[recent_start:]))
            prior_max_abs  = prior_start + int(np.argmax(c[prior_start:recent_start]))
            if recent_max_abs > 14 and prior_max_abs > 14:
                rsi_at_recent_high = calc_rsi_single(c[:recent_max_abs + 1]) or 50
                rsi_at_prior_high  = calc_rsi_single(c[:prior_max_abs + 1]) or 50
                price_higher_high = float(c[recent_max_abs]) > float(c[prior_max_abs]) * 1.01
                rsi_lower_high    = rsi_at_recent_high < rsi_at_prior_high - 3
                if price_higher_high and rsi_lower_high and rsi_val > 50:
                    events.append({'type': 'bearish_divergence', 'text': f'RSI Dusus Uyumsuzlugu: Fiyat zirve yaparken RSI dustü ({sf(rsi_at_prior_high)}->{sf(rsi_at_recent_high)}) - Guclu satis sinyali', 'impact': 'very_negative'})
                    event_score -= 4; sell_count += 1

        macd = calc_macd(c)
        if macd.get('signalType') == 'buy':
            events.append({'type': 'macd_cross', 'text': 'MACD alis kesisimi - Yukari momentum basladi', 'impact': 'positive'})
            event_score += 2; buy_count += 1
        elif macd.get('signalType') == 'sell':
            events.append({'type': 'macd_cross', 'text': 'MACD satis kesisimi - Asagi momentum basladi', 'impact': 'negative'})
            event_score -= 2; sell_count += 1

        if n >= 200:
            ema50  = pd.Series(c).ewm(span=50).mean().values
            ema200 = pd.Series(c).ewm(span=200).mean().values
            if ema50[-1] > ema200[-1] and ema50[-2] <= ema200[-2]:
                events.append({'type': 'golden_cross', 'text': 'ALTIN KESISIM! EMA50 > EMA200 - Guclu uzun vadeli alis sinyali', 'impact': 'very_positive'})
                event_score += 5; buy_count += 1
            elif ema50[-1] < ema200[-1] and ema50[-2] >= ema200[-2]:
                events.append({'type': 'death_cross', 'text': 'OLUM KESISIMI! EMA50 < EMA200 - Guclu uzun vadeli satis sinyali', 'impact': 'very_negative'})
                event_score -= 5; sell_count += 1

        adx_data = calc_adx(h, l, c)
        adx_val  = float(adx_data.get('value', 25))
        plus_di  = float(adx_data.get('plusDI', 0))
        minus_di = float(adx_data.get('minusDI', 0))
        sideways_market = adx_val < 15
        if adx_val > 30:
            if plus_di > minus_di:
                events.append({'type': 'adx_strong_bull', 'text': f'ADX={sf(adx_val)} - Guclu yukselis trendi (+DI={sf(plus_di)} > -DI={sf(minus_di)})', 'impact': 'positive'})
                event_score += 2; buy_count += 1
            else:
                events.append({'type': 'adx_strong_bear', 'text': f'ADX={sf(adx_val)} - Guclu dusus trendi (-DI={sf(minus_di)} > +DI={sf(plus_di)})', 'impact': 'negative'})
                event_score -= 2; sell_count += 1
        elif sideways_market:
            event_score = int(event_score * 0.7)

        stoch   = calc_stochastic(c, h, l)
        stoch_k = float(stoch.get('k', 50))
        if stoch_k < 20:
            events.append({'type': 'stoch_oversold', 'text': f'Stochastic %K={sf(stoch_k)} - Asiri satim bolgesinde, donus bekleniyor', 'impact': 'positive'})
            event_score += 2; buy_count += 1
        elif stoch_k > 80:
            events.append({'type': 'stoch_overbought', 'text': f'Stochastic %K={sf(stoch_k)} - Asiri alim bolgesinde', 'impact': 'negative'})
            event_score -= 2; sell_count += 1

        if n >= 20:
            vol_avg   = np.mean(v[-20:])
            vol_today = v[-1]
            if vol_avg > 0 and vol_today > vol_avg * 2:
                ratio     = sf(vol_today / vol_avg)
                direction = 'yukselis' if c[-1] > c[-2] else 'dusus'
                impact    = 'positive' if c[-1] > c[-2] else 'negative'
                events.append({'type': 'volume_spike', 'text': f'Hacim patlamasi ({ratio}x ortalama) + {direction} hareketi', 'impact': impact})
                if c[-1] > c[-2]:
                    event_score += 2; buy_count += 1
                else:
                    event_score -= 2; sell_count += 1

        bb = calc_bollinger(c, cp)
        if bb.get('lower', 0) > 0 and cp < bb['lower']:
            events.append({'type': 'bb_break_lower', 'text': f'Fiyat alt Bollinger bandinin altinda ({sf(bb["lower"])}) - Toparlanma bekleniyor', 'impact': 'positive'})
            event_score += 2; buy_count += 1
        elif bb.get('upper', 0) > 0 and cp > bb['upper']:
            events.append({'type': 'bb_break_upper', 'text': f'Fiyat ust Bollinger bandini asti ({sf(bb["upper"])}) - Asiri alim', 'impact': 'negative'})
            event_score -= 1; sell_count += 1

        w52     = calc_52w(hist)
        w52_pos = w52.get('currentPct', 50)
        if w52_pos < 10:
            events.append({'type': '52w_low', 'text': f'52 haftalik dibin %{sf(w52_pos)} uzerinde - Tarihi dip bolgesi', 'impact': 'positive'})
            event_score += 2; buy_count += 1
        elif w52_pos > 90:
            events.append({'type': '52w_high', 'text': f'52 haftalik zirveye %{sf(100 - w52_pos)} mesafede', 'impact': 'neutral'})

        sr = calc_support_resistance(hist)
        if sr.get('resistances'):
            nearest_res = sr['resistances'][0]
            if cp > nearest_res * 0.99 and c[-2] < nearest_res:
                events.append({'type': 'resistance_break', 'text': f'Direnc kirdi ({sf(nearest_res)} TL) - Yukari kirilim', 'impact': 'positive'})
                event_score += 3; buy_count += 1
        if sr.get('supports'):
            nearest_sup = sr['supports'][0]
            if cp < nearest_sup * 1.01 and c[-2] > nearest_sup:
                events.append({'type': 'support_break', 'text': f'Destek kirildi ({sf(nearest_sup)} TL) - Asagi kirilim', 'impact': 'negative'})
                event_score -= 3; sell_count += 1

        if n >= 50:
            s_pd    = pd.Series(c)
            ema20_val = float(s_pd.ewm(span=20).mean().iloc[-1])
            ema50_val = float(s_pd.ewm(span=50).mean().iloc[-1])
            uptrend   = cp > ema20_val > ema50_val
            downtrend = cp < ema20_val < ema50_val
            if event_score > 0 and uptrend:
                events.append({'type': 'trend_aligned_bull', 'text': 'Trend teyidi: Yukselis trendinde alis sinyali (EMA20 > EMA50) - Guclu uyum', 'impact': 'positive'})
                event_score += 2
            elif event_score > 0 and downtrend:
                events.append({'type': 'trend_counter_bull', 'text': 'Trend uyarisi: Dusus trendinde alis denemesi (EMA20 < EMA50) - Dikkat!', 'impact': 'neutral'})
                event_score -= 1
            elif event_score < 0 and downtrend:
                events.append({'type': 'trend_aligned_bear', 'text': 'Trend teyidi: Dusus trendinde satis sinyali (EMA20 < EMA50) - Guclu uyum', 'impact': 'negative'})
                event_score -= 2
            elif event_score < 0 and uptrend:
                events.append({'type': 'trend_counter_bear', 'text': 'Trend uyarisi: Yukselis trendinde satis denemesi (EMA20 > EMA50) - Dikkat!', 'impact': 'neutral'})
                event_score += 1

        o_arr   = hist['Open'].values.astype(float) if 'Open' in hist.columns else c.copy()
        candles = calc_candlestick_patterns(o_arr, h, l, c)
        for p in candles.get('patterns', []):
            impact = 'positive' if p['type'] == 'bullish' else ('negative' if p['type'] == 'bearish' else 'neutral')
            events.append({'type': 'candlestick', 'text': f"Mum Formasyonu: {p['name']} - {p['description']}", 'impact': impact})
            if p['type'] == 'bullish':
                event_score += p['strength']; buy_count += 1
            elif p['type'] == 'bearish':
                event_score -= p['strength']; sell_count += 1

        dominant_count = max(buy_count, sell_count)
        if dominant_count < 2:
            return None

        MIN_SCORE = 4
        if abs(event_score) < MIN_SCORE:
            return None

        MAX_POSSIBLE = 28
        opp_score     = min(100, int(abs(event_score) / MAX_POSSIBLE * 100))
        opp_direction = 'buy' if event_score > 0 else 'sell'

        returns = {}
        for label, days in [('daily', 1), ('weekly', 5), ('monthly', 22), ('yearly', 252)]:
            actual_days = min(days, n - 1)
            if actual_days > 0:
                ret = ((c[-1] - c[-1 - actual_days]) / c[-1 - actual_days]) * 100
                returns[label] = sf(ret)
            else:
                returns[label] = 0

        dyn = calc_dynamic_thresholds(c, h, l, v)

        return {
            'code': sym,
            'name': BIST100_STOCKS.get(sym, sym),
            'price': sf(cp),
            'changePct': stock.get('changePct', 0),
            'eventScore': event_score,
            'opportunityScore': opp_score,
            'direction': opp_direction,
            'events': events,
            'eventCount': len(events),
            'buySignals': buy_count,
            'sellSignals': sell_count,
            'returns': returns,
            'rsi': sf(rsi_val),
            'adx': sf(adx_val),
            'stochastic': sf(stoch_k),
            'macdSignal': macd.get('signalType', 'neutral'),
            'sidewaysMarket': sideways_market,
            'dynamicThresholds': dyn,
            'candlestickPatterns': candles.get('patterns', []),
            'tradePlan': calc_trade_plan(hist, symbol=sym),
        }
    except Exception as e:
        print(f"[OPP-COMPUTE] {sym} hata: {e}")
        return None


def _compute_strategy_for_stock(stock):
    """Tek hisse icin 3 strateji hesapla (thread-safe)"""
    sym = stock['code']
    try:
        hist = _cget_hist(f"{sym}_1y")
        if hist is None:
            return None

        c  = hist['Close'].values.astype(float)
        h  = hist['High'].values.astype(float)
        l  = hist['Low'].values.astype(float)
        cp = float(c[-1])
        n  = len(c)
        result = {'ma_cross': None, 'breakout': None, 'mean_reversion': None}

        if n >= 50:
            s     = pd.Series(c)
            ema20 = s.ewm(span=20).mean().values
            ema50 = s.ewm(span=50).mean().values
            signal = None
            if ema20[-1] > ema50[-1] and ema20[-2] <= ema50[-2]: signal = 'AL'
            elif ema20[-1] < ema50[-1] and ema20[-2] >= ema50[-2]: signal = 'SAT'
            elif ema20[-1] > ema50[-1]: signal = 'ALIS POZISYONUNDA'
            elif ema20[-1] < ema50[-1]: signal = 'SATIS POZISYONUNDA'
            distance = sf(((ema20[-1] - ema50[-1]) / ema50[-1]) * 100)
            result['ma_cross'] = {
                'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                'price': sf(cp), 'changePct': stock.get('changePct', 0),
                'signal': signal, 'ema20': sf(ema20[-1]), 'ema50': sf(ema50[-1]),
                'distance': distance, 'freshSignal': signal in ('AL', 'SAT'),
            }

        if n >= 20:
            high_20 = float(np.max(h[-20:]))
            low_20  = float(np.min(l[-20:]))
            signal  = None
            if cp >= high_20 * 0.99: signal = 'YUKARI KIRILIM'
            elif cp <= low_20 * 1.01: signal = 'ASAGI KIRILIM'
            else:
                pos    = ((cp - low_20) / (high_20 - low_20) * 100) if high_20 != low_20 else 50
                signal = f'BANT ICINDE (%{sf(pos)})'
            result['breakout'] = {
                'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                'price': sf(cp), 'changePct': stock.get('changePct', 0),
                'signal': signal, 'high20': sf(high_20), 'low20': sf(low_20),
                'freshSignal': 'KIRILIM' in (signal or ''),
            }

        if n >= 15:
            rsi    = calc_rsi(c).get('value', 50)
            signal = None
            if rsi < 30: signal = 'ASIRI SATIM → AL'
            elif rsi < 40: signal = 'SATIM BOLGESI → ALIS FIRSATI'
            elif rsi > 70: signal = 'ASIRI ALIM → SAT'
            elif rsi > 60: signal = 'ALIM BOLGESI → DIKKAT'
            else: signal = 'NOTR BOLGE'
            result['mean_reversion'] = {
                'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                'price': sf(cp), 'changePct': stock.get('changePct', 0),
                'signal': signal, 'rsi': sf(rsi), 'freshSignal': rsi < 30 or rsi > 70,
            }

        return result
    except Exception as e:
        print(f"[STRAT-COMPUTE] {sym} hata: {e}")
        return None


# =====================================================================
# ROUTES
# Backtest, heatmap, rapor, temettü, sinyal performansı ve kalibrasyon
# → routes_analysis_reports.py'a taşındı (700 satır kuralı)
# =====================================================================

