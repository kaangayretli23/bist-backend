"""
Heavy analysis routes: signals, opportunities, strategies, backtest, heatmap, report, dividends, etc.
"""
import threading, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Blueprint, jsonify, request

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pass

try:
    import yfinance as yf
except ImportError:
    yf = None

from config import (
    YF_OK as _YF_OK,
    BIST100_STOCKS, BIST30, SECTOR_MAP, PARALLEL_WORKERS,
    _stock_cache, _hist_cache,
    _cget, _cget_hist, _plan_lock_cache, _plan_lock_cache_lock,
    _get_stocks, _get_indices,
    PLAN_LOCK_CONFIG, CACHE_TTL,
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
    print(f"[HATA] routes_analysis indicators import: {e}")
try:
    from signals import (
        calc_recommendation, calc_52w, calc_signal_backtest,
        calc_market_regime, calc_sector_relative_strength,
        check_signal_alerts, calc_ml_confidence,
    )
except ImportError as e:
    print(f"[HATA] routes_analysis signals import: {e}")
try:
    from trade_plans import calc_trade_plan
except ImportError as e:
    print(f"[HATA] routes_analysis trade_plans import: {e}")
try:
    from data_fetcher import _fetch_hist_df
except ImportError as e:
    print(f"[HATA] routes_analysis data_fetcher import: {e}")

analysis_bp = Blueprint('analysis', __name__)

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

        rec = calc_recommendation(hist, ind)
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
        ml_conf = calc_ml_confidence(hist, ind, score, sig_type)
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
    except Exception:
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
    except Exception:
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
    except Exception:
        return None


# =====================================================================
# ROUTES
# =====================================================================

@analysis_bp.route('/api/backtest', methods=['POST'])
def backtest():
    try:
        d           = request.json or {}
        sym         = d.get('symbol', '').upper()
        strategy    = d.get('strategy', 'ma_cross')
        params      = d.get('params', {})
        period      = d.get('period', '1y')
        commission  = float(d.get('commission', 0.001))
        initial_capital = float(d.get('initialCapital', 100000))

        if not sym:
            return jsonify({'error': 'Hisse kodu gerekli'}), 400

        hist = _fetch_hist_df(sym, period)
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{sym} icin yeterli veri yok'}), 400

        closes  = hist['Close'].values.astype(float)
        highs   = hist['High'].values.astype(float)
        lows    = hist['Low'].values.astype(float)
        dates   = [d2.strftime('%Y-%m-%d') for d2 in hist.index]
        n       = len(closes)
        signals = [0] * n

        if strategy == 'ma_cross':
            fast_p = int(params.get('fast', 20))
            slow_p = int(params.get('slow', 50))
            s = pd.Series(closes)
            fast_ma = s.rolling(fast_p).mean().values
            slow_ma = s.rolling(slow_p).mean().values
            for i in range(slow_p + 1, n):
                if fast_ma[i] > slow_ma[i] and fast_ma[i - 1] <= slow_ma[i - 1]:
                    signals[i] = 1
                elif fast_ma[i] < slow_ma[i] and fast_ma[i - 1] >= slow_ma[i - 1]:
                    signals[i] = -1

        elif strategy == 'breakout':
            lookback = int(params.get('lookback', 20))
            for i in range(lookback, n):
                high_n = max(highs[i - lookback:i])
                low_n  = min(lows[i - lookback:i])
                if closes[i] > high_n: signals[i] = 1
                elif closes[i] < low_n: signals[i] = -1

        elif strategy == 'mean_reversion':
            rsi_low  = float(params.get('rsi_low', 30))
            rsi_high = float(params.get('rsi_high', 70))
            for i in range(15, n):
                rsi = calc_rsi_single(closes[:i + 1])
                if rsi is not None:
                    if rsi < rsi_low: signals[i] = 1
                    elif rsi > rsi_high: signals[i] = -1

        cash    = initial_capital
        shares  = 0
        equity_curve = []
        trades  = []
        peak_equity = initial_capital
        max_dd  = 0
        wins = losses = 0

        for i in range(n):
            price = closes[i]
            if signals[i] == 1 and shares == 0:
                shares = int(cash * (1 - commission) / price)
                if shares > 0:
                    cost = shares * price * (1 + commission)
                    cash -= cost
                    trades.append({'date': dates[i], 'action': 'AL', 'price': sf(price), 'shares': shares, 'pnl': 0})
            elif signals[i] == -1 and shares > 0:
                revenue = shares * price * (1 - commission)
                pnl = revenue - (trades[-1]['shares'] * trades[-1]['price'] * (1 + commission)) if trades else 0
                cash += revenue
                trades.append({'date': dates[i], 'action': 'SAT', 'price': sf(price), 'shares': shares, 'pnl': sf(pnl)})
                if pnl > 0: wins += 1
                else: losses += 1
                shares = 0

            equity = cash + shares * price
            equity_curve.append({'date': dates[i], 'equity': sf(equity)})
            if equity > peak_equity: peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100
            if dd > max_dd: max_dd = dd

        final_equity = cash + shares * closes[-1]
        total_return = sf(((final_equity - initial_capital) / initial_capital) * 100)
        bh_return    = sf(((closes[-1] - closes[0]) / closes[0]) * 100)
        years        = n / 252
        cagr         = sf(((final_equity / initial_capital) ** (1 / years) - 1) * 100) if years > 0 else 0

        daily_returns = np.diff([e['equity'] for e in equity_curve]) / np.array([e['equity'] for e in equity_curve[:-1]])
        sharpe = sf(float(np.mean(daily_returns) / np.std(daily_returns) * (252 ** 0.5))) if len(daily_returns) > 0 and np.std(daily_returns) > 0 else 0

        total_trades = wins + losses
        win_rate     = sf(wins / total_trades * 100) if total_trades > 0 else 0
        alpha        = sf(float(total_return) - float(bh_return))

        in_market_days = sum(1 for i in range(n) if (any(t['action'] == 'AL' and dates.index(t['date']) <= i for t in trades if t['date'] in dates)))
        exposure_pct   = sf(in_market_days / n * 100) if n > 0 else 0

        trade_pnls  = [float(t['pnl']) for t in trades if t['action'] == 'SAT' and t['pnl'] != 0]
        avg_trade   = sf(np.mean(trade_pnls)) if trade_pnls else 0
        avg_win     = sf(np.mean([p for p in trade_pnls if p > 0])) if [p for p in trade_pnls if p > 0] else 0
        avg_loss    = sf(np.mean([p for p in trade_pnls if p < 0])) if [p for p in trade_pnls if p < 0] else 0
        profit_factor = sf(abs(sum(p for p in trade_pnls if p > 0) / sum(p for p in trade_pnls if p < 0))) if any(p < 0 for p in trade_pnls) else 999

        return jsonify(safe_dict({
            'success': True,
            'results': {
                'totalReturn': total_return, 'cagr': cagr, 'sharpeRatio': sharpe,
                'maxDrawdown': sf(-max_dd), 'winRate': win_rate, 'totalTrades': total_trades,
                'buyAndHoldReturn': bh_return, 'alpha': alpha,
                'finalEquity': sf(final_equity), 'initialCapital': sf(initial_capital),
                'exposure': exposure_pct, 'avgTrade': avg_trade,
                'avgWin': avg_win, 'avgLoss': avg_loss, 'profitFactor': profit_factor,
                'commission': sf(commission * 100),
            },
            'equityCurve': equity_curve[::max(1, len(equity_curve) // 200)],
            'trades': trades[-50:],
            'warnings': [
                'Hayatta kalma yanliligi (Survivorship Bias): Bu backtest sadece bugun BIST100 endeksinde bulunan hisseleri kapsamaktadir.',
                'Kayma (Slippage): Gercek islemlerde emir fiyati ile gerceklesen fiyat arasinda fark olabilir.',
                'Komisyon: Backtest %' + str(sf(commission * 100)) + ' komisyon varsayimi kullanmaktadir.',
            ],
        }))
    except Exception as e:
        import traceback
        print(f"[BACKTEST] Hata: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/heatmap')
def heatmap():
    """Sektor bazli isi haritasi verisi"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'sectors': [], 'loading': True})

        stock_map = {s['code']: s for s in stocks}
        sectors   = []
        sector_display = {
            'bankacilik': 'Bankacilik', 'havacilik': 'Havacilik',
            'otomotiv': 'Otomotiv', 'enerji': 'Enerji',
            'holding': 'Holding', 'perakende': 'Perakende',
            'teknoloji': 'Teknoloji', 'telekom': 'Telekom',
            'demir_celik': 'Demir Celik', 'gida': 'Gida',
            'insaat': 'Insaat', 'gayrimenkul': 'Gayrimenkul',
        }

        for sector_name, symbols in SECTOR_MAP.items():
            sector_stocks = []
            total_change  = 0
            count = 0
            for sym in symbols:
                if sym in stock_map:
                    s = stock_map[sym]
                    sector_stocks.append({'code': s['code'], 'name': s['name'], 'price': s['price'], 'changePct': s['changePct'], 'volume': s['volume']})
                    total_change += s['changePct']
                    count += 1
            avg_change = sf(total_change / count) if count > 0 else 0
            sectors.append({
                'name': sector_name,
                'displayName': sector_display.get(sector_name, sector_name),
                'avgChange': avg_change,
                'stockCount': count,
                'stocks': sorted(sector_stocks, key=lambda x: x['changePct'], reverse=True),
            })

        sectors.sort(key=lambda x: x['avgChange'], reverse=True)
        return jsonify(safe_dict({'success': True, 'sectors': sectors}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/report')
def daily_report():
    """Gunluk piyasa raporu"""
    try:
        stocks  = _get_stocks()
        indices = _get_indices()

        if not stocks:
            return jsonify({'success': True, 'loading': True, 'message': 'Veriler yukleniyor'})

        up        = [s for s in stocks if s.get('changePct', 0) > 0]
        down      = [s for s in stocks if s.get('changePct', 0) < 0]
        unchanged = [s for s in stocks if s.get('changePct', 0) == 0]

        sorted_up   = sorted(stocks, key=lambda x: x.get('changePct', 0), reverse=True)
        sorted_down = sorted(stocks, key=lambda x: x.get('changePct', 0))
        sorted_vol  = sorted(stocks, key=lambda x: x.get('volume', 0), reverse=True)

        gap_up   = sorted([s for s in stocks if s.get('gapPct', 0) > 1],  key=lambda x: x.get('gapPct', 0), reverse=True)
        gap_down = sorted([s for s in stocks if s.get('gapPct', 0) < -1], key=lambda x: x.get('gapPct', 0))

        all_changes = [s.get('changePct', 0) for s in stocks]
        avg_change  = sf(np.mean(all_changes)) if all_changes else 0

        sector_perf = []
        stock_map   = {s['code']: s for s in stocks}
        for sname, syms in SECTOR_MAP.items():
            changes = [stock_map[s]['changePct'] for s in syms if s in stock_map]
            if changes:
                sector_perf.append({
                    'name': sname, 'avgChange': sf(np.mean(changes)),
                    'bestStock': max([(s, stock_map[s]['changePct']) for s in syms if s in stock_map], key=lambda x: x[1])[0],
                })
        sector_perf.sort(key=lambda x: x['avgChange'], reverse=True)

        report_lines = []
        bist100 = indices.get('XU100', {})
        if bist100:
            direction = 'yukselis' if bist100.get('changePct', 0) > 0 else 'dusus'
            report_lines.append(f"BIST 100 endeksi %{bist100.get('changePct', 0)} {direction} gosteriyor.")
        report_lines.append(f"Toplam {len(stocks)} hisseden {len(up)} yukselen, {len(down)} dusen, {len(unchanged)} degismez.")
        report_lines.append(f"Piyasa ortalama degisimi: %{avg_change}")
        if sorted_up:   report_lines.append(f"Gunun yildizi: {sorted_up[0]['code']} (%{sorted_up[0]['changePct']})")
        if sorted_down: report_lines.append(f"Gunun kaybi: {sorted_down[0]['code']} (%{sorted_down[0]['changePct']})")
        if sector_perf: report_lines.append(f"En iyi sektor: {sector_perf[0]['name']} (%{sector_perf[0]['avgChange']})")

        return jsonify(safe_dict({
            'success': True,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'summary': ' '.join(report_lines),
            'reportLines': report_lines,
            'marketBreadth': {'advancing': len(up), 'declining': len(down), 'unchanged': len(unchanged), 'total': len(stocks), 'avgChange': avg_change},
            'topGainers':    [{'code': s['code'], 'name': s['name'], 'changePct': s['changePct'], 'price': s['price']} for s in sorted_up[:5]],
            'topLosers':     [{'code': s['code'], 'name': s['name'], 'changePct': s['changePct'], 'price': s['price']} for s in sorted_down[:5]],
            'volumeLeaders': [{'code': s['code'], 'name': s['name'], 'volume': s['volume'], 'changePct': s['changePct']} for s in sorted_vol[:5]],
            'gapUp':         [{'code': s['code'], 'gapPct': s.get('gapPct', 0)} for s in gap_up[:5]],
            'gapDown':       [{'code': s['code'], 'gapPct': s.get('gapPct', 0)} for s in gap_down[:5]],
            'sectorPerformance': sector_perf,
            'indices': indices,
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/signals')
def signal_scanner():
    """Tum hisselerin sinyal taramasi - composite score ile sirali (PARALEL)"""
    try:
        timeframe   = request.args.get('timeframe', 'weekly')
        min_score   = float(request.args.get('minScore', 0))
        signal_type = request.args.get('type', 'all')

        with _signals_cache_lock:
            sc = _signals_cache
            if sc['data'] and (time.time() - sc['ts']) < COMPUTED_CACHE_TTL:
                cached   = sc['data']
                filtered = cached['all_results']
                if signal_type == 'buy':  filtered = [r for r in filtered if float(r['score']) > 0]
                elif signal_type == 'sell': filtered = [r for r in filtered if float(r['score']) < 0]
                if min_score > 0: filtered = [r for r in filtered if abs(float(r['score'])) >= min_score]
                return jsonify(safe_dict({'success': True, 'timeframe': timeframe, 'totalScanned': cached['totalScanned'], 'signalCount': len(filtered), 'signals': filtered, 'marketRegime': calc_market_regime(), 'timestamp': cached['timestamp'], 'meta': _api_meta()}))

        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True, 'signals': [], 'message': 'Veriler yukleniyor...'})

        hist_ready = sum(1 for s in stocks if _cget_hist(f"{s['code']}_1y") is not None)
        if hist_ready < 10:
            return jsonify({'success': True, 'loading': True, 'signals': [], 'message': f'Tarihsel veriler hazirlaniyor ({hist_ready}/{len(stocks)})...'})

        results = []
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_compute_signal_for_stock, s, timeframe): s for s in stocks}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        results.sort(key=lambda x: float(x['score']), reverse=True)

        with _signals_cache_lock:
            _signals_cache['data'] = {'all_results': results, 'totalScanned': len(stocks), 'timestamp': datetime.now().isoformat()}
            _signals_cache['ts'] = time.time()

        filtered = results
        if signal_type == 'buy':   filtered = [r for r in results if float(r['score']) > 0]
        elif signal_type == 'sell': filtered = [r for r in results if float(r['score']) < 0]
        if min_score > 0: filtered = [r for r in filtered if abs(float(r['score'])) >= min_score]

        return jsonify(safe_dict({'success': True, 'timeframe': timeframe, 'totalScanned': len(stocks), 'signalCount': len(filtered), 'signals': filtered, 'marketRegime': calc_market_regime(), 'timestamp': datetime.now().isoformat(), 'meta': _api_meta()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/opportunities')
def opportunities():
    """Coklu zaman dilimli firsat raporu - PARALEL hesaplama"""
    try:
        with _opps_cache_lock:
            oc = _opps_cache
            if oc['data'] and (time.time() - oc['ts']) < COMPUTED_CACHE_TTL:
                return jsonify(safe_dict(oc['data']))

        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True, 'message': 'Veriler yukleniyor...'})

        hist_ready = sum(1 for s in stocks if _cget_hist(f"{s['code']}_1y") is not None)
        if hist_ready < 10:
            return jsonify({'success': True, 'loading': True, 'message': f'Tarihsel veriler hazirlaniyor ({hist_ready}/{len(stocks)})...'})

        opportunities_list = []
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_compute_opportunity_for_stock, s): s for s in stocks}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    opportunities_list.append(result)

        opportunities_list.sort(key=lambda x: x.get('opportunityScore', abs(x['eventScore'])), reverse=True)
        buy_opps  = [o for o in opportunities_list if o['eventScore'] > 0]
        sell_opps = [o for o in opportunities_list if o['eventScore'] < 0]

        result_data = {
            'success': True,
            'totalScanned': len(stocks),
            'buyOpportunities': buy_opps[:20],
            'sellOpportunities': sell_opps[:20],
            'marketRegime': calc_market_regime(),
            'timestamp': datetime.now().isoformat(),
            'meta': _api_meta(),
        }

        with _opps_cache_lock:
            _opps_cache['data'] = result_data
            _opps_cache['ts']   = time.time()

        return jsonify(safe_dict(result_data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/locked-plans')
def locked_plans():
    """Kilitli yatirım planlarını döner (plan kilitleme sistemi)"""
    try:
        now    = time.time()
        result = []
        with _plan_lock_cache_lock:
            items = list(_plan_lock_cache.items())
        for key, entry in items:
            parts = key.rsplit('_', 1)
            sym   = parts[0]
            tf    = parts[1] if len(parts) == 2 else 'daily'
            cfg   = PLAN_LOCK_CONFIG.get(tf, PLAN_LOCK_CONFIG['daily'])
            age   = now - entry['locked_at']
            if age > cfg['max_lock']:
                continue
            remaining = max(0, cfg['max_lock'] - age)
            result.append({
                'symbol':       sym,
                'name':         BIST100_STOCKS.get(sym, sym),
                'timeframe':    tf,
                'lockedAt':     entry['locked_at'],
                'lockedPrice':  entry['locked_price'],
                'signal':       entry['signal'],
                'ageSeconds':   int(age),
                'remainingSec': int(remaining),
                'maxLockSec':   cfg['max_lock'],
                'tfPlan':       entry.get('tf_plan', {}),
            })
        result.sort(key=lambda x: x['lockedAt'], reverse=True)
        return jsonify({'success': True, 'plans': result, 'count': len(result)})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/strategies/live')
def live_strategies():
    """3 stratejiyi tum hisselere canli uygular (PARALEL)"""
    try:
        with _strat_cache_lock:
            sc = _strat_cache
            if sc['data'] and (time.time() - sc['ts']) < COMPUTED_CACHE_TTL:
                return jsonify(safe_dict(sc['data']))

        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True, 'message': 'Veriler yukleniyor...'})

        results = {'ma_cross': [], 'breakout': [], 'mean_reversion': []}
        strategy_names = {
            'ma_cross': 'Hareketli Ortalama Kesisimi',
            'breakout': 'Kirilim Stratejisi',
            'mean_reversion': 'Ortalamaya Donus (RSI)',
        }

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_compute_strategy_for_stock, s): s for s in stocks}
            for future in as_completed(futures):
                r = future.result()
                if r:
                    for key in ('ma_cross', 'breakout', 'mean_reversion'):
                        if r[key]:
                            results[key].append(r[key])

        for key in results:
            results[key].sort(key=lambda x: (not x.get('freshSignal', False), -abs(x.get('changePct', 0))))

        result_data = {
            'success': True,
            'strategies': results,
            'strategyNames': strategy_names,
            'totalStocks': len(stocks),
            'timestamp': datetime.now().isoformat(),
        }

        with _strat_cache_lock:
            _strat_cache['data'] = result_data
            _strat_cache['ts']   = time.time()

        return jsonify(safe_dict(result_data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/dividends')
def dividend_calendar():
    """BIST hisselerinin temettu bilgileri (yfinance dividends)"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True, 'dividends': [], 'message': 'Veriler yukleniyor...'})

        dividends_list = []
        for stock in stocks:
            sym = stock['code']
            try:
                ticker_sym = sym + '.IS'
                if _YF_OK and yf:
                    tkr  = yf.Ticker(ticker_sym)
                    divs = tkr.dividends
                    if divs is not None and len(divs) > 0:
                        cp = stock.get('price', 0)
                        recent_divs = []
                        total_div_1y = 0
                        one_year_ago = datetime.now() - timedelta(days=365)
                        for dt, amt in divs.items():
                            div_date = dt.to_pydatetime().replace(tzinfo=None) if hasattr(dt, 'to_pydatetime') else dt
                            recent_divs.append({'date': div_date.strftime('%Y-%m-%d'), 'amount': sf(float(amt)), 'year': div_date.year})
                            if div_date >= one_year_ago:
                                total_div_1y += float(amt)

                        if recent_divs:
                            div_yield = sf((total_div_1y / cp * 100) if cp > 0 else 0)
                            last_div  = recent_divs[-1]
                            yearly_divs = {}
                            for dv in recent_divs:
                                yr = dv['year']
                                yearly_divs[yr] = yearly_divs.get(yr, 0) + float(dv['amount'])
                            dividends_list.append({
                                'code': sym,
                                'name': BIST100_STOCKS.get(sym, sym),
                                'price': sf(cp),
                                'lastDividend': last_div,
                                'dividendYield': div_yield,
                                'totalDiv1Y': sf(total_div_1y),
                                'history': recent_divs[-10:],
                                'yearlyTotals': {str(k): sf(v) for k, v in sorted(yearly_divs.items())},
                                'changePct': stock.get('changePct', 0),
                            })
            except Exception:
                continue

        dividends_list.sort(key=lambda x: float(x.get('dividendYield', 0)), reverse=True)
        return jsonify(safe_dict({'success': True, 'count': len(dividends_list), 'dividends': dividends_list, 'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/market/regime')
def market_regime_endpoint():
    """Piyasa rejimi (boga/ayi/yatay)"""
    try:
        regime = calc_market_regime()
        return jsonify(safe_dict({'success': True, **regime}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/sectors/analysis')
def sectors_analysis():
    """Sektor bazli goreceli guc analizi"""
    try:
        result = calc_sector_relative_strength()
        return jsonify(safe_dict({'success': True, **result}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/alerts/signals')
def signal_alerts_endpoint():
    """Sinyal bazli otomatik uyarilar (mum formasyonu, altin kesisim vb.)"""
    try:
        alerts = check_signal_alerts()
        return jsonify(safe_dict({
            'success': True,
            'alerts': alerts[:30],
            'totalAlerts': len(alerts),
            'marketRegime': calc_market_regime(),
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/signals/performance')
def signals_performance():
    """Tum hisselerin sinyal performans ozeti"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True})

        results = []
        for stock in stocks[:50]:
            sym = stock['code']
            try:
                hist = _cget_hist(f"{sym}_1y")
                if hist is None or len(hist) < 60:
                    continue
                bt = calc_signal_backtest(hist)
                if bt.get('totalSignals', 0) > 3:
                    overall = bt.get('overall', {})
                    results.append({
                        'code': sym,
                        'name': BIST100_STOCKS.get(sym, sym),
                        'totalSignals': bt['totalSignals'],
                        'winRate5d':    overall.get('winRate5d', 0),
                        'winRate10d':   overall.get('winRate10d', 0),
                        'winRate20d':   overall.get('winRate20d', 0),
                        'avgReturn10d': overall.get('avgRet10d', 0),
                    })
            except Exception:
                continue

        results.sort(key=lambda x: float(x.get('winRate10d', 0)), reverse=True)
        return jsonify(safe_dict({'success': True, 'performance': results, 'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/signals/calibration')
def signals_calibration():
    """BIST indikatör kalibrasyonu - tekli veya toplu hisse analizi"""
    try:
        symbol = request.args.get('symbol', '').upper().strip()

        if symbol:
            hist = _cget_hist(f"{symbol}_1y")
            if hist is None:
                hist = _fetch_hist_df(symbol, '1y')
            if hist is None or len(hist) < 60:
                return jsonify({'error': f'{symbol} icin yeterli veri yok'}), 400
            bt = calc_signal_backtest(hist)
            return jsonify(safe_dict({
                'success': True, 'symbol': symbol,
                'rsiCalibration': bt.get('rsiCalibration', {}),
                'bestRsiThreshold': bt.get('bestRsiThreshold', '30/70'),
                'rankedIndicators': bt.get('rankedIndicators', [])[:5],
                'benchmark': bt.get('benchmark', {}),
                'totalSignals': bt.get('totalSignals', 0),
                'timestamp': datetime.now().isoformat(),
            }))

        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True})

        agg_rsi = {}
        agg_ind = {}
        processed = 0

        for stock in stocks[:20]:
            sym = stock['code']
            try:
                hist = _cget_hist(f"{sym}_1y")
                if hist is None or len(hist) < 60:
                    continue
                bt = calc_signal_backtest(hist)
                if bt.get('totalSignals', 0) < 5:
                    continue
                for thresh, stats in bt.get('rsiCalibration', {}).items():
                    agg_rsi.setdefault(thresh, []).append(float(stats.get('profitFactor10d', 0)))
                for ind in bt.get('rankedIndicators', []):
                    name = ind.get('reason', '')
                    if name:
                        agg_ind.setdefault(name, []).append(float(ind.get('profitFactor10d', 0)))
                processed += 1
            except Exception:
                continue

        if processed == 0:
            return jsonify({'success': True, 'loading': True, 'message': 'Veri hazırlanıyor'})

        rsi_summary = {}
        for thresh, pfs in agg_rsi.items():
            avg_pf = float(np.mean(pfs)) if pfs else 0
            rsi_summary[thresh] = {'avgProfitFactor10d': sf(avg_pf), 'stockCount': len(pfs)}
        best_rsi_bulk = max(rsi_summary, key=lambda k: float(rsi_summary[k]['avgProfitFactor10d'])) if rsi_summary else '30/70'

        ind_summary = []
        for name, pfs in agg_ind.items():
            avg_pf = float(np.mean(pfs)) if pfs else 0
            ind_summary.append({'reason': name, 'avgProfitFactor10d': sf(avg_pf), 'stockCount': len(pfs)})
        ind_summary.sort(key=lambda x: float(x['avgProfitFactor10d']), reverse=True)

        return jsonify(safe_dict({
            'success': True,
            'processedStocks': processed,
            'rsiCalibrationSummary': rsi_summary,
            'bestRsiThreshold': best_rsi_bulk,
            'topIndicators': ind_summary[:5],
            'allIndicators': ind_summary,
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500
