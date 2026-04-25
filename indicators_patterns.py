"""
BIST Pro - Pattern Detection Indicators
Divergence, Volume Profile, SMC, Chart Patterns
"""
import numpy as np
import pandas as pd
from config import sf
from indicators_basic import (
    _resample_to_tf, calc_rsi, calc_rsi_single, calc_macd, calc_bollinger,
)


def _rsi_series(closes, period=14):
    """Wilder yumuşatma ile tam RSI serisi (vektörel)"""
    c = np.array(closes, dtype=float)
    if len(c) < period + 1:
        return np.full(len(c), 50.0)
    delta = np.diff(c)
    gains  = np.where(delta > 0,  delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    avg_g  = np.zeros(len(delta))
    avg_l  = np.zeros(len(delta))
    avg_g[period - 1] = np.mean(gains[:period])
    avg_l[period - 1] = np.mean(losses[:period])
    for i in range(period, len(delta)):
        avg_g[i] = (avg_g[i-1] * (period-1) + gains[i])  / period
        avg_l[i] = (avg_l[i-1] * (period-1) + losses[i]) / period
    rs  = np.where(avg_l == 0, np.inf, avg_g / avg_l)
    rsi = np.where(avg_l == 0, 100.0, 100 - 100 / (1 + rs))
    result = np.full(len(c), np.nan)
    result[period:] = rsi[period - 1:]
    return result

def _find_peaks(arr, window=5):
    """Lokal zirve indekslerini döndür"""
    peaks = []
    for i in range(window, len(arr) - window):
        if arr[i] == max(arr[i-window:i+window+1]):
            peaks.append(i)
    return peaks

def _find_troughs(arr, window=5):
    """Lokal dip indekslerini döndür"""
    troughs = []
    for i in range(window, len(arr) - window):
        if arr[i] == min(arr[i-window:i+window+1]):
            troughs.append(i)
    return troughs


# =====================================================================
# MTF (Multi-TimeFrame) SİNYAL
# =====================================================================
def calc_mtf_signal(hist_daily):
    """Gercek coklu zaman dilimi sinyali: daily/weekly/monthly"""
    def _tf_signal(hist):
        if hist is None or len(hist) < 10:
            return {'signal': 'neutral', 'score': 0, 'rsi': 50,
                    'macd': 'neutral', 'ema': 'neutral', 'bars': 0}
        try:
            c = hist['Close'].values.astype(float)
            h = hist['High'].values.astype(float) if 'High' in hist.columns else c.copy()
            l = hist['Low'].values.astype(float) if 'Low' in hist.columns else c.copy()
            h = np.where(np.isnan(h), c, h)
            l = np.where(np.isnan(l), c, l)
            n = len(c)
            score = 0
            rsi_d = calc_rsi(c)
            rsi_val = float(rsi_d.get('value', 50))
            if   rsi_val < 35: score += 2
            elif rsi_val < 45: score += 1
            elif rsi_val > 65: score -= 2
            elif rsi_val > 55: score -= 1
            macd_sig = 'neutral'
            if n >= 26:
                md = calc_macd(c)
                hist_val = float(md.get('histogram', 0))
                if   hist_val > 0: score += 1; macd_sig = 'buy'
                elif hist_val < 0: score -= 1; macd_sig = 'sell'
            ema_sig = 'neutral'
            if n >= 50:
                s = pd.Series(c)
                e20 = float(s.ewm(span=20, adjust=False).mean().iloc[-1])
                e50 = float(s.ewm(span=50, adjust=False).mean().iloc[-1])
                cur = float(c[-1])
                if   cur > e20 and e20 > e50: score += 1; ema_sig = 'buy'
                elif cur < e20 and e20 < e50: score -= 1; ema_sig = 'sell'
            if n >= 20:
                bb = calc_bollinger(c, float(c[-1]))
                bbl = float(bb.get('lower', 0))
                bbu = float(bb.get('upper', 0))
                cp = float(c[-1])
                if bbl > 0 and cp < bbl: score += 1
                elif bbu > 0 and cp > bbu: score -= 1
            signal = 'buy' if score >= 2 else ('sell' if score <= -2 else 'neutral')
            return {
                'signal': signal, 'score': sf(score),
                'rsi': sf(rsi_val), 'macd': macd_sig, 'ema': ema_sig,
                'bars': n, 'currentPrice': sf(float(c[-1])),
            }
        except Exception as e:
            return {'signal': 'neutral', 'score': 0, 'rsi': 50,
                    'macd': 'neutral', 'ema': 'neutral', 'bars': 0, 'error': str(e)}

    try:
        daily_sig   = _tf_signal(hist_daily)
        weekly_sig  = _tf_signal(_resample_to_tf(hist_daily, 'weekly'))
        monthly_sig = _tf_signal(_resample_to_tf(hist_daily, 'monthly'))
        sigs = [daily_sig['signal'], weekly_sig['signal'], monthly_sig['signal']]
        buy_c  = sigs.count('buy')
        sell_c = sigs.count('sell')
        if   buy_c >= 2:  dominant = 'buy';  mtf_score = buy_c
        elif sell_c >= 2: dominant = 'sell'; mtf_score = sell_c
        else:             dominant = 'neutral'; mtf_score = 0
        alignment = f'{max(buy_c, sell_c)}/3'
        strength  = ('Guclu' if max(buy_c, sell_c) == 3
                     else ('Orta' if max(buy_c, sell_c) == 2 else 'Uyumsuz'))
        return {
            'daily': daily_sig, 'weekly': weekly_sig, 'monthly': monthly_sig,
            'mtfScore': mtf_score, 'mtfAlignment': alignment,
            'mtfDirection': dominant, 'mtfStrength': strength,
            'description': (
                f'Gunluk: {daily_sig["signal"]} | '
                f'Haftalik: {weekly_sig["signal"]} | '
                f'Aylik: {monthly_sig["signal"]} '
                f'→ {alignment} uyum ({strength})'
            ),
        }
    except Exception as e:
        print(f"  [MTF] Hata: {e}")
        return {
            'daily': {'signal': 'neutral'}, 'weekly': {'signal': 'neutral'},
            'monthly': {'signal': 'neutral'}, 'mtfScore': 0,
            'mtfAlignment': '0/3', 'mtfDirection': 'neutral',
            'mtfStrength': 'Uyumsuz', 'error': str(e),
        }


# =====================================================================
# UYUMSUZLUK (DIVERGENCE) TESPİTİ
# =====================================================================
def calc_divergence(hist, lookback=90):
    """RSI + MACD uyumsuzluk tespiti."""
    try:
        c  = hist['Close'].values.astype(float)
        n  = len(c)
        if n < 50:
            return {'divergences': [], 'recentDivergences': [],
                    'summary': {'bullish': 0, 'bearish': 0, 'signal': 'neutral', 'count': 0, 'hasRecent': False}}

        lb   = min(lookback, n)
        c_lb = c[-lb:]
        rsi_arr  = _rsi_series(c_lb)
        rsi_vals = np.where(np.isnan(rsi_arr), 50.0, rsi_arr)

        s          = pd.Series(c_lb, dtype=float)
        macd_line  = s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()
        sig_line   = macd_line.ewm(span=9, adjust=False).mean()
        macd_hist_arr = (macd_line - sig_line).values

        divergences = []
        window = 5
        price_peaks   = _find_peaks(c_lb, window)
        price_troughs = _find_troughs(c_lb, window)

        # Regular Bearish: Fiyat HH, RSI LH
        if len(price_peaks) >= 2:
            p1, p2 = price_peaks[-2], price_peaks[-1]
            if c_lb[p2] > c_lb[p1] and rsi_vals[p2] < rsi_vals[p1]:
                divergences.append({
                    'type': 'regular_bearish', 'label': 'Klasik Ayi Uyumsuzlugu', 'signal': 'sell',
                    'description': f'Fiyat yeni zirve ({sf(c_lb[p2])}) ama RSI dusuyor ({sf(rsi_vals[p2])} < {sf(rsi_vals[p1])})',
                    'strength': sf(abs(rsi_vals[p1] - rsi_vals[p2])), 'recency': int(lb - p2),
                    'priceBar1': sf(c_lb[p1]), 'priceBar2': sf(c_lb[p2]),
                    'rsiBar1': sf(rsi_vals[p1]), 'rsiBar2': sf(rsi_vals[p2]),
                })

        # Regular Bullish: Fiyat LL, RSI HL
        if len(price_troughs) >= 2:
            t1, t2 = price_troughs[-2], price_troughs[-1]
            if c_lb[t2] < c_lb[t1] and rsi_vals[t2] > rsi_vals[t1]:
                divergences.append({
                    'type': 'regular_bullish', 'label': 'Klasik Boga Uyumsuzlugu', 'signal': 'buy',
                    'description': f'Fiyat yeni dip ({sf(c_lb[t2])}) ama RSI yukseliyor ({sf(rsi_vals[t2])} > {sf(rsi_vals[t1])})',
                    'strength': sf(abs(rsi_vals[t2] - rsi_vals[t1])), 'recency': int(lb - t2),
                    'priceBar1': sf(c_lb[t1]), 'priceBar2': sf(c_lb[t2]),
                    'rsiBar1': sf(rsi_vals[t1]), 'rsiBar2': sf(rsi_vals[t2]),
                })

        # Hidden Bullish: Fiyat HL, RSI LL
        if len(price_troughs) >= 2:
            t1, t2 = price_troughs[-2], price_troughs[-1]
            if c_lb[t2] > c_lb[t1] and rsi_vals[t2] < rsi_vals[t1]:
                divergences.append({
                    'type': 'hidden_bullish', 'label': 'Gizli Boga Uyumsuzlugu', 'signal': 'buy',
                    'description': f'Fiyat yuksek dip ({sf(c_lb[t2])}) ama RSI dusuk -> Uptrend devam',
                    'strength': sf(abs(rsi_vals[t2] - rsi_vals[t1])), 'recency': int(lb - t2),
                    'priceBar1': sf(c_lb[t1]), 'priceBar2': sf(c_lb[t2]),
                    'rsiBar1': sf(rsi_vals[t1]), 'rsiBar2': sf(rsi_vals[t2]),
                })

        # Hidden Bearish: Fiyat LH, RSI HH
        if len(price_peaks) >= 2:
            p1, p2 = price_peaks[-2], price_peaks[-1]
            if c_lb[p2] < c_lb[p1] and rsi_vals[p2] > rsi_vals[p1]:
                divergences.append({
                    'type': 'hidden_bearish', 'label': 'Gizli Ayi Uyumsuzlugu', 'signal': 'sell',
                    'description': f'Fiyat dusuk zirve ({sf(c_lb[p2])}) ama RSI yukseliyor -> Downtrend devam',
                    'strength': sf(abs(rsi_vals[p2] - rsi_vals[p1])), 'recency': int(lb - p2),
                    'priceBar1': sf(c_lb[p1]), 'priceBar2': sf(c_lb[p2]),
                    'rsiBar1': sf(rsi_vals[p1]), 'rsiBar2': sf(rsi_vals[p2]),
                })

        # MACD Bearish Divergence
        if len(price_peaks) >= 2:
            p1, p2 = price_peaks[-2], price_peaks[-1]
            if (p1 < len(macd_hist_arr) and p2 < len(macd_hist_arr) and
                    c_lb[p2] > c_lb[p1] and macd_hist_arr[p2] < macd_hist_arr[p1]):
                divergences.append({
                    'type': 'macd_bearish', 'label': 'MACD Ayi Uyumsuzlugu', 'signal': 'sell',
                    'description': f'Fiyat HH ama MACD histogram dusuk ({sf(float(macd_hist_arr[p2]))} < {sf(float(macd_hist_arr[p1]))})',
                    'strength': sf(abs(float(macd_hist_arr[p1]) - float(macd_hist_arr[p2]))),
                    'recency': int(lb - p2),
                })

        # MACD Bullish Divergence
        if len(price_troughs) >= 2:
            t1, t2 = price_troughs[-2], price_troughs[-1]
            if (t1 < len(macd_hist_arr) and t2 < len(macd_hist_arr) and
                    c_lb[t2] < c_lb[t1] and macd_hist_arr[t2] > macd_hist_arr[t1]):
                divergences.append({
                    'type': 'macd_bullish', 'label': 'MACD Boga Uyumsuzlugu', 'signal': 'buy',
                    'description': f'Fiyat LL ama MACD histogram yukseliyor ({sf(float(macd_hist_arr[t2]))} > {sf(float(macd_hist_arr[t1]))})',
                    'strength': sf(abs(float(macd_hist_arr[t2]) - float(macd_hist_arr[t1]))),
                    'recency': int(lb - t2),
                })

        recent    = [d for d in divergences if d.get('recency', 999) <= 20]
        bull_cnt  = sum(1 for d in divergences if d['signal'] == 'buy')
        bear_cnt  = sum(1 for d in divergences if d['signal'] == 'sell')
        overall   = 'buy' if bull_cnt > bear_cnt else ('sell' if bear_cnt > bull_cnt else 'neutral')

        return {
            'divergences': divergences, 'recentDivergences': recent,
            'summary': {'bullish': bull_cnt, 'bearish': bear_cnt, 'signal': overall,
                        'count': len(divergences), 'hasRecent': len(recent) > 0},
            'currentRsi': sf(float(rsi_vals[-1])),
            'currentMacdHist': sf(float(macd_hist_arr[-1])),
        }
    except Exception as e:
        print(f"  [DIVERGENCE] Hata: {e}")
        return {'divergences': [], 'recentDivergences': [],
                'summary': {'bullish': 0, 'bearish': 0, 'signal': 'neutral', 'count': 0, 'hasRecent': False},
                'error': str(e)}


# =====================================================================
# HACİM PROFİLİ & VWAP
# =====================================================================
def calc_volume_profile(hist, bins=20):
    """Hacim Profili: VWAP, POC, VAH, VAL, Anomaly"""
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float) if 'High' in hist.columns else c.copy()
        l = hist['Low'].values.astype(float) if 'Low' in hist.columns else c.copy()
        v = hist['Volume'].values.astype(float) if 'Volume' in hist.columns else np.ones(len(c))
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        v = np.where(np.isnan(v) | (v <= 0), 0, v)
        n = len(c)

        typical = (h + l + c) / 3
        cum_vol = np.cumsum(v)
        cum_tpv = np.cumsum(typical * v)
        vwap_ser = np.where(cum_vol > 0, cum_tpv / cum_vol, typical)
        vwap = float(vwap_ser[-1])
        cur_price = float(c[-1])
        vwap_pct = sf((cur_price - vwap) / vwap * 100) if vwap > 0 else 0
        vwap_sig = ('buy' if cur_price < vwap * 0.99
                    else ('sell' if cur_price > vwap * 1.01 else 'neutral'))

        price_min = float(np.min(l))
        price_max = float(np.max(h))
        if price_max <= price_min:
            price_max = price_min * 1.01
        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_volumes = np.zeros(bins)
        for i in range(n):
            bar_range = h[i] - l[i]
            if bar_range <= 0:
                idx = min(max(int(np.searchsorted(bin_edges, c[i], side='right') - 1), 0), bins - 1)
                bin_volumes[idx] += v[i]
            else:
                for b in range(bins):
                    ov_lo = max(l[i], bin_edges[b])
                    ov_hi = min(h[i], bin_edges[b + 1])
                    if ov_hi > ov_lo:
                        bin_volumes[b] += v[i] * (ov_hi - ov_lo) / bar_range

        poc_idx = int(np.argmax(bin_volumes))
        poc_price = sf((bin_edges[poc_idx] + bin_edges[poc_idx + 1]) / 2)
        poc_volume = sf(bin_volumes[poc_idx])

        total_vol = float(np.sum(bin_volumes))
        va_target = total_vol * 0.70
        va_vol = bin_volumes[poc_idx]
        lo_i, hi_i = poc_idx, poc_idx
        while va_vol < va_target and (lo_i > 0 or hi_i < bins - 1):
            add_lo = bin_volumes[lo_i - 1] if lo_i > 0 else 0.0
            add_hi = bin_volumes[hi_i + 1] if hi_i < bins - 1 else 0.0
            if add_hi >= add_lo and hi_i < bins - 1:
                hi_i += 1; va_vol += bin_volumes[hi_i]
            elif lo_i > 0:
                lo_i -= 1; va_vol += bin_volumes[lo_i]
            else:
                hi_i += 1; va_vol += bin_volumes[hi_i]

        vah = sf((bin_edges[hi_i] + bin_edges[hi_i + 1]) / 2)
        val = sf((bin_edges[lo_i] + bin_edges[lo_i + 1]) / 2)

        avg_vol_20 = float(np.mean(v[-20:])) if n >= 20 else float(np.mean(v))
        last_vol = float(v[-1])
        vol_ratio = sf(last_vol / avg_vol_20) if avg_vol_20 > 0 else 0
        vol_anomaly = last_vol > avg_vol_20 * 2
        vol_trend = ('artiyor' if n >= 6 and float(np.mean(v[-3:])) > float(np.mean(v[-6:-3]))
                     else 'azaliyor')

        profile = [
            {'priceLevel': sf((bin_edges[i] + bin_edges[i + 1]) / 2),
             'volume': sf(bin_volumes[i]), 'isPOC': i == poc_idx,
             'isVAH': i == hi_i, 'isVAL': i == lo_i, 'inValueArea': lo_i <= i <= hi_i}
            for i in range(bins)
        ]
        return {
            'vwap': sf(vwap), 'vwapSignal': vwap_sig, 'vwapPct': vwap_pct,
            'poc': poc_price, 'pocVolume': poc_volume, 'vah': vah, 'val': val,
            'profile': profile, 'volumeAnomaly': vol_anomaly, 'volumeRatio': vol_ratio,
            'volumeTrend': vol_trend, 'avgVolume20': sf(avg_vol_20), 'lastVolume': sf(last_vol),
            'currentPrice': sf(cur_price), 'priceVsVwap': vwap_pct,
            'priceVsVAH': sf((cur_price - float(vah)) / float(vah) * 100) if float(vah) > 0 else 0,
            'priceVsVAL': sf((cur_price - float(val)) / float(val) * 100) if float(val) > 0 else 0,
            'priceVsPOC': sf((cur_price - float(poc_price)) / float(poc_price) * 100) if float(poc_price) > 0 else 0,
        }
    except Exception as e:
        print(f"  [VOL-PROFILE] Hata: {e}")
        return {'error': str(e), 'vwap': 0, 'poc': 0, 'vah': 0, 'val': 0,
                'volumeAnomaly': False, 'volumeRatio': 0}


# =====================================================================
# SMART MONEY CONCEPTS (SMC)

# calc_smc, calc_chart_patterns -> indicators_smc.py
from indicators_smc import calc_smc, calc_chart_patterns  # re-export
