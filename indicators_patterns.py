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
# =====================================================================
def calc_smc(hist, lookback=120):
    """SMC: FVG, Order Block, BOS, CHoCH"""
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float) if 'High' in hist.columns else c.copy()
        l = hist['Low'].values.astype(float) if 'Low' in hist.columns else c.copy()
        o = hist['Open'].values.astype(float) if 'Open' in hist.columns else c.copy()
        h = np.where(np.isnan(h), c, h); l = np.where(np.isnan(l), c, l); o = np.where(np.isnan(o), c, o)
        n = len(c); lb = min(lookback, n)
        c_lb, h_lb, l_lb, o_lb = c[-lb:], h[-lb:], l[-lb:], o[-lb:]

        # FVG
        fvgs = []
        for i in range(2, lb):
            if h_lb[i - 2] < l_lb[i]:
                gap_pct = (l_lb[i] - h_lb[i - 2]) / h_lb[i - 2] * 100
                filled = float(np.min(l_lb[i:])) < h_lb[i - 2]
                fvgs.append({'type': 'bullish_fvg', 'label': 'Boga FVG',
                    'top': sf(l_lb[i]), 'bottom': sf(h_lb[i - 2]),
                    'midpoint': sf((l_lb[i] + h_lb[i - 2]) / 2),
                    'sizePct': sf(gap_pct), 'filled': filled, 'barsAgo': int(lb - i)})
            elif l_lb[i - 2] > h_lb[i]:
                gap_pct = (l_lb[i - 2] - h_lb[i]) / h_lb[i] * 100
                filled = float(np.max(h_lb[i:])) > l_lb[i - 2]
                fvgs.append({'type': 'bearish_fvg', 'label': 'Ayi FVG',
                    'top': sf(l_lb[i - 2]), 'bottom': sf(h_lb[i]),
                    'midpoint': sf((l_lb[i - 2] + h_lb[i]) / 2),
                    'sizePct': sf(gap_pct), 'filled': filled, 'barsAgo': int(lb - i)})
        active_fvgs = sorted([f for f in fvgs if not f['filled'] and f['barsAgo'] <= 30],
                             key=lambda x: x['barsAgo'])[:5]

        # Order Blocks
        obs = []
        imp_thr = 0.015
        for i in range(1, lb - 3):
            if c_lb[i] < o_lb[i]:
                nxt_hi = float(np.max(h_lb[i + 1:min(i + 4, lb)]))
                if (nxt_hi - c_lb[i]) / c_lb[i] > imp_thr:
                    obs.append({'type': 'bullish_ob', 'label': 'Boga Order Block',
                        'top': sf(max(o_lb[i], c_lb[i])), 'bottom': sf(min(o_lb[i], c_lb[i])),
                        'midpoint': sf((o_lb[i] + c_lb[i]) / 2),
                        'impulseStrength': sf((nxt_hi - c_lb[i]) / c_lb[i] * 100), 'barsAgo': int(lb - i)})
            elif c_lb[i] > o_lb[i]:
                nxt_lo = float(np.min(l_lb[i + 1:min(i + 4, lb)]))
                if (c_lb[i] - nxt_lo) / c_lb[i] > imp_thr:
                    obs.append({'type': 'bearish_ob', 'label': 'Ayi Order Block',
                        'top': sf(max(o_lb[i], c_lb[i])), 'bottom': sf(min(o_lb[i], c_lb[i])),
                        'midpoint': sf((o_lb[i] + c_lb[i]) / 2),
                        'impulseStrength': sf((c_lb[i] - nxt_lo) / c_lb[i] * 100), 'barsAgo': int(lb - i)})
        recent_obs = sorted([ob for ob in obs if ob['barsAgo'] <= 40], key=lambda x: x['barsAgo'])[:5]

        # BOS
        swing_highs = _find_peaks(h_lb, 5)
        swing_lows = _find_troughs(l_lb, 5)
        bos_events = []
        structure_trend = 'neutral'
        if len(swing_highs) >= 1:
            last_sh = swing_highs[-1]
            if c_lb[-1] > h_lb[last_sh]:
                bos_events.append({'type': 'bullish_bos', 'label': 'Yukari BOS',
                    'description': f'Kapanis ({sf(c_lb[-1])}) swing high kirdi ({sf(h_lb[last_sh])})',
                    'level': sf(h_lb[last_sh]), 'barsAgo': int(lb - 1 - last_sh)})
                structure_trend = 'bullish'
        if len(swing_lows) >= 1:
            last_sl = swing_lows[-1]
            if c_lb[-1] < l_lb[last_sl]:
                bos_events.append({'type': 'bearish_bos', 'label': 'Asagi BOS',
                    'description': f'Kapanis ({sf(c_lb[-1])}) swing low kirdi ({sf(l_lb[last_sl])})',
                    'level': sf(l_lb[last_sl]), 'barsAgo': int(lb - 1 - last_sl)})
                structure_trend = 'bearish'

        # CHoCH
        choch_events = []
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            sh1, sh2 = swing_highs[-2], swing_highs[-1]
            sl1, sl2 = swing_lows[-2], swing_lows[-1]
            if h_lb[sh2] < h_lb[sh1] and l_lb[sl2] < l_lb[sl1] and c_lb[-1] > h_lb[sh2]:
                choch_events.append({'type': 'bullish_choch', 'label': 'Boga CHoCH',
                    'description': "Downtrend'te swing high kirildi - Olasi trend degisimi",
                    'level': sf(h_lb[sh2])})
            elif h_lb[sh2] > h_lb[sh1] and l_lb[sl2] > l_lb[sl1] and c_lb[-1] < l_lb[sl2]:
                choch_events.append({'type': 'bearish_choch', 'label': 'Ayi CHoCH',
                    'description': "Uptrend'de swing low kirildi - Olasi trend degisimi",
                    'level': sf(l_lb[sl2])})

        # Entry Zones
        cur = float(c_lb[-1])
        entry_zones = []
        for ob in recent_obs:
            if ob['type'] == 'bullish_ob' and float(ob['bottom']) < cur:
                entry_zones.append({'source': 'bullish_ob', 'level': ob['midpoint'], 'top': ob['top'], 'bottom': ob['bottom']})
            elif ob['type'] == 'bearish_ob' and float(ob['top']) > cur:
                entry_zones.append({'source': 'bearish_ob', 'level': ob['midpoint'], 'top': ob['top'], 'bottom': ob['bottom']})
        for fg in active_fvgs:
            if fg['type'] == 'bullish_fvg' and float(fg['bottom']) < cur:
                entry_zones.append({'source': 'fvg_support', 'level': fg['midpoint'], 'top': fg['top'], 'bottom': fg['bottom']})
            elif fg['type'] == 'bearish_fvg' and float(fg['top']) > cur:
                entry_zones.append({'source': 'fvg_resistance', 'level': fg['midpoint'], 'top': fg['top'], 'bottom': fg['bottom']})

        bull_score = (sum(1 for f in active_fvgs if f['type'] == 'bullish_fvg') +
                      sum(1 for ob in recent_obs if ob['type'] == 'bullish_ob') +
                      sum(1 for b in bos_events if b['type'] == 'bullish_bos') +
                      sum(1 for cc in choch_events if cc['type'] == 'bullish_choch'))
        bear_score = (sum(1 for f in active_fvgs if f['type'] == 'bearish_fvg') +
                      sum(1 for ob in recent_obs if ob['type'] == 'bearish_ob') +
                      sum(1 for b in bos_events if b['type'] == 'bearish_bos') +
                      sum(1 for cc in choch_events if cc['type'] == 'bearish_choch'))
        smc_signal = ('buy' if bull_score > bear_score
                      else ('sell' if bear_score > bull_score else 'neutral'))

        return {
            'signal': smc_signal, 'structureTrend': structure_trend,
            'bullScore': bull_score, 'bearScore': bear_score,
            'fvgs': active_fvgs, 'orderBlocks': recent_obs,
            'bosEvents': bos_events, 'chochEvents': choch_events,
            'entryZones': entry_zones[:4],
            'summary': {'activeFvgCount': len(active_fvgs), 'activeObCount': len(recent_obs),
                        'hasBOS': len(bos_events) > 0, 'hasCHoCH': len(choch_events) > 0},
        }
    except Exception as e:
        print(f"  [SMC] Hata: {e}")
        return {'signal': 'neutral', 'error': str(e),
                'fvgs': [], 'orderBlocks': [], 'bosEvents': [], 'chochEvents': [],
                'summary': {'activeFvgCount': 0, 'activeObCount': 0, 'hasBOS': False, 'hasCHoCH': False}}


# =====================================================================
# KLASİK GRAFİK FORMASYONLARI
# =====================================================================
def calc_chart_patterns(hist, lookback=120):
    """Cift Tepe/Dip, OBO, Ucgen, Bayrak"""
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float) if 'High' in hist.columns else c.copy()
        l = hist['Low'].values.astype(float) if 'Low' in hist.columns else c.copy()
        h = np.where(np.isnan(h), c, h); l = np.where(np.isnan(l), c, l)
        n = len(c); lb = min(lookback, n)
        c_lb, h_lb, l_lb = c[-lb:], h[-lb:], l[-lb:]
        patterns = []
        tol = 0.03
        peaks = _find_peaks(h_lb, 5)
        troughs = _find_troughs(l_lb, 5)

        # Cift Tepe
        if len(peaks) >= 2:
            p1, p2 = peaks[-2], peaks[-1]
            if abs(h_lb[p2] - h_lb[p1]) / h_lb[p1] <= tol and p2 - p1 >= 10:
                neckline = float(np.min(l_lb[p1:p2 + 1]))
                completed = bool(c_lb[-1] < neckline)
                height = float(h_lb[p2]) - neckline
                patterns.append({'type': 'double_top', 'label': 'Cift Tepe', 'signal': 'sell',
                    'reliability': 'high', 'completed': completed,
                    'description': f'Iki benzer zirve ({sf(h_lb[p1])}, {sf(h_lb[p2])}) Neckline: {sf(neckline)}' +
                                   (' -> TAMAMLANDI' if completed else ' -> Neckline kirilmasi bekleniyor'),
                    'peak1': sf(h_lb[p1]), 'peak2': sf(h_lb[p2]),
                    'neckline': sf(neckline), 'target': sf(neckline - height), 'barsAgo': int(lb - p2)})

        # Cift Dip
        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            if abs(l_lb[t2] - l_lb[t1]) / l_lb[t1] <= tol and t2 - t1 >= 10:
                neckline = float(np.max(h_lb[t1:t2 + 1]))
                completed = bool(c_lb[-1] > neckline)
                height = neckline - float(l_lb[t2])
                patterns.append({'type': 'double_bottom', 'label': 'Cift Dip', 'signal': 'buy',
                    'reliability': 'high', 'completed': completed,
                    'description': f'Iki benzer dip ({sf(l_lb[t1])}, {sf(l_lb[t2])}) Neckline: {sf(neckline)}' +
                                   (' -> TAMAMLANDI' if completed else ' -> Neckline kirilmasi bekleniyor'),
                    'trough1': sf(l_lb[t1]), 'trough2': sf(l_lb[t2]),
                    'neckline': sf(neckline), 'target': sf(neckline + height), 'barsAgo': int(lb - t2)})

        # Omuz-Bas-Omuz
        if len(peaks) >= 3:
            ls, hd, rs = peaks[-3], peaks[-2], peaks[-1]
            lsh, hp, rsh = float(h_lb[ls]), float(h_lb[hd]), float(h_lb[rs])
            if hp > lsh and hp > rsh and abs(rsh - lsh) / lsh <= 0.05 and rs - ls >= 20:
                neckline = (float(np.min(l_lb[ls:hd + 1])) + float(np.min(l_lb[hd:rs + 1]))) / 2
                completed = bool(c_lb[-1] < neckline)
                patterns.append({'type': 'head_shoulders', 'label': 'Omuz-Bas-Omuz', 'signal': 'sell',
                    'reliability': 'very_high', 'completed': completed,
                    'description': f'Sol omuz ({sf(lsh)}), Bas ({sf(hp)}), Sag omuz ({sf(rsh)}) Neckline: {sf(neckline)}' +
                                   (' -> TAMAMLANDI' if completed else ''),
                    'leftShoulder': sf(lsh), 'head': sf(hp), 'rightShoulder': sf(rsh),
                    'neckline': sf(neckline), 'target': sf(neckline - (hp - neckline)), 'barsAgo': int(lb - rs)})

        # Ters OBO
        if len(troughs) >= 3:
            ls, hd, rs = troughs[-3], troughs[-2], troughs[-1]
            lsh, hp, rsh = float(l_lb[ls]), float(l_lb[hd]), float(l_lb[rs])
            if hp < lsh and hp < rsh and abs(rsh - lsh) / lsh <= 0.05 and rs - ls >= 20:
                neckline = (float(np.max(h_lb[ls:hd + 1])) + float(np.max(h_lb[hd:rs + 1]))) / 2
                completed = bool(c_lb[-1] > neckline)
                patterns.append({'type': 'inv_head_shoulders', 'label': 'Ters Omuz-Bas-Omuz', 'signal': 'buy',
                    'reliability': 'very_high', 'completed': completed,
                    'description': f'Sol omuz ({sf(lsh)}), Bas ({sf(hp)}), Sag omuz ({sf(rsh)}) Neckline: {sf(neckline)}' +
                                   (' -> TAMAMLANDI' if completed else ''),
                    'leftShoulder': sf(lsh), 'head': sf(hp), 'rightShoulder': sf(rsh),
                    'neckline': sf(neckline), 'target': sf(neckline + (neckline - hp)), 'barsAgo': int(lb - rs)})

        # Ucgen Formasyonlari
        if lb >= 30:
            x = np.arange(30, dtype=float)
            h_seg = h_lb[-30:].astype(float); l_seg = l_lb[-30:].astype(float)
            h_slope = float(np.polyfit(x, h_seg, 1)[0])
            l_slope = float(np.polyfit(x, l_seg, 1)[0])
            h_pct = h_slope / float(np.mean(h_seg)) * 100
            l_pct = l_slope / float(np.mean(l_seg)) * 100
            if abs(h_pct) < 0.08 and l_pct > 0.08:
                res = sf(float(np.max(h_seg[-10:])))
                rng = float(np.max(h_seg)) - float(np.min(l_seg))
                patterns.append({'type': 'ascending_triangle', 'label': 'Yukselen Ucgen', 'signal': 'buy',
                    'reliability': 'medium', 'completed': bool(c_lb[-1] > float(np.max(h_seg))),
                    'description': f'Duz direnc ({res}) + yukselen dip -> Yukari kirilim beklenir',
                    'resistance': res, 'target': sf(float(np.max(h_seg)) + rng), 'barsAgo': 0})
            elif abs(l_pct) < 0.08 and h_pct < -0.08:
                sup = sf(float(np.min(l_seg[-10:])))
                rng = float(np.max(h_seg)) - float(np.min(l_seg))
                patterns.append({'type': 'descending_triangle', 'label': 'Alcalan Ucgen', 'signal': 'sell',
                    'reliability': 'medium', 'completed': bool(c_lb[-1] < float(np.min(l_seg))),
                    'description': f'Duz destek ({sup}) + dusen zirve -> Asagi kirilim beklenir',
                    'support': sup, 'target': sf(float(np.min(l_seg)) - rng), 'barsAgo': 0})
            elif h_pct < -0.05 and l_pct > 0.05:
                apex = sf((float(np.max(h_seg[-5:])) + float(np.min(l_seg[-5:]))) / 2)
                patterns.append({'type': 'symmetrical_triangle', 'label': 'Simetrik Ucgen', 'signal': 'neutral',
                    'reliability': 'medium', 'completed': False,
                    'description': f'Daralan fiyat araligi -> Guclu kirilim bekleniyor (apex: {apex})',
                    'apex': apex, 'barsAgo': 0})

        # Bayrak
        if lb >= 26:
            pre_move_pct = (float(c_lb[-16]) - float(c_lb[-26])) / max(float(c_lb[-26]), 1) * 100
            consol_range = ((float(np.max(h_lb[-15:])) - float(np.min(l_lb[-15:]))) /
                            max(float(c_lb[-15]), 1) * 100)
            if abs(pre_move_pct) > 5 and consol_range < 4:
                is_bull = pre_move_pct > 0
                patterns.append({'type': 'bull_flag' if is_bull else 'bear_flag',
                    'label': 'Boga Bayragi' if is_bull else 'Ayi Bayragi',
                    'signal': 'buy' if is_bull else 'sell',
                    'reliability': 'medium', 'completed': False,
                    'description': f'{sf(abs(pre_move_pct))}% on hareket + {sf(consol_range)}% konsolidasyon -> Trend devam bekleniyor',
                    'priorMovePct': sf(pre_move_pct), 'consolidationRangePct': sf(consol_range), 'barsAgo': 0})

        completed_p = [p for p in patterns if p.get('completed', False)]
        bull_patt = [p for p in patterns if p['signal'] == 'buy']
        bear_patt = [p for p in patterns if p['signal'] == 'sell']
        if completed_p: overall = completed_p[0]['signal']
        elif len(bull_patt) > len(bear_patt): overall = 'buy'
        elif len(bear_patt) > len(bull_patt): overall = 'sell'
        else: overall = 'neutral'

        return {
            'signal': overall, 'patterns': patterns,
            'completedPatterns': completed_p,
            'pendingPatterns': [p for p in patterns if not p.get('completed', False)],
            'summary': {'total': len(patterns), 'bullish': len(bull_patt),
                        'bearish': len(bear_patt), 'completed': len(completed_p)},
        }
    except Exception as e:
        print(f"  [PATTERNS] Hata: {e}")
        return {'signal': 'neutral', 'patterns': [], 'completedPatterns': [], 'pendingPatterns': [],
                'summary': {'total': 0, 'bullish': 0, 'bearish': 0, 'completed': 0}, 'error': str(e)}
