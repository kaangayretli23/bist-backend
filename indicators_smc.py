"""
BIST Pro - SMC & Chart Pattern Indicators
calc_smc, calc_chart_patterns
indicators_patterns.py'dan ayrıştırıldı (600 satır kuralı).
"""
import numpy as np
from config import sf
from indicators_patterns import _find_peaks, _find_troughs


def calc_smc(hist, lookback=120):
    """SMC: FVG, Order Block, BOS, CHoCH"""
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)  if 'High' in hist.columns else c.copy()
        l = hist['Low'].values.astype(float)   if 'Low'  in hist.columns else c.copy()
        o = hist['Open'].values.astype(float)  if 'Open' in hist.columns else c.copy()
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        o = np.where(np.isnan(o), c, o)
        n = len(c); lb = min(lookback, n)
        c_lb, h_lb, l_lb, o_lb = c[-lb:], h[-lb:], l[-lb:], o[-lb:]

        # FVG
        fvgs = []
        for i in range(2, lb):
            if h_lb[i - 2] < l_lb[i]:
                gap_pct = (l_lb[i] - h_lb[i - 2]) / h_lb[i - 2] * 100
                filled  = float(np.min(l_lb[i:])) < h_lb[i - 2]
                fvgs.append({'type': 'bullish_fvg', 'label': 'Boga FVG',
                    'top': sf(l_lb[i]), 'bottom': sf(h_lb[i - 2]),
                    'midpoint': sf((l_lb[i] + h_lb[i - 2]) / 2),
                    'sizePct': sf(gap_pct), 'filled': filled, 'barsAgo': int(lb - i)})
            elif l_lb[i - 2] > h_lb[i]:
                gap_pct = (l_lb[i - 2] - h_lb[i]) / h_lb[i] * 100
                filled  = float(np.max(h_lb[i:])) > l_lb[i - 2]
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
        swing_lows  = _find_troughs(l_lb, 5)
        bos_events = []
        structure_trend = 'neutral'
        if swing_highs and c_lb[-1] > h_lb[swing_highs[-1]]:
            bos_events.append({'type': 'bullish_bos', 'label': 'Yukari BOS',
                'description': f'Kapanis ({sf(c_lb[-1])}) swing high kirdi ({sf(h_lb[swing_highs[-1]])})',
                'level': sf(h_lb[swing_highs[-1]]), 'barsAgo': int(lb - 1 - swing_highs[-1])})
            structure_trend = 'bullish'
        if swing_lows and c_lb[-1] < l_lb[swing_lows[-1]]:
            bos_events.append({'type': 'bearish_bos', 'label': 'Asagi BOS',
                'description': f'Kapanis ({sf(c_lb[-1])}) swing low kirdi ({sf(l_lb[swing_lows[-1]])})',
                'level': sf(l_lb[swing_lows[-1]]), 'barsAgo': int(lb - 1 - swing_lows[-1])})
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
                      sum(1 for b in bos_events  if b['type'] == 'bullish_bos') +
                      sum(1 for cc in choch_events if cc['type'] == 'bullish_choch'))
        bear_score = (sum(1 for f in active_fvgs if f['type'] == 'bearish_fvg') +
                      sum(1 for ob in recent_obs if ob['type'] == 'bearish_ob') +
                      sum(1 for b in bos_events  if b['type'] == 'bearish_bos') +
                      sum(1 for cc in choch_events if cc['type'] == 'bearish_choch'))
        smc_signal = 'buy' if bull_score > bear_score else ('sell' if bear_score > bull_score else 'neutral')

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


def calc_chart_patterns(hist, lookback=120):
    """Cift Tepe/Dip, OBO, Ucgen, Bayrak"""
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float) if 'High' in hist.columns else c.copy()
        l = hist['Low'].values.astype(float)  if 'Low'  in hist.columns else c.copy()
        h = np.where(np.isnan(h), c, h); l = np.where(np.isnan(l), c, l)
        n = len(c); lb = min(lookback, n)
        c_lb, h_lb, l_lb = c[-lb:], h[-lb:], l[-lb:]
        patterns = []
        tol = 0.03
        peaks   = _find_peaks(h_lb, 5)
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
                res = sf(float(np.max(h_seg[-10:]))); rng = float(np.max(h_seg)) - float(np.min(l_seg))
                patterns.append({'type': 'ascending_triangle', 'label': 'Yukselen Ucgen', 'signal': 'buy',
                    'reliability': 'medium', 'completed': bool(c_lb[-1] > float(np.max(h_seg))),
                    'description': f'Duz direnc ({res}) + yukselen dip -> Yukari kirilim beklenir',
                    'resistance': res, 'target': sf(float(np.max(h_seg)) + rng), 'barsAgo': 0})
            elif abs(l_pct) < 0.08 and h_pct < -0.08:
                sup = sf(float(np.min(l_seg[-10:]))); rng = float(np.max(h_seg)) - float(np.min(l_seg))
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
        if completed_p:                       overall = completed_p[0]['signal']
        elif len(bull_patt) > len(bear_patt): overall = 'buy'
        elif len(bear_patt) > len(bull_patt): overall = 'sell'
        else:                                 overall = 'neutral'
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
