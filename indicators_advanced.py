"""
BIST Pro - Advanced Indicators
Fibonacci Adv, Pivot Adv, Advanced Indicators, Dynamic Thresholds,
Candlestick Patterns, Chart Data Prep
"""
import numpy as np
import pandas as pd
from config import sf, si
from indicators_basic import calc_rsi_single


# =====================================================================
# FİBONACCİ RETRACEMENT & EXTENSION
# =====================================================================
def calc_fibonacci_adv(hist, lookback=60):
    """Fibonacci retracement ve extension seviyeleri."""
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float) if 'High' in hist.columns else c.copy()
        l = hist['Low'].values.astype(float) if 'Low' in hist.columns else c.copy()
        h = np.where(np.isnan(h), c, h); l = np.where(np.isnan(l), c, l)
        n = len(c); lb = min(lookback, n)
        seg_h, seg_l = h[-lb:], l[-lb:]
        hi_idx = int(np.argmax(seg_h)); lo_idx = int(np.argmin(seg_l))
        swing_high = float(seg_h[hi_idx]); swing_low = float(seg_l[lo_idx])
        diff = swing_high - swing_low; cur = float(c[-1])

        trend = 'uptrend' if hi_idx > lo_idx else 'downtrend'
        base, top = swing_low, swing_high
        ret_ratios = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
        ext_ratios = [1.272, 1.618, 2.618]

        def label_level(lvl):
            if lvl < cur - diff * 0.01: return 'support'
            elif lvl > cur + diff * 0.01: return 'resistance'
            return 'current'

        retracements = []
        for r in ret_ratios:
            lvl = sf(top - diff * r)
            retracements.append({'ratio': r, 'label': f'Fib {r:.3f}', 'level': lvl,
                'role': label_level(float(lvl)),
                'distPct': sf((float(lvl) - cur) / cur * 100)})

        extensions = []
        for r in ext_ratios:
            lvl = sf(top + diff * (r - 1)) if trend == 'uptrend' else sf(base - diff * (r - 1))
            extensions.append({'ratio': r, 'label': f'Fib {r:.3f}', 'level': lvl,
                'role': 'extension_target',
                'distPct': sf((float(lvl) - cur) / cur * 100)})

        supports = sorted([lv for lv in retracements if lv['role'] == 'support'],
                          key=lambda x: -float(x['level']))[:3]
        resistances = sorted([lv for lv in retracements if lv['role'] == 'resistance'],
                             key=lambda x: float(x['level']))[:3]
        golden_top = sf(top - diff * 0.618)
        golden_bot = sf(top - diff * 0.65)
        in_golden = float(golden_bot) <= cur <= float(golden_top)

        return {
            'trend': trend, 'swingHigh': sf(swing_high), 'swingLow': sf(swing_low),
            'currentPrice': sf(cur), 'retracements': retracements, 'extensions': extensions,
            'nearestSupports': supports, 'nearestResistances': resistances,
            'goldenPocket': {'top': golden_top, 'bottom': golden_bot, 'inZone': in_golden},
        }
    except Exception as e:
        print(f"  [FIB] Hata: {e}")
        return {'error': str(e), 'retracements': [], 'extensions': []}


# =====================================================================
# PİVOT NOKTALARI (DETAYLI)
# =====================================================================
def calc_pivot_points_adv(hist):
    """Klasik, Camarilla ve Woodie Pivot Noktalari (detayli)."""
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float) if 'High' in hist.columns else c.copy()
        l = hist['Low'].values.astype(float) if 'Low' in hist.columns else c.copy()
        o = hist['Open'].values.astype(float) if 'Open' in hist.columns else c.copy()
        h = np.where(np.isnan(h), c, h); l = np.where(np.isnan(l), c, l); o = np.where(np.isnan(o), c, o)

        H, L, C, O = float(h[-2]), float(l[-2]), float(c[-2]), float(o[-2])
        rng = H - L; cur = float(c[-1])

        def _role(lvl):
            return 'support' if lvl < cur else ('resistance' if lvl > cur else 'pivot')

        pp_c = (H + L + C) / 3
        classic = {'pp': sf(pp_c), 'r1': sf(2 * pp_c - L), 'r2': sf(pp_c + rng),
                   'r3': sf(H + 2 * (pp_c - L)), 's1': sf(2 * pp_c - H),
                   's2': sf(pp_c - rng), 's3': sf(L - 2 * (H - pp_c))}

        cam = {'pp': sf(pp_c),
               'r1': sf(C + rng * 1.1 / 12), 'r2': sf(C + rng * 1.1 / 6),
               'r3': sf(C + rng * 1.1 / 4), 'r4': sf(C + rng * 1.1 / 2),
               's1': sf(C - rng * 1.1 / 12), 's2': sf(C - rng * 1.1 / 6),
               's3': sf(C - rng * 1.1 / 4), 's4': sf(C - rng * 1.1 / 2)}

        pp_w = (H + L + 2 * C) / 4
        woodie = {'pp': sf(pp_w), 'r1': sf(2 * pp_w - L), 'r2': sf(pp_w + rng),
                  's1': sf(2 * pp_w - H), 's2': sf(pp_w - rng)}

        all_levels = [{'model': 'classic', 'name': name.upper(), 'level': val, 'role': _role(float(val))}
                      for name, val in classic.items()]
        supports = sorted([lv for lv in all_levels if lv['role'] == 'support'],
                          key=lambda x: -float(x['level']))[:3]
        resistances = sorted([lv for lv in all_levels if lv['role'] == 'resistance'],
                             key=lambda x: float(x['level']))[:3]
        bias = 'bullish' if cur > float(classic['pp']) else 'bearish'

        return {'currentPrice': sf(cur), 'bias': bias,
                'classic': classic, 'camarilla': cam, 'woodie': woodie,
                'nearestSupports': supports, 'nearestResistances': resistances}
    except Exception as e:
        print(f"  [PIVOT] Hata: {e}")
        return {'error': str(e), 'classic': {}, 'camarilla': {}, 'woodie': {}}


# =====================================================================
# İLERİ TEKNİK İNDİKATÖRLER
# =====================================================================
def calc_advanced_indicators(hist):
    """Ichimoku Cloud, Stochastic (14,3,3), Williams %R (14)"""
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float) if 'High' in hist.columns else c.copy()
        l = hist['Low'].values.astype(float) if 'Low' in hist.columns else c.copy()
        h = np.where(np.isnan(h), c, h); l = np.where(np.isnan(l), c, l)
        n = len(c)
        result = {}

        # Ichimoku
        if n >= 52:
            def mid(arr, period):
                return (pd.Series(arr).rolling(period).max() + pd.Series(arr).rolling(period).min()) / 2
            tenkan = mid(h, 9).values; kijun = mid(h, 26).values
            senkou_a = ((pd.Series(tenkan) + pd.Series(kijun)) / 2).shift(26).values
            senkou_b = mid(h, 52).shift(26).values
            cur_price = float(c[-1])
            sa = float(senkou_a[-27]) if not np.isnan(senkou_a[-27]) else 0
            sb = float(senkou_b[-27]) if not np.isnan(senkou_b[-27]) else 0
            cloud_top, cloud_bot = max(sa, sb), min(sa, sb)
            tk = float(tenkan[-1]) if not np.isnan(tenkan[-1]) else cur_price
            kj = float(kijun[-1]) if not np.isnan(kijun[-1]) else cur_price
            above_cloud = cur_price > cloud_top
            below_cloud = cur_price < cloud_bot
            in_cloud = cloud_bot <= cur_price <= cloud_top
            tk_kj_cross = 'bullish' if tk > kj else ('bearish' if tk < kj else 'neutral')
            ich_signal = ('buy' if above_cloud and tk > kj
                          else ('sell' if below_cloud and tk < kj else 'neutral'))
            result['ichimoku'] = {
                'tenkan': sf(tk), 'kijun': sf(kj), 'senkouA': sf(sa), 'senkouB': sf(sb),
                'cloudTop': sf(cloud_top), 'cloudBottom': sf(cloud_bot),
                'aboveCloud': above_cloud, 'belowCloud': below_cloud, 'inCloud': in_cloud,
                'tkKjCross': tk_kj_cross, 'signal': ich_signal}
        else:
            result['ichimoku'] = {'signal': 'neutral', 'error': 'Yetersiz veri (min 52 bar)'}

        # Stochastic (14, 3, 3)
        if n >= 17:
            h_ser, l_ser, c_ser = pd.Series(h), pd.Series(l), pd.Series(c)
            highest_h = h_ser.rolling(14).max(); lowest_l = l_ser.rolling(14).min()
            raw_k = 100 * (c_ser - lowest_l) / (highest_h - lowest_l + 1e-10)
            k_line = raw_k.rolling(3).mean(); d_line = k_line.rolling(3).mean()
            k_val, d_val = sf(float(k_line.iloc[-1])), sf(float(d_line.iloc[-1]))
            if float(k_val) < 20 and float(d_val) < 20: sto_signal = 'buy'
            elif float(k_val) > 80 and float(d_val) > 80: sto_signal = 'sell'
            elif float(k_val) > float(d_val) and float(k_val) < 50: sto_signal = 'buy'
            elif float(k_val) < float(d_val) and float(k_val) > 50: sto_signal = 'sell'
            else: sto_signal = 'neutral'
            result['stochastic'] = {'k': k_val, 'd': d_val,
                'overbought': float(k_val) > 80, 'oversold': float(k_val) < 20, 'signal': sto_signal}
        else:
            result['stochastic'] = {'signal': 'neutral', 'k': 50, 'd': 50}

        # Williams %R (14)
        if n >= 14:
            highest_h = float(np.max(h[-14:])); lowest_l = float(np.min(l[-14:]))
            wr = sf(((highest_h - float(c[-1])) / (highest_h - lowest_l + 1e-10)) * -100)
            wr_signal = 'buy' if float(wr) < -80 else ('sell' if float(wr) > -20 else 'neutral')
            result['williamsR'] = {'value': wr, 'overbought': float(wr) > -20,
                'oversold': float(wr) < -80, 'signal': wr_signal}
        else:
            result['williamsR'] = {'signal': 'neutral', 'value': -50}

        signals = [result.get('ichimoku', {}).get('signal', 'neutral'),
                   result.get('stochastic', {}).get('signal', 'neutral'),
                   result.get('williamsR', {}).get('signal', 'neutral')]
        buy_cnt, sell_cnt = signals.count('buy'), signals.count('sell')
        result['summary'] = {'signal': 'buy' if buy_cnt > sell_cnt else ('sell' if sell_cnt > buy_cnt else 'neutral'),
                             'buyCount': buy_cnt, 'sellCount': sell_cnt}
        return result
    except Exception as e:
        print(f"  [ADV-IND] Hata: {e}")
        return {'error': str(e), 'summary': {'signal': 'neutral', 'buyCount': 0, 'sellCount': 0}}


# =====================================================================
# DİNAMİK EŞİKLER
# =====================================================================
def calc_dynamic_thresholds(closes, highs, lows, volumes):
    """Her hisse icin tarihsel dagilima gore adaptif RSI/BB/Volume esikleri"""
    try:
        n = len(closes)
        if n < 60:
            return {'rsi_oversold': 30, 'rsi_overbought': 70, 'vol_spike': 2.0, 'bb_std': 2.0}
        rsi_values = []
        for i in range(20, n):
            rv = calc_rsi_single(closes[:i+1])
            if rv is not None:
                rsi_values.append(rv)
        if len(rsi_values) < 20:
            return {'rsi_oversold': 30, 'rsi_overbought': 70, 'vol_spike': 2.0, 'bb_std': 2.0}
        rsi_arr = np.array(rsi_values)
        rsi_oversold = max(20, min(40, float(sf(np.percentile(rsi_arr, 10)))))
        rsi_overbought = max(60, min(85, float(sf(np.percentile(rsi_arr, 90)))))
        if n >= 60:
            daily_returns = np.diff(closes[-60:]) / closes[-60:-1]
            vol = float(np.std(daily_returns))
            bb_std = max(1.5, min(3.0, 2.0 * (vol / 0.02)))
        else:
            bb_std = 2.0
        if n >= 30:
            vol_mean = float(np.mean(volumes[-30:]))
            vol_std = float(np.std(volumes[-30:]))
            vol_spike = max(1.5, min(3.0, (vol_mean + vol_std) / vol_mean if vol_mean > 0 else 2.0))
        else:
            vol_spike = 2.0
        return {'rsi_oversold': sf(rsi_oversold), 'rsi_overbought': sf(rsi_overbought),
                'vol_spike': sf(vol_spike), 'bb_std': sf(bb_std)}
    except Exception:
        return {'rsi_oversold': 30, 'rsi_overbought': 70, 'vol_spike': 2.0, 'bb_std': 2.0}


# =====================================================================
# MUM FORMASYONLARI
# =====================================================================
def calc_candlestick_patterns(opens, highs, lows, closes):
    """Mum formasyonlarini tespit et"""
    try:
        n = len(closes)
        if n < 5:
            return {'patterns': [], 'signal': 'neutral'}
        patterns = []
        o = [float(x) for x in opens[-5:]]
        h = [float(x) for x in highs[-5:]]
        l = [float(x) for x in lows[-5:]]
        c = [float(x) for x in closes[-5:]]
        body = [c[i] - o[i] for i in range(5)]
        body_abs = [abs(b) for b in body]
        upper_shadow = [h[i] - max(o[i], c[i]) for i in range(5)]
        lower_shadow = [min(o[i], c[i]) - l[i] for i in range(5)]
        total_range = [h[i] - l[i] for i in range(5)]
        i = 4
        avg_body = np.mean(body_abs[:4]) if np.mean(body_abs[:4]) > 0 else 0.01

        if body_abs[i] < total_range[i] * 0.1 and total_range[i] > 0:
            patterns.append({'name': 'Doji', 'type': 'neutral', 'description': 'Kararsizlik formasyonu.', 'strength': 2})
        if lower_shadow[i] > body_abs[i] * 2 and upper_shadow[i] < body_abs[i] * 0.5 and body[i-1] < 0:
            patterns.append({'name': 'Cekic (Hammer)', 'type': 'bullish', 'description': 'Dusus sonrasi toparlanma sinyali.', 'strength': 3})
        if upper_shadow[i] > body_abs[i] * 2 and lower_shadow[i] < body_abs[i] * 0.5 and body[i-1] > 0:
            patterns.append({'name': 'Kayan Yildiz (Shooting Star)', 'type': 'bearish', 'description': 'Yukselis sonrasi satis baskisi.', 'strength': 3})
        if body[i] > 0 and body[i-1] < 0 and o[i] <= c[i-1] and c[i] >= o[i-1] and body_abs[i] > body_abs[i-1]:
            patterns.append({'name': 'Yukari Yutan (Bullish Engulfing)', 'type': 'bullish', 'description': 'Guclu alis formasyonu.', 'strength': 4})
        if body[i] < 0 and body[i-1] > 0 and o[i] >= c[i-1] and c[i] <= o[i-1] and body_abs[i] > body_abs[i-1]:
            patterns.append({'name': 'Asagi Yutan (Bearish Engulfing)', 'type': 'bearish', 'description': 'Guclu satis formasyonu.', 'strength': 4})
        if n >= 3 and body[i-2] < 0 and body_abs[i-2] > avg_body and body_abs[i-1] < avg_body * 0.5 and body[i] > 0 and body_abs[i] > avg_body:
            patterns.append({'name': 'Sabah Yildizi (Morning Star)', 'type': 'bullish', 'description': 'Guclu 3 mumlu dip formasyonu.', 'strength': 5})
        if n >= 3 and body[i-2] > 0 and body_abs[i-2] > avg_body and body_abs[i-1] < avg_body * 0.5 and body[i] < 0 and body_abs[i] > avg_body:
            patterns.append({'name': 'Aksam Yildizi (Evening Star)', 'type': 'bearish', 'description': 'Guclu 3 mumlu tepe formasyonu.', 'strength': 5})
        if body[i] > 0 and body[i-1] > 0 and body[i-2] > 0 and c[i] > c[i-1] > c[i-2] and body_abs[i] > avg_body * 0.5 and body_abs[i-1] > avg_body * 0.5:
            patterns.append({'name': 'Uc Beyaz Asker', 'type': 'bullish', 'description': 'Art arda 3 guclu yukselis mumu.', 'strength': 4})
        if body[i] < 0 and body[i-1] < 0 and body[i-2] < 0 and c[i] < c[i-1] < c[i-2] and body_abs[i] > avg_body * 0.5 and body_abs[i-1] > avg_body * 0.5:
            patterns.append({'name': 'Uc Kara Karga', 'type': 'bearish', 'description': 'Art arda 3 guclu dusus mumu.', 'strength': 4})
        if total_range[i] > 0:
            shadow_ratio = (upper_shadow[i] + lower_shadow[i]) / total_range[i]
            if shadow_ratio < 0.1 and body_abs[i] > avg_body * 1.5:
                mtype = 'bullish' if body[i] > 0 else 'bearish'
                patterns.append({'name': f'Marubozu ({"Yukari" if mtype == "bullish" else "Asagi"})', 'type': mtype,
                    'description': 'Golgesiz guclu mum. Trend devami beklenir.', 'strength': 3})

        bullish = sum(1 for p in patterns if p['type'] == 'bullish')
        bearish = sum(1 for p in patterns if p['type'] == 'bearish')
        signal = 'buy' if bullish > bearish else ('sell' if bearish > bullish else 'neutral')
        return {'patterns': patterns, 'signal': signal, 'bullishCount': bullish, 'bearishCount': bearish}
    except Exception as e:
        print(f"  [CANDLE] Hata: {e}")
        return {'patterns': [], 'signal': 'neutral'}


# Market regime cache (signals.py tarafindan kullanilir)
_market_regime_cache = {'regime': None, 'ts': 0}


def prepare_chart_data(hist):
    try:
        cs=[{'date':d.strftime('%Y-%m-%d'),'open':sf(r['Open']),'high':sf(r['High']),'low':sf(r['Low']),'close':sf(r['Close']),'volume':si(r['Volume'])} for d,r in hist.iterrows()]
        return {'candlestick':cs,'dates':[c['date'] for c in cs],'prices':[c['close'] for c in cs],'volumes':[c['volume'] for c in cs],'dataPoints':len(cs)}
    except Exception: return {'candlestick':[],'dates':[],'prices':[],'volumes':[],'dataPoints':0}
