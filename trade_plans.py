"""
BIST Pro - Trade Plan Module
"""
import time, json, traceback
import numpy as np
from config import (
    sf, _lock, _plan_lock_cache, _plan_lock_cache_lock,
    PLAN_LOCK_CONFIG, PLAN_MAX_LOCK_SECONDS, _hist_cache, _cget_hist
)
from indicators import *

def _is_plan_valid(lock_entry, cur_price, cur_signal, cfg=None):
    """Kilitli bir planin hala gecerli olup olmadigini kontrol et.
    cfg: PLAN_LOCK_CONFIG[timeframe] — yoksa daily config kullanilir.
    Returns (valid: bool, reason: str)"""
    if cfg is None:
        cfg = PLAN_LOCK_CONFIG['daily']
    now = time.time()
    age = now - lock_entry['locked_at']

    if age > cfg['max_lock']:
        return False, 'max_lock_exceeded'

    if age < cfg['min_lock']:
        return True, 'min_lock_active'

    locked_signal = lock_entry['signal']
    tf            = lock_entry.get('timeframe', 'daily')
    tf_plan       = lock_entry.get('tf_plan', {})  # Tek timeframe plani

    # Sinyal yonu degistiyse kilidi ac
    if cur_signal and cur_signal != locked_signal and cur_signal not in ('BEKLE', 'neutral'):
        return False, 'signal_changed'

    # Stop-loss tetiklendi mi?
    if locked_signal == 'AL':
        sl = tf_plan.get('buy', {}).get('stopLoss', 0)
        if sl and float(sl) > 0 and cur_price < float(sl):
            return False, 'stop_loss_hit'
    elif locked_signal == 'SAT':
        sl = tf_plan.get('sell', {}).get('stopLoss', 0)
        if sl and float(sl) > 0 and cur_price > float(sl):
            return False, 'stop_loss_hit'

    return True, 'valid'


def calc_trade_plan(hist, indicators=None, symbol=None):
    """Her hisse icin gunluk/haftalik/aylik bazda detayli al-sat plani
    Entry, stop-loss, 3 hedef, risk/reward orani hesaplar.
    symbol verilirse plan kilitlenir, her guncellemede degismez."""
    try:
        cur_price_init = float(hist['Close'].values[-1])
        cur_signal_init = (indicators or {}).get('recommendation', {}).get('signal', '')

        # Per-timeframe kilit kontrolu: hangi tf'ler hala kilitli?
        tf_locked = {}  # { 'daily': tf_plan_dict, ... }
        if symbol:
            for tf_label, cfg in PLAN_LOCK_CONFIG.items():
                lock_key = f"{symbol}_{tf_label}"
                with _plan_lock_cache_lock:
                    lock_entry = _plan_lock_cache.get(lock_key)
                if lock_entry:
                    valid, reason = _is_plan_valid(lock_entry, cur_price_init, cur_signal_init, cfg)
                    if valid:
                        tf_locked[tf_label] = lock_entry['tf_plan']
                    else:
                        with _plan_lock_cache_lock:
                            _plan_lock_cache.pop(lock_key, None)
                        print(f"[PLAN-LOCK] {symbol}_{tf_label} kilidi acildi: {reason}")

        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        # NaN temizligi
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        v = np.where(np.isnan(v), 0, v)
        n = len(c)
        cur = float(c[-1])

        if n < 20:
            return {}

        # Destek/Direnc hesapla
        sr = calc_support_resistance(hist)
        supports = sr.get('supports', [])
        resistances = sr.get('resistances', [])

        # Fibonacci
        fib = calc_fibonacci(hist)
        fib_levels = fib.get('levels', {})
        fib_values = sorted([float(v) for v in fib_levels.values() if v > 0])

        # Pivot Points
        pp = calc_pivot_points(hist)
        classic = pp.get('classic', {})

        # ATR (volatilite bazli hedef)
        atr_data = calc_atr(h, l, c)
        atr_val = float(atr_data.get('value', 0))
        atr_pct = float(atr_data.get('pct', 2))

        # Bollinger
        bb = calc_bollinger(c, cur)
        bb_upper = float(bb.get('upper', 0))
        bb_lower = float(bb.get('lower', 0))
        bb_middle = float(bb.get('middle', 0))

        # EMA seviyeleri
        s_pd = pd.Series(c)
        ema20 = float(s_pd.ewm(span=20).mean().iloc[-1]) if n >= 20 else cur
        ema50 = float(s_pd.ewm(span=50).mean().iloc[-1]) if n >= 50 else cur
        ema200 = float(s_pd.ewm(span=200).mean().iloc[-1]) if n >= 200 else cur

        # RSI
        rsi_val = float(calc_rsi(c).get('value', 50))

        # Dinamik esikler
        dyn = calc_dynamic_thresholds(c, h, l, v)
        dyn_oversold = float(dyn.get('rsi_oversold', 30))
        dyn_overbought = float(dyn.get('rsi_overbought', 70))

        plans = {}

        # Kilitli timeframe'leri direkt kullan
        for tf_label, cached_tf in tf_locked.items():
            plans[tf_label] = cached_tf

        for tf_label, tf_days, atr_mult, target_mult in [
            ('daily', 1, 1.0, [1.0, 1.5, 2.5]),
            ('weekly', 5, 1.5, [1.5, 2.5, 4.0]),
            ('monthly', 22, 2.5, [2.5, 4.0, 6.0]),
        ]:
            # Zaten kilitliyse hesaplama
            if tf_label in tf_locked:
                continue

            # Bu zaman dilimi icin yeterli veri var mi
            if n < tf_days + 20:
                continue

            # Zaman dilimi bazli getiri ve momentum
            tf_slice = c[-tf_days:] if tf_days <= n else c
            tf_high = float(np.max(h[-tf_days:])) if tf_days <= n else float(np.max(h))
            tf_low = float(np.min(l[-tf_days:])) if tf_days <= n else float(np.min(l))
            tf_ret = ((cur - float(c[-tf_days-1])) / float(c[-tf_days-1]) * 100) if n > tf_days else 0
            tf_range = tf_high - tf_low

            # Trend yonu (bu zaman diliminde)
            if n >= tf_days + 20:
                sma_short = float(np.mean(c[-min(10, tf_days):]))
                sma_long = float(np.mean(c[-min(tf_days+10, n):]))
                trend = 'yukari' if sma_short > sma_long else ('asagi' if sma_short < sma_long else 'yatay')
            else:
                trend = 'yatay'

            # ========== ALIS PLANI ==========
            buy_entry = None
            buy_sl = None
            buy_targets = []
            buy_reasons = []
            buy_strategy = ''

            # En yakin destek = alis noktasi
            nearby_supports = [s for s in supports if s < cur and s > cur * 0.90]
            nearby_resistances = [r for r in resistances if r > cur and r < cur * 1.20]

            # Fib destek seviyeleri
            fib_supports = [v for v in fib_values if v < cur and v > cur * 0.90]
            fib_resistances = [v for v in fib_values if v > cur and v < cur * 1.20]

            # -- Strateji 1: Destek'ten alis
            if nearby_supports:
                best_support = nearby_supports[0]
                buy_entry = sf(best_support)
                buy_sl = sf(best_support - atr_val * atr_mult)
                buy_reasons.append(f'Destek seviyesinden alis ({sf(best_support)} TL)')
                buy_strategy = 'destek_alis'

            # -- Strateji 2: Bollinger alt bant
            elif bb_lower > 0 and cur < bb_middle:
                buy_entry = sf(bb_lower)
                buy_sl = sf(bb_lower - atr_val * atr_mult)
                buy_reasons.append(f'Bollinger alt bant ({sf(bb_lower)} TL) yakininda alis')
                buy_strategy = 'bollinger_alis'

            # -- Strateji 3: EMA geri cekilme
            elif cur > ema20 > ema50:
                buy_entry = sf(ema20)
                buy_sl = sf(ema50 - atr_val * 0.5)
                buy_reasons.append(f'EMA20 ({sf(ema20)} TL) geri cekilmesinde alis')
                buy_strategy = 'ema_pullback'

            # -- Strateji 4: Momentum alis (mevcut fiyattan)
            elif rsi_val < dyn_oversold + 10 and trend != 'asagi':
                buy_entry = sf(cur)
                buy_sl = sf(cur - atr_val * atr_mult)
                buy_reasons.append(f'RSI={sf(rsi_val)} dusuk, momentum alisi')
                buy_strategy = 'momentum_alis'

            # -- Fallback: Pivot S1 seviyesinden
            else:
                s1 = float(classic.get('s1', cur * 0.97))
                buy_entry = sf(s1)
                buy_sl = sf(float(classic.get('s2', s1 - atr_val)))
                buy_reasons.append(f'Pivot S1 ({sf(s1)} TL) seviyesinden alis')
                buy_strategy = 'pivot_alis'

            # Giris fiyati mevcut fiyattan cok uzaksa (>%1.5) mevcut fiyati kullan
            # %1.5'ten fazla aşağıda bir destek varsa beklemek yerine şu anki fiyattan gir
            if buy_entry and float(buy_entry) < cur * 0.985:
                buy_entry = sf(cur)
                buy_sl = sf(cur - atr_val * atr_mult)
                if 'destek_alis' in buy_strategy or 'bollinger_alis' in buy_strategy or 'pivot_alis' in buy_strategy:
                    buy_reasons.append(f'Giris mevcut fiyata ({sf(cur)} TL) ayarlandi')

            # Alis hedefleri
            entry_price = float(buy_entry)
            sl_price = float(buy_sl)
            risk = entry_price - sl_price

            if risk > 0:
                for i, mult in enumerate(target_mult):
                    raw_target = entry_price + risk * mult
                    # Direnc seviyesine yakin mi?
                    snapped = raw_target
                    for r in (nearby_resistances + fib_resistances):
                        if abs(r - raw_target) < atr_val:
                            snapped = r
                            break
                    buy_targets.append(sf(snapped))
                    buy_reasons.append(f'Hedef {i+1}: {sf(snapped)} TL (R/R {mult:.1f}x)')

            # Risk/Reward orani
            if risk > 0 and buy_targets:
                rr_ratio = sf((float(buy_targets[-1]) - entry_price) / risk)
            else:
                rr_ratio = 0

            # ========== SATIS PLANI ==========
            sell_entry = None
            sell_sl = None
            sell_targets = []
            sell_reasons = []
            sell_strategy = ''

            # -- Strateji 1: Direncten satis
            if nearby_resistances:
                best_res = nearby_resistances[0]
                sell_entry = sf(best_res)
                sell_sl = sf(best_res + atr_val * atr_mult)
                sell_reasons.append(f'Direnc seviyesinde satis ({sf(best_res)} TL)')
                sell_strategy = 'direnc_satis'

            # -- Strateji 2: Bollinger ust bant
            elif bb_upper > 0 and cur > bb_middle:
                sell_entry = sf(bb_upper)
                sell_sl = sf(bb_upper + atr_val * atr_mult)
                sell_reasons.append(f'Bollinger ust bant ({sf(bb_upper)} TL) yakininda satis')
                sell_strategy = 'bollinger_satis'

            # -- Strateji 3: RSI asiri alim
            elif rsi_val > dyn_overbought - 5:
                sell_entry = sf(cur)
                sell_sl = sf(cur + atr_val * atr_mult)
                sell_reasons.append(f'RSI={sf(rsi_val)} yuksek, satis baskisi')
                sell_strategy = 'rsi_satis'

            # -- Fallback: Pivot R1
            else:
                r1 = float(classic.get('r1', cur * 1.03))
                sell_entry = sf(r1)
                sell_sl = sf(float(classic.get('r2', r1 + atr_val)))
                sell_reasons.append(f'Pivot R1 ({sf(r1)} TL) seviyesinde satis')
                sell_strategy = 'pivot_satis'

            # Satis girisi mevcut fiyattan cok uzaksa (>%1.5 yukarda) mevcut fiyati kullan
            if sell_entry and float(sell_entry) > cur * 1.015:
                sell_entry = sf(cur)
                sell_sl = sf(cur + atr_val * atr_mult)
                if 'direnc_satis' in sell_strategy or 'bollinger_satis' in sell_strategy or 'pivot_satis' in sell_strategy:
                    sell_reasons.append(f'Giris mevcut fiyata ({sf(cur)} TL) ayarlandi')

            # Satis hedefleri (asagi)
            s_entry = float(sell_entry)
            s_risk = float(sell_sl) - s_entry

            if s_risk > 0:
                for i, mult in enumerate(target_mult):
                    raw_target = s_entry - s_risk * mult
                    # Destege yakin mi?
                    snapped = raw_target
                    for sup in (nearby_supports + fib_supports):
                        if abs(sup - raw_target) < atr_val:
                            snapped = sup
                            break
                    sell_targets.append(sf(max(snapped, 0.01)))
                    sell_reasons.append(f'Hedef {i+1}: {sf(max(snapped, 0.01))} TL')

            # Sell R/R
            if s_risk > 0 and sell_targets:
                sell_rr = sf((s_entry - float(sell_targets[-1])) / s_risk)
            else:
                sell_rr = 0

            # Mevcut fiyatin plana gore durumu
            if float(buy_entry) > 0:
                entry_dist = sf(((cur - float(buy_entry)) / float(buy_entry)) * 100)
            else:
                entry_dist = 0

            # Genel sinyal (bu zaman dilimi icin)
            if trend == 'yukari' and rsi_val < 60:
                tf_signal = 'AL'
                tf_signal_desc = 'Trend yukari, geri cekilmede alis firsati'
            elif trend == 'yukari' and rsi_val >= 60:
                tf_signal = 'BEKLE'
                tf_signal_desc = 'Trend yukari ama asiri alimda, geri cekilme bekle'
            elif trend == 'asagi' and rsi_val > 40:
                tf_signal = 'SAT'
                tf_signal_desc = 'Trend asagi, direncte satis firsati'
            elif trend == 'asagi' and rsi_val <= 40:
                tf_signal = 'BEKLE'
                tf_signal_desc = 'Trend asagi ama asiri satimda, toparlanma bekle'
            else:
                tf_signal = 'BEKLE'
                tf_signal_desc = 'Belirsiz yonde, net sinyal icin bekle'

            plans[tf_label] = {
                'trend': trend,
                'signal': tf_signal,
                'signalDescription': tf_signal_desc,
                'momentum': sf(tf_ret),
                'range': sf(tf_range),
                'atr': sf(atr_val),
                'rsi': sf(rsi_val),

                # Alis plani
                'buy': {
                    'entry': buy_entry,
                    'stopLoss': buy_sl,
                    'targets': buy_targets,
                    'riskReward': rr_ratio,
                    'strategy': buy_strategy,
                    'reasons': buy_reasons[:5],
                    'risk': sf(risk) if risk > 0 else 0,
                    'distanceFromEntry': entry_dist,
                },

                # Satis plani
                'sell': {
                    'entry': sell_entry,
                    'stopLoss': sell_sl,
                    'targets': sell_targets,
                    'riskReward': sell_rr,
                    'strategy': sell_strategy,
                    'reasons': sell_reasons[:5],
                },

                # Onemli seviyeler
                'keyLevels': {
                    'supports': supports[:3],
                    'resistances': resistances[:3],
                    'ema20': sf(ema20),
                    'ema50': sf(ema50),
                    'ema200': sf(ema200) if n >= 200 else None,
                    'bbUpper': sf(bb_upper),
                    'bbLower': sf(bb_lower),
                    'bbMiddle': sf(bb_middle),
                    'pivotPP': classic.get('pp', 0),
                    'pivotR1': classic.get('r1', 0),
                    'pivotS1': classic.get('s1', 0),
                },
            }

        # Her timeframe'i ayri ayri kilitle (sadece yeni hesaplananlar)
        if symbol and plans:
            now_ts = time.time()
            cur_p  = float(c[-1])
            changed = False
            for tf_label, tf_plan in plans.items():
                if tf_label in tf_locked:
                    continue  # Zaten kilitliydi, dokunma
                cfg = PLAN_LOCK_CONFIG.get(tf_label, PLAN_LOCK_CONFIG['daily'])
                lock_key = f"{symbol}_{tf_label}"
                with _plan_lock_cache_lock:
                    _plan_lock_cache[lock_key] = {
                        'tf_plan':      tf_plan,
                        'locked_at':    now_ts,
                        'locked_price': cur_p,
                        'signal':       tf_plan.get('signal', 'BEKLE'),
                        'timeframe':    tf_label,
                        'max_lock':     cfg['max_lock'],
                        'min_lock':     cfg['min_lock'],
                    }
                changed = True
            if changed:
                threading.Thread(target=_db_save_plan_locks, daemon=True).start()

        return plans
    except Exception as e:
        print(f"  [TRADE-PLAN] Hata: {e}")
        return {}


def _db_save_plan_locks():
    """Plan kilitleme cache'ini DB'ye kaydet"""
    try:
        with _plan_lock_cache_lock:
            snap = dict(_plan_lock_cache)
        _db_upsert_cache('plan_locks', json.dumps(snap), time.time())
    except Exception as e:
        print(f"[PLAN-LOCK-SAVE] Hata: {e}")


def prepare_chart_data(hist):
    try:
        cs=[{'date':d.strftime('%Y-%m-%d'),'open':sf(r['Open']),'high':sf(r['High']),'low':sf(r['Low']),'close':sf(r['Close']),'volume':si(r['Volume'])} for d,r in hist.iterrows()]
        return {'candlestick':cs,'dates':[c['date'] for c in cs],'prices':[c['close'] for c in cs],'volumes':[c['volume'] for c in cs],'dataPoints':len(cs)}
    except: return {'candlestick':[],'dates':[],'prices':[],'volumes':[],'dataPoints':0}


