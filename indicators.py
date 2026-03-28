"""
BIST Pro - Technical Indicators Module (Facade)
Tum indicator fonksiyonlarini alt modullerden re-export eder.
Mevcut importlar bozulmasin diye bu dosya korunur.
"""
import numpy as np
import pandas as pd
from config import sf, si

# =====================================================================
# ALT MODULLERDEN RE-EXPORT
# =====================================================================

# Basic indicators
from indicators_basic import (
    _resample_to_tf, EMA_PERIODS,
    calc_rsi, calc_rsi_single, calc_macd, calc_macd_history,
    calc_bollinger, calc_bollinger_history,
    calc_ema, calc_ema_history,
    calc_stochastic, calc_stochastic_history,
    calc_atr, calc_adx, calc_obv,
    calc_williams_r, calc_cci, calc_mfi, calc_vwap,
    calc_ichimoku, calc_psar,
    calc_support_resistance, calc_fibonacci, calc_pivot_points,
    calc_roc, calc_aroon, calc_trix, calc_dmi,
)

# Pattern detection
from indicators_patterns import (
    _rsi_series, _find_peaks, _find_troughs,
    calc_mtf_signal, calc_divergence, calc_volume_profile,
    calc_smc, calc_chart_patterns,
)

# Advanced indicators
from indicators_advanced import (
    calc_fibonacci_adv, calc_pivot_points_adv,
    calc_advanced_indicators,
    calc_dynamic_thresholds, calc_candlestick_patterns,
    prepare_chart_data, _market_regime_cache,
)


# =====================================================================
# ORCHESTRATOR: calc_all_indicators
# =====================================================================
def calc_all_indicators(hist, cp):
    """Tum teknik indikatorleri tek seferde hesapla"""
    c = hist['Close'].values.astype(float)
    h = hist['High'].values.astype(float)
    l = hist['Low'].values.astype(float)
    v = hist['Volume'].values.astype(float)
    o = hist['Open'].values.astype(float) if 'Open' in hist.columns else c.copy()
    h = np.where(np.isnan(h), c, h)
    l = np.where(np.isnan(l), c, l)
    v = np.where(np.isnan(v), 0, v)
    o = np.where(np.isnan(o), c, o)
    cp = float(cp)

    rsi_h = [{'date': hist.index[i].strftime('%Y-%m-%d'), 'value': rv}
             for i in range(14, len(c))
             if (rv := calc_rsi_single(c[:i+1])) is not None]

    dyn_thresholds = calc_dynamic_thresholds(c, h, l, v)

    rsi_data = calc_rsi(c)
    rsi_val = rsi_data.get('value', 50)
    dyn_oversold = float(dyn_thresholds.get('rsi_oversold', 30))
    dyn_overbought = float(dyn_thresholds.get('rsi_overbought', 70))
    if rsi_val < dyn_oversold:
        rsi_data['signal'] = 'buy'
        rsi_data['dynamicNote'] = f'Dinamik esik: <{dyn_oversold}'
    elif rsi_val > dyn_overbought:
        rsi_data['signal'] = 'sell'
        rsi_data['dynamicNote'] = f'Dinamik esik: >{dyn_overbought}'
    rsi_data['dynamicOversold'] = dyn_oversold
    rsi_data['dynamicOverbought'] = dyn_overbought

    ind = {
        'rsi': rsi_data, 'rsiHistory': rsi_h,
        'macd': calc_macd(c), 'macdHistory': calc_macd_history(c),
        'bollinger': calc_bollinger(c, cp), 'bollingerHistory': calc_bollinger_history(c),
        'stochastic': calc_stochastic(c, h, l), 'stochasticHistory': calc_stochastic_history(c, h, l),
        'ema': calc_ema(c, cp), 'emaHistory': calc_ema_history(c),
        'atr': calc_atr(h, l, c),
        'adx': calc_adx(h, l, c),
        'obv': calc_obv(c, v),
        'williamsR': calc_williams_r(c, h, l),
        'cci': calc_cci(c, h, l),
        'mfi': calc_mfi(c, h, l, v),
        'vwap': calc_vwap(c, h, l, v),
        'ichimoku': calc_ichimoku(c, h, l),
        'psar': calc_psar(c, h, l),
        'roc': calc_roc(c),
        'aroon': calc_aroon(h, l),
        'trix': calc_trix(c),
        'dmi': calc_dmi(h, l, c),
        'candlestick': calc_candlestick_patterns(o, h, l, c),
        'dynamicThresholds': dyn_thresholds,
    }
    sigs = [x.get('signal', 'neutral') for x in ind.values() if isinstance(x, dict) and 'signal' in x]
    bc, sc = sigs.count('buy'), sigs.count('sell')
    t = len(sigs)
    ind['summary'] = {
        'overall': 'buy' if bc > sc and bc >= t * 0.4 else ('sell' if sc > bc and sc >= t * 0.4 else 'neutral'),
        'buySignals': bc, 'sellSignals': sc,
        'neutralSignals': t - bc - sc, 'totalIndicators': t,
    }
    return ind
