"""
BIST Pro - Signal & Recommendation Module (Facade)
Tum signal fonksiyonlarini alt modullerden re-export eder.
"""
import numpy as np
import time, traceback
from config import sf, si, _lock, _stock_cache, _index_cache, _cget, _get_stocks, BIST100_STOCKS, SECTOR_MAP
from indicators import *
from indicators import _market_regime_cache, _resample_to_tf

# Core signals
from signals_core import (
    calc_recommendation, calc_fundamentals, calc_52w,
    fetch_fundamental_data, check_signal_alerts, calc_ml_confidence,
)

# Backtest & market analysis
from signals_backtest import (
    calc_signal_backtest, calc_market_regime, calc_sector_relative_strength,
)
