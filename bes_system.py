"""
BIST Pro - BES Fund Analysis Module (Facade)
Alt modullerden re-export eder.
"""
# Data, cache, analysis fonksiyonlari
from bes_data import (
    _bes_cache, _bes_cache_lock, BES_CACHE_TTL, _tefas_semaphore,
    _bes_bg_loading, _bes_bg_error,
    TEFAS_API_URL, TEFAS_ALLOC_URL, TEFAS_COMPARE_URL, TEFAS_HEADERS,
    BES_FUND_GROUPS,
    _bes_cache_get, _bes_cache_set,
    _bes_bg_analyze_top, _classify_fund,
    _fetch_tefas_funds, _fetch_tefas_compare,
    _fetch_tefas_allocation, _fetch_tefas_history_chunked,
    _get_tefas_field, _parse_fund_row, _parse_compare_row, _parse_tefas_date,
    _analyze_fund_performance, _bes_optimize, _fund_reasoning, _simulate_bes,
)

# Flask API routes (import sırasında register olur)
import bes_routes
