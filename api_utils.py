"""
Shared API utility functions used across route modules.
"""
import time
from datetime import datetime
from config import _lock, _stock_cache, CACHE_TTL, CACHE_STALE_TTL, _status


def _cache_freshness():
    """Cache durumunu dondur: fresh/stale/loading"""
    now = time.time()
    with _lock:
        if not _stock_cache:
            return 'loading', None
        ages = [now - v['ts'] for v in _stock_cache.values()]
    avg_age = sum(ages) / len(ages) if ages else 9999
    if avg_age < CACHE_TTL:
        return 'fresh', datetime.fromtimestamp(now - avg_age).isoformat()
    elif avg_age < CACHE_STALE_TTL:
        return 'stale', datetime.fromtimestamp(now - avg_age).isoformat()
    else:
        return 'expired', None


def _api_meta(data_quality=None, extra=None):
    """Tum API response'larina eklenecek meta bilgisi"""
    freshness, last_updated = _cache_freshness()
    meta = {
        'lastUpdated': last_updated or _status.get('lastRun'),
        'snapshotTimestamp': datetime.now().isoformat(),
        'dataQuality': data_quality or freshness,
        'loaderPhase': _status.get('phase', 'idle'),
    }
    if extra:
        meta.update(extra)
    return meta
