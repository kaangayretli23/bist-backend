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

try:
    from analysis_helpers import (
        _signals_cache, _signals_cache_lock,
        _opps_cache, _opps_cache_lock,
        _strat_cache, _strat_cache_lock,
        COMPUTED_CACHE_TTL,
        _compute_signal_for_stock,
        _compute_opportunity_for_stock,
        _compute_strategy_for_stock,
    )
except ImportError as e:
    print(f"[HATA] routes_analysis analysis_helpers import: {e}")
try:
    from signals import calc_market_regime, calc_sector_relative_strength, check_signal_alerts
except ImportError:
    pass

try:
    from signal_tracker import log_signals_batch, get_signal_stats, get_recent_signal_log
    _TRACKER_OK = True
except ImportError as e:
    print(f"[UYARI] signal_tracker import: {e}")
    _TRACKER_OK = False


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

        # Sinyal performans takibi: AL/SAT sinyallerini kaydet
        if _TRACKER_OK:
            try:
                log_signals_batch(results, timeframe)
            except Exception:
                pass

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


@analysis_bp.route('/api/market/regime')
def market_regime():
    """Piyasa rejimi (bull/bear/sideways) — frontend için ayrı endpoint"""
    try:
        regime = calc_market_regime()
        return jsonify({'success': True, **regime})
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


@analysis_bp.route('/api/signal-stats')
def signal_stats():
    """Sinyal performans istatistikleri (kazanma oranı, ort. getiri, aylık trend)"""
    if not _TRACKER_OK:
        return jsonify({'error': 'signal_tracker modülü yüklenemedi'}), 500
    try:
        stats = get_signal_stats()
        return jsonify({'success': True, **stats})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/paper-stats')
def paper_stats():
    """Paper trading (auto_positions) istatistikleri"""
    try:
        from config import get_db
        db = get_db()

        # Genel özet
        summary = db.execute("""
            SELECT COUNT(*) as total,
                   SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins,
                   SUM(CASE WHEN pnl<=0 THEN 1 ELSE 0 END) as losses,
                   SUM(pnl) as net_pnl,
                   AVG(pnl_pct) as avg_pct,
                   MAX(pnl_pct) as best_pct,
                   MIN(pnl_pct) as worst_pct
            FROM auto_positions WHERE status='closed'
        """).fetchone()

        # Çıkış nedenine göre
        by_reason = db.execute("""
            SELECT
                CASE
                    WHEN close_reason LIKE 'Stop-Loss%' THEN 'Stop-Loss'
                    WHEN close_reason LIKE 'TP3%' THEN 'TP3'
                    WHEN close_reason LIKE 'TP2%' THEN 'TP2'
                    WHEN close_reason LIKE 'TP1%' THEN 'TP1'
                    WHEN close_reason LIKE 'Trailing%' THEN 'Trailing-Stop'
                    ELSE close_reason
                END as reason,
                COUNT(*) as cnt,
                SUM(pnl) as total_pnl,
                AVG(pnl_pct) as avg_pct,
                SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins
            FROM auto_positions WHERE status='closed'
            GROUP BY reason ORDER BY total_pnl DESC
        """).fetchall()

        # Sembole göre
        by_symbol = db.execute("""
            SELECT symbol,
                   COUNT(*) as cnt,
                   SUM(pnl) as total_pnl,
                   AVG(pnl_pct) as avg_pct,
                   SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins
            FROM auto_positions WHERE status='closed'
            GROUP BY symbol ORDER BY total_pnl DESC
        """).fetchall()

        # Son 10 işlem
        recent = db.execute("""
            SELECT symbol, entry_price, close_price, pnl, pnl_pct,
                   close_reason, opened_at, closed_at
            FROM auto_positions WHERE status='closed'
            ORDER BY closed_at DESC LIMIT 10
        """).fetchall()

        # Açık pozisyonlar
        open_pos = db.execute("""
            SELECT symbol, entry_price, stop_loss, take_profit1,
                   take_profit2, take_profit3, opened_at
            FROM auto_positions WHERE status='open'
            ORDER BY opened_at DESC
        """).fetchall()

        db.close()

        total = int(summary['total'] or 0)
        wins  = int(summary['wins']  or 0)

        def _fmt_reason(rows):
            result = []
            for r in rows:
                cnt = int(r['cnt'] or 0)
                w   = int(r['wins'] or 0)
                result.append({
                    'reason':   r['reason'],
                    'cnt':      cnt,
                    'wins':     w,
                    'winRate':  round(w / cnt * 100, 1) if cnt else 0,
                    'totalPnl': round(float(r['total_pnl'] or 0), 2),
                    'avgPct':   round(float(r['avg_pct']   or 0), 2),
                })
            return result

        return jsonify({
            'success': True,
            'summary': {
                'total':    total,
                'wins':     wins,
                'losses':   total - wins,
                'winRate':  round(wins / total * 100, 1) if total else 0,
                'netPnl':   round(float(summary['net_pnl']   or 0), 2),
                'avgPct':   round(float(summary['avg_pct']   or 0), 2),
                'bestPct':  round(float(summary['best_pct']  or 0), 2),
                'worstPct': round(float(summary['worst_pct'] or 0), 2),
            },
            'byReason': _fmt_reason(by_reason),
            'bySymbol': [{
                'symbol':   r['symbol'],
                'cnt':      int(r['cnt'] or 0),
                'wins':     int(r['wins'] or 0),
                'winRate':  round(int(r['wins'] or 0) / int(r['cnt'] or 1) * 100, 1),
                'totalPnl': round(float(r['total_pnl'] or 0), 2),
                'avgPct':   round(float(r['avg_pct']   or 0), 2),
            } for r in by_symbol],
            'recent': [{
                'symbol':      r['symbol'],
                'entryPrice':  round(float(r['entry_price']  or 0), 2),
                'closePrice':  round(float(r['close_price']  or 0), 2),
                'pnl':         round(float(r['pnl']          or 0), 2),
                'pnlPct':      round(float(r['pnl_pct']      or 0), 2),
                'closeReason': r['close_reason'] or '',
                'closedAt':    r['closed_at'] or '',
            } for r in recent],
            'openPositions': [{
                'symbol':     r['symbol'],
                'entryPrice': round(float(r['entry_price'] or 0), 2),
                'sl':         round(float(r['stop_loss']   or 0), 2),
                'tp1':        round(float(r['take_profit1'] or 0), 2),
                'tp2':        round(float(r['take_profit2'] or 0), 2),
                'tp3':        round(float(r['take_profit3'] or 0), 2),
                'openedAt':   r['opened_at'] or '',
            } for r in open_pos],
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/signal-log')
def signal_log_endpoint():
    """Son sinyal kayıtları + sonuçları"""
    if not _TRACKER_OK:
        return jsonify({'error': 'signal_tracker modülü yüklenemedi'}), 500
    try:
        limit     = int(request.args.get('limit', 100))
        timeframe = request.args.get('timeframe') or None
        action    = request.args.get('action') or None
        symbol    = request.args.get('symbol') or None
        logs = get_recent_signal_log(limit=limit, timeframe=timeframe, action=action, symbol=symbol)
        return jsonify({'success': True, 'count': len(logs), 'signals': logs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_bp.route('/api/briefing')
def briefing_endpoint():
    """Sabah Briefing: rejim + top AL/SAT + alerts + sentiment + auto-trader ozeti — tek endpoint"""
    try:
        uid = request.args.get('userId', '').strip()
        timeframe = request.args.get('timeframe', 'weekly')

        # 1) Market regime
        try:
            regime = calc_market_regime()
        except Exception:
            regime = {'regime': 'unknown', 'description': 'Hesaplanamadi'}

        # 2) Top signals: signals cache'inden (hesaplanmadiysa loading)
        top_buy, top_sell, total_scanned, sig_loading = [], [], 0, True
        try:
            with _signals_cache_lock:
                sc = _signals_cache
                if sc['data'] and (time.time() - sc['ts']) < COMPUTED_CACHE_TTL:
                    sigs = sc['data']['all_results']
                    total_scanned = sc['data']['totalScanned']
                    sig_loading = False
                    buys  = [r for r in sigs if float(r.get('score', 0)) > 0]
                    sells = [r for r in sigs if float(r.get('score', 0)) < 0]
                    buys.sort(key=lambda r: float(r.get('score', 0)), reverse=True)
                    sells.sort(key=lambda r: float(r.get('score', 0)))
                    def _slim(r):
                        return {
                            'symbol':     r.get('symbol', ''),
                            'name':       r.get('name', ''),
                            'price':      r.get('price', 0),
                            'score':      r.get('score', 0),
                            'confidence': r.get('confidence', 0),
                            'action':     r.get('action', ''),
                            'sector':     r.get('sector', ''),
                        }
                    top_buy  = [_slim(r) for r in buys[:3]]
                    top_sell = [_slim(r) for r in sells[:3]]
        except Exception:
            pass

        # 3) Alerts (cache'li, max 5 gösteriyoruz)
        alerts_top, alerts_total = [], 0
        try:
            all_alerts = check_signal_alerts(max_alerts=50)
            alerts_total = len(all_alerts)
            alerts_top = all_alerts[:5]
        except Exception:
            pass

        # 4) News sentiment — pazar geneli
        sentiment = None
        try:
            from news_sentiment import get_market_sentiment
            sentiment = get_market_sentiment()
        except Exception:
            sentiment = None

        # 5) Auto-trader ozet (userId verildiyse)
        auto_trader = None
        if uid:
            try:
                from auto_trader import _auto_get_config, _auto_get_open_positions, _auto_get_daily_trade_count
                cfg = _auto_get_config(uid)
                if cfg:
                    positions = _auto_get_open_positions(uid)
                    daily_count = _auto_get_daily_trade_count(uid)
                    open_pnl = 0.0
                    for pos in positions:
                        stock = _cget(_stock_cache, pos['symbol'])
                        cp = float(stock.get('price', 0)) if stock else 0
                        if cp > 0 and pos['entryPrice'] > 0:
                            open_pnl += (cp - pos['entryPrice']) * pos['quantity']
                    auto_trader = {
                        'enabled': bool(cfg.get('enabled')),
                        'positionsCount': len(positions),
                        'maxPositions': int(cfg.get('maxPositions', 5)),
                        'dailyTradeCount': daily_count,
                        'maxDailyTrades': int(cfg.get('maxDailyTrades', 3)),
                        'openPnL': round(open_pnl, 2),
                        'minScore': float(cfg.get('minScore', 0)),
                        'minConfidence': float(cfg.get('minConfidence', 0)),
                    }
            except Exception:
                auto_trader = None

        return jsonify(safe_dict({
            'success': True,
            'timeframe': timeframe,
            'regime': regime,
            'signalsLoading': sig_loading,
            'totalScanned': total_scanned,
            'topBuy': top_buy,
            'topSell': top_sell,
            'alertsTotal': alerts_total,
            'alertsTop': alerts_top,
            'marketSentiment': sentiment,
            'autoTrader': auto_trader,
            'timestamp': datetime.now().isoformat(),
            'meta': _api_meta(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


