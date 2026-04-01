"""Gelismis hisse analizi, sinyal performans ve kalibrasyon route'lari"""
import numpy as np
from datetime import datetime
from flask import request, jsonify

from config import app, safe_dict, sf, BIST100_STOCKS
from data_fetcher import _get_stocks, _cget_hist, _fetch_hist_df
from indicators import (
    calc_all_indicators, calc_mtf_signal, calc_divergence,
    calc_volume_profile, calc_smc, calc_chart_patterns,
    calc_fibonacci_adv, calc_pivot_points_adv, calc_advanced_indicators,
    calc_support_resistance, calc_candlestick_patterns,
)
from signals import (
    calc_recommendation, calc_fundamentals, calc_market_regime,
    calc_signal_backtest, calc_sector_relative_strength,
    fetch_fundamental_data, check_signal_alerts, calc_52w,
)


# =====================================================================
# ADVANCED STOCK ANALYSIS ENDPOINTS
# =====================================================================
@app.route('/api/stock/<symbol>/backtest-signals')
def stock_signal_backtest(symbol):
    """Hisse bazli sinyal backtest sonuclari"""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 50:
            return jsonify({'error': f'{symbol} icin yeterli veri yok'}), 400
        result = calc_signal_backtest(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/mtf')
def stock_mtf(symbol):
    """Coklu zaman dilimi (MTF) analizi"""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 30 bar gerekli)'}), 400
        mtf = calc_mtf_signal(hist)
        return jsonify(safe_dict({
            'success': True, 'symbol': symbol, 'mtf': mtf,
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/divergence')
def stock_divergence(symbol):
    """RSI ve MACD uyumsuzluk (divergence) analizi"""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 50:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 50 bar)'}), 400
        result = calc_divergence(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/volume-profile')
def stock_volume_profile(symbol):
    """Hacim Profili ve VWAP analizi"""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 20:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 20 bar)'}), 400
        result = calc_volume_profile(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/smc')
def stock_smc(symbol):
    """Smart Money Concepts (SMC) analizi"""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 30 bar)'}), 400
        result = calc_smc(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/patterns')
def stock_chart_patterns(symbol):
    """Grafik formasyon analizi"""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 30 bar)'}), 400
        result = calc_chart_patterns(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/fibonacci')
def stock_fibonacci(symbol):
    """Fibonacci retracement ve extension seviyeleri"""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 20:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 20 bar)'}), 400
        result = calc_fibonacci_adv(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/pivots')
def stock_pivot_points(symbol):
    """Klasik, Camarilla ve Woodie Pivot Noktalari"""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 3:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 3 bar)'}), 400
        result = calc_pivot_points_adv(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/advanced-indicators')
def stock_advanced_indicators(symbol):
    """Ileri teknik indikatörler: Ichimoku, Stochastic, Williams %R"""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 14:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 14 bar)'}), 400
        result = calc_advanced_indicators(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result,
                                  'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/full-analysis')
def stock_full_analysis(symbol):
    """Kapsamli tam analiz: Tum fazlari tek endpoint'te birlestirir"""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 30 bar)'}), 400

        stocks = _get_stocks()
        stock_info = next((s for s in stocks if s.get('code') == symbol), {})
        cp = float(hist['Close'].iloc[-1])

        def _safe(fn, *args, **kwargs):
            try: return fn(*args, **kwargs)
            except Exception as ex: return {'error': str(ex)}

        ind          = _safe(calc_all_indicators, hist, cp)
        mtf          = _safe(calc_mtf_signal,       hist)
        div          = _safe(calc_divergence,        hist)
        vp           = _safe(calc_volume_profile,    hist)
        smc          = _safe(calc_smc,               hist)
        patterns     = _safe(calc_chart_patterns,    hist)
        fib          = _safe(calc_fibonacci_adv,         hist)
        pivots       = _safe(calc_pivot_points_adv,      hist)
        adv          = _safe(calc_advanced_indicators, hist)
        sr           = _safe(calc_support_resistance, hist)
        candles      = _safe(calc_candlestick_patterns, hist)
        backtest     = _safe(calc_signal_backtest,   hist)

        all_signals = [
            mtf.get('mtfDirection', 'neutral'),
            div.get('summary', {}).get('signal', 'neutral'),
            vp.get('vwapSignal', 'neutral'),
            smc.get('signal', 'neutral'),
            patterns.get('signal', 'neutral'),
            adv.get('summary', {}).get('signal', 'neutral'),
            pivots.get('bias', 'neutral') if 'bias' in pivots else 'neutral',
        ]
        buy_votes  = all_signals.count('buy')
        sell_votes = all_signals.count('sell')
        consensus  = ('buy'  if buy_votes > sell_votes
                      else ('sell' if sell_votes > buy_votes else 'neutral'))
        confidence = round(max(buy_votes, sell_votes) / len(all_signals) * 100)

        return jsonify(safe_dict({
            'success':     True,
            'symbol':      symbol,
            'name':        BIST100_STOCKS.get(symbol, symbol),
            'price':       sf(cp),
            'changePct':   stock_info.get('changePct', 0),
            'consensus':   consensus,
            'buyVotes':    buy_votes,
            'sellVotes':   sell_votes,
            'neutralVotes': len(all_signals) - buy_votes - sell_votes,
            'confidence':  confidence,
            'indicators':         ind,
            'mtf':                mtf,
            'divergence':         div,
            'volumeProfile':      vp,
            'smc':                smc,
            'chartPatterns':      patterns,
            'fibonacci':          fib,
            'pivots':             pivots,
            'advancedIndicators': adv,
            'supportResistance':  sr,
            'candlestickPatterns': candles,
            'backtest':           backtest,
            'timestamp':          datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/fundamentals')
def stock_fundamentals_endpoint(symbol):
    """Hisse temel analiz verileri (F/K, PD/DD)"""
    try:
        symbol = symbol.upper()
        result = fetch_fundamental_data(symbol)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, 'fundamentals': result}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# MARKET REGIME, SECTOR ANALYSIS, SIGNAL ALERTS
# =====================================================================
@app.route('/api/market/regime')
def market_regime_endpoint():
    """Piyasa rejimi (boga/ayi/yatay)"""
    try:
        regime = calc_market_regime()
        return jsonify(safe_dict({'success': True, **regime}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sectors/analysis')
def sectors_analysis():
    """Sektor bazli goreceli guc analizi"""
    try:
        result = calc_sector_relative_strength()
        return jsonify(safe_dict({'success': True, **result}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/signals')
def signal_alerts_endpoint():
    """Sinyal bazli otomatik uyarilar (mum formasyonu, altin kesisim vb.)"""
    try:
        alerts = check_signal_alerts()
        return jsonify(safe_dict({
            'success': True,
            'alerts': alerts[:30],
            'totalAlerts': len(alerts),
            'marketRegime': calc_market_regime(),
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# SINYAL PERFORMANS & KALIBRASYON
# =====================================================================
@app.route('/api/signals/performance')
def signals_performance():
    """Tum hisselerin sinyal performans ozeti"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True})

        results = []
        for stock in stocks[:50]:
            sym = stock['code']
            try:
                hist = _cget_hist(f"{sym}_1y")
                if hist is None or len(hist) < 60:
                    continue
                bt = calc_signal_backtest(hist)
                if bt.get('totalSignals', 0) > 3:
                    overall = bt.get('overall', {})
                    results.append({
                        'code': sym,
                        'name': BIST100_STOCKS.get(sym, sym),
                        'totalSignals': bt['totalSignals'],
                        'winRate5d': overall.get('winRate5d', 0),
                        'winRate10d': overall.get('winRate10d', 0),
                        'winRate20d': overall.get('winRate20d', 0),
                        'avgReturn10d': overall.get('avgRet10d', 0),
                    })
            except Exception:
                continue

        results.sort(key=lambda x: float(x.get('winRate10d', 0)), reverse=True)
        return jsonify(safe_dict({
            'success': True,
            'performance': results,
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals/calibration')
def signals_calibration():
    """BIST indikatör kalibrasyonu"""
    try:
        symbol = request.args.get('symbol', '').upper().strip()

        if symbol:
            hist = _cget_hist(f"{symbol}_1y")
            if hist is None:
                hist = _fetch_hist_df(symbol, '1y')
            if hist is None or len(hist) < 60:
                return jsonify({'error': f'{symbol} icin yeterli veri yok'}), 400
            bt = calc_signal_backtest(hist)
            return jsonify(safe_dict({
                'success': True,
                'symbol': symbol,
                'rsiCalibration': bt.get('rsiCalibration', {}),
                'bestRsiThreshold': bt.get('bestRsiThreshold', '30/70'),
                'rankedIndicators': bt.get('rankedIndicators', [])[:5],
                'benchmark': bt.get('benchmark', {}),
                'totalSignals': bt.get('totalSignals', 0),
                'timestamp': datetime.now().isoformat(),
            }))

        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True})

        agg_rsi = {}
        agg_ind = {}
        processed = 0

        for stock in stocks[:20]:
            sym = stock['code']
            try:
                hist = _cget_hist(f"{sym}_1y")
                if hist is None or len(hist) < 60:
                    continue
                bt = calc_signal_backtest(hist)
                if bt.get('totalSignals', 0) < 5:
                    continue

                for thresh, stats in bt.get('rsiCalibration', {}).items():
                    agg_rsi.setdefault(thresh, []).append(float(stats.get('profitFactor10d', 0)))

                for ind in bt.get('rankedIndicators', []):
                    name = ind.get('reason', '')
                    if name:
                        agg_ind.setdefault(name, []).append(float(ind.get('profitFactor10d', 0)))

                processed += 1
            except Exception:
                continue

        if processed == 0:
            return jsonify({'success': True, 'loading': True, 'message': 'Veri hazırlanıyor'})

        rsi_summary = {}
        for thresh, pfs in agg_rsi.items():
            avg_pf = float(np.mean(pfs)) if pfs else 0
            rsi_summary[thresh] = {
                'avgProfitFactor10d': sf(avg_pf),
                'stockCount': len(pfs),
            }
        best_rsi_bulk = max(rsi_summary, key=lambda k: float(rsi_summary[k]['avgProfitFactor10d'])) if rsi_summary else '30/70'

        ind_summary = []
        for name, pfs in agg_ind.items():
            avg_pf = float(np.mean(pfs)) if pfs else 0
            ind_summary.append({
                'reason': name,
                'avgProfitFactor10d': sf(avg_pf),
                'stockCount': len(pfs),
            })
        ind_summary.sort(key=lambda x: float(x['avgProfitFactor10d']), reverse=True)

        return jsonify(safe_dict({
            'success': True,
            'processedStocks': processed,
            'rsiCalibrationSummary': rsi_summary,
            'bestRsiThreshold': best_rsi_bulk,
            'topIndicators': ind_summary[:5],
            'allIndicators': ind_summary,
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500
