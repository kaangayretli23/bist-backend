"""
Per-stock advanced analysis routes: MTF, divergence, SMC, patterns, fibonacci, pivots, etc.
"""
from datetime import datetime
from flask import Blueprint, jsonify, request
from config import (
    BIST100_STOCKS, _cget_hist, _get_stocks, safe_dict, sf,
)
try:
    from data_fetcher import _fetch_hist_df
except ImportError as e:
    print(f"[HATA] routes_stock data_fetcher import: {e}")
try:
    from indicators import (
        calc_all_indicators, calc_mtf_signal, calc_divergence,
        calc_volume_profile, calc_smc, calc_chart_patterns,
        calc_fibonacci_adv, calc_pivot_points_adv, calc_advanced_indicators,
        calc_support_resistance, calc_candlestick_patterns,
    )
except ImportError as e:
    print(f"[HATA] routes_stock indicators import: {e}")
try:
    from signals import (
        calc_signal_backtest, fetch_fundamental_data,
    )
except ImportError as e:
    print(f"[HATA] routes_stock signals import: {e}")

stock_bp = Blueprint('stock', __name__)


@stock_bp.route('/api/stock/<symbol>/backtest-signals')
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


@stock_bp.route('/api/stock/<symbol>/mtf')
def stock_mtf(symbol):
    """
    Gercek coklu zaman dilimi (MTF) analizi.
    Gunluk OHLCV verisini haftalik ve aylik bara resample ederek
    her zaman diliminde RSI/MACD/EMA/Bollinger indikatörlerini hesaplar.
    """
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 30 bar gerekli)'}), 400
        mtf = calc_mtf_signal(hist)
        return jsonify(safe_dict({
            'success': True, 'symbol': symbol,
            'mtf': mtf, 'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stock_bp.route('/api/stock/<symbol>/divergence')
def stock_divergence(symbol):
    """RSI ve MACD uyumsuzluk (divergence) analizi."""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 50:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 50 bar)'}), 400
        result = calc_divergence(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result, 'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stock_bp.route('/api/stock/<symbol>/volume-profile')
def stock_volume_profile(symbol):
    """Hacim Profili ve VWAP analizi."""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 20:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 20 bar)'}), 400
        result = calc_volume_profile(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result, 'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stock_bp.route('/api/stock/<symbol>/smc')
def stock_smc(symbol):
    """Smart Money Concepts (SMC) analizi: FVG, Order Block, BOS, CHoCH."""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 30 bar)'}), 400
        result = calc_smc(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result, 'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stock_bp.route('/api/stock/<symbol>/patterns')
def stock_chart_patterns(symbol):
    """Grafik formasyon analizi: Çift Tepe/Dip, OBO, Üçgen, Bayrak."""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 30 bar)'}), 400
        result = calc_chart_patterns(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result, 'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stock_bp.route('/api/stock/<symbol>/fibonacci')
def stock_fibonacci(symbol):
    """Fibonacci retracement ve extension seviyeleri. Golden Pocket (0.618-0.65) dahil."""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 20:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 20 bar)'}), 400
        result = calc_fibonacci_adv(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result, 'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stock_bp.route('/api/stock/<symbol>/pivots')
def stock_pivot_points(symbol):
    """Klasik, Camarilla ve Woodie Pivot Noktalari."""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 3:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 3 bar)'}), 400
        result = calc_pivot_points_adv(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result, 'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stock_bp.route('/api/stock/<symbol>/advanced-indicators')
def stock_advanced_indicators(symbol):
    """Ileri teknik indikatörler: Ichimoku, Stochastic (14,3,3), Williams %R (14)."""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 14:
            return jsonify({'error': f'{symbol} icin yeterli veri yok (min 14 bar)'}), 400
        result = calc_advanced_indicators(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result, 'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@stock_bp.route('/api/stock/<symbol>/full-analysis')
def stock_full_analysis(symbol):
    """
    Kapsamli tam analiz: Tum fazlari tek endpoint'te birlestirir.
    MTF + Divergence + Volume Profile + SMC + Patterns +
    Fibonacci + Pivots + Advanced Indicators + Temel Gostergeler.
    """
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
            try:
                return fn(*args, **kwargs)
            except Exception as ex:
                return {'error': str(ex)}

        ind      = _safe(calc_all_indicators, hist, cp)
        mtf      = _safe(calc_mtf_signal,           hist)
        div      = _safe(calc_divergence,            hist)
        vp       = _safe(calc_volume_profile,        hist)
        smc      = _safe(calc_smc,                   hist)
        patterns = _safe(calc_chart_patterns,        hist)
        fib      = _safe(calc_fibonacci_adv,         hist)
        pivots   = _safe(calc_pivot_points_adv,      hist)
        adv      = _safe(calc_advanced_indicators,   hist)
        sr       = _safe(calc_support_resistance,    hist)
        candles  = _safe(calc_candlestick_patterns,  hist)
        backtest = _safe(calc_signal_backtest,       hist)

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
        consensus  = ('buy' if buy_votes > sell_votes else ('sell' if sell_votes > buy_votes else 'neutral'))
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


@stock_bp.route('/api/stock/<symbol>/fundamentals')
def stock_fundamentals_endpoint(symbol):
    """Hisse temel analiz verileri (F/K, PD/DD)"""
    try:
        symbol = symbol.upper()
        result = fetch_fundamental_data(symbol)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, 'fundamentals': result}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500
