"""
Analiz Raporlama Endpoint'leri
Blueprint: analysis_reports_bp
Backtest, ısı haritası, günlük rapor, temettü, sinyal performansı ve kalibrasyon.
routes_analysis.py'dan ayrıştırıldı (700 satır kuralı).
"""
import threading
from datetime import datetime, timedelta
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
    BIST100_STOCKS, SECTOR_MAP,
    _cget_hist, _get_stocks, _get_indices,
    sf, safe_dict,
)

try:
    from indicators import calc_rsi_single
except ImportError as e:
    print(f"[HATA] routes_analysis_reports indicators import: {e}")
try:
    from signals import calc_signal_backtest
except ImportError as e:
    print(f"[HATA] routes_analysis_reports signals import: {e}")
try:
    from data_fetcher import _fetch_hist_df
except ImportError as e:
    print(f"[HATA] routes_analysis_reports data_fetcher import: {e}")

analysis_reports_bp = Blueprint('analysis_reports', __name__)


# =====================================================================
# BACKTEST
# =====================================================================

@analysis_reports_bp.route('/api/backtest', methods=['POST'])
def backtest():
    try:
        d               = request.json or {}
        sym             = d.get('symbol', '').upper()
        strategy        = d.get('strategy', 'ma_cross')
        params          = d.get('params', {})
        period          = d.get('period', '1y')
        commission      = float(d.get('commission', 0.001))
        initial_capital = float(d.get('initialCapital', 100000))

        if not sym:
            return jsonify({'error': 'Hisse kodu gerekli'}), 400

        hist = _fetch_hist_df(sym, period)
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{sym} icin yeterli veri yok'}), 400

        closes  = hist['Close'].values.astype(float)
        highs   = hist['High'].values.astype(float)
        lows    = hist['Low'].values.astype(float)
        dates   = [d2.strftime('%Y-%m-%d') for d2 in hist.index]
        n       = len(closes)
        signals = [0] * n

        if strategy == 'ma_cross':
            fast_p = int(params.get('fast', 20))
            slow_p = int(params.get('slow', 50))
            s = pd.Series(closes)
            fast_ma = s.rolling(fast_p).mean().values
            slow_ma = s.rolling(slow_p).mean().values
            for i in range(slow_p + 1, n):
                if fast_ma[i] > slow_ma[i] and fast_ma[i - 1] <= slow_ma[i - 1]:
                    signals[i] = 1
                elif fast_ma[i] < slow_ma[i] and fast_ma[i - 1] >= slow_ma[i - 1]:
                    signals[i] = -1

        elif strategy == 'breakout':
            lookback = int(params.get('lookback', 20))
            for i in range(lookback, n):
                high_n = max(highs[i - lookback:i])
                low_n  = min(lows[i - lookback:i])
                if closes[i] > high_n: signals[i] = 1
                elif closes[i] < low_n: signals[i] = -1

        elif strategy == 'mean_reversion':
            rsi_low  = float(params.get('rsi_low', 30))
            rsi_high = float(params.get('rsi_high', 70))
            for i in range(15, n):
                rsi = calc_rsi_single(closes[:i + 1])
                if rsi is not None:
                    if rsi < rsi_low: signals[i] = 1
                    elif rsi > rsi_high: signals[i] = -1

        elif strategy == 'system':
            # BIST Pro composite sistemi — her bar için calc_recommendation çalıştır
            from indicators import calc_all_indicators
            from signals_core import calc_recommendation
            tf_key     = params.get('timeframe', 'weekly')
            min_score_raw = float(params.get('min_score', 3.0))
            # Score formül [-14, 14] clamp'li — saçma eşik gelirse 3.0'a normalize et
            min_score  = 3.0 if (min_score_raw <= 0 or min_score_raw > 14) else min_score_raw
            step       = max(1, int(params.get('step', 1)))
            start_bar  = max(60, int(params.get('start_bar', 60)))
            in_pos = False
            for i in range(start_bar, n, step):
                try:
                    h_slice = hist.iloc[:i + 1]
                    if len(h_slice) < 30:
                        continue
                    cp = float(closes[i])
                    ind = calc_all_indicators(h_slice, cp)
                    rec = calc_recommendation(h_slice, ind, symbol=sym)
                    tf_rec = rec.get(tf_key, rec.get('weekly', {}))
                    action = tf_rec.get('action', 'NOTR')
                    score  = float(tf_rec.get('score', 0))
                    if not in_pos:
                        if action in ('AL', 'GÜÇLÜ AL') and score >= min_score:
                            signals[i] = 1
                            in_pos = True
                    else:
                        if action in ('SAT', 'GÜÇLÜ SAT', 'TUTUN/SAT') or score <= -min_score:
                            signals[i] = -1
                            in_pos = False
                except Exception:
                    continue

        cash         = initial_capital
        shares       = 0
        equity_curve = []
        trades       = []
        peak_equity  = initial_capital
        max_dd       = 0
        wins = losses = 0

        for i in range(n):
            price = closes[i]
            if signals[i] == 1 and shares == 0:
                shares = int(cash * (1 - commission) / price)
                if shares > 0:
                    cost = shares * price * (1 + commission)
                    cash -= cost
                    trades.append({'date': dates[i], 'action': 'AL', 'price': sf(price), 'shares': shares, 'pnl': 0})
            elif signals[i] == -1 and shares > 0:
                revenue = shares * price * (1 - commission)
                last_buy = next((t for t in reversed(trades) if t['action'] == 'AL'), None)
                pnl = revenue - (last_buy['shares'] * last_buy['price'] * (1 + commission)) if last_buy else 0
                cash += revenue
                trades.append({'date': dates[i], 'action': 'SAT', 'price': sf(price), 'shares': shares, 'pnl': sf(pnl)})
                if pnl > 0: wins += 1
                else: losses += 1
                shares = 0

            equity = cash + shares * price
            equity_curve.append({'date': dates[i], 'equity': sf(equity)})
            if equity > peak_equity: peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100
            if dd > max_dd: max_dd = dd

        final_equity  = cash + shares * closes[-1]
        total_return  = sf(((final_equity - initial_capital) / initial_capital) * 100)
        bh_return     = sf(((closes[-1] - closes[0]) / closes[0]) * 100)
        years         = n / 252
        cagr          = sf(((final_equity / initial_capital) ** (1 / years) - 1) * 100) if years > 0 else 0

        daily_returns = np.diff([e['equity'] for e in equity_curve]) / np.array([e['equity'] for e in equity_curve[:-1]])
        sharpe = sf(float(np.mean(daily_returns) / np.std(daily_returns) * (252 ** 0.5))) if len(daily_returns) > 0 and np.std(daily_returns) > 0 else 0

        total_trades  = wins + losses
        win_rate      = sf(wins / total_trades * 100) if total_trades > 0 else 0
        alpha         = sf(float(total_return) - float(bh_return))

        in_market_days = sum(1 for i in range(n) if (any(t['action'] == 'AL' and dates.index(t['date']) <= i for t in trades if t['date'] in dates)))
        exposure_pct   = sf(in_market_days / n * 100) if n > 0 else 0

        trade_pnls    = [float(t['pnl']) for t in trades if t['action'] == 'SAT' and t['pnl'] != 0]
        avg_trade     = sf(np.mean(trade_pnls)) if trade_pnls else 0
        avg_win       = sf(np.mean([p for p in trade_pnls if p > 0])) if [p for p in trade_pnls if p > 0] else 0
        avg_loss      = sf(np.mean([p for p in trade_pnls if p < 0])) if [p for p in trade_pnls if p < 0] else 0
        profit_factor = sf(abs(sum(p for p in trade_pnls if p > 0) / sum(p for p in trade_pnls if p < 0))) if any(p < 0 for p in trade_pnls) else 999

        return jsonify(safe_dict({
            'success': True,
            'results': {
                'totalReturn': total_return, 'cagr': cagr, 'sharpeRatio': sharpe,
                'maxDrawdown': sf(-max_dd), 'winRate': win_rate, 'totalTrades': total_trades,
                'buyAndHoldReturn': bh_return, 'alpha': alpha,
                'finalEquity': sf(final_equity), 'initialCapital': sf(initial_capital),
                'exposure': exposure_pct, 'avgTrade': avg_trade,
                'avgWin': avg_win, 'avgLoss': avg_loss, 'profitFactor': profit_factor,
                'commission': sf(commission * 100),
            },
            'equityCurve': equity_curve[::max(1, len(equity_curve) // 200)],
            'trades': trades[-50:],
            'warnings': [
                'Hayatta kalma yanliligi: Bu backtest sadece bugun BIST100 endeksinde bulunan hisseleri kapsar.',
                'Kayma (Slippage): Gercek islemlerde emir fiyati ile gerceklesen fiyat arasinda fark olabilir.',
                f"Komisyon: %{sf(commission * 100)} varsayimi kullanilmaktadir.",
            ],
        }))
    except Exception as e:
        import traceback
        print(f"[BACKTEST] Hata: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


# =====================================================================
# ISI HARİTASI
# =====================================================================

@analysis_reports_bp.route('/api/heatmap')
def heatmap():
    """Sektor bazli isi haritasi verisi"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'sectors': [], 'loading': True})

        stock_map = {s['code']: s for s in stocks}
        sectors   = []
        sector_display = {
            'bankacilik': 'Bankacilik', 'havacilik': 'Havacilik',
            'otomotiv': 'Otomotiv', 'enerji': 'Enerji',
            'holding': 'Holding', 'perakende': 'Perakende',
            'teknoloji': 'Teknoloji', 'telekom': 'Telekom',
            'demir_celik': 'Demir Celik', 'gida': 'Gida',
            'insaat': 'Insaat', 'gayrimenkul': 'Gayrimenkul',
        }

        for sector_name, symbols in SECTOR_MAP.items():
            sector_stocks = []
            total_change  = 0
            count = 0
            for sym in symbols:
                if sym in stock_map:
                    s = stock_map[sym]
                    sector_stocks.append({
                        'code': s['code'], 'name': s['name'],
                        'price': s['price'], 'changePct': s['changePct'],
                        'volume': s['volume'],
                    })
                    total_change += s['changePct']
                    count += 1
            avg_change = sf(total_change / count) if count > 0 else 0
            sectors.append({
                'name': sector_name,
                'displayName': sector_display.get(sector_name, sector_name),
                'avgChange': avg_change,
                'stockCount': count,
                'stocks': sorted(sector_stocks, key=lambda x: x['changePct'], reverse=True),
            })

        sectors.sort(key=lambda x: x['avgChange'], reverse=True)
        return jsonify(safe_dict({'success': True, 'sectors': sectors}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# GÜNLÜK RAPOR
# =====================================================================

@analysis_reports_bp.route('/api/report')
def daily_report():
    """Gunluk piyasa raporu"""
    try:
        stocks  = _get_stocks()
        indices = _get_indices()

        if not stocks:
            return jsonify({'success': True, 'loading': True, 'message': 'Veriler yukleniyor'})

        up        = [s for s in stocks if s.get('changePct', 0) > 0]
        down      = [s for s in stocks if s.get('changePct', 0) < 0]
        unchanged = [s for s in stocks if s.get('changePct', 0) == 0]

        sorted_up   = sorted(stocks, key=lambda x: x.get('changePct', 0), reverse=True)
        sorted_down = sorted(stocks, key=lambda x: x.get('changePct', 0))
        sorted_vol  = sorted(stocks, key=lambda x: x.get('volume', 0), reverse=True)

        gap_up   = sorted([s for s in stocks if s.get('gapPct', 0) > 1],  key=lambda x: x.get('gapPct', 0), reverse=True)
        gap_down = sorted([s for s in stocks if s.get('gapPct', 0) < -1], key=lambda x: x.get('gapPct', 0))

        all_changes = [s.get('changePct', 0) for s in stocks]
        avg_change  = sf(np.mean(all_changes)) if all_changes else 0

        sector_perf = []
        stock_map   = {s['code']: s for s in stocks}
        for sname, syms in SECTOR_MAP.items():
            changes = [stock_map[s]['changePct'] for s in syms if s in stock_map]
            if changes:
                sector_perf.append({
                    'name': sname, 'avgChange': sf(np.mean(changes)),
                    'bestStock': max(
                        [(s, stock_map[s]['changePct']) for s in syms if s in stock_map],
                        key=lambda x: x[1]
                    )[0],
                })
        sector_perf.sort(key=lambda x: x['avgChange'], reverse=True)

        report_lines = []
        bist100 = indices.get('XU100', {})
        if bist100:
            direction = 'yukselis' if bist100.get('changePct', 0) > 0 else 'dusus'
            report_lines.append(f"BIST 100 endeksi %{bist100.get('changePct', 0)} {direction} gosteriyor.")
        report_lines.append(f"Toplam {len(stocks)} hisseden {len(up)} yukselen, {len(down)} dusen, {len(unchanged)} degismez.")
        report_lines.append(f"Piyasa ortalama degisimi: %{avg_change}")
        if sorted_up:   report_lines.append(f"Gunun yildizi: {sorted_up[0]['code']} (%{sorted_up[0]['changePct']})")
        if sorted_down: report_lines.append(f"Gunun kaybi: {sorted_down[0]['code']} (%{sorted_down[0]['changePct']})")
        if sector_perf: report_lines.append(f"En iyi sektor: {sector_perf[0]['name']} (%{sector_perf[0]['avgChange']})")

        return jsonify(safe_dict({
            'success': True,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'summary': ' '.join(report_lines),
            'reportLines': report_lines,
            'marketBreadth': {
                'advancing': len(up), 'declining': len(down),
                'unchanged': len(unchanged), 'total': len(stocks), 'avgChange': avg_change,
            },
            'topGainers':    [{'code': s['code'], 'name': s['name'], 'changePct': s['changePct'], 'price': s['price']} for s in sorted_up[:5]],
            'topLosers':     [{'code': s['code'], 'name': s['name'], 'changePct': s['changePct'], 'price': s['price']} for s in sorted_down[:5]],
            'volumeLeaders': [{'code': s['code'], 'name': s['name'], 'volume': s['volume'], 'changePct': s['changePct']} for s in sorted_vol[:5]],
            'gapUp':         [{'code': s['code'], 'gapPct': s.get('gapPct', 0)} for s in gap_up[:5]],
            'gapDown':       [{'code': s['code'], 'gapPct': s.get('gapPct', 0)} for s in gap_down[:5]],
            'sectorPerformance': sector_perf,
            'indices': indices,
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# TEMETTÜ TAKVİMİ
# =====================================================================

@analysis_reports_bp.route('/api/dividends')
def dividend_calendar():
    """BIST hisselerinin temettu bilgileri (yfinance dividends)"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True, 'dividends': [], 'message': 'Veriler yukleniyor...'})

        dividends_list = []
        for stock in stocks:
            sym = stock['code']
            try:
                if _YF_OK and yf:
                    tkr  = yf.Ticker(sym + '.IS')
                    divs = tkr.dividends
                    if divs is not None and len(divs) > 0:
                        cp           = stock.get('price', 0)
                        recent_divs  = []
                        total_div_1y = 0
                        one_year_ago = datetime.now() - timedelta(days=365)
                        for dt, amt in divs.items():
                            div_date = dt.to_pydatetime().replace(tzinfo=None) if hasattr(dt, 'to_pydatetime') else dt
                            recent_divs.append({'date': div_date.strftime('%Y-%m-%d'), 'amount': sf(float(amt)), 'year': div_date.year})
                            if div_date >= one_year_ago:
                                total_div_1y += float(amt)

                        if recent_divs:
                            div_yield   = sf((total_div_1y / cp * 100) if cp > 0 else 0)
                            last_div    = recent_divs[-1]
                            yearly_divs = {}
                            for dv in recent_divs:
                                yr = dv['year']
                                yearly_divs[yr] = yearly_divs.get(yr, 0) + float(dv['amount'])
                            dividends_list.append({
                                'code': sym,
                                'name': BIST100_STOCKS.get(sym, sym),
                                'price': sf(cp),
                                'lastDividend': last_div,
                                'dividendYield': div_yield,
                                'totalDiv1Y': sf(total_div_1y),
                                'history': recent_divs[-10:],
                                'yearlyTotals': {str(k): sf(v) for k, v in sorted(yearly_divs.items())},
                                'changePct': stock.get('changePct', 0),
                            })
            except Exception:
                continue

        dividends_list.sort(key=lambda x: float(x.get('dividendYield', 0)), reverse=True)
        return jsonify(safe_dict({
            'success': True, 'count': len(dividends_list),
            'dividends': dividends_list, 'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# SİNYAL PERFORMANSI & KALİBRASYON
# =====================================================================

@analysis_reports_bp.route('/api/signals/performance')
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
                        'winRate5d':    overall.get('winRate5d', 0),
                        'winRate10d':   overall.get('winRate10d', 0),
                        'winRate20d':   overall.get('winRate20d', 0),
                        'avgReturn10d': overall.get('avgRet10d', 0),
                    })
            except Exception:
                continue

        results.sort(key=lambda x: float(x.get('winRate10d', 0)), reverse=True)
        return jsonify(safe_dict({'success': True, 'performance': results, 'timestamp': datetime.now().isoformat()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@analysis_reports_bp.route('/api/signals/calibration')
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
                'success': True, 'symbol': symbol,
                'rsiCalibration':    bt.get('rsiCalibration', {}),
                'bestRsiThreshold':  bt.get('bestRsiThreshold', '30/70'),
                'rankedIndicators':  bt.get('rankedIndicators', [])[:5],
                'benchmark':         bt.get('benchmark', {}),
                'totalSignals':      bt.get('totalSignals', 0),
                'timestamp':         datetime.now().isoformat(),
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
            rsi_summary[thresh] = {'avgProfitFactor10d': sf(avg_pf), 'stockCount': len(pfs)}
        best_rsi = max(rsi_summary, key=lambda k: float(rsi_summary[k]['avgProfitFactor10d'])) if rsi_summary else '30/70'

        ind_summary = []
        for name, pfs in agg_ind.items():
            avg_pf = float(np.mean(pfs)) if pfs else 0
            ind_summary.append({'reason': name, 'avgProfitFactor10d': sf(avg_pf), 'stockCount': len(pfs)})
        ind_summary.sort(key=lambda x: float(x['avgProfitFactor10d']), reverse=True)

        return jsonify(safe_dict({
            'success': True,
            'processedStocks':       processed,
            'rsiCalibrationSummary': rsi_summary,
            'bestRsiThreshold':      best_rsi,
            'topIndicators':         ind_summary[:5],
            'allIndicators':         ind_summary,
            'timestamp':             datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500
