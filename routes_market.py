"""
Market data routes: dashboard, indices, stock detail, commodity, compare, screener, sectors, search.
"""
import traceback
from datetime import datetime
from flask import Blueprint, jsonify, request
import requests as req_lib

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pass

from config import (
    YF_OK, BIST100_STOCKS, BIST30, SECTOR_MAP,
    _stock_cache, _index_cache, _hist_cache,
    _cget, _cget_hist, _cset,
    _get_stocks, _get_indices, _status,
    sf, si, safe_dict,
)
from api_utils import _api_meta
try:
    from indicators import (
        calc_rsi, calc_macd, calc_ema,
        calc_all_indicators, calc_support_resistance, calc_fibonacci,
        calc_pivot_points, prepare_chart_data, calc_candlestick_patterns,
    )
except ImportError as e:
    print(f"[HATA] routes_market indicators import: {e}")
try:
    from signals import (
        calc_recommendation, calc_fundamentals, calc_52w,
        calc_signal_backtest, calc_market_regime, calc_ml_confidence,
    )
except ImportError as e:
    print(f"[HATA] routes_market signals import: {e}")
try:
    from trade_plans import calc_trade_plan
except ImportError as e:
    print(f"[HATA] routes_market trade_plans import: {e}")
try:
    from data_fetcher import _fetch_hist_df, IS_YATIRIM_HEADERS
except ImportError as e:
    print(f"[HATA] routes_market data_fetcher import: {e}")

market_bp = Blueprint('market', __name__)


@market_bp.route('/api/dashboard')
def dashboard():
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify(safe_dict({
                'success': True, 'loading': True, 'stockCount': 0,
                'message': f"Veriler yukleniyor ({_status['loaded']}/{_status['total']})...",
                'movers': {'topGainers': [], 'topLosers': [], 'volumeLeaders': [], 'gapStocks': []},
                'marketBreadth': {'advancing': 0, 'declining': 0, 'unchanged': 0, 'advDecRatio': 0},
                'allStocks': [], 'meta': _api_meta('loading'),
            }))
        sbc = sorted(stocks, key=lambda x: x.get('changePct', 0), reverse=True)
        adv = sum(1 for s in stocks if s.get('changePct', 0) > 0)
        dec = sum(1 for s in stocks if s.get('changePct', 0) < 0)
        return jsonify(safe_dict({
            'success': True, 'loading': False, 'stockCount': len(stocks),
            'timestamp': datetime.now().isoformat(),
            'movers': {
                'topGainers': sbc[:5], 'topLosers': sbc[-5:][::-1],
                'volumeLeaders': sorted(stocks, key=lambda x: x.get('volume', 0), reverse=True)[:5],
                'gapStocks': sorted(stocks, key=lambda x: abs(x.get('gapPct', 0)), reverse=True)[:5],
            },
            'marketBreadth': {
                'advancing': adv, 'declining': dec,
                'unchanged': len(stocks) - adv - dec,
                'advDecRatio': sf(adv / dec if dec > 0 else adv),
            },
            'allStocks': sbc, 'meta': _api_meta(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@market_bp.route('/api/indices')
def indices():
    try:
        data = _get_indices()
        if not data:
            return jsonify(safe_dict({'success': True, 'loading': True, 'indices': {}, 'message': 'Endeksler yukleniyor...'}))

        usdtry_data = data.get('USDTRY')
        gold_data = data.get('GOLD')
        silver_data = data.get('SILVER')
        if usdtry_data and gold_data:
            usd_rate = usdtry_data.get('value', 0)
            if usd_rate > 0:
                gold_usd = gold_data.get('value', 0)
                gold_tl = sf(gold_usd * usd_rate / 31.1035, 2)
                data['GOLDTL'] = {
                    'name': 'Altin/TL (gram)', 'value': gold_tl, 'change': 0,
                    'changePct': gold_data.get('changePct', 0), 'volume': 0,
                }
        if usdtry_data and silver_data:
            usd_rate = usdtry_data.get('value', 0)
            if usd_rate > 0:
                silver_usd = silver_data.get('value', 0)
                silver_tl = sf(silver_usd * usd_rate / 31.1035, 2)
                data['SILVERTL'] = {
                    'name': 'Gumus/TL (gram)', 'value': silver_tl, 'change': 0,
                    'changePct': silver_data.get('changePct', 0), 'volume': 0,
                }
        return jsonify(safe_dict({'success': True, 'indices': data}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@market_bp.route('/api/bist100')
def bist100():
    try:
        sector = request.args.get('sector')
        sort_by = request.args.get('sort', 'code')
        order = request.args.get('order', 'asc')
        stocks = _get_stocks(SECTOR_MAP[sector]) if sector and sector in SECTOR_MAP else _get_stocks()
        if not stocks:
            return jsonify(safe_dict({
                'success': True, 'stocks': [], 'count': 0,
                'sectors': list(SECTOR_MAP.keys()), 'loading': True,
                'message': f"Hisse verileri yukleniyor ({_status['loaded']}/{_status['total']})...",
            }))
        rev = (order == 'desc')
        km = {'change': 'changePct', 'volume': 'volume', 'price': 'price'}
        sk = km.get(sort_by, 'code')
        stocks.sort(key=lambda x: x.get(sk, 0) if sk != 'code' else x.get('code', ''), reverse=rev)
        return jsonify(safe_dict({'success': True, 'stocks': stocks, 'count': len(stocks), 'sectors': list(SECTOR_MAP.keys()), 'meta': _api_meta()}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@market_bp.route('/api/bist30')
def bist30():
    try:
        stocks = _get_stocks(BIST30)
        if not stocks:
            return jsonify(safe_dict({'success': True, 'stocks': [], 'count': 0, 'loading': True}))
        stocks.sort(key=lambda x: x.get('code', ''))
        return jsonify(safe_dict({'success': True, 'stocks': stocks, 'count': len(stocks)}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@market_bp.route('/api/stock/<symbol>')
def stock_detail(symbol):
    """Tek hisse detay - SENKRON, 3 katmanli fallback (IsYatirim -> Yahoo -> yfinance)"""
    try:
        symbol = symbol.upper()
        period = request.args.get('period', '1y')

        hist = _cget_hist(f"{symbol}_{period}")

        if hist is None:
            print(f"[DETAIL] {symbol} {period} cekiliyor...")
            hist = _fetch_hist_df(symbol, period)
            if hist is not None and len(hist) >= 2:
                _cset(_hist_cache, f"{symbol}_{period}", hist)
                print(f"[DETAIL] {symbol} OK: {len(hist)} bar")
            else:
                hist = None

        if hist is None:
            quick = _cget(_stock_cache, symbol)
            if quick:
                return jsonify(safe_dict({
                    'success': True, 'code': symbol, 'name': quick['name'],
                    'price': quick['price'], 'change': quick['change'],
                    'changePercent': quick['changePct'],
                    'volume': quick['volume'], 'dayOpen': quick['open'],
                    'dayHigh': quick['high'], 'dayLow': quick['low'],
                    'prevClose': quick['prevClose'], 'currency': 'TRY',
                    'period': period, 'dataPoints': 0,
                    'indicators': {}, 'chartData': {'candlestick': [], 'dates': [], 'prices': [], 'volumes': [], 'dataPoints': 0},
                    'fibonacci': {'levels': {}}, 'supportResistance': {'supports': [], 'resistances': []},
                    'pivotPoints': {'classic': {}, 'camarilla': {}, 'woodie': {}, 'current': 0},
                    'recommendation': {
                        'weekly': {'action': 'neutral', 'confidence': 0, 'reasons': []},
                        'monthly': {'action': 'neutral', 'confidence': 0, 'reasons': []},
                        'yearly': {'action': 'neutral', 'confidence': 0, 'reasons': []},
                    },
                }))
            return jsonify({'error': f'{symbol} verisi bulunamadi'}), 404

        if hist[['Open', 'High', 'Low']].isna().any().any():
            hist['Open'] = hist['Open'].fillna(hist['Close'])
            hist['High'] = hist['High'].fillna(hist['Close'])
            hist['Low'] = hist['Low'].fillna(hist['Close'])
            hist['Volume'] = hist['Volume'].fillna(0)

        cp = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else cp
        # Live override: piyasa acikken IS Yatirim hist endpoint'i bugunun barini
        # henuz vermez (sadece kapanis sonrasi). _stock_cache live quote'tan price/change
        # gun ici tazedir → fiyat/degisim icin onu kullan, OHLCV/indicator hesaplari hist'ten.
        live_price = cp
        live_change = cp - prev
        live_pct = (cp - prev) / prev * 100 if prev else 0.0
        live_volume = si(hist['Volume'].iloc[-1])
        live_high = sf(hist['High'].iloc[-1])
        live_low = sf(hist['Low'].iloc[-1])
        live_open = sf(hist['Open'].iloc[-1])
        try:
            live_q = _cget(_stock_cache, symbol)
            if live_q and live_q.get('price'):
                live_price = float(live_q.get('price') or cp)
                if 'change' in live_q: live_change = float(live_q.get('change') or 0)
                if 'changePct' in live_q: live_pct = float(live_q.get('changePct') or 0)
                if 'prevClose' in live_q and live_q.get('prevClose'): prev = float(live_q['prevClose'])
                if live_q.get('volume'): live_volume = si(live_q['volume'])
                if live_q.get('high'): live_high = sf(live_q['high'])
                if live_q.get('low'): live_low = sf(live_q['low'])
                if live_q.get('open'): live_open = sf(live_q['open'])
        except Exception:
            pass
        cp = live_price
        w52 = calc_52w(hist)
        ind = calc_all_indicators(hist, cp)
        rec = calc_recommendation(hist, ind, symbol=symbol)

        ml_conf = {}
        for tf_label in ['weekly', 'monthly', 'yearly']:
            tf_rec = rec.get(tf_label, {})
            tf_action = tf_rec.get('action', 'NOTR')
            if tf_action in ('AL', 'TUTUN/AL'):
                ml_conf[tf_label] = calc_ml_confidence(hist, ind, float(tf_rec.get('score', 0)), 'buy', symbol=symbol)
            elif tf_action in ('SAT', 'TUTUN/SAT'):
                ml_conf[tf_label] = calc_ml_confidence(hist, ind, float(tf_rec.get('score', 0)), 'sell', symbol=symbol)
            else:
                ml_conf[tf_label] = {'confidence': 50, 'grade': 'C', 'factors': []}

        return jsonify(safe_dict({
            'success': True, 'code': symbol,
            'name': BIST100_STOCKS.get(symbol, symbol),
            'price': sf(cp), 'change': sf(live_change),
            'changePercent': sf(live_pct),
            'volume': live_volume,
            'dayHigh': live_high,
            'dayLow': live_low,
            'dayOpen': live_open,
            'prevClose': sf(prev), 'currency': 'TRY',
            'period': period, 'dataPoints': len(hist),
            'week52': w52,
            # NOT: marketValue = price * volume → bu islem hacmi (TL), market cap DEGIL.
            # Market cap = price * shares_outstanding (bu veri Is Yatirim API'sinde yok).
            # Frontend mevcut isim ile kullaniyor → degistirme riski var, sadece niyet aciklamasi.
            'marketValue': sf(cp * si(hist['Volume'].iloc[-1])),
            'indicators': ind,
            'chartData': prepare_chart_data(hist),
            'fibonacci': calc_fibonacci(hist),
            'supportResistance': calc_support_resistance(hist),
            'pivotPoints': calc_pivot_points(hist),
            'recommendation': rec,
            'mlConfidence': ml_conf,
            'signalBacktest': calc_signal_backtest(hist),
            'tradePlan': calc_trade_plan(hist, ind, symbol=symbol),
            'marketRegime': calc_market_regime(),
            'fundamentals': calc_fundamentals(hist, symbol),
            'meta': _api_meta(),
        }))
    except Exception as e:
        print(f"STOCK {symbol}: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@market_bp.route('/api/commodity/<symbol>')
def commodity_detail(symbol):
    """Emtia detay - Altin, Gumus, GOLDTL, SILVERTL icin hisse gibi tam analiz"""
    try:
        symbol = symbol.upper()
        period = request.args.get('period', '1y')

        COMMODITY_MAP = {
            'GOLD': ('GC=F', 'Altin (USD/ons)', 'USD'),
            'SILVER': ('SI=F', 'Gumus (USD/ons)', 'USD'),
            'GOLDTL': ('GC=F', 'Altin/TL (gram)', 'TRY'),
            'SILVERTL': ('SI=F', 'Gumus/TL (gram)', 'TRY'),
            'USDTRY': ('USDTRY=X', 'Dolar/TL', 'TRY'),
            'EURTRY': ('EURTRY=X', 'Euro/TL', 'TRY'),
        }

        if symbol not in COMMODITY_MAP:
            return jsonify({'error': f'{symbol} desteklenmiyor. Desteklenen: {list(COMMODITY_MAP.keys())}'}), 400

        yahoo_sym, name, currency = COMMODITY_MAP[symbol]
        period_days = {'1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825}.get(period, 365)

        cache_key = f"COMMODITY_{symbol}_{period}"
        hist = _cget_hist(cache_key)

        if hist is None:
            from data_fetcher import _fetch_yahoo_http_df
            print(f"[COMMODITY] {symbol} ({yahoo_sym}) {period} cekiliyor...")
            hist = _fetch_yahoo_http_df(yahoo_sym, period_days)
            if (hist is None or len(hist) < 10) and YF_OK and yf:
                try:
                    h = yf.Ticker(yahoo_sym).history(period=period, timeout=15)
                    if h is not None and not h.empty and len(h) >= 10:
                        hist = h
                        print(f"  [YF-COMMODITY] {symbol} OK: {len(hist)} bar")
                except Exception as e:
                    print(f"  [YF-COMMODITY] {symbol}: {e}")

            if hist is not None and len(hist) >= 2:
                if symbol in ('GOLDTL', 'SILVERTL'):
                    usd_hist = _fetch_yahoo_http_df('USDTRY=X', period_days)
                    if usd_hist is None and YF_OK and yf:
                        try:
                            usd_hist = yf.Ticker('USDTRY=X').history(period=period, timeout=15)
                        except Exception:
                            pass
                    if usd_hist is not None and len(usd_hist) >= 2:
                        hist.index = hist.index.normalize()
                        usd_hist.index = usd_hist.index.normalize()
                        hist = hist[~hist.index.duplicated(keep='last')]
                        usd_hist = usd_hist[~usd_hist.index.duplicated(keep='last')]
                        common_dates = hist.index.intersection(usd_hist.index)
                        print(f"  [COMMODITY] {symbol} tarih eslestirme: hist={len(hist)}, usd={len(usd_hist)}, ortak={len(common_dates)}")
                        if len(common_dates) >= 10:
                            hist = hist.loc[common_dates].copy()
                            usd_rates = usd_hist.loc[common_dates, 'Close']
                            ons_to_gram = 31.1035
                            for col in ['Open', 'High', 'Low', 'Close']:
                                hist[col] = hist[col] * usd_rates.values / ons_to_gram
                            print(f"  [COMMODITY] {symbol} TL donusumu OK: {len(hist)} bar, son fiyat: {hist['Close'].iloc[-1]:.2f}")
                        else:
                            print(f"  [COMMODITY] {symbol} TL donusumu: ortak tarih az, son kur ile donusturuluyor")
                            last_usd_rate = float(usd_hist['Close'].iloc[-1])
                            ons_to_gram = 31.1035
                            for col in ['Open', 'High', 'Low', 'Close']:
                                hist[col] = hist[col] * last_usd_rate / ons_to_gram
                    else:
                        print(f"  [COMMODITY] {symbol}: USDTRY hist yok, cache'den kur aliniyor")
                        usd_idx = _cget(_index_cache, 'USDTRY')
                        if usd_idx:
                            last_rate = usd_idx.get('value', 0)
                            if last_rate > 0:
                                ons_to_gram = 31.1035
                                for col in ['Open', 'High', 'Low', 'Close']:
                                    hist[col] = hist[col] * last_rate / ons_to_gram

                _cset(_hist_cache, cache_key, hist)
                print(f"[COMMODITY] {symbol} OK: {len(hist)} bar")
            else:
                hist = None

        if hist is None:
            idx = _cget(_index_cache, symbol) or _cget(_index_cache, symbol.replace('TL', ''))
            if idx:
                return jsonify(safe_dict({
                    'success': True, 'code': symbol, 'name': name,
                    'price': idx.get('value', 0), 'change': idx.get('change', 0),
                    'changePercent': idx.get('changePct', 0), 'volume': 0,
                    'currency': currency, 'period': period, 'dataPoints': 0,
                    'indicators': {}, 'chartData': {'dates': [], 'prices': [], 'volumes': [], 'dataPoints': 0},
                    'fibonacci': {'levels': {}}, 'supportResistance': {'supports': [], 'resistances': []},
                    'pivotPoints': {'classic': {}, 'camarilla': {}, 'woodie': {}, 'current': 0},
                    'recommendation': {
                        'weekly': {'action': 'neutral', 'confidence': 0, 'reasons': []},
                        'monthly': {'action': 'neutral', 'confidence': 0, 'reasons': []},
                        'yearly': {'action': 'neutral', 'confidence': 0, 'reasons': []},
                    },
                }))
            return jsonify({'error': f'{symbol} verisi bulunamadi'}), 404

        cp = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else cp
        w52 = calc_52w(hist)

        return jsonify(safe_dict({
            'success': True, 'code': symbol, 'name': name,
            'price': sf(cp, 4 if currency == 'USD' else 2),
            'change': sf(cp - prev, 4 if currency == 'USD' else 2),
            'changePercent': sf((cp - prev) / prev * 100 if prev else 0),
            'volume': si(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
            'dayHigh': sf(hist['High'].iloc[-1], 4 if currency == 'USD' else 2),
            'dayLow': sf(hist['Low'].iloc[-1], 4 if currency == 'USD' else 2),
            'dayOpen': sf(hist['Open'].iloc[-1], 4 if currency == 'USD' else 2),
            'prevClose': sf(prev, 4 if currency == 'USD' else 2),
            'currency': currency,
            'period': period, 'dataPoints': len(hist),
            'week52': w52,
            'indicators': (cmd_indics := calc_all_indicators(hist, cp)),
            'chartData': prepare_chart_data(hist),
            'fibonacci': calc_fibonacci(hist),
            'supportResistance': calc_support_resistance(hist),
            'pivotPoints': calc_pivot_points(hist),
            'recommendation': calc_recommendation(hist, cmd_indics),
            'fundamentals': calc_fundamentals(hist, symbol),
        }))
    except Exception as e:
        print(f"COMMODITY {symbol}: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@market_bp.route('/api/stock/<symbol>/events')
def stock_events(symbol):
    return jsonify({'success': True, 'symbol': symbol.upper(), 'events': {'dividends': [], 'splits': []}})


@market_bp.route('/api/stock/<symbol>/kap')
def stock_kap(symbol):
    """KAP bildirimlerini scrape et"""
    symbol = symbol.upper()
    try:
        url = f"https://www.kap.org.tr/tr/api/disclosures?company={symbol}&type=FR&lang=tr"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://www.kap.org.tr/',
        }
        try:
            resp = req_lib.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                notifications = []
                items = data if isinstance(data, list) else data.get('disclosures', data.get('data', []))
                for item in items[:20]:
                    notifications.append({
                        'title': item.get('title', item.get('subject', '')),
                        'date': item.get('publishDate', item.get('date', '')),
                        'type': item.get('type', item.get('disclosureType', '')),
                        'summary': item.get('summary', '')[:200],
                    })
                if notifications:
                    return jsonify({'success': True, 'symbol': symbol, 'notifications': notifications})
        except Exception as e:
            print(f"[KAP-API] {symbol}: {e}")

        try:
            url2 = f"https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HaberlerHisseTekil?hession={symbol}&startdate={datetime.now().strftime('%d-%m-%Y')}&enddate={datetime.now().strftime('%d-%m-%Y')}"
            # verify=False: isyatirim cert chain sorunu (platform bağımlı); public haber verisi, credential akmaz
            resp2 = req_lib.get(url2, headers=IS_YATIRIM_HEADERS, timeout=10, verify=False)
            if resp2.status_code == 200:
                data2 = resp2.json()
                news = data2.get('value', [])
                notifications = [{'title': n.get('BASLIK', ''), 'date': n.get('TARIH', ''), 'type': 'haber', 'summary': ''} for n in news[:10]]
                return jsonify({'success': True, 'symbol': symbol, 'notifications': notifications})
        except Exception as e:
            print(f"[KAP-ISYATIRIM] {symbol}: {e}")

        return jsonify({'success': True, 'symbol': symbol, 'notifications': [], 'message': 'KAP verisi alinamadi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@market_bp.route('/api/compare', methods=['POST'])
def compare():
    """Hisseleri detayli karsilastir"""
    try:
        data = request.json or {}
        symbols = data.get('symbols', [])
        if len(symbols) < 2:
            return jsonify({'error': 'En az 2 hisse gerekli'}), 400

        results = []
        for sym in symbols[:5]:
            sym = sym.upper()
            s = _cget(_stock_cache, sym)
            if not s:
                continue

            hist = _cget_hist(f"{sym}_1y")
            indicators = {}
            if hist is not None and len(hist) >= 14:
                c = hist['Close'].values.astype(float)
                h = hist['High'].values.astype(float)
                l = hist['Low'].values.astype(float)
                rsi = calc_rsi(c)
                macd = calc_macd(c)
                ema = calc_ema(c, float(c[-1]))
                w52 = calc_52w(hist)
                indicators = {
                    'rsi': rsi.get('value', 0),
                    'rsiSignal': rsi.get('signal', 'neutral'),
                    'macdSignal': macd.get('signalType', 'neutral'),
                    'ema20': ema.get('ema20', 0),
                    'ema50': ema.get('ema50', 0),
                    'high52w': w52.get('high52w', 0),
                    'low52w': w52.get('low52w', 0),
                    'pos52w': w52.get('currentPct', 50),
                }

            results.append({
                'code': sym, 'name': s['name'], 'price': s['price'],
                'change': s['change'], 'changePct': s['changePct'],
                'volume': s['volume'], 'open': s.get('open', 0),
                'high': s.get('high', 0), 'low': s.get('low', 0),
                'prevClose': s.get('prevClose', 0),
                'gap': s.get('gap', 0), 'gapPct': s.get('gapPct', 0),
                **indicators,
            })

        return jsonify(safe_dict({'success': True, 'comparison': results}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@market_bp.route('/api/screener', methods=['POST'])
def screener():
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'matches': [], 'message': 'Veriler yukleniyor'})
        conditions = request.json.get('conditions', []) if request.json else []
        matches = []
        for s in stocks:
            ok = True
            for cd in conditions:
                ind, op, val = cd.get('indicator', ''), cd.get('operator', '>'), float(cd.get('value', 0))
                sv = s.get(ind, s.get('changePct', 0))
                try:
                    if op == '>' and not (float(sv) > val):
                        ok = False; break
                    elif op == '<' and not (float(sv) < val):
                        ok = False; break
                except Exception:
                    ok = False; break
            if ok:
                matches.append(s)
        return jsonify(safe_dict({'success': True, 'matches': matches[:50], 'totalMatches': len(matches)}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@market_bp.route('/api/sectors')
def sectors():
    try:
        sd = []
        for sn, syms in SECTOR_MAP.items():
            stocks = _get_stocks(syms)
            changes = [s['changePct'] for s in stocks if 'changePct' in s]
            sd.append({'name': sn, 'stockCount': len(syms), 'avgChange': sf(np.mean(changes)) if changes else 0, 'symbols': syms})
        sd.sort(key=lambda x: x['avgChange'], reverse=True)
        return jsonify(safe_dict({'success': True, 'sectors': sd}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@market_bp.route('/api/search')
def search():
    q = request.args.get('q', '').upper()
    return (
        jsonify({'success': True, 'results': [{'code': c, 'name': n} for c, n in BIST100_STOCKS.items() if q in c or q in n.upper()][:10]})
        if q else jsonify({'results': []})
    )
