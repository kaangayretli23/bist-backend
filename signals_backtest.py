"""
BIST Pro - Signal Backtest & Market Analysis
calc_signal_backtest, calc_market_regime, calc_sector_relative_strength
"""
import numpy as np
import pandas as pd
import time
from config import sf, si, _lock, _stock_cache, _index_cache, _cget, _get_stocks, BIST100_STOCKS, SECTOR_MAP
from indicators import *
from indicators import _market_regime_cache, _resample_to_tf

def calc_signal_backtest(hist, lookback_days=252):
    """Enhanced backtest: 9 indikatör, Profit Factor / Sharpe / benchmark, BIST RSI kalibrasyonu"""
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        v = np.where(np.isnan(v), 0, v)
        n = len(c)
        if n < 60:
            return {'totalSignals': 0, 'message': 'Yeterli veri yok'}

        # ---- Metrik yardimcilari ----
        def _pf(rets):
            """Profit Factor = toplam kazanc / toplam kayip"""
            wins   = sum(r for r in rets if r > 0)
            losses = sum(-r for r in rets if r < 0)
            if losses == 0:
                return sf(99.0 if wins > 0 else 0.0)
            return sf(wins / losses)

        def _sharpe(rets, period_days=10):
            """Yillik Sharpe orani"""
            if len(rets) < 3:
                return 0.0
            m = float(np.mean(rets))
            s = float(np.std(rets))
            return sf(m / s * float(np.sqrt(252.0 / period_days)) if s > 0 else 0.0)

        def calc_stats_v2(sigs):
            if not sigs:
                return {
                    'count': 0,
                    'winRate5d': 0, 'winRate10d': 0, 'winRate20d': 0,
                    'avgRet5d': 0, 'avgRet10d': 0, 'avgRet20d': 0,
                    'profitFactor5d': 0, 'profitFactor10d': 0, 'profitFactor20d': 0,
                    'sharpe5d': 0, 'sharpe10d': 0, 'sharpe20d': 0,
                    'avgWin10d': 0, 'avgLoss10d': 0, 'grade': '-',
                }
            r5  = [float(s['ret5d'])  for s in sigs]
            r10 = [float(s['ret10d']) for s in sigs]
            r20 = [float(s['ret20d']) for s in sigs]
            wr5  = sf(sum(1 for s in sigs if s['win5d'])  / len(sigs) * 100)
            wr10 = sf(sum(1 for s in sigs if s['win10d']) / len(sigs) * 100)
            wr20 = sf(sum(1 for s in sigs if s['win20d']) / len(sigs) * 100)
            pf10 = _pf(r10)
            sh10 = _sharpe(r10, 10)
            avg_win  = sf(float(np.mean([r for r in r10 if r > 0])) if any(r > 0 for r in r10) else 0.0)
            avg_loss = sf(float(np.mean([r for r in r10 if r < 0])) if any(r < 0 for r in r10) else 0.0)
            grade = ('Guclu' if float(pf10) >= 1.5 and float(wr10) >= 55
                     else ('Orta' if float(pf10) >= 1.0 and float(wr10) >= 50 else 'Zayif'))
            return {
                'count': len(sigs),
                'winRate5d': wr5, 'winRate10d': wr10, 'winRate20d': wr20,
                'avgRet5d':  sf(float(np.mean(r5))),
                'avgRet10d': sf(float(np.mean(r10))),
                'avgRet20d': sf(float(np.mean(r20))),
                'profitFactor5d':  _pf(r5),
                'profitFactor10d': pf10,
                'profitFactor20d': _pf(r20),
                'sharpe5d':  _sharpe(r5, 5),
                'sharpe10d': sh10,
                'sharpe20d': _sharpe(r20, 20),
                'avgWin10d':  avg_win,
                'avgLoss10d': avg_loss,
                'grade': grade,
            }

        # ---- Indikatör dizilerini onceden hesapla (vektörel) ----

        # 1. RSI (Wilder smoothing)
        def _rsi_arr(closes, period=14):
            delta = np.diff(closes)
            g  = np.where(delta > 0, delta, 0.0)
            lo = np.where(delta < 0, -delta, 0.0)
            arr = np.full(len(closes), 50.0)
            if len(delta) < period:
                return arr
            ag, al = float(np.mean(g[:period])), float(np.mean(lo[:period]))
            for i in range(period, len(delta)):
                ag = (ag*(period-1) + g[i]) / period
                al = (al*(period-1) + lo[i]) / period
                rs = ag/al if al > 0 else 100.0
                arr[i+1] = 100.0 - (100.0/(1.0+rs))
            return arr

        # 2. MACD
        def _macd_arr(closes):
            s = pd.Series(closes)
            mv = (s.ewm(span=12, adjust=False).mean() - s.ewm(span=26, adjust=False).mean()).values
            sv = pd.Series(mv).ewm(span=9, adjust=False).mean().values
            return mv, sv

        # 3. Bollinger Bands
        def _boll_arr(closes, period=20, mult=2.0):
            s   = pd.Series(closes)
            mid = s.rolling(period).mean().values
            std = s.rolling(period).std(ddof=1).values
            return mid + mult*std, mid, mid - mult*std

        # 4. Stochastic %K
        def _stoch_arr(closes, highs, lows, period=14):
            k = np.full(len(closes), 50.0)
            for i in range(period-1, len(closes)):
                hi = float(np.max(highs[i-period+1:i+1]))
                lo = float(np.min(lows[i-period+1:i+1]))
                k[i] = ((closes[i]-lo)/(hi-lo))*100.0 if hi != lo else 50.0
            return k

        # 5. EMA
        def _ema(closes, span):
            return pd.Series(closes).ewm(span=span, adjust=False).mean().values

        # 6. Williams %R
        def _wpr_arr(closes, highs, lows, period=14):
            w = np.full(len(closes), -50.0)
            for i in range(period-1, len(closes)):
                hh = float(np.max(highs[i-period+1:i+1]))
                ll = float(np.min(lows[i-period+1:i+1]))
                w[i] = ((hh-closes[i])/(hh-ll))*-100.0 if hh != ll else -50.0
            return w

        # 7. CCI
        def _cci_arr(closes, highs, lows, period=20):
            tp  = (highs + lows + closes) / 3.0
            arr = np.zeros(len(closes))
            for i in range(period-1, len(closes)):
                tp_w = tp[i-period+1:i+1]
                sma  = float(np.mean(tp_w))
                md   = float(np.mean(np.abs(tp_w - sma)))
                arr[i] = (tp[i]-sma)/(0.015*md) if md > 0 else 0.0
            return arr

        # 8. MFI
        def _mfi_arr(closes, highs, lows, volumes, period=14):
            tp  = (highs + lows + closes) / 3.0
            mf  = tp * volumes
            arr = np.full(len(closes), 50.0)
            for i in range(period, len(closes)):
                # w_tp ve w_prev ayni boyutta olmali
                w_tp   = tp[i-period+1:i+1]   # [i-period+1 .. i]  shape=(period,)
                w_prev = tp[i-period:i]         # [i-period   .. i-1] shape=(period,)
                w_mf   = mf[i-period+1:i+1]
                pmf = float(np.sum(w_mf[w_tp > w_prev]))
                nmf = float(np.sum(w_mf[w_tp <= w_prev]))
                arr[i] = 100.0 - (100.0/(1.0+pmf/nmf)) if nmf > 0 else 100.0
            return arr

        # 9. OBV
        def _obv_arr(closes, volumes):
            obv = np.zeros(len(closes))
            for i in range(1, len(closes)):
                if   closes[i] > closes[i-1]: obv[i] = obv[i-1] + volumes[i]
                elif closes[i] < closes[i-1]: obv[i] = obv[i-1] - volumes[i]
                else:                          obv[i] = obv[i-1]
            return obv

        rsi_a          = _rsi_arr(c)
        macd_v, macd_s = _macd_arr(c)
        bb_u, _bb_m, bb_l = _boll_arr(c)
        stoch_k        = _stoch_arr(c, h, l)
        ema20          = _ema(c, 20)
        ema50          = _ema(c, 50)
        wpr_a          = _wpr_arr(c, h, l)
        cci_a          = _cci_arr(c, h, l)
        mfi_a          = _mfi_arr(c, h, l, v)
        obv_a          = _obv_arr(c, v)

        # ---- Sinyal uretimi ----
        start_i = 60     # 60 bar stabilite suresi
        end_i   = n - 20 # 20 bar gelecegi gormek icin
        signals = []

        for i in range(start_i, end_i):
            ep  = float(c[i])
            r5  = ((float(c[min(i+5,  n-1)]) - ep) / ep) * 100.0
            r10 = ((float(c[min(i+10, n-1)]) - ep) / ep) * 100.0
            r20 = ((float(c[min(i+20, n-1)]) - ep) / ep) * 100.0

            def _add(stype, reason):
                m5, m10, m20 = r5, r10, r20
                if stype == 'sell':
                    m5, m10, m20 = -m5, -m10, -m20
                signals.append({
                    'day': i, 'type': stype, 'reason': reason,
                    'price': sf(ep),
                    'ret5d':  sf(m5),  'ret10d': sf(m10), 'ret20d': sf(m20),
                    'win5d':  m5 > 0, 'win10d': m10 > 0, 'win20d': m20 > 0,
                })

            # RSI
            rsi = rsi_a[i]
            if   rsi < 30: _add('buy',  'RSI < 30')
            elif rsi > 70: _add('sell', 'RSI > 70')

            # MACD kesisim
            if i > 0:
                if   macd_v[i] > macd_s[i] and macd_v[i-1] <= macd_s[i-1]: _add('buy',  'MACD Kesisim')
                elif macd_v[i] < macd_s[i] and macd_v[i-1] >= macd_s[i-1]: _add('sell', 'MACD Kesisim')

            # Bollinger Bantlari
            if not np.isnan(bb_l[i]) and bb_l[i] > 0:
                if   ep < bb_l[i]: _add('buy',  'Bollinger Alt Bant')
                elif ep > bb_u[i]: _add('sell', 'Bollinger Ust Bant')

            # Stochastic
            if   stoch_k[i] < 20: _add('buy',  'Stochastic Asiri Satim')
            elif stoch_k[i] > 80: _add('sell', 'Stochastic Asiri Alim')

            # EMA kesisim (yeni kesisim aninda tetikle)
            if i > 0:
                now_bull = ep > ema20[i] and ema20[i] > ema50[i]
                now_bear = ep < ema20[i] and ema20[i] < ema50[i]
                prv_bull = float(c[i-1]) > ema20[i-1] and ema20[i-1] > ema50[i-1]
                prv_bear = float(c[i-1]) < ema20[i-1] and ema20[i-1] < ema50[i-1]
                if   now_bull and not prv_bull: _add('buy',  'EMA Yukari Kesisim')
                elif now_bear and not prv_bear: _add('sell', 'EMA Asagi Kesisim')

            # Williams %R
            if   wpr_a[i] < -80: _add('buy',  'Williams %R Asiri Satim')
            elif wpr_a[i] > -20: _add('sell', 'Williams %R Asiri Alim')

            # CCI
            if   cci_a[i] < -100: _add('buy',  'CCI Asiri Satim')
            elif cci_a[i] >  100: _add('sell', 'CCI Asiri Alim')

            # MFI
            if   mfi_a[i] < 20: _add('buy',  'MFI Asiri Satim')
            elif mfi_a[i] > 80: _add('sell', 'MFI Asiri Alim')

            # OBV diverjans (10-gunluk egim)
            if i >= 10:
                obv_slope   = float(obv_a[i]  - obv_a[i-10])
                price_slope = float(c[i]) - float(c[i-10])
                if   obv_slope > 0 and price_slope < 0: _add('buy',  'OBV Pozitif Diverjans')
                elif obv_slope < 0 and price_slope > 0: _add('sell', 'OBV Negatif Diverjans')

        if not signals:
            return {'totalSignals': 0, 'message': 'Sinyal bulunamadi'}

        buy_sigs  = [s for s in signals if s['type'] == 'buy']
        sell_sigs = [s for s in signals if s['type'] == 'sell']

        # Her indikatör için istatistik
        by_reason = {}
        for s in signals:
            by_reason.setdefault(s['reason'], []).append(s)

        reason_stats = {r: {**calc_stats_v2(sigs), 'reason': r}
                        for r, sigs in by_reason.items()}

        # Profit Factor'a göre sırala
        ranked = sorted(
            reason_stats.values(),
            key=lambda x: (float(x.get('profitFactor10d', 0)),
                           float(x.get('winRate10d', 0))),
            reverse=True,
        )

        # ---- Buy-and-Hold Benchmark ----
        # Rastgele giris yapilsaydi ortalama 10-gunluk getiri ne olurdu?
        baseline_rets = [
            ((float(c[min(i+10, n-1)]) - float(c[i])) / float(c[i])) * 100.0
            for i in range(start_i, end_i)
        ]
        baseline_avg  = sf(float(np.mean(baseline_rets))) if baseline_rets else 0
        full_period_r = sf(((float(c[-1]) - float(c[start_i])) / float(c[start_i])) * 100.0)

        # ---- BIST RSI Kalibrasyonu ----
        # Hangi RSI esigi BIST'te daha iyi calisıyor?
        rsi_calib = {}
        for lo_th, hi_th in [(25, 75), (30, 70), (35, 65)]:
            cal = []
            for i in range(start_i, end_i):
                rv = rsi_a[i]
                if   rv < lo_th: st = 'buy'
                elif rv > hi_th: st = 'sell'
                else: continue
                ep_c = float(c[i])
                r = ((float(c[min(i+10, n-1)]) - ep_c) / ep_c) * 100.0
                if st == 'sell':
                    r = -r
                cal.append(r)
            if cal:
                wins   = [r for r in cal if r > 0]
                losses = [abs(r) for r in cal if r < 0]
                rsi_calib[f'{lo_th}/{hi_th}'] = {
                    'signalCount':     len(cal),
                    'winRate10d':      sf(len(wins)/len(cal)*100),
                    'profitFactor10d': sf(sum(wins)/sum(losses) if losses else 99.0),
                    'avgReturn10d':    sf(float(np.mean(cal))),
                }
        best_rsi = (max(rsi_calib, key=lambda k: float(rsi_calib[k].get('profitFactor10d', 0)))
                    if rsi_calib else '30/70')

        return {
            'totalSignals': len(signals),
            'buySignals':   calc_stats_v2(buy_sigs),
            'sellSignals':  calc_stats_v2(sell_sigs),
            'overall':      calc_stats_v2(signals),
            'byReason':     reason_stats,
            'rankedIndicators': ranked,
            'recentSignals': signals[-10:],
            'benchmark': {
                'avgRandom10dReturn': baseline_avg,
                'fullPeriodReturn':   full_period_r,
                'note': 'avgRandom10dReturn: rastgele giris olsaydi beklenen 10-gunluk ortalama getiri',
            },
            'rsiCalibration':  rsi_calib,
            'bestRsiThreshold': best_rsi,
        }
    except Exception as e:
        print(f"  [BACKTEST] Hata: {e}")
        import traceback; traceback.print_exc()
        return {'totalSignals': 0, 'error': str(e)}


# =====================================================================
# FEATURE 2: DYNAMIC THRESHOLDS (Hisse bazli adaptif esikler)
# =====================================================================

def calc_market_regime():
    """BIST100 trend durumunu analiz et: bull/bear/sideways"""
    try:
        # Cache kontrol (5 dakika)
        if _market_regime_cache['regime'] and time.time() - _market_regime_cache['ts'] < 300:
            return _market_regime_cache['regime']

        # XU100 tarihsel verisini al
        hist = _cget_hist("XU100_1y")
        if hist is None:
            # Cache'de yoksa senkron olarak cek
            try:
                xu_df = _fetch_isyatirim_df("XU100", 365)
                if xu_df is not None and len(xu_df) >= 30:
                    _cset(_hist_cache, "XU100_1y", xu_df)
                    hist = xu_df
                    print("[REGIME] XU100 verisi senkron olarak cekildi")
            except Exception as xe:
                print(f"[REGIME] XU100 senkron cekme hatasi: {xe}")

        if hist is None:
            # Hala yoksa stock cache'den basit rejim hesapla
            stocks = _get_stocks()
            if stocks:
                advancing = sum(1 for s in stocks if s.get('changePct', 0) > 0)
                declining = sum(1 for s in stocks if s.get('changePct', 0) < 0)
                total = len(stocks)
                ratio = advancing / max(declining, 1)
                if ratio > 2:
                    regime_name, desc = 'strong_bull', 'Guclu Boga Piyasasi'
                elif ratio > 1.3:
                    regime_name, desc = 'bull', 'Boga Piyasasi'
                elif ratio < 0.5:
                    regime_name, desc = 'strong_bear', 'Guclu Ayi Piyasasi'
                elif ratio < 0.8:
                    regime_name, desc = 'bear', 'Ayi Piyasasi'
                else:
                    regime_name, desc = 'sideways', 'Yatay Piyasa'
                return {
                    'regime': regime_name, 'strength': sf(min(abs(ratio - 1) * 50, 100)),
                    'description': desc,
                    'reasons': [f'Yukselen: {advancing}, Dusen: {declining} (toplam {total})',
                                f'A/D orani: {sf(ratio)}'],
                    'indicators': {'breadthRatio': sf(ratio)},
                }
            return {'regime': 'unknown', 'strength': 0, 'description': 'Piyasa verisi mevcut degil'}

        c = hist['Close'].values.astype(float)
        n = len(c)
        if n < 50:
            return {'regime': 'unknown', 'strength': 0, 'description': 'Yeterli veri yok'}

        cur = float(c[-1])

        # SMA hesapla
        sma20 = float(np.mean(c[-20:])) if n >= 20 else cur
        sma50 = float(np.mean(c[-50:])) if n >= 50 else sma20
        sma200 = float(np.mean(c[-200:])) if n >= 200 else sma50

        # EMA hesapla
        s = pd.Series(c)
        ema20 = float(s.ewm(span=20).mean().iloc[-1])
        ema50 = float(s.ewm(span=50).mean().iloc[-1])

        # ADX (trend gucu)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        adx_data = calc_adx(h, l, c)
        adx_val = float(adx_data.get('value', 25))
        plus_di = float(adx_data.get('plusDI', 0))
        minus_di = float(adx_data.get('minusDI', 0))

        # RSI
        rsi = calc_rsi(c).get('value', 50)

        # Son 20 gun momentum
        ret_20d = ((cur - float(c[-20])) / float(c[-20])) * 100 if n >= 20 else 0
        ret_50d = ((cur - float(c[-50])) / float(c[-50])) * 100 if n >= 50 else 0

        # Volatilite
        if n >= 20:
            daily_returns = np.diff(c[-30:]) / c[-30:-1]
            volatility = float(np.std(daily_returns)) * (252 ** 0.5) * 100
        else:
            volatility = 25

        # Piyasa genisligi (cache'deki hisselerden)
        stocks = _get_stocks()
        advancing = sum(1 for s in stocks if s.get('changePct', 0) > 0)
        declining = sum(1 for s in stocks if s.get('changePct', 0) < 0)
        breadth_ratio = advancing / max(declining, 1)

        # Rejim belirleme skoru
        score = 0
        reasons = []

        # Fiyat > SMA pozisyonu
        if cur > sma20: score += 1; reasons.append('Fiyat SMA20 uzerinde')
        else: score -= 1; reasons.append('Fiyat SMA20 altinda')

        if cur > sma50: score += 1; reasons.append('Fiyat SMA50 uzerinde')
        else: score -= 1; reasons.append('Fiyat SMA50 altinda')

        if n >= 200:
            if cur > sma200: score += 1.5; reasons.append('Fiyat SMA200 uzerinde (uzun vadeli boga)')
            else: score -= 1.5; reasons.append('Fiyat SMA200 altinda (uzun vadeli ayi)')

        # SMA siralamasi
        if sma20 > sma50: score += 1; reasons.append('SMA20 > SMA50 (yukari trend)')
        else: score -= 1; reasons.append('SMA20 < SMA50 (asagi trend)')

        # Momentum
        if ret_20d > 5: score += 1; reasons.append(f'20 gunluk getiri: %{sf(ret_20d)} (guclu)')
        elif ret_20d > 0: score += 0.5
        elif ret_20d < -5: score -= 1; reasons.append(f'20 gunluk getiri: %{sf(ret_20d)} (zayif)')
        else: score -= 0.5

        # ADX trend gucu
        if adx_val > 25:
            if plus_di > minus_di: score += 1; reasons.append(f'ADX={sf(adx_val)}: Guclu yukari trend')
            else: score -= 1; reasons.append(f'ADX={sf(adx_val)}: Guclu asagi trend')

        # Piyasa genisligi
        if breadth_ratio > 1.5: score += 0.5; reasons.append(f'Piyasa genisligi pozitif ({advancing}/{declining})')
        elif breadth_ratio < 0.7: score -= 0.5; reasons.append(f'Piyasa genisligi negatif ({advancing}/{declining})')

        # Rejim siniflandirma
        if score >= 3:
            regime = 'strong_bull'
            desc = 'Guclu Boga Piyasasi - Alis sinyalleri daha guvenilir'
        elif score >= 1:
            regime = 'bull'
            desc = 'Boga Piyasasi - Genel yukari trend'
        elif score <= -3:
            regime = 'strong_bear'
            desc = 'Guclu Ayi Piyasasi - Satis sinyalleri daha guvenilir'
        elif score <= -1:
            regime = 'bear'
            desc = 'Ayi Piyasasi - Genel asagi trend'
        else:
            regime = 'sideways'
            desc = 'Yatay Piyasa - Belirsizlik hakim, dikkatli olun'

        # Sinyal guven carpani
        if regime in ('strong_bull', 'bull'):
            buy_confidence_mult = 1.2
            sell_confidence_mult = 0.8
        elif regime in ('strong_bear', 'bear'):
            buy_confidence_mult = 0.8
            sell_confidence_mult = 1.2
        else:
            buy_confidence_mult = 1.0
            sell_confidence_mult = 1.0

        result = {
            'regime': regime,
            'score': sf(score),
            'strength': sf(min(abs(score) / 5 * 100, 100)),
            'description': desc,
            'reasons': reasons[:6],
            'indicators': {
                'sma20': sf(sma20), 'sma50': sf(sma50), 'sma200': sf(sma200) if n >= 200 else None,
                'adx': sf(adx_val), 'rsi': sf(rsi),
                'ret20d': sf(ret_20d), 'ret50d': sf(ret_50d),
                'volatility': sf(volatility),
                'breadthRatio': sf(breadth_ratio),
            },
            'confidence_multiplier': {
                'buy': buy_confidence_mult,
                'sell': sell_confidence_mult,
            },
        }

        _market_regime_cache['regime'] = result
        _market_regime_cache['ts'] = time.time()
        return result
    except Exception as e:
        print(f"  [REGIME] Hata: {e}")
        return {'regime': 'unknown', 'strength': 0, 'description': str(e)}


# =====================================================================
# FEATURE 5: SECTOR ANALYSIS & RELATIVE STRENGTH

def calc_sector_relative_strength():
    """Sektor bazli goreceli guc analizi"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return {'sectors': []}

        stock_map = {s['code']: s for s in stocks}
        sector_results = []

        for sector_name, symbols in SECTOR_MAP.items():
            sector_stocks = []
            returns_1d = []
            returns_1w = []
            returns_1m = []

            for sym in symbols:
                s = stock_map.get(sym)
                if not s:
                    continue

                hist = _cget_hist(f"{sym}_1y")
                stock_info = {'code': sym, 'name': s.get('name', sym), 'price': s['price'], 'changePct': s.get('changePct', 0)}

                if hist is not None and len(hist) >= 22:
                    c = hist['Close'].values.astype(float)
                    n = len(c)
                    stock_info['ret1w'] = sf(((float(c[-1]) - float(c[-5])) / float(c[-5])) * 100) if n >= 5 else 0
                    stock_info['ret1m'] = sf(((float(c[-1]) - float(c[-22])) / float(c[-22])) * 100) if n >= 22 else 0
                    stock_info['ret3m'] = sf(((float(c[-1]) - float(c[-66])) / float(c[-66])) * 100) if n >= 66 else 0

                    # RSI
                    rsi = calc_rsi(c)
                    stock_info['rsi'] = rsi.get('value', 50)
                    stock_info['rsiSignal'] = rsi.get('signal', 'neutral')

                    returns_1d.append(s.get('changePct', 0))
                    returns_1w.append(float(stock_info.get('ret1w', 0)))
                    returns_1m.append(float(stock_info.get('ret1m', 0)))

                sector_stocks.append(stock_info)

            if not sector_stocks:
                continue

            avg_1d = sf(np.mean(returns_1d)) if returns_1d else 0
            avg_1w = sf(np.mean(returns_1w)) if returns_1w else 0
            avg_1m = sf(np.mean(returns_1m)) if returns_1m else 0

            # Relative Strength Index (sektor bazli)
            rs_score = float(avg_1d) * 0.2 + float(avg_1w) * 0.3 + float(avg_1m) * 0.5

            sector_results.append({
                'name': sector_name,
                'displayName': {
                    'bankacilik': 'Bankacilik', 'havacilik': 'Havacilik',
                    'otomotiv': 'Otomotiv', 'enerji': 'Enerji',
                    'holding': 'Holding', 'perakende': 'Perakende',
                    'teknoloji': 'Teknoloji', 'telekom': 'Telekom',
                    'demir_celik': 'Demir Celik', 'gida': 'Gida',
                    'insaat': 'Insaat', 'gayrimenkul': 'Gayrimenkul',
                }.get(sector_name, sector_name),
                'avgChange1d': avg_1d,
                'avgChange1w': avg_1w,
                'avgChange1m': avg_1m,
                'relativeStrength': sf(rs_score),
                'stockCount': len(sector_stocks),
                'stocks': sorted(sector_stocks, key=lambda x: float(x.get('ret1m', 0)), reverse=True),
                'topPerformer': max(sector_stocks, key=lambda x: float(x.get('ret1m', 0)))['code'] if sector_stocks else '',
                'worstPerformer': min(sector_stocks, key=lambda x: float(x.get('ret1m', 0)))['code'] if sector_stocks else '',
            })

        sector_results.sort(key=lambda x: float(x['relativeStrength']), reverse=True)

        # Sektor rotasyonu tespit
        rotation = 'neutral'
        if sector_results:
            top_sectors = sector_results[:3]
            defensive = ['perakende', 'gida', 'telekom']
            cyclical = ['bankacilik', 'otomotiv', 'enerji', 'holding']
            top_names = [s['name'] for s in top_sectors]
            if any(s in top_names for s in cyclical):
                rotation = 'risk_on'
            elif any(s in top_names for s in defensive):
                rotation = 'risk_off'

        return {
            'sectors': sector_results,
            'rotation': rotation,
            'rotationDescription': {
                'risk_on': 'Dongusel sektorler lider - Risk istahi yuksek',
                'risk_off': 'Defansif sektorler lider - Temkinli piyasa',
                'neutral': 'Belirgin sektor rotasyonu yok',
            }.get(rotation, ''),
        }
    except Exception as e:
        print(f"  [SECTOR-RS] Hata: {e}")
        return {'sectors': [], 'error': str(e)}
