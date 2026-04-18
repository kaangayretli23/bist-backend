"""
BIST Pro - Signal Backtest
calc_signal_backtest
calc_market_regime, calc_sector_relative_strength → signals_market.py
"""
import numpy as np
import pandas as pd
import time
from config import sf, si, _lock, _stock_cache, _index_cache, _cget, _get_stocks, BIST100_STOCKS, SECTOR_MAP

# Işlem maliyeti: round-trip komisyon + BSMV + spread tahmini (BIST ortalaması ~%0.2)
FEE_PCT = 0.2
# BIST'te ac iga satis (short selling) cogu kurumda kisitli — sell sinyalleri "executable" degil
SHORT_EXECUTABLE = False

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
            avg_loss = sf(abs(float(np.mean([r for r in r10 if r < 0]))) if any(r < 0 for r in r10) else 0.0)
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
                arr[i] = 100.0 - (100.0/(1.0+pmf/nmf)) if nmf > 0 else (50.0 if pmf == 0 else 100.0)
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
                # Islem maliyeti: round-trip giris+cikis komisyonu net getiriyi dusurur
                m5  -= FEE_PCT
                m10 -= FEE_PCT
                m20 -= FEE_PCT
                signals.append({
                    'day': i, 'type': stype, 'reason': reason,
                    'price': sf(ep),
                    'ret5d':  sf(m5),  'ret10d': sf(m10), 'ret20d': sf(m20),
                    'win5d':  m5 > 0, 'win10d': m10 > 0, 'win20d': m20 > 0,
                    'executable': (stype == 'buy') or SHORT_EXECUTABLE,
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

        # ---- Aktif setup'lar (son 20 bar, ileri getiri yok — canli izleme icin) ----
        active_setups = []
        for i in range(end_i, n):
            ep = float(c[i])
            triggers = []
            rsi = rsi_a[i]
            if   rsi < 30: triggers.append(('buy',  'RSI < 30'))
            elif rsi > 70: triggers.append(('sell', 'RSI > 70'))
            if i > 0:
                if   macd_v[i] > macd_s[i] and macd_v[i-1] <= macd_s[i-1]: triggers.append(('buy',  'MACD Kesisim'))
                elif macd_v[i] < macd_s[i] and macd_v[i-1] >= macd_s[i-1]: triggers.append(('sell', 'MACD Kesisim'))
            if not np.isnan(bb_l[i]) and bb_l[i] > 0:
                if   ep < bb_l[i]: triggers.append(('buy',  'Bollinger Alt Bant'))
                elif ep > bb_u[i]: triggers.append(('sell', 'Bollinger Ust Bant'))
            if   stoch_k[i] < 20: triggers.append(('buy',  'Stochastic Asiri Satim'))
            elif stoch_k[i] > 80: triggers.append(('sell', 'Stochastic Asiri Alim'))
            if i > 0:
                now_bull = ep > ema20[i] and ema20[i] > ema50[i]
                now_bear = ep < ema20[i] and ema20[i] < ema50[i]
                prv_bull = float(c[i-1]) > ema20[i-1] and ema20[i-1] > ema50[i-1]
                prv_bear = float(c[i-1]) < ema20[i-1] and ema20[i-1] < ema50[i-1]
                if   now_bull and not prv_bull: triggers.append(('buy',  'EMA Yukari Kesisim'))
                elif now_bear and not prv_bear: triggers.append(('sell', 'EMA Asagi Kesisim'))
            if   wpr_a[i] < -80: triggers.append(('buy',  'Williams %R Asiri Satim'))
            elif wpr_a[i] > -20: triggers.append(('sell', 'Williams %R Asiri Alim'))
            if   cci_a[i] < -100: triggers.append(('buy',  'CCI Asiri Satim'))
            elif cci_a[i] >  100: triggers.append(('sell', 'CCI Asiri Alim'))
            if   mfi_a[i] < 20: triggers.append(('buy',  'MFI Asiri Satim'))
            elif mfi_a[i] > 80: triggers.append(('sell', 'MFI Asiri Alim'))
            if i >= 10:
                obv_slope   = float(obv_a[i]  - obv_a[i-10])
                price_slope = float(c[i]) - float(c[i-10])
                if   obv_slope > 0 and price_slope < 0: triggers.append(('buy',  'OBV Pozitif Diverjans'))
                elif obv_slope < 0 and price_slope > 0: triggers.append(('sell', 'OBV Negatif Diverjans'))
            for stype, reason in triggers:
                active_setups.append({
                    'day': i, 'barsFromEnd': n - 1 - i,
                    'type': stype, 'reason': reason, 'price': sf(ep),
                    'executable': (stype == 'buy') or SHORT_EXECUTABLE,
                })

        if not signals:
            return {'totalSignals': 0, 'activeSetups': active_setups, 'message': 'Sinyal bulunamadi'}

        # Clustering dedupe: aynı bar+yön tekrarı (RSI<30 AND BB-alt AND Stoch<20 → 1 sinyal)
        # overall/buy/sell istatistikleri bağımsız gözlemlerden hesaplansın
        def _dedupe_by_bar(sigs):
            seen = {}
            for s in sigs:
                key = (s['day'], s['type'])
                if key not in seen:
                    seen[key] = s
            return list(seen.values())

        buy_sigs  = [s for s in signals if s['type'] == 'buy']
        sell_sigs = [s for s in signals if s['type'] == 'sell']
        uniq_buy  = _dedupe_by_bar(buy_sigs)
        uniq_sell = _dedupe_by_bar(sell_sigs)
        uniq_all  = _dedupe_by_bar(signals)

        # Her indikatör için istatistik (byReason clustering'e rağmen per-indikatör anlamlı)
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
        # Rastgele giris yapilsaydi ortalama 10-gunluk net getiri ne olurdu? (round-trip komisyon dahil)
        baseline_rets = [
            ((float(c[min(i+10, n-1)]) - float(c[i])) / float(c[i])) * 100.0 - FEE_PCT
            for i in range(start_i, end_i)
        ]
        baseline_avg  = sf(float(np.mean(baseline_rets))) if baseline_rets else 0
        full_period_r = sf(((float(c[-1]) - float(c[start_i])) / float(c[start_i])) * 100.0 - FEE_PCT)

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
                r -= FEE_PCT
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
            'totalSignals':    len(uniq_all),   # dedupe edilmiş (bar+yön bazında)
            'rawSignalCount':  len(signals),    # her indikatör tetiklemesi ayrı (eskisi)
            'buySignals':   calc_stats_v2(uniq_buy),
            'sellSignals':  calc_stats_v2(uniq_sell),
            'overall':      calc_stats_v2(uniq_all),
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
            'activeSetups':   active_setups,  # son 20 bar canli setup'lar (ileri getiri yok)
            'feePct':         FEE_PCT,
            'shortExecutable': SHORT_EXECUTABLE,
        }
    except Exception as e:
        print(f"  [BACKTEST] Hata: {e}")
        import traceback; traceback.print_exc()
        return {'totalSignals': 0, 'error': str(e)}

# calc_market_regime ve calc_sector_relative_strength → signals_market.py

from signals_market import calc_market_regime, calc_sector_relative_strength  # re-export
