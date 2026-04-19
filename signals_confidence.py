"""
BIST Pro - Signal Confidence & Alerts
check_signal_alerts, calc_ml_confidence
signals_core.py'dan ayrıştırıldı (600 satır kuralı).
"""
import time
import numpy as np
import pandas as pd
from config import sf, _cget_hist, _get_stocks
from indicators import calc_rsi, calc_support_resistance, calc_dynamic_thresholds, calc_candlestick_patterns
from indicators_patterns import calc_mtf_signal
from signals_market import calc_market_regime, REGIMES_BULLISH, REGIMES_BEARISH

# Alert uretimi pahalidir (her hisse icin 1y hist + 4 indikatör) — 5 dakikalik TTL cache
_ALERTS_CACHE = {'alerts': None, 'ts': 0.0}
_ALERTS_CACHE_TTL = 300


def check_signal_alerts(max_alerts=200):
    """Sinyal bazli otomatik uyari kontrolu - enhanced"""
    now_ts = time.time()
    cached = _ALERTS_CACHE.get('alerts')
    if cached is not None and now_ts - _ALERTS_CACHE.get('ts', 0.0) < _ALERTS_CACHE_TTL:
        return cached[:max_alerts]

    stocks = _get_stocks()
    if not stocks:
        return []

    alerts = []
    regime = calc_market_regime()

    for stock in stocks:
        sym = stock['code']
        try:
            hist = _cget_hist(f"{sym}_1y")
            if hist is None or len(hist) < 50:
                continue

            c = hist['Close'].values.astype(float)
            h = hist['High'].values.astype(float)
            l = hist['Low'].values.astype(float)
            o = hist['Open'].values.astype(float) if 'Open' in hist.columns else c.copy()
            v = hist['Volume'].values.astype(float)
            cp = float(c[-1])
            n = len(c)

            # Mum formasyonlari
            candles = calc_candlestick_patterns(o, h, l, c)
            for p in candles.get('patterns', []):
                if p.get('strength', 0) >= 4:
                    alerts.append({
                        'symbol': sym,
                        'type': 'candlestick',
                        'pattern_name': p['name'],
                        'signal': p['type'],
                        'message': f"{sym} ({sf(cp)} TL): {p['name']} - {p['description']}",
                        'strength': p['strength'],
                    })

            # Altin/Olum kesisim
            if n >= 200:
                ema50 = pd.Series(c).ewm(span=50).mean().values
                ema200 = pd.Series(c).ewm(span=200).mean().values
                if ema50[-1] > ema200[-1] and ema50[-2] <= ema200[-2]:
                    alerts.append({'symbol': sym, 'type': 'golden_cross', 'signal': 'bullish',
                        'message': f"ALTIN KESISIM: {sym} ({sf(cp)} TL) - EMA50 > EMA200", 'strength': 5})
                elif ema50[-1] < ema200[-1] and ema50[-2] >= ema200[-2]:
                    alerts.append({'symbol': sym, 'type': 'death_cross', 'signal': 'bearish',
                        'message': f"OLUM KESISIMI: {sym} ({sf(cp)} TL) - EMA50 < EMA200", 'strength': 5})

            # Dinamik RSI esikleri
            thresholds = calc_dynamic_thresholds(c, h, l, v)
            rsi_val = calc_rsi(c).get('value', 50)
            if rsi_val < float(thresholds['rsi_oversold']):
                alerts.append({'symbol': sym, 'type': 'rsi_dynamic', 'signal': 'bullish',
                    'message': f"RSI ASIRI SATIM: {sym} ({sf(cp)} TL) RSI={sf(rsi_val)} < {thresholds['rsi_oversold']} (dinamik esik)", 'strength': 3})
            elif rsi_val > float(thresholds['rsi_overbought']):
                alerts.append({'symbol': sym, 'type': 'rsi_dynamic', 'signal': 'bearish',
                    'message': f"RSI ASIRI ALIM: {sym} ({sf(cp)} TL) RSI={sf(rsi_val)} > {thresholds['rsi_overbought']} (dinamik esik)", 'strength': 3})

        except Exception:
            continue

    alerts.sort(key=lambda x: x.get('strength', 0), reverse=True)
    _ALERTS_CACHE['alerts'] = alerts
    _ALERTS_CACHE['ts'] = now_ts
    return alerts[:max_alerts]


def calc_ml_confidence(hist, indicators, recommendation_score, signal_type='buy', symbol=None):
    """Sinyal guven skorunu coklu faktore gore hesapla (ML-inspired weighted scoring)"""
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        n = len(c)
        if n < 50:
            return {'confidence': 50, 'grade': 'C', 'factors': []}

        factors = []
        score = 0
        max_score = 0

        # 1. Indikatör Konsensüsü (agirlik: 25%)
        summary = indicators.get('summary', {})
        bc = summary.get('buySignals', 0)
        sc = summary.get('sellSignals', 0)
        total = summary.get('totalIndicators', 1)
        consensus = (bc if signal_type == 'buy' else sc) / total * 100
        consensus_score = min(consensus / 100 * 25, 25)
        score += consensus_score; max_score += 25
        factors.append({'name': 'Indikatör Konsensüsü', 'value': sf(consensus), 'score': sf(consensus_score), 'max': 25})

        # 2. Piyasa Rejimi Uyumu (agirlik: 15%)
        regime = calc_market_regime()
        regime_type = regime.get('regime', 'unknown')
        if signal_type == 'buy' and regime_type in REGIMES_BULLISH:
            regime_score = 15
        elif signal_type == 'sell' and regime_type in REGIMES_BEARISH:
            regime_score = 15
        elif regime_type == 'sideways':
            regime_score = 7.5
        elif (signal_type == 'buy' and regime_type in REGIMES_BEARISH) or \
             (signal_type == 'sell' and regime_type in REGIMES_BULLISH):
            regime_score = 3
        else:
            regime_score = 10
        score += regime_score; max_score += 15
        factors.append({'name': 'Piyasa Rejimi Uyumu', 'value': regime_type, 'score': sf(regime_score), 'max': 15})

        # 3. Hacim Teyidi (agirlik: 15%)
        vol_ratio = 1.0
        vol_score = 5
        if n >= 20:
            vol_avg = float(np.mean(v[-20:]))
            vol_recent = float(np.mean(v[-3:]))
            vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 1
            if vol_ratio > 1.2:
                price_direction = 'up' if c[-1] > c[-3] else 'down'
                vol_score = 15 if (signal_type == 'buy') == (price_direction == 'up') else 5
            elif vol_ratio > 1.0:
                vol_score = 10
        score += vol_score; max_score += 15
        factors.append({'name': 'Hacim Teyidi', 'value': sf(vol_ratio), 'score': sf(vol_score), 'max': 15})

        # 4. Trend Uyumu (agirlik: 15%)
        trend_score = 0
        if n >= 50:
            s = pd.Series(c)
            ema20 = float(s.ewm(span=20).mean().iloc[-1])
            ema50 = float(s.ewm(span=50).mean().iloc[-1])
            cp = float(c[-1])
            if signal_type == 'buy':
                if cp > ema20 > ema50: trend_score = 15
                elif cp > ema20: trend_score = 10
                elif cp > ema50: trend_score = 7
                else: trend_score = 3
            else:
                if cp < ema20 < ema50: trend_score = 15
                elif cp < ema20: trend_score = 10
                elif cp < ema50: trend_score = 7
                else: trend_score = 3
        score += trend_score; max_score += 15
        factors.append({'name': 'Trend Uyumu', 'value': 'EMA20/50', 'score': sf(trend_score), 'max': 15})

        # 5. Mum Formasyon Teyidi (agirlik: 10%)
        o = hist['Open'].values.astype(float) if 'Open' in hist.columns else c.copy()
        candles = calc_candlestick_patterns(o, h, l, c)
        matching = [p for p in candles.get('patterns', [])
                    if (p['type'] == 'bullish' and signal_type == 'buy') or
                       (p['type'] == 'bearish' and signal_type == 'sell')]
        candle_score = min(max(p.get('strength', 0) for p in matching) * 2, 10) if matching else 0
        score += candle_score; max_score += 10
        factors.append({'name': 'Mum Formasyonu', 'value': matching[0]['name'] if matching else 'Yok', 'score': sf(candle_score), 'max': 10})

        # 6. Destek/Direnc Yakinligi (agirlik: 10%)
        sr_score = 0
        cp = float(c[-1])
        sr = calc_support_resistance(hist)
        if cp > 0:
            if signal_type == 'buy' and sr.get('supports'):
                dist = abs(cp - sr['supports'][0]) / cp * 100
                sr_score = 10 if dist < 2 else (7 if dist < 5 else 3)
            elif signal_type == 'sell' and sr.get('resistances'):
                dist = abs(sr['resistances'][0] - cp) / cp * 100
                sr_score = 10 if dist < 2 else (7 if dist < 5 else 3)
        score += sr_score; max_score += 10
        factors.append({'name': 'Destek/Direnc', 'value': 'Yakin' if sr_score >= 7 else 'Uzak', 'score': sf(sr_score), 'max': 10})

        # 7. MTF Uyumu (agirlik: 10%, max 10 ile clamp — confidence >%100 olmasın)
        bt_score = 5
        mtf_label = 'N/A'
        try:
            mtf_res = calc_mtf_signal(hist)
            mtf_dir = mtf_res.get('mtfDirection', 'neutral')
            mtf_sc = mtf_res.get('mtfScore', 0)
            if (signal_type == 'buy' and mtf_dir == 'buy') or (signal_type == 'sell' and mtf_dir == 'sell'):
                bt_score = min(5 + mtf_sc * 2.5, 10)
            elif mtf_dir == 'neutral':
                bt_score = 5
            else:
                bt_score = 2
            mtf_label = mtf_res.get('mtfAlignment', 'N/A')
        except Exception:
            pass
        score += bt_score; max_score += 10
        factors.append({'name': 'MTF Uyumu', 'value': mtf_label, 'score': sf(bt_score), 'max': 10})

        # 8. Haber Sentiment modifier — confidence'a ±5 etki (calc_recommendation ile uyumlu)
        sent_mod = 0.0
        sent_label = 'Yok'
        if symbol:
            try:
                from news_sentiment import get_sentiment_score_for_signal
                raw_sent = get_sentiment_score_for_signal(symbol)
                if raw_sent is not None:
                    _rs = float(raw_sent)
                    if signal_type == 'sell':
                        _rs = -_rs
                    sent_mod = max(-5.0, min(5.0, _rs * 10.0))
                    sent_label = f'{_rs:+.2f}'
            except Exception:
                pass
        factors.append({'name': 'Haber Sentiment', 'value': sent_label, 'score': sf(sent_mod), 'max': 5})

        _conf_raw = (score / max_score * 100) if max_score > 0 else 50
        conf_val = max(0.0, min(100.0, _conf_raw + sent_mod))
        confidence = sf(conf_val)
        grade = 'A' if conf_val >= 80 else ('B' if conf_val >= 65 else ('C' if conf_val >= 50 else ('D' if conf_val >= 35 else 'F')))
        return {'confidence': confidence, 'grade': grade, 'score': sf(score), 'maxScore': sf(max_score), 'factors': factors}
    except Exception as e:
        print(f"  [ML-CONF] Hata: {e}")
        return {'confidence': 50, 'grade': 'C', 'factors': [], 'error': str(e)}
