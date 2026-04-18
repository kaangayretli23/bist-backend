"""
BIST Pro - Signal Core Module
calc_recommendation
check_signal_alerts, calc_ml_confidence → signals_confidence.py
"""
import numpy as np
import time, traceback, threading
from config import sf, si, _lock, _stock_cache, _index_cache, _cget, _cget_hist, _get_stocks, BIST100_STOCKS, SECTOR_MAP
from indicators import (
    calc_rsi, calc_macd, calc_bollinger, calc_stochastic, calc_adx,
    calc_support_resistance, calc_fibonacci, calc_dynamic_thresholds,
    calc_candlestick_patterns, calc_divergence, calc_mtf_signal,
    calc_ichimoku, calc_psar,
)
from signals_market import calc_market_regime, REGIMES_BULLISH, REGIMES_BEARISH

# İşlem stiline göre indikatör parametreleri
_TF_PARAMS = {
    'daily':   {'days': 1,   'rsi': 9,  'bb': 10, 'stoch': 9,  'adx': 7,  'atr': 7,  'macd': (6, 13, 5),   'sma': (5, 20, 50),    'vol': 5},
    'weekly':  {'days': 5,   'rsi': 14, 'bb': 20, 'stoch': 14, 'adx': 14, 'atr': 14, 'macd': (12, 26, 9),  'sma': (20, 50, 200),  'vol': 20},
    'monthly': {'days': 22,  'rsi': 21, 'bb': 20, 'stoch': 14, 'adx': 14, 'atr': 14, 'macd': (12, 26, 9),  'sma': (50, 200, 200), 'vol': 60},
    'yearly':  {'days': 252, 'rsi': 21, 'bb': 20, 'stoch': 14, 'adx': 14, 'atr': 14, 'macd': (12, 26, 9),  'sma': (50, 200, 200), 'vol': 60},
}

def _splice_live_close(hist, live_price):
    """Son bar'ın Close/High/Low'unu canlı fiyatla güncelle (intraday resample).
    Hist'in kopyası döner; orijinal mutate edilmez. live_price geçersizse hist'i olduğu gibi döner."""
    try:
        if hist is None or len(hist) == 0 or live_price is None:
            return hist
        lp = float(live_price)
        if lp <= 0:
            return hist
        h2 = hist.copy()
        last_idx = h2.index[-1]
        h2.at[last_idx, 'Close'] = lp
        try:
            cur_high = float(h2.at[last_idx, 'High'])
            if lp > cur_high:
                h2.at[last_idx, 'High'] = lp
        except Exception:
            h2.at[last_idx, 'High'] = lp
        try:
            cur_low = float(h2.at[last_idx, 'Low'])
            if lp < cur_low:
                h2.at[last_idx, 'Low'] = lp
        except Exception:
            h2.at[last_idx, 'Low'] = lp
        return h2
    except Exception:
        return hist


def calc_recommendation(hist, indicators, symbol=None):
    """Haftalik/Aylik/Yillik al-sat onerisi - guclendirilmis analiz + detayli reason"""
    try:
        c=hist['Close'].values.astype(float)
        h=hist['High'].values.astype(float)
        l=hist['Low'].values.astype(float)
        v=hist['Volume'].values.astype(float)
        o=hist['Open'].values.astype(float) if 'Open' in hist.columns else c.copy()
        # NaN temizligi: Close ile doldur
        h=np.where(np.isnan(h), c, h)
        l=np.where(np.isnan(l), c, l)
        v=np.where(np.isnan(v), 0, v)
        o=np.where(np.isnan(o), c, o)
        n=len(c)
        cur=float(c[-1])
        recommendations={}

        # Destek/direnc hesapla (tum periyotlar icin ortak)
        try:
            sr = calc_support_resistance(hist)
        except Exception:
            sr = {'supports': [], 'resistances': [], 'current': 0}
        supports = sr.get('supports', [])
        resistances = sr.get('resistances', [])
        try:
            fib = calc_fibonacci(hist)
        except Exception:
            fib = {'levels': {}}
        fib_sup = fib.get('nearestSupport')
        fib_res = fib.get('nearestResistance')

        # Dinamik esikler
        try:
            dyn = calc_dynamic_thresholds(c, h, l, v) if n >= 60 else {'rsi_oversold': 30, 'rsi_overbought': 70}
        except Exception:
            dyn = {'rsi_oversold': 30, 'rsi_overbought': 70}
        dyn_oversold = float(dyn.get('rsi_oversold', 30))
        dyn_overbought = float(dyn.get('rsi_overbought', 70))

        # Mum formasyonlari
        try:
            candle_data = calc_candlestick_patterns(o, h, l, c) if n >= 5 else {'patterns': [], 'signal': 'neutral'}
        except Exception:
            candle_data = {'patterns': [], 'signal': 'neutral'}

        # Piyasa rejimi — calc_market_regime signals_market'tan module-level import edildi
        try:
            regime = calc_market_regime()
        except Exception:
            regime = {'regime': 'unknown', 'description': ''}
        regime_type = regime.get('regime', 'unknown')

        # Diverjans hesapla (tum periyotlar icin ortak)
        try:
            div_data = calc_divergence(hist)
            div_summary = div_data.get('summary', {})
            div_signal = div_summary.get('signal', 'neutral')
            div_has_recent = div_summary.get('hasRecent', False)
            div_recent = div_data.get('recentDivergences', [])
        except Exception:
            div_signal = 'neutral'; div_has_recent = False; div_recent = []

        # MTF sinyal hesapla (tum periyotlar icin ortak)
        try:
            mtf_data = calc_mtf_signal(hist)
            mtf_direction = mtf_data.get('mtfDirection', 'neutral')
            mtf_score_val = mtf_data.get('mtfScore', 0)
            mtf_strength = mtf_data.get('mtfStrength', 'Uyumsuz')
        except Exception:
            mtf_direction = 'neutral'; mtf_score_val = 0; mtf_strength = 'Uyumsuz'; mtf_data = {}

        for label in ['daily', 'weekly', 'monthly', 'yearly']:
            p = _TF_PARAMS[label]; days = p['days']
            if n<days+14: recommendations[label]={'action':'neutral','confidence':0,'reasons':[],'score':0,'strategy':'Yeterli veri yok','reason':'Yeterli veri yok','indicatorBreakdown':{}}; continue

            sl=slice(-days,None)
            sc=c[sl]; sh=h[sl]; slow=l[sl]; sv=v[sl]

            try:
                bb = calc_bollinger(c, cur, period=p['bb'])
            except Exception:
                bb = {'upper': 0, 'lower': 0, 'middle': 0}
            bb_upper = bb.get('upper', 0)
            bb_lower = bb.get('lower', 0)
            bb_middle = bb.get('middle', 0)

            score=0; reasons=[]; strategy_parts=[]
            buy_indicators = 0; sell_indicators = 0; total_indicators = 0

            # 1. Trend (SMA) - tek bir gösterge sayılır (3 alt-check birleşik, consensus'u şişirmesin)
            ps, pm, pl = p['sma']
            sma_s = np.mean(c[-ps:]) if n >= ps else c[-1]
            sma_m = np.mean(c[-pm:]) if n >= pm else sma_s
            sma_l = np.mean(c[-pl:]) if n >= pl else sma_m
            sma_buy_sub = 0; sma_sell_sub = 0
            if cur > sma_s:
                score += 1; sma_buy_sub += 1
                reasons.append(f'Fiyat ({sf(cur)}) SMA{ps} ({sf(sma_s)}) uzerinde')
            else:
                score -= 1; sma_sell_sub += 1
                reasons.append(f'Fiyat ({sf(cur)}) SMA{ps} ({sf(sma_s)}) altinda')
            if sma_s > sma_m:
                score += 1; sma_buy_sub += 1
                reasons.append(f'SMA{ps} ({sf(sma_s)}) > SMA{pm} ({sf(sma_m)}) → Yukari trend')
            else:
                score -= 1; sma_sell_sub += 1
                reasons.append(f'SMA{ps} ({sf(sma_s)}) < SMA{pm} ({sf(sma_m)}) → Asagi trend')
            if n >= pl:
                if cur > sma_l:
                    score += 0.5; sma_buy_sub += 1
                    reasons.append(f'Fiyat SMA{pl} ({sf(sma_l)}) uzerinde → Uzun vadeli boga')
                else:
                    score -= 0.5; sma_sell_sub += 1
                    reasons.append(f'Fiyat SMA{pl} ({sf(sma_l)}) altinda → Uzun vadeli ayi')
            total_indicators += 1
            if sma_buy_sub > sma_sell_sub:
                buy_indicators += 1
            elif sma_sell_sub > sma_buy_sub:
                sell_indicators += 1

            # 2. RSI (dinamik eşiklere göre bucket sınırları — süreksizlik yok)
            rsi_val=calc_rsi(c, period=p['rsi'])
            rsi_v = rsi_val.get('value', 50)
            total_indicators += 1
            _rsi_lo_mid = (dyn_oversold + 50) / 2
            _rsi_hi_mid = (dyn_overbought + 50) / 2
            if rsi_v < dyn_oversold:
                score+=2; buy_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Asiri satim bolgesi (<{sf(dyn_oversold)}) → Guclu alis firsati')
            elif rsi_v < _rsi_lo_mid:
                score+=1; buy_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Zayif bolge (<{sf(_rsi_lo_mid)}) → Toparlanma bekleniyor')
            elif rsi_v > dyn_overbought:
                score-=2; sell_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Asiri alim bolgesi (>{sf(dyn_overbought)}) → Kar realizasyonu bekleniyor')
            elif rsi_v > _rsi_hi_mid:
                score-=0.5; sell_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Guclu bolge (>{sf(_rsi_hi_mid)})')
            elif rsi_v >= 50:
                score+=0.5; buy_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Notr-pozitif')
            else:
                score-=0.5; sell_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Notr-negatif')

            # 3. MACD
            macd=calc_macd(c, *p['macd'])
            macd_type = macd.get('signalType', 'neutral')
            macd_hist = macd.get('histogram', 0)
            total_indicators += 1
            if macd_type=='buy':
                score+=1.5; buy_indicators+=1
                reasons.append(f'MACD alis sinyali (histogram: {macd_hist})')
            elif macd_type=='sell':
                score-=1.5; sell_indicators+=1
                reasons.append(f'MACD satis sinyali (histogram: {macd_hist})')

            # 4. Bollinger — sadece band dışı sinyal consensus'a sayılır; orta bant sadece bilgi
            if bb_lower > 0 and cur < bb_lower:
                total_indicators += 1
                score+=1; buy_indicators+=1
                reasons.append(f'Fiyat ({sf(cur)}) alt Bollinger bandinin ({sf(bb_lower)}) altinda → Toparlanma bekleniyor')
            elif bb_upper > 0 and cur > bb_upper:
                total_indicators += 1
                score-=1; sell_indicators+=1
                reasons.append(f'Fiyat ({sf(cur)}) ust Bollinger bandinin ({sf(bb_upper)}) uzerinde → Geri cekilme bekleniyor')
            elif bb_middle > 0:
                # Orta bant konumu — skora/consensus'a etkisi yok, sadece bilgi amaçlı
                if cur > bb_middle:
                    reasons.append(f'Fiyat Bollinger orta bant ({sf(bb_middle)}) uzerinde')
                else:
                    reasons.append(f'Fiyat Bollinger orta bant ({sf(bb_middle)}) altinda')

            # 5. Hacim trendi — son kısa pencere vs daha önceki uzun pencere (overlap yok)
            _vol_recent_n = min(5, len(sv))
            _vol_base_n   = max(p['vol'], _vol_recent_n + 5)
            if len(sv) > _vol_recent_n + 5:
                vol_recent = float(np.mean(sv[-_vol_recent_n:]))
                # Taban: son N bar öncesindeki daha geniş pencere
                _base_end = len(sv) - _vol_recent_n
                _base_start = max(0, _base_end - _vol_base_n)
                _base_slice = sv[_base_start:_base_end]
                vol_avg = float(np.mean(_base_slice)) if len(_base_slice) > 0 else vol_recent
                vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 1
                total_indicators += 1
                if vol_ratio > 1.2:
                    vol_pts = 1.5 if vol_ratio > 2.0 else 1.0
                    if c[-1]>c[-5]:
                        score+=vol_pts; buy_indicators+=1
                        reasons.append(f'Hacim ortalamanin {sf(vol_ratio)}x uzerinde + yukari hareket → Guclu alis')
                    else:
                        score-=vol_pts; sell_indicators+=1
                        reasons.append(f'Hacim ortalamanin {sf(vol_ratio)}x uzerinde + dusus → Guclu satis baskisi')
                elif vol_ratio < 0.5:
                    reasons.append(f'Hacim ortalamanin altinda ({sf(vol_ratio)}x) → Dusuk ilgi, sinyal gucsuslesiyor')

            # 6. Momentum (periyoda gore)
            if len(sc)>=days:
                period_return=sf(((c[-1]-sc[0])/sc[0])*100)
                total_indicators += 1
                if period_return>10: score+=1.5; buy_indicators+=1; reasons.append(f'{label} getiri: %{period_return} (guclu yukselis)')
                elif period_return>5: score+=1; buy_indicators+=1; reasons.append(f'{label} getiri: %{period_return} (pozitif)')
                elif period_return<-10: score-=1.5; sell_indicators+=1; reasons.append(f'{label} getiri: %{period_return} (sert dusus)')
                elif period_return<-5: score-=1; sell_indicators+=1; reasons.append(f'{label} getiri: %{period_return} (negatif)')

            # 7. Stochastic
            stoch=calc_stochastic(c,h,l, period=p['stoch'])
            stoch_k = stoch.get('k', 50)
            total_indicators += 1
            if stoch_k<20: score+=1; buy_indicators+=1; reasons.append(f'Stochastic K={sf(stoch_k)}: Asiri satim bolgesi')
            elif stoch_k>80: score-=1; sell_indicators+=1; reasons.append(f'Stochastic K={sf(stoch_k)}: Asiri alim bolgesi')

            # 8. ADX - Trend gucu (arttirilmis agirlik: ADX gucune gore 0.5-1.5 puan)
            adx_data = calc_adx(h, l, c, period=p['adx'])
            adx_val = adx_data.get('value', 25)
            total_indicators += 1
            if adx_val > 25:
                trend_dir = 'yukari' if adx_data.get('plusDI', 0) > adx_data.get('minusDI', 0) else 'asagi'
                adx_pts = 1.5 if adx_val > 40 else (1.0 if adx_val > 30 else 0.5)
                reasons.append(f'ADX={sf(adx_val)}: Guclu {trend_dir} trend (agirlik: {adx_pts})')
                if trend_dir == 'yukari': score += adx_pts; buy_indicators += 1
                else: score -= adx_pts; sell_indicators += 1
            else:
                reasons.append(f'ADX={sf(adx_val)}: Zayif trend (<25), yatay piyasa - sinyaller zayifliyor')

            # 9. Ichimoku (eger yeterli veri varsa)
            if n >= 52:
                ichi = calc_ichimoku(c, h, l)
                total_indicators += 1
                if ichi.get('signal') == 'buy':
                    score += 1; buy_indicators += 1
                    reasons.append(f'Ichimoku alis sinyali (fiyat bulutun uzerinde)')
                elif ichi.get('signal') == 'sell':
                    score -= 1; sell_indicators += 1
                    reasons.append(f'Ichimoku satis sinyali (fiyat bulutun altinda)')

            # 10. Parabolic SAR
            if n >= 5:
                psar = calc_psar(c, h, l)
                total_indicators += 1
                if psar.get('signal') == 'buy':
                    score += 0.5; buy_indicators += 1
                    reasons.append(f'Parabolic SAR yukari trend (SAR={sf(psar.get("value", 0))})')
                elif psar.get('signal') == 'sell':
                    score -= 0.5; sell_indicators += 1
                    reasons.append(f'Parabolic SAR asagi trend (SAR={sf(psar.get("value", 0))})')

            # 11. Mum formasyonlari (p değişkeni dış döngüde _TF_PARAMS; shadowing önleme)
            for pat in candle_data.get('patterns', []):
                if pat.get('strength', 0) >= 3:
                    total_indicators += 1
                    if pat['type'] == 'bullish':
                        score += 0.5 * (pat['strength'] / 5)
                        buy_indicators += 1
                        reasons.append(f'Mum: {pat["name"]} → {pat["description"][:60]}')
                    elif pat['type'] == 'bearish':
                        score -= 0.5 * (pat['strength'] / 5)
                        sell_indicators += 1
                        reasons.append(f'Mum: {pat["name"]} → {pat["description"][:60]}')

            # 12. Diverjans sinyali (±2.0 puan - guclu ve nadir sinyal)
            if n >= 50:
                total_indicators += 1
                if div_signal == 'buy':
                    div_pts = 2.0 if div_has_recent else 1.0
                    score += div_pts; buy_indicators += 1
                    recent_labels = [d['label'] for d in div_recent if d['signal'] == 'buy'][:2]
                    reasons.append(f'Boga diverjans{"i (son 20 bar icinde)" if div_has_recent else ""}: {", ".join(recent_labels) if recent_labels else "RSI/MACD uyumsuzlugu"}')
                elif div_signal == 'sell':
                    div_pts = 2.0 if div_has_recent else 1.0
                    score -= div_pts; sell_indicators += 1
                    recent_labels = [d['label'] for d in div_recent if d['signal'] == 'sell'][:2]
                    reasons.append(f'Ayi diverjans{"i (son 20 bar icinde)" if div_has_recent else ""}: {", ".join(recent_labels) if recent_labels else "RSI/MACD uyumsuzlugu"}')

            # 13. MTF (Coklu Zaman Dilimi) uyum sinyali (±1.5 puan)
            if mtf_strength != 'Uyumsuz':
                total_indicators += 1
                if mtf_direction == 'buy':
                    mtf_pts = 1.5 if mtf_score_val == 3 else 1.0
                    score += mtf_pts; buy_indicators += 1
                    reasons.append(f'MTF uyumu: {mtf_data.get("description", "")} → {mtf_strength} alis')
                elif mtf_direction == 'sell':
                    mtf_pts = 1.5 if mtf_score_val == 3 else 1.0
                    score -= mtf_pts; sell_indicators += 1
                    reasons.append(f'MTF uyumu: {mtf_data.get("description", "")} → {mtf_strength} satis')

            # 14. Piyasa rejimi etkisi (final clamp 394'te score'u ±14'e sınırlar; sentiment muted olmasın)
            if regime_type in REGIMES_BULLISH and score > 0:
                score = score * 1.15
                reasons.append(f'Piyasa rejimi: {regime.get("description", "")} → Alis sinyali gucleniyor')
            elif regime_type in REGIMES_BEARISH and score < 0:
                score = score * 1.15
                reasons.append(f'Piyasa rejimi: {regime.get("description", "")} → Satis sinyali gucleniyor')
            elif regime_type in REGIMES_BEARISH and score > 0:
                score = score * 0.85
                reasons.append(f'Piyasa rejimi: {regime.get("description", "")} → Alis sinyali zayifliyor (ayi piyasasi)')

            # 15. Destek/Direnc bazli yorumlar
            if supports:
                nearest_sup = supports[0]
                sup_dist = sf(((cur - nearest_sup) / nearest_sup) * 100) if nearest_sup > 0 else sf(0)
                if float(sup_dist) < 2:
                    score += 1
                    reasons.append(f'Fiyat destege ({sf(nearest_sup)}) cok yakin (%{sup_dist}) → Destek bolgesi')
                    strategy_parts.append(f'{sf(nearest_sup)} TL desteginden alis yapilabilir')
                elif float(sup_dist) < 5:
                    strategy_parts.append(f'{sf(nearest_sup)} TL destegine yaklasirsa alis firsati')

            if resistances and cur > 0:
                nearest_res = resistances[0]
                res_dist = sf(((nearest_res - cur) / cur) * 100)
                if float(res_dist) < 2:
                    score -= 1
                    reasons.append(f'Fiyat dirence ({sf(nearest_res)}) cok yakin (%{res_dist}) → Satis baskisi')
                    strategy_parts.append(f'{sf(nearest_res)} TL direncinde satis/kar realizasyonu')
                elif float(res_dist) < 5:
                    strategy_parts.append(f'{sf(nearest_res)} TL direncini kirarsa alis guclenir')

            # 16. Fibonacci bazli yorumlar
            if fib_sup and fib_sup.get('price'):
                fib_sup_dist = sf(((cur - fib_sup['price']) / cur) * 100)
                if float(fib_sup_dist) < 3:
                    strategy_parts.append(f"Fibonacci {fib_sup['level']} destegi ({fib_sup['price']} TL) yakininda")
            if fib_res and fib_res.get('price'):
                fib_res_dist = sf(((fib_res['price'] - cur) / cur) * 100)
                if float(fib_res_dist) < 3:
                    strategy_parts.append(f"Fibonacci {fib_res['level']} direnci ({fib_res['price']} TL) yakininda")

            # Haber sentiment modifier (opsiyonel): -0.5..+0.5 → ±1.5 puan
            sentiment_bonus = 0.0
            if symbol:
                try:
                    from news_sentiment import get_sentiment_score_for_signal
                    raw_sent = get_sentiment_score_for_signal(symbol)
                    if raw_sent is not None:
                        sentiment_bonus = max(-1.5, min(1.5, float(raw_sent) * 3.0))
                        if sentiment_bonus > 0.3:
                            reasons.append(f'Pozitif haber sentiment (+{sf(sentiment_bonus)})')
                        elif sentiment_bonus < -0.3:
                            reasons.append(f'Negatif haber sentiment ({sf(sentiment_bonus)})')
                        score += sentiment_bonus
                except Exception:
                    pass

            # Sonuc
            max_score = 14.0
            score = max(-14.0, min(14.0, score))  # skor siniri
            conf = min(abs(score) / max_score * 100, 100)

            # Confluence filtresi: AL/SAT icin en az %40 gosterge uyumu gerekli
            consensus_pct = (buy_indicators / total_indicators * 100) if total_indicators > 0 else 50
            sell_consensus_pct = (sell_indicators / total_indicators * 100) if total_indicators > 0 else 50

            if score >= 7 and consensus_pct >= 55: action = 'GÜÇLÜ AL'
            elif score >= 3 and consensus_pct >= 40: action = 'AL'
            elif score >= 1.5: action = 'TUTUN/AL'
            elif score <= -7 and sell_consensus_pct >= 55: action = 'GÜÇLÜ SAT'
            elif score <= -3 and sell_consensus_pct >= 40: action = 'SAT'
            elif score <= -1.5: action = 'TUTUN/SAT'
            else: action = 'NOTR'

            # KISA OZET REASON (tek satirlik aciklama)
            if action == 'AL':
                top_buy = [r for r in reasons if any(k in r.lower() for k in ['alis','yukari','pozitif','topar','destek','asiri satim','uzerinde'])][:3]
                reason_summary = f"AL: {buy_indicators}/{total_indicators} gosterge alis yonunde. " + (top_buy[0] if top_buy else reasons[0] if reasons else '')
            elif action == 'SAT':
                top_sell = [r for r in reasons if any(k in r.lower() for k in ['satis','asagi','negatif','dusus','direnc','asiri alim','altinda'])][:3]
                reason_summary = f"SAT: {sell_indicators}/{total_indicators} gosterge satis yonunde. " + (top_sell[0] if top_sell else reasons[0] if reasons else '')
            elif action == 'TUTUN/AL':
                reason_summary = f"ZAYIF ALIS: {buy_indicators}/{total_indicators} gosterge alis yonunde. Destek bolgesi bekleniyor."
            elif action == 'TUTUN/SAT':
                reason_summary = f"ZAYIF SATIS: {sell_indicators}/{total_indicators} gosterge satis yonunde. Direnc bolgesi bekleniyor."
            else:
                reason_summary = f"NOTR: {buy_indicators} alis vs {sell_indicators} satis sinyali. Belirgin yon yok."

            # Strateji olustur
            if not strategy_parts:
                if action == 'AL':
                    strategy_parts.append('Teknik gostergeler alis yonunde')
                    if supports: strategy_parts.append(f'Stop-loss: {sf(supports[0])} TL altinda')
                    if resistances: strategy_parts.append(f'Hedef: {sf(resistances[0])} TL')
                elif action == 'SAT':
                    strategy_parts.append('Teknik gostergeler satis yonunde')
                    if resistances: strategy_parts.append(f'{sf(resistances[0])} TL direnci asagi sari')
                    if supports: strategy_parts.append(f'{sf(supports[0])} TL destegi kirilirsa satis guclenir')
                else:
                    strategy_parts.append('Belirgin bir sinyal yok, bekle-gor stratejisi')

            recommendations[label]={
                'action':action,'score':sf(score),'confidence':sf(conf),
                'reasons':reasons[:10],
                'reason': reason_summary,
                'strategy': ' | '.join(strategy_parts[:4]),
                'indicatorBreakdown': {
                    'buy': buy_indicators,
                    'sell': sell_indicators,
                    'total': total_indicators,
                    'consensus': sf(buy_indicators / total_indicators * 100) if total_indicators > 0 else 50,
                },
                'keyLevels': {
                    'supports': supports[:3],
                    'resistances': resistances[:3],
                    f'sma{ps}': sf(sma_s),
                    f'sma{pm}': sf(sma_m),
                    f'sma{pl}': sf(sma_l) if n >= pl else None,
                    'bollingerUpper': sf(bb_upper),
                    'bollingerLower': sf(bb_lower),
                },
                'dynamicRSI': {'oversold': sf(dyn_oversold), 'overbought': sf(dyn_overbought), 'current': sf(rsi_v)},
            }

        return recommendations
    except Exception as e:
        print(f"  [REC] Hata: {e}")
        return {'weekly':{'action':'neutral','confidence':0,'reasons':[],'strategy':'','reason':'Hesaplama hatasi','indicatorBreakdown':{}},'monthly':{'action':'neutral','confidence':0,'reasons':[],'strategy':'','reason':'Hesaplama hatasi','indicatorBreakdown':{}},'yearly':{'action':'neutral','confidence':0,'reasons':[],'strategy':'','reason':'Hesaplama hatasi','indicatorBreakdown':{}}}

# check_signal_alerts ve calc_ml_confidence → signals_confidence.py

