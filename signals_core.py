"""
BIST Pro - Signal Core Module
calc_recommendation, calc_fundamentals, calc_52w,
fetch_fundamental_data, check_signal_alerts, calc_ml_confidence
"""
import numpy as np
import time, traceback, threading
from config import sf, si, _lock, _stock_cache, _index_cache, _cget, _get_stocks, BIST100_STOCKS, SECTOR_MAP
from indicators import *
from indicators import _market_regime_cache, _resample_to_tf

# Fundamental data cache
_fundamental_cache = {}
_fundamental_cache_lock = threading.Lock()

def calc_recommendation(hist, indicators):
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

        # Bollinger bantlari
        try:
            bb = calc_bollinger(c, cur)
        except Exception:
            bb = {'upper': 0, 'lower': 0, 'middle': 0}
        bb_upper = bb.get('upper', 0)
        bb_lower = bb.get('lower', 0)
        bb_middle = bb.get('middle', 0)

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

        # Piyasa rejimi (lazy import - signals_backtest'te)
        try:
            from signals_backtest import calc_market_regime
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

        for label, days in [('weekly',5),('monthly',22),('yearly',252)]:
            if n<days+14: recommendations[label]={'action':'neutral','confidence':0,'reasons':[],'score':0,'strategy':'Yeterli veri yok','reason':'Yeterli veri yok','indicatorBreakdown':{}}; continue

            sl=slice(-days,None)
            sc=c[sl]; sh=h[sl]; slow=l[sl]; sv=v[sl]

            score=0; reasons=[]; strategy_parts=[]
            buy_indicators = 0; sell_indicators = 0; total_indicators = 0

            # 1. Trend (SMA) - Agirlik: 2 puan
            sma20=np.mean(c[-20:]) if n>=20 else c[-1]
            sma50=np.mean(c[-50:]) if n>=50 else sma20
            sma200=np.mean(c[-200:]) if n>=200 else sma50
            total_indicators += 1
            if cur>sma20:
                score+=1; buy_indicators+=1
                reasons.append(f'Fiyat ({sf(cur)}) SMA20 ({sf(sma20)}) uzerinde')
            else:
                score-=1; sell_indicators+=1
                reasons.append(f'Fiyat ({sf(cur)}) SMA20 ({sf(sma20)}) altinda')

            total_indicators += 1
            if sma20>sma50:
                score+=1; buy_indicators+=1
                reasons.append(f'SMA20 ({sf(sma20)}) > SMA50 ({sf(sma50)}) → Yukari trend')
            else:
                score-=1; sell_indicators+=1
                reasons.append(f'SMA20 ({sf(sma20)}) < SMA50 ({sf(sma50)}) → Asagi trend')

            # SMA200 bonus (uzun vadeli trend)
            if n >= 200:
                total_indicators += 1
                if cur > sma200:
                    score += 0.5; buy_indicators += 1
                    reasons.append(f'Fiyat SMA200 ({sf(sma200)}) uzerinde → Uzun vadeli boga')
                else:
                    score -= 0.5; sell_indicators += 1
                    reasons.append(f'Fiyat SMA200 ({sf(sma200)}) altinda → Uzun vadeli ayi')

            # 2. RSI (Dinamik esikler ile)
            rsi_val=calc_rsi(c)
            rsi_v = rsi_val.get('value', 50)
            total_indicators += 1
            if rsi_v < dyn_oversold:
                score+=2; buy_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Asiri satim bolgesi (<{sf(dyn_oversold)}) → Guclu alis firsati')
            elif rsi_v<40:
                score+=1; buy_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Zayif bolge → Toparlanma bekleniyor')
            elif rsi_v > dyn_overbought:
                score-=2; sell_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Asiri alim bolgesi (>{sf(dyn_overbought)}) → Kar realizasyonu bekleniyor')
            elif rsi_v>60:
                score-=0.5; sell_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Guclu bolge (60-70)')
            elif rsi_v>=50:
                score+=0.5; buy_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Notr-pozitif')
            else:
                score-=0.5; sell_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Notr-negatif')

            # 3. MACD
            macd=calc_macd(c)
            macd_type = macd.get('signalType', 'neutral')
            macd_hist = macd.get('histogram', 0)
            total_indicators += 1
            if macd_type=='buy':
                score+=1.5; buy_indicators+=1
                reasons.append(f'MACD alis sinyali (histogram: {macd_hist})')
            elif macd_type=='sell':
                score-=1.5; sell_indicators+=1
                reasons.append(f'MACD satis sinyali (histogram: {macd_hist})')

            # 4. Bollinger
            total_indicators += 1
            if bb_lower > 0 and cur < bb_lower:
                score+=1; buy_indicators+=1
                reasons.append(f'Fiyat ({sf(cur)}) alt Bollinger bandinin ({sf(bb_lower)}) altinda → Toparlanma bekleniyor')
            elif bb_upper > 0 and cur > bb_upper:
                score-=1; sell_indicators+=1
                reasons.append(f'Fiyat ({sf(cur)}) ust Bollinger bandinin ({sf(bb_upper)}) uzerinde → Geri cekilme bekleniyor')
            elif bb_middle > 0:
                if cur > bb_middle:
                    buy_indicators+=1
                    reasons.append(f'Fiyat Bollinger orta bant ({sf(bb_middle)}) uzerinde')
                else:
                    sell_indicators+=1
                    reasons.append(f'Fiyat Bollinger orta bant ({sf(bb_middle)}) altinda')

            # 5. Hacim trendi
            if len(sv)>5:
                vol_avg=np.mean(sv[-20:]) if len(sv)>=20 else np.mean(sv)
                vol_recent=np.mean(sv[-5:])
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
            stoch=calc_stochastic(c,h,l)
            stoch_k = stoch.get('k', 50)
            total_indicators += 1
            if stoch_k<20: score+=1; buy_indicators+=1; reasons.append(f'Stochastic K={sf(stoch_k)}: Asiri satim bolgesi')
            elif stoch_k>80: score-=1; sell_indicators+=1; reasons.append(f'Stochastic K={sf(stoch_k)}: Asiri alim bolgesi')

            # 8. ADX - Trend gucu (arttirilmis agirlik: ADX gucune gore 0.5-1.5 puan)
            adx_data = calc_adx(h, l, c)
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

            # 11. Mum formasyonlari
            for p in candle_data.get('patterns', []):
                if p.get('strength', 0) >= 3:
                    total_indicators += 1
                    if p['type'] == 'bullish':
                        score += 0.5 * (p['strength'] / 5)
                        buy_indicators += 1
                        reasons.append(f'Mum: {p["name"]} → {p["description"][:60]}')
                    elif p['type'] == 'bearish':
                        score -= 0.5 * (p['strength'] / 5)
                        sell_indicators += 1
                        reasons.append(f'Mum: {p["name"]} → {p["description"][:60]}')

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

            # 14. Piyasa rejimi etkisi (skor +-14 ile sinirlandirilir)
            if regime_type in ('strong_bull', 'bull') and score > 0:
                score = min(score * 1.15, 14.0)
                reasons.append(f'Piyasa rejimi: {regime.get("description", "")} → Alis sinyali gucleniyor')
            elif regime_type in ('strong_bear', 'bear') and score < 0:
                score = max(score * 1.15, -14.0)
                reasons.append(f'Piyasa rejimi: {regime.get("description", "")} → Satis sinyali gucleniyor')
            elif regime_type in ('strong_bear', 'bear') and score > 0:
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

            # Sonuc
            max_score = 14.0
            score = max(-14.0, min(14.0, score))  # skor siniri
            conf = min(abs(score) / max_score * 100, 100)

            # Confluence filtresi: AL/SAT icin en az %40 gosterge uyumu gerekli
            consensus_pct = (buy_indicators / total_indicators * 100) if total_indicators > 0 else 50
            sell_consensus_pct = (sell_indicators / total_indicators * 100) if total_indicators > 0 else 50

            if score >= 3 and consensus_pct >= 40: action = 'AL'
            elif score >= 1.5: action = 'TUTUN/AL'
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
                    'sma20': sf(sma20),
                    'sma50': sf(sma50),
                    'sma200': sf(sma200) if n >= 200 else None,
                    'bollingerUpper': sf(bb_upper),
                    'bollingerLower': sf(bb_lower),
                },
                'dynamicRSI': {'oversold': sf(dyn_oversold), 'overbought': sf(dyn_overbought), 'current': sf(rsi_v)},
            }

        return recommendations
    except Exception as e:
        print(f"  [REC] Hata: {e}")
        return {'weekly':{'action':'neutral','confidence':0,'reasons':[],'strategy':'','reason':'Hesaplama hatasi','indicatorBreakdown':{}},'monthly':{'action':'neutral','confidence':0,'reasons':[],'strategy':'','reason':'Hesaplama hatasi','indicatorBreakdown':{}},'yearly':{'action':'neutral','confidence':0,'reasons':[],'strategy':'','reason':'Hesaplama hatasi','indicatorBreakdown':{}}}

def calc_fundamentals(hist, symbol):
    """Temel verileri mevcut fiyat/hacim verisinden hesapla"""
    try:
        c = hist['Close'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        # NaN temizligi
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        v = np.where(np.isnan(v), 0, v)
        n = len(c)
        cur = float(c[-1])

        # Ortalama gunluk hacim (son 20 gun)
        avg_vol_20 = float(np.mean(v[-20:])) if n >= 20 else float(np.mean(v))
        avg_vol_60 = float(np.mean(v[-60:])) if n >= 60 else avg_vol_20

        # Volatilite (yillik)
        if n >= 20:
            daily_returns = np.diff(c[-60:]) / c[-60:-1] if n >= 60 else np.diff(c) / c[:-1]
            volatility = sf(float(np.std(daily_returns)) * (252 ** 0.5) * 100)
        else:
            volatility = 0

        # Beta hesapla (BIST100'e gore) - basit yaklasim: volatilite bazli
        beta = sf(volatility / 25) if volatility else 1.0  # BIST100 avg vol ~25%

        # Ortalama islem hacmi (TL)
        avg_turnover = sf(cur * avg_vol_20)

        # 1 aylik, 3 aylik, 6 aylik, 1 yillik getiri
        returns = {}
        for label, days in [('1ay', 22), ('3ay', 66), ('6ay', 132), ('1yil', 252)]:
            if n > days:
                ret = sf(((cur - float(c[-days])) / float(c[-days])) * 100)
                returns[label] = ret

        # Gunluk ortalama aralik (ATR benzeri) - NaN-safe
        if n >= 14:
            daily_range = [(float(h[i]) - float(l[i])) for i in range(-14, 0) if h[i] == h[i] and l[i] == l[i]]
            avg_daily_range = sf(np.mean(daily_range)) if daily_range else 0
            avg_daily_range_pct = sf(avg_daily_range / cur * 100) if cur > 0 else 0
        else:
            avg_daily_range = 0
            avg_daily_range_pct = 0

        # 52 haftalik high/low'dan uzaklik (NaN-safe)
        if n >= 252:
            hi52 = float(np.nanmax(h[-252:]))
            lo52 = float(np.nanmin(l[-252:]))
        else:
            hi52 = float(np.nanmax(h))
            lo52 = float(np.nanmin(l))
        # NaN kontrolu
        if hi52 != hi52: hi52 = cur  # NaN ise cur kullan
        if lo52 != lo52: lo52 = cur
        dist_from_high = sf(((cur - hi52) / hi52) * 100) if hi52 else 0
        dist_from_low = sf(((cur - lo52) / lo52) * 100) if lo52 else 0

        return {
            'avgVolume20': si(avg_vol_20),
            'avgVolume60': si(avg_vol_60),
            'avgTurnover': avg_turnover,
            'volatility': volatility,
            'beta': beta,
            'returns': returns,
            'avgDailyRange': avg_daily_range,
            'avgDailyRangePct': avg_daily_range_pct,
            'distFromHigh52w': dist_from_high,
            'distFromLow52w': dist_from_low,
        }
    except Exception as e:
        print(f"  [FUND] {symbol} hata: {e}")
        return {}

def calc_52w(hist):
    """52 hafta (veya mevcut veri) high/low hesapla - NaN-safe"""
    try:
        h=hist['High'].values.astype(float)
        l=hist['Low'].values.astype(float)
        c=float(hist['Close'].iloc[-1])
        hi52=sf(float(np.nanmax(h))); lo52=sf(float(np.nanmin(l)))
        # NaN fallback
        if hi52 == 0 and c > 0: hi52 = sf(c)
        if lo52 == 0 and c > 0: lo52 = sf(c)
        rng=hi52-lo52
        pos=sf((c-lo52)/rng*100 if rng>0 else 50)
        return {'high52w':hi52,'low52w':lo52,'currentPct':pos,'range':sf(rng)}
    except Exception: return {'high52w':0,'low52w':0,'currentPct':50,'range':0}


def fetch_fundamental_data(symbol):
    """Is Yatirim'dan temel analiz verilerini cek (F/K, PD/DD vs.)"""
    try:
        # Cache kontrol (1 saat)
        with _fundamental_cache_lock:
            cached = _fundamental_cache.get(symbol)
        if cached and time.time() - cached['ts'] < 3600:
            return cached['data']

        url = f"https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/MaliTablo?hession={symbol}&doession=2024&dession=4"
        headers = IS_YATIRIM_HEADERS.copy()

        try:
            resp = req_lib.get(url, headers=headers, timeout=10, verify=False)
            if resp.status_code == 200:
                data = resp.json()
                rows = data.get('value', [])
                if rows:
                    result = _parse_fundamental_data(rows, symbol)
                    with _fundamental_cache_lock:
                        _fundamental_cache[symbol] = {'data': result, 'ts': time.time()}
                    return result
        except Exception as e:
            print(f"  [FUNDAMENTAL] {symbol} Is Yatirim hata: {e}")

        # Fallback: yfinance info
        if YF_OK:
            try:
                tkr = yf.Ticker(f"{symbol}.IS")
                info = tkr.info or {}
                result = {
                    'pe': sf(info.get('trailingPE', 0)),
                    'forwardPE': sf(info.get('forwardPE', 0)),
                    'pb': sf(info.get('priceToBook', 0)),
                    'marketCap': info.get('marketCap', 0),
                    'dividendYield': sf(info.get('dividendYield', 0) * 100) if info.get('dividendYield') else 0,
                    'debtToEquity': sf(info.get('debtToEquity', 0)),
                    'roe': sf(info.get('returnOnEquity', 0) * 100) if info.get('returnOnEquity') else 0,
                    'roa': sf(info.get('returnOnAssets', 0) * 100) if info.get('returnOnAssets') else 0,
                    'profitMargin': sf(info.get('profitMargins', 0) * 100) if info.get('profitMargins') else 0,
                    'source': 'yfinance',
                }
                with _fundamental_cache_lock:
                    _fundamental_cache[symbol] = {'data': result, 'ts': time.time()}
                return result
            except Exception as e:
                print(f"  [FUNDAMENTAL] {symbol} yfinance hata: {e}")

        return {}
    except Exception as e:
        print(f"  [FUNDAMENTAL] {symbol}: {e}")
        return {}

def _parse_fundamental_data(rows, symbol):
    """Is Yatirim mali tablo verisini parse et"""
    try:
        result = {'source': 'isyatirim'}
        for row in rows:
            key = str(row.get('KALEM', '')).upper()
            val = row.get('DEGER', 0)
            if 'NET KAR' in key or 'NET DONEM' in key:
                result['netProfit'] = sf(float(val)) if val else 0
            elif 'SATIS' in key or 'GELIR' in key:
                result['revenue'] = sf(float(val)) if val else 0
            elif 'OZKAYN' in key:
                result['equity'] = sf(float(val)) if val else 0
        return result
    except Exception:
        return {'source': 'isyatirim'}


# =====================================================================
# FEATURE 7: ENHANCED TELEGRAM/EMAIL ALERT SYSTEM
# =====================================================================

def check_signal_alerts():
    """Sinyal bazli otomatik uyari kontrolu - enhanced"""
    stocks = _get_stocks()
    if not stocks:
        return []

    alerts = []
    regime = calc_market_regime()
    regime_str = regime.get('regime', 'unknown')

    for stock in stocks:
        sym = stock['code']
        try:
            hist = _cget_hist(f"{sym}_1y")
            if hist is None or len(hist) < 50:
                continue

            c = hist['Close'].values.astype(float)
            h = hist['High'].values.astype(float)
            l = hist['Low'].values.astype(float)
            o = hist['Open'].values.astype(float)
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

    # Strength'e gore sirala
    alerts.sort(key=lambda x: x.get('strength', 0), reverse=True)
    return alerts


# =====================================================================
# FEATURE 8: ML-BASED SIGNAL CONFIDENCE SCORING
# =====================================================================

def calc_ml_confidence(hist, indicators, recommendation_score, signal_type='buy'):
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
        if signal_type == 'buy':
            consensus = bc / total * 100
        else:
            consensus = sc / total * 100
        consensus_score = min(consensus / 100 * 25, 25)
        score += consensus_score
        max_score += 25
        factors.append({'name': 'Indikatör Konsensüsü', 'value': sf(consensus), 'score': sf(consensus_score), 'max': 25})

        # 2. Piyasa Rejimi Uyumu (agirlik: 15%)
        regime = calc_market_regime()
        regime_type = regime.get('regime', 'unknown')
        regime_score = 0
        if signal_type == 'buy' and regime_type in ('strong_bull', 'bull'):
            regime_score = 15
        elif signal_type == 'sell' and regime_type in ('strong_bear', 'bear'):
            regime_score = 15
        elif regime_type == 'sideways':
            regime_score = 7.5
        elif signal_type == 'buy' and regime_type in ('strong_bear', 'bear'):
            regime_score = 3
        elif signal_type == 'sell' and regime_type in ('strong_bull', 'bull'):
            regime_score = 3
        else:
            regime_score = 10
        score += regime_score
        max_score += 15
        factors.append({'name': 'Piyasa Rejimi Uyumu', 'value': regime_type, 'score': sf(regime_score), 'max': 15})

        # 3. Hacim Teyidi (agirlik: 15%)
        vol_score = 0
        if n >= 20:
            vol_avg = float(np.mean(v[-20:]))
            vol_recent = float(np.mean(v[-3:]))
            vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 1
            if vol_ratio > 1.2:
                # Hacim teyidi var
                price_direction = 'up' if c[-1] > c[-3] else 'down'
                if (signal_type == 'buy' and price_direction == 'up') or (signal_type == 'sell' and price_direction == 'down'):
                    vol_score = 15  # Hacim sinyal yonunu destekliyor
                else:
                    vol_score = 5  # Hacim ters yonde
            elif vol_ratio > 1.0:
                vol_score = 10  # Normal hacim
            else:
                vol_score = 5  # Dusuk hacim
        score += vol_score
        max_score += 15
        factors.append({'name': 'Hacim Teyidi', 'value': sf(vol_ratio) if n >= 20 else '-', 'score': sf(vol_score), 'max': 15})

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
        score += trend_score
        max_score += 15
        factors.append({'name': 'Trend Uyumu', 'value': 'EMA20/50', 'score': sf(trend_score), 'max': 15})

        # 5. Mum Formasyon Teyidi (agirlik: 10%)
        o = hist['Open'].values.astype(float)
        candles = calc_candlestick_patterns(o, h, l, c)
        candle_score = 0
        matching_patterns = [p for p in candles.get('patterns', [])
                            if (p['type'] == 'bullish' and signal_type == 'buy') or
                               (p['type'] == 'bearish' and signal_type == 'sell')]
        if matching_patterns:
            best_strength = max(p['strength'] for p in matching_patterns)
            candle_score = min(best_strength * 2, 10)
        score += candle_score
        max_score += 10
        factors.append({'name': 'Mum Formasyonu', 'value': matching_patterns[0]['name'] if matching_patterns else 'Yok', 'score': sf(candle_score), 'max': 10})

        # 6. Destek/Direnc Yakinligi (agirlik: 10%)
        sr_score = 0
        sr = calc_support_resistance(hist)
        cp = float(c[-1])
        if signal_type == 'buy' and sr.get('supports'):
            nearest_sup = sr['supports'][0]
            dist = abs(cp - nearest_sup) / cp * 100
            if dist < 2: sr_score = 10  # Desteğe çok yakin
            elif dist < 5: sr_score = 7
            else: sr_score = 3
        elif signal_type == 'sell' and sr.get('resistances'):
            nearest_res = sr['resistances'][0]
            dist = abs(nearest_res - cp) / cp * 100
            if dist < 2: sr_score = 10  # Dirence çok yakin
            elif dist < 5: sr_score = 7
            else: sr_score = 3
        score += sr_score
        max_score += 10
        factors.append({'name': 'Destek/Direnc', 'value': 'Yakin' if sr_score >= 7 else 'Uzak', 'score': sf(sr_score), 'max': 10})

        # 7. Coklu Zaman Dilimi (MTF) Uyumu (agirlik: 10%)
        # Backtest win rate yerine MTF kullaniyoruz - circular logic'i onler
        bt_score = 0
        mtf_label = 'N/A'
        try:
            mtf_res = calc_mtf_signal(hist)
            mtf_dir = mtf_res.get('mtfDirection', 'neutral')
            mtf_sc = mtf_res.get('mtfScore', 0)
            if signal_type == 'buy' and mtf_dir == 'buy':
                bt_score = 5 + mtf_sc * 2.5  # 2/3: 7.5, 3/3: 10
            elif signal_type == 'sell' and mtf_dir == 'sell':
                bt_score = 5 + mtf_sc * 2.5
            elif mtf_dir == 'neutral':
                bt_score = 5
            else:
                bt_score = 2  # Ters yon
            mtf_label = mtf_res.get('mtfAlignment', 'N/A')
        except Exception:
            bt_score = 5
        score += bt_score
        max_score += 10
        factors.append({'name': 'MTF Uyumu', 'value': mtf_label, 'score': sf(bt_score), 'max': 10})

        # Final confidence
        confidence = sf(score / max_score * 100) if max_score > 0 else 50

        # Grade
        conf_val = float(confidence)
        if conf_val >= 80: grade = 'A'
        elif conf_val >= 65: grade = 'B'
        elif conf_val >= 50: grade = 'C'
        elif conf_val >= 35: grade = 'D'
        else: grade = 'F'

        return {
            'confidence': confidence,
            'grade': grade,
            'score': sf(score),
            'maxScore': sf(max_score),
            'factors': factors,
        }
    except Exception as e:
        print(f"  [ML-CONF] Hata: {e}")
        return {'confidence': 50, 'grade': 'C', 'factors': [], 'error': str(e)}


# =====================================================================
# FEATURE 9: DETAYLI TRADE PLAN (Gunluk/Haftalik/Aylik giriş/çıkış)
# =====================================================================

