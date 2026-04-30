"""
BIST Pro - Market Regime & Sector Analysis
calc_market_regime, calc_sector_relative_strength
signals_backtest.py'dan ayrıştırıldı (600 satır kuralı).
"""
import numpy as np
import pandas as pd
import time

# config ve data_fetcher_raw bağımlılıkları fonksiyon içinde lazy import edilir
# (circular import sorununu önlemek için — Flask startup sırasında kısmi yükleme riski var)
from indicators import calc_rsi, calc_adx
from indicators import _market_regime_cache

# Piyasa rejimi sabitleri — tek kaynak (typo riski yok, tuple üyeliği yerine sabit)
REGIME_STRONG_BULL = 'strong_bull'
REGIME_BULL = 'bull'
REGIME_SIDEWAYS = 'sideways'
REGIME_BEAR = 'bear'
REGIME_STRONG_BEAR = 'strong_bear'
REGIME_UNKNOWN = 'unknown'
REGIMES_BULLISH = (REGIME_STRONG_BULL, REGIME_BULL)
REGIMES_BEARISH = (REGIME_STRONG_BEAR, REGIME_BEAR)
REGIMES_STRONG = (REGIME_STRONG_BULL, REGIME_STRONG_BEAR)


def calc_market_regime():
    """BIST100 trend durumunu analiz et: bull/bear/sideways"""
    try:
        # Lazy import — circular import veya kısmi yükleme sorununu önler
        from config import _cget_hist as _cgh, _cset as _cs, _hist_cache as _hc, _get_stocks as _gs, sf, SECTOR_MAP
        from data_fetcher_raw import _fetch_isyatirim_df as _fetch_isy
        from data_fetcher_raw import _fetch_yahoo_http_df as _fetch_yh

        # Cache: sadece gerçek sonuç varsa döndür (hata sonucu değil)
        cached = _market_regime_cache.get('regime')
        if cached and isinstance(cached, dict) and cached.get('regime') not in (None, 'unknown') \
                and time.time() - _market_regime_cache.get('ts', 0) < 300:
            return cached

        hist = _cgh("XU100_1y")
        if hist is None:
            try:
                xu_df = _fetch_isy("XU100", 365)
                if xu_df is not None and len(xu_df) >= 30:
                    _cs(_hc, "XU100_1y", xu_df)
                    hist = xu_df
                    print("[REGIME] XU100 verisi Is Yatirim'dan cekildi")
            except Exception as xe:
                print(f"[REGIME] XU100 Is Yatirim hatasi: {xe}")
        if hist is None:
            try:
                xu_df = _fetch_yh("XU100.IS", 365)
                if xu_df is not None and len(xu_df) >= 30:
                    _cs(_hc, "XU100_1y", xu_df)
                    hist = xu_df
                    print("[REGIME] XU100 verisi Yahoo'dan cekildi")
            except Exception as ye:
                print(f"[REGIME] XU100 Yahoo hatasi: {ye}")

        if hist is None:
            stocks = _gs()
            if stocks:
                advancing = sum(1 for s in stocks if s.get('changePct', 0) > 0)
                declining = sum(1 for s in stocks if s.get('changePct', 0) < 0)
                total = len(stocks)
                ratio = advancing / max(declining, 1)
                if ratio > 2:       regime_name, desc = 'strong_bull', 'Guclu Boga Piyasasi'
                elif ratio > 1.3:   regime_name, desc = 'bull', 'Boga Piyasasi'
                elif ratio < 0.5:   regime_name, desc = 'strong_bear', 'Guclu Ayi Piyasasi'
                elif ratio < 0.8:   regime_name, desc = 'bear', 'Ayi Piyasasi'
                else:               regime_name, desc = 'sideways', 'Yatay Piyasa'
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

        # cur: live XU100 önceliklidir — hist son barı piyasa açıkken Salı kapanışıdır.
        # SMA20/50/200 kıyaslamaları "fiyat SMA üzerinde" gibi → live kullanmadan
        # bugünkü hareket yansımaz, rejim hep dünden hesaplanır.
        cur = float(c[-1])
        try:
            from config import _cget as _cg, _index_cache as _ixc
            xu_live = _cg(_ixc, 'XU100')
            if xu_live and xu_live.get('value'):
                live_val = float(xu_live['value'])
                if live_val > 0:
                    cur = live_val
        except Exception:
            pass
        sma20  = float(np.mean(c[-20:]))  if n >= 20  else cur
        sma50  = float(np.mean(c[-50:]))  if n >= 50  else sma20
        sma200 = float(np.mean(c[-200:])) if n >= 200 else sma50
        s = pd.Series(c)
        ema20 = float(s.ewm(span=20).mean().iloc[-1])
        ema50 = float(s.ewm(span=50).mean().iloc[-1])

        h_arr = hist['High'].values.astype(float)
        l_arr = hist['Low'].values.astype(float)
        adx_data = calc_adx(h_arr, l_arr, c)
        adx_val   = float(adx_data.get('value', 25))
        plus_di   = float(adx_data.get('plusDI', 0))
        minus_di  = float(adx_data.get('minusDI', 0))
        rsi = calc_rsi(c).get('value', 50)

        ret_20d = ((cur - float(c[-20])) / float(c[-20])) * 100 if n >= 20 else 0
        ret_50d = ((cur - float(c[-50])) / float(c[-50])) * 100 if n >= 50 else 0

        if n >= 20:
            daily_returns = np.diff(c[-30:]) / c[-30:-1]
            volatility = float(np.std(daily_returns)) * (252 ** 0.5) * 100
        else:
            volatility = 25

        stocks = _gs()
        advancing = sum(1 for s in stocks if s.get('changePct', 0) > 0)
        declining = sum(1 for s in stocks if s.get('changePct', 0) < 0)
        breadth_ratio = advancing / max(declining, 1)

        score = 0
        reasons = []

        if cur > sma20:  score += 1;   reasons.append('Fiyat SMA20 uzerinde')
        else:            score -= 1;   reasons.append('Fiyat SMA20 altinda')
        if cur > sma50:  score += 1;   reasons.append('Fiyat SMA50 uzerinde')
        else:            score -= 1;   reasons.append('Fiyat SMA50 altinda')
        if n >= 200:
            if cur > sma200: score += 1.5; reasons.append('Fiyat SMA200 uzerinde (uzun vadeli boga)')
            else:            score -= 1.5; reasons.append('Fiyat SMA200 altinda (uzun vadeli ayi)')
        if sma20 > sma50: score += 1; reasons.append('SMA20 > SMA50 (yukari trend)')
        else:             score -= 1; reasons.append('SMA20 < SMA50 (asagi trend)')
        if ret_20d > 5:   score += 1; reasons.append(f'20 gunluk getiri: %{sf(ret_20d)} (guclu)')
        elif ret_20d > 0: score += 0.5
        elif ret_20d < -5: score -= 1; reasons.append(f'20 gunluk getiri: %{sf(ret_20d)} (zayif)')
        else:             score -= 0.5
        if adx_val > 25:
            if plus_di > minus_di: score += 1; reasons.append(f'ADX={sf(adx_val)}: Guclu yukari trend')
            else:                  score -= 1; reasons.append(f'ADX={sf(adx_val)}: Guclu asagi trend')
        if breadth_ratio > 1.5:   score += 0.5; reasons.append(f'Piyasa genisligi pozitif ({advancing}/{declining})')
        elif breadth_ratio < 0.7: score -= 0.5; reasons.append(f'Piyasa genisligi negatif ({advancing}/{declining})')

        if score >= 3:    regime, desc = 'strong_bull', 'Guclu Boga Piyasasi - Alis sinyalleri daha guvenilir'
        elif score >= 1:  regime, desc = 'bull',        'Boga Piyasasi - Genel yukari trend'
        elif score <= -3: regime, desc = 'strong_bear', 'Guclu Ayi Piyasasi - Satis sinyalleri daha guvenilir'
        elif score <= -1: regime, desc = 'bear',        'Ayi Piyasasi - Genel asagi trend'
        else:             regime, desc = 'sideways',    'Yatay Piyasa - Belirsizlik hakim, dikkatli olun'

        # Intraday override: XU100 gun ici sert dusus (>%2) → rejimi bir kademe asagi cek
        intraday_pct = None
        try:
            from config import _cget as _cg, _index_cache as _ixc, _stock_cache as _stc
            xu_live = _cg(_ixc, 'XU100') or _cg(_stc, 'XU100')
            if xu_live and 'changePct' in xu_live:
                intraday_pct = float(xu_live.get('changePct', 0))
        except Exception:
            intraday_pct = None

        if intraday_pct is not None and intraday_pct <= -2.0:
            reasons.insert(0, f'Gun ici XU100 sert dusus: %{sf(intraday_pct)} → rejim risk-off')
            downshift = {
                'strong_bull': ('bull',     'Gun ici sert dusus - Bogaya ragmen risk-off'),
                'bull':        ('sideways', 'Gun ici sert dusus - Belirsizlik, temkinli'),
                'sideways':    ('bear',     'Gun ici sert dusus - Ayi baskisi'),
                'bear':        ('strong_bear', 'Gun ici sert dusus - Guclu ayi'),
            }
            if regime in downshift:
                regime, desc = downshift[regime]

        if regime in REGIMES_BULLISH:
            buy_cm, sell_cm = 1.2, 0.8
        elif regime in REGIMES_BEARISH:
            buy_cm, sell_cm = 0.8, 1.2
        else:
            buy_cm, sell_cm = 1.0, 1.0

        result = {
            'regime': regime, 'score': sf(score),
            'strength': sf(min(abs(score) / 5 * 100, 100)),
            'description': desc, 'reasons': reasons[:6],
            'indicators': {
                'sma20': sf(sma20), 'sma50': sf(sma50), 'sma200': sf(sma200) if n >= 200 else None,
                'adx': sf(adx_val), 'rsi': sf(rsi),
                'ret20d': sf(ret_20d), 'ret50d': sf(ret_50d),
                'volatility': sf(volatility), 'breadthRatio': sf(breadth_ratio),
                'intradayPct': sf(intraday_pct) if intraday_pct is not None else None,
            },
            'confidence_multiplier': {'buy': buy_cm, 'sell': sell_cm},
        }
        _market_regime_cache['regime'] = result
        _market_regime_cache['ts'] = time.time()
        return result
    except Exception as e:
        print(f"[REGIME] Hata: {e}")
        return {'regime': 'unknown', 'strength': 0, 'description': str(e)}


def calc_sector_relative_strength():
    """Sektor bazli goreceli guc analizi"""
    try:
        from config import _get_stocks as _gs2, _cget_hist as _cgh2, sf, SECTOR_MAP
        stocks = _gs2()
        if not stocks:
            return {'sectors': []}

        stock_map = {s['code']: s for s in stocks}
        sector_results = []
        _DISPLAY = {
            'bankacilik': 'Bankacilik', 'havacilik': 'Havacilik',
            'otomotiv': 'Otomotiv', 'enerji': 'Enerji',
            'holding': 'Holding', 'perakende': 'Perakende',
            'teknoloji': 'Teknoloji', 'telekom': 'Telekom',
            'demir_celik': 'Demir Celik', 'gida': 'Gida',
            'insaat': 'Insaat', 'gayrimenkul': 'Gayrimenkul',
        }

        for sector_name, symbols in SECTOR_MAP.items():
            sector_stocks, returns_1d, returns_1w, returns_1m = [], [], [], []
            for sym in symbols:
                s = stock_map.get(sym)
                if not s:
                    continue
                hist = _cgh2(f"{sym}_1y")
                si = {'code': sym, 'name': s.get('name', sym), 'price': s['price'], 'changePct': s.get('changePct', 0)}
                if hist is not None and len(hist) >= 22:
                    c = hist['Close'].values.astype(float)
                    n = len(c)
                    si['ret1w'] = sf(((float(c[-1]) - float(c[-5]))  / float(c[-5]))  * 100) if n >= 5  else 0
                    si['ret1m'] = sf(((float(c[-1]) - float(c[-22])) / float(c[-22])) * 100) if n >= 22 else 0
                    si['ret3m'] = sf(((float(c[-1]) - float(c[-66])) / float(c[-66])) * 100) if n >= 66 else 0
                    rsi = calc_rsi(c)
                    si['rsi'] = rsi.get('value', 50)
                    si['rsiSignal'] = rsi.get('signal', 'neutral')
                    returns_1d.append(s.get('changePct', 0))
                    returns_1w.append(float(si.get('ret1w', 0)))
                    returns_1m.append(float(si.get('ret1m', 0)))
                sector_stocks.append(si)

            if not sector_stocks:
                continue
            avg_1d = sf(np.mean(returns_1d)) if returns_1d else 0
            avg_1w = sf(np.mean(returns_1w)) if returns_1w else 0
            avg_1m = sf(np.mean(returns_1m)) if returns_1m else 0
            rs_score = float(avg_1d) * 0.2 + float(avg_1w) * 0.3 + float(avg_1m) * 0.5
            sector_results.append({
                'name': sector_name,
                'displayName': _DISPLAY.get(sector_name, sector_name),
                'avgChange1d': avg_1d, 'avgChange1w': avg_1w, 'avgChange1m': avg_1m,
                'relativeStrength': sf(rs_score),
                'stockCount': len(sector_stocks),
                'stocks': sorted(sector_stocks, key=lambda x: float(x.get('ret1m', 0)), reverse=True),
                'topPerformer': max(sector_stocks, key=lambda x: float(x.get('ret1m', 0)))['code'] if sector_stocks else '',
                'worstPerformer': min(sector_stocks, key=lambda x: float(x.get('ret1m', 0)))['code'] if sector_stocks else '',
            })

        sector_results.sort(key=lambda x: float(x['relativeStrength']), reverse=True)
        rotation = 'neutral'
        if sector_results:
            top_names = [s['name'] for s in sector_results[:3]]
            if any(s in top_names for s in ['bankacilik', 'otomotiv', 'enerji', 'holding']):
                rotation = 'risk_on'
            elif any(s in top_names for s in ['perakende', 'gida', 'telekom']):
                rotation = 'risk_off'
        return {
            'sectors': sector_results,
            'rotation': rotation,
            'rotationDescription': {
                'risk_on':  'Dongusel sektorler lider - Risk istahi yuksek',
                'risk_off': 'Defansif sektorler lider - Temkinli piyasa',
                'neutral':  'Belirgin sektor rotasyonu yok',
            }.get(rotation, ''),
        }
    except Exception as e:
        print(f"  [SECTOR-RS] Hata: {e}")
        return {'sectors': [], 'error': str(e)}
