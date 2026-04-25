"""
BES Fon Performans Analizi ve Optimizasyon
_analyze_fund_performance, _bes_optimize, _fund_reasoning, _simulate_bes
bes_data.py'dan ayrıştırıldı (700 satır kuralı).
"""
import time, traceback
from datetime import datetime, timedelta
try:
    import numpy as np
except ImportError:
    pass
from config import sf, si

def _analyze_fund_performance(history_data, fund_code):
    """Fon gecmis verisinden performans metrikleri hesapla"""
    if not history_data or len(history_data) < 2:
        return None

    # Fiyatlari cikar ve sırala
    prices = []
    for row in history_data:
        parsed = _parse_fund_row(row)
        if parsed and parsed['price'] and parsed['price'] > 0:
            dt = _parse_tefas_date(parsed['date'])
            prices.append({
                'date': parsed['date'],
                'date_parsed': dt,
                'price': parsed['price'],
                'code': parsed['code'],
                'name': parsed['name'],
            })

    if len(prices) < 2:
        return None

    # Tarihe gore sirala (en eski en basta) - datetime objeleriyle dogru siralama
    try:
        prices.sort(key=lambda x: x['date_parsed'] or datetime.min)
    except Exception:
        # Fallback: string siralama
        prices.sort(key=lambda x: x['date'])

    current_price = prices[-1]['price']
    first_price = prices[0]['price']
    total_return = ((current_price - first_price) / first_price) * 100 if first_price > 0 else 0
    total_days = len(prices)

    # Donemsel getiriler - tarih bazli lookback (takvim gunu -> islem gunu donusumu)
    returns = {}
    period_map = {'1h': 7, '1a': 30, '3a': 90, '6a': 180, '1y': 365}
    for label, cal_days in period_map.items():
        # Tarih bazli lookback: en son tarihten cal_days gun oncesine en yakin veri noktasini bul
        target_date = prices[-1]['date_parsed'] - timedelta(days=cal_days) if prices[-1].get('date_parsed') else None
        found_price = None

        if target_date:
            # Target tarihine en yakin (ve oncesindeki) veri noktasini bul
            for p in prices:
                if p.get('date_parsed') and p['date_parsed'] <= target_date:
                    found_price = p['price']
                # Once erisince devam et (sirali oldugu icin son eslesme en yakin olacak)

        # Tarih bazli bulunamadiysa, index bazli fallback
        if not found_price:
            # Tahmini islem gunu: takvim gunu * 5/7 (hafta ici orani)
            approx_trading_days = max(1, int(cal_days * 5 / 7))
            if len(prices) >= approx_trading_days:
                found_price = prices[-min(approx_trading_days, len(prices))]['price']
            elif len(prices) >= max(cal_days // 4, 2):
                # En az ceyrek kadar veri varsa mevcut verinin basindan hesapla
                found_price = prices[0]['price']

        if found_price and found_price > 0:
            returns[label] = sf(((current_price - found_price) / found_price) * 100)
        else:
            returns[label] = None

    # Volatilite (gunluk fiyat degisim std sapma)
    daily_returns = []
    for i in range(1, len(prices)):
        if prices[i-1]['price'] > 0:
            dr = (prices[i]['price'] - prices[i-1]['price']) / prices[i-1]['price']
            daily_returns.append(dr)

    volatility = 0
    if daily_returns:
        mean_r = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_r)**2 for r in daily_returns) / len(daily_returns)
        volatility = sf(variance ** 0.5 * (252 ** 0.5) * 100)  # Yillik volatilite %

    # Max drawdown
    max_dd = 0
    peak = prices[0]['price']
    for p in prices:
        if p['price'] > peak:
            peak = p['price']
        dd = ((peak - p['price']) / peak) * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe benzeri metrik (risksiz oran ~%45 TRY)
    risk_free_daily = 0.45 / 252
    avg_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
    std_daily = (sum((r - avg_daily_return)**2 for r in daily_returns) / len(daily_returns)) ** 0.5 if daily_returns else 1
    sharpe = sf(((avg_daily_return - risk_free_daily) / std_daily * (252 ** 0.5)) if std_daily > 0 else 0)

    return {
        'code': fund_code,
        'name': prices[-1].get('name', ''),
        'category': _classify_fund(prices[-1].get('name', '')),
        'currentPrice': current_price,
        'firstPrice': first_price,
        'totalReturn': sf(total_return),
        'totalDays': total_days,
        'returns': returns,
        'volatility': volatility,
        'maxDrawdown': sf(max_dd),
        'sharpe': sharpe,
        'dailyReturns': daily_returns[-30:] if daily_returns else [],
        'priceHistory': [{'date': p['date'], 'price': p['price']} for p in prices[-90:]],
    }

def _bes_optimize(funds_perf, risk_profile='moderate', horizon_months=12):
    """
    BES fon dagilim optimizasyonu.
    Risk profiline gore ideal fon agirliklarini hesaplar.
    """
    if not funds_perf:
        return []

    risk_weights = {
        'conservative': {'return_w': 0.2, 'risk_w': 0.6, 'sharpe_w': 0.2, 'max_equity': 20},
        'moderate':     {'return_w': 0.35, 'risk_w': 0.35, 'sharpe_w': 0.3, 'max_equity': 50},
        'aggressive':   {'return_w': 0.5, 'risk_w': 0.15, 'sharpe_w': 0.35, 'max_equity': 80},
    }
    weights = risk_weights.get(risk_profile, risk_weights['moderate'])

    scored_funds = []
    for f in funds_perf:
        if not f:
            continue
        rets = f.get('returns', {})
        ret_6m = rets.get('6a') or rets.get('3a') or rets.get('1a') or 0
        vol = f.get('volatility', 50)
        sharpe = f.get('sharpe', 0)
        category = f.get('category', 'diger')

        # Normalizing: return pozitif iyi, vol dusuk iyi, sharpe yuksek iyi
        ret_score = min(max(ret_6m, -50), 100) / 100
        vol_score = 1 - min(vol, 100) / 100
        sharpe_score = min(max(sharpe, -3), 3) / 3

        total_score = (
            ret_score * weights['return_w'] +
            vol_score * weights['risk_w'] +
            sharpe_score * weights['sharpe_w']
        )

        scored_funds.append({
            'code': f['code'],
            'name': f['name'],
            'category': category,
            'categoryLabel': BES_FUND_GROUPS.get(category, 'Diğer'),
            'score': sf(total_score * 100),
            'return6m': sf(ret_6m),
            'volatility': f['volatility'],
            'sharpe': f['sharpe'],
            'maxDrawdown': f['maxDrawdown'],
            'currentPrice': f['currentPrice'],
        })

    scored_funds.sort(key=lambda x: x['score'], reverse=True)

    # Kategori bazli dagilim onerisi
    category_targets = {
        'conservative': {'borclanma': 40, 'para_piyasasi': 25, 'altin': 15, 'katilim': 10, 'hisse': 5, 'diger': 5},
        'moderate':     {'hisse': 25, 'borclanma': 25, 'altin': 15, 'katilim': 15, 'para_piyasasi': 10, 'diger': 10},
        'aggressive':   {'hisse': 40, 'endeks': 15, 'altin': 15, 'doviz': 10, 'borclanma': 10, 'karma': 10},
    }
    targets = category_targets.get(risk_profile, category_targets['moderate'])

    # Her kategori icin en iyi fonu sec
    recommendations = []
    used_categories = set()
    for cat, target_pct in sorted(targets.items(), key=lambda x: x[1], reverse=True):
        best_in_cat = [f for f in scored_funds if f['category'] == cat]
        if best_in_cat:
            pick = best_in_cat[0]
            pick['recommendedPct'] = target_pct
            pick['reasoning'] = _fund_reasoning(pick, cat, target_pct, risk_profile)
            recommendations.append(pick)
            used_categories.add(cat)

    # Hedef kategoride fon bulunamazsa en iyi genel fonlardan tamamla
    total_allocated = sum(r['recommendedPct'] for r in recommendations)
    if total_allocated < 100:
        remaining = 100 - total_allocated
        remaining_funds = [f for f in scored_funds if f['code'] not in [r['code'] for r in recommendations]]
        if remaining_funds:
            pick = remaining_funds[0]
            pick['recommendedPct'] = remaining
            pick['reasoning'] = f"Kalan %{remaining} oran portföy dengeleme amacıyla önerildi."
            recommendations.append(pick)

    return recommendations

def _fund_reasoning(fund, category, pct, risk_profile):
    """Fon onerisi icin aciklama metni uret"""
    cat_label = BES_FUND_GROUPS.get(category, category)
    sharpe = fund.get('sharpe', 0)
    vol = fund.get('volatility', 0)
    ret = fund.get('return6m', 0)

    reasons = []
    if ret > 10: reasons.append(f"6 aylık getirisi %{ret} ile güçlü")
    elif ret > 0: reasons.append(f"6 aylık %{ret} pozitif getiri")
    else: reasons.append(f"6 aylık getiri %{ret}")

    if vol < 10: reasons.append("düşük volatilite")
    elif vol < 20: reasons.append("orta seviye volatilite")
    else: reasons.append(f"%{vol} volatilite")

    if sharpe > 1: reasons.append("yüksek risk-getiri oranı")
    elif sharpe > 0: reasons.append("pozitif risk-getiri dengesi")

    profile_labels = {'conservative': 'muhafazakar', 'moderate': 'dengeli', 'aggressive': 'agresif'}
    profile_label = profile_labels.get(risk_profile, 'dengeli')

    return f"{cat_label} kategorisinde {profile_label} profil için %{pct} ağırlık. " + ", ".join(reasons) + "."

def _simulate_bes(recommendations, monthly_contribution, horizon_months):
    """BES portfoy simulasyonu: aylık katkı ile birikim projeksiyonu"""
    if not recommendations or monthly_contribution <= 0 or horizon_months <= 0:
        return {'error': 'Geçersiz parametreler'}

    # Devlet katkisi (%30, yillik max limit - 2024 icin ~27.000 TL civarı)
    yearly_contribution = monthly_contribution * 12
    devlet_katkisi_rate = 0.30
    devlet_katkisi_yearly_max = 30000  # Yaklasik yillik limit
    devlet_katkisi_yearly = min(yearly_contribution * devlet_katkisi_rate, devlet_katkisi_yearly_max)
    devlet_katkisi_monthly = devlet_katkisi_yearly / 12

    # Her fon icin aylik getiri tahmini (gecmis veriden)
    total_monthly = monthly_contribution + devlet_katkisi_monthly
    monthly_results = []
    fund_balances = {}

    for rec in recommendations:
        pct = rec.get('recommendedPct', 0) / 100
        ret_6m = rec.get('return6m', 0) or 0
        # 6 aylik getiriyi aylik getiriye cevir
        monthly_return = ((1 + ret_6m / 100) ** (1/6) - 1)
        fund_balances[rec['code']] = {
            'name': rec['name'],
            'code': rec['code'],
            'pct': pct,
            'monthlyReturn': monthly_return,
            'balance': 0,
            'totalContribution': 0,
        }

    total_balance = 0
    total_contributed = 0
    total_devlet = 0
    total_gain = 0

    for month in range(1, horizon_months + 1):
        month_contribution = monthly_contribution
        month_devlet = devlet_katkisi_monthly
        total_contributed += month_contribution
        total_devlet += month_devlet

        for code, fb in fund_balances.items():
            contrib = (month_contribution + month_devlet) * fb['pct']
            fb['totalContribution'] += contrib
            # Birikim: onceki bakiye * (1 + aylik getiri) + yeni katki
            fb['balance'] = fb['balance'] * (1 + fb['monthlyReturn']) + contrib

        total_balance = sum(fb['balance'] for fb in fund_balances.values())
        total_gain = total_balance - total_contributed - total_devlet

        if month % 3 == 0 or month == horizon_months or month <= 3:
            monthly_results.append({
                'month': month,
                'totalBalance': sf(total_balance),
                'totalContributed': sf(total_contributed),
                'devletKatkisi': sf(total_devlet),
                'totalGain': sf(total_gain),
                'gainPct': sf((total_gain / (total_contributed + total_devlet)) * 100) if (total_contributed + total_devlet) > 0 else 0,
            })

    fund_details = []
    for code, fb in fund_balances.items():
        gain = fb['balance'] - fb['totalContribution']
        fund_details.append({
            'code': fb['code'],
            'name': fb['name'],
            'pct': sf(fb['pct'] * 100),
            'balance': sf(fb['balance']),
            'totalContribution': sf(fb['totalContribution']),
            'gain': sf(gain),
            'gainPct': sf((gain / fb['totalContribution']) * 100) if fb['totalContribution'] > 0 else 0,
            'monthlyReturn': sf(fb['monthlyReturn'] * 100, 3),
        })

    return {
        'totalBalance': sf(total_balance),
        'totalContributed': sf(total_contributed),
        'devletKatkisi': sf(total_devlet),
        'totalGain': sf(total_gain),
        'totalGainPct': sf((total_gain / (total_contributed + total_devlet)) * 100) if (total_contributed + total_devlet) > 0 else 0,
        'horizonMonths': horizon_months,
        'monthlyContribution': monthly_contribution,
        'monthlyDevlet': sf(devlet_katkisi_monthly),
        'timeline': monthly_results,
        'fundDetails': fund_details,
    }
