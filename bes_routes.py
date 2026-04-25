"""
BIST Pro - BES Fund API Routes
"""
from flask import jsonify, request
from config import sf, si, safe_dict, app
from bes_data import (
    _bes_cache_get, _bes_cache_set, _bes_bg_loading, _bes_bg_error,
    _classify_fund, _fetch_tefas_funds, _fetch_tefas_compare,
    _fetch_tefas_allocation, _fetch_tefas_history_chunked,
    _get_tefas_field, _parse_fund_row, _parse_compare_row, _parse_tefas_date,
    _bes_bg_analyze_top, _bes_cache, _bes_cache_lock, BES_CACHE_TTL,
    BES_FUND_GROUPS, _tefas_semaphore,
)
from bes_analysis import _analyze_fund_performance, _bes_optimize, _fund_reasoning, _simulate_bes
import bes_data
import time, threading, traceback, json
from datetime import datetime, timedelta
import numpy as np



# ---- BES API ROUTES ----

@app.route('/api/bes/funds')
def bes_funds():
    """Tum BES fonlarini listele (guncel fiyat, getiri)"""
    try:
        cache_key = 'bes_funds_all'
        cached = _bes_cache_get(cache_key)
        if cached:
            return jsonify(safe_dict(cached))

        today = datetime.now()
        start = (today - timedelta(days=30)).strftime('%d.%m.%Y')
        end = today.strftime('%d.%m.%Y')
        raw = _fetch_tefas_funds(start, end)

        if not raw:
            return jsonify({'success': False, 'error': 'TEFAS verisi alinamadi', 'funds': []})

        # En son tarihteki fonlari al
        funds_map = {}
        for row in (raw if isinstance(raw, list) else []):
            parsed = _parse_fund_row(row)
            if parsed and parsed['code']:
                code = parsed['code']
                if code not in funds_map or parsed['date'] > funds_map[code]['date']:
                    funds_map[code] = parsed

        funds_list = list(funds_map.values())
        for f in funds_list:
            f['category'] = _classify_fund(f.get('name', ''))
            f['categoryLabel'] = BES_FUND_GROUPS.get(f['category'], 'Diğer')

        funds_list.sort(key=lambda x: x.get('total_value', 0), reverse=True)

        result = {
            'success': True,
            'count': len(funds_list),
            'funds': funds_list,
            'timestamp': datetime.now().isoformat(),
            'categories': BES_FUND_GROUPS,
        }
        _bes_cache_set(cache_key, result)
        return jsonify(safe_dict(result))
    except Exception as e:
        print(f"[BES] funds hata: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e), 'funds': []}), 500


@app.route('/api/bes/fund/<code>')
def bes_fund_detail(code):
    """Tek bir BES fonunun detayli analizi"""
    try:
        days = int(request.args.get('days', 365))
        cache_key = f'bes_fund_{code}_{days}'
        cached = _bes_cache_get(cache_key)
        if cached:
            return jsonify(safe_dict(cached))

        history = _fetch_tefas_history_chunked(code, days=days)
        if not history:
            return jsonify({'success': False, 'error': f'{code} fon verisi bulunamadi'})

        perf = _analyze_fund_performance(history, code)
        if not perf:
            return jsonify({'success': False, 'error': f'{code} analiz yapilamadi'})

        # Portfoy dagilimi
        today = datetime.now()
        alloc_start = (today - timedelta(days=7)).strftime('%d.%m.%Y')
        alloc_end = today.strftime('%d.%m.%Y')
        allocation = _fetch_tefas_allocation(code, alloc_start, alloc_end)

        result = {
            'success': True,
            'fund': perf,
            'allocation': allocation,
            'timestamp': datetime.now().isoformat(),
        }
        _bes_cache_set(cache_key, result)
        return jsonify(safe_dict(result))
    except Exception as e:
        print(f"[BES] fund detail hata: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bes/analyze', methods=['POST'])
def bes_analyze():
    """
    Kullanicinin BES bilgilerini alip analiz & oneri yap.
    Input: { funds: [{code, pct}], monthlyContribution, horizonMonths, riskProfile }
    """
    try:
        body = request.get_json(force=True)
        user_funds = body.get('funds', [])
        monthly_contribution = float(body.get('monthlyContribution', 1000))
        horizon_months = int(body.get('horizonMonths', 36))
        risk_profile = body.get('riskProfile', 'moderate')

        if risk_profile not in ('conservative', 'moderate', 'aggressive'):
            risk_profile = 'moderate'

        # Genel fon havuzundan en iyileri bul (cache'den)
        all_funds_cache = _bes_cache_get('bes_analysis_pool')
        if not all_funds_cache:
            # Cache yok - background thread baslatilmis mi kontrol et
            if not _bes_bg_loading:
                threading.Thread(target=_bes_bg_analyze_top, daemon=True).start()
            return jsonify({'success': True, 'loading': True, 'message': 'Fon verileri hazırlaniyor. Lütfen BES sekmesini açik birakin, veriler hazir olunca otomatik yüklenecek.'})

        all_perfs = all_funds_cache

        # Kullanicinin mevcut fonlarini analiz et (sadece cache hazirsa)
        current_analysis = []
        if user_funds:
            for uf in user_funds:
                code = uf.get('code', '').strip().upper()
                if not code:
                    continue
                # Oncelikle cache pool'dan bul
                cached_perf = next((f for f in all_perfs if f['code'] == code), None)
                if cached_perf:
                    perf = dict(cached_perf)
                else:
                    # Cache'de yoksa kisa sureli veri cek (timeout riski dusuk)
                    history = _fetch_tefas_history_chunked(code, days=90)
                    perf = _analyze_fund_performance(history, code)
                if perf:
                    perf['userPct'] = uf.get('pct', 0)
                    current_analysis.append(perf)

        # Optimizasyon
        recommendations = _bes_optimize(all_perfs if all_perfs else current_analysis, risk_profile, horizon_months)

        # Simulasyon
        simulation = _simulate_bes(recommendations, monthly_contribution, horizon_months)

        # Mevcut portfoy vs onerilen karsilastirma
        current_score = 0
        if current_analysis:
            for ca in current_analysis:
                w = ca.get('userPct', 0) / 100
                s = ca.get('sharpe', 0)
                current_score += w * s

        recommended_score = 0
        if recommendations:
            for rec in recommendations:
                w = rec.get('recommendedPct', 0) / 100
                s = rec.get('sharpe', 0)
                recommended_score += w * s

        result = {
            'success': True,
            'currentPortfolio': {
                'funds': [{
                    'code': ca['code'],
                    'name': ca['name'],
                    'category': ca['category'],
                    'categoryLabel': BES_FUND_GROUPS.get(ca['category'], 'Diğer'),
                    'userPct': ca.get('userPct', 0),
                    'returns': ca['returns'],
                    'volatility': ca['volatility'],
                    'sharpe': ca['sharpe'],
                    'maxDrawdown': ca['maxDrawdown'],
                } for ca in current_analysis],
                'overallSharpe': sf(current_score),
            },
            'recommendations': recommendations,
            'simulation': simulation,
            'riskProfile': risk_profile,
            'riskProfileLabel': {'conservative': 'Muhafazakar', 'moderate': 'Dengeli', 'aggressive': 'Agresif'}.get(risk_profile, 'Dengeli'),
            'horizonMonths': horizon_months,
            'monthlyContribution': monthly_contribution,
            'comparison': {
                'currentScore': sf(current_score * 100),
                'recommendedScore': sf(recommended_score * 100),
                'improvement': sf((recommended_score - current_score) * 100),
            },
            'timestamp': datetime.now().isoformat(),
        }
        return jsonify(safe_dict(result))
    except Exception as e:
        print(f"[BES] analyze hata: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bes/simulate', methods=['POST'])
def bes_simulate():
    """Hizli simulasyon: verilen fonlar ve oranlarla birikim projeksiyonu"""
    try:
        body = request.get_json(force=True)
        funds = body.get('funds', [])
        monthly = float(body.get('monthlyContribution', 1000))
        months = int(body.get('horizonMonths', 36))

        # Her fonun performansini cek
        fund_perfs = []
        for f in funds:
            code = f.get('code', '').strip().upper()
            pct = float(f.get('pct', 0))
            if not code:
                continue
            history = _fetch_tefas_history_chunked(code, days=180)
            perf = _analyze_fund_performance(history, code)
            if perf:
                perf['recommendedPct'] = pct
                perf['return6m'] = perf.get('returns', {}).get('6a', 0)
                fund_perfs.append(perf)

        simulation = _simulate_bes(fund_perfs, monthly, months)
        return jsonify(safe_dict({
            'success': True,
            'simulation': simulation,
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bes/debug')
def bes_debug():
    """BES veri pipeline debug endpoint"""
    try:
        results = {}
        today = datetime.now()

        # 1. Compare API testi
        try:
            compare = _fetch_tefas_compare()
            if compare and isinstance(compare, list) and len(compare) > 0:
                sample = compare[0] if isinstance(compare[0], dict) else {}
                parsed = _parse_compare_row(compare[0]) if sample else None
                results['compare_api'] = {
                    'success': True, 'count': len(compare),
                    'sample_keys': list(sample.keys())[:15],
                    'parsed_sample': parsed,
                }
            else:
                results['compare_api'] = {'success': False, 'data': str(compare)[:200]}
        except Exception as e:
            results['compare_api'] = {'success': False, 'error': str(e)}

        # 2. History API testi (7 gun)
        try:
            start = (today - timedelta(days=7)).strftime('%d.%m.%Y')
            end = today.strftime('%d.%m.%Y')
            hist = _fetch_tefas_funds(start, end)
            if hist and isinstance(hist, list) and len(hist) > 0:
                sample = hist[0] if isinstance(hist[0], dict) else {}
                parsed = _parse_fund_row(hist[0]) if sample else None
                results['history_api'] = {
                    'success': True, 'count': len(hist),
                    'sample_keys': list(sample.keys())[:15],
                    'parsed_sample': parsed,
                }
            else:
                results['history_api'] = {'success': False, 'data': str(hist)[:200]}
        except Exception as e:
            results['history_api'] = {'success': False, 'error': str(e)}

        # 3. Cache durumu
        pool = _bes_cache_get('bes_analysis_pool')
        results['cache'] = {
            'has_pool': pool is not None,
            'pool_size': len(pool) if pool else 0,
            'bg_loading': _bes_bg_loading,
            'bg_error': _bes_bg_error,
        }
        if pool and len(pool) > 0:
            sample = pool[0]
            results['cache']['sample_fund'] = {
                'code': sample.get('code'), 'name': sample.get('name', '')[:40],
                'returns': sample.get('returns'), 'volatility': sample.get('volatility'),
                'sharpe': sample.get('sharpe'), 'price': sample.get('currentPrice'),
            }

        return jsonify(safe_dict(results))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bes/top')
def bes_top():
    """Kategorilere gore en iyi BES fonlari - background thread ile Render timeout bypass"""
    global _bes_bg_loading, _bes_bg_error
    try:
        category = request.args.get('category', '')
        period = request.args.get('period', '3a')
        limit_n = int(request.args.get('limit', 20))

        cache_key = f'bes_top_{category}_{period}'
        cached = _bes_cache_get(cache_key)
        if cached:
            return jsonify(safe_dict(cached))

        pool = _bes_cache_get('bes_analysis_pool')
        if not pool:
            # Cache yok - background thread ile analiz baslat
            if _bes_bg_loading:
                # Zaten calisiyor, polling devam etsin
                return jsonify({'success': True, 'loading': True, 'message': 'Fonlar analiz ediliyor, lutfen bekleyin...'})

            if _bes_bg_error:
                err = _bes_bg_error
                # Hata sonrasi tekrar dene - thread'i yeniden baslat
                _bes_bg_error = ''
                print(f"[BES-TOP] Onceki hata: {err}, yeniden deneniyor...")
                t = threading.Thread(target=_bes_bg_analyze_top, daemon=True)
                t.start()
                return jsonify({'success': True, 'loading': True, 'message': f'Tekrar deneniyor... ({err})'})

            # Background thread baslat
            print("[BES-TOP] Background analiz thread baslatiliyor...")
            t = threading.Thread(target=_bes_bg_analyze_top, daemon=True)
            t.start()
            return jsonify({'success': True, 'loading': True, 'message': 'BES fon analizi baslatildi, birkaç saniye sonra hazir olacak...'})

        # Pool hazir - filtrele ve sirala
        filtered = pool
        if category:
            filtered = [f for f in pool if f.get('category') == category]

        def sort_key(f):
            returns = f.get('returns', {})
            return returns.get(period, 0) or 0

        filtered.sort(key=sort_key, reverse=True)

        top_list = []
        for f in filtered[:limit_n]:
            top_list.append({
                'code': f['code'],
                'name': f['name'],
                'category': f['category'],
                'categoryLabel': BES_FUND_GROUPS.get(f['category'], 'Diğer'),
                'currentPrice': f['currentPrice'],
                'returns': f['returns'],
                'volatility': f['volatility'],
                'sharpe': f['sharpe'],
                'maxDrawdown': f['maxDrawdown'],
            })

        result = {
            'success': True,
            'category': category,
            'categoryLabel': BES_FUND_GROUPS.get(category, 'Tümü'),
            'period': period,
            'count': len(top_list),
            'funds': top_list,
            'timestamp': datetime.now().isoformat(),
        }
        _bes_cache_set(cache_key, result)
        return jsonify(safe_dict(result))
    except Exception as e:
        print(f"[BES-TOP] HATA: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# =====================================================================

