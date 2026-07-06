"""
BIST Pro - Auto Trader Performans Analiz Endpoint'leri
Win-rate by hour / score-band / top symbols. auto_trader_routes.py'dan ayri (600 satir kurali).
backend.py `import auto_trader_routes_analytics` ile yukler.

Lazy: config, auto_trader, get_db modul-icinde fonksiyon ic.
"""
from flask import jsonify, request
from config import app, get_db, safe_dict
from auth_middleware import require_user


@app.route('/api/auto-trade/analytics')
@require_user
def auto_trade_analytics():
    """Performans analizi: win-rate by hour/score-band, top symbols.
    Query: userId, days (default 30, 1..365 araligi).

    Returns:
      hourly: [{hour:10..18, trades:n, wins:n, winRate:%}]
      scoreBands: [{band:'5-7'/..., trades:n, wins:n, winRate:%, avgPnl:tl}]
      topWinners: [{symbol, trades, totalPnl, avgPct}] (top 5)
      topLosers: [{symbol, trades, totalPnl, avgPct}] (bottom 5)
      summary: {totalTrades, wins, losses, winRate, totalPnl, avgWinPct, avgLossPct, expectancy}
    """
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400
        try:
            days = min(max(int(request.args.get('days', 30)), 1), 365)
        except (TypeError, ValueError):
            days = 30
        from datetime import datetime, timedelta
        cutoff = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')

        db = get_db()
        # BUY trade'lerini kapanmis pozisyonlarla join et
        rows = db.execute(
            "SELECT t.created_at AS buy_at, t.signal_score AS score, t.confidence AS conf, "
            "       p.symbol AS sym, p.pnl AS pnl, p.pnl_pct AS pnl_pct, p.entry_price AS entry, "
            "       p.quantity AS qty, p.closed_at AS close_at, p.close_reason AS reason "
            "FROM auto_trades t "
            "JOIN auto_positions p ON t.position_id = p.id "
            "WHERE t.user_id=? AND t.action='BUY' AND p.status='closed' "
            "      AND t.created_at >= ? AND p.pnl IS NOT NULL "
            "ORDER BY t.created_at DESC",
            (uid, cutoff)
        ).fetchall()
        db.close()

        # Hour bucket (10-18)
        hourly = {h: {'trades': 0, 'wins': 0, 'pnl': 0.0} for h in range(10, 19)}
        # Score bands
        bands = {
            '5-7':  {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pcts': []},
            '7-8':  {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pcts': []},
            '8-9':  {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pcts': []},
            '9+':   {'trades': 0, 'wins': 0, 'pnl': 0.0, 'pcts': []},
        }
        # Symbol aggregates
        sym_agg = {}
        # Summary
        total_pnl = 0.0
        wins, losses = 0, 0
        win_pcts, loss_pcts = [], []

        for r in rows:
            pnl = float(r['pnl'] or 0)
            pnl_pct = float(r['pnl_pct'] or 0)
            sc = float(r['score'] or 0)
            sym = r['sym']
            total_pnl += pnl
            if pnl > 0:
                wins += 1
                win_pcts.append(pnl_pct)
            elif pnl < 0:
                losses += 1
                loss_pcts.append(pnl_pct)

            # Hour parse — created_at format: 'YYYY-MM-DD HH:MM:SS' veya ISO
            buy_at = r['buy_at'] or ''
            hour = -1
            try:
                # ISO 'T' ya da bosluk: split her ikisi de calissin
                _t = str(buy_at).replace('T', ' ').split(' ')
                if len(_t) >= 2:
                    hour = int(_t[1].split(':')[0])
            except Exception:
                pass
            if 10 <= hour <= 18:
                hourly[hour]['trades'] += 1
                hourly[hour]['pnl'] += pnl
                if pnl > 0:
                    hourly[hour]['wins'] += 1

            # Score band
            band = '5-7' if sc < 7 else '7-8' if sc < 8 else '8-9' if sc < 9 else '9+'
            bands[band]['trades'] += 1
            bands[band]['pnl'] += pnl
            bands[band]['pcts'].append(pnl_pct)
            if pnl > 0:
                bands[band]['wins'] += 1

            # Symbol
            if sym not in sym_agg:
                sym_agg[sym] = {'trades': 0, 'pnl': 0.0, 'pcts': []}
            sym_agg[sym]['trades'] += 1
            sym_agg[sym]['pnl'] += pnl
            sym_agg[sym]['pcts'].append(pnl_pct)

        total_trades = wins + losses
        win_rate = round(wins / total_trades * 100, 1) if total_trades > 0 else 0
        avg_win = round(sum(win_pcts) / len(win_pcts), 2) if win_pcts else 0
        avg_loss = round(sum(loss_pcts) / len(loss_pcts), 2) if loss_pcts else 0
        # Expectancy = winRate * avgWin + (1-winRate) * avgLoss (yuzde olarak)
        wr = (wins / total_trades) if total_trades > 0 else 0
        expectancy = round(wr * avg_win + (1 - wr) * avg_loss, 2)

        hourly_out = []
        for h in range(10, 19):
            t = hourly[h]['trades']
            w = hourly[h]['wins']
            hourly_out.append({
                'hour': h,
                'trades': t,
                'wins': w,
                'winRate': round(w / t * 100, 1) if t > 0 else 0,
                'pnl': round(hourly[h]['pnl'], 2),
            })

        bands_out = []
        for b, d in bands.items():
            t = d['trades']
            avg_pct = round(sum(d['pcts']) / len(d['pcts']), 2) if d['pcts'] else 0
            bands_out.append({
                'band': b,
                'trades': t,
                'wins': d['wins'],
                'winRate': round(d['wins'] / t * 100, 1) if t > 0 else 0,
                'pnl': round(d['pnl'], 2),
                'avgPct': avg_pct,
            })

        sym_list = []
        for sym, d in sym_agg.items():
            avg_pct = round(sum(d['pcts']) / len(d['pcts']), 2) if d['pcts'] else 0
            sym_list.append({
                'symbol': sym,
                'trades': d['trades'],
                'totalPnl': round(d['pnl'], 2),
                'avgPct': avg_pct,
            })
        # PnL'e gore sirala
        sym_list.sort(key=lambda x: x['totalPnl'], reverse=True)
        top_winners = [s for s in sym_list if s['totalPnl'] > 0][:5]
        top_losers = sorted([s for s in sym_list if s['totalPnl'] < 0],
                            key=lambda x: x['totalPnl'])[:5]

        return jsonify(safe_dict({
            'success': True,
            'days': days,
            'summary': {
                'totalTrades': total_trades,
                'wins': wins,
                'losses': losses,
                'winRate': win_rate,
                'totalPnl': round(total_pnl, 2),
                'avgWinPct': avg_win,
                'avgLossPct': avg_loss,
                'expectancy': expectancy,
            },
            'hourly': hourly_out,
            'scoreBands': bands_out,
            'topWinners': top_winners,
            'topLosers': top_losers,
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auto-trade/signal-backtest')
@require_user
def auto_trade_signal_backtest():
    """#4 Sinyal backtest: skor kovasına göre başarı oranı (TÜM AL sinyalleri, sadece açılanlar değil).

    Kaynak: signal_log (üretilen her AL sinyali) ⨝ signal_outcomes (horizon sonrası sonuç).
    Bu, /analytics'ten farklı — orası GERÇEKLEŞEN trade'leri ölçer, burası sistemin SİNYAL
    kalitesini ölçer (skor yükseldikçe başarı artıyor mu?).

    Query:
      horizon: 1|3|5|10|20 gün (default 10)
      days:    geriye bakış penceresi (default 365, 7..1095)

    Returns:
      scoreBuckets: [{bucket:'3-5'|'5-7'|'7+'|'<3', n, wins, winRate,
                      avgReturn, avgExcess, excessN}]   (excess = getiri - XU100)
      aiDecisions:  [{decision:'APPROVE_CANDIDATE'|'WAIT'|'REJECT', n, avgScore, avgConfidence}]
      aiNote: AI kararı-outcome bağı için not (henüz az veri / bağlantı yok).
    """
    try:
        import time as _t
        try:
            horizon = int(request.args.get('horizon', 10))
        except (TypeError, ValueError):
            horizon = 10
        if horizon not in (1, 3, 5, 10, 20):
            horizon = 10
        try:
            days = min(max(int(request.args.get('days', 365)), 7), 1095)
        except (TypeError, ValueError):
            days = 365
        cutoff_epoch = _t.time() - days * 86400

        db = get_db()
        rows = db.execute(
            "SELECT s.score AS score, o.return_pct AS ret, o.win AS win, o.index_return_pct AS idx "
            "FROM signal_log s JOIN signal_outcomes o ON o.signal_id = s.id "
            "WHERE o.horizon_days=? AND s.action='AL' AND s.logged_at >= ? "
            "      AND o.return_pct IS NOT NULL",
            (horizon, cutoff_epoch)
        ).fetchall()
        ai_rows = db.execute(
            "SELECT ai_decision AS dec, COUNT(*) AS n, AVG(score) AS avg_score, "
            "       AVG(ai_confidence) AS avg_conf "
            "FROM ai_trade_reviews GROUP BY ai_decision"
        ).fetchall()
        db.close()

        # Skor kovaları — düşükten yükseğe gradient (yüksek skor = yüksek başarı mı?)
        order = ['<3', '3-5', '5-7', '7+']
        buckets = {k: {'n': 0, 'wins': 0, 'rets': [], 'excess': []} for k in order}

        def _bucket_of(sc):
            if sc < 3:
                return '<3'
            if sc < 5:
                return '3-5'
            if sc < 7:
                return '5-7'
            return '7+'

        for r in rows:
            sc = float(r['score'] or 0)
            ret = float(r['ret'] or 0)
            d = buckets[_bucket_of(sc)]
            d['n'] += 1
            d['wins'] += int(r['win'] or 0)
            d['rets'].append(ret)
            if r['idx'] is not None:
                d['excess'].append(ret - float(r['idx']))

        score_buckets = []
        for b in order:
            d = buckets[b]
            n = d['n']
            score_buckets.append({
                'bucket': b,
                'n': n,
                'wins': d['wins'],
                'winRate': round(d['wins'] / n * 100, 1) if n else 0,
                'avgReturn': round(sum(d['rets']) / len(d['rets']), 2) if d['rets'] else 0,
                'avgExcess': round(sum(d['excess']) / len(d['excess']), 2) if d['excess'] else None,
                'excessN': len(d['excess']),
            })

        ai_decisions = [{
            'decision': r['dec'] or '?',
            'n': int(r['n'] or 0),
            'avgScore': round(float(r['avg_score']), 1) if r['avg_score'] is not None else None,
            'avgConfidence': round(float(r['avg_conf']), 0) if r['avg_conf'] is not None else None,
        } for r in ai_rows]
        ai_total = sum(a['n'] for a in ai_decisions)

        return jsonify(safe_dict({
            'success': True,
            'horizon': horizon,
            'days': days,
            'totalSignals': sum(b['n'] for b in score_buckets),
            'scoreBuckets': score_buckets,
            'aiDecisions': ai_decisions,
            'aiNote': (
                f"AI kararı sayısı: {ai_total}. AI kararları henüz outcome'a bağlı değil "
                "(ai_trade_reviews ↔ signal_outcomes join yok); yeterli veri birikince "
                "APPROVE/WAIT/REJECT başarı oranı eklenecek. Şimdilik sadece dağılım."
            ),
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
