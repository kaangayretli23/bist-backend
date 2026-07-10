"""
BIST Pro - Auto Trader Raporlama & Mutabakat Endpoint'leri
Performans / PnL özeti / reconcile (portfolios↔auto_positions karşılaştır+düzelt).
auto_trader_routes_positions.py'dan ayrıştırıldı (700 satır kuralı).

backend.py `import auto_trader_routes_reports` ile yüklenir;
@app.route decoratorları importta rotaları Flask app'e kaydeder.
"""
from flask import jsonify, request
from config import app, get_db, safe_dict, _stock_cache, _cget
from auth_middleware import require_user


@app.route('/api/auto-trade/performance', methods=['GET'])
@require_user
def auto_trade_performance():
    """C3 — Performans dashboard backend: kullanıcının kapanmış pozisyonlarının istatistikleri.
    Query: userId, days (default 30, max 365)
    Returns: winRate, avgWin, avgLoss, profitFactor, sharpeRatio (basit),
             totalPnL, totalTrades, bestTrade, worstTrade, avgHoldDays,
             distByReason (SL/TP1/TP2/TP3/MANUAL/TIME/PANIC sayilari).
    """
    try:
        uid = request.args.get('userId', '')
        days = max(1, min(365, int(request.args.get('days', 30))))
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400

        from datetime import datetime as _dt, timedelta as _td
        cutoff = (_dt.now() - _td(days=days)).strftime('%Y-%m-%d %H:%M:%S')

        db = get_db()
        rows = db.execute(
            "SELECT id, symbol, entry_price, close_price, quantity, pnl, pnl_pct, "
            "       opened_at, closed_at, close_reason "
            "FROM auto_positions "
            "WHERE user_id=? AND status='closed' AND closed_at>=? "
            "ORDER BY closed_at DESC",
            (uid, cutoff)
        ).fetchall()
        db.close()

        if not rows:
            return jsonify({
                'success': True,
                'days': days,
                'totalTrades': 0,
                'message': f'Son {days} günde kapanmış pozisyon yok'
            })

        wins = [r for r in rows if float(r['pnl'] or 0) > 0]
        losses = [r for r in rows if float(r['pnl'] or 0) < 0]
        total_pnl = sum(float(r['pnl'] or 0) for r in rows)
        gross_win = sum(float(r['pnl'] or 0) for r in wins)
        gross_loss = abs(sum(float(r['pnl'] or 0) for r in losses))
        win_rate = (len(wins) / len(rows) * 100) if rows else 0
        avg_win = (gross_win / len(wins)) if wins else 0
        avg_loss = (gross_loss / len(losses)) if losses else 0
        profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (gross_win if gross_win > 0 else 0)

        # PnL% bazli Sharpe (basit, gunluk ret yok — trade-bazli)
        pnl_pcts = [float(r['pnl_pct'] or 0) for r in rows]
        avg_pnl_pct = sum(pnl_pcts) / len(pnl_pcts) if pnl_pcts else 0
        try:
            import statistics as _stat
            std_pnl = _stat.stdev(pnl_pcts) if len(pnl_pcts) > 1 else 0
        except Exception:
            std_pnl = 0
        sharpe = (avg_pnl_pct / std_pnl) if std_pnl > 0 else 0

        # Hold sure (gun)
        hold_days = []
        for r in rows:
            try:
                _o = _dt.strptime((r['opened_at'] or '')[:19], '%Y-%m-%d %H:%M:%S')
                _c = _dt.strptime((r['closed_at'] or '')[:19], '%Y-%m-%d %H:%M:%S')
                hold_days.append((_c - _o).total_seconds() / 86400)
            except Exception:
                pass
        avg_hold = (sum(hold_days) / len(hold_days)) if hold_days else 0

        best = max(rows, key=lambda r: float(r['pnl'] or 0)) if rows else None
        worst = min(rows, key=lambda r: float(r['pnl'] or 0)) if rows else None

        # Sebep dagilimi
        dist = {}
        for r in rows:
            reason = (r['close_reason'] or 'unknown').lower()
            key = 'manual'
            if 'stop-loss' in reason or 'sl' in reason and 'tp' not in reason: key = 'sl'
            elif 'tp3' in reason: key = 'tp3'
            elif 'tp2' in reason: key = 'tp2'
            elif 'tp1' in reason: key = 'tp1'
            elif 'trailing' in reason: key = 'trailing'
            elif 'panic' in reason: key = 'panic'
            elif 'time' in reason: key = 'time_exit'
            elif 'manuel' in reason or 'manual' in reason: key = 'manual'
            dist[key] = dist.get(key, 0) + 1

        return jsonify(safe_dict({
            'success': True,
            'days': days,
            'totalTrades': len(rows),
            'wins': len(wins),
            'losses': len(losses),
            'winRate': round(win_rate, 2),
            'totalPnL': round(total_pnl, 2),
            'avgWin': round(avg_win, 2),
            'avgLoss': round(avg_loss, 2),
            'profitFactor': round(profit_factor, 2),
            'avgPnLPct': round(avg_pnl_pct, 2),
            'sharpeRatio': round(sharpe, 2),
            'avgHoldDays': round(avg_hold, 1),
            'bestTrade': {
                'symbol': best['symbol'], 'pnl': round(float(best['pnl'] or 0), 2),
                'pnlPct': round(float(best['pnl_pct'] or 0), 2),
                'closedAt': best['closed_at']
            } if best else None,
            'worstTrade': {
                'symbol': worst['symbol'], 'pnl': round(float(worst['pnl'] or 0), 2),
                'pnlPct': round(float(worst['pnl_pct'] or 0), 2),
                'closedAt': worst['closed_at']
            } if worst else None,
            'distByReason': dist,
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auto-trade/reconcile', methods=['GET'])
@require_user
def auto_trade_reconcile():
    """auto_positions (open) ile portfolios'u karsilastir. Sapma varsa flag'le.
    Query: ?userId=X
    Donus: {success, rows: [{symbol, autoQty, autoAvg, portfolioQty, portfolioAvg, status, delta}]}
    status: 'match' | 'missing_in_portfolio' | 'missing_in_auto' | 'qty_mismatch' | 'price_mismatch'
    """
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400

        db = get_db()
        auto_rows = db.execute(
            "SELECT symbol, SUM(quantity) as qty, "
            "SUM(entry_price*quantity)/NULLIF(SUM(quantity),0) as avg_cost "
            "FROM auto_positions WHERE user_id=? AND status='open' GROUP BY symbol",
            (uid,)
        ).fetchall()
        port_rows = db.execute(
            "SELECT symbol, quantity, avg_cost FROM portfolios WHERE user_id=?",
            (uid,)
        ).fetchall()
        db.close()

        auto_map = {r['symbol']: (float(r['qty'] or 0), float(r['avg_cost'] or 0)) for r in auto_rows}
        port_map = {r['symbol']: (float(r['quantity'] or 0), float(r['avg_cost'] or 0)) for r in port_rows}

        all_syms = sorted(set(auto_map.keys()) | set(port_map.keys()))
        rows = []
        mismatch_count = 0
        for sym in all_syms:
            a_qty, a_avg = auto_map.get(sym, (0.0, 0.0))
            p_qty, p_avg = port_map.get(sym, (0.0, 0.0))

            if sym not in port_map:
                status = 'missing_in_portfolio'
            elif sym not in auto_map:
                status = 'missing_in_auto'
            elif abs(a_qty - p_qty) > 0.01:
                status = 'qty_mismatch'
            elif a_avg > 0 and p_avg > 0 and abs(a_avg - p_avg) / max(a_avg, p_avg) > 0.01:
                status = 'price_mismatch'
            else:
                status = 'match'

            if status != 'match':
                mismatch_count += 1

            rows.append({
                'symbol': sym,
                'autoQty': round(a_qty, 4),
                'autoAvg': round(a_avg, 4),
                'portfolioQty': round(p_qty, 4),
                'portfolioAvg': round(p_avg, 4),
                'qtyDelta': round(a_qty - p_qty, 4),
                'status': status,
            })

        return jsonify({
            'success': True,
            'rows': rows,
            'mismatchCount': mismatch_count,
            'totalSymbols': len(all_syms),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auto-trade/reconcile/fix', methods=['POST'])
@require_user
def auto_trade_reconcile_fix():
    """Secilen sembolu portfolios'u auto_positions'a esitle (auto source of truth).
    Body: { userId, symbol }
    auto_positions'taki acik pozisyonun toplam qty + avg'i ile portfolios satirini overwrite eder.
    Sembol auto'da yoksa portfolios'tan siler.
    """
    try:
        data = request.json or {}
        uid = data.get('userId', '')
        sym = (data.get('symbol', '') or '').strip().upper()
        if not uid or not sym:
            return jsonify({'success': False, 'error': 'userId ve symbol gerekli'}), 400

        db = get_db()
        auto_row = db.execute(
            "SELECT SUM(quantity) as qty, "
            "SUM(entry_price*quantity)/NULLIF(SUM(quantity),0) as avg_cost "
            "FROM auto_positions WHERE user_id=? AND symbol=? AND status='open'",
            (uid, sym)
        ).fetchone()
        a_qty = float(auto_row['qty'] or 0) if auto_row else 0.0
        a_avg = float(auto_row['avg_cost'] or 0) if auto_row else 0.0

        existing = db.execute(
            "SELECT id FROM portfolios WHERE user_id=? AND symbol=?", (uid, sym)
        ).fetchone()

        if a_qty <= 0:
            # auto'da yok → portfolios'tan sil
            if existing:
                db.execute("DELETE FROM portfolios WHERE id=?", (existing['id'],))
                db.commit()
                db.close()
                return jsonify({'success': True, 'message': f'{sym} portfolios\'tan silindi'})
            db.close()
            return jsonify({'success': True, 'message': f'{sym} her iki yerde de yok'})

        if existing:
            db.execute(
                "UPDATE portfolios SET quantity=?, avg_cost=? WHERE id=?",
                (round(a_qty, 4), round(a_avg, 4), existing['id'])
            )
        else:
            db.execute(
                "INSERT INTO portfolios (user_id, symbol, quantity, avg_cost) VALUES (?,?,?,?)",
                (uid, sym, round(a_qty, 4), round(a_avg, 4))
            )
        db.commit()
        db.close()
        return jsonify({
            'success': True,
            'message': f'{sym}: portfolios güncellendi ({a_qty} lot @ {a_avg:.2f})',
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auto-trade/pnl-summary', methods=['GET'])
@require_user
def auto_trade_pnl_summary():
    """Hizli PnL ozeti: bugun/hafta/ay/yil toplam realized PnL + trade sayisi.
    Acik pozisyonlarin unrealized PnL'i de eklenir (snapshot).
    Query: userId
    """
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400

        from datetime import datetime as _dt, timedelta as _td
        now = _dt.now()
        # Hafta basi: pazartesi 00:00 (BIST takvimine yakin)
        _wk_start = (now - _td(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        _today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        _month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        _year_start = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        _fmt = '%Y-%m-%d %H:%M:%S'

        windows = {
            'today': _today_start.strftime(_fmt),
            'week':  _wk_start.strftime(_fmt),
            'month': _month_start.strftime(_fmt),
            'year':  _year_start.strftime(_fmt),
        }

        db = get_db()
        out = {}
        for key, cutoff in windows.items():
            row = db.execute(
                "SELECT COALESCE(SUM(pnl), 0) AS pnl, COUNT(*) AS cnt, "
                "       COALESCE(SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END), 0) AS wins, "
                "       COALESCE(SUM(CASE WHEN pnl<0 THEN 1 ELSE 0 END), 0) AS losses "
                "FROM auto_positions "
                "WHERE user_id=? AND status='closed' AND closed_at>=?",
                (uid, cutoff),
            ).fetchone()
            _pnl = float(row['pnl']) if row else 0.0
            _cnt = int(row['cnt']) if row else 0
            _wins = int(row['wins']) if row else 0
            _losses = int(row['losses']) if row else 0
            _wr = round(_wins / _cnt * 100, 1) if _cnt > 0 else 0.0
            out[key] = {
                'realizedPnL': round(_pnl, 2),
                'trades': _cnt,
                'wins': _wins,
                'losses': _losses,
                'winRate': _wr,
            }
        # Acik pozisyonlarin unrealized PnL'i (anlik)
        open_rows = db.execute(
            "SELECT symbol, entry_price, quantity FROM auto_positions "
            "WHERE user_id=? AND status='open'",
            (uid,),
        ).fetchall()
        db.close()
        unreal = 0.0
        open_count = 0
        for r in open_rows:
            sym = r['symbol']
            entry = float(r['entry_price'])
            qty = float(r['quantity'])
            cur = 0.0
            try:
                stk = _cget(_stock_cache, sym) or {}
                cur = float(stk.get('price', 0) or 0)
            except Exception:
                cur = 0.0
            if cur > 0:
                unreal += (cur - entry) * qty
            open_count += 1

        return jsonify({
            'success': True,
            'today': out['today'],
            'week':  out['week'],
            'month': out['month'],
            'year':  out['year'],
            'openPositions': open_count,
            'unrealizedPnL': round(unreal, 2),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
