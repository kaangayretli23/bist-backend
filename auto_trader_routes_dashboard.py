"""
BIST Pro - Auto Trader Dashboard Endpoints
status + daily-summary (en agir iki endpoint).
auto_trader_routes.py'dan ayri (600 satir kurali).
backend.py `import auto_trader_routes_dashboard` ile yukler.
"""
from flask import jsonify, request
from config import app, get_db, safe_dict, _stock_cache, _cget, _cset
from auth_middleware import require_user
from auto_trader import (
    _auto_get_config, _auto_get_open_positions, _auto_get_daily_trade_count,
)


@app.route('/api/auto-trade/status')
@require_user
def auto_trade_status():
    """Oto-trade durumunu, acik pozisyonlari ve performansi getir"""
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400

        cfg = _auto_get_config(uid)
        positions = _auto_get_open_positions(uid)
        daily_count = _auto_get_daily_trade_count(uid)

        # Acik pozisyonlara guncel fiyat ekle
        # Once realtime (WebSocket/yfinance fresh), sonra batch cache, en son fetch.
        try:
            from realtime_prices import get_price as _rt_get
        except Exception:
            _rt_get = None
        total_pnl = 0
        for pos in positions:
            cur_price = 0.0
            if _rt_get:
                try:
                    rt = _rt_get(pos['symbol'])
                    if rt and rt > 0:
                        cur_price = float(rt)
                except Exception:
                    pass
            if cur_price <= 0:
                stock = _cget(_stock_cache, pos['symbol'])
                cur_price = float(stock.get('price', 0)) if stock else 0
            # Cache'de yoksa direkt fetch dene (portföydeki hisse)
            if cur_price == 0:
                try:
                    from data_fetcher import _process_stock
                    _, fresh = _process_stock(pos['symbol'], retry_count=1)
                    if fresh:
                        _cset(_stock_cache, pos['symbol'], fresh)
                        cur_price = float(fresh.get('price', 0))
                except Exception:
                    pass
            pos['currentPrice'] = cur_price
            entry = pos.get('entryPrice', 0) or 0
            if cur_price > 0 and entry > 0:
                pos['pnl'] = round((cur_price - entry) * pos['quantity'], 2)
                pos['pnlPct'] = round((cur_price - entry) / entry * 100, 2)
                total_pnl += pos['pnl']
            else:
                pos['pnl'] = 0
                pos['pnlPct'] = 0

        # Kapatilmis pozisyon istatistikleri
        db = get_db()
        closed = db.execute("SELECT * FROM auto_positions WHERE user_id=? AND status='closed' ORDER BY closed_at DESC LIMIT 50", (uid,)).fetchall()
        db.close()

        total_closed_pnl = 0
        win_count = 0
        loss_count = 0
        for r in closed:
            p = float(r['pnl'] or 0)
            total_closed_pnl += p
            if p > 0:
                win_count += 1
            elif p < 0:
                loss_count += 1

        total_trades = win_count + loss_count
        win_rate = round(win_count / total_trades * 100, 1) if total_trades > 0 else 0

        return jsonify(safe_dict({
            'success': True,
            'config': cfg,
            'positions': positions,
            'dailyTradeCount': daily_count,
            'performance': {
                'openPnL': round(total_pnl, 2),
                'closedPnL': round(total_closed_pnl, 2),
                'totalPnL': round(total_pnl + total_closed_pnl, 2),
                'winCount': win_count,
                'lossCount': loss_count,
                'winRate': win_rate,
                'totalTrades': total_trades,
            },
            'recentClosed': [{
                'id': int(r['id']),
                'symbol': r['symbol'], 'entryPrice': float(r['entry_price']),
                'quantity': float(r['quantity']),
                'closePrice': float(r['close_price'] or 0), 'pnl': float(r['pnl'] or 0),
                'pnlPct': float(r['pnl_pct'] or 0), 'reason': r['close_reason'],
                'closedAt': r['closed_at'],
            } for r in closed[:20]],
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auto-trade/daily-summary')
@require_user
def auto_trade_daily_summary():
    """Bugunku tum aktiviteyi tek ekranda topla:
       - kapali isemler P&L (gross/net/komisyon)
       - acik pozisyon unrealized P&L
       - bugunki BUY/SELL sayisi
       - karar log dagilimi (skip/buy/pending)
       - aktif reject cooldown listesi
    """
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400
        from datetime import datetime as _dt
        today = _dt.now().strftime('%Y-%m-%d')
        today_floor = today + ' 00:00:00'

        db = get_db()
        # Bugun kapatilan pozisyonlar (PnL)
        closed = db.execute(
            "SELECT symbol, entry_price, close_price, quantity, pnl, pnl_pct, close_reason, closed_at "
            "FROM auto_positions WHERE user_id=? AND status='closed' AND closed_at >= ? "
            "ORDER BY closed_at DESC",
            (uid, today_floor)
        ).fetchall()
        realized_pnl = sum(float(r['pnl'] or 0) for r in closed)
        winners = sum(1 for r in closed if float(r['pnl'] or 0) > 0)
        losers = sum(1 for r in closed if float(r['pnl'] or 0) < 0)

        # Bugunku trade log (buy + sell)
        trades_today = db.execute(
            "SELECT COUNT(*) AS c, action FROM auto_trades "
            "WHERE user_id=? AND created_at >= ? GROUP BY action",
            (uid, today_floor)
        ).fetchall()
        trade_counts = {r['action']: int(r['c']) for r in trades_today}

        # Karar dagilimi
        dec_summary = db.execute(
            "SELECT decision, reason, COUNT(*) AS c FROM auto_decisions "
            "WHERE user_id=? AND created_at >= ? GROUP BY decision, reason ORDER BY c DESC",
            (uid, today_floor)
        ).fetchall()
        decisions = [{'decision': r['decision'], 'reason': r['reason'] or '', 'count': int(r['c'])}
                     for r in dec_summary]
        # Decision aggregate
        dec_total = {'BUY': 0, 'PENDING': 0, 'SKIP': 0}
        for d in decisions:
            if d['decision'] in dec_total:
                dec_total[d['decision']] += d['count']

        # Acik pozisyon unrealized P&L — realtime fiyat oncelikli
        try:
            from realtime_prices import get_price as _rt_get
        except Exception:
            _rt_get = None
        open_positions = _auto_get_open_positions(uid) or []
        unrealized = 0.0
        for p in open_positions:
            cur = 0.0
            if _rt_get:
                try:
                    rt = _rt_get(p['symbol'])
                    if rt and rt > 0:
                        cur = float(rt)
                except Exception:
                    pass
            if cur <= 0:
                stock = _cget(_stock_cache, p['symbol'])
                cur = float(stock.get('price', 0)) if stock else 0
            if cur <= 0:
                continue
            unrealized += (cur - float(p['entryPrice'])) * float(p['quantity'])

        # Aktif reject cooldown listesi (ram-resident)
        from auto_trader_risk import _reject_cooldown
        import time as _t
        now_ts = _t.time()
        cooldown_active = []
        for k, (exp, reason) in list(_reject_cooldown.items()):
            if not k.startswith(uid + '_'):
                continue
            if exp <= now_ts:
                continue
            sym = k[len(uid)+1:]
            cooldown_active.append({
                'symbol': sym, 'reason': reason,
                'remainingMin': int((exp - now_ts) / 60),
            })
        cooldown_active.sort(key=lambda x: -x['remainingMin'])

        # Piyasa rejimi (cache'li)
        try:
            from auto_trader_regime import get_market_regime
            _regime_mode, _regime_detail = get_market_regime()
        except Exception:
            _regime_mode, _regime_detail = 'risk-on', 'Rejim modulü yüklenemedi'

        # Komisyon toplam (closed today)
        from auto_trader import _calc_trade_costs
        commission_today = 0.0
        for r in closed:
            entry = float(r['entry_price'] or 0)
            cp = float(r['close_price'] or 0)
            qty = float(r['quantity'] or 0)
            commission_today += _calc_trade_costs(uid, entry * qty, cp * qty)

        db.close()

        return jsonify(safe_dict({
            'success': True,
            'date': today,
            'pnl': {
                'realized': round(realized_pnl, 2),
                'unrealized': round(unrealized, 2),
                'total': round(realized_pnl + unrealized, 2),
                'commissionToday': round(commission_today, 2),
            },
            'trades': {
                'closedToday': len(closed),
                'winners': winners,
                'losers': losers,
                'buyCount': trade_counts.get('BUY', 0),
                'sellCount': sum(trade_counts.get(a, 0) for a in trade_counts if a.startswith('SELL')),
            },
            'closedPositions': [{
                'symbol': r['symbol'], 'entry': float(r['entry_price'] or 0),
                'close': float(r['close_price'] or 0), 'qty': float(r['quantity'] or 0),
                'pnl': float(r['pnl'] or 0), 'pnlPct': float(r['pnl_pct'] or 0),
                'reason': r['close_reason'] or '', 'closedAt': r['closed_at'],
            } for r in closed],
            'decisions': decisions,
            'decisionTotal': dec_total,
            'cooldownActive': cooldown_active,
            'openPositions': len(open_positions),
            'regime': {'mode': _regime_mode, 'detail': _regime_detail},
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
