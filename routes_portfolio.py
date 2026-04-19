"""
Portfolio, watchlist, and alerts routes.
"""
import sqlite3
import os
from datetime import datetime
from flask import Blueprint, jsonify, request
import requests as req_lib

from config import (
    _stock_cache, _cget, get_db, db_conn, safe_dict, sf,
)
from auth_middleware import require_user

TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')

portfolio_bp = Blueprint('portfolio', __name__)


def _send_telegram_alerts(user_id, triggered_alerts):
    """Tetiklenen uyarilari Telegram'a gonder"""
    if not TELEGRAM_BOT_TOKEN:
        return
    try:
        db = get_db()
        user = db.execute("SELECT telegram_chat_id FROM users WHERE id=?", (user_id,)).fetchone()
        db.close()
        if not user or not user['telegram_chat_id']:
            return
        chat_id = user['telegram_chat_id']
        for alert in triggered_alerts:
            text = f"🔔 *BIST Pro Uyari*\n\n{alert['message']}"
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            try:
                req_lib.post(url, json={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'}, timeout=5)
            except Exception:
                pass
    except Exception:
        pass


@portfolio_bp.route('/api/portfolio', methods=['GET'])
@require_user
def get_portfolio():
    try:
        uid = request.args.get('userId', request.args.get('user', ''))
        if not uid:
            return jsonify(safe_dict({
                'success': True, 'positions': [],
                'summary': {'totalValue': 0, 'totalCost': 0, 'totalPnL': 0, 'totalPnLPct': 0, 'positionCount': 0},
                'needsLogin': True,
            }))
        with db_conn() as db:
            rows = db.execute("SELECT * FROM portfolios WHERE user_id=?", (uid,)).fetchall()
        pos = []; tv = tc = 0
        for r in rows:
            cd = _cget(_stock_cache, r['symbol'])
            if not cd: continue
            q, ac, cp = r['quantity'], r['avg_cost'], float(cd.get('price') or 0)
            if cp <= 0: continue
            mv = cp * q; cb = ac * q; upnl = mv - cb; tv += mv; tc += cb
            pos.append({
                'id': r['id'], 'symbol': r['symbol'], 'name': cd.get('name', r['symbol']),
                'quantity': q, 'avgCost': sf(ac), 'currentPrice': sf(cp),
                'marketValue': sf(mv), 'costBasis': sf(cb),
                'unrealizedPnL': sf(upnl), 'unrealizedPnLPct': sf(upnl / cb * 100 if cb else 0),
                'changePct': float(cd.get('changePct') or 0), 'weight': 0,
            })
        for p in pos:
            p['weight'] = sf(float(p['marketValue']) / tv * 100 if tv > 0 else 0)
        tp = tv - tc
        dp = sum(float(p['marketValue']) * p['changePct'] / 100 for p in pos)
        return jsonify(safe_dict({
            'success': True, 'positions': pos,
            'summary': {
                'totalValue': sf(tv), 'totalCost': sf(tc),
                'totalPnL': sf(tp), 'totalPnLPct': sf(tp / tc * 100 if tc else 0),
                'dailyPnL': sf(dp), 'positionCount': len(pos),
            },
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@portfolio_bp.route('/api/portfolio', methods=['POST'])
@require_user
def add_portfolio():
    try:
        d = request.json or {}
        uid = d.get('userId', d.get('user', ''))
        sym = d.get('symbol', '').upper()
        qty = float(d.get('quantity', 0))
        ac = float(d.get('avgCost', 0))
        if not uid: return jsonify({'error': 'Giris yapmaniz gerekli'}), 401
        if not sym or qty <= 0 or ac <= 0: return jsonify({'error': 'Gecersiz veri'}), 400
        with db_conn() as db:
            existing = db.execute("SELECT * FROM portfolios WHERE user_id=? AND symbol=?", (uid, sym)).fetchone()
            if existing:
                new_qty = existing['quantity'] + qty
                new_avg = (existing['avg_cost'] * existing['quantity'] + ac * qty) / new_qty
                db.execute("UPDATE portfolios SET quantity=?, avg_cost=? WHERE id=?", (new_qty, new_avg, existing['id']))
            else:
                db.execute("INSERT INTO portfolios (user_id, symbol, quantity, avg_cost) VALUES (?, ?, ?, ?)", (uid, sym, qty, ac))
            db.commit()
        return jsonify({'success': True, 'message': f'{sym} portfoye eklendi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@portfolio_bp.route('/api/portfolio', methods=['DELETE'])
@require_user
def del_portfolio():
    try:
        d = request.json or {}
        uid = d.get('userId', d.get('user', ''))
        sym = d.get('symbol', '').upper()
        pid = d.get('id')
        with db_conn() as db:
            if pid:
                db.execute("DELETE FROM portfolios WHERE id=?", (pid,))
            elif uid and sym:
                db.execute("DELETE FROM portfolios WHERE user_id=? AND symbol=?", (uid, sym))
            db.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@portfolio_bp.route('/api/portfolio/risk')
def portfolio_risk():
    return jsonify({'success': True, 'risk': {'message': 'Risk analizi yukleniyor'}})


@portfolio_bp.route('/api/watchlist', methods=['GET'])
@require_user
def get_watchlist():
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify(safe_dict({'success': True, 'watchlist': [], 'symbols': [], 'needsLogin': True}))
        with db_conn() as db:
            rows = db.execute("SELECT symbol FROM watchlists WHERE user_id=?", (uid,)).fetchall()
        symbols = [r['symbol'] for r in rows]
        from config import _get_stocks
        stocks = _get_stocks(symbols) if symbols else []
        return jsonify(safe_dict({'success': True, 'watchlist': stocks, 'symbols': symbols}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@portfolio_bp.route('/api/watchlist', methods=['POST'])
@require_user
def update_watchlist():
    try:
        d = request.json or {}
        uid = d.get('userId', '')
        sym = d.get('symbol', '').upper()
        action = d.get('action', 'add')
        if not uid: return jsonify({'error': 'Giris yapmaniz gerekli'}), 401
        if not sym: return jsonify({'error': 'Hisse kodu gerekli'}), 400
        with db_conn() as db:
            if action == 'add':
                try:
                    db.execute("INSERT INTO watchlists (user_id, symbol) VALUES (?, ?)", (uid, sym))
                except sqlite3.IntegrityError:
                    pass
            elif action == 'remove':
                db.execute("DELETE FROM watchlists WHERE user_id=? AND symbol=?", (uid, sym))
            elif action == 'toggle':
                existing = db.execute("SELECT id FROM watchlists WHERE user_id=? AND symbol=?", (uid, sym)).fetchone()
                if existing:
                    db.execute("DELETE FROM watchlists WHERE id=?", (existing['id'],))
                    action = 'removed'
                else:
                    db.execute("INSERT INTO watchlists (user_id, symbol) VALUES (?, ?)", (uid, sym))
                    action = 'added'
            db.commit()
        return jsonify({'success': True, 'action': action, 'symbol': sym})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@portfolio_bp.route('/api/alerts', methods=['GET'])
@require_user
def get_alerts():
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify(safe_dict({'success': True, 'alerts': [], 'needsLogin': True}))
        with db_conn() as db:
            rows = db.execute("SELECT * FROM alerts WHERE user_id=? ORDER BY created_at DESC", (uid,)).fetchall()
        alerts = [{
            'id': r['id'], 'symbol': r['symbol'], 'condition': r['condition'],
            'targetValue': r['target_value'], 'active': bool(r['active']),
            'triggered': bool(r['triggered']), 'triggeredAt': r['triggered_at'],
            'createdAt': r['created_at'],
        } for r in rows]
        return jsonify(safe_dict({'success': True, 'alerts': alerts}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@portfolio_bp.route('/api/alerts', methods=['POST'])
@require_user
def add_alert():
    try:
        d = request.json or {}
        uid = d.get('userId', '')
        sym = d.get('symbol', '').upper()
        condition = d.get('condition', 'price_above')
        target = float(d.get('targetValue', d.get('threshold', 0)))
        if not uid: return jsonify({'error': 'Giris yapmaniz gerekli'}), 401
        if not sym or target <= 0: return jsonify({'error': 'Gecersiz veri'}), 400
        with db_conn() as db:
            db.execute("INSERT INTO alerts (user_id, symbol, condition, target_value) VALUES (?, ?, ?, ?)",
                       (uid, sym, condition, target))
            db.commit()
        return jsonify({'success': True, 'message': f'{sym} icin uyari eklendi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@portfolio_bp.route('/api/alerts/<int:aid>', methods=['DELETE'])
@require_user
def del_alert(aid):
    try:
        uid = (request.json or {}).get('userId', request.args.get('userId', ''))
        if not uid:
            return jsonify({'error': 'Giris yapmaniz gerekli'}), 401
        with db_conn() as db:
            db.execute("DELETE FROM alerts WHERE id=? AND user_id=?", (aid, uid))
            db.commit()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@portfolio_bp.route('/api/alerts/check', methods=['POST'])
def check_alerts():
    """Tetiklenen uyarilari kontrol et"""
    try:
        d = request.json or {}
        uid = d.get('userId', '')
        if not uid:
            return jsonify({'success': True, 'triggered': []})
        triggered = []
        with db_conn() as db:
            rows = db.execute("SELECT * FROM alerts WHERE user_id=? AND active=1 AND triggered=0", (uid,)).fetchall()
            for r in rows:
                stock = _cget(_stock_cache, r['symbol'])
                if not stock:
                    continue
                price = stock['price']
                fire = False
                if r['condition'] == 'price_above' and price >= r['target_value']:
                    fire = True
                elif r['condition'] == 'price_below' and price <= r['target_value']:
                    fire = True
                elif r['condition'] == 'change_above' and stock.get('changePct', 0) >= r['target_value']:
                    fire = True
                elif r['condition'] == 'change_below' and stock.get('changePct', 0) <= r['target_value']:
                    fire = True

                if fire:
                    db.execute("UPDATE alerts SET triggered=1, triggered_at=? WHERE id=?",
                               (datetime.now().isoformat(), r['id']))
                    triggered.append({
                        'id': r['id'], 'symbol': r['symbol'], 'condition': r['condition'],
                        'targetValue': r['target_value'], 'currentPrice': price,
                        'message': f"{r['symbol']} uyarisi tetiklendi: {r['condition']} {r['target_value']} (Guncel: {price})",
                    })
            db.commit()

        if triggered:
            _send_telegram_alerts(uid, triggered)

        return jsonify(safe_dict({'success': True, 'triggered': triggered}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500
