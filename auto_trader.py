"""
BIST Pro - Auto Trading Engine Module
"""
import threading, traceback, time
from datetime import datetime
from flask import jsonify, request
from config import (
    get_db, USE_POSTGRES, PG_OK, sf, safe_dict,
    _lock, _stock_cache, _hist_cache, _cget, _get_stocks, app
)
from trade_plans import calc_trade_plan

# =====================================================================
# AUTO TRADING ENGINE — Otomatik Alim-Satim Motoru
# =====================================================================

_auto_trade_lock = threading.Lock()

def _auto_get_config(user_id):
    """Kullanicinin oto-trade konfigurasyonunu getir"""
    try:
        db = get_db()
        row = db.execute("SELECT * FROM auto_config WHERE user_id=?", (user_id,)).fetchone()
        db.close()
        if row:
            return {
                'enabled': bool(row['enabled']),
                'capital': float(row['capital']),
                'maxPositions': int(row['max_positions']),
                'riskPerTrade': float(row['risk_per_trade']),
                'minScore': float(row['min_score']),
                'minConfidence': float(row['min_confidence']),
                'tradeStyle': row['trade_style'],
                'stopLossPct': float(row['stop_loss_pct']),
                'takeProfitPct': float(row['take_profit_pct']),
                'trailingStop': bool(row['trailing_stop']),
                'trailingPct': float(row['trailing_pct']),
                'allowedSymbols': row['allowed_symbols'],
                'blockedSymbols': row['blocked_symbols'],
                'maxDailyTrades': int(row['max_daily_trades']),
            }
        return None
    except Exception as e:
        print(f"[AUTO-TRADE] Config getirme hatasi: {e}")
        return None

def _auto_get_open_positions(user_id):
    """Kullanicinin acik pozisyonlarini getir"""
    try:
        db = get_db()
        rows = db.execute("SELECT * FROM auto_positions WHERE user_id=? AND status='open'", (user_id,)).fetchall()
        db.close()
        positions = []
        for r in rows:
            positions.append({
                'id': r['id'], 'symbol': r['symbol'], 'side': r['side'],
                'entryPrice': float(r['entry_price']), 'quantity': float(r['quantity']),
                'stopLoss': float(r['stop_loss'] or 0), 'takeProfit1': float(r['take_profit1'] or 0),
                'takeProfit2': float(r['take_profit2'] or 0), 'takeProfit3': float(r['take_profit3'] or 0),
                'trailingStop': float(r['trailing_stop'] or 0), 'highestPrice': float(r['highest_price'] or 0),
                'openedAt': r['opened_at'],
            })
        return positions
    except Exception as e:
        print(f"[AUTO-TRADE] Pozisyon getirme hatasi: {e}")
        return []

def _auto_get_daily_trade_count(user_id):
    """Buguncu trade sayisini getir"""
    try:
        db = get_db()
        today = datetime.now().strftime('%Y-%m-%d')
        if USE_POSTGRES and PG_OK:
            row = db.execute("SELECT COUNT(*) as cnt FROM auto_trades WHERE user_id=? AND created_at::date=?::date", (user_id, today)).fetchone()
        else:
            row = db.execute("SELECT COUNT(*) as cnt FROM auto_trades WHERE user_id=? AND date(created_at)=?", (user_id, today)).fetchone()
        db.close()
        return int(row['cnt']) if row else 0
    except Exception as e:
        print(f"[AUTO-TRADE] Trade sayisi hatasi: {e}")
        return 0

def _auto_log_trade(user_id, symbol, action, price, quantity, reason, score, confidence, position_id=None):
    """Trade logunu DB'ye kaydet"""
    try:
        db = get_db()
        db.execute(
            "INSERT INTO auto_trades (user_id, symbol, action, price, quantity, reason, signal_score, confidence, position_id) VALUES (?,?,?,?,?,?,?,?,?)",
            (user_id, symbol, action, price, quantity, reason, score, confidence, position_id)
        )
        db.commit()
        db.close()
        print(f"[AUTO-TRADE] {action}: {symbol} @ {price} x{quantity} - {reason}")
    except Exception as e:
        print(f"[AUTO-TRADE] Trade log hatasi: {e}")

def _auto_open_position(user_id, symbol, price, quantity, stop_loss, tp1, tp2, tp3, trailing_sl):
    """Yeni pozisyon ac"""
    try:
        db = get_db()
        db.execute(
            """INSERT INTO auto_positions
               (user_id, symbol, side, entry_price, quantity, stop_loss, take_profit1, take_profit2, take_profit3, trailing_stop, highest_price)
               VALUES (?,?,'long',?,?,?,?,?,?,?,?)""",
            (user_id, symbol, price, quantity, stop_loss, tp1, tp2, tp3, trailing_sl, price)
        )
        db.commit()
        # Son eklenen ID'yi al
        if USE_POSTGRES and PG_OK:
            row = db.execute("SELECT MAX(id) as mid FROM auto_positions WHERE user_id=? AND symbol=?", (user_id, symbol)).fetchone()
        else:
            row = db.execute("SELECT last_insert_rowid() as mid").fetchone()
        pos_id = int(row['mid']) if row else 0
        db.close()
        return pos_id
    except Exception as e:
        print(f"[AUTO-TRADE] Pozisyon acma hatasi: {e}")
        return 0

def _auto_close_position(position_id, close_price, reason):
    """Pozisyon kapat"""
    try:
        db = get_db()
        row = db.execute("SELECT * FROM auto_positions WHERE id=?", (position_id,)).fetchone()
        if not row:
            db.close()
            return
        entry = float(row['entry_price'])
        qty = float(row['quantity'])
        pnl = (close_price - entry) * qty
        pnl_pct = ((close_price - entry) / entry * 100) if entry > 0 else 0
        now_str = datetime.now().isoformat()
        db.execute(
            "UPDATE auto_positions SET status='closed', closed_at=?, close_price=?, close_reason=?, pnl=?, pnl_pct=? WHERE id=?",
            (now_str, close_price, reason, round(pnl, 2), round(pnl_pct, 2), position_id)
        )
        db.commit()
        db.close()
        print(f"[AUTO-TRADE] Pozisyon kapatildi #{position_id}: {row['symbol']} PnL={pnl:.2f} TL ({pnl_pct:.1f}%) - {reason}")
    except Exception as e:
        print(f"[AUTO-TRADE] Pozisyon kapatma hatasi: {e}")

def _auto_update_trailing(position_id, new_trailing, new_highest):
    """Trailing stop guncelle"""
    try:
        db = get_db()
        db.execute(
            "UPDATE auto_positions SET trailing_stop=?, highest_price=? WHERE id=?",
            (new_trailing, new_highest, position_id)
        )
        db.commit()
        db.close()
    except Exception as e:
        print(f"[AUTO-TRADE] Trailing guncelleme hatasi: {e}")

def _auto_engine_cycle():
    """Ana oto-trade dongusu — her 5dk'da bir cagrilir.
    1) Config'i aktif kullanicilar icin oku
    2) Acik pozisyonlari kontrol et (SL/TP/trailing)
    3) Yeni firsatlari tara ve pozisyon ac
    """
    with _auto_trade_lock:
        try:
            # Tum aktif auto-trade kullanicilarini bul
            db = get_db()
            users = db.execute("SELECT * FROM auto_config WHERE enabled=1").fetchall()
            db.close()
            if not users:
                return

            for user_row in users:
                uid = user_row['user_id']
                cfg = _auto_get_config(uid)
                if not cfg or not cfg['enabled']:
                    continue

                # ---- ADIM 1: Acik pozisyonlari kontrol et ----
                positions = _auto_get_open_positions(uid)
                for pos in positions:
                    sym = pos['symbol']
                    stock = _cget(_stock_cache, sym)
                    if not stock:
                        continue
                    cur_price = float(stock.get('price', 0))
                    if cur_price <= 0:
                        continue

                    # Stop-Loss kontrolu
                    sl = pos['stopLoss']
                    if sl > 0 and cur_price <= sl:
                        _auto_close_position(pos['id'], cur_price, f"Stop-Loss tetiklendi ({sl:.2f})")
                        _auto_log_trade(uid, sym, 'SELL_SL', cur_price, pos['quantity'],
                                       f"SL tetiklendi: {cur_price:.2f} <= {sl:.2f}", 0, 0, pos['id'])
                        continue

                    # Take-Profit kontrolu (kademe kademe)
                    tp3 = pos['takeProfit3']
                    tp2 = pos['takeProfit2']
                    tp1 = pos['takeProfit1']
                    if tp3 > 0 and cur_price >= tp3:
                        _auto_close_position(pos['id'], cur_price, f"TP3 hedef ({tp3:.2f})")
                        _auto_log_trade(uid, sym, 'SELL_TP3', cur_price, pos['quantity'],
                                       f"TP3: {cur_price:.2f} >= {tp3:.2f}", 0, 0, pos['id'])
                        continue
                    elif tp2 > 0 and cur_price >= tp2:
                        # TP2'de yarisi kapat
                        half_qty = round(pos['quantity'] / 2, 2)
                        if half_qty > 0:
                            _auto_close_position(pos['id'], cur_price, f"TP2 kismi kapanis ({tp2:.2f})")
                            _auto_log_trade(uid, sym, 'SELL_TP2', cur_price, half_qty,
                                           f"TP2 kismi: {cur_price:.2f} >= {tp2:.2f}", 0, 0, pos['id'])
                        continue

                    # Trailing Stop kontrolu
                    if cfg['trailingStop']:
                        highest = pos['highestPrice']
                        if cur_price > highest:
                            new_highest = cur_price
                            new_trailing = cur_price * (1 - cfg['trailingPct'] / 100)
                            _auto_update_trailing(pos['id'], new_trailing, new_highest)
                        else:
                            trailing_sl = pos['trailingStop']
                            if trailing_sl > 0 and cur_price <= trailing_sl:
                                _auto_close_position(pos['id'], cur_price, f"Trailing-Stop ({trailing_sl:.2f})")
                                _auto_log_trade(uid, sym, 'SELL_TRAIL', cur_price, pos['quantity'],
                                               f"Trailing SL: {cur_price:.2f} <= {trailing_sl:.2f}", 0, 0, pos['id'])
                                continue

                # ---- ADIM 2: Yeni firsatlari tara ----
                open_positions = _auto_get_open_positions(uid)
                open_symbols = {p['symbol'] for p in open_positions}

                if len(open_positions) >= cfg['maxPositions']:
                    continue  # Max pozisyon doldu

                daily_trades = _auto_get_daily_trade_count(uid)
                if daily_trades >= cfg['maxDailyTrades']:
                    continue  # Gunluk limit doldu

                # Allowed/blocked symbol filtreleri
                allowed = set(cfg['allowedSymbols'].split(',')) if cfg['allowedSymbols'] else set()
                blocked = set(cfg['blockedSymbols'].split(',')) if cfg['blockedSymbols'] else set()

                # Sinyal skoru yuksek hisseleri bul
                candidates = []
                stocks = _get_stocks()
                for s in stocks:
                    sym = s.get('code', '')
                    if not sym or sym in open_symbols:
                        continue
                    if allowed and sym not in allowed:
                        continue
                    if sym in blocked:
                        continue

                    rec = s.get('recommendation', {})
                    signal = rec.get('signal', '')
                    score = float(rec.get('score', 0))
                    confidence = float(rec.get('confidence', 0))

                    if signal != 'AL' or score < cfg['minScore'] or confidence < cfg['minConfidence']:
                        continue

                    candidates.append({
                        'symbol': sym, 'price': float(s.get('price', 0)),
                        'score': score, 'confidence': confidence,
                        'name': s.get('name', sym),
                    })

                # Skora gore sirala, en iyi adaylari al
                candidates.sort(key=lambda x: x['score'], reverse=True)
                slots = cfg['maxPositions'] - len(open_positions)
                daily_remaining = cfg['maxDailyTrades'] - daily_trades

                for cand in candidates[:min(slots, daily_remaining)]:
                    sym = cand['symbol']
                    price = cand['price']
                    if price <= 0:
                        continue

                    # Pozisyon buyuklugu hesapla (risk-bazli)
                    risk_amount = cfg['capital'] * (cfg['riskPerTrade'] / 100)
                    sl_distance = price * (cfg['stopLossPct'] / 100)
                    if sl_distance <= 0:
                        continue
                    quantity = round(risk_amount / sl_distance, 2)
                    position_cost = quantity * price

                    # Sermaye kontrolu (mevcut acik pozisyonlar + bu)
                    used_capital = sum(p['entryPrice'] * p['quantity'] for p in open_positions)
                    if used_capital + position_cost > cfg['capital']:
                        continue

                    # Stop-Loss ve Take-Profit hesapla
                    stop_loss = round(price * (1 - cfg['stopLossPct'] / 100), 2)
                    tp1 = round(price * (1 + cfg['takeProfitPct'] / 100), 2)
                    tp2 = round(price * (1 + cfg['takeProfitPct'] * 1.5 / 100), 2)
                    tp3 = round(price * (1 + cfg['takeProfitPct'] * 2 / 100), 2)
                    trailing_sl = round(price * (1 - cfg['trailingPct'] / 100), 2) if cfg['trailingStop'] else 0

                    # Trade planini kontrol et (daha iyi SL/TP varsa kullan)
                    hist = _hist_cache.get(sym, {}).get('data')
                    if hist is not None and len(hist) >= 20:
                        try:
                            plan = calc_trade_plan(hist, symbol=sym)
                            tf_key = {'daily': 'daily', 'swing': 'weekly', 'monthly': 'monthly'}.get(cfg['tradeStyle'], 'daily')
                            tf_plan = plan.get(tf_key, {})
                            buy_plan = tf_plan.get('buyPlan', {})
                            if buy_plan:
                                plan_sl = float(buy_plan.get('stopLoss', 0))
                                if plan_sl > 0:
                                    stop_loss = plan_sl
                                targets = buy_plan.get('targets', [])
                                if len(targets) >= 1:
                                    tp1 = float(targets[0].get('price', tp1))
                                if len(targets) >= 2:
                                    tp2 = float(targets[1].get('price', tp2))
                                if len(targets) >= 3:
                                    tp3 = float(targets[2].get('price', tp3))
                        except Exception:
                            pass

                    # Pozisyon ac
                    pos_id = _auto_open_position(uid, sym, price, quantity, stop_loss, tp1, tp2, tp3, trailing_sl)
                    if pos_id:
                        _auto_log_trade(uid, sym, 'BUY', price, quantity,
                                       f"Skor={cand['score']:.1f}, Guven=%{cand['confidence']:.0f}, SL={stop_loss:.2f}, TP1={tp1:.2f}",
                                       cand['score'], cand['confidence'], pos_id)
                        open_positions = _auto_get_open_positions(uid)
                        open_symbols.add(sym)

            print(f"[AUTO-TRADE] Dongu tamamlandi: {len(users)} kullanici tarandi")
        except Exception as e:
            print(f"[AUTO-TRADE] Engine hatasi: {e}")
            traceback.print_exc()


# ---- AUTO-TRADE API ENDPOINTS ----

@app.route('/api/auto-trade/config', methods=['GET', 'POST'])
def auto_trade_config():
    """Oto-trade konfigurasyonunu getir/guncelle"""
    try:
        uid = request.args.get('userId', '') or (request.json or {}).get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400

        if request.method == 'GET':
            cfg = _auto_get_config(uid)
            if cfg:
                return jsonify(safe_dict({'success': True, 'config': cfg}))
            return jsonify(safe_dict({'success': True, 'config': None, 'message': 'Henuz konfigürasyon yok'}))

        # POST — guncelle veya olustur
        data = request.json or {}
        db = get_db()
        existing = db.execute("SELECT * FROM auto_config WHERE user_id=?", (uid,)).fetchone()

        if existing:
            db.execute("""UPDATE auto_config SET
                enabled=?, capital=?, max_positions=?, risk_per_trade=?,
                min_score=?, min_confidence=?, trade_style=?,
                stop_loss_pct=?, take_profit_pct=?, trailing_stop=?, trailing_pct=?,
                allowed_symbols=?, blocked_symbols=?, max_daily_trades=?
                WHERE user_id=?""",
                (int(data.get('enabled', existing['enabled'])),
                 float(data.get('capital', existing['capital'])),
                 int(data.get('maxPositions', existing['max_positions'])),
                 float(data.get('riskPerTrade', existing['risk_per_trade'])),
                 float(data.get('minScore', existing['min_score'])),
                 float(data.get('minConfidence', existing['min_confidence'])),
                 data.get('tradeStyle', existing['trade_style']),
                 float(data.get('stopLossPct', existing['stop_loss_pct'])),
                 float(data.get('takeProfitPct', existing['take_profit_pct'])),
                 int(data.get('trailingStop', existing['trailing_stop'])),
                 float(data.get('trailingPct', existing['trailing_pct'])),
                 data.get('allowedSymbols', existing['allowed_symbols']),
                 data.get('blockedSymbols', existing['blocked_symbols']),
                 int(data.get('maxDailyTrades', existing['max_daily_trades'])),
                 uid))
        else:
            db.execute("""INSERT INTO auto_config
                (user_id, enabled, capital, max_positions, risk_per_trade,
                 min_score, min_confidence, trade_style,
                 stop_loss_pct, take_profit_pct, trailing_stop, trailing_pct,
                 allowed_symbols, blocked_symbols, max_daily_trades)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (uid,
                 int(data.get('enabled', 0)),
                 float(data.get('capital', 100000)),
                 int(data.get('maxPositions', 5)),
                 float(data.get('riskPerTrade', 2.0)),
                 float(data.get('minScore', 8.0)),
                 float(data.get('minConfidence', 60)),
                 data.get('tradeStyle', 'swing'),
                 float(data.get('stopLossPct', 3.0)),
                 float(data.get('takeProfitPct', 6.0)),
                 int(data.get('trailingStop', 1)),
                 float(data.get('trailingPct', 2.0)),
                 data.get('allowedSymbols', ''),
                 data.get('blockedSymbols', ''),
                 int(data.get('maxDailyTrades', 3))))
        db.commit()
        db.close()
        return jsonify({'success': True, 'message': 'Konfigürasyon kaydedildi'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auto-trade/toggle', methods=['POST'])
def auto_trade_toggle():
    """Oto-trade'i ac/kapat"""
    try:
        data = request.json or {}
        uid = data.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400
        enabled = int(data.get('enabled', 0))
        db = get_db()
        existing = db.execute("SELECT * FROM auto_config WHERE user_id=?", (uid,)).fetchone()
        if existing:
            db.execute("UPDATE auto_config SET enabled=? WHERE user_id=?", (enabled, uid))
            db.commit()
        else:
            db.execute("""INSERT INTO auto_config (user_id, enabled) VALUES (?,?)""", (uid, enabled))
            db.commit()
        db.close()
        status = 'aktif' if enabled else 'deaktif'
        print(f"[AUTO-TRADE] {uid}: {status}")
        return jsonify({'success': True, 'enabled': bool(enabled), 'message': f'Oto-trade {status}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auto-trade/status')
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
        total_pnl = 0
        for pos in positions:
            stock = _cget(_stock_cache, pos['symbol'])
            cur_price = float(stock.get('price', 0)) if stock else 0
            pos['currentPrice'] = cur_price
            if cur_price > 0:
                pos['pnl'] = round((cur_price - pos['entryPrice']) * pos['quantity'], 2)
                pos['pnlPct'] = round((cur_price - pos['entryPrice']) / pos['entryPrice'] * 100, 2)
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
                'symbol': r['symbol'], 'entryPrice': float(r['entry_price']),
                'closePrice': float(r['close_price'] or 0), 'pnl': float(r['pnl'] or 0),
                'pnlPct': float(r['pnl_pct'] or 0), 'reason': r['close_reason'],
                'closedAt': r['closed_at'],
            } for r in closed[:20]],
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auto-trade/trades')
def auto_trade_trades():
    """Trade gecmisini getir"""
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400
        limit = int(request.args.get('limit', 50))
        db = get_db()
        rows = db.execute("SELECT * FROM auto_trades WHERE user_id=? ORDER BY created_at DESC LIMIT ?", (uid, limit)).fetchall()
        db.close()
        trades = [{
            'id': r['id'], 'symbol': r['symbol'], 'action': r['action'],
            'price': float(r['price']), 'quantity': float(r['quantity']),
            'reason': r['reason'], 'score': float(r['signal_score'] or 0),
            'confidence': float(r['confidence'] or 0), 'createdAt': r['created_at'],
        } for r in rows]
        return jsonify(safe_dict({'success': True, 'trades': trades}))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auto-trade/close', methods=['POST'])
def auto_trade_manual_close():
    """Manuel pozisyon kapatma"""
    try:
        data = request.json or {}
        uid = data.get('userId', '')
        pos_id = int(data.get('positionId', 0))
        if not uid or not pos_id:
            return jsonify({'success': False, 'error': 'userId ve positionId gerekli'}), 400

        db = get_db()
        row = db.execute("SELECT * FROM auto_positions WHERE id=? AND user_id=? AND status='open'", (pos_id, uid)).fetchone()
        db.close()
        if not row:
            return jsonify({'success': False, 'error': 'Pozisyon bulunamadi'}), 404

        sym = row['symbol']
        stock = _cget(_stock_cache, sym)
        cur_price = float(stock.get('price', 0)) if stock else float(row['entry_price'])
        _auto_close_position(pos_id, cur_price, "Manuel kapanis")
        _auto_log_trade(uid, sym, 'SELL_MANUAL', cur_price, float(row['quantity']),
                       "Kullanici tarafindan manuel kapatildi", 0, 0, pos_id)
        return jsonify({'success': True, 'message': f'{sym} pozisyonu kapatildi @ {cur_price:.2f}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
