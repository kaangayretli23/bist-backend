"""
BIST Pro - Auto Trading Engine Module
"""
import threading, traceback, time
from datetime import datetime
try:
    from zoneinfo import ZoneInfo
    _TZ_IST = ZoneInfo('Europe/Istanbul')
except Exception:
    _TZ_IST = None


def _now_ist():
    """Europe/Istanbul saatinde timezone-aware datetime. Sunucu UTC'de olsa bile BIST takvimine gore."""
    if _TZ_IST:
        return datetime.now(_TZ_IST)
    return datetime.now()


def _today_ist():
    return _now_ist().strftime('%Y-%m-%d')

from flask import jsonify, request
from config import (
    get_db, USE_POSTGRES, PG_OK, sf, safe_dict,
    _lock, _stock_cache, _hist_cache, _cget, _cget_hist, _cset, _get_stocks, app
)
from trade_plans import calc_trade_plan
from auth_middleware import require_user

# =====================================================================
# AUTO TRADING ENGINE — Otomatik Alim-Satim Motoru
# =====================================================================

_auto_trade_lock = threading.Lock()


def _row_get(row, key, default=None):
    """Sqlite Row / dict / Postgres row'dan güvenli get — kolon yoksa default döner."""
    try:
        val = row[key]
        return val if val is not None else default
    except (KeyError, IndexError, TypeError):
        return default

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
                'panicSellEnabled': bool(_row_get(row, 'panic_sell_enabled', 0)),
                'panicDropPct': float(_row_get(row, 'panic_drop_pct', 2.0) or 2.0),
                'panicWindowMin': int(_row_get(row, 'panic_window_min', 5) or 5),
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
    """Bugün açılmış pozisyon sayısı (sadece BUY).
    SELL/SELL_TRAIL/SELL_TP dahil DEĞİL — SL/TP tetiklemelerinin sınırlanması istenmiyor.
    maxDailyTrades = gün içinde kullanıcıya uyarı/oto-open limiti (yeni giriş)."""
    try:
        db = get_db()
        today = _today_ist()
        if USE_POSTGRES and PG_OK:
            row = db.execute("SELECT COUNT(*) as cnt FROM auto_trades WHERE user_id=%s AND action='BUY' AND (created_at AT TIME ZONE 'Europe/Istanbul')::date=%s::date", (user_id, today)).fetchone()
        else:
            row = db.execute("SELECT COUNT(*) as cnt FROM auto_trades WHERE user_id=? AND action='BUY' AND date(created_at)=?", (user_id, today)).fetchone()
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
    """Yeni pozisyon ac. Capital/max_positions yarış koşulu için son-an DB kontrolü yapar."""
    try:
        db = get_db()
        # Yarış koşulu önlemi: INSERT'ten hemen önce mevcut açık pozisyon sayısını + toplam kullanılan sermayeyi doğrula
        try:
            cfg_row = db.execute(
                "SELECT max_positions, capital FROM auto_config WHERE user_id=?", (user_id,)
            ).fetchone()
            if cfg_row:
                max_pos = int(cfg_row['max_positions'] or 5)
                capital = float(cfg_row['capital'] or 0)
                count_row = db.execute(
                    "SELECT COUNT(*) AS c, COALESCE(SUM(entry_price*quantity), 0) AS used "
                    "FROM auto_positions WHERE user_id=? AND status='open'",
                    (user_id,)
                ).fetchone()
                cur_count = int(count_row['c']) if count_row else 0
                used_capital = float(count_row['used']) if count_row else 0.0
                cost = price * quantity
                if cur_count >= max_pos:
                    print(f"[AUTO-TRADE] Pozisyon açılmadı — max_positions sınırına ulaşıldı ({cur_count}/{max_pos})")
                    db.close()
                    return 0
                if capital > 0 and (used_capital + cost) > capital * 1.001:  # %0.1 tolerans
                    print(f"[AUTO-TRADE] Pozisyon açılmadı — sermaye yetersiz "
                          f"(kullanılan={used_capital:.0f} + yeni={cost:.0f} > kapital={capital:.0f})")
                    db.close()
                    return 0
        except Exception as _guard_err:
            print(f"[AUTO-TRADE] Son-an kontrol hatası (devam ediliyor): {_guard_err}")
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
        now_str = _now_ist().isoformat()
        db.execute(
            "UPDATE auto_positions SET status='closed', closed_at=?, close_price=?, close_reason=?, pnl=?, pnl_pct=? WHERE id=?",
            (now_str, close_price, reason, round(pnl, 2), round(pnl_pct, 2), position_id)
        )
        db.commit()
        db.close()
        print(f"[AUTO-TRADE] Pozisyon kapatildi #{position_id}: {row['symbol']} PnL={pnl:.2f} TL ({pnl_pct:.1f}%) - {reason}")
        # Telegram bildirimi
        try:
            from routes_telegram import send_position_closed_notification
            send_position_closed_notification(row['symbol'], close_price, round(pnl, 2), round(pnl_pct, 2), reason)
        except Exception:
            pass
    except Exception as e:
        print(f"[AUTO-TRADE] Pozisyon kapatma hatasi: {e}")

def _auto_partial_close(position_id, sell_qty, price, reason, clear_tp_field=None):
    """Pozisyonun bir kısmını sat (TP1/TP2 kademeli kâr alma).
    sell_qty: satılacak miktar. Kalan quantity DB'de güncellenir.
    clear_tp_field: 'take_profit1' veya 'take_profit2' → 0'a set eder (tekrar tetiklemesin)
    """
    try:
        db = get_db()
        row = db.execute("SELECT * FROM auto_positions WHERE id=?", (position_id,)).fetchone()
        if not row:
            db.close()
            return
        entry = float(row['entry_price'])
        cur_qty = float(row['quantity'])
        actual_sell = min(sell_qty, cur_qty)
        if actual_sell <= 0:
            db.close()
            return
        remaining = round(cur_qty - actual_sell, 2)
        pnl = (price - entry) * actual_sell
        pnl_pct = ((price - entry) / entry * 100) if entry > 0 else 0

        if remaining <= 0.01:
            # Kalan yok — tam kapat
            from auto_trader import _auto_close_position
            db.close()
            _auto_close_position(position_id, price, reason)
            return

        # Quantity azalt + tetiklenen TP'yi sıfırla (SQL whitelist: kullanıcı input'undan bağımsız)
        _ALLOWED_CLEAR_TP = {
            'take_profit1': "UPDATE auto_positions SET quantity=?, take_profit1=0 WHERE id=?",
            'take_profit2': "UPDATE auto_positions SET quantity=?, take_profit2=0 WHERE id=?",
        }
        if clear_tp_field and clear_tp_field in _ALLOWED_CLEAR_TP:
            db.execute(_ALLOWED_CLEAR_TP[clear_tp_field], [remaining, position_id])
        else:
            db.execute("UPDATE auto_positions SET quantity=? WHERE id=?", [remaining, position_id])
        db.commit()
        db.close()
        print(f"[AUTO-TRADE] Kısmi satış #{position_id}: {row['symbol']} "
              f"{actual_sell:.2f} lot @ {price:.2f}, kalan={remaining:.2f}, "
              f"PnL={pnl:.2f} ({pnl_pct:.1f}%) - {reason}")
        try:
            from routes_telegram import send_position_closed_notification
            send_position_closed_notification(
                row['symbol'], price, round(pnl, 2), round(pnl_pct, 2),
                f"{reason} (kısmi — kalan {remaining:.2f} lot)")
        except Exception:
            pass
    except Exception as e:
        print(f"[AUTO-TRADE] Kısmi satış hatası: {e}")


def _auto_update_trailing(position_id, new_trailing, new_highest):
    """Trailing stop + highest price güncelle (onay sonrası çağrılır)"""
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


def _auto_update_highest_price(position_id, new_highest):
    """Sadece highest_price güncelle (trailing onay beklerken zirveyi kaçırma)"""
    try:
        db = get_db()
        db.execute(
            "UPDATE auto_positions SET highest_price=? WHERE id=?",
            (new_highest, position_id)
        )
        db.commit()
        db.close()
    except Exception as e:
        print(f"[AUTO-TRADE] Highest price guncelleme hatasi: {e}")


@app.route('/api/auto-trade/config', methods=['GET', 'POST'])
@require_user
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
                allowed_symbols=?, blocked_symbols=?, max_daily_trades=?,
                panic_sell_enabled=?, panic_drop_pct=?, panic_window_min=?
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
                 int(data.get('panicSellEnabled', _row_get(existing, 'panic_sell_enabled', 0))),
                 float(data.get('panicDropPct', _row_get(existing, 'panic_drop_pct', 2.0) or 2.0)),
                 int(data.get('panicWindowMin', _row_get(existing, 'panic_window_min', 5) or 5)),
                 uid))
        else:
            db.execute("""INSERT INTO auto_config
                (user_id, enabled, capital, max_positions, risk_per_trade,
                 min_score, min_confidence, trade_style,
                 stop_loss_pct, take_profit_pct, trailing_stop, trailing_pct,
                 allowed_symbols, blocked_symbols, max_daily_trades,
                 panic_sell_enabled, panic_drop_pct, panic_window_min)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (uid,
                 int(data.get('enabled', 0)),
                 float(data.get('capital', 100000)),
                 int(data.get('maxPositions', 5)),
                 float(data.get('riskPerTrade', 2.0)),
                 float(data.get('minScore', 5.0)),
                 float(data.get('minConfidence', 60)),
                 data.get('tradeStyle', 'swing'),
                 float(data.get('stopLossPct', 3.0)),
                 float(data.get('takeProfitPct', 6.0)),
                 int(data.get('trailingStop', 1)),
                 float(data.get('trailingPct', 2.0)),
                 data.get('allowedSymbols', ''),
                 data.get('blockedSymbols', ''),
                 int(data.get('maxDailyTrades', 3)),
                 int(data.get('panicSellEnabled', 0)),
                 float(data.get('panicDropPct', 2.0)),
                 int(data.get('panicWindowMin', 5))))
        db.commit()
        db.close()
        return jsonify({'success': True, 'message': 'Konfigürasyon kaydedildi'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auto-trade/toggle', methods=['POST'])
@require_user
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
            db.execute("""INSERT INTO auto_config
                (user_id, enabled, capital, max_positions, risk_per_trade,
                 min_score, min_confidence, trade_style,
                 stop_loss_pct, take_profit_pct, trailing_stop, trailing_pct,
                 allowed_symbols, blocked_symbols, max_daily_trades)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (uid, enabled, 100000, 5, 2.0, 5.0, 60.0, 'swing',
                 3.0, 6.0, 1, 2.0, '', '', 3))
            db.commit()
        db.close()
        status = 'aktif' if enabled else 'deaktif'
        print(f"[AUTO-TRADE] {uid}: {status}")
        return jsonify({'success': True, 'enabled': bool(enabled), 'message': f'Oto-trade {status}'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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
        total_pnl = 0
        for pos in positions:
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
                'symbol': r['symbol'], 'entryPrice': float(r['entry_price']),
                'closePrice': float(r['close_price'] or 0), 'pnl': float(r['pnl'] or 0),
                'pnlPct': float(r['pnl_pct'] or 0), 'reason': r['close_reason'],
                'closedAt': r['closed_at'],
            } for r in closed[:20]],
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/auto-trade/trades')
@require_user
def auto_trade_trades():
    """Trade gecmisini getir"""
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400
        # Limit bound: DoS onlemi — 1..500 araligina clamp
        try:
            limit = min(max(int(request.args.get('limit', 50)), 1), 500)
        except (TypeError, ValueError):
            limit = 50
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

@app.route('/api/auto-trade/run', methods=['POST'])
@require_user
def auto_trade_run_now():
    """Motoru hemen tetikle (test/debug icin)"""
    try:
        from auto_trader_engine import _auto_engine_cycle
        _auto_engine_cycle()
        return jsonify({'success': True, 'message': 'Motor calistirildi, loglari kontrol edin'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auto-trade/debug')
@require_user
def auto_trade_debug():
    """Sistemin mevcut durumunu raporla"""
    try:
        uid = request.args.get('userId', '')

        # Cache durumu
        hist_count = len(_hist_cache)
        stock_count = len(_stock_cache)

        # DB'de aktif kullanicilar
        db = get_db()
        active_users = db.execute("SELECT user_id, enabled, min_score, min_confidence FROM auto_config WHERE enabled=1").fetchall()
        db.close()

        # Kullanicinin config'i
        cfg = _auto_get_config(uid) if uid else None

        # Hist cache olan hisselerden ornek AL sinyalleri (ilk 5)
        sample_signals = []
        if uid and cfg:
            try:
                from indicators import calc_all_indicators
                from signals import calc_recommendation as _calc_rec
                _tf_map = {'daily': 'daily', 'swing': 'weekly', 'monthly': 'monthly'}
                tf_key = _tf_map.get(cfg.get('tradeStyle', 'swing'), 'weekly')
                stocks = _get_stocks()
                checked = 0
                for s in stocks[:30]:  # ilk 30 hisse kontrol et
                    sym = s.get('code', '')
                    hist = _cget_hist(f"{sym}_1y")
                    if hist is None or len(hist) < 30:
                        try:
                            from data_fetcher import _fetch_hist_df

                            hist = _fetch_hist_df(sym, '1y')
                            if hist is not None and len(hist) >= 30:
                                _cset(_hist_cache, f"{sym}_1y", hist)
                            else:
                                continue
                        except Exception:
                            continue
                    checked += 1
                    try:
                        cp = float(s.get('price', 0)) or float(hist['Close'].iloc[-1])
                        indics = calc_all_indicators(hist, cp)
                        recs = _calc_rec(hist, indics)
                        rec = recs.get(tf_key) or recs.get('weekly', {})
                        sample_signals.append({
                            'symbol': sym,
                            'action': rec.get('action', '?'),
                            'score': rec.get('score', 0),
                            'confidence': rec.get('confidence', 0),
                        })
                    except Exception as e:
                        sample_signals.append({'symbol': sym, 'error': str(e)})
                    if len(sample_signals) >= 5:
                        break
            except Exception as e:
                sample_signals = [{'error': str(e)}]

        return jsonify(safe_dict({
            'success': True,
            'cache': {
                'stockCount': stock_count,
                'histCount': hist_count,
            },
            'activeUsers': [dict(r) for r in active_users],
            'userConfig': cfg,
            'sampleSignals': sample_signals,
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auto-trade/reset', methods=['POST'])
@require_user
def auto_trade_reset():
    """Tüm geçmiş işlemleri ve PnL'i sıfırla. Sermaye ayarı korunur."""
    try:
        data = request.json or {}
        uid = data.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400

        db = get_db()
        # Açık pozisyon sayısını logla
        open_count = db.execute(
            "SELECT COUNT(*) as c FROM auto_positions WHERE user_id=? AND status='open'", (uid,)
        ).fetchone()['c']
        pos_count = db.execute(
            "SELECT COUNT(*) as c FROM auto_positions WHERE user_id=?", (uid,)
        ).fetchone()['c']
        trade_count = db.execute(
            "SELECT COUNT(*) as c FROM auto_trades WHERE user_id=?", (uid,)
        ).fetchone()['c']

        # Tüm pozisyon ve trade geçmişini sil
        db.execute("DELETE FROM auto_positions WHERE user_id=?", (uid,))
        db.execute("DELETE FROM auto_trades WHERE user_id=?", (uid,))
        db.commit()
        db.close()

        print(f"[AUTO-TRADE] {uid} sıfırlandı: {pos_count} pozisyon, {trade_count} işlem silindi ({open_count} açık pozisyon vardı)")
        return jsonify({
            'success': True,
            'message': 'Oto-trade gecmisi sıfırlandı. Sermaye ayarı korundu.',
            'deletedPositions': pos_count,
            'deletedTrades': trade_count,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auto-trade/close', methods=['POST'])
@require_user
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


@app.route('/api/auto-trade/position/edit', methods=['POST'])
@require_user
def auto_trade_edit_position():
    """Açık pozisyonun giriş fiyatı/adedini manuel düzelt (Midas'ta gerçekleşen doluma göre).
    Body: { userId, positionId, entryPrice?, quantity?, adjustSlTp? (bool) }
    adjustSlTp True ise SL/TP/trailing/highest orantılı kaydırılır.
    """
    try:
        data = request.json or {}
        uid = data.get('userId', '')
        pos_id = int(data.get('positionId', 0))
        if not uid or not pos_id:
            return jsonify({'success': False, 'error': 'userId ve positionId gerekli'}), 400

        db = get_db()
        row = db.execute(
            "SELECT * FROM auto_positions WHERE id=? AND user_id=? AND status='open'",
            (pos_id, uid)
        ).fetchone()
        if not row:
            db.close()
            return jsonify({'success': False, 'error': 'Açık pozisyon bulunamadı'}), 404

        old_entry = float(row['entry_price'])
        old_qty   = float(row['quantity'])
        new_entry = float(data.get('entryPrice', old_entry))
        new_qty   = float(data.get('quantity', old_qty))
        adjust    = bool(data.get('adjustSlTp', True))

        if new_entry <= 0 or new_qty <= 0:
            db.close()
            return jsonify({'success': False, 'error': 'Geçersiz fiyat veya adet'}), 400

        new_sl = float(row['stop_loss'] or 0)
        new_tp1 = float(row['take_profit1'] or 0)
        new_tp2 = float(row['take_profit2'] or 0)
        new_tp3 = float(row['take_profit3'] or 0)
        new_trail = float(row['trailing_stop'] or 0)
        new_high = float(row['highest_price'] or new_entry)

        if adjust and old_entry > 0 and abs(new_entry - old_entry) > 1e-6:
            ratio = new_entry / old_entry
            new_sl    = round(new_sl * ratio, 2)    if new_sl > 0    else new_sl
            new_tp1   = round(new_tp1 * ratio, 2)   if new_tp1 > 0   else new_tp1
            new_tp2   = round(new_tp2 * ratio, 2)   if new_tp2 > 0   else new_tp2
            new_tp3   = round(new_tp3 * ratio, 2)   if new_tp3 > 0   else new_tp3
            new_trail = round(new_trail * ratio, 2) if new_trail > 0 else new_trail
            # highest_price: yeni giriş fiyatına sıfırla (gelecek trailing güncellemelerine referans)
            new_high = new_entry

        db.execute(
            """UPDATE auto_positions SET
               entry_price=?, quantity=?, stop_loss=?,
               take_profit1=?, take_profit2=?, take_profit3=?,
               trailing_stop=?, highest_price=?
               WHERE id=?""",
            (new_entry, new_qty, new_sl, new_tp1, new_tp2, new_tp3, new_trail, new_high, pos_id)
        )
        db.commit()
        sym = row['symbol']
        db.close()

        _auto_log_trade(
            uid, sym, 'EDIT', new_entry, new_qty,
            f"Manuel düzelt: entry {old_entry:.2f}→{new_entry:.2f}, qty {old_qty}→{new_qty}"
            + (f", SL/TP orantılı kaydırıldı ({ratio:.4f})" if adjust and old_entry > 0 and abs(new_entry - old_entry) > 1e-6 else ""),
            0, 0, pos_id
        )
        return jsonify({'success': True, 'message': f'{sym} güncellendi',
                        'position': {
                            'entryPrice': new_entry, 'quantity': new_qty,
                            'stopLoss': new_sl, 'takeProfit1': new_tp1,
                            'takeProfit2': new_tp2, 'takeProfit3': new_tp3,
                            'trailingStop': new_trail, 'highestPrice': new_high,
                        }})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
