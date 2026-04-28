"""
BIST Pro - Auto Trader HTTP Endpoints
config / capital-adjust / toggle / trades / decisions / run / debug / reset.
status + daily-summary -> auto_trader_routes_dashboard.py (600 satir kurali).

backend.py `import auto_trader_routes` satırı ile bu modülü yükler;
decoratorlar modül importunda çalışıp rotaları Flask app'e kaydeder.
"""
from flask import jsonify, request
from config import (
    app, get_db, safe_dict,
    _stock_cache, _hist_cache, _cget_hist, _cset, _get_stocks,
)
from auth_middleware import require_user
from auto_trader import _auto_get_config, _row_get


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
            # NOT: capital UPDATE'ten cikarildi — bakiye sadece /capital/adjust ile
            # degisir. Boylece "Kaydet" butonu yanlislikla bakiyeyi sifirlamaz.
            db.execute("""UPDATE auto_config SET
                enabled=?, max_positions=?, risk_per_trade=?,
                min_score=?, min_confidence=?, trade_style=?,
                stop_loss_pct=?, take_profit_pct=?, trailing_stop=?, trailing_pct=?,
                allowed_symbols=?, blocked_symbols=?, max_daily_trades=?,
                panic_sell_enabled=?, panic_drop_pct=?, panic_window_min=?,
                commission_pct=?, bsmv_pct=?, tight_trailing_pct=?, max_per_sector=?,
                min_turnover_tl=?
                WHERE user_id=?""",
                (int(data.get('enabled', existing['enabled'])),
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
                 float(data.get('commissionPct', _row_get(existing, 'commission_pct', 0) or 0)),
                 float(data.get('bsmvPct', _row_get(existing, 'bsmv_pct', 5) or 5)),
                 float(data.get('tightTrailingPct', _row_get(existing, 'tight_trailing_pct', 1.0) or 1.0)),
                 int(data.get('maxPerSector', _row_get(existing, 'max_per_sector', 2) or 2)),
                 float(data.get('minTurnoverTL', _row_get(existing, 'min_turnover_tl', 1_000_000) or 1_000_000)),
                 uid))
        else:
            db.execute("""INSERT INTO auto_config
                (user_id, enabled, capital, max_positions, risk_per_trade,
                 min_score, min_confidence, trade_style,
                 stop_loss_pct, take_profit_pct, trailing_stop, trailing_pct,
                 allowed_symbols, blocked_symbols, max_daily_trades,
                 panic_sell_enabled, panic_drop_pct, panic_window_min,
                 commission_pct, bsmv_pct, tight_trailing_pct, max_per_sector,
                 min_turnover_tl)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                 int(data.get('panicWindowMin', 5)),
                 float(data.get('commissionPct', 0)),
                 float(data.get('bsmvPct', 5)),
                 float(data.get('tightTrailingPct', 1.0)),
                 int(data.get('maxPerSector', 2)),
                 float(data.get('minTurnoverTL', 1_000_000))))
        db.commit()
        db.close()
        return jsonify({'success': True, 'message': 'Konfigürasyon kaydedildi'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auto-trade/capital/adjust', methods=['POST'])
@require_user
def auto_trade_capital_adjust():
    """Oto-trade sermayesini delta kadar artir/azalt. Diger ayarlar korunur.
    Body: { userId, delta (TL; pozitif=ekle, negatif=cikar), mode? ('add'|'set') }
    mode='set' verilirse delta yeni capital olur.
    """
    try:
        data = request.json or {}
        uid = data.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400
        try:
            delta = float(data.get('delta', 0))
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Gecersiz delta'}), 400
        mode = (data.get('mode') or 'add').lower()

        db = get_db()
        existing = db.execute("SELECT capital FROM auto_config WHERE user_id=?", (uid,)).fetchone()
        if not existing:
            db.close()
            return jsonify({'success': False, 'error': 'Oto-trade config bulunamadi'}), 404
        old = float(existing['capital'])
        new = delta if mode == 'set' else old + delta
        if new < 0:
            db.close()
            return jsonify({'success': False, 'error': f'Sermaye negatif olamaz (sonuc: {new:.0f})'}), 400
        db.execute("UPDATE auto_config SET capital=? WHERE user_id=?", (new, uid))
        db.commit()
        db.close()
        print(f"[AUTO-TRADE] {uid} sermaye: {old:.0f} -> {new:.0f} TL ({mode})")
        return jsonify({
            'success': True,
            'oldCapital': old, 'newCapital': new,
            'message': f'Sermaye güncellendi: {old:.0f} → {new:.0f} TL',
        })
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


@app.route('/api/auto-trade/decisions')
@require_user
def auto_trade_decisions():
    """Karar log'u — bugun veya verilen aralik icin scanner/plan kararlari.
    Query: userId, limit (1..500, default 200), since (ISO datetime opsiyonel),
           decision ('SKIP'|'BUY'|'PENDING' opsiyonel filtre)
    """
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400
        try:
            limit = min(max(int(request.args.get('limit', 200)), 1), 500)
        except (TypeError, ValueError):
            limit = 200
        decision_filter = (request.args.get('decision', '') or '').upper().strip()
        since = (request.args.get('since', '') or '').strip()

        sql = "SELECT * FROM auto_decisions WHERE user_id=?"
        params = [uid]
        if decision_filter in ('SKIP', 'BUY', 'PENDING'):
            sql += " AND decision=?"
            params.append(decision_filter)
        if since:
            sql += " AND created_at >= ?"
            params.append(since)
        else:
            # default: bugun (UTC tabanli, basit)
            from datetime import datetime as _dt
            sql += " AND created_at >= ?"
            params.append(_dt.now().strftime('%Y-%m-%d 00:00:00'))
        sql += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        db = get_db()
        rows = db.execute(sql, tuple(params)).fetchall()
        # Reason ozet (bugunku)
        summary_rows = db.execute(
            "SELECT decision, reason, COUNT(*) AS c FROM auto_decisions "
            "WHERE user_id=? AND created_at >= ? GROUP BY decision, reason ORDER BY c DESC",
            (uid, params[-2] if since else params[-2])
        ).fetchall() if rows else []
        db.close()

        items = [{
            'id': r['id'], 'symbol': r['symbol'], 'tf': r['timeframe'] or '',
            'decision': r['decision'], 'reason': r['reason'] or '',
            'detail': r['detail'] or '',
            'price': float(r['price'] or 0), 'score': float(r['score'] or 0),
            'confidence': float(r['confidence'] or 0),
            'createdAt': r['created_at'],
        } for r in rows]
        summary = [{'decision': r['decision'], 'reason': r['reason'] or '', 'count': int(r['c'])}
                   for r in summary_rows]
        return jsonify(safe_dict({'success': True, 'decisions': items, 'summary': summary, 'count': len(items)}))
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

