"""
BIST Pro - Auto Trader Pozisyon Endpoint'leri
Pozisyon kapatma/ekleme/reopen/edit HTTP endpointleri.
auto_trader_routes.py'dan ayrıştırıldı (600 satır kuralı).

backend.py `import auto_trader_routes_positions` ile yüklenir;
@app.route decoratorları importta rotaları Flask app'e kaydeder.
"""
from flask import jsonify, request
from config import app, get_db, _stock_cache, _cget
from auth_middleware import require_user
from auto_trader import (
    _auto_get_config,
    _auto_log_trade, _auto_close_position, _auto_partial_close,
)


@app.route('/api/auto-trade/close', methods=['POST'])
@require_user
def auto_trade_manual_close():
    """Manuel pozisyon kapatma.
    Body: { userId, positionId, price? (TL, opsiyonel — verilmezse cache fiyat), quantity? (opsiyonel — verilmezse tam kapat) }
    """
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
        pos_qty = float(row['quantity'])

        # Fiyat: override verilmişse onu kullan, yoksa cache'ten, o da yoksa entry'den
        price_raw = data.get('price', None)
        if price_raw is not None and str(price_raw).strip() != '':
            try:
                close_price = float(price_raw)
            except (TypeError, ValueError):
                return jsonify({'success': False, 'error': 'Gecersiz fiyat'}), 400
            if close_price <= 0:
                return jsonify({'success': False, 'error': 'Fiyat 0 dan buyuk olmali'}), 400
        else:
            stock = _cget(_stock_cache, sym)
            close_price = float(stock.get('price', 0)) if stock else 0
            if close_price <= 0:
                close_price = float(row['entry_price'])

        # Adet: override verilmişse onu kullan, yoksa tam kapat
        qty_raw = data.get('quantity', None)
        if qty_raw is not None and str(qty_raw).strip() != '':
            try:
                sell_qty = float(qty_raw)
            except (TypeError, ValueError):
                return jsonify({'success': False, 'error': 'Gecersiz adet'}), 400
            if sell_qty <= 0:
                return jsonify({'success': False, 'error': 'Adet 0 dan buyuk olmali'}), 400
            if sell_qty > pos_qty + 0.01:
                return jsonify({'success': False, 'error': f'Adet pozisyon miktarindan fazla ({pos_qty})'}), 400
        else:
            sell_qty = pos_qty

        # Erken/duygusal kapanis koruma — confirm bayragi yoksa, pozisyon SL'ye
        # yari mesafe gelmediyse veya TP1'in yarisini gecmediyse uyari don.
        # Mantik: bugun ARCLK vakasinda -%1'de elini cektin; sistem -%3'te SL'di.
        confirm = bool(data.get('confirm', False))
        if not confirm:
            entry = float(row['entry_price'] or 0)
            sl = float(row['stop_loss'] or 0)
            tp1 = float(row['take_profit1'] or 0)
            if entry > 0:
                pnl_pct = (close_price - entry) / entry * 100
                sl_pct = ((entry - sl) / entry * 100) if sl > 0 else 0
                tp1_pct = ((tp1 - entry) / entry * 100) if tp1 > 0 else 0
                # Zarardayken: SL'ye yari mesafe gelmediyse uyar
                if pnl_pct < 0 and sl_pct > 0 and abs(pnl_pct) < sl_pct * 0.5:
                    return jsonify({
                        'success': False,
                        'needsConfirm': True,
                        'warning': (
                            f"⚠️ Erken cikis: pozisyon zararda %{pnl_pct:.2f}, "
                            f"otomatik SL %{sl_pct:.2f}'de ({sl:.2f} TL).\n"
                            f"Sistem henuz SL'ye %{sl_pct - abs(pnl_pct):.2f} mesafede. "
                            f"Disipline guvenip beklemek mi, simdi kapatmak mi? "
                            f"Devam icin tekrar Onay'a bas."
                        )
                    }), 200
                # Az kardayken: TP1'in yarisina gelmediyse uyar
                if pnl_pct > 0 and tp1_pct > 0 and pnl_pct < tp1_pct * 0.5:
                    return jsonify({
                        'success': False,
                        'needsConfirm': True,
                        'warning': (
                            f"⚠️ Erken kar realize: pozisyon karda +%{pnl_pct:.2f}, "
                            f"TP1 +%{tp1_pct:.2f}'de ({tp1:.2f} TL).\n"
                            f"Hedefin yarisindan azindasin. Devam icin tekrar Onay'a bas."
                        )
                    }), 200

        is_partial = sell_qty < pos_qty - 0.01
        if is_partial:
            _auto_partial_close(pos_id, sell_qty, close_price, "Manuel kismi kapanis")
            _auto_log_trade(uid, sym, 'SELL_MANUAL', close_price, sell_qty,
                           f"Kullanici tarafindan kismi manuel satis ({sell_qty}/{pos_qty})", 0, 0, pos_id)
            return jsonify({'success': True, 'message': f'{sym} {sell_qty} adet kapatildi @ {close_price:.2f}', 'partial': True})

        _auto_close_position(pos_id, close_price, "Manuel kapanis")
        _auto_log_trade(uid, sym, 'SELL_MANUAL', close_price, sell_qty,
                       "Kullanici tarafindan manuel kapatildi", 0, 0, pos_id)
        return jsonify({'success': True, 'message': f'{sym} pozisyonu kapatildi @ {close_price:.2f}', 'partial': False})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


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


@app.route('/api/auto-trade/close-all', methods=['POST'])
@require_user
def auto_trade_close_all():
    """C2 — Acil çıkış: kullanıcının tüm açık pozisyonlarını piyasa fiyatından kapat.
    Body: { userId, reason?: str (default 'Acil çıkış') }
    Gerçek emir Midas'tan elle yapılacak, bu sadece DB temizliği + tracking yapar.
    """
    try:
        data = request.json or {}
        uid = data.get('userId') or request.args.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400
        reason = data.get('reason') or 'Acil çıkış (close-all)'

        db = get_db()
        rows = db.execute(
            "SELECT id, symbol, quantity, entry_price FROM auto_positions "
            "WHERE user_id=? AND status='open'",
            (uid,)
        ).fetchall()
        db.close()

        if not rows:
            return jsonify({'success': True, 'message': 'Açık pozisyon yok', 'closed': 0})

        closed = []
        failed = []
        for r in rows:
            sym = r['symbol']
            pid = r['id']
            qty = float(r['quantity'])
            # Anlik fiyat al
            cp = 0.0
            try:
                from realtime_prices import get_price as _rt_get
                _rp = _rt_get(sym)
                if _rp and _rp > 0:
                    cp = float(_rp)
            except Exception:
                pass
            if cp <= 0:
                from config import _cget, _stock_cache
                _stk = _cget(_stock_cache, sym)
                cp = float(_stk.get('price', 0)) if _stk else 0
            if cp <= 0:
                failed.append({'symbol': sym, 'reason': 'fiyat alinamadi'})
                continue
            try:
                _auto_close_position(pid, cp, reason)
                _auto_log_trade(uid, sym, 'SELL_MANUAL', cp, qty,
                                f"Close-All: {reason}", 0, 0, pid)
                closed.append({'symbol': sym, 'qty': qty, 'price': cp})
            except Exception as ce:
                failed.append({'symbol': sym, 'reason': str(ce)})

        # Telegram ozet
        try:
            from routes_telegram import send_telegram
            _msg_lines = [f"🚨 <b>ACİL ÇIKIŞ — {len(closed)} pozisyon kapandı</b>"]
            for c in closed:
                _msg_lines.append(f"  {c['symbol']}: {c['qty']:.0f} adet @ {c['price']:.2f}")
            if failed:
                _msg_lines.append(f"\n⚠️ {len(failed)} pozisyon kapatılamadı:")
                for f in failed:
                    _msg_lines.append(f"  {f['symbol']}: {f['reason']}")
            _msg_lines.append(f"\nSebep: {reason}")
            _msg_lines.append("⚠️ Gerçek emirlerini Midas'tan kontrol et!")
            send_telegram('\n'.join(_msg_lines))
        except Exception:
            pass

        return jsonify({
            'success': True,
            'closed': len(closed),
            'closedDetails': closed,
            'failed': failed,
            'message': f"{len(closed)} pozisyon kapatıldı, {len(failed)} başarısız"
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auto-trade/position/add', methods=['POST'])
@require_user
def auto_trade_add_position():
    """Kullanicinin elindeki hisseyi manuel olarak oto-trade takibine ekle.
    Kullanim: Midas'ta zaten alinmis pozisyonu sisteme kaydetmek icin. Sermaye kontrolu atlanir
    (kullanici zaten odemis), ama max_positions ve 'ayni sembol acik' kontrolleri calisir.
    Body: { userId, symbol, entryPrice, quantity, stopLoss?, takeProfit1?, takeProfit2?, takeProfit3?, trailingStop? }
    SL/TP verilmezse config'deki stopLossPct/takeProfitPct/trailingPct kullanilir.
    """
    try:
        data = request.json or {}
        uid = data.get('userId', '')
        sym = (data.get('symbol', '') or '').strip().upper()
        if not uid or not sym:
            return jsonify({'success': False, 'error': 'userId ve symbol gerekli'}), 400

        try:
            entry = float(data.get('entryPrice', 0))
            qty = float(data.get('quantity', 0))
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Gecersiz fiyat/adet'}), 400
        if entry <= 0 or qty <= 0:
            return jsonify({'success': False, 'error': 'Fiyat ve adet 0 dan buyuk olmali'}), 400

        cfg = _auto_get_config(uid)
        if not cfg:
            return jsonify({'success': False, 'error': 'Oto-trade config bulunamadi'}), 400

        db = get_db()
        # Ayni sembolde acik pozisyon varsa engelle
        existing = db.execute(
            "SELECT id, quantity FROM auto_positions WHERE user_id=? AND symbol=? AND status='open'",
            (uid, sym)
        ).fetchone()
        if existing:
            db.close()
            return jsonify({
                'success': False,
                'error': f'{sym} icin zaten acik pozisyon var (#{existing["id"]}, {existing["quantity"]} lot). "Düzelt" ile adet/fiyat güncelleyebilirsin.'
            }), 400

        # Max positions kontrolu
        open_count = int(db.execute(
            "SELECT COUNT(*) as c FROM auto_positions WHERE user_id=? AND status='open'", (uid,)
        ).fetchone()['c'])
        if open_count >= int(cfg.get('maxPositions', 5)):
            db.close()
            return jsonify({
                'success': False,
                'error': f'Acik pozisyon limitine ulasildi ({open_count}/{cfg["maxPositions"]}). Limiti artirin veya bir pozisyon kapatin.'
            }), 400

        # SL/TP/trailing — override yoksa config oranlarindan hesapla
        def _opt_float(key):
            v = data.get(key)
            if v is None or str(v).strip() == '':
                return None
            try:
                fv = float(v)
                return fv if fv > 0 else None
            except (TypeError, ValueError):
                return None

        sl = _opt_float('stopLoss') or round(entry * (1 - float(cfg.get('stopLossPct', 3)) / 100), 2)
        tp_pct = float(cfg.get('takeProfitPct', 6))
        tp1 = _opt_float('takeProfit1') or round(entry * (1 + tp_pct / 100), 2)
        tp2 = _opt_float('takeProfit2') or round(entry * (1 + tp_pct * 1.5 / 100), 2)
        tp3 = _opt_float('takeProfit3') or round(entry * (1 + tp_pct * 2.0 / 100), 2)
        trail_override = _opt_float('trailingStop')
        if trail_override is not None:
            trailing_sl = trail_override
        elif cfg.get('trailingStop'):
            trailing_sl = round(entry * (1 - float(cfg.get('trailingPct', 2)) / 100), 2)
        else:
            trailing_sl = 0

        db.execute(
            """INSERT INTO auto_positions
               (user_id, symbol, side, entry_price, quantity, stop_loss, take_profit1, take_profit2, take_profit3, trailing_stop, highest_price)
               VALUES (?,?,'long',?,?,?,?,?,?,?,?)""",
            (uid, sym, entry, qty, sl, tp1, tp2, tp3, trailing_sl, entry)
        )
        db.commit()
        row = db.execute("SELECT last_insert_rowid() as mid").fetchone()
        pos_id = int(row['mid']) if row else 0
        db.close()

        _auto_log_trade(
            uid, sym, 'BUY', entry, qty,
            f"Manuel pozisyon ekleme (kullanici zaten satin almis) | SL={sl:.2f}, TP1={tp1:.2f}",
            0, 0, pos_id
        )
        # Portfoye senkronla
        try:
            from auto_trader_sync import _sync_portfolio_buy
            _sync_portfolio_buy(uid, sym, qty, entry)
        except Exception as _ps_err:
            print(f"[AUTO-TRADE] Manuel ekle portfoy sync hatasi: {_ps_err}")
        # Gercek zamanli fiyat takibine abone ol
        try:
            from realtime_prices import subscribe as _rt_sub
            _rt_sub(sym)
        except Exception:
            pass

        return jsonify({
            'success': True,
            'message': f'{sym} pozisyonu eklendi ({qty} lot @ {entry:.2f}, SL {sl:.2f}, TP1 {tp1:.2f})',
            'positionId': pos_id,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/auto-trade/reopen', methods=['POST'])
@require_user
def auto_trade_reopen_position():
    """Kapatilmis pozisyonu geri ac (yanlislikla/otomatik satis telafisi).
    Body: { userId, positionId }
    Pozisyonu 'open' statusune dondurur, SL cooldown varsa temizler, REOPEN trade logu atar.
    """
    try:
        data = request.json or {}
        uid = data.get('userId', '')
        pos_id = int(data.get('positionId', 0))
        if not uid or not pos_id:
            return jsonify({'success': False, 'error': 'userId ve positionId gerekli'}), 400

        db = get_db()
        row = db.execute(
            "SELECT * FROM auto_positions WHERE id=? AND user_id=? AND status='closed'",
            (pos_id, uid)
        ).fetchone()
        if not row:
            db.close()
            return jsonify({'success': False, 'error': 'Kapatilmis pozisyon bulunamadi'}), 404

        sym = row['symbol']
        qty = float(row['quantity'])
        entry = float(row['entry_price'])
        prev_close_price = float(row['close_price'] or 0)
        prev_reason = row['close_reason'] or ''

        # Ayni sembolde zaten acik pozisyon varsa engelle (duplicate onle)
        existing = db.execute(
            "SELECT id FROM auto_positions WHERE user_id=? AND symbol=? AND status='open'",
            (uid, sym)
        ).fetchone()
        if existing:
            db.close()
            return jsonify({'success': False, 'error': f'{sym} icin zaten acik bir pozisyon var (#{existing["id"]}). Onu duzelt.'}), 400

        db.execute(
            "UPDATE auto_positions SET status='open', closed_at=NULL, close_price=NULL, "
            "close_reason=NULL, pnl=0, pnl_pct=0 WHERE id=?",
            (pos_id,)
        )
        db.commit()
        db.close()

        # SL cooldown varsa temizle (pozisyon tekrar acik olduguna gore yeniden taranabilmeli)
        try:
            from auto_trader_risk import _sl_cooldown
            from config import get_db as _get_db
            key = f"{uid}_{sym}"
            if key in _sl_cooldown:
                del _sl_cooldown[key]
            _db = _get_db()
            try:
                _db.execute("DELETE FROM sl_cooldown WHERE uid_sym=?", [key])
                _db.commit()
            finally:
                _db.close()
        except Exception as e:
            print(f"[AUTO-TRADE] Reopen SL cooldown temizleme uyarisi: {e}")

        # Portfoye geri ekle (kapatilirken silinmisti)
        try:
            from auto_trader_sync import _sync_portfolio_buy
            _sync_portfolio_buy(uid, sym, qty, entry)
        except Exception as _ps_err:
            print(f"[AUTO-TRADE] Reopen portfoy sync hatasi: {_ps_err}")

        _auto_log_trade(
            uid, sym, 'REOPEN', prev_close_price or entry, qty,
            f"Kapanan pozisyon geri alindi (onceki sebep: {prev_reason})",
            0, 0, pos_id
        )
        return jsonify({'success': True, 'message': f'{sym} pozisyonu geri acildi ({qty} lot @ {entry:.2f} giris)'})
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
        # Portfoy senkron: eski lot'u cikar, yeni lot'u ekle (avg_cost yeniden hesaplanir)
        try:
            from auto_trader_sync import _sync_portfolio_diff
            _sync_portfolio_diff(uid, sym, old_qty, old_entry, new_qty, new_entry)
        except Exception as _ps_err:
            print(f"[AUTO-TRADE] Edit portfoy sync hatasi: {_ps_err}")
        return jsonify({'success': True, 'message': f'{sym} güncellendi',
                        'position': {
                            'entryPrice': new_entry, 'quantity': new_qty,
                            'stopLoss': new_sl, 'takeProfit1': new_tp1,
                            'takeProfit2': new_tp2, 'takeProfit3': new_tp3,
                            'trailingStop': new_trail, 'highestPrice': new_high,
                        }})
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
