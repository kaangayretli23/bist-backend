"""
BIST Pro - Auto Trading Engine — Veri Katmanı
Tz yardımcıları, _auto_trade_lock, DB getter/mutator fonksiyonları.

HTTP endpointleri `auto_trader_routes.py` modülüne ayrıştırıldı (600 satır kuralı).
Engine cycle adımları ise `auto_trader_engine.py` / `_positions` / `_plan` / `_scanner`
modüllerine bölündü.

backend.py bu modülü `import auto_trader` ile yükler; routes için ayrıca
`import auto_trader_routes` eklenmiştir.
"""
import threading
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

from config import get_db, USE_POSTGRES, PG_OK

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
                'commissionPct': float(_row_get(row, 'commission_pct', 0) or 0),
                'bsmvPct': float(_row_get(row, 'bsmv_pct', 5) or 5),
                'tightTrailingPct': float(_row_get(row, 'tight_trailing_pct', 1.0) or 1.0),
                'maxPerSector': int(_row_get(row, 'max_per_sector', 2) or 2),
                'minTurnoverTL': float(_row_get(row, 'min_turnover_tl', 1_000_000) or 1_000_000),
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
                # Sermaye kontrolu: artik gercek alim Midas'ta yapildigi icin BLOK degil sadece UYARI
                if capital > 0 and (used_capital + cost) > capital * 1.001:
                    print(f"[AUTO-TRADE] UYARI — sermaye limiti asildi ama yine de aciliyor "
                          f"(kullanılan={used_capital:.0f} + yeni={cost:.0f} > kapital={capital:.0f}). "
                          "Risk takibi icin capital ayarini guncelleyin.")
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
        # Portfoy senkronu: gercek alim oldugu icin manuel portfoye de ekle
        try:
            from auto_trader_sync import _sync_portfolio_buy
            _sync_portfolio_buy(user_id, symbol, quantity, price)
        except Exception as _ps_err:
            print(f"[AUTO-TRADE] Portfoy BUY sync hatasi: {_ps_err}")
        # Realtime fiyat takibi: tüm açılış path'leri (scanner, plan, telegram approve) için ortak nokta
        try:
            from realtime_prices import subscribe as _rt_sub
            _rt_sub(symbol)
        except Exception:
            pass
        return pos_id
    except Exception as e:
        print(f"[AUTO-TRADE] Pozisyon acma hatasi: {e}")
        return 0

def _calc_trade_costs(user_id, notional_buy, notional_sell):
    """Komisyon + BSMV'yi hesapla (BUY ve SELL tarafinda ayri ayri).
    Midas BIST'te 0 — cfg bos/yoksa 0 dondurur. BSMV komisyon uzerinden alinir.
    """
    try:
        cfg = _auto_get_config(user_id) or {}
        c_pct = float(cfg.get('commissionPct', 0) or 0)
        b_pct = float(cfg.get('bsmvPct', 0) or 0)
        if c_pct <= 0:
            return 0.0
        comm = (notional_buy + notional_sell) * (c_pct / 100.0)
        bsmv = comm * (b_pct / 100.0)
        return round(comm + bsmv, 2)
    except Exception:
        return 0.0


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
        gross = (close_price - entry) * qty
        costs = _calc_trade_costs(row['user_id'], entry * qty, close_price * qty)
        pnl = gross - costs
        pnl_pct = ((close_price - entry) / entry * 100) if entry > 0 else 0
        if entry > 0 and qty > 0 and costs > 0:
            pnl_pct = (pnl / (entry * qty)) * 100
        now_str = _now_ist().isoformat()
        db.execute(
            "UPDATE auto_positions SET status='closed', closed_at=?, close_price=?, close_reason=?, pnl=?, pnl_pct=? WHERE id=?",
            (now_str, close_price, reason, round(pnl, 2), round(pnl_pct, 2), position_id)
        )
        db.commit()
        db.close()
        # Portfoy senkronu: SELL -> portfolios'tan dus
        try:
            from auto_trader_sync import _sync_portfolio_sell
            _sync_portfolio_sell(row['user_id'], row['symbol'], qty)
        except Exception as _ps_err:
            print(f"[AUTO-TRADE] Portfoy SELL sync hatasi: {_ps_err}")
        # Alert state cleanup (RAM+DB) — her durumda yapılır (kullanıcı bazlı)
        try:
            from realtime_prices import clear_alert_state as _rt_clear
            _rt_clear(row['symbol'], row['user_id'], position_id)
        except Exception:
            pass
        # Realtime takibi durdur — başka açık pozisyon yoksa
        try:
            from realtime_prices import unsubscribe as _rt_unsub
            db2 = get_db()
            try:
                rem = db2.execute(
                    "SELECT COUNT(*) AS c FROM auto_positions WHERE symbol=? AND status='open'",
                    (row['symbol'],)
                ).fetchone()
                if rem and int(rem['c']) == 0:
                    _rt_unsub(row['symbol'])
            finally:
                db2.close()
        except Exception:
            pass
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
        gross = (price - entry) * actual_sell
        costs = _calc_trade_costs(row['user_id'], entry * actual_sell, price * actual_sell)
        pnl = gross - costs
        pnl_pct = ((price - entry) / entry * 100) if entry > 0 else 0
        if entry > 0 and actual_sell > 0 and costs > 0:
            pnl_pct = (pnl / (entry * actual_sell)) * 100

        if remaining <= 0.01:
            # Kalan yok — tam kapat
            db.close()
            _auto_close_position(position_id, price, reason)
            return

        # Quantity azalt + tetiklenen TP'yi sıfırla (SQL whitelist: kullanıcı input'undan bağımsız)
        # TP1 tetiklenirse: SL'yi giriş fiyatına çek (break-even) — kar geri verilmesin.
        # Sadece mevcut SL girişin altındaysa yukarı taşı; trailing yukarı sürüklediyse dokunma.
        _BUMP_SL_TO_BE = clear_tp_field == 'take_profit1'
        _be_price = round(entry * 1.001, 2)  # %0.1 tampon
        _ALLOWED_CLEAR_TP = {
            'take_profit1': "UPDATE auto_positions SET quantity=?, take_profit1=0 WHERE id=?",
            'take_profit2': "UPDATE auto_positions SET quantity=?, take_profit2=0 WHERE id=?",
        }
        if clear_tp_field and clear_tp_field in _ALLOWED_CLEAR_TP:
            db.execute(_ALLOWED_CLEAR_TP[clear_tp_field], [remaining, position_id])
        else:
            db.execute("UPDATE auto_positions SET quantity=? WHERE id=?", [remaining, position_id])
        # Break-even SL: TP1 sonrası, mevcut SL girişin altındaysa yukarı çek
        _be_applied = False
        if _BUMP_SL_TO_BE:
            cur_sl = float(row['stop_loss'] or 0)
            if cur_sl < _be_price:
                db.execute("UPDATE auto_positions SET stop_loss=? WHERE id=?", (_be_price, position_id))
                _be_applied = True
        db.commit()
        db.close()
        # Portfoy senkronu: kismi SELL -> portfolios'tan actual_sell kadar dus
        try:
            from auto_trader_sync import _sync_portfolio_sell
            _sync_portfolio_sell(row['user_id'], row['symbol'], actual_sell)
        except Exception as _ps_err:
            print(f"[AUTO-TRADE] Portfoy partial SELL sync hatasi: {_ps_err}")
        _be_msg = f", SL→BE ({_be_price:.2f})" if _be_applied else ""
        print(f"[AUTO-TRADE] Kısmi satış #{position_id}: {row['symbol']} "
              f"{actual_sell:.2f} lot @ {price:.2f}, kalan={remaining:.2f}, "
              f"PnL={pnl:.2f} ({pnl_pct:.1f}%) - {reason}{_be_msg}")
        try:
            from routes_telegram import send_position_closed_notification
            _be_tail = f"\n🛡 SL break-even'a çekildi: {_be_price:.2f}" if _be_applied else ""
            send_position_closed_notification(
                row['symbol'], price, round(pnl, 2), round(pnl_pct, 2),
                f"{reason} (kısmi — kalan {remaining:.2f} lot){_be_tail}")
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
