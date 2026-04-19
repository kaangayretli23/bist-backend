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
