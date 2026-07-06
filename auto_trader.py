"""
BIST Pro - Auto Trading Engine — Veri Katmanı
Tz yardımcıları, _auto_trade_lock, DB getter/mutator fonksiyonları.

HTTP endpointleri `auto_trader_routes.py` modülüne ayrıştırıldı (600 satır kuralı).
Engine cycle adımları ise `auto_trader_engine.py` / `_positions` / `_plan` / `_scanner`
modüllerine bölündü.

backend.py bu modülü `import auto_trader` ile yükler; routes için ayrıca
`import auto_trader_routes` eklenmiştir.
"""
import os
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
                'tpStrategy': str(_row_get(row, 'tp_strategy', 'auto') or 'auto'),
                'drawdownFreezePct': float(_row_get(row, 'drawdown_freeze_pct', 0) or 0),
                'drawdownFreezeWindowDays': int(_row_get(row, 'drawdown_freeze_window_days', 7) or 7),
                'maxConsecutiveLosses': int(_row_get(row, 'max_consecutive_losses', 3) or 3),
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
                'tpStrategy': (r['tp_strategy'] if 'tp_strategy' in r.keys() else 'staged') or 'staged',
            })
        return positions
    except Exception as e:
        print(f"[AUTO-TRADE] Pozisyon getirme hatasi: {e}")
        return []

def _auto_get_daily_trade_count(user_id):
    """Bugün (TR saati) açılmış pozisyon sayısı (sadece BUY).
    SELL/SELL_TRAIL/SELL_TP dahil DEĞİL — SL/TP tetiklemelerinin sınırlanması istenmiyor.
    maxDailyTrades = gün içinde kullanıcıya uyarı/oto-open limiti (yeni giriş).

    NOT (Y2): SQLite tarafi UTC parse eder; sunucu UTC'de calisirken TR saati 00:00-03:00
    araliginda acilan trade'ler 'dunkune' sayilirdi. '+3 hours' modifier ile Europe/Istanbul'a
    cevriliyor; Postgres tarafi zaten 'AT TIME ZONE'.
    """
    try:
        db = get_db()
        today = _today_ist()
        if USE_POSTGRES and PG_OK:
            row = db.execute("SELECT COUNT(*) as cnt FROM auto_trades WHERE user_id=%s AND action='BUY' AND (created_at AT TIME ZONE 'Europe/Istanbul')::date=%s::date", (user_id, today)).fetchone()
        else:
            row = db.execute(
                "SELECT COUNT(*) as cnt FROM auto_trades "
                "WHERE user_id=? AND action='BUY' AND date(created_at, '+3 hours')=?",
                (user_id, today)
            ).fetchone()
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

def _decide_tp_strategy(user_id, entry_price, tp1, tp3):
    """TP strateji karari: 'auto' modunda sistem heuristic ile secer.
    Returns: 'staged' (50/25/25 - kar kossun) | 'all_at_tp1' (tek seferde sat).

    Heuristic:
      - Piyasa rejimi 'neutral' → temkinli, hizli kar al → 'all_at_tp1'
      - TP1-TP3 araligi <%3 → dar hedef, fragmentasyon mantiksiz → 'all_at_tp1'
      - Default → 'staged'

    Kullanici override (config.tp_strategy='staged'/'all_at_tp1') varsa o oncelik alir.
    Sadece config.tp_strategy='auto' ise bu fonksiyon devreye girer.
    """
    # 1. Kullanici manuel override
    try:
        cfg = _auto_get_config(user_id) or {}
        cfg_strategy = (cfg.get('tpStrategy') or 'auto').lower()
        if cfg_strategy in ('staged', 'all_at_tp1'):
            return cfg_strategy
    except Exception:
        pass
    # 2. Auto karar
    # Piyasa rejimi
    try:
        from auto_trader_regime import get_market_regime
        mode, _ = get_market_regime()
    except Exception:
        mode = 'risk-on'
    # TP1↔TP3 spread (relative)
    spread = 0.0
    try:
        if tp1 and tp3 and tp1 > 0:
            spread = (float(tp3) - float(tp1)) / float(tp1)
    except Exception:
        pass
    # Karar agacli
    if mode == 'neutral':
        return 'all_at_tp1'
    if 0 < spread < 0.03:
        return 'all_at_tp1'
    return 'staged'


def _auto_open_position(user_id, symbol, price, quantity, stop_loss, tp1, tp2, tp3, trailing_sl):
    """Yeni pozisyon ac. Capital/max_positions yarış koşulu için son-an DB kontrolü yapar.

    Concurrency: son-an kontrol + INSERT atomik blok icinde calismali (Scanner thread
    ile Telegram approve thread paralel acmaya kalkmasin -> max_positions asabilirdi).
    _auto_trade_lock ile sariliyor; pratikte 1-2ms blok suresi, scanner thread
    zaten 5 dk cycle'da.
    """
    pos_id = 0
    tp_strategy = 'staged'
    try:
        # ─── K2: Lock altinda son-an kontrol + INSERT ───────────────────
        with _auto_trade_lock:
            db = get_db()
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
                    # Sermaye kontrolu (K3 — tek cikis noktasi): AUTO_STRICT_CAPITAL=1 ise burada
                    # HARD BLOCK. _auto_open_position 4 yoldan cagriliyor (scanner, plan, Telegram
                    # approve); strict guard sadece scanner'daydi — approve/plan yollari bypass ediyordu.
                    # Blogu tek cokus noktasina koyarak 3 yolu da koruyoruz.
                    if capital > 0 and (used_capital + cost) > capital * 1.001:
                        if os.environ.get('AUTO_STRICT_CAPITAL', '0') == '1':
                            print(f"[AUTO-TRADE] Pozisyon açılmadı — sermaye limiti aşıldı "
                                  f"(kullanılan={used_capital:.0f} + yeni={cost:.0f} > kapital={capital:.0f}, "
                                  "AUTO_STRICT_CAPITAL=1 → engellendi)")
                            db.close()
                            return 0
                        print(f"[AUTO-TRADE] UYARI — sermaye limiti asildi ama yine de aciliyor "
                              f"(kullanılan={used_capital:.0f} + yeni={cost:.0f} > kapital={capital:.0f}). "
                              "Risk takibi icin capital ayarini guncelleyin.")
            except Exception as _guard_err:
                print(f"[AUTO-TRADE] Son-an kontrol hatası (devam ediliyor): {_guard_err}")

            # TP strateji karari (auto/staged/all_at_tp1) — INSERT'ten once
            try:
                tp_strategy = _decide_tp_strategy(user_id, price, tp1, tp3)
            except Exception as _ts_err:
                print(f"[AUTO-TRADE] TP strategy karar hatasi (default staged): {_ts_err}")
                tp_strategy = 'staged'

            db.execute(
                """INSERT INTO auto_positions
                   (user_id, symbol, side, entry_price, quantity, stop_loss,
                    take_profit1, take_profit2, take_profit3, trailing_stop, highest_price, tp_strategy)
                   VALUES (?,?,'long',?,?,?,?,?,?,?,?,?)""",
                (user_id, symbol, price, quantity, stop_loss, tp1, tp2, tp3, trailing_sl, price, tp_strategy)
            )
            db.commit()
            print(f"[AUTO-TRADE] Pozisyon acildi: {symbol} qty={quantity} @ {price}, tp_strategy={tp_strategy}")
            # Son eklenen ID'yi al (lock icinde okumamiz lazim — baska INSERT araya girmesin)
            if USE_POSTGRES and PG_OK:
                row = db.execute("SELECT MAX(id) as mid FROM auto_positions WHERE user_id=? AND symbol=?", (user_id, symbol)).fetchone()
            else:
                row = db.execute("SELECT last_insert_rowid() as mid").fetchone()
            pos_id = int(row['mid']) if row else 0
            db.close()
        # ─── Lock burada birakilir, network IO buradan sonra ───────────
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

def _calc_trade_costs(user_id, notional_buy, notional_sell, cfg=None):
    """Komisyon + BSMV'yi hesapla (BUY ve SELL tarafinda ayri ayri).
    Midas BIST'te 0 — cfg bos/yoksa 0 dondurur. BSMV komisyon uzerinden alinir.

    O3: cfg opsiyonel parametre — caller cfg'yi zaten elinde tutuyorsa
    geçirsin ki extra DB conn acmayalim. cfg=None -> _auto_get_config DB hit.
    Partial close/close path'leri 5-6 conn aciyordu; cfg passthrough ile -1 conn.
    """
    try:
        if cfg is None:
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


def _auto_close_position(position_id, close_price, reason, cfg=None):
    """Pozisyon kapat. cfg opsiyonel — passthrough ile _calc_trade_costs extra DB conn acmaz (O3)."""
    try:
        db = get_db()
        row = db.execute("SELECT * FROM auto_positions WHERE id=?", (position_id,)).fetchone()
        if not row:
            db.close()
            return False
        # #4d Idempotency: RT-monitor + step1 ayni pozisyonu ayni anda SL'de gorebilir (iki thread).
        # Zaten kapanmissa tekrar kapatma → cift portfoy-sat / cift bildirim / cift log olmasin.
        if (row['status'] if 'status' in row.keys() else 'open') != 'open':
            db.close()
            return False
        entry = float(row['entry_price'])
        qty = float(row['quantity'])
        gross = (close_price - entry) * qty
        costs = _calc_trade_costs(row['user_id'], entry * qty, close_price * qty, cfg=cfg)
        pnl = gross - costs
        pnl_pct = ((close_price - entry) / entry * 100) if entry > 0 else 0
        if entry > 0 and qty > 0 and costs > 0:
            pnl_pct = (pnl / (entry * qty)) * 100
        now_str = _now_ist().isoformat()
        # CAS: yalniz hala 'open' ise kapat. rowcount==0 → baska thread bizden once kapatmis,
        # yan etkileri (portfoy/bildirim/log) TEKRAR calistirma.
        cur = db.execute(
            "UPDATE auto_positions SET status='closed', closed_at=?, close_price=?, close_reason=?, pnl=?, pnl_pct=? "
            "WHERE id=? AND status='open'",
            (now_str, close_price, reason, round(pnl, 2), round(pnl_pct, 2), position_id)
        )
        if cur.rowcount == 0:
            db.close()
            return False
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
        # TP icra 'asked' dedup kayitlarini temizle (reopen ayni pos_id'yi kullanir;
        # stale key sonraki TP onay istegini bloklamasin)
        try:
            import telegram_state as _ts
            with _ts._pending_tp_exec_lock:
                for _f in ('take_profit1', 'take_profit2', 'take_profit3', 'trailing'):
                    _ts._tp_exec_asked.discard(f"{position_id}_{_f}")
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
        return True   # gerçekten KAPATILDI → caller log/cooldown bir kez çalışsın (#4d)
    except Exception as e:
        print(f"[AUTO-TRADE] Pozisyon kapatma hatasi: {e}")
        return False

def _auto_partial_close(position_id, sell_qty, price, reason, clear_tp_field=None, cfg=None):
    """Pozisyonun bir kısmını sat (TP1/TP2 kademeli kâr alma).
    sell_qty: satılacak miktar. Kalan quantity DB'de güncellenir.
    clear_tp_field: 'take_profit1' veya 'take_profit2' → 0'a set eder (tekrar tetiklemesin)
    cfg: opsiyonel oto-trade config (O3 — _calc_trade_costs extra conn acmasin).

    Concurrency: quantity guncellemesi atomic CAS ile (WHERE quantity>=?).
    Parallel partial close (orn. TP1+trailing ayni cycle'da) ezme olmaz —
    ikinci thread'in UPDATE'i rowcount=0 doner, sessizce iptal olur.
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

        # Yaklasik kalan (CAS sonrasi gercek deger DB'den okunacak)
        estimated_remaining = round(cur_qty - actual_sell, 2)

        # Eger neredeyse hepsi satilacaksa (1 lottan az kalir) → tam kapat
        # Threshold 0.01'den 1.0'a yukseldi: BIST'te integer lot kurali,
        # 0.01 float artefakti yakalanmiyordu (Y5).
        if estimated_remaining < 1.0:
            db.close()
            _auto_close_position(position_id, price, reason, cfg=cfg)
            return

        # Atomic CAS update: WHERE quantity>=actual_sell. Baska thread daha onceden
        # quantity'i dusurmusse (race condition) bizim UPDATE rowcount=0 doner,
        # PnL/log/notification BIZIM tarafimizdan tetiklenmez — diger thread halletti.
        _ALLOWED_CLEAR_TP_CAS = {
            'take_profit1': "UPDATE auto_positions SET quantity=quantity-?, take_profit1=0 WHERE id=? AND quantity>=?",
            'take_profit2': "UPDATE auto_positions SET quantity=quantity-?, take_profit2=0 WHERE id=? AND quantity>=?",
        }
        _BUMP_SL_TO_BE = clear_tp_field == 'take_profit1'
        _be_price = round(entry * 1.001, 2)  # %0.1 tampon
        _cur_sl = float(row['stop_loss'] or 0)

        if clear_tp_field and clear_tp_field in _ALLOWED_CLEAR_TP_CAS:
            cur = db.execute(_ALLOWED_CLEAR_TP_CAS[clear_tp_field],
                             (actual_sell, position_id, actual_sell))
        else:
            cur = db.execute(
                "UPDATE auto_positions SET quantity=quantity-? WHERE id=? AND quantity>=?",
                (actual_sell, position_id, actual_sell)
            )

        if cur.rowcount == 0:
            # Race: baska thread quantity'i bizim okudugumuz andan beri dusurmus.
            db.close()
            print(f"[AUTO-TRADE] #{position_id} {row['symbol']} kismi satis atlandi "
                  f"(race condition — diger thread halletti)")
            return

        # CRITICAL: commit'i UPDATE'ten hemen sonra cagir — paralel thread'lerin
        # bizim CAS'imizi gormesi icin. Aksi takdirde:
        #   - SQLite: write lock biz commit edene kadar tutulur (BUSY error riski)
        #   - Postgres: ikinci UPDATE bizim commit'i bekler, ama timeout olabilir
        # Asagidaki PnL hesabi/_calc_trade_costs ayri connection acar; commit'siz
        # uncommitted state'i goremez.
        db.commit()

        # Gercek kalan (CAS sonrasi DB'den oku — local estimated stale olabilir)
        try:
            _row2 = db.execute("SELECT quantity FROM auto_positions WHERE id=?",
                               (position_id,)).fetchone()
            remaining = float(_row2['quantity']) if _row2 else estimated_remaining
        except Exception:
            remaining = estimated_remaining
        db.close()

        # PnL hesabi (CAS basariliydi — bu satis bizim)
        gross = (price - entry) * actual_sell
        costs = _calc_trade_costs(row['user_id'], entry * actual_sell, price * actual_sell, cfg=cfg)
        pnl = gross - costs
        # PnL%: net (entry*qty notional uzerinden). costs=0 olsa bile formul tutarli.
        pnl_pct = (pnl / (entry * actual_sell) * 100) if entry > 0 and actual_sell > 0 else 0

        # CAS sonrasi remaining gercekte 1'in altinda kaldiysa (float artefakti
        # veya baska partial paralel) tam kapatma cagrisi:
        if remaining < 1.0:
            _auto_close_position(position_id, price, f"{reason} (kalan {remaining:.2f} lot temizlik)", cfg=cfg)
            return

        # Break-even SL onerisi: TP1 sonrasi mevcut SL girisin altindaysa,
        # Telegram'dan onay iste (otomatik UPDATE yok). Kullanici onaylarsa
        # _auto_update_level cagrilir; reddederse veya 60dk yanit gelmezse
        # eski SL korunur.
        _be_requested = False
        if _BUMP_SL_TO_BE and _cur_sl < _be_price:
            try:
                from telegram_notifications import send_sl_change_request
                _be_requested = bool(send_sl_change_request(
                    position_id=position_id,
                    symbol=row['symbol'],
                    field='stop_loss',
                    old_val=_cur_sl,
                    new_val=_be_price,
                    reason='TP1 hit -> SL break-even (kâr koruma)',
                    cur_price=price,
                    expires_min=60,
                ))
            except Exception as _be_err:
                print(f"[AUTO-TRADE] BE onay istegi gonderilemedi: {_be_err}")
        # Portfoy senkronu: kismi SELL -> portfolios'tan actual_sell kadar dus
        try:
            from auto_trader_sync import _sync_portfolio_sell
            _sync_portfolio_sell(row['user_id'], row['symbol'], actual_sell)
        except Exception as _ps_err:
            print(f"[AUTO-TRADE] Portfoy partial SELL sync hatasi: {_ps_err}")
        _be_msg = f", SL→BE onayi istendi ({_cur_sl:.2f}→{_be_price:.2f})" if _be_requested else ""
        print(f"[AUTO-TRADE] Kısmi satış #{position_id}: {row['symbol']} "
              f"{actual_sell:.2f} lot @ {price:.2f}, kalan={remaining:.2f}, "
              f"PnL={pnl:.2f} ({pnl_pct:.1f}%) - {reason}{_be_msg}")
        try:
            from routes_telegram import send_position_closed_notification
            _be_tail = (f"\n🛡 SL break-even icin Telegram'dan onay istendi "
                        f"({_be_price:.2f})") if _be_requested else ""
            send_position_closed_notification(
                row['symbol'], price, round(pnl, 2), round(pnl_pct, 2),
                f"{reason} (kısmi — kalan {remaining:.2f} lot){_be_tail}")
        except Exception:
            pass
    except Exception as e:
        print(f"[AUTO-TRADE] Kısmi satış hatası: {e}")


def _auto_update_trailing(position_id, new_trailing, new_highest):
    """Trailing stop + highest price güncelle (onay sonrası çağrılır).
    B#3: trailing SADECE YUKARI çekilir — stale/eski bir onay stop'u geriye çekemez.
    highest_price de yalnız yukarı. DB-agnostik (max Python'da)."""
    try:
        db = get_db()
        row = db.execute("SELECT trailing_stop, highest_price FROM auto_positions WHERE id=?",
                         (position_id,)).fetchone()
        _cur = float(row['trailing_stop'] or 0) if row else 0
        _cur_hi = float(row['highest_price'] or 0) if row else 0
        _new_hi = max(_cur_hi, float(new_highest or 0))
        if new_trailing is None or (_cur > 0 and float(new_trailing) <= _cur):
            # Trailing iyileşmiyor (stale onay) → stop'a DOKUNMA, sadece zirveyi güncelle.
            db.execute("UPDATE auto_positions SET highest_price=? WHERE id=?", (_new_hi, position_id))
            print(f"[AUTO-TRADE] #{position_id} trailing geriye çekilmedi (stale onay): "
                  f"yeni {new_trailing} <= mevcut {_cur:.2f}")
        else:
            db.execute("UPDATE auto_positions SET trailing_stop=?, highest_price=? WHERE id=?",
                       (float(new_trailing), _new_hi, position_id))
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


_AUTO_LEVEL_FIELDS = {
    'stop_loss':    'UPDATE auto_positions SET stop_loss=? WHERE id=?',
    'take_profit1': 'UPDATE auto_positions SET take_profit1=? WHERE id=?',
    'take_profit2': 'UPDATE auto_positions SET take_profit2=? WHERE id=?',
    'take_profit3': 'UPDATE auto_positions SET take_profit3=? WHERE id=?',
}


def _auto_update_level(position_id, field, new_val):
    """SL/TP seviyesini guncelle (Telegram onayi sonrasi cagrilir).
    field whitelist: stop_loss / take_profit1 / take_profit2 / take_profit3.
    """
    sql = _AUTO_LEVEL_FIELDS.get(field)
    if not sql:
        print(f"[AUTO-TRADE] Gecersiz seviye alani: {field}")
        return
    try:
        db = get_db()
        db.execute(sql, (float(new_val), position_id))
        db.commit()
        db.close()
        print(f"[AUTO-TRADE] Pozisyon #{position_id} {field} -> {float(new_val):.2f} (onay sonrasi)")
    except Exception as e:
        print(f"[AUTO-TRADE] Seviye guncelleme hatasi ({field}): {e}")


# =====================================================================
# TP ICRA ONAYI — TP hedefine ulasinca otomatik satis/TP-degisimi YOK.
# Kullanici Telegram'dan onaylar; iki monitor de (RT + motor) bu kapidan gecer.
# =====================================================================

_TP_FIELD_LABEL = {'take_profit1': 'TP1', 'take_profit2': 'TP2', 'take_profit3': 'TP3',
                   'trailing': 'Trailing-Stop'}


def _tp_take_profit(uid, position_id, symbol, kind, tp_field, sell_qty, price, tp_target):
    """TP hedefi tetiklendi — ONAYSIZ icra YOK. Telegram'dan onay iste; kullanici
    onaylayinca execute_tp_exec() kismi/tam satisi yapar. Telegram yoksa pozisyon
    dokunulmaz (kullanici elle yonetir).

    kind: 'partial' (TP1/TP2 %50 sat + ilgili TP'yi kapat) | 'full' (TP3 / all-at-tp1 tam kapat).
    """
    label = _TP_FIELD_LABEL.get(tp_field, tp_field)
    reason = (f"{label} tetiklendi ({float(tp_target):.2f})" if tp_field == 'trailing'
              else f"{label} hedef ({float(tp_target):.2f})")
    try:
        from telegram_notifications import send_tp_exec_request
        send_tp_exec_request(position_id, uid, symbol, kind, tp_field,
                             sell_qty, price, tp_target, reason)
    except Exception as e:
        print(f"[AUTO-TRADE] {symbol} {label} onay istegi gonderilemedi: {e}")


def execute_tp_exec(chg):
    """Telegram onayi gelince TP icrasini gerceklestir (kismi sat veya tam kapat).
    chg: telegram_state._pending_tp_exec kaydi. Onay handler'i cagirir.
    Returns (ok: bool, human_msg: str)."""
    pos_id = chg.get('position_id')
    uid = chg.get('uid', '')
    sym = chg.get('symbol', '')
    tp_field = chg.get('tp_field', '')
    kind = chg.get('kind', 'partial')
    reason = chg.get('reason') or f"{_TP_FIELD_LABEL.get(tp_field, tp_field)} hedef"

    # Icra fiyati: onay anindaki canli fiyat daha dogru; yoksa istek anindaki fiyat
    price = float(chg.get('price') or 0)
    try:
        from realtime_prices import get_price
        live = get_price(sym)
        if live and live > 0:
            price = float(live)
    except Exception:
        pass

    action = {'take_profit1': 'SELL_TP1', 'take_profit2': 'SELL_TP2',
              'take_profit3': 'SELL_TP3', 'trailing': 'SELL_TRAIL'}.get(tp_field, 'SELL_TP')
    sell_qty = float(chg.get('sell_qty') or 0)
    try:
        if kind == 'full':
            _auto_close_position(pos_id, price, f"{reason} (onayli)")
            _auto_log_trade(uid, sym, action, price, sell_qty,
                            f"{reason} — onayli tam kapat", 0, 0, pos_id)
            try:
                from auto_trader_risk import _panic_clear
                _panic_clear(pos_id)
            except Exception:
                pass
        else:
            _auto_partial_close(pos_id, sell_qty, price,
                                f"{reason} (onayli)", clear_tp_field=tp_field)
            _auto_log_trade(uid, sym, action, price, sell_qty,
                            f"{reason} — onayli kismi", 0, 0, pos_id)
        # RT monitor alert state'ini sifirla (flag'ler tazelensin)
        try:
            from realtime_prices import clear_alert_state
            clear_alert_state(sym, uid, pos_id)
        except Exception:
            pass
        # 'asked' dedup kaydini temizle
        try:
            import telegram_state as _ts
            with _ts._pending_tp_exec_lock:
                _ts._tp_exec_asked.discard(f"{pos_id}_{tp_field}")
        except Exception:
            pass
        human = (f"{sym}: tamamı kapatıldı @ {price:.2f}" if kind == 'full'
                 else f"{sym}: {sell_qty:.0f} lot satıldı @ {price:.2f} "
                      f"({_TP_FIELD_LABEL.get(tp_field, tp_field)} kapandı)")
        print(f"[AUTO-TRADE] TP icra (onayli): {human}")
        return True, human
    except Exception as e:
        print(f"[AUTO-TRADE] TP icra hatasi ({sym} {tp_field}): {e}")
        return False, str(e)
