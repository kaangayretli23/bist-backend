"""
realtime_prices.py - Anlık BIST fiyat izleme
Öncelik sırası:
  1. TradingView WebSocket (borsapy) — cookie ile giriş
  2. yfinance poll — sadece açık pozisyonlar, 8 sn timeout, paralel fetch
"""
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

# TradingView credentials (.env'den)
TV_SESSION      = os.environ.get('TV_SESSION', '')
TV_SESSION_SIGN = os.environ.get('TV_SESSION_SIGN', '')
TV_USERNAME     = os.environ.get('TV_USERNAME', '')
TV_PASSWORD     = os.environ.get('TV_PASSWORD', '')

# ── Fiyat cache ──
# {symbol: {'price': float, 'ts': float, 'source': 'ws'|'yf'}}
_rt_cache: dict = {}
_rt_lock = threading.Lock()
_RT_STALE_SEC = 90   # 90 sn'den eski → stale, yeniden fetch

# ── WebSocket stream ──
_stream = None
_stream_lock = threading.Lock()
_stream_ok = False
_subscribed: set = set()

# ── yfinance poll thread havuzu ──
# Sadece açık pozisyonlar için, max 10 eş zamanlı fetch
_yf_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix='yf-price')
_YF_TIMEOUT  = 8   # saniye — bu sürede cevap gelmezse eski fiyatı kullan


# =====================================================================
# WEBSOCKET STREAM
# =====================================================================

def _on_price_update(symbol: str, quote: dict) -> None:
    price = quote.get('last') or quote.get('lp') or quote.get('close')
    if price:
        try:
            p = float(price)
            if p > 0:
                with _rt_lock:
                    _rt_cache[symbol] = {
                        'price': p,
                        'change_pct': float(quote.get('chp') or quote.get('ch') or 0),
                        'ts': time.time(),
                        'source': 'ws',
                    }
        except Exception:
            pass


def _get_stream():
    global _stream, _stream_ok
    with _stream_lock:
        if _stream is not None and getattr(_stream, 'is_connected', False):
            return _stream
        try:
            from borsapy import TradingViewStream, set_tradingview_auth
            if TV_SESSION and TV_SESSION_SIGN:
                try:
                    set_tradingview_auth(session=TV_SESSION, session_sign=TV_SESSION_SIGN)
                    print("[RT-PRICES] TradingView cookie ile giriş yapıldı")
                except Exception as e:
                    print(f"[RT-PRICES] TV cookie hatası: {e}")
            elif TV_USERNAME and TV_PASSWORD:
                try:
                    set_tradingview_auth(username=TV_USERNAME, password=TV_PASSWORD)
                    print(f"[RT-PRICES] TradingView giriş yapıldı: {TV_USERNAME}")
                except Exception as e:
                    print(f"[RT-PRICES] TV giriş hatası: {e}")
            _stream = TradingViewStream()
            _stream.connect()
            _stream_ok = True
            print("[RT-PRICES] WebSocket bağlantısı kuruldu")
            return _stream
        except Exception as e:
            print(f"[RT-PRICES] Stream başlatılamadı: {e}")
            _stream_ok = False
            return None


def subscribe(symbol: str) -> bool:
    if symbol in _subscribed:
        return True
    try:
        stream = _get_stream()
        if stream is None:
            return False
        stream.subscribe(symbol)
        stream.on_quote(symbol, lambda s, q: _on_price_update(s, q))
        with _rt_lock:
            _subscribed.add(symbol)
        print(f"[RT-PRICES] {symbol} abone olundu")
        return True
    except Exception as e:
        print(f"[RT-PRICES] {symbol} abone hatası: {e}")
        return False


def unsubscribe(symbol: str) -> None:
    try:
        stream = _get_stream()
        if stream:
            try:
                stream.unsubscribe(symbol)
            except Exception:
                pass
        with _rt_lock:
            _subscribed.discard(symbol)
        print(f"[RT-PRICES] {symbol} abonelik iptal")
    except Exception:
        pass


def sync_subscriptions(symbols: list) -> None:
    target = set(symbols)
    with _rt_lock:
        current = set(_subscribed)
    for s in target - current:
        subscribe(s)
    for s in current - target:
        unsubscribe(s)


# =====================================================================
# YFİNANCE FALLBACK — timeout korumalı, paralel
# =====================================================================

def _yf_fetch_one(symbol: str) -> float | None:
    """Tek sembol için yfinance fetch — bu fonksiyon thread'de çalışır."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(f"{symbol}.IS")
        hist = ticker.history(period='1d', interval='1m')
        if hist is not None and len(hist) > 0:
            return float(hist['Close'].iloc[-1])
    except Exception:
        pass
    return None


def _yf_price_safe(symbol: str) -> float | None:
    """
    yfinance fetch, _YF_TIMEOUT saniye timeout ile.
    Donma olmaz — timeout'ta None döner, son cache kullanılır.
    """
    # Cache'de taze veri varsa fetch'e gitme
    with _rt_lock:
        cached = _rt_cache.get(symbol)
    if cached and (time.time() - cached['ts']) < _RT_STALE_SEC:
        return cached['price']

    try:
        future = _yf_executor.submit(_yf_fetch_one, symbol)
        price = future.result(timeout=_YF_TIMEOUT)
        if price and price > 0:
            with _rt_lock:
                _rt_cache[symbol] = {
                    'price': price,
                    'change_pct': 0.0,
                    'ts': time.time(),
                    'source': 'yf',
                }
            return price
    except FuturesTimeout:
        print(f"[RT-PRICES] {symbol} yfinance timeout ({_YF_TIMEOUT}s) — son cache kullanılıyor")
    except Exception as e:
        print(f"[RT-PRICES] {symbol} yfinance hatası: {e}")

    # Timeout veya hata: stale cache varsa onu kullan
    with _rt_lock:
        cached = _rt_cache.get(symbol)
    return cached['price'] if cached else None


def _yf_poll_batch(symbols: list) -> None:
    """Sembol listesini paralel olarak yfinance'tan çek."""
    if not symbols:
        return
    futures = {_yf_executor.submit(_yf_fetch_one, sym): sym for sym in symbols}
    for future, sym in futures.items():
        try:
            price = future.result(timeout=_YF_TIMEOUT)
            if price and price > 0:
                with _rt_lock:
                    _rt_cache[sym] = {
                        'price': price,
                        'change_pct': 0.0,
                        'ts': time.time(),
                        'source': 'yf',
                    }
        except FuturesTimeout:
            print(f"[RT-PRICES] {sym} poll timeout — atlandı")
        except Exception:
            pass


# =====================================================================
# FİYAT OKUMA (dışarıdan kullanılan tek fonksiyon)
# =====================================================================

def get_price(symbol: str) -> float | None:
    """
    En güncel fiyatı döner:
    1. WebSocket cache tazeyse → anında döner
    2. Değilse → yfinance (timeout korumalı)
    3. yfinance de başarısızsa → stale cache (en son bilinen)
    """
    with _rt_lock:
        cached = _rt_cache.get(symbol)
    if cached and (time.time() - cached['ts']) < _RT_STALE_SEC:
        return cached['price']
    return _yf_price_safe(symbol)


def get_quote(symbol: str) -> dict | None:
    with _rt_lock:
        return dict(_rt_cache[symbol]) if symbol in _rt_cache else None


def is_realtime() -> bool:
    return _stream_ok and bool(TV_SESSION or TV_USERNAME)


# =====================================================================
# POZİSYON MONİTÖRÜ
# =====================================================================

_alert_state: dict = {}
_alert_lock = threading.Lock()


def _load_persisted_alert_state():
    """Startup: DB'den alert state'leri RAM'a al. {pos_key: state} formatina cevir.
    pos_key = f'{user_id}_{symbol}'"""
    try:
        from database import _db_load_alert_states
        loaded = _db_load_alert_states()  # {(uid, sym, pid): state}
        with _alert_lock:
            for (uid, sym, _pid), st in loaded.items():
                _alert_state[f"{uid}_{sym}"] = st
        if loaded:
            print(f"[RT-MONITOR] {len(loaded)} alert state DB'den yuklendi")
    except Exception as e:
        print(f"[RT-MONITOR] Alert state yukleme hatasi: {e}")


def _get_open_positions():
    """DB'den açık pozisyonları güvenli şekilde çek."""
    try:
        from config import get_db
        db = get_db()
        rows = db.execute(
            "SELECT ap.id, ap.user_id, ap.symbol, ap.entry_price, ap.quantity, "
            "ap.stop_loss, ap.take_profit1, ap.take_profit2, ap.take_profit3 "
            "FROM auto_positions ap WHERE ap.status='open'"
        ).fetchall()
        db.close()
        return rows
    except Exception as e:
        print(f"[RT-MONITOR] DB okuma hatası: {e}")
        return []


def _enabled_user_styles() -> dict:
    """{user_id: trade_style} — sadece auto-trade aktif kullanicilar.
    SL/TP tetiklendiginde otomatik kapatma sadece bu kullanicilarda calisir."""
    try:
        from config import get_db
        db = get_db()
        rows = db.execute(
            "SELECT user_id, trade_style FROM auto_config WHERE enabled=1"
        ).fetchall()
        db.close()
        return {r['user_id']: (r['trade_style'] or 'swing') for r in rows}
    except Exception:
        return {}


# Spike koruması: SL/TP tetiklemesi için ardışık tick sayısı.
# Tek bir saçma tick'le kapatmayalım; iki ardışık tick istiyoruz.
_RT_CONFIRM_TICKS = 2


def _check_positions_once():
    """
    Açık pozisyonları kontrol et:
      1) Telegram uyarısı (her durumda — auto-trade kapalı kullanıcılar için de)
      2) Auto-trade aktif kullanıcılarda SL/TP tetiklenince otomatik kapat
         (ardışık 2 tick koşulu — spike koruması).
    """
    positions = _get_open_positions()
    if not positions:
        return

    try:
        from routes_telegram import send_telegram
    except Exception:
        send_telegram = None

    enabled_styles = _enabled_user_styles()

    for pos in positions:
        sym   = pos['symbol']
        entry = float(pos['entry_price'] or 0)
        qty   = float(pos['quantity']    or 0)
        sl    = float(pos['stop_loss']   or 0)
        tp1   = float(pos['take_profit1'] or 0)
        tp2   = float(pos['take_profit2'] or 0)
        tp3   = float(pos['take_profit3'] or 0)
        uid   = pos['user_id']
        pid   = pos['id']

        cur = get_price(sym)
        if not cur or cur <= 0 or not entry:
            continue

        pnl_pct = (cur - entry) / entry * 100
        pos_key = f"{uid}_{sym}"

        with _alert_lock:
            state = _alert_state.setdefault(pos_key, {
                'sl_warned': False, 'sl_hit': False, 'sl_confirm': 0, 'sl_executed': False,
                'tp1_warned': False, 'tp1_hit': False, 'tp1_confirm': 0, 'tp1_executed': False,
                'tp2_warned': False, 'tp2_hit': False, 'tp2_confirm': 0, 'tp2_executed': False,
                'tp3_hit': False, 'tp3_confirm': 0, 'tp3_executed': False,
            })
            # Geriye-uyumluluk: eski state'lerde yeni alanlar yoksa ekle
            for k in ('sl_confirm','sl_executed','tp1_confirm','tp1_executed',
                      'tp2_hit','tp2_confirm','tp2_executed','tp3_confirm','tp3_executed'):
                state.setdefault(k, 0 if k.endswith('_confirm') else False)

        msgs = []
        auto_exec = uid in enabled_styles
        style = enabled_styles.get(uid, 'swing')

        # ---- STOP-LOSS ----
        if sl > 0:
            if cur <= sl:
                state['sl_confirm'] = state['sl_confirm'] + 1
                if not state['sl_hit']:
                    msgs.append(
                        f"🚨 <b>STOP-LOSS — {sym}</b>\n"
                        f"Fiyat: {cur:.2f} ≤ SL: {sl:.2f}\n"
                        f"📉 Zarar: %{pnl_pct:.1f}\n"
                        f"⚠️ <b>SATIŞ YAPINIZ</b>"
                    )
                    state['sl_hit'] = True
                # Auto-execute: 2 ardışık tick + cfg.enabled
                if (auto_exec and not state['sl_executed']
                        and state['sl_confirm'] >= _RT_CONFIRM_TICKS):
                    try:
                        from auto_trader import _auto_close_position, _auto_log_trade
                        from auto_trader_risk import _sl_cooldown_block, _panic_clear
                        _auto_close_position(pid, cur, f"Stop-Loss tetiklendi ({sl:.2f})")
                        _auto_log_trade(uid, sym, 'SELL_SL', cur, qty,
                                        f"SL RT: {cur:.2f} <= {sl:.2f}", 0, 0, pid)
                        _sl_cooldown_block(uid, sym, style)
                        _panic_clear(pid)
                        state['sl_executed'] = True
                        print(f"[RT-EXEC] {sym} SL otomatik kapatildi @ {cur:.2f}")
                        # Pozisyon kapandı → state'i temizle (RAM + DB)
                        with _alert_lock:
                            _alert_state.pop(pos_key, None)
                        try:
                            from database import _db_delete_alert_state
                            _db_delete_alert_state(uid, sym, pid)
                        except Exception:
                            pass
                        unsubscribe(sym)
                        # Telegram'a son haber
                        if send_telegram and msgs:
                            for m in msgs:
                                try: send_telegram(m)
                                except Exception: pass
                            msgs = []
                        continue
                    except Exception as ex:
                        print(f"[RT-EXEC] {sym} SL kapatma hatasi: {ex}")
            else:
                state['sl_confirm'] = 0
                if (not state['sl_hit'] and not state['sl_warned']
                        and cur <= sl * 1.02):
                    msgs.append(
                        f"⚠️ <b>SL Yaklaşıyor — {sym}</b>\n"
                        f"Fiyat: {cur:.2f} | SL: {sl:.2f} "
                        f"(%{abs((cur - sl) / sl * 100):.1f} uzakta)"
                    )
                    state['sl_warned'] = True

        # ---- TAKE-PROFIT (yuksekten alcaga) ----
        if tp3 > 0 and cur >= tp3:
            state['tp3_confirm'] = state['tp3_confirm'] + 1
            if not state['tp3_hit']:
                msgs.append(
                    f"🎯 <b>TP3 HEDEFİ — {sym}</b>\n"
                    f"Fiyat: {cur:.2f} ≥ TP3: {tp3:.2f}\n"
                    f"📈 Kâr: %{pnl_pct:.1f} — <b>SATIŞ ÖNERİLİR</b>"
                )
                state['tp3_hit'] = True
            if (auto_exec and not state['tp3_executed']
                    and state['tp3_confirm'] >= _RT_CONFIRM_TICKS):
                try:
                    from auto_trader import _auto_close_position, _auto_log_trade
                    from auto_trader_risk import _panic_clear
                    _auto_close_position(pid, cur, f"TP3 hedef ({tp3:.2f})")
                    _auto_log_trade(uid, sym, 'SELL_TP3', cur, qty,
                                    f"TP3 RT: {cur:.2f} >= {tp3:.2f}", 0, 0, pid)
                    _panic_clear(pid)
                    state['tp3_executed'] = True
                    print(f"[RT-EXEC] {sym} TP3 otomatik kapatildi @ {cur:.2f}")
                    with _alert_lock:
                        _alert_state.pop(pos_key, None)
                    try:
                        from database import _db_delete_alert_state
                        _db_delete_alert_state(uid, sym, pid)
                    except Exception:
                        pass
                    unsubscribe(sym)
                    if send_telegram and msgs:
                        for m in msgs:
                            try: send_telegram(m)
                            except Exception: pass
                        msgs = []
                    continue
                except Exception as ex:
                    print(f"[RT-EXEC] {sym} TP3 kapatma hatasi: {ex}")
        elif tp2 > 0 and cur >= tp2:
            state['tp2_confirm'] = state['tp2_confirm'] + 1
            if not state['tp2_hit']:
                msgs.append(
                    f"🎯 <b>TP2 HEDEFİ — {sym}</b>\n"
                    f"Fiyat: {cur:.2f} ≥ TP2: {tp2:.2f}\n"
                    f"📈 Kâr: %{pnl_pct:.1f} — <b>SATIŞ ÖNERİLİR</b>"
                )
                state['tp2_hit'] = True
                state['tp2_warned'] = True
            if (auto_exec and not state['tp2_executed']
                    and state['tp2_confirm'] >= _RT_CONFIRM_TICKS):
                try:
                    from auto_trader import _auto_partial_close, _auto_log_trade
                    sell_qty = int(qty * 0.5) or int(qty)
                    _auto_partial_close(pid, sell_qty, cur,
                                        f"TP2 hedef ({tp2:.2f})", clear_tp_field='take_profit2')
                    _auto_log_trade(uid, sym, 'SELL_TP2', cur, sell_qty,
                                    f"TP2 RT kismi: {cur:.2f} >= {tp2:.2f}", 0, 0, pid)
                    state['tp2_executed'] = True
                    print(f"[RT-EXEC] {sym} TP2 partial close @ {cur:.2f} ({sell_qty} adet)")
                except Exception as ex:
                    print(f"[RT-EXEC] {sym} TP2 partial hatasi: {ex}")
        elif tp1 > 0 and cur >= tp1:
            state['tp1_confirm'] = state['tp1_confirm'] + 1
            if not state['tp1_hit']:
                msgs.append(
                    f"🎯 <b>TP1 HEDEFİ — {sym}</b>\n"
                    f"Fiyat: {cur:.2f} ≥ TP1: {tp1:.2f}\n"
                    f"📈 Kâr: %{pnl_pct:.1f}"
                )
                state['tp1_hit'] = True
            if (auto_exec and not state['tp1_executed']
                    and state['tp1_confirm'] >= _RT_CONFIRM_TICKS):
                try:
                    from auto_trader import _auto_partial_close, _auto_log_trade
                    sell_qty = int(qty * 0.5) or int(qty)
                    _auto_partial_close(pid, sell_qty, cur,
                                        f"TP1 hedef ({tp1:.2f})", clear_tp_field='take_profit1')
                    _auto_log_trade(uid, sym, 'SELL_TP1', cur, sell_qty,
                                    f"TP1 RT kismi: {cur:.2f} >= {tp1:.2f}", 0, 0, pid)
                    state['tp1_executed'] = True
                    print(f"[RT-EXEC] {sym} TP1 partial close @ {cur:.2f} ({sell_qty} adet)")
                except Exception as ex:
                    print(f"[RT-EXEC] {sym} TP1 partial hatasi: {ex}")
        else:
            # Hicbir TP'de degil — confirm sayaclarini sifirla (spike kayboldu)
            state['tp1_confirm'] = state['tp2_confirm'] = state['tp3_confirm'] = 0
            if (tp1 > 0 and not state['tp1_warned'] and not state['tp1_hit']
                    and cur >= tp1 * 0.98):
                msgs.append(
                    f"📍 <b>TP1 Yaklaşıyor — {sym}</b>\n"
                    f"Fiyat: {cur:.2f} | TP1: {tp1:.2f}"
                )
                state['tp1_warned'] = True

        with _alert_lock:
            _alert_state[pos_key] = state

        # Persist (yalnizca flag degismis ise) — DB'ye yazma maliyetini azaltmak icin
        # her tick degil, anlamli flag transition'larinda yazariz.
        try:
            if any(state.get(k) for k in (
                'sl_warned', 'sl_hit', 'sl_executed',
                'tp1_warned', 'tp1_hit', 'tp1_executed',
                'tp2_warned', 'tp2_hit', 'tp2_executed',
                'tp3_hit', 'tp3_executed',
            )):
                from database import _db_save_alert_state
                _db_save_alert_state(uid, sym, pid, state)
        except Exception:
            pass

        if send_telegram:
            for msg in msgs:
                try:
                    send_telegram(msg)
                except Exception:
                    pass


def reset_position_alerts(symbol: str, user_id: str, position_id: int | None = None) -> None:
    """Pozisyon kapanınca uyarı state'ini temizle (RAM+DB) ve aboneliği iptal et."""
    with _alert_lock:
        _alert_state.pop(f"{user_id}_{symbol}", None)
    if position_id is not None:
        try:
            from database import _db_delete_alert_state
            _db_delete_alert_state(user_id, symbol, position_id)
        except Exception:
            pass
    unsubscribe(symbol)


# =====================================================================
# ARKA PLAN THREAD
# =====================================================================

_monitor_started = False
_monitor_lock = threading.Lock()


def start_realtime_monitor():
    global _monitor_started
    with _monitor_lock:
        if _monitor_started:
            return
        _monitor_started = True

    # Restart sonrasi alert state'leri DB'den geri yukle (TP1/SL flagleri)
    _load_persisted_alert_state()

    def _loop():
        # Başlangıçta stream'i bağla
        time.sleep(8)
        _get_stream()

        tick = 0
        while True:
            try:
                positions = _get_open_positions()
                syms = list({p['symbol'] for p in positions})

                # WebSocket aboneliğini senkronize et
                if syms:
                    sync_subscriptions(syms)

                # WebSocket cache'i stale olan sembolleri yfinance ile tazele
                stale = []
                now = time.time()
                for sym in syms:
                    with _rt_lock:
                        cached = _rt_cache.get(sym)
                    if not cached or (now - cached['ts']) > _RT_STALE_SEC:
                        stale.append(sym)

                if stale:
                    # Paralel, timeout korumalı yfinance fetch
                    _yf_poll_batch(stale)

                # SL/TP kontrol
                _check_positions_once()

                # Stream bağlantısını her 5 dakikada bir doğrula
                if tick % 10 == 0:
                    stream = _get_stream()
                    if stream and not getattr(stream, 'is_connected', True):
                        print("[RT-PRICES] Stream koptu, yeniden bağlanıyor...")
                        with _stream_lock:
                            global _stream, _stream_ok, _subscribed
                            # Eski stream'i kapat (callback'lerin çift tetiklenmesini önler)
                            old_stream = _stream
                            _stream = None
                            _stream_ok = False
                            _subscribed = set()
                            if old_stream is not None:
                                for _close_method in ('disconnect', 'close', 'stop'):
                                    try:
                                        fn = getattr(old_stream, _close_method, None)
                                        if callable(fn):
                                            fn()
                                            break
                                    except Exception:
                                        pass
                        # Tek seferde yeniden bağlan ki arada subscribe/unsubscribe yarış koşuluna girmesin
                        _get_stream()

            except Exception as e:
                print(f"[RT-MONITOR] Loop hatası: {e}")

            tick += 1
            time.sleep(30)  # Her 30 saniyede bir kontrol

    t = threading.Thread(target=_loop, daemon=True, name='rt-monitor')
    t.start()
    mode = "WebSocket + yfinance yedek" if (TV_SESSION or TV_USERNAME) else "yfinance (TV credentials yok)"
    print(f"[RT-MONITOR] Başlatıldı — mod: {mode}")
