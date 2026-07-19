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

# TradingView tick'lerini _stock_cache'e koprule (mevcut endpoint'ler taze veri gorur).
# Sorun cikarsa USE_TV_LIVE=0 ile devre disi birakilir → eski Is Yatirim akisina doner.
USE_TV_LIVE = os.environ.get('USE_TV_LIVE', '1') not in ('0', 'false', 'False', '')

# D3: Operasyonel istatistik sayaclari — /api/system/diagnostic ile gosterilir.
_stats = {
    'tick_count': 0,           # toplam tick alindi
    'tick_first_ts': 0.0,      # ilk tick zamani (uptime hesabi)
    'tick_last_ts': 0.0,       # son tick zamani
    'reconnect_count': 0,      # WebSocket reconnect sayisi
    'reconnect_last_ts': 0.0,  # son reconnect zamani
    'subscribe_count': 0,      # toplam subscribe yapildi
    'subscribe_fail_count': 0, # subscribe basarisiz
    'started_at': time.time(), # modul yuklendigi zaman
}
_stats_lock = threading.Lock()

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
    if not price:
        return
    try:
        p = float(price)
        if p <= 0:
            return
    except Exception:
        return
    # borsapy quote keys: 'last', 'change', 'change_percent', 'open', 'high', 'low'
    # (alternatifler: TradingView ham → 'lp', 'ch', 'chp')
    tv_ch_raw = quote.get('change')
    if tv_ch_raw is None:
        tv_ch_raw = quote.get('ch')
    tv_chp_raw = quote.get('change_percent')
    if tv_chp_raw is None:
        tv_chp_raw = quote.get('chp')
    try:
        tv_ch = float(tv_ch_raw) if tv_ch_raw is not None else None
    except Exception:
        tv_ch = None
    try:
        tv_chp = float(tv_chp_raw) if tv_chp_raw is not None else None
    except Exception:
        tv_chp = None
    tv_open = quote.get('open')
    tv_high = quote.get('high')
    tv_low = quote.get('low')
    # change_pct: tv_chp oncelikli (zaten yuzde), yoksa tv_ch (mutlak deger — yuzde degil
    # ama yaklaşık olarak yon bilgisi olur), o da yoksa 0. `or` yerine `is not None` —
    # tv_chp=0.0 (gercek hareketsiz hisse) durumunda fallback'e dusmesin.
    if tv_chp is not None:
        rt_change_pct = tv_chp
    elif tv_ch is not None:
        rt_change_pct = tv_ch
    else:
        rt_change_pct = 0.0
    _now_ts = time.time()
    with _rt_lock:
        _rt_cache[symbol] = {
            'price': p,
            'change_pct': float(rt_change_pct),
            'ts': _now_ts,
            'source': 'ws',
        }
    with _stats_lock:
        _stats['tick_count'] += 1
        _stats['tick_last_ts'] = _now_ts
        if _stats['tick_first_ts'] == 0.0:
            _stats['tick_first_ts'] = _now_ts
    # ── Koprule: _stock_cache'i de live fiyatla guncelle (USE_TV_LIVE=1 ise) ──
    if not USE_TV_LIVE:
        return
    try:
        from config import _stock_cache, _cset, sf, si
        cur_item = _stock_cache.get(symbol)
        existing = (cur_item or {}).get('data', {}) if cur_item else {}
        prev_close = existing.get('prevClose')
        if not prev_close or float(prev_close) <= 0:
            return  # Loader baseline'i henuz dolmadi → atla
        prev_close = float(prev_close)
        chg_abs = tv_ch if tv_ch is not None else (p - prev_close)
        chg_pct = tv_chp if tv_chp is not None else (chg_abs / prev_close * 100)
        # OHLC: TradingView quote'undan tercih et, yoksa loader baseline'inden + extend
        try: new_open = float(tv_open) if tv_open is not None else float(existing.get('open') or p)
        except Exception: new_open = float(existing.get('open') or p)
        try: tv_hi = float(tv_high) if tv_high is not None else None
        except Exception: tv_hi = None
        try: tv_lo = float(tv_low) if tv_low is not None else None
        except Exception: tv_lo = None
        old_hi = float(existing.get('high') or p)
        old_lo = float(existing.get('low') or p)
        new_hi = max(old_hi, p, tv_hi or p)
        candidates_lo = [v for v in (old_lo, p, tv_lo) if v and v > 0]
        new_lo = min(candidates_lo) if candidates_lo else p
        merged = dict(existing)
        merged.update({
            'price': sf(p),
            'change': sf(chg_abs),
            'changePct': sf(chg_pct),
            'open': sf(new_open),
            'high': sf(new_hi),
            'low': sf(new_lo),
        })
        _cset(_stock_cache, symbol, merged)
    except Exception:
        pass  # Bridge basarisiz olsa _rt_cache hala calisiyor — sessiz gec


def _get_stream():
    global _stream, _stream_ok
    with _stream_lock:
        # is_connected default=True: borsapy versiyonu attribute saglamiyorsa
        # mevcut stream'i kabul et (False olsa her cagride yeni stream → leak).
        # Reconnect logic'i _loop icinde ayrica calisir (line 656).
        if _stream is not None and getattr(_stream, 'is_connected', True):
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
    # Atomik check-and-add: iki thread ayni anda subscribe(SAME) cagirsa,
    # ikinci thread False return alsin. on_quote çift kayıt → cift tick callback'i onlenir.
    with _rt_lock:
        if symbol in _subscribed:
            return True
        _subscribed.add(symbol)  # placeholder claim — ileride fail olursa rollback
    try:
        stream = _get_stream()
        if stream is None:
            with _rt_lock:
                _subscribed.discard(symbol)  # rollback
            return False
        stream.subscribe(symbol)
        stream.on_quote(symbol, lambda s, q: _on_price_update(s, q))
        with _stats_lock:
            _stats['subscribe_count'] += 1
        print(f"[RT-PRICES] {symbol} abone olundu")
        return True
    except Exception as e:
        with _rt_lock:
            _subscribed.discard(symbol)  # rollback — sonraki retry icin
        with _stats_lock:
            _stats['subscribe_fail_count'] += 1
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


def sync_subscriptions(symbols: list, throttle_sec: float = 0.05) -> None:
    """Hedef sembol kumesine senkronize et. Aralarda ufak gecikme = TV rate-limit guvenligi."""
    target = set(symbols)
    with _rt_lock:
        current = set(_subscribed)
    to_add = list(target - current)
    to_remove = list(current - target)
    if to_add:
        print(f"[RT-PRICES] sync: {len(to_add)} yeni abonelik ekleniyor...")
    for i, s in enumerate(to_add):
        subscribe(s)
        if throttle_sec and i < len(to_add) - 1:
            time.sleep(throttle_sec)
    for s in to_remove:
        unsubscribe(s)
    if to_add:
        print(f"[RT-PRICES] sync: tamamlandi (toplam abone={len(_subscribed)})")


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
    """Sembol listesini paralel olarak yfinance'tan çek.
    `as_completed` ile bitenler sirayla islenir — eskiden `futures.items()` iteration
    order'a gore bekliyordu, ilk yavas future digerlerini bloke ediyordu."""
    if not symbols:
        return
    from concurrent.futures import as_completed
    futures = {_yf_executor.submit(_yf_fetch_one, sym): sym for sym in symbols}
    # Toplam timeout: tek timeout × 2 (paralel oldugu icin generally cok daha hizli biter)
    try:
        for future in as_completed(futures, timeout=_YF_TIMEOUT * 2):
            sym = futures[future]
            try:
                price = future.result(timeout=0.1)  # zaten tamamlandi, hizli al
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
    except FuturesTimeout:
        # Toplam timeout — geri kalan future'lar yarida kalir
        unfinished = [futures[f] for f in futures if not f.done()]
        if unfinished:
            print(f"[RT-PRICES] _yf_poll_batch toplam timeout — bitmeyen: {len(unfinished)}")


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


# NOT (FAZ 2): is_realtime() kaldırıldı — hiç çağrılmıyordu. Canlılık bilgisi
# get_quote()'un döndürdüğü kayıttaki kaynak/zaman alanlarından okunuyor.

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
            "ap.stop_loss, ap.take_profit1, ap.take_profit2, ap.take_profit3, "
            "ap.tp_strategy "
            "FROM auto_positions ap WHERE ap.status='open'"
        ).fetchall()
        db.close()
        return rows
    except Exception as e:
        print(f"[RT-MONITOR] DB okuma hatası: {e}")
        return []


# _enabled_user_styles + _check_positions_once -> realtime_monitor.py (modul satir siniri).
# _loop bu fonksiyonu cagiriyor; isim alanina geri import (circular: realtime_monitor
# realtime_prices YUKLENDIKTEN sonra import edilir, get_price/_alert_state vb. zaten tanimli).
from realtime_monitor import _check_positions_once


# NOT (FAZ 2): reset_position_alerts() kaldırıldı — hiç çağrılmıyordu ve kendi docstring'i
# tehlikeli olduğunu söylüyordu (unsubscribe, başka kullanıcının açık pozisyonunun takibini de
# durduruyordu). Kullanımdaki güvenli yol: clear_alert_state() (unsubscribe yapmaz).

def clear_alert_state(symbol: str, user_id: str, position_id: int | None = None) -> None:
    """Sadece alert state cleanup (RAM+DB). Unsubscribe yapmaz.
    Manuel/otomatik kapatma sonrası RAM leak'i onler."""
    with _alert_lock:
        _alert_state.pop(f"{user_id}_{symbol}", None)
    if position_id is not None:
        try:
            from database import _db_delete_alert_state
            _db_delete_alert_state(user_id, symbol, position_id)
        except Exception:
            pass


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
        # global deklarasyonu fonksiyon basinda olmali — asagida _subscribed = set() atamasi var
        global _stream, _stream_ok, _subscribed
        # Başlangıçta stream'i bağla
        time.sleep(8)
        _get_stream()

        tick = 0
        while True:
            try:
                positions = _get_open_positions()
                syms = list({p['symbol'] for p in positions})

                # WebSocket aboneliğini SADECE EKLE — sync yapma!
                # sync_subscriptions, syms'de olmayanlari unsubscribe eder; bu da BIST100
                # genel aboneligimizi (backend startup + loader FAZE 5c) ezerdi.
                # Acik pozisyonlar zaten BIST100'un alt kumesi, abone ol yeterli.
                for s in syms:
                    if s not in _subscribed:
                        subscribe(s)

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

                # P0: Piyasa-geneli erken uyarı (hız alarmı + endeks devre kesici) — BIST100∪BIST30
                try:
                    from market_alerts import check_market_alerts_once
                    check_market_alerts_once()
                except Exception as e:
                    print(f"[MARKET-ALERTS] hata: {e}")

                # Stream bağlantısını her 5 dakikada bir doğrula
                if tick % 10 == 0:
                    stream = _get_stream()
                    if stream and not getattr(stream, 'is_connected', True):
                        print("[RT-PRICES] Stream koptu, yeniden bağlanıyor...")
                        with _stats_lock:
                            _stats['reconnect_count'] += 1
                            _stats['reconnect_last_ts'] = time.time()
                        # D1: Telegram'a kopuş bildirimi (sadece ilk kopuşta)
                        try:
                            from routes_telegram import send_telegram
                            send_telegram(
                                "🔌 <b>TV WebSocket koptu</b>\n"
                                "Sistem otomatik yeniden bağlanmaya çalışıyor.\n"
                                "Live fiyat akışı geçici olarak durdu — birkaç saniye içinde geri dönmeli."
                            )
                        except Exception:
                            pass
                        # ÖNCE: eski abonelik listesini sakla (reconnect sonrasi geri al)
                        with _rt_lock:
                            previously_subscribed = set(_subscribed)
                        with _stream_lock:
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
                        # Eski abonelikleri geri yukle (BIST100 cache canli kalsin)
                        if previously_subscribed:
                            print(f"[RT-PRICES] Reconnect: {len(previously_subscribed)} sembol yeniden abone olunacak")
                            for sym in previously_subscribed:
                                subscribe(sym)
                                time.sleep(0.05)  # rate-limit guvenligi
                            print(f"[RT-PRICES] Reconnect: aboneler geri yuklendi (toplam={len(_subscribed)})")
                        # D1: Reconnect başarılıysa Telegram'a bildir
                        try:
                            if _stream_ok:
                                from routes_telegram import send_telegram
                                send_telegram(
                                    f"✅ <b>TV WebSocket tekrar bağlı</b>\n"
                                    f"{len(_subscribed)} sembol yeniden abone, live akış aktif."
                                )
                        except Exception:
                            pass

            except Exception as e:
                print(f"[RT-MONITOR] Loop hatası: {e}")

            tick += 1
            # P2: Volatil seansta (yakın zamanda uyarı çıktıysa) polling hızlanır: 30sn → 10sn.
            try:
                from market_alerts import is_volatile_mode
                _sleep = 10 if is_volatile_mode() else 30
            except Exception:
                _sleep = 30
            time.sleep(_sleep)  # Normal 30sn; volatil modda 10sn

    t = threading.Thread(target=_loop, daemon=True, name='rt-monitor')
    t.start()
    mode = "WebSocket + yfinance yedek" if (TV_SESSION or TV_USERNAME) else "yfinance (TV credentials yok)"
    print(f"[RT-MONITOR] Başlatıldı — mod: {mode}")
