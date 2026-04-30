"""
BIST Pro - Data Fetcher & Background Loader Module
Ham veri çekme fonksiyonları → data_fetcher_raw.py
"""
import os, time, threading, traceback, json, re, logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    import yfinance as yf
    YF_OK = True
    # yfinance "$XYZ.IS: possibly delisted" log spam'ini bastir — biz zaten
    # alternatif kaynaklara fallback ediyoruz, kullaniciyi rahatsiz etmesin.
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)
    logging.getLogger('yfinance.utils').setLevel(logging.CRITICAL)
except ImportError:
    YF_OK = False
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pass
from config import (
    sf, si, _lock, _stock_cache, _index_cache, _hist_cache,
    _cget, _cset, _ctouch, _cget_hist, _get_stocks,
    _loader_started, _status, CACHE_TTL, CACHE_STALE_TTL, HIST_CACHE_TTL,
    BIST100_STOCKS, BIST30, INDEX_TICKERS, PARALLEL_WORKERS, app, BASE_DIR
)
from data_fetcher_raw import (
    _normalize_ohlcv_df,
    _fetch_isyatirim_df, _fetch_isyatirim_quick,
    _fetch_yahoo_http, _fetch_yahoo_http_df,
    _fetch_borsapy_quick, _fetch_borsapy_hist,
    IS_YATIRIM_HEADERS,
)

# Lazy imports to avoid circular dependencies
def _get_db_save_snapshot():
    from database import _db_save_market_snapshot
    return _db_save_market_snapshot

def _get_db_load_snapshot():
    from database import _db_load_market_snapshot
    return _db_load_market_snapshot

def _get_start_telegram_thread():
    try:
        from routes_telegram import _start_telegram_thread
        return _start_telegram_thread
    except Exception:
        return None

def _get_db():
    from config import get_db
    return get_db

def _get_send_telegram_alerts():
    try:
        from routes_portfolio import _send_telegram_alerts
        return _send_telegram_alerts
    except Exception:
        return None

def _get_auto_engine():
    try:
        from auto_trader_engine import _auto_engine_cycle
        return _auto_engine_cycle
    except Exception:
        return None

import urllib.request
import urllib.error
import requests as req_lib
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---- BIRLESIK FETCHER (4 katmanli fallback) ----
def _fetch_stock_data(sym, retry_count=2):
    """Hisse verisi cek: 1.borsapy -> 2.IsYatirim -> 3.Yahoo HTTP -> 4.yfinance"""
    for attempt in range(1, retry_count + 1):
        # 1. borsapy (birincil - en guncel)
        data = _fetch_borsapy_quick(sym)
        if data:
            return data

        # 2. Is Yatirim
        data = _fetch_isyatirim_quick(sym)
        if data:
            return data

        # 3. Yahoo HTTP
        for ticker in [f"{sym}.IS", sym]:
            data = _fetch_yahoo_http(ticker)
            if data:
                print(f"  [YAHOO] {sym} OK")
                return data

        # 4. yfinance (son care)
        if YF_OK:
            try:
                h = yf.Ticker(f"{sym}.IS").history(period="5d", timeout=6)
                if h is not None and not h.empty and len(h) >= 2:
                    cur, prev = float(h['Close'].iloc[-1]), float(h['Close'].iloc[-2])
                    if prev > 0:
                        print(f"  [YF] {sym} OK: {cur}")
                        return {'close': cur, 'prev': prev, 'open': float(h['Open'].iloc[-1]),
                                'high': float(h['High'].iloc[-1]), 'low': float(h['Low'].iloc[-1]),
                                'volume': int(h['Volume'].iloc[-1])}
            except Exception as e:
                print(f"  [YF] {sym}: {e}")

        if attempt < retry_count:
            wait = attempt * 0.5
            print(f"  [RETRY] {sym} deneme {attempt}/{retry_count} basarisiz, {wait}s bekleniyor...")
            time.sleep(wait)

    print(f"  [ALL] {sym} BASARISIZ ({retry_count} deneme)")
    return None


def _fetch_hist_df(sym, period='1y'):
    """Tam DataFrame cek (indicators icin): 1.borsapy -> 2.IsYatirim -> 3.Yahoo HTTP DF -> 4.yfinance"""
    period_days = {'1mo':30,'3mo':90,'6mo':180,'1y':365,'2y':730,'5y':1825}.get(period, 365)

    # 1. borsapy (birincil)
    df = _fetch_borsapy_hist(sym, period)
    if df is not None and len(df) >= 10:
        return df

    # 2. Is Yatirim
    df = _fetch_isyatirim_df(sym, period_days)
    if df is not None and len(df) >= 10:
        return df

    # 3. Yahoo HTTP DataFrame
    for ticker in [f"{sym}.IS", sym]:
        df = _fetch_yahoo_http_df(ticker, period_days)
        if df is not None and len(df) >= 10:
            return df

    # 3. yfinance
    if YF_OK:
        try:
            h = yf.Ticker(f"{sym}.IS").history(period=period, timeout=10)
            if h is not None and not h.empty and len(h) >= 10:
                # NaN temizligi: yfinance verisi de NaN icerebilir
                if 'Close' in h.columns:
                    h = h.dropna(subset=['Close'])
                    if 'Open' in h.columns: h['Open'] = h['Open'].fillna(h['Close'])
                    if 'High' in h.columns: h['High'] = h['High'].fillna(h['Close'])
                    if 'Low' in h.columns: h['Low'] = h['Low'].fillna(h['Close'])
                    if 'Volume' in h.columns: h['Volume'] = h['Volume'].fillna(0)
                print(f"  [YF-HIST] {sym} OK: {len(h)} bar")
                return h
        except Exception as e:
            print(f"  [YF-HIST] {sym}: {e}")

    return None


def _process_stock(sym, retry_count=2):
    """Tek hisseyi cek ve cache formatinda dondur (thread-safe)"""
    try:
        data = _fetch_stock_data(sym, retry_count=retry_count)
        if data:
            cur, prev = sf(data['close']), sf(data['prev'])
            if prev > 0:
                ch = sf(cur - prev); o = sf(data.get('open', cur))
                return sym, {
                    'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                    'price': cur, 'prevClose': prev,
                    'change': ch, 'changePct': sf(ch / prev * 100),
                    'volume': si(data.get('volume', 0)),
                    'open': o, 'high': sf(data.get('high', cur)),
                    'low': sf(data.get('low', cur)),
                    'gap': sf(o - prev), 'gapPct': sf((o - prev) / prev * 100),
                }
    except Exception as e:
        print(f"  [WORKER] {sym} hata: {e}")
    return sym, None

def _fetch_stocks_parallel(symbols, label="STOCKS", retry_count=2):
    """Hisse listesini paralel cek, basarisizlari dondur"""
    ok_count = 0
    fail_list = []
    total = len(symbols)

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(_process_stock, sym, retry_count): sym for sym in symbols}
        done_count = 0
        for future in as_completed(futures):
            sym, result = future.result()
            done_count += 1
            if result:
                _cset(_stock_cache, sym, result)
                ok_count += 1
            else:
                fail_list.append(sym)
                # NOT: _ctouch ile TS guncellemiyoruz — fetch fail olunca eski TS
                # ile 30 dk (CACHE_STALE_TTL) stale serve edilir, sonra dusurulur.
                # _ctouch eskiden "stale → taze gibi gorunsun" yaratiyordu (yaniltici).
            _status['loaded'] = _status.get('loaded', 0) + 1
            if done_count % 20 == 0 or done_count == total:
                print(f"  [{label}] {done_count}/{total}, cache={len(_stock_cache)}, fail={len(fail_list)}")

    print(f"[LOADER] {label}: {ok_count} OK, {len(fail_list)} basarisiz")
    if fail_list:
        print(f"[LOADER] {label} basarisiz: {fail_list}")
    return fail_list

def _background_loop():
    print(f"[LOADER] Thread basliyor, YF={YF_OK}, workers={PARALLEL_WORKERS}")

    while True:
        t0 = time.time()
        try:
            # === FAZE 1: Endeksler (paralel) ===
            _status['phase'] = 'indices'
            _status['error'] = ''
            print(f"\n[LOADER] ====== FAZE 1: Endeksler ======")

            def _fetch_one_index(item):
                key, (tsym, name) = item
                data = _fetch_index_data(key, tsym, name)
                return key, tsym, name, data

            with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                for key, tsym, name, data in executor.map(_fetch_one_index, INDEX_TICKERS.items()):
                    if data:
                        cur, prev = sf(data['close'], 4), sf(data['prev'], 4)
                        _cset(_index_cache, key, {
                            'name': name, 'value': cur,
                            'change': sf(cur - prev, 4),
                            'changePct': sf((cur - prev) / prev * 100 if prev else 0),
                            'volume': si(data.get('volume', 0)),
                        })

            print(f"[LOADER] Endeksler: {len(_index_cache)}/{len(INDEX_TICKERS)}")

            # === FAZE 2: BIST30 hisseleri (paralel) ===
            _status['phase'] = 'stocks'
            _status['total'] = len(BIST100_STOCKS)
            _status['loaded'] = 0
            print(f"\n[LOADER] ====== FAZE 2: BIST30 ({len(BIST30)} hisse) [paralel x{PARALLEL_WORKERS}] ======")

            bist30_fail = _fetch_stocks_parallel(BIST30, label="BIST30")

            # === FAZE 3: Kalan BIST100 hisseleri (paralel) ===
            with _lock:
                cached_and_valid = {s for s in BIST100_STOCKS.keys()
                                    if s in _stock_cache and time.time() - _stock_cache[s]['ts'] < CACHE_TTL}
            remaining = [s for s in BIST100_STOCKS.keys() if s not in cached_and_valid]
            phase3_fail = []
            if remaining:
                print(f"\n[LOADER] ====== FAZE 3: {len(remaining)} kalan hisse [paralel x{PARALLEL_WORKERS}] ======")
                phase3_fail = _fetch_stocks_parallel(remaining, label="BIST100")

            # === FAZE 4: Retry failed stocks (paralel) ===
            all_failed = list(set(bist30_fail + phase3_fail))
            if all_failed:
                print(f"\n[LOADER] ====== FAZE 4: {len(all_failed)} basarisiz hisse yeniden deneniyor ======")
                retry_fail = _fetch_stocks_parallel(all_failed, label="RETRY", retry_count=2)
                print(f"[LOADER] FAZE 4: {len(all_failed) - len(retry_fail)}/{len(all_failed)} kurtarildi")

            elapsed = round(time.time() - t0, 1)
            _status['phase'] = 'done'
            _status['lastRun'] = datetime.now().isoformat()
            print(f"\n[LOADER] ====== SONUC: {len(_stock_cache)} hisse, {len(_index_cache)} endeks ({elapsed}s) ======\n")

            # === FAZE 5: Tarihsel veri on-yukleme (ayri thread - ana donguyu BLOKLAMAZ) ===
            threading.Thread(target=_preload_hist_data, daemon=True).start()

            # === FAZE 5b: DB snapshot kaydet (restart sonrası hızlı preload için) ===
            _save_fn = _get_db_save_snapshot()
            if _save_fn:
                threading.Thread(target=_save_fn, daemon=True).start()

            # === FAZE 5c: TradingView WebSocket — BIST100 tum hisselere abone (live tick) ===
            # Loader'in doldurdugu prevClose baseline ustune TV tick'leri _stock_cache'i
            # gercek-zamanli gunceller (realtime_prices._on_price_update bridge'i).
            try:
                def _tv_sync():
                    try:
                        from realtime_prices import sync_subscriptions
                        sync_subscriptions(list(BIST100_STOCKS.keys()), throttle_sec=0.05)
                    except Exception as tve:
                        print(f"[RT-PRICES] BIST100 sync hatasi: {tve}")
                threading.Thread(target=_tv_sync, daemon=True).start()
            except Exception as e:
                print(f"[RT-PRICES] sync thread baslatilamadi: {e}")

            # === FAZE 6: Otomatik alert kontrolu (cooldown destekli) ===
            _auto_check_all_alerts()

            # === FAZE 7: Otomatik alim-satim motoru ===
            try:
                _engine_fn = _get_auto_engine()
                if _engine_fn:
                    _engine_fn()
            except Exception as ate:
                print(f"[AUTO-TRADE] Engine hatasi: {ate}")

        except Exception as e:
            print(f"[LOADER] FATAL: {e}")
            traceback.print_exc()
            _status['phase'] = 'error'
            _status['error'] = str(e)

        time.sleep(300)


def _auto_check_all_alerts():
    """Background loop'ta tum aktif alert'leri kontrol et (cooldown destekli)"""
    try:
        db = _get_db()()
        rows = db.execute("SELECT * FROM alerts WHERE active=1").fetchall()
        if not rows:
            db.close()
            return

        now = datetime.now()
        triggered_count = 0
        for r in rows:
            # Cooldown kontrolu (ayni alert 30dk icinde tekrar tetiklenmez)
            cooldown_until = r['cooldown_until'] if 'cooldown_until' in r.keys() else None
            if cooldown_until:
                try:
                    cd_time = datetime.fromisoformat(cooldown_until)
                    if now < cd_time:
                        continue
                except Exception:
                    pass

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
                cooldown_end = (now + timedelta(minutes=30)).isoformat()
                db.execute("UPDATE alerts SET triggered_at=?, cooldown_until=? WHERE id=?",
                           (now.isoformat(), cooldown_end, r['id']))
                triggered_count += 1

                # Telegram bildirim
                _tg = _get_send_telegram_alerts()
                if _tg: _tg(r['user_id'], [{
                    'symbol': r['symbol'], 'condition': r['condition'],
                    'targetValue': r['target_value'], 'currentPrice': price,
                    'message': f"{r['symbol']} uyarisi tetiklendi: {r['condition']} {r['target_value']} (Guncel: {price})"
                }])

        if triggered_count > 0:
            db.commit()
            print(f"[ALERT-AUTO] {triggered_count} uyari tetiklendi")
        db.close()
    except Exception as e:
        print(f"[ALERT-AUTO] Hata: {e}")

def _preload_one_hist(sym):
    """Tek hisse icin tarihsel veriyi cek ve cache'le"""
    try:
        cached = _cget_hist(f"{sym}_1y")
        if cached is not None:
            return sym, True
        df = _fetch_hist_df(sym, '1y')
        if df is not None and len(df) >= 30:
            _cset(_hist_cache, f"{sym}_1y", df)
            return sym, True
    except Exception as e:
        print(f"  [HIST] {sym}: {e}")
    return sym, False

_hist_preload_running = False

def _preload_hist_data():
    """Tum hisselerin tarihsel verisini paralel on-yukle"""
    global _hist_preload_running
    if _hist_preload_running:
        print("[HIST-PRELOAD] Zaten calisiyor, atlandi")
        return
    _hist_preload_running = True
    try:
        _preload_hist_data_inner()
    finally:
        _hist_preload_running = False

def _preload_hist_data_inner():
    # XU100 endeks verisini de yukle (market regime icin)
    if _cget_hist("XU100_1y") is None:
        try:
            xu_df = _fetch_isyatirim_df("XU100", 365)
            if xu_df is not None and len(xu_df) >= 30:
                _cset(_hist_cache, "XU100_1y", xu_df)
                print("[HIST-PRELOAD] XU100 endeks verisi yuklendi")
        except Exception as e:
            print(f"[HIST-PRELOAD] XU100 hata: {e}")

    symbols = list(BIST100_STOCKS.keys())
    # Sadece cache'de olmayanlari cek
    to_fetch = [s for s in symbols if _cget_hist(f"{s}_1y") is None]
    if not to_fetch:
        print(f"[HIST-PRELOAD] Tum {len(symbols)} hisse zaten cache'de")
        return

    print(f"\n[HIST-PRELOAD] ====== {len(to_fetch)} hisse icin tarihsel veri cekilecek [paralel x{PARALLEL_WORKERS}] ======")
    t0 = time.time()
    ok = 0
    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
        futures = {executor.submit(_preload_one_hist, sym): sym for sym in to_fetch}
        done = 0
        for future in as_completed(futures):
            sym, success = future.result()
            done += 1
            if success:
                ok += 1
            if done % 30 == 0 or done == len(to_fetch):
                print(f"  [HIST-PRELOAD] {done}/{len(to_fetch)}, OK={ok}")
    elapsed = round(time.time() - t0, 1)
    print(f"[HIST-PRELOAD] Tamamlandi: {ok}/{len(to_fetch)} ({elapsed}s)\n")

_loader_init_lock = threading.Lock()

def _ensure_loader():
    global _loader_started
    # Atomik check-and-set: iki request ayni anda before_request'i tetiklerse
    # iki loader thread baslamasini onle (cache concurrent write race olur).
    with _loader_init_lock:
        if _loader_started:
            return
        _loader_started = True
        # config modulundeki _loader_started'i de guncelle — `from config import
        # _loader_started` ile baska modullerde stale binding kaliyordu, attribute
        # olarak set edersek getattr(config, '_loader_started') taze deger doner.
        import config as _cfg
        _cfg._loader_started = True
    # Cold-start: DB snapshot'ından cache'i önceden doldur (kullanıcı anında veri görür)
    try:
        _get_db_load_snapshot()()
    except Exception as e:
        print(f"[WARN] Market snapshot yuklenemedi: {e}")
    print("[LOADER] Thread baslatiliyor (before_request)")
    t = threading.Thread(target=_background_loop, daemon=True)
    t.start()

@app.before_request
def before_req():
    """Her request'te loader'in calistigini garanti et"""
    _ensure_loader()
    fn = _get_start_telegram_thread()
    if fn:
        fn()


def _fetch_index_data(key, tsym, name):
    """Endeks verisi cek.
    SIRA: Yahoo HTTP (intraday tick) → Is Yatirim daily → yfinance.
    Yahoo onceliklidir cunku Is Yatirim daily endpoint piyasa acikken bugunun barini
    dondurmuyor → 'cur=Salı close, prev=Pazartesi close' yaniltici."""
    # 1. Yahoo HTTP (intraday — piyasa acikken bugunun barini icerir)
    data = _fetch_yahoo_http(tsym)
    if data:
        print(f"  [YAHOO-IDX] {key} OK: {data['close']}")
        return data

    # 2. Is Yatirim daily (fallback — piyasa kapaliyken Salı kapanışı doğru)
    isyatirim_map = {'XU100':'XU100','XU030':'XU030','XBANK':'XBANK'}
    is_sym = isyatirim_map.get(key)
    if is_sym:
        data = _fetch_isyatirim_quick(is_sym)
        if data:
            print(f"  [ISYATIRIM-IDX] {key} OK: {data['close']}")
            return data

    # 3. yfinance son care
    if YF_OK:
        try:
            h = yf.Ticker(tsym).history(period="5d", timeout=10)
            if h is not None and not h.empty and len(h) >= 2:
                cur, prev = float(h['Close'].iloc[-1]), float(h['Close'].iloc[-2])
                print(f"  [YF-IDX] {key} OK: {cur}")
                return {'close': cur, 'prev': prev,
                        'volume': int(h['Volume'].iloc[-1]) if 'Volume' in h.columns else 0}
        except Exception as e:
            print(f"  [YF-IDX] {key}: {e}")

    return None




