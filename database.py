"""
database.py - Database initialization and market cache persistence
Extracted from backend.py
"""
import json
import time
import gzip
import traceback
import threading
import io
import os
import sqlite3
import base64

from datetime import datetime

try:
    import pandas as pd
    PD_OK = True
except ImportError:
    PD_OK = False

from config import (get_db, USE_POSTGRES, PG_OK, DATABASE_URL, DB_PATH,
                    _lock, _stock_cache, _index_cache, _hist_cache,
                    _plan_lock_cache, _plan_lock_cache_lock, app,
                    CACHE_STALE_TTL, HIST_CACHE_TTL, PLAN_MAX_LOCK_SECONDS,
                    _cget_hist, _cset)


# =====================================================================
# DATABASE INITIALIZATION
# =====================================================================

def init_db():
    if USE_POSTGRES and PG_OK:
        db = get_db()
        db.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                telegram_chat_id TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS portfolios (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_cost REAL NOT NULL,
                added_at TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS watchlists (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                symbol TEXT NOT NULL,
                added_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(user_id, symbol)
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                symbol TEXT NOT NULL,
                condition TEXT NOT NULL,
                target_value REAL NOT NULL,
                active INTEGER DEFAULT 1,
                triggered INTEGER DEFAULT 0,
                triggered_at TEXT,
                cooldown_until TEXT,
                created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS market_cache (
                cache_key TEXT PRIMARY KEY,
                data      TEXT NOT NULL,
                saved_at  DOUBLE PRECISION NOT NULL
            );
            CREATE TABLE IF NOT EXISTS auto_config (
                user_id TEXT PRIMARY KEY REFERENCES users(id),
                enabled INTEGER DEFAULT 0,
                capital REAL DEFAULT 100000,
                max_positions INTEGER DEFAULT 5,
                risk_per_trade REAL DEFAULT 2.0,
                min_score REAL DEFAULT 5.0,
                min_confidence REAL DEFAULT 60,
                trade_style TEXT DEFAULT 'swing',
                stop_loss_pct REAL DEFAULT 3.0,
                take_profit_pct REAL DEFAULT 6.0,
                trailing_stop INTEGER DEFAULT 1,
                trailing_pct REAL DEFAULT 2.0,
                allowed_symbols TEXT DEFAULT '',
                blocked_symbols TEXT DEFAULT '',
                max_daily_trades INTEGER DEFAULT 3,
                updated_at TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS auto_positions (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                symbol TEXT NOT NULL,
                side TEXT NOT NULL DEFAULT 'long',
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                stop_loss REAL,
                take_profit1 REAL,
                take_profit2 REAL,
                take_profit3 REAL,
                trailing_stop REAL,
                highest_price REAL,
                status TEXT DEFAULT 'open',
                opened_at TIMESTAMP DEFAULT NOW(),
                closed_at TIMESTAMP,
                close_price REAL,
                close_reason TEXT,
                pnl REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS auto_trades (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id),
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                reason TEXT,
                signal_score REAL,
                confidence REAL,
                position_id INTEGER,
                created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE TABLE IF NOT EXISTS signal_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                action TEXT NOT NULL,
                score REAL NOT NULL,
                price_at_signal REAL,
                rsi REAL,
                macd REAL,
                logged_at REAL NOT NULL,
                outcome_checked INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL REFERENCES signal_log(id),
                horizon_days INTEGER NOT NULL,
                price_at_horizon REAL,
                return_pct REAL,
                win INTEGER DEFAULT 0,
                UNIQUE(signal_id, horizon_days)
            );
            CREATE TABLE IF NOT EXISTS auto_decisions (
                id SERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT,
                decision TEXT NOT NULL,
                reason TEXT,
                detail TEXT,
                price REAL,
                score REAL,
                confidence REAL,
                created_at TIMESTAMP DEFAULT NOW()
            );
            CREATE INDEX IF NOT EXISTS idx_auto_decisions_user_time
                ON auto_decisions(user_id, created_at DESC);
        ''')
        db.commit()
        db.close()
        print(f"[DB] PostgreSQL hazir: {DATABASE_URL[:40]}...")
    else:
        db = get_db()
        db.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT,
                telegram_chat_id TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS portfolios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_cost REAL NOT NULL,
                added_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS watchlists (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                added_at TEXT DEFAULT (datetime('now')),
                UNIQUE(user_id, symbol),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                condition TEXT NOT NULL,
                target_value REAL NOT NULL,
                active INTEGER DEFAULT 1,
                triggered INTEGER DEFAULT 0,
                triggered_at TEXT,
                cooldown_until TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS market_cache (
                cache_key TEXT PRIMARY KEY,
                data      TEXT NOT NULL,
                saved_at  REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS auto_config (
                user_id TEXT PRIMARY KEY,
                enabled INTEGER DEFAULT 0,
                capital REAL DEFAULT 100000,
                max_positions INTEGER DEFAULT 5,
                risk_per_trade REAL DEFAULT 2.0,
                min_score REAL DEFAULT 5.0,
                min_confidence REAL DEFAULT 60,
                trade_style TEXT DEFAULT 'swing',
                stop_loss_pct REAL DEFAULT 3.0,
                take_profit_pct REAL DEFAULT 6.0,
                trailing_stop INTEGER DEFAULT 1,
                trailing_pct REAL DEFAULT 2.0,
                allowed_symbols TEXT DEFAULT '',
                blocked_symbols TEXT DEFAULT '',
                max_daily_trades INTEGER DEFAULT 3,
                updated_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS auto_positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL DEFAULT 'long',
                entry_price REAL NOT NULL,
                quantity REAL NOT NULL,
                stop_loss REAL,
                take_profit1 REAL,
                take_profit2 REAL,
                take_profit3 REAL,
                trailing_stop REAL,
                highest_price REAL,
                status TEXT DEFAULT 'open',
                opened_at TEXT DEFAULT (datetime('now')),
                closed_at TEXT,
                close_price REAL,
                close_reason TEXT,
                pnl REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS auto_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                reason TEXT,
                signal_score REAL,
                confidence REAL,
                position_id INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS signal_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                action TEXT NOT NULL,
                score REAL NOT NULL,
                price_at_signal REAL,
                rsi REAL,
                macd REAL,
                logged_at REAL NOT NULL,
                outcome_checked INTEGER DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS signal_outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id INTEGER NOT NULL,
                horizon_days INTEGER NOT NULL,
                price_at_horizon REAL,
                return_pct REAL,
                win INTEGER DEFAULT 0,
                UNIQUE(signal_id, horizon_days),
                FOREIGN KEY(signal_id) REFERENCES signal_log(id)
            );
            CREATE TABLE IF NOT EXISTS auto_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT,
                decision TEXT NOT NULL,
                reason TEXT,
                detail TEXT,
                price REAL,
                score REAL,
                confidence REAL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_auto_decisions_user_time
                ON auto_decisions(user_id, created_at DESC);
        ''')
        db.commit()
        db.close()
        print("[DB] SQLite hazir:", DB_PATH)

    # Migrations — idempotent ALTER TABLE (both Postgres & SQLite tolerate IF NOT EXISTS via try/except)
    _migrations = [
        ("ALTER TABLE auto_config ADD COLUMN panic_sell_enabled INTEGER DEFAULT 0", True),
        ("ALTER TABLE auto_config ADD COLUMN panic_drop_pct REAL DEFAULT 2.0", True),
        ("ALTER TABLE auto_config ADD COLUMN panic_window_min INTEGER DEFAULT 5", True),
        # Komisyon + BSMV (Midas BIST'te 0; broker degisirse veya US stocks/Matriks icin doldurulur)
        ("ALTER TABLE auto_config ADD COLUMN commission_pct REAL DEFAULT 0", True),
        ("ALTER TABLE auto_config ADD COLUMN bsmv_pct REAL DEFAULT 5", True),
        # Runner-mode trailing: TP2 sonrasi pozisyonda daha siki trail (%1 default vs normal %2)
        ("ALTER TABLE auto_config ADD COLUMN tight_trailing_pct REAL DEFAULT 1.0", True),
        # Sektor cap: ayni sektorden max N pozisyon (korelasyon riski)
        ("ALTER TABLE auto_config ADD COLUMN max_per_sector INTEGER DEFAULT 2", True),
        # Likidite filtresi: 20 gunluk ortalama TL turnover (Close*Volume) esigi.
        # Cok sig hisseler (1M TL altinda gunluk islem hacmi) filtrelenir.
        ("ALTER TABLE auto_config ADD COLUMN min_turnover_tl REAL DEFAULT 1000000", True),
        # min_score default 8→5 bumpup (yalnız 8.0 bırakılmış default kayıtlar)
        ("UPDATE auto_config SET min_score=5.0 WHERE min_score=8.0", False),
        # SL cooldown persistence (restart sonrası kaybolmasın)
        ("CREATE TABLE IF NOT EXISTS sl_cooldown ("
         "  uid_sym TEXT PRIMARY KEY,"
         "  hit_at REAL NOT NULL,"
         "  trade_style TEXT NOT NULL"
         ")", True),
    ]
    for _mig, _idempotent in _migrations:
        try:
            _mdb = get_db()
            try:
                _mdb.execute(_mig)
                _mdb.commit()
            except Exception as _me:
                if not _idempotent:
                    print(f"[DB-MIGRATION] Uyarı: '{_mig[:60]}…' hata: {_me}")
                # Idempotent ALTER TABLE'larda 'column already exists' beklenen hata
            finally:
                _mdb.close()
        except Exception as _oe:
            print(f"[DB-MIGRATION] Bağlantı hatası: {_oe}")


# =====================================================================
# MARKET CACHE — DB'ye snapshot kaydet / yükle (restart sonrası preload)
# =====================================================================

def _db_upsert_cache(key, data, ts):
    """market_cache tablosuna yaz (Postgres ve SQLite uyumlu)"""
    try:
        if USE_POSTGRES and PG_OK:
            # PgConnection.execute() ? → %s dönüşümü yapar
            db = get_db()
            db.execute(
                "INSERT INTO market_cache (cache_key, data, saved_at) VALUES (?, ?, ?) "
                "ON CONFLICT (cache_key) DO UPDATE SET data=EXCLUDED.data, saved_at=EXCLUDED.saved_at",
                (key, data, ts)
            )
            db.commit()
            db.close()
        else:
            db = sqlite3.connect(DB_PATH)
            db.execute(
                "INSERT OR REPLACE INTO market_cache (cache_key, data, saved_at) VALUES (?,?,?)",
                (key, data, ts)
            )
            db.commit()
            db.close()
    except Exception as e:
        print(f"[CACHE-SAVE] DB yazma hatası: {e}")


def _db_get_cache(key):
    """market_cache'den oku → (data, saved_at) tuple veya None"""
    try:
        db = get_db()
        row = db.execute(
            "SELECT data, saved_at FROM market_cache WHERE cache_key=?", (key,)
        ).fetchone()
        db.close()
        if row:
            return row['data'], row['saved_at']
        return None
    except Exception:
        return None


def _db_save_market_snapshot():
    """Her load cycle sonunda fiyat + endeks + tarihsel veriyi DB'ye kaydet"""
    # _cget_hist and _cset imported from config
    try:
        with _lock:
            stocks_snap = {k: v for k, v in _stock_cache.items()}
            indices_snap = {k: v for k, v in _index_cache.items()}

        price_payload = json.dumps({'stocks': stocks_snap, 'indices': indices_snap})

        # Tarihsel veri (gzip sıkıştırmalı)
        with _lock:
            hist_keys = list(_hist_cache.keys())
        hist_payload = {}
        for hk in hist_keys:
            df = _cget_hist(hk)
            if df is not None:
                try:
                    raw = df.to_json(orient='split')
                    compressed = base64.b64encode(gzip.compress(raw.encode())).decode()
                    hist_payload[hk] = compressed
                except Exception:
                    pass
        hist_data = json.dumps(hist_payload)

        now = time.time()
        _db_upsert_cache('price_snapshot', price_payload, now)
        _db_upsert_cache('hist_snapshot', hist_data, now)
        print(f"[CACHE-SAVE] {len(stocks_snap)} hisse, {len(hist_payload)} tarihsel veri kaydedildi")
    except Exception as e:
        print(f"[CACHE-SAVE] Hata: {e}")


def _db_load_market_snapshot():
    """Cold-start: DB snapshot'ından in-memory cache'i hızla doldur"""
    # Constants imported from config
    # _cget_hist and _cset imported from config
    try:
        price_row = _db_get_cache('price_snapshot')
        if price_row:
            payload, saved_at = price_row
            age = time.time() - float(saved_at)
            if age < CACHE_STALE_TTL:  # 30 dakikadan tazeyse yükle
                snapshot = json.loads(payload)
                now = time.time()
                with _lock:
                    for k, v in snapshot.get('stocks', {}).items():
                        _stock_cache[k] = {'data': v['data'], 'ts': now}
                    for k, v in snapshot.get('indices', {}).items():
                        _index_cache[k] = {'data': v['data'], 'ts': now}
                print(f"[CACHE-LOAD] {len(snapshot.get('stocks',{}))} hisse, "
                      f"{len(snapshot.get('indices',{}))} endeks DB'den yüklendi (yaş: {int(age)}s)")
            else:
                print(f"[CACHE-LOAD] Snapshot çok eski ({int(age)}s), atlanıyor")
    except Exception as e:
        print(f"[CACHE-LOAD] Fiyat snapshot hatası: {e}")

    try:
        hist_row = _db_get_cache('hist_snapshot')
        if hist_row:
            payload, saved_at = hist_row
            age = time.time() - float(saved_at)
            if age < HIST_CACHE_TTL:  # 1 saatten tazeyse yükle
                hist_payload = json.loads(payload)
                loaded = 0
                for hk, compressed in hist_payload.items():
                    try:
                        raw = gzip.decompress(base64.b64decode(compressed)).decode()
                        df = pd.read_json(io.StringIO(raw), orient='split')
                        if len(df) >= 10:
                            _cset(_hist_cache, hk, df)
                            loaded += 1
                    except Exception:
                        pass
                print(f"[CACHE-LOAD] {loaded} tarihsel veri DB'den yüklendi (yaş: {int(age)}s)")
    except Exception as e:
        print(f"[CACHE-LOAD] Tarihsel snapshot hatası: {e}")

    try:
        plan_row = _db_get_cache('plan_locks')
        if plan_row:
            payload, saved_at = plan_row
            age = time.time() - float(saved_at)
            if age < PLAN_MAX_LOCK_SECONDS:
                snap = json.loads(payload)
                now = time.time()
                loaded_plans = 0
                with _plan_lock_cache_lock:
                    for key, entry in snap.items():
                        # Kaydedildiginden bu yana gecen sureyi locked_at'a ekle
                        # (kilitleme suresi restart sonrasi da devam eder)
                        entry_age = now - float(entry.get('locked_at', now))
                        if entry_age < PLAN_MAX_LOCK_SECONDS:
                            _plan_lock_cache[key] = entry
                            loaded_plans += 1
                print(f"[CACHE-LOAD] {loaded_plans} plan kilidi DB'den yüklendi (yaş: {int(age)}s)")
    except Exception as e:
        print(f"[CACHE-LOAD] Plan kilidi snapshot hatası: {e}")


# =====================================================================
# PLAN LOCK PERSISTENCE
# =====================================================================

def _db_save_plan_locks():
    """Plan kilitleme cache'ini DB'ye kaydet"""
    try:
        with _plan_lock_cache_lock:
            snap = dict(_plan_lock_cache)
        _db_upsert_cache('plan_locks', json.dumps(snap), time.time())
    except Exception as e:
        print(f"[PLAN-LOCK-SAVE] Hata: {e}")
