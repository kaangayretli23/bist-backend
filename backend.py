"""
BIST Pro v7.0.0 - IS YATIRIM API PRIMARY + Advanced Analytics
Thread guvenli: before_request ile lazy start.
Hicbir route yfinance CAGIRMAZ.
SQLite veritabani, kullanici sistemi, backtest, KAP haberleri
"""
from flask import Flask, jsonify, request, send_from_directory, make_response, session
from flask_cors import CORS
import traceback, os, time, threading, json, hashlib, sqlite3, uuid, re, gzip, io
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YF_OK = True
except ImportError:
    YF_OK = False

try:
    import pandas as pd
    import numpy as np
except ImportError:
    pass

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Postgres: DATABASE_URL env var (Render Postgres addon otomatik ekler)
# Fallback: SQLite (lokal gelistirme)
DATABASE_URL = os.environ.get('DATABASE_URL', '')
USE_POSTGRES = bool(DATABASE_URL)

# SQLite fallback
DB_PATH = os.environ.get('DB_PATH', os.path.join(BASE_DIR, 'bist.db'))
os.makedirs(os.path.dirname(DB_PATH) or '.', exist_ok=True)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'bist-pro-secret-' + str(hash(BASE_DIR)))
CORS(app, supports_credentials=True)

# =====================================================================
# DATABASE (Postgres birincil, SQLite yedek)
# =====================================================================
try:
    import psycopg2
    import psycopg2.extras
    PG_OK = True
except ImportError:
    PG_OK = False

class PgRowWrapper:
    """psycopg2 row'u sqlite3.Row gibi dict-like erisilebilir yapan wrapper"""
    def __init__(self, row_dict):
        self._d = row_dict or {}
    def __getitem__(self, key):
        return self._d[key]
    def __contains__(self, key):
        return key in self._d
    def get(self, key, default=None):
        return self._d.get(key, default)
    def keys(self):
        return self._d.keys()

class PgConnection:
    """Postgres baglantisi - sqlite3 arayuzuyle uyumlu wrapper"""
    def __init__(self, conn):
        self._conn = conn
        self._cursor = None
    def execute(self, sql, params=None):
        # SQLite ? parametrelerini Postgres %s'e cevir
        sql = sql.replace('?', '%s')
        cur = self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params or ())
        self._cursor = cur
        return self
    def executescript(self, sql):
        cur = self._conn.cursor()
        cur.execute(sql)
        self._cursor = cur
    def fetchall(self):
        if self._cursor:
            return [PgRowWrapper(r) for r in self._cursor.fetchall()]
        return []
    def fetchone(self):
        if self._cursor:
            r = self._cursor.fetchone()
            return PgRowWrapper(r) if r else None
        return None
    def commit(self):
        self._conn.commit()
    def close(self):
        self._conn.close()
    @property
    def lastrowid(self):
        return self._cursor.lastrowid if self._cursor else None

def get_db():
    if USE_POSTGRES and PG_OK:
        conn = psycopg2.connect(DATABASE_URL)
        return PgConnection(conn)
    else:
        db = sqlite3.connect(DB_PATH)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

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
        ''')
        db.commit()
        db.close()
        print("[DB] SQLite hazir:", DB_PATH)

init_db()

def hash_password(pw):
    return hashlib.sha256((pw + app.secret_key).encode()).hexdigest()

@app.after_request
def after_req(resp):
    # 4xx/5xx HTML -> JSON
    if resp.status_code >= 400 and 'text/html' in (resp.content_type or ''):
        resp = make_response(json.dumps({"error": f"HTTP {resp.status_code}"}), resp.status_code)
        resp.headers['Content-Type'] = 'application/json'
    # gzip sikistirma (>1KB ve henuz sikistirilmamis)
    if (resp.status_code == 200
        and 'Content-Encoding' not in resp.headers
        and resp.content_length and resp.content_length > 1024
        and 'gzip' in request.headers.get('Accept-Encoding', '')):
        ct = resp.content_type or ''
        if 'json' in ct or 'javascript' in ct or 'text/' in ct:
            data = resp.get_data()
            buf = io.BytesIO()
            with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as gz:
                gz.write(data)
            compressed = buf.getvalue()
            if len(compressed) < len(data):
                resp.set_data(compressed)
                resp.headers['Content-Encoding'] = 'gzip'
                resp.headers['Content-Length'] = len(compressed)
                resp.headers['Vary'] = 'Accept-Encoding'
    return resp

@app.errorhandler(404)
def e404(e): return jsonify({'error':'Not found'}), 404
@app.errorhandler(500)
def e500(e): return jsonify({'error':str(e)}), 500
@app.errorhandler(Exception)
def eall(e):
    print(f"ERR: {traceback.format_exc()}")
    return jsonify({'error':str(e)}), 500

def sf(v, d=2):
    try:
        f = float(v)
        return 0.0 if f != f else round(f, d)
    except: return 0.0

def si(v):
    try: return int(v)
    except: return 0

def safe_dict(d):
    if isinstance(d, dict): return {str(k): safe_dict(v) for k, v in d.items()}
    if isinstance(d, list): return [safe_dict(i) for i in d]
    if hasattr(d, 'item'): return d.item()
    if isinstance(d, float): return 0.0 if d != d else round(d, 4)
    return d

# =====================================================================
# RESPONSE WRAPPER: lastUpdated + stale + data_quality
# =====================================================================
def _cache_freshness():
    """Cache durumunu dondur: fresh/stale/loading"""
    now = time.time()
    with _lock:
        if not _stock_cache:
            return 'loading', None
        ages = [now - v['ts'] for v in _stock_cache.values()]
    avg_age = sum(ages) / len(ages) if ages else 9999
    if avg_age < CACHE_TTL:
        return 'fresh', datetime.fromtimestamp(now - avg_age).isoformat()
    elif avg_age < CACHE_STALE_TTL:
        return 'stale', datetime.fromtimestamp(now - avg_age).isoformat()
    else:
        return 'expired', None

def _api_meta(data_quality=None, extra=None):
    """Tum API response'larina eklenecek meta bilgisi"""
    freshness, last_updated = _cache_freshness()
    meta = {
        'lastUpdated': last_updated or _status.get('lastRun'),
        'snapshotTimestamp': datetime.now().isoformat(),
        'dataQuality': data_quality or freshness,
        'loaderPhase': _status.get('phase', 'idle'),
    }
    if extra:
        meta.update(extra)
    return meta

BIST100_STOCKS = {
    'THYAO':'Turk Hava Yollari','GARAN':'Garanti BBVA','ISCTR':'Is Bankasi (C)',
    'AKBNK':'Akbank','TUPRS':'Tupras','BIMAS':'BIM','SAHOL':'Sabanci Holding',
    'KCHOL':'Koc Holding','EREGL':'Eregli Demir Celik','SISE':'Sise Cam',
    'PETKM':'Petkim','ASELS':'Aselsan','TOASO':'Tofas','TCELL':'Turkcell',
    'ENKAI':'Enka Insaat','KOZAL':'Koza Altin','KRDMD':'Kardemir (D)',
    'TTKOM':'Turk Telekom','ARCLK':'Arcelik','SOKM':'Sok Marketler',
    'HEKTS':'Hektas','PGSUS':'Pegasus','TAVHL':'TAV Havalimanlari',
    'DOHOL':'Dogan Holding','VESTL':'Vestel','MGROS':'Migros',
    'FROTO':'Ford Otosan','SASA':'SASA Polyester','EKGYO':'Emlak Konut GYO',
    'TTRAK':'Turk Traktor','AEFES':'Anadolu Efes','YKBNK':'Yapi Kredi',
    'VAKBN':'Vakifbank','HALKB':'Halkbank','TSKB':'TSKB',
    'ALARK':'Alarko Holding','OTKAR':'Otokar','CEMTS':'Cemtas',
    'CIMSA':'Cimsa','GUBRF':'Gubre Fabrikalari','KONTR':'Kontrolmatik',
    'KOZAA':'Koza Anadolu','SKBNK':'Sekerbank','ISGYO':'Is GYO',
    'ULKER':'Ulker','CCOLA':'Coca-Cola Icecek','TRGYO':'Torunlar GYO',
    'ENJSA':'Enerjisa','AKSEN':'Aksa Enerji','ODAS':'Odas Elektrik',
    'AGHOL':'AG Anadolu Grubu','DOAS':'Dogus Otomotiv','BERA':'Bera Holding',
    'MPARK':'MLP Saglik','ANHYT':'Anadolu Hayat','BTCIM':'Baticim',
    'BUCIM':'Bursa Cimento','EGEEN':'Ege Endustri','ISMEN':'Is Yatirim Menkul',
    'LOGO':'Logo Yazilim','MAVI':'Mavi Giyim','NETAS':'Netas Telekom',
    'TKFEN':'Tekfen Holding','TURSG':'Turkiye Sigorta','ZOREN':'Zorlu Enerji',
    'AKFGY':'Akfen GYO','AKFYE':'Akfen Yenilenebilir','ALFAS':'Alfa Solar',
    'ALTNY':'Altin Yunus','AYDEM':'Aydem Enerji','BASGZ':'Basgazete',
    'BIOEN':'Biotrend Enerji','BRSAN':'Borusan Mannesmann','CANTE':'Cantek Holding',
    'DESA':'Desa Deri','EBEBK':'Ebebek','ECILC':'Eczacibasi Ilac',
    'EUPWR':'Europower Enerji','GESAN':'Giresun San.','GLYHO':'Global Yatirim Holding',
    'GWIND':'Galata Wind','IPEKE':'Ipek Dogal Enerji','ISDMR':'Iskenderun Demir',
    'KAYSE':'Kayseri Seker','KLSER':'Kaleseramik','KMPUR':'Kimpur Kimya',
    'KZBGY':'Kizilbuk GYO','LILAK':'Lilak Saglik','LMKDC':'Limak Dogu Cimento',
    'MAGEN':'MedyaGrup Enerji','MAKIM':'Makina Takim','MIATK':'Mia Teknoloji',
    'OBAMS':'Oba Makarnacilik','PAPIL':'Papilon Savunma','REEDR':'Reedr Teknoloji',
    'RGYAS':'Reysas GYO','ROYAL':'Royal Hali',
    'TABGD':'Tablo Gida','TGSAS':'TGS Dis Ticaret','TRILC':'Turk Ilac Serum',
    'TUKAS':'Tukas','YEOTK':'Yeo Teknoloji','ASSEN':'Assan Aluminyum',
    'ASTOR':'Astor Enerji','AGROT':'Agrotech','AHGAZ':'Ahlatci Dogalgaz',
    'AKCNS':'Akcansa Cimento','AKSA':'Aksa Akrilik','ALBRK':'Albaraka Turk',
    'ANSGR':'Anadolu Sigorta','ARDYZ':'Ard Grup Bilisim','BFREN':'Bosch Fren',
    'BIENY':'Bien Yapi','BOBET':'Bogazici Beton','BRYAT':'Borusan Yatirim',
    'CWENE':'CW Enerji','ECZYT':'Eczacibasi Yatirim','ENERY':'Enerya Enerji',
    'EUREN':'Europen Endustri','IZENR':'Izdemir Enerji','KCAER':'Kocaer Celik',
    'KONYA':'Konya Cimento','OYAKC':'Oyak Cimento','QUAGR':'Qua Granite',
    'SAYAS':'Say Yenilenebilir','SDTTR':'SDT Uzay Savunma','SELEC':'Selecuk Ecza',
    'SMRTG':'Smartiks Yazilim','TKNSA':'Teknosa','VESBE':'Vestel Beyaz Esya',
    'YUNSA':'Yunsa'
}
BIST30 = ['THYAO','GARAN','ISCTR','AKBNK','TUPRS','BIMAS','SAHOL','KCHOL',
    'EREGL','SISE','ASELS','TCELL','ENKAI','FROTO','PGSUS','TAVHL',
    'TOASO','ARCLK','PETKM','TTKOM','KOZAL','YKBNK','VAKBN','HALKB',
    'EKGYO','SASA','MGROS','SOKM','DOHOL','VESTL']
SECTOR_MAP = {
    'bankacilik':['GARAN','ISCTR','AKBNK','YKBNK','VAKBN','HALKB','TSKB','SKBNK'],
    'havacilik':['THYAO','PGSUS','TAVHL'],
    'otomotiv':['TOASO','FROTO','DOAS','TTRAK','OTKAR'],
    'enerji':['TUPRS','PETKM','AKSEN','ENJSA','ODAS','ZOREN','ASTOR'],
    'holding':['SAHOL','KCHOL','DOHOL','AGHOL','ALARK'],
    'perakende':['BIMAS','SOKM','MGROS','MAVI'],
    'teknoloji':['ASELS','LOGO','NETAS'],
    'telekom':['TCELL','TTKOM'],
    'demir_celik':['EREGL','KRDMD','CEMTS'],
    'gida':['ULKER','CCOLA','AEFES'],
    'insaat':['ENKAI','TKFEN'],
    'gayrimenkul':['EKGYO','TRGYO','ISGYO'],
}
INDEX_TICKERS = {
    'XU100':('XU100.IS','BIST 100'),'XU030':('XU030.IS','BIST 30'),
    'XBANK':('XBANK.IS','Bankacilik'),'USDTRY':('USDTRY=X','USD/TRY'),
    'GOLD':('GC=F','Altin (USD)'),
    'SILVER':('SI=F','Gumus (USD)'),
    'EURTRY':('EURTRY=X','EUR/TRY'),
}

# =====================================================================
# CACHE
# =====================================================================
_lock = threading.Lock()
_stock_cache = {}
_index_cache = {}
_hist_cache = {}
_loader_started = False
_status = {'phase':'idle','loaded':0,'total':0,'lastRun':None,'error':''}
CACHE_TTL = 600
CACHE_STALE_TTL = 1800  # Stale data served up to 30 min (prevents stocks from disappearing on fetch failure)
HIST_CACHE_TTL = 3600   # Tarihsel veri 1 saat gecerli (gun ici degismez)

def _cget(store, key):
    with _lock:
        item = store.get(key)
        if item and time.time() - item['ts'] < CACHE_TTL:
            return item['data']
    return None

def _cget_hist(key):
    """Tarihsel veri icin uzun TTL'li cache okuma"""
    with _lock:
        item = _hist_cache.get(key)
        if item and time.time() - item['ts'] < HIST_CACHE_TTL:
            return item['data']
    return None

def _cset(store, key, data):
    with _lock:
        store[key] = {'data': data, 'ts': time.time()}

def _ctouch(store, key):
    """Refresh timestamp of existing cache entry (keeps stale data alive on fetch failure)"""
    with _lock:
        if key in store:
            store[key]['ts'] = time.time()
            return True
    return False

def _get_stocks(symbols=None):
    with _lock:
        now = time.time()
        if symbols:
            return [_stock_cache[s]['data'] for s in symbols
                    if s in _stock_cache and now - _stock_cache[s]['ts'] < CACHE_STALE_TTL]
        return [v['data'] for v in _stock_cache.values()
                if now - v['ts'] < CACHE_STALE_TTL]

def _get_indices():
    with _lock:
        now = time.time()
        return {k:v['data'] for k,v in _index_cache.items()
                if now - v['ts'] < CACHE_STALE_TTL}


# =====================================================================
# DATA FETCHER - Is Yatirim API (birincil) + Yahoo HTTP (yedek) + yfinance (son care)
# =====================================================================
import urllib.request
import urllib.error
import requests as req_lib

IS_YATIRIM_BASE = "https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HisseTekil"
IS_YATIRIM_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://www.isyatirim.com.tr/',
    'Accept': 'application/json',
}

# ---- IS YATIRIM API ----
def _fetch_isyatirim_df(symbol, days=365):
    """Is Yatirim'dan OHLCV DataFrame cek - BIRINCIL KAYNAK"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        sd = start_date.strftime('%d-%m-%Y')
        ed = end_date.strftime('%d-%m-%Y')
        url = f"{IS_YATIRIM_BASE}?hisse={symbol}&startdate={sd}&enddate={ed}"

        print(f"  [ISYATIRIM] {symbol} {days}d cekiliyor...")
        resp = None
        last_err = None
        for http_attempt in range(2):
            try:
                resp = req_lib.get(url, headers=IS_YATIRIM_HEADERS, timeout=10)
                resp.raise_for_status()
                break
            except Exception as http_e:
                last_err = http_e
                if http_attempt < 1:
                    print(f"  [ISYATIRIM] {symbol} HTTP hata ({http_e}), 0.5s sonra tekrar...")
                    time.sleep(0.5)
        if resp is None:
            print(f"  [ISYATIRIM] {symbol} HTTP denemesi basarisiz: {last_err}")
            return None

        try:
            data = resp.json()
        except Exception as json_e:
            print(f"  [ISYATIRIM] {symbol}: JSON parse hatasi: {json_e}, ilk 200 karakter: {resp.text[:200]}")
            return None

        # API yanit formatini kontrol et (value, d, veya dogrudan liste)
        rows = data.get('value', [])
        if not rows:
            rows = data.get('d', [])
        if not rows and isinstance(data, list):
            rows = data

        if not rows or len(rows) < 2:
            print(f"  [ISYATIRIM] {symbol}: bos veri ({len(rows) if rows else 0} satir), response keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
            return None

        # Ilk satirdaki TUM kolonlari logla
        if len(rows) > 0:
            cols = list(rows[0].keys())
            print(f"  [ISYATIRIM] {symbol}: {len(rows)} satir, kolonlar: {cols}")

        # DataFrame olustur - kolon isimlerini otomatik kesfet
        df_raw = pd.DataFrame(rows)

        # Tarih kolonu - esnek arama
        date_col = None
        for c in df_raw.columns:
            cu = c.upper()
            if 'TARIH' in cu or 'DATE' in cu or 'TARH' in cu:
                date_col = c; break
        if not date_col:
            print(f"  [ISYATIRIM] {symbol}: tarih kolonu bulunamadi, kolonlar: {list(df_raw.columns)}")
            return None

        # OHLCV kolonlari - esnek mapping (API degisikliklerine karsi guclu)
        col_map = {}
        for c in df_raw.columns:
            cu = c.upper()
            # Close: KAPANIS_FIYATI, KAPANIS, HISSE_KAPANIS, vb.
            if 'Close' not in col_map and 'KAPANIS' in cu and 'DUZELTILMIS' not in cu:
                col_map['Close'] = c
            # Open: ACILIS_FIYATI, ACILIS, vb.
            elif 'Open' not in col_map and 'ACILIS' in cu:
                col_map['Open'] = c
            # High: EN_YUKSEK, YUKSEK, YUKSEK_FIYAT, vb.
            elif 'High' not in col_map and ('YUKSEK' in cu or 'EN_YUKSEK' in cu or 'HIGH' in cu):
                col_map['High'] = c
            # Low: EN_DUSUK, DUSUK, DUSUK_FIYAT, vb.
            elif 'Low' not in col_map and ('DUSUK' in cu or 'EN_DUSUK' in cu or 'LOW' in cu):
                col_map['Low'] = c
            # Volume: HACIM_LOT, HACIM (LOT), ISLEM_HACMI, vb.
            elif 'Volume' not in col_map and 'HACIM' in cu and 'TL' not in cu:
                col_map['Volume'] = c

        # Volume bulunamadiysa HACIM iceren herhangi bir kolonu dene
        if 'Volume' not in col_map:
            for c in df_raw.columns:
                cu = c.upper()
                if 'HACIM' in cu or 'VOLUME' in cu or 'ADET' in cu:
                    col_map['Volume'] = c; break

        # Close bulunamadiysa DUZELTILMIS KAPANIS veya herhangi kapanis
        if 'Close' not in col_map:
            for c in df_raw.columns:
                if 'KAPANIS' in c.upper():
                    col_map['Close'] = c; break
            if 'Close' not in col_map:
                # Son care: numerik kolonlari dene (CLOSE, PRICE, FIYAT)
                for c in df_raw.columns:
                    cu = c.upper()
                    if 'CLOSE' in cu or 'FIYAT' in cu or 'PRICE' in cu:
                        col_map['Close'] = c; break
            if 'Close' not in col_map:
                print(f"  [ISYATIRIM] {symbol}: Close kolonu bulunamadi, kolonlar: {list(df_raw.columns)}")
                return None

        print(f"  [ISYATIRIM] {symbol} mapping: {col_map}")

        # DataFrame build
        df = pd.DataFrame(index=pd.DatetimeIndex(pd.to_datetime(df_raw[date_col])))
        df['Close'] = pd.to_numeric(df_raw[col_map['Close']].values, errors='coerce')
        df['Open'] = pd.to_numeric(df_raw[col_map.get('Open', col_map['Close'])].values, errors='coerce')
        df['High'] = pd.to_numeric(df_raw[col_map.get('High', col_map['Close'])].values, errors='coerce')
        df['Low'] = pd.to_numeric(df_raw[col_map.get('Low', col_map['Close'])].values, errors='coerce')

        if 'Volume' in col_map:
            df['Volume'] = pd.to_numeric(df_raw[col_map['Volume']].values, errors='coerce').fillna(0).astype(int)
        else:
            df['Volume'] = 0

        # NaN degerlerini doldur: Open/High/Low icin Close kullan (kritik fix)
        df['Open'] = df['Open'].fillna(df['Close'])
        df['High'] = df['High'].fillna(df['Close'])
        df['Low'] = df['Low'].fillna(df['Close'])

        # High en az Close kadar, Low en fazla Close kadar olmali
        df['High'] = df[['High', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Close']].min(axis=1)

        df = df.dropna(subset=['Close']).sort_index()

        if len(df) < 2:
            return None

        # Veri kalitesi kontrolu
        nan_counts = df[['Open','High','Low','Close']].isna().sum()
        total_nan = nan_counts.sum()
        if total_nan > 0:
            print(f"  [ISYATIRIM] {symbol} UYARI: {total_nan} NaN deger doldurulamadi: {nan_counts.to_dict()}")

        # Son fiyat kontrolu
        last_close = float(df['Close'].iloc[-1])
        if last_close <= 0:
            print(f"  [ISYATIRIM] {symbol} UYARI: Son kapanis fiyati 0 veya negatif: {last_close}")

        print(f"  [ISYATIRIM] {symbol} OK: {len(df)} bar, {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}, son fiyat: {last_close}")
        return df

    except Exception as e:
        print(f"  [ISYATIRIM] {symbol} HATA: {e}")
        import traceback
        traceback.print_exc()
        return None


def _fetch_isyatirim_quick(symbol):
    """Is Yatirim'dan son fiyat bilgisi (quick - dashboard icin)"""
    try:
        df = _fetch_isyatirim_df(symbol, days=5)
        if df is None or len(df) < 2:
            return None
        cur = float(df['Close'].iloc[-1])
        prev = float(df['Close'].iloc[-2])
        return {
            'close': cur, 'prev': prev,
            'open': float(df['Open'].iloc[-1]),
            'high': float(df['High'].iloc[-1]),
            'low': float(df['Low'].iloc[-1]),
            'volume': int(df['Volume'].iloc[-1]),
        }
    except Exception as e:
        print(f"  [ISYATIRIM-Q] {symbol}: {e}")
        return None


# ---- YAHOO HTTP API (YEDEK) ----
def _fetch_yahoo_http(symbol, period1_days=14):
    """Yahoo Finance v8 chart API - yedek kaynak"""
    try:
        now = int(time.time())
        p1 = now - (period1_days * 86400)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={now}&interval=1d"
        r = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(r, timeout=8) as resp:
            raw = json.loads(resp.read().decode())

        result = raw.get('chart', {}).get('result', [])
        if not result: return None

        data = result[0]
        quote = data.get('indicators', {}).get('quote', [{}])[0]
        closes = quote.get('close', [])
        if not closes or len(closes) < 2: return None

        valid = [(i, c) for i, c in enumerate(closes) if c is not None]
        if len(valid) < 2: return None

        last_i, cur = valid[-1]; prev_i, prev = valid[-2]
        opens = quote.get('open', []); highs = quote.get('high', [])
        lows = quote.get('low', []); volumes = quote.get('volume', [])

        o_val = float(opens[last_i]) if last_i < len(opens) and opens[last_i] else float(cur)
        h_val = float(highs[last_i]) if last_i < len(highs) and highs[last_i] else float(cur)
        l_val = float(lows[last_i]) if last_i < len(lows) and lows[last_i] else float(cur)
        v_val = int(volumes[last_i]) if last_i < len(volumes) and volumes[last_i] else 0
        return {
            'close': float(cur), 'prev': float(prev),
            'open': o_val, 'high': max(h_val, float(cur)), 'low': min(l_val, float(cur)),
            'volume': v_val,
        }
    except Exception as e:
        print(f"  [YAHOO-HTTP] {symbol}: {e}")
        return None


def _fetch_yahoo_http_df(symbol, period1_days=365):
    """Yahoo Finance v8 - tam DataFrame (yedek)"""
    try:
        now = int(time.time())
        p1 = now - (period1_days * 86400)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={now}&interval=1d"
        r = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(r, timeout=12) as resp:
            raw = json.loads(resp.read().decode())

        result = raw.get('chart', {}).get('result', [])
        if not result: return None

        data = result[0]
        timestamps = data.get('timestamp', [])
        quote = data.get('indicators', {}).get('quote', [{}])[0]
        if not timestamps or not quote.get('close'): return None

        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
        df = pd.DataFrame({
            'Open': [float(v) if v else np.nan for v in quote.get('open', [])],
            'High': [float(v) if v else np.nan for v in quote.get('high', [])],
            'Low': [float(v) if v else np.nan for v in quote.get('low', [])],
            'Close': [float(v) if v else np.nan for v in quote.get('close', [])],
            'Volume': [int(v) if v else 0 for v in quote.get('volume', [])],
        }, index=pd.DatetimeIndex(dates))
        df = df.dropna(subset=['Close'])

        # NaN degerlerini doldur: Open/High/Low icin Close kullan
        df['Open'] = df['Open'].fillna(df['Close'])
        df['High'] = df['High'].fillna(df['Close'])
        df['Low'] = df['Low'].fillna(df['Close'])
        df['High'] = df[['High', 'Close']].max(axis=1)
        df['Low'] = df[['Low', 'Close']].min(axis=1)

        if len(df) < 10: return None
        print(f"  [YAHOO-DF] {symbol}: {len(df)} bar")
        return df
    except Exception as e:
        print(f"  [YAHOO-DF] {symbol}: {e}")
        return None


# ---- BIRLESIK FETCHER (3 katmanli fallback) ----
def _fetch_stock_data(sym, retry_count=2):
    """Hisse verisi cek: 1.IsYatirim -> 2.Yahoo HTTP -> 3.yfinance (retry destekli)"""
    for attempt in range(1, retry_count + 1):
        # 1. Is Yatirim
        data = _fetch_isyatirim_quick(sym)
        if data:
            return data

        # 2. Yahoo HTTP
        for ticker in [f"{sym}.IS", sym]:
            data = _fetch_yahoo_http(ticker)
            if data:
                print(f"  [YAHOO] {sym} OK")
                return data

        # 3. yfinance (son care)
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
    """Tam DataFrame cek (indicators icin): 1.IsYatirim -> 2.Yahoo HTTP DF -> 3.yfinance"""
    period_days = {'1mo':30,'3mo':90,'6mo':180,'1y':365,'2y':730,'5y':1825}.get(period, 365)

    # 1. Is Yatirim
    df = _fetch_isyatirim_df(sym, period_days)
    if df is not None and len(df) >= 10:
        return df

    # 2. Yahoo HTTP DataFrame
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


def _fetch_index_data(key, tsym, name):
    """Endeks verisi cek"""
    # Endeksler icin Is Yatirim: XU100 = BIST100 endeksi
    isyatirim_map = {'XU100':'XU100','XU030':'XU030','XBANK':'XBANK'}
    is_sym = isyatirim_map.get(key)

    if is_sym:
        data = _fetch_isyatirim_quick(is_sym)
        if data:
            print(f"  [ISYATIRIM-IDX] {key} OK: {data['close']}")
            return data

    # Yahoo HTTP fallback
    data = _fetch_yahoo_http(tsym)
    if data:
        print(f"  [YAHOO-IDX] {key} OK: {data['close']}")
        return data

    # yfinance fallback
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


# =====================================================================
# BACKGROUND LOADER (paralel - ThreadPoolExecutor)
# =====================================================================
PARALLEL_WORKERS = 12  # Paralel hisse cekme sayisi (performans iyilestirmesi)

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
                _ctouch(_stock_cache, sym)  # Keep stale data alive
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

            # === FAZE 6: Otomatik alert kontrolu (cooldown destekli) ===
            _auto_check_all_alerts()

        except Exception as e:
            print(f"[LOADER] FATAL: {e}")
            traceback.print_exc()
            _status['phase'] = 'error'
            _status['error'] = str(e)

        time.sleep(300)


def _auto_check_all_alerts():
    """Background loop'ta tum aktif alert'leri kontrol et (cooldown destekli)"""
    try:
        db = get_db()
        rows = db.execute("SELECT * FROM alerts WHERE active=1 AND triggered=0").fetchall()
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
                except:
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
                db.execute("UPDATE alerts SET triggered=1, triggered_at=?, cooldown_until=? WHERE id=?",
                           (now.isoformat(), cooldown_end, r['id']))
                triggered_count += 1

                # Telegram bildirim
                _send_telegram_alerts(r['user_id'], [{
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

def _preload_hist_data():
    """Tum hisselerin tarihsel verisini paralel on-yukle"""
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

def _ensure_loader():
    global _loader_started
    if _loader_started: return
    _loader_started = True
    print("[LOADER] Thread baslatiliyor (before_request)")
    t = threading.Thread(target=_background_loop, daemon=True)
    t.start()

@app.before_request
def before_req():
    """Her request'te loader'in calistigini garanti et"""
    _ensure_loader()
    _start_telegram_thread()

# =====================================================================
# INDICATORS
# =====================================================================
def calc_rsi(closes, period=14):
    if len(closes)<period+1: return {'name':'RSI','value':50.0,'signal':'neutral'}
    d=np.diff(closes)
    ag=float(np.mean(np.where(d>0,d,0)[-period:]))
    al=float(np.mean(np.where(d<0,-d,0)[-period:]))
    rsi=100.0 if al==0 else sf(100-100/(1+ag/al))
    return {'name':'RSI','value':rsi,'signal':'buy' if rsi<30 else ('sell' if rsi>70 else 'neutral')}

def calc_rsi_single(closes, period=14):
    if len(closes)<period+1: return None
    d=np.diff(closes)
    ag=float(np.mean(np.where(d>0,d,0)[-period:]))
    al=float(np.mean(np.where(d<0,-d,0)[-period:]))
    return 100.0 if al==0 else sf(100-100/(1+ag/al))

def calc_macd(closes):
    if len(closes)<26: return {'name':'MACD','macd':0,'signal':0,'histogram':0,'signalType':'neutral'}
    s=pd.Series(list(closes),dtype=float); ml=s.ewm(span=12).mean()-s.ewm(span=26).mean(); sl=ml.ewm(span=9).mean()
    mv,sv=sf(ml.iloc[-1]),sf(sl.iloc[-1])
    return {'name':'MACD','macd':mv,'signal':sv,'histogram':sf(mv-sv),'signalType':'buy' if mv>sv else ('sell' if mv<sv else 'neutral')}

def calc_macd_history(closes):
    if len(closes)<26: return []
    s=pd.Series(list(closes),dtype=float); ml=s.ewm(span=12).mean()-s.ewm(span=26).mean(); sl=ml.ewm(span=9).mean()
    return [{'macd':sf(ml.iloc[i]),'signal':sf(sl.iloc[i]),'histogram':sf((ml-sl).iloc[i])} for i in range(26,len(closes))]

def calc_bollinger(closes, cp, period=20):
    if len(closes)<period: return {'name':'Bollinger','upper':0,'middle':0,'lower':0,'signal':'neutral','bandwidth':0}
    r=closes[-period:]; sma,std=float(np.mean(r)),float(np.std(r))
    u,m,lo=sf(sma+2*std),sf(sma),sf(sma-2*std)
    return {'name':'Bollinger','upper':u,'middle':m,'lower':lo,'bandwidth':sf((u-lo)/m*100 if m else 0),'signal':'buy' if float(cp)<lo else ('sell' if float(cp)>u else 'neutral')}

def calc_bollinger_history(closes, period=20):
    r=[]
    for i in range(period,len(closes)):
        w=closes[i-period:i]; sma,std=float(np.mean(w)),float(np.std(w))
        r.append({'upper':sf(sma+2*std),'middle':sf(sma),'lower':sf(sma-2*std)})
    return r

EMA_PERIODS = [5, 10, 20, 50, 100, 200]

def calc_ema(closes, cp):
    result={'name':'EMA','signal':'neutral'}
    s=pd.Series(list(closes),dtype=float)
    for p in EMA_PERIODS:
        if len(closes)>=p:
            result[f'ema{p}']=sf(s.ewm(span=p).mean().iloc[-1])
    e20,e50=result.get('ema20',cp),result.get('ema50',cp)
    if float(cp)>e20>e50: result['signal']='buy'
    elif float(cp)<e20<e50: result['signal']='sell'
    return result

def calc_ema_history(closes):
    s=pd.Series(list(closes),dtype=float)
    emas={}
    for p in EMA_PERIODS:
        if len(closes)>=p:
            emas[f'ema{p}']=s.ewm(span=p).mean()
    r=[]
    for i in range(len(closes)):
        pt={}
        for k,v in emas.items():
            if i<len(v): pt[k]=sf(v.iloc[i])
        r.append(pt)
    return r

def calc_stochastic(closes, highs, lows, period=14):
    if len(closes)<period: return {'name':'Stochastic','k':50,'d':50,'signal':'neutral'}
    hi,lo,cur=float(np.max(highs[-period:])),float(np.min(lows[-period:])),float(closes[-1])
    k=sf(((cur-lo)/(hi-lo))*100 if hi!=lo else 50)
    return {'name':'Stochastic','k':k,'d':k,'signal':'buy' if k<20 else ('sell' if k>80 else 'neutral')}

def calc_stochastic_history(closes, highs, lows, period=14):
    r=[]
    for i in range(period, len(closes)):
        hi=float(np.max(highs[i-period:i+1]))
        lo=float(np.min(lows[i-period:i+1]))
        cur=float(closes[i])
        k=sf(((cur-lo)/(hi-lo))*100 if hi!=lo else 50)
        r.append({'k':k})
    # Smooth %D (3-period SMA of %K)
    for i in range(len(r)):
        if i >= 2:
            r[i]['d'] = sf((r[i]['k'] + r[i-1]['k'] + r[i-2]['k']) / 3)
        else:
            r[i]['d'] = r[i]['k']
    return r

def calc_atr(highs, lows, closes, period=14):
    if len(closes)<period+1: return {'name':'ATR','value':0,'pct':0,'signal':'neutral'}
    tr=[max(float(highs[i])-float(lows[i]),abs(float(highs[i])-float(closes[i-1])),abs(float(lows[i])-float(closes[i-1]))) for i in range(1,len(closes))]
    atr=sf(np.mean(tr[-period:]))
    return {'name':'ATR','value':atr,'pct':sf(atr/float(closes[-1])*100 if closes[-1] else 0),'signal':'neutral'}

def calc_adx(highs, lows, closes, period=14):
    n=len(closes)
    if n<period+1: return {'name':'ADX','value':25,'plusDI':0,'minusDI':0,'signal':'neutral'}
    tr,pdm,mdm=[],[],[]
    for i in range(1,n):
        hv,lv,phv,plv,pcv=float(highs[i]),float(lows[i]),float(highs[i-1]),float(lows[i-1]),float(closes[i-1])
        tr.append(max(hv-lv,abs(hv-pcv),abs(lv-pcv)))
        um,dm=hv-phv,plv-lv
        pdm.append(um if um>dm and um>0 else 0); mdm.append(dm if dm>um and dm>0 else 0)
    if len(tr)<period: return {'name':'ADX','value':25,'plusDI':0,'minusDI':0,'signal':'neutral'}
    atr_s,pdm_s,mdm_s=float(np.mean(tr[:period])),float(np.mean(pdm[:period])),float(np.mean(mdm[:period]))
    for i in range(period,len(tr)):
        atr_s=(atr_s*(period-1)+tr[i])/period; pdm_s=(pdm_s*(period-1)+pdm[i])/period; mdm_s=(mdm_s*(period-1)+mdm[i])/period
    pdi=sf((pdm_s/atr_s)*100 if atr_s else 0); mdi=sf((mdm_s/atr_s)*100 if atr_s else 0)
    ds=pdi+mdi; adx=sf(abs(pdi-mdi)/ds*100 if ds else 0)
    return {'name':'ADX','value':adx,'plusDI':pdi,'minusDI':mdi,'signal':'buy' if pdi>mdi and adx>25 else ('sell' if mdi>pdi and adx>25 else 'neutral')}

def calc_obv(closes, volumes):
    if len(closes)<10: return {'name':'OBV','value':0,'trend':'neutral','signal':'neutral'}
    obv,vals=0,[0]
    for i in range(1,len(closes)):
        if float(closes[i])>float(closes[i-1]): obv+=int(volumes[i])
        elif float(closes[i])<float(closes[i-1]): obv-=int(volumes[i])
        vals.append(obv)
    trend='up' if vals[-1]>vals[-min(10,len(vals)-1)] else 'down'
    return {'name':'OBV','value':si(abs(vals[-1])),'trend':trend,'signal':'buy' if trend=='up' else 'sell'}

def calc_support_resistance(hist):
    try:
        c,h,l=hist['Close'].values.astype(float),hist['High'].values.astype(float),hist['Low'].values.astype(float)
        # NaN temizligi: NaN degerleri Close ile degistir
        h_clean = np.where(np.isnan(h), c, h)
        l_clean = np.where(np.isnan(l), c, l)
        n=min(90,len(c)); rh,rl=h_clean[-n:],l_clean[-n:]; sups,ress=[],[]
        for i in range(2,n-2):
            if rh[i]>rh[i-1] and rh[i]>rh[i-2] and rh[i]>rh[i+1] and rh[i]>rh[i+2]: ress.append(float(rh[i]))
            if rl[i]<rl[i-1] and rl[i]<rl[i-2] and rl[i]<rl[i+1] and rl[i]<rl[i+2]: sups.append(float(rl[i]))
        cur=float(c[-1])
        return {'supports':[sf(s) for s in sorted([s for s in sups if s<cur],reverse=True)[:3]],'resistances':[sf(r) for r in sorted([r for r in ress if r>cur])[:3]],'current':sf(cur)}
    except: return {'supports':[],'resistances':[],'current':0}

def calc_fibonacci(hist):
    try:
        c=hist['Close'].values.astype(float)
        h=hist['High'].values.astype(float)
        l=hist['Low'].values.astype(float)

        # Birden fazla periyot dene: 90, 60, 30, tum veri
        for lookback in [90, 60, 30, len(c)]:
            n=min(lookback, len(c))
            if n < 5: continue
            rc, rh, rl = c[-n:], h[-n:], l[-n:]
            hi = float(np.nanmax(rh))
            lo = float(np.nanmin(rl))
            if hi != hi: hi = float(np.nanmax(rc))  # NaN fallback
            if lo != lo: lo = float(np.nanmin(rc))
            d = hi - lo
            if d > 0: break
        else:
            return {'levels':{},'currentZone':'-','trend':'-','description':'Yeterli veri yok'}

        # Trend yonu belirle
        mid_idx = n // 2
        first_half_avg = float(np.mean(rc[:mid_idx]))
        second_half_avg = float(np.mean(rc[mid_idx:]))
        trend = 'yukari' if second_half_avg > first_half_avg else 'asagi'

        # Yukari trendde: dip->tepe (retracement), asagi trendde: tepe->dip
        if trend == 'yukari':
            levels = {
                '0.0 (Tepe)': sf(hi),
                '23.6': sf(hi - d * 0.236),
                '38.2': sf(hi - d * 0.382),
                '50.0': sf(hi - d * 0.5),
                '61.8': sf(hi - d * 0.618),
                '78.6': sf(hi - d * 0.786),
                '100.0 (Dip)': sf(lo)
            }
        else:
            levels = {
                '0.0 (Dip)': sf(lo),
                '23.6': sf(lo + d * 0.236),
                '38.2': sf(lo + d * 0.382),
                '50.0': sf(lo + d * 0.5),
                '61.8': sf(lo + d * 0.618),
                '78.6': sf(lo + d * 0.786),
                '100.0 (Tepe)': sf(hi)
            }

        cur = float(c[-1])
        zone = "Belirsiz"
        lk = list(levels.keys())
        lv = list(levels.values())
        sorted_vals = sorted([(lk[i], lv[i]) for i in range(len(lv))], key=lambda x: -x[1])

        for i in range(len(sorted_vals) - 1):
            if cur <= sorted_vals[i][1] and cur >= sorted_vals[i+1][1]:
                zone = f"{sorted_vals[i][0]} - {sorted_vals[i+1][0]}"
                break

        # Destek ve direnc seviyeleri
        nearest_support = None
        nearest_resistance = None
        for k, v in sorted(levels.items(), key=lambda x: x[1]):
            if v < cur:
                nearest_support = {'level': k, 'price': v}
            elif v > cur and nearest_resistance is None:
                nearest_resistance = {'level': k, 'price': v}

        # Aciklama olustur
        desc_parts = [f"Son {n} barda analiz"]
        desc_parts.append(f"Trend: {'Yukari' if trend == 'yukari' else 'Asagi'}")
        desc_parts.append(f"Aralik: {sf(lo)} - {sf(hi)} ({sf(d)} TL fark)")
        if nearest_support:
            desc_parts.append(f"En yakin destek: {nearest_support['price']} TL ({nearest_support['level']})")
        if nearest_resistance:
            desc_parts.append(f"En yakin direnc: {nearest_resistance['price']} TL ({nearest_resistance['level']})")
        pos_pct = sf((cur - lo) / d * 100) if d > 0 else 50
        desc_parts.append(f"Fiyat araliktaki konum: %{pos_pct}")

        return {
            'levels': levels, 'high': sf(hi), 'low': sf(lo),
            'currentZone': zone, 'trend': trend,
            'nearestSupport': nearest_support,
            'nearestResistance': nearest_resistance,
            'positionPct': pos_pct,
            'lookbackBars': n,
            'description': ' | '.join(desc_parts)
        }
    except Exception as e:
        print(f"  [FIB] Hata: {e}")
        return {'levels':{},'currentZone':'-','trend':'-','description':'Hesaplama hatasi'}

def calc_williams_r(closes, highs, lows, period=14):
    if len(closes)<period: return {'name':'Williams %R','value':-50,'signal':'neutral'}
    hh=float(np.max(highs[-period:])); ll=float(np.min(lows[-period:])); cur=float(closes[-1])
    wr=sf(((hh-cur)/(hh-ll))*-100 if hh!=ll else -50)
    sig='buy' if wr<-80 else ('sell' if wr>-20 else 'neutral')
    return {'name':'Williams %R','value':wr,'signal':sig}

def calc_cci(closes, highs, lows, period=20):
    if len(closes)<period: return {'name':'CCI','value':0,'signal':'neutral'}
    tp=[(float(highs[i])+float(lows[i])+float(closes[i]))/3 for i in range(len(closes))]
    tp_r=tp[-period:]
    sma=np.mean(tp_r); md=np.mean(np.abs(np.array(tp_r)-sma))
    cci=sf((tp[-1]-sma)/(0.015*md) if md>0 else 0)
    sig='buy' if cci<-100 else ('sell' if cci>100 else 'neutral')
    return {'name':'CCI','value':cci,'signal':sig}

def calc_mfi(closes, highs, lows, volumes, period=14):
    if len(closes)<period+1: return {'name':'MFI','value':50,'signal':'neutral'}
    tp=[(float(highs[i])+float(lows[i])+float(closes[i]))/3 for i in range(len(closes))]
    pmf=nmf=0
    for i in range(-period,0):
        mf=tp[i]*float(volumes[i])
        if tp[i]>tp[i-1]: pmf+=mf
        else: nmf+=mf
    mfi=sf(100-(100/(1+pmf/nmf)) if nmf>0 else 100)
    sig='buy' if mfi<20 else ('sell' if mfi>80 else 'neutral')
    return {'name':'MFI','value':mfi,'signal':sig}

def calc_vwap(closes, highs, lows, volumes, period=20):
    if len(closes)<period: return {'name':'VWAP','value':0,'signal':'neutral'}
    tp=np.array([(float(highs[i])+float(lows[i])+float(closes[i]))/3 for i in range(-period,0)])
    vol=np.array([float(volumes[i]) for i in range(-period,0)])
    vwap=sf(np.sum(tp*vol)/np.sum(vol) if np.sum(vol)>0 else float(closes[-1]))
    cp=float(closes[-1])
    sig='buy' if cp>vwap else ('sell' if cp<vwap else 'neutral')
    return {'name':'VWAP','value':vwap,'signal':sig}

def calc_ichimoku(closes, highs, lows):
    n=len(closes)
    if n<52: return {'name':'Ichimoku','tenkan':0,'kijun':0,'signal':'neutral'}
    hh9=float(np.max(highs[-9:])); ll9=float(np.min(lows[-9:]))
    hh26=float(np.max(highs[-26:])); ll26=float(np.min(lows[-26:]))
    hh52=float(np.max(highs[-52:])); ll52=float(np.min(lows[-52:]))
    tenkan=sf((hh9+ll9)/2); kijun=sf((hh26+ll26)/2)
    ssa=sf((tenkan+kijun)/2); ssb=sf((hh52+ll52)/2)
    cp=float(closes[-1])
    if cp>ssa and cp>ssb and tenkan>kijun: sig='buy'
    elif cp<ssa and cp<ssb and tenkan<kijun: sig='sell'
    else: sig='neutral'
    return {'name':'Ichimoku','tenkan':tenkan,'kijun':kijun,'senkouA':ssa,'senkouB':ssb,'signal':sig}

def calc_psar(closes, highs, lows, af_start=0.02, af_step=0.02, af_max=0.2):
    n=len(closes)
    if n<5: return {'name':'Parabolic SAR','value':0,'trend':'neutral','signal':'neutral'}
    bull=True; af=af_start; ep=float(highs[0]); sar=float(lows[0])
    for i in range(1,n):
        hi,lo,cl=float(highs[i]),float(lows[i]),float(closes[i])
        prev_sar=sar; sar=prev_sar+af*(ep-prev_sar)
        if bull:
            if lo<sar: bull=False; sar=ep; ep=lo; af=af_start
            else:
                if hi>ep: ep=hi; af=min(af+af_step,af_max)
        else:
            if hi>sar: bull=True; sar=ep; ep=hi; af=af_start
            else:
                if lo<ep: ep=lo; af=min(af+af_step,af_max)
    trend='up' if bull else 'down'
    return {'name':'Parabolic SAR','value':sf(sar),'trend':trend,'signal':'buy' if bull else 'sell'}

def calc_pivot_points(hist):
    """Klasik, Camarilla, Woodie pivot noktalari"""
    try:
        c=float(hist['Close'].iloc[-1])
        h=float(hist['High'].iloc[-1])
        l=float(hist['Low'].iloc[-1])
        o=float(hist['Open'].iloc[-1])
        # NaN fallback: Close kullan
        if h != h: h = c
        if l != l: l = c
        if o != o: o = c
        pp=(h+l+c)/3
        classic={
            'pp':sf(pp),'r1':sf(2*pp-l),'r2':sf(pp+(h-l)),'r3':sf(h+2*(pp-l)),
            's1':sf(2*pp-h),'s2':sf(pp-(h-l)),'s3':sf(l-2*(h-pp))
        }
        d=h-l
        camarilla={
            'pp':sf(pp),'r1':sf(c+d*1.1/12),'r2':sf(c+d*1.1/6),'r3':sf(c+d*1.1/4),'r4':sf(c+d*1.1/2),
            's1':sf(c-d*1.1/12),'s2':sf(c-d*1.1/6),'s3':sf(c-d*1.1/4),'s4':sf(c-d*1.1/2)
        }
        wpp=(h+l+2*c)/4
        woodie={
            'pp':sf(wpp),'r1':sf(2*wpp-l),'r2':sf(wpp+(h-l)),
            's1':sf(2*wpp-h),'s2':sf(wpp-(h-l))
        }
        return {'classic':classic,'camarilla':camarilla,'woodie':woodie,'current':sf(c)}
    except: return {'classic':{},'camarilla':{},'woodie':{},'current':0}

def calc_roc(closes, period=12):
    """Rate of Change - momentum osilatoru"""
    if len(closes)<period+1: return {'name':'ROC','value':0,'signal':'neutral'}
    cur,prev=float(closes[-1]),float(closes[-period-1])
    roc=sf(((cur-prev)/prev)*100 if prev!=0 else 0)
    sig='buy' if roc>5 else ('sell' if roc<-5 else 'neutral')
    return {'name':'ROC','value':roc,'signal':sig,'period':period}

def calc_aroon(highs, lows, period=25):
    """Aroon Up/Down - trend yonu gostergesi"""
    if len(highs)<period+1: return {'name':'Aroon','up':50,'down':50,'signal':'neutral'}
    h_slice=list(highs[-period-1:]); l_slice=list(lows[-period-1:])
    days_since_high=period-h_slice.index(max(h_slice))
    days_since_low=period-l_slice.index(min(l_slice))
    up=sf((days_since_high/period)*100); down=sf((days_since_low/period)*100)
    if up>70 and down<30: sig='buy'
    elif down>70 and up<30: sig='sell'
    else: sig='neutral'
    return {'name':'Aroon','up':up,'down':down,'signal':sig,'oscillator':sf(up-down)}

def calc_trix(closes, period=15):
    """TRIX - triple smoothed EMA oscillator"""
    if len(closes)<period*3: return {'name':'TRIX','value':0,'signal':'neutral'}
    def ema_arr(data, p):
        result=[float(data[0])]; k=2/(p+1)
        for i in range(1,len(data)):
            result.append(float(data[i])*k + result[-1]*(1-k))
        return result
    e1=ema_arr(closes,period); e2=ema_arr(e1,period); e3=ema_arr(e2,period)
    if len(e3)<2 or e3[-2]==0: return {'name':'TRIX','value':0,'signal':'neutral'}
    trix=sf(((e3[-1]-e3[-2])/e3[-2])*10000)
    sig='buy' if trix>0 else ('sell' if trix<0 else 'neutral')
    return {'name':'TRIX','value':trix,'signal':sig}

def calc_dmi(highs, lows, closes, period=14):
    """Directional Movement - trend gucu (ADX'in detayli hali)"""
    n=len(closes)
    if n<period+1: return {'name':'DMI','diPlus':0,'diMinus':0,'adx':0,'signal':'neutral'}
    pDM,nDM,tr_list=[],[],[]
    for i in range(1,n):
        hi,lo,cl=float(highs[i]),float(lows[i]),float(closes[i])
        phi,plo,pcl=float(highs[i-1]),float(lows[i-1]),float(closes[i-1])
        up_move=hi-phi; down_move=plo-lo
        pDM.append(up_move if up_move>down_move and up_move>0 else 0)
        nDM.append(down_move if down_move>up_move and down_move>0 else 0)
        tr_list.append(max(hi-lo,abs(hi-pcl),abs(lo-pcl)))
    if len(pDM)<period: return {'name':'DMI','diPlus':0,'diMinus':0,'adx':0,'signal':'neutral'}
    atr=np.mean(tr_list[-period:]); s_pDM=np.mean(pDM[-period:]); s_nDM=np.mean(nDM[-period:])
    diP=sf((s_pDM/atr)*100 if atr>0 else 0); diM=sf((s_nDM/atr)*100 if atr>0 else 0)
    dx=abs(diP-diM)/(diP+diM)*100 if (diP+diM)>0 else 0
    sig='buy' if diP>diM and dx>20 else ('sell' if diM>diP and dx>20 else 'neutral')
    return {'name':'DMI','diPlus':diP,'diMinus':diM,'adx':sf(dx),'signal':sig}

def calc_recommendation(hist, indicators):
    """Haftalik/Aylik/Yillik al-sat onerisi - guclendirilmis analiz + detayli reason"""
    try:
        c=hist['Close'].values.astype(float)
        h=hist['High'].values.astype(float)
        l=hist['Low'].values.astype(float)
        v=hist['Volume'].values.astype(float)
        o=hist['Open'].values.astype(float) if 'Open' in hist.columns else c.copy()
        # NaN temizligi: Close ile doldur
        h=np.where(np.isnan(h), c, h)
        l=np.where(np.isnan(l), c, l)
        v=np.where(np.isnan(v), 0, v)
        o=np.where(np.isnan(o), c, o)
        n=len(c)
        cur=float(c[-1])
        recommendations={}

        # Destek/direnc hesapla (tum periyotlar icin ortak)
        try:
            sr = calc_support_resistance(hist)
        except:
            sr = {'supports': [], 'resistances': [], 'current': 0}
        supports = sr.get('supports', [])
        resistances = sr.get('resistances', [])
        try:
            fib = calc_fibonacci(hist)
        except:
            fib = {'levels': {}}
        fib_sup = fib.get('nearestSupport')
        fib_res = fib.get('nearestResistance')

        # Bollinger bantlari
        try:
            bb = calc_bollinger(c, cur)
        except:
            bb = {'upper': 0, 'lower': 0, 'middle': 0}
        bb_upper = bb.get('upper', 0)
        bb_lower = bb.get('lower', 0)
        bb_middle = bb.get('middle', 0)

        # Dinamik esikler
        try:
            dyn = calc_dynamic_thresholds(c, h, l, v) if n >= 60 else {'rsi_oversold': 30, 'rsi_overbought': 70}
        except:
            dyn = {'rsi_oversold': 30, 'rsi_overbought': 70}
        dyn_oversold = float(dyn.get('rsi_oversold', 30))
        dyn_overbought = float(dyn.get('rsi_overbought', 70))

        # Mum formasyonlari
        try:
            candle_data = calc_candlestick_patterns(o, h, l, c) if n >= 5 else {'patterns': [], 'signal': 'neutral'}
        except:
            candle_data = {'patterns': [], 'signal': 'neutral'}

        # Piyasa rejimi
        try:
            regime = calc_market_regime()
        except:
            regime = {'regime': 'unknown', 'description': ''}
        regime_type = regime.get('regime', 'unknown')

        for label, days in [('weekly',5),('monthly',22),('yearly',252)]:
            if n<days+14: recommendations[label]={'action':'neutral','confidence':0,'reasons':[],'score':0,'strategy':'Yeterli veri yok','reason':'Yeterli veri yok','indicatorBreakdown':{}}; continue

            sl=slice(-days,None)
            sc=c[sl]; sh=h[sl]; slow=l[sl]; sv=v[sl]

            score=0; reasons=[]; strategy_parts=[]
            buy_indicators = 0; sell_indicators = 0; total_indicators = 0

            # 1. Trend (SMA) - Agirlik: 2 puan
            sma20=np.mean(c[-20:]) if n>=20 else c[-1]
            sma50=np.mean(c[-50:]) if n>=50 else sma20
            sma200=np.mean(c[-200:]) if n>=200 else sma50
            total_indicators += 1
            if cur>sma20:
                score+=1; buy_indicators+=1
                reasons.append(f'Fiyat ({sf(cur)}) SMA20 ({sf(sma20)}) uzerinde')
            else:
                score-=1; sell_indicators+=1
                reasons.append(f'Fiyat ({sf(cur)}) SMA20 ({sf(sma20)}) altinda')

            total_indicators += 1
            if sma20>sma50:
                score+=1; buy_indicators+=1
                reasons.append(f'SMA20 ({sf(sma20)}) > SMA50 ({sf(sma50)}) → Yukari trend')
            else:
                score-=1; sell_indicators+=1
                reasons.append(f'SMA20 ({sf(sma20)}) < SMA50 ({sf(sma50)}) → Asagi trend')

            # SMA200 bonus (uzun vadeli trend)
            if n >= 200:
                total_indicators += 1
                if cur > sma200:
                    score += 0.5; buy_indicators += 1
                    reasons.append(f'Fiyat SMA200 ({sf(sma200)}) uzerinde → Uzun vadeli boga')
                else:
                    score -= 0.5; sell_indicators += 1
                    reasons.append(f'Fiyat SMA200 ({sf(sma200)}) altinda → Uzun vadeli ayi')

            # 2. RSI (Dinamik esikler ile)
            rsi_val=calc_rsi(c)
            rsi_v = rsi_val.get('value', 50)
            total_indicators += 1
            if rsi_v < dyn_oversold:
                score+=2; buy_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Asiri satim bolgesi (<{sf(dyn_oversold)}) → Guclu alis firsati')
            elif rsi_v<40:
                score+=1; buy_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Zayif bolge → Toparlanma bekleniyor')
            elif rsi_v > dyn_overbought:
                score-=2; sell_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Asiri alim bolgesi (>{sf(dyn_overbought)}) → Kar realizasyonu bekleniyor')
            elif rsi_v>60:
                score-=0.5; sell_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Guclu bolge (60-70)')
            elif rsi_v>=50:
                score+=0.5; buy_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Notr-pozitif')
            else:
                score-=0.5; sell_indicators+=1
                reasons.append(f'RSI={sf(rsi_v)}: Notr-negatif')

            # 3. MACD
            macd=calc_macd(c)
            macd_type = macd.get('signalType', 'neutral')
            macd_hist = macd.get('histogram', 0)
            total_indicators += 1
            if macd_type=='buy':
                score+=1.5; buy_indicators+=1
                reasons.append(f'MACD alis sinyali (histogram: {macd_hist})')
            elif macd_type=='sell':
                score-=1.5; sell_indicators+=1
                reasons.append(f'MACD satis sinyali (histogram: {macd_hist})')

            # 4. Bollinger
            total_indicators += 1
            if bb_lower > 0 and cur < bb_lower:
                score+=1; buy_indicators+=1
                reasons.append(f'Fiyat ({sf(cur)}) alt Bollinger bandinin ({sf(bb_lower)}) altinda → Toparlanma bekleniyor')
            elif bb_upper > 0 and cur > bb_upper:
                score-=1; sell_indicators+=1
                reasons.append(f'Fiyat ({sf(cur)}) ust Bollinger bandinin ({sf(bb_upper)}) uzerinde → Geri cekilme bekleniyor')
            elif bb_middle > 0:
                if cur > bb_middle:
                    buy_indicators+=1
                    reasons.append(f'Fiyat Bollinger orta bant ({sf(bb_middle)}) uzerinde')
                else:
                    sell_indicators+=1
                    reasons.append(f'Fiyat Bollinger orta bant ({sf(bb_middle)}) altinda')

            # 5. Hacim trendi
            if len(sv)>5:
                vol_avg=np.mean(sv[-20:]) if len(sv)>=20 else np.mean(sv)
                vol_recent=np.mean(sv[-5:])
                vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 1
                total_indicators += 1
                if vol_ratio > 1.5:
                    if c[-1]>c[-5]:
                        score+=1; buy_indicators+=1
                        reasons.append(f'Hacim ortalamanin {sf(vol_ratio)}x uzerinde + yukari hareket → Guclu alis')
                    else:
                        score-=1; sell_indicators+=1
                        reasons.append(f'Hacim ortalamanin {sf(vol_ratio)}x uzerinde + dusus → Guclu satis baskisi')
                elif vol_ratio < 0.5:
                    reasons.append(f'Hacim ortalamanin altinda ({sf(vol_ratio)}x) → Dusuk ilgi, sinyal gucsuslesiyor')

            # 6. Momentum (periyoda gore)
            if len(sc)>=days:
                period_return=sf(((c[-1]-sc[0])/sc[0])*100)
                total_indicators += 1
                if period_return>10: score+=1.5; buy_indicators+=1; reasons.append(f'{label} getiri: %{period_return} (guclu yukselis)')
                elif period_return>5: score+=1; buy_indicators+=1; reasons.append(f'{label} getiri: %{period_return} (pozitif)')
                elif period_return<-10: score-=1.5; sell_indicators+=1; reasons.append(f'{label} getiri: %{period_return} (sert dusus)')
                elif period_return<-5: score-=1; sell_indicators+=1; reasons.append(f'{label} getiri: %{period_return} (negatif)')

            # 7. Stochastic
            stoch=calc_stochastic(c,h,l)
            stoch_k = stoch.get('k', 50)
            total_indicators += 1
            if stoch_k<20: score+=1; buy_indicators+=1; reasons.append(f'Stochastic K={sf(stoch_k)}: Asiri satim bolgesi')
            elif stoch_k>80: score-=1; sell_indicators+=1; reasons.append(f'Stochastic K={sf(stoch_k)}: Asiri alim bolgesi')

            # 8. ADX - Trend gucu
            adx_data = calc_adx(h, l, c)
            adx_val = adx_data.get('value', 25)
            total_indicators += 1
            if adx_val > 25:
                trend_dir = 'yukari' if adx_data.get('plusDI', 0) > adx_data.get('minusDI', 0) else 'asagi'
                reasons.append(f'ADX={sf(adx_val)}: Guclu {trend_dir} trend')
                if trend_dir == 'yukari': score += 0.5; buy_indicators += 1
                else: score -= 0.5; sell_indicators += 1
            else:
                reasons.append(f'ADX={sf(adx_val)}: Zayif trend (<25), yatay piyasa')

            # 9. Ichimoku (eger yeterli veri varsa)
            if n >= 52:
                ichi = calc_ichimoku(c, h, l)
                total_indicators += 1
                if ichi.get('signal') == 'buy':
                    score += 1; buy_indicators += 1
                    reasons.append(f'Ichimoku alis sinyali (fiyat bulutun uzerinde)')
                elif ichi.get('signal') == 'sell':
                    score -= 1; sell_indicators += 1
                    reasons.append(f'Ichimoku satis sinyali (fiyat bulutun altinda)')

            # 10. Parabolic SAR
            if n >= 5:
                psar = calc_psar(c, h, l)
                total_indicators += 1
                if psar.get('signal') == 'buy':
                    score += 0.5; buy_indicators += 1
                    reasons.append(f'Parabolic SAR yukari trend (SAR={sf(psar.get("value", 0))})')
                elif psar.get('signal') == 'sell':
                    score -= 0.5; sell_indicators += 1
                    reasons.append(f'Parabolic SAR asagi trend (SAR={sf(psar.get("value", 0))})')

            # 11. Mum formasyonlari
            for p in candle_data.get('patterns', []):
                if p.get('strength', 0) >= 3:
                    total_indicators += 1
                    if p['type'] == 'bullish':
                        score += 0.5 * (p['strength'] / 5)
                        buy_indicators += 1
                        reasons.append(f'Mum: {p["name"]} → {p["description"][:60]}')
                    elif p['type'] == 'bearish':
                        score -= 0.5 * (p['strength'] / 5)
                        sell_indicators += 1
                        reasons.append(f'Mum: {p["name"]} → {p["description"][:60]}')

            # 12. Piyasa rejimi etkisi
            if regime_type in ('strong_bull', 'bull') and score > 0:
                score *= 1.15
                reasons.append(f'Piyasa rejimi: {regime.get("description", "")} → Alis sinyali gucleniyor')
            elif regime_type in ('strong_bear', 'bear') and score < 0:
                score *= 1.15
                reasons.append(f'Piyasa rejimi: {regime.get("description", "")} → Satis sinyali gucleniyor')
            elif regime_type in ('strong_bear', 'bear') and score > 0:
                score *= 0.85
                reasons.append(f'Piyasa rejimi: {regime.get("description", "")} → Alis sinyali zayifliyor (ayi piyasasi)')

            # 13. Destek/Direnc bazli yorumlar
            if supports:
                nearest_sup = supports[0]
                sup_dist = sf(((cur - nearest_sup) / nearest_sup) * 100)
                if float(sup_dist) < 2:
                    score += 1
                    reasons.append(f'Fiyat destege ({sf(nearest_sup)}) cok yakin (%{sup_dist}) → Destek bolgesi')
                    strategy_parts.append(f'{sf(nearest_sup)} TL desteginden alis yapilabilir')
                elif float(sup_dist) < 5:
                    strategy_parts.append(f'{sf(nearest_sup)} TL destegine yaklasirsa alis firsati')

            if resistances:
                nearest_res = resistances[0]
                res_dist = sf(((nearest_res - cur) / cur) * 100)
                if float(res_dist) < 2:
                    score -= 1
                    reasons.append(f'Fiyat dirence ({sf(nearest_res)}) cok yakin (%{res_dist}) → Satis baskisi')
                    strategy_parts.append(f'{sf(nearest_res)} TL direncinde satis/kar realizasyonu')
                elif float(res_dist) < 5:
                    strategy_parts.append(f'{sf(nearest_res)} TL direncini kirarsa alis guclenir')

            # 14. Fibonacci bazli yorumlar
            if fib_sup and fib_sup.get('price'):
                fib_sup_dist = sf(((cur - fib_sup['price']) / cur) * 100)
                if float(fib_sup_dist) < 3:
                    strategy_parts.append(f"Fibonacci {fib_sup['level']} destegi ({fib_sup['price']} TL) yakininda")
            if fib_res and fib_res.get('price'):
                fib_res_dist = sf(((fib_res['price'] - cur) / cur) * 100)
                if float(fib_res_dist) < 3:
                    strategy_parts.append(f"Fibonacci {fib_res['level']} direnci ({fib_res['price']} TL) yakininda")

            # Sonuc
            max_score=14.0
            conf=min(abs(score)/max_score*100, 100)
            if score>=3: action='AL'
            elif score>=1.5: action='TUTUN/AL'
            elif score<=-3: action='SAT'
            elif score<=-1.5: action='TUTUN/SAT'
            else: action='NOTR'

            # KISA OZET REASON (tek satirlik aciklama)
            if action == 'AL':
                top_buy = [r for r in reasons if any(k in r.lower() for k in ['alis','yukari','pozitif','topar','destek','asiri satim','uzerinde'])][:3]
                reason_summary = f"AL: {buy_indicators}/{total_indicators} gosterge alis yonunde. " + (top_buy[0] if top_buy else reasons[0] if reasons else '')
            elif action == 'SAT':
                top_sell = [r for r in reasons if any(k in r.lower() for k in ['satis','asagi','negatif','dusus','direnc','asiri alim','altinda'])][:3]
                reason_summary = f"SAT: {sell_indicators}/{total_indicators} gosterge satis yonunde. " + (top_sell[0] if top_sell else reasons[0] if reasons else '')
            elif action == 'TUTUN/AL':
                reason_summary = f"ZAYIF ALIS: {buy_indicators}/{total_indicators} gosterge alis yonunde. Destek bolgesi bekleniyor."
            elif action == 'TUTUN/SAT':
                reason_summary = f"ZAYIF SATIS: {sell_indicators}/{total_indicators} gosterge satis yonunde. Direnc bolgesi bekleniyor."
            else:
                reason_summary = f"NOTR: {buy_indicators} alis vs {sell_indicators} satis sinyali. Belirgin yon yok."

            # Strateji olustur
            if not strategy_parts:
                if action == 'AL':
                    strategy_parts.append('Teknik gostergeler alis yonunde')
                    if supports: strategy_parts.append(f'Stop-loss: {sf(supports[0])} TL altinda')
                    if resistances: strategy_parts.append(f'Hedef: {sf(resistances[0])} TL')
                elif action == 'SAT':
                    strategy_parts.append('Teknik gostergeler satis yonunde')
                    if resistances: strategy_parts.append(f'{sf(resistances[0])} TL direnci asagi sari')
                    if supports: strategy_parts.append(f'{sf(supports[0])} TL destegi kirilirsa satis guclenir')
                else:
                    strategy_parts.append('Belirgin bir sinyal yok, bekle-gor stratejisi')

            recommendations[label]={
                'action':action,'score':sf(score),'confidence':sf(conf),
                'reasons':reasons[:10],
                'reason': reason_summary,
                'strategy': ' | '.join(strategy_parts[:4]),
                'indicatorBreakdown': {
                    'buy': buy_indicators,
                    'sell': sell_indicators,
                    'total': total_indicators,
                    'consensus': sf(buy_indicators / total_indicators * 100) if total_indicators > 0 else 50,
                },
                'keyLevels': {
                    'supports': supports[:3],
                    'resistances': resistances[:3],
                    'sma20': sf(sma20),
                    'sma50': sf(sma50),
                    'sma200': sf(sma200) if n >= 200 else None,
                    'bollingerUpper': sf(bb_upper),
                    'bollingerLower': sf(bb_lower),
                },
                'dynamicRSI': {'oversold': sf(dyn_oversold), 'overbought': sf(dyn_overbought), 'current': sf(rsi_v)},
            }

        return recommendations
    except Exception as e:
        print(f"  [REC] Hata: {e}")
        return {'weekly':{'action':'neutral','confidence':0,'reasons':[],'strategy':'','reason':'Hesaplama hatasi','indicatorBreakdown':{}},'monthly':{'action':'neutral','confidence':0,'reasons':[],'strategy':'','reason':'Hesaplama hatasi','indicatorBreakdown':{}},'yearly':{'action':'neutral','confidence':0,'reasons':[],'strategy':'','reason':'Hesaplama hatasi','indicatorBreakdown':{}}}

def calc_fundamentals(hist, symbol):
    """Temel verileri mevcut fiyat/hacim verisinden hesapla"""
    try:
        c = hist['Close'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        # NaN temizligi
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        v = np.where(np.isnan(v), 0, v)
        n = len(c)
        cur = float(c[-1])

        # Ortalama gunluk hacim (son 20 gun)
        avg_vol_20 = float(np.mean(v[-20:])) if n >= 20 else float(np.mean(v))
        avg_vol_60 = float(np.mean(v[-60:])) if n >= 60 else avg_vol_20

        # Volatilite (yillik)
        if n >= 20:
            daily_returns = np.diff(c[-60:]) / c[-60:-1] if n >= 60 else np.diff(c) / c[:-1]
            volatility = sf(float(np.std(daily_returns)) * (252 ** 0.5) * 100)
        else:
            volatility = 0

        # Beta hesapla (BIST100'e gore) - basit yaklasim: volatilite bazli
        beta = sf(volatility / 25) if volatility else 1.0  # BIST100 avg vol ~25%

        # Ortalama islem hacmi (TL)
        avg_turnover = sf(cur * avg_vol_20)

        # 1 aylik, 3 aylik, 6 aylik, 1 yillik getiri
        returns = {}
        for label, days in [('1ay', 22), ('3ay', 66), ('6ay', 132), ('1yil', 252)]:
            if n > days:
                ret = sf(((cur - float(c[-days])) / float(c[-days])) * 100)
                returns[label] = ret

        # Gunluk ortalama aralik (ATR benzeri) - NaN-safe
        if n >= 14:
            daily_range = [(float(h[i]) - float(l[i])) for i in range(-14, 0) if h[i] == h[i] and l[i] == l[i]]
            avg_daily_range = sf(np.mean(daily_range)) if daily_range else 0
            avg_daily_range_pct = sf(avg_daily_range / cur * 100) if cur > 0 else 0
        else:
            avg_daily_range = 0
            avg_daily_range_pct = 0

        # 52 haftalik high/low'dan uzaklik (NaN-safe)
        if n >= 252:
            hi52 = float(np.nanmax(h[-252:]))
            lo52 = float(np.nanmin(l[-252:]))
        else:
            hi52 = float(np.nanmax(h))
            lo52 = float(np.nanmin(l))
        # NaN kontrolu
        if hi52 != hi52: hi52 = cur  # NaN ise cur kullan
        if lo52 != lo52: lo52 = cur
        dist_from_high = sf(((cur - hi52) / hi52) * 100) if hi52 else 0
        dist_from_low = sf(((cur - lo52) / lo52) * 100) if lo52 else 0

        return {
            'avgVolume20': si(avg_vol_20),
            'avgVolume60': si(avg_vol_60),
            'avgTurnover': avg_turnover,
            'volatility': volatility,
            'beta': beta,
            'returns': returns,
            'avgDailyRange': avg_daily_range,
            'avgDailyRangePct': avg_daily_range_pct,
            'distFromHigh52w': dist_from_high,
            'distFromLow52w': dist_from_low,
        }
    except Exception as e:
        print(f"  [FUND] {symbol} hata: {e}")
        return {}

def calc_52w(hist):
    """52 hafta (veya mevcut veri) high/low hesapla - NaN-safe"""
    try:
        h=hist['High'].values.astype(float)
        l=hist['Low'].values.astype(float)
        c=float(hist['Close'].iloc[-1])
        hi52=sf(float(np.nanmax(h))); lo52=sf(float(np.nanmin(l)))
        # NaN fallback
        if hi52 == 0 and c > 0: hi52 = sf(c)
        if lo52 == 0 and c > 0: lo52 = sf(c)
        rng=hi52-lo52
        pos=sf((c-lo52)/rng*100 if rng>0 else 50)
        return {'high52w':hi52,'low52w':lo52,'currentPct':pos,'range':sf(rng)}
    except: return {'high52w':0,'low52w':0,'currentPct':50,'range':0}

def calc_all_indicators(hist, cp):
    c,h,l,v=hist['Close'].values.astype(float),hist['High'].values.astype(float),hist['Low'].values.astype(float),hist['Volume'].values.astype(float)
    o=hist['Open'].values.astype(float) if 'Open' in hist.columns else c.copy()
    # NaN temizligi
    h=np.where(np.isnan(h), c, h); l=np.where(np.isnan(l), c, l)
    v=np.where(np.isnan(v), 0, v); o=np.where(np.isnan(o), c, o)
    cp=float(cp)
    rsi_h=[{'date':hist.index[i].strftime('%Y-%m-%d'),'value':rv} for i in range(14,len(c)) if (rv:=calc_rsi_single(c[:i+1])) is not None]

    # Dinamik esikler hesapla
    dyn_thresholds = calc_dynamic_thresholds(c, h, l, v)

    # RSI'yi dinamik esiklerle hesapla
    rsi_data = calc_rsi(c)
    rsi_val = rsi_data.get('value', 50)
    dyn_oversold = float(dyn_thresholds.get('rsi_oversold', 30))
    dyn_overbought = float(dyn_thresholds.get('rsi_overbought', 70))
    if rsi_val < dyn_oversold:
        rsi_data['signal'] = 'buy'
        rsi_data['dynamicNote'] = f'Dinamik esik: <{dyn_oversold}'
    elif rsi_val > dyn_overbought:
        rsi_data['signal'] = 'sell'
        rsi_data['dynamicNote'] = f'Dinamik esik: >{dyn_overbought}'
    rsi_data['dynamicOversold'] = dyn_oversold
    rsi_data['dynamicOverbought'] = dyn_overbought

    ind={
        'rsi':rsi_data,'rsiHistory':rsi_h,
        'macd':calc_macd(c),'macdHistory':calc_macd_history(c),
        'bollinger':calc_bollinger(c,cp),'bollingerHistory':calc_bollinger_history(c),
        'stochastic':calc_stochastic(c,h,l),'stochasticHistory':calc_stochastic_history(c,h,l),
        'ema':calc_ema(c,cp),'emaHistory':calc_ema_history(c),
        'atr':calc_atr(h,l,c),
        'adx':calc_adx(h,l,c),
        'obv':calc_obv(c,v),
        'williamsR':calc_williams_r(c,h,l),
        'cci':calc_cci(c,h,l),
        'mfi':calc_mfi(c,h,l,v),
        'vwap':calc_vwap(c,h,l,v),
        'ichimoku':calc_ichimoku(c,h,l),
        'psar':calc_psar(c,h,l),
        'roc':calc_roc(c),
        'aroon':calc_aroon(h,l),
        'trix':calc_trix(c),
        'dmi':calc_dmi(h,l,c),
        'candlestick':calc_candlestick_patterns(o,h,l,c),
        'dynamicThresholds':dyn_thresholds,
    }
    sigs=[x.get('signal','neutral') for x in ind.values() if isinstance(x,dict) and 'signal' in x]
    bc,sc=sigs.count('buy'),sigs.count('sell'); t=len(sigs)
    ind['summary']={'overall':'buy' if bc>sc and bc>=t*0.4 else ('sell' if sc>bc and sc>=t*0.4 else 'neutral'),'buySignals':bc,'sellSignals':sc,'neutralSignals':t-bc-sc,'totalIndicators':t}
    return ind


# =====================================================================
# FEATURE 1: SIGNAL BACKTESTING & PERFORMANCE TRACKING
# =====================================================================
def calc_signal_backtest(hist, lookback_days=252):
    """Gecmis sinyallerin performansini olc: her sinyal sonrasi 5/10/20 gun getiri"""
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        # NaN temizligi
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        v = np.where(np.isnan(v), 0, v)
        n = len(c)
        if n < 60:
            return {'totalSignals': 0, 'message': 'Yeterli veri yok'}

        signals = []
        # Her gunde geriye donuk indikatoru hesapla ve sinyal uret
        for i in range(50, n - 20):  # 20 gun sonrasini gormek icin -20
            sc = c[:i+1]
            sh = h[:i+1]
            sl = l[:i+1]
            sv = v[:i+1]

            rsi = calc_rsi(sc)
            macd = calc_macd(sc)

            signal_type = None
            signal_reason = ''

            # RSI asiri satim
            if rsi.get('value', 50) < 30:
                signal_type = 'buy'
                signal_reason = 'RSI < 30'
            elif rsi.get('value', 50) > 70:
                signal_type = 'sell'
                signal_reason = 'RSI > 70'
            # MACD kesisim
            elif macd.get('signalType') == 'buy' and len(sc) > 26:
                prev_macd = calc_macd(sc[:-1])
                if prev_macd.get('signalType') != 'buy':
                    signal_type = 'buy'
                    signal_reason = 'MACD Kesisim'
            elif macd.get('signalType') == 'sell' and len(sc) > 26:
                prev_macd = calc_macd(sc[:-1])
                if prev_macd.get('signalType') != 'sell':
                    signal_type = 'sell'
                    signal_reason = 'MACD Kesisim'
            # Bollinger alt bant
            elif len(sc) >= 20:
                bb = calc_bollinger(sc, float(sc[-1]))
                if bb.get('lower', 0) > 0 and float(sc[-1]) < bb['lower']:
                    signal_type = 'buy'
                    signal_reason = 'Bollinger Alt Bant'
                elif bb.get('upper', 0) > 0 and float(sc[-1]) > bb['upper']:
                    signal_type = 'sell'
                    signal_reason = 'Bollinger Ust Bant'

            if signal_type:
                entry_price = float(c[i])
                ret_5d = sf(((float(c[min(i+5, n-1)]) - entry_price) / entry_price) * 100)
                ret_10d = sf(((float(c[min(i+10, n-1)]) - entry_price) / entry_price) * 100)
                ret_20d = sf(((float(c[min(i+20, n-1)]) - entry_price) / entry_price) * 100)

                # Satis sinyali icin getiriyi ters cevir
                if signal_type == 'sell':
                    ret_5d, ret_10d, ret_20d = -ret_5d, -ret_10d, -ret_20d

                signals.append({
                    'day': i,
                    'type': signal_type,
                    'reason': signal_reason,
                    'price': sf(entry_price),
                    'ret5d': float(ret_5d),
                    'ret10d': float(ret_10d),
                    'ret20d': float(ret_20d),
                    'win5d': float(ret_5d) > 0,
                    'win10d': float(ret_10d) > 0,
                    'win20d': float(ret_20d) > 0,
                })

        if not signals:
            return {'totalSignals': 0, 'message': 'Sinyal bulunamadi'}

        buy_signals = [s for s in signals if s['type'] == 'buy']
        sell_signals = [s for s in signals if s['type'] == 'sell']

        def calc_stats(sigs):
            if not sigs:
                return {'count': 0, 'winRate5d': 0, 'winRate10d': 0, 'winRate20d': 0, 'avgRet5d': 0, 'avgRet10d': 0, 'avgRet20d': 0}
            return {
                'count': len(sigs),
                'winRate5d': sf(sum(1 for s in sigs if s['win5d']) / len(sigs) * 100),
                'winRate10d': sf(sum(1 for s in sigs if s['win10d']) / len(sigs) * 100),
                'winRate20d': sf(sum(1 for s in sigs if s['win20d']) / len(sigs) * 100),
                'avgRet5d': sf(np.mean([s['ret5d'] for s in sigs])),
                'avgRet10d': sf(np.mean([s['ret10d'] for s in sigs])),
                'avgRet20d': sf(np.mean([s['ret20d'] for s in sigs])),
            }

        # Sinyal tiplerine gore istatistik
        by_reason = {}
        for s in signals:
            r = s['reason']
            if r not in by_reason:
                by_reason[r] = []
            by_reason[r].append(s)

        reason_stats = {}
        for reason, sigs in by_reason.items():
            reason_stats[reason] = calc_stats(sigs)

        return {
            'totalSignals': len(signals),
            'buySignals': calc_stats(buy_signals),
            'sellSignals': calc_stats(sell_signals),
            'overall': calc_stats(signals),
            'byReason': reason_stats,
            'recentSignals': signals[-10:],
        }
    except Exception as e:
        print(f"  [BACKTEST] Hata: {e}")
        return {'totalSignals': 0, 'error': str(e)}


# =====================================================================
# FEATURE 2: DYNAMIC THRESHOLDS (Hisse bazli adaptif esikler)
# =====================================================================
def calc_dynamic_thresholds(closes, highs, lows, volumes):
    """Her hisse icin tarihsel dagılıma gore adaptif RSI/BB/Volume esikleri"""
    try:
        n = len(closes)
        if n < 60:
            return {'rsi_oversold': 30, 'rsi_overbought': 70, 'vol_spike': 2.0, 'bb_std': 2.0}

        # Tum RSI degerlerini hesapla
        rsi_values = []
        for i in range(20, n):
            rv = calc_rsi_single(closes[:i+1])
            if rv is not None:
                rsi_values.append(rv)

        if len(rsi_values) < 20:
            return {'rsi_oversold': 30, 'rsi_overbought': 70, 'vol_spike': 2.0, 'bb_std': 2.0}

        rsi_arr = np.array(rsi_values)
        # Percentile bazli esikler: %10 ve %90
        rsi_oversold = sf(np.percentile(rsi_arr, 10))
        rsi_overbought = sf(np.percentile(rsi_arr, 90))

        # Minimum/maximum sinirlari
        rsi_oversold = max(20, min(40, float(rsi_oversold)))
        rsi_overbought = max(60, min(85, float(rsi_overbought)))

        # Volatilite bazli Bollinger genisligi
        if n >= 60:
            daily_returns = np.diff(closes[-60:]) / closes[-60:-1]
            vol = float(np.std(daily_returns))
            # Dusuk volatilite -> daha dar bantlar, yuksek -> daha genis
            bb_std = max(1.5, min(3.0, 2.0 * (vol / 0.02)))
        else:
            bb_std = 2.0

        # Hacim spike esigi: medyan bazli
        if n >= 30:
            vol_mean = float(np.mean(volumes[-30:]))
            vol_std = float(np.std(volumes[-30:]))
            vol_spike = max(1.5, min(3.0, (vol_mean + vol_std) / vol_mean if vol_mean > 0 else 2.0))
        else:
            vol_spike = 2.0

        return {
            'rsi_oversold': sf(rsi_oversold),
            'rsi_overbought': sf(rsi_overbought),
            'vol_spike': sf(vol_spike),
            'bb_std': sf(bb_std),
        }
    except:
        return {'rsi_oversold': 30, 'rsi_overbought': 70, 'vol_spike': 2.0, 'bb_std': 2.0}


# =====================================================================
# FEATURE 3: CANDLESTICK PATTERN RECOGNITION
# =====================================================================
def calc_candlestick_patterns(opens, highs, lows, closes):
    """Mum formasyonlarini tespit et"""
    try:
        n = len(closes)
        if n < 5:
            return {'patterns': [], 'signal': 'neutral'}

        patterns = []
        o, h, l, c = [float(x) for x in opens[-5:]], [float(x) for x in highs[-5:]], [float(x) for x in lows[-5:]], [float(x) for x in closes[-5:]]
        body = [c[i] - o[i] for i in range(5)]
        body_abs = [abs(b) for b in body]
        upper_shadow = [h[i] - max(o[i], c[i]) for i in range(5)]
        lower_shadow = [min(o[i], c[i]) - l[i] for i in range(5)]
        total_range = [h[i] - l[i] for i in range(5)]

        # Son mum (index 4)
        i = 4
        avg_body = np.mean(body_abs[:4]) if np.mean(body_abs[:4]) > 0 else 0.01

        # 1. Doji - Gövde çok küçük
        if body_abs[i] < total_range[i] * 0.1 and total_range[i] > 0:
            patterns.append({'name': 'Doji', 'type': 'neutral', 'description': 'Kararsizlik formasyonu. Trend donusu habercisi olabilir.', 'strength': 2})

        # 2. Hammer (Cekic) - Uzun alt golge, kucuk govde, ust kisminda
        if (lower_shadow[i] > body_abs[i] * 2 and upper_shadow[i] < body_abs[i] * 0.5 and
            body[i-1] < 0):  # Onceki mum dusus
            patterns.append({'name': 'Cekic (Hammer)', 'type': 'bullish', 'description': 'Dusus sonrasi toparlanma sinyali. Guclu alis formasyonu.', 'strength': 3})

        # 3. Shooting Star (Kayan Yildiz) - Uzun ust golge, kucuk govde, alt kisminda
        if (upper_shadow[i] > body_abs[i] * 2 and lower_shadow[i] < body_abs[i] * 0.5 and
            body[i-1] > 0):  # Onceki mum yukselis
            patterns.append({'name': 'Kayan Yildiz (Shooting Star)', 'type': 'bearish', 'description': 'Yukselis sonrasi satis baskisi. Dusus sinyali.', 'strength': 3})

        # 4. Bullish Engulfing (Yukari Yutan)
        if (body[i] > 0 and body[i-1] < 0 and
            o[i] <= c[i-1] and c[i] >= o[i-1] and
            body_abs[i] > body_abs[i-1]):
            patterns.append({'name': 'Yukari Yutan (Bullish Engulfing)', 'type': 'bullish', 'description': 'Guclu alis formasyonu. Alicilar kontrolu ele aldi.', 'strength': 4})

        # 5. Bearish Engulfing (Asagi Yutan)
        if (body[i] < 0 and body[i-1] > 0 and
            o[i] >= c[i-1] and c[i] <= o[i-1] and
            body_abs[i] > body_abs[i-1]):
            patterns.append({'name': 'Asagi Yutan (Bearish Engulfing)', 'type': 'bearish', 'description': 'Guclu satis formasyonu. Saticilar kontrolu ele aldi.', 'strength': 4})

        # 6. Morning Star (Sabah Yildizi) - 3 mumlu boğa formasyonu
        if (n >= 3 and body[i-2] < 0 and body_abs[i-2] > avg_body and
            body_abs[i-1] < avg_body * 0.5 and
            body[i] > 0 and body_abs[i] > avg_body):
            patterns.append({'name': 'Sabah Yildizi (Morning Star)', 'type': 'bullish', 'description': 'Guclu 3 mumlu dip formasyonu. Trendin donusu bekleniyor.', 'strength': 5})

        # 7. Evening Star (Aksam Yildizi) - 3 mumlu ayı formasyonu
        if (n >= 3 and body[i-2] > 0 and body_abs[i-2] > avg_body and
            body_abs[i-1] < avg_body * 0.5 and
            body[i] < 0 and body_abs[i] > avg_body):
            patterns.append({'name': 'Aksam Yildizi (Evening Star)', 'type': 'bearish', 'description': 'Guclu 3 mumlu tepe formasyonu. Dusus bekleniyor.', 'strength': 5})

        # 8. Three White Soldiers (Uc Beyaz Asker)
        if (body[i] > 0 and body[i-1] > 0 and body[i-2] > 0 and
            c[i] > c[i-1] > c[i-2] and
            body_abs[i] > avg_body * 0.5 and body_abs[i-1] > avg_body * 0.5):
            patterns.append({'name': 'Uc Beyaz Asker', 'type': 'bullish', 'description': 'Art arda 3 guclu yukselis mumu. Guclu alis trendi.', 'strength': 4})

        # 9. Three Black Crows (Uc Kara Karga)
        if (body[i] < 0 and body[i-1] < 0 and body[i-2] < 0 and
            c[i] < c[i-1] < c[i-2] and
            body_abs[i] > avg_body * 0.5 and body_abs[i-1] > avg_body * 0.5):
            patterns.append({'name': 'Uc Kara Karga', 'type': 'bearish', 'description': 'Art arda 3 guclu dusus mumu. Guclu satis baskisi.', 'strength': 4})

        # 10. Marubozu (gölgesiz güçlü mum)
        if total_range[i] > 0:
            shadow_ratio = (upper_shadow[i] + lower_shadow[i]) / total_range[i]
            if shadow_ratio < 0.1 and body_abs[i] > avg_body * 1.5:
                mtype = 'bullish' if body[i] > 0 else 'bearish'
                patterns.append({'name': f'Marubozu ({"Yukari" if mtype == "bullish" else "Asagi"})', 'type': mtype,
                    'description': 'Golgesiz guclu mum. Tek yonlu baski. Trend devami beklenir.', 'strength': 3})

        # Genel sinyal
        bullish = sum(1 for p in patterns if p['type'] == 'bullish')
        bearish = sum(1 for p in patterns if p['type'] == 'bearish')
        signal = 'buy' if bullish > bearish else ('sell' if bearish > bullish else 'neutral')

        return {
            'patterns': patterns,
            'signal': signal,
            'bullishCount': bullish,
            'bearishCount': bearish,
        }
    except Exception as e:
        print(f"  [CANDLE] Hata: {e}")
        return {'patterns': [], 'signal': 'neutral'}


# =====================================================================
# FEATURE 4: MARKET REGIME DETECTION (Piyasa Rejimi)
# =====================================================================
_market_regime_cache = {'regime': None, 'ts': 0}

def calc_market_regime():
    """BIST100 trend durumunu analiz et: bull/bear/sideways"""
    try:
        # Cache kontrol (5 dakika)
        if _market_regime_cache['regime'] and time.time() - _market_regime_cache['ts'] < 300:
            return _market_regime_cache['regime']

        # XU100 tarihsel verisini al
        hist = _cget_hist("XU100_1y")
        if hist is None:
            # Cache'de yoksa senkron olarak cek
            try:
                xu_df = _fetch_isyatirim_df("XU100", 365)
                if xu_df is not None and len(xu_df) >= 30:
                    _cset(_hist_cache, "XU100_1y", xu_df)
                    hist = xu_df
                    print("[REGIME] XU100 verisi senkron olarak cekildi")
            except Exception as xe:
                print(f"[REGIME] XU100 senkron cekme hatasi: {xe}")

        if hist is None:
            # Hala yoksa stock cache'den basit rejim hesapla
            stocks = _get_stocks()
            if stocks:
                advancing = sum(1 for s in stocks if s.get('changePct', 0) > 0)
                declining = sum(1 for s in stocks if s.get('changePct', 0) < 0)
                total = len(stocks)
                ratio = advancing / max(declining, 1)
                if ratio > 2:
                    regime_name, desc = 'strong_bull', 'Guclu Boga Piyasasi'
                elif ratio > 1.3:
                    regime_name, desc = 'bull', 'Boga Piyasasi'
                elif ratio < 0.5:
                    regime_name, desc = 'strong_bear', 'Guclu Ayi Piyasasi'
                elif ratio < 0.8:
                    regime_name, desc = 'bear', 'Ayi Piyasasi'
                else:
                    regime_name, desc = 'sideways', 'Yatay Piyasa'
                return {
                    'regime': regime_name, 'strength': sf(min(abs(ratio - 1) * 50, 100)),
                    'description': desc,
                    'reasons': [f'Yukselen: {advancing}, Dusen: {declining} (toplam {total})',
                                f'A/D orani: {sf(ratio)}'],
                    'indicators': {'breadthRatio': sf(ratio)},
                }
            return {'regime': 'unknown', 'strength': 0, 'description': 'Piyasa verisi mevcut degil'}

        c = hist['Close'].values.astype(float)
        n = len(c)
        if n < 50:
            return {'regime': 'unknown', 'strength': 0, 'description': 'Yeterli veri yok'}

        cur = float(c[-1])

        # SMA hesapla
        sma20 = float(np.mean(c[-20:])) if n >= 20 else cur
        sma50 = float(np.mean(c[-50:])) if n >= 50 else sma20
        sma200 = float(np.mean(c[-200:])) if n >= 200 else sma50

        # EMA hesapla
        s = pd.Series(c)
        ema20 = float(s.ewm(span=20).mean().iloc[-1])
        ema50 = float(s.ewm(span=50).mean().iloc[-1])

        # ADX (trend gucu)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        adx_data = calc_adx(h, l, c)
        adx_val = float(adx_data.get('value', 25))
        plus_di = float(adx_data.get('plusDI', 0))
        minus_di = float(adx_data.get('minusDI', 0))

        # RSI
        rsi = calc_rsi(c).get('value', 50)

        # Son 20 gun momentum
        ret_20d = ((cur - float(c[-20])) / float(c[-20])) * 100 if n >= 20 else 0
        ret_50d = ((cur - float(c[-50])) / float(c[-50])) * 100 if n >= 50 else 0

        # Volatilite
        if n >= 20:
            daily_returns = np.diff(c[-30:]) / c[-30:-1]
            volatility = float(np.std(daily_returns)) * (252 ** 0.5) * 100
        else:
            volatility = 25

        # Piyasa genisligi (cache'deki hisselerden)
        stocks = _get_stocks()
        advancing = sum(1 for s in stocks if s.get('changePct', 0) > 0)
        declining = sum(1 for s in stocks if s.get('changePct', 0) < 0)
        breadth_ratio = advancing / max(declining, 1)

        # Rejim belirleme skoru
        score = 0
        reasons = []

        # Fiyat > SMA pozisyonu
        if cur > sma20: score += 1; reasons.append('Fiyat SMA20 uzerinde')
        else: score -= 1; reasons.append('Fiyat SMA20 altinda')

        if cur > sma50: score += 1; reasons.append('Fiyat SMA50 uzerinde')
        else: score -= 1; reasons.append('Fiyat SMA50 altinda')

        if n >= 200:
            if cur > sma200: score += 1.5; reasons.append('Fiyat SMA200 uzerinde (uzun vadeli boga)')
            else: score -= 1.5; reasons.append('Fiyat SMA200 altinda (uzun vadeli ayi)')

        # SMA siralamasi
        if sma20 > sma50: score += 1; reasons.append('SMA20 > SMA50 (yukari trend)')
        else: score -= 1; reasons.append('SMA20 < SMA50 (asagi trend)')

        # Momentum
        if ret_20d > 5: score += 1; reasons.append(f'20 gunluk getiri: %{sf(ret_20d)} (guclu)')
        elif ret_20d > 0: score += 0.5
        elif ret_20d < -5: score -= 1; reasons.append(f'20 gunluk getiri: %{sf(ret_20d)} (zayif)')
        else: score -= 0.5

        # ADX trend gucu
        if adx_val > 25:
            if plus_di > minus_di: score += 1; reasons.append(f'ADX={sf(adx_val)}: Guclu yukari trend')
            else: score -= 1; reasons.append(f'ADX={sf(adx_val)}: Guclu asagi trend')

        # Piyasa genisligi
        if breadth_ratio > 1.5: score += 0.5; reasons.append(f'Piyasa genisligi pozitif ({advancing}/{declining})')
        elif breadth_ratio < 0.7: score -= 0.5; reasons.append(f'Piyasa genisligi negatif ({advancing}/{declining})')

        # Rejim siniflandirma
        if score >= 3:
            regime = 'strong_bull'
            desc = 'Guclu Boga Piyasasi - Alis sinyalleri daha guvenilir'
        elif score >= 1:
            regime = 'bull'
            desc = 'Boga Piyasasi - Genel yukari trend'
        elif score <= -3:
            regime = 'strong_bear'
            desc = 'Guclu Ayi Piyasasi - Satis sinyalleri daha guvenilir'
        elif score <= -1:
            regime = 'bear'
            desc = 'Ayi Piyasasi - Genel asagi trend'
        else:
            regime = 'sideways'
            desc = 'Yatay Piyasa - Belirsizlik hakim, dikkatli olun'

        # Sinyal guven carpani
        if regime in ('strong_bull', 'bull'):
            buy_confidence_mult = 1.2
            sell_confidence_mult = 0.8
        elif regime in ('strong_bear', 'bear'):
            buy_confidence_mult = 0.8
            sell_confidence_mult = 1.2
        else:
            buy_confidence_mult = 1.0
            sell_confidence_mult = 1.0

        result = {
            'regime': regime,
            'score': sf(score),
            'strength': sf(min(abs(score) / 5 * 100, 100)),
            'description': desc,
            'reasons': reasons[:6],
            'indicators': {
                'sma20': sf(sma20), 'sma50': sf(sma50), 'sma200': sf(sma200) if n >= 200 else None,
                'adx': sf(adx_val), 'rsi': sf(rsi),
                'ret20d': sf(ret_20d), 'ret50d': sf(ret_50d),
                'volatility': sf(volatility),
                'breadthRatio': sf(breadth_ratio),
            },
            'confidence_multiplier': {
                'buy': buy_confidence_mult,
                'sell': sell_confidence_mult,
            },
        }

        _market_regime_cache['regime'] = result
        _market_regime_cache['ts'] = time.time()
        return result
    except Exception as e:
        print(f"  [REGIME] Hata: {e}")
        return {'regime': 'unknown', 'strength': 0, 'description': str(e)}


# =====================================================================
# FEATURE 5: SECTOR ANALYSIS & RELATIVE STRENGTH
# =====================================================================
def calc_sector_relative_strength():
    """Sektor bazli goreceli guc analizi"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return {'sectors': []}

        stock_map = {s['code']: s for s in stocks}
        sector_results = []

        for sector_name, symbols in SECTOR_MAP.items():
            sector_stocks = []
            returns_1d = []
            returns_1w = []
            returns_1m = []

            for sym in symbols:
                s = stock_map.get(sym)
                if not s:
                    continue

                hist = _cget_hist(f"{sym}_1y")
                stock_info = {'code': sym, 'name': s.get('name', sym), 'price': s['price'], 'changePct': s.get('changePct', 0)}

                if hist is not None and len(hist) >= 22:
                    c = hist['Close'].values.astype(float)
                    n = len(c)
                    stock_info['ret1w'] = sf(((float(c[-1]) - float(c[-5])) / float(c[-5])) * 100) if n >= 5 else 0
                    stock_info['ret1m'] = sf(((float(c[-1]) - float(c[-22])) / float(c[-22])) * 100) if n >= 22 else 0
                    stock_info['ret3m'] = sf(((float(c[-1]) - float(c[-66])) / float(c[-66])) * 100) if n >= 66 else 0

                    # RSI
                    rsi = calc_rsi(c)
                    stock_info['rsi'] = rsi.get('value', 50)
                    stock_info['rsiSignal'] = rsi.get('signal', 'neutral')

                    returns_1d.append(s.get('changePct', 0))
                    returns_1w.append(float(stock_info.get('ret1w', 0)))
                    returns_1m.append(float(stock_info.get('ret1m', 0)))

                sector_stocks.append(stock_info)

            if not sector_stocks:
                continue

            avg_1d = sf(np.mean(returns_1d)) if returns_1d else 0
            avg_1w = sf(np.mean(returns_1w)) if returns_1w else 0
            avg_1m = sf(np.mean(returns_1m)) if returns_1m else 0

            # Relative Strength Index (sektor bazli)
            rs_score = float(avg_1d) * 0.2 + float(avg_1w) * 0.3 + float(avg_1m) * 0.5

            sector_results.append({
                'name': sector_name,
                'displayName': {
                    'bankacilik': 'Bankacilik', 'havacilik': 'Havacilik',
                    'otomotiv': 'Otomotiv', 'enerji': 'Enerji',
                    'holding': 'Holding', 'perakende': 'Perakende',
                    'teknoloji': 'Teknoloji', 'telekom': 'Telekom',
                    'demir_celik': 'Demir Celik', 'gida': 'Gida',
                    'insaat': 'Insaat', 'gayrimenkul': 'Gayrimenkul',
                }.get(sector_name, sector_name),
                'avgChange1d': avg_1d,
                'avgChange1w': avg_1w,
                'avgChange1m': avg_1m,
                'relativeStrength': sf(rs_score),
                'stockCount': len(sector_stocks),
                'stocks': sorted(sector_stocks, key=lambda x: float(x.get('ret1m', 0)), reverse=True),
                'topPerformer': max(sector_stocks, key=lambda x: float(x.get('ret1m', 0)))['code'] if sector_stocks else '',
                'worstPerformer': min(sector_stocks, key=lambda x: float(x.get('ret1m', 0)))['code'] if sector_stocks else '',
            })

        sector_results.sort(key=lambda x: float(x['relativeStrength']), reverse=True)

        # Sektor rotasyonu tespit
        rotation = 'neutral'
        if sector_results:
            top_sectors = sector_results[:3]
            defensive = ['perakende', 'gida', 'telekom']
            cyclical = ['bankacilik', 'otomotiv', 'enerji', 'holding']
            top_names = [s['name'] for s in top_sectors]
            if any(s in top_names for s in cyclical):
                rotation = 'risk_on'
            elif any(s in top_names for s in defensive):
                rotation = 'risk_off'

        return {
            'sectors': sector_results,
            'rotation': rotation,
            'rotationDescription': {
                'risk_on': 'Dongusel sektorler lider - Risk istahi yuksek',
                'risk_off': 'Defansif sektorler lider - Temkinli piyasa',
                'neutral': 'Belirgin sektor rotasyonu yok',
            }.get(rotation, ''),
        }
    except Exception as e:
        print(f"  [SECTOR-RS] Hata: {e}")
        return {'sectors': [], 'error': str(e)}


# =====================================================================
# FEATURE 6: FUNDAMENTAL ANALYSIS (F/K, PD/DD) - Is Yatirim Scraping
# =====================================================================
_fundamental_cache = {}

def fetch_fundamental_data(symbol):
    """Is Yatirim'dan temel analiz verilerini cek (F/K, PD/DD vs.)"""
    try:
        # Cache kontrol (1 saat)
        cached = _fundamental_cache.get(symbol)
        if cached and time.time() - cached['ts'] < 3600:
            return cached['data']

        url = f"https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/MaliTablo?hession={symbol}&doession=2024&dession=4"
        headers = IS_YATIRIM_HEADERS.copy()

        try:
            resp = req_lib.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                rows = data.get('value', [])
                if rows:
                    result = _parse_fundamental_data(rows, symbol)
                    _fundamental_cache[symbol] = {'data': result, 'ts': time.time()}
                    return result
        except Exception as e:
            print(f"  [FUNDAMENTAL] {symbol} Is Yatirim hata: {e}")

        # Fallback: yfinance info
        if YF_OK:
            try:
                tkr = yf.Ticker(f"{symbol}.IS")
                info = tkr.info or {}
                result = {
                    'pe': sf(info.get('trailingPE', 0)),
                    'forwardPE': sf(info.get('forwardPE', 0)),
                    'pb': sf(info.get('priceToBook', 0)),
                    'marketCap': info.get('marketCap', 0),
                    'dividendYield': sf(info.get('dividendYield', 0) * 100) if info.get('dividendYield') else 0,
                    'debtToEquity': sf(info.get('debtToEquity', 0)),
                    'roe': sf(info.get('returnOnEquity', 0) * 100) if info.get('returnOnEquity') else 0,
                    'roa': sf(info.get('returnOnAssets', 0) * 100) if info.get('returnOnAssets') else 0,
                    'profitMargin': sf(info.get('profitMargins', 0) * 100) if info.get('profitMargins') else 0,
                    'source': 'yfinance',
                }
                _fundamental_cache[symbol] = {'data': result, 'ts': time.time()}
                return result
            except Exception as e:
                print(f"  [FUNDAMENTAL] {symbol} yfinance hata: {e}")

        return {}
    except Exception as e:
        print(f"  [FUNDAMENTAL] {symbol}: {e}")
        return {}

def _parse_fundamental_data(rows, symbol):
    """Is Yatirim mali tablo verisini parse et"""
    try:
        result = {'source': 'isyatirim'}
        for row in rows:
            key = str(row.get('KALEM', '')).upper()
            val = row.get('DEGER', 0)
            if 'NET KAR' in key or 'NET DONEM' in key:
                result['netProfit'] = sf(float(val)) if val else 0
            elif 'SATIS' in key or 'GELIR' in key:
                result['revenue'] = sf(float(val)) if val else 0
            elif 'OZKAYN' in key:
                result['equity'] = sf(float(val)) if val else 0
        return result
    except:
        return {'source': 'isyatirim'}


# =====================================================================
# FEATURE 7: ENHANCED TELEGRAM/EMAIL ALERT SYSTEM
# =====================================================================
def check_signal_alerts():
    """Sinyal bazli otomatik uyari kontrolu - enhanced"""
    stocks = _get_stocks()
    if not stocks:
        return []

    alerts = []
    regime = calc_market_regime()
    regime_str = regime.get('regime', 'unknown')

    for stock in stocks:
        sym = stock['code']
        try:
            hist = _cget_hist(f"{sym}_1y")
            if hist is None or len(hist) < 50:
                continue

            c = hist['Close'].values.astype(float)
            h = hist['High'].values.astype(float)
            l = hist['Low'].values.astype(float)
            o = hist['Open'].values.astype(float)
            v = hist['Volume'].values.astype(float)
            cp = float(c[-1])
            n = len(c)

            # Mum formasyonlari
            candles = calc_candlestick_patterns(o, h, l, c)
            for p in candles.get('patterns', []):
                if p.get('strength', 0) >= 4:
                    alerts.append({
                        'symbol': sym,
                        'type': 'candlestick',
                        'signal': p['type'],
                        'message': f"{sym} ({sf(cp)} TL): {p['name']} - {p['description']}",
                        'strength': p['strength'],
                    })

            # Altin/Olum kesisim
            if n >= 200:
                ema50 = pd.Series(c).ewm(span=50).mean().values
                ema200 = pd.Series(c).ewm(span=200).mean().values
                if ema50[-1] > ema200[-1] and ema50[-2] <= ema200[-2]:
                    alerts.append({'symbol': sym, 'type': 'golden_cross', 'signal': 'bullish',
                        'message': f"ALTIN KESISIM: {sym} ({sf(cp)} TL) - EMA50 > EMA200", 'strength': 5})
                elif ema50[-1] < ema200[-1] and ema50[-2] >= ema200[-2]:
                    alerts.append({'symbol': sym, 'type': 'death_cross', 'signal': 'bearish',
                        'message': f"OLUM KESISIMI: {sym} ({sf(cp)} TL) - EMA50 < EMA200", 'strength': 5})

            # Dinamik RSI esikleri
            thresholds = calc_dynamic_thresholds(c, h, l, v)
            rsi_val = calc_rsi(c).get('value', 50)
            if rsi_val < float(thresholds['rsi_oversold']):
                alerts.append({'symbol': sym, 'type': 'rsi_dynamic', 'signal': 'bullish',
                    'message': f"RSI ASIRI SATIM: {sym} ({sf(cp)} TL) RSI={sf(rsi_val)} < {thresholds['rsi_oversold']} (dinamik esik)", 'strength': 3})
            elif rsi_val > float(thresholds['rsi_overbought']):
                alerts.append({'symbol': sym, 'type': 'rsi_dynamic', 'signal': 'bearish',
                    'message': f"RSI ASIRI ALIM: {sym} ({sf(cp)} TL) RSI={sf(rsi_val)} > {thresholds['rsi_overbought']} (dinamik esik)", 'strength': 3})

        except:
            continue

    # Strength'e gore sirala
    alerts.sort(key=lambda x: x.get('strength', 0), reverse=True)
    return alerts


# =====================================================================
# FEATURE 8: ML-BASED SIGNAL CONFIDENCE SCORING
# =====================================================================
def calc_ml_confidence(hist, indicators, recommendation_score, signal_type='buy'):
    """Sinyal guven skorunu coklu faktore gore hesapla (ML-inspired weighted scoring)"""
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        n = len(c)
        if n < 50:
            return {'confidence': 50, 'grade': 'C', 'factors': []}

        factors = []
        score = 0
        max_score = 0

        # 1. Indikatör Konsensüsü (agirlik: 25%)
        summary = indicators.get('summary', {})
        bc = summary.get('buySignals', 0)
        sc = summary.get('sellSignals', 0)
        total = summary.get('totalIndicators', 1)
        if signal_type == 'buy':
            consensus = bc / total * 100
        else:
            consensus = sc / total * 100
        consensus_score = min(consensus / 100 * 25, 25)
        score += consensus_score
        max_score += 25
        factors.append({'name': 'Indikatör Konsensüsü', 'value': sf(consensus), 'score': sf(consensus_score), 'max': 25})

        # 2. Piyasa Rejimi Uyumu (agirlik: 15%)
        regime = calc_market_regime()
        regime_type = regime.get('regime', 'unknown')
        regime_score = 0
        if signal_type == 'buy' and regime_type in ('strong_bull', 'bull'):
            regime_score = 15
        elif signal_type == 'sell' and regime_type in ('strong_bear', 'bear'):
            regime_score = 15
        elif regime_type == 'sideways':
            regime_score = 7.5
        elif signal_type == 'buy' and regime_type in ('strong_bear', 'bear'):
            regime_score = 3
        elif signal_type == 'sell' and regime_type in ('strong_bull', 'bull'):
            regime_score = 3
        else:
            regime_score = 10
        score += regime_score
        max_score += 15
        factors.append({'name': 'Piyasa Rejimi Uyumu', 'value': regime_type, 'score': sf(regime_score), 'max': 15})

        # 3. Hacim Teyidi (agirlik: 15%)
        vol_score = 0
        if n >= 20:
            vol_avg = float(np.mean(v[-20:]))
            vol_recent = float(np.mean(v[-3:]))
            vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 1
            if vol_ratio > 1.5:
                # Hacim teyidi var
                price_direction = 'up' if c[-1] > c[-3] else 'down'
                if (signal_type == 'buy' and price_direction == 'up') or (signal_type == 'sell' and price_direction == 'down'):
                    vol_score = 15  # Hacim sinyal yonunu destekliyor
                else:
                    vol_score = 5  # Hacim ters yonde
            elif vol_ratio > 1.0:
                vol_score = 10  # Normal hacim
            else:
                vol_score = 5  # Dusuk hacim
        score += vol_score
        max_score += 15
        factors.append({'name': 'Hacim Teyidi', 'value': sf(vol_ratio) if n >= 20 else '-', 'score': sf(vol_score), 'max': 15})

        # 4. Trend Uyumu (agirlik: 15%)
        trend_score = 0
        if n >= 50:
            s = pd.Series(c)
            ema20 = float(s.ewm(span=20).mean().iloc[-1])
            ema50 = float(s.ewm(span=50).mean().iloc[-1])
            cp = float(c[-1])
            if signal_type == 'buy':
                if cp > ema20 > ema50: trend_score = 15
                elif cp > ema20: trend_score = 10
                elif cp > ema50: trend_score = 7
                else: trend_score = 3
            else:
                if cp < ema20 < ema50: trend_score = 15
                elif cp < ema20: trend_score = 10
                elif cp < ema50: trend_score = 7
                else: trend_score = 3
        score += trend_score
        max_score += 15
        factors.append({'name': 'Trend Uyumu', 'value': 'EMA20/50', 'score': sf(trend_score), 'max': 15})

        # 5. Mum Formasyon Teyidi (agirlik: 10%)
        o = hist['Open'].values.astype(float)
        candles = calc_candlestick_patterns(o, h, l, c)
        candle_score = 0
        matching_patterns = [p for p in candles.get('patterns', [])
                            if (p['type'] == 'bullish' and signal_type == 'buy') or
                               (p['type'] == 'bearish' and signal_type == 'sell')]
        if matching_patterns:
            best_strength = max(p['strength'] for p in matching_patterns)
            candle_score = min(best_strength * 2, 10)
        score += candle_score
        max_score += 10
        factors.append({'name': 'Mum Formasyonu', 'value': matching_patterns[0]['name'] if matching_patterns else 'Yok', 'score': sf(candle_score), 'max': 10})

        # 6. Destek/Direnc Yakinligi (agirlik: 10%)
        sr_score = 0
        sr = calc_support_resistance(hist)
        cp = float(c[-1])
        if signal_type == 'buy' and sr.get('supports'):
            nearest_sup = sr['supports'][0]
            dist = abs(cp - nearest_sup) / cp * 100
            if dist < 2: sr_score = 10  # Desteğe çok yakin
            elif dist < 5: sr_score = 7
            else: sr_score = 3
        elif signal_type == 'sell' and sr.get('resistances'):
            nearest_res = sr['resistances'][0]
            dist = abs(nearest_res - cp) / cp * 100
            if dist < 2: sr_score = 10  # Dirence çok yakin
            elif dist < 5: sr_score = 7
            else: sr_score = 3
        score += sr_score
        max_score += 10
        factors.append({'name': 'Destek/Direnc', 'value': 'Yakin' if sr_score >= 7 else 'Uzak', 'score': sf(sr_score), 'max': 10})

        # 7. Geçmiş Sinyal Performansı (agirlik: 10%)
        backtest = calc_signal_backtest(hist)
        bt_score = 0
        if backtest.get('totalSignals', 0) > 5:
            if signal_type == 'buy':
                win_rate = float(backtest.get('buySignals', {}).get('winRate10d', 50))
            else:
                win_rate = float(backtest.get('sellSignals', {}).get('winRate10d', 50))
            bt_score = min(win_rate / 100 * 10, 10)
        else:
            bt_score = 5  # Yeterli veri yok, notr
        score += bt_score
        max_score += 10
        factors.append({'name': 'Gecmis Performans', 'value': f'{sf(win_rate)}%' if backtest.get('totalSignals', 0) > 5 else 'N/A', 'score': sf(bt_score), 'max': 10})

        # Final confidence
        confidence = sf(score / max_score * 100) if max_score > 0 else 50

        # Grade
        conf_val = float(confidence)
        if conf_val >= 80: grade = 'A'
        elif conf_val >= 65: grade = 'B'
        elif conf_val >= 50: grade = 'C'
        elif conf_val >= 35: grade = 'D'
        else: grade = 'F'

        return {
            'confidence': confidence,
            'grade': grade,
            'score': sf(score),
            'maxScore': sf(max_score),
            'factors': factors,
        }
    except Exception as e:
        print(f"  [ML-CONF] Hata: {e}")
        return {'confidence': 50, 'grade': 'C', 'factors': [], 'error': str(e)}


# =====================================================================
# FEATURE 9: DETAYLI TRADE PLAN (Gunluk/Haftalik/Aylik giriş/çıkış)
# =====================================================================
def calc_trade_plan(hist, indicators=None):
    """Her hisse icin gunluk/haftalik/aylik bazda detayli al-sat plani
    Entry, stop-loss, 3 hedef, risk/reward orani hesaplar"""
    try:
        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        # NaN temizligi
        h = np.where(np.isnan(h), c, h)
        l = np.where(np.isnan(l), c, l)
        v = np.where(np.isnan(v), 0, v)
        n = len(c)
        cur = float(c[-1])

        if n < 20:
            return {}

        # Destek/Direnc hesapla
        sr = calc_support_resistance(hist)
        supports = sr.get('supports', [])
        resistances = sr.get('resistances', [])

        # Fibonacci
        fib = calc_fibonacci(hist)
        fib_levels = fib.get('levels', {})
        fib_values = sorted([float(v) for v in fib_levels.values() if v > 0])

        # Pivot Points
        pp = calc_pivot_points(hist)
        classic = pp.get('classic', {})

        # ATR (volatilite bazli hedef)
        atr_data = calc_atr(h, l, c)
        atr_val = float(atr_data.get('value', 0))
        atr_pct = float(atr_data.get('pct', 2))

        # Bollinger
        bb = calc_bollinger(c, cur)
        bb_upper = float(bb.get('upper', 0))
        bb_lower = float(bb.get('lower', 0))
        bb_middle = float(bb.get('middle', 0))

        # EMA seviyeleri
        s_pd = pd.Series(c)
        ema20 = float(s_pd.ewm(span=20).mean().iloc[-1]) if n >= 20 else cur
        ema50 = float(s_pd.ewm(span=50).mean().iloc[-1]) if n >= 50 else cur
        ema200 = float(s_pd.ewm(span=200).mean().iloc[-1]) if n >= 200 else cur

        # RSI
        rsi_val = float(calc_rsi(c).get('value', 50))

        # Dinamik esikler
        dyn = calc_dynamic_thresholds(c, h, l, v)
        dyn_oversold = float(dyn.get('rsi_oversold', 30))
        dyn_overbought = float(dyn.get('rsi_overbought', 70))

        plans = {}

        for tf_label, tf_days, atr_mult, target_mult in [
            ('daily', 1, 1.0, [1.0, 1.5, 2.5]),
            ('weekly', 5, 1.5, [1.5, 2.5, 4.0]),
            ('monthly', 22, 2.5, [2.5, 4.0, 6.0]),
        ]:
            # Bu zaman dilimi icin yeterli veri var mi
            if n < tf_days + 20:
                continue

            # Zaman dilimi bazli getiri ve momentum
            tf_slice = c[-tf_days:] if tf_days <= n else c
            tf_high = float(np.max(h[-tf_days:])) if tf_days <= n else float(np.max(h))
            tf_low = float(np.min(l[-tf_days:])) if tf_days <= n else float(np.min(l))
            tf_ret = ((cur - float(c[-tf_days-1])) / float(c[-tf_days-1]) * 100) if n > tf_days else 0
            tf_range = tf_high - tf_low

            # Trend yonu (bu zaman diliminde)
            if n >= tf_days + 20:
                sma_short = float(np.mean(c[-min(10, tf_days):]))
                sma_long = float(np.mean(c[-min(tf_days+10, n):]))
                trend = 'yukari' if sma_short > sma_long else ('asagi' if sma_short < sma_long else 'yatay')
            else:
                trend = 'yatay'

            # ========== ALIS PLANI ==========
            buy_entry = None
            buy_sl = None
            buy_targets = []
            buy_reasons = []
            buy_strategy = ''

            # En yakin destek = alis noktasi
            nearby_supports = [s for s in supports if s < cur and s > cur * 0.90]
            nearby_resistances = [r for r in resistances if r > cur and r < cur * 1.20]

            # Fib destek seviyeleri
            fib_supports = [v for v in fib_values if v < cur and v > cur * 0.90]
            fib_resistances = [v for v in fib_values if v > cur and v < cur * 1.20]

            # -- Strateji 1: Destek'ten alis
            if nearby_supports:
                best_support = nearby_supports[0]
                buy_entry = sf(best_support)
                buy_sl = sf(best_support - atr_val * atr_mult)
                buy_reasons.append(f'Destek seviyesinden alis ({sf(best_support)} TL)')
                buy_strategy = 'destek_alis'

            # -- Strateji 2: Bollinger alt bant
            elif bb_lower > 0 and cur < bb_middle:
                buy_entry = sf(bb_lower)
                buy_sl = sf(bb_lower - atr_val * atr_mult)
                buy_reasons.append(f'Bollinger alt bant ({sf(bb_lower)} TL) yakininda alis')
                buy_strategy = 'bollinger_alis'

            # -- Strateji 3: EMA geri cekilme
            elif cur > ema20 > ema50:
                buy_entry = sf(ema20)
                buy_sl = sf(ema50 - atr_val * 0.5)
                buy_reasons.append(f'EMA20 ({sf(ema20)} TL) geri cekilmesinde alis')
                buy_strategy = 'ema_pullback'

            # -- Strateji 4: Momentum alis (mevcut fiyattan)
            elif rsi_val < dyn_oversold + 10 and trend != 'asagi':
                buy_entry = sf(cur)
                buy_sl = sf(cur - atr_val * atr_mult)
                buy_reasons.append(f'RSI={sf(rsi_val)} dusuk, momentum alisi')
                buy_strategy = 'momentum_alis'

            # -- Fallback: Pivot S1 seviyesinden
            else:
                s1 = float(classic.get('s1', cur * 0.97))
                buy_entry = sf(s1)
                buy_sl = sf(float(classic.get('s2', s1 - atr_val)))
                buy_reasons.append(f'Pivot S1 ({sf(s1)} TL) seviyesinden alis')
                buy_strategy = 'pivot_alis'

            # Giris fiyati mevcut fiyattan cok uzaksa (>%3) mevcut fiyati kullan
            if buy_entry and float(buy_entry) < cur * 0.97:
                buy_entry = sf(cur)
                buy_sl = sf(cur - atr_val * atr_mult)
                if 'destek_alis' in buy_strategy or 'bollinger_alis' in buy_strategy or 'pivot_alis' in buy_strategy:
                    buy_reasons.append(f'Giris mevcut fiyata ({sf(cur)} TL) ayarlandi')

            # Alis hedefleri
            entry_price = float(buy_entry)
            sl_price = float(buy_sl)
            risk = entry_price - sl_price

            if risk > 0:
                for i, mult in enumerate(target_mult):
                    raw_target = entry_price + risk * mult
                    # Direnc seviyesine yakin mi?
                    snapped = raw_target
                    for r in (nearby_resistances + fib_resistances):
                        if abs(r - raw_target) < atr_val:
                            snapped = r
                            break
                    buy_targets.append(sf(snapped))
                    buy_reasons.append(f'Hedef {i+1}: {sf(snapped)} TL (R/R {mult:.1f}x)')

            # Risk/Reward orani
            if risk > 0 and buy_targets:
                rr_ratio = sf((float(buy_targets[-1]) - entry_price) / risk)
            else:
                rr_ratio = 0

            # ========== SATIS PLANI ==========
            sell_entry = None
            sell_sl = None
            sell_targets = []
            sell_reasons = []
            sell_strategy = ''

            # -- Strateji 1: Direncten satis
            if nearby_resistances:
                best_res = nearby_resistances[0]
                sell_entry = sf(best_res)
                sell_sl = sf(best_res + atr_val * atr_mult)
                sell_reasons.append(f'Direnc seviyesinde satis ({sf(best_res)} TL)')
                sell_strategy = 'direnc_satis'

            # -- Strateji 2: Bollinger ust bant
            elif bb_upper > 0 and cur > bb_middle:
                sell_entry = sf(bb_upper)
                sell_sl = sf(bb_upper + atr_val * atr_mult)
                sell_reasons.append(f'Bollinger ust bant ({sf(bb_upper)} TL) yakininda satis')
                sell_strategy = 'bollinger_satis'

            # -- Strateji 3: RSI asiri alim
            elif rsi_val > dyn_overbought - 5:
                sell_entry = sf(cur)
                sell_sl = sf(cur + atr_val * atr_mult)
                sell_reasons.append(f'RSI={sf(rsi_val)} yuksek, satis baskisi')
                sell_strategy = 'rsi_satis'

            # -- Fallback: Pivot R1
            else:
                r1 = float(classic.get('r1', cur * 1.03))
                sell_entry = sf(r1)
                sell_sl = sf(float(classic.get('r2', r1 + atr_val)))
                sell_reasons.append(f'Pivot R1 ({sf(r1)} TL) seviyesinde satis')
                sell_strategy = 'pivot_satis'

            # Satis girisi mevcut fiyattan cok uzaksa (>%3 yukarda) mevcut fiyati kullan
            if sell_entry and float(sell_entry) > cur * 1.03:
                sell_entry = sf(cur)
                sell_sl = sf(cur + atr_val * atr_mult)
                if 'direnc_satis' in sell_strategy or 'bollinger_satis' in sell_strategy or 'pivot_satis' in sell_strategy:
                    sell_reasons.append(f'Giris mevcut fiyata ({sf(cur)} TL) ayarlandi')

            # Satis hedefleri (asagi)
            s_entry = float(sell_entry)
            s_risk = float(sell_sl) - s_entry

            if s_risk > 0:
                for i, mult in enumerate(target_mult):
                    raw_target = s_entry - s_risk * mult
                    # Destege yakin mi?
                    snapped = raw_target
                    for sup in (nearby_supports + fib_supports):
                        if abs(sup - raw_target) < atr_val:
                            snapped = sup
                            break
                    sell_targets.append(sf(max(snapped, 0.01)))
                    sell_reasons.append(f'Hedef {i+1}: {sf(max(snapped, 0.01))} TL')

            # Sell R/R
            if s_risk > 0 and sell_targets:
                sell_rr = sf((s_entry - float(sell_targets[-1])) / s_risk)
            else:
                sell_rr = 0

            # Mevcut fiyatin plana gore durumu
            if float(buy_entry) > 0:
                entry_dist = sf(((cur - float(buy_entry)) / float(buy_entry)) * 100)
            else:
                entry_dist = 0

            # Genel sinyal (bu zaman dilimi icin)
            if trend == 'yukari' and rsi_val < 60:
                tf_signal = 'AL'
                tf_signal_desc = 'Trend yukari, geri cekilmede alis firsati'
            elif trend == 'yukari' and rsi_val >= 60:
                tf_signal = 'BEKLE'
                tf_signal_desc = 'Trend yukari ama asiri alimda, geri cekilme bekle'
            elif trend == 'asagi' and rsi_val > 40:
                tf_signal = 'SAT'
                tf_signal_desc = 'Trend asagi, direncte satis firsati'
            elif trend == 'asagi' and rsi_val <= 40:
                tf_signal = 'BEKLE'
                tf_signal_desc = 'Trend asagi ama asiri satimda, toparlanma bekle'
            else:
                tf_signal = 'BEKLE'
                tf_signal_desc = 'Belirsiz yonde, net sinyal icin bekle'

            plans[tf_label] = {
                'trend': trend,
                'signal': tf_signal,
                'signalDescription': tf_signal_desc,
                'momentum': sf(tf_ret),
                'range': sf(tf_range),
                'atr': sf(atr_val),
                'rsi': sf(rsi_val),

                # Alis plani
                'buy': {
                    'entry': buy_entry,
                    'stopLoss': buy_sl,
                    'targets': buy_targets,
                    'riskReward': rr_ratio,
                    'strategy': buy_strategy,
                    'reasons': buy_reasons[:5],
                    'risk': sf(risk) if risk > 0 else 0,
                    'distanceFromEntry': entry_dist,
                },

                # Satis plani
                'sell': {
                    'entry': sell_entry,
                    'stopLoss': sell_sl,
                    'targets': sell_targets,
                    'riskReward': sell_rr,
                    'strategy': sell_strategy,
                    'reasons': sell_reasons[:5],
                },

                # Onemli seviyeler
                'keyLevels': {
                    'supports': supports[:3],
                    'resistances': resistances[:3],
                    'ema20': sf(ema20),
                    'ema50': sf(ema50),
                    'ema200': sf(ema200) if n >= 200 else None,
                    'bbUpper': sf(bb_upper),
                    'bbLower': sf(bb_lower),
                    'bbMiddle': sf(bb_middle),
                    'pivotPP': classic.get('pp', 0),
                    'pivotR1': classic.get('r1', 0),
                    'pivotS1': classic.get('s1', 0),
                },
            }

        return plans
    except Exception as e:
        print(f"  [TRADE-PLAN] Hata: {e}")
        return {}


def prepare_chart_data(hist):
    try:
        cs=[{'date':d.strftime('%Y-%m-%d'),'open':sf(r['Open']),'high':sf(r['High']),'low':sf(r['Low']),'close':sf(r['Close']),'volume':si(r['Volume'])} for d,r in hist.iterrows()]
        return {'candlestick':cs,'dates':[c['date'] for c in cs],'prices':[c['close'] for c in cs],'volumes':[c['volume'] for c in cs],'dataPoints':len(cs)}
    except: return {'candlestick':[],'dates':[],'prices':[],'volumes':[],'dataPoints':0}

# =====================================================================
# ROUTES - SIFIR YF CAGRISI
# =====================================================================
@app.route('/api/health')
def health():
    now = time.time()
    with _lock:
        fresh = [k for k, v in _stock_cache.items() if now - v['ts'] < CACHE_TTL]
        stale = [k for k, v in _stock_cache.items() if CACHE_TTL <= now - v['ts'] < CACHE_STALE_TTL]
    hist_ready = sum(1 for s in BIST100_STOCKS if _cget_hist(f"{s}_1y") is not None)
    return jsonify({
        'status': 'ok', 'version': '7.0.0', 'yf': YF_OK,
        'time': datetime.now().isoformat(),
        'loader': _status,
        'loaderStarted': _loader_started,
        'stockCache': len(_stock_cache),
        'stockCacheFresh': len(fresh),
        'stockCacheStale': len(stale),
        'indexCache': len(_index_cache),
        'histCache': hist_ready,
        'histCacheTotal': len(_hist_cache),
        'cachedStocks': list(_stock_cache.keys()),
        'cachedIndices': list(_index_cache.keys()),
        'totalDefined': len(BIST100_STOCKS),
        'missingStocks': [s for s in BIST100_STOCKS.keys() if s not in _stock_cache],
    })

@app.route('/api/debug')
def debug():
    """Detayli debug - Render loglarinda ne olduğunu goster"""
    now = time.time()
    stock_details = {}
    with _lock:
        for k, v in _stock_cache.items():
            age = round(now - v['ts'], 1)
            status = 'fresh' if age < CACHE_TTL else ('stale' if age < CACHE_STALE_TTL else 'expired')
            stock_details[k] = {'price': v['data']['price'], 'age_sec': age, 'status': status}
    index_details = {}
    with _lock:
        for k, v in _index_cache.items():
            age = round(now - v['ts'], 1)
            index_details[k] = {'value': v['data']['value'], 'age_sec': age}
    missing = [s for s in BIST100_STOCKS.keys() if s not in _stock_cache]
    return jsonify({
        'loaderStarted': _loader_started,
        'status': _status,
        'stockCache': stock_details,
        'indexCache': index_details,
        'totalStocks': len(stock_details),
        'totalDefined': len(BIST100_STOCKS),
        'missingStocks': missing,
        'totalMissing': len(missing),
        'totalIndices': len(index_details),
        'yfinance': YF_OK,
        'time': datetime.now().isoformat(),
    })

@app.route('/api/test-fetch/<symbol>')
def test_fetch(symbol):
    """Veri cekme pipeline'ini test et - debug icin"""
    symbol = symbol.upper()
    results = {}

    # 1. Is Yatirim quick
    try:
        data = _fetch_isyatirim_quick(symbol)
        results['isyatirim_quick'] = {'success': data is not None, 'data': data}
    except Exception as e:
        results['isyatirim_quick'] = {'success': False, 'error': str(e)}

    # 2. Is Yatirim DF (30 gun)
    try:
        df = _fetch_isyatirim_df(symbol, days=30)
        if df is not None:
            nan_counts = df[['Open','High','Low','Close','Volume']].isna().sum().to_dict()
            results['isyatirim_df'] = {
                'success': True, 'rows': len(df),
                'last_close': float(df['Close'].iloc[-1]),
                'last_high': float(df['High'].iloc[-1]),
                'last_low': float(df['Low'].iloc[-1]),
                'nan_counts': nan_counts,
                'date_range': f"{df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}",
            }
        else:
            results['isyatirim_df'] = {'success': False, 'data': None}
    except Exception as e:
        results['isyatirim_df'] = {'success': False, 'error': str(e)}

    # 3. Yahoo HTTP
    try:
        data = _fetch_yahoo_http(f"{symbol}.IS")
        results['yahoo_http'] = {'success': data is not None, 'data': data}
    except Exception as e:
        results['yahoo_http'] = {'success': False, 'error': str(e)}

    # 4. Yahoo DF
    try:
        df = _fetch_yahoo_http_df(f"{symbol}.IS", period1_days=30)
        if df is not None:
            results['yahoo_df'] = {'success': True, 'rows': len(df), 'last_close': float(df['Close'].iloc[-1])}
        else:
            results['yahoo_df'] = {'success': False, 'data': None}
    except Exception as e:
        results['yahoo_df'] = {'success': False, 'error': str(e)}

    # 5. Cache durumu
    cached = _cget(_stock_cache, symbol)
    results['cache'] = {'has_cache': cached is not None}
    if cached:
        results['cache']['data'] = cached

    return jsonify(safe_dict(results))

# --- index.html bellekte gzipli cache ---
_index_cache = {'raw': None, 'gz': None, 'mtime': 0}

def _load_index_html():
    """index.html'i diskten oku, gziple, bellekte tut"""
    fpath = os.path.join(BASE_DIR, 'index.html')
    try:
        mt = os.path.getmtime(fpath)
        if _index_cache['raw'] and _index_cache['mtime'] == mt:
            return True
        with open(fpath, 'rb') as f:
            raw = f.read()
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode='wb', compresslevel=6) as gz:
            gz.write(raw)
        _index_cache['raw'] = raw
        _index_cache['gz'] = buf.getvalue()
        _index_cache['mtime'] = mt
        print(f"[INDEX] Cached: {len(raw)} bytes -> gzip {len(_index_cache['gz'])} bytes")
        return True
    except Exception as e:
        print(f"[INDEX] Load error: {e}")
        return False

_load_index_html()

@app.route('/')
def index():
    if not _index_cache['raw']:
        if not _load_index_html():
            return jsonify({'error':'index.html bulunamadi'}), 500
    # gzip destegi varsa sikistirilmis gonder
    ae = request.headers.get('Accept-Encoding', '')
    if 'gzip' in ae and _index_cache['gz']:
        resp = make_response(_index_cache['gz'])
        resp.headers['Content-Encoding'] = 'gzip'
    else:
        resp = make_response(_index_cache['raw'])
    resp.headers['Content-Type'] = 'text/html; charset=utf-8'
    resp.headers['Cache-Control'] = 'public, max-age=300'
    resp.headers['Vary'] = 'Accept-Encoding'
    return resp

@app.route('/api/dashboard')
def dashboard():
    try:
        stocks=_get_stocks()
        if not stocks:
            return jsonify(safe_dict({'success':True,'loading':True,'stockCount':0,'message':f"Veriler yukleniyor ({_status['loaded']}/{_status['total']})...",'movers':{'topGainers':[],'topLosers':[],'volumeLeaders':[],'gapStocks':[]},'marketBreadth':{'advancing':0,'declining':0,'unchanged':0,'advDecRatio':0},'allStocks':[],'meta':_api_meta('loading')}))
        sbc=sorted(stocks,key=lambda x:x.get('changePct',0),reverse=True)
        adv=sum(1 for s in stocks if s.get('changePct',0)>0)
        dec=sum(1 for s in stocks if s.get('changePct',0)<0)
        return jsonify(safe_dict({'success':True,'loading':False,'stockCount':len(stocks),'timestamp':datetime.now().isoformat(),'movers':{'topGainers':sbc[:5],'topLosers':sbc[-5:][::-1],'volumeLeaders':sorted(stocks,key=lambda x:x.get('volume',0),reverse=True)[:5],'gapStocks':sorted(stocks,key=lambda x:abs(x.get('gapPct',0)),reverse=True)[:5]},'marketBreadth':{'advancing':adv,'declining':dec,'unchanged':len(stocks)-adv-dec,'advDecRatio':sf(adv/dec if dec>0 else adv)},'allStocks':sbc,'meta':_api_meta()}))
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/indices')
def indices():
    try:
        data=_get_indices()
        if not data:
            return jsonify(safe_dict({'success':True,'loading':True,'indices':{},'message':'Endeksler yukleniyor...'}))

        # Gold/TL ve Silver/TL hesapla
        usdtry_data = data.get('USDTRY')
        gold_data = data.get('GOLD')
        silver_data = data.get('SILVER')
        if usdtry_data and gold_data:
            usd_rate = usdtry_data.get('value', 0)
            if usd_rate > 0:
                gold_usd = gold_data.get('value', 0)
                gold_tl = sf(gold_usd * usd_rate / 31.1035, 2)  # ons -> gram (1 ons = 31.1035 gram)
                data['GOLDTL'] = {
                    'name': 'Altin/TL (gram)',
                    'value': gold_tl,
                    'change': 0,
                    'changePct': gold_data.get('changePct', 0),
                    'volume': 0,
                }
        if usdtry_data and silver_data:
            usd_rate = usdtry_data.get('value', 0)
            if usd_rate > 0:
                silver_usd = silver_data.get('value', 0)
                silver_tl = sf(silver_usd * usd_rate / 31.1035, 2)  # ons -> gram
                data['SILVERTL'] = {
                    'name': 'Gumus/TL (gram)',
                    'value': silver_tl,
                    'change': 0,
                    'changePct': silver_data.get('changePct', 0),
                    'volume': 0,
                }

        return jsonify(safe_dict({'success':True,'indices':data}))
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/bist100')
def bist100():
    try:
        sector=request.args.get('sector'); sort_by=request.args.get('sort','code'); order=request.args.get('order','asc')
        stocks=_get_stocks(SECTOR_MAP[sector]) if sector and sector in SECTOR_MAP else _get_stocks()
        if not stocks:
            return jsonify(safe_dict({'success':True,'stocks':[],'count':0,'sectors':list(SECTOR_MAP.keys()),'loading':True,'message':f"Hisse verileri yukleniyor ({_status['loaded']}/{_status['total']})..."}))
        rev=(order=='desc'); km={'change':'changePct','volume':'volume','price':'price'}; sk=km.get(sort_by,'code')
        stocks.sort(key=lambda x:x.get(sk,0) if sk!='code' else x.get('code',''),reverse=rev)
        return jsonify(safe_dict({'success':True,'stocks':stocks,'count':len(stocks),'sectors':list(SECTOR_MAP.keys()),'meta':_api_meta()}))
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/bist30')
def bist30():
    try:
        stocks=_get_stocks(BIST30)
        if not stocks: return jsonify(safe_dict({'success':True,'stocks':[],'count':0,'loading':True}))
        stocks.sort(key=lambda x:x.get('code','')); return jsonify(safe_dict({'success':True,'stocks':stocks,'count':len(stocks)}))
    except Exception as e:
        return jsonify({'error':str(e)}),500

@app.route('/api/stock/<symbol>')
def stock_detail(symbol):
    """Tek hisse detay - SENKRON, 3 katmanli fallback (IsYatirim -> Yahoo -> yfinance)"""
    try:
        symbol=symbol.upper(); period=request.args.get('period','1y')

        # 1. Once hist cache'e bak
        hist=_cget_hist(f"{symbol}_{period}")

        # 2. Cache'de yoksa SENKRON cek (birlesik fetcher)
        if hist is None:
            print(f"[DETAIL] {symbol} {period} cekiliyor...")
            hist = _fetch_hist_df(symbol, period)
            if hist is not None and len(hist) >= 2:
                _cset(_hist_cache, f"{symbol}_{period}", hist)
                print(f"[DETAIL] {symbol} OK: {len(hist)} bar")
            else:
                hist = None

        # 3. Hist yoksa quick cache'den don
        if hist is None:
            quick=_cget(_stock_cache, symbol)
            if quick:
                return jsonify(safe_dict({
                    'success':True,'code':symbol,'name':quick['name'],
                    'price':quick['price'],'change':quick['change'],
                    'changePercent':quick['changePct'],
                    'volume':quick['volume'],'dayOpen':quick['open'],
                    'dayHigh':quick['high'],'dayLow':quick['low'],
                    'prevClose':quick['prevClose'],'currency':'TRY',
                    'period':period,'dataPoints':0,
                    'indicators':{},'chartData':{'candlestick':[],'dates':[],'prices':[],'volumes':[],'dataPoints':0},
                    'fibonacci':{'levels':{}},'supportResistance':{'supports':[],'resistances':[]},
                    'pivotPoints':{'classic':{},'camarilla':{},'woodie':{},'current':0},
                    'recommendation':{'weekly':{'action':'neutral','confidence':0,'reasons':[]},'monthly':{'action':'neutral','confidence':0,'reasons':[]},'yearly':{'action':'neutral','confidence':0,'reasons':[]}}
                }))
            return jsonify({'error':f'{symbol} verisi bulunamadi'}),404

        # 4. Hist var - TAM ANALIZ yap
        # Son NaN temizligi (guvenlik katmani)
        if hist[['Open','High','Low']].isna().any().any():
            hist['Open'] = hist['Open'].fillna(hist['Close'])
            hist['High'] = hist['High'].fillna(hist['Close'])
            hist['Low'] = hist['Low'].fillna(hist['Close'])
            hist['Volume'] = hist['Volume'].fillna(0)

        cp=float(hist['Close'].iloc[-1])
        prev=float(hist['Close'].iloc[-2]) if len(hist)>1 else cp
        w52=calc_52w(hist)
        ind=calc_all_indicators(hist,cp)
        rec=calc_recommendation(hist, ind)

        # ML Confidence (sadece aktif sinyal varsa)
        ml_conf = {}
        for tf_label in ['weekly', 'monthly', 'yearly']:
            tf_rec = rec.get(tf_label, {})
            tf_action = tf_rec.get('action', 'NOTR')
            if tf_action in ('AL', 'TUTUN/AL'):
                ml_conf[tf_label] = calc_ml_confidence(hist, ind, float(tf_rec.get('score', 0)), 'buy')
            elif tf_action in ('SAT', 'TUTUN/SAT'):
                ml_conf[tf_label] = calc_ml_confidence(hist, ind, float(tf_rec.get('score', 0)), 'sell')
            else:
                ml_conf[tf_label] = {'confidence': 50, 'grade': 'C', 'factors': []}

        return jsonify(safe_dict({
            'success':True,'code':symbol,
            'name':BIST100_STOCKS.get(symbol,symbol),
            'price':sf(cp),'change':sf(cp-prev),
            'changePercent':sf((cp-prev)/prev*100 if prev else 0),
            'volume':si(hist['Volume'].iloc[-1]),
            'dayHigh':sf(hist['High'].iloc[-1]),
            'dayLow':sf(hist['Low'].iloc[-1]),
            'dayOpen':sf(hist['Open'].iloc[-1]),
            'prevClose':sf(prev),'currency':'TRY',
            'period':period,'dataPoints':len(hist),
            'week52':w52,
            'marketValue':sf(cp * si(hist['Volume'].iloc[-1])),
            'indicators':ind,
            'chartData':prepare_chart_data(hist),
            'fibonacci':calc_fibonacci(hist),
            'supportResistance':calc_support_resistance(hist),
            'pivotPoints':calc_pivot_points(hist),
            'recommendation':rec,
            'mlConfidence':ml_conf,
            'signalBacktest':calc_signal_backtest(hist),
            'tradePlan':calc_trade_plan(hist, ind),
            'marketRegime':calc_market_regime(),
            'fundamentals':calc_fundamentals(hist, symbol),
            'meta':_api_meta(),
        }))
    except Exception as e:
        print(f"STOCK {symbol}: {traceback.format_exc()}")
        return jsonify({'error':str(e)}),500

@app.route('/api/commodity/<symbol>')
def commodity_detail(symbol):
    """Emtia detay - Altin, Gumus, GOLDTL, SILVERTL icin hisse gibi tam analiz"""
    try:
        symbol = symbol.upper()
        period = request.args.get('period', '1y')

        COMMODITY_MAP = {
            'GOLD': ('GC=F', 'Altin (USD/ons)', 'USD'),
            'SILVER': ('SI=F', 'Gumus (USD/ons)', 'USD'),
            'GOLDTL': ('GC=F', 'Altin/TL (gram)', 'TRY'),
            'SILVERTL': ('SI=F', 'Gumus/TL (gram)', 'TRY'),
            'USDTRY': ('USDTRY=X', 'Dolar/TL', 'TRY'),
            'EURTRY': ('EURTRY=X', 'Euro/TL', 'TRY'),
        }

        if symbol not in COMMODITY_MAP:
            return jsonify({'error': f'{symbol} desteklenmiyor. Desteklenen: {list(COMMODITY_MAP.keys())}'}), 400

        yahoo_sym, name, currency = COMMODITY_MAP[symbol]
        period_days = {'1mo':30,'3mo':90,'6mo':180,'1y':365,'2y':730,'5y':1825}.get(period, 365)

        # Hist cache key
        cache_key = f"COMMODITY_{symbol}_{period}"
        hist = _cget_hist(cache_key)

        if hist is None:
            print(f"[COMMODITY] {symbol} ({yahoo_sym}) {period} cekiliyor...")
            # Yahoo HTTP
            hist = _fetch_yahoo_http_df(yahoo_sym, period_days)
            # yfinance fallback
            if (hist is None or len(hist) < 10) and YF_OK:
                try:
                    h = yf.Ticker(yahoo_sym).history(period=period, timeout=15)
                    if h is not None and not h.empty and len(h) >= 10:
                        hist = h
                        print(f"  [YF-COMMODITY] {symbol} OK: {len(hist)} bar")
                except Exception as e:
                    print(f"  [YF-COMMODITY] {symbol}: {e}")

            if hist is not None and len(hist) >= 2:
                # GOLDTL / SILVERTL icin: USD fiyat * USDTRY / 31.1035 (ons->gram)
                if symbol in ('GOLDTL', 'SILVERTL'):
                    usd_hist = _fetch_yahoo_http_df('USDTRY=X', period_days)
                    if usd_hist is None and YF_OK:
                        try:
                            usd_hist = yf.Ticker('USDTRY=X').history(period=period, timeout=15)
                        except:
                            pass
                    if usd_hist is not None and len(usd_hist) >= 2:
                        # Normalize dates (remove time component for matching)
                        hist.index = hist.index.normalize()
                        usd_hist.index = usd_hist.index.normalize()
                        # Remove duplicate dates (keep last)
                        hist = hist[~hist.index.duplicated(keep='last')]
                        usd_hist = usd_hist[~usd_hist.index.duplicated(keep='last')]
                        common_dates = hist.index.intersection(usd_hist.index)
                        print(f"  [COMMODITY] {symbol} tarih eslestirme: hist={len(hist)}, usd={len(usd_hist)}, ortak={len(common_dates)}")
                        if len(common_dates) >= 10:
                            hist = hist.loc[common_dates].copy()
                            usd_rates = usd_hist.loc[common_dates, 'Close']
                            ons_to_gram = 31.1035
                            for col in ['Open', 'High', 'Low', 'Close']:
                                hist[col] = hist[col] * usd_rates.values / ons_to_gram
                            print(f"  [COMMODITY] {symbol} TL donusumu OK: {len(hist)} bar, son fiyat: {hist['Close'].iloc[-1]:.2f}")
                        else:
                            # Fallback: tek kur ile tum seriyi donustur
                            print(f"  [COMMODITY] {symbol} TL donusumu: ortak tarih az, son kur ile donusturuluyor")
                            last_usd_rate = float(usd_hist['Close'].iloc[-1])
                            ons_to_gram = 31.1035
                            for col in ['Open', 'High', 'Low', 'Close']:
                                hist[col] = hist[col] * last_usd_rate / ons_to_gram
                    else:
                        # USDTRY verisi yok, indeks cache'den kur al
                        print(f"  [COMMODITY] {symbol}: USDTRY hist yok, cache'den kur aliniyor")
                        usd_idx = _cget(_index_cache, 'USDTRY')
                        if usd_idx:
                            last_rate = usd_idx.get('value', 0)
                            if last_rate > 0:
                                ons_to_gram = 31.1035
                                for col in ['Open', 'High', 'Low', 'Close']:
                                    hist[col] = hist[col] * last_rate / ons_to_gram

                _cset(_hist_cache, cache_key, hist)
                print(f"[COMMODITY] {symbol} OK: {len(hist)} bar")
            else:
                hist = None

        if hist is None:
            # Fallback: indeks cache'den basit veri don
            idx = _cget(_index_cache, symbol) or _cget(_index_cache, symbol.replace('TL',''))
            if idx:
                return jsonify(safe_dict({
                    'success': True, 'code': symbol, 'name': name,
                    'price': idx.get('value', 0), 'change': idx.get('change', 0),
                    'changePercent': idx.get('changePct', 0), 'volume': 0,
                    'currency': currency, 'period': period, 'dataPoints': 0,
                    'indicators': {}, 'chartData': {'dates':[],'prices':[],'volumes':[],'dataPoints':0},
                    'fibonacci': {'levels':{}}, 'supportResistance': {'supports':[],'resistances':[]},
                    'pivotPoints': {'classic':{},'camarilla':{},'woodie':{},'current':0},
                    'recommendation': {'weekly':{'action':'neutral','confidence':0,'reasons':[]},'monthly':{'action':'neutral','confidence':0,'reasons':[]},'yearly':{'action':'neutral','confidence':0,'reasons':[]}}
                }))
            return jsonify({'error': f'{symbol} verisi bulunamadi'}), 404

        # Tam analiz (hisse gibi)
        cp = float(hist['Close'].iloc[-1])
        prev = float(hist['Close'].iloc[-2]) if len(hist) > 1 else cp
        w52 = calc_52w(hist)

        return jsonify(safe_dict({
            'success': True, 'code': symbol, 'name': name,
            'price': sf(cp, 4 if currency == 'USD' else 2),
            'change': sf(cp - prev, 4 if currency == 'USD' else 2),
            'changePercent': sf((cp - prev) / prev * 100 if prev else 0),
            'volume': si(hist['Volume'].iloc[-1]) if 'Volume' in hist.columns else 0,
            'dayHigh': sf(hist['High'].iloc[-1], 4 if currency == 'USD' else 2),
            'dayLow': sf(hist['Low'].iloc[-1], 4 if currency == 'USD' else 2),
            'dayOpen': sf(hist['Open'].iloc[-1], 4 if currency == 'USD' else 2),
            'prevClose': sf(prev, 4 if currency == 'USD' else 2),
            'currency': currency,
            'period': period, 'dataPoints': len(hist),
            'week52': w52,
            'indicators': calc_all_indicators(hist, cp),
            'chartData': prepare_chart_data(hist),
            'fibonacci': calc_fibonacci(hist),
            'supportResistance': calc_support_resistance(hist),
            'pivotPoints': calc_pivot_points(hist),
            'recommendation': calc_recommendation(hist, None),
            'fundamentals': calc_fundamentals(hist, symbol),
        }))
    except Exception as e:
        print(f"COMMODITY {symbol}: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stock/<symbol>/events')
def stock_events(symbol):
    return jsonify({'success':True,'symbol':symbol.upper(),'events':{'dividends':[],'splits':[]}})

@app.route('/api/stock/<symbol>/kap')
def stock_kap(symbol):
    """KAP bildirimlerini scrape et"""
    symbol = symbol.upper()
    try:
        url = f"https://www.kap.org.tr/tr/api/disclosures?company={symbol}&type=FR&lang=tr"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://www.kap.org.tr/',
        }
        try:
            resp = req_lib.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                notifications = []
                items = data if isinstance(data, list) else data.get('disclosures', data.get('data', []))
                for item in items[:20]:
                    notifications.append({
                        'title': item.get('title', item.get('subject', '')),
                        'date': item.get('publishDate', item.get('date', '')),
                        'type': item.get('type', item.get('disclosureType', '')),
                        'summary': item.get('summary', '')[:200],
                    })
                if notifications:
                    return jsonify({'success': True, 'symbol': symbol, 'notifications': notifications})
        except Exception as e:
            print(f"[KAP-API] {symbol}: {e}")

        # Fallback: Is Yatirim haberler
        try:
            url2 = f"https://www.isyatirim.com.tr/_layouts/15/Isyatirim.Website/Common/Data.aspx/HaberlerHisseTekil?hession={symbol}&startdate={datetime.now().strftime('%d-%m-%Y')}&enddate={datetime.now().strftime('%d-%m-%Y')}"
            resp2 = req_lib.get(url2, headers=IS_YATIRIM_HEADERS, timeout=10)
            if resp2.status_code == 200:
                data2 = resp2.json()
                news = data2.get('value', [])
                notifications = [{'title': n.get('BASLIK', ''), 'date': n.get('TARIH', ''), 'type': 'haber', 'summary': ''} for n in news[:10]]
                return jsonify({'success': True, 'symbol': symbol, 'notifications': notifications})
        except Exception as e:
            print(f"[KAP-ISYATIRIM] {symbol}: {e}")

        return jsonify({'success': True, 'symbol': symbol, 'notifications': [], 'message': 'KAP verisi alinamadi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare():
    """Hisseleri detayli karsilastir"""
    try:
        data = request.json or {}
        symbols = data.get('symbols', [])
        if len(symbols) < 2:
            return jsonify({'error': 'En az 2 hisse gerekli'}), 400

        results = []
        for sym in symbols[:5]:
            sym = sym.upper()
            s = _cget(_stock_cache, sym)
            if not s:
                continue

            # Indicator hesapla
            hist = _cget_hist(f"{sym}_1y")
            indicators = {}
            if hist is not None and len(hist) >= 14:
                c = hist['Close'].values.astype(float)
                h = hist['High'].values.astype(float)
                l = hist['Low'].values.astype(float)
                rsi = calc_rsi(c)
                macd = calc_macd(c)
                ema = calc_ema(c, float(c[-1]))
                w52 = calc_52w(hist)
                indicators = {
                    'rsi': rsi.get('value', 0),
                    'rsiSignal': rsi.get('signal', 'neutral'),
                    'macdSignal': macd.get('signalType', 'neutral'),
                    'ema20': ema.get('ema20', 0),
                    'ema50': ema.get('ema50', 0),
                    'high52w': w52.get('high52w', 0),
                    'low52w': w52.get('low52w', 0),
                    'pos52w': w52.get('currentPct', 50),
                }

            results.append({
                'code': sym, 'name': s['name'], 'price': s['price'],
                'change': s['change'], 'changePct': s['changePct'],
                'volume': s['volume'], 'open': s.get('open', 0),
                'high': s.get('high', 0), 'low': s.get('low', 0),
                'prevClose': s.get('prevClose', 0),
                'gap': s.get('gap', 0), 'gapPct': s.get('gapPct', 0),
                **indicators
            })

        return jsonify(safe_dict({'success': True, 'comparison': results}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/screener', methods=['POST'])
def screener():
    try:
        stocks=_get_stocks()
        if not stocks: return jsonify({'success':True,'matches':[],'message':'Veriler yukleniyor'})
        conditions=request.json.get('conditions',[]) if request.json else []
        matches=[]
        for s in stocks:
            ok=True
            for cd in conditions:
                ind,op,val=cd.get('indicator',''),cd.get('operator','>'),float(cd.get('value',0))
                sv=s.get(ind,s.get('changePct',0))
                try:
                    if op=='>' and not(float(sv)>val): ok=False; break
                    elif op=='<' and not(float(sv)<val): ok=False; break
                except: ok=False; break
            if ok: matches.append(s)
        return jsonify(safe_dict({'success':True,'matches':matches[:50],'totalMatches':len(matches)}))
    except Exception as e:
        return jsonify({'error':str(e)}),500

# ---- PORTFOLIO / ALERTS / WATCHLIST (DB-backed, user-aware) ----
@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    try:
        uid = request.args.get('userId', request.args.get('user', ''))
        if not uid:
            return jsonify(safe_dict({'success':True,'positions':[],'summary':{'totalValue':0,'totalCost':0,'totalPnL':0,'totalPnLPct':0,'positionCount':0},'needsLogin':True}))
        db = get_db()
        rows = db.execute("SELECT * FROM portfolios WHERE user_id=?", (uid,)).fetchall()
        db.close()
        pos=[]; tv=tc=0
        for r in rows:
            cd=_cget(_stock_cache, r['symbol'])
            if not cd: continue
            q,ac,cp=r['quantity'],r['avg_cost'],cd['price']
            mv=cp*q; cb=ac*q; upnl=mv-cb; tv+=mv; tc+=cb
            pos.append({'id':r['id'],'symbol':r['symbol'],'name':cd['name'],'quantity':q,'avgCost':sf(ac),'currentPrice':sf(cp),'marketValue':sf(mv),'costBasis':sf(cb),'unrealizedPnL':sf(upnl),'unrealizedPnLPct':sf(upnl/cb*100 if cb else 0),'changePct':cd.get('changePct',0),'weight':0})
        for p in pos: p['weight']=sf(float(p['marketValue'])/tv*100 if tv>0 else 0)
        tp=tv-tc
        dp=sum(float(p['marketValue'])*p['changePct']/100 for p in pos)
        return jsonify(safe_dict({'success':True,'positions':pos,'summary':{'totalValue':sf(tv),'totalCost':sf(tc),'totalPnL':sf(tp),'totalPnLPct':sf(tp/tc*100 if tc else 0),'dailyPnL':sf(dp),'positionCount':len(pos)}}))
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/portfolio', methods=['POST'])
def add_portfolio():
    try:
        d=request.json or {}
        uid=d.get('userId',d.get('user',''))
        sym=d.get('symbol','').upper(); qty=float(d.get('quantity',0)); ac=float(d.get('avgCost',0))
        if not uid: return jsonify({'error':'Giris yapmaniz gerekli'}),401
        if not sym or qty<=0 or ac<=0: return jsonify({'error':'Gecersiz veri'}),400
        db = get_db()
        existing = db.execute("SELECT * FROM portfolios WHERE user_id=? AND symbol=?", (uid, sym)).fetchone()
        if existing:
            new_qty = existing['quantity'] + qty
            new_avg = (existing['avg_cost'] * existing['quantity'] + ac * qty) / new_qty
            db.execute("UPDATE portfolios SET quantity=?, avg_cost=? WHERE id=?", (new_qty, new_avg, existing['id']))
        else:
            db.execute("INSERT INTO portfolios (user_id, symbol, quantity, avg_cost) VALUES (?, ?, ?, ?)", (uid, sym, qty, ac))
        db.commit(); db.close()
        return jsonify({'success':True,'message':f'{sym} portfoye eklendi'})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/portfolio', methods=['DELETE'])
def del_portfolio():
    try:
        d=request.json or {}
        uid=d.get('userId',d.get('user',''))
        sym=d.get('symbol','').upper()
        pid=d.get('id')
        db = get_db()
        if pid:
            db.execute("DELETE FROM portfolios WHERE id=?", (pid,))
        elif uid and sym:
            db.execute("DELETE FROM portfolios WHERE user_id=? AND symbol=?", (uid, sym))
        db.commit(); db.close()
        return jsonify({'success':True})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/portfolio/risk')
def portfolio_risk():
    return jsonify({'success':True,'risk':{'message':'Risk analizi yukleniyor'}})

@app.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify(safe_dict({'success':True,'watchlist':[],'symbols':[],'needsLogin':True}))
        db = get_db()
        rows = db.execute("SELECT symbol FROM watchlists WHERE user_id=?", (uid,)).fetchall()
        db.close()
        symbols = [r['symbol'] for r in rows]
        stocks = _get_stocks(symbols) if symbols else []
        return jsonify(safe_dict({'success':True,'watchlist':stocks,'symbols':symbols}))
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/watchlist', methods=['POST'])
def update_watchlist():
    try:
        d=request.json or {}
        uid=d.get('userId','')
        sym=d.get('symbol','').upper()
        action=d.get('action','add')
        if not uid: return jsonify({'error':'Giris yapmaniz gerekli'}),401
        if not sym: return jsonify({'error':'Hisse kodu gerekli'}),400
        db = get_db()
        if action == 'add':
            try:
                db.execute("INSERT INTO watchlists (user_id, symbol) VALUES (?, ?)", (uid, sym))
            except sqlite3.IntegrityError:
                pass
        elif action == 'remove':
            db.execute("DELETE FROM watchlists WHERE user_id=? AND symbol=?", (uid, sym))
        elif action == 'toggle':
            existing = db.execute("SELECT id FROM watchlists WHERE user_id=? AND symbol=?", (uid, sym)).fetchone()
            if existing:
                db.execute("DELETE FROM watchlists WHERE id=?", (existing['id'],))
                action = 'removed'
            else:
                db.execute("INSERT INTO watchlists (user_id, symbol) VALUES (?, ?)", (uid, sym))
                action = 'added'
        db.commit(); db.close()
        return jsonify({'success':True,'action':action,'symbol':sym})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify(safe_dict({'success':True,'alerts':[],'needsLogin':True}))
        db = get_db()
        rows = db.execute("SELECT * FROM alerts WHERE user_id=? ORDER BY created_at DESC", (uid,)).fetchall()
        db.close()
        alerts = [{
            'id':r['id'],'symbol':r['symbol'],'condition':r['condition'],
            'targetValue':r['target_value'],'active':bool(r['active']),
            'triggered':bool(r['triggered']),'triggeredAt':r['triggered_at'],
            'createdAt':r['created_at'],
        } for r in rows]
        return jsonify(safe_dict({'success':True,'alerts':alerts}))
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/alerts', methods=['POST'])
def add_alert():
    try:
        d=request.json or {}
        uid=d.get('userId','')
        sym=d.get('symbol','').upper()
        condition=d.get('condition','price_above')
        target=float(d.get('targetValue',d.get('threshold',0)))
        if not uid: return jsonify({'error':'Giris yapmaniz gerekli'}),401
        if not sym or target<=0: return jsonify({'error':'Gecersiz veri'}),400
        db = get_db()
        db.execute("INSERT INTO alerts (user_id, symbol, condition, target_value) VALUES (?, ?, ?, ?)",
                   (uid, sym, condition, target))
        db.commit(); db.close()
        return jsonify({'success':True,'message':f'{sym} icin uyari eklendi'})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/alerts/<int:aid>', methods=['DELETE'])
def del_alert(aid):
    try:
        db = get_db()
        db.execute("DELETE FROM alerts WHERE id=?", (aid,))
        db.commit(); db.close()
        return jsonify({'success':True})
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/sectors')
def sectors():
    try:
        sd=[]
        for sn,syms in SECTOR_MAP.items():
            stocks=_get_stocks(syms); changes=[s['changePct'] for s in stocks if 'changePct' in s]
            sd.append({'name':sn,'stockCount':len(syms),'avgChange':sf(np.mean(changes)) if changes else 0,'symbols':syms})
        sd.sort(key=lambda x:x['avgChange'],reverse=True)
        return jsonify(safe_dict({'success':True,'sectors':sd}))
    except Exception as e: return jsonify({'error':str(e)}),500

@app.route('/api/search')
def search():
    q=request.args.get('q','').upper()
    return jsonify({'success':True,'results':[{'code':c,'name':n} for c,n in BIST100_STOCKS.items() if q in c or q in n.upper()][:10]}) if q else jsonify({'results':[]})

@app.route('/api/backtest', methods=['POST'])
def backtest():
    try:
        d = request.json or {}
        sym = d.get('symbol', '').upper()
        strategy = d.get('strategy', 'ma_cross')
        params = d.get('params', {})
        period = d.get('period', '1y')
        commission = float(d.get('commission', 0.001))
        initial_capital = float(d.get('initialCapital', 100000))

        if not sym:
            return jsonify({'error': 'Hisse kodu gerekli'}), 400

        hist = _fetch_hist_df(sym, period)
        if hist is None or len(hist) < 30:
            return jsonify({'error': f'{sym} icin yeterli veri yok'}), 400

        closes = hist['Close'].values.astype(float)
        highs = hist['High'].values.astype(float)
        lows = hist['Low'].values.astype(float)
        dates = [d.strftime('%Y-%m-%d') for d in hist.index]
        n = len(closes)

        # Sinyal uret
        signals = [0] * n  # 1=al, -1=sat, 0=bekle

        if strategy == 'ma_cross':
            fast_p = int(params.get('fast', 20))
            slow_p = int(params.get('slow', 50))
            s = pd.Series(closes)
            fast_ma = s.rolling(fast_p).mean().values
            slow_ma = s.rolling(slow_p).mean().values
            for i in range(slow_p + 1, n):
                if fast_ma[i] > slow_ma[i] and fast_ma[i-1] <= slow_ma[i-1]:
                    signals[i] = 1
                elif fast_ma[i] < slow_ma[i] and fast_ma[i-1] >= slow_ma[i-1]:
                    signals[i] = -1

        elif strategy == 'breakout':
            lookback = int(params.get('lookback', 20))
            for i in range(lookback, n):
                high_n = max(highs[i-lookback:i])
                low_n = min(lows[i-lookback:i])
                if closes[i] > high_n:
                    signals[i] = 1
                elif closes[i] < low_n:
                    signals[i] = -1

        elif strategy == 'mean_reversion':
            rsi_low = float(params.get('rsi_low', 30))
            rsi_high = float(params.get('rsi_high', 70))
            for i in range(15, n):
                rsi = calc_rsi_single(closes[:i+1])
                if rsi is not None:
                    if rsi < rsi_low: signals[i] = 1
                    elif rsi > rsi_high: signals[i] = -1

        # Backtest calistir
        cash = initial_capital
        shares = 0
        equity_curve = []
        trades = []
        peak_equity = initial_capital
        max_dd = 0
        wins = 0
        losses = 0

        for i in range(n):
            price = closes[i]
            if signals[i] == 1 and shares == 0:
                shares = int(cash * (1 - commission) / price)
                if shares > 0:
                    cost = shares * price * (1 + commission)
                    cash -= cost
                    trades.append({'date': dates[i], 'action': 'AL', 'price': sf(price), 'shares': shares, 'pnl': 0})
            elif signals[i] == -1 and shares > 0:
                revenue = shares * price * (1 - commission)
                pnl = revenue - (trades[-1]['shares'] * trades[-1]['price'] * (1 + commission)) if trades else 0
                cash += revenue
                trades.append({'date': dates[i], 'action': 'SAT', 'price': sf(price), 'shares': shares, 'pnl': sf(pnl)})
                if pnl > 0: wins += 1
                else: losses += 1
                shares = 0

            equity = cash + shares * price
            equity_curve.append({'date': dates[i], 'equity': sf(equity)})
            if equity > peak_equity: peak_equity = equity
            dd = (peak_equity - equity) / peak_equity * 100
            if dd > max_dd: max_dd = dd

        final_equity = cash + shares * closes[-1]
        total_return = sf(((final_equity - initial_capital) / initial_capital) * 100)
        bh_return = sf(((closes[-1] - closes[0]) / closes[0]) * 100)
        years = n / 252
        cagr = sf(((final_equity / initial_capital) ** (1 / years) - 1) * 100) if years > 0 else 0

        daily_returns = np.diff([e['equity'] for e in equity_curve]) / np.array([e['equity'] for e in equity_curve[:-1]])
        sharpe = sf(float(np.mean(daily_returns) / np.std(daily_returns) * (252**0.5))) if len(daily_returns) > 0 and np.std(daily_returns) > 0 else 0

        total_trades = wins + losses
        win_rate = sf(wins / total_trades * 100) if total_trades > 0 else 0
        alpha = sf(float(total_return) - float(bh_return))

        # Exposure (piyasada kalma orani)
        in_market_days = sum(1 for i in range(n) if (any(t['action'] == 'AL' and dates.index(t['date']) <= i for t in trades if t['date'] in dates)))
        exposure_pct = sf(in_market_days / n * 100) if n > 0 else 0

        # Ortalama trade getirisi
        trade_pnls = [float(t['pnl']) for t in trades if t['action'] == 'SAT' and t['pnl'] != 0]
        avg_trade = sf(np.mean(trade_pnls)) if trade_pnls else 0
        avg_win = sf(np.mean([p for p in trade_pnls if p > 0])) if [p for p in trade_pnls if p > 0] else 0
        avg_loss = sf(np.mean([p for p in trade_pnls if p < 0])) if [p for p in trade_pnls if p < 0] else 0
        profit_factor = sf(abs(sum(p for p in trade_pnls if p > 0) / sum(p for p in trade_pnls if p < 0))) if any(p < 0 for p in trade_pnls) else 999

        return jsonify(safe_dict({
            'success': True,
            'results': {
                'totalReturn': total_return, 'cagr': cagr, 'sharpeRatio': sharpe,
                'maxDrawdown': sf(-max_dd), 'winRate': win_rate, 'totalTrades': total_trades,
                'buyAndHoldReturn': bh_return, 'alpha': alpha,
                'finalEquity': sf(final_equity), 'initialCapital': sf(initial_capital),
                'exposure': exposure_pct,
                'avgTrade': avg_trade,
                'avgWin': avg_win,
                'avgLoss': avg_loss,
                'profitFactor': profit_factor,
                'commission': sf(commission * 100),
            },
            'equityCurve': equity_curve[::max(1, len(equity_curve)//200)],
            'trades': trades[-50:],
            'warnings': [
                'Hayatta kalma yanliligi (Survivorship Bias): Bu backtest sadece bugun BIST100 endeksinde bulunan hisseleri kapsamaktadir. Gecmiste endeksten cikarilmis (iflas, birlesme, borsadan cikarma) hisseler dahil degildir. Gercek performans bu sonuclardan daha dusuk olabilir.',
                'Kayma (Slippage): Gercek islemlerde emir fiyati ile gerceklesen fiyat arasinda fark olabilir. Ozellikle dusuk hacimli hisselerde bu etki belirgindir.',
                'Komisyon: Backtest %' + str(sf(commission * 100)) + ' komisyon varsayimi kullanmaktadir.',
            ],
        }))
    except Exception as e:
        print(f"[BACKTEST] Hata: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

# =====================================================================
# ISI HARITASI (HEATMAP)
# =====================================================================
@app.route('/api/heatmap')
def heatmap():
    """Sektor bazli isi haritasi verisi"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'sectors': [], 'loading': True})

        stock_map = {s['code']: s for s in stocks}
        sectors = []

        for sector_name, symbols in SECTOR_MAP.items():
            sector_stocks = []
            total_change = 0
            count = 0
            for sym in symbols:
                if sym in stock_map:
                    s = stock_map[sym]
                    sector_stocks.append({
                        'code': s['code'], 'name': s['name'],
                        'price': s['price'], 'changePct': s['changePct'],
                        'volume': s['volume'],
                    })
                    total_change += s['changePct']
                    count += 1

            avg_change = sf(total_change / count) if count > 0 else 0
            sectors.append({
                'name': sector_name,
                'displayName': {
                    'bankacilik': 'Bankacilik', 'havacilik': 'Havacilik',
                    'otomotiv': 'Otomotiv', 'enerji': 'Enerji',
                    'holding': 'Holding', 'perakende': 'Perakende',
                    'teknoloji': 'Teknoloji', 'telekom': 'Telekom',
                    'demir_celik': 'Demir Celik', 'gida': 'Gida',
                    'insaat': 'Insaat', 'gayrimenkul': 'Gayrimenkul',
                }.get(sector_name, sector_name),
                'avgChange': avg_change,
                'stockCount': count,
                'stocks': sorted(sector_stocks, key=lambda x: x['changePct'], reverse=True),
            })

        sectors.sort(key=lambda x: x['avgChange'], reverse=True)
        return jsonify(safe_dict({'success': True, 'sectors': sectors}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# GUNLUK RAPOR
# =====================================================================
@app.route('/api/report')
def daily_report():
    """Gunluk piyasa raporu"""
    try:
        stocks = _get_stocks()
        indices = _get_indices()

        if not stocks:
            return jsonify({'success': True, 'loading': True, 'message': 'Veriler yukleniyor'})

        # Piyasa genisligi
        up = [s for s in stocks if s.get('changePct', 0) > 0]
        down = [s for s in stocks if s.get('changePct', 0) < 0]
        unchanged = [s for s in stocks if s.get('changePct', 0) == 0]

        # En cok artanlar / azalanlar
        sorted_up = sorted(stocks, key=lambda x: x.get('changePct', 0), reverse=True)
        sorted_down = sorted(stocks, key=lambda x: x.get('changePct', 0))

        # Hacim liderleri
        sorted_vol = sorted(stocks, key=lambda x: x.get('volume', 0), reverse=True)

        # Gap analizi
        gap_up = sorted([s for s in stocks if s.get('gapPct', 0) > 1], key=lambda x: x.get('gapPct', 0), reverse=True)
        gap_down = sorted([s for s in stocks if s.get('gapPct', 0) < -1], key=lambda x: x.get('gapPct', 0))

        # Ortalama degisim
        all_changes = [s.get('changePct', 0) for s in stocks]
        avg_change = sf(np.mean(all_changes)) if all_changes else 0

        # Sektor performansi
        sector_perf = []
        stock_map = {s['code']: s for s in stocks}
        for sname, syms in SECTOR_MAP.items():
            changes = [stock_map[s]['changePct'] for s in syms if s in stock_map]
            if changes:
                sector_perf.append({
                    'name': sname, 'avgChange': sf(np.mean(changes)),
                    'bestStock': max([(s, stock_map[s]['changePct']) for s in syms if s in stock_map], key=lambda x: x[1])[0] if changes else '',
                })
        sector_perf.sort(key=lambda x: x['avgChange'], reverse=True)

        # Rapor metni olustur
        report_lines = []
        bist100 = indices.get('XU100', {})
        if bist100:
            direction = 'yukselis' if bist100.get('changePct', 0) > 0 else 'dusus'
            report_lines.append(f"BIST 100 endeksi %{bist100.get('changePct', 0)} {direction} gosteriyor.")

        report_lines.append(f"Toplam {len(stocks)} hisseden {len(up)} yukselen, {len(down)} dusen, {len(unchanged)} degismez.")
        report_lines.append(f"Piyasa ortalama degisimi: %{avg_change}")

        if sorted_up:
            report_lines.append(f"Gunun yildizi: {sorted_up[0]['code']} (%{sorted_up[0]['changePct']})")
        if sorted_down:
            report_lines.append(f"Gunun kaybi: {sorted_down[0]['code']} (%{sorted_down[0]['changePct']})")

        if sector_perf:
            report_lines.append(f"En iyi sektor: {sector_perf[0]['name']} (%{sector_perf[0]['avgChange']})")

        return jsonify(safe_dict({
            'success': True,
            'date': datetime.now().strftime('%Y-%m-%d'),
            'summary': ' '.join(report_lines),
            'reportLines': report_lines,
            'marketBreadth': {
                'advancing': len(up), 'declining': len(down),
                'unchanged': len(unchanged), 'total': len(stocks),
                'avgChange': avg_change,
            },
            'topGainers': [{'code': s['code'], 'name': s['name'], 'changePct': s['changePct'], 'price': s['price']} for s in sorted_up[:5]],
            'topLosers': [{'code': s['code'], 'name': s['name'], 'changePct': s['changePct'], 'price': s['price']} for s in sorted_down[:5]],
            'volumeLeaders': [{'code': s['code'], 'name': s['name'], 'volume': s['volume'], 'changePct': s['changePct']} for s in sorted_vol[:5]],
            'gapUp': [{'code': s['code'], 'gapPct': s.get('gapPct', 0)} for s in gap_up[:5]],
            'gapDown': [{'code': s['code'], 'gapPct': s.get('gapPct', 0)} for s in gap_down[:5]],
            'sectorPerformance': sector_perf,
            'indices': indices,
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# SINYAL TARAMA (Tum hisseleri tarar, guclu AL/SAT sinyallerini bulur)
# =====================================================================
_signals_cache = {'data': None, 'ts': 0}
_signals_cache_lock = threading.Lock()
_opps_cache = {'data': None, 'ts': 0}
_opps_cache_lock = threading.Lock()
COMPUTED_CACHE_TTL = 120  # 2 dakika - agir hesaplamalar icin

def _compute_signal_for_stock(stock, timeframe):
    """Tek hisse icin sinyal hesapla (thread-safe, paralel calisir)"""
    sym = stock['code']
    try:
        hist = _cget_hist(f"{sym}_1y")
        if hist is None:
            return None

        c = hist['Close'].values.astype(float)
        cp = float(c[-1])

        ind = calc_all_indicators(hist, cp)
        summary = ind.get('summary', {})
        buy_count = summary.get('buySignals', 0)
        sell_count = summary.get('sellSignals', 0)
        total_ind = summary.get('totalIndicators', 1)

        rec = calc_recommendation(hist, ind)
        tf_rec = rec.get(timeframe, {})
        action = tf_rec.get('action', 'NOTR')
        score = float(tf_rec.get('score', 0))
        confidence = float(tf_rec.get('confidence', 0))

        consensus_pct = (buy_count / total_ind * 100) if total_ind > 0 else 50
        if score < 0:
            consensus_pct = 100 - consensus_pct
        composite = min(100, (abs(score) / 12 * 60) + (consensus_pct * 0.4))

        rsi_val = ind.get('rsi', {}).get('value', 50)
        macd_type = ind.get('macd', {}).get('signalType', 'neutral')
        macd_hist = ind.get('macd', {}).get('histogram', 0)

        sr = calc_support_resistance(hist)
        supports = sr.get('supports', [])[:2]
        resistances = sr.get('resistances', [])[:2]
        stop_loss = supports[0] if supports else sf(cp * 0.95)
        target = resistances[0] if resistances else sf(cp * 1.10)

        sig_type = 'buy' if score > 0 else 'sell'
        ml_conf = calc_ml_confidence(hist, ind, score, sig_type)
        candle_patterns = ind.get('candlestick', {}).get('patterns', [])

        return {
            'code': sym,
            'name': BIST100_STOCKS.get(sym, sym),
            'price': sf(cp),
            'changePct': stock.get('changePct', 0),
            'volume': stock.get('volume', 0),
            'action': action,
            'score': sf(score),
            'confidence': sf(confidence),
            'composite': sf(composite),
            'mlConfidence': ml_conf.get('confidence', 50),
            'mlGrade': ml_conf.get('grade', 'C'),
            'buySignals': buy_count,
            'sellSignals': sell_count,
            'totalIndicators': total_ind,
            'rsi': sf(rsi_val),
            'macdSignal': macd_type,
            'macdHistogram': macd_hist,
            'stopLoss': stop_loss,
            'target': target,
            'supports': supports,
            'resistances': resistances,
            'reasons': tf_rec.get('reasons', [])[:5],
            'reason': tf_rec.get('reason', ''),
            'indicatorBreakdown': tf_rec.get('indicatorBreakdown', {}),
            'strategy': tf_rec.get('strategy', ''),
            'candlestickPatterns': candle_patterns[:3],
            'dynamicThresholds': ind.get('dynamicThresholds', {}),
            'tradePlan': calc_trade_plan(hist, ind),
        }
    except:
        return None

@app.route('/api/signals')
def signal_scanner():
    """Tum hisselerin sinyal taramasi - composite score ile sirali (PARALEL)"""
    try:
        timeframe = request.args.get('timeframe', 'weekly')
        min_score = float(request.args.get('minScore', 0))
        signal_type = request.args.get('type', 'all')

        # Computed cache kontrol (2 dk)
        with _signals_cache_lock:
            sc = _signals_cache
            if sc['data'] and (time.time() - sc['ts']) < COMPUTED_CACHE_TTL:
                cached = sc['data']
                # Client-side filtreleme uygula
                filtered = cached['all_results']
                if signal_type == 'buy':
                    filtered = [r for r in filtered if float(r['score']) > 0]
                elif signal_type == 'sell':
                    filtered = [r for r in filtered if float(r['score']) < 0]
                if min_score > 0:
                    filtered = [r for r in filtered if abs(float(r['score'])) >= min_score]
                return jsonify(safe_dict({
                    'success': True, 'timeframe': timeframe,
                    'totalScanned': cached['totalScanned'],
                    'signalCount': len(filtered), 'signals': filtered,
                    'marketRegime': calc_market_regime(),
                    'timestamp': cached['timestamp'], 'meta': _api_meta(),
                }))

        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True, 'signals': [], 'message': 'Veriler yukleniyor...'})

        hist_ready = sum(1 for s in stocks if _cget_hist(f"{s['code']}_1y") is not None)
        if hist_ready < 10:
            return jsonify({'success': True, 'loading': True, 'signals': [], 'message': f'Tarihsel veriler hazirlaniyor ({hist_ready}/{len(stocks)})...'})

        # PARALEL sinyal hesaplama - ThreadPoolExecutor ile ~5x hiz
        results = []
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_compute_signal_for_stock, s, timeframe): s for s in stocks}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

        results.sort(key=lambda x: float(x['score']), reverse=True)

        # Cache'e kaydet (filtresiz tam liste)
        with _signals_cache_lock:
            _signals_cache['data'] = {
                'all_results': results,
                'totalScanned': len(stocks),
                'timestamp': datetime.now().isoformat(),
            }
            _signals_cache['ts'] = time.time()

        # Filtrele
        filtered = results
        if signal_type == 'buy':
            filtered = [r for r in results if float(r['score']) > 0]
        elif signal_type == 'sell':
            filtered = [r for r in results if float(r['score']) < 0]
        if min_score > 0:
            filtered = [r for r in filtered if abs(float(r['score'])) >= min_score]

        return jsonify(safe_dict({
            'success': True, 'timeframe': timeframe,
            'totalScanned': len(stocks),
            'signalCount': len(filtered), 'signals': filtered,
            'marketRegime': calc_market_regime(),
            'timestamp': datetime.now().isoformat(), 'meta': _api_meta(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# FIRSAT OZETI (Gunluk/Haftalik/Aylik/Yillik)
# =====================================================================
def _compute_opportunity_for_stock(stock):
    """Tek hisse icin firsat analizi (thread-safe, paralel calisir)"""
    sym = stock['code']
    try:
        hist = _cget_hist(f"{sym}_1y")
        if hist is None:
            return None

        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        cp = float(c[-1])
        n = len(c)

        events = []
        event_score = 0

        rsi_val = calc_rsi(c).get('value', 50)
        if rsi_val < 30:
            events.append({'type': 'rsi_oversold', 'text': f'RSI {sf(rsi_val)} - Asiri satim bolgesinde, toparlanma bekleniyor', 'impact': 'positive'})
            event_score += 3
        elif rsi_val > 70:
            events.append({'type': 'rsi_overbought', 'text': f'RSI {sf(rsi_val)} - Asiri alim bolgesinde, duzeltme gelebilir', 'impact': 'negative'})
            event_score -= 3

        macd = calc_macd(c)
        if macd.get('signalType') == 'buy':
            events.append({'type': 'macd_cross', 'text': 'MACD alis kesisimi - Yukari momentum basladi', 'impact': 'positive'})
            event_score += 2
        elif macd.get('signalType') == 'sell':
            events.append({'type': 'macd_cross', 'text': 'MACD satis kesisimi - Asagi momentum basladi', 'impact': 'negative'})
            event_score -= 2

        if n >= 200:
            ema50 = pd.Series(c).ewm(span=50).mean().values
            ema200 = pd.Series(c).ewm(span=200).mean().values
            if ema50[-1] > ema200[-1] and ema50[-2] <= ema200[-2]:
                events.append({'type': 'golden_cross', 'text': 'ALTIN KESISIM! EMA50 > EMA200 - Guclu uzun vadeli alis sinyali', 'impact': 'very_positive'})
                event_score += 5
            elif ema50[-1] < ema200[-1] and ema50[-2] >= ema200[-2]:
                events.append({'type': 'death_cross', 'text': 'OLUM KESISIMI! EMA50 < EMA200 - Guclu uzun vadeli satis sinyali', 'impact': 'very_negative'})
                event_score -= 5

        if n >= 20:
            vol_avg = np.mean(v[-20:])
            vol_today = v[-1]
            if vol_avg > 0 and vol_today > vol_avg * 2:
                ratio = sf(vol_today / vol_avg)
                direction = 'yukselis' if c[-1] > c[-2] else 'dusus'
                impact = 'positive' if c[-1] > c[-2] else 'negative'
                events.append({'type': 'volume_spike', 'text': f'Hacim patlamasi ({ratio}x ortalama) + {direction} hareketi', 'impact': impact})
                event_score += 2 if c[-1] > c[-2] else -2

        bb = calc_bollinger(c, cp)
        if bb.get('lower', 0) > 0 and cp < bb['lower']:
            events.append({'type': 'bb_break_lower', 'text': f'Fiyat alt Bollinger bandinin altinda ({sf(bb["lower"])}) - Toparlanma bekleniyor', 'impact': 'positive'})
            event_score += 2
        elif bb.get('upper', 0) > 0 and cp > bb['upper']:
            events.append({'type': 'bb_break_upper', 'text': f'Fiyat ust Bollinger bandini asti ({sf(bb["upper"])}) - Asiri alim', 'impact': 'negative'})
            event_score -= 1

        w52 = calc_52w(hist)
        w52_pos = w52.get('position', 50)
        if w52_pos < 10:
            events.append({'type': '52w_low', 'text': f'52 haftalik dibin %{sf(w52_pos)} uzerinde - Tarihi dip bolgesi', 'impact': 'positive'})
            event_score += 2
        elif w52_pos > 90:
            events.append({'type': '52w_high', 'text': f'52 haftalik zirveye %{sf(100-w52_pos)} mesafede', 'impact': 'neutral'})

        sr = calc_support_resistance(hist)
        if sr.get('resistances'):
            nearest_res = sr['resistances'][0]
            if cp > nearest_res * 0.99 and c[-2] < nearest_res:
                events.append({'type': 'resistance_break', 'text': f'Direnc kirdi ({sf(nearest_res)} TL) - Yukari kirilim', 'impact': 'positive'})
                event_score += 3
        if sr.get('supports'):
            nearest_sup = sr['supports'][0]
            if cp < nearest_sup * 1.01 and c[-2] > nearest_sup:
                events.append({'type': 'support_break', 'text': f'Destek kirildi ({sf(nearest_sup)} TL) - Asagi kirilim', 'impact': 'negative'})
                event_score -= 3

        returns = {}
        for label, days in [('daily', 1), ('weekly', 5), ('monthly', 22), ('yearly', 252)]:
            # Esnek index: yeterli veri yoksa mevcut verinin basindan hesapla
            actual_days = min(days, n - 1)
            if actual_days > 0:
                ret = ((c[-1] - c[-1-actual_days]) / c[-1-actual_days]) * 100
                returns[label] = sf(ret)
            else:
                returns[label] = 0

        o_arr = hist['Open'].values.astype(float) if 'Open' in hist.columns else c.copy()
        candles = calc_candlestick_patterns(o_arr, h, l, c)
        for p in candles.get('patterns', []):
            impact = 'positive' if p['type'] == 'bullish' else ('negative' if p['type'] == 'bearish' else 'neutral')
            events.append({'type': 'candlestick', 'text': f"Mum Formasyonu: {p['name']} - {p['description']}", 'impact': impact})
            event_score += p['strength'] if p['type'] == 'bullish' else -p['strength']

        dyn = calc_dynamic_thresholds(c, h, l, v)

        if events:
            return {
                'code': sym,
                'name': BIST100_STOCKS.get(sym, sym),
                'price': sf(cp),
                'changePct': stock.get('changePct', 0),
                'eventScore': event_score,
                'events': events,
                'eventCount': len(events),
                'returns': returns,
                'rsi': sf(rsi_val),
                'dynamicThresholds': dyn,
                'candlestickPatterns': candles.get('patterns', []),
                'tradePlan': calc_trade_plan(hist),
            }
        return None
    except:
        return None

@app.route('/api/opportunities')
def opportunities():
    """Coklu zaman dilimli firsat raporu - PARALEL hesaplama"""
    try:
        # Computed cache kontrol (2 dk)
        with _opps_cache_lock:
            oc = _opps_cache
            if oc['data'] and (time.time() - oc['ts']) < COMPUTED_CACHE_TTL:
                return jsonify(safe_dict(oc['data']))

        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True, 'message': 'Veriler yukleniyor...'})

        hist_ready = sum(1 for s in stocks if _cget_hist(f"{s['code']}_1y") is not None)
        if hist_ready < 10:
            return jsonify({'success': True, 'loading': True, 'message': f'Tarihsel veriler hazirlaniyor ({hist_ready}/{len(stocks)})...'})

        # PARALEL firsat hesaplama
        opportunities_list = []
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_compute_opportunity_for_stock, s): s for s in stocks}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    opportunities_list.append(result)

        opportunities_list.sort(key=lambda x: abs(x['eventScore']), reverse=True)
        buy_opps = [o for o in opportunities_list if o['eventScore'] > 0]
        sell_opps = [o for o in opportunities_list if o['eventScore'] < 0]

        result_data = {
            'success': True,
            'totalScanned': len(stocks),
            'buyOpportunities': buy_opps[:20],
            'sellOpportunities': sell_opps[:20],
            'marketRegime': calc_market_regime(),
            'timestamp': datetime.now().isoformat(),
            'meta': _api_meta(),
        }

        with _opps_cache_lock:
            _opps_cache['data'] = result_data
            _opps_cache['ts'] = time.time()

        return jsonify(safe_dict(result_data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# CANLI STRATEJI SIMULASYONU
# =====================================================================
_strat_cache = {'data': None, 'ts': 0}
_strat_cache_lock = threading.Lock()

def _compute_strategy_for_stock(stock):
    """Tek hisse icin 3 strateji hesapla (thread-safe)"""
    sym = stock['code']
    try:
        hist = _cget_hist(f"{sym}_1y")
        if hist is None:
            return None

        c = hist['Close'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
        cp = float(c[-1])
        n = len(c)
        result = {'ma_cross': None, 'breakout': None, 'mean_reversion': None}

        if n >= 50:
            s = pd.Series(c)
            ema20 = s.ewm(span=20).mean().values
            ema50 = s.ewm(span=50).mean().values
            signal = None
            if ema20[-1] > ema50[-1] and ema20[-2] <= ema50[-2]: signal = 'AL'
            elif ema20[-1] < ema50[-1] and ema20[-2] >= ema50[-2]: signal = 'SAT'
            elif ema20[-1] > ema50[-1]: signal = 'ALIS POZISYONUNDA'
            elif ema20[-1] < ema50[-1]: signal = 'SATIS POZISYONUNDA'
            distance = sf(((ema20[-1] - ema50[-1]) / ema50[-1]) * 100)
            result['ma_cross'] = {
                'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                'price': sf(cp), 'changePct': stock.get('changePct', 0),
                'signal': signal, 'ema20': sf(ema20[-1]), 'ema50': sf(ema50[-1]),
                'distance': distance, 'freshSignal': signal in ('AL', 'SAT'),
            }

        if n >= 20:
            high_20 = float(np.max(h[-20:]))
            low_20 = float(np.min(l[-20:]))
            signal = None
            if cp >= high_20 * 0.99: signal = 'YUKARI KIRILIM'
            elif cp <= low_20 * 1.01: signal = 'ASAGI KIRILIM'
            else:
                pos = ((cp - low_20) / (high_20 - low_20) * 100) if high_20 != low_20 else 50
                signal = f'BANT ICINDE (%{sf(pos)})'
            result['breakout'] = {
                'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                'price': sf(cp), 'changePct': stock.get('changePct', 0),
                'signal': signal, 'high20': sf(high_20), 'low20': sf(low_20),
                'freshSignal': 'KIRILIM' in (signal or ''),
            }

        if n >= 15:
            rsi = calc_rsi(c).get('value', 50)
            signal = None
            if rsi < 30: signal = 'ASIRI SATIM → AL'
            elif rsi < 40: signal = 'SATIM BOLGESI → ALIS FIRSATI'
            elif rsi > 70: signal = 'ASIRI ALIM → SAT'
            elif rsi > 60: signal = 'ALIM BOLGESI → DIKKAT'
            else: signal = 'NOTR BOLGE'
            result['mean_reversion'] = {
                'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                'price': sf(cp), 'changePct': stock.get('changePct', 0),
                'signal': signal, 'rsi': sf(rsi), 'freshSignal': rsi < 30 or rsi > 70,
            }

        return result
    except:
        return None

@app.route('/api/strategies/live')
def live_strategies():
    """3 stratejiyi tum hisselere canli uygular (PARALEL)"""
    try:
        # Computed cache kontrol (2 dk)
        with _strat_cache_lock:
            sc = _strat_cache
            if sc['data'] and (time.time() - sc['ts']) < COMPUTED_CACHE_TTL:
                return jsonify(safe_dict(sc['data']))

        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True, 'message': 'Veriler yukleniyor...'})

        results = {'ma_cross': [], 'breakout': [], 'mean_reversion': []}
        strategy_names = {
            'ma_cross': 'Hareketli Ortalama Kesisimi',
            'breakout': 'Kirilim Stratejisi',
            'mean_reversion': 'Ortalamaya Donus (RSI)',
        }

        # PARALEL strateji hesaplama
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
            futures = {executor.submit(_compute_strategy_for_stock, s): s for s in stocks}
            for future in as_completed(futures):
                r = future.result()
                if r:
                    for key in ('ma_cross', 'breakout', 'mean_reversion'):
                        if r[key]:
                            results[key].append(r[key])

        for key in results:
            results[key].sort(key=lambda x: (not x.get('freshSignal', False), -abs(x.get('changePct', 0))))

        result_data = {
            'success': True,
            'strategies': results,
            'strategyNames': strategy_names,
            'totalStocks': len(stocks),
            'timestamp': datetime.now().isoformat(),
        }

        with _strat_cache_lock:
            _strat_cache['data'] = result_data
            _strat_cache['ts'] = time.time()

        return jsonify(safe_dict(result_data))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# TEMETTU TAKVIMLERI
# =====================================================================
@app.route('/api/dividends')
def dividend_calendar():
    """BIST hisselerinin temettu bilgileri (yfinance dividends)"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True, 'dividends': [], 'message': 'Veriler yukleniyor...'})

        # Eksik tarihsel verileri paralel cek
        # Sadece cache'deki veriyi kullan - HTTP fetch YAPMA
        dividends_list = []
        for stock in stocks:
            sym = stock['code']
            try:
                ticker_sym = sym + '.IS'
                hist = _cget_hist(f"{sym}_1y")

                # yfinance Ticker ile temettu bilgisi
                if YF_OK:
                    tkr = yf.Ticker(ticker_sym)
                    divs = tkr.dividends
                    if divs is not None and len(divs) > 0:
                        cp = stock.get('price', 0)
                        # Son 3 yilin temettuleri
                        recent_divs = []
                        total_div_1y = 0
                        one_year_ago = datetime.now() - timedelta(days=365)
                        for dt, amt in divs.items():
                            div_date = dt.to_pydatetime().replace(tzinfo=None) if hasattr(dt, 'to_pydatetime') else dt
                            recent_divs.append({
                                'date': div_date.strftime('%Y-%m-%d'),
                                'amount': sf(float(amt)),
                                'year': div_date.year,
                            })
                            if div_date >= one_year_ago:
                                total_div_1y += float(amt)

                        if recent_divs:
                            # Temettu verimi
                            div_yield = sf((total_div_1y / cp * 100) if cp > 0 else 0)
                            # Son temettu
                            last_div = recent_divs[-1]

                            # Yillik temettu ozeti
                            yearly_divs = {}
                            for d in recent_divs:
                                yr = d['year']
                                yearly_divs[yr] = yearly_divs.get(yr, 0) + float(d['amount'])

                            dividends_list.append({
                                'code': sym,
                                'name': BIST100_STOCKS.get(sym, sym),
                                'price': sf(cp),
                                'lastDividend': last_div,
                                'dividendYield': div_yield,
                                'totalDiv1Y': sf(total_div_1y),
                                'history': recent_divs[-10:],  # Son 10 temettu
                                'yearlyTotals': {str(k): sf(v) for k, v in sorted(yearly_divs.items())},
                                'changePct': stock.get('changePct', 0),
                            })
            except:
                continue

        # Temettu verimine gore sirala
        dividends_list.sort(key=lambda x: float(x.get('dividendYield', 0)), reverse=True)

        return jsonify(safe_dict({
            'success': True,
            'count': len(dividends_list),
            'dividends': dividends_list,
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# TELEGRAM OTOMATIK BILDIRIM
# =====================================================================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

def send_telegram(message):
    """Telegram mesaji gonder"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    try:
        import requests as req
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        req.post(url, json={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'HTML'}, timeout=10)
        return True
    except:
        return False

def _auto_signal_check():
    """Arka planda guclu sinyalleri tespit edip Telegram bildirim gonder - Enhanced v7"""
    while True:
        try:
            time.sleep(600)  # 10dk arayla kontrol
            if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
                continue

            stocks = _get_stocks()
            if not stocks:
                continue

            # Enhanced: check_signal_alerts kullan (mum formasyonu, dinamik esikler dahil)
            signal_alerts = check_signal_alerts()
            if not signal_alerts:
                continue

            # Piyasa rejimi
            regime = calc_market_regime()
            regime_emoji = {'strong_bull': '🐂🐂', 'bull': '🐂', 'strong_bear': '🐻🐻', 'bear': '🐻', 'sideways': '↔️'}.get(regime.get('regime', ''), '❓')

            alerts_text = []
            for alert in signal_alerts[:15]:
                emoji = '🟢' if alert.get('signal') == 'bullish' else ('🔴' if alert.get('signal') == 'bearish' else '⚪')
                alerts_text.append(f"{emoji} {alert['message']}")

            if alerts_text:
                header = f"📊 <b>BIST Sinyal Raporu v7</b> ({datetime.now().strftime('%H:%M')})\n"
                header += f"{regime_emoji} Piyasa: {regime.get('description', 'Bilinmiyor')}\n\n"
                msg = header + '\n'.join(alerts_text)
                send_telegram(msg)

        except:
            continue

# Telegram bildirim thread'ini baslat
_telegram_thread_started = False
def _start_telegram_thread():
    global _telegram_thread_started
    if _telegram_thread_started:
        return
    _telegram_thread_started = True
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        t = threading.Thread(target=_auto_signal_check, daemon=True)
        t.start()
        print("[TELEGRAM] Otomatik sinyal bildirimi aktif")

@app.route('/api/telegram/test')
def test_telegram():
    """Telegram baglantisini test et"""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return jsonify({'success': False, 'error': 'TELEGRAM_BOT_TOKEN ve TELEGRAM_CHAT_ID env degiskenleri gerekli',
                        'setup': 'Render Dashboard > Environment Variables:\n1. TELEGRAM_BOT_TOKEN = @BotFather\'dan alinan token\n2. TELEGRAM_CHAT_ID = @userinfobot\'tan alinan chat ID'})
    ok = send_telegram("✅ BIST Pro Telegram bildirimi calisiyor!")
    return jsonify({'success': ok, 'message': 'Test mesaji gonderildi' if ok else 'Gonderilemedi'})

@app.route('/api/telegram/send-report', methods=['POST'])
def send_telegram_report():
    """Manuel sinyal raporu gonder"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'error': 'Veriler henuz yuklenmedi'}), 400

        strong_buys = []
        strong_sells = []
        for stock in stocks:
            sym = stock['code']
            try:
                hist = _cget_hist(f"{sym}_1y")
                if hist is None:
                    continue
                c = hist['Close'].values.astype(float)
                cp = float(c[-1])
                ind = calc_all_indicators(hist, cp)
                summary = ind.get('summary', {})
                bc = summary.get('buySignals', 0)
                sc = summary.get('sellSignals', 0)
                total = summary.get('totalIndicators', 1)

                if bc >= total * 0.6:
                    strong_buys.append((sym, sf(cp), bc, total, stock.get('changePct', 0)))
                elif sc >= total * 0.6:
                    strong_sells.append((sym, sf(cp), sc, total, stock.get('changePct', 0)))
            except:
                continue

        msg = f"📊 <b>BIST Gunluk Sinyal Raporu</b>\n📅 {datetime.now().strftime('%d.%m.%Y %H:%M')}\n\n"

        if strong_buys:
            msg += "🟢 <b>GUCLU ALIS SINYALLERI:</b>\n"
            for sym, price, bc, total, chg in sorted(strong_buys, key=lambda x: -x[2]):
                msg += f"  • {sym}: {price} TL (%{chg}) - {bc}/{total} AL\n"

        if strong_sells:
            msg += "\n🔴 <b>GUCLU SATIS SINYALLERI:</b>\n"
            for sym, price, sc, total, chg in sorted(strong_sells, key=lambda x: -x[2]):
                msg += f"  • {sym}: {price} TL (%{chg}) - {sc}/{total} SAT\n"

        if not strong_buys and not strong_sells:
            msg += "Guclu sinyal bulunamadi. Piyasa notr gorunuyor."

        ok = send_telegram(msg)
        return jsonify({'success': ok, 'message': msg})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# KULLANICI SISTEMI (AUTH)
# =====================================================================
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        d = request.json or {}
        username = d.get('username', '').strip()
        password = d.get('password', '')
        email = d.get('email', '').strip()

        if not username or len(username) < 3:
            return jsonify({'error': 'Kullanici adi en az 3 karakter olmali'}), 400
        if not password or len(password) < 4:
            return jsonify({'error': 'Sifre en az 4 karakter olmali'}), 400

        db = get_db()
        existing = db.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
        if existing:
            db.close()
            return jsonify({'error': 'Bu kullanici adi zaten alinmis'}), 400

        user_id = str(uuid.uuid4())[:8]
        db.execute("INSERT INTO users (id, username, password_hash, email) VALUES (?, ?, ?, ?)",
                   (user_id, username, hash_password(password), email))
        db.commit()
        db.close()

        return jsonify({'success': True, 'userId': user_id, 'username': username})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        d = request.json or {}
        username = d.get('username', '').strip()
        password = d.get('password', '')

        if not username or not password:
            return jsonify({'error': 'Kullanici adi ve sifre gerekli'}), 400

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

        if not user:
            db.close()
            return jsonify({'error': 'Kullanici bulunamadi. Kayit olmaniz gerekebilir.'}), 401

        db.close()
        if user['password_hash'] != hash_password(password):
            return jsonify({'error': 'Sifre hatali'}), 401

        return jsonify({'success': True, 'userId': user['id'], 'username': user['username'], 'email': user['email'] or ''})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/profile', methods=['POST'])
def update_profile():
    try:
        d = request.json or {}
        user_id = d.get('userId', '')
        if not user_id:
            return jsonify({'error': 'Giris yapmaniz gerekli'}), 401

        db = get_db()
        user = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
        if not user:
            db.close()
            return jsonify({'error': 'Kullanici bulunamadi'}), 404

        email = d.get('email', user['email'] or '')
        telegram = d.get('telegramChatId', user['telegram_chat_id'] or '')
        db.execute("UPDATE users SET email=?, telegram_chat_id=? WHERE id=?", (email, telegram, user_id))
        db.commit()
        db.close()
        return jsonify({'success': True, 'message': 'Profil guncellendi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/check')
def check_session():
    """localStorage oturumunun hala gecerli olup olmadigini kontrol et"""
    uid = request.args.get('userId', '')
    if not uid:
        return jsonify({'valid': False})
    try:
        db = get_db()
        user = db.execute("SELECT id FROM users WHERE id=?", (uid,)).fetchone()
        db.close()
        return jsonify({'valid': user is not None})
    except:
        return jsonify({'valid': False})


# =====================================================================
# ALERT CHECK (tetikleme kontrolu)
# =====================================================================
@app.route('/api/alerts/check', methods=['POST'])
def check_alerts():
    """Tetiklenen uyarilari kontrol et"""
    try:
        d = request.json or {}
        uid = d.get('userId', '')
        if not uid:
            return jsonify({'success': True, 'triggered': []})
        db = get_db()
        rows = db.execute("SELECT * FROM alerts WHERE user_id=? AND active=1 AND triggered=0", (uid,)).fetchall()
        triggered = []
        for r in rows:
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
                db.execute("UPDATE alerts SET triggered=1, triggered_at=? WHERE id=?",
                           (datetime.now().isoformat(), r['id']))
                triggered.append({
                    'id': r['id'], 'symbol': r['symbol'], 'condition': r['condition'],
                    'targetValue': r['target_value'], 'currentPrice': price,
                    'message': f"{r['symbol']} uyarisi tetiklendi: {r['condition']} {r['target_value']} (Guncel: {price})"
                })
        db.commit()
        db.close()

        # Telegram bildirim gonder
        if triggered:
            _send_telegram_alerts(uid, triggered)

        return jsonify(safe_dict({'success': True, 'triggered': triggered}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# TELEGRAM BILDIRIM
# =====================================================================
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')

def _send_telegram_alerts(user_id, triggered_alerts):
    """Tetiklenen uyarilari Telegram'a gonder"""
    if not TELEGRAM_BOT_TOKEN:
        return
    try:
        db = get_db()
        user = db.execute("SELECT telegram_chat_id FROM users WHERE id=?", (user_id,)).fetchone()
        db.close()
        if not user or not user['telegram_chat_id']:
            return

        chat_id = user['telegram_chat_id']
        for alert in triggered_alerts:
            text = f"🔔 *BIST Pro Uyari*\n\n{alert['message']}"
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            try:
                req_lib.post(url, json={'chat_id': chat_id, 'text': text, 'parse_mode': 'Markdown'}, timeout=5)
            except:
                pass
    except:
        pass


# =====================================================================
# YENI ENDPOINTLER - Sinyal Performans, Sektor RS, Market Regime, Alerts
# =====================================================================
@app.route('/api/market/regime')
def market_regime_endpoint():
    """Piyasa rejimi (boga/ayi/yatay)"""
    try:
        regime = calc_market_regime()
        return jsonify(safe_dict({'success': True, **regime}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/sectors/analysis')
def sectors_analysis():
    """Sektor bazli goreceli guc analizi"""
    try:
        result = calc_sector_relative_strength()
        return jsonify(safe_dict({'success': True, **result}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/backtest-signals')
def stock_signal_backtest(symbol):
    """Hisse bazli sinyal backtest sonuclari"""
    try:
        symbol = symbol.upper()
        hist = _cget_hist(f"{symbol}_1y")
        if hist is None:
            hist = _fetch_hist_df(symbol, '1y')
        if hist is None or len(hist) < 50:
            return jsonify({'error': f'{symbol} icin yeterli veri yok'}), 400
        result = calc_signal_backtest(hist)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, **result}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>/fundamentals')
def stock_fundamentals_endpoint(symbol):
    """Hisse temel analiz verileri (F/K, PD/DD)"""
    try:
        symbol = symbol.upper()
        result = fetch_fundamental_data(symbol)
        return jsonify(safe_dict({'success': True, 'symbol': symbol, 'fundamentals': result}))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/signals')
def signal_alerts_endpoint():
    """Sinyal bazli otomatik uyarilar (mum formasyonu, altin kesisim vb.)"""
    try:
        alerts = check_signal_alerts()
        return jsonify(safe_dict({
            'success': True,
            'alerts': alerts[:30],
            'totalAlerts': len(alerts),
            'marketRegime': calc_market_regime(),
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals/performance')
def signals_performance():
    """Tum hisselerin sinyal performans ozeti"""
    try:
        stocks = _get_stocks()
        if not stocks:
            return jsonify({'success': True, 'loading': True})

        results = []
        for stock in stocks[:50]:  # Performans icin ilk 50
            sym = stock['code']
            try:
                hist = _cget_hist(f"{sym}_1y")
                if hist is None or len(hist) < 60:
                    continue
                bt = calc_signal_backtest(hist)
                if bt.get('totalSignals', 0) > 3:
                    overall = bt.get('overall', {})
                    results.append({
                        'code': sym,
                        'name': BIST100_STOCKS.get(sym, sym),
                        'totalSignals': bt['totalSignals'],
                        'winRate5d': overall.get('winRate5d', 0),
                        'winRate10d': overall.get('winRate10d', 0),
                        'winRate20d': overall.get('winRate20d', 0),
                        'avgReturn10d': overall.get('avgRet10d', 0),
                    })
            except:
                continue

        results.sort(key=lambda x: float(x.get('winRate10d', 0)), reverse=True)
        return jsonify(safe_dict({
            'success': True,
            'performance': results,
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# =====================================================================
# BES (Bireysel Emeklilik Sistemi) FON ANALIZ MODULU
# TEFAS API uzerinden fon verisi cekilir, analiz & oneri yapilir
# =====================================================================
_bes_cache = {}
_bes_cache_lock = threading.Lock()
BES_CACHE_TTL = 1800  # 30 dakika
_tefas_semaphore = threading.Semaphore(1)  # TEFAS API rate limiter - tek seferde 1 istek

# BES background analiz thread state
_bes_bg_loading = False
_bes_bg_error = ''

TEFAS_API_URL = "https://www.tefas.gov.tr/api/DB/BindHistoryInfo"
TEFAS_ALLOC_URL = "https://www.tefas.gov.tr/api/DB/BindHistoryAllocation"
TEFAS_COMPARE_URL = "https://www.tefas.gov.tr/api/DB/BindComparisonFundReturns"
TEFAS_HEADERS = {
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://www.tefas.gov.tr/TarihselVeriler.aspx',
    'Origin': 'https://www.tefas.gov.tr',
}

# BES Fon Gruplari
BES_FUND_GROUPS = {
    'hisse': 'Hisse Senedi',
    'borclanma': 'Borçlanma Araçları',
    'katilim': 'Katılım',
    'karma': 'Karma / Dengeli',
    'doviz': 'Döviz',
    'altin': 'Altın / Kıymetli Maden',
    'endeks': 'Endeks',
    'para_piyasasi': 'Para Piyasası / Likit',
    'standart': 'Standart',
    'diger': 'Diğer',
}

def _bes_cache_get(key):
    with _bes_cache_lock:
        item = _bes_cache.get(key)
        if item and time.time() - item['ts'] < BES_CACHE_TTL:
            return item['data']
    return None

def _bes_cache_set(key, data):
    with _bes_cache_lock:
        _bes_cache[key] = {'data': data, 'ts': time.time()}

def _bes_bg_analyze_top():
    """Arka planda BES fon analizi yap ve cache'e kaydet (Render 30s timeout bypass)
    Strateji:
    1. TEFAS Compare API ile resmi getirileri cek (en guvenilir)
    2. Basarisizsa broad fetch + manual hesaplama
    3. Son care: fallback veri"""
    global _bes_bg_loading, _bes_bg_error
    _bes_bg_loading = True
    _bes_bg_error = ''
    try:
        today = datetime.now()
        pool = []

        # ===== YONTEM 1: TEFAS Compare API (resmi getiriler) =====
        print("[BES-BG] ADIM 1: TEFAS Compare API deneniyor...")
        compare_data = _fetch_tefas_compare()

        if compare_data and isinstance(compare_data, list) and len(compare_data) > 0:
            print(f"[BES-BG] Compare API'den {len(compare_data)} fon alindi")

            # Compare verisini parse et
            parsed_funds = []
            for row in compare_data:
                parsed = _parse_compare_row(row)
                if parsed and parsed['code'] and (parsed['price'] > 0 or parsed['total_value'] > 0):
                    parsed_funds.append(parsed)

            if parsed_funds:
                # Buyukluge gore sirala
                parsed_funds.sort(key=lambda x: x['total_value'], reverse=True)
                print(f"[BES-BG] {len(parsed_funds)} fon parse edildi (Compare API)")

                for f in parsed_funds[:30]:
                    rets = f['returns']
                    # Volatilite tahmini: gunluk getiriden
                    vol_est = abs(f['daily_return']) * (252 ** 0.5) * 100 if f['daily_return'] else 0
                    if vol_est < 1 and rets.get('3a'):
                        vol_est = abs(rets['3a']) / 3 * 2  # Kaba tahmin

                    # Sharpe tahmini
                    ret_6m = rets.get('6a') or rets.get('3a') or rets.get('1a') or 0
                    sharpe_est = sf((ret_6m / max(vol_est, 1)) * 0.5) if vol_est > 0 else 0

                    pool.append({
                        'code': f['code'],
                        'name': f['name'],
                        'category': _classify_fund(f['name']),
                        'currentPrice': f['price'],
                        'firstPrice': f['price'],
                        'totalReturn': rets.get('1y') or rets.get('6a') or 0,
                        'totalDays': 252,
                        'returns': {
                            '1h': rets.get('1h'),
                            '1a': rets.get('1a'),
                            '3a': rets.get('3a'),
                            '6a': rets.get('6a'),
                            '1y': rets.get('1y'),
                        },
                        'volatility': sf(vol_est),
                        'maxDrawdown': 0,
                        'sharpe': sharpe_est,
                        'dailyReturns': [],
                        'priceHistory': [],
                    })

                if pool:
                    _bes_cache_set('bes_analysis_pool', pool)
                    print(f"[BES-BG] Compare API basarili: {len(pool)} fon cache'e yazildi")
                    # Ornek getiriler logla
                    for p in pool[:3]:
                        print(f"[BES-BG]   {p['code']}: 1a={p['returns'].get('1a')}% 3a={p['returns'].get('3a')}% 6a={p['returns'].get('6a')}% 1y={p['returns'].get('1y')}%")
                    return

        print("[BES-BG] Compare API basarisiz veya bos, ADIM 2'ye geciliyor...")

        # ===== YONTEM 2: Broad fetch + manual hesaplama =====
        print("[BES-BG] ADIM 2: TEFAS broad fetch deneniyor...")
        raw = None
        for days_back in [90, 60, 30, 14, 7]:
            start = (today - timedelta(days=days_back)).strftime('%d.%m.%Y')
            end = today.strftime('%d.%m.%Y')
            raw = _fetch_tefas_funds(start, end)
            if raw and isinstance(raw, list) and len(raw) > 0:
                print(f"[BES-BG] TEFAS broad fetch basarili: {len(raw)} satir ({days_back} gun)")
                break

        if raw:
            # Fon koduna gore grupla
            fund_map = {}
            parse_ok = 0
            parse_fail = 0
            for row in (raw if isinstance(raw, list) else []):
                parsed = _parse_fund_row(row)
                if parsed and parsed['code']:
                    code = parsed['code']
                    if code not in fund_map:
                        fund_map[code] = {'rows': [], 'meta': parsed}
                    fund_map[code]['rows'].append(row)
                    if parsed.get('total_value', 0) >= fund_map[code]['meta'].get('total_value', 0):
                        fund_map[code]['meta'] = parsed
                    if parsed.get('price', 0) > 0:
                        parse_ok += 1
                    else:
                        parse_fail += 1

            print(f"[BES-BG] {len(fund_map)} unique fon, price>0: {parse_ok}, price=0: {parse_fail}")

            if fund_map:
                sorted_funds = sorted(fund_map.values(), key=lambda x: x['meta'].get('total_value', 0), reverse=True)

                # Debug: ilk 3 fonun verisini logla
                for fd in sorted_funds[:3]:
                    m = fd['meta']
                    print(f"[BES-BG] DEBUG fon: {m['code']} name={m['name'][:30]} price={m['price']} rows={len(fd['rows'])}")

                for fund_data in sorted_funds[:20]:
                    code = fund_data['meta']['code']
                    rows = fund_data['rows']
                    if len(rows) >= 2:
                        try:
                            perf = _analyze_fund_performance(rows, code)
                            if perf:
                                pool.append(perf)
                        except Exception as fe:
                            print(f"[BES-BG] Analiz hatasi {code}: {fe}")

                print(f"[BES-BG] Broad fetch'ten {len(pool)} fon analiz edildi")

                # Yetersizse sequential fetch
                if len(pool) < 5:
                    remaining = [fd for fd in sorted_funds[:20] if fd['meta']['code'] not in [p['code'] for p in pool]]
                    for fund_data in remaining[:5]:
                        code = fund_data['meta']['code']
                        try:
                            print(f"[BES-BG] Sequential fetch: {code}")
                            history = _fetch_tefas_history_chunked(code, days=200)
                            perf = _analyze_fund_performance(history, code)
                            if perf:
                                pool.append(perf)
                            time.sleep(1.5)
                        except Exception as fe:
                            print(f"[BES-BG] Sequential fetch hatasi {code}: {fe}")
                            time.sleep(2)

        # ===== YONTEM 3: Fallback =====
        if not pool:
            print("[BES-BG] Tum yontemler basarisiz, fallback moda geciliyor...")
            if raw and isinstance(raw, list):
                fund_map_fb = {}
                for row in raw:
                    parsed = _parse_fund_row(row)
                    if parsed and parsed['code']:
                        code = parsed['code']
                        if code not in fund_map_fb or parsed.get('total_value', 0) > fund_map_fb[code].get('total_value', 0):
                            fund_map_fb[code] = parsed
                for f in sorted(fund_map_fb.values(), key=lambda x: x.get('total_value', 0), reverse=True)[:15]:
                    pool.append({
                        'code': f['code'],
                        'name': f.get('name', ''),
                        'category': _classify_fund(f.get('name', '')),
                        'currentPrice': f.get('price', 0),
                        'firstPrice': f.get('price', 0),
                        'totalReturn': 0, 'totalDays': 1,
                        'returns': {'1h': None, '1a': None, '3a': None, '6a': None, '1y': None},
                        'volatility': 0, 'maxDrawdown': 0, 'sharpe': 0,
                        'dailyReturns': [], 'priceHistory': [],
                    })

        if pool:
            _bes_cache_set('bes_analysis_pool', pool)
            print(f"[BES-BG] Analiz tamamlandi: {len(pool)} fon cache'e yazildi")
        else:
            _bes_bg_error = 'Fon analizi yapilamadi - TEFAS API yanit vermiyor'
            print("[BES-BG] Hicbir fon analiz edilemedi")
    except Exception as e:
        _bes_bg_error = str(e)
        print(f"[BES-BG] HATA: {e}")
        traceback.print_exc()
    finally:
        _bes_bg_loading = False

def _classify_fund(fund_name):
    """Fon adina gore kategori tahmini"""
    name_upper = fund_name.upper() if fund_name else ''
    if any(k in name_upper for k in ['HİSSE', 'HISSE', 'EQUITY', 'PAY']): return 'hisse'
    if any(k in name_upper for k in ['BORÇLANMA', 'BORCLANMA', 'TAHVİL', 'TAHVIL', 'BONO', 'BOND']): return 'borclanma'
    if any(k in name_upper for k in ['KATILIM', 'KATKIM', 'SUKUK']): return 'katilim'
    if any(k in name_upper for k in ['KARMA', 'DENGELİ', 'DENGELI', 'MIX', 'BALANCED']): return 'karma'
    if any(k in name_upper for k in ['DÖVİZ', 'DOVIZ', 'EURO', 'DOLAR', 'USD', 'EUR', 'FX']): return 'doviz'
    if any(k in name_upper for k in ['ALTIN', 'KIYMETLI', 'GOLD', 'PRECIOUS', 'GÜMÜŞ', 'GUMUS']): return 'altin'
    if any(k in name_upper for k in ['ENDEKS', 'INDEX']): return 'endeks'
    if any(k in name_upper for k in ['LİKİT', 'LIKIT', 'PARA PİYASASI', 'PARA PIYASASI', 'MONEY']): return 'para_piyasasi'
    if any(k in name_upper for k in ['STANDART', 'STANDARD']): return 'standart'
    return 'diger'

def _fetch_tefas_funds(start_date, end_date, fund_code=''):
    """TEFAS API'den BES fon verisi cek (max 90 gun) - retry destekli, semaphore ile rate limited"""
    _tefas_semaphore.acquire()
    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = {
                    'fontip': 'EMK',
                    'sfontur': '',
                    'fonkod': fund_code,
                    'fongrup': '',
                    'bastarih': start_date,
                    'bittarih': end_date,
                    'fonturkod': '',
                    'fonunvantip': '',
                }
                resp = req_lib.post(TEFAS_API_URL, headers=TEFAS_HEADERS, data=data, timeout=30)
                if resp.status_code in (403, 429, 503):
                    wait = (attempt + 1) * 3
                    print(f"[BES] TEFAS rate limit ({resp.status_code}), {wait}s bekleniyor... (deneme {attempt+1}/{max_retries})")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                result = resp.json()
                # Debug: ilk cagrilarda API yapisini logla
                if fund_code and isinstance(result, dict):
                    keys = list(result.keys())[:5]
                    print(f"[BES] TEFAS response keys ({fund_code}): {keys}")
                if isinstance(result, dict) and 'data' in result:
                    rows = result['data']
                    if rows and isinstance(rows, list) and len(rows) > 0:
                        print(f"[BES] TEFAS {fund_code or 'ALL'}: {len(rows)} satir, ornek keys: {list(rows[0].keys())[:8] if isinstance(rows[0], dict) else 'not-dict'}")
                    return rows
                if isinstance(result, list):
                    if result and isinstance(result[0], dict):
                        print(f"[BES] TEFAS {fund_code or 'ALL'}: {len(result)} satir (list), ornek keys: {list(result[0].keys())[:8]}")
                    return result
                return result
            except Exception as e:
                print(f"[BES] TEFAS fetch hata (deneme {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
        return []
    finally:
        _tefas_semaphore.release()
        time.sleep(0.8)  # Her TEFAS cagrisi arasinda 0.8s bekleme


def _fetch_tefas_compare():
    """TEFAS Compare API'den tum EMK fonlarinin donemsel getirilerini cek.
    Bu endpoint resmi hesaplanmis getirileri dogrudan dondurur."""
    _tefas_semaphore.acquire()
    try:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = {
                    'fontip': 'EMK',
                    'sfontur': '',
                    'fonkod': '',
                    'fongrup': '',
                    'fonturkod': '',
                    'fonunvantip': '',
                }
                resp = req_lib.post(TEFAS_COMPARE_URL, headers=TEFAS_HEADERS, data=data, timeout=30)
                if resp.status_code in (403, 429, 503):
                    wait = (attempt + 1) * 3
                    print(f"[BES-CMP] TEFAS rate limit ({resp.status_code}), {wait}s bekleniyor...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                result = resp.json()
                rows = result.get('data', result) if isinstance(result, dict) else result
                if rows and isinstance(rows, list) and len(rows) > 0:
                    sample = rows[0] if isinstance(rows[0], dict) else {}
                    print(f"[BES-CMP] TEFAS Compare basarili: {len(rows)} fon, ornek keys: {list(sample.keys())[:12]}")
                    # Ilk satirdan tum key'leri logla (debug)
                    if sample:
                        print(f"[BES-CMP] Ornek fon: {dict(list(sample.items())[:10])}")
                    return rows
                print(f"[BES-CMP] TEFAS Compare bos yanit: {type(result)}")
                return []
            except Exception as e:
                print(f"[BES-CMP] TEFAS Compare hata (deneme {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
        return []
    finally:
        _tefas_semaphore.release()
        time.sleep(0.8)


def _parse_compare_row(row):
    """TEFAS Compare API satirindan getiri bilgilerini cek"""
    if not isinstance(row, dict):
        return None

    code = _get_tefas_field(row, 'FonKodu', 'fonkodu', 'FONKODU', 'FonKod', default='')
    name = _get_tefas_field(row, 'FonUnvani', 'fonunvani', 'FONUNVANI', 'FonAdi', 'Fon', default='')
    price = sf(_get_tefas_field(row, 'BirimPayDegeri', 'birimpay', 'BirimPayDeger', 'FonFiyat', default=0))
    total_value = sf(_get_tefas_field(row, 'ToplamDeger', 'toplamdeger', 'TOPLAMDEGER', default=0))

    # Getiri alanlari - TEFAS farkli isimler kullanabilir
    ret_1h = _get_tefas_field(row, 'HaftalikGetiri', 'haftalikgetiri', 'BirHaftalikGetiri',
                               '1HaftalikGetiri', 'Haftalik', default=None)
    ret_1a = _get_tefas_field(row, 'AylikGetiri', 'aylikgetiri', 'BirAylikGetiri',
                               '1AylikGetiri', 'Aylik', default=None)
    ret_3a = _get_tefas_field(row, 'UcAylikGetiri', 'ucaylikgetiri', '3AylikGetiri',
                               'UcAylik', default=None)
    ret_6a = _get_tefas_field(row, 'AltiAylikGetiri', 'altiaylikgetiri', '6AylikGetiri',
                               'AltiAylik', default=None)
    ret_1y = _get_tefas_field(row, 'YillikGetiri', 'yillikgetiri', 'BirYillikGetiri',
                               '1YillikGetiri', 'Yillik', 'YilBasindanGetiri', default=None)
    ret_daily = _get_tefas_field(row, 'GunlukGetiri', 'gunlukgetiri', 'Gunluk', default=None)

    # Sayisal deger alanlari da dene (bazi TEFAS versiyonlari farkli isimlendirme kullaniyor)
    for key, val_ref in row.items():
        kl = key.lower().replace('İ', 'i').replace('ı', 'i')
        if val_ref is not None and isinstance(val_ref, (int, float)):
            if ret_1h is None and ('hafta' in kl or '1h' in kl or '1w' in kl): ret_1h = val_ref
            elif ret_1a is None and ('1ay' in kl or '1a' in kl or '1m' in kl) and 'aylik' not in kl: ret_1a = val_ref
            elif ret_3a is None and ('3ay' in kl or 'ucay' in kl or '3a' in kl or '3m' in kl): ret_3a = val_ref
            elif ret_6a is None and ('6ay' in kl or 'altiay' in kl or '6a' in kl or '6m' in kl): ret_6a = val_ref
            elif ret_1y is None and ('yil' in kl or '1y' in kl or '12' in kl) and 'basindan' not in kl: ret_1y = val_ref

    returns = {}
    if ret_1h is not None: returns['1h'] = sf(float(ret_1h))
    if ret_1a is not None: returns['1a'] = sf(float(ret_1a))
    if ret_3a is not None: returns['3a'] = sf(float(ret_3a))
    if ret_6a is not None: returns['6a'] = sf(float(ret_6a))
    if ret_1y is not None: returns['1y'] = sf(float(ret_1y))

    return {
        'code': code or '',
        'name': name or '',
        'price': price or 0,
        'total_value': total_value or 0,
        'returns': returns,
        'daily_return': sf(float(ret_daily)) if ret_daily is not None else 0,
    }

def _fetch_tefas_allocation(fund_code, start_date, end_date):
    """TEFAS API'den fon portfoy dagilimini cek"""
    try:
        data = {
            'fontip': 'EMK',
            'fonkod': fund_code,
            'bastarih': start_date,
            'bittarih': end_date,
        }
        resp = req_lib.post(TEFAS_ALLOC_URL, headers=TEFAS_HEADERS, data=data, timeout=30)
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, dict) and 'data' in result:
            return result['data']
        if isinstance(result, list):
            return result
        return result
    except Exception as e:
        print(f"[BES] TEFAS allocation hata: {e}")
        return []

def _fetch_tefas_history_chunked(fund_code, days=365):
    """90 gunluk chunk'larla uzun sureli fon gecmisi cek - semaphore ile rate limited"""
    all_data = []
    end = datetime.now()
    start = end - timedelta(days=days)
    chunk_start = start
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=89), end)
        sd = chunk_start.strftime('%d.%m.%Y')
        ed = chunk_end.strftime('%d.%m.%Y')
        chunk_data = _fetch_tefas_funds(sd, ed, fund_code)
        if chunk_data and isinstance(chunk_data, list):
            all_data.extend(chunk_data)
        chunk_start = chunk_end + timedelta(days=1)
        # Rate limiting artik _fetch_tefas_funds icinde semaphore ile yapiliyor
    return all_data

def _get_tefas_field(row, *keys, default=None):
    """TEFAS API field'ini case-insensitive olarak bul"""
    for key in keys:
        if key in row:
            return row[key]
    # Case-insensitive fallback
    row_lower = {k.lower(): v for k, v in row.items()}
    for key in keys:
        kl = key.lower()
        if kl in row_lower:
            return row_lower[kl]
    return default

def _parse_fund_row(row):
    """TEFAS API'den donen bir fund row'unu parse et - genis field name destegi"""
    if isinstance(row, dict):
        code = _get_tefas_field(row, 'FonKodu', 'fonkodu', 'FONKODU', 'FonKod', default='')
        name = _get_tefas_field(row, 'FonUnvani', 'fonunvani', 'FONUNVANI', 'FonAdi', default='')
        date_raw = _get_tefas_field(row, 'Tarih', 'tarih', 'TARIH', default='')
        price = sf(_get_tefas_field(row, 'BirimPayDegeri', 'birimpay', 'BIRIMPAY', 'BirimPayDeger', default=0))
        total_value = sf(_get_tefas_field(row, 'ToplamDeger', 'toplamdeger', 'TOPLAMDEGER', default=0))
        investors = si(_get_tefas_field(row, 'YatirimciSayisi', 'yatirimcisayisi', 'YATIRIMCISAYISI', default=0))
        shares = sf(_get_tefas_field(row, 'PaySayisi', 'paysayisi', 'PAYSAYISI', default=0))

        # Eger price 0 ama total_value var ise, total_value/shares'den hesapla
        if (not price or price <= 0) and total_value and total_value > 0 and shares and shares > 0:
            price = sf(total_value / shares)

        # Tarih WCF formatinda olabilir
        if date_raw and isinstance(date_raw, str) and '/Date(' in date_raw:
            dt = _parse_tefas_date(date_raw)
            if dt:
                date_raw = dt.strftime('%d.%m.%Y')

        return {
            'code': code or '',
            'name': name or '',
            'date': date_raw or '',
            'price': price or 0,
            'total_value': total_value or 0,
            'investors': investors or 0,
            'shares': shares or 0,
        }
    return None

def _parse_tefas_date(date_val):
    """TEFAS tarih formatlarini parse et: 'dd.MM.yyyy', 'yyyy-MM-dd', '/Date(timestamp)/' """
    if not date_val:
        return None
    if isinstance(date_val, str):
        # WCF date format: /Date(1645488000000)/
        wcf_match = re.match(r'/Date\((\-?\d+)\)/', date_val)
        if wcf_match:
            ts = int(wcf_match.group(1)) / 1000
            return datetime.fromtimestamp(ts)
        for fmt in ['%d.%m.%Y', '%Y-%m-%d', '%Y-%m-%dT%H:%M:%S']:
            try:
                return datetime.strptime(date_val, fmt)
            except:
                continue
    return None

def _analyze_fund_performance(history_data, fund_code):
    """Fon gecmis verisinden performans metrikleri hesapla"""
    if not history_data or len(history_data) < 2:
        return None

    # Fiyatlari cikar ve sırala
    prices = []
    for row in history_data:
        parsed = _parse_fund_row(row)
        if parsed and parsed['price'] and parsed['price'] > 0:
            dt = _parse_tefas_date(parsed['date'])
            prices.append({
                'date': parsed['date'],
                'date_parsed': dt,
                'price': parsed['price'],
                'code': parsed['code'],
                'name': parsed['name'],
            })

    if len(prices) < 2:
        return None

    # Tarihe gore sirala (en eski en basta) - datetime objeleriyle dogru siralama
    try:
        prices.sort(key=lambda x: x['date_parsed'] or datetime.min)
    except:
        # Fallback: string siralama
        prices.sort(key=lambda x: x['date'])

    current_price = prices[-1]['price']
    first_price = prices[0]['price']
    total_return = ((current_price - first_price) / first_price) * 100 if first_price > 0 else 0
    total_days = len(prices)

    # Donemsel getiriler - tarih bazli lookback (takvim gunu -> islem gunu donusumu)
    returns = {}
    period_map = {'1h': 7, '1a': 30, '3a': 90, '6a': 180, '1y': 365}
    for label, cal_days in period_map.items():
        # Tarih bazli lookback: en son tarihten cal_days gun oncesine en yakin veri noktasini bul
        target_date = prices[-1]['date_parsed'] - timedelta(days=cal_days) if prices[-1].get('date_parsed') else None
        found_price = None

        if target_date:
            # Target tarihine en yakin (ve oncesindeki) veri noktasini bul
            for p in prices:
                if p.get('date_parsed') and p['date_parsed'] <= target_date:
                    found_price = p['price']
                # Once erisince devam et (sirali oldugu icin son eslesme en yakin olacak)

        # Tarih bazli bulunamadiysa, index bazli fallback
        if not found_price:
            # Tahmini islem gunu: takvim gunu * 5/7 (hafta ici orani)
            approx_trading_days = max(1, int(cal_days * 5 / 7))
            if len(prices) >= approx_trading_days:
                found_price = prices[-min(approx_trading_days, len(prices))]['price']
            elif len(prices) >= max(cal_days // 4, 2):
                # En az ceyrek kadar veri varsa mevcut verinin basindan hesapla
                found_price = prices[0]['price']

        if found_price and found_price > 0:
            returns[label] = sf(((current_price - found_price) / found_price) * 100)
        else:
            returns[label] = None

    # Volatilite (gunluk fiyat degisim std sapma)
    daily_returns = []
    for i in range(1, len(prices)):
        if prices[i-1]['price'] > 0:
            dr = (prices[i]['price'] - prices[i-1]['price']) / prices[i-1]['price']
            daily_returns.append(dr)

    volatility = 0
    if daily_returns:
        mean_r = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_r)**2 for r in daily_returns) / len(daily_returns)
        volatility = sf(variance ** 0.5 * (252 ** 0.5) * 100)  # Yillik volatilite %

    # Max drawdown
    max_dd = 0
    peak = prices[0]['price']
    for p in prices:
        if p['price'] > peak:
            peak = p['price']
        dd = ((peak - p['price']) / peak) * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Sharpe benzeri metrik (risksiz oran ~%45 TRY)
    risk_free_daily = 0.45 / 252
    avg_daily_return = sum(daily_returns) / len(daily_returns) if daily_returns else 0
    std_daily = (sum((r - avg_daily_return)**2 for r in daily_returns) / len(daily_returns)) ** 0.5 if daily_returns else 1
    sharpe = sf(((avg_daily_return - risk_free_daily) / std_daily * (252 ** 0.5)) if std_daily > 0 else 0)

    return {
        'code': fund_code,
        'name': prices[-1].get('name', ''),
        'category': _classify_fund(prices[-1].get('name', '')),
        'currentPrice': current_price,
        'firstPrice': first_price,
        'totalReturn': sf(total_return),
        'totalDays': total_days,
        'returns': returns,
        'volatility': volatility,
        'maxDrawdown': sf(max_dd),
        'sharpe': sharpe,
        'dailyReturns': daily_returns[-30:] if daily_returns else [],
        'priceHistory': [{'date': p['date'], 'price': p['price']} for p in prices[-90:]],
    }

def _bes_optimize(funds_perf, risk_profile='moderate', horizon_months=12):
    """
    BES fon dagilim optimizasyonu.
    Risk profiline gore ideal fon agirliklarini hesaplar.
    """
    if not funds_perf:
        return []

    risk_weights = {
        'conservative': {'return_w': 0.2, 'risk_w': 0.6, 'sharpe_w': 0.2, 'max_equity': 20},
        'moderate':     {'return_w': 0.35, 'risk_w': 0.35, 'sharpe_w': 0.3, 'max_equity': 50},
        'aggressive':   {'return_w': 0.5, 'risk_w': 0.15, 'sharpe_w': 0.35, 'max_equity': 80},
    }
    weights = risk_weights.get(risk_profile, risk_weights['moderate'])

    scored_funds = []
    for f in funds_perf:
        if not f:
            continue
        rets = f.get('returns', {})
        ret_6m = rets.get('6a') or rets.get('3a') or rets.get('1a') or 0
        vol = f.get('volatility', 50)
        sharpe = f.get('sharpe', 0)
        category = f.get('category', 'diger')

        # Normalizing: return pozitif iyi, vol dusuk iyi, sharpe yuksek iyi
        ret_score = min(max(ret_6m, -50), 100) / 100
        vol_score = 1 - min(vol, 100) / 100
        sharpe_score = min(max(sharpe, -3), 3) / 3

        total_score = (
            ret_score * weights['return_w'] +
            vol_score * weights['risk_w'] +
            sharpe_score * weights['sharpe_w']
        )

        scored_funds.append({
            'code': f['code'],
            'name': f['name'],
            'category': category,
            'categoryLabel': BES_FUND_GROUPS.get(category, 'Diğer'),
            'score': sf(total_score * 100),
            'return6m': sf(ret_6m),
            'volatility': f['volatility'],
            'sharpe': f['sharpe'],
            'maxDrawdown': f['maxDrawdown'],
            'currentPrice': f['currentPrice'],
        })

    scored_funds.sort(key=lambda x: x['score'], reverse=True)

    # Kategori bazli dagilim onerisi
    category_targets = {
        'conservative': {'borclanma': 40, 'para_piyasasi': 25, 'altin': 15, 'katilim': 10, 'hisse': 5, 'diger': 5},
        'moderate':     {'hisse': 25, 'borclanma': 25, 'altin': 15, 'katilim': 15, 'para_piyasasi': 10, 'diger': 10},
        'aggressive':   {'hisse': 40, 'endeks': 15, 'altin': 15, 'doviz': 10, 'borclanma': 10, 'karma': 10},
    }
    targets = category_targets.get(risk_profile, category_targets['moderate'])

    # Her kategori icin en iyi fonu sec
    recommendations = []
    used_categories = set()
    for cat, target_pct in sorted(targets.items(), key=lambda x: x[1], reverse=True):
        best_in_cat = [f for f in scored_funds if f['category'] == cat]
        if best_in_cat:
            pick = best_in_cat[0]
            pick['recommendedPct'] = target_pct
            pick['reasoning'] = _fund_reasoning(pick, cat, target_pct, risk_profile)
            recommendations.append(pick)
            used_categories.add(cat)

    # Hedef kategoride fon bulunamazsa en iyi genel fonlardan tamamla
    total_allocated = sum(r['recommendedPct'] for r in recommendations)
    if total_allocated < 100:
        remaining = 100 - total_allocated
        remaining_funds = [f for f in scored_funds if f['code'] not in [r['code'] for r in recommendations]]
        if remaining_funds:
            pick = remaining_funds[0]
            pick['recommendedPct'] = remaining
            pick['reasoning'] = f"Kalan %{remaining} oran portföy dengeleme amacıyla önerildi."
            recommendations.append(pick)

    return recommendations

def _fund_reasoning(fund, category, pct, risk_profile):
    """Fon onerisi icin aciklama metni uret"""
    cat_label = BES_FUND_GROUPS.get(category, category)
    sharpe = fund.get('sharpe', 0)
    vol = fund.get('volatility', 0)
    ret = fund.get('return6m', 0)

    reasons = []
    if ret > 10: reasons.append(f"6 aylık getirisi %{ret} ile güçlü")
    elif ret > 0: reasons.append(f"6 aylık %{ret} pozitif getiri")
    else: reasons.append(f"6 aylık getiri %{ret}")

    if vol < 10: reasons.append("düşük volatilite")
    elif vol < 20: reasons.append("orta seviye volatilite")
    else: reasons.append(f"%{vol} volatilite")

    if sharpe > 1: reasons.append("yüksek risk-getiri oranı")
    elif sharpe > 0: reasons.append("pozitif risk-getiri dengesi")

    profile_labels = {'conservative': 'muhafazakar', 'moderate': 'dengeli', 'aggressive': 'agresif'}
    profile_label = profile_labels.get(risk_profile, 'dengeli')

    return f"{cat_label} kategorisinde {profile_label} profil için %{pct} ağırlık. " + ", ".join(reasons) + "."

def _simulate_bes(recommendations, monthly_contribution, horizon_months):
    """BES portfoy simulasyonu: aylık katkı ile birikim projeksiyonu"""
    if not recommendations or monthly_contribution <= 0 or horizon_months <= 0:
        return {'error': 'Geçersiz parametreler'}

    # Devlet katkisi (%30, yillik max limit - 2024 icin ~27.000 TL civarı)
    yearly_contribution = monthly_contribution * 12
    devlet_katkisi_rate = 0.30
    devlet_katkisi_yearly_max = 30000  # Yaklasik yillik limit
    devlet_katkisi_yearly = min(yearly_contribution * devlet_katkisi_rate, devlet_katkisi_yearly_max)
    devlet_katkisi_monthly = devlet_katkisi_yearly / 12

    # Her fon icin aylik getiri tahmini (gecmis veriden)
    total_monthly = monthly_contribution + devlet_katkisi_monthly
    monthly_results = []
    fund_balances = {}

    for rec in recommendations:
        pct = rec.get('recommendedPct', 0) / 100
        ret_6m = rec.get('return6m', 0) or 0
        # 6 aylik getiriyi aylik getiriye cevir
        monthly_return = ((1 + ret_6m / 100) ** (1/6) - 1)
        fund_balances[rec['code']] = {
            'name': rec['name'],
            'code': rec['code'],
            'pct': pct,
            'monthlyReturn': monthly_return,
            'balance': 0,
            'totalContribution': 0,
        }

    total_balance = 0
    total_contributed = 0
    total_devlet = 0
    total_gain = 0

    for month in range(1, horizon_months + 1):
        month_contribution = monthly_contribution
        month_devlet = devlet_katkisi_monthly
        total_contributed += month_contribution
        total_devlet += month_devlet

        for code, fb in fund_balances.items():
            contrib = (month_contribution + month_devlet) * fb['pct']
            fb['totalContribution'] += contrib
            # Birikim: onceki bakiye * (1 + aylik getiri) + yeni katki
            fb['balance'] = fb['balance'] * (1 + fb['monthlyReturn']) + contrib

        total_balance = sum(fb['balance'] for fb in fund_balances.values())
        total_gain = total_balance - total_contributed - total_devlet

        if month % 3 == 0 or month == horizon_months or month <= 3:
            monthly_results.append({
                'month': month,
                'totalBalance': sf(total_balance),
                'totalContributed': sf(total_contributed),
                'devletKatkisi': sf(total_devlet),
                'totalGain': sf(total_gain),
                'gainPct': sf((total_gain / (total_contributed + total_devlet)) * 100) if (total_contributed + total_devlet) > 0 else 0,
            })

    fund_details = []
    for code, fb in fund_balances.items():
        gain = fb['balance'] - fb['totalContribution']
        fund_details.append({
            'code': fb['code'],
            'name': fb['name'],
            'pct': sf(fb['pct'] * 100),
            'balance': sf(fb['balance']),
            'totalContribution': sf(fb['totalContribution']),
            'gain': sf(gain),
            'gainPct': sf((gain / fb['totalContribution']) * 100) if fb['totalContribution'] > 0 else 0,
            'monthlyReturn': sf(fb['monthlyReturn'] * 100, 3),
        })

    return {
        'totalBalance': sf(total_balance),
        'totalContributed': sf(total_contributed),
        'devletKatkisi': sf(total_devlet),
        'totalGain': sf(total_gain),
        'totalGainPct': sf((total_gain / (total_contributed + total_devlet)) * 100) if (total_contributed + total_devlet) > 0 else 0,
        'horizonMonths': horizon_months,
        'monthlyContribution': monthly_contribution,
        'monthlyDevlet': sf(devlet_katkisi_monthly),
        'timeline': monthly_results,
        'fundDetails': fund_details,
    }


# ---- BES API ROUTES ----

@app.route('/api/bes/funds')
def bes_funds():
    """Tum BES fonlarini listele (guncel fiyat, getiri)"""
    try:
        cache_key = 'bes_funds_all'
        cached = _bes_cache_get(cache_key)
        if cached:
            return jsonify(safe_dict(cached))

        today = datetime.now()
        start = (today - timedelta(days=30)).strftime('%d.%m.%Y')
        end = today.strftime('%d.%m.%Y')
        raw = _fetch_tefas_funds(start, end)

        if not raw:
            return jsonify({'success': False, 'error': 'TEFAS verisi alinamadi', 'funds': []})

        # En son tarihteki fonlari al
        funds_map = {}
        for row in (raw if isinstance(raw, list) else []):
            parsed = _parse_fund_row(row)
            if parsed and parsed['code']:
                code = parsed['code']
                if code not in funds_map or parsed['date'] > funds_map[code]['date']:
                    funds_map[code] = parsed

        funds_list = list(funds_map.values())
        for f in funds_list:
            f['category'] = _classify_fund(f.get('name', ''))
            f['categoryLabel'] = BES_FUND_GROUPS.get(f['category'], 'Diğer')

        funds_list.sort(key=lambda x: x.get('total_value', 0), reverse=True)

        result = {
            'success': True,
            'count': len(funds_list),
            'funds': funds_list,
            'timestamp': datetime.now().isoformat(),
            'categories': BES_FUND_GROUPS,
        }
        _bes_cache_set(cache_key, result)
        return jsonify(safe_dict(result))
    except Exception as e:
        print(f"[BES] funds hata: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e), 'funds': []}), 500


@app.route('/api/bes/fund/<code>')
def bes_fund_detail(code):
    """Tek bir BES fonunun detayli analizi"""
    try:
        days = int(request.args.get('days', 365))
        cache_key = f'bes_fund_{code}_{days}'
        cached = _bes_cache_get(cache_key)
        if cached:
            return jsonify(safe_dict(cached))

        history = _fetch_tefas_history_chunked(code, days=days)
        if not history:
            return jsonify({'success': False, 'error': f'{code} fon verisi bulunamadi'})

        perf = _analyze_fund_performance(history, code)
        if not perf:
            return jsonify({'success': False, 'error': f'{code} analiz yapilamadi'})

        # Portfoy dagilimi
        today = datetime.now()
        alloc_start = (today - timedelta(days=7)).strftime('%d.%m.%Y')
        alloc_end = today.strftime('%d.%m.%Y')
        allocation = _fetch_tefas_allocation(code, alloc_start, alloc_end)

        result = {
            'success': True,
            'fund': perf,
            'allocation': allocation,
            'timestamp': datetime.now().isoformat(),
        }
        _bes_cache_set(cache_key, result)
        return jsonify(safe_dict(result))
    except Exception as e:
        print(f"[BES] fund detail hata: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bes/analyze', methods=['POST'])
def bes_analyze():
    """
    Kullanicinin BES bilgilerini alip analiz & oneri yap.
    Input: { funds: [{code, pct}], monthlyContribution, horizonMonths, riskProfile }
    """
    try:
        body = request.get_json(force=True)
        user_funds = body.get('funds', [])
        monthly_contribution = float(body.get('monthlyContribution', 1000))
        horizon_months = int(body.get('horizonMonths', 36))
        risk_profile = body.get('riskProfile', 'moderate')

        if risk_profile not in ('conservative', 'moderate', 'aggressive'):
            risk_profile = 'moderate'

        # Genel fon havuzundan en iyileri bul (cache'den)
        all_funds_cache = _bes_cache_get('bes_analysis_pool')
        if not all_funds_cache:
            # Cache yok - background thread baslatilmis mi kontrol et
            if not _bes_bg_loading:
                threading.Thread(target=_bes_bg_analyze_top, daemon=True).start()
            return jsonify({'success': True, 'loading': True, 'message': 'Fon verileri hazırlaniyor. Lütfen BES sekmesini açik birakin, veriler hazir olunca otomatik yüklenecek.'})

        all_perfs = all_funds_cache

        # Kullanicinin mevcut fonlarini analiz et (sadece cache hazirsa)
        current_analysis = []
        if user_funds:
            for uf in user_funds:
                code = uf.get('code', '').strip().upper()
                if not code:
                    continue
                # Oncelikle cache pool'dan bul
                cached_perf = next((f for f in all_perfs if f['code'] == code), None)
                if cached_perf:
                    perf = dict(cached_perf)
                else:
                    # Cache'de yoksa kisa sureli veri cek (timeout riski dusuk)
                    history = _fetch_tefas_history_chunked(code, days=90)
                    perf = _analyze_fund_performance(history, code)
                if perf:
                    perf['userPct'] = uf.get('pct', 0)
                    current_analysis.append(perf)

        # Optimizasyon
        recommendations = _bes_optimize(all_perfs if all_perfs else current_analysis, risk_profile, horizon_months)

        # Simulasyon
        simulation = _simulate_bes(recommendations, monthly_contribution, horizon_months)

        # Mevcut portfoy vs onerilen karsilastirma
        current_score = 0
        if current_analysis:
            for ca in current_analysis:
                w = ca.get('userPct', 0) / 100
                s = ca.get('sharpe', 0)
                current_score += w * s

        recommended_score = 0
        if recommendations:
            for rec in recommendations:
                w = rec.get('recommendedPct', 0) / 100
                s = rec.get('sharpe', 0)
                recommended_score += w * s

        result = {
            'success': True,
            'currentPortfolio': {
                'funds': [{
                    'code': ca['code'],
                    'name': ca['name'],
                    'category': ca['category'],
                    'categoryLabel': BES_FUND_GROUPS.get(ca['category'], 'Diğer'),
                    'userPct': ca.get('userPct', 0),
                    'returns': ca['returns'],
                    'volatility': ca['volatility'],
                    'sharpe': ca['sharpe'],
                    'maxDrawdown': ca['maxDrawdown'],
                } for ca in current_analysis],
                'overallSharpe': sf(current_score),
            },
            'recommendations': recommendations,
            'simulation': simulation,
            'riskProfile': risk_profile,
            'riskProfileLabel': {'conservative': 'Muhafazakar', 'moderate': 'Dengeli', 'aggressive': 'Agresif'}.get(risk_profile, 'Dengeli'),
            'horizonMonths': horizon_months,
            'monthlyContribution': monthly_contribution,
            'comparison': {
                'currentScore': sf(current_score * 100),
                'recommendedScore': sf(recommended_score * 100),
                'improvement': sf((recommended_score - current_score) * 100),
            },
            'timestamp': datetime.now().isoformat(),
        }
        return jsonify(safe_dict(result))
    except Exception as e:
        print(f"[BES] analyze hata: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bes/simulate', methods=['POST'])
def bes_simulate():
    """Hizli simulasyon: verilen fonlar ve oranlarla birikim projeksiyonu"""
    try:
        body = request.get_json(force=True)
        funds = body.get('funds', [])
        monthly = float(body.get('monthlyContribution', 1000))
        months = int(body.get('horizonMonths', 36))

        # Her fonun performansini cek
        fund_perfs = []
        for f in funds:
            code = f.get('code', '').strip().upper()
            pct = float(f.get('pct', 0))
            if not code:
                continue
            history = _fetch_tefas_history_chunked(code, days=180)
            perf = _analyze_fund_performance(history, code)
            if perf:
                perf['recommendedPct'] = pct
                perf['return6m'] = perf.get('returns', {}).get('6a', 0)
                fund_perfs.append(perf)

        simulation = _simulate_bes(fund_perfs, monthly, months)
        return jsonify(safe_dict({
            'success': True,
            'simulation': simulation,
            'timestamp': datetime.now().isoformat(),
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/bes/debug')
def bes_debug():
    """BES veri pipeline debug endpoint"""
    try:
        results = {}
        today = datetime.now()

        # 1. Compare API testi
        try:
            compare = _fetch_tefas_compare()
            if compare and isinstance(compare, list) and len(compare) > 0:
                sample = compare[0] if isinstance(compare[0], dict) else {}
                parsed = _parse_compare_row(compare[0]) if sample else None
                results['compare_api'] = {
                    'success': True, 'count': len(compare),
                    'sample_keys': list(sample.keys())[:15],
                    'parsed_sample': parsed,
                }
            else:
                results['compare_api'] = {'success': False, 'data': str(compare)[:200]}
        except Exception as e:
            results['compare_api'] = {'success': False, 'error': str(e)}

        # 2. History API testi (7 gun)
        try:
            start = (today - timedelta(days=7)).strftime('%d.%m.%Y')
            end = today.strftime('%d.%m.%Y')
            hist = _fetch_tefas_funds(start, end)
            if hist and isinstance(hist, list) and len(hist) > 0:
                sample = hist[0] if isinstance(hist[0], dict) else {}
                parsed = _parse_fund_row(hist[0]) if sample else None
                results['history_api'] = {
                    'success': True, 'count': len(hist),
                    'sample_keys': list(sample.keys())[:15],
                    'parsed_sample': parsed,
                }
            else:
                results['history_api'] = {'success': False, 'data': str(hist)[:200]}
        except Exception as e:
            results['history_api'] = {'success': False, 'error': str(e)}

        # 3. Cache durumu
        pool = _bes_cache_get('bes_analysis_pool')
        results['cache'] = {
            'has_pool': pool is not None,
            'pool_size': len(pool) if pool else 0,
            'bg_loading': _bes_bg_loading,
            'bg_error': _bes_bg_error,
        }
        if pool and len(pool) > 0:
            sample = pool[0]
            results['cache']['sample_fund'] = {
                'code': sample.get('code'), 'name': sample.get('name', '')[:40],
                'returns': sample.get('returns'), 'volatility': sample.get('volatility'),
                'sharpe': sample.get('sharpe'), 'price': sample.get('currentPrice'),
            }

        return jsonify(safe_dict(results))
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bes/top')
def bes_top():
    """Kategorilere gore en iyi BES fonlari - background thread ile Render timeout bypass"""
    global _bes_bg_loading, _bes_bg_error
    try:
        category = request.args.get('category', '')
        period = request.args.get('period', '3a')
        limit_n = int(request.args.get('limit', 20))

        cache_key = f'bes_top_{category}_{period}'
        cached = _bes_cache_get(cache_key)
        if cached:
            return jsonify(safe_dict(cached))

        pool = _bes_cache_get('bes_analysis_pool')
        if not pool:
            # Cache yok - background thread ile analiz baslat
            if _bes_bg_loading:
                # Zaten calisiyor, polling devam etsin
                return jsonify({'success': True, 'loading': True, 'message': 'Fonlar analiz ediliyor, lutfen bekleyin...'})

            if _bes_bg_error:
                err = _bes_bg_error
                # Hata sonrasi tekrar dene - thread'i yeniden baslat
                _bes_bg_error = ''
                print(f"[BES-TOP] Onceki hata: {err}, yeniden deneniyor...")
                t = threading.Thread(target=_bes_bg_analyze_top, daemon=True)
                t.start()
                return jsonify({'success': True, 'loading': True, 'message': f'Tekrar deneniyor... ({err})'})

            # Background thread baslat
            print("[BES-TOP] Background analiz thread baslatiliyor...")
            t = threading.Thread(target=_bes_bg_analyze_top, daemon=True)
            t.start()
            return jsonify({'success': True, 'loading': True, 'message': 'BES fon analizi baslatildi, birkaç saniye sonra hazir olacak...'})

        # Pool hazir - filtrele ve sirala
        filtered = pool
        if category:
            filtered = [f for f in pool if f.get('category') == category]

        def sort_key(f):
            returns = f.get('returns', {})
            return returns.get(period, 0) or 0

        filtered.sort(key=sort_key, reverse=True)

        top_list = []
        for f in filtered[:limit_n]:
            top_list.append({
                'code': f['code'],
                'name': f['name'],
                'category': f['category'],
                'categoryLabel': BES_FUND_GROUPS.get(f['category'], 'Diğer'),
                'currentPrice': f['currentPrice'],
                'returns': f['returns'],
                'volatility': f['volatility'],
                'sharpe': f['sharpe'],
                'maxDrawdown': f['maxDrawdown'],
            })

        result = {
            'success': True,
            'category': category,
            'categoryLabel': BES_FUND_GROUPS.get(category, 'Tümü'),
            'period': period,
            'count': len(top_list),
            'funds': top_list,
            'timestamp': datetime.now().isoformat(),
        }
        _bes_cache_set(cache_key, result)
        return jsonify(safe_dict(result))
    except Exception as e:
        print(f"[BES-TOP] HATA: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/docs')
def docs():
    return jsonify({'name': 'BIST Pro v7.1.0', 'endpoints': [
        '/api/health', '/api/debug', '/api/dashboard', '/api/indices',
        '/api/bist100', '/api/bist30', '/api/stock/<sym>', '/api/stock/<sym>/kap',
        '/api/stock/<sym>/backtest-signals', '/api/stock/<sym>/fundamentals',
        '/api/commodity/<sym>', '/api/compare', '/api/screener', '/api/heatmap', '/api/report',
        '/api/backtest', '/api/sectors', '/api/sectors/analysis', '/api/search',
        '/api/signals', '/api/signals/performance', '/api/opportunities',
        '/api/market/regime', '/api/alerts/signals',
        '/api/auth/register', '/api/auth/login', '/api/auth/profile',
        '/api/portfolio', '/api/watchlist', '/api/alerts', '/api/alerts/check',
        '/api/bes/funds', '/api/bes/fund/<code>', '/api/bes/analyze',
        '/api/bes/simulate', '/api/bes/top',
    ]})

# NO MODULE-LEVEL THREAD START - before_request handles it
print("[STARTUP] BIST Pro v7.1.0 ready - batch loader + SQLite + uyelik + advanced analytics + BES")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"[STARTUP] Port {port}")
    app.run(host='0.0.0.0', port=port)
