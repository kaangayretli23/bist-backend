"""
BIST Pro - Config & Foundation Module
Extracted from backend.py: imports, DB config, cache, constants, utilities.
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

# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================
def hash_password(pw):
    return hashlib.sha256((pw + app.secret_key).encode()).hexdigest()

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
# CONSTANTS: BIST STOCKS, SECTORS, INDICES
# =====================================================================
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
# PARALLEL WORKERS
# =====================================================================
PARALLEL_WORKERS = 12  # Paralel hisse cekme sayisi (performans iyilestirmesi)

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

# Plan kilitleme cache: { 'THYAO_daily': { 'plan': {...}, 'locked_at': ts, 'locked_price': 287.0, 'signal': 'AL' } }
_plan_lock_cache = {}
_plan_lock_cache_lock = threading.Lock()

# Her timeframe icin ayri kilit suresi (saniye)
# min_lock: bu sure dolmadan sinyal degisimi yok sayilir
# max_lock: plan bu sureden sonra yenilenir
PLAN_LOCK_CONFIG = {
    'daily':   {'min_lock': 900,    'max_lock': 14400},   # 15dk min / 4 saat max  (intraday)
    'weekly':  {'min_lock': 7200,   'max_lock': 259200},  # 2sa min  / 3 gun max   (swing)
    'monthly': {'min_lock': 21600,  'max_lock': 604800},  # 6sa min  / 7 gun max   (pozisyon)
}
# Geriye donuk uyumluluk icin eski isimler
PLAN_MIN_LOCK_SECONDS = PLAN_LOCK_CONFIG['daily']['min_lock']
PLAN_MAX_LOCK_SECONDS = PLAN_LOCK_CONFIG['daily']['max_lock']

# HTML page cache
_html_page_cache = {'raw': None, 'gz': None, 'mtime': 0}
_html_page_cache_lock = threading.Lock()

# =====================================================================
# CACHE ACCESSOR FUNCTIONS
# =====================================================================
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
