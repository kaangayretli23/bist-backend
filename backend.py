"""
BIST Pro v6.0.0 - IS YATIRIM API PRIMARY
Thread guvenli: before_request ile lazy start.
Hicbir route yfinance CAGIRMAZ.
SQLite veritabani, kullanici sistemi, backtest, KAP haberleri
"""
from flask import Flask, jsonify, request, send_from_directory, make_response, session
from flask_cors import CORS
import traceback, os, time, threading, json, hashlib, sqlite3, uuid
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
DB_PATH = os.path.join(BASE_DIR, 'bist.db')
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'bist-pro-secret-' + str(hash(BASE_DIR)))
CORS(app, supports_credentials=True)

# =====================================================================
# DATABASE (SQLite)
# =====================================================================
def get_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    return db

def init_db():
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
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
    ''')
    db.commit()
    db.close()
    print("[DB] Veritabani hazir:", DB_PATH)

init_db()

def hash_password(pw):
    return hashlib.sha256((pw + app.secret_key).encode()).hexdigest()

@app.after_request
def force_json(resp):
    if resp.status_code >= 400 and 'text/html' in (resp.content_type or ''):
        resp = make_response(json.dumps({"error": f"HTTP {resp.status_code}"}), resp.status_code)
        resp.headers['Content-Type'] = 'application/json'
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

def _cget(store, key):
    with _lock:
        item = store.get(key)
        if item and time.time() - item['ts'] < CACHE_TTL:
            return item['data']
    return None

def _cset(store, key, data):
    with _lock:
        store[key] = {'data': data, 'ts': time.time()}

def _get_stocks(symbols=None):
    with _lock:
        if symbols:
            return [_stock_cache[s]['data'] for s in symbols
                    if s in _stock_cache and time.time()-_stock_cache[s]['ts']<CACHE_TTL]
        return [v['data'] for v in _stock_cache.values()
                if time.time()-v['ts']<CACHE_TTL]

def _get_indices():
    with _lock:
        return {k:v['data'] for k,v in _index_cache.items()
                if time.time()-v['ts']<CACHE_TTL}


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
        url = f"{IS_YATIRIM_BASE}?hisse={symbol}&startdate={sd}&enddate={ed}.json"

        print(f"  [ISYATIRIM] {symbol} {days}d cekiliyor...")
        resp = None
        last_err = None
        for http_attempt in range(3):
            try:
                resp = req_lib.get(url, headers=IS_YATIRIM_HEADERS, timeout=15)
                resp.raise_for_status()
                break
            except Exception as http_e:
                last_err = http_e
                if http_attempt < 2:
                    wait = (http_attempt + 1) * 1.5
                    print(f"  [ISYATIRIM] {symbol} HTTP hata ({http_e}), {wait}s sonra tekrar...")
                    time.sleep(wait)
        if resp is None:
            print(f"  [ISYATIRIM] {symbol} 3 HTTP denemesi basarisiz: {last_err}")
            return None

        data = resp.json()
        rows = data.get('value', [])
        if not rows or len(rows) < 2:
            print(f"  [ISYATIRIM] {symbol}: bos veri ({len(rows)} satir)")
            return None

        # Ilk satirdaki tum kolonlari logla (ilk seferde kesfet)
        if len(rows) > 0:
            cols = list(rows[0].keys())
            print(f"  [ISYATIRIM] {symbol}: {len(rows)} satir, kolonlar: {cols[:10]}")

        # DataFrame olustur - kolon isimlerini otomatik kesfet
        df_raw = pd.DataFrame(rows)

        # Tarih kolonu
        date_col = None
        for c in df_raw.columns:
            if 'TARIH' in c.upper():
                date_col = c; break
        if not date_col:
            print(f"  [ISYATIRIM] {symbol}: tarih kolonu bulunamadi")
            return None

        # OHLCV kolonlari - esnek mapping
        col_map = {}
        for c in df_raw.columns:
            cu = c.upper()
            if 'KAPANIS' in cu and 'DUZELTILMIS' not in cu: col_map['Close'] = c
            elif 'ACILIS' in cu: col_map['Open'] = c
            elif 'YUKSEK' in cu: col_map['High'] = c
            elif 'DUSUK' in cu: col_map['Low'] = c
            elif 'HACIM' in cu and 'TL' not in cu and 'LOT' in cu: col_map['Volume'] = c
            elif 'HACIM' in cu and 'LOT' not in cu and 'TL' not in cu: col_map['Volume'] = c

        # Volume bulunamadiysa HACIM iceren herhangi bir kolonu dene
        if 'Volume' not in col_map:
            for c in df_raw.columns:
                if 'HACIM' in c.upper():
                    col_map['Volume'] = c; break

        if 'Close' not in col_map:
            # Son care: DUZELTILMIS KAPANIS
            for c in df_raw.columns:
                if 'KAPANIS' in c.upper():
                    col_map['Close'] = c; break
            if 'Close' not in col_map:
                print(f"  [ISYATIRIM] {symbol}: Close kolonu bulunamadi")
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

        df = df.dropna(subset=['Close']).sort_index()

        if len(df) < 2:
            return None

        print(f"  [ISYATIRIM] {symbol} OK: {len(df)} bar, {df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')}")
        return df

    except Exception as e:
        print(f"  [ISYATIRIM] {symbol}: {e}")
        return None


def _fetch_isyatirim_quick(symbol):
    """Is Yatirim'dan son fiyat bilgisi (quick - dashboard icin)"""
    try:
        df = _fetch_isyatirim_df(symbol, days=7)
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
def _fetch_yahoo_http(symbol, period1_days=7):
    """Yahoo Finance v8 chart API - yedek kaynak"""
    try:
        now = int(time.time())
        p1 = now - (period1_days * 86400)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={p1}&period2={now}&interval=1d"
        r = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        with urllib.request.urlopen(r, timeout=15) as resp:
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

        return {
            'close': float(cur), 'prev': float(prev),
            'open': float(opens[last_i]) if opens[last_i] else float(cur),
            'high': float(highs[last_i]) if highs[last_i] else float(cur),
            'low': float(lows[last_i]) if lows[last_i] else float(cur),
            'volume': int(volumes[last_i]) if volumes[last_i] else 0,
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
        with urllib.request.urlopen(r, timeout=20) as resp:
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
                h = yf.Ticker(f"{sym}.IS").history(period="5d", timeout=10)
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
            wait = attempt * 2
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
            h = yf.Ticker(f"{sym}.IS").history(period=period, timeout=15)
            if h is not None and not h.empty and len(h) >= 10:
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
# BACKGROUND LOADER
# =====================================================================
def _background_loop():
    print(f"[LOADER] Thread basliyor, YF={YF_OK}")
    time.sleep(3)

    while True:
        try:
            # === FAZE 1: Endeksler ===
            _status['phase'] = 'indices'
            _status['error'] = ''
            print(f"\n[LOADER] ====== FAZE 1: Endeksler ======")

            for key, (tsym, name) in INDEX_TICKERS.items():
                data = _fetch_index_data(key, tsym, name)
                if data:
                    cur, prev = sf(data['close'], 4), sf(data['prev'], 4)
                    _cset(_index_cache, key, {
                        'name': name, 'value': cur,
                        'change': sf(cur - prev, 4),
                        'changePct': sf((cur - prev) / prev * 100 if prev else 0),
                        'volume': si(data.get('volume', 0)),
                    })
                time.sleep(0.5)

            print(f"[LOADER] Endeksler: {len(_index_cache)}/{len(INDEX_TICKERS)}")

            # === FAZE 2: BIST30 hisseleri ===
            _status['phase'] = 'stocks'
            _status['total'] = len(BIST100_STOCKS)
            _status['loaded'] = 0
            fail_count = 0
            fail_list = []
            print(f"\n[LOADER] ====== FAZE 2: BIST30 ({len(BIST30)} hisse) ======")

            for i, sym in enumerate(BIST30):
                data = _fetch_stock_data(sym)
                if data:
                    cur, prev = sf(data['close']), sf(data['prev'])
                    if prev > 0:
                        ch = sf(cur - prev); o = sf(data.get('open', cur))
                        _cset(_stock_cache, sym, {
                            'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                            'price': cur, 'prevClose': prev,
                            'change': ch, 'changePct': sf(ch / prev * 100),
                            'volume': si(data.get('volume', 0)),
                            'open': o, 'high': sf(data.get('high', cur)),
                            'low': sf(data.get('low', cur)),
                            'gap': sf(o - prev), 'gapPct': sf((o - prev) / prev * 100),
                        })
                    else:
                        fail_count += 1; fail_list.append(sym)
                        print(f"  [SKIP] {sym}: prev <= 0 (prev={prev})")
                else:
                    fail_count += 1; fail_list.append(sym)
                _status['loaded'] = i + 1
                if (i + 1) % 10 == 0:
                    print(f"  [BIST30] {i+1}/{len(BIST30)}, cache={len(_stock_cache)}, fail={fail_count}")
                time.sleep(0.8)

            print(f"[LOADER] BIST30: {len(_stock_cache)} hisse cache'de, {fail_count} basarisiz")
            if fail_list:
                print(f"[LOADER] BIST30 basarisiz: {fail_list}")

            # === FAZE 3: Kalan BIST100 hisseleri ===
            # Cache key'i olsa bile TTL dolmussa yeniden cek
            with _lock:
                cached_and_valid = {s for s in BIST100_STOCKS.keys()
                                    if s in _stock_cache and time.time() - _stock_cache[s]['ts'] < CACHE_TTL}
            remaining = [s for s in BIST100_STOCKS.keys() if s not in cached_and_valid]
            if remaining:
                print(f"\n[LOADER] ====== FAZE 3: {len(remaining)} kalan hisse ======")
                phase3_fail = 0
                phase3_fail_list = []
                for i, sym in enumerate(remaining):
                    data = _fetch_stock_data(sym)
                    if data:
                        cur, prev = sf(data['close']), sf(data['prev'])
                        if prev > 0:
                            ch = sf(cur - prev); o = sf(data.get('open', cur))
                            _cset(_stock_cache, sym, {
                                'code': sym, 'name': BIST100_STOCKS.get(sym, sym),
                                'price': cur, 'prevClose': prev,
                                'change': ch, 'changePct': sf(ch / prev * 100),
                                'volume': si(data.get('volume', 0)),
                                'open': o, 'high': sf(data.get('high', cur)),
                                'low': sf(data.get('low', cur)),
                                'gap': sf(o - prev), 'gapPct': sf((o - prev) / prev * 100),
                            })
                        else:
                            phase3_fail += 1; phase3_fail_list.append(sym)
                            print(f"  [SKIP] {sym}: prev <= 0 (prev={prev})")
                    else:
                        phase3_fail += 1; phase3_fail_list.append(sym)
                    if (i + 1) % 10 == 0:
                        print(f"  [BIST100] {i+1}/{len(remaining)}, toplam cache={len(_stock_cache)}, fail={phase3_fail}")
                    time.sleep(0.8)
                print(f"[LOADER] BIST100 kalan: {phase3_fail} basarisiz")
                if phase3_fail_list:
                    print(f"[LOADER] BIST100 basarisiz: {phase3_fail_list}")

            _status['phase'] = 'done'
            _status['lastRun'] = datetime.now().isoformat()
            print(f"\n[LOADER] ====== SONUC: {len(_stock_cache)} hisse, {len(_index_cache)} endeks ======\n")

        except Exception as e:
            print(f"[LOADER] FATAL: {e}")
            traceback.print_exc()
            _status['phase'] = 'error'
            _status['error'] = str(e)

        time.sleep(300)

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
        n=min(90,len(c)); rh,rl=h[-n:],l[-n:]; sups,ress=[],[]
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
            hi = float(np.max(rh))
            lo = float(np.min(rl))
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
        h,l,c=float(hist['High'].iloc[-1]),float(hist['Low'].iloc[-1]),float(hist['Close'].iloc[-1])
        o=float(hist['Open'].iloc[-1])
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
    """Haftalik/Aylik/Yillik al-sat onerisi - destek/direnc yorumlu"""
    try:
        c=hist['Close'].values.astype(float)
        h=hist['High'].values.astype(float)
        l=hist['Low'].values.astype(float)
        v=hist['Volume'].values.astype(float)
        n=len(c)
        cur=float(c[-1])
        recommendations={}

        # Destek/direnc hesapla (tum periyotlar icin ortak)
        sr = calc_support_resistance(hist)
        supports = sr.get('supports', [])
        resistances = sr.get('resistances', [])
        fib = calc_fibonacci(hist)
        fib_sup = fib.get('nearestSupport')
        fib_res = fib.get('nearestResistance')

        # Bollinger bantlari
        bb = calc_bollinger(c, cur)
        bb_upper = bb.get('upper', 0)
        bb_lower = bb.get('lower', 0)
        bb_middle = bb.get('middle', 0)

        for label, days in [('weekly',5),('monthly',22),('yearly',252)]:
            if n<days+14: recommendations[label]={'action':'neutral','confidence':0,'reasons':[],'score':0,'strategy':'Yeterli veri yok'}; continue

            sl=slice(-days,None)
            sc=c[sl]; sh=h[sl]; slow=l[sl]; sv=v[sl]

            score=0; reasons=[]; strategy_parts=[]

            # 1. Trend (SMA)
            sma20=np.mean(c[-20:]) if n>=20 else c[-1]
            sma50=np.mean(c[-50:]) if n>=50 else sma20
            sma200=np.mean(c[-200:]) if n>=200 else sma50
            if cur>sma20: score+=1; reasons.append(f'Fiyat ({sf(cur)}) SMA20 ({sf(sma20)}) uzerinde')
            else: score-=1; reasons.append(f'Fiyat ({sf(cur)}) SMA20 ({sf(sma20)}) altinda')

            if sma20>sma50: score+=1; reasons.append(f'SMA20 ({sf(sma20)}) > SMA50 ({sf(sma50)}) → Yukari trend')
            else: score-=1; reasons.append(f'SMA20 ({sf(sma20)}) < SMA50 ({sf(sma50)}) → Asagi trend')

            # 2. RSI
            rsi_val=calc_rsi(c)
            rsi_v = rsi_val.get('value', 50)
            if rsi_v<30: score+=2; reasons.append(f'RSI={sf(rsi_v)}: Asiri satim bolgesi (<30) → Alis firsati olabilir')
            elif rsi_v<40: score+=1; reasons.append(f'RSI={sf(rsi_v)}: Zayif bolge (30-40) → Toparlanma bekleniyor')
            elif rsi_v>70: score-=2; reasons.append(f'RSI={sf(rsi_v)}: Asiri alim bolgesi (>70) → Kar realizasyonu bekleniyor')
            elif rsi_v>60: score-=0.5; reasons.append(f'RSI={sf(rsi_v)}: Guclu bolge (60-70)')
            elif rsi_v>=50: score+=0.5; reasons.append(f'RSI={sf(rsi_v)}: Notr-pozitif')
            else: score-=0.5; reasons.append(f'RSI={sf(rsi_v)}: Notr-negatif')

            # 3. MACD
            macd=calc_macd(c)
            macd_type = macd.get('signalType', 'neutral')
            macd_hist = macd.get('histogram', 0)
            if macd_type=='buy':
                score+=1.5
                reasons.append(f'MACD alis sinyali (histogram: {macd_hist})')
            elif macd_type=='sell':
                score-=1.5
                reasons.append(f'MACD satis sinyali (histogram: {macd_hist})')

            # 4. Bollinger
            if bb_lower > 0 and cur < bb_lower:
                score+=1; reasons.append(f'Fiyat ({sf(cur)}) alt Bollinger bandinin ({sf(bb_lower)}) altinda → Toparlanma bekleniyor')
            elif bb_upper > 0 and cur > bb_upper:
                score-=1; reasons.append(f'Fiyat ({sf(cur)}) ust Bollinger bandinin ({sf(bb_upper)}) uzerinde → Geri cekilme bekleniyor')
            elif bb_middle > 0:
                if cur > bb_middle:
                    reasons.append(f'Fiyat Bollinger orta bant ({sf(bb_middle)}) uzerinde')
                else:
                    reasons.append(f'Fiyat Bollinger orta bant ({sf(bb_middle)}) altinda')

            # 5. Hacim trendi
            if len(sv)>5:
                vol_avg=np.mean(sv[-20:]) if len(sv)>=20 else np.mean(sv)
                vol_recent=np.mean(sv[-5:])
                vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 1
                if vol_ratio > 1.5:
                    if c[-1]>c[-5]: score+=1; reasons.append(f'Hacim ortalamanin {sf(vol_ratio)}x uzerinde + yukari hareket → Guclu alis')
                    else: score-=1; reasons.append(f'Hacim ortalamanin {sf(vol_ratio)}x uzerinde + dusus → Guclu satis baskisi')

            # 6. Momentum (periyoda gore)
            if len(sc)>=days:
                period_return=sf(((c[-1]-sc[0])/sc[0])*100)
                if period_return>10: score+=1.5; reasons.append(f'{label} getiri: %{period_return} (guclu yukselis)')
                elif period_return>5: score+=1; reasons.append(f'{label} getiri: %{period_return} (pozitif)')
                elif period_return<-10: score-=1.5; reasons.append(f'{label} getiri: %{period_return} (sert dusus)')
                elif period_return<-5: score-=1; reasons.append(f'{label} getiri: %{period_return} (negatif)')

            # 7. Stochastic
            stoch=calc_stochastic(c,h,l)
            stoch_k = stoch.get('k', 50)
            if stoch_k<20: score+=1; reasons.append(f'Stochastic K={sf(stoch_k)}: Asiri satim bolgesi')
            elif stoch_k>80: score-=1; reasons.append(f'Stochastic K={sf(stoch_k)}: Asiri alim bolgesi')

            # 8. ADX - Trend gucu
            adx_data = calc_adx(h, l, c)
            adx_val = adx_data.get('value', 25)
            if adx_val > 25:
                trend_dir = 'yukari' if adx_data.get('plusDI', 0) > adx_data.get('minusDI', 0) else 'asagi'
                reasons.append(f'ADX={sf(adx_val)}: Guclu {trend_dir} trend')
                if trend_dir == 'yukari': score += 0.5
                else: score -= 0.5

            # 9. Destek/Direnc bazli yorumlar
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

            # 10. Fibonacci bazli yorumlar
            if fib_sup and fib_sup.get('price'):
                fib_sup_dist = sf(((cur - fib_sup['price']) / cur) * 100)
                if float(fib_sup_dist) < 3:
                    strategy_parts.append(f"Fibonacci {fib_sup['level']} destegi ({fib_sup['price']} TL) yakininda")
            if fib_res and fib_res.get('price'):
                fib_res_dist = sf(((fib_res['price'] - cur) / cur) * 100)
                if float(fib_res_dist) < 3:
                    strategy_parts.append(f"Fibonacci {fib_res['level']} direnci ({fib_res['price']} TL) yakininda")

            # Sonuc
            max_score=12.0
            conf=min(abs(score)/max_score*100, 100)
            if score>=3: action='AL'
            elif score>=1.5: action='TUTUN/AL'
            elif score<=-3: action='SAT'
            elif score<=-1.5: action='TUTUN/SAT'
            else: action='NOTR'

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
                'reasons':reasons[:8],
                'strategy': ' | '.join(strategy_parts[:4]),
                'keyLevels': {
                    'supports': supports[:3],
                    'resistances': resistances[:3],
                    'sma20': sf(sma20),
                    'sma50': sf(sma50),
                    'bollingerUpper': sf(bb_upper),
                    'bollingerLower': sf(bb_lower),
                }
            }

        return recommendations
    except Exception as e:
        print(f"  [REC] Hata: {e}")
        return {'weekly':{'action':'neutral','confidence':0,'reasons':[],'strategy':''},'monthly':{'action':'neutral','confidence':0,'reasons':[],'strategy':''},'yearly':{'action':'neutral','confidence':0,'reasons':[],'strategy':''}}

def calc_fundamentals(hist, symbol):
    """Temel verileri mevcut fiyat/hacim verisinden hesapla"""
    try:
        c = hist['Close'].values.astype(float)
        v = hist['Volume'].values.astype(float)
        h = hist['High'].values.astype(float)
        l = hist['Low'].values.astype(float)
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

        # Gunluk ortalama aralik (ATR benzeri)
        if n >= 14:
            daily_range = [(float(h[i]) - float(l[i])) for i in range(-14, 0)]
            avg_daily_range = sf(np.mean(daily_range))
            avg_daily_range_pct = sf(avg_daily_range / cur * 100)
        else:
            avg_daily_range = 0
            avg_daily_range_pct = 0

        # 52 haftalik high/low'dan uzaklik
        if n >= 252:
            hi52 = float(np.max(h[-252:]))
            lo52 = float(np.min(l[-252:]))
        else:
            hi52 = float(np.max(h))
            lo52 = float(np.min(l))
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
    """52 hafta (veya mevcut veri) high/low hesapla"""
    try:
        h=hist['High'].values.astype(float)
        l=hist['Low'].values.astype(float)
        c=float(hist['Close'].iloc[-1])
        hi52=sf(np.max(h)); lo52=sf(np.min(l))
        rng=hi52-lo52
        pos=sf((c-lo52)/rng*100 if rng>0 else 50)
        return {'high52w':hi52,'low52w':lo52,'currentPct':pos,'range':sf(rng)}
    except: return {'high52w':0,'low52w':0,'currentPct':50,'range':0}

def calc_all_indicators(hist, cp):
    c,h,l,v=hist['Close'].values.astype(float),hist['High'].values.astype(float),hist['Low'].values.astype(float),hist['Volume'].values.astype(float)
    cp=float(cp)
    rsi_h=[{'date':hist.index[i].strftime('%Y-%m-%d'),'value':rv} for i in range(14,len(c)) if (rv:=calc_rsi_single(c[:i+1])) is not None]
    ind={
        'rsi':calc_rsi(c),'rsiHistory':rsi_h,
        'macd':calc_macd(c),'macdHistory':calc_macd_history(c),
        'bollinger':calc_bollinger(c,cp),'bollingerHistory':calc_bollinger_history(c),
        'stochastic':calc_stochastic(c,h,l),
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
    }
    sigs=[x.get('signal','neutral') for x in ind.values() if isinstance(x,dict) and 'signal' in x]
    bc,sc=sigs.count('buy'),sigs.count('sell'); t=len(sigs)
    ind['summary']={'overall':'buy' if bc>sc and bc>=t*0.4 else ('sell' if sc>bc and sc>=t*0.4 else 'neutral'),'buySignals':bc,'sellSignals':sc,'neutralSignals':t-bc-sc,'totalIndicators':t}
    return ind

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
    return jsonify({
        'status': 'ok', 'version': '4.1.0', 'yf': YF_OK,
        'time': datetime.now().isoformat(),
        'loader': _status,
        'loaderStarted': _loader_started,
        'stockCache': len(_stock_cache),
        'indexCache': len(_index_cache),
        'cachedStocks': list(_stock_cache.keys()),
        'cachedIndices': list(_index_cache.keys()),
    })

@app.route('/api/debug')
def debug():
    """Detayli debug - Render loglarinda ne olduğunu goster"""
    stock_details = {}
    with _lock:
        for k, v in _stock_cache.items():
            age = round(time.time() - v['ts'], 1)
            stock_details[k] = {'price': v['data']['price'], 'age_sec': age}
    index_details = {}
    with _lock:
        for k, v in _index_cache.items():
            age = round(time.time() - v['ts'], 1)
            index_details[k] = {'value': v['data']['value'], 'age_sec': age}
    return jsonify({
        'loaderStarted': _loader_started,
        'status': _status,
        'stockCache': stock_details,
        'indexCache': index_details,
        'totalStocks': len(stock_details),
        'totalIndices': len(index_details),
        'yfinance': YF_OK,
        'time': datetime.now().isoformat(),
    })

@app.route('/')
def index():
    try: return send_from_directory(BASE_DIR, 'index.html')
    except: return jsonify({'error':'index.html bulunamadi'}),500

@app.route('/api/dashboard')
def dashboard():
    try:
        stocks=_get_stocks()
        if not stocks:
            return jsonify(safe_dict({'success':True,'loading':True,'stockCount':0,'message':f"Veriler yukleniyor ({_status['loaded']}/{_status['total']})...",'movers':{'topGainers':[],'topLosers':[],'volumeLeaders':[],'gapStocks':[]},'marketBreadth':{'advancing':0,'declining':0,'unchanged':0,'advDecRatio':0},'allStocks':[]}))
        sbc=sorted(stocks,key=lambda x:x.get('changePct',0),reverse=True)
        adv=sum(1 for s in stocks if s.get('changePct',0)>0)
        dec=sum(1 for s in stocks if s.get('changePct',0)<0)
        return jsonify(safe_dict({'success':True,'loading':False,'stockCount':len(stocks),'timestamp':datetime.now().isoformat(),'movers':{'topGainers':sbc[:5],'topLosers':sbc[-5:][::-1],'volumeLeaders':sorted(stocks,key=lambda x:x.get('volume',0),reverse=True)[:5],'gapStocks':sorted(stocks,key=lambda x:abs(x.get('gapPct',0)),reverse=True)[:5]},'marketBreadth':{'advancing':adv,'declining':dec,'unchanged':len(stocks)-adv-dec,'advDecRatio':sf(adv/dec if dec>0 else adv)},'allStocks':sbc}))
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
        return jsonify(safe_dict({'success':True,'stocks':stocks,'count':len(stocks),'sectors':list(SECTOR_MAP.keys())}))
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
        hist=_cget(_hist_cache, f"{symbol}_{period}")

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
        cp=float(hist['Close'].iloc[-1])
        prev=float(hist['Close'].iloc[-2]) if len(hist)>1 else cp
        w52=calc_52w(hist)
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
            'indicators':calc_all_indicators(hist,cp),
            'chartData':prepare_chart_data(hist),
            'fibonacci':calc_fibonacci(hist),
            'supportResistance':calc_support_resistance(hist),
            'pivotPoints':calc_pivot_points(hist),
            'recommendation':calc_recommendation(hist, None),
            'fundamentals':calc_fundamentals(hist, symbol),
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
        hist = _cget(_hist_cache, cache_key)

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
            hist = _cget(_hist_cache, f"{sym}_1y")
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

        return jsonify(safe_dict({
            'success': True,
            'results': {
                'totalReturn': total_return, 'cagr': cagr, 'sharpeRatio': sharpe,
                'maxDrawdown': sf(-max_dd), 'winRate': win_rate, 'totalTrades': total_trades,
                'buyAndHoldReturn': bh_return, 'alpha': alpha,
                'finalEquity': sf(final_equity), 'initialCapital': sf(initial_capital),
            },
            'equityCurve': equity_curve[::max(1, len(equity_curve)//200)],
            'trades': trades[-50:],
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
        db.close()

        if not user or user['password_hash'] != hash_password(password):
            return jsonify({'error': 'Kullanici adi veya sifre hatali'}), 401

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


@app.route('/api/docs')
def docs():
    return jsonify({'name': 'BIST Pro v6.0.0', 'endpoints': [
        '/api/health', '/api/debug', '/api/dashboard', '/api/indices',
        '/api/bist100', '/api/bist30', '/api/stock/<sym>', '/api/stock/<sym>/kap',
        '/api/commodity/<sym>', '/api/compare', '/api/screener', '/api/heatmap', '/api/report',
        '/api/backtest', '/api/sectors', '/api/search',
        '/api/auth/register', '/api/auth/login', '/api/auth/profile',
        '/api/portfolio', '/api/watchlist', '/api/alerts', '/api/alerts/check',
    ]})

# NO MODULE-LEVEL THREAD START - before_request handles it
print("[STARTUP] BIST Pro v6.0.0 ready - batch loader + SQLite + uyelik")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    print(f"[STARTUP] Port {port}")
    app.run(host='0.0.0.0', port=port)
