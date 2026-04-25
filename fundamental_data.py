"""
Temel Analiz Verisi — Yahoo Finance (HABERLER/RAPORLAMA İÇİN)

Bu modül: Sadece yfinance, 24h cache, Türkçe alan adları.
Döndürdüğü alanlar: fk, pd_dd, piyasa_degeri, temettü_verimi, eps, sektor, sirket_adi, ciro, net_kar.
Haber raporları ve Telegram metinlerinde okunabilir format için kullanılır.

⚠ Bkz. signals_fundamental.py: AYNI İŞ AMA FARKLI MODÜL!
signals_fundamental.py → İş Yatırım birincil + yfinance fallback, İngilizce alanlar (pe, pb, roe),
    sinyal skoru pipeline'ı için. Bu iki modülü KARIŞTIRMA.
"""
import time
import threading

try:
    import yfinance as yf
except ImportError:
    yf = None

# =====================================================================
# CACHE
# =====================================================================
_fund_cache: dict[str, dict] = {}   # symbol → {data, fetched_at}
_fund_lock = threading.Lock()
FUND_CACHE_TTL = 86400   # 24 saat (temel veriler günlük değişir)
FUND_FETCH_TIMEOUT = 12  # saniye

# =====================================================================
# YF ALAN HARİTASI
# =====================================================================
_YF_FIELDS = {
    'trailingPE':        'fk',           # F/K oranı
    'priceToBook':       'pd_dd',        # PD/DD oranı
    'marketCap':         'piyasa_degeri',
    'dividendYield':     'temettü_verimi',
    'trailingEps':       'eps',
    'totalRevenue':      'ciro',
    'netIncomeToCommon': 'net_kar',
    'debtToEquity':      'borc_oz_kaynak',
    'returnOnEquity':    'roe',
    'currentRatio':      'cari_oran',
    'sector':            'sektor',
    'industry':          'endustri',
    'longName':          'sirket_adi',
    'country':           'ulke',
    'fullTimeEmployees': 'calisan_sayisi',
}

# F/K değerlendirme eşikleri (BIST için uyarlandı)
FK_CHEAP   = 10   # F/K < 10 → çok ucuz
FK_FAIR    = 20   # F/K 10-20 → makul
FK_PRICEY  = 35   # F/K > 35 → pahalı

PD_DD_CHEAP = 1.5
PD_DD_FAIR  = 3.0

DIV_GOOD    = 3.0  # %3+ temettü verimi iyi


# =====================================================================
# ÇEKME FONKSİYONLARI
# =====================================================================

def fetch_fundamentals(symbol: str) -> dict | None:
    """
    Hisse için temel verileri çek (önce cache'e bak).
    Returns dict veya None (başarısız ise).
    """
    # Cache kontrolü
    with _fund_lock:
        cached = _fund_cache.get(symbol)
        if cached and time.time() - cached['fetched_at'] < FUND_CACHE_TTL:
            return cached['data']

    if yf is None:
        return None

    try:
        ticker = yf.Ticker(f"{symbol}.IS")
        info   = ticker.info or {}

        if not info or info.get('regularMarketPrice') is None and not info.get('trailingPE'):
            return None

        data = {'symbol': symbol, 'fetched_at': time.time()}
        for yf_key, local_key in _YF_FIELDS.items():
            val = info.get(yf_key)
            data[local_key] = val

        # Piyasa değerini milyar TL'ye çevir (ham değer TL cinsinden)
        if data.get('piyasa_degeri'):
            data['piyasa_degeri_milyar'] = round(data['piyasa_degeri'] / 1e9, 2)
        else:
            data['piyasa_degeri_milyar'] = None

        # Temettü verimini yüzdeye çevir
        if data.get('temettü_verimi'):
            data['temettü_verimi_pct'] = round(data['temettü_verimi'] * 100, 2)
        else:
            data['temettü_verimi_pct'] = None

        # Temel skor hesapla
        data['temel_skor'] = _calc_fundamental_score(data)
        data['temel_ozet'] = _build_summary(data)

        with _fund_lock:
            _fund_cache[symbol] = {'data': data, 'fetched_at': time.time()}

        return data

    except Exception as e:
        print(f"[FUND] {symbol} veri çekme hatası: {e}")
        return None


def fetch_fundamentals_batch(symbols: list[str], delay: float = 0.3) -> dict[str, dict]:
    """
    Birden fazla hisse için temel veri çek.
    Returns {symbol: data}
    """
    result = {}
    for sym in symbols:
        data = fetch_fundamentals(sym)
        if data:
            result[sym] = data
        time.sleep(delay)
    return result


# =====================================================================
# TEMEL SKOR (0–3 arası, teknik skora eklenebilir)
# =====================================================================

def _calc_fundamental_score(data: dict) -> float:
    """
    0-3 arası temel analiz skoru.
    3 = çok güçlü temel, 0 = zayıf veya veri yok.
    """
    score = 0.0

    # F/K değerlendirmesi (+0-1)
    fk = data.get('fk')
    if fk and fk > 0:
        if fk < FK_CHEAP:
            score += 1.0
        elif fk < FK_FAIR:
            score += 0.6
        elif fk < FK_PRICEY:
            score += 0.2
        # F/K > 35 → 0 puan

    # PD/DD değerlendirmesi (+0-0.75)
    pd_dd = data.get('pd_dd')
    if pd_dd and pd_dd > 0:
        if pd_dd < PD_DD_CHEAP:
            score += 0.75
        elif pd_dd < PD_DD_FAIR:
            score += 0.4
        elif pd_dd < 5.0:
            score += 0.1

    # Temettü verimi (+0-0.5)
    div = data.get('temettü_verimi_pct')
    if div and div > 0:
        if div >= DIV_GOOD:
            score += 0.5
        elif div >= 1.0:
            score += 0.25

    # ROE değerlendirmesi (+0-0.5)
    roe = data.get('roe')
    if roe and roe > 0:
        if roe >= 0.20:   # %20+ ROE
            score += 0.5
        elif roe >= 0.10:
            score += 0.25

    # Borç/öz kaynak cezası (-0.25)
    de = data.get('borc_oz_kaynak')
    if de and de > 150:   # çok yüksek kaldıraç
        score -= 0.25

    return round(min(max(score, 0), 3), 2)


def _build_summary(data: dict) -> str:
    """Temel verilerin kısa metin özeti"""
    parts = []

    fk = data.get('fk')
    if fk and fk > 0:
        parts.append(f"F/K={fk:.1f}")

    pd_dd = data.get('pd_dd')
    if pd_dd and pd_dd > 0:
        parts.append(f"PD/DD={pd_dd:.1f}")

    div = data.get('temettü_verimi_pct')
    if div and div > 0:
        parts.append(f"Temettü=%{div:.1f}")

    roe = data.get('roe')
    if roe:
        parts.append(f"ROE=%{roe*100:.0f}")

    mv = data.get('piyasa_degeri_milyar')
    if mv:
        parts.append(f"PD={mv:.1f}Mrd TL")

    return '  |  '.join(parts) if parts else 'Veri yok'


# =====================================================================
# KOLAY ERİŞİM FONKSİYONLARI
# =====================================================================

def get_fundamental_score(symbol: str) -> float:
    """Sadece temel skoru döndür (0-3). Hızlı erişim için."""
    data = fetch_fundamentals(symbol)
    return data.get('temel_skor', 0.0) if data else 0.0


def get_valuation_label(symbol: str) -> str:
    """
    'ucuz' | 'makul' | 'pahalı' | 'bilinmiyor'
    F/K ve PD/DD'ye göre değerleme etiketi.
    """
    data = fetch_fundamentals(symbol)
    if not data:
        return 'bilinmiyor'

    fk    = data.get('fk', 0) or 0
    pd_dd = data.get('pd_dd', 0) or 0
    score = data.get('temel_skor', 0)

    if score >= 2.0:
        return 'ucuz'
    elif score >= 1.0:
        return 'makul'
    elif fk > FK_PRICEY or pd_dd > 6.0:
        return 'pahalı'
    return 'makul'


def format_fundamentals_message(symbol: str) -> str:
    """Telegram veya UI için formatlanmış temel veri mesajı"""
    data = fetch_fundamentals(symbol)
    if not data:
        return f"<b>{symbol}</b>: Temel veri bulunamadı."

    lines = [
        f"📊 <b>{symbol} — Temel Analiz</b>",
        f"🏢 {data.get('sirket_adi', symbol)}",
        f"━━━━━━━━━━━━━━━━━━",
    ]

    if data.get('fk'):
        lines.append(f"F/K Oranı: <b>{data['fk']:.1f}x</b>")
    if data.get('pd_dd'):
        lines.append(f"PD/DD: <b>{data['pd_dd']:.1f}x</b>")
    if data.get('temettü_verimi_pct'):
        lines.append(f"Temettü Verimi: <b>%{data['temettü_verimi_pct']:.2f}</b>")
    if data.get('eps'):
        lines.append(f"EPS: {data['eps']:.2f} TL")
    if data.get('roe'):
        lines.append(f"ROE: %{data['roe']*100:.1f}")
    if data.get('borc_oz_kaynak'):
        lines.append(f"Borç/Öz Kaynak: {data['borc_oz_kaynak']:.0f}%")
    if data.get('cari_oran'):
        lines.append(f"Cari Oran: {data['cari_oran']:.2f}")
    if data.get('piyasa_degeri_milyar'):
        lines.append(f"Piyasa Değeri: {data['piyasa_degeri_milyar']:.1f} Mrd TL")
    if data.get('ciro'):
        lines.append(f"Ciro: {data['ciro']/1e9:.1f} Mrd TL")

    lines += [
        f"━━━━━━━━━━━━━━━━━━",
        f"⭐ Temel Skor: <b>{data['temel_skor']:.1f}/3.0</b>  "
        f"({get_valuation_label(symbol).upper()})",
        f"📝 {data.get('temel_ozet', '')}",
    ]

    return '\n'.join(lines)


# =====================================================================
# CACHE TEMİZLEME
# =====================================================================

def clear_fundamental_cache(symbol: str = None):
    """Cache'i temizle. symbol=None ise tümünü temizler."""
    with _fund_lock:
        if symbol:
            _fund_cache.pop(symbol, None)
        else:
            _fund_cache.clear()
