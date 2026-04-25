"""
Haber Sentiment Analizi
Google News RSS üzerinden Türkçe finansal haberler çeker,
keyword bazlı sentiment skoru üretir.
Sinyal motoruna ek girdi sağlar: -1.0 (çok negatif) → +1.0 (çok pozitif)
"""
import time
import threading
import xml.etree.ElementTree as ET

try:
    import requests as _req
except ImportError:
    _req = None

# =====================================================================
# TÜRKÇE FİNANSAL SENTIMENT KELİME LİSTELERİ
# =====================================================================

POSITIVE_KEYWORDS = {
    # Kârlılık
    'rekor kâr': 3, 'net kâr arttı': 3, 'kâr açıkladı': 2,
    'güçlü büyüme': 2, 'olumlu': 1, 'yükseliş': 1, 'artış': 1,
    # Kurumsal
    'temettü': 2, 'bedelsiz': 2, 'hisse geri alım': 2,
    'ortaklık': 1, 'anlaşma imzaladı': 2, 'sözleşme': 1,
    'ihracat rekoru': 3, 'kapasite artışı': 2,
    # Analist
    'hedef fiyat artırıldı': 2, 'al tavsiyesi': 2, 'endekse eklendi': 2,
    'yükseltildi': 1, 'güçlü tut': 1,
    # Piyasa
    'yükseldi': 1, 'güçlendi': 1, 'rallisi': 1, 'zirve': 1,
}

NEGATIVE_KEYWORDS = {
    # Kârlılık
    'zarar açıkladı': -3, 'net zarar': -3, 'kâr düştü': -2,
    'gelirler azaldı': -2, 'zayıf': -1, 'hayal kırıklığı': -2,
    # Hukuki/finansal sıkıntı
    'iflas': -4, 'konkordato': -4, 'temerrüt': -3, 'icra': -3,
    'soruşturma': -2, 'dava açıldı': -2, 'ceza': -2,
    'borç yapılandırması': -2, 'sermaye azaltımı': -2,
    # Analist
    'hedef fiyat düşürüldü': -2, 'sat tavsiyesi': -2,
    'endeksten çıkarıldı': -2, 'düşürüldü': -1,
    # Piyasa
    'düştü': -1, 'geriledi': -1, 'sattı': -1, 'çöküş': -3,
    # Makro/sektör
    'enflasyon baskısı': -1, 'kur riski': -1, 'faiz artışı': -1,
}

NEUTRAL_AMPLIFIERS = {
    'beklentinin üzerinde': 1, 'beklentinin altında': -1,
    'tahminlerin üzerinde': 1, 'tahminlerin altında': -1,
}

# =====================================================================
# CACHE
# =====================================================================
_sentiment_cache: dict[str, dict] = {}
_sentiment_lock = threading.Lock()
SENTIMENT_CACHE_TTL = 900   # 15 dakika (haberler sık güncellenir)

_market_sentiment_cache: dict = {}
_market_sentiment_ts: float = 0
MARKET_SENTIMENT_TTL = 1800  # 30 dakika


# =====================================================================
# HAM HABER ÇEKME
# =====================================================================

_rss_cache: dict[str, dict] = {}   # query → {items, fetched_at}
_rss_cache_lock = threading.Lock()
RSS_CACHE_TTL = 600   # 10 dk: news_sentiment ve kap_scraper arasında tek hit paylaşımı


def _fetch_news_rss(query: str, max_items: int = 30) -> list[dict]:
    """Google News RSS'ten ham haber listesi çek (query-bazlı 10dk cache — kap_scraper da kullanır)"""
    if _req is None:
        return []
    # Cache kontrol — aynı query son 10dk içinde çekildiyse yeniden çekme
    now = time.time()
    with _rss_cache_lock:
        entry = _rss_cache.get(query)
    if entry and (now - entry['fetched_at']) < RSS_CACHE_TTL:
        return entry['items'][:max_items]
    try:
        url = (
            f'https://news.google.com/rss/search'
            f'?q={query.replace(" ", "+")}&hl=tr&gl=TR&ceid=TR:tr'
        )
        resp = _req.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        if resp.status_code != 200:
            return []

        root = ET.fromstring(resp.text)
        items = []
        for item in root.findall('.//item')[:max_items]:
            title_el  = item.find('title')
            link_el   = item.find('link')
            date_el   = item.find('pubDate')
            source_el = item.find('source')

            if title_el is None:
                continue

            title = (title_el.text or '').strip()
            # Google News title genelde "Başlık - Kaynak" formatında
            if ' - ' in title:
                title = title.rsplit(' - ', 1)[0].strip()

            items.append({
                'title':     title,
                'url':       link_el.text if link_el is not None else '',
                'published': date_el.text if date_el is not None else '',
                'source':    source_el.text if source_el is not None else '',
            })
        with _rss_cache_lock:
            _rss_cache[query] = {'items': items, 'fetched_at': time.time()}
            # Cap: farklı query sayısı sınırsız büyümesin
            if len(_rss_cache) > 500:
                oldest = sorted(_rss_cache.items(), key=lambda x: x[1]['fetched_at'])
                for k, _ in oldest[:100]:
                    _rss_cache.pop(k, None)
        return items
    except Exception:
        return []


# =====================================================================
# SENTIMENT HESAPLAMA
# =====================================================================

def _score_text(text: str) -> float:
    """Bir başlık için ham sentiment skoru (-10 to +10)"""
    text_lower = text.lower()
    score = 0.0

    for phrase, weight in POSITIVE_KEYWORDS.items():
        if phrase in text_lower:
            score += weight

    for phrase, weight in NEGATIVE_KEYWORDS.items():
        if phrase in text_lower:
            score += weight

    for phrase, weight in NEUTRAL_AMPLIFIERS.items():
        if phrase in text_lower:
            score += weight

    return score


def _normalize_score(raw: float, article_count: int) -> float:
    """
    Ham skoru -1.0 → +1.0 aralığına normalize et.
    Makul bir baseline: makası ±5 alıyoruz.
    """
    if article_count == 0:
        return 0.0
    avg = raw / max(article_count, 1)
    return max(-1.0, min(1.0, avg / 5.0))


def _sentiment_label(score: float) -> str:
    if score >= 0.4:
        return 'çok pozitif'
    elif score >= 0.15:
        return 'pozitif'
    elif score <= -0.4:
        return 'çok negatif'
    elif score <= -0.15:
        return 'negatif'
    return 'nötr'


# =====================================================================
# ANA FONKSİYONLAR
# =====================================================================

def get_stock_news_sentiment(symbol: str) -> dict:
    """
    Bir hisse için haber sentiment analizi yap.
    Returns:
        symbol, score (-1 to 1), label, article_count,
        positive_count, negative_count, top_headlines
    """
    with _sentiment_lock:
        cached = _sentiment_cache.get(symbol)
        if cached and time.time() - cached['fetched_at'] < SENTIMENT_CACHE_TTL:
            return cached['result']

    # İki sorgu ile kapsamlı haber toplama
    articles = _fetch_news_rss(f'{symbol} hisse borsa')
    articles += _fetch_news_rss(f'{symbol} şirket')

    # Tekrarlı başlıkları filtrele
    seen_titles = set()
    unique = []
    for a in articles:
        key = a['title'][:60]
        if key not in seen_titles:
            seen_titles.add(key)
            unique.append(a)

    total_raw   = 0.0
    pos_count   = 0
    neg_count   = 0
    top_headlines = []

    scored = []
    for a in unique:
        s = _score_text(a['title'])
        a['sentiment_score'] = s
        total_raw += s
        if s > 0:
            pos_count += 1
        elif s < 0:
            neg_count += 1
        scored.append(a)

    # En çarpıcı haberleri öne al (mutlak skor büyüklüğüne göre)
    scored.sort(key=lambda x: -abs(x['sentiment_score']))
    top_headlines = [a['title'] for a in scored[:5]]

    norm_score = _normalize_score(total_raw, len(unique))
    result = {
        'symbol':          symbol,
        'score':           round(norm_score, 3),
        'label':           _sentiment_label(norm_score),
        'article_count':   len(unique),
        'positive_count':  pos_count,
        'negative_count':  neg_count,
        'top_headlines':   top_headlines,
        'fetched_at':      time.time(),
    }

    with _sentiment_lock:
        _sentiment_cache[symbol] = {'result': result, 'fetched_at': time.time()}

    return result


def get_market_sentiment() -> dict:
    """
    Genel piyasa sentiment'i (BIST, borsa, ekonomi haberleri).
    Returns {score, label, article_count, summary}
    """
    global _market_sentiment_cache, _market_sentiment_ts

    if time.time() - _market_sentiment_ts < MARKET_SENTIMENT_TTL and _market_sentiment_cache:
        return _market_sentiment_cache

    queries = [
        'BIST borsa istanbul bugün',
        'borsa hisse senedi piyasa',
        'Türkiye ekonomi enflasyon faiz',
    ]

    all_articles = []
    for q in queries:
        all_articles += _fetch_news_rss(q, max_items=20)

    total_raw = 0.0
    pos_count = 0
    neg_count = 0
    for a in all_articles:
        s = _score_text(a['title'])
        total_raw += s
        if s > 0:
            pos_count += 1
        elif s < 0:
            neg_count += 1

    norm = _normalize_score(total_raw, len(all_articles))
    result = {
        'score':          round(norm, 3),
        'label':          _sentiment_label(norm),
        'article_count':  len(all_articles),
        'positive_count': pos_count,
        'negative_count': neg_count,
    }

    _market_sentiment_cache = result
    _market_sentiment_ts = time.time()
    return result


def get_sentiment_score_for_signal(symbol: str) -> float:
    """
    Sinyal motoruna eklenecek sentiment katkısı.
    Returns -0.5 → +0.5 (teknik skora toplanır)
    """
    try:
        sent = get_stock_news_sentiment(symbol)
        return round(sent['score'] * 0.5, 3)
    except Exception:
        return 0.0


# =====================================================================
# CACHE YÖNETİMİ
# =====================================================================

def clear_sentiment_cache(symbol: str = None):
    with _sentiment_lock:
        if symbol:
            _sentiment_cache.pop(symbol, None)
        else:
            _sentiment_cache.clear()
