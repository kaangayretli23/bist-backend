"""
Şirket Haber Monitörü (Google News RSS)
NOT: Modül ismi 'kap_scraper' olsa da, gerçek KAP (kap.org.tr) API'sine bağlanmaz.
Google News RSS üzerinden hisse bazlı KAP/duyuru/önemli haberleri keyword filtreyle tarar.
Resmi KAP bildirimi yerine indirekt (haber sitelerinin KAP haberi raporlaması) kullanılır.
Önemli duyurularda Telegram bildirimi gönderir. Arka planda 30 dakikada bir portföyü tarar.
"""
import time
import threading
import hashlib
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

try:
    import requests as _req
except ImportError:
    _req = None

# =====================================================================
# ANAHTAR KELİMELER
# =====================================================================

# Bu kelimelerden biri başlıkta varsa → önemli KAP duyurusu sayılır
KAP_KEYWORDS = [
    'temettü', 'kar payı', 'bedelsiz', 'sermaye artırımı', 'sermaye azaltımı',
    'birleşme', 'devralma', 'satın alma', 'ortaklık', 'genel kurul',
    'kap bildirdi', 'özel durum', 'içsel bilgi', 'pay geri alım',
    'ihraç', 'halka arz', 'borsaya kotasyon', 'endeks', 'yönetim değişikliği',
    'ceo', 'genel müdür', 'iflas', 'konkordato', 'temerrüt', 'icra',
    'soruşturma', 'dava', 'ceza', 'sözleşme', 'anlaşma', 'ihale',
    'kapasite artışı', 'fabrika', 'yatırım', 'rekor', 'zarar açıkladı',
]

# Pozitif/negatif ağırlık
POSITIVE_WORDS = {
    'temettü': 2, 'bedelsiz': 2, 'kar payı': 2, 'rekor': 2,
    'sermaye artırımı': 1, 'sözleşme': 1, 'anlaşma': 1,
    'yatırım': 1, 'büyüme': 1, 'kapasite artışı': 1,
}
NEGATIVE_WORDS = {
    'iflas': -3, 'konkordato': -3, 'temerrüt': -2, 'icra': -2,
    'soruşturma': -2, 'zarar açıkladı': -2, 'ceza': -1,
    'dava': -1, 'sermaye azaltımı': -1,
}

# =====================================================================
# CACHE
# =====================================================================
_seen_articles: dict[str, float] = {}   # url_hash → timestamp
_seen_lock = threading.Lock()
_SEEN_MAX = 20000   # Hard cap; aşılırsa en eski %20 atılır
_kap_cache: dict[str, dict] = {}        # symbol → {articles, fetched_at}
_kap_cache_lock = threading.Lock()
KAP_CACHE_TTL = 1800   # 30 dakika

# Haber bildirimi cooldown: aynı hisse için kaç saniyede bir bildirim atılabilir
_news_notified: dict[str, float] = {}   # symbol → son bildirim timestamp
_news_notified_lock = threading.Lock()
NEWS_NOTIFY_COOLDOWN = 21600  # 6 saat — aynı hisse için AL önerisi tekrarı

_kap_thread_started = False
_kap_thread_lock = threading.Lock()


# =====================================================================
# YARDIMCI FONKSİYONLAR
# =====================================================================

def _url_hash(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()


def _clean_title(title: str) -> str:
    """HTML entity ve gereksiz karakterleri temizle"""
    replacements = {
        '&#39;': "'", '&amp;': '&', '&lt;': '<', '&gt;': '>',
        '&quot;': '"', '&nbsp;': ' ',
    }
    for k, v in replacements.items():
        title = title.replace(k, v)
    return title.strip()


def _score_article(title: str) -> int:
    """Başlığa göre önem skoru hesapla. 0 = önemsiz, >0 = önemli"""
    title_lower = title.lower()
    score = 0
    for word, weight in POSITIVE_WORDS.items():
        if word in title_lower:
            score += weight
    for word, weight in NEGATIVE_WORDS.items():
        if word in title_lower:
            score += weight  # weight zaten negatif
    # KAP kelimesi varsa +1 bonus
    if 'kap' in title_lower:
        score += 1
    return score


def _is_important(title: str) -> bool:
    """Başlık önemli bir duyuru mu?"""
    title_lower = title.lower()
    return any(kw in title_lower for kw in KAP_KEYWORDS)


# =====================================================================
# HABER ÇEKME
# =====================================================================

def fetch_kap_news(symbol: str, max_items: int = 20) -> list[dict]:
    """
    Bir hisse için Google News RSS'ten son KAP/önemli haberlerini çek.
    Returns list of {title, url, published, score, is_important}
    """
    # Cache kontrol
    with _kap_cache_lock:
        cached = _kap_cache.get(symbol)
        if cached and time.time() - cached['fetched_at'] < KAP_CACHE_TTL:
            return cached['articles']

    if _req is None:
        return []

    articles = []
    # İki sorgu: genel haber + KAP odaklı
    # news_sentiment._fetch_news_rss kullanırız — query-bazlı ortak cache ile duplicate fetch yok
    try:
        from news_sentiment import _fetch_news_rss
    except Exception:
        _fetch_news_rss = None

    queries = [
        f'{symbol} hisse borsa',
        f'{symbol} KAP bildiri temettü sermaye',
    ]

    seen_urls = set()
    for query in queries:
        try:
            if _fetch_news_rss is not None:
                raw_items = _fetch_news_rss(query, max_items=max_items)
            else:
                raw_items = []

            for it in raw_items:
                title    = _clean_title(it.get('title', ''))
                link     = it.get('url', '')
                pub_date = it.get('published', '')
                if link in seen_urls:
                    continue
                seen_urls.add(link)

                score = _score_article(title)
                articles.append({
                    'symbol':       symbol,
                    'title':        title,
                    'url':          link,
                    'published':    pub_date,
                    'score':        score,
                    'is_important': _is_important(title),
                })
        except Exception:
            continue

    # Skora göre sırala
    articles.sort(key=lambda x: (-x['score'], x['published']), reverse=False)
    articles.sort(key=lambda x: -x['score'])

    with _kap_cache_lock:
        _kap_cache[symbol] = {'articles': articles, 'fetched_at': time.time()}

    return articles


def get_stock_sentiment(symbol: str) -> dict:
    """
    Hisse için özet sentiment döndür.
    Returns {symbol, score, label, important_count, article_count, top_headlines}
    """
    articles = fetch_kap_news(symbol)
    if not articles:
        return {'symbol': symbol, 'score': 0, 'label': 'nötr', 'important_count': 0,
                'article_count': 0, 'top_headlines': []}

    total_score   = sum(a['score'] for a in articles)
    important     = [a for a in articles if a['is_important']]
    top_headlines = [a['title'] for a in articles[:3]]

    if total_score >= 4:
        label = 'pozitif'
    elif total_score <= -2:
        label = 'negatif'
    else:
        label = 'nötr'

    return {
        'symbol':          symbol,
        'score':           total_score,
        'label':           label,
        'important_count': len(important),
        'article_count':   len(articles),
        'top_headlines':   top_headlines,
        'important_news':  [{'title': a['title'], 'published': a['published']} for a in important[:5]],
    }


# =====================================================================
# ARKA PLAN MONİTÖR
# =====================================================================

def _check_new_important_news(symbol: str) -> list[dict]:
    """Daha önce görülmemiş önemli haberleri döndür"""
    articles = fetch_kap_news(symbol)
    new_important = []
    now = time.time()

    with _seen_lock:
        for a in articles:
            if not a['is_important']:
                continue
            h = _url_hash(a['url'])
            if h not in _seen_articles:
                _seen_articles[h] = now
                new_important.append(a)

    return new_important


def _format_kap_alert(symbol: str, article: dict) -> str:
    score = article['score']
    if score >= 2:
        emoji = '🚨'
    elif score >= 1:
        emoji = '📢'
    elif score <= -2:
        emoji = '🔴'
    elif score <= -1:
        emoji = '⚠️'
    else:
        emoji = '📰'

    return (
        f"{emoji} <b>KAP/HABER: {symbol}</b>\n"
        f"📰 {article['title']}\n"
        f"🕐 {article['published'][:25] if article['published'] else ''}"
    )


def _kap_monitor_loop():
    """
    Arka plan thread'i.
    Her 30 dakikada BIST100 hisselerini tarar, yeni önemli duyuruları Telegram'a gönderir.
    İlk çalışmada mevcut haberleri 'görüldü' olarak işaretler (spam engeli).
    """
    from config import BIST100_STOCKS

    # İlk çalışma: mevcut haberleri görüldü olarak işaretle, bildirim gönderme
    first_run = True
    print("[KAP] Monitör başlatıldı — ilk tarama haberleri seeding ediyor")

    while True:
        try:
            symbols = list(BIST100_STOCKS.keys())

            for symbol in symbols:
                try:
                    articles = fetch_kap_news(symbol)
                    with _seen_lock:
                        for a in articles:
                            if a['is_important']:
                                h = _url_hash(a['url'])
                                if h not in _seen_articles:
                                    if first_run:
                                        # Seed: sadece kaydet, bildirim gönderme
                                        _seen_articles[h] = time.time()
                                    else:
                                        _seen_articles[h] = time.time()
                                        _evaluate_and_notify(symbol, a)
                except Exception:
                    pass
                time.sleep(0.5)  # Rate limit

            first_run = False
            print(f"[KAP] Tarama tamamlandı: {len(symbols)} hisse | "
                  f"{datetime.now().strftime('%H:%M')}")

            # Eski 'görüldü' kayıtlarını temizle (7 günden eski)
            cutoff = time.time() - 7 * 86400
            with _seen_lock:
                stale = [h for h, ts in _seen_articles.items() if ts < cutoff]
                for h in stale:
                    del _seen_articles[h]
                # Hard cap: büyük burst olursa en eski %20'yi at
                if len(_seen_articles) > _SEEN_MAX:
                    sorted_items = sorted(_seen_articles.items(), key=lambda x: x[1])
                    drop_n = len(sorted_items) // 5
                    for h, _ in sorted_items[:drop_n]:
                        _seen_articles.pop(h, None)

        except Exception as e:
            print(f"[KAP] Monitor hatası: {e}")

        time.sleep(1800)  # 30 dakika bekle


def _get_signal_for_symbol(symbol: str) -> dict:
    """
    Hisse için hızlı sinyal hesapla.
    Returns: {action, score, confidence, price} veya None
    """
    try:
        from config import _cget_hist, sf
        from indicators import calc_all_indicators
        from signals import calc_recommendation, calc_ml_confidence

        hist = _cget_hist(f"{symbol}_1y")
        if hist is None or len(hist) < 30:
            return None

        cp = float(hist['Close'].values[-1])
        ind = calc_all_indicators(hist, cp)
        rec = calc_recommendation(hist, ind, symbol=symbol)

        # Haftalık sinyal kullan (swing trade odaklı)
        tf_rec = rec.get('weekly', rec.get('daily', {}))
        action = tf_rec.get('action', 'NOTR')
        score = float(tf_rec.get('score', 0))
        confidence = float(tf_rec.get('confidence', 0))

        return {
            'action': action,
            'score': score,
            'confidence': confidence,
            'price': sf(cp),
        }
    except Exception as e:
        print(f"[KAP] Sinyal hesaplama hatası ({symbol}): {e}")
        return None


def _is_in_portfolio(symbol: str) -> bool:
    """Hisse aktif auto_positions'da var mı? (Postgres + SQLite uyumlu)"""
    try:
        from config import db_conn
        with db_conn() as conn:
            # PgConnection ve sqlite3.Connection ikisinde de execute() arayüzü var;
            # cursor() yerine doğrudan execute kullan (PgConnection wrapper cursor() desteklemiyor)
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM auto_positions WHERE symbol=? AND status='open'",
                (symbol,)
            ).fetchone()
            if row is None:
                return False
            # PgConnection → dict-like, sqlite3.Row → int-indexable
            try:
                count = int(row['c']) if 'c' in (row.keys() if hasattr(row, 'keys') else []) else int(row[0])
            except Exception:
                count = int(row[0])
            return count > 0
    except Exception:
        return False


def _check_news_cooldown(symbol: str, kind: str) -> bool:
    """True dönerse bildirim gönderilebilir, False ise cooldown devam ediyor."""
    key = f"{kind}_{symbol}"
    now = time.time()
    with _news_notified_lock:
        last = _news_notified.get(key, 0)
        if now - last < NEWS_NOTIFY_COOLDOWN:
            remaining = int((NEWS_NOTIFY_COOLDOWN - (now - last)) / 3600)
            print(f"[KAP] {symbol} {kind} cooldown — {remaining}s kaldı, sessiz kalındı")
            return False
        _news_notified[key] = now
        return True


def _evaluate_and_notify(symbol: str, article: dict):
    """
    Haber + sinyal korelasyonu yaparak Telegram bildirimi gönder.
    - Pozitif haber + AL sinyali → AL fırsatı (6 saatte 1 kez)
    - Negatif haber + portföyde var → Uyarı (6 saatte 1 kez)
    - Diğer durumlarda → sessiz kal
    """
    try:
        from routes_telegram import send_news_telegram, send_telegram

        news_score = article['score']

        # --- POZİTİF HABER: AL fırsatı araştır ---
        # Eşikler: haber skoru >= 2 (gerçekten önemli haber) VE
        #          sinyal skoru >= 7 VE güven >= 65% (auto-trade standardı)
        if news_score >= 2:
            if not _check_news_cooldown(symbol, 'AL'):
                return
            sig = _get_signal_for_symbol(symbol)
            if (sig
                    and sig['action'] in ('AL', 'GÜÇLÜ AL')
                    and sig['score'] >= 7
                    and sig['confidence'] >= 65):
                strength_label = '🚀 GÜÇLÜ AL' if sig['action'] == 'GÜÇLÜ AL' else '✅ AL'
                msg = (
                    f"📰 <b>HABER DESTEKLİ AL FIRSATI: {symbol}</b>\n\n"
                    f"📣 <b>Haber:</b> {article['title']}\n\n"
                    f"📊 <b>Sinyal:</b> {strength_label} "
                    f"(skor: {sig['score']:.1f}/10, güven: %{sig['confidence']:.0f})\n"
                    f"💰 <b>Fiyat:</b> {sig['price']} TL\n\n"
                    f"🕐 {article['published'][:25] if article['published'] else ''}"
                )
                send_news_telegram(msg)
                print(f"[KAP] AL bildirimi gönderildi: {symbol} | haber: {news_score} | skor: {sig['score']:.1f} | güven: %{sig['confidence']:.0f}")
            else:
                with _news_notified_lock:
                    _news_notified.pop(f'AL_{symbol}', None)
                reason = 'sinyal yok' if not sig else f"skor={sig.get('score',0):.1f} güven=%{sig.get('confidence',0):.0f}"
                print(f"[KAP] {symbol}: pozitif haber ama eşik altı ({reason}) — cooldown geri alındı")

        # --- NEGATİF HABER: Portföy uyarısı ---
        elif news_score < 0 and _is_in_portfolio(symbol):
            if not _check_news_cooldown(symbol, 'NEG'):
                return
            severity = '🔴 KRİTİK' if news_score <= -2 else '⚠️ DİKKAT'
            msg = (
                f"{severity} <b>PORTFÖYDEKİ HİSSEDE OLUMSUZ HABER: {symbol}</b>\n\n"
                f"📰 {article['title']}\n"
                f"🕐 {article['published'][:25] if article['published'] else ''}"
            )
            # Negatif haber portföy uyarısı → trade chat'e de git (kritik olabilir)
            send_telegram(msg)
            send_news_telegram(msg)
            print(f"[KAP] Uyarı gönderildi: {symbol} | haber skoru: {news_score}")

        else:
            print(f"[KAP] {symbol}: haber filtrelendi (skor: {news_score})")

    except Exception as e:
        print(f"[KAP] Değerlendirme hatası ({symbol}): {e}")


def start_kap_monitor():
    """KAP monitör thread'ini başlat (tek seferlik)"""
    global _kap_thread_started
    with _kap_thread_lock:
        if _kap_thread_started:
            return
        _kap_thread_started = True

    t = threading.Thread(target=_kap_monitor_loop, daemon=True)
    t.start()
    print("[KAP] Duyuru monitörü başlatıldı")
