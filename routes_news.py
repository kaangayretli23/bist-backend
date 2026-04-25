"""
Haber, KAP ve Temel Analiz Endpoint'leri
Blueprint: news_bp
Prefix: /api/news, /api/fundamentals
"""
from flask import Blueprint, jsonify, request
from config import safe_dict, BIST100_STOCKS
from auth_middleware import require_user

news_bp = Blueprint('news', __name__)


# =====================================================================
# KAP / HABER ENDPOİNT'LERİ
# =====================================================================

@news_bp.route('/api/news/<symbol>')
def get_news(symbol: str):
    """
    Bir hisse için son haberler ve sentiment.
    GET /api/news/THYAO
    """
    try:
        from kap_scraper import fetch_kap_news, get_stock_sentiment
        symbol = symbol.upper()
        articles = fetch_kap_news(symbol, max_items=15)
        sentiment = get_stock_sentiment(symbol)
        return jsonify(safe_dict({
            'success':   True,
            'symbol':    symbol,
            'sentiment': sentiment,
            'articles':  articles[:15],
            'count':     len(articles),
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@news_bp.route('/api/news/sentiment/<symbol>')
def get_sentiment(symbol: str):
    """
    Detaylı haber sentiment analizi.
    GET /api/news/sentiment/GARAN
    """
    try:
        from news_sentiment import get_stock_news_sentiment
        symbol = symbol.upper()
        result = get_stock_news_sentiment(symbol)
        return jsonify(safe_dict({'success': True, **result}))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@news_bp.route('/api/news/market-sentiment')
def get_market_sentiment():
    """
    Genel piyasa sentiment skoru.
    GET /api/news/market-sentiment
    """
    try:
        from news_sentiment import get_market_sentiment
        result = get_market_sentiment()
        return jsonify(safe_dict({'success': True, **result}))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@news_bp.route('/api/news/scan', methods=['POST'])
def scan_news():
    """
    Birden fazla hisse için hızlı sentiment taraması.
    POST /api/news/scan  body: {"symbols": ["THYAO","GARAN"]}
    Boş body → ilk 20 BIST100 hissesi taranır.
    """
    try:
        from news_sentiment import get_stock_news_sentiment
        body    = request.get_json(silent=True) or {}
        symbols = body.get('symbols') or list(BIST100_STOCKS.keys())[:20]
        symbols = [s.upper() for s in symbols]

        results = []
        for sym in symbols:
            try:
                sent = get_stock_news_sentiment(sym)
                results.append(sent)
            except Exception:
                pass

        # Skora göre sırala
        results.sort(key=lambda x: -x.get('score', 0))
        return jsonify(safe_dict({'success': True, 'results': results, 'count': len(results)}))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =====================================================================
# TEMEL ANALİZ ENDPOİNT'LERİ
# =====================================================================

@news_bp.route('/api/fundamentals/<symbol>')
def get_fundamentals(symbol: str):
    """
    Bir hisse için temel analiz verileri.
    GET /api/fundamentals/THYAO
    """
    try:
        from fundamental_data import fetch_fundamentals, get_valuation_label
        symbol = symbol.upper()
        data   = fetch_fundamentals(symbol)
        if not data:
            return jsonify({'success': False, 'error': f'{symbol} için temel veri bulunamadı'}), 404
        return jsonify(safe_dict({'success': True, 'data': data,
                                  'valuation': get_valuation_label(symbol)}))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@news_bp.route('/api/fundamentals/<symbol>/message')
def get_fundamentals_message(symbol: str):
    """
    Telegram formatında temel analiz mesajı.
    GET /api/fundamentals/THYAO/message
    """
    try:
        from fundamental_data import format_fundamentals_message
        symbol = symbol.upper()
        msg    = format_fundamentals_message(symbol)
        return jsonify({'success': True, 'message': msg})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@news_bp.route('/api/fundamentals/scan', methods=['POST'])
def scan_fundamentals():
    """
    Birden fazla hisse için temel skor taraması.
    POST /api/fundamentals/scan  body: {"symbols": [...]}
    Boş body → BIST100'ün tamamı (yavaş, cache yok ise)
    """
    try:
        from fundamental_data import fetch_fundamentals_batch, get_valuation_label
        body    = request.get_json(silent=True) or {}
        symbols = body.get('symbols') or list(BIST100_STOCKS.keys())
        symbols = [s.upper() for s in symbols]

        results = fetch_fundamentals_batch(symbols, delay=0.2)
        output  = []
        for sym, data in results.items():
            output.append({
                'symbol':       sym,
                'temel_skor':   data.get('temel_skor', 0),
                'valuation':    get_valuation_label(sym),
                'fk':           data.get('fk'),
                'pd_dd':        data.get('pd_dd'),
                'temettü_verimi_pct': data.get('temettü_verimi_pct'),
                'temel_ozet':   data.get('temel_ozet', ''),
            })

        output.sort(key=lambda x: -x['temel_skor'])
        return jsonify(safe_dict({'success': True, 'results': output, 'count': len(output)}))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =====================================================================
# KOMBİNE ANALİZ
# =====================================================================

@news_bp.route('/api/analysis/full/<symbol>')
def full_analysis(symbol: str):
    """
    Tek endpointte teknik + temel + sentiment özeti.
    GET /api/analysis/full/THYAO
    """
    try:
        from fundamental_data import fetch_fundamentals, get_valuation_label
        from news_sentiment   import get_stock_news_sentiment
        from kap_scraper      import get_stock_sentiment as get_kap_sentiment
        from config           import _get_stocks

        symbol = symbol.upper()

        # Teknik veri (cache'den)
        stocks = _get_stocks([symbol])
        tech   = stocks[0] if stocks else {}

        # Temel veri
        fund   = fetch_fundamentals(symbol) or {}
        val    = get_valuation_label(symbol)

        # Sentiment
        news_s = get_stock_news_sentiment(symbol)
        kap_s  = get_kap_sentiment(symbol)

        return jsonify(safe_dict({
            'success': True,
            'symbol':  symbol,
            'technical': {
                'price':     tech.get('price'),
                'changePct': tech.get('changePct'),
            },
            'fundamental': {
                'temel_skor': fund.get('temel_skor', 0),
                'valuation':  val,
                'ozet':       fund.get('temel_ozet', ''),
                'fk':         fund.get('fk'),
                'pd_dd':      fund.get('pd_dd'),
                'temettü_verimi_pct': fund.get('temettü_verimi_pct'),
            },
            'news_sentiment': {
                'score':          news_s.get('score', 0),
                'label':          news_s.get('label', 'nötr'),
                'article_count':  news_s.get('article_count', 0),
                'top_headlines':  news_s.get('top_headlines', []),
            },
            'kap': {
                'important_count': kap_s.get('important_count', 0),
                'important_news':  kap_s.get('important_news', []),
            },
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# =====================================================================
# TELEGRAM BİLDİRİM TETİKLEYİCİLERİ
# =====================================================================

@news_bp.route('/api/news/send-report/<symbol>', methods=['POST'])
@require_user
def send_news_report(symbol: str):
    """
    Bir hisse için temel + haber raporunu Telegram'a gönder.
    POST /api/news/send-report/THYAO
    """
    try:
        from routes_telegram    import send_news_telegram as send_telegram
        from fundamental_data   import format_fundamentals_message
        from news_sentiment     import get_stock_news_sentiment
        from kap_scraper        import get_stock_sentiment as get_kap_sentiment

        symbol   = symbol.upper()
        fund_msg = format_fundamentals_message(symbol)
        news_s   = get_stock_news_sentiment(symbol)
        kap_s    = get_kap_sentiment(symbol)

        sentiment_line = (
            f"📰 Haber Sentiment: <b>{news_s.get('label','nötr').upper()}</b> "
            f"(skor {news_s.get('score',0):+.2f}  |  "
            f"{news_s.get('article_count',0)} haber)"
        )

        headlines = news_s.get('top_headlines', [])[:3]
        hl_text   = '\n'.join(f"  • {h}" for h in headlines)

        kap_line = ''
        if kap_s.get('important_count', 0) > 0:
            kap_line = f"\n🚨 <b>Önemli KAP duyurusu var!</b>"
            for n in kap_s.get('important_news', [])[:2]:
                kap_line += f"\n  📌 {n['title']}"

        full_msg = f"{fund_msg}\n\n{sentiment_line}\n{hl_text}{kap_line}"
        ok       = send_telegram(full_msg)
        return jsonify({'success': ok, 'symbol': symbol})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
