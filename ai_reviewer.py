"""
ai_reviewer.py — OpenAI ikincil AI filtre/inceleme katmanı (Kaan, lokal tek kullanıcı).

GÜVENLİK SÖZLEŞMESİ:
  • Sadece analiz/yorum üretir. ASLA pozisyon açmaz, broker'a/Midas'a bağlanmaz, emir vermez.
  • Nihai karar her zaman kullanıcı onayındadır (Telegram approve/reject).
  • OPENAI_API_KEY yalnızca backend .env'de tutulur; frontend'e asla gönderilmez.
  • Maliyet kontrolü: günlük çağrı limiti + aylık tahmini USD limiti + sembol cooldown/cache.
  • KEY yoksa / disabled ise / limit doluysa: hiç çağrı yapılmaz, güvenli fallback döner.

Model varsayılanı OPENAI_MODEL (env). Fiyat env'den override edilebilir
(AI_PRICE_INPUT_PER_1M / AI_PRICE_OUTPUT_PER_1M) — model fiyatı değişirse tek satırla güncellenir.
"""
import json
import os
import time
from datetime import datetime

from config import get_db


# =====================================================================
# CONFIG (env)
# =====================================================================
def _env(key, default=None):
    return os.environ.get(key, default)


def _env_int(key, default):
    try:
        return int(_env(key, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(key, default):
    try:
        return float(_env(key, str(default)))
    except (TypeError, ValueError):
        return default


def _model():
    return _env('OPENAI_MODEL', 'gpt-5.4-mini')


def is_ai_review_enabled():
    """AI review açık mı? Hem toggle hem API key şart (key yoksa hiç deneme)."""
    return _env('AI_REVIEW_ENABLED', '0') == '1' and bool(_env('OPENAI_API_KEY'))


def key_status():
    """API key'in GÜVENLİ durumu — değer ASLA dönmez, sadece var/yok+prefix+uzunluk."""
    k = _env('OPENAI_API_KEY')
    if not k:
        return {'loaded': False, 'prefix': None, 'length': 0}
    k = k.strip()
    return {'loaded': True, 'prefix': k[:8] + '...', 'length': len(k)}


def log_key_status():
    """Başlangıçta güvenli tek satır log (değer yazılmaz)."""
    s = key_status()
    if s['loaded']:
        print(f"[AI-REVIEW] OPENAI_API_KEY loaded: true | prefix: {s['prefix']} | "
              f"length: {s['length']} | model: {_model()} | enabled: {is_ai_review_enabled()}")
    else:
        print("[AI-REVIEW] OPENAI_API_KEY loaded: false — .env kontrol et "
              "(C:\\Users\\Kaan\\bist-backend\\.env)")


# OpenAI hata tipleri (opsiyonel import — paket yoksa None)
try:
    from openai import AuthenticationError as _OpenAIAuthError, NotFoundError as _OpenAINotFound
except Exception:
    _OpenAIAuthError = _OpenAINotFound = None


def _friendly_error(e):
    """OpenAI exception'ını kullanıcıya gösterilecek kısa Türkçe mesaja çevir."""
    msg = str(e).lower()
    if _OpenAIAuthError is not None and isinstance(e, _OpenAIAuthError):
        return 'API key geçersiz veya okunamadı'
    if 'invalid_api_key' in msg or 'incorrect api key' in msg or '401' in msg or 'authentication' in msg:
        return 'API key geçersiz veya okunamadı'
    if (_OpenAINotFound is not None and isinstance(e, _OpenAINotFound)) or \
       ('model' in msg and ('not found' in msg or 'does not exist' in msg)):
        return f'Model bulunamadı: {_model()} (OPENAI_MODEL değerini kontrol et)'
    if 'rate limit' in msg or '429' in msg:
        return 'OpenAI hız/kota limiti (rate limit) — biraz sonra tekrar dene'
    if 'timeout' in msg or 'timed out' in msg:
        return 'OpenAI zaman aşımı — bağlantı/timeout'
    return f'AI hata: {type(e).__name__}'


# =====================================================================
# MALİYET
# =====================================================================
def estimate_ai_cost(model, input_tokens, output_tokens, cached_tokens=0):
    """Tahmini USD maliyet. Fiyat env'den (1M token başına USD).

    OpenAI'da prompt_tokens cached token'ları DA içerir; cached kısım daha ucuza fatura edilir.
    Bu yüzden: non_cached_input = input - cached, cached ayrı (indirimli) fiyattan hesaplanır.
    """
    price_in = _env_float('AI_PRICE_INPUT_PER_1M', 0.75)
    price_cached = _env_float('AI_PRICE_CACHED_INPUT_PER_1M', price_in)  # yoksa normal input fiyatı
    price_out = _env_float('AI_PRICE_OUTPUT_PER_1M', 4.50)
    inp = int(input_tokens or 0)
    out = int(output_tokens or 0)
    cached = max(0, min(int(cached_tokens or 0), inp))  # cached, input'un alt kümesi
    non_cached = inp - cached
    return round(
        (non_cached / 1_000_000.0) * price_in
        + (cached / 1_000_000.0) * price_cached
        + (out / 1_000_000.0) * price_out,
        6,
    )


# =====================================================================
# KULLANIM TAKİBİ (DB)
# =====================================================================
def _day_start_epoch():
    now = datetime.now()
    return datetime(now.year, now.month, now.day).timestamp()


def _month_start_epoch():
    now = datetime.now()
    return datetime(now.year, now.month, 1).timestamp()


def _usage_since(since_epoch):
    db = get_db()
    try:
        row = db.execute(
            "SELECT COUNT(*) AS c, COALESCE(SUM(estimated_usd), 0) AS u "
            "FROM ai_usage_log WHERE created_at >= ? AND success = 1",
            (since_epoch,),
        ).fetchone()
        return {'calls': int(row['c'] or 0), 'usd': float(row['u'] or 0.0)}
    except Exception as e:
        print(f"[AI-REVIEW] usage sorgu hatasi: {e}")
        return {'calls': 0, 'usd': 0.0}
    finally:
        db.close()


def get_ai_usage_today():
    return _usage_since(_day_start_epoch())


def get_ai_usage_month():
    return _usage_since(_month_start_epoch())


def _log_usage(symbol, purpose, model, in_tok, out_tok, usd, success, error):
    try:
        db = get_db()
        try:
            db.execute(
                "INSERT INTO ai_usage_log "
                "(created_at, symbol, purpose, model, input_tokens, output_tokens, estimated_usd, success, error) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (time.time(), symbol, purpose, model, int(in_tok or 0), int(out_tok or 0),
                 float(usd or 0.0), 1 if success else 0, (str(error)[:500] if error else None)),
            )
            db.commit()
        finally:
            db.close()
    except Exception as e:
        print(f"[AI-REVIEW] usage log hatasi: {e}")


def _log_review(symbol, signal, score, confidence, verdict, usd, price=None):
    try:
        db = get_db()
        try:
            db.execute(
                "INSERT INTO ai_trade_reviews "
                "(created_at, symbol, signal, score, confidence, ai_decision, ai_confidence, "
                " ai_summary, risk_flags_json, raw_json, estimated_usd, price) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (time.time(), symbol, signal, float(score or 0), float(confidence or 0),
                 verdict.get('decision', 'WAIT'), float(verdict.get('confidence', 0) or 0),
                 (verdict.get('summary', '') or '')[:1000],
                 json.dumps(verdict.get('risk_flags', []), ensure_ascii=False),
                 json.dumps(verdict, ensure_ascii=False), float(usd or 0.0),
                 (float(price) if price is not None else None)),
            )
            db.commit()
        finally:
            db.close()
    except Exception as e:
        print(f"[AI-REVIEW] review log hatasi: {e}")


# =====================================================================
# LİMİT / COOLDOWN / CACHE
# =====================================================================
def _within_budget():
    """(bool, sebep) — günlük çağrı + aylık USD limiti kontrolü (sembol-bağımsız)."""
    daily_limit = _env_int('AI_DAILY_CALL_LIMIT', 50)
    if daily_limit > 0 and get_ai_usage_today()['calls'] >= daily_limit:
        return (False, f'gunluk_cagri_limiti ({daily_limit})')
    monthly_usd = _env_float('AI_MONTHLY_USD_LIMIT', 10.0)
    if monthly_usd > 0 and get_ai_usage_month()['usd'] >= monthly_usd:
        return (False, f'aylik_usd_limiti (${monthly_usd:.2f})')
    return (True, 'ok')


def can_call_ai(symbol, purpose='trade_review', signal=None, score=None, price=None):
    """(bool, sebep) — AI çağrısı yapılabilir mi? Limit/cooldown kontrolü.

    K3: cooldown da bağlam-duyarlı — sinyal/skor/fiyat MATERYAL değiştiyse cooldown içinde
    olsa bile taze inceleme serbest (yoksa cache-miss'i cooldown bloklar, bayat WAIT döner).
    Toplam maliyet yine AI_DAILY_CALL_LIMIT + AI_MONTHLY_USD_LIMIT ile sınırlı.
    """
    if not is_ai_review_enabled():
        return (False, 'ai_disabled')

    ok, reason = _within_budget()
    if not ok:
        return (False, reason)

    cooldown_min = _env_int('AI_COOLDOWN_SAME_SYMBOL_MIN', 60)
    if cooldown_min > 0:
        db = get_db()
        try:
            row = db.execute(
                "SELECT created_at, signal, score, price FROM ai_trade_reviews "
                "WHERE symbol=? ORDER BY created_at DESC LIMIT 1",
                (symbol,),
            ).fetchone()
            if (row and (time.time() - float(row['created_at'])) < cooldown_min * 60
                    and _cache_context_matches(row, signal, score, price)):
                return (False, f'cooldown ({cooldown_min}dk)')
        except Exception:
            pass
        finally:
            db.close()

    return (True, 'ok')


def _cache_context_matches(row, signal, score, price):
    """K3: Son kayit ile mevcut aday MATERYAL olarak ayni mi? (cache/cooldown gecerliligi)

    Sadece sembol yeterli degil — sinyal/skor/fiyat degistiyse eski AI yorumu bayattir.
    Kaba band'ler kullaniriz (maliyet korunur, materyal degisimde yenilenir):
      • signal: tam eslesme (AL→GÜÇLÜ AL degisince yenile)
      • score:  tam sayiya yuvarla (1.0 puanlik band)
      • price:  ~%2 band
    None gecilen kriter (veya kayitta NULL) atlanir → geriye donuk uyumlu.
    """
    try:
        if signal is not None and (row['signal'] or '') != signal:
            return False
        if score is not None and row['score'] is not None:
            if round(float(row['score'])) != round(float(score)):
                return False
        if price is not None and row['price'] is not None:
            rp = float(row['price'])
            if rp > 0 and abs(float(price) - rp) / rp > 0.02:
                return False
    except Exception:
        # Karsilastirma yapilamazsa "eslesmiyor" say — guvenli taraf (taze inceleme yapilir)
        return False
    return True


def _get_cached_review(symbol, signal=None, score=None, price=None):
    """Cache TTL içinde aynı sembol+bağlam için taze inceleme varsa onu döndür (çağrı yapma).

    Bağlam (signal/score/price) verilirse ve son kayıttan MATERYAL farklıysa cache MISS →
    bayat yorum reuse edilmez (K3). Bağlam None ise eski davranış (sembol-bazlı).
    """
    ttl_min = _env_int('AI_CACHE_TTL_MIN', 60)
    if ttl_min <= 0:
        return None
    db = get_db()
    try:
        row = db.execute(
            "SELECT raw_json, created_at, signal, score, price FROM ai_trade_reviews "
            "WHERE symbol=? ORDER BY created_at DESC LIMIT 1",
            (symbol,),
        ).fetchone()
        if row and (time.time() - float(row['created_at'])) < ttl_min * 60:
            if not _cache_context_matches(row, signal, score, price):
                return None
            try:
                v = json.loads(row['raw_json'])
                v['_cached'] = True
                return v
            except Exception:
                return None
    except Exception:
        pass
    finally:
        db.close()
    return None


# =====================================================================
# FALLBACK (güvenli varsayılan — asla APPROVE etmez)
# =====================================================================
def fallback_review(reason):
    """AI çağrılamadığında/başarısız olduğunda güvenli fallback. Asla otomatik onay vermez."""
    return {
        'decision': 'WAIT',
        'confidence': 0,
        'summary': f'AI incelemesi yapılamadı ({reason}). Manuel değerlendirin.',
        'bullish_points': [],
        'risk_flags': [f'ai_yok:{reason}'],
        'conditions': [],
        'telegram_text': f'🤖 AI inceleme atlandı ({reason}) — manuel karar.',
        'requires_human_confirmation': True,
        '_fallback': True,
        '_reason': reason,
    }


# =====================================================================
# BAĞLAM (LLM'e verilecek sistem verisi)
# =====================================================================
def build_candidate_context(candidate):
    """Adaydan LLM'e verilecek kompakt, yapılandırılmış bağlam üretir (sadece sistem verisi)."""
    def _r(v, nd=2):
        try:
            return round(float(v), nd)
        except (TypeError, ValueError):
            return v

    ctx = {
        'symbol': candidate.get('symbol'),
        'name': candidate.get('name'),
        'signal': candidate.get('signal'),
        'price': _r(candidate.get('price')),
        'score': _r(candidate.get('score'), 1),
        'confidence': _r(candidate.get('confidence'), 0),
        'stop_loss': _r(candidate.get('stop_loss')),
        'targets': {
            'tp1': _r(candidate.get('tp1')),
            'tp2': _r(candidate.get('tp2')),
            'tp3': _r(candidate.get('tp3')),
        },
        'risk_reward': _r(candidate.get('rr'), 2),
        'quantity': candidate.get('quantity'),
        'position_cost': _r(candidate.get('position_cost')),
        'sector': candidate.get('sector'),
        'sector_momentum_1m': candidate.get('sector_1m'),
        'market_regime': candidate.get('regime'),
        'mtf': candidate.get('mtf'),
        'score_breakdown': candidate.get('score_breakdown'),
        'volume_ratio': candidate.get('volume_ratio'),
        'turnover_tl': candidate.get('turnover'),
        'kap_sentiment': candidate.get('kap_sentiment'),
    }
    # None alanları at (prompt kısa kalsın)
    return {k: v for k, v in ctx.items() if v is not None}


_SYSTEM_PROMPT = (
    "Sen bir BIST hisse senedi ikinci-göz analiz asistanısın. Bir kural-tabanlı teknik sistem "
    "zaten bir ADAY üretti; senin işin bu adayı verilen SİSTEM VERİLERİNE dayanarak eleştirel "
    "değerlendirmek. Kesin yatırım tavsiyesi VERME. Sadece verilen veriye dayan, veri yoksa "
    "varsayma. Emin değilsen WAIT ya da REJECT seç. Şunları değerlendir: risk/ödül oranı, "
    "stop-loss mesafesi, hedefler, hacim/likidite, trend ve çoklu zaman dilimi uyumu, sektör "
    "momentumu, piyasa rejimi ve KAP/haber riski. Nihai karar kullanıcı onayı gerektirir.\n\n"
    "SADECE şu şemada, JSON dışında HİÇBİR metin olmadan yanıt ver:\n"
    "{\n"
    '  "decision": "APPROVE_CANDIDATE | WAIT | REJECT",\n'
    '  "confidence": 0-100,\n'
    '  "summary": "Kısa Türkçe özet (max 2 cümle)",\n'
    '  "bullish_points": ["kısa madde", ...],\n'
    '  "risk_flags": ["kısa madde", ...],\n'
    '  "conditions": ["kısa madde", ...],\n'
    '  "telegram_text": "Telegram için çok kısa (max 240 karakter) Türkçe özet",\n'
    '  "requires_human_confirmation": true\n'
    "}"
)


def _normalize_verdict(raw):
    """LLM JSON çıktısını güvenli şemaya normalize et."""
    dec = str(raw.get('decision', 'WAIT')).upper().strip()
    if dec not in ('APPROVE_CANDIDATE', 'WAIT', 'REJECT'):
        dec = 'WAIT'
    try:
        conf = max(0, min(100, int(float(raw.get('confidence', 0)))))
    except (TypeError, ValueError):
        conf = 0

    def _list(v):
        if isinstance(v, list):
            return [str(x)[:200] for x in v][:6]
        return []

    return {
        'decision': dec,
        'confidence': conf,
        'summary': str(raw.get('summary', ''))[:1000],
        'bullish_points': _list(raw.get('bullish_points')),
        'risk_flags': _list(raw.get('risk_flags')),
        'conditions': _list(raw.get('conditions')),
        'telegram_text': str(raw.get('telegram_text', ''))[:400],
        'requires_human_confirmation': True,  # her zaman true — güvenlik sözleşmesi
    }


def _call_openai(system_prompt, user_payload, timeout_sec, max_out):
    """OpenAI çağrısı. (verdict_raw, usage) döner. Hata fırlatabilir."""
    from openai import OpenAI
    client = OpenAI(api_key=_env('OPENAI_API_KEY'), timeout=timeout_sec)
    base = dict(
        model=_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    # Token limiti parametre adı modele göre değişiyor (max_tokens vs max_completion_tokens).
    try:
        resp = client.chat.completions.create(max_completion_tokens=max_out, **base)
    except Exception as e1:
        if 'max_completion_tokens' in str(e1) or 'max_tokens' in str(e1) or 'Unsupported' in str(e1):
            resp = client.chat.completions.create(max_tokens=max_out, **base)
        else:
            raise
    content = resp.choices[0].message.content or '{}'
    return json.loads(content), _usage_tokens(resp)


def _usage_tokens(resp):
    """(input, output, cached) token sayıları. cached: prompt_tokens_details.cached_tokens."""
    usage = getattr(resp, 'usage', None)
    if not usage:
        return (0, 0, 0)
    in_tok = getattr(usage, 'prompt_tokens', 0) or 0
    out_tok = getattr(usage, 'completion_tokens', 0) or 0
    details = getattr(usage, 'prompt_tokens_details', None)
    cached = 0
    if details is not None:
        cached = (getattr(details, 'cached_tokens', None)
                  or (details.get('cached_tokens') if isinstance(details, dict) else 0)
                  or 0)
    return (in_tok, out_tok, cached)


# =====================================================================
# ANA GİRİŞ
# =====================================================================
def review_trade_candidate(candidate):
    """
    Bir AL adayını AI ile incele. Her zaman bir verdict dict döner (asla exception fırlatmaz).
    Pozisyon AÇMAZ — sadece yorum. decision ∈ {APPROVE_CANDIDATE, WAIT, REJECT}.
    """
    symbol = candidate.get('symbol', '?')
    purpose = 'trade_review'
    # K3: bağlam — sinyal/skor/fiyat değişince bayat cache/cooldown reuse edilmesin
    c_signal = candidate.get('signal')
    c_score = candidate.get('score')
    c_price = candidate.get('price')

    # 1) Cache — TTL içinde AYNI bağlamlı taze inceleme varsa tekrar çağırma
    cached = _get_cached_review(symbol, c_signal, c_score, c_price)
    if cached is not None:
        return cached

    # 2) Limit/cooldown kontrolü (bağlam-duyarlı cooldown)
    ok, reason = can_call_ai(symbol, purpose, c_signal, c_score, c_price)
    if not ok:
        # Limit/cooldown/disabled → fallback (çağrı yapılmaz, para harcanmaz)
        return fallback_review(reason)

    # 3) Çağrı
    model = _model()
    timeout_sec = _env_int('AI_REVIEW_TIMEOUT_SEC', 15)
    max_out = _env_int('AI_MAX_OUTPUT_TOKENS', 700)
    ctx = build_candidate_context(candidate)
    try:
        raw, (in_tok, out_tok, cached_tok) = _call_openai(_SYSTEM_PROMPT, ctx, timeout_sec, max_out)
        usd = estimate_ai_cost(model, in_tok, out_tok, cached_tok)
        verdict = _normalize_verdict(raw)
        _log_usage(symbol, purpose, model, in_tok, out_tok, usd, True, None)
        _log_review(symbol, c_signal, c_score,
                    candidate.get('confidence'), verdict, usd, c_price)
        verdict['_usd'] = usd
        return verdict
    except Exception as e:
        # Başarısız çağrı — usage'a hata olarak yaz (maliyet 0), güvenli fallback dön
        print(f"[AI-REVIEW] {symbol} OpenAI çağrı hatası: {e}")
        _log_usage(symbol, purpose, model, 0, 0, 0.0, False, e)
        return fallback_review(_friendly_error(e))


# =====================================================================
# READ-ONLY ASİSTAN (web chat) — emir vermez, sadece sistem verisini açıklar
# =====================================================================
_ASSISTANT_SYSTEM = (
    "Sen BIST Pro sisteminin READ-ONLY yardımcı asistanısın. Kullanıcının kendi lokal trading "
    "sistemindeki verileri (sinyaller, skorlar, açık pozisyonlar, kararlar) açıklarsın. "
    "KURALLAR: Sadece sana verilen SİSTEM VERİLERİNE dayan; veri yoksa 'elimde veri yok' de, uydurma. "
    "Emir OLUŞTURMA, Midas'a/broker'a bağlanma, şifre/2FA isteme. Kesin al/sat tavsiyesi verme; "
    "riskleri ve sistemin gerekçesini açıkla. Kısa, net, Türkçe yanıt ver. Nihai karar kullanıcınındır."
)


def ask_assistant(question, context_data, symbol=None):
    """Read-only web asistanı. (ok/answer/error) döner. Emir vermez, sadece açıklar."""
    if not is_ai_review_enabled():
        return {'ok': False, 'error': 'AI kapalı (AI_REVIEW_ENABLED=1 ve OPENAI_API_KEY gerekli).'}
    ok, reason = _within_budget()
    if not ok:
        return {'ok': False, 'error': f'Limit doldu: {reason}'}

    model = _model()
    timeout_sec = _env_int('AI_REVIEW_TIMEOUT_SEC', 15)
    max_out = _env_int('AI_MAX_OUTPUT_TOKENS', 700)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=_env('OPENAI_API_KEY'), timeout=timeout_sec)
        user = (f"SORU: {question}\n\nSİSTEM VERİLERİ (JSON):\n"
                f"{json.dumps(context_data, ensure_ascii=False, default=str)}")
        base = dict(
            model=model,
            messages=[{"role": "system", "content": _ASSISTANT_SYSTEM},
                      {"role": "user", "content": user}],
            temperature=0.3,
        )
        try:
            resp = client.chat.completions.create(max_completion_tokens=max_out, **base)
        except Exception as e1:
            if 'token' in str(e1).lower() or 'Unsupported' in str(e1):
                resp = client.chat.completions.create(max_tokens=max_out, **base)
            else:
                raise
        answer = resp.choices[0].message.content or ''
        in_tok, out_tok, cached_tok = _usage_tokens(resp)
        usd = estimate_ai_cost(model, in_tok, out_tok, cached_tok)
        _log_usage(symbol or 'ASSISTANT', 'assistant', model, in_tok, out_tok, usd, True, None)
        return {'ok': True, 'answer': answer, 'usd': usd}
    except Exception as e:
        print(f"[AI-ASSISTANT] hata: {e}")
        _log_usage(symbol or 'ASSISTANT', 'assistant', model, 0, 0, 0.0, False, e)
        return {'ok': False, 'error': _friendly_error(e)}
