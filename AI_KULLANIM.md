# AI İkincil Filtre & Read-Only Asistan — Kullanım

Lokal, tek kullanıcılık BIST Pro sistemine eklenen OpenAI tabanlı **ikinci göz** katmanı.
AI **asla emir vermez / pozisyon açmaz**; sadece analiz/yorum üretir. Nihai karar sende.

## Kurulum
1. `pip install -r requirements.txt` (openai eklendi).
2. `.env` içine anahtar ve ayarlar (bkz. `.env.example`). En kritik:
   - `OPENAI_API_KEY=sk-...` (sadece backend'de; frontend'e gitmez)
   - `OPENAI_MODEL=gpt-5.4-mini` → **OpenAI'daki güncel model adıyla doğrula**, yanlışsa AI çağrısı hata verir (sistem çökmez, fallback devreye girer).
   - `AI_REVIEW_ENABLED=1` (0 yaparsan AI tamamen kapalı, eski akış).
3. Backend'i yeniden başlat: `python backend.py`.

## Nasıl çalışır (scanner ikincil filtre)
- Teknik sistem tüm BIST'i tarar, elemeleri yapar. **AI sadece** son elemeden geçen,
  `AL/GÜÇLÜ AL`, skoru `AI_MIN_SCORE_FOR_REVIEW`+ ve **en güçlü `AI_REVIEW_TOP_N`** adayda çalışır.
- AI kararı:
  - **APPROVE_CANDIDATE** → Telegram kartına "AI 2. Göz" özeti eklenir, normal onay akışı.
  - **WAIT / REJECT** → pozisyon açılmaz, loglanır, bilgi amaçlı Telegram notu (butonsuz).
  - **Hata/limit** → `AI_REVIEW_FAIL_MODE=manual_review` ise manuel incelemeye bırakılır (otomatik açmaz).
- Nihai karar her zaman **senin Telegram onayında**.

## Maliyet kontrolü
- `AI_DAILY_CALL_LIMIT` (günlük çağrı) + `AI_MONTHLY_USD_LIMIT` (aylık tahmini USD) aşılırsa çağrı yapılmaz.
- Aynı sembol `AI_COOLDOWN_SAME_SYMBOL_MIN` içinde tekrar çağrılmaz; `AI_CACHE_TTL_MIN` içinde cache kullanılır.
- Her çağrının token/maliyeti `ai_usage_log` tablosuna, her inceleme `ai_trade_reviews` tablosuna yazılır.
- ⚠️ **ChatGPT Plus üyeliği API'ye dahil değildir** — API ayrı ücretlendirilir; limitler bu yüzden var.
- Fiyat tahmini `AI_PRICE_INPUT_PER_1M` / `AI_PRICE_OUTPUT_PER_1M` env'inden; modelin gerçek fiyatıyla güncelle.

## Read-only web asistanı
- Sağ altta 🤖 butonu → chat drawer. Örnek: "THYAO neden AL verdi?", "Bugün neden işlem açmadı?",
  "Portföy riskim nasıl?", "En güçlü 5 aday?".
- Sadece sistem verisini okur/açıklar; emir oluşturmaz, Midas'a bağlanmaz, şifre istemez.
- Endpoint: `POST /api/ai/chat {message, symbol?}`, durum: `GET /api/ai/status` (key döndürmez).

## Güvenlik notları
- `.env` **git'e girmez** (`.gitignore`). API key'i paylaşırsan platform.openai.com'dan **rotate et**.
- AI sadece Telegram/analiz üretir. Broker otomasyonu, şifre/cookie saklama YOK.

## Kapatma
- `AI_REVIEW_ENABLED=0` → AI tamamen devre dışı, sistem eski haliyle çalışır (hiçbir şey bozulmaz).
