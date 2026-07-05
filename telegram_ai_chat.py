"""
telegram_ai_chat.py — Telegram üzerinden read-only AI asistan.

Bota yazılan düz metni (buton/komut DEĞİL) ai_reviewer.ask_assistant'a bağlar ve cevabı
Telegram'a geri gönderir. Web chatbox'ın Telegram karşılığı — aynı read-only context'i kullanır.

GÜVENLİK: Sadece TELEGRAM_CHAT_ID'deki kullanıcının mesajları işlenir. AI emir vermez,
sadece sistem verisini açıklar. Her mesaj bir AI çağrısıdır → limit/cooldown ai_reviewer'da.
"""
import os


def handle_text_message(update):
    """
    update içindeki 'message' metnini AI'ya sorup cevabı Telegram'a gönderir.
    True  = metin mesajı olarak işlendi (callback akışına gitmesin).
    False = işlenmedi (buton callback'i / komut / boş / yetkisiz) → normal akış devam etsin.
    """
    msg = update.get('message') or update.get('edited_message') or {}
    text = (msg.get('text') or '').strip()
    if not text:
        return False
    # /komutlar burada işlenmez (ileride /help vb. eklenebilir)
    if text.startswith('/'):
        return False

    # K1: sadece tanımlı chat_id'deki kullanıcı (callback auth ile aynı prensip)
    chat_id = str((msg.get('chat') or {}).get('id', ''))
    allowed = str(os.environ.get('TELEGRAM_CHAT_ID', ''))
    if not allowed or chat_id != allowed:
        print(f"[TG-AI] Yetkisiz metin mesaji bloklandi (chat_id={chat_id})")
        return True  # işlendi say (cevap verme) — normal akışa düşmesin

    try:
        from telegram_notifications import send_telegram
    except Exception:
        return True

    try:
        import ai_reviewer
        if not ai_reviewer.is_ai_review_enabled():
            send_telegram('🤖 AI şu an kapalı (OPENAI_API_KEY / AI_REVIEW_ENABLED kontrol et).')
            return True

        # Web asistanıyla aynı read-only context + sembol tespiti (kod tekrarı yok)
        from routes_ai import _gather_context, _detect_symbol
        symbol = _detect_symbol(text)
        ctx = _gather_context(symbol)
        res = ai_reviewer.ask_assistant(text, ctx, symbol)

        if res.get('ok'):
            answer = (res.get('answer') or '(boş yanıt)').strip()
            send_telegram(f'🤖 <b>AI</b>\n{answer}')
        else:
            send_telegram(f'🤖 {res.get("error", "AI yanıt vermedi.")}')
    except Exception as e:
        print(f"[TG-AI] hata: {e}")
        try:
            from telegram_notifications import send_telegram as _st
            _st(f'🤖 AI hata: {type(e).__name__}')
        except Exception:
            pass
    return True
