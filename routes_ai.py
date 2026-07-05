"""
routes_ai.py — Read-only BIST Asistanı (web chat)
Blueprint: ai_bp   Prefix: /api/ai

GÜVENLİK: Sadece OKUR ve açıklar. Emir oluşturmaz, Midas'a bağlanmaz, şifre/2FA istemez.
OPENAI_API_KEY yalnızca backend'de; frontend'e ASLA gönderilmez (endpoint sadece cevabı döner).
"""
from flask import Blueprint, jsonify, request

from config import safe_dict, SOLO_USER_ID, get_db

ai_bp = Blueprint('ai', __name__)


def _detect_symbol(message):
    """Mesajdaki büyük harfli 4-6 harflik token'ı bilinen BIST koduyla eşleştir."""
    import re
    try:
        from config import _get_stocks
        codes = {s.get('code', '').upper() for s in _get_stocks()}
    except Exception:
        codes = set()
    for tok in re.findall(r'\b[A-ZĞÜŞİÖÇ]{4,6}\b', (message or '').upper()):
        if tok in codes:
            return tok
    return None


def _gather_context(symbol=None):
    """Read-only sistem verisi topla (DB + açık pozisyonlar). AI'ya sadece bu verilir."""
    uid = SOLO_USER_ID
    ctx = {'open_positions': [], 'recent_decisions': [], 'config': {}}

    # Açık pozisyonlar
    try:
        from auto_trader import _auto_get_open_positions
        for p in _auto_get_open_positions(uid):
            ctx['open_positions'].append({
                'symbol': p.get('symbol'), 'qty': p.get('quantity'),
                'entry': p.get('entryPrice'), 'sl': p.get('stopLoss'),
                'tp1': p.get('takeProfit1'),
            })
    except Exception:
        pass

    # Son otomatik kararlar (neden açtı/açmadı)
    try:
        db = get_db()
        try:
            rows = db.execute(
                "SELECT symbol, decision, reason, detail, score, confidence "
                "FROM auto_decisions WHERE user_id=? ORDER BY created_at DESC LIMIT 25",
                (uid,),
            ).fetchall()
            ctx['recent_decisions'] = [dict(r) for r in rows]
        finally:
            db.close()
    except Exception:
        pass

    # Config özeti (min skor vb.) — read-only
    try:
        from auto_trader import _auto_get_config
        cfg = _auto_get_config(uid) or {}
        ctx['config'] = {
            'minScore': cfg.get('minScore'), 'minConfidence': cfg.get('minConfidence'),
            'maxPositions': cfg.get('maxPositions'), 'capital': cfg.get('capital'),
            'tradeStyle': cfg.get('tradeStyle'), 'enabled': cfg.get('enabled'),
        }
    except Exception:
        pass

    # En güçlü son adaylar (signal_log'dan yüksek skorlu son AL sinyalleri)
    try:
        db = get_db()
        try:
            rows = db.execute(
                "SELECT symbol, action, score, ml_confidence, timeframe "
                "FROM signal_log WHERE action IN ('AL','GÜÇLÜ AL') "
                "ORDER BY logged_at DESC LIMIT 40",
            ).fetchall()
            seen, top = set(), []
            for r in sorted((dict(x) for x in rows), key=lambda d: d.get('score') or 0, reverse=True):
                if r['symbol'] in seen:
                    continue
                seen.add(r['symbol'])
                top.append(r)
                if len(top) >= 8:
                    break
            ctx['top_candidates'] = top
        finally:
            db.close()
    except Exception:
        pass

    # Sembole özel: son sinyal kaydı (read-only, yeniden hesaplamaz)
    if symbol:
        try:
            db = get_db()
            try:
                row = db.execute(
                    "SELECT symbol, action, score, rsi, macd, factors, ml_confidence, timeframe, logged_at "
                    "FROM signal_log WHERE symbol=? ORDER BY logged_at DESC LIMIT 1",
                    (symbol,),
                ).fetchone()
                if row:
                    ctx['symbol_signal'] = dict(row)
            finally:
                db.close()
        except Exception:
            pass

    return ctx


@ai_bp.route('/api/ai/chat', methods=['POST'])
def ai_chat():
    """
    Read-only BIST asistanı. POST {message, symbol?}.
    Örnek: "THYAO neden AL verdi?", "Bugün neden işlem açmadı?", "Portföy riskim nasıl?"
    """
    import ai_reviewer
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    symbol = (data.get('symbol') or '').strip().upper() or None
    if not message:
        return jsonify({'ok': False, 'error': 'Boş mesaj.'}), 400

    if not ai_reviewer.is_ai_review_enabled():
        return jsonify({
            'ok': False,
            'error': 'AI kapalı. .env içine OPENAI_API_KEY ve AI_REVIEW_ENABLED=1 ekleyin.',
        }), 200

    if symbol is None:
        symbol = _detect_symbol(message)

    ctx = _gather_context(symbol)
    result = ai_reviewer.ask_assistant(message, ctx, symbol)
    return jsonify(safe_dict({**result, 'symbol': symbol})), 200


@ai_bp.route('/api/ai/status')
def ai_status():
    """AI durum + günlük/aylık kullanım (frontend panel için, key DÖNMEZ)."""
    import ai_reviewer
    try:
        ks = ai_reviewer.key_status()  # değer DÖNMEZ; sadece loaded/prefix/length
        return jsonify(safe_dict({
            'ok': True,
            'enabled': ai_reviewer.is_ai_review_enabled(),
            'model': ai_reviewer._model(),
            'key_loaded': ks['loaded'],
            'key_prefix': ks['prefix'],
            'key_length': ks['length'],
            'usage_today': ai_reviewer.get_ai_usage_today(),
            'usage_month': ai_reviewer.get_ai_usage_month(),
        }))
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500
