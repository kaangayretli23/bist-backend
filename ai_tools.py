"""
ai_tools.py — Read-only AI asistan araç kayıt defteri (OpenAI function calling).

Asistan bir soruya cevap verirken CANLI sistem verisine ihtiyaç duyarsa bu araçları çağırır;
biz sunucu tarafında çalıştırıp sonucu geri veririz, o da cevabı kurar.

GÜVENLİK SÖZLEŞMESİ (değişmez):
  • TÜM araçlar SADECE OKUR. Emir OLUŞTURMAZ, pozisyon açmaz/kapatmaz, Midas/broker'a bağlanmaz,
    config değiştirmez, DB'ye yazmaz.
  • Yeni araç eklerken: yalnızca okuma yapan, para yolunu değiştirmeyen fonksiyonlar bağla.
  • Aksiyon (al/sat/kapat) gerekiyorsa AI DİREKT yapmaz — ayrı bir insan-onay (Telegram approve)
    akışına düşürülmeli; bu modüle YAZMA aracı EKLENMEZ.

Her araç: OpenAI şeması (TOOL_SCHEMAS) + handler (TOOL_HANDLERS). Handler JSON-serileştirilebilir
dict döner; hata olursa {'error': ...} döner (asla exception fırlatıp döngüyü kırmaz).
Lazy import: ağır modüller (config, indicators, realtime_prices) fonksiyon içinde import edilir.
"""
import json


# =====================================================================
# ARAÇ HANDLER'LARI (hepsi read-only)
# =====================================================================
def _tool_canli_fiyat(symbol=None):
    """Bir hissenin anlık/son fiyatı, günlük değişim %, veri yaşı."""
    sym = (symbol or '').strip().upper()
    if not sym:
        return {'error': 'symbol gerekli'}
    out = {'symbol': sym}
    try:
        import time
        from realtime_prices import get_price, get_quote
        p = get_price(sym)
        if p:
            out['price'] = round(float(p), 4)
        q = get_quote(sym)
        if q:
            out['source'] = q.get('source')
            if q.get('ts'):
                out['veri_yasi_sn'] = round(time.time() - float(q['ts']))
    except Exception as e:
        out['rt_error'] = str(e)
    try:
        from config import _cget, _stock_cache
        st = _cget(_stock_cache, sym)
        if st:
            out.setdefault('price', st.get('price'))
            out['gunluk_degisim_pct'] = st.get('changePct')
            out['isim'] = st.get('name')
    except Exception:
        pass
    if 'price' not in out:
        return {'symbol': sym, 'error': 'fiyat bulunamadı (veri yok)'}
    return out


def _tool_teknik_ozet(symbol=None):
    """Teknik görünüm: RSI, MACD, ATR, ADX, indikatör özeti, ATR-bazlı önerilen SL, son sistem sinyali."""
    sym = (symbol or '').strip().upper()
    if not sym:
        return {'error': 'symbol gerekli'}
    from config import _cget_hist, _cget, _stock_cache
    hist = _cget_hist(f"{sym}_1y")
    if hist is None or len(hist) < 30:
        try:
            from data_fetcher import _fetch_hist_df
            hist = _fetch_hist_df(sym, '1y')
        except Exception:
            hist = None
    if hist is None or len(hist) < 30:
        return {'symbol': sym, 'error': 'yeterli tarihsel veri yok'}

    price = None
    try:
        from realtime_prices import get_price
        price = get_price(sym)
    except Exception:
        pass
    if not price:
        st = _cget(_stock_cache, sym)
        price = (st or {}).get('price') or float(hist['Close'].iloc[-1])
    price = float(price)

    from indicators import calc_all_indicators
    ind = calc_all_indicators(hist, price)
    rsi = ind.get('rsi', {}) or {}
    macd = ind.get('macd', {}) or {}
    atr = ind.get('atr', {}) or {}
    adx = ind.get('adx', {}) or {}
    summ = ind.get('summary', {}) or {}

    atr_val = float(atr.get('value') or 0)
    sl_2atr = round(price - 2 * atr_val, 2) if atr_val > 0 else None

    last_sig = None
    try:
        from config import get_db
        db = get_db()
        try:
            r = db.execute(
                "SELECT action, score, timeframe FROM signal_log WHERE symbol=? ORDER BY logged_at DESC LIMIT 1",
                (sym,),
            ).fetchone()
            if r:
                last_sig = {'action': r['action'], 'score': r['score'], 'tf': r['timeframe']}
        finally:
            db.close()
    except Exception:
        pass

    return {
        'symbol': sym,
        'fiyat': round(price, 4),
        'rsi': rsi.get('value'), 'rsi_sinyal': rsi.get('signal'),
        'macd_histogram': macd.get('histogram'), 'macd_sinyal': macd.get('signalType'),
        'atr': atr.get('value'), 'atr_pct': atr.get('pct'),
        'adx': adx.get('value'), 'adx_sinyal': adx.get('signal'),
        'onerilen_sl_2xatr': sl_2atr,
        'indikator_ozeti': summ.get('overall'),
        'al_sinyali_sayisi': summ.get('buySignals'), 'sat_sinyali_sayisi': summ.get('sellSignals'),
        'son_sistem_sinyali': last_sig,
    }


def _tool_portfoy_ozeti():
    """Açık pozisyonlar: giriş, canlı fiyat, P&L %, stop-loss'a mesafe %, toplam değer/P&L."""
    from config import SOLO_USER_ID, _cget, _stock_cache
    uid = SOLO_USER_ID
    try:
        from auto_trader import _auto_get_open_positions
        positions = _auto_get_open_positions(uid) or []
    except Exception as e:
        return {'error': f'pozisyonlar okunamadı: {e}'}

    try:
        from realtime_prices import get_price
    except Exception:
        get_price = None

    out_positions = []
    total_cost = 0.0
    total_val = 0.0
    for p in positions:
        sym = p.get('symbol')
        qty = float(p.get('quantity') or 0)
        entry = float(p.get('entryPrice') or 0)
        sl = float(p.get('stopLoss') or 0)
        cur = None
        if get_price:
            try:
                cur = get_price(sym)
            except Exception:
                cur = None
        if not cur:
            st = _cget(_stock_cache, sym)
            cur = (st or {}).get('price')
        cur = float(cur or 0)
        pnl_pct = round((cur - entry) / entry * 100, 2) if entry > 0 and cur > 0 else None
        to_sl = round((cur - sl) / cur * 100, 2) if cur > 0 and sl > 0 else None
        cost = entry * qty
        val = (cur * qty) if cur > 0 else cost
        total_cost += cost
        total_val += val
        out_positions.append({
            'symbol': sym, 'adet': qty, 'giris': entry,
            'canli_fiyat': round(cur, 4) if cur > 0 else None,
            'stop_loss': sl, 'pnl_pct': pnl_pct, 'stop_mesafe_pct': to_sl,
        })

    return {
        'pozisyon_sayisi': len(out_positions),
        'toplam_maliyet': round(total_cost, 2),
        'toplam_deger': round(total_val, 2),
        'toplam_pnl': round(total_val - total_cost, 2),
        'toplam_pnl_pct': round((total_val - total_cost) / total_cost * 100, 2) if total_cost > 0 else None,
        'pozisyonlar': out_positions,
    }


def _tool_tarama_onizle(limit=8):
    """Sistemin son ürettiği en yüksek skorlu AL adayları (ilk N). Canlı tarama değil, son loglar."""
    try:
        n = int(limit or 8)
    except (TypeError, ValueError):
        n = 8
    n = max(1, min(n, 20))
    from config import get_db
    db = get_db()
    try:
        rows = db.execute(
            "SELECT symbol, action, score, ml_confidence, timeframe FROM signal_log "
            "WHERE action IN ('AL','GÜÇLÜ AL') ORDER BY logged_at DESC LIMIT 60"
        ).fetchall()
    finally:
        db.close()
    seen, top = set(), []
    for r in sorted((dict(x) for x in rows), key=lambda d: d.get('score') or 0, reverse=True):
        if r['symbol'] in seen:
            continue
        seen.add(r['symbol'])
        top.append({'symbol': r['symbol'], 'sinyal': r['action'], 'skor': r['score'],
                    'ml_guven': r['ml_confidence'], 'tf': r['timeframe']})
        if len(top) >= n:
            break
    return {'adaylar': top,
            'not': 'Son loglanan AL sinyallerinden skorca en yüksekler (o anki canlı tarama değil).'}


def _tool_bugun_kararlar():
    """Bugün otomatik trader hangi kararları verdi (BUY/PENDING/SKIP) + sebep dağılımı."""
    from config import SOLO_USER_ID, get_db
    from datetime import datetime
    uid = SOLO_USER_ID
    floor = datetime.now().strftime('%Y-%m-%d 00:00:00')
    db = get_db()
    try:
        rows = db.execute(
            "SELECT decision, reason, COUNT(*) AS c FROM auto_decisions "
            "WHERE user_id=? AND created_at >= ? GROUP BY decision, reason ORDER BY c DESC",
            (uid, floor),
        ).fetchall()
    finally:
        db.close()
    return {'tarih': floor[:10],
            'dagilim': [{'karar': r['decision'], 'sebep': r['reason'] or '', 'adet': int(r['c'])} for r in rows]}


# =====================================================================
# KAYIT DEFTERİ (OpenAI şema + handler eşlemesi)
# =====================================================================
TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "canli_fiyat",
        "description": "Bir BIST hissesinin anlık/son fiyatı, günlük değişim yüzdesi ve verinin kaç saniye eski olduğu.",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "BIST hisse kodu, örn: THYAO"}
        }, "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "teknik_ozet",
        "description": "Bir hissenin teknik görünümü: RSI, MACD, ATR, ADX, indikatör özeti, ATR-bazlı önerilen stop-loss ve son sistem sinyali (skor/aksiyon).",
        "parameters": {"type": "object", "properties": {
            "symbol": {"type": "string", "description": "BIST hisse kodu, örn: ASELS"}
        }, "required": ["symbol"]},
    }},
    {"type": "function", "function": {
        "name": "portfoy_ozeti",
        "description": "Kullanıcının açık pozisyonları: giriş fiyatı, canlı fiyat, P&L yüzdesi, stop-loss'a kalan mesafe yüzdesi ve toplam değer/P&L. Stop-loss analizi için kullan.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "tarama_onizle",
        "description": "Sistemin son ürettiği en yüksek skorlu AL adaylarını (ilk N hisse) döndürür. 'ilk 5 hisse', 'en iyi adaylar' gibi sorular için.",
        "parameters": {"type": "object", "properties": {
            "limit": {"type": "integer", "description": "Kaç aday döndürülsün (varsayılan 8, max 20)"}
        }},
    }},
    {"type": "function", "function": {
        "name": "bugun_kararlar",
        "description": "Bugün otomatik trader'ın verdiği kararların (BUY/PENDING/SKIP) sebep dağılımı. 'bugün neden işlem açmadı' gibi sorular için.",
        "parameters": {"type": "object", "properties": {}},
    }},
]

TOOL_HANDLERS = {
    "canli_fiyat": _tool_canli_fiyat,
    "teknik_ozet": _tool_teknik_ozet,
    "portfoy_ozeti": _tool_portfoy_ozeti,
    "tarama_onizle": _tool_tarama_onizle,
    "bugun_kararlar": _tool_bugun_kararlar,
}


def execute_tool(name, arguments):
    """Aracı güvenli çalıştır. Her zaman JSON-serileştirilebilir dict döner (asla raise etmez)."""
    fn = TOOL_HANDLERS.get(name)
    if not fn:
        return {'error': f'bilinmeyen araç: {name}'}
    if not isinstance(arguments, dict):
        arguments = {}
    try:
        result = fn(**arguments)
        # JSON-serileştirilebilir olduğundan emin ol
        json.dumps(result, ensure_ascii=False, default=str)
        return result
    except TypeError as e:
        return {'error': f'araç parametre hatası ({name}): {e}'}
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'}
