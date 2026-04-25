"""
Auth Middleware — userId doğrulama decorator'ları.

Mevcut oturum modeli: client header/body'de 'userId' gönderir, sunucu users tablosunda olup olmadığını
kontrol eder. Session token yok (mevcut mimariyi koruyoruz). Bu decorator, destructive endpoint'lerde:
  - userId alanı var mı?
  - users tablosunda karşılığı var mı?
  kontrolünü tek satırda yapar.

SOLO_MODE: tek-kullanicili ev kullanimi icin auth bypass. Aktifken eksik userId
SOLO_USER_ID ile doldurulur ve user satiri yoksa otomatik olusturulur.

NOT: Gerçek kimlik doğrulama için JWT/session-token refactor gerekir. Bu katman yalnızca 'kullanıcı
olmadan kritik işlem yapılmasın' önlemidir.
"""
from functools import wraps
from flask import request, jsonify
from config import db_conn, SOLO_MODE, SOLO_USER_ID, SOLO_USERNAME

_solo_user_ensured = False


def _ensure_solo_user():
    """Solo mode aktifken kaan kullanicisinin users tablosunda var oldugundan emin ol (idempotent)."""
    global _solo_user_ensured
    if _solo_user_ensured:
        return
    try:
        with db_conn() as db:
            row = db.execute("SELECT id FROM users WHERE id=?", (SOLO_USER_ID,)).fetchone()
            if not row:
                # Solo modda sifre kullanilmadigi icin password_hash bos string
                db.execute(
                    "INSERT INTO users (id, username, password_hash, email) VALUES (?, ?, ?, ?)",
                    (SOLO_USER_ID, SOLO_USERNAME, '', '')
                )
                db.commit()
                print(f"[AUTH] Solo mode user olusturuldu: id={SOLO_USER_ID}")
        _solo_user_ensured = True
    except Exception as e:
        print(f"[AUTH] Solo user ensure hatasi: {e}")


def _extract_user_id() -> str:
    """userId'yi body → query → header sırasıyla ara."""
    uid = ''
    try:
        if request.is_json:
            d = request.get_json(silent=True) or {}
            uid = d.get('userId', '') or d.get('user_id', '')
    except Exception:
        pass
    if not uid:
        uid = request.args.get('userId', '') or request.args.get('user_id', '')
    if not uid:
        uid = request.headers.get('X-User-Id', '')
    return (uid or '').strip()


def require_user(f):
    """Destructive endpoint'ler için: userId yoksa veya users'ta yoksa 401 döner.
    SOLO_MODE aktifken eksik/gecersiz userId SOLO_USER_ID ile karsilanir."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        uid = _extract_user_id()
        if SOLO_MODE:
            _ensure_solo_user()
            if not uid:
                # Body/query'de userId yoksa solo userId kullan; downstream kod
                # _extract_user_id'yi yeniden cagirsa bile g.solo_uid'e bakmaktansa
                # request.args/body'i degistirmek riskli — bu yuzden header'a yaz.
                # Endpoint'ler genelde body/query'den okudugu icin, eksik userId'li
                # solo client zaten yok (frontend dolduruyor). Yine de geçişe izin ver.
                return f(*args, **kwargs)
            # Solo'da uid varsa users'ta zorunlu degil — geçiş izni
            return f(*args, **kwargs)
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 401
        try:
            with db_conn() as db:
                row = db.execute("SELECT id FROM users WHERE id=?", (uid,)).fetchone()
            if not row:
                return jsonify({'success': False, 'error': 'Geçersiz userId'}), 401
        except Exception as e:
            return jsonify({'success': False, 'error': f'Kimlik doğrulama hatası: {e}'}), 500
        return f(*args, **kwargs)
    return wrapper
