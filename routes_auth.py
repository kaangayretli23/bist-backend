"""
Authentication routes: register, login, profile, session check.
"""
import uuid
from flask import Blueprint, jsonify, request
from config import get_db, db_conn, hash_password, safe_dict, SOLO_MODE, SOLO_USER_ID, SOLO_USERNAME

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/api/auth/solo-info', methods=['GET'])
def solo_info():
    """Frontend bu endpoint'e bakarak solo mode'da otomatik giris yapar."""
    return jsonify({
        'solo': bool(SOLO_MODE),
        'userId': SOLO_USER_ID if SOLO_MODE else '',
        'username': SOLO_USERNAME if SOLO_MODE else '',
    })


@auth_bp.route('/api/auth/register', methods=['POST'])
def register():
    try:
        d = request.json or {}
        username = d.get('username', '').strip()
        password = d.get('password', '')
        email = d.get('email', '').strip()

        if not username or len(username) < 3:
            return jsonify({'error': 'Kullanici adi en az 3 karakter olmali'}), 400
        if not password or len(password) < 4:
            return jsonify({'error': 'Sifre en az 4 karakter olmali'}), 400

        with db_conn() as db:
            existing = db.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
            if existing:
                return jsonify({'error': 'Bu kullanici adi zaten alinmis'}), 400
            user_id = str(uuid.uuid4())
            db.execute("INSERT INTO users (id, username, password_hash, email) VALUES (?, ?, ?, ?)",
                       (user_id, username, hash_password(password), email))
            db.commit()
        return jsonify({'success': True, 'userId': user_id, 'username': username})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/api/auth/login', methods=['POST'])
def login():
    try:
        d = request.json or {}
        username = d.get('username', '').strip()
        password = d.get('password', '')

        if not username or not password:
            return jsonify({'error': 'Kullanici adi ve sifre gerekli'}), 400

        with db_conn() as db:
            user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

        if not user:
            return jsonify({'error': 'Kullanici bulunamadi. Kayit olmaniz gerekebilir.'}), 401
        if user['password_hash'] != hash_password(password):
            return jsonify({'error': 'Sifre hatali'}), 401

        return jsonify({'success': True, 'userId': user['id'], 'username': user['username'], 'email': user['email'] or ''})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/api/auth/profile', methods=['POST'])
def update_profile():
    try:
        d = request.json or {}
        user_id = d.get('userId', '')
        if not user_id:
            return jsonify({'error': 'Giris yapmaniz gerekli'}), 401

        with db_conn() as db:
            user = db.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone()
            if not user:
                return jsonify({'error': 'Kullanici bulunamadi'}), 404
            email = d.get('email', user['email'] or '')
            telegram = d.get('telegramChatId', user['telegram_chat_id'] or '')
            db.execute("UPDATE users SET email=?, telegram_chat_id=? WHERE id=?", (email, telegram, user_id))
            db.commit()
        return jsonify({'success': True, 'message': 'Profil guncellendi'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@auth_bp.route('/api/auth/check')
def check_session():
    """localStorage oturumunun hala gecerli olup olmadigini kontrol et.
    Solo mode'da her zaman valid doner (auth bypass)."""
    if SOLO_MODE:
        return jsonify({'valid': True, 'solo': True, 'userId': SOLO_USER_ID})
    uid = request.args.get('userId', '')
    if not uid:
        return jsonify({'valid': False})
    try:
        with db_conn() as db:
            user = db.execute("SELECT id FROM users WHERE id=?", (uid,)).fetchone()
        return jsonify({'valid': user is not None})
    except Exception:
        return jsonify({'valid': False})
