"""
database_state.py - Restart resilience: alert_state + pending_signals persistence.
database.py'den ayristirildi (modul satir siniri). database.py geriye-uyum icin re-export eder.
"""
import json
import time
import sqlite3
from datetime import datetime
from config import get_db, USE_POSTGRES, PG_OK, DB_PATH


def _db_save_alert_state(user_id: str, symbol: str, pos_id: int, state: dict) -> None:
    """rt_alert_state tablosuna upsert. _confirm sayaclari ramdan tutulur,
    persist edilen sadece flag'ler (hit/warned/executed) — JSON olarak."""
    try:
        # Persist flagleri sadece (counter'lar transient)
        persist = {k: v for k, v in state.items() if not k.endswith('_confirm')}
        payload = json.dumps(persist)
        now = time.time()
        if USE_POSTGRES and PG_OK:
            db = get_db()
            db.execute(
                "INSERT INTO rt_alert_state (user_id, symbol, pos_id, state_json, updated_at) "
                "VALUES (?,?,?,?,?) "
                "ON CONFLICT (user_id, symbol, pos_id) DO UPDATE SET "
                "state_json=EXCLUDED.state_json, updated_at=EXCLUDED.updated_at",
                (user_id, symbol, int(pos_id), payload, now)
            )
            db.commit()
            db.close()
        else:
            db = sqlite3.connect(DB_PATH)
            db.execute(
                "INSERT OR REPLACE INTO rt_alert_state "
                "(user_id, symbol, pos_id, state_json, updated_at) VALUES (?,?,?,?,?)",
                (user_id, symbol, int(pos_id), payload, now)
            )
            db.commit()
            db.close()
    except Exception as e:
        print(f"[ALERT-STATE-SAVE] Hata: {e}")


def _db_delete_alert_state(user_id: str, symbol: str, pos_id: int) -> None:
    """Pozisyon kapaninca alert state'i sil."""
    try:
        db = get_db()
        db.execute(
            "DELETE FROM rt_alert_state WHERE user_id=? AND symbol=? AND pos_id=?",
            (user_id, symbol, int(pos_id))
        )
        db.commit()
        db.close()
    except Exception as e:
        print(f"[ALERT-STATE-DELETE] Hata: {e}")


def _db_load_alert_states() -> dict:
    """Tum alert state'leri yukle → {(user_id, symbol, pos_id): state_dict}.
    Sadece acik pozisyonlara ait olanlar dondurulur (yetim kayitlar silinir)."""
    try:
        db = get_db()
        # Once yetim kayitlari temizle (acik olmayan pozisyonlara ait olanlar)
        db.execute(
            "DELETE FROM rt_alert_state WHERE pos_id NOT IN "
            "(SELECT id FROM auto_positions WHERE status='open')"
        )
        db.commit()
        rows = db.execute(
            "SELECT user_id, symbol, pos_id, state_json FROM rt_alert_state"
        ).fetchall()
        db.close()
        out = {}
        for r in rows:
            try:
                state = json.loads(r['state_json'])
                # Confirm sayaclari ramdan baslar (0)
                for k in ('sl_confirm', 'tp1_confirm', 'tp2_confirm', 'tp3_confirm'):
                    state.setdefault(k, 0)
                out[(r['user_id'], r['symbol'], int(r['pos_id']))] = state
            except Exception:
                pass
        return out
    except Exception as e:
        print(f"[ALERT-STATE-LOAD] Hata: {e}")
        return {}


def _db_save_pending_signal(signal_id: str, payload: dict) -> None:
    """pending_signals tablosuna upsert. expires_at datetime ise epoch'a cevir."""
    try:
        # datetime → ISO string serialize
        payload_copy = dict(payload)
        exp = payload_copy.get('expires_at')
        if isinstance(exp, datetime):
            exp_epoch = exp.timestamp()
            payload_copy['expires_at'] = exp.isoformat()
        else:
            try:
                exp_epoch = float(exp) if exp else (time.time() + 900)
            except Exception:
                exp_epoch = time.time() + 900
        payload_json = json.dumps(payload_copy)
        if USE_POSTGRES and PG_OK:
            db = get_db()
            db.execute(
                "INSERT INTO pending_signals (signal_id, payload_json, expires_at) "
                "VALUES (?,?,?) "
                "ON CONFLICT (signal_id) DO UPDATE SET "
                "payload_json=EXCLUDED.payload_json, expires_at=EXCLUDED.expires_at",
                (signal_id, payload_json, exp_epoch)
            )
            db.commit()
            db.close()
        else:
            db = sqlite3.connect(DB_PATH)
            db.execute(
                "INSERT OR REPLACE INTO pending_signals "
                "(signal_id, payload_json, expires_at) VALUES (?,?,?)",
                (signal_id, payload_json, exp_epoch)
            )
            db.commit()
            db.close()
    except Exception as e:
        print(f"[PENDING-SAVE] Hata: {e}")


def _db_delete_pending_signal(signal_id: str) -> None:
    """Onaylanmis veya iptal edilmis sinyali sil."""
    try:
        db = get_db()
        db.execute("DELETE FROM pending_signals WHERE signal_id=?", (signal_id,))
        db.commit()
        db.close()
    except Exception as e:
        print(f"[PENDING-DELETE] Hata: {e}")


def _db_load_pending_signals() -> dict:
    """Suresi gecmemis tum bekleyen sinyalleri yukle → {signal_id: payload}."""
    try:
        db = get_db()
        # Onceden suresi dolmuş kayıtlari temizle
        db.execute("DELETE FROM pending_signals WHERE expires_at < ?", (time.time(),))
        db.commit()
        rows = db.execute(
            "SELECT signal_id, payload_json FROM pending_signals"
        ).fetchall()
        db.close()
        out = {}
        for r in rows:
            try:
                payload = json.loads(r['payload_json'])
                # ISO string → datetime
                exp = payload.get('expires_at')
                if isinstance(exp, str):
                    try:
                        payload['expires_at'] = datetime.fromisoformat(exp)
                    except Exception:
                        pass
                out[r['signal_id']] = payload
            except Exception:
                pass
        return out
    except Exception as e:
        print(f"[PENDING-LOAD] Hata: {e}")
        return {}
