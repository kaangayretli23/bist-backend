"""
realtime_monitor.py - Acik pozisyon SL/TP/panic monitoru (_check_positions_once).
realtime_prices.py'den ayristirildi (modul satir siniri). Fiyat cache + alert-state
realtime_prices'ta; bu modul onlari import eder. realtime_prices._loop bu modulu
YALNIZ kendisi (gerekli isimler tanimlandiktan sonra) import eder -> circular yok.
"""
import time
from realtime_prices import get_price, _get_open_positions, _alert_state, _alert_lock


def _enabled_user_styles() -> dict:
    """{user_id: trade_style} — sadece auto-trade aktif kullanicilar.
    SL/TP tetiklendiginde otomatik kapatma sadece bu kullanicilarda calisir."""
    try:
        from config import get_db
        db = get_db()
        rows = db.execute(
            "SELECT user_id, trade_style FROM auto_config WHERE enabled=1"
        ).fetchall()
        db.close()
        return {r['user_id']: (r['trade_style'] or 'swing') for r in rows}
    except Exception:
        return {}


# Spike koruması: SL/TP tetiklemesi için ardışık tick sayısı.
# Tek bir saçma tick'le kapatmayalım; iki ardışık tick istiyoruz.
_RT_CONFIRM_TICKS = 2


def _check_positions_once():
    """
    Açık pozisyonları kontrol et:
      1) Telegram uyarısı (her durumda — auto-trade kapalı kullanıcılar için de)
      2) Auto-trade aktif kullanıcılarda SL/TP tetiklenince otomatik kapat
         (ardışık 2 tick koşulu — spike koruması).
    """
    positions = _get_open_positions()
    if not positions:
        return

    try:
        from routes_telegram import send_telegram
    except Exception:
        send_telegram = None

    enabled_styles = _enabled_user_styles()

    for pos in positions:
        sym   = pos['symbol']
        entry = float(pos['entry_price'] or 0)
        qty   = float(pos['quantity']    or 0)
        sl    = float(pos['stop_loss']   or 0)
        tp1   = float(pos['take_profit1'] or 0)
        tp2   = float(pos['take_profit2'] or 0)
        tp3   = float(pos['take_profit3'] or 0)
        uid   = pos['user_id']
        pid   = pos['id']
        # TP strategy: 'staged' (default — 50/25/25 kademeli) | 'all_at_tp1' (TP1'de tamamini sat)
        # NULL ise eski pozisyonlar icin staged davranisi (geriye donuk uyum).
        tp_strategy = (pos['tp_strategy'] if 'tp_strategy' in pos.keys() else None) or 'staged'

        cur = get_price(sym)
        if not cur or cur <= 0 or not entry:
            continue

        # P1: Bayat-fiyat koruması — canlı fiyat çok eskiyse SL'i donmuş fiyatla KIYASLAMA.
        # Feed çöküş anında donarsa, eski (çöküş-öncesi) fiyatla SL asla tetiklenmez → sessiz körlük.
        # Bunu sesli uyarıya çevir; bu tick için SL/TP kararı verme.
        try:
            from market_alerts import price_too_stale, should_warn_stale
            _stale, _age = price_too_stale(sym)
            if _stale:
                if send_telegram and should_warn_stale(sym):
                    _age_txt = f"{int(_age)}sn önce" if _age is not None else "veri yok"
                    try:
                        send_telegram(
                            f"🛑 <b>KORUMASIZ — {sym}</b>\n"
                            f"Canlı fiyat akmıyor (son veri {_age_txt}).\n"
                            f"SL/TP otomatik koruması ŞU AN GÜVENİLİR DEĞİL — elle takip et."
                        )
                    except Exception:
                        pass
                continue
        except Exception:
            pass

        pnl_pct = (cur - entry) / entry * 100
        pos_key = f"{uid}_{sym}"

        with _alert_lock:
            state = _alert_state.setdefault(pos_key, {
                'sl_warned': False, 'sl_hit': False, 'sl_confirm': 0, 'sl_executed': False,
                'tp1_warned': False, 'tp1_hit': False, 'tp1_confirm': 0, 'tp1_executed': False,
                'tp2_warned': False, 'tp2_hit': False, 'tp2_confirm': 0, 'tp2_executed': False,
                'tp3_hit': False, 'tp3_confirm': 0, 'tp3_executed': False,
            })
            # Geriye-uyumluluk: eski state'lerde yeni alanlar yoksa ekle
            for k in ('sl_confirm','sl_executed','tp1_confirm','tp1_executed',
                      'tp2_hit','tp2_confirm','tp2_executed','tp3_confirm','tp3_executed'):
                state.setdefault(k, 0 if k.endswith('_confirm') else False)

        msgs = []
        auto_exec = uid in enabled_styles
        style = enabled_styles.get(uid, 'swing')

        # ---- STOP-LOSS ----
        if sl > 0:
            if cur <= sl:
                state['sl_confirm'] = state['sl_confirm'] + 1
                if not state['sl_hit']:
                    if auto_exec:
                        msgs.append(
                            f"🤖 <b>STOP-LOSS — {sym}</b>\n"
                            f"Fiyat: {cur:.2f} ≤ SL: {sl:.2f}\n"
                            f"📉 Zarar: %{pnl_pct:.1f}\n"
                            f"⏳ Bot otomatik kapatıyor (1 tick onay daha)..."
                        )
                    else:
                        msgs.append(
                            f"🚨 <b>STOP-LOSS — {sym}</b>\n"
                            f"Fiyat: {cur:.2f} ≤ SL: {sl:.2f}\n"
                            f"📉 Zarar: %{pnl_pct:.1f}\n"
                            f"⚠️ <b>SATIŞ YAPINIZ</b>"
                        )
                    state['sl_hit'] = True
                # Auto-execute: 2 ardışık tick + cfg.enabled
                if (auto_exec and not state['sl_executed']
                        and state['sl_confirm'] >= _RT_CONFIRM_TICKS):
                    try:
                        from auto_trader import _auto_close_position, _auto_log_trade
                        from auto_trader_risk import _sl_cooldown_block, _panic_clear
                        # #4d: yalniz GERÇEKTEN kapatan thread log+cooldown yazsin (cift SELL_SL onle)
                        if _auto_close_position(pid, cur, f"Stop-Loss tetiklendi ({sl:.2f})"):
                            _auto_log_trade(uid, sym, 'SELL_SL', cur, qty,
                                            f"SL RT: {cur:.2f} <= {sl:.2f}", 0, 0, pid)
                            _sl_cooldown_block(uid, sym, style)
                        _panic_clear(pid)
                        state['sl_executed'] = True
                        print(f"[RT-EXEC] {sym} SL otomatik kapatildi @ {cur:.2f}")
                        # Pozisyon kapandı → state'i temizle (RAM + DB)
                        with _alert_lock:
                            _alert_state.pop(pos_key, None)
                        try:
                            from database import _db_delete_alert_state
                            _db_delete_alert_state(uid, sym, pid)
                        except Exception:
                            pass
                        # NOT: unsubscribe burada YAPILMAZ — _auto_close_position icinde zaten
                        # "baska acik pozisyon yoksa" kontroluyle yapiliyor. Buradan ek cagri
                        # cift unsubscribe yaratir + BIST100 sync zaten yine abone yapacaktir.
                        # Telegram'a son haber
                        if send_telegram and msgs:
                            for m in msgs:
                                try: send_telegram(m)
                                except Exception: pass
                            msgs = []
                        continue
                    except Exception as ex:
                        print(f"[RT-EXEC] {sym} SL kapatma hatasi: {ex}")
            else:
                state['sl_confirm'] = 0
                if (not state['sl_hit'] and not state['sl_warned']
                        and cur <= sl * 1.02):
                    msgs.append(
                        f"⚠️ <b>SL Yaklaşıyor — {sym}</b>\n"
                        f"Fiyat: {cur:.2f} | SL: {sl:.2f} "
                        f"(%{abs((cur - sl) / sl * 100):.1f} uzakta)"
                    )
                    state['sl_warned'] = True

        # ---- TAKE-PROFIT (yuksekten alcaga) ----
        if tp3 > 0 and cur >= tp3:
            state['tp3_confirm'] = state['tp3_confirm'] + 1
            if not state['tp3_hit']:
                if auto_exec:
                    msgs.append(
                        f"🎯 <b>TP3 HEDEFİ — {sym}</b>\n"
                        f"Fiyat: {cur:.2f} ≥ TP3: {tp3:.2f}\n"
                        f"📈 Kâr: %{pnl_pct:.1f}\n"
                        f"📲 Onayın için Telegram'a istek gönderiliyor..."
                    )
                else:
                    msgs.append(
                        f"🎯 <b>TP3 HEDEFİ — {sym}</b>\n"
                        f"Fiyat: {cur:.2f} ≥ TP3: {tp3:.2f}\n"
                        f"📈 Kâr: %{pnl_pct:.1f} — <b>SATIŞ ÖNERİLİR</b>"
                    )
                state['tp3_hit'] = True
            # Onaysiz icra YOK — Telegram onayi iste (tek sefer; onay akisi devralir)
            if (auto_exec and not state['tp3_executed']
                    and state['tp3_confirm'] >= _RT_CONFIRM_TICKS):
                try:
                    from auto_trader import _tp_take_profit
                    _tp_take_profit(uid, pid, sym, 'full', 'take_profit3', qty, cur, tp3)
                    state['tp3_executed'] = True
                    print(f"[RT-EXEC] {sym} TP3 onay istegi gonderildi @ {cur:.2f}")
                except Exception as ex:
                    print(f"[RT-EXEC] {sym} TP3 onay istegi hatasi: {ex}")
        elif tp2 > 0 and cur >= tp2:
            state['tp2_confirm'] = state['tp2_confirm'] + 1
            if not state['tp2_hit']:
                if auto_exec:
                    msgs.append(
                        f"🎯 <b>TP2 HEDEFİ — {sym}</b>\n"
                        f"Fiyat: {cur:.2f} ≥ TP2: {tp2:.2f}\n"
                        f"📈 Kâr: %{pnl_pct:.1f}\n"
                        f"📲 Onayın için Telegram'a istek gönderiliyor..."
                    )
                else:
                    msgs.append(
                        f"🎯 <b>TP2 HEDEFİ — {sym}</b>\n"
                        f"Fiyat: {cur:.2f} ≥ TP2: {tp2:.2f}\n"
                        f"📈 Kâr: %{pnl_pct:.1f} — <b>SATIŞ ÖNERİLİR</b>"
                    )
                state['tp2_hit'] = True
                state['tp2_warned'] = True
            # Onaysiz icra YOK — Telegram onayi iste
            if (auto_exec and not state['tp2_executed']
                    and state['tp2_confirm'] >= _RT_CONFIRM_TICKS):
                try:
                    from auto_trader import _tp_take_profit
                    sell_qty = int(qty * 0.5) or int(qty)
                    _tp_take_profit(uid, pid, sym, 'partial', 'take_profit2', sell_qty, cur, tp2)
                    state['tp2_executed'] = True
                    print(f"[RT-EXEC] {sym} TP2 onay istegi gonderildi @ {cur:.2f}")
                except Exception as ex:
                    print(f"[RT-EXEC] {sym} TP2 onay istegi hatasi: {ex}")
        elif tp1 > 0 and cur >= tp1:
            state['tp1_confirm'] = state['tp1_confirm'] + 1
            # tp_strategy='all_at_tp1' ise TP1'de tamamini sat (full close, partial degil)
            _all_at_tp1 = (tp_strategy == 'all_at_tp1')
            if not state['tp1_hit']:
                if auto_exec:
                    _t1_tail = "tüm pozisyon" if _all_at_tp1 else "%50 kısmi"
                    msgs.append(
                        f"🎯 <b>TP1 HEDEFİ — {sym}</b>\n"
                        f"Fiyat: {cur:.2f} ≥ TP1: {tp1:.2f}\n"
                        f"📈 Kâr: %{pnl_pct:.1f}\n"
                        f"📲 Onayın için Telegram'a istek gönderiliyor ({_t1_tail})..."
                    )
                else:
                    msgs.append(
                        f"🎯 <b>TP1 HEDEFİ — {sym}</b>\n"
                        f"Fiyat: {cur:.2f} ≥ TP1: {tp1:.2f}\n"
                        f"📈 Kâr: %{pnl_pct:.1f}"
                    )
                state['tp1_hit'] = True
            # Onaysiz icra YOK — Telegram onayi iste (all-at-tp1 ise tam, degilse %50 kismi)
            if (auto_exec and not state['tp1_executed']
                    and state['tp1_confirm'] >= _RT_CONFIRM_TICKS):
                try:
                    from auto_trader import _tp_take_profit
                    if _all_at_tp1:
                        _tp_take_profit(uid, pid, sym, 'full', 'take_profit1', qty, cur, tp1)
                        # Ayni anda TP2/TP3 icin ek istek cikmasin
                        state['tp2_executed'] = state['tp3_executed'] = True
                    else:
                        sell_qty = int(qty * 0.5) or int(qty)
                        _tp_take_profit(uid, pid, sym, 'partial', 'take_profit1', sell_qty, cur, tp1)
                    state['tp1_executed'] = True
                    print(f"[RT-EXEC] {sym} TP1 onay istegi gonderildi @ {cur:.2f}")
                except Exception as ex:
                    print(f"[RT-EXEC] {sym} TP1 onay istegi hatasi: {ex}")
        else:
            # Hicbir TP'de degil — confirm sayaclarini sifirla (spike kayboldu)
            state['tp1_confirm'] = state['tp2_confirm'] = state['tp3_confirm'] = 0
            if (tp1 > 0 and not state['tp1_warned'] and not state['tp1_hit']
                    and cur >= tp1 * 0.98):
                msgs.append(
                    f"📍 <b>TP1 Yaklaşıyor — {sym}</b>\n"
                    f"Fiyat: {cur:.2f} | TP1: {tp1:.2f}"
                )
                state['tp1_warned'] = True

        with _alert_lock:
            _alert_state[pos_key] = state

        # Persist (yalnizca flag degismis ise) — DB'ye yazma maliyetini azaltmak icin
        # her tick degil, anlamli flag transition'larinda yazariz.
        try:
            if any(state.get(k) for k in (
                'sl_warned', 'sl_hit', 'sl_executed',
                'tp1_warned', 'tp1_hit', 'tp1_executed',
                'tp2_warned', 'tp2_hit', 'tp2_executed',
                'tp3_hit', 'tp3_executed',
            )):
                from database import _db_save_alert_state
                _db_save_alert_state(uid, sym, pid, state)
        except Exception:
            pass

        if send_telegram:
            for msg in msgs:
                try:
                    send_telegram(msg)
                except Exception:
                    pass
