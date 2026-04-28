"""
BIST Pro - Auto Trader: Acik Pozisyon Yönetimi (ADIM 1)
SL/TP/trailing/panic-sell kontrollerini yürütür.
auto_trader_engine.py'dan ayrıştırıldı (600 satır kuralı).
"""
# Not: config, auto_trader, routes_telegram, data_fetcher, realtime_prices
# fonksiyon içinde lazy import edilir (circular/partial load önlemi).
from auto_trader_risk import (
    _panic_track_and_check, _panic_clear, _sl_cooldown_block,
)


def _step1_manage_positions(uid, cfg, positions):
    """Açık pozisyonları SL/TP/trailing açısından kontrol et."""
    from config import _cget, _cset, _stock_cache
    from auto_trader import (
        _auto_close_position, _auto_partial_close, _auto_log_trade,
        _auto_update_trailing, _auto_update_highest_price,
    )
    for pos in positions:
        sym = pos['symbol']
        # Gerçek zamanlı fiyatı önce dene, yoksa cache/fetch
        cur_price = 0.0
        try:
            from realtime_prices import get_price as _rt_get
            rt_price = _rt_get(sym)
            if rt_price and rt_price > 0:
                cur_price = rt_price
        except Exception:
            pass
        if cur_price <= 0:
            stock = _cget(_stock_cache, sym)
            cur_price = float(stock.get('price', 0)) if stock else 0
        if cur_price <= 0:
            try:
                from data_fetcher import _process_stock
                _, fresh = _process_stock(sym, retry_count=1)
                if fresh:
                    _cset(_stock_cache, sym, fresh)
                    cur_price = float(fresh.get('price', 0))
            except Exception:
                pass
        if cur_price <= 0:
            print(f"[AUTO-TRADE] {sym} fiyat alinamadi, SL/TP kontrolu atlandi")
            continue

        # Stop-Loss kontrolu
        sl = pos['stopLoss']
        if sl > 0 and cur_price <= sl:
            _auto_close_position(pos['id'], cur_price, f"Stop-Loss tetiklendi ({sl:.2f})")
            _auto_log_trade(uid, sym, 'SELL_SL', cur_price, pos['quantity'],
                           f"SL tetiklendi: {cur_price:.2f} <= {sl:.2f}", 0, 0, pos['id'])
            _sl_cooldown_block(uid, sym, cfg.get('tradeStyle', 'swing'))
            _panic_clear(pos['id'])
            continue

        # SL yaklasma uyarisi (%2 icinde)
        if sl > 0 and cur_price <= sl * 1.02:
            try:
                from routes_telegram import send_sl_warning
                send_sl_warning(sym, cur_price, sl)
            except Exception:
                pass

        # Take-Profit kontrolu — kademeli kâr alma
        # TP1: %50 sat, TP2: kalanın %50'si, TP3: tamamını kapat
        tp3 = pos['takeProfit3']
        tp2 = pos['takeProfit2']
        tp1 = pos['takeProfit1']
        qty = pos['quantity']
        if tp3 > 0 and cur_price >= tp3:
            _auto_close_position(pos['id'], cur_price, f"TP3 hedef ({tp3:.2f})")
            _auto_log_trade(uid, sym, 'SELL_TP3', cur_price, qty,
                           f"TP3: {cur_price:.2f} >= {tp3:.2f}", 0, 0, pos['id'])
            _panic_clear(pos['id'])
            continue
        elif tp2 > 0 and cur_price >= tp2:
            sell_qty = int(qty * 0.5)
            if sell_qty < 1:
                sell_qty = qty  # çok az kaldıysa hepsini sat
            _auto_partial_close(pos['id'], sell_qty, cur_price,
                               f"TP2 hedef ({tp2:.2f})", clear_tp_field='take_profit2')
            _auto_log_trade(uid, sym, 'SELL_TP2', cur_price, sell_qty,
                           f"TP2 kısmi: {cur_price:.2f} >= {tp2:.2f}", 0, 0, pos['id'])
        elif tp1 > 0 and cur_price >= tp1:
            sell_qty = int(qty * 0.5)
            if sell_qty < 1:
                sell_qty = qty
            _auto_partial_close(pos['id'], sell_qty, cur_price,
                               f"TP1 hedef ({tp1:.2f})", clear_tp_field='take_profit1')
            _auto_log_trade(uid, sym, 'SELL_TP1', cur_price, sell_qty,
                           f"TP1 kısmi: {cur_price:.2f} >= {tp1:.2f}", 0, 0, pos['id'])

        # TP yaklasma bildirimi (%2 icinde, henuz tetiklenmemis)
        try:
            from routes_telegram import send_tp_approaching
            if tp1 > 0 and cur_price >= tp1 * 0.98 and cur_price < tp1:
                send_tp_approaching(sym, cur_price, tp1, "TP1")
            elif tp2 > 0 and cur_price >= tp2 * 0.98 and cur_price < tp2:
                send_tp_approaching(sym, cur_price, tp2, "TP2")
            elif tp3 > 0 and cur_price >= tp3 * 0.98 and cur_price < tp3:
                send_tp_approaching(sym, cur_price, tp3, "TP3")
        except Exception:
            pass

        # Trailing Stop kontrolu
        # Runner mode: TP2 hit edilmis pozisyon (tp1=0 AND tp2=0 AND tp3>0) →
        #   sıkı trail (tightTrailingPct), Telegram onayı bypass, master flag bypass.
        # Mantik: TP2 sonrasi kazanan pozisyon, momentum verince TP3'e ya da daha
        # uzaga gidebilir; ama donerse hizli kilitlemeliyiz (kar geri verilmesin).
        _tp2_hit = (tp1 == 0 and tp2 == 0 and tp3 > 0)
        _trailing_enabled = bool(cfg['trailingStop']) or _tp2_hit
        _trail_pct = float(cfg.get('tightTrailingPct', 1.0)) if _tp2_hit else float(cfg['trailingPct'])
        # Tick-aware trail: dusuk fiyatli (mikro) hisselerde nominal %3 trail cok dar
        # kaliyor (orn. TUKAS @2.60 + %3 = 0.078 TL = ~8 tick — normal volatilite).
        # 10 TL alti hisselerde min trail %5; 5 TL alti %6 olsun. Runner mode'da
        # (tightTrailing) bu bypass'i UYGULAMA — TP2 sonrasi sikiyiz.
        if not _tp2_hit:
            if cur_price < 5.0:
                _trail_pct = max(_trail_pct, 6.0)
            elif cur_price < 10.0:
                _trail_pct = max(_trail_pct, 5.0)
        if _trailing_enabled:
            highest = pos['highestPrice']
            if cur_price > highest:
                new_trailing = cur_price * (1 - _trail_pct / 100)
                import os as _os_tg
                _tg_configured = bool(_os_tg.environ.get('TELEGRAM_BOT_TOKEN') and _os_tg.environ.get('TELEGRAM_CHAT_ID'))
                if _tg_configured and not _tp2_hit:
                    # Telegram var ve henüz TP2 öncesi: highest update + onay bekle
                    _auto_update_highest_price(pos['id'], cur_price)
                    try:
                        from routes_telegram import send_trailing_update
                        send_trailing_update(sym, new_trailing, cur_price, pos.get('entryPrice'),
                                             tp1=tp1, tp2=tp2, tp3=tp3, position_id=pos['id'])
                    except Exception as _tr_err:
                        print(f"[AUTO-TRADER] Trailing bildirim hatası ({sym}): {_tr_err}")
                else:
                    # Runner mode VEYA Telegram yok: trail'i doğrudan güncelle
                    _auto_update_trailing(pos['id'], round(new_trailing, 2), cur_price)
                    if _tp2_hit:
                        print(f"[AUTO-TRADE] {sym} runner trail: {new_trailing:.2f} "
                              f"(TP2 sonrası, %{_trail_pct} sıkı trail)")
            else:
                # Acilis ilk 15 dk trailing tetiklemesini atla (gap koruması);
                # highest yine guncellenir ama stop tetiklenmez.
                from datetime import datetime, timezone, timedelta
                _tr_now = datetime.now(timezone(timedelta(hours=3)))
                _opening_window = (_tr_now.weekday() < 5 and
                                   _tr_now.hour == 10 and _tr_now.minute < 15)
                trailing_sl = pos['trailingStop']
                if (not _opening_window) and trailing_sl > 0 and cur_price <= trailing_sl:
                    _auto_close_position(pos['id'], cur_price, f"Trailing-Stop ({trailing_sl:.2f})")
                    _auto_log_trade(uid, sym, 'SELL_TRAIL', cur_price, pos['quantity'],
                                   f"Trailing SL: {cur_price:.2f} <= {trailing_sl:.2f}", 0, 0, pos['id'])
                    _sl_cooldown_block(uid, sym, cfg.get('tradeStyle', 'swing'))
                    _panic_clear(pos['id'])
                    continue
                if _opening_window and trailing_sl > 0 and cur_price <= trailing_sl:
                    print(f"[AUTO-TRADE] {sym} trailing stop tetiklendi fakat acilis penceresinde (ilk 15dk) — atlaniyor")

        # Panic-Sell kontrolü (ani ters dönüş — SL beklemeden çık)
        if cfg.get('panicSellEnabled'):
            try:
                drop_pct = float(cfg.get('panicDropPct', 2.0))
                window_min = int(cfg.get('panicWindowMin', 5))
                triggered, peak, drop = _panic_track_and_check(
                    pos['id'], cur_price, pos['entryPrice'], drop_pct, window_min
                )
                if triggered:
                    reason = f"Panic-Sell: {window_min}dk içinde zirveden %{drop:.1f} düşüş ({peak:.2f}→{cur_price:.2f})"
                    _auto_close_position(pos['id'], cur_price, reason)
                    _auto_log_trade(uid, sym, 'SELL_PANIC', cur_price, pos['quantity'],
                                   reason, 0, 0, pos['id'])
                    _panic_clear(pos['id'])
                    try:
                        from routes_telegram import send_telegram
                        send_telegram(
                            f"🚨 <b>PANİK SATIŞ: {sym}</b>\n"
                            f"🔝 Zirve: {peak:.2f} TL\n"
                            f"💰 Şu an: {cur_price:.2f} TL  (-%{drop:.1f})\n"
                            f"⏱ Pencere: {window_min} dk\n"
                            f"⚡ Midas'ta hemen sat — ani ters dönüş tespit edildi."
                        )
                    except Exception:
                        pass
                    continue
            except Exception as _panic_err:
                print(f"[AUTO-TRADE] Panic-sell kontrolü hatası ({sym}): {_panic_err}")
