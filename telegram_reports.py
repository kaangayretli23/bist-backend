"""
Telegram Otomatik Rapor Sistemleri
Günlük/haftalık performans raporları ve BIST piyasa brifingleri.
routes_telegram.py'dan ayrıştırıldı (700 satır kuralı).
"""
import time
from datetime import datetime, timedelta

from routes_telegram import send_telegram, send_news_telegram, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TELEGRAM_NEWS_BOT_TOKEN, TELEGRAM_NEWS_CHAT_ID
from config import _get_stocks
from signals_market import REGIMES_STRONG

try:
    from signals import calc_market_regime, check_signal_alerts
except ImportError:
    pass


def _build_market_report(period='close'):
    """Gunluk BIST piyasa raporu.
    period:
      'yesterday' — 09:25 raporu: dünün kapanış verisi (cache'deki son veri)
      'open'      — 10:25 raporu: bugünün canlı açılış verisi + sinyaller
      'close'     — 18:30 raporu: günün kapanışı + en çok değişenler
    """
    try:
        from config import _get_stocks, _get_indices, SECTOR_MAP
        stocks  = _get_stocks()
        indices = _get_indices()

        now      = datetime.now()
        now_str  = now.strftime('%d.%m.%Y %H:%M')
        yest_str = (now - timedelta(days=1)).strftime('%d.%m.%Y')

        if not stocks:
            return f"📊 <b>BIST Piyasa Raporu</b> ({now_str})\n⚠️ Hisse verisi henüz yüklenmedi."

        # Endeks verileri
        index_lines = []
        for key, data in (indices or {}).items():
            if key in ('XU100', 'XU030', 'XBANK', 'USDTRY'):
                chg   = data.get('changePct', 0)
                emoji = '🟢' if chg > 0 else ('🔴' if chg < 0 else '⚪')
                index_lines.append(
                    f"{emoji} <b>{data.get('name', key)}</b>: "
                    f"{data.get('value', 0):.2f}  ({'+'if chg>=0 else ''}{chg:.2f}%)"
                )

        valid       = [s for s in stocks if s.get('changePct') is not None and s.get('price', 0) > 0]
        top_gainers = sorted(valid, key=lambda x: x.get('changePct', 0), reverse=True)[:5]
        top_losers  = sorted(valid, key=lambda x: x.get('changePct', 0))[:5]

        # Sektor performansi
        sector_lines = []
        for sector, syms in SECTOR_MAP.items():
            ss = [s for s in stocks if s.get('code') in syms]
            if ss:
                avg_chg = sum(s.get('changePct', 0) for s in ss) / len(ss)
                emoji   = '🟢' if avg_chg > 0.5 else ('🔴' if avg_chg < -0.5 else '⚪')
                sector_lines.append((avg_chg, f"{emoji} {sector.replace('_',' ').title()}: {'+'if avg_chg>=0 else ''}{avg_chg:.1f}%"))
        sector_lines.sort(key=lambda x: -x[0])

        rising  = sum(1 for s in valid if s.get('changePct', 0) > 0)
        falling = sum(1 for s in valid if s.get('changePct', 0) < 0)
        flat    = len(valid) - rising - falling

        # ── BAŞLIK ──
        if period == 'yesterday':
            title    = "🌙 <b>DÜNÜN KAPANIŞ ÖZETİ</b>"
            subtitle = f"📅 {yest_str} kapanışı"
        elif period == 'open':
            title    = "🌅 <b>BUGÜN AÇILIŞ BRİFİNGİ</b>"
            subtitle = f"📅 {now_str}"
        else:
            title    = "📉 <b>GÜNLÜK KAPANIŞ RAPORU</b>"
            subtitle = f"📅 {now_str}"

        lines = [title, subtitle, "━━━━━━━━━━━━━━━━━━"]

        if index_lines:
            lines.append("<b>Endeksler:</b>")
            lines += index_lines

        lines += [
            "━━━━━━━━━━━━━━━━━━",
            f"<b>Piyasa Geneli:</b>  🟢 {rising}  🔴 {falling}  ⚪ {flat}",
            "━━━━━━━━━━━━━━━━━━",
        ]

        if period == 'yesterday':
            # Dün: sadece özet + sektörler (açılış sinyali yok, veri eski)
            lines.append("<b>Dünün En Çok Yükselenleri:</b>")
            for s in top_gainers:
                lines.append(f"  🚀 {s['code']}: {s['price']:.2f} TL  (+%{s['changePct']:.2f})")
            lines.append("<b>Dünün En Çok Düşenleri:</b>")
            for s in top_losers:
                lines.append(f"  🔻 {s['code']}: {s['price']:.2f} TL  (%{s['changePct']:.2f})")
            if sector_lines:
                lines += ["━━━━━━━━━━━━━━━━━━", "<b>Sektör Performansı (Dün):</b>"]
                lines += [sl[1] for sl in sector_lines[:8]]
            lines += ["━━━━━━━━━━━━━━━━━━", "⏰ <i>10:25'te bugünün açılış raporu gelecek.</i>"]

        elif period == 'open':
            # 10:25 raporu: canlı veri + sinyaller
            lines.append("<b>Bugünün En Çok Yükselenleri:</b>")
            for s in top_gainers:
                lines.append(f"  🚀 {s['code']}: {s['price']:.2f} TL  (+%{s['changePct']:.2f})")
            lines.append("<b>Bugünün En Çok Düşenleri:</b>")
            for s in top_losers:
                lines.append(f"  🔻 {s['code']}: {s['price']:.2f} TL  (%{s['changePct']:.2f})")
            if sector_lines:
                lines += ["━━━━━━━━━━━━━━━━━━", "<b>Sektörler:</b>"]
                lines += [sl[1] for sl in sector_lines[:8]]
            # Güçlü sinyaller
            try:
                signal_alerts = check_signal_alerts()
                bulls = [a for a in (signal_alerts or []) if a.get('signal') == 'bullish'][:5]
                if bulls:
                    lines += ["━━━━━━━━━━━━━━━━━━", "<b>Bugün İzlenecek Fırsatlar:</b>"]
                    for a in bulls:
                        lines.append(f"  🎯 {a['message']}")
            except Exception:
                pass

        else:  # close
            lines.append("<b>En Çok Yükselen:</b>")
            for s in top_gainers:
                lines.append(f"  🚀 {s['code']}: {s['price']:.2f} TL  (+%{s['changePct']:.2f})")
            lines.append("<b>En Çok Düşen:</b>")
            for s in top_losers:
                lines.append(f"  🔻 {s['code']}: {s['price']:.2f} TL  (%{s['changePct']:.2f})")
            if sector_lines:
                lines += ["━━━━━━━━━━━━━━━━━━", "<b>Sektörler:</b>"]
                lines += [sl[1] for sl in sector_lines[:8]]

        return '\n'.join(lines)
    except Exception as e:
        return f"[Piyasa raporu hatası: {e}]"


def _build_performance_report(days=1):
    """Gunluk veya haftalik performans raporu olustur"""
    try:
        import sqlite3
        conn = sqlite3.connect('bist.db')

        since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d %H:%M:%S')
        label = "Günlük" if days == 1 else "Haftalık"
        period = datetime.now().strftime('%d.%m.%Y') if days == 1 else \
            f"{(datetime.now() - timedelta(days=7)).strftime('%d.%m')} – {datetime.now().strftime('%d.%m.%Y')}"

        # Kapanan pozisyonlar
        closed = conn.execute('''
            SELECT symbol, entry_price, close_price, pnl, pnl_pct, close_reason, opened_at, closed_at
            FROM auto_positions
            WHERE status = 'closed' AND closed_at >= ?
            ORDER BY closed_at DESC
        ''', (since,)).fetchall()

        # Açık pozisyonlar
        open_pos = conn.execute('''
            SELECT symbol, entry_price, quantity, stop_loss, take_profit1
            FROM auto_positions WHERE status = 'open'
        ''').fetchall()

        # Config - sermaye
        cfg = conn.execute('SELECT capital FROM auto_config LIMIT 1').fetchone()
        capital = cfg[0] if cfg else 100000

        conn.close()

        wins = [r for r in closed if r[3] > 0]
        losses = [r for r in closed if r[3] <= 0]
        total_pnl = sum(r[3] for r in closed)
        win_rate = (len(wins) / len(closed) * 100) if closed else 0

        lines = [
            f"📊 <b>BIST Pro {label} Raporu</b>",
            f"📅 {period}",
            f"━━━━━━━━━━━━━━━━━━",
        ]

        if closed:
            lines += [
                f"✅ Kazanan: {len(wins)}  |  ❌ Kaybeden: {len(losses)}",
                f"🎯 İsabet Oranı: %{win_rate:.0f}",
                f"💰 Dönem PnL: <b>{'+'if total_pnl>=0 else ''}{total_pnl:.0f} TL</b>  "
                f"(%{total_pnl/capital*100:.2f} sermaye)",
                f"━━━━━━━━━━━━━━━━━━",
                f"<b>İşlem Detayları:</b>",
            ]
            for r in closed:
                emoji = "💚" if r[3] > 0 else "🔴"
                lines.append(
                    f"{emoji} {r[0]}: {r[1]:.2f}→{r[2]:.2f}  "
                    f"({'+'if r[3]>=0 else ''}{r[3]:.0f} TL / %{r[4]:.1f})  [{r[5]}]"
                )
        else:
            lines.append(f"Bu dönemde kapanan işlem yok.")

        if open_pos:
            lines += [
                f"━━━━━━━━━━━━━━━━━━",
                f"<b>Açık Pozisyonlar ({len(open_pos)}):</b>",
            ]
            for p in open_pos:
                lines.append(f"  • {p[0]}: giriş {p[1]:.2f}  SL {p[2]:.2f}  TP1 {p[3]:.2f}")

        return '\n'.join(lines)
    except Exception as e:
        return f"[Rapor hatası: {e}]"


def _performance_reporter():
    """Otomatik rapor zamanlayici:
       - Her gun 09:15: BIST acilis brifing + guclu sinyaller
       - Her gun 18:30: BIST kapanis raporu + gunluk performans
       - Her Pazartesi 09:00: Haftalik performans ozeti
    """
    last_yesterday = None   # 09:25 dünün kapanış özeti
    last_open      = None   # 10:25 bugünün açılış brifingı
    last_close     = None   # 18:30 kapanış raporu
    last_weekly    = None   # Pazartesi 09:00 haftalık

    while True:
        try:
            time.sleep(60)
            if not TELEGRAM_NEWS_BOT_TOKEN or not TELEGRAM_NEWS_CHAT_ID:
                continue

            now     = datetime.now()
            today   = now.date()
            weekday = now.weekday()  # 0=Pazartesi, 5=Cumartesi, 6=Pazar

            # Hafta sonu rapor gonderme
            if weekday >= 5:
                continue

            # 09:25 — Dünün kapanış özeti (09:25-09:54 arası)
            if now.hour == 9 and 25 <= now.minute <= 54 and last_yesterday != today:
                last_yesterday = today
                msg = _build_market_report(period='yesterday')
                send_news_telegram(msg)
                print("[TELEGRAM] Dünün kapanış özeti gönderildi")

            # 10:25 — Bugünün açılış brifingı, canlı veri (10:25-10:54 arası)
            if now.hour == 10 and 25 <= now.minute <= 54 and last_open != today:
                last_open = today
                msg = _build_market_report(period='open')
                send_news_telegram(msg)
                print("[TELEGRAM] Açılış brifingı gönderildi")

            # 18:30 — Kapanış + günlük performans (18:30-18:59 arası)
            if now.hour == 18 and 30 <= now.minute <= 59 and last_close != today:
                last_close = today
                market_msg = _build_market_report(period='close')
                perf_msg   = _build_performance_report(days=1)
                send_news_telegram(market_msg)
                time.sleep(2)
                send_news_telegram(perf_msg)
                print("[TELEGRAM] Kapanış + günlük performans gönderildi")

            # Pazartesi 09:00 — Haftalık performans (09:00-09:24 arası)
            if weekday == 0 and now.hour == 9 and now.minute <= 24 and last_weekly != today:
                last_weekly = today
                msg = _build_performance_report(days=7)
                send_news_telegram(msg)
                print("[TELEGRAM] Haftalık performans gönderildi")

        except Exception as _rep_err:
            print(f"[TELEGRAM-REPORTER] Hata: {_rep_err}")


# Gönderilmiş alert takibi: "SYM_type" → gönderilme zamanı (timestamp)
# Aynı alert 4 saat içinde tekrar gönderilmez.
_sent_alert_cache: dict = {}
_ALERT_COOLDOWN_SECS = 4 * 3600  # 4 saat

# Piyasa genelinde yaygın olduğu için tek başına anlamlı olmayan pattern adları.
# Bull piyasada onlarca hisse aynı anda tetikler → raporu tekrar doldurur.
# signals_confidence.py'daki p['name'] değerleriyle eşleşir.
_NOISY_PATTERNS = {'Uc Beyaz Asker', 'Uc Kara Karga', 'Rising Three', 'Falling Three'}

# Tip başına maksimum alert sayısı (raporun çeşitli kalması için)
_MAX_PER_TYPE = 3

# Minimum strength eşiği — bu değerin altındaki alertler gönderilmez
_MIN_STRENGTH = 4


def _auto_signal_check():
    """Arka planda güçlü sinyalleri tespit edip Telegram bildirimi gönder.

    Filtreler:
      - strength >= _MIN_STRENGTH (varsayılan 4)
      - Noisy (piyasa geneli) pattern'lar hariç
      - Aynı symbol+type kombinasyonu 4 saat içinde tekrar gönderilmez
      - Her tip için max _MAX_PER_TYPE alert
      - Sadece ≥3 benzersiz alert varsa rapor gönderilir (gürültü bastırma)
    """
    while True:
        try:
            time.sleep(1800)  # 30 dakikada bir kontrol (eskiden 10 dk)
            if not TELEGRAM_NEWS_BOT_TOKEN or not TELEGRAM_NEWS_CHAT_ID:
                continue

            stocks = _get_stocks()
            if not stocks:
                continue

            signal_alerts = check_signal_alerts()
            if not signal_alerts:
                continue

            regime = calc_market_regime()
            regime_type = regime.get('regime', '')

            now_ts = time.time()

            # Süresi dolan cache girdilerini temizle
            expired = [k for k, ts in _sent_alert_cache.items() if now_ts - ts > _ALERT_COOLDOWN_SECS]
            for k in expired:
                del _sent_alert_cache[k]

            # Filtreleme
            type_counts: dict = {}
            filtered = []
            for alert in signal_alerts:
                strength = alert.get('strength', 0)
                alert_type = alert.get('type', '')
                sym = alert.get('symbol', '')

                # Güven eşiği
                if strength < _MIN_STRENGTH:
                    continue

                # Piyasa geneli noisy candlestick pattern'lar — sadece güçlü rejim değişiminde gönder
                if alert.get('pattern_name') in _NOISY_PATTERNS:
                    if regime_type not in REGIMES_STRONG:
                        continue

                # Dedup: aynı symbol+type 4 saatte bir
                cache_key = f"{sym}_{alert_type}"
                if cache_key in _sent_alert_cache:
                    continue

                # Tip başına limit
                type_counts[alert_type] = type_counts.get(alert_type, 0) + 1
                if type_counts[alert_type] > _MAX_PER_TYPE:
                    continue

                filtered.append(alert)

            # Strength'e göre sırala: en güçlü sinyaller önce
            filtered.sort(key=lambda x: x.get('strength', 0), reverse=True)

            # Minimum 3 anlamlı alert yoksa gönderme
            if len(filtered) < 3:
                continue

            regime_emoji = {
                'strong_bull': '🐂🐂', 'bull': '🐂',
                'strong_bear': '🐻🐻', 'bear': '🐻',
                'sideways': '↔️',
            }.get(regime_type, '❓')

            alerts_text = []
            for alert in filtered[:12]:
                emoji = '🟢' if alert.get('signal') == 'bullish' else ('🔴' if alert.get('signal') == 'bearish' else '⚪')
                alerts_text.append(f"{emoji} {alert['message']}")
                # Cache'e kaydet
                _sent_alert_cache[f"{alert.get('symbol', '')}_{alert.get('type', '')}"] = now_ts

            header = f"📊 <b>BIST Sinyal Raporu</b> ({datetime.now().strftime('%H:%M')})\n"
            header += f"{regime_emoji} Piyasa: {regime.get('description', 'Bilinmiyor')}\n\n"
            send_news_telegram(header + '\n'.join(alerts_text))

        except Exception as _sig_err:
            print(f"[TELEGRAM-SIGNAL-CHECK] Hata: {_sig_err}")
            continue

