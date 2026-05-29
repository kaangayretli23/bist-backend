"""
BIST Pro - Vergi Donemi Islem Raporu
Yillik kapatilmis pozisyon ozeti (bilgi amacli, resmi vergi beyani DEGIL).

Endpoints:
  GET /api/auto-trade/tax-report?userId=X&year=2026     -> JSON ozet + detay
  GET /api/auto-trade/tax-report.pdf?userId=X&year=2026 -> PDF dosyasi

reportlab kullanir. requirements.txt'e ekli.
Turkce karakter destegi icin Windows Arial fontunu register eder; bulunamazsa
Helvetica'ya duser (Turkce karakter bozulabilir).
"""
import io
import os
from datetime import datetime
from flask import jsonify, request, send_file

from config import app, get_db, safe_dict
from auth_middleware import require_user
from auto_trader import _auto_get_config


def _fetch_closed_in_year(uid: str, year: int):
    """O yil icinde kapatilmis tum pozisyonlari getir (closed_at o yila aitse)."""
    start = f"{year}-01-01 00:00:00"
    end = f"{year + 1}-01-01 00:00:00"
    db = get_db()
    try:
        rows = db.execute(
            "SELECT id, symbol, entry_price, close_price, quantity, "
            "pnl, pnl_pct, close_reason, created_at, closed_at "
            "FROM auto_positions WHERE user_id=? AND status='closed' "
            "AND closed_at >= ? AND closed_at < ? "
            "ORDER BY closed_at ASC",
            (uid, start, end)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        db.close()


def _compute_tax_summary(uid: str, year: int):
    """Yillik ozet + detay listesi."""
    rows = _fetch_closed_in_year(uid, year)
    cfg = _auto_get_config(uid) or {}
    c_pct = float(cfg.get('commissionPct', 0) or 0)
    b_pct = float(cfg.get('bsmvPct', 0) or 0)

    total_buy = 0.0
    total_sell = 0.0
    total_gross = 0.0
    total_costs = 0.0
    total_net = 0.0
    winners = 0
    losers = 0
    by_symbol: dict = {}

    detail = []
    for r in rows:
        entry = float(r['entry_price'] or 0)
        close = float(r['close_price'] or 0)
        qty = float(r['quantity'] or 0)
        notional_buy = entry * qty
        notional_sell = close * qty
        gross = (close - entry) * qty
        # Komisyon + BSMV (yillik raporda hesabi yeniden yap, cunku DB'deki pnl
        # kismi satislarda farkli notional kullanmis olabilir — ozet icin tutarli olsun)
        if c_pct > 0:
            comm = (notional_buy + notional_sell) * (c_pct / 100.0)
            bsmv = comm * (b_pct / 100.0)
            costs = comm + bsmv
        else:
            costs = 0.0
        net = gross - costs

        total_buy += notional_buy
        total_sell += notional_sell
        total_gross += gross
        total_costs += costs
        total_net += net
        if net > 0: winners += 1
        elif net < 0: losers += 1
        sym = r['symbol']
        s = by_symbol.setdefault(sym, {'symbol': sym, 'trades': 0, 'netPnL': 0.0, 'volume': 0.0})
        s['trades'] += 1
        s['netPnL'] += net
        s['volume'] += notional_buy + notional_sell

        detail.append({
            'id': int(r['id']),
            'symbol': sym,
            'openedAt': r['created_at'],
            'closedAt': r['closed_at'],
            'entryPrice': round(entry, 4),
            'closePrice': round(close, 4),
            'quantity': qty,
            'notionalBuy': round(notional_buy, 2),
            'notionalSell': round(notional_sell, 2),
            'grossPnL': round(gross, 2),
            'costs': round(costs, 2),
            'netPnL': round(net, 2),
            'pnlPct': float(r['pnl_pct'] or 0),
            'reason': r['close_reason'] or '',
        })

    total_trades = winners + losers
    win_rate = round(winners / total_trades * 100, 1) if total_trades > 0 else 0.0
    by_symbol_list = sorted(by_symbol.values(), key=lambda x: x['netPnL'], reverse=True)
    for s in by_symbol_list:
        s['netPnL'] = round(s['netPnL'], 2)
        s['volume'] = round(s['volume'], 2)

    summary = {
        'year': year,
        'totalTrades': total_trades,
        'winners': winners,
        'losers': losers,
        'winRate': win_rate,
        'totalBuy': round(total_buy, 2),
        'totalSell': round(total_sell, 2),
        'totalVolume': round(total_buy + total_sell, 2),
        'grossPnL': round(total_gross, 2),
        'totalCosts': round(total_costs, 2),
        'netPnL': round(total_net, 2),
        'commissionPct': c_pct,
        'bsmvPct': b_pct,
        'generatedAt': datetime.now().isoformat(timespec='seconds'),
    }
    return {'summary': summary, 'detail': detail, 'bySymbol': by_symbol_list}


@app.route('/api/auto-trade/tax-report')
@require_user
def auto_trade_tax_report():
    """Yillik islem ozeti — JSON (frontend'de onizleme icin)."""
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400
        try:
            year = int(request.args.get('year', datetime.now().year))
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Gecersiz yil'}), 400
        if not (2000 <= year <= 2100):
            return jsonify({'success': False, 'error': 'Yil 2000..2100 araliginda olmali'}), 400
        result = _compute_tax_summary(uid, year)
        return jsonify(safe_dict({'success': True, **result}))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


def _register_tr_font():
    """Windows Arial fontunu reportlab'e kaydet (Turkce karakter destegi).
    Bulunamazsa False doner -> Helvetica fallback."""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        candidates = [
            ('Arial',     r'C:\Windows\Fonts\arial.ttf'),
            ('Arial-Bold', r'C:\Windows\Fonts\arialbd.ttf'),
        ]
        ok = True
        for name, path in candidates:
            if os.path.exists(path):
                try:
                    pdfmetrics.registerFont(TTFont(name, path))
                except Exception:
                    ok = False
            else:
                ok = False
        return ok
    except Exception:
        return False


def _build_pdf(uid: str, year: int) -> bytes:
    """Yillik ozet PDF'i uret. Bytes doner."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
    )

    has_arial = _register_tr_font()
    font_normal = 'Arial' if has_arial else 'Helvetica'
    font_bold   = 'Arial-Bold' if has_arial else 'Helvetica-Bold'

    data = _compute_tax_summary(uid, year)
    s = data['summary']
    detail = data['detail']
    by_sym = data['bySymbol']

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.5 * cm, bottomMargin=1.5 * cm,
        title=f"BIST Pro - {year} Yillik Islem Ozeti",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'Title', parent=styles['Heading1'], fontName=font_bold,
        fontSize=18, alignment=1, spaceAfter=6,
    )
    sub_style = ParagraphStyle(
        'Sub', parent=styles['Normal'], fontName=font_normal,
        fontSize=10, alignment=1, textColor=colors.grey, spaceAfter=18,
    )
    h2 = ParagraphStyle(
        'H2', parent=styles['Heading2'], fontName=font_bold,
        fontSize=13, spaceBefore=14, spaceAfter=8,
    )
    body = ParagraphStyle(
        'Body', parent=styles['Normal'], fontName=font_normal,
        fontSize=10, leading=14,
    )
    note = ParagraphStyle(
        'Note', parent=styles['Normal'], fontName=font_normal,
        fontSize=8, textColor=colors.grey, leading=11, spaceBefore=10,
    )

    story = []
    story.append(Paragraph(f"{year} Yıllık İşlem Özeti", title_style))
    story.append(Paragraph(
        f"BIST Pro · Oluşturulma: {s['generatedAt']} · Kullanıcı: {uid}",
        sub_style
    ))

    # ÖZET TABLOSU
    story.append(Paragraph("📊 Özet", h2))
    summary_rows = [
        ['İşlem Sayısı', f"{s['totalTrades']}"],
        ['Kazanan / Kaybeden', f"{s['winners']} / {s['losers']}  (Başarı: %{s['winRate']:.1f})"],
        ['Toplam Alış Hacmi (TL)', f"{s['totalBuy']:,.2f}"],
        ['Toplam Satış Hacmi (TL)', f"{s['totalSell']:,.2f}"],
        ['Toplam İşlem Hacmi (TL)', f"{s['totalVolume']:,.2f}"],
        ['Brüt Kâr/Zarar (TL)', f"{s['grossPnL']:,.2f}"],
        [f"Komisyon + BSMV (TL)  [%{s['commissionPct']:.3f} + BSMV %{s['bsmvPct']:.1f}]",
         f"{s['totalCosts']:,.2f}"],
        ['Net Kâr/Zarar (TL)', f"{s['netPnL']:,.2f}"],
    ]
    summary_table = Table(summary_rows, colWidths=[10 * cm, 6 * cm])
    summary_table.setStyle(TableStyle([
        ('FONT', (0, 0), (-1, -1), font_normal, 10),
        ('FONT', (0, -1), (-1, -1), font_bold, 11),
        ('BACKGROUND', (0, -1), (-1, -1),
         colors.HexColor('#e8f5e9') if s['netPnL'] >= 0 else colors.HexColor('#ffebee')),
        ('TEXTCOLOR', (1, -1), (1, -1),
         colors.HexColor('#1b5e20') if s['netPnL'] >= 0 else colors.HexColor('#b71c1c')),
        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.append(summary_table)

    # HİSSE BAŞINA TOPLAM
    if by_sym:
        story.append(Paragraph("🏷 Hisse Bazında Net P&L", h2))
        sym_header = ['Hisse', 'İşlem', 'Hacim (TL)', 'Net P&L (TL)']
        sym_rows = [sym_header] + [
            [r['symbol'], str(r['trades']),
             f"{r['volume']:,.2f}", f"{r['netPnL']:+,.2f}"]
            for r in by_sym
        ]
        sym_table = Table(sym_rows, colWidths=[3 * cm, 2 * cm, 5 * cm, 5 * cm], repeatRows=1)
        sym_style = [
            ('FONT', (0, 0), (-1, 0), font_bold, 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#37474f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONT', (0, 1), (-1, -1), font_normal, 9),
            ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
            ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f5f5f5')]),
        ]
        # Net PnL renklendir
        for i, r in enumerate(by_sym, start=1):
            color = colors.HexColor('#1b5e20') if r['netPnL'] >= 0 else colors.HexColor('#b71c1c')
            sym_style.append(('TEXTCOLOR', (3, i), (3, i), color))
        sym_table.setStyle(TableStyle(sym_style))
        story.append(sym_table)

    # DETAY TABLOSU (yeni sayfa)
    if detail:
        story.append(PageBreak())
        story.append(Paragraph("📋 İşlem Detayı", h2))
        det_header = ['#', 'Hisse', 'Açılış', 'Kapanış', 'Adet',
                      'Giriş', 'Çıkış', 'Brüt P&L', 'Net P&L', '%']
        det_rows = [det_header]
        for i, t in enumerate(detail, start=1):
            opened = (t['openedAt'] or '')[:10]
            closed = (t['closedAt'] or '')[:10]
            det_rows.append([
                str(i), t['symbol'], opened, closed,
                f"{t['quantity']:g}",
                f"{t['entryPrice']:.2f}",
                f"{t['closePrice']:.2f}",
                f"{t['grossPnL']:+,.2f}",
                f"{t['netPnL']:+,.2f}",
                f"{t['pnlPct']:+.1f}",
            ])
        det_table = Table(
            det_rows,
            colWidths=[0.8 * cm, 1.8 * cm, 2.0 * cm, 2.0 * cm, 1.4 * cm,
                       1.6 * cm, 1.6 * cm, 2.2 * cm, 2.2 * cm, 1.4 * cm],
            repeatRows=1,
        )
        det_style = [
            ('FONT', (0, 0), (-1, 0), font_bold, 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#37474f')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONT', (0, 1), (-1, -1), font_normal, 8),
            ('GRID', (0, 0), (-1, -1), 0.2, colors.grey),
            ('ALIGN', (2, 0), (-1, -1), 'RIGHT'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1),
             [colors.white, colors.HexColor('#f5f5f5')]),
        ]
        # Net PnL ve % renklendir
        for i, t in enumerate(detail, start=1):
            color = colors.HexColor('#1b5e20') if t['netPnL'] >= 0 else colors.HexColor('#b71c1c')
            det_style.append(('TEXTCOLOR', (7, i), (9, i), color))
        det_table.setStyle(TableStyle(det_style))
        story.append(det_table)

    # ALT NOT
    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "<b>Önemli Uyarı:</b> Bu rapor yalnızca bilgilendirme amaçlıdır ve resmi "
        "vergi beyanı yerine geçmez. Türkiye'de BIST hisse senedi alım-satım kazançlarında "
        "tevkifat oranı dönemsel olarak değişebilir; resmi beyan için aracı kurum "
        "hesap özetinizi ve mali müşavirinizi referans alın.",
        note
    ))

    doc.build(story)
    return buf.getvalue()


@app.route('/api/auto-trade/tax-report.pdf')
@require_user
def auto_trade_tax_report_pdf():
    """Yillik islem ozeti — PDF dosya indirme."""
    try:
        uid = request.args.get('userId', '')
        if not uid:
            return jsonify({'success': False, 'error': 'userId gerekli'}), 400
        try:
            year = int(request.args.get('year', datetime.now().year))
        except (TypeError, ValueError):
            return jsonify({'success': False, 'error': 'Gecersiz yil'}), 400
        if not (2000 <= year <= 2100):
            return jsonify({'success': False, 'error': 'Yil 2000..2100 araliginda olmali'}), 400

        pdf_bytes = _build_pdf(uid, year)
        return send_file(
            io.BytesIO(pdf_bytes),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f"bistpro_islem_ozeti_{year}.pdf",
        )
    except ImportError as e:
        return jsonify({
            'success': False,
            'error': f'reportlab kutuphanesi yuklu degil: {e}. requirements.txt yukleyin.'
        }), 500
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500
