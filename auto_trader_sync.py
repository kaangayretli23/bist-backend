"""
BIST Pro - Auto Trader Portfoy Senkronizasyonu
Auto-trader BUY/SELL islemlerini kullanicinin manuel `portfolios` tablosuna yansitir.

Sebep: Artik paper-trading yapmiyoruz. Kullanici Midas'ta gercek alim-satim yapiyor;
oto-trader sinyal/kapatma karari verdiginde portfoy de essenkron guncellenmeli.

avg_cost weighted average kuralini kullanir (routes_portfolio.add_portfolio ile ayni).
SELL'de qty azaltir; sifirlanirsa satir silinir; avg_cost korunur (kalan ayni maliyetli).
"""
from config import db_conn


def _sync_portfolio_buy(uid: str, symbol: str, quantity: float, price: float) -> None:
    """Auto-trader BUY -> portfolios'a ekle (weighted avg ile birlestir)."""
    if not uid or not symbol or quantity <= 0 or price <= 0:
        return
    sym = symbol.upper()
    try:
        with db_conn() as db:
            existing = db.execute(
                "SELECT id, quantity, avg_cost FROM portfolios WHERE user_id=? AND symbol=?",
                (uid, sym)
            ).fetchone()
            if existing:
                old_qty = float(existing['quantity'])
                old_avg = float(existing['avg_cost'])
                new_qty = old_qty + quantity
                new_avg = (old_avg * old_qty + price * quantity) / new_qty if new_qty > 0 else price
                db.execute(
                    "UPDATE portfolios SET quantity=?, avg_cost=? WHERE id=?",
                    (round(new_qty, 4), round(new_avg, 4), existing['id'])
                )
            else:
                db.execute(
                    "INSERT INTO portfolios (user_id, symbol, quantity, avg_cost) VALUES (?,?,?,?)",
                    (uid, sym, round(quantity, 4), round(price, 4))
                )
            db.commit()
        print(f"[PORTFOLIO-SYNC] BUY {sym} +{quantity} @ {price:.2f} -> portfoy guncellendi")
    except Exception as e:
        print(f"[PORTFOLIO-SYNC] BUY hatasi ({sym}): {e}")


def _sync_portfolio_sell(uid: str, symbol: str, quantity: float) -> None:
    """Auto-trader SELL -> portfolios'tan dus. Sifirlanirsa satir silinir."""
    if not uid or not symbol or quantity <= 0:
        return
    sym = symbol.upper()
    try:
        with db_conn() as db:
            existing = db.execute(
                "SELECT id, quantity FROM portfolios WHERE user_id=? AND symbol=?",
                (uid, sym)
            ).fetchone()
            if not existing:
                print(f"[PORTFOLIO-SYNC] SELL {sym} -{quantity}: portfoyde bulunamadi (atlandi)")
                return
            old_qty = float(existing['quantity'])
            new_qty = round(old_qty - quantity, 4)
            if new_qty <= 0.001:
                db.execute("DELETE FROM portfolios WHERE id=?", (existing['id'],))
                print(f"[PORTFOLIO-SYNC] SELL {sym} -{quantity} -> portfoyden silindi")
            else:
                db.execute(
                    "UPDATE portfolios SET quantity=? WHERE id=?",
                    (new_qty, existing['id'])
                )
                print(f"[PORTFOLIO-SYNC] SELL {sym} -{quantity} -> kalan {new_qty}")
            db.commit()
    except Exception as e:
        print(f"[PORTFOLIO-SYNC] SELL hatasi ({sym}): {e}")


def _sync_portfolio_diff(uid: str, symbol: str, old_qty: float, old_price: float,
                          new_qty: float, new_price: float) -> None:
    """Edit position -> eski qty cikar, yeni qty ekle (avg recalculate).
    Net etki: portfoyde sembol icin (new_qty - old_qty) ekleme veya cikarma + avg_cost guncelleme.
    """
    if not uid or not symbol:
        return
    sym = symbol.upper()
    try:
        # Once eski lot'u tamamen cikar (manuel pozisyon eski avg_cost ile girmisti)
        if old_qty > 0:
            _sync_portfolio_sell(uid, sym, old_qty)
        # Sonra yeni lot'u ekle
        if new_qty > 0 and new_price > 0:
            _sync_portfolio_buy(uid, sym, new_qty, new_price)
    except Exception as e:
        print(f"[PORTFOLIO-SYNC] DIFF hatasi ({sym}): {e}")
