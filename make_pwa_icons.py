"""
make_pwa_icons.py — BIST Pro PWA ikonlarını üretir (Pillow).
Tema: koyu arka plan (#0a0e17) + yeşil yükselen mum motifi. Maskable safe-zone korunur.
Çalıştır: python make_pwa_icons.py  → pwa/icon-192.png, icon-512.png, apple-touch-icon.png
"""
import os
from PIL import Image, ImageDraw

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'pwa')
os.makedirs(OUT, exist_ok=True)

BG = (10, 14, 23)       # --bg-primary
GREEN = (34, 197, 94)   # yükseliş
GREEN_D = (21, 128, 61)
RED = (239, 68, 68)


def draw_icon(size, maskable=False):
    img = Image.new('RGBA', (size, size), BG + (255,))
    d = ImageDraw.Draw(img)
    # Yuvarlak köşeli arka plan paneli (maskable'da tam kare bg kalır, motif merkezde)
    if not maskable:
        r = int(size * 0.18)
        d.rounded_rectangle([0, 0, size - 1, size - 1], radius=r, fill=BG + (255,))

    # Maskable için içerik merkez %70'te kalsın (safe zone)
    pad = size * (0.20 if maskable else 0.14)
    area = size - 2 * pad
    # 4 yükselen mum (candlestick) — yeşil ağırlıklı, biri kırmızı
    n = 4
    gap = area / n
    cw = gap * 0.42                     # mum gövde genişliği
    heights = [0.42, 0.60, 0.50, 0.80]  # gövde yükseklik oranları (yükselen trend)
    wick = [0.62, 0.82, 0.72, 0.98]     # fitil oranları
    base_y = pad + area                 # alt hiza
    colors = [GREEN, GREEN, RED, GREEN]
    for i in range(n):
        cx = pad + gap * i + gap / 2
        bh = area * heights[i]
        wh = area * wick[i]
        top = base_y - bh
        col = colors[i]
        # fitil
        d.line([(cx, base_y - wh), (cx, base_y)], fill=col + (255,), width=max(2, int(size * 0.012)))
        # gövde
        d.rounded_rectangle([cx - cw / 2, top, cx + cw / 2, base_y],
                            radius=max(2, int(cw * 0.18)), fill=col + (255,))
    # yükseliş oku (ince, mumların üstünden geçen)
    ax0, ay0 = pad, base_y - area * 0.35
    ax1, ay1 = pad + area, pad + area * 0.10
    d.line([(ax0, ay0), (ax1, ay1)], fill=GREEN + (255,), width=max(3, int(size * 0.02)))
    # ok başı
    ah = size * 0.05
    d.polygon([(ax1, ay1), (ax1 - ah, ay1 - ah * 0.2), (ax1 - ah * 0.2, ay1 + ah)], fill=GREEN + (255,))
    return img


for sz in (192, 512):
    draw_icon(sz, maskable=False).save(os.path.join(OUT, f'icon-{sz}.png'))
    draw_icon(sz, maskable=True).save(os.path.join(OUT, f'icon-{sz}-maskable.png'))
# Apple touch icon (köşe yuvarlamayı iOS kendi yapar → düz kare, safe pad)
draw_icon(180, maskable=True).save(os.path.join(OUT, 'apple-touch-icon.png'))
# favicon
draw_icon(64, maskable=False).save(os.path.join(OUT, 'favicon.png'))

print('PWA ikonlari uretildi:', os.listdir(OUT))
