# Telefondan App Gibi Erişim — Tailscale + PWA + Otomatik Başlatma

Amaç: BIST Pro'ya **telefondan, TeamViewer olmadan**, native app hissiyle eriş.
Yöntem: özel VPN (Tailscale) + tarayıcıdan "ana ekrana ekle" (PWA). **Public internet'e açılmaz**,
router port yönlendirmesi **yapılmaz**.

---

## 1) Tailscale — özel erişim (port forwarding YOK)

Tailscale, cihazlarını şifreli özel bir ağda (WireGuard) buluşturur. Telefon, PC'ye
`100.x.y.z` Tailscale IP'siyle doğrudan ulaşır; internete hiçbir port açılmaz.

**PC (Windows):**
1. https://tailscale.com/download → Windows sürümünü kur.
2. Aynı hesapla giriş yap. Sistem tepsisinde Tailscale'in **PC IP'sini** not et (örn. `100.101.102.103`).

**Telefon (iPhone/Android):**
1. App Store / Play Store → **Tailscale** kur, **aynı hesapla** giriş yap.
2. VPN iznini onayla. Artık telefon PC'yi görebilir.

**Test:** Telefon tarayıcısında `http://100.101.102.103:5000` (kendi PC IP'nle) → BIST Pro açılmalı.

> Backend zaten `0.0.0.0:5000` dinliyor (`backend.py`), yani Tailscale/lokal ağdan erişime hazır.
> Ekstra kod değişikliği gerekmez.

**Windows Firewall:** İlk erişimde 5000 portu için izin sorabilir → **İzin ver** (özel ağ).
Sormazsa: Denetim Masası > Windows Defender Firewall > Gelişmiş > Gelen Kuralları >
Yeni Kural > Port > TCP 5000 > İzin ver.

---

## 2) PWA — "Ana ekrana ekle" (app ikonu)

Artık sistem PWA. Telefonda tam ekran app gibi açılır.

**iPhone (Safari):**
1. `http://<PC-Tailscale-IP>:5000` aç.
2. Paylaş butonu (kare+ok) → **Ana Ekrana Ekle** → Ekle.
3. Ana ekranda 🟢 BIST Pro ikonu belirir; dokun → adres çubuğu olmadan tam ekran açılır.

**Android (Chrome):**
1. Adresi aç → sağ üst ⋮ menü → **Uygulamayı yükle** / **Ana ekrana ekle**.

> iOS'ta PWA'yı **Safari** ile eklemelisin (Chrome iOS add-to-home-screen desteği kısıtlı).

**İçerik:** manifest.json + service worker + ikonlar hazır. Servis worker `/api/` çağrılarını
**asla cache'lemez** (borsa verisi hep taze); sadece app kabuğunu/ikonları cache'ler.

---

## 3) PC açılınca backend otomatik başlasın (Task Scheduler)

`start_backend.bat` eklendi — `pythonw.exe` ile **pencere açmadan** arka planda çalışır,
çıktı `backend.log`'a yazılır.

**Kaydet (yönetici PowerShell'de tek komut — oturum açılınca başlar):**
```powershell
schtasks /Create /TN "BIST Pro Backend" /TR "C:\Users\Kaan\bist-backend\start_backend.bat" /SC ONLOGON /RL HIGHEST /F
```

**Yönet:**
```powershell
schtasks /Run    /TN "BIST Pro Backend"   # hemen başlat
schtasks /Query  /TN "BIST Pro Backend"   # durum
schtasks /End    /TN "BIST Pro Backend"   # durdur
schtasks /Delete /TN "BIST Pro Backend" /F  # kaldır
```

- `ONLOGON`: sen Windows'a giriş yapınca başlar (en güvenilir; ev PC'si için önerilen).
- Girişten önce (kullanıcı yokken) de çalışsın istersen `/SC ONSTART` kullan **ve** görevi
  "kullanıcı oturum açmasa da çalıştır" seçeneğiyle Task Scheduler GUI'den ayarla.
  Ancak veri kaynakları/venv genelde kullanıcı bağlamında sorunsuz olduğu için `ONLOGON` yeterli.

**Çalışıyor mu?** Tarayıcıda `http://localhost:5000` veya `backend.log` son satırlarına bak.
`pythonw` pencere açmaz; durdurmak için Görev Yöneticisi'nden `pythonw.exe` ya da `schtasks /End`.

---

## 4) Güvenlik özeti

- ✅ Erişim yalnızca **Tailscale ağındaki cihazlarından** (senin telefonun/PC'n). Public yok.
- ✅ Router'da **port açılmadı**, internete **expose edilmedi**.
- ✅ Telegram bildirimleri aynen çalışır (giden bağlantı, etkilenmez).
- ✅ OpenAI API key backend'de (`.env`), telefona/frontend'e gitmez.
- ⚠️ Tailscale hesabını koru (2FA aç). O hesabın erişimi = sistemin erişimi.
- ⚠️ `SOLO_MODE=1` (auth yok) sadece bu özel ağ için güvenli; sistemi asla public'e açma.

---

## Sık sorun / çözüm

| Sorun | Çözüm |
|---|---|
| Telefon açmıyor | İki cihaz da Tailscale'de "connected" mi? PC IP doğru mu? Firewall 5000 izinli mi? |
| "Ana ekrana ekle" yok (iPhone) | Safari kullan (Chrome değil). |
| Eski arayüz görünüyor | PWA'yı sil + tekrar ekle; ya da SW sürümü `sw.js` içinde `bistpro-v1` → `v2` yap. |
| Backend açılışta gelmedi | `schtasks /Query /TN "BIST Pro Backend"`; `backend.log`'a bak. |
