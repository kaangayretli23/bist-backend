/* BIST Pro service worker — app shell cache + network-first.
   ÖNEMLİ: /api/ istekleri ASLA cache'lenmez (canlı borsa verisi hep taze gelsin). */
const CACHE = 'bistpro-v1';
const SHELL = ['/', '/manifest.json', '/pwa/icon-192.png', '/pwa/icon-512.png'];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE)
      .then((c) => c.addAll(SHELL).catch(() => {}))  // biri gelmezse install patlamasın
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);

  // Canlı veri / AI çağrıları: dokunma → tarayıcı doğrudan ağa gitsin (asla stale)
  if (url.pathname.startsWith('/api/')) return;

  // Sayfa gezinmesi: network-first, çevrimdışıysa app shell'i göster
  if (req.mode === 'navigate') {
    e.respondWith(fetch(req).catch(() => caches.match('/')));
    return;
  }

  // Statik varlıklar (ikon/manifest): cache-first, yoksa ağdan çek ve cache'le
  e.respondWith(
    caches.match(req).then((hit) => hit || fetch(req).then((res) => {
      if (res && res.ok && (url.pathname.startsWith('/pwa/') || url.pathname === '/manifest.json')) {
        const copy = res.clone();
        caches.open(CACHE).then((c) => c.put(req, copy));
      }
      return res;
    }).catch(() => hit))
  );
});
