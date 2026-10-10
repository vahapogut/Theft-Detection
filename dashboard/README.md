# TheftGuard Dashboard

Next.js 16 ve React 19 paneli. Kurulum, kullanıcı oluşturma ve sunucu yapılandırması için [ana README](../README.md) dosyasını okuyun.

```bash
npm ci
npm run dev
```

Tarayıcı istekleri varsayılan olarak `/api` üzerinden FastAPI sunucusuna iletilir. `BACKEND_URL` sunucu hedefini; isteğe bağlı `NEXT_PUBLIC_API_URL` ayrı API dağıtımını belirler. Oturum çerezi API, WebSocket ve kanıt erişiminde kullanılır. İlk yönetici hesabı sunucu terminalinden oluşturulur.

Ekranlar: canlı görüntü ve istatistikler, kamera sağlığı/bölgeler/maskeler, yüz listeleri, olay inceleme, sistem ayarları ve yöneticiye özel kullanıcı yönetimi. Yazı tipleri yereldir; derleme sırasında harici font indirilmez.

```bash
npm run lint
npx tsc --noEmit
npm run build
npm run start
```

## Katkıda bulunanlar

- [vahapogut](https://github.com/vahapogut)
