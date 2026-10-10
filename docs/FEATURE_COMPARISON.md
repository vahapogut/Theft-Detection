# Benzer projeler ve özellik karşılaştırması

Bu karşılaştırma, 10 Ekim 2026 tarihinde aşağıdaki projelerin resmi GitHub kaynakları okunarak hazırlandı. Özellikler bağımsız olarak geliştirildi; bu projelerden kaynak kod alınmadı. Mevcut JPEG görüntü kaydı, tek ROI ve CSV indirme zaten vardı; tabloda bunların üzerine eklenen işlevler belirtiliyor.

| Alan | İncelenen kaynak | Önceki durum | Eklenen özellik |
| --- | --- | --- | --- |
| Olay inceleme | [Frigate review](https://github.com/blakeblackshear/frigate/blob/dev/docs/docs/configuration/review.md), [ZoneMinder etiketleri](https://github.com/ZoneMinder/zoneminder/blob/master/docs/userguide/tags.rst) | İnceleme durumu veya not yoktu | Yeni, incelendi, yanlış alarm durumları; inceleyen kullanıcı, not ve arşiv koruması |
| Arama ve dışa aktarma | [ZoneMinder filtreleri](https://github.com/ZoneMinder/zoneminder/blob/master/docs/userguide/filterevents.rst) | Son 100 kayıt içinde tarayıcı filtresi; tarih ve sayfalama düğmeleri işlevsizdi | Sunucuda metin, tarih, kamera, tür, durum ve arşiv filtresi; gerçek sayfalama; filtrelenen tüm kayıtların güvenli CSV çıktısı |
| Kamera kurtarma | [Viseron kamera yönetimi](https://github.com/roflcoopter/viseron/blob/dev/viseron/components/ffmpeg/camera.py), [ZoneMinder ağ ayarları](https://github.com/ZoneMinder/zoneminder/blob/master/docs/userguide/options/options_network.rst) | İlk bağlantı başarısızlığından sonra yeniden deneme yoktu | Sınırlandırılmış artan bekleme ile yeniden bağlanma, elle yeniden bağlama, son kare zamanı, FPS ve bağlantı durumu |
| Bölgeler ve yanlış alarm kontrolü | [Frigate bölgeleri](https://github.com/blakeblackshear/frigate/blob/dev/docs/docs/configuration/zones.md) | Tek ROI, sabit bekleme süresi; ROI kaydetme kilitleniyordu | Birden fazla adlandırılmış bölge; bölgeye özel bekleme; güven, ardışık kare ve tekrar alarm aralığı; kamera bazlı kural anahtarları; kilitlenme düzeltmesi |
| Gizlilik | [motionEye arayüzü](https://github.com/motioneye-project/motioneye/blob/dev/motioneye/templates/main.html), [Frigate maskeleri](https://github.com/blakeblackshear/frigate/blob/dev/docs/docs/configuration/masks.md) | Yayın ve bildirim görüntüleri için maske/ek tercihi yoktu | Analizden, yayından ve kayıttan önce siyah gizlilik maskeleri; e-posta ve Telegram bildirimlerinde görüntü ekini kapatma |
| Video kanıtı | [Frigate kayıt ayarları](https://github.com/blakeblackshear/frigate/blob/dev/docs/docs/configuration/record.md), [ZoneMinder olay videosu](https://github.com/ZoneMinder/zoneminder/blob/master/docs/userguide/viewevents.rst) | Yalnızca JPEG görüntü | Sınırlandırılmış bellek tamponuyla alarm öncesi/sonrası H.264 MP4 klip; arka planda kodlama ve kayıt yeteneği kontrolü |
| Saklama yönetimi | [Viseron depolama ayarları](https://github.com/roflcoopter/viseron/blob/dev/viseron/components/storage/config.py) | Dosyalar ve kayıtlar süresiz birikiyordu | Yaşa göre temizleme önizlemesi; kullanıcıya bağlı, süreli, tek kullanımlık onay; arşivlenmiş ve hazırlanmakta olan kanıtların korunması |
| Kullanıcı yetkileri | [ZoneMinder kullanıcıları](https://github.com/ZoneMinder/zoneminder/blob/master/docs/userguide/options/options_users.rst) | API, canlı yayın ve kanıtlar giriş gerektirmiyordu | Yönetici, operatör, gözlemci rolleri; parola özeti; süreli oturumlar; API, WebSocket ve kanıtlarda yetki denetimi |
| Gerçek istatistikler | Olay inceleme akışının tamamlayıcısı | Saatlik grafik rastgeleydi; yanlış alarm sayısı toplamın yüzde 15'i olarak üretiliyordu | Veritabanından gerçek saatlik/haftalık sayılar, inceleme durumları ve gerçek CPU/RAM ölçümleri |

## Tasarım farkları ve sınırlar

- Gizlilik maskeleri ilgili pikselleri tamamen karartır. Frigate'in nesne filtre maskeleriyle aynı işlev olduğu iddia edilmez.
- Saklama temizliği otomatik çalışmaz; yönetici önizleme yapıp silmeyi açıkça onaylar. Sürekli kayıt, depolama katmanları ve boyut kotası bu sürümün kapsamında değildir.
- Roller uygulama genelindedir; kullanıcıya özel kamera listesi henüz yoktur.
- Klip kaydı FFmpeg'in H.264 kodlayıcısını gerektirir. Kullanılamazsa durum arayüzde görünür; JPEG kanıtı çalışmaya devam eder.
- Maskelerde yapılan değişiklikler yeni görüntülere uygulanır; daha önce kaydedilmiş kanıtlar geriye dönük değiştirilmez.
- Algılama sonuçları inceleme gerektiren şüphe kayıtlarıdır; doğrulanmış suç kararı değildir. Kurallar ve maskeler sahaya göre ayarlanmalıdır.
- Yerel video dosyasının sonuna gelindiğinde kamera çevrim dışına geçer; dosyayı tekrar oynatmak için yeniden bağlama kullanılır.
- Kamera sürücüsü kendi okuma zaman aşımını uygulamıyorsa sistem eski kareyi göstermeyi keser; yerel sürücünün bloke olan çağrısı tamamlanana kadar aynı iş parçacığı zorla kapatılmaz.

## Kaynak ve lisans notları

Frigate ve Viseron MIT; ZoneMinder GPL v2; motionEye GPL v3 lisanslıdır. Bağlantılar özellik araştırmasının kaynaklarıdır ve bu projenin katkıda bulunanlar listesi değildir. Shinobi'nin resmi GitHub README dosyası GitLab'a taşındığını bildirdiği için güncel özellik karşılaştırmasına dahil edilmedi.
