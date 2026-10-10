# TheftGuard

Kamera görüntülerindeki şüpheli hareketleri, bölge ihlallerini ve bekleme davranışlarını incelemek için Python/FastAPI ve Next.js tabanlı güvenlik paneli.

![Dashboard](docs/dashboard.png)

## Özellikler

- Çoklu kamera, kesinti sonrası otomatik yeniden bağlanma, bağlantı durumu, son kare zamanı ve FPS.
- YOLO nesne/poz tespiti ve kamera başına ayrı takip durumu.
- Kamera bazında güven eşiği, ardışık kare koşulu, alarm tekrar aralığı ve kural anahtarları.
- Adlandırılmış yasaklı alan/bekleme bölgeleri ve bölgeye özel süreler.
- Görüntü analizinden, canlı yayından ve kayıttan önce uygulanan gizlilik maskeleri.
- Olay inceleme durumları, notlar, arşiv koruması, tarih/kamera/tür/durum filtreleri ve gerçek sayfalama.
- Filtrelenen tüm olayların CSV çıktısı; JPEG görüntü ve alarm öncesi/sonrası H.264 MP4 klip.
- Silmeden önizlenen, arşivleri koruyan saklama temizliği.
- Gerçek veritabanı verilerinden saatlik/haftalık grafikler ve sistem kaynak ölçümleri.
- Yönetici, operatör ve gözlemci rolleri; API, WebSocket ve kanıt dosyalarında oturum kontrolü.
- Yüz listeleri; isteğe bağlı e-posta/Telegram bildirimleri ve görüntü ekini kapatma seçeneği.

[Kaynaklı özellik karşılaştırması ve sınırlar](docs/FEATURE_COMPARISON.md)

Algılama sonuçları insan incelemesine yönelik şüphe kayıtlarıdır. Model çıktısı tek başına doğrulanmış bir hırsızlık kararı değildir.

## Gereksinimler

- Python 3.11 ve Node.js 20.9 veya daha yeni bir sürüm; Node.js LTS önerilir.
- `dlib` kurulumu için C++ derleyicisi ve CMake.
- Video klipleri için `libx264` kodlayıcılı FFmpeg. Kullanılabilirlik Ayarlar ekranında gösterilir.
- CPU çalışması desteklenir; CUDA GPU isteğe bağlıdır.

Dahil edilen modeller: `yolov8n.pt` ve `yolov8n-pose.pt`. İsteğe bağlı `shoplifting.pt` dosyası mevcutsa kullanılır. Yalnızca güvendiğiniz model dosyalarını yükleyin.

## Kurulum

```bash
python3.11 -m venv venv
source venv/bin/activate
python -m pip install 'cmake<4' 'setuptools<81' wheel
python -m pip install -r requirements.txt
cd dashboard
npm ci
cd ..
```

CPU kurulumu için Python bağımlılıklarını yüklemeden önce PyTorch CPU paketlerini kurabilirsiniz:

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Windows'ta sanal ortamı `venv\Scripts\activate` ile etkinleştirin; C++ Build Tools gereklidir. Mevcut `start_system.bat` başlatıcısı kullanılabilir; önce aşağıdaki yönetici hesabı adımını tamamlayın.

## İlk yönetici hesabı

Varsayılan kullanıcı veya parola bulunmaz. Arka uçla aynı çalışma dizininde ve aynı ortam değişkenleriyle çalıştırın:

```bash
python manage_users.py create-admin --username vahapogut
```

Parola terminalde gizli olarak iki kez istenir; en az 12 karakter gerekir. Diğer hesaplar Kullanıcılar ekranından oluşturulur.

```bash
python manage_users.py reset-password --username vahapogut
```

Parola, rol veya etkinlik durumu değiştiğinde mevcut oturumlar iptal edilir. Son etkin yönetici devre dışı bırakılamaz veya daha düşük bir role geçirilemez.

| Rol | İzinler |
| --- | --- |
| Yönetici | Tüm ekranlar; kamera, algılama, yüz, kullanıcı ve sistem ayarları; kanıt temizliği |
| Operatör | Canlı yayın ve olayları görme, dışa aktarma, inceleme/not/arşiv işlemleri |
| Gözlemci | Canlı yayın, olaylar, kanıtlar ve istatistikleri görme; CSV dışa aktarma |

## Çalıştırma

İki terminal kullanın:

```bash
# Depo kökünde, Python sanal ortamı etkinleştirilmişken
python backend.py
```

```bash
cd dashboard
npm run dev
```

Panel 3000, arka uç 8000 portunu kullanır. Tarayıcıdaki API ve WebSocket istekleri aynı kökendeki `/api` yolu üzerinden iletilir. Arka uç varsayılan olarak yalnızca `127.0.0.1` üzerinde dinler. Fiziksel kamera yoksa panelde sinyal yok görüntüsü beklenir; bağlı olmayan kamera otomatik olarak yeniden denenir.

Aynı veri dizini için tek arka uç süreci çalıştırın. Yeniden başlatıldığında yarıda kalan klipler başarısız olarak işaretlenir; mevcut kanıt dosyaları korunur. Kamera yapılandırması bozuksa dosyanın üzerine yazılmaz; `cameras.json` düzeltildikten sonra uygulamayı yeniden başlatın.

### Verileri ayrı bir dizinde saklama

`THEFTGUARD_DATA_DIR` ayarları, kamera tanımlarını, olay veritabanını ve kanıt dosyalarını taşır. Yönetici oluşturma komutunda da aynı değişkeni kullanın. Model dosyaları varsayılan olarak depo kökünden okunur.

```bash
export THEFTGUARD_DATA_DIR="$PWD/data/runtime"
python manage_users.py create-admin --username vahapogut
python backend.py
```

Ayrı veri dizini belirtilmezse mevcut çalışma dizini kullanılır. Eski dört sütunlu `alerts` tablosu kayıtlar korunarak otomatik genişletilir. Eski kayıtların kamera bilgisi bulunmadığından bu alanlar boş kalabilir.

### Ortam değişkenleri

| Değişken | Varsayılan / amaç |
| --- | --- |
| `THEFTGUARD_DATA_DIR` | Çalışma dizini; ayarlar, olay veritabanı ve kanıtlar |
| `THEFTGUARD_MODEL_DIR` | Depo kökü; YOLO model dosyaları |
| `AUTH_DB_PATH` | Veri dizini altında `data/auth.sqlite3`; isteğe bağlı kullanıcı veritabanı yolu |
| `BIND_HOST`, `PORT` | `127.0.0.1`, `8000` |
| `BACKEND_URL` | Next.js sunucusunun arka uç adresi; `http://127.0.0.1:8000` |
| `NEXT_PUBLIC_API_URL` | Boş bırakıldığında aynı kökendeki `/api`; ayrı dağıtım için isteğe bağlı |
| `ALLOWED_ORIGINS` | Virgülle ayrılmış izinli tarayıcı kökenleri; varsayılan `http://localhost:3000,http://127.0.0.1:3000` |
| `SESSION_COOKIE_SECURE` | Yerelde `false`; HTTPS dağıtımında `true` |
| `SMTP_PASSWORD`, `TELEGRAM_BOT_TOKEN` | Bildirimler etkinleştirildiğinde gerekli gizli değerler |

Uzak erişimde HTTPS kullanın, `ALLOWED_ORIGINS` değerini gerçek panel kökeniyle ayarlayın ve güvenli oturum çerezlerini etkinleştirin. Kamera kaynak adresleri API yanıtlarında gizlenir. Bildirim parolaları JSON ayar dosyasında veya API yanıtlarında açık olarak tutulmaz.

## Olay ve saklama akışı

Geçmiş ekranında olayları süzün; bir olayı incelendi veya yanlış alarm olarak işaretleyin, not ekleyin ve korunması gereken kanıtı arşivleyin. CSV dışa aktarımı yalnızca görünen sayfayı değil tüm filtrelenmiş sonuçları içerir.

Ayarlar ekranında klibin alarm öncesi/sonrası süresini seçin. Kamera sayısı, klip kuyruğu ve bellek tamponu sınırlıdır; kaynak doluluğu veya kodlayıcı eksikliği sessiz bir başarılı kayıt olarak gösterilmez.

Saklama temizliği otomatik çalışmaz. Önce yaş sınırıyla önizleme alın, ardından silmeyi onaylayın. Önizleme 10 dakika geçerlidir ve bir kez kullanılabilir. Aday kayıtlar veya dosyalar değişmişse yeni önizleme gerekir. Arşivlenmiş olaylar, paylaşılan kanıt dosyaları ve hazırlanmakta olan klipler korunur.

Gizlilik maskesi değiştirildiğinde o kameranın eski klip tamponu ve tamamlanmamış klipleri temizlenir. Önceden tamamlanmış kanıt dosyaları geriye dönük değiştirilmez.

## Doğrulama

```bash
# Depo kökünde
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python test_setup.py
python test_pose.py

cd dashboard
npm run lint
npx tsc --noEmit
npm run build
```

Eski `test_setup.py` ve `test_pose.py` hata yakaladığında sıfır çıkış kodu verebilir; çıktıda `SETUP_SUCCESS` ve `Model loaded successfully` işaretlerini de kontrol edin. Yeni testler geçici veritabanları, sentetik kameralar ve sahte bildirim istemcileri kullanır; dışarıya bildirim göndermez.

## Katkıda bulunanlar

- [vahapogut](https://github.com/vahapogut)

## Lisans

[MIT](LICENSE)
