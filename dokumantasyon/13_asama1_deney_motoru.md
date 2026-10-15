# Aşama 1 — Deney Motoru Sağlamlaştırma

T4 kapasite benchmarkı tamamlandıktan sonra büyük pilotlara geçmeden önce deney motorunun veri bütünlüğü, süre ölçümü ve kesinti dayanıklılığı güçlendirilmiştir.

## Tamamlanan ilk paket

### CIFAR-10 split doğrulaması

Sabit 45.000 eğitim / 5.000 validation ayrımı:
- boyut,
- tekrarlanan indeks,
- train/validation çakışması,
- 50.000 örneğin eksiksiz kapsanması

açısından otomatik doğrulanır.

Split dosyasına SHA-256 özeti ve sınıf dağılımları eklenir.

### Eğitim alt kümesinden normalizasyon

CIFAR-10 mean/std değerleri artık hazır sabitlerden alınmaz.

Yalnızca 45.000 örneklik eğitim alt kümesi kullanılarak hesaplanır ve JSON dosyasında dondurulur. Validation ve test görüntüleri normalizasyon istatistiğinin hesaplanmasına katılmaz.

Normalizasyon dosyası:
- split seed
- eğitim indeks özeti
- mean
- std

bilgilerini içerir. Başka bir split ile yanlışlıkla kullanılırsa çalışma durur.

### Kalıcı Colab yolları

Aşağıdaki ortam değişkenleri desteklenir:
- `TEZ_VERI_KLASORU`
- `TEZ_SPLIT_DOSYASI`
- `TEZ_STATS_DOSYASI`
- `TEZ_CIKTI_KLASORU`

Böylece Colab oturum dosya sistemi yerine Google Drive üzerinde kalıcı veri/ayar/sonuç kaydı kullanılabilir.

### CUDA süre ölçümü

CUDA işlemleri asenkron olduğu için epoch başlangıcı ve bitişinde GPU senkronizasyonu uygulanır.

Eğitim ve validation süreleri ayrı ayrı kaydedilir; toplam epoch süresi de ayrıca tutulur.

### Checkpoint / resume

Her başarılı epoch sonunda:
- model ağırlıkları,
- optimizer durumu,
- tamamlanan epoch,
- geçmiş metrik kayıtları,
- Python rastgelelik durumu,
- NumPy rastgelelik durumu,
- PyTorch CPU/CUDA RNG durumu,
- DataLoader generator durumu

checkpoint dosyasına yazılır.

Oturum kesilirse aynı run tekrar başlatıldığında sonraki epoch'tan devam edilir. Run tamamlandığında geçici checkpoint silinir ve `tamamlandi.json` nihai kayıt olarak kalır.

### Atomik kayıt

JSON ve CSV sonuçları geçici dosyaya yazılıp tek işlemle hedef dosyanın yerine geçirilir. Amaç, Colab oturumu yazma sırasında kesilirse yarım JSON/CSV bırakma riskini azaltmaktır.

## Henüz yapılmayan Aşama 1 işleri

İkinci pakette:
- macro precision / recall / F1
- AULC
- yakınsama eşikleri
- gradyan probe sistemi
- parametre güncelleme ölçümleri
- ayrıntılı ortam metadata
- sonuç bütünlük kontrolü

eklenecektir.

Büyük 60 koşuluk LR pilotu, Aşama 1 tamamlanmadan başlatılmayacaktır.


## Teknik doğrulama notebooku

`colab/01_deney_motoru_dogrulama.ipynb`

aynı 3 epochluk koşuyu iki biçimde çalıştırır:

- kesintisiz referans,
- 1. epoch sonunda checkpoint alıp yeni model/loader ile devam.

Final model parametreleri ve epoch metrikleri karşılaştırılır. Tolerans içinde eşleşme sağlanmazsa doğrulama başarısız sayılır ve LR pilotuna geçilmez.

Doğrulama sonucu Google Drive'da:

`MyDrive/tez_deney_sonuclari/asama1_deney_motoru_dogrulama/dogrulama_sonucu.json`

olarak saklanır.
