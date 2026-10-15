# Colab T4 Benchmark — Aşama 0

Bu aşamanın amacı tez sonucu üretmek değil, kullanılacak NVIDIA T4 ortamının gerçek kapasitesini ölçmektir.

## Neden önce benchmark yapılıyor?

Maksimum deney planı 1.035 koşuya kadar genişleyebilmektedir. Bu kapsamın uygulanabilirliği masa başında tahmin edilmemelidir. Önce gerçek T4 üzerinde:

- CNN-v1 epoch süresi
- CIFAR uyarlanmış ResNet-18 epoch süresi
- validation süresi
- tepe GPU bellek kullanımı
- PyTorch / CUDA / cuDNN ortamı

ölçülür.

## Colab ayarı

Colab içinde:

`Runtime > Change runtime type > T4 GPU`

seçilmelidir.

Benchmark scripti CUDA yoksa çalışmayı durdurur. T4 dışında bir GPU verilirse bunu ayrıca uyarı olarak bildirir.

## GitHub erişimi

Repo public olduğu için GitHub tokenı veya Colab Secret kullanılmaz. Notebook GitHub public API'sinden `main` dalının commit SHA'sını alır ve public arşivi indirir; böylece `git clone` bağımlılığı olmadan kullanılan commit benchmark metadata'sına kaydedilir.

## Google Drive veri kaynağı

Colab deneylerinde ortak veri kökü:

`/content/drive/MyDrive/tez_datasets`

olarak kullanılır.

CIFAR-10 bu klasörden okunur ve Colab oturumlarında yeniden internetten indirilmez. Yükleyici aşağıdaki iki standart yapıyı destekler:

- `tez_datasets/cifar-10-batches-py/` — çıkarılmış veri seti
- `tez_datasets/cifar-10-python.tar.gz` — standart arşiv; ilk kullanımda Drive üzerinde bir kez açılır

Bu iki yapıdan hiçbiri yoksa çalışma ağdan veri indirmek yerine açık hata ile durur.

Yerel Windows kullanımında `TEZ_VERI_KLASORU` tanımlı olmadığı için mevcut `veriler/` davranışı korunur.

## Benchmark ayarları

Varsayılan:
- CIFAR-10
- aynı 45.000 / 5.000 split sistemi
- batch size 128
- seed 42
- 1 warmup epoch
- 3 ölçüm epochu
- ReLU
- SGD, lr=0.01, weight_decay=0

Bu ayarlar bilimsel kıyaslama sonucu değildir. Yalnız süre ve bellek ölçümü içindir.

## Üretilen çıktı

`sonuclar/benchmark_colab_t4/benchmark_t4.json`

dosyasında:
- donanım ve yazılım sürümleri
- model başına epoch süreleri
- epoch süre standart sapması
- tepe GPU bellek kullanımı
- pilot ve ana CIFAR deneyleri için kaba GPU-saat tahmini

kaydedilir.

MNIST, Animals-10 ve 5 Flower Types süreleri bu ilk benchmark üzerinden kesin kabul edilmez. Bu veri setleri ilgili aşamaya gelindiğinde ayrıca ölçülecektir.

## Sonraki karar

Benchmark sonucuna göre:
1. 150 veya 300 ana CIFAR koşusunun süresi,
2. pilotların toplam maliyeti,
3. ResNet-18 doğrulama kapsamı,
4. 1.035 maksimum koşunun ne kadarının gerçekçi olduğu

sayısal olarak kararlaştırılacaktır.
