# Colab T4 Benchmark Sonucu ve Deney Kapsamı Kararı

Bu belge, gerçek Google Colab NVIDIA T4 benchmark sonucuna dayanır.

## Ölçülen ortam

- GPU: Tesla T4
- PyTorch: 2.11.0+cu130
- CUDA: 13.0
- cuDNN: 92700
- batch size: 128
- num_workers: 0
- seed: 42

## Ölçülen hızlar

### CNN-v1
- Ortalama eğitim süresi: 24.53 sn/epoch
- Ortalama validation süresi: 1.43 sn/epoch
- Toplam ortalama: 25.97 sn/epoch
- Tepe GPU belleği: 355 MB

### CIFAR uyarlanmış ResNet-18
- Ortalama eğitim süresi: 69.96 sn/epoch
- Ortalama validation süresi: 2.47 sn/epoch
- Toplam ortalama: 72.43 sn/epoch
- Tepe GPU belleği: 743 MB

Bu benchmark bilimsel performans sonucu değildir; hesaplama bütçesi içindir.

## Mevcut tasarımın gerçek T4 maliyeti

- LR pilotu, 60 x 30 epoch: yaklaşık 12.98 GPU-saat
- Weight decay pilotu, 45 x 30 epoch: yaklaşık 9.74 GPU-saat
- CIFAR-10 ana deney, 150 x 100 epoch: yaklaşık 108.20 GPU-saat
- CIFAR-10 ana deney, 300 x 100 epoch: yaklaşık 216.39 GPU-saat
- ResNet-18, 120 x 100 epoch: yaklaşık 241.44 GPU-saat

Yalnız LR + WD + 300 CIFAR + 120 ResNet yaklaşık 480.55 GPU-saat etmektedir.
MNIST, Animals-10, 5 Flower Types ve sağlamlık deneyleri bu hesaba dahil değildir.

## Karar: 1.035 koşu varsayılan plan değildir

1.035 koşuluk önceki plan maksimum teorik üst sınır olarak korunabilir; ancak gerçek T4 benchmarkına göre doğrudan bu kapsamla başlamak verimli değildir.

Araştırma üç seviyede yürütülecektir.

## Seviye A — Çekirdek tez

Zorunlu ve en yüksek bilimsel öncelikli bölüm:

- LR pilotu: 60 koşu
- Weight decay pilotu: 45 koşu
- CIFAR-10 ana deney: 30 kombinasyon x 5 seed = 150 koşu

Toplam: 255 koşu.

Mevcut T4 ölçümüne göre yalnız bu bölüm yaklaşık 131 GPU-saat gerektirir.

Bu seviye tamamlandığında tez önerisindeki ana faktöriyel araştırma sorusu, en az 3 tekrar şartından daha güçlü biçimde 5 seed ile karşılanmış olur.

## Seviye B — Veri setleri arası doğrulama

Tez önerisindeki:
- MNIST
- Animals-10
- 5 Flower Types

ayrı ayrı benchmark edilecektir.

Her veri setinde başlangıç hedefi:
30 kombinasyon x 3 seed = 90 koşu.

Üç veri seti toplamı: 270 koşu.

Her veri seti için epoch bütçesi ayrı pilotla belirlenip kendi içinde sabitlenecektir. CIFAR-10'daki 100 epoch sayısı körlemesine diğer veri setlerine taşınmayacaktır.

Kaynak bütçesi uygunsa bu veri setleri daha sonra 5 seed'e çıkarılabilir.

## Seviye C — Derin mimari doğrulaması

ResNet-18'in maliyeti CNN-v1'in yaklaşık 2.79 katıdır.

Bu nedenle 120 ResNet koşusu varsayılan plandan çıkarılmıştır.

İlk hedef:
- 6 önceden tanımlanmış temsilci aktivasyon-optimizer kombinasyonu
- 3 seed
- CIFAR-10

Toplam: 18 ResNet-18 koşusu.

100 epoch varsayımında yaklaşık 36.2 GPU-saat eder.

Seçim kuralı ana deney sonuçları görülmeden önce tanımlanacak veya seçim sonuç-temelli olacaksa bu durum açık biçimde raporlanacaktır.

## Genişletme sırası

Kaynak yeterliyse deney sayısı şu sırayla artırılır:

1. CIFAR-10: 5 seed -> 10 seed
2. Ek veri setleri: 3 seed -> 5 seed
3. ResNet-18 temsilci kombinasyon sayısı / seed sayısı
4. Önceden tanımlanmış sağlamlık ve duyarlılık koşulları

Bu sıra, yalnız toplam koşu sayısını büyütmek yerine istatistiksel tekrar ve dış geçerliliği önceliklendirir.

## Neden 10 seed hemen kullanılmıyor?

Tez önerisi en az 3 tekrar ister. 5 seed:
- minimum şartın üzerindedir,
- aynı seedlerin tüm kombinasyonlarda kullanıldığı eşleştirilmiş tasarımı destekler,
- standart sapma ve güven aralığı hesabını güçlendirir,
- T4 kaynak maliyetini yönetilebilir tutar.

10 seed yalnız çekirdek deneyler tamamlandıktan ve kaynak bütçesi uygun görüldükten sonra genişletme olarak kullanılacaktır.

## Colab oturum etkisi

Colab çalışma süreleri ve GPU erişimi dinamik olduğundan tek uzun çalışma yerine run bazlı checkpoint/resume sistemi kullanılacaktır.

Ana motor:
- tamamlanan runı tekrar çalıştırmamalı,
- yarım runı checkpointten sürdürmeli,
- her epoch sonrası kalıcı özet yazmalı,
- sonuçları Drive üzerinde saklamalıdır.

## Sonraki teknik adım

Aşama 1 deney motoru aşağıdaki özelliklerle tamamlanacaktır:

1. checkpoint/resume
2. CUDA doğru zamanlama
3. test seti izolasyonu
4. eğitim alt kümesinden normalizasyon istatistiği
5. macro precision/recall/F1
6. AULC ve yakınsama ölçümleri
7. gradyan probe sistemi
8. ortam/Git commit metadata kaydı
9. otomatik sonuç doğrulama
10. Colab oturumu kesintisine dayanıklılık

Bu işler tamamlandıktan sonra LR pilotuna geçilecektir.
