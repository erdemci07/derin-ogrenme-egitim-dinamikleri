# Deney Planı

## Aşama 1 — Deney altyapısının kurulması

Amaç, araştırma sonucu üretmek değil; sistemin doğru çalıştığını göstermektir.

İlk duman testi:

- Veri seti: CIFAR-10
- Model: CNN-v1
- Aktivasyon: ReLU
- Optimizasyon: SGD
- Seed: 42
- Epoch: 2

Kontrol edilecekler:
- Veri yükleme
- Eğitim / doğrulama ayrımı
- Modelin ileri yayılımı
- Kayıp hesabı
- Geri yayılım
- Optimizasyon adımı
- Metrik kaydı
- Checkpoint oluşturma
- Aynı seed ile tekrar üretilebilirlik

Bu 2 epoch'luk çalışma tezde performans sonucu olarak kullanılmayacaktır.

## Aşama 2 — Öğrenme oranı pilotu

Temsilci aktivasyonlar:
- ReLU
- GELU
- Mish

Her optimizer için uygun logaritmik öğrenme oranı aralıkları eşit arama bütçesiyle denenecektir.

Pilotun amacı, ana deneyde kullanılacak optimizer-bazlı öğrenme oranlarını belirlemektir. Aktivasyonlara ayrı ayrı özel learning rate seçilmeyecektir.

## Aşama 3 — Weight decay pilotu

Öğrenme oranları belirlendikten sonra, bütün optimizerlar için aynı aday weight decay değerleri ve aynı tuning bütçesi kullanılacaktır.

Kesin değerler pilot başlamadan önce kayıt altına alınacaktır.

## Aşama 4 — Ana CIFAR-10 deneyleri

30 aktivasyon-optimizasyon kombinasyonu sabit koşullar altında çalıştırılacaktır.

Hedef:
- 30 kombinasyon
- 5 seed
- Toplam 150 ana run

Her run ayrı bir deney kimliğiyle kaydedilecektir.

## Aşama 5 — İstatistik ve eğitim dinamiği analizi

Ana sonuçlar üzerinde:
- İki yönlü ANOVA
- Etkileşim analizi
- Gerekli post-hoc analizler
- Yakınsama eğrileri
- Gradyan ve aktivasyon ölçümleri

uygulanacaktır.

## Aşama 6 — Doğrulama

Ana sonuçların tek bir veri setine veya küçük CNN'e özgü olup olmadığını incelemek amacıyla:

- MNIST
- Animals-10
- 5 Flower Types

ve seçilmiş kombinasyonlarda:

- ResNet-18

doğrulama deneyleri planlanacaktır.

Bu aşamanın tam kapsamı ana deneylerden önce danışmanla netleştirilmelidir.
