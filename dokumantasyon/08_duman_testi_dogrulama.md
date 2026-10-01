# Duman Testi Doğrulama Kaydı

## Amaç

Bu kayıt, ana bilimsel deneylere geçmeden önce deney altyapısının teknik olarak çalıştığını ve aynı seed ile tekrar üretilebilir sonuç verdiğini belgelemek için tutulmaktadır.

Bu test **tez performans sonucu değildir**.

## Çalışma koşulu

- Deney kodu: `DUMAN-CIFAR10-001`
- Veri seti: CIFAR-10
- Model: CNN-v1
- Aktivasyon: ReLU
- Optimizasyon: SGD
- Learning rate: 0.01
- Weight decay: 0
- Seed: 42
- Epoch: 2
- Cihaz: CPU

## İlk çalıştırma

| Epoch | Train accuracy | Validation accuracy |
|---|---:|---:|
| 1 | 0.2427 | 0.3440 |
| 2 | 0.3542 | 0.4026 |

## İkinci çalıştırma

Aynı kod ve aynı seed ile ikinci kez çalıştırıldığında:

| Epoch | Train accuracy | Validation accuracy |
|---|---:|---:|
| 1 | 0.2427 | 0.3440 |
| 2 | 0.3542 | 0.4026 |

Sonuçlar aynı değerleri vermiştir.

## Sonuç

Duman testi başarılı kabul edilmiştir.

Doğrulanan temel zincir:

- veri yükleme
- train / validation ayrımı
- model forward işlemi
- kayıp hesabı
- backpropagation
- optimizer adımı
- validation
- metrik kaydı
- aynı seed ile tekrar üretilebilirlik

Bu sonuç ana aktivasyon × optimizer karşılaştırmasının bilimsel bulgusu değildir; yalnızca deney altyapısının teknik doğrulamasıdır.

## Sonraki aşama

Bir sonraki aşama, ana deneylerde kullanılacak optimizer-bazlı öğrenme oranlarını seçmek için **learning-rate pilotudur**.
