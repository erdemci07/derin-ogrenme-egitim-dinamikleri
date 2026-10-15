# Aşama 1 — LR Pilotu Entegrasyon Testi Sonucu

Gerçek `calistir_lr_pilotu.py` motoru Google Colab Tesla T4 üzerinde tek koşu / üç epoch teknik entegrasyon testinde başarıyla çıktı üretmiştir.

Test koşusu:
- run: LRP-SGD-RELU-LR0p003-S42
- optimizer: SGD
- aktivasyon: ReLU
- learning rate: 0.003
- seed: 42
- epoch: 3
- cihaz: CUDA / Tesla T4

Üretilen nihai kayıt:
- validation accuracy
- macro precision
- macro recall
- macro F1
- normalized AULC
- epoch-to-95-own-best
- PyTorch/CUDA/GPU
- Git commit

alanlarını içerdi.

Bu değerler bilimsel LR pilotu sonucu değildir; yalnız entegrasyon doğrulamasıdır.

## Aşama 1 durumu

Üç teknik kapı tamamlanmıştır:

1. T4 kapasite benchmarkı: tamamlandı
2. checkpoint/resume ve tekrar üretilebilirlik: geçti
3. metrik/yakınsama/gradyan ölçümleri ve gerçek LR runner entegrasyonu: geçti

Bundan sonra gerçek 60 koşuluk LR pilotu başlatılabilir.

## Gerçek pilot

Notebook:

`colab/04_lr_pilotu_60_kosu.ipynb`

Toplam:
- 3 aktivasyon
- 5 optimizer
- optimizer başına 4 LR
- 30 epoch
- seed 42
- 60 koşu

Colab oturumları arasında Drive checkpoint ve tamamlandı kayıtları kullanılarak devam eder.

## LR seçim kuralı

Her optimizer için dört LR adayı, ReLU/GELU/Mish üzerindeki 30. epoch sonuçlarıyla değerlendirilir.

Öncelik:
1. en yüksek ortalama validation accuracy
2. eşitlikte en düşük ortalama validation loss
3. tam eşitlikte daha düşük LR

AULC LR seçim kriteri değildir; yalnız yakınsama çıktısı olarak kaydedilir.

## Ortak yakınsama eşiği

Ana deneyde kullanılacak ikincil ortak eşik, LR pilotu tamamlandıktan sonra:

`0.95 × seçilmiş LR'lerdeki 15 temsilci koşunun own-best validation accuracy değerlerinin minimumu`

olarak hesaplanır ve ana deney başlamadan önce dondurulur.

AULC ana yakınsama ölçülerinden biri olmaya devam eder; ortak eşik tamamlayıcı bir hız ölçüsüdür.
