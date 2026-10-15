# Aşama 1 — Metrik, Yakınsama ve Gradyan Doğrulama Sonucu

Google Colab NVIDIA T4 üzerinde Aşama 1'in ikinci teknik doğrulaması başarıyla tamamlandı.

Bu çalışma tez performans bulgusu değildir. Ana deneylerde kullanılacak ölçüm altyapısının teknik doğrulamasıdır.

## Ortam

- GPU: Tesla T4
- Python: 3.13.15
- PyTorch: 2.11.0+cu130
- CUDA: 13.0
- cuDNN: 92700
- batch size: 128
- probe batch size: 128
- deney seed: 42
- split seed: 2026

## Sınıflandırma metrikleri

Üç epoch boyunca validation accuracy ile birlikte:

- macro precision
- macro recall
- macro F1

başarıyla hesaplandı.

3. epoch teknik doğrulama değerleri:
- validation accuracy: 0.4726
- macro precision: 0.47693878070630896
- macro recall: 0.4726
- macro F1: 0.46738920877446105

Bu değerler bilimsel sonuç olarak kullanılmayacaktır.

## Yakınsama ölçümü

Üç epochluk teknik eğri için:
- normalized AULC: 0.41885
- kendi en iyi validation accuracy değerinin %95'ine ulaşma epochu: 3
- gözlenen en iyi validation accuracy: 0.4726

Ana deneyde AULC yakınsama çıktısı olarak kullanılacaktır. Ortak performans eşiği LR pilotu sonrasında önceden tanımlanmış kuralla seçilip ana deneyden önce dondurulacaktır.

## Gradyan ve aktivasyon probe

Sabit, augmentation uygulanmayan training probe batch üzerinde epoch 1 ve epoch 3 ölçümleri başarıyla alındı.

Global gradient L2:
- epoch 1: 3.3370183314944697
- epoch 3: 3.424154318517744

Beş ReLU katmanında:
- activation mean/std
- zero fraction
- near-zero derivative fraction
- derivative absolute mean
- dead-unit fraction

başarıyla kaydedildi.

Probe işleminin orijinal model parametrelerini değiştirmediği hash kontrolüyle doğrulandı.

## Parametre güncelleme probe

Orijinal model ve optimizer değiştirilmeden kopya üzerinde tek optimizer adımı ölçüldü.

- global update L2: 0.03424154106092055
- global weight L2 before: 25.562777126309808
- global relative update: 0.00133950786691632

Katman/parametre bazında:
- update L2
- önceki weight L2
- relative update

başarıyla üretildi.

## Otomatik kontroller

Aşağıdaki kontrollerin tamamı geçti:

- epoch sayısı doğru
- macro metrikler 0-1 aralığında
- AULC 0-1 aralığında
- epoch-to-95 ölçümü geçerli
- iki probe kaydı üretildi
- probe orijinal modeli değiştirmedi
- global gradyan sonlu ve pozitif
- beş ReLU aktivasyon katmanı ölçüldü
- aktivasyon oranları geçerli
- update probe orijinal modeli değiştirmedi
- global update sonlu ve pozitif
- relative update geçerli

**Sonuç: GEÇTİ**

## Aşama 1 sonucu

Aşama 1'in iki teknik doğrulama kapısı tamamlanmıştır:

1. checkpoint/resume ve tekrarlanabilirlik: GEÇTİ
2. metrik/yakınsama/gradyan ölçümleri: GEÇTİ

Bundan sonraki adım, LR pilotu çalıştırıcısını bu doğrulanmış altyapıyla birleştirip tek koşuluk entegrasyon testi yapmaktır. Entegrasyon testi geçmeden 60 koşuluk LR pilotu başlatılmayacaktır.
