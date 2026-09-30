# Araştırma Protokolü 0.1

## Amaç

Bu çalışma, farklı aktivasyon fonksiyonları ile birinci dereceden optimizasyon algoritmaları arasındaki etkileşimin derin öğrenme modellerinin yakınsama davranışı ve genelleme performansı üzerindeki etkisini teorik ve deneysel olarak incelemeyi amaçlamaktadır.

## Ana faktörler

### Aktivasyon fonksiyonları
1. ReLU
2. Leaky ReLU
3. ELU
4. GELU
5. Swish / SiLU
6. Mish

### Optimizasyon algoritmaları
1. SGD
2. Momentum SGD
3. RMSProp
4. Adam
5. AdamW

Ana faktöriyel yapı 6 × 5 = 30 kombinasyondan oluşacaktır.

## Ana veri seti

Ana kontrollü deneylerde CIFAR-10 kullanılacaktır.

CIFAR-10'un resmi eğitim verisi eğitim ve doğrulama bölümlerine ayrılacak; resmi test verisi hiperparametre seçimi sırasında kullanılmayacaktır.

Geçici plan:
- 45.000 eğitim
- 5.000 doğrulama
- 10.000 test

Bu ayrım ana deneylerden önce sabit dosyalar halinde kaydedilecektir.

## Ana model

Ana deneylerde sabit bir CNN-v1 mimarisi kullanılacaktır. Aktivasyon fonksiyonu ve optimizasyon algoritması dışında kalan mimari unsurlar bütün kombinasyonlarda aynı tutulacaktır.

Batch normalization ve dropout ilk ana deney tasarımında kullanılmayacaktır. Amaç, araştırılan iki faktörün etkisini mümkün olduğunca doğrudan gözlemlemektir.

## Tekrarlar

Tez önerisindeki en az üç tekrar koşulunu güçlendirmek amacıyla imkanlar elverirse beş sabit seed kullanılacaktır.

Önerilen seed'ler:
- 42
- 123
- 2026
- 3407
- 7319

## Eğitim bütçesi

Ana deney için 100 epoch geçici başlangıç değeridir. Kesin epoch sayısı pilot deneylerden sonra, ana deney başlamadan önce dondurulacaktır.

Early stopping ana deneyin temel karşılaştırmasında kullanılmayacaktır.

## Birincil değerlendirme ölçütleri

- CIFAR-10 test accuracy
- Validation öğrenme eğrisi alanı (AULC)

## İkincil ölçütler

- Macro precision
- Macro recall
- Macro F1
- Training loss
- Validation loss
- Test loss
- Eğitim süresi
- Ortak başarı eşiğine ulaşma süresi
- Kendi en iyi doğrulama başarısının %95'ine ulaşma epoch'u
- Train-test genelleme farkı

## Eğitim dinamiği ölçümleri

Seçilmiş epoch noktalarında:
- Global gradient norm
- Katman bazlı gradient norm
- Aktivasyon ortalaması ve standart sapması
- Sıfır / sıfıra yakın aktivasyon oranları
- Sıfıra yakın türev oranı
- ReLU için ölü nöron oranı
- Parametre güncelleme büyüklüğü
- Güncelleme / ağırlık normu oranı

## İstatistiksel plan

Ana analiz iki yönlü ANOVA olacaktır:

- Aktivasyon ana etkisi
- Optimizasyon algoritması ana etkisi
- Aktivasyon × optimizasyon etkileşim etkisi

Etkileşim anlamlı bulunduğunda uygun basit etki ve çoklu karşılaştırma analizleri uygulanacaktır. Seed kaynaklı değişkenliğe karşı ek sağlamlık analizi ayrıca değerlendirilecektir.

## Temel araştırma ilkesi

Sonuçlara bakılarak deney koşulları değiştirilmeyecektir. Pilot aşamada belirlenen ve ana deney öncesinde dondurulan protokol, bütün ana koşullarda aynı biçimde uygulanacaktır.
