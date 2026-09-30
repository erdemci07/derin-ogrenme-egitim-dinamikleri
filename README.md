# Derin Öğrenme Eğitim Dinamikleri

Bu depo, yüksek lisans tez önerisinde tanımlanan araştırma sürecinin deneysel ve teknik altyapısını düzenli, izlenebilir ve tekrar üretilebilir biçimde yürütmek için oluşturulmuştur.

## Araştırmanın kısa özeti

Çalışmada altı aktivasyon fonksiyonu ile beş optimizasyon algoritmasının derin öğrenme modellerinin eğitim sürecine birlikte nasıl etki ettiği incelenecektir.

**Aktivasyon fonksiyonları**
- ReLU
- Leaky ReLU
- ELU
- GELU
- Swish / SiLU
- Mish

**Optimizasyon algoritmaları**
- SGD
- Momentum SGD
- RMSProp
- Adam
- AdamW

Ana deneylerde aynı CNN mimarisi, aynı veri bölünmesi ve aynı eğitim bütçesi korunacaktır. Böylece gözlenen farkların mümkün olduğunca aktivasyon fonksiyonu ve optimizasyon algoritması seçiminden kaynaklanması amaçlanmaktadır.

## Temel araştırma sorusu

Bir aktivasyon fonksiyonunun başarısı, birlikte kullanıldığı optimizasyon algoritmasına göre değişiyor mu; bu etkileşim yakınsama hızı, eğitim kararlılığı ve genelleme performansını nasıl etkiliyor?

## Çalışma sırası

1. Araştırma protokolünün kesinleştirilmesi
2. Deney altyapısının kurulması
3. Duman testi ile sistemin doğrulanması
4. Öğrenme oranı ve gerekli hiperparametreler için pilot deneyler
5. Ana CIFAR-10 deneyleri
6. İstatistiksel analizler
7. Gradyan ve eğitim dinamiği incelemeleri
8. Diğer veri setleriyle doğrulama
9. Seçilmiş deneylerin ResNet-18 ile tekrarı
10. Bulguların tez metnine aktarılması

## Nereden başlanmalı?

İlk kez inceleyen biri için önerilen sıra:

1. `dokumantasyon/01_arastirma_protokolu.md`
2. `dokumantasyon/02_deney_plani.md`
3. `dokumantasyon/03_terimler_sozlugu.md`
4. `dokumantasyon/04_deneylerin_nedenleri.md`
5. `dokumantasyon/05_tez_sahibi_anlatim_notlari.md`

## Önemli ilke

Ana deneyler başlamadan önce veri bölünmesi, model mimarisi, seed'ler, öğrenme oranları, eğitim bütçesi ve değerlendirme ölçütleri dondurulacaktır. Ana sonuçlar görüldükten sonra deney koşulları sonuca göre değiştirilmeyecektir.

## Durum

Şu anda çalışma **deney altyapısının kurulması ve pilot protokolün hazırlanması** aşamasındadır.
