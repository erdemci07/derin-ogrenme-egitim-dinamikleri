# Deneylerin Nedenleri

Bu belge, "neden böyle yaptınız?" sorusuna açık cevap verebilmek için tutulmaktadır.

## Neden aynı CNN kullanılıyor?
Araştırılan ana faktörler aktivasyon fonksiyonu ve optimizasyon algoritmasıdır. Mimari de sürekli değiştirilirse hangi farkın hangi değişkenden kaynaklandığını ayırmak zorlaşır.

## Neden birden fazla seed kullanılıyor?
Sinir ağlarının başlangıç ağırlıkları, veri sırası ve bazı eğitim işlemleri rastgelelik içerir. Tek bir çalıştırmanın sonucu tesadüfen iyi veya kötü olabilir. Aynı koşulun farklı seed'lerle tekrarlanması, sonucun kararlılığını görmemizi sağlar.

## Neden test seti pilot aşamada kullanılmıyor?
Test seti nihai değerlendirme için ayrılmıştır. Öğrenme oranı veya başka ayarlar test sonucuna göre seçilirse test verisi dolaylı biçimde eğitim kararlarına karışmış olur.

## Neden bütün optimizerlara körlemesine aynı learning rate verilmiyor?
Farklı optimizerlar aynı learning rate değerinde eşdeğer davranmaz. Bu nedenle her optimizer için eşit tuning bütçesiyle uygun learning rate bölgesi pilot aşamada belirlenir.

## Neden activation başına ayrı learning rate seçilmiyor?
Her aktivasyon-optimizer çiftini ayrı ayrı optimize etmek, araştırılan etkileşime ek bir tuning farkı sokabilir. Bu nedenle seçilen optimizer learning rate'i bütün aktivasyonlarda aynı tutulur.

## Neden ana deneyde early stopping yok?
Tez önerisi aynı eğitim bütçesiyle karşılaştırmayı esas alır. Bir modeli 30, diğerini 90 epoch çalıştırmak eğitim süresi ve yakınsama karşılaştırmasını zorlaştırır.

## Neden BatchNorm ve dropout ilk CNN'de yok?
Bu bileşenler eğitim dinamiğini ve aktivasyon dağılımlarını değiştirir. Ana kontrollü deneyde üçüncü değişken sayısını azaltmak amaçlanmaktadır. Daha karmaşık mimari davranışı ayrıca ResNet-18 doğrulamasında incelenecektir.

## Neden gradient ölçüyoruz?
Sadece hangi kombinasyonun daha yüksek accuracy verdiğini değil, neden farklı davrandığını da anlamak istiyoruz. Gradient ve aktivasyon ölçümleri teorik açıklama ile deneysel sonucu birbirine bağlamaya yardımcı olur.

## Neden iki yönlü ANOVA?
Çünkü yalnızca "hangi activation daha iyi?" veya "hangi optimizer daha iyi?" sorusuna değil, optimizer etkisinin activation seçimine bağlı olup olmadığına da bakmak istiyoruz.
