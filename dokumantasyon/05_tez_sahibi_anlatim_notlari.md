# Tez Sahibi Anlatım Notları

Bu belge, yapılan çalışmanın danışmana veya jüriye kendi cümleleriyle açıklanabilmesini kolaylaştırmak için hazırlanmıştır.

## Çalışmayı tek paragrafta nasıl anlatırım?
Bu tezde farklı aktivasyon fonksiyonları ile optimizasyon algoritmalarının birlikte kullanımının derin öğrenme modelinin öğrenme hızı ve genelleme başarısı üzerindeki etkisini inceliyorum. Karşılaştırmanın adil olması için ana deneylerde aynı CNN mimarisi, aynı veri bölünmesi ve aynı eğitim bütçesi kullanılıyor. Her kombinasyon birden fazla rastgele başlangıçla tekrarlanıyor. Sadece son doğruluk değerine değil, öğrenme eğrilerine, eğitim süresine ve gradyan davranışına da bakıyorum. Daha sonra aktivasyon ve optimizerın hem ayrı etkilerini hem de birbirleriyle etkileşimini istatistiksel olarak değerlendiriyorum.

## "Yeni bir algoritma mı geliştirdiniz?"
Hayır. Tezin temel amacı yeni bir optimizer veya yeni bir aktivasyon fonksiyonu icat etmek değil. Mevcut yöntemlerin birlikte kullanımında ortaya çıkan eğitim davranışını kontrollü deneylerle ve teorik analizle incelemek.

## "Neden CIFAR-10?"
Ana deney için yaygın kullanılan ve kontrollü karşılaştırmaya uygun bir görüntü sınıflandırma veri seti olması nedeniyle seçildi. Diğer veri setleri daha sonra bulguların farklı problem yapılarına taşınıp taşınamadığını kontrol etmek için kullanılacak.

## "Neden aynı modeli kullanıyorsunuz?"
Model mimarisi sürekli değişirse performans farkının activationdan mı, optimizerdan mı yoksa mimariden mi geldiğini ayırmak zorlaşır. Bu yüzden ana karşılaştırmada model sabit tutuluyor.

## "Neden 5 seed?"
Tez önerisi en az üç tekrar öngörüyor. İmkan olduğu ölçüde beş tekrar kullanarak tek bir rastgele başlangıcın sonucu belirlemesi riskini azaltmak istiyoruz.

## "Neden ANOVA?"
Çünkü yalnız ortalama accuracy değerlerini yan yana koymak yeterli değil. Gözlenen farkların rastlantısal değişkenliğin ötesinde olup olmadığını incelemek ve özellikle activation ile optimizer arasında istatistiksel bir etkileşim bulunup bulunmadığını test etmek gerekiyor.

## "Yakınsamayı nasıl ölçüyorsunuz?"
Yalnız son epoch'a bakmıyoruz. Modelin öğrenme eğrisi, belirli performans seviyelerine kaç epoch veya kaç saniyede ulaştığı ve öğrenme eğrisinin genel davranışı birlikte değerlendiriliyor.

## "Test setini neden en son kullanıyorsunuz?"
Çünkü test verisini ayar seçerken kullanmak nihai performans ölçümünü taraflı hale getirebilir. Ayarlar train ve validation ile belirleniyor; test seti nihai değerlendirmede kullanılıyor.

## Not
Bu dosya ezber metni değildir. Amaç, yapılan her tercihin mantığını gerçekten anlayıp doğal biçimde açıklayabilmektir.
