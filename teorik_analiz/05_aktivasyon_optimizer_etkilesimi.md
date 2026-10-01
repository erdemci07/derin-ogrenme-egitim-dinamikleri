# Aktivasyon ve Optimizer Etkileşiminin Matematiksel Çerçevesi

Bu tezde asıl araştırılan konu aktivasyon fonksiyonlarının veya optimizerların tek başına performansı değildir. Temel soru, bu iki bileşenin birlikte kullanıldığında nasıl bir eğitim dinamiği oluşturduğudur.

## Temel ilişki

Bir aktivasyon fonksiyonunun türevi, geri yayılımla oluşan gradyanın bir parçasıdır:

$$
g_t
=
\nabla_{\theta}L_t
$$

Bu gradyan daha sonra optimizerın güncelleme kuralına girer.

SGD için:

$$
\Delta\theta_t=-\eta g_t
$$

Adam için ise yaklaşık olarak:

$$
\Delta\theta_t
=
-\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

şeklindedir.

Aynı aktivasyonun ürettiği gradyan, SGD'de doğrudan adım büyüklüğünü belirlerken Adam veya RMSProp'ta geçmiş gradyan istatistikleri üzerinden yeniden ölçeklenir.

Bu nedenle aynı (g_t) dağılımının farklı optimizerlarda aynı parametre hareketini üretmesi beklenmez.

## ReLU ve SGD açısından

ReLU'nun negatif bölgede sıfır türev üretmesi, bazı parametreler için seyrek veya sıfır gradyanlara neden olabilir.

SGD bu gradyanı doğrudan kullandığı için bu durum parametre güncellemesine doğrudan yansır.

Bu, ReLU + SGD kombinasyonunun kötü olduğu anlamına gelmez. Buradaki teorik beklenti yalnızca, sıfır gradyan bölgelerinin SGD'nin güncelleme mekanizmasında doğrudan görülebileceğidir.

## Leaky ReLU ve Momentum açısından

Leaky ReLU negatif bölgede küçük de olsa gradyan taşır.

Momentum ise geçmiş güncelleme yönlerini biriktirir.

Bu iki özellik birlikte değerlendirildiğinde, ReLU'ya göre daha sürekli bir güncelleme akışı oluşması teorik olarak mümkündür.

Ancak bunun daha iyi doğruluk veya genelleme üretip üretmediği deneysel olarak test edilecektir.

## ELU ve adaptif optimizerlar açısından

ELU negatif uçlarda doygunluğa yaklaşır ve türevi küçülebilir.

RMSProp ve Adam, gradyanların ikinci moment bilgisi üzerinden etkin adım büyüklüğünü ölçeklediği için küçük veya değişken gradyan büyüklüklerine SGD'den farklı tepki verir.

Bu nedenle ELU'nun gradyan dağılımı ile adaptif ölçekleme mekanizmasının birlikte incelenmesi önemlidir.

## GELU, Swish ve Mish açısından

Bu üç fonksiyon ReLU'ya göre daha yumuşak türev davranışına sahiptir.

Teorik olarak daha sürekli bir gradyan dağılımı oluşturabilirler.

Adam ve RMSProp gibi optimizerlar ise bu dağılımın hem ortalama yönünü hem de büyüklük geçmişini farklı biçimlerde işler.

Bu nedenle tezde özellikle modern aktivasyonların adaptif optimizerlarla etkileşimi yalnız final accuracy üzerinden değil, yakınsama eğrileri ve gradyan ölçümleriyle de değerlendirilecektir.

## Neden tek bir "en iyi" teorik eşleşme önceden ilan edilmiyor?

Teorik yapı bize mekanizma hakkında beklenti verir; ancak genelleme performansı yalnız yerel türev özelliklerinden çıkarsanamaz.

Veri dağılımı, ağ mimarisi, öğrenme oranı, başlangıç ağırlıkları ve eğitim süreci sonucu etkiler.

Bu nedenle bu bölümün amacı sonuçları önceden ilan etmek değil, deneylerde görülecek davranışlar için test edilebilir açıklamalar oluşturmaktır.

## Tez sahibinin kısa anlatımı

"Tezde iki bileşeni ayrı ayrı puanlamıyoruz. Aktivasyon fonksiyonu gradyanın yapısını değiştiriyor, optimizer ise bu gradyanı farklı bir güncelleme kuralıyla işliyor. Dolayısıyla gerçek araştırma sorusu bu iki mekanizmanın birlikte nasıl davrandığı."
