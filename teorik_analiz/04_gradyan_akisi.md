# Gradyan Akışı ve Zincir Kuralı

Derin bir sinir ağında parametre güncellemesi tek bir katmanın yerel davranışına bağlı değildir. Kayıp fonksiyonundan geriye doğru gelen türevler zincir kuralı ile katmanlar boyunca çarpılarak ilerler.

Basitleştirilmiş bir ağda:

$$
a^{(l)}=f(z^{(l)})
$$

ve

$$
z^{(l)}=W^{(l)}a^{(l-1)}+b^{(l)}
$$

olsun.

Bir önceki katmana aktarılan gradyanın temel yapısı:

$$
\frac{\partial L}{\partial z^{(l)}}
=
\frac{\partial L}{\partial a^{(l)}}
\odot
f'(z^{(l)})
$$

şeklindedir.

Buradaki (odot), eleman bazlı çarpımı ifade eder.

Ağ boyunca geri gidildiğinde aktivasyon türevleri ve ağırlık matrisleri ardışık olarak zincire girer. Bu nedenle çok küçük türev değerlerinin art arda gelmesi gradyanın küçülmesine; büyük değerlerin zincir boyunca büyümesi ise gradyanın kontrolsüz artmasına katkıda bulunabilir.

## Vanishing gradient

Geri yayılım sırasında ardışık çarpanların mutlak değerleri çoğunlukla 1'in altında olduğunda gradyan katmanlar geriye gittikçe küçülebilir.

Sonuçta ilk katmanlar çok küçük güncellemeler alabilir.

Bu tezde yalnız klasik sigmoid/tanh karşılaştırması yapılmadığı için vanishing gradient tek başına araştırma konusu değildir. Ancak ELU, GELU, Swish ve Mish gibi fonksiyonların farklı bölgelerdeki türev büyüklükleri nedeniyle gradyan akışı yine ölçülecektir.

## Sıfır gradyan bölgesi

ReLU için negatif bölgede:

$$
f'(x)=0
$$

olduğundan o nörondan geriye aktarılan yerel gradyan kesilir.

Bir nöron eğitim boyunca sürekli negatif bölgede kalırsa parametre güncellemeleri ciddi biçimde sınırlanabilir.

Bu nedenle deneylerde ReLU için sıfır aktivasyon ve dead-neuron oranı ayrıca takip edilecektir.

## Yumuşak aktivasyonlar

GELU, Swish ve Mish'in geçişleri parçalı ve keskin değildir. Türevleri girişe göre sürekli değişir.

Bu özellik teorik olarak optimizera daha sürekli bir gradyan sinyali sağlayabilir. Ancak bunun otomatik olarak daha yüksek doğruluk veya daha iyi genelleme anlamına geldiği varsayılmayacaktır.

Tezde bu ilişki doğrudan deneysel olarak sınanacaktır.

## Ölçülecek büyüklükler

Ana deneylerde belirli epoch noktalarında sabit bir probe batch üzerinde:

$$
\lVert g \rVert_2
$$

global gradyan normu,

$$
\lVert g_l \rVert_2
$$

katman bazlı gradyan normu,

ve parametre güncellemesi için:

$$
\lVert \Delta W \rVert_2
$$

ile göreli güncelleme:

$$
\frac{\lVert\Delta W\rVert_2}{\lVert W\rVert_2}
$$

hesaplanacaktır.

Bunlara ek olarak aktivasyonların:

- ortalaması,
- standart sapması,
- sıfır oranı,
- sıfıra yakın türev oranı

kaydedilecektir.

## Tez sahibinin kısa anlatımı

"Zincir kuralı nedeniyle aktivasyon türevi sadece bulunduğu katmanı değil, geriye doğru taşınan gradyanı da etkiliyor. Biz bu etkiyi teorik olarak açıklayıp deneylerde doğrudan gradyan normu, aktivasyon dağılımı ve parametre güncellemesi üzerinden ölçeceğiz."
