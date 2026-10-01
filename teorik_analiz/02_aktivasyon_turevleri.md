# Aktivasyon Fonksiyonlarının Türevleri ve Gradyan Davranışı

Geri yayılım sırasında bir katmandaki gradyan, zincir kuralı ile katmanlar boyunca taşınır. Aktivasyon fonksiyonunun türevi bu zincirin doğrudan bir parçasıdır.

Basitleştirilmiş biçimde:

[
\frac{\partial L}{\partial z}
=
\frac{\partial L}{\partial a}
\cdot
\frac{\partial a}{\partial z}
]

Burada:

- (L): kayıp fonksiyonu,
- (z): aktivasyon öncesi değer,
- (a=f(z)): aktivasyon çıktısıdır.

Dolayısıyla (f'(z)) çok küçük, sıfır veya büyük olduğunda geri yayılımın davranışı da değişir.

## ReLU türevi

[
f'(x)=
\begin{cases}
0, & x<0 \\
1, & x>0
\end{cases}
]

(x=0) noktasında klasik anlamda türev yoktur. Pratik derin öğrenme uygulamalarında bu noktaya uygun bir alt-türev değeri atanır.

Temel sonuç:

- pozitif bölgede gradyan korunur,
- negatif bölgede gradyan tamamen kesilir.

Bu nedenle çok sayıda nöron sürekli negatif bölgede kalırsa öğrenme yavaşlayabilir.

## Leaky ReLU türevi

[
f'(x)=
\begin{cases}
\alpha, & x<0 \\
1, & x>0
\end{cases}
]

Bu çalışmada (alpha=0.01).

Negatif bölgede gradyan tamamen sıfırlanmadığı için ReLU'ya göre daha sürekli bir güncelleme akışı sağlar.

## ELU türevi

[
f'(x)=
\begin{cases}
1, & x>0 \\
\alpha e^x, & x\leq0
\end{cases}
]

Negatif (x) değerleri büyüdükçe (e^x) sıfıra yaklaşır. Bu nedenle ELU negatif uç bölgede doygunluk gösterebilir.

ReLU'dan farklı olarak negatif bölgede türev bir anda sıfıra düşmez; ancak çok negatif değerlerde oldukça küçülebilir.

## GELU türevi

[
f(x)=x\Phi(x)
]

olduğundan çarpım kuralıyla:

[
f'(x)=\Phi(x)+x\phi(x)
]

Burada (phi(x)) standart normal dağılımın yoğunluk fonksiyonudur:

[
\phi(x)=\frac{1}{\sqrt{2\pi}}e^{-x^2/2}
]

GELU'nun türevi ReLU gibi iki sabit bölgeden oluşmaz. Geçiş bölgesi yumuşaktır ve girdinin büyüklüğüne bağlıdır.

## Swish türevi

[
f(x)=x\sigma(x)
]

için:

[
f'(x)
=
\sigma(x)
+
x\sigma(x)(1-\sigma(x))
]

eşdeğer biçimde:

[
f'(x)=\sigma(x)\left[1+x(1-\sigma(x))\right]
]

şeklinde yazılabilir.

Bu yapı, gradyanın giriş değerine göre sürekli değişmesine neden olur.

## Mish türevi

[
f(x)=x\tanh(\operatorname{softplus}(x))
]

olsun.

[
s=\operatorname{softplus}(x)=\ln(1+e^x)
]

tanımıyla:

[
f'(x)
=
\tanh(s)
+
x\sigma(x)\left(1-\tanh^2(s)\right)
]

elde edilir.

Mish'in türevi de sürekli ve yumuşaktır. ReLU'daki gibi geniş bir bölgede doğrudan sıfır türev oluşmaz; ancak türev büyüklüğü giriş değerine göre değişir.

## Karşılaştırmanın mantığı

Bu tez için önemli olan yalnızca türev formüllerinin farklı olması değildir. Bu farklılıkların eğitim sırasında hangi ölçülebilir davranışlara dönüştüğüdür.

Bu nedenle deneylerde özellikle şu noktalar izlenecektir:

- global gradyan normu,
- katman bazlı gradyan normu,
- aktivasyonların ortalama ve standart sapması,
- sıfır aktivasyon oranı,
- sıfıra yakın türev oranı,
- ReLU için dead-neuron oranı,
- parametre güncelleme büyüklüğü.

## Tez sahibinin kısa anlatımı

"Optimizer, geri yayılımla gelen gradyanı kullanıyor. Aktivasyon fonksiyonunun türevi de bu gradyanın oluştuğu zincirin içinde. Bu nedenle iki farklı aktivasyon aynı optimizerla kullanıldığında optimizerın gördüğü gradyan dağılımı aynı olmak zorunda değil. Tezin teorik tarafında önce bunu açıklayıp sonra deneylerde gerçekten oluşup oluşmadığına bakıyoruz."
