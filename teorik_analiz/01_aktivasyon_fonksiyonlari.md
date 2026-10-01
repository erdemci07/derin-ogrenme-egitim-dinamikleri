# Aktivasyon Fonksiyonlarının Matematiksel Yapısı

Bu bölümde tez önerisinde yer alan ReLU, Leaky ReLU, ELU, GELU, Swish ve Mish fonksiyonlarının temel matematiksel yapısı özetlenmektedir.

Amaç yalnızca fonksiyonların formüllerini vermek değildir. Daha sonraki deneylerde görülecek yakınsama, kararlılık ve genelleme farklarını açıklayabilmek için bu fonksiyonların çıktı ve türev davranışlarını önceden tanımlamaktır.

## 1. ReLU

ReLU (Rectified Linear Unit):

$$
f(x)=\max(0,x)
$$

Parçalı gösterimi:

$$
f(x)=
\begin{cases}
0, & x<0 \\
x, & x\geq 0
\end{cases}
$$

Pozitif bölgede doğrusal davranır. Negatif bölgede çıktı sıfırdır. Hesaplama açısından basittir ve derin ağlarda yaygın olarak kullanılmasının temel nedenlerinden biri budur.

Negatif bölgede türevin de sıfır olması, bazı nöronların uzun süre güncellenememesine yol açabilir. Bu durum uygulamada "dead ReLU" problemi olarak bilinir.

## 2. Leaky ReLU

Leaky ReLU, ReLU'nun negatif bölgede tamamen sıfır olmak yerine küçük bir eğim bırakılmış halidir:

$$
f(x)=
\begin{cases}
x, & x\geq 0 \\
\alpha x, & x<0
\end{cases}
$$

Bu çalışmada:

$$
\alpha=0.01
$$

olarak kullanılacaktır.

Negatif bölgede küçük de olsa gradyan akışı sürdüğü için ReLU'daki tamamen sıfır türev bölgesini azaltmayı amaçlar.

## 3. ELU

ELU (Exponential Linear Unit):

$$
f(x)=
\begin{cases}
x, & x>0 \\
\alpha(e^x-1), & x\leq 0
\end{cases}
$$

Bu çalışmada:

$$
\alpha=1
$$

olarak kullanılacaktır.

Pozitif bölgede ReLU gibi doğrusal davranırken negatif bölgede üstel ve yumuşak bir geçiş gösterir. Negatif değer üretmesi, aktivasyonların sıfır çevresinde daha dengeli dağılmasına katkı sağlayabilir.

## 4. GELU

GELU (Gaussian Error Linear Unit) girdiyi doğrudan sıfırlamak yerine, girdinin büyüklüğüne bağlı yumuşak bir ağırlıklandırma uygular:

$$
f(x)=x\Phi(x)
$$

Burada $\Phi(x)$, standart normal dağılımın kümülatif dağılım fonksiyonudur.

Eşdeğer biçimde:

$$
\Phi(x)=\frac{1}{2}\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

olduğundan:

$$
f(x)=\frac{x}{2}\left(1+\operatorname{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
$$

şeklinde yazılabilir.

Bu çalışmada PyTorch'un yaklaşık olmayan GELU tanımı kullanılacaktır.

## 5. Swish

Tez önerisinde kullanılan tanım:

$$
f(x)=x\,\sigma(x)
$$

Burada sigmoid fonksiyonu:

$$
\sigma(x)=\frac{1}{1+e^{-x}}
$$

şeklindedir.

Bu fonksiyon PyTorch'ta SiLU adıyla uygulanmaktadır. Bu çalışmada Swish ve SiLU aynı fonksiyon olarak ele alınacaktır.

Swish, ReLU gibi keskin bir geçişe sahip değildir. Negatif bölgede küçük negatif çıktılara izin verir ve türevi süreklidir.

## 6. Mish

Mish fonksiyonu:

$$
f(x)=x\tanh(\operatorname{softplus}(x))
$$

şeklinde tanımlanır.

Softplus:

$$
\operatorname{softplus}(x)=\ln(1+e^x)
$$

olduğundan:

$$
f(x)=x\tanh(\ln(1+e^x))
$$

biçiminde de yazılabilir.

Mish de Swish gibi yumuşak ve türevlenebilir bir yapıdadır. Negatif bölgede küçük negatif çıktılara izin verir.

## Bu bölüm tez açısından neden gerekli?

Tez önerisinin temel iddiası, aktivasyon fonksiyonlarının optimizerlardan bağımsız değerlendirilmemesi gerektiğidir. Bu nedenle fonksiyonların yalnız son doğruluk üzerindeki etkisine değil, gradyanın büyüklüğünü ve yönünü nasıl değiştirdiğine de bakılacaktır.

Sonraki bölümde bu altı fonksiyonun türevleri ele alınmaktadır.

## Tez sahibinin kısa anlatımı

"Burada önce kullandığımız altı aktivasyon fonksiyonunun matematiksel yapısını ortaya koyuyoruz. Çünkü optimizer ağırlıkları doğrudan aktivasyon değerlerine göre değil, geri yayılım sırasında oluşan gradyanlara göre güncelliyor. Dolayısıyla aktivasyonun türev yapısı, optimizerın eline ulaşan bilgiyi doğrudan etkiliyor."
