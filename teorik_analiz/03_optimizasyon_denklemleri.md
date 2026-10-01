# Birinci Dereceden Optimizasyon Algoritmalarının Güncelleme Denklemleri

Tez önerisinde SGD, Momentum SGD, RMSProp, Adam ve AdamW birlikte incelenmektedir.

Bu yöntemlerin ortak noktası, parametreleri birinci dereceden türev yani gradyan bilgisini kullanarak güncellemeleridir. Farkları ise mevcut ve geçmiş gradyanları nasıl işledikleridir.

Aşağıda:

- (	heta_t): (t). adımdaki model parametreleri,
- (g_t=\nabla_{\theta}L_t): mevcut gradyan,
- (eta): öğrenme oranı

olarak kullanılacaktır.

## 1. SGD

En temel güncelleme:

$$
\theta_{t+1}
=
\theta_t
-
\eta g_t
$$

şeklindedir.

SGD yalnızca o adımdaki gradyanı kullanır. Bu nedenle aktivasyon fonksiyonunun ürettiği gradyan büyüklüğü ve yönü güncellemeye doğrudan yansır.

## 2. Momentum SGD

Momentum, geçmiş gradyanların etkisini de taşıyan bir hız terimi kullanır:

$$
v_t
=
\mu v_{t-1}
+
g_t
$$

$$
\theta_{t+1}
=
\theta_t
-
\eta v_t
$$

Burada (mu) momentum katsayısıdır.

Bu çalışmada:

$$
\mu=0.9
$$

kullanılacaktır.

Momentumun temel etkisi, ardışık adımlarda benzer yöndeki gradyanları biriktirerek güncellemeyi hızlandırmak; sık yön değiştiren bileşenleri ise görece yumuşatmaktır.

## 3. RMSProp

RMSProp, gradyanların karelerinin hareketli ortalamasını tutar:

$$
v_t
=
\rho v_{t-1}
+
(1-\rho)g_t^2
$$

ardından parametre güncellemesi:

$$
\theta_{t+1}
=
\theta_t
-
\eta
\frac{g_t}{\sqrt{v_t}+\epsilon}
$$

şeklinde yapılır.

Bu çalışmada temel ayar:

$$
\rho=0.99
$$

olarak kullanılacaktır.

RMSProp'un önemli özelliği, her parametre için etkin adım büyüklüğünü gradyan geçmişine göre ölçeklemesidir.

## 4. Adam

Adam hem birinci moment hem ikinci moment tahmini kullanır.

Birinci moment:

$$
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
$$

İkinci moment:

$$
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
$$

Başlangıç yanlılığı düzeltmeleri:

$$
\hat m_t
=
\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t
=
\frac{v_t}{1-\beta_2^t}
$$

Parametre güncellemesi:

$$
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

Bu çalışmada:

$$
\beta_1=0.9,
\qquad
\beta_2=0.999
$$

kullanılacaktır.

## 5. AdamW

AdamW'nin temel farkı weight decay işlemini gradyan tabanlı adaptif güncellemeden ayırmasıdır.

Basitleştirilmiş güncelleme:

$$
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
-
\eta\lambda\theta_t
$$

Burada (lambda) weight decay katsayısıdır.

Bu ayrım nedeniyle Adam ve AdamW aynı moment mekanizmasına sahip olsalar da düzenlileştirme etkileri aynı biçimde uygulanmaz.

## Bu denklemler tez açısından neden önemli?

Aktivasyon fonksiyonu geri yayılım sırasında (g_t) gradyanının yapısını etkiler. Optimizer ise tam olarak bu (g_t) üzerinden güncelleme üretir.

Bu nedenle tezde incelenen etkileşim matematiksel olarak şu zincir üzerinden okunabilir:

$$
\text{aktivasyon türevi}
\rightarrow
g_t
\rightarrow
\text{optimizer güncelleme kuralı}
\rightarrow
\Delta\theta_t
\rightarrow
\text{yakınsama ve genelleme}
$$

## Tez sahibinin kısa anlatımı

"Beş optimizerın hepsi gradyanı kullanıyor ama aynı biçimde kullanmıyor. SGD mevcut gradyanı doğrudan kullanırken Momentum geçmiş yönleri de taşıyor. RMSProp ve Adam ise gradyanın büyüklüğüne göre adımı parametre bazında ölçekliyor. Dolayısıyla aktivasyonun ürettiği gradyan dağılımı farklıysa, optimizerların buna verdikleri tepki de farklılaşabiliyor."
