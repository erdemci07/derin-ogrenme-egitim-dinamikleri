# Deney Öncesi Teorik Beklentiler

Bu belge ana deneyler tamamlanmadan önce hazırlanmıştır.

Amaç, deney sonuçları görüldükten sonra geriye dönük açıklama üretmek yerine, teorik yapıdan çıkan beklentileri önceden kayıt altına almaktır.

Buradaki ifadeler kesin sonuç değildir. Deneylerde doğrulanması veya reddedilmesi gereken hipotez niteliğindedir.

## Beklenti 1 — ReLU'da sıfır gradyan bölgeleri daha belirgin olacaktır

ReLU negatif bölgede sıfır türev verdiği için diğer yumuşak aktivasyonlara göre:

- sıfır aktivasyon oranının,
- sıfır veya sıfıra çok yakın türev oranının,
- bazı katmanlarda düşük gradyan normlarının

daha belirgin görülmesi beklenebilir.

Bu beklenti final accuracy hakkında doğrudan hüküm vermez.

## Beklenti 2 — Leaky ReLU negatif bölgede gradyan akışını sürdürecektir

Negatif bölgede türev (0.01) olduğundan ReLU'ya göre tamamen kesilen gradyan sayısının azalması beklenir.

Bu farkın özellikle ilk katmanlardaki gradyan ölçümlerinde gözlenip gözlenmediği incelenecektir.

## Beklenti 3 — ELU negatif uçlarda doygunluk gösterebilir

ELU için negatif bölgede türev:

[
\alpha e^x
]

olduğu için çok negatif girdilerde türev küçülebilir.

Bu nedenle ELU'nun ReLU'dan farklı fakat tamamen sorunsuz olmayan bir negatif-bölge gradyan davranışı göstermesi beklenir.

## Beklenti 4 — GELU, Swish ve Mish daha yumuşak gradyan değişimleri gösterebilir

Bu fonksiyonların türevleri girişe göre sürekli değiştiği için epochlar boyunca gradyan normu ve güncelleme büyüklüğünde ReLU'ya kıyasla daha yumuşak değişimler görülmesi mümkündür.

Bu durum "kesin olarak daha kararlı" şeklinde kabul edilmeyecek; deneysel ölçümlerle sınanacaktır.

## Beklenti 5 — Adaptif optimizerlar gradyan büyüklüğü farklarını kısmen yeniden ölçekleyebilir

RMSProp, Adam ve AdamW ikinci moment bilgisini kullandıkları için farklı aktivasyonların oluşturduğu gradyan büyüklüğü farklarının parametre güncellemesine etkisini SGD'den farklı biçimde işleyebilir.

Bu nedenle aktivasyonlar arasındaki farkların optimizer değiştiğinde aynı büyüklükte kalması beklenmemelidir.

## Beklenti 6 — Aktivasyon × optimizer etkileşimi ölçülebilir olabilir

Tezin temel hipotezlerinden biri, aktivasyonun etkisinin optimizerdan tamamen bağımsız olmadığıdır.

İstatistiksel olarak bu durum iki yönlü ANOVA'daki:

[
\text{aktivasyon} \times \text{optimizer}
]

etkileşim terimiyle değerlendirilecektir.

Etkileşim anlamlı çıkarsa, bir aktivasyonun etkisinin optimizera göre değiştiğine dair deneysel destek elde edilmiş olacaktır.

## Beklenti 7 — Yakınsama ve final doğruluk aynı şeyi göstermeyebilir

Bir kombinasyon erken epochlarda hızlı ilerleyip daha sonra plato yapabilir.

Başka bir kombinasyon başlangıçta daha yavaş fakat son epochlarda daha yüksek doğruluğa ulaşabilir.

Bu nedenle tezde:

- final performans,
- öğrenme eğrisinin alanı,
- belirli eşiklere ulaşma süresi,
- eğitim kararlılığı

ayrı boyutlar olarak ele alınacaktır.

## Deney sonuçları geldikten sonra ne yapılacak?

Her beklenti için:

1. ilgili deneysel metrik bulunacak,
2. gözlenen sonuç beklentiyle karşılaştırılacak,
3. uyum varsa mekanizma açıklanacak,
4. uyum yoksa bu da açık biçimde raporlanacaktır.

Teorik bölüm, deney sonucunu zorla doğrulamak için kullanılmayacaktır.

## Tez sahibinin kısa anlatımı

"Deneyleri görmeden önce hangi mekanizmaları beklediğimizi yazıyoruz. Böylece sonuç geldikten sonra hikâye uydurmak yerine, teorik beklentiyi deneyle karşılaştırabiliyoruz. Uymayan sonuç da tez açısından değerlidir; çünkü o zaman teorik mekanizmanın tek başına yeterli olmadığını görürüz."
