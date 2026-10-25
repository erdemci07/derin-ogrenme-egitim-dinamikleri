# ResNet-18 seçili doğrulama protokolü

## Amaç ve kapsam

Tez önerisindeki ana deneyler, dört veri setinde basit CNN mimarisiyle yürütülmüştür. Öneride belirtildiği üzere, bulguların ağ derinliği arttığında korunup korunmadığını incelemek için **seçilmiş deneyler** ResNet-18 üzerinde tekrar edilir.

Bu aşama yeni bir hiperparametre taraması veya ikinci bir 6 aktivasyon × 5 optimizer ana deney seti değildir. Seçim, tamamlanmış CNN ana deneylerinin özetleri kullanılarak ResNet çalıştırılmadan önce sabitlenmiştir.

## Veri, tekrar ve sabit tutulan koşullar

- Veri seti: CIFAR-10; tez önerisindeki temel veri seti.
- Mimari: CIFAR-10 girdisi için uyarlanmış ResNet-18.
- Tekrar: Her kombinasyon için seed 101, 202 ve 303; toplam 18 koşu.
- Epoch, batch büyüklüğü, veri bölünmesi, augmentation, optimizer ayarları, loss ve test protokolü: ayarlar/frozen_main_config.yaml ile ana CNN deneyindeki gibi sabit.
- Scheduler, early stopping ve gradyan kırpma kullanılmaz.
- ResNet-18'in BatchNorm ve kısa-yol bağlantıları mimarinin sabit unsurlarıdır. Aktivasyon fonksiyonu başlangıç katmanı ve tüm artık bloklarda aynı kuralla uygulanır.
- Başlatma: Aktivasyonlar arası taraf tutmamak için tüm konvolüsyonel/doğrusal katmanlarda Xavier uniform (gain=1); BatchNorm ölçeği 1, bias 0.

## Seçim mantığı

Ana CNN sonuçlarının dört veri seti üzerindeki eşit ağırlıklı özetiyle aşağıdaki altı tekil kombinasyon belirlenmiştir.

| Kombinasyon | ResNet-18'e taşınma gerekçesi |
|---|---|
| Adam + GELU | Geçerli koşular arasında en yüksek genel performans; aynı zamanda etkileşim karşılaştırmasının kontrol kolu |
| AdamW + GELU | En hızlı yakınsama (eşitlikte daha yüksek AULC ile seçildi) |
| SGD + ReLU | Geçerli koşular arasında en düşük genel performans ve en yavaş yakınsama |
| Momentum SGD + ELU | En güçlü pozitif aktivasyon × optimizer etkileşimi |
| Adam + ELU | Etkileşim 2×2 karşıt kolu |
| Momentum SGD + GELU | Etkileşim 2×2 kontrol kolu |

Görünürdeki beş seçim ölçütü altı kombinasyona dönüşür; çünkü etkileşim tek bir hücreyle değil, en az iki optimizer ve iki aktivasyondan oluşan 2×2 karşılaştırmayla yorumlanabilir. Bu nedenle eklenen iki kol, tezin ana sorusu olan aktivasyon × optimizer etkileşimini daha derin mimaride test edilebilir kılar.

MNIST–SGD–ELU–S101 koşulu NaN/Inf nedeniyle kararsız kaydedildiği için, tam üç seedli geçerli ortalama yoktur. Bu kombinasyon “en düşük” veya “en yavaş” temsilcisi olarak seçilmez; ek kararlılık analizinde ayrı raporlanır. MNIST–AdamW–ReLU–S202 ise yeniden yürütülmüş, 50 epoch CSV ve sonuç JSON'u birbiriyle doğrulanmıştır.

## Beklenen raporlama

Her kombinasyon için üç seed üzerinden ortalama ± standart sapma ile test accuracy, macro precision/recall/F1, test loss, AULC, kendi en iyi validation accuracy değerinin %95'ine ulaşma epoch'u ve eğitim süresi raporlanır. ResNet-18 sonuçları, CNN bulgularını “doğrular” ya da “çürütür” diye peşinen ifade etmez; bulgunun mimari derinlik altında korunma, zayıflama veya değişme yönünde kanıt üretir.

Bu seçili alt küme için betimsel karşılaştırmalar ve önceden belirlenmiş 2×2 etkileşim kontrastı sunulur. Tam faktöriyel iki yönlü ANOVA, ana CNN deneyinin 6 × 5 tasarımında raporlanır; 6 hücrelik ResNet-18 alt kümesinde 6 × 5 ANOVA yapılmaz.
