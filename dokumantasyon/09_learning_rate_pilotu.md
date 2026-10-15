# Learning-Rate Pilotu

## Neden yapılıyor?

Ana 6 × 5 aktivasyon-optimizasyon deneyine geçmeden önce her optimizer ailesi için makul bir learning-rate seçmek gerekir.

Amaç, bir aktivasyona özel avantaj sağlayacak şekilde ayrı ayrı ayar yapmak değildir. Her optimizer için bir LR seçilecek ve ana deneyde o optimizerın bütün aktivasyon koşullarında aynı değer kullanılacaktır.

## Pilot tasarımı

Temsilci aktivasyonlar:

- ReLU
- GELU
- Mish

Optimizerlar:

- SGD
- Momentum SGD
- RMSProp
- Adam
- AdamW

Her optimizer için dört LR adayı vardır.

Toplam:

`3 aktivasyon × 5 optimizer × 4 LR = 60 koşu`

Her koşu:

- CIFAR-10
- CNN-v1
- seed 42
- 30 epoch
- weight decay 0
- aynı train/validation split
- test setine dokunulmadan

çalıştırılacaktır.

## LR adayları

SGD / Momentum SGD:

- 0.003
- 0.01
- 0.03
- 0.1

RMSProp / Adam / AdamW:

- 0.0001
- 0.0003
- 0.001
- 0.003

Optimizer ailelerinin tipik LR ölçekleri farklı olduğu için aynı sayısal grid zorlanmamıştır; ancak her optimizer için arama bütçesi eşittir.

## Seçim kuralı

Birincil ölçüt:

**30. epoch validation accuracy değerinin ReLU, GELU ve Mish üzerindeki ortalaması**

İkincil ölçüt:

**30. epoch validation loss ortalaması**

NaN, Inf veya belirgin sayısal kararsızlık gösteren koşular elenir.

AULC bu aşamada LR seçmek için kullanılmaz. Çünkü AULC daha sonra yakınsama davranışını ölçmek için araştırma sonucu olarak kullanılacaktır; LR'yi AULC ile seçmek yakınsama karşılaştırmasını tuning aşamasına taşır.

## Çalıştırmadan önce

Önce yalnız planı görmek için:

```powershell
.\.venv\Scripts\python.exe calistir_lr_pilotu.py
```

Bu komut eğitim başlatmaz.

Tek bir koşuyla teknik kontrol yapmak için:

```powershell
.\.venv\Scripts\python.exe calistir_lr_pilotu.py --calistir --maksimum-run 1
```

Tüm 60 koşuyu başlatmak için:

```powershell
.\.venv\Scripts\python.exe calistir_lr_pilotu.py --calistir
```

CPU üzerinde tam pilotun çalıştırılması önerilmez. Ana pilot GPU ortamında çalıştırılmalıdır.


## Aşama 1 sonrası ölçüm güncellemesi

LR pilotunda validation accuracy/loss yanında macro precision, macro recall ve macro F1 da kaydedilir.

AULC ve kendi en iyi validation değerinin %95'ine ulaşma epochu teknik olarak hesaplanır ve arşivlenir. Ancak **AULC kaydedilir ancak learning-rate seçimi için kullanılmaz**. LR seçimi önceden tanımlanan kurala göre 30. epoch validation accuracy üzerinden yapılır; validation loss yalnız ikincil ölçüttür.

Gradyan/aktivasyon probe ölçümleri ana yakınsama analizi için doğrulanmış olmakla birlikte LR seçim kriterine dahil edilmez.
