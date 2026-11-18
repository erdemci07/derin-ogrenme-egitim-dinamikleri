# Derin Öğrenme Eğitim Dinamikleri

Bu depo, Van Yüzüncü Yıl Üniversitesi Fen Bilimleri Enstitüsü, Yapay Zeka ve Robotik Anabilim Dalı yüksek lisans tezinin deney altyapısıdır:

> **Derin Öğrenme Modellerinde Aktivasyon Fonksiyonları ve Optimizasyon Algoritmalarının Yakınsama Dinamikleri ve Genelleme Performansı Üzerindeki Etkilerinin Teorik ve Deneysel Analizi**

## Durum

**Deneyler tamamlanmıştır (Ekim 2026). Depo arşiv amaçlıdır; tezdeki sonuçları üreten kod sürümleri aşağıdaki commit'lerde sabittir.**

| Aşama | Koşu | Durum |
|---|---|---|
| T4 benchmark ve teknik doğrulamalar | — | Tamamlandı |
| LR pilotu (6 aktivasyon × 5 optimizer × 4 LR, CIFAR-10, 30 epoch) | 120 | Tamamlandı (4 kararsız koşu elendi) |
| AdamW weight decay = 0,01 LR doğrulaması | 24 | Tamamlandı |
| Ana CNN deneyi (4 veri seti × 30 kombinasyon × 3 seed, 50 epoch) | 360 | 359 tamamlandı (MNIST–SGD–ELU–S101 sayısal kararsızlık) |
| ResNet-18 seçili doğrulama (CIFAR-10, 6 kombinasyon × 3 seed) | 18 | Tamamlandı |
| Ek kararlılık analizi (MNIST–SGD–ELU–S101, gradyan kırpma) | 1 | Tamamlandı, ANOVA'ya dahil değil |
| İki yönlü ANOVA, Tukey, basit etkiler | — | Tamamlandı |

## Araştırmanın özeti

Altı aktivasyon fonksiyonu (ReLU, Leaky ReLU, ELU, GELU, Swish/SiLU, Mish) ile beş optimizasyon algoritması (SGD, Momentum SGD, RMSProp, Adam, AdamW) 6 × 5 tam çapraz tasarımda eşleştirilmiştir. Bütün kombinasyonlarda aynı CNN, aynı veri bölünmesi ve aynı eğitim bütçesi kullanılmıştır.

**Temel soru:** Bir aktivasyon fonksiyonunun başarısı birlikte kullanıldığı optimizasyon algoritmasına göre değişiyor mu (aktivasyon × optimizer etkileşimi)?

**Ana bulgular (ayrıntı tez metnindedir):**
- Etkileşim **AULC** (öğrenme eğrisi altındaki alan) ölçütünde dört veri setinin tamamında anlamlı ve büyüktür (kısmi η² = 0,80–0,94). Final test doğruluğunda yalnızca Animals-10'da anlamlıdır.
- ELU uyarlanabilir olmayan yöntemlerle (SGD, Momentum), GELU uyarlanabilir yöntemlerle (RMSProp, Adam, AdamW) göreli olarak avantajlıdır.
- Bütün veri setlerinde üstün tek bir kombinasyon yoktur.
- ResNet-18 seçili kombinasyonlarda doğruluğu 4,87–7,29 yüzde puan artırmış, CNN'deki etkileşim yapısını açıkça doğrulamamıştır.

## Dondurulmuş protokol

`ayarlar/frozen_main_config.yaml` (18 Ekim 2026, LR pilotlarından sonra, ana deneylerden önce)

| Bileşen | Değer | Kaynak |
|---|---|---|
| Model | CNN-v1 (4 conv + 2 maxpool + FC 256), BatchNorm/Dropout yok | Tez önerisi; üçüncü değişkeni dışlamak |
| Başlatma | Xavier uniform (gain = 1), bias = 0, bütün aktivasyonlarda aynı | Adil karşılaştırma |
| Epoch / batch | 50 / 128, scheduler ve early stopping yok | T4 bütçesi; eşit eğitim bütçesi |
| Seed | 101, 202, 303 (bölme seed'i 2026) | Pilottaki seed 42'den bağımsız |
| SGD | LR 0,1 | 120 koşuluk LR pilotu |
| Momentum SGD | LR 0,01, momentum 0,9 | LR pilotu |
| RMSProp | LR 0,001, α = 0,99 | LR pilotu |
| Adam | LR 0,001, β₁ = 0,9, β₂ = 0,999 | LR pilotu |
| AdamW | LR 0,001, weight decay 0,01 | 24 koşuluk AdamW doğrulaması |

LR seçim kuralı: 6 aktivasyon üzerinde 30. epoch ortalama **validation** doğruluğu; eşitlikte düşük validation kaybı, tam eşitlikte düşük LR. Test seti ayar seçiminde kullanılmamıştır. Seçilen LR'ler dört veri setinde değiştirilmeden kullanılmıştır.

## Sonuçları üreten kod sürümleri

Her Colab notebook'u kodu GitHub'dan **sabit bir commit** üzerinden indirir ve bu commit'i her koşunun sonuç dosyasına (`tamamlandi.json` → `git_commit`) yazar. Ayrıca her sonuç dosyası dondurulmuş protokolün SHA256 özetini (`frozen_config_sha256`) içerir.

| Notebook | Deney | Kod commit'i |
|---|---|---|
| `colab/00`–`06` | Benchmark, doğrulamalar, LR pilotları | O tarihteki `main` (commit sonuç dosyalarında kayıtlı) |
| `colab/07_ana_deney_cifar10_90.ipynb` | Ana deney, CIFAR-10 | `1a30f6a` |
| `colab/08_ana_deney_mnist_90.ipynb` | Ana deney, MNIST | `1314d43` |
| `colab/09_ana_deney_animals10_90.ipynb` | Ana deney, Animals-10 | `1314d43` |
| `colab/10_ana_deney_flowers5_90.ipynb` | Ana deney, Flowers-5 | `8b2a37a` |
| `colab/11_resnet18_secilmis_dogrulama_cifar10.ipynb` | ResNet-18 | `9ba373b` |

`1a30f6a` → `1314d43` arasında yalnızca veri indirme yöntemi (Kaggle bağımlılığı kaldırıldı), `1314d43` → `8b2a37a` arasında yalnızca Flowers veri kaynağı metadatası değişmiştir. Model, optimizer, LR, epoch, batch ve seed değerleri bütün ana deneylerde aynıdır.

## Depo yapısı

```
kaynak_kod/        Model (CNN-v1, ResNet-18), aktivasyonlar, optimizerlar, eğitim döngüsü,
                   checkpoint/resume, yakınsama ve gradyan ölçümleri, veri hazırlama
ayarlar/           Pilot ve dondurulmuş ana deney yapılandırmaları
calistir_*.py      Deney çalıştırıcıları (LR pilotu, ana deney, ResNet-18)
colab/             Deneylerin yürütüldüğü Colab T4 notebook'ları (00–11)
analiz/            İstatistik analiz notebook'u (iki yönlü ANOVA, Tukey, basit etkiler)
sonuclar/ozet/     Tezde kullanılan özet çıktılar (ANOVA, LR seçimi, ResNet planı, benchmark)
teorik_analiz/     Aktivasyon türevleri, optimizer denklemleri, gradyan akışı, teorik beklentiler
dokumantasyon/     Protokol, deney planı, kararların gerekçeleri ve aşama raporları
```

## Ham sonuçlar (Google Drive)

Koşu bazındaki ham kayıtlar (her koşu için `tamamlandi.json`, epoch CSV'leri, checkpoint'ler) boyut nedeniyle depoda değil, Google Drive'da `MyDrive/tez_deney_sonuclari/` altındadır:

```
benchmark_colab_t4/                T4 hız ölçümü
ortak_ayarlar/                     Veri bölünmeleri (split) ve normalizasyon istatistikleri
asama1_*/, lr_pilotu_entegrasyon_testi/   Teknik doğrulamalar
lr_pilotu/, lr_pilotu_kalan_60/    120 LR koşusu
lr_pilotu_120_analiz/              LR seçimi
adamw_wd001_lr_pilotu/, _analiz/   AdamW doğrulaması
ana_deney/{cifar10,mnist,animals10,flowers5}/   359 ana koşu
ek_kararlilik_analizi/             Gradyan kırpmalı ek koşu
resnet18_secilmis_dogrulama/       18 ResNet-18 koşusu
anova_ciktilari/                   ANOVA, Tukey ve basit etki çıktıları
sonuc_gorselleri/                  Tez şekilleri
```

Veri setleri `MyDrive/tez_datasets/` altında tutulmuştur.

## Tekrar üretme

1. Google Colab'da (NVIDIA T4) ilgili `colab/` notebook'unu açın.
2. Notebook kodu kendi sabit commit'inden indirir, Drive'ı bağlar ve tamamlanmış koşuları atlar; yarım koşuları checkpoint'ten sürdürür.
3. Bir koşuyu yeniden üretmek için ilgili koşunun Drive'daki klasörünü başka bir yere taşıyıp notebook'u çalıştırın.
4. İstatistik analizi için `analiz/anova_veri_seti_basina_v2.ipynb` notebook'unu Colab'da çalıştırın; Drive'daki `ana_deney/` kayıtlarını okur ve `anova_ciktilari/` klasörüne yazar.

Not: `analiz/` notebook'u final test doğruluğu üzerinden ANOVA üretir. Tezdeki AULC/T95 analizleri ve şekiller aynı `tamamlandi.json` kayıtlarındaki `aulc_normalized` ve `epoch_to_95_own_best` alanlarından hesaplanmıştır.

## Nereden başlanmalı?

1. `dokumantasyon/00_tez_onerisi_uyum_matrisi.md`
2. `dokumantasyon/01_arastirma_protokolu.md`
3. `dokumantasyon/04_deneylerin_nedenleri.md`
4. `dokumantasyon/12_t4_benchmark_sonucu_ve_kapsam_karari.md`
5. `dokumantasyon/16_lr_pilotu_entegrasyon_ve_gercek_pilot.md`
6. `dokumantasyon/18_resnet18_secilmis_dogrulama.md`
7. `teorik_analiz/01`–`06`
