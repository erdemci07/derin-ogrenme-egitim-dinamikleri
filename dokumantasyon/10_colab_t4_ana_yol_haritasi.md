# Colab T4 Ana Yol Haritası

Bu belge, tez önerisindeki kapsam korunarak deneylerin Google Colab üzerinde NVIDIA T4 GPU ile adım adım yürütülmesi için hazırlanmıştır.

Temel ilke: Büyük deney paketleri tek seferde başlatılmayacaktır. Her aşama teknik ve bilimsel olarak doğrulandıktan sonra bir sonraki aşamaya geçilecektir.

## Hesaplama ortamı

Ana deney ortamı:
- Google Colab
- NVIDIA T4 GPU
- PyTorch
- CUDA
- GitHub üzerinden sürüm kontrollü kaynak kod
- Google Drive üzerinde kalıcı deney çıktıları

Colab oturumlarının kesilebilmesi nedeniyle bütün uzun koşular checkpoint/resume destekli olacaktır. Her koşu tamamlandığında sonuç, metadata ve özet kayıtları kalıcı depoya yazılacaktır.

## Aşama 0 — T4 ortam doğrulama ve gerçek süre ölçümü

Amaç:
- Gerçekten NVIDIA T4 atandığını doğrulamak
- PyTorch/CUDA sürümlerini kaydetmek
- GPU belleğini kaydetmek
- CIFAR-10 + CNN-v1 için gerçek epoch süresini ölçmek
- ResNet-18 için temsilci süre ölçümü yapmak
- GPU bellek kullanımını ölçmek
- Aynı seed tekrarını T4 üzerinde doğrulamak

Bu aşama tamamlanmadan toplam deney süresi veya maliyeti kesin kabul edilmeyecektir.

## Aşama 1 — Deney motorunu ana çalışmaya hazır hale getirme

Tamamlanacak teknik işler:
- eğitim alt kümesinden normalizasyon istatistiklerini hesaplama ve dondurma
- split bütünlük ve sınıf dağılımı kontrolü
- CUDA zaman ölçümünde senkronizasyon
- checkpoint/resume
- yarım kalan run devam ettirme
- NaN/Inf ve sayısal kararsızlık kontrolü
- deney kimliği ve tekrar koşu koruması
- ortam metadata kaydı
- GPU adı, PyTorch, CUDA ve Git commit kaydı
- accuracy, macro precision, recall ve F1
- train/validation/test loss
- AULC
- eşik yakınsama ölçümleri
- gradyan ve parametre güncelleme probe sistemi

## Aşama 2 — Teorik analiz doğrulaması

Altı aktivasyon fonksiyonunun:
- fonksiyonu
- birinci türevi
- süreklilik özellikleri
- doygunluk / ölü bölge davranışı
- sıfıra yakın türev davranışı

beş optimizerın:
- güncelleme kuralı
- moment mekanizması
- adaptif ölçekleme
- weight decay davranışı

analitik olarak incelenecektir.

Elle yazılan türevler seçilmiş noktalarda PyTorch autograd ile sayısal olarak kontrol edilecektir.

## Aşama 3 — Hiperparametre pilotları

### Learning-rate pilotu
3 temsilci aktivasyon x 5 optimizer x 4 LR = 60 koşu.

### Weight-decay pilotu
3 temsilci aktivasyon x 5 optimizer x 3 WD = 45 koşu.

Toplam pilot: 105 koşu.

Test seti hiperparametre seçiminde kullanılmayacaktır.

Pilotlar tamamlandıktan sonra ana deney ayarları dondurulacaktır.

## Aşama 4 — Ana CIFAR-10 deneyleri

İlk güvenli hedef:
30 kombinasyon x 5 seed = 150 koşu.

Hesaplama bütçesi uygunsa:
30 kombinasyon x 10 seed = 300 koşu.

Ana sonuçlar:
- test accuracy
- macro F1
- validation AULC
- yakınsama süresi
- eğitim süresi
- genelleme farkı
- gradyan normları
- aktivasyon dağılımları
- parametre güncelleme büyüklükleri

## Aşama 5 — Ek veri setleri

Maksimum kapsam:
- MNIST: 30 kombinasyon x 5 seed = 150
- Animals-10: 30 kombinasyon x 5 seed = 150
- 5 Flower Types: 30 kombinasyon x 5 seed = 150

Toplam: 450 koşu.

## Aşama 6 — ResNet-18 doğrulaması

Ana CNN sonuçlarından önceden tanımlanmış seçim kuralıyla 12 temsilci aktivasyon-optimizer kombinasyonu seçilecektir.

Maksimum plan:
12 kombinasyon x 5 seed x 2 doğrulama veri seti = 120 koşu.

Bu bölüm ana faktöriyel deneyin yerine geçmez; ağ derinliği arttığında gözlenen ilişkinin korunup korunmadığını sınar.

## Aşama 7 — Sağlamlık / duyarlılık deneyleri

Maksimum 60 ek koşu.

Bu koşular yalnız önceden tanımlanmış sorular için kullanılacaktır. Sırf deney sayısını artırmak amacıyla koşu eklenmeyecektir.

Örnek sorular:
- seçilen epoch bütçesine duyarlılık
- scheduler kullanılmadığında / seçilmiş duyarlılık koşulunda davranış
- farklı başlangıç seed sayısının sonuç sıralamasına etkisi

## Maksimum deney kapsamı

- LR pilotu: 60
- WD pilotu: 45
- CIFAR-10: 300
- üç ek veri seti: 450
- ResNet-18: 120
- sağlamlık: 60

Toplam maksimum: 1.035 koşu.

Bu sayı hedef üst sınırdır. T4 benchmark sonuçları ve Colab kaynak tüketimi uygun değilse bilimsel değeri en yüksek koşular korunarak kapsam azaltılacaktır.

## Aşama 8 — İstatistiksel analiz

Ana faktöriyel sonuçlarda:
- two-way ANOVA
- aktivasyon ana etkisi
- optimizer ana etkisi
- aktivasyon x optimizer etkileşimi
- etki büyüklükleri
- güven aralıkları
- varsayım kontrolleri
- gerekli simple-effects / post-hoc analizleri
- çoklu karşılaştırma düzeltmesi
- seed etkisini değerlendiren karma model sağlamlık analizi

uygulanacaktır.

## Aşama 9 — Tez üretimi

Deney süreci devam ederken tez metni paralel geliştirilecektir.

Ana bölümler:
- giriş ve araştırma problemi
- literatür
- matematiksel / teorik analiz
- materyal ve yöntem
- deney tasarımı
- bulgular
- istatistiksel analiz
- tartışma
- sonuç ve öneriler
- kaynakça
- gerekli ekler

Nihai biçim Van Yüzüncü Yıl Üniversitesi Fen Bilimleri Enstitüsü güncel tez şablonuna göre hazırlanacaktır.

## Aşama 10 — Savunma paketi

Sonuçlardan sonra:
- tez PDF/DOCX
- yüksek çözünürlüklü şekil ve çizelgeler
- ana sonuç özeti
- yöntem akış şeması
- literatür farkı tablosu
- teorik beklenti / deney bulgusu eşleştirmesi
- 18–24 slaytlık savunma sunumu
- olası jüri soruları için teknik kontrol listesi

hazırlanacaktır.

## Geçiş kuralı

Her aşamada:
1. kod doğrulanır,
2. küçük ölçekli test yapılır,
3. çıktı kontrol edilir,
4. sonuç GitHub'da belgelenir,
5. ancak bundan sonra sonraki aşamaya geçilir.

İlk sıradaki iş: Colab T4 benchmark ve süre ölçümü.


## Benchmark sonrası güncel kapsam kararı

Gerçek Tesla T4 benchmarkı tamamlandı. 1.035 koşu maksimum teorik üst sınır olarak korunmakla birlikte varsayılan çalışma planı değildir.

Güncel yürütme sırası:
- Seviye A: 255 çekirdek koşu
- Seviye B: ek veri setlerinde başlangıçta 3 seed
- Seviye C: seçilmiş ResNet-18 doğrulaması
- kaynak uygunsa seed ve kapsam genişletmesi

Ayrıntı: `dokumantasyon/12_t4_benchmark_sonucu_ve_kapsam_karari.md`
