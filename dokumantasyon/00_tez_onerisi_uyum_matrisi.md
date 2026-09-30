# Tez Önerisi Uyum Matrisi

Bu belge, GitHub deposundaki teknik çalışmaların yüklenen **"Copy of Tez onerisi.docx"** belgesindeki tez önerisiyle uyumunu izlemek için tutulmaktadır.

Bu depo için temel kural şudur:

> Tez önerisinde açıkça tanımlanan amaç, kapsam ve yöntem ana çerçevedir. Teknik uygulamada öneride bulunmayan bir karar alınırsa bu karar ayrıca belirtilir ve gerekçesi dokümante edilir.

## Öneriden doğrudan gelen ana unsurlar

| Tez önerisindeki unsur | Depodaki karşılığı | Durum |
|---|---|---|
| 6 aktivasyon fonksiyonu | `kaynak_kod/aktivasyonlar.py` | Eklendi |
| 5 optimizasyon algoritması | `kaynak_kod/optimizasyon_algoritmalari.py` | Eklendi |
| 6 × 5 = 30 kombinasyon | `dokumantasyon/01_arastirma_protokolu.md` | Planlandı |
| Aynı ağ mimarisi | `kaynak_kod/cnn_modeli.py` | CNN-v1 oluşturuldu |
| Aynı veri bölünmesi | `kaynak_kod/veri_hazirlama.py` | Sabit split sistemi eklendi |
| Aynı eğitim bütçesi | Ana deney ayarlarında dondurulacak | Pilot sonrası |
| En az 3 tekrar | Hedef 5 seed olarak genişletildi | Pilot sonrası kesinleşecek |
| CIFAR-10 ana deney | Veri hazırlama altyapısı | Eklendi |
| MNIST | Doğrulama aşaması | Daha sonra |
| Animals-10 | Doğrulama aşaması | Daha sonra |
| 5 Flower Types | Doğrulama aşaması | Daha sonra |
| Basit CNN | CNN-v1 | Eklendi |
| Seçilmiş ResNet-18 deneyleri | Doğrulama aşaması | Daha sonra |
| Accuracy | Metrik sistemi | Geliştirilecek |
| Precision | Metrik sistemi | Geliştirilecek |
| Recall | Metrik sistemi | Geliştirilecek |
| F1 | Metrik sistemi | Geliştirilecek |
| Training / validation loss | Eğitim döngüsü | Temel hali eklendi |
| Yakınsama hızı | `yakinsama_analizi.py` | Sonraki aşama |
| Eğitim süresi | Epoch süresi ölçümü | Temel hali eklendi |
| Two-way ANOVA | İstatistiksel analiz | Sonraki aşama |
| Aktivasyon × optimizer etkileşimi | Ana istatistiksel araştırma sorusu | Protokolde tanımlandı |
| Türev ve gradyan davranışı | Gradyan analizi | Sonraki aşama |
| Teorik + deneysel bütünleştirme | Tez tartışma yapısı | Süreç boyunca |

## Tez önerisinde bulunmayan ama teknik uygulama için eklediğimiz kararlar

Aşağıdaki noktalar tez önerisinin doğrudan cümleleri değildir. Deneyin kontrollü, tekrarlanabilir ve açıklanabilir olması için teknik tasarım kararı olarak eklenmiştir:

- 45.000 eğitim / 5.000 doğrulama ayrımı
- Hedef olarak 5 seed kullanılması
- CNN-v1 katman ve kanal sayıları
- BatchNorm ve Dropout'un ana küçük CNN'de kullanılmaması
- 2 epoch duman testi
- Pilot öğrenme oranı araması
- Test setinin hiperparametre seçiminden ayrılması
- AULC gibi yakınsama ölçülerinin eklenmesi
- Sonuçların GitHub üzerinde sürüm ve deney kimliğiyle izlenmesi

Bu kararların hiçbiri tez önerisinin ana araştırma sorusunu değiştirmek amacıyla eklenmemiştir. Ana deney başlamadan önce gerekli olanlar danışmanla netleştirilip dondurulacaktır.

## Şu anki aşama

Tez önerisindeki çalışma takvimine paralel olarak şu anda:

**veri setlerinin hazırlanması + deney ortamının oluşturulması + CNN mimarisinin geliştirilmesi/doğrulanması**

aşamalarına karşılık gelen teknik altyapı kurulmaktadır.

Henüz ana aktivasyon-optimizasyon deneyleri başlatılmamıştır.
