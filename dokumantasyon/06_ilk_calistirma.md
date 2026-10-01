# İlk Çalıştırma — Duman Testi

Bu aşamada araştırma sonucu üretilmez. Amaç, deney altyapısının teknik olarak doğru çalıştığını doğrulamaktır.

## Ne çalıştırılıyor?

- Veri seti: CIFAR-10
- Model: CNN-v1
- Aktivasyon: ReLU
- Optimizasyon: SGD
- Seed: 42
- Epoch: 2
- Test seti: kullanılmıyor

## Kurulum

Python ortamında:

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python calistir_duman_testi.py
```

İlk çalıştırmada CIFAR-10 otomatik indirilecektir.

## Beklenen çıktılar

`sonuclar/duman_testi/` altında:

- `epoch_sonuclari.csv`
- `deney_bilgileri.json`
- `son_model.pt`

oluşmalıdır.

## Başarı ölçütü

Bu aşamada yüksek doğruluk beklenmez. İki epoch yalnızca sistem kontrolüdür.

Kontrol listesi:

- CIFAR-10 indirildi mi?
- 45.000 eğitim / 5.000 doğrulama ayrımı oluştu mu?
- Eğitim iki epoch tamamlandı mı?
- Loss ve accuracy değerleri kaydedildi mi?
- Son model kaydedildi mi?
- Aynı seed ile tekrar çalıştırıldığında sonuçlar tutarlı mı?
