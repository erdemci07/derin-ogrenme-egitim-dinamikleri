# Windows ve VS Code Kurulumu

Bu sayfa, projeyi yeni bir Windows bilgisayarda veya farklı bir VS Code kurulumunda ilk kez açan kişi içindir.

## En kolay yöntem

Depoyu bilgisayara indirdikten veya GitHub'dan clone ettikten sonra proje klasöründe:

`kurulum_windows.cmd`

dosyasına çift tıklayın.

Bu işlem otomatik olarak:

1. Proje içinde `.venv` adlı ayrı bir Python sanal ortamı oluşturur.
2. `pip`, `setuptools` ve `wheel` araçlarını günceller.
3. `requirements.txt` içindeki bütün gerekli Python kütüphanelerini kurar.
4. Kurulan kütüphanelerin çalıştığını kontrol eder.
5. Kullanılan PyTorch/CUDA/GPU durumunu ekrana yazar.

PowerShell script çalıştırma kısıtlaması olan bilgisayarlarda `.cmd` dosyası gerekli geçici izinle betiği başlatır; Windows'un genel Execution Policy ayarı kalıcı olarak değiştirilmez.

## VS Code'da Python seçimi

Kurulumdan sonra VS Code içinde:

`Ctrl + Shift + P`

ardından:

`Python: Select Interpreter`

seçin ve aşağıdaki yorumlayıcıyı kullanın:

`.venv\Scripts\python.exe`

## Duman testini çalıştırma

VS Code terminalinde:

```powershell
.\.venv\Scripts\python.exe calistir_duman_testi.py
```

İlk çalıştırmada CIFAR-10 veri seti otomatik olarak indirilecektir.

## Neden sanal ortam kullanılıyor?

Bu proje için gerekli paketleri bilgisayardaki diğer Python projelerinden ayırmak ve aynı çalışma ortamını farklı bilgisayarlarda daha kolay yeniden kurabilmek için `.venv` kullanılmaktadır.

## GPU hakkında

İlk duman testi CPU'da da çalışabilir. Ana deneylerin hesaplama yükü daha yüksek olduğu için GPU kullanılacaktır. Kurulum betiği mevcut PyTorch kurulumunun CUDA GPU'yu görüp görmediğini raporlar.

PyTorch'un uygun CUDA sürümü bilgisayardaki NVIDIA sürücüsüne göre değişebildiğinden, GPU kurulumu ana deneylere geçmeden önce ayrıca doğrulanacaktır. Bu aşamada amaç, projenin temel Python ortamını tek komutla kurabilmektir.
