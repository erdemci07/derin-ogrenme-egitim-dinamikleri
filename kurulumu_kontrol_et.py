"""Kurulumdan sonra temel kütüphanelerin erişilebilirliğini doğrular.

Bu dosya bilimsel deney çalıştırmaz. Yalnızca ortamın hazır olup olmadığını kontrol eder.
"""

from __future__ import annotations

import importlib
import platform
import shutil
import sys


KUTUPHANELER = {
    "torch": "PyTorch",
    "torchvision": "torchvision",
    "numpy": "NumPy",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "scipy": "SciPy",
    "statsmodels": "statsmodels",
    "matplotlib": "Matplotlib",
    "yaml": "PyYAML",
}


def ana() -> None:
    print("\nKurulum kontrolu basliyor...\n")

    hata_var = False

    for modul_adi, gorunen_ad in KUTUPHANELER.items():
        try:
            modul = importlib.import_module(modul_adi)
            surum = getattr(modul, "__version__", "surum bilgisi yok")
            print(f"[OK] {gorunen_ad}: {surum}")
        except Exception as hata:
            hata_var = True
            print(f"[HATA] {gorunen_ad}: {hata}")

    print("\nPython:", sys.version.split()[0])
    print("Isletim sistemi:", platform.platform())

    try:
        import torch

        print("CUDA kullanilabilir:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("CUDA surumu:", torch.version.cuda)
        else:
            if shutil.which("nvidia-smi"):
                hata_var = True
                print(
                    "[HATA] NVIDIA GPU algilandi ancak PyTorch CUDA'yi kullanamiyor. "
                    "CUDA destekli PyTorch kurulumu tamamlanmamis olabilir."
                )
            else:
                print(
                    "Not: NVIDIA CUDA GPU algilanmadi. CPU modu kullanilacak."
                )
    except Exception:
        pass

    if hata_var:
        raise SystemExit("\nKurulum tamamlanmadi. Yukaridaki HATA satirlarini kontrol edin.")

    print("\nTum temel kutuphaneler hazir.")


if __name__ == "__main__":
    ana()
