"""Ana kontrollü deneylerde kullanılacak CNN-v1 modeli."""

from __future__ import annotations

import torch
from torch import nn

from .aktivasyonlar import aktivasyon_olustur


class CNNv1(nn.Module):
    """CIFAR-10 için sabit ve yorumlanabilir temel CNN.

    BatchNorm ve Dropout bilerek kullanılmamaktadır. Amaç, ana kontrollü
    deneyde aktivasyon ve optimizer dışındaki değişken sayısını azaltmaktır.
    """

    def __init__(self, aktivasyon: str, sinif_sayisi: int = 10) -> None:
        super().__init__()

        if sinif_sayisi <= 1:
            raise ValueError("sinif_sayisi en az 2 olmalıdır.")

        self.ozellikler = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            aktivasyon_olustur(aktivasyon),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            aktivasyon_olustur(aktivasyon),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            aktivasyon_olustur(aktivasyon),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            aktivasyon_olustur(aktivasyon),
            nn.MaxPool2d(kernel_size=2),
        )

        self.siniflandirici = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            aktivasyon_olustur(aktivasyon),
            nn.Linear(256, sinif_sayisi),
        )

        self._ortak_agirlik_baslatma()

    def _ortak_agirlik_baslatma(self) -> None:
        """Bütün aktivasyonlarda aynı başlangıç kuralını uygular.

        Aktivasyona özel initialization kullanılmaz; böylece karşılaştırmaya
        üçüncü bir değişken eklenmemiş olur.
        """
        for modul in self.modules():
            if isinstance(modul, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(modul.weight)
                if modul.bias is not None:
                    nn.init.zeros_(modul.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ozellikler(x)
        return self.siniflandirici(x)
