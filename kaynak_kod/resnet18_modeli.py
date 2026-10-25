"""CIFAR-10 uyarlamalı ResNet-18 ve aktivasyon-fonksiyonu yerleşimi.

Bu modül, tez önerisindeki seçili deneylerin daha derin bir mimaride
tekrar edilmesi için kullanılır. BatchNorm ve kısa-yol bağlantıları ResNet-18
mimarisinin sabit parçalarıdır; aktivasyon fonksiyonu ise tüm ana bloklarda
aynı biçimde değiştirilir.
"""

from __future__ import annotations

import torch
from torch import nn

from kaynak_kod.aktivasyonlar import aktivasyon_olustur


def _conv3x3(giris_kanal: int, cikis_kanal: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        giris_kanal,
        cikis_kanal,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class TemelBlok(nn.Module):
    """ResNet-18 için iki evrişimli temel blok."""

    expansion = 1

    def __init__(
        self,
        giris_kanal: int,
        cikis_kanal: int,
        aktivasyon: str,
        stride: int = 1,
    ) -> None:
        super().__init__()
        self.conv1 = _conv3x3(giris_kanal, cikis_kanal, stride)
        self.bn1 = nn.BatchNorm2d(cikis_kanal)
        self.aktivasyon1 = aktivasyon_olustur(aktivasyon)

        self.conv2 = _conv3x3(cikis_kanal, cikis_kanal)
        self.bn2 = nn.BatchNorm2d(cikis_kanal)
        self.aktivasyon2 = aktivasyon_olustur(aktivasyon)

        if stride != 1 or giris_kanal != cikis_kanal:
            self.kisayol = nn.Sequential(
                nn.Conv2d(
                    giris_kanal,
                    cikis_kanal,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(cikis_kanal),
            )
        else:
            self.kisayol = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cikti = self.aktivasyon1(self.bn1(self.conv1(x)))
        cikti = self.bn2(self.conv2(cikti))
        cikti = cikti + self.kisayol(x)
        return self.aktivasyon2(cikti)


class ResNet18CIFAR(nn.Module):
    """CIFAR-10 görüntü boyutuna uyarlanmış ResNet-18.

    ImageNet başlangıç katmanı ve max-pooling kullanılmaz; bunun yerine
    32x32 girdiler için 3x3, stride=1 başlangıç evrişimi kullanılır.
    Ağırlık başlatma aktivasyonlar arasında tarafsız kalması için tüm
    konvolüsyonel ve doğrusal katmanlarda Xavier uniform olarak sabittir.
    """

    def __init__(self, aktivasyon: str, sinif_sayisi: int = 10) -> None:
        super().__init__()
        self.aktivasyon_adi = aktivasyon

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.stem_aktivasyon = aktivasyon_olustur(aktivasyon)

        self.katman1 = self._katman_olustur(64, 64, aktivasyon, blok_sayisi=2)
        self.katman2 = self._katman_olustur(64, 128, aktivasyon, blok_sayisi=2, stride=2)
        self.katman3 = self._katman_olustur(128, 256, aktivasyon, blok_sayisi=2, stride=2)
        self.katman4 = self._katman_olustur(256, 512, aktivasyon, blok_sayisi=2, stride=2)

        self.havuz = nn.AdaptiveAvgPool2d((1, 1))
        self.siniflandirici = nn.Linear(512, sinif_sayisi)
        self._agirliklari_baslat()

    @staticmethod
    def _katman_olustur(
        giris_kanal: int,
        cikis_kanal: int,
        aktivasyon: str,
        blok_sayisi: int,
        stride: int = 1,
    ) -> nn.Sequential:
        bloklar = [TemelBlok(giris_kanal, cikis_kanal, aktivasyon, stride)]
        bloklar.extend(
            TemelBlok(cikis_kanal, cikis_kanal, aktivasyon)
            for _ in range(blok_sayisi - 1)
        )
        return nn.Sequential(*bloklar)

    def _agirliklari_baslat(self) -> None:
        for katman in self.modules():
            if isinstance(katman, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(katman.weight)
                if katman.bias is not None:
                    nn.init.zeros_(katman.bias)
            elif isinstance(katman, nn.BatchNorm2d):
                nn.init.ones_(katman.weight)
                nn.init.zeros_(katman.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_aktivasyon(self.bn1(self.conv1(x)))
        x = self.katman1(x)
        x = self.katman2(x)
        x = self.katman3(x)
        x = self.katman4(x)
        x = self.havuz(x)
        x = torch.flatten(x, 1)
        return self.siniflandirici(x)
