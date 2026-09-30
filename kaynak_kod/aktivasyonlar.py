"""Tez önerisinde yer alan altı aktivasyon fonksiyonunun ortak üreticisi."""

from __future__ import annotations

from torch import nn


AKTIVASYON_ADLARI = (
    "relu",
    "leaky_relu",
    "elu",
    "gelu",
    "swish",
    "mish",
)


def aktivasyon_olustur(ad: str) -> nn.Module:
    """Verilen kısa ada göre aktivasyon modülünü oluşturur.

    Sabit parametreler:
    - LeakyReLU negative_slope = 0.01
    - ELU alpha = 1.0
    - Swish = SiLU (beta=1 eşdeğeri)
    - GELU exact form (approximate="none")
    """
    ad = ad.lower().strip()

    if ad == "relu":
        return nn.ReLU()
    if ad == "leaky_relu":
        return nn.LeakyReLU(negative_slope=0.01)
    if ad == "elu":
        return nn.ELU(alpha=1.0)
    if ad == "gelu":
        return nn.GELU(approximate="none")
    if ad == "swish":
        return nn.SiLU()
    if ad == "mish":
        return nn.Mish()

    raise ValueError(
        f"Bilinmeyen aktivasyon: {ad}. "
        f"Geçerli seçenekler: {', '.join(AKTIVASYON_ADLARI)}"
    )
