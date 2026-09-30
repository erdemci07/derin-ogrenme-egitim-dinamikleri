"""Tez önerisinde yer alan beş optimizasyon algoritmasının ortak üreticisi."""

from __future__ import annotations

from collections.abc import Iterable

import torch


OPTIMIZASYON_ADLARI = (
    "sgd",
    "momentum_sgd",
    "rmsprop",
    "adam",
    "adamw",
)


def optimizasyon_olustur(
    ad: str,
    parametreler: Iterable[torch.nn.Parameter],
    learning_rate: float,
    weight_decay: float = 0.0,
) -> torch.optim.Optimizer:
    """Seçilen optimizerı protokolde sabitlenen temel parametrelerle oluşturur."""
    ad = ad.lower().strip()

    if ad == "sgd":
        return torch.optim.SGD(
            parametreler,
            lr=learning_rate,
            momentum=0.0,
            weight_decay=weight_decay,
        )

    if ad == "momentum_sgd":
        return torch.optim.SGD(
            parametreler,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )

    if ad == "rmsprop":
        return torch.optim.RMSprop(
            parametreler,
            lr=learning_rate,
            alpha=0.99,
            momentum=0.0,
            weight_decay=weight_decay,
        )

    if ad == "adam":
        return torch.optim.Adam(
            parametreler,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

    if ad == "adamw":
        return torch.optim.AdamW(
            parametreler,
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )

    raise ValueError(
        f"Bilinmeyen optimizasyon algoritması: {ad}. "
        f"Geçerli seçenekler: {', '.join(OPTIMIZASYON_ADLARI)}"
    )
