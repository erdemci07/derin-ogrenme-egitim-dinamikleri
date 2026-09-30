"""Tek bir deney koşusunun eğitim ve doğrulama döngüleri."""

from __future__ import annotations

import time

import torch
from torch import nn


def _accuracy(logitler: torch.Tensor, hedefler: torch.Tensor) -> float:
    tahminler = logitler.argmax(dim=1)
    return (tahminler == hedefler).float().mean().item()


def bir_epoch_egit(
    model: nn.Module,
    loader,
    kayip_fonksiyonu: nn.Module,
    optimizer: torch.optim.Optimizer,
    cihaz: torch.device,
) -> dict:
    model.train()

    toplam_kayip = 0.0
    toplam_dogru = 0
    toplam_ornek = 0
    baslangic = time.perf_counter()

    for girdiler, hedefler in loader:
        girdiler = girdiler.to(cihaz, non_blocking=True)
        hedefler = hedefler.to(cihaz, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logitler = model(girdiler)
        kayip = kayip_fonksiyonu(logitler, hedefler)
        kayip.backward()
        optimizer.step()

        batch_boyutu = hedefler.size(0)
        toplam_kayip += kayip.item() * batch_boyutu
        toplam_dogru += (logitler.argmax(dim=1) == hedefler).sum().item()
        toplam_ornek += batch_boyutu

    sure = time.perf_counter() - baslangic

    return {
        "loss": toplam_kayip / toplam_ornek,
        "accuracy": toplam_dogru / toplam_ornek,
        "seconds": sure,
    }


@torch.no_grad()
def dogrula(
    model: nn.Module,
    loader,
    kayip_fonksiyonu: nn.Module,
    cihaz: torch.device,
) -> dict:
    model.eval()

    toplam_kayip = 0.0
    toplam_dogru = 0
    toplam_ornek = 0

    for girdiler, hedefler in loader:
        girdiler = girdiler.to(cihaz, non_blocking=True)
        hedefler = hedefler.to(cihaz, non_blocking=True)

        logitler = model(girdiler)
        kayip = kayip_fonksiyonu(logitler, hedefler)

        batch_boyutu = hedefler.size(0)
        toplam_kayip += kayip.item() * batch_boyutu
        toplam_dogru += (logitler.argmax(dim=1) == hedefler).sum().item()
        toplam_ornek += batch_boyutu

    return {
        "loss": toplam_kayip / toplam_ornek,
        "accuracy": toplam_dogru / toplam_ornek,
    }
