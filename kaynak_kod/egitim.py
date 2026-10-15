"""Tek bir deney koşusunun eğitim ve doğrulama döngüleri."""

from __future__ import annotations

import time

import torch
from torch import nn


def _cihaz_senkronize(cihaz: torch.device) -> None:
    """CUDA asenkron işlemlerini süre ölçümü sınırında tamamlar."""
    if cihaz.type == "cuda":
        torch.cuda.synchronize(cihaz)


def _macro_metrikler(confusion: torch.Tensor) -> dict[str, float]:
    """Confusion matrix üzerinden macro precision, recall ve F1 hesaplar."""
    confusion = confusion.to(dtype=torch.float64)
    tp = torch.diag(confusion)
    fp = confusion.sum(dim=0) - tp
    fn = confusion.sum(dim=1) - tp

    precision_payda = tp + fp
    recall_payda = tp + fn

    precision = torch.where(
        precision_payda > 0,
        tp / precision_payda,
        torch.zeros_like(tp),
    )
    recall = torch.where(
        recall_payda > 0,
        tp / recall_payda,
        torch.zeros_like(tp),
    )

    f1_payda = precision + recall
    f1 = torch.where(
        f1_payda > 0,
        2.0 * precision * recall / f1_payda,
        torch.zeros_like(tp),
    )

    return {
        "macro_precision": float(precision.mean().item()),
        "macro_recall": float(recall.mean().item()),
        "macro_f1": float(f1.mean().item()),
    }


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

    _cihaz_senkronize(cihaz)
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

    _cihaz_senkronize(cihaz)
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
    """Loss, accuracy ve macro sınıflandırma metriklerini hesaplar."""
    model.eval()

    toplam_kayip = 0.0
    toplam_dogru = 0
    toplam_ornek = 0
    confusion = None

    _cihaz_senkronize(cihaz)
    baslangic = time.perf_counter()

    for girdiler, hedefler in loader:
        girdiler = girdiler.to(cihaz, non_blocking=True)
        hedefler = hedefler.to(cihaz, non_blocking=True)

        logitler = model(girdiler)
        kayip = kayip_fonksiyonu(logitler, hedefler)
        tahminler = logitler.argmax(dim=1)

        if confusion is None:
            sinif_sayisi = int(logitler.shape[1])
            confusion = torch.zeros(
                (sinif_sayisi, sinif_sayisi),
                dtype=torch.int64,
                device=cihaz,
            )

        indeks = hedefler * confusion.shape[0] + tahminler
        confusion += torch.bincount(
            indeks,
            minlength=confusion.numel(),
        ).reshape_as(confusion)

        batch_boyutu = hedefler.size(0)
        toplam_kayip += kayip.item() * batch_boyutu
        toplam_dogru += (tahminler == hedefler).sum().item()
        toplam_ornek += batch_boyutu

    _cihaz_senkronize(cihaz)
    sure = time.perf_counter() - baslangic

    if confusion is None:
        raise ValueError("Doğrulama DataLoader'ı boş.")

    macro = _macro_metrikler(confusion)

    return {
        "loss": toplam_kayip / toplam_ornek,
        "accuracy": toplam_dogru / toplam_ornek,
        "seconds": sure,
        **macro,
    }
