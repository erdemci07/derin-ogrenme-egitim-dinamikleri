"""CIFAR-10 veri hazırlama ve sabit train/validation ayrımı."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .tekrarlanabilirlik import dataloader_generator


# CIFAR-10 için yaygın kullanılan kanal istatistikleri.
# Ana deneyden önce istersek yalnız eğitim alt kümesinden yeniden hesaplayıp
# config dosyasında donduracağız.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _donusumler():
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    degerlendirme_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    return train_transform, degerlendirme_transform


def _split_indeksleri(
    etiketler: np.ndarray,
    split_dosyasi: Path,
    split_seed: int,
) -> tuple[list[int], list[int]]:
    """45.000/5.000 stratified ayrımı bir kez üretir ve kaydeder."""
    if split_dosyasi.exists():
        veri = json.loads(split_dosyasi.read_text(encoding="utf-8"))
        return veri["train_indices"], veri["validation_indices"]

    tum_indeksler = np.arange(len(etiketler))
    train_idx, val_idx = train_test_split(
        tum_indeksler,
        test_size=5000,
        random_state=split_seed,
        stratify=etiketler,
        shuffle=True,
    )

    split_dosyasi.parent.mkdir(parents=True, exist_ok=True)
    split_dosyasi.write_text(
        json.dumps(
            {
                "split_seed": split_seed,
                "train_indices": train_idx.tolist(),
                "validation_indices": val_idx.tolist(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return train_idx.tolist(), val_idx.tolist()


def cifar10_yukleyicileri(
    veri_klasoru: str,
    split_dosyasi: str,
    batch_size: int,
    split_seed: int,
    loader_seed: int,
    num_workers: int = 0,
):
    """Train ve validation DataLoader nesnelerini oluşturur.

    Test seti pilot/duman testi sırasında bilerek döndürülmez.
    """
    train_transform, val_transform = _donusumler()

    train_taban = datasets.CIFAR10(
        root=veri_klasoru,
        train=True,
        download=True,
        transform=train_transform,
    )
    val_taban = datasets.CIFAR10(
        root=veri_klasoru,
        train=True,
        download=False,
        transform=val_transform,
    )

    etiketler = np.asarray(train_taban.targets)
    train_idx, val_idx = _split_indeksleri(
        etiketler,
        Path(split_dosyasi),
        split_seed,
    )

    train_set = Subset(train_taban, train_idx)
    val_set = Subset(val_taban, val_idx)

    generator = dataloader_generator(loader_seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
