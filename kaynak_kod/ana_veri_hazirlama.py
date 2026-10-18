"""Ana 360 koşu için veri setlerini tek ve dondurulmuş protokolle hazırlar.

İlkeler:
- Splitler bir kez oluşturulur ve JSON olarak saklanır.
- Normalizasyon yalnız TRAIN alt kümesinden hesaplanır.
- Validation/test verileri augmentation almaz.
- Tüm görüntüler CNN-v1 için 3x32x32 biçimine getirilir.
- Test seti hiperparametre seçimi için kullanılmaz; yalnız final değerlendirmede kullanılır.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import ImageFile
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .tekrarlanabilirlik import dataloader_generator
from .veri_hazirlama import (
    _cifar10_drive_verisini_hazirla,
    _donusumler,
    _istatistikleri_al,
    _split_hash,
    _split_indeksleri,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass
class AnaVeriPaketi:
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    sinif_sayisi: int
    sinif_adlari: list[str]
    meta: dict[str, Any]


def _json_yaz(yol: Path, veri: dict[str, Any]) -> None:
    yol.parent.mkdir(parents=True, exist_ok=True)
    gecici = yol.with_suffix(yol.suffix + ".tmp")
    gecici.write_text(
        json.dumps(veri, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(gecici, yol)


def _indeks_hash(*indeks_gruplari: list[int]) -> str:
    ham = json.dumps(indeks_gruplari, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(ham).hexdigest()


def _data_loader(
    dataset,
    indeksler: list[int] | None,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    generator: torch.Generator | None = None,
) -> DataLoader:
    veri = Subset(dataset, indeksler) if indeksler is not None else dataset
    return DataLoader(
        veri,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator if shuffle else None,
    )


def _kanal_istatistikleri(
    dataset,
    train_idx: list[int],
    *,
    batch_size: int = 512,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    kanal_toplami = None
    kanal_kare_toplami = None
    piksel_sayisi = 0

    for goruntuler, _ in loader:
        goruntuler = goruntuler.to(dtype=torch.float64)
        if kanal_toplami is None:
            kanal_sayisi = goruntuler.shape[1]
            kanal_toplami = torch.zeros(kanal_sayisi, dtype=torch.float64)
            kanal_kare_toplami = torch.zeros(kanal_sayisi, dtype=torch.float64)

        kanal_toplami += goruntuler.sum(dim=(0, 2, 3))
        kanal_kare_toplami += (goruntuler * goruntuler).sum(dim=(0, 2, 3))
        piksel_sayisi += (
            goruntuler.shape[0] * goruntuler.shape[2] * goruntuler.shape[3]
        )

    if kanal_toplami is None or kanal_kare_toplami is None or piksel_sayisi == 0:
        raise ValueError("Normalizasyon istatistikleri için eğitim verisi boş.")

    ortalama = kanal_toplami / piksel_sayisi
    varyans = kanal_kare_toplami / piksel_sayisi - ortalama * ortalama
    std = torch.sqrt(torch.clamp(varyans, min=0.0))
    return tuple(float(x) for x in ortalama), tuple(float(x) for x in std)


def _stats_al_veya_hesapla(
    *,
    dataset_adi: str,
    stats_dosyasi: Path,
    split_seed: int,
    train_idx: list[int],
    stats_dataset,
    dataset_fingerprint: str,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    train_hash = _indeks_hash(train_idx)

    if stats_dosyasi.exists():
        veri = json.loads(stats_dosyasi.read_text(encoding="utf-8"))
        if veri.get("dataset") != dataset_adi:
            raise ValueError("Normalizasyon dosyası başka veri setine ait.")
        if int(veri["split_seed"]) != split_seed:
            raise ValueError("Normalizasyon split seed değeri uyuşmuyor.")
        if veri.get("train_index_sha256") != train_hash:
            raise ValueError("Normalizasyon eğitim indeksleri mevcut split ile uyuşmuyor.")
        if veri.get("dataset_fingerprint") != dataset_fingerprint:
            raise ValueError("Normalizasyon dosyası mevcut veri sürümüyle uyuşmuyor.")

        mean = tuple(float(x) for x in veri["mean"])
        std = tuple(float(x) for x in veri["std"])
        if len(mean) != 3 or len(std) != 3 or any(x <= 0 for x in std):
            raise ValueError("Normalizasyon istatistik dosyası geçersiz.")
        return mean, std

    print(
        f"{dataset_adi}: normalizasyon yalnız eğitim alt kümesinden hesaplanıyor...",
        flush=True,
    )
    mean, std = _kanal_istatistikleri(stats_dataset, train_idx)
    _json_yaz(
        stats_dosyasi,
        {
            "dataset": dataset_adi,
            "source": "training subset only",
            "split_seed": split_seed,
            "train_index_sha256": train_hash,
            "dataset_fingerprint": dataset_fingerprint,
            "mean": mean,
            "std": std,
        },
    )
    return mean, std


def _stratified_train_val_split(
    *,
    etiketler: np.ndarray,
    split_dosyasi: Path,
    split_seed: int,
    train_sayisi: int,
    val_sayisi: int,
    dataset_adi: str,
) -> tuple[list[int], list[int]]:
    toplam = len(etiketler)
    if train_sayisi + val_sayisi != toplam:
        raise ValueError("Train/validation sayıları toplam örnek sayısıyla uyuşmuyor.")

    if split_dosyasi.exists():
        veri = json.loads(split_dosyasi.read_text(encoding="utf-8"))
        if veri.get("dataset") != dataset_adi:
            raise ValueError("Split dosyası başka veri setine ait.")
        if int(veri["split_seed"]) != split_seed:
            raise ValueError("Split seed uyuşmuyor.")
        train_idx = [int(x) for x in veri["train_indices"]]
        val_idx = [int(x) for x in veri["validation_indices"]]
    else:
        tum = np.arange(toplam)
        train_np, val_np = train_test_split(
            tum,
            test_size=val_sayisi,
            random_state=split_seed,
            stratify=etiketler,
            shuffle=True,
        )
        train_idx = train_np.tolist()
        val_idx = val_np.tolist()
        _json_yaz(
            split_dosyasi,
            {
                "dataset": dataset_adi,
                "split_seed": split_seed,
                "split_sha256": _indeks_hash(train_idx, val_idx),
                "train_count": len(train_idx),
                "validation_count": len(val_idx),
                "train_indices": train_idx,
                "validation_indices": val_idx,
            },
        )

    if len(train_idx) != train_sayisi or len(val_idx) != val_sayisi:
        raise ValueError(
            f"{dataset_adi} split boyutu beklenenden farklı: "
            f"train={len(train_idx)}, val={len(val_idx)}"
        )
    if set(train_idx).intersection(val_idx):
        raise ValueError(f"{dataset_adi} train ve validation splitleri çakışıyor.")
    if len(set(train_idx).union(val_idx)) != toplam:
        raise ValueError(f"{dataset_adi} split tüm eğitim örneklerini kapsamıyor.")

    return train_idx, val_idx


def _imagefolder_fingerprint(taban: datasets.ImageFolder, kok: Path) -> str:
    kayitlar = []
    for yol, etiket in taban.samples:
        p = Path(yol)
        try:
            rel = p.relative_to(kok).as_posix()
        except ValueError:
            rel = p.name
        kayitlar.append((rel, int(etiket), int(p.stat().st_size)))
    ham = json.dumps(
        {"classes": taban.classes, "samples": kayitlar},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(ham).hexdigest()


def _imagefolder_split(
    *,
    taban: datasets.ImageFolder,
    kok: Path,
    split_dosyasi: Path,
    split_seed: int,
    dataset_adi: str,
) -> tuple[list[int], list[int], list[int], str]:
    fingerprint = _imagefolder_fingerprint(taban, kok)
    etiketler = np.asarray(taban.targets)
    toplam = len(etiketler)

    if split_dosyasi.exists():
        veri = json.loads(split_dosyasi.read_text(encoding="utf-8"))
        if veri.get("dataset") != dataset_adi:
            raise ValueError("Split dosyası başka veri setine ait.")
        if int(veri["split_seed"]) != split_seed:
            raise ValueError("Split seed uyuşmuyor.")
        if veri.get("dataset_fingerprint") != fingerprint:
            raise ValueError(
                "Veri setinin dosya içeriği önceki split oluşturulduğundan beri değişmiş."
            )
        train_idx = [int(x) for x in veri["train_indices"]]
        val_idx = [int(x) for x in veri["validation_indices"]]
        test_idx = [int(x) for x in veri["test_indices"]]
    else:
        tum = np.arange(toplam)
        train_np, temp_np = train_test_split(
            tum,
            test_size=0.20,
            random_state=split_seed,
            stratify=etiketler,
            shuffle=True,
        )
        temp_labels = etiketler[temp_np]
        val_np, test_np = train_test_split(
            temp_np,
            test_size=0.50,
            random_state=split_seed + 1,
            stratify=temp_labels,
            shuffle=True,
        )
        train_idx = train_np.tolist()
        val_idx = val_np.tolist()
        test_idx = test_np.tolist()

        _json_yaz(
            split_dosyasi,
            {
                "dataset": dataset_adi,
                "split_seed": split_seed,
                "dataset_fingerprint": fingerprint,
                "split_sha256": _indeks_hash(train_idx, val_idx, test_idx),
                "train_count": len(train_idx),
                "validation_count": len(val_idx),
                "test_count": len(test_idx),
                "class_names": list(taban.classes),
                "train_indices": train_idx,
                "validation_indices": val_idx,
                "test_indices": test_idx,
            },
        )

    a, b, c = set(train_idx), set(val_idx), set(test_idx)
    if a & b or a & c or b & c:
        raise ValueError(f"{dataset_adi} splitleri birbiriyle çakışıyor.")
    if len(a | b | c) != toplam:
        raise ValueError(f"{dataset_adi} splitleri veri setinin tamamını kapsamıyor.")

    return train_idx, val_idx, test_idx, fingerprint


def _cifar10_paketi(
    *,
    veri_koku: Path,
    split_dosyasi: Path,
    stats_dosyasi: Path,
    batch_size: int,
    split_seed: int,
    loader_seed: int,
    num_workers: int,
) -> AnaVeriPaketi:
    _cifar10_drive_verisini_hazirla(veri_koku)

    indeks_taban = datasets.CIFAR10(
        root=str(veri_koku),
        train=True,
        download=False,
        transform=None,
    )
    etiketler = np.asarray(indeks_taban.targets)
    train_idx, val_idx = _split_indeksleri(etiketler, split_dosyasi, split_seed)

    mean, std = _istatistikleri_al(
        veri_koku,
        train_idx,
        split_seed,
        stats_dosyasi,
    )
    train_transform, eval_transform = _donusumler(mean, std)

    train_taban = datasets.CIFAR10(
        root=str(veri_koku), train=True, download=False, transform=train_transform
    )
    val_taban = datasets.CIFAR10(
        root=str(veri_koku), train=True, download=False, transform=eval_transform
    )
    test_taban = datasets.CIFAR10(
        root=str(veri_koku), train=False, download=False, transform=eval_transform
    )

    generator = dataloader_generator(loader_seed)
    return AnaVeriPaketi(
        train_loader=_data_loader(
            train_taban,
            train_idx,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        ),
        validation_loader=_data_loader(
            val_taban,
            val_idx,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        test_loader=_data_loader(
            test_taban,
            None,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        sinif_sayisi=10,
        sinif_adlari=list(train_taban.classes),
        meta={
            "dataset": "cifar10",
            "train_count": len(train_idx),
            "validation_count": len(val_idx),
            "test_count": len(test_taban),
            "split_seed": split_seed,
            "split_sha256": _split_hash(train_idx, val_idx),
            "input_shape": [3, 32, 32],
            "normalization_mean": list(mean),
            "normalization_std": list(std),
        },
    )


def _mnist_paketi(
    *,
    veri_koku: Path,
    split_dosyasi: Path,
    stats_dosyasi: Path,
    batch_size: int,
    split_seed: int,
    loader_seed: int,
    num_workers: int,
) -> AnaVeriPaketi:
    mnist_koku = veri_koku / "mnist"
    mnist_koku.mkdir(parents=True, exist_ok=True)

    indeks_taban = datasets.MNIST(
        root=str(mnist_koku),
        train=True,
        download=True,
        transform=None,
    )
    etiketler = np.asarray(indeks_taban.targets)
    train_idx, val_idx = _stratified_train_val_split(
        etiketler=etiketler,
        split_dosyasi=split_dosyasi,
        split_seed=split_seed,
        train_sayisi=54000,
        val_sayisi=6000,
        dataset_adi="mnist",
    )

    temel_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
        ]
    )
    stats_taban = datasets.MNIST(
        root=str(mnist_koku),
        train=True,
        download=False,
        transform=temel_transform,
    )
    fingerprint = hashlib.sha256(b"torchvision-mnist-official-train-v1").hexdigest()
    mean, std = _stats_al_veya_hesapla(
        dataset_adi="mnist",
        stats_dosyasi=stats_dosyasi,
        split_seed=split_seed,
        train_idx=train_idx,
        stats_dataset=stats_taban,
        dataset_fingerprint=fingerprint,
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_taban = datasets.MNIST(
        root=str(mnist_koku), train=True, download=False, transform=train_transform
    )
    val_taban = datasets.MNIST(
        root=str(mnist_koku), train=True, download=False, transform=eval_transform
    )
    test_taban = datasets.MNIST(
        root=str(mnist_koku), train=False, download=True, transform=eval_transform
    )

    generator = dataloader_generator(loader_seed)
    return AnaVeriPaketi(
        train_loader=_data_loader(
            train_taban,
            train_idx,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        ),
        validation_loader=_data_loader(
            val_taban,
            val_idx,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        test_loader=_data_loader(
            test_taban,
            None,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        sinif_sayisi=10,
        sinif_adlari=[str(i) for i in range(10)],
        meta={
            "dataset": "mnist",
            "train_count": len(train_idx),
            "validation_count": len(val_idx),
            "test_count": len(test_taban),
            "split_seed": split_seed,
            "split_sha256": _indeks_hash(train_idx, val_idx),
            "input_shape": [3, 32, 32],
            "normalization_mean": list(mean),
            "normalization_std": list(std),
        },
    )


def _imagefolder_paketi(
    *,
    dataset_adi: str,
    beklenen_sinif_sayisi: int,
    imagefolder_koku: Path,
    split_dosyasi: Path,
    stats_dosyasi: Path,
    batch_size: int,
    split_seed: int,
    loader_seed: int,
    num_workers: int,
) -> AnaVeriPaketi:
    if not imagefolder_koku.is_dir():
        raise FileNotFoundError(
            f"{dataset_adi} ImageFolder kökü bulunamadı: {imagefolder_koku}"
        )

    indeks_taban = datasets.ImageFolder(root=str(imagefolder_koku), transform=None)
    if len(indeks_taban.classes) != beklenen_sinif_sayisi:
        raise ValueError(
            f"{dataset_adi}: {beklenen_sinif_sayisi} sınıf bekleniyordu, "
            f"{len(indeks_taban.classes)} bulundu: {indeks_taban.classes}"
        )

    train_idx, val_idx, test_idx, fingerprint = _imagefolder_split(
        taban=indeks_taban,
        kok=imagefolder_koku,
        split_dosyasi=split_dosyasi,
        split_seed=split_seed,
        dataset_adi=dataset_adi,
    )

    temel_transform = transforms.Compose(
        [transforms.Resize((32, 32)), transforms.ToTensor()]
    )
    stats_taban = datasets.ImageFolder(
        root=str(imagefolder_koku),
        transform=temel_transform,
    )
    mean, std = _stats_al_veya_hesapla(
        dataset_adi=dataset_adi,
        stats_dosyasi=stats_dosyasi,
        split_seed=split_seed,
        train_idx=train_idx,
        stats_dataset=stats_taban,
        dataset_fingerprint=fingerprint,
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    train_taban = datasets.ImageFolder(
        root=str(imagefolder_koku), transform=train_transform
    )
    eval_taban = datasets.ImageFolder(
        root=str(imagefolder_koku), transform=eval_transform
    )

    generator = dataloader_generator(loader_seed)
    return AnaVeriPaketi(
        train_loader=_data_loader(
            train_taban,
            train_idx,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=generator,
        ),
        validation_loader=_data_loader(
            eval_taban,
            val_idx,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        test_loader=_data_loader(
            eval_taban,
            test_idx,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        sinif_sayisi=beklenen_sinif_sayisi,
        sinif_adlari=list(indeks_taban.classes),
        meta={
            "dataset": dataset_adi,
            "train_count": len(train_idx),
            "validation_count": len(val_idx),
            "test_count": len(test_idx),
            "total_count": len(indeks_taban),
            "split_seed": split_seed,
            "dataset_fingerprint": fingerprint,
            "split_sha256": _indeks_hash(train_idx, val_idx, test_idx),
            "input_shape": [3, 32, 32],
            "class_names": list(indeks_taban.classes),
            "normalization_mean": list(mean),
            "normalization_std": list(std),
        },
    )


def ana_veri_yukleyicileri(
    *,
    veri_seti: str,
    veri_klasoru: str,
    split_dosyasi: str,
    stats_dosyasi: str,
    batch_size: int,
    split_seed: int,
    loader_seed: int,
    num_workers: int,
) -> AnaVeriPaketi:
    """İstenen veri seti için train/validation/test yükleyicilerini döndürür."""
    veri_seti = veri_seti.lower().strip()
    veri_koku = Path(os.environ.get("TEZ_VERI_KLASORU", veri_klasoru))
    split = Path(os.environ.get("TEZ_SPLIT_DOSYASI", split_dosyasi))
    stats = Path(os.environ.get("TEZ_STATS_DOSYASI", stats_dosyasi))

    if veri_seti == "cifar10":
        return _cifar10_paketi(
            veri_koku=veri_koku,
            split_dosyasi=split,
            stats_dosyasi=stats,
            batch_size=batch_size,
            split_seed=split_seed,
            loader_seed=loader_seed,
            num_workers=num_workers,
        )

    if veri_seti == "mnist":
        return _mnist_paketi(
            veri_koku=veri_koku,
            split_dosyasi=split,
            stats_dosyasi=stats,
            batch_size=batch_size,
            split_seed=split_seed,
            loader_seed=loader_seed,
            num_workers=num_workers,
        )

    if veri_seti in {"animals10", "flowers5"}:
        env_kok = os.environ.get("TEZ_IMAGEFOLDER_KOKU")
        if not env_kok:
            raise RuntimeError(
                f"{veri_seti} için TEZ_IMAGEFOLDER_KOKU ortam değişkeni gerekli."
            )
        return _imagefolder_paketi(
            dataset_adi=veri_seti,
            beklenen_sinif_sayisi=10 if veri_seti == "animals10" else 5,
            imagefolder_koku=Path(env_kok),
            split_dosyasi=split,
            stats_dosyasi=stats,
            batch_size=batch_size,
            split_seed=split_seed,
            loader_seed=loader_seed,
            num_workers=num_workers,
        )

    raise ValueError(
        "Bilinmeyen veri_seti. Geçerli seçenekler: "
        "cifar10, mnist, animals10, flowers5"
    )
