"""CIFAR-10 veri hazırlama, sabit split ve eğitim-istatistiği yönetimi."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .tekrarlanabilirlik import dataloader_generator


def _veri_kokunu_coz(veri_klasoru: str) -> tuple[Path, bool]:
    """Veri kökünü ve bunun harici/kalıcı bir kök olup olmadığını döndürür."""
    ortam_yolu = os.environ.get("TEZ_VERI_KLASORU")
    if ortam_yolu:
        return Path(ortam_yolu), True
    return Path(veri_klasoru), False


def _cifar10_drive_verisini_hazirla(veri_koku: Path) -> None:
    """Drive'daki standart CIFAR-10 arşivini/klasörünü kullanılabilir hale getirir."""
    cikartilmis = veri_koku / "cifar-10-batches-py"
    if cikartilmis.is_dir():
        return

    arsiv = veri_koku / "cifar-10-python.tar.gz"
    if arsiv.is_file():
        print(
            f"CIFAR-10 arşivi bulundu, Drive üzerinde bir kez açılıyor: {arsiv}",
            flush=True,
        )
        with tarfile.open(arsiv, "r:gz") as tar:
            tar.extractall(veri_koku)
        if cikartilmis.is_dir():
            return

    raise FileNotFoundError(
        "CIFAR-10 Google Drive üzerinde bulunamadı. Beklenen konum: "
        f"{cikartilmis} veya {arsiv}."
    )


def _split_indeksleri(
    etiketler: np.ndarray,
    split_dosyasi: Path,
    split_seed: int,
) -> tuple[list[int], list[int]]:
    """45.000/5.000 stratified ayrımı üretir, doğrular ve kaydeder."""
    if split_dosyasi.exists():
        veri = json.loads(split_dosyasi.read_text(encoding="utf-8"))
        if int(veri.get("split_seed", split_seed)) != split_seed:
            raise ValueError(
                f"Split seed uyuşmuyor: dosya={veri.get('split_seed')} "
                f"beklenen={split_seed}"
            )
        train_idx = [int(x) for x in veri["train_indices"]]
        val_idx = [int(x) for x in veri["validation_indices"]]
    else:
        tum_indeksler = np.arange(len(etiketler))
        train_idx_np, val_idx_np = train_test_split(
            tum_indeksler,
            test_size=5000,
            random_state=split_seed,
            stratify=etiketler,
            shuffle=True,
        )
        train_idx = train_idx_np.tolist()
        val_idx = val_idx_np.tolist()

    _split_dogrula(train_idx, val_idx, len(etiketler))

    split_hash = _split_hash(train_idx, val_idx)
    sinif_dagilimi = {
        "train": np.bincount(etiketler[train_idx], minlength=10).tolist(),
        "validation": np.bincount(etiketler[val_idx], minlength=10).tolist(),
    }

    split_dosyasi.parent.mkdir(parents=True, exist_ok=True)
    split_dosyasi.write_text(
        json.dumps(
            {
                "split_seed": split_seed,
                "split_sha256": split_hash,
                "train_count": len(train_idx),
                "validation_count": len(val_idx),
                "class_distribution": sinif_dagilimi,
                "train_indices": train_idx,
                "validation_indices": val_idx,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return train_idx, val_idx


def _split_dogrula(
    train_idx: list[int],
    val_idx: list[int],
    toplam_ornek: int,
) -> None:
    if len(train_idx) != 45000 or len(val_idx) != 5000:
        raise ValueError(
            f"Beklenmeyen split boyutu: train={len(train_idx)}, val={len(val_idx)}"
        )

    train_set = set(train_idx)
    val_set = set(val_idx)

    if len(train_set) != len(train_idx) or len(val_set) != len(val_idx):
        raise ValueError("Split içinde tekrarlanan indeks bulundu.")

    if train_set.intersection(val_set):
        raise ValueError("Train ve validation splitleri çakışıyor.")

    birlesim = train_set.union(val_set)
    if len(birlesim) != toplam_ornek or min(birlesim) != 0 or max(birlesim) != toplam_ornek - 1:
        raise ValueError("Split, CIFAR-10 eğitim setinin tamamını tam olarak kapsamıyor.")


def _split_hash(train_idx: list[int], val_idx: list[int]) -> str:
    ham = json.dumps(
        {"train_indices": train_idx, "validation_indices": val_idx},
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(ham).hexdigest()


def _egitim_istatistiklerini_hesapla(
    veri_koku: Path,
    train_idx: list[int],
    batch_size: int = 512,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Normalize değerlerini yalnız 45k eğitim alt kümesinden hesaplar."""
    taban = datasets.CIFAR10(
        root=str(veri_koku),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )
    alt_kume = Subset(taban, train_idx)
    loader = DataLoader(
        alt_kume,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    kanal_toplami = torch.zeros(3, dtype=torch.float64)
    kanal_kare_toplami = torch.zeros(3, dtype=torch.float64)
    piksel_sayisi = 0

    for goruntuler, _ in loader:
        goruntuler = goruntuler.to(dtype=torch.float64)
        kanal_toplami += goruntuler.sum(dim=(0, 2, 3))
        kanal_kare_toplami += (goruntuler * goruntuler).sum(dim=(0, 2, 3))
        piksel_sayisi += goruntuler.shape[0] * goruntuler.shape[2] * goruntuler.shape[3]

    ortalama = kanal_toplami / piksel_sayisi
    varyans = kanal_kare_toplami / piksel_sayisi - ortalama * ortalama
    std = torch.sqrt(torch.clamp(varyans, min=0.0))

    return tuple(float(x) for x in ortalama), tuple(float(x) for x in std)


def _istatistikleri_al(
    veri_koku: Path,
    train_idx: list[int],
    split_seed: int,
    stats_dosyasi: Path,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    split_hash = _split_hash(train_idx, [])

    if stats_dosyasi.exists():
        veri = json.loads(stats_dosyasi.read_text(encoding="utf-8"))
        if int(veri["split_seed"]) != split_seed:
            raise ValueError("Normalizasyon istatistiğinin split seed değeri uyuşmuyor.")
        if veri.get("train_index_sha256") != split_hash:
            raise ValueError(
                "Normalizasyon istatistiği mevcut eğitim splitine ait değil."
            )
        mean = tuple(float(x) for x in veri["mean"])
        std = tuple(float(x) for x in veri["std"])
        if len(mean) != 3 or len(std) != 3 or any(x <= 0 for x in std):
            raise ValueError("Normalizasyon istatistik dosyası geçersiz.")
        return mean, std

    print(
        "CIFAR-10 normalizasyon değerleri yalnız 45.000 eğitim örneğinden hesaplanıyor...",
        flush=True,
    )
    mean, std = _egitim_istatistiklerini_hesapla(veri_koku, train_idx)

    stats_dosyasi.parent.mkdir(parents=True, exist_ok=True)
    stats_dosyasi.write_text(
        json.dumps(
            {
                "dataset": "CIFAR-10",
                "source": "45k training subset only",
                "split_seed": split_seed,
                "train_index_sha256": split_hash,
                "mean": mean,
                "std": std,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Normalizasyon kaydedildi: mean={mean}, std={std}", flush=True)
    return mean, std


def _donusumler(
    mean: tuple[float, ...],
    std: tuple[float, ...],
):
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    degerlendirme_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, degerlendirme_transform


def cifar10_yukleyicileri(
    veri_klasoru: str,
    split_dosyasi: str,
    batch_size: int,
    split_seed: int,
    loader_seed: int,
    num_workers: int = 0,
    stats_dosyasi: str = "ayarlar/cifar10_stats.json",
):
    """Train ve validation DataLoader nesnelerini oluşturur.

    Test seti pilot/duman testi sırasında bilerek döndürülmez.
    Normalizasyon değerleri yalnızca sabit 45k eğitim alt kümesinden hesaplanır.
    """
    veri_koku, harici_kok = _veri_kokunu_coz(veri_klasoru)
    veri_koku.mkdir(parents=True, exist_ok=True)

    if harici_kok:
        _cifar10_drive_verisini_hazirla(veri_koku)

    indeks_taban = datasets.CIFAR10(
        root=str(veri_koku),
        train=True,
        download=not harici_kok,
        transform=None,
    )
    etiketler = np.asarray(indeks_taban.targets)

    etkin_split_dosyasi = Path(
        os.environ.get("TEZ_SPLIT_DOSYASI", split_dosyasi)
    )
    etkin_stats_dosyasi = Path(
        os.environ.get("TEZ_STATS_DOSYASI", stats_dosyasi)
    )

    train_idx, val_idx = _split_indeksleri(
        etiketler,
        etkin_split_dosyasi,
        split_seed,
    )

    mean, std = _istatistikleri_al(
        veri_koku,
        train_idx,
        split_seed,
        etkin_stats_dosyasi,
    )
    train_transform, val_transform = _donusumler(mean, std)

    train_taban = datasets.CIFAR10(
        root=str(veri_koku),
        train=True,
        download=False,
        transform=train_transform,
    )
    val_taban = datasets.CIFAR10(
        root=str(veri_koku),
        train=True,
        download=False,
        transform=val_transform,
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


def cifar10_probe_batch(
    veri_klasoru: str,
    split_dosyasi: str,
    stats_dosyasi: str,
    split_seed: int,
    probe_boyutu: int = 128,
):
    """Yalnız eğitim alt kümesinden sabit ve augmentation'sız probe batch üretir."""
    if probe_boyutu <= 0:
        raise ValueError("probe_boyutu pozitif olmalıdır.")

    veri_koku, harici_kok = _veri_kokunu_coz(veri_klasoru)
    veri_koku.mkdir(parents=True, exist_ok=True)

    if harici_kok:
        _cifar10_drive_verisini_hazirla(veri_koku)

    indeks_taban = datasets.CIFAR10(
        root=str(veri_koku),
        train=True,
        download=not harici_kok,
        transform=None,
    )
    etiketler = np.asarray(indeks_taban.targets)

    etkin_split_dosyasi = Path(
        os.environ.get("TEZ_SPLIT_DOSYASI", split_dosyasi)
    )
    etkin_stats_dosyasi = Path(
        os.environ.get("TEZ_STATS_DOSYASI", stats_dosyasi)
    )

    train_idx, _ = _split_indeksleri(
        etiketler,
        etkin_split_dosyasi,
        split_seed,
    )
    mean, std = _istatistikleri_al(
        veri_koku,
        train_idx,
        split_seed,
        etkin_stats_dosyasi,
    )

    _, probe_transform = _donusumler(mean, std)
    probe_taban = datasets.CIFAR10(
        root=str(veri_koku),
        train=True,
        download=False,
        transform=probe_transform,
    )

    # Split dosyasındaki ilk örnekler sabittir; augmentation uygulanmaz.
    secilen = train_idx[: min(probe_boyutu, len(train_idx))]
    probe_set = Subset(probe_taban, secilen)
    probe_loader = DataLoader(
        probe_set,
        batch_size=len(secilen),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    return next(iter(probe_loader))
