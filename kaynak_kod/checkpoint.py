"""Uzun deney koşularını güvenli biçimde durdurup devam ettirme yardımcıları."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def checkpoint_kaydet(
    yol: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    kayitlar: list[dict[str, Any]],
    train_loader,
    run_id: str,
) -> None:
    """Model/optimizer ve rastgelelik durumunu atomik biçimde kaydeder."""
    hedef = Path(yol)
    hedef.parent.mkdir(parents=True, exist_ok=True)
    gecici = hedef.with_suffix(hedef.suffix + ".tmp")

    loader_generator = getattr(train_loader, "generator", None)

    durum = {
        "run_id": run_id,
        "epoch": int(epoch),
        "kayitlar": kayitlar,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "loader_generator_state": (
            loader_generator.get_state() if loader_generator is not None else None
        ),
    }

    torch.save(durum, gecici)
    os.replace(gecici, hedef)


def checkpoint_yukle(
    yol: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader,
    cihaz: torch.device,
    beklenen_run_id: str,
) -> tuple[int, list[dict[str, Any]]]:
    """Checkpoint'i yükler ve sonraki epoch numarası ile geçmiş kayıtları döndürür."""
    hedef = Path(yol)
    if not hedef.exists():
        return 1, []

    durum = torch.load(hedef, map_location=cihaz, weights_only=False)

    if durum.get("run_id") != beklenen_run_id:
        raise ValueError(
            "Checkpoint başka bir koşuya ait: "
            f"{durum.get('run_id')} != {beklenen_run_id}"
        )

    model.load_state_dict(durum["model_state_dict"])
    optimizer.load_state_dict(durum["optimizer_state_dict"])

    random.setstate(durum["python_random_state"])
    np.random.set_state(durum["numpy_random_state"])
    # map_location=cihaz, checkpoint içindeki RNG ByteTensor'larını da
    # CUDA'ya taşıyabilir. CPU RNG ve DataLoader generator durumları ise
    # CPU ByteTensor bekler. Bu nedenle açıkça CPU'ya geri alınır.
    torch_rng_state = durum["torch_rng_state"].detach().cpu()
    torch.set_rng_state(torch_rng_state)

    if torch.cuda.is_available() and durum.get("cuda_rng_state_all") is not None:
        cuda_rng_state_all = [
            rng_durumu.detach().cpu()
            for rng_durumu in durum["cuda_rng_state_all"]
        ]
        torch.cuda.set_rng_state_all(cuda_rng_state_all)

    loader_generator = getattr(train_loader, "generator", None)
    if loader_generator is not None and durum.get("loader_generator_state") is not None:
        loader_generator_state = durum["loader_generator_state"].detach().cpu()
        loader_generator.set_state(loader_generator_state)

    tamamlanan_epoch = int(durum["epoch"])
    kayitlar = list(durum.get("kayitlar", []))
    return tamamlanan_epoch + 1, kayitlar
