"""Aşama 1 ikinci paket: metrik, yakınsama ve gradyan probe doğrulaması.

Bu araç bilimsel tez sonucu üretmez. Amaç, tez önerisinde belirtilen
precision/recall/F1, yakınsama ve teorik gradyan mekanizması ölçümlerinin
ana deneylerden önce teknik olarak doğrulanmasıdır.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kaynak_kod.cnn_modeli import CNNv1
from kaynak_kod.egitim import bir_epoch_egit, dogrula
from kaynak_kod.gradyan_analizi import (
    gradyan_probe,
    parametre_guncelleme_probe,
)
from kaynak_kod.kayit_sistemi import json_kaydet, klasor_hazirla
from kaynak_kod.optimizasyon_algoritmalari import optimizasyon_olustur
from kaynak_kod.tekrarlanabilirlik import sabit_seed_ayarla
from kaynak_kod.veri_hazirlama import (
    cifar10_probe_batch,
    cifar10_yukleyicileri,
)
from kaynak_kod.yakinsama_analizi import yakinsama_ozeti


def _model_sha256(model: nn.Module) -> str:
    ozet = hashlib.sha256()
    for ad, tensor in sorted(model.state_dict().items()):
        ozet.update(ad.encode("utf-8"))
        ozet.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return ozet.hexdigest()


def _sonlu(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def ana(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--veri-klasoru", required=True)
    parser.add_argument("--split-dosyasi", required=True)
    parser.add_argument("--stats-dosyasi", required=True)
    parser.add_argument("--cikti-klasoru", required=True)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--probe-boyutu", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        raise RuntimeError("Bu doğrulama Colab T4/CUDA üzerinde çalıştırılmalıdır.")

    cihaz = torch.device("cuda")
    cikti = klasor_hazirla(args.cikti_klasoru)
    sonuc_yolu = Path(cikti) / "metrik_yakinsama_gradyan_dogrulama.json"

    os.environ["TEZ_VERI_KLASORU"] = args.veri_klasoru
    os.environ["TEZ_SPLIT_DOSYASI"] = args.split_dosyasi
    os.environ["TEZ_STATS_DOSYASI"] = args.stats_dosyasi

    print("=" * 72, flush=True)
    print("AŞAMA 1 — METRİK / YAKINSAMA / GRADYAN DOĞRULAMASI", flush=True)
    print("=" * 72, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    sabit_seed_ayarla(args.seed)

    train_loader, val_loader = cifar10_yukleyicileri(
        veri_klasoru=args.veri_klasoru,
        split_dosyasi=args.split_dosyasi,
        stats_dosyasi=args.stats_dosyasi,
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        loader_seed=args.seed,
        num_workers=args.num_workers,
    )

    probe_batch = cifar10_probe_batch(
        veri_klasoru=args.veri_klasoru,
        split_dosyasi=args.split_dosyasi,
        stats_dosyasi=args.stats_dosyasi,
        split_seed=args.split_seed,
        probe_boyutu=args.probe_boyutu,
    )

    model = CNNv1("relu").to(cihaz)
    optimizer = optimizasyon_olustur(
        "sgd",
        model.parameters(),
        learning_rate=0.01,
        weight_decay=0.0,
    )
    kayip_fonksiyonu = nn.CrossEntropyLoss()

    epoch_kayitlari: list[dict[str, Any]] = []
    probe_kayitlari: list[dict[str, Any]] = []

    for epoch in range(1, args.epoch + 1):
        train = bir_epoch_egit(
            model,
            train_loader,
            kayip_fonksiyonu,
            optimizer,
            cihaz,
        )
        val = dogrula(
            model,
            val_loader,
            kayip_fonksiyonu,
            cihaz,
        )

        satir = {
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "validation_loss": val["loss"],
            "validation_accuracy": val["accuracy"],
            "macro_precision": val["macro_precision"],
            "macro_recall": val["macro_recall"],
            "macro_f1": val["macro_f1"],
            "train_seconds": train["seconds"],
            "validation_seconds": val["seconds"],
        }
        epoch_kayitlari.append(satir)

        print(
            f"{epoch}/{args.epoch} | "
            f"val_acc={val['accuracy']:.6f} | "
            f"macro_f1={val['macro_f1']:.6f}",
            flush=True,
        )

        if epoch in {1, args.epoch}:
            hash_once = _model_sha256(model)
            probe = gradyan_probe(
                model,
                kayip_fonksiyonu,
                probe_batch,
                cihaz,
            )
            hash_sonra = _model_sha256(model)
            probe["epoch"] = epoch
            probe["model_unchanged"] = hash_once == hash_sonra
            probe_kayitlari.append(probe)

    validation_accuracy = [
        float(x["validation_accuracy"]) for x in epoch_kayitlari
    ]
    yakinsama = yakinsama_ozeti(validation_accuracy)

    hash_once = _model_sha256(model)
    update_probe = parametre_guncelleme_probe(
        model,
        optimizer,
        kayip_fonksiyonu,
        probe_batch,
        cihaz,
    )
    hash_sonra = _model_sha256(model)
    update_probe["original_model_unchanged"] = hash_once == hash_sonra

    kontroller = {
        "epoch_sayisi_dogru": len(epoch_kayitlari) == args.epoch,
        "macro_metrikler_0_1": all(
            0.0 <= float(satir[alan]) <= 1.0
            for satir in epoch_kayitlari
            for alan in ("macro_precision", "macro_recall", "macro_f1")
        ),
        "aulc_0_1": 0.0 <= float(yakinsama["aulc_normalized"]) <= 1.0,
        "epoch_to_95_gecerli": (
            1 <= int(yakinsama["epoch_to_95_own_best"]) <= args.epoch
        ),
        "probe_sayisi_dogru": len(probe_kayitlari) == 2,
        "probe_modeli_degistirmedi": all(
            bool(p["model_unchanged"]) for p in probe_kayitlari
        ),
        "global_gradyan_pozitif": all(
            _sonlu(p["global_gradient_l2"])
            and float(p["global_gradient_l2"]) > 0
            for p in probe_kayitlari
        ),
        "relu_aktivasyon_katmani_5": all(
            len(p["activations"]) == 5 for p in probe_kayitlari
        ),
        "aktivasyon_oranlari_gecerli": all(
            0.0 <= float(a["zero_fraction"]) <= 1.0
            and 0.0 <= float(a["near_zero_derivative_fraction"]) <= 1.0
            and (
                a["dead_unit_fraction"] is None
                or 0.0 <= float(a["dead_unit_fraction"]) <= 1.0
            )
            for p in probe_kayitlari
            for a in p["activations"]
        ),
        "update_probe_modeli_degistirmedi": bool(
            update_probe["original_model_unchanged"]
        ),
        "global_update_pozitif": (
            _sonlu(update_probe["global_update_l2"])
            and float(update_probe["global_update_l2"]) > 0
        ),
        "relative_update_gecerli": (
            _sonlu(update_probe["global_relative_update"])
            and float(update_probe["global_relative_update"]) >= 0
        ),
    }

    gecti = all(kontroller.values())

    sonuc = {
        "amac": "Metrik, yakınsama ve mekanizma ölçümleri teknik doğrulaması",
        "bilimsel_sonuc_degil": True,
        "gecti": gecti,
        "ortam": {
            "git_commit": os.environ.get("TEZ_GIT_COMMIT", "unknown"),
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "batch_size": args.batch_size,
            "probe_batch_size": args.probe_boyutu,
            "seed": args.seed,
            "split_seed": args.split_seed,
        },
        "epoch_kayitlari": epoch_kayitlari,
        "yakinsama": yakinsama,
        "gradyan_probe": probe_kayitlari,
        "parametre_guncelleme_probe": update_probe,
        "kontroller": kontroller,
    }

    json_kaydet(sonuc_yolu, sonuc)

    print("\nYakınsama özeti:", yakinsama, flush=True)
    print(
        "Final global gradient L2:",
        probe_kayitlari[-1]["global_gradient_l2"],
        flush=True,
    )
    print(
        "Global relative update:",
        update_probe["global_relative_update"],
        flush=True,
    )
    print("\nDOĞRULAMA:", "GEÇTİ" if gecti else "BAŞARISIZ", flush=True)
    print("Sonuç:", sonuc_yolu, flush=True)

    if not gecti:
        basarisiz = [k for k, v in kontroller.items() if not v]
        raise RuntimeError(
            "Aşama 1 ikinci paket doğrulaması başarısız: "
            + ", ".join(basarisiz)
        )


if __name__ == "__main__":
    ana()
