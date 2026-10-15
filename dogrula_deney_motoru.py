"""Aşama 1 deney motoru için checkpoint/resume ve tekrar üretilebilirlik doğrulaması.

Bu araç tez performans sonucu üretmez. Aynı üç epochluk eğitimi:
1) kesintisiz,
2) 1. epoch sonunda checkpoint alıp yeni model/loader oluşturarak devam eden
iki akışla çalıştırır ve sonuçların eşleşmesini denetler.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any

import torch
from torch import nn

from kaynak_kod.checkpoint import checkpoint_kaydet, checkpoint_yukle
from kaynak_kod.cnn_modeli import CNNv1
from kaynak_kod.egitim import bir_epoch_egit, dogrula
from kaynak_kod.kayit_sistemi import json_kaydet, klasor_hazirla
from kaynak_kod.optimizasyon_algoritmalari import optimizasyon_olustur
from kaynak_kod.tekrarlanabilirlik import sabit_seed_ayarla
from kaynak_kod.veri_hazirlama import cifar10_yukleyicileri


def _model_sha256(model: nn.Module) -> str:
    ozet = hashlib.sha256()
    for ad, tensor in sorted(model.state_dict().items()):
        ozet.update(ad.encode("utf-8"))
        ozet.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return ozet.hexdigest()


def _model_max_fark(a: nn.Module, b: nn.Module) -> float:
    en_buyuk = 0.0
    durum_a = a.state_dict()
    durum_b = b.state_dict()

    if durum_a.keys() != durum_b.keys():
        raise ValueError("Karşılaştırılan modellerin state_dict anahtarları farklı.")

    for anahtar in durum_a:
        ta = durum_a[anahtar].detach().cpu()
        tb = durum_b[anahtar].detach().cpu()
        if ta.shape != tb.shape:
            raise ValueError(f"Parametre şekli farklı: {anahtar}")
        if ta.numel():
            fark = float((ta - tb).abs().max().item())
            en_buyuk = max(en_buyuk, fark)

    return en_buyuk


def _metrik_satiri(epoch: int, train: dict, val: dict) -> dict[str, float | int]:
    return {
        "epoch": epoch,
        "train_loss": float(train["loss"]),
        "train_accuracy": float(train["accuracy"]),
        "validation_loss": float(val["loss"]),
        "validation_accuracy": float(val["accuracy"]),
        "train_seconds": float(train["seconds"]),
        "validation_seconds": float(val["seconds"]),
    }


def _metrik_farklari(
    temel: list[dict[str, Any]],
    devam: list[dict[str, Any]],
) -> dict[str, float]:
    alanlar = (
        "train_loss",
        "train_accuracy",
        "validation_loss",
        "validation_accuracy",
    )
    farklar = {alan: 0.0 for alan in alanlar}

    if len(temel) != len(devam):
        raise ValueError("Epoch kayıt sayıları eşleşmiyor.")

    for a, b in zip(temel, devam):
        if int(a["epoch"]) != int(b["epoch"]):
            raise ValueError("Epoch numaraları eşleşmiyor.")
        for alan in alanlar:
            farklar[alan] = max(
                farklar[alan],
                abs(float(a[alan]) - float(b[alan])),
            )

    return farklar


def _yukleyiciler(
    *,
    veri_klasoru: str,
    split_dosyasi: str,
    stats_dosyasi: str,
    batch_size: int,
    split_seed: int,
    seed: int,
    num_workers: int,
):
    return cifar10_yukleyicileri(
        veri_klasoru=veri_klasoru,
        split_dosyasi=split_dosyasi,
        stats_dosyasi=stats_dosyasi,
        batch_size=batch_size,
        split_seed=split_seed,
        loader_seed=seed,
        num_workers=num_workers,
    )


def _model_optimizer(cihaz: torch.device):
    model = CNNv1("relu").to(cihaz)
    optimizer = optimizasyon_olustur(
        "sgd",
        model.parameters(),
        learning_rate=0.01,
        weight_decay=0.0,
    )
    return model, optimizer


def ana(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--veri-klasoru", required=True)
    parser.add_argument("--split-dosyasi", required=True)
    parser.add_argument("--stats-dosyasi", required=True)
    parser.add_argument("--cikti-klasoru", required=True)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--kesinti-epoch", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerans", type=float, default=1e-7)
    args = parser.parse_args(argv)

    if args.epoch < 2:
        raise ValueError("Doğrulama için en az 2 epoch gerekir.")
    if not 1 <= args.kesinti_epoch < args.epoch:
        raise ValueError("kesinti-epoch, 1 ile toplam epoch-1 arasında olmalıdır.")

    if not torch.cuda.is_available():
        raise RuntimeError("Bu doğrulama Colab T4/CUDA üzerinde çalıştırılmalıdır.")

    cihaz = torch.device("cuda")
    cikti = klasor_hazirla(args.cikti_klasoru)
    checkpoint_yolu = Path(cikti) / "kesinti_checkpoint.pt"
    sonuc_yolu = Path(cikti) / "dogrulama_sonucu.json"

    os.environ["TEZ_VERI_KLASORU"] = args.veri_klasoru
    os.environ["TEZ_SPLIT_DOSYASI"] = args.split_dosyasi
    os.environ["TEZ_STATS_DOSYASI"] = args.stats_dosyasi

    print("=" * 68, flush=True)
    print("AŞAMA 1 — DENEY MOTORU CHECKPOINT/RESUME DOĞRULAMASI", flush=True)
    print("=" * 68, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Veri:", args.veri_klasoru, flush=True)
    print("Split:", args.split_dosyasi, flush=True)
    print("Stats:", args.stats_dosyasi, flush=True)

    kayip = nn.CrossEntropyLoss()

    # A) Kesintisiz referans
    print("\n[A] Kesintisiz referans çalışması başlıyor...", flush=True)
    sabit_seed_ayarla(args.seed)
    train_loader_a, val_loader_a = _yukleyiciler(
        veri_klasoru=args.veri_klasoru,
        split_dosyasi=args.split_dosyasi,
        stats_dosyasi=args.stats_dosyasi,
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    model_a, optimizer_a = _model_optimizer(cihaz)
    kayit_a: list[dict[str, Any]] = []

    for epoch in range(1, args.epoch + 1):
        train = bir_epoch_egit(model_a, train_loader_a, kayip, optimizer_a, cihaz)
        val = dogrula(model_a, val_loader_a, kayip, cihaz)
        satir = _metrik_satiri(epoch, train, val)
        kayit_a.append(satir)
        print(
            f"[A] {epoch}/{args.epoch} | "
            f"train_acc={satir['train_accuracy']:.6f} | "
            f"val_acc={satir['validation_accuracy']:.6f}",
            flush=True,
        )

    hash_a = _model_sha256(model_a)

    # B1) Kesintiye kadar
    print("\n[B1] Kesintili çalışma başlangıcı...", flush=True)
    sabit_seed_ayarla(args.seed)
    train_loader_b1, val_loader_b1 = _yukleyiciler(
        veri_klasoru=args.veri_klasoru,
        split_dosyasi=args.split_dosyasi,
        stats_dosyasi=args.stats_dosyasi,
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    model_b1, optimizer_b1 = _model_optimizer(cihaz)
    kayit_b: list[dict[str, Any]] = []

    for epoch in range(1, args.kesinti_epoch + 1):
        train = bir_epoch_egit(model_b1, train_loader_b1, kayip, optimizer_b1, cihaz)
        val = dogrula(model_b1, val_loader_b1, kayip, cihaz)
        satir = _metrik_satiri(epoch, train, val)
        kayit_b.append(satir)
        checkpoint_kaydet(
            checkpoint_yolu,
            model=model_b1,
            optimizer=optimizer_b1,
            epoch=epoch,
            kayitlar=kayit_b,
            train_loader=train_loader_b1,
            run_id="ASAMA1-RESUME-TEST",
        )
        print(
            f"[B1] {epoch}/{args.epoch} tamamlandı ve checkpoint yazıldı.",
            flush=True,
        )

    del model_b1, optimizer_b1, train_loader_b1, val_loader_b1
    gc.collect()
    torch.cuda.empty_cache()

    # B2) Yeni nesneler + checkpointten devam
    print("\n[B2] Yeni model/loader oluşturuluyor ve checkpoint yükleniyor...", flush=True)
    sabit_seed_ayarla(args.seed)
    train_loader_b2, val_loader_b2 = _yukleyiciler(
        veri_klasoru=args.veri_klasoru,
        split_dosyasi=args.split_dosyasi,
        stats_dosyasi=args.stats_dosyasi,
        batch_size=args.batch_size,
        split_seed=args.split_seed,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    model_b2, optimizer_b2 = _model_optimizer(cihaz)

    baslangic_epoch, kayit_b = checkpoint_yukle(
        checkpoint_yolu,
        model=model_b2,
        optimizer=optimizer_b2,
        train_loader=train_loader_b2,
        cihaz=cihaz,
        beklenen_run_id="ASAMA1-RESUME-TEST",
    )
    print(f"[B2] Epoch {baslangic_epoch}'den devam ediliyor.", flush=True)

    for epoch in range(baslangic_epoch, args.epoch + 1):
        train = bir_epoch_egit(model_b2, train_loader_b2, kayip, optimizer_b2, cihaz)
        val = dogrula(model_b2, val_loader_b2, kayip, cihaz)
        satir = _metrik_satiri(epoch, train, val)
        kayit_b.append(satir)
        checkpoint_kaydet(
            checkpoint_yolu,
            model=model_b2,
            optimizer=optimizer_b2,
            epoch=epoch,
            kayitlar=kayit_b,
            train_loader=train_loader_b2,
            run_id="ASAMA1-RESUME-TEST",
        )
        print(
            f"[B2] {epoch}/{args.epoch} | "
            f"train_acc={satir['train_accuracy']:.6f} | "
            f"val_acc={satir['validation_accuracy']:.6f}",
            flush=True,
        )

    hash_b = _model_sha256(model_b2)
    max_parametre_farki = _model_max_fark(model_a, model_b2)
    metrik_farklari = _metrik_farklari(kayit_a, kayit_b)

    split_veri = json.loads(Path(args.split_dosyasi).read_text(encoding="utf-8"))
    stats_veri = json.loads(Path(args.stats_dosyasi).read_text(encoding="utf-8"))

    gecti = (
        max_parametre_farki <= args.tolerans
        and all(v <= args.tolerans for v in metrik_farklari.values())
        and len(kayit_a) == args.epoch
        and len(kayit_b) == args.epoch
    )

    sonuc = {
        "amac": "Checkpoint/resume ve tekrarlanabilirlik teknik doğrulaması",
        "bilimsel_sonuc_degil": True,
        "gecti": gecti,
        "tolerans": args.tolerans,
        "ortam": {
            "git_commit": os.environ.get("TEZ_GIT_COMMIT", "unknown"),
            "gpu": torch.cuda.get_device_name(0),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
            "split_seed": args.split_seed,
        },
        "veri_dogrulama": {
            "train_count": split_veri["train_count"],
            "validation_count": split_veri["validation_count"],
            "split_sha256": split_veri["split_sha256"],
            "class_distribution": split_veri["class_distribution"],
            "normalization_mean": stats_veri["mean"],
            "normalization_std": stats_veri["std"],
            "normalization_source": stats_veri["source"],
        },
        "kesintisiz": {
            "model_sha256": hash_a,
            "epoch_kayitlari": kayit_a,
        },
        "checkpoint_resume": {
            "model_sha256": hash_b,
            "epoch_kayitlari": kayit_b,
        },
        "karsilastirma": {
            "model_hash_esit": hash_a == hash_b,
            "max_parametre_mutlak_farki": max_parametre_farki,
            "maksimum_metrik_farklari": metrik_farklari,
        },
    }

    json_kaydet(sonuc_yolu, sonuc)

    print("\n" + "=" * 68, flush=True)
    print("DOĞRULAMA SONUCU:", "GEÇTİ" if gecti else "BAŞARISIZ", flush=True)
    print("Model hash eşit:", hash_a == hash_b, flush=True)
    print("Maksimum parametre farkı:", max_parametre_farki, flush=True)
    print("Maksimum metrik farkları:", metrik_farklari, flush=True)
    print("Sonuç dosyası:", sonuc_yolu, flush=True)

    if not gecti:
        raise RuntimeError(
            "Checkpoint/resume doğrulaması tolerans içinde eşleşmedi. "
            "LR pilotuna geçilmemelidir."
        )


if __name__ == "__main__":
    ana()
