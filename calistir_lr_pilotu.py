"""Learning-rate pilotunu planlar ve isteğe bağlı olarak çalıştırır.

Bu pilotta:
- temsilci aktivasyonlar: ReLU, GELU, Mish
- optimizerlar: SGD, Momentum SGD, RMSProp, Adam, AdamW
- her optimizer için 4 LR adayı
- tek pilot seed: 42
- 30 epoch
- weight decay: 0

Toplam: 3 x 5 x 4 = 60 koşu.

Test seti kullanılmaz.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
import yaml
from torch import nn

from kaynak_kod.checkpoint import checkpoint_kaydet, checkpoint_yukle
from kaynak_kod.cnn_modeli import CNNv1
from kaynak_kod.egitim import bir_epoch_egit, dogrula
from kaynak_kod.kayit_sistemi import epoch_csv_kaydet, json_kaydet, klasor_hazirla
from kaynak_kod.optimizasyon_algoritmalari import optimizasyon_olustur
from kaynak_kod.tekrarlanabilirlik import sabit_seed_ayarla
from kaynak_kod.veri_hazirlama import cifar10_yukleyicileri


def run_id_olustur(optimizer: str, aktivasyon: str, lr: float, seed: int) -> str:
    lr_metin = f"{lr:.4g}".replace(".", "p")
    return f"LRP-{optimizer.upper()}-{aktivasyon.upper()}-LR{lr_metin}-S{seed}"


def deney_plani(ayarlar: dict) -> list[dict]:
    plan = []
    seed = int(ayarlar["egitim"]["seed"])

    for optimizer, lr_listesi in ayarlar["learning_rate_adaylari"].items():
        for aktivasyon in ayarlar["model"]["temsilci_aktivasyonlar"]:
            for lr in lr_listesi:
                plan.append(
                    {
                        "run_id": run_id_olustur(optimizer, aktivasyon, float(lr), seed),
                        "optimizer": optimizer,
                        "aktivasyon": aktivasyon,
                        "learning_rate": float(lr),
                        "seed": seed,
                    }
                )

    return plan


def tek_kosu(ayarlar: dict, kosu: dict, cihaz: torch.device) -> None:
    seed = kosu["seed"]
    sabit_seed_ayarla(seed)

    varsayilan_cikti = Path(ayarlar["cikti"]["klasor"])
    cikti_koku = Path(os.environ.get("TEZ_CIKTI_KLASORU", str(varsayilan_cikti)))
    cikti = cikti_koku / kosu["run_id"]
    tamamlandi = cikti / "tamamlandi.json"
    checkpoint_dosyasi = cikti / "checkpoint.pt"

    if tamamlandi.exists():
        print(f"[ATLA] {kosu['run_id']} daha önce tamamlanmış.")
        return

    klasor_hazirla(cikti)

    train_loader, val_loader = cifar10_yukleyicileri(
        veri_klasoru=ayarlar["veri"]["veri_klasoru"],
        split_dosyasi=ayarlar["veri"]["split_dosyasi"],
        batch_size=int(ayarlar["veri"]["batch_size"]),
        split_seed=int(ayarlar["veri"]["split_seed"]),
        loader_seed=seed,
        num_workers=int(ayarlar["veri"]["num_workers"]),
    )

    model = CNNv1(kosu["aktivasyon"]).to(cihaz)
    kayip_fonksiyonu = nn.CrossEntropyLoss()

    optimizer = optimizasyon_olustur(
        kosu["optimizer"],
        model.parameters(),
        learning_rate=kosu["learning_rate"],
        weight_decay=float(ayarlar["egitim"]["weight_decay"]),
    )

    baslangic_epoch, kayitlar = checkpoint_yukle(
        checkpoint_dosyasi,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        cihaz=cihaz,
        beklenen_run_id=kosu["run_id"],
    )

    if baslangic_epoch > 1:
        print(
            f"[DEVAM] {kosu['run_id']} checkpoint bulundu; "
            f"epoch {baslangic_epoch}'den devam ediliyor.",
            flush=True,
        )

    for epoch in range(baslangic_epoch, int(ayarlar["egitim"]["epoch"]) + 1):
        train = bir_epoch_egit(model, train_loader, kayip_fonksiyonu, optimizer, cihaz)
        val = dogrula(model, val_loader, kayip_fonksiyonu, cihaz)

        satir = {
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "validation_loss": val["loss"],
            "validation_accuracy": val["accuracy"],
            "learning_rate": kosu["learning_rate"],
            "train_seconds": train["seconds"],
            "validation_seconds": val["seconds"],
            "epoch_seconds": train["seconds"] + val["seconds"],
        }
        kayitlar.append(satir)
        epoch_csv_kaydet(cikti / "epoch_sonuclari.csv", kayitlar)

        checkpoint_kaydet(
            checkpoint_dosyasi,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            kayitlar=kayitlar,
            train_loader=train_loader,
            run_id=kosu["run_id"],
        )

        if not all(
            math.isfinite(float(satir[alan]))
            for alan in (
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
            )
        ):
            json_kaydet(
                cikti / "kararsiz.json",
                {**kosu, "epoch": epoch, "neden": "NaN veya Inf tespit edildi."},
            )
            print(f"[KARARSIZ] {kosu['run_id']} epoch={epoch}")
            return

        print(
            f"{kosu['run_id']} | {epoch:02d}/{ayarlar['egitim']['epoch']} | "
            f"train_acc={train['accuracy']:.4f} | val_acc={val['accuracy']:.4f}"
        )

    son = kayitlar[-1]
    json_kaydet(
        tamamlandi,
        {
            **kosu,
            "epoch": int(ayarlar["egitim"]["epoch"]),
            "final_train_loss": son["train_loss"],
            "final_train_accuracy": son["train_accuracy"],
            "final_validation_loss": son["validation_loss"],
            "final_validation_accuracy": son["validation_accuracy"],
            "cihaz": str(cihaz),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "pytorch_surumu": torch.__version__,
            "cuda_surumu": torch.version.cuda,
            "git_commit": os.environ.get("TEZ_GIT_COMMIT", "unknown"),
        },
    )

    if checkpoint_dosyasi.exists():
        checkpoint_dosyasi.unlink()


def ana() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calistir",
        action="store_true",
        help="Verilmezse yalnızca 60 koşuluk pilot planını gösterir.",
    )
    parser.add_argument(
        "--maksimum-run",
        type=int,
        default=None,
        help="Teknik kontrol için yalnız ilk N koşuyu çalıştırır.",
    )
    args = parser.parse_args()

    ayarlar = yaml.safe_load(
        Path("ayarlar/lr_pilotu.yaml").read_text(encoding="utf-8")
    )
    plan = deney_plani(ayarlar)

    print(f"Toplam planlanan koşu: {len(plan)}")
    for sira, kosu in enumerate(plan, start=1):
        print(
            f"{sira:02d}. {kosu['run_id']} | "
            f"{kosu['optimizer']} | {kosu['aktivasyon']} | lr={kosu['learning_rate']}"
        )

    if not args.calistir:
        print(
            "\nYalnız plan gösterildi. Gerçek pilotu başlatmak için "
            "'--calistir' parametresini kullanın."
        )
        return

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nKullanılan cihaz: {cihaz}")

    if cihaz.type == "cpu":
        raise RuntimeError(
            "LR pilotu CPU modunda baslatilmadi. "
            "Ana pilot icin CUDA destekli NVIDIA GPU gereklidir."
        )

    calisilacak = plan
    if args.maksimum_run is not None:
        calisilacak = plan[: args.maksimum_run]

    for sira, kosu in enumerate(calisilacak, start=1):
        print(f"\n=== Koşu {sira}/{len(calisilacak)}: {kosu['run_id']} ===")
        tek_kosu(ayarlar, kosu, cihaz)


if __name__ == "__main__":
    ana()
