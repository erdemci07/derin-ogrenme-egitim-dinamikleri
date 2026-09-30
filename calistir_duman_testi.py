"""İlk teknik duman testini çalıştırır.

Bu çalışma tez bulgusu üretmez. Amaç; veri hazırlama, model oluşturma,
eğitim, doğrulama ve kayıt zincirinin doğru çalıştığını kontrol etmektir.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import torch
import yaml
from torch import nn

from kaynak_kod.cnn_modeli import CNNv1
from kaynak_kod.egitim import bir_epoch_egit, dogrula
from kaynak_kod.kayit_sistemi import epoch_csv_kaydet, json_kaydet, klasor_hazirla
from kaynak_kod.optimizasyon_algoritmalari import optimizasyon_olustur
from kaynak_kod.tekrarlanabilirlik import sabit_seed_ayarla
from kaynak_kod.veri_hazirlama import cifar10_yukleyicileri


def ana() -> None:
    ayar_yolu = Path("ayarlar/duman_testi.yaml")
    ayarlar = yaml.safe_load(ayar_yolu.read_text(encoding="utf-8"))

    seed = int(ayarlar["egitim"]["seed"])
    sabit_seed_ayarla(seed)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {cihaz}")

    train_loader, val_loader = cifar10_yukleyicileri(
        veri_klasoru=ayarlar["veri"]["veri_klasoru"],
        split_dosyasi=ayarlar["veri"]["split_dosyasi"],
        batch_size=int(ayarlar["veri"]["batch_size"]),
        split_seed=int(ayarlar["veri"]["split_seed"]),
        loader_seed=seed,
        num_workers=int(ayarlar["veri"]["num_workers"]),
    )

    model = CNNv1(ayarlar["model"]["aktivasyon"]).to(cihaz)
    kayip_fonksiyonu = nn.CrossEntropyLoss()

    optimizer = optimizasyon_olustur(
        ayarlar["optimizasyon"]["algoritma"],
        model.parameters(),
        learning_rate=float(ayarlar["optimizasyon"]["learning_rate"]),
        weight_decay=float(ayarlar["optimizasyon"]["weight_decay"]),
    )

    epoch_kayitlari = []

    for epoch in range(1, int(ayarlar["egitim"]["epoch"]) + 1):
        train_sonuc = bir_epoch_egit(
            model,
            train_loader,
            kayip_fonksiyonu,
            optimizer,
            cihaz,
        )
        val_sonuc = dogrula(
            model,
            val_loader,
            kayip_fonksiyonu,
            cihaz,
        )

        kayit = {
            "epoch": epoch,
            "train_loss": round(train_sonuc["loss"], 6),
            "train_accuracy": round(train_sonuc["accuracy"], 6),
            "validation_loss": round(val_sonuc["loss"], 6),
            "validation_accuracy": round(val_sonuc["accuracy"], 6),
            "epoch_seconds": round(train_sonuc["seconds"], 3),
        }
        epoch_kayitlari.append(kayit)

        print(
            f"Epoch {epoch}: "
            f"train_acc={kayit['train_accuracy']:.4f}, "
            f"val_acc={kayit['validation_accuracy']:.4f}"
        )

    cikti_klasoru = klasor_hazirla(ayarlar["cikti"]["klasor"])

    epoch_csv_kaydet(
        cikti_klasoru / "epoch_sonuclari.csv",
        epoch_kayitlari,
    )

    metadata = {
        "deney_kodu": ayarlar["deney_kodu"],
        "aciklama": ayarlar["aciklama"],
        "seed": seed,
        "aktivasyon": ayarlar["model"]["aktivasyon"],
        "optimizer": ayarlar["optimizasyon"]["algoritma"],
        "learning_rate": ayarlar["optimizasyon"]["learning_rate"],
        "weight_decay": ayarlar["optimizasyon"]["weight_decay"],
        "epoch": ayarlar["egitim"]["epoch"],
        "cihaz": str(cihaz),
        "python_surumu": platform.python_version(),
        "pytorch_surumu": torch.__version__,
        "cuda_surumu": torch.version.cuda,
        "platform": sys.platform,
    }
    json_kaydet(cikti_klasoru / "deney_bilgileri.json", metadata)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "ayarlar": ayarlar,
        },
        cikti_klasoru / "son_model.pt",
    )

    print("\nDuman testi tamamlandı.")
    print(f"Sonuç klasörü: {cikti_klasoru}")


if __name__ == "__main__":
    ana()
