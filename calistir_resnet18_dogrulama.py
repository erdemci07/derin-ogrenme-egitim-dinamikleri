"""ResNet-18 üzerinde seçilmiş CIFAR-10 kombinasyonlarının doğrulaması.

Ana 6x5 CNN deneyinden sonra, tez önerisindeki "seçilmiş deneyler"
ifadesini sınır kabul ederek yalnızca önceden gerekçelendirilmiş altı
tekil kombinasyon CIFAR-10 üzerinde üç sabit seed ile tekrarlanır.
Bu modül yeni bir hiperparametre araması veya ana deney değildir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from torch import nn

from calistir_ana_deney import (
    _config_oku,
    _genel_config_dogrula,
    _optimizer_olustur,
    _train_loader_olustur,
    bir_epoch_egit,
    dogrula,
    sabit_seed_ayarla,
    yakinsama_ozeti,
)
from kaynak_kod.ana_veri_hazirlama import ana_veri_yukleyicileri
from kaynak_kod.checkpoint import checkpoint_kaydet, checkpoint_yukle
from kaynak_kod.kayit_sistemi import (
    atomik_json_yaz,
    atomik_satir_yaz,
    cihaza_tasima,
    epoch_csv_basligi,
    kararsiz_kaydi_yaz,
)
from kaynak_kod.resnet18_modeli import ResNet18CIFAR


MIMARI_ADI = "resnet18_cifar"
VERI_SETI = "cifar10"
SECILI_KOMBINASYONLAR: tuple[dict[str, Any], ...] = (
    {
        "optimizer": "adam",
        "activation": "gelu",
        "roller": (
            "genel_performans_en_yuksek",
            "etkilesim_2x2_kontrol",
        ),
    },
    {
        "optimizer": "adamw",
        "activation": "gelu",
        "roller": ("en_hizli_yakinsama",),
    },
    {
        "optimizer": "sgd",
        "activation": "relu",
        "roller": (
            "genel_performans_en_dusuk_gecerli",
            "en_yavas_yakinsama_gecerli",
        ),
    },
    {
        "optimizer": "momentum_sgd",
        "activation": "elu",
        "roller": ("en_guclu_pozitif_aktivasyon_optimizer_etkilesimi",),
    },
    {
        "optimizer": "adam",
        "activation": "elu",
        "roller": ("etkilesim_2x2_karsit_kol",),
    },
    {
        "optimizer": "momentum_sgd",
        "activation": "gelu",
        "roller": ("etkilesim_2x2_kontrol",),
    },
)


def _sha256(dosya: Path) -> str:
    return hashlib.sha256(dosya.read_bytes()).hexdigest()


def _ana_kok() -> Path:
    return Path(__file__).resolve().parent


def _cikti_koku() -> Path:
    varsayilan = _ana_kok() / "sonuclar" / "resnet18_secilmis_dogrulama" / VERI_SETI
    return Path(os.environ.get("TEZ_CIKTI_KLASORU", varsayilan))


def _calisma_kodu(optimizer: str, aktivasyon: str, seed: int) -> str:
    return f"RESNET18-CIFAR10-{optimizer.upper()}-{aktivasyon.upper()}-S{seed}"


def _plan_olustur(seeds: list[int]) -> list[dict[str, Any]]:
    return [
        {
            **kombinasyon,
            "seed": seed,
            "run_id": _calisma_kodu(
                kombinasyon["optimizer"], kombinasyon["activation"], seed
            ),
        }
        for kombinasyon in SECILI_KOMBINASYONLAR
        for seed in seeds
    ]


def _config_dogrula(config: dict[str, Any], config_yolu: Path) -> None:
    """Ana deneyin dondurulmuş ayarlarını aynen kullanmayı zorunlu kılar."""

    _genel_config_dogrula(config, config_yolu)
    if int(config["deney"]["epoch_sayisi"]) != 50:
        raise ValueError("ResNet-18 doğrulaması ana deneydeki 50 epoch bütçesini kullanır.")
    if list(config["deney"]["seeds"]) != [101, 202, 303]:
        raise ValueError("ResNet-18 doğrulaması ana deneyle aynı üç sabit seed'i kullanır.")
    if config["deney"]["data_sets"] != [VERI_SETI]:
        # Ana YAML çoklu veri seti içerebilir; bu kontrol yalnızca CIFAR'ın varlığını ister.
        if VERI_SETI not in config["deney"]["data_sets"]:
            raise ValueError("Dondurulmuş ana yapılandırmada CIFAR-10 bulunmalıdır.")


def _donanim_dogrula(cihaz: torch.device) -> None:
    if cihaz.type != "cuda":
        raise RuntimeError("Bu doğrulama Colab T4 GPU üzerinde çalıştırılmalıdır.")
    ad = torch.cuda.get_device_name(cihaz)
    if "T4" not in ad.upper():
        raise RuntimeError(f"Beklenen GPU T4, bulunan GPU: {ad}")


def _veri_yollarini_dogrula() -> tuple[Path, Path, Path]:
    veri_koku = os.environ.get("TEZ_VERI_KLASORU")
    split_dosyasi = os.environ.get("TEZ_SPLIT_DOSYASI")
    istatistik_dosyasi = os.environ.get("TEZ_STATS_DOSYASI")
    if not all((veri_koku, split_dosyasi, istatistik_dosyasi)):
        raise RuntimeError(
            "TEZ_VERI_KLASORU, TEZ_SPLIT_DOSYASI ve TEZ_STATS_DOSYASI "
            "ortam değişkenleri ayarlanmalıdır."
        )

    yollar = tuple(Path(yol) for yol in (veri_koku, split_dosyasi, istatistik_dosyasi))
    eksikler = [str(yol) for yol in yollar if not yol.exists()]
    if eksikler:
        raise FileNotFoundError(f"Paylaşımlı veri/split dosyası bulunamadı: {eksikler}")
    return yollar


def _veri_paketi_olustur(seed: int, config: dict[str, Any]) -> dict[str, Any]:
    veri_koku, split_dosyasi, istatistik_dosyasi = _veri_yollarini_dogrula()
    return ana_veri_yukleyicileri(
        veri_seti=VERI_SETI,
        veri_koku=veri_koku,
        split_dosyasi=split_dosyasi,
        istatistik_dosyasi=istatistik_dosyasi,
        batch_size=int(config["deney"]["batch_size"]),
        seed=seed,
        num_workers=int(config["deney"]["num_workers"]),
    )


def on_kontrol(config: dict[str, Any], cihaz: torch.device) -> None:
    """Gerçek koşulardan önce veri, model, kayıp ve optimizer uyumunu sınar."""

    _donanim_dogrula(cihaz)
    sabit_seed_ayarla(int(config["deney"]["seeds"][0]))
    paket = _veri_paketi_olustur(int(config["deney"]["seeds"][0]), config)
    girdi, hedef = next(iter(paket["train_loader"]))
    girdi = cihaza_tasima(girdi, cihaz)
    hedef = cihaza_tasima(hedef, cihaz)

    for aktivasyon in sorted({x["activation"] for x in SECILI_KOMBINASYONLAR}):
        model = ResNet18CIFAR(aktivasyon).to(cihaz)
        logits = model(girdi)
        if logits.shape != (girdi.shape[0], 10):
            raise RuntimeError(f"Beklenmeyen ResNet-18 çıktı şekli: {tuple(logits.shape)}")
        kayip = nn.CrossEntropyLoss()(logits, hedef)
        if not torch.isfinite(kayip):
            raise RuntimeError(f"Ön kontrolde sonlu olmayan kayıp: aktivasyon={aktivasyon}")

    for kombinasyon in SECILI_KOMBINASYONLAR:
        model = ResNet18CIFAR(kombinasyon["activation"]).to(cihaz)
        optimizer = _optimizer_olustur(model, kombinasyon["optimizer"], config)
        if optimizer is None:
            raise RuntimeError("Optimizer oluşturulamadı.")

    print(
        "Ön kontrol tamam: CIFAR-10, ResNet-18, 6 seçili kombinasyon ve "
        "CrossEntropyLoss uyumlu."
    )


def _tamamlanmis_mi(calisma_klasoru: Path) -> bool:
    return (calisma_klasoru / "tamamlandi.json").exists()


def _onceki_checkpoint(calisma_klasoru: Path) -> dict[str, Any] | None:
    checkpoint_yolu = calisma_klasoru / "checkpoint.pt"
    if checkpoint_yolu.exists():
        return checkpoint_yukle(checkpoint_yolu)
    return None


def _calismayi_yurut(
    kosu: dict[str, Any],
    config: dict[str, Any],
    config_yolu: Path,
    cihaz: torch.device,
) -> str:
    run_id = str(kosu["run_id"])
    calisma_klasoru = _cikti_koku() / run_id
    tamamlandi = calisma_klasoru / "tamamlandi.json"
    kararsiz = calisma_klasoru / "kararsiz.json"

    if tamamlandi.exists():
        print(f"[ATLA] {run_id}: tamamlandı.")
        return "atlandi"
    if kararsiz.exists():
        print(f"[ATLA] {run_id}: önceki kararsızlık kaydı korunuyor.")
        return "kararsiz"

    calisma_klasoru.mkdir(parents=True, exist_ok=True)
    epoch_csv = calisma_klasoru / "epoch_sonuclari.csv"
    checkpoint_yolu = calisma_klasoru / "checkpoint.pt"

    sabit_seed_ayarla(int(kosu["seed"]))
    paket = _veri_paketi_olustur(int(kosu["seed"]), config)
    model = ResNet18CIFAR(str(kosu["activation"])).to(cihaz)
    optimizer = _optimizer_olustur(model, str(kosu["optimizer"]), config)
    kayip_fonksiyonu = nn.CrossEntropyLoss()

    baslangic_epoch = 1
    epoch_kayitlari: list[dict[str, Any]] = []
    onceki = _onceki_checkpoint(calisma_klasoru)
    if onceki is not None:
        model.load_state_dict(onceki["model_state"])
        optimizer.load_state_dict(onceki["optimizer_state"])
        baslangic_epoch = int(onceki["epoch"]) + 1
        epoch_kayitlari = list(onceki.get("epoch_kayitlari", []))
        print(f"[DEVAM] {run_id}: epoch {baslangic_epoch}.")

    if not epoch_csv.exists():
        atomik_satir_yaz(epoch_csv, epoch_csv_basligi(), baslik=True)

    epoch_sayisi = int(config["deney"]["epoch_sayisi"])
    baslangic_zamani = time.perf_counter()
    try:
        for epoch in range(baslangic_epoch, epoch_sayisi + 1):
            train_loader = _train_loader_olustur(
                paket["train_dataset"],
                int(config["deney"]["batch_size"]),
                int(kosu["seed"]),
                epoch,
                int(config["deney"]["num_workers"]),
            )
            train_metrikler = bir_epoch_egit(
                model, train_loader, optimizer, kayip_fonksiyonu, cihaz
            )
            val_metrikler = dogrula(
                model, paket["val_loader"], kayip_fonksiyonu, cihaz
            )

            metrikler = {
                "epoch": epoch,
                "train_loss": float(train_metrikler["loss"]),
                "train_accuracy": float(train_metrikler["accuracy"]),
                "val_loss": float(val_metrikler["loss"]),
                "val_accuracy": float(val_metrikler["accuracy"]),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            if not all(math.isfinite(deger) for deger in metrikler.values() if isinstance(deger, float)):
                raise FloatingPointError(f"Sonlu olmayan metrik: {metrikler}")

            atomik_satir_yaz(epoch_csv, metrikler)
            epoch_kayitlari.append(metrikler)
            checkpoint_kaydet(
                checkpoint_yolu,
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch_kayitlari": epoch_kayitlari,
                },
            )
            print(
                f"[{run_id}] {epoch:02d}/{epoch_sayisi} "
                f"train_acc={metrikler['train_accuracy']:.4f} "
                f"val_acc={metrikler['val_accuracy']:.4f}"
            )

    except (FloatingPointError, RuntimeError) as hata:
        kararsiz_kaydi_yaz(
            kararsiz,
            {
                "run_id": run_id,
                "mimari": MIMARI_ADI,
                "veri_seti": VERI_SETI,
                "optimizer": kosu["optimizer"],
                "activation": kosu["activation"],
                "seed": kosu["seed"],
                "hata": str(hata),
                "epoch_kayit_sayisi": len(epoch_kayitlari),
                "gradyan_kirpma": "yok",
            },
        )
        print(f"[KARARSIZ] {run_id}: {hata}")
        return "kararsiz"

    test_metrikler = dogrula(model, paket["test_loader"], kayip_fonksiyonu, cihaz)
    yakinsama = yakinsama_ozeti(epoch_kayitlari)
    toplam_sure = time.perf_counter() - baslangic_zamani
    config_sha = _sha256(config_yolu)

    sonuc = {
        "run_id": run_id,
        "mimari": MIMARI_ADI,
        "veri_seti": VERI_SETI,
        "optimizer": kosu["optimizer"],
        "activation": kosu["activation"],
        "seed": kosu["seed"],
        "secim_rollleri": list(kosu["roller"]),
        "secim_kaynagi": "ana_CNN_sonuclari",
        "config_dosyasi": str(config_yolu.name),
        "config_sha256": config_sha,
        "epoch_sayisi": epoch_sayisi,
        "gradyan_kirpma": "yok",
        "scheduler": "yok",
        "early_stopping": "yok",
        "test_loss": float(test_metrikler["loss"]),
        "test_accuracy": float(test_metrikler["accuracy"]),
        "test_precision_macro": float(test_metrikler["precision_macro"]),
        "test_recall_macro": float(test_metrikler["recall_macro"]),
        "test_f1_macro": float(test_metrikler["f1_macro"]),
        "toplam_egitim_suresi_saniye": float(toplam_sure),
        "yakinsama": yakinsama,
        "son_epoch": epoch_kayitlari[-1],
    }
    atomik_json_yaz(tamamlandi, sonuc)
    if checkpoint_yolu.exists():
        checkpoint_yolu.unlink()
    print(
        f"[TAMAM] {run_id}: test_acc={sonuc['test_accuracy']:.4f}, "
        f"test_f1={sonuc['test_f1_macro']:.4f}"
    )
    return "tamamlandi"


def ana(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ayar",
        type=Path,
        default=_ana_kok() / "ayarlar" / "frozen_main_config.yaml",
    )
    parser.add_argument("--sadece-on-kontrol", action="store_true")
    parser.add_argument(
        "--maksimum-run",
        type=int,
        default=None,
        help="Kesintiden sonra sınırlı sayıda koşu yürütmek için isteğe bağlı sınır.",
    )
    args = parser.parse_args(argv)

    config_yolu = args.ayar.resolve()
    config = _config_oku(config_yolu)
    _config_dogrula(config, config_yolu)
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    on_kontrol(config, cihaz)
    if args.sadece_on_kontrol:
        return

    plan = _plan_olustur(list(config["deney"]["seeds"]))
    if args.maksimum_run is not None:
        plan = plan[: args.maksimum_run]

    _cikti_koku().mkdir(parents=True, exist_ok=True)
    plan_bilgisi = {
        "mimari": MIMARI_ADI,
        "veri_seti": VERI_SETI,
        "kombinasyon_sayisi": len(SECILI_KOMBINASYONLAR),
        "seedler": list(config["deney"]["seeds"]),
        "toplam_kosu": len(plan),
        "secili_kombinasyonlar": [
            {
                "optimizer": x["optimizer"],
                "activation": x["activation"],
                "roller": list(x["roller"]),
            }
            for x in SECILI_KOMBINASYONLAR
        ],
        "config_sha256": _sha256(config_yolu),
    }
    atomik_json_yaz(_cikti_koku() / "plan.json", plan_bilgisi)

    sayac = {"tamamlandi": 0, "atlandi": 0, "kararsiz": 0}
    for kosu in plan:
        durum = _calismayi_yurut(kosu, config, config_yolu, cihaz)
        sayac[durum] += 1
    print(f"ResNet-18 seçili doğrulama özeti: {sayac}")


if __name__ == "__main__":
    ana()
