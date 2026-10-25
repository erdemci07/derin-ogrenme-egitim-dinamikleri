"""ResNet-18 üzerinde seçilmiş CIFAR-10 kombinasyonlarının doğrulaması.

Ana 6x5 CNN deneyinden sonra, tez önerisindeki "seçilmiş deneyler"
ifadesini sınır kabul ederek yalnızca önceden gerekçelendirilmiş altı
tekil kombinasyon CIFAR-10 üzerinde üç sabit seed ile tekrarlanır.
Bu modül yeni bir hiperparametre araması veya ana deney değildir.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from calistir_ana_deney import _config_oku, _genel_config_dogrula
from kaynak_kod.ana_veri_hazirlama import ana_veri_yukleyicileri
from kaynak_kod.checkpoint import checkpoint_kaydet, checkpoint_yukle
from kaynak_kod.egitim import bir_epoch_egit, dogrula
from kaynak_kod.kayit_sistemi import epoch_csv_kaydet, json_kaydet, klasor_hazirla
from kaynak_kod.optimizasyon_algoritmalari import optimizasyon_olustur
from kaynak_kod.resnet18_modeli import ResNet18CIFAR
from kaynak_kod.tekrarlanabilirlik import sabit_seed_ayarla
from kaynak_kod.yakinsama_analizi import yakinsama_ozeti


MIMARI_ADI = "resnet18_cifar"
VERI_SETI = "cifar10"
SECILI_KOMBINASYONLAR: tuple[dict[str, Any], ...] = (
    {
        "optimizer": "adam",
        "activation": "gelu",
        "roller": ("genel_performans_en_yuksek", "etkilesim_2x2_kontrol"),
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


def _cikti_koku() -> Path:
    varsayilan = Path("sonuclar") / "resnet18_secilmis_dogrulama" / VERI_SETI
    return Path(os.environ.get("TEZ_CIKTI_KLASORU", varsayilan))


def _calisma_kodu(optimizer: str, aktivasyon: str, seed: int) -> str:
    return f"RESNET18-CIFAR10-{optimizer.upper()}-{aktivasyon.upper()}-S{seed}"


def _plan_olustur(cfg: dict) -> list[dict[str, Any]]:
    plan = []
    for seed in cfg["egitim"]["seeds"]:
        for kombinasyon in SECILI_KOMBINASYONLAR:
            optimizer = kombinasyon["optimizer"]
            opt_cfg = cfg["optimizerlar"][optimizer]
            plan.append(
                {
                    **kombinasyon,
                    "seed": int(seed),
                    "learning_rate": float(opt_cfg["learning_rate"]),
                    "weight_decay": float(opt_cfg["weight_decay"]),
                    "run_id": _calisma_kodu(
                        optimizer, kombinasyon["activation"], int(seed)
                    ),
                }
            )
    return plan


def _veri_paketi(cfg: dict, seed: int):
    egitim = cfg["egitim"]
    return ana_veri_yukleyicileri(
        veri_seti=VERI_SETI,
        veri_klasoru="veriler",
        split_dosyasi=f"ayarlar/{VERI_SETI}_main_split.json",
        stats_dosyasi=f"ayarlar/{VERI_SETI}_main_stats.json",
        batch_size=int(egitim["batch_size"]),
        split_seed=int(egitim["split_seed"]),
        loader_seed=int(seed),
        num_workers=int(egitim["num_workers"]),
    )


def _donanim_dogrula(cihaz: torch.device) -> str:
    if cihaz.type != "cuda":
        raise RuntimeError("Bu doğrulama Colab T4 GPU üzerinde çalıştırılmalıdır.")
    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu.upper():
        raise RuntimeError(f"Beklenen GPU T4, bulunan GPU: {gpu}")
    return gpu


def on_kontrol(cfg: dict, cihaz: torch.device) -> None:
    """Koşulardan önce CIFAR-10, ResNet-18, kayıp ve optimizer zincirini sınar."""

    _donanim_dogrula(cihaz)
    seed = int(cfg["egitim"]["seeds"][0])
    sabit_seed_ayarla(seed)
    paket = _veri_paketi(cfg, seed)
    girdiler, hedefler = next(iter(paket.train_loader))
    if girdiler.ndim != 4 or tuple(girdiler.shape[1:]) != (3, 32, 32):
        raise RuntimeError(f"Beklenen giriş N×3×32×32, gelen: {tuple(girdiler.shape)}")

    mini_x = girdiler[: min(8, len(girdiler))].to(cihaz)
    mini_y = hedefler[: len(mini_x)].to(cihaz)
    kayip_fonksiyonu = nn.CrossEntropyLoss()

    for aktivasyon in sorted({x["activation"] for x in SECILI_KOMBINASYONLAR}):
        sabit_seed_ayarla(seed)
        model = ResNet18CIFAR(aktivasyon, sinif_sayisi=paket.sinif_sayisi).to(cihaz)
        logits = model(mini_x)
        if logits.shape != (len(mini_x), paket.sinif_sayisi):
            raise RuntimeError(
                f"{aktivasyon}: beklenmeyen model çıktısı {tuple(logits.shape)}"
            )
        kayip = kayip_fonksiyonu(logits, mini_y)
        if not math.isfinite(float(kayip.item())):
            raise RuntimeError(f"{aktivasyon}: ön kontrol kaybı sonlu değil.")

    for kombinasyon in SECILI_KOMBINASYONLAR:
        model = ResNet18CIFAR(
            kombinasyon["activation"], sinif_sayisi=paket.sinif_sayisi
        ).to(cihaz)
        opt_cfg = cfg["optimizerlar"][kombinasyon["optimizer"]]
        optimizer = optimizasyon_olustur(
            kombinasyon["optimizer"],
            model.parameters(),
            learning_rate=float(opt_cfg["learning_rate"]),
            weight_decay=float(opt_cfg["weight_decay"]),
        )
        optimizer.zero_grad(set_to_none=True)
        kayip = kayip_fonksiyonu(model(mini_x), mini_y)
        kayip.backward()
        optimizer.step()
        if not math.isfinite(float(kayip.item())):
            raise RuntimeError(f"{kombinasyon['optimizer']}: optimizer ön kontrolü başarısız.")

    print(
        "ÖN KONTROL BAŞARILI | CIFAR-10 | ResNet-18 | "
        f"train={paket.meta['train_count']} | val={paket.meta['validation_count']} | "
        f"test={paket.meta['test_count']}",
        flush=True,
    )


def _calismayi_yurut(
    cfg: dict,
    config_sha256: str,
    kosu: dict[str, Any],
    cihaz: torch.device,
    gpu: str,
) -> str:
    run_id = str(kosu["run_id"])
    cikti = _cikti_koku() / run_id
    tamamlandi = cikti / "tamamlandi.json"
    kararsiz = cikti / "kararsiz.json"
    checkpoint_dosyasi = cikti / "checkpoint.pt"

    if tamamlandi.exists():
        print(f"[ATLA] {run_id} daha önce tamamlanmış.", flush=True)
        return "atlandi"
    if kararsiz.exists():
        print(f"[ATLA-KARARSIZ] {run_id} önceki kararsızlık kaydı korunuyor.", flush=True)
        return "kararsiz"

    klasor_hazirla(cikti)
    sabit_seed_ayarla(int(kosu["seed"]))
    paket = _veri_paketi(cfg, int(kosu["seed"]))

    # Veri yükleyici kurulumu RNG tüketse bile model başlangıcı seed içinde sabit kalır.
    sabit_seed_ayarla(int(kosu["seed"]))
    model = ResNet18CIFAR(
        str(kosu["activation"]), sinif_sayisi=paket.sinif_sayisi
    ).to(cihaz)
    kayip_fonksiyonu = nn.CrossEntropyLoss()
    optimizer = optimizasyon_olustur(
        str(kosu["optimizer"]),
        model.parameters(),
        learning_rate=float(kosu["learning_rate"]),
        weight_decay=float(kosu["weight_decay"]),
    )

    baslangic_epoch, kayitlar = checkpoint_yukle(
        checkpoint_dosyasi,
        model=model,
        optimizer=optimizer,
        train_loader=paket.train_loader,
        cihaz=cihaz,
        beklenen_run_id=run_id,
    )
    if baslangic_epoch > 1:
        print(f"[DEVAM] {run_id} epoch {baslangic_epoch}'den sürdürülüyor.", flush=True)

    toplam_epoch = int(cfg["egitim"]["epoch"])
    for epoch in range(baslangic_epoch, toplam_epoch + 1):
        train = bir_epoch_egit(
            model, paket.train_loader, kayip_fonksiyonu, optimizer, cihaz
        )
        val = dogrula(model, paket.validation_loader, kayip_fonksiyonu, cihaz)
        satir = {
            "epoch": epoch,
            "train_loss": train["loss"],
            "train_accuracy": train["accuracy"],
            "validation_loss": val["loss"],
            "validation_accuracy": val["accuracy"],
            "macro_precision": val["macro_precision"],
            "macro_recall": val["macro_recall"],
            "macro_f1": val["macro_f1"],
            "learning_rate": kosu["learning_rate"],
            "weight_decay": kosu["weight_decay"],
            "train_seconds": train["seconds"],
            "validation_seconds": val["seconds"],
            "epoch_seconds": train["seconds"] + val["seconds"],
        }
        kayitlar.append(satir)
        epoch_csv_kaydet(cikti / "epoch_sonuclari.csv", kayitlar)

        kontrol = (
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        )
        if not all(math.isfinite(float(satir[alan])) for alan in kontrol):
            json_kaydet(
                kararsiz,
                {
                    **kosu,
                    "mimari": MIMARI_ADI,
                    "veri_seti": VERI_SETI,
                    "epoch": epoch,
                    "neden": "NaN veya Inf tespit edildi.",
                    "gradyan_kirpma": "yok",
                    "frozen_config_sha256": config_sha256,
                    "dataset_meta": paket.meta,
                },
            )
            print(f"[KARARSIZ] {run_id} epoch={epoch}", flush=True)
            return "kararsiz"

        checkpoint_kaydet(
            checkpoint_dosyasi,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            kayitlar=kayitlar,
            train_loader=paket.train_loader,
            run_id=run_id,
        )
        print(
            f"{run_id} | {epoch:02d}/{toplam_epoch} | "
            f"train_acc={train['accuracy']:.4f} | val_acc={val['accuracy']:.4f}",
            flush=True,
        )

    test = dogrula(model, paket.test_loader, kayip_fonksiyonu, cihaz)
    test_alanlari = ("loss", "accuracy", "macro_precision", "macro_recall", "macro_f1")
    if not all(math.isfinite(float(test[alan])) for alan in test_alanlari):
        json_kaydet(
            kararsiz,
            {
                **kosu,
                "mimari": MIMARI_ADI,
                "veri_seti": VERI_SETI,
                "epoch": toplam_epoch,
                "neden": "Final test değerlendirmesinde NaN veya Inf tespit edildi.",
                "gradyan_kirpma": "yok",
                "frozen_config_sha256": config_sha256,
                "dataset_meta": paket.meta,
            },
        )
        return "kararsiz"

    son = kayitlar[-1]
    yakinsama = yakinsama_ozeti(
        [float(x["validation_accuracy"]) for x in kayitlar]
    )
    json_kaydet(
        tamamlandi,
        {
            **kosu,
            "mimari": MIMARI_ADI,
            "veri_seti": VERI_SETI,
            "epoch": toplam_epoch,
            "secim_rollleri": list(kosu["roller"]),
            "secim_kaynagi": "ana_CNN_sonuclari",
            "gradyan_kirpma": "yok",
            "scheduler": "yok",
            "early_stopping": False,
            "final_train_loss": son["train_loss"],
            "final_train_accuracy": son["train_accuracy"],
            "final_validation_loss": son["validation_loss"],
            "final_validation_accuracy": son["validation_accuracy"],
            "final_validation_macro_precision": son["macro_precision"],
            "final_validation_macro_recall": son["macro_recall"],
            "final_validation_macro_f1": son["macro_f1"],
            "test_loss": test["loss"],
            "test_accuracy": test["accuracy"],
            "test_macro_precision": test["macro_precision"],
            "test_macro_recall": test["macro_recall"],
            "test_macro_f1": test["macro_f1"],
            "aulc_normalized": yakinsama["aulc_normalized"],
            "epoch_to_95_own_best": yakinsama["epoch_to_95_own_best"],
            "own_best_validation_accuracy": yakinsama["own_best_validation_accuracy"],
            "total_train_seconds": sum(float(x["train_seconds"]) for x in kayitlar),
            "total_validation_seconds": sum(
                float(x["validation_seconds"]) for x in kayitlar
            ),
            "dataset_meta": paket.meta,
            "class_names": paket.sinif_adlari,
            "frozen_config_sha256": config_sha256,
            "gpu": gpu,
            "pytorch_surumu": torch.__version__,
            "cuda_surumu": torch.version.cuda,
            "git_commit": os.environ.get("TEZ_GIT_COMMIT", "unknown"),
        },
    )
    if checkpoint_dosyasi.exists():
        checkpoint_dosyasi.unlink()

    print(
        f"[TAMAM] {run_id} | test_acc={test['accuracy']:.4f} | "
        f"test_f1={test['macro_f1']:.4f}",
        flush=True,
    )
    return "tamamlandi"


def _ilerleme(plan: list[dict[str, Any]]) -> tuple[int, int, int]:
    tamam = sum((_cikti_koku() / x["run_id"] / "tamamlandi.json").exists() for x in plan)
    kararsiz = sum(
        (_cikti_koku() / x["run_id"] / "kararsiz.json").exists()
        and not (_cikti_koku() / x["run_id"] / "tamamlandi.json").exists()
        for x in plan
    )
    return tamam, kararsiz, tamam + kararsiz


def ana(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="ayarlar/frozen_main_config.yaml",
        help="Dondurulmuş ana deney ayar dosyası.",
    )
    parser.add_argument("--sadece-on-kontrol", action="store_true")
    parser.add_argument("--maksimum-run", type=int, default=None)
    args = parser.parse_args(argv)

    cfg, config_sha256 = _config_oku(Path(args.config))
    _genel_config_dogrula(cfg)
    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = _donanim_dogrula(cihaz)
    on_kontrol(cfg, cihaz)
    if args.sadece_on_kontrol:
        return

    plan = _plan_olustur(cfg)
    if len(plan) != 18:
        raise RuntimeError(f"18 ResNet-18 koşusu bekleniyordu, {len(plan)} üretildi.")
    if args.maksimum_run is not None:
        if args.maksimum_run <= 0:
            raise ValueError("--maksimum-run pozitif olmalıdır.")
        plan = plan[: args.maksimum_run]

    klasor_hazirla(_cikti_koku())
    json_kaydet(
        _cikti_koku() / "plan.json",
        {
            "mimari": MIMARI_ADI,
            "veri_seti": VERI_SETI,
            "kombinasyon_sayisi": len(SECILI_KOMBINASYONLAR),
            "toplam_kosu": len(plan),
            "seeds": list(cfg["egitim"]["seeds"]),
            "secili_kombinasyonlar": [
                {
                    "optimizer": x["optimizer"],
                    "activation": x["activation"],
                    "roller": list(x["roller"]),
                }
                for x in SECILI_KOMBINASYONLAR
            ],
            "frozen_config_sha256": config_sha256,
        },
    )

    tamam, kararsiz, deger = _ilerleme(plan)
    print(
        f"Başlangıç ilerleme: {deger}/{len(plan)} "
        f"({tamam} tamam, {kararsiz} kararsız).",
        flush=True,
    )
    for sira, kosu in enumerate(plan, start=1):
        print(f"=== {sira}/{len(plan)} | {kosu['run_id']} ===", flush=True)
        _calismayi_yurut(cfg, config_sha256, kosu, cihaz, gpu)

    tamam, kararsiz, deger = _ilerleme(plan)
    print(
        f"Son ilerleme: {deger}/{len(plan)} "
        f"({tamam} tamam, {kararsiz} kararsız).",
        flush=True,
    )


if __name__ == "__main__":
    ana()
