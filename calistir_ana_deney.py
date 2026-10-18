"""Dondurulmuş ana CNN deneylerini çalıştırır.

Her veri seti için:
6 aktivasyon x 5 optimizer x 3 bağımsız ana seed = 90 koşu.

Dört veri seti toplamı 360 ana koşudur. Hiperparametreler
ayarlar/frozen_main_config.yaml dosyasından okunur ve ana sonuçlar
başladıktan sonra değiştirilmez.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path

import torch
import yaml
from torch import nn

from kaynak_kod.ana_veri_hazirlama import ana_veri_yukleyicileri
from kaynak_kod.checkpoint import checkpoint_kaydet, checkpoint_yukle
from kaynak_kod.cnn_modeli import CNNv1
from kaynak_kod.egitim import bir_epoch_egit, dogrula
from kaynak_kod.kayit_sistemi import epoch_csv_kaydet, json_kaydet, klasor_hazirla
from kaynak_kod.optimizasyon_algoritmalari import optimizasyon_olustur
from kaynak_kod.tekrarlanabilirlik import sabit_seed_ayarla
from kaynak_kod.yakinsama_analizi import yakinsama_ozeti


GECERLI_VERI_SETLERI = ("cifar10", "mnist", "animals10", "flowers5")


def _config_oku(yol: Path) -> tuple[dict, str]:
    ham = yol.read_bytes()
    cfg = yaml.safe_load(ham.decode("utf-8"))
    return cfg, hashlib.sha256(ham).hexdigest()


def _optimizer_config_dogrula(cfg: dict) -> None:
    """Kod ile dondurulmuş optimizer parametrelerinin birebir uyumunu doğrular."""
    beklenen = {
        "sgd": {
            "learning_rate": 0.1,
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
        "momentum_sgd": {
            "learning_rate": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0,
        },
        "rmsprop": {
            "learning_rate": 0.001,
            "alpha": 0.99,
            "eps": 1e-8,
            "momentum": 0.0,
            "weight_decay": 0.0,
        },
        "adam": {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 0.0,
        },
        "adamw": {
            "learning_rate": 0.001,
            "beta1": 0.9,
            "beta2": 0.999,
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    }

    if set(cfg["optimizerlar"]) != set(beklenen):
        raise ValueError("Dondurulmuş optimizer listesi beklenen protokolle uyuşmuyor.")

    for ad, exp in beklenen.items():
        gercek = cfg["optimizerlar"][ad]
        for alan, deger in exp.items():
            if alan not in gercek:
                raise ValueError(f"{ad}.{alan} frozen config içinde yok.")
            if not math.isclose(
                float(gercek[alan]),
                float(deger),
                rel_tol=0.0,
                abs_tol=1e-15,
            ):
                raise ValueError(
                    f"{ad}.{alan} değiştirilmiş: {gercek[alan]} != {deger}"
                )


def _genel_config_dogrula(cfg: dict) -> None:
    aktivasyonlar = tuple(cfg["model"]["aktivasyonlar"])
    if aktivasyonlar != (
        "relu",
        "leaky_relu",
        "elu",
        "gelu",
        "swish",
        "mish",
    ):
        raise ValueError("Dondurulmuş aktivasyon listesi değiştirilmiş.")

    egitim = cfg["egitim"]
    if int(egitim["epoch"]) != 50:
        raise ValueError("Ana deney epoch değeri 50 olmalıdır.")
    if int(egitim["batch_size"]) != 128:
        raise ValueError("Ana deney batch size değeri 128 olmalıdır.")
    if [int(x) for x in egitim["seeds"]] != [101, 202, 303]:
        raise ValueError("Ana deney seedleri [101, 202, 303] olmalıdır.")
    if int(egitim["split_seed"]) != 2026:
        raise ValueError("Split seed 2026 olmalıdır.")
    if egitim.get("scheduler") is not None:
        raise ValueError("Ana deneyde scheduler kullanılmamalıdır.")
    if bool(egitim.get("early_stopping")):
        raise ValueError("Ana deneyde early stopping kullanılmamalıdır.")

    _optimizer_config_dogrula(cfg)


def run_id_olustur(
    veri_seti: str,
    optimizer: str,
    aktivasyon: str,
    seed: int,
) -> str:
    return (
        f"MAIN-{veri_seti.upper()}-{optimizer.upper()}-"
        f"{aktivasyon.upper()}-S{seed}"
    )


def deney_plani(cfg: dict, veri_seti: str) -> list[dict]:
    plan = []
    for seed in cfg["egitim"]["seeds"]:
        for optimizer in cfg["optimizerlar"]:
            for aktivasyon in cfg["model"]["aktivasyonlar"]:
                opt = cfg["optimizerlar"][optimizer]
                plan.append(
                    {
                        "run_id": run_id_olustur(
                            veri_seti, optimizer, aktivasyon, int(seed)
                        ),
                        "veri_seti": veri_seti,
                        "optimizer": optimizer,
                        "aktivasyon": aktivasyon,
                        "learning_rate": float(opt["learning_rate"]),
                        "weight_decay": float(opt["weight_decay"]),
                        "seed": int(seed),
                    }
                )
    return plan


def _veri_paketi(cfg: dict, veri_seti: str, seed: int):
    egitim = cfg["egitim"]
    return ana_veri_yukleyicileri(
        veri_seti=veri_seti,
        veri_klasoru="veriler",
        split_dosyasi=f"ayarlar/{veri_seti}_main_split.json",
        stats_dosyasi=f"ayarlar/{veri_seti}_main_stats.json",
        batch_size=int(egitim["batch_size"]),
        split_seed=int(egitim["split_seed"]),
        loader_seed=int(seed),
        num_workers=int(egitim["num_workers"]),
    )


def on_kontrol(cfg: dict, veri_seti: str, cihaz: torch.device) -> None:
    """90 koşu başlamadan veri/model/optimizer zincirini tek batch ile doğrular."""
    seed = int(cfg["egitim"]["seeds"][0])
    sabit_seed_ayarla(seed)
    paket = _veri_paketi(cfg, veri_seti, seed)

    girdiler, hedefler = next(iter(paket.train_loader))
    if girdiler.ndim != 4 or tuple(girdiler.shape[1:]) != (3, 32, 32):
        raise RuntimeError(
            f"Beklenen giriş N×3×32×32, gelen: {tuple(girdiler.shape)}"
        )
    if hedefler.numel() == 0:
        raise RuntimeError("Ön kontrolde boş batch geldi.")
    if int(hedefler.min()) < 0 or int(hedefler.max()) >= paket.sinif_sayisi:
        raise RuntimeError("Etiketler sınıf aralığı dışında.")

    mini_x = girdiler[: min(8, len(girdiler))].to(cihaz)
    mini_y = hedefler[: len(mini_x)].to(cihaz)
    loss_fn = nn.CrossEntropyLoss()

    for aktivasyon in cfg["model"]["aktivasyonlar"]:
        sabit_seed_ayarla(seed)
        model = CNNv1(
            aktivasyon,
            sinif_sayisi=paket.sinif_sayisi,
        ).to(cihaz)
        logit = model(mini_x)
        if logit.shape != (len(mini_x), paket.sinif_sayisi):
            raise RuntimeError(
                f"{aktivasyon}: beklenmeyen model çıktısı {tuple(logit.shape)}"
            )
        loss = loss_fn(logit, mini_y)
        if not math.isfinite(float(loss.item())):
            raise RuntimeError(f"{aktivasyon}: ön kontrol loss sonlu değil.")

    # Her optimizerı frozen LR/weight-decay ile bir kez gerçekten kur.
    for optimizer_adi, opt_cfg in cfg["optimizerlar"].items():
        model = CNNv1("relu", sinif_sayisi=paket.sinif_sayisi).to(cihaz)
        optimizer = optimizasyon_olustur(
            optimizer_adi,
            model.parameters(),
            learning_rate=float(opt_cfg["learning_rate"]),
            weight_decay=float(opt_cfg["weight_decay"]),
        )
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(mini_x), mini_y)
        loss.backward()
        optimizer.step()
        if not math.isfinite(float(loss.item())):
            raise RuntimeError(f"{optimizer_adi}: optimizer ön kontrolü başarısız.")

    print(
        "ÖN KONTROL BAŞARILI | "
        f"dataset={veri_seti} | sınıf={paket.sinif_sayisi} | "
        f"train={paket.meta['train_count']} | "
        f"val={paket.meta['validation_count']} | "
        f"test={paket.meta['test_count']} | "
        f"batch={tuple(girdiler.shape)}",
        flush=True,
    )
    print("Sınıflar:", paket.sinif_adlari, flush=True)


def tek_kosu(
    cfg: dict,
    config_sha256: str,
    kosu: dict,
    cihaz: torch.device,
) -> None:
    sabit_seed_ayarla(kosu["seed"])

    cikti_koku = Path(
        os.environ.get(
            "TEZ_CIKTI_KLASORU",
            f"sonuclar/ana_deney/{kosu['veri_seti']}",
        )
    )
    cikti = cikti_koku / kosu["run_id"]
    tamamlandi = cikti / "tamamlandi.json"
    kararsiz = cikti / "kararsiz.json"
    checkpoint_dosyasi = cikti / "checkpoint.pt"

    if tamamlandi.exists():
        print(f"[ATLA] {kosu['run_id']} daha önce tamamlanmış.", flush=True)
        return

    if kararsiz.exists():
        print(
            f"[ATLA-KARARSIZ] {kosu['run_id']} daha önce NaN/Inf ile "
            "kararsız olarak kaydedilmiş.",
            flush=True,
        )
        return

    klasor_hazirla(cikti)
    paket = _veri_paketi(cfg, kosu["veri_seti"], kosu["seed"])

    # Split/normalizasyon hazırlığının olası RNG tüketimi model başlangıcını
    # etkilemesin. Aynı seed içindeki tüm kombinasyonlar aynı başlangıç
    # rastgelelik durumundan model kurmaya başlar.
    sabit_seed_ayarla(kosu["seed"])

    model = CNNv1(
        kosu["aktivasyon"],
        sinif_sayisi=paket.sinif_sayisi,
    ).to(cihaz)
    kayip_fonksiyonu = nn.CrossEntropyLoss()
    optimizer = optimizasyon_olustur(
        kosu["optimizer"],
        model.parameters(),
        learning_rate=kosu["learning_rate"],
        weight_decay=kosu["weight_decay"],
    )

    baslangic_epoch, kayitlar = checkpoint_yukle(
        checkpoint_dosyasi,
        model=model,
        optimizer=optimizer,
        train_loader=paket.train_loader,
        cihaz=cihaz,
        beklenen_run_id=kosu["run_id"],
    )

    if baslangic_epoch > 1:
        print(
            f"[DEVAM] {kosu['run_id']} epoch {baslangic_epoch}'den sürdürülüyor.",
            flush=True,
        )

    toplam_epoch = int(cfg["egitim"]["epoch"])
    for epoch in range(baslangic_epoch, toplam_epoch + 1):
        train = bir_epoch_egit(
            model, paket.train_loader, kayip_fonksiyonu, optimizer, cihaz
        )
        val = dogrula(
            model, paket.validation_loader, kayip_fonksiyonu, cihaz
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
            "learning_rate": kosu["learning_rate"],
            "weight_decay": kosu["weight_decay"],
            "train_seconds": train["seconds"],
            "validation_seconds": val["seconds"],
            "epoch_seconds": train["seconds"] + val["seconds"],
        }
        kayitlar.append(satir)
        epoch_csv_kaydet(cikti / "epoch_sonuclari.csv", kayitlar)

        kontrol_alanlari = (
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        )
        if not all(
            math.isfinite(float(satir[alan])) for alan in kontrol_alanlari
        ):
            json_kaydet(
                kararsiz,
                {
                    **kosu,
                    "epoch": epoch,
                    "neden": "NaN veya Inf tespit edildi.",
                    "frozen_config_sha256": config_sha256,
                    "dataset_meta": paket.meta,
                },
            )
            print(f"[KARARSIZ] {kosu['run_id']} epoch={epoch}", flush=True)
            return

        checkpoint_kaydet(
            checkpoint_dosyasi,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            kayitlar=kayitlar,
            train_loader=paket.train_loader,
            run_id=kosu["run_id"],
        )

        print(
            f"{kosu['run_id']} | {epoch:02d}/{toplam_epoch} | "
            f"train_acc={train['accuracy']:.4f} | "
            f"val_acc={val['accuracy']:.4f}",
            flush=True,
        )

    if not kayitlar:
        raise RuntimeError(f"{kosu['run_id']}: epoch kaydı bulunamadı.")

    # Test seti yalnız eğitim tamamen bittikten sonra, bir kez değerlendirilir.
    test = dogrula(model, paket.test_loader, kayip_fonksiyonu, cihaz)
    if not all(
        math.isfinite(float(test[k]))
        for k in ("loss", "accuracy", "macro_precision", "macro_recall", "macro_f1")
    ):
        json_kaydet(
            kararsiz,
            {
                **kosu,
                "epoch": toplam_epoch,
                "neden": "Final test değerlendirmesinde NaN veya Inf tespit edildi.",
                "frozen_config_sha256": config_sha256,
                "dataset_meta": paket.meta,
            },
        )
        return

    son = kayitlar[-1]
    yakinsama = yakinsama_ozeti(
        [float(x["validation_accuracy"]) for x in kayitlar]
    )

    json_kaydet(
        tamamlandi,
        {
            **kosu,
            "epoch": toplam_epoch,
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
            "own_best_validation_accuracy": yakinsama[
                "own_best_validation_accuracy"
            ],
            "total_train_seconds": sum(float(x["train_seconds"]) for x in kayitlar),
            "total_validation_seconds": sum(
                float(x["validation_seconds"]) for x in kayitlar
            ),
            "dataset_meta": paket.meta,
            "class_names": paket.sinif_adlari,
            "frozen_config_sha256": config_sha256,
            "cihaz": str(cihaz),
            "gpu": torch.cuda.get_device_name(0),
            "pytorch_surumu": torch.__version__,
            "cuda_surumu": torch.version.cuda,
            "git_commit": os.environ.get("TEZ_GIT_COMMIT", "unknown"),
        },
    )

    if checkpoint_dosyasi.exists():
        checkpoint_dosyasi.unlink()


def _ilerleme(plan: list[dict], cikti_koku: Path) -> tuple[int, int, int]:
    tamam = sum(
        (cikti_koku / k["run_id"] / "tamamlandi.json").exists()
        for k in plan
    )
    kararsiz = sum(
        (cikti_koku / k["run_id"] / "kararsiz.json").exists()
        and not (cikti_koku / k["run_id"] / "tamamlandi.json").exists()
        for k in plan
    )
    return tamam, kararsiz, tamam + kararsiz


def ana(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--veri-seti", required=True, choices=GECERLI_VERI_SETLERI)
    p.add_argument(
        "--config",
        default="ayarlar/frozen_main_config.yaml",
        help="Dondurulmuş ana deney ayar dosyası.",
    )
    p.add_argument(
        "--sadece-on-kontrol",
        action="store_true",
        help="90 koşuyu başlatmadan veri/model/optimizer zincirini doğrular.",
    )
    p.add_argument(
        "--maksimum-run",
        type=int,
        default=None,
        help="Yalnız teknik test amacıyla ilk N koşuyu çalıştırır.",
    )
    args = p.parse_args(argv)

    config_yolu = Path(args.config)
    cfg, config_sha256 = _config_oku(config_yolu)
    _genel_config_dogrula(cfg)

    cihaz = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cihaz.type != "cuda":
        raise RuntimeError("Ana deney CPU'da başlatılmaz; NVIDIA T4 gereklidir.")

    gpu = torch.cuda.get_device_name(0)
    if "T4" not in gpu.upper():
        raise RuntimeError(
            f"Ana deney protokolü T4'e donduruldu; mevcut GPU: {gpu}"
        )

    if args.sadece_on_kontrol:
        on_kontrol(cfg, args.veri_seti, cihaz)
        return

    plan = deney_plani(cfg, args.veri_seti)
    if len(plan) != 90:
        raise RuntimeError(f"90 ana koşu bekleniyordu, {len(plan)} üretildi.")

    print(
        f"ANA DENEY | dataset={args.veri_seti} | koşu=90 | "
        f"epoch={cfg['egitim']['epoch']} | seeds={cfg['egitim']['seeds']} | "
        f"GPU={gpu}",
        flush=True,
    )
    print("Frozen config SHA256:", config_sha256, flush=True)

    calisilacak = plan
    if args.maksimum_run is not None:
        if args.maksimum_run <= 0:
            raise ValueError("--maksimum-run pozitif olmalıdır.")
        calisilacak = plan[: args.maksimum_run]

    cikti_koku = Path(
        os.environ.get(
            "TEZ_CIKTI_KLASORU",
            f"sonuclar/ana_deney/{args.veri_seti}",
        )
    )
    tamam, kararsiz, deger = _ilerleme(plan, cikti_koku)
    print(
        f"Başlangıç ilerleme: {deger}/90 "
        f"({tamam} tamam, {kararsiz} kararsız).",
        flush=True,
    )

    for sira, kosu in enumerate(calisilacak, start=1):
        print(
            f"\n=== {sira}/{len(calisilacak)} | {kosu['run_id']} ===",
            flush=True,
        )
        tek_kosu(cfg, config_sha256, kosu, cihaz)

    tamam, kararsiz, deger = _ilerleme(plan, cikti_koku)
    print(
        f"\nSon ilerleme: {deger}/90 "
        f"({tamam} tamam, {kararsiz} kararsız).",
        flush=True,
    )
    if deger == 90:
        print("Bu veri setinin 90 ana koşusu değerlendirildi.", flush=True)
    else:
        print(
            "Kesinti olursa aynı notebooku yeniden çalıştırın; tamamlanan "
            "koşular atlanır ve checkpoint varsa yarım koşu sürdürülür.",
            flush=True,
        )


if __name__ == "__main__":
    ana()
