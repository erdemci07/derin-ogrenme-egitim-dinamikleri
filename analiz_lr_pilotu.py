"""Tamamlanan 60 koşuluk learning-rate pilotunu özetler ve LR seçimini yapar."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


BEKLENEN_OPTIMIZERLAR = (
    "sgd",
    "momentum_sgd",
    "rmsprop",
    "adam",
    "adamw",
)

BEKLENEN_AKTIVASYONLAR = ("relu", "gelu", "mish")

BEKLENEN_LR = {
    "sgd": (0.003, 0.01, 0.03, 0.1),
    "momentum_sgd": (0.003, 0.01, 0.03, 0.1),
    "rmsprop": (0.0001, 0.0003, 0.001, 0.003),
    "adam": (0.0001, 0.0003, 0.001, 0.003),
    "adamw": (0.0001, 0.0003, 0.001, 0.003),
}


def _json_oku(yol: Path) -> dict:
    return json.loads(yol.read_text(encoding="utf-8"))


def _csv_yaz(yol: Path, satirlar: list[dict]) -> None:
    if not satirlar:
        return
    yol.parent.mkdir(parents=True, exist_ok=True)
    with yol.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(satirlar[0].keys()))
        w.writeheader()
        w.writerows(satirlar)


def ana(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pilot-klasoru", required=True)
    p.add_argument("--cikti-klasoru", required=True)
    args = p.parse_args(argv)

    pilot = Path(args.pilot_klasoru)
    cikti = Path(args.cikti_klasoru)
    cikti.mkdir(parents=True, exist_ok=True)

    # Google Drive aynı ada sahip klasörler oluşturabildiği için kayıtları
    # run_id bazında tekilleştiriyoruz. Sağlıklı tamamlanmış kayıt varsa önceliklidir.
    tamamlanan_by_run = {}
    kararsiz_by_run = {}

    for run_dir in sorted(pilot.iterdir() if pilot.exists() else []):
        if not run_dir.is_dir():
            continue

        tamam = run_dir / "tamamlandi.json"
        kararsiz = run_dir / "kararsiz.json"

        if tamam.exists():
            veri = _json_oku(tamam)
            veri["run_klasoru"] = run_dir.name
            run_id = veri["run_id"]
            tamamlanan_by_run[run_id] = veri
            kararsiz_by_run.pop(run_id, None)
        elif kararsiz.exists():
            veri = _json_oku(kararsiz)
            veri["run_klasoru"] = run_dir.name
            run_id = veri["run_id"]
            if run_id not in tamamlanan_by_run:
                kararsiz_by_run.setdefault(run_id, veri)

    tamamlanan = list(tamamlanan_by_run.values())
    kararsizlar = list(kararsiz_by_run.values())

    beklenen_toplam = (
        len(BEKLENEN_OPTIMIZERLAR)
        * len(BEKLENEN_AKTIVASYONLAR)
        * 4
    )

    degerlendirilen = len(tamamlanan) + len(kararsizlar)
    print(
        f"Değerlendirilen: {degerlendirilen}/{beklenen_toplam} "
        f"({len(tamamlanan)} sağlıklı, {len(kararsizlar)} kararsız/uygunsuz)"
    )

    if len(tamamlanan) + len(kararsizlar) < beklenen_toplam:
        raise RuntimeError(
            "LR pilotu henüz tamamlanmadı; seçim analizi yapılmadı."
        )

    # Her optimizer-LR adayı için üç temsilci aktivasyonun tamamı sağlıklı olmalı.
    gruplar = defaultdict(list)
    for r in tamamlanan:
        gruplar[(r["optimizer"], float(r["learning_rate"]))].append(r)

    ozet_satirlari = []
    secimler = {}

    for optimizer in BEKLENEN_OPTIMIZERLAR:
        adaylar = []

        for lr in BEKLENEN_LR[optimizer]:
            kosular = gruplar.get((optimizer, float(lr)), [])
            aktivasyonlar = {r["aktivasyon"] for r in kosular}
            uygun = (
                len(kosular) == 3
                and aktivasyonlar == set(BEKLENEN_AKTIVASYONLAR)
                and all(
                    math.isfinite(float(r["final_validation_accuracy"]))
                    and math.isfinite(float(r["final_validation_loss"]))
                    for r in kosular
                )
            )

            if uygun:
                acc = [float(r["final_validation_accuracy"]) for r in kosular]
                loss = [float(r["final_validation_loss"]) for r in kosular]
                satir = {
                    "optimizer": optimizer,
                    "learning_rate": float(lr),
                    "uygun": True,
                    "aktivasyon_sayisi": 3,
                    "mean_final_validation_accuracy": mean(acc),
                    "std_final_validation_accuracy": pstdev(acc),
                    "mean_final_validation_loss": mean(loss),
                    "std_final_validation_loss": pstdev(loss),
                }
                adaylar.append(satir)
            else:
                satir = {
                    "optimizer": optimizer,
                    "learning_rate": float(lr),
                    "uygun": False,
                    "aktivasyon_sayisi": len(kosular),
                    "mean_final_validation_accuracy": "",
                    "std_final_validation_accuracy": "",
                    "mean_final_validation_loss": "",
                    "std_final_validation_loss": "",
                }

            ozet_satirlari.append(satir)

        if not adaylar:
            raise RuntimeError(
                f"{optimizer} için üç aktivasyonun da tamamlandığı geçerli LR adayı yok."
            )

        # Önceden belirlenen seçim:
        # 1) en yüksek ortalama epoch-30 val accuracy
        # 2) en düşük ortalama epoch-30 val loss
        # 3) tam eşitlikte daha düşük LR
        adaylar.sort(
            key=lambda x: (
                -float(x["mean_final_validation_accuracy"]),
                float(x["mean_final_validation_loss"]),
                float(x["learning_rate"]),
            )
        )
        secimler[optimizer] = adaylar[0]

    _csv_yaz(cikti / "lr_aday_ozeti.csv", ozet_satirlari)

    secilen_lr = {
        optimizer: float(secimler[optimizer]["learning_rate"])
        for optimizer in BEKLENEN_OPTIMIZERLAR
    }

    # Ortak yakınsama eşiği: seçilmiş LR'lerdeki 15 temsilci koşunun
    # own-best değerlerinin minimumunun %95'i. Bu, ana deneyden önce dondurulur.
    secili_kosular = []
    for r in tamamlanan:
        if math.isclose(
            float(r["learning_rate"]),
            secilen_lr[r["optimizer"]],
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            secili_kosular.append(r)

    if len(secili_kosular) != 15:
        raise RuntimeError(
            f"Ortak eşik için 15 seçilmiş koşu bekleniyordu, {len(secili_kosular)} bulundu."
        )

    min_own_best = min(
        float(r["own_best_validation_accuracy"]) for r in secili_kosular
    )
    ortak_esik = 0.95 * min_own_best

    sonuc = {
        "protokol": {
            "birincil": "epoch 30 mean validation accuracy across ReLU/GELU/Mish",
            "ikincil": "epoch 30 mean validation loss",
            "tam_esitlik": "lower learning rate",
            "aulc_lr_seciminde_kullanildi": False,
        },
        "secilen_learning_rate": secilen_lr,
        "optimizer_ayrintilari": secimler,
        "ortak_yakinsama_esigi": {
            "kural": (
                "0.95 x minimum own-best validation accuracy among the "
                "15 selected-LR representative pilot runs"
            ),
            "minimum_own_best": min_own_best,
            "threshold": ortak_esik,
            "ana_deneyden_once_dondur": True,
        },
        "tamamlanan_kosu": len(tamamlanan),
        "kararsiz_kosu": len(kararsizlar),
    }

    (cikti / "lr_secim_sonucu.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSeçilen LR değerleri:")
    for optimizer in BEKLENEN_OPTIMIZERLAR:
        s = secimler[optimizer]
        print(
            f"- {optimizer}: {s['learning_rate']} | "
            f"mean val acc={s['mean_final_validation_accuracy']:.6f} | "
            f"mean val loss={s['mean_final_validation_loss']:.6f}"
        )

    print(f"\nOrtak yakınsama eşiği: {ortak_esik:.6f}")
    print("Sonuç:", cikti / "lr_secim_sonucu.json")


if __name__ == "__main__":
    ana()
