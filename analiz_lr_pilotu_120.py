"""Tam 120 koşuluk learning-rate pilotunu özetler ve nihai LR seçimini yapar.

İlk 60: ReLU, GELU, Mish
İkinci 60: Leaky ReLU, ELU, Swish

Toplam: 6 aktivasyon x 5 optimizer x 4 LR = 120 koşu.
Test seti kullanılmaz.
"""

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

BEKLENEN_AKTIVASYONLAR = (
    "relu",
    "leaky_relu",
    "elu",
    "gelu",
    "swish",
    "mish",
)

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


def _kayitlari_topla(klasorler: list[Path]) -> tuple[list[dict], list[dict]]:
    # Google Drive aynı run_id için aynı ada sahip birden fazla klasör
    # oluşturmuş olsa bile bilimsel olarak yalnız bir koşu sayılır.
    tamamlanan_by_run: dict[str, dict] = {}
    kararsiz_by_run: dict[str, dict] = {}

    for pilot in klasorler:
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

    return list(tamamlanan_by_run.values()), list(kararsiz_by_run.values())


def ana(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ilk-60-klasoru", required=True)
    p.add_argument("--ikinci-60-klasoru", required=True)
    p.add_argument("--cikti-klasoru", required=True)
    args = p.parse_args(argv)

    ilk = Path(args.ilk_60_klasoru)
    ikinci = Path(args.ikinci_60_klasoru)
    cikti = Path(args.cikti_klasoru)
    cikti.mkdir(parents=True, exist_ok=True)

    tamamlanan, kararsizlar = _kayitlari_topla([ilk, ikinci])

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

    if degerlendirilen != beklenen_toplam:
        raise RuntimeError(
            "Tam 120 LR pilotu henüz tamamlanmadı; nihai LR seçimi yapılmadı."
        )

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
                len(kosular) == len(BEKLENEN_AKTIVASYONLAR)
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
                    "aktivasyon_sayisi": len(BEKLENEN_AKTIVASYONLAR),
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
                f"{optimizer} için altı aktivasyonun tamamında sağlıklı kalan LR adayı yok."
            )

        # Önceden dondurulmuş basit seçim kuralı:
        # 1) en yüksek ortalama epoch-30 validation accuracy
        # 2) en düşük ortalama epoch-30 validation loss
        # 3) tam eşitlikte daha düşük LR
        adaylar.sort(
            key=lambda x: (
                -float(x["mean_final_validation_accuracy"]),
                float(x["mean_final_validation_loss"]),
                float(x["learning_rate"]),
            )
        )
        secimler[optimizer] = adaylar[0]

    _csv_yaz(cikti / "lr_aday_ozeti_120.csv", ozet_satirlari)

    secilen_lr = {
        optimizer: float(secimler[optimizer]["learning_rate"])
        for optimizer in BEKLENEN_OPTIMIZERLAR
    }

    sonuc = {
        "protokol": {
            "kapsam": "6 aktivasyon x 5 optimizer x 4 LR = 120 pilot koşusu",
            "birincil": "epoch 30 mean validation accuracy across all 6 activations",
            "ikincil": "epoch 30 mean validation loss across all 6 activations",
            "tam_esitlik": "lower learning rate",
            "kararsiz_aday": "NaN/Inf üreten optimizer-LR adayı uygun sayılmaz",
            "test_seti_kullanildi": False,
        },
        "secilen_learning_rate": secilen_lr,
        "optimizer_ayrintilari": secimler,
        "tamamlanan_kosu": len(tamamlanan),
        "kararsiz_kosu": len(kararsizlar),
        "degerlendirilen_toplam": degerlendirilen,
    }

    (cikti / "lr_secim_sonucu_120.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nNihai LR değerleri:")
    for optimizer in BEKLENEN_OPTIMIZERLAR:
        s = secimler[optimizer]
        print(
            f"- {optimizer}: {s['learning_rate']} | "
            f"mean val acc={s['mean_final_validation_accuracy']:.6f} | "
            f"mean val loss={s['mean_final_validation_loss']:.6f}"
        )

    print("\nSonuç:", cikti / "lr_secim_sonucu_120.json")


if __name__ == "__main__":
    ana()
