"""Deney sonuçlarını güvenli ve insan tarafından okunabilir biçimde kaydeder."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def klasor_hazirla(klasor: str | Path) -> Path:
    yol = Path(klasor)
    yol.mkdir(parents=True, exist_ok=True)
    return yol


def _atomik_degistir(gecici: Path, hedef: Path) -> None:
    gecici.flush if False else None
    os.replace(gecici, hedef)


def json_kaydet(yol: str | Path, veri: dict[str, Any]) -> None:
    hedef = Path(yol)
    hedef.parent.mkdir(parents=True, exist_ok=True)
    gecici = hedef.with_suffix(hedef.suffix + ".tmp")
    gecici.write_text(
        json.dumps(veri, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(gecici, hedef)


def epoch_csv_kaydet(yol: str | Path, satirlar: list[dict[str, Any]]) -> None:
    if not satirlar:
        return

    hedef = Path(yol)
    hedef.parent.mkdir(parents=True, exist_ok=True)
    gecici = hedef.with_suffix(hedef.suffix + ".tmp")
    alanlar = list(satirlar[0].keys())

    with gecici.open("w", encoding="utf-8", newline="") as dosya:
        yazici = csv.DictWriter(dosya, fieldnames=alanlar)
        yazici.writeheader()
        yazici.writerows(satirlar)
        dosya.flush()
        os.fsync(dosya.fileno())

    os.replace(gecici, hedef)
