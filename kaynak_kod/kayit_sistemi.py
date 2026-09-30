"""Deney sonuçlarını insan tarafından okunabilir biçimde kaydeder."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def klasor_hazirla(klasor: str | Path) -> Path:
    yol = Path(klasor)
    yol.mkdir(parents=True, exist_ok=True)
    return yol


def json_kaydet(yol: str | Path, veri: dict[str, Any]) -> None:
    Path(yol).write_text(
        json.dumps(veri, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def epoch_csv_kaydet(yol: str | Path, satirlar: list[dict[str, Any]]) -> None:
    if not satirlar:
        return

    alanlar = list(satirlar[0].keys())
    with Path(yol).open("w", encoding="utf-8", newline="") as dosya:
        yazici = csv.DictWriter(dosya, fieldnames=alanlar)
        yazici.writeheader()
        yazici.writerows(satirlar)
