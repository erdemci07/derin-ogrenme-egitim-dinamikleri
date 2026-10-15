"""Yakınsama eğrilerinden önceden tanımlı özet ölçüler üretir."""

from __future__ import annotations

from collections.abc import Sequence


def normalize_aulc(degerler: Sequence[float]) -> float:
    """Epochlar arası trapez alanının epoch aralığına bölünmüş halini döndürür.

    Accuracy 0-1 ölçeğindeyse sonuç da 0-1 ölçeğinde kalır.
    Tek epoch varsa o epoch değeri döndürülür.
    """
    y = [float(v) for v in degerler]
    if not y:
        raise ValueError("AULC için en az bir değer gerekir.")
    if len(y) == 1:
        return y[0]

    alan = sum((y[i - 1] + y[i]) / 2.0 for i in range(1, len(y)))
    return alan / (len(y) - 1)


def epoch_to_own_best_fraction(
    degerler: Sequence[float],
    fraction: float = 0.95,
) -> int:
    """Kendi gözlenen en iyi değerinin belirli oranına ilk ulaşılan epoch."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction 0 ile 1 arasında olmalıdır.")

    y = [float(v) for v in degerler]
    if not y:
        raise ValueError("Yakınsama hesabı için en az bir değer gerekir.")

    hedef = max(y) * fraction
    for i, deger in enumerate(y, start=1):
        if deger >= hedef:
            return i

    return len(y)


def epoch_to_common_threshold(
    degerler: Sequence[float],
    threshold: float,
) -> int | None:
    """Önceden dondurulmuş ortak eşiğe ilk ulaşılan epoch; ulaşılmazsa None."""
    y = [float(v) for v in degerler]
    if not y:
        raise ValueError("Yakınsama hesabı için en az bir değer gerekir.")

    for i, deger in enumerate(y, start=1):
        if deger >= float(threshold):
            return i

    return None


def yakinsama_ozeti(
    validation_accuracy: Sequence[float],
    common_threshold: float | None = None,
) -> dict:
    """Bir validation accuracy eğrisi için temel yakınsama özetini üretir."""
    sonuc = {
        "aulc_normalized": normalize_aulc(validation_accuracy),
        "epoch_to_95_own_best": epoch_to_own_best_fraction(
            validation_accuracy,
            fraction=0.95,
        ),
        "own_best_validation_accuracy": max(float(x) for x in validation_accuracy),
    }

    if common_threshold is not None:
        sonuc["common_threshold"] = float(common_threshold)
        sonuc["epoch_to_common_threshold"] = epoch_to_common_threshold(
            validation_accuracy,
            common_threshold,
        )

    return sonuc
