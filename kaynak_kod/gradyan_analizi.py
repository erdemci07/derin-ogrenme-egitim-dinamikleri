"""Tezdeki teorik beklentileri deneysel mekanizma ölçümleriyle sınamak için probe araçları."""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


AKTIVASYON_TURLERI = (
    nn.ReLU,
    nn.LeakyReLU,
    nn.ELU,
    nn.GELU,
    nn.SiLU,
    nn.Mish,
)


def _aktivasyon_turevi(modul: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(modul, nn.ReLU):
        return (x > 0).to(dtype=x.dtype)

    if isinstance(modul, nn.LeakyReLU):
        return torch.where(
            x > 0,
            torch.ones_like(x),
            torch.full_like(x, float(modul.negative_slope)),
        )

    if isinstance(modul, nn.ELU):
        return torch.where(
            x > 0,
            torch.ones_like(x),
            float(modul.alpha) * torch.exp(x),
        )

    if isinstance(modul, nn.GELU):
        # Bu tezde approximate="none" kullanılmaktadır.
        phi_cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        phi_pdf = torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        return phi_cdf + x * phi_pdf

    if isinstance(modul, nn.SiLU):
        sigma = torch.sigmoid(x)
        return sigma * (1.0 + x * (1.0 - sigma))

    if isinstance(modul, nn.Mish):
        s = F.softplus(x)
        tanh_s = torch.tanh(s)
        return tanh_s + x * torch.sigmoid(x) * (1.0 - tanh_s * tanh_s)

    raise TypeError(f"Desteklenmeyen aktivasyon modülü: {type(modul).__name__}")


def _dead_unit_fraction(cikti: torch.Tensor) -> float | None:
    """Probe batch üzerinde tamamen sıfır kalan kanal/özellik oranı."""
    if cikti.ndim < 2:
        return None

    # N,C,... biçiminde batch ve uzamsal boyutlar boyunca birimi değerlendir.
    reduce_dims = [0] + list(range(2, cikti.ndim))
    dead = (cikti == 0).all(dim=reduce_dims)
    return float(dead.float().mean().item())


def gradyan_probe(
    model: nn.Module,
    kayip_fonksiyonu: nn.Module,
    probe_batch,
    cihaz: torch.device,
    *,
    turev_esigi: float = 1e-6,
    sifir_esigi: float = 1e-12,
) -> dict[str, Any]:
    """Modeli güncellemeden sabit batch üzerinde gradyan/aktivasyon ölçümleri alır."""
    onceki_mod = model.training
    model.eval()
    model.zero_grad(set_to_none=True)

    aktivasyon_kayitlari: list[dict[str, Any]] = []
    hooklar = []

    def _hook_uret(ad: str, modul: nn.Module):
        def _hook(_modul, girdiler, cikti):
            x = girdiler[0].detach()
            y = cikti.detach()
            turev = _aktivasyon_turevi(modul, x)

            kayit = {
                "layer": ad,
                "activation_type": type(modul).__name__,
                "activation_mean": float(y.mean().item()),
                "activation_std": float(y.std(unbiased=False).item()),
                "zero_fraction": float(
                    (y.abs() <= sifir_esigi).float().mean().item()
                ),
                "near_zero_derivative_fraction": float(
                    (turev.abs() <= turev_esigi).float().mean().item()
                ),
                "derivative_abs_mean": float(turev.abs().mean().item()),
                "dead_unit_fraction": (
                    _dead_unit_fraction(y) if isinstance(modul, nn.ReLU) else None
                ),
            }
            aktivasyon_kayitlari.append(kayit)

        return _hook

    for ad, modul in model.named_modules():
        if isinstance(modul, AKTIVASYON_TURLERI):
            hooklar.append(modul.register_forward_hook(_hook_uret(ad, modul)))

    girdiler, hedefler = probe_batch
    girdiler = girdiler.to(cihaz, non_blocking=True)
    hedefler = hedefler.to(cihaz, non_blocking=True)

    try:
        logitler = model(girdiler)
        kayip = kayip_fonksiyonu(logitler, hedefler)
        kayip.backward()

        katman_gradyanlari = []
        global_kare_toplam = 0.0

        for ad, parametre in model.named_parameters():
            if parametre.grad is None:
                continue
            norm = float(parametre.grad.detach().norm(2).item())
            global_kare_toplam += norm * norm
            katman_gradyanlari.append(
                {
                    "parameter": ad,
                    "gradient_l2": norm,
                    "gradient_mean": float(parametre.grad.detach().mean().item()),
                    "gradient_std": float(
                        parametre.grad.detach().std(unbiased=False).item()
                    ),
                }
            )

        sonuc = {
            "probe_loss": float(kayip.item()),
            "global_gradient_l2": math.sqrt(global_kare_toplam),
            "parameter_gradients": katman_gradyanlari,
            "activations": aktivasyon_kayitlari,
        }
    finally:
        for hook in hooklar:
            hook.remove()
        model.zero_grad(set_to_none=True)
        model.train(onceki_mod)

    return sonuc


def parametre_guncelleme_probe(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    kayip_fonksiyonu: nn.Module,
    probe_batch,
    cihaz: torch.device,
) -> dict[str, Any]:
    """Orijinal modeli değiştirmeden tek varsayımsal optimizer adımını ölçer.

    Model ve optimizer birlikte deepcopy edilir; böylece optimizer state'i ile
    kopya model parametreleri arasındaki bağ korunur.
    """
    model_kopya, optimizer_kopya = copy.deepcopy((model, optimizer))
    model_kopya.to(cihaz)
    model_kopya.train()

    girdiler, hedefler = probe_batch
    girdiler = girdiler.to(cihaz, non_blocking=True)
    hedefler = hedefler.to(cihaz, non_blocking=True)

    once = {
        ad: p.detach().clone()
        for ad, p in model_kopya.named_parameters()
        if p.requires_grad
    }

    optimizer_kopya.zero_grad(set_to_none=True)
    logitler = model_kopya(girdiler)
    kayip = kayip_fonksiyonu(logitler, hedefler)
    kayip.backward()
    optimizer_kopya.step()

    katmanlar = []
    delta_kare = 0.0
    agirlik_kare = 0.0

    for ad, p in model_kopya.named_parameters():
        if ad not in once:
            continue

        eski = once[ad]
        delta = p.detach() - eski
        delta_norm = float(delta.norm(2).item())
        agirlik_norm = float(eski.norm(2).item())
        oran = (
            delta_norm / agirlik_norm
            if agirlik_norm > 0
            else None
        )

        delta_kare += delta_norm * delta_norm
        agirlik_kare += agirlik_norm * agirlik_norm

        katmanlar.append(
            {
                "parameter": ad,
                "update_l2": delta_norm,
                "weight_l2_before": agirlik_norm,
                "relative_update": oran,
            }
        )

    global_delta = math.sqrt(delta_kare)
    global_agirlik = math.sqrt(agirlik_kare)

    return {
        "probe_loss": float(kayip.item()),
        "global_update_l2": global_delta,
        "global_weight_l2_before": global_agirlik,
        "global_relative_update": (
            global_delta / global_agirlik if global_agirlik > 0 else None
        ),
        "parameter_updates": katmanlar,
    }
