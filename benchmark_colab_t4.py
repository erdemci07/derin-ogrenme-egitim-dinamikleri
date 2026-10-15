"""Google Colab NVIDIA T4 üzerinde gerçek eğitim süresini ölçer.

Bu dosya bilimsel sonuç üretmez. Amaç, ana tez deneylerinden önce:
- GPU/CUDA ortamını doğrulamak,
- CNN-v1 ve CIFAR uyarlanmış ResNet-18 için gerçek epoch sürelerini ölçmek,
- GPU bellek kullanımını kaydetmek,
- geniş deney planının hesaplama bütçesini gerçek ölçüme dayandırmaktır.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from statistics import mean, pstdev

import torch
from torch import nn
from torchvision.models import resnet18

from kaynak_kod.cnn_modeli import CNNv1
from kaynak_kod.egitim import bir_epoch_egit, dogrula
from kaynak_kod.tekrarlanabilirlik import sabit_seed_ayarla
from kaynak_kod.veri_hazirlama import cifar10_yukleyicileri


def _nvidia_smi() -> str:
    try:
        sonuc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return sonuc.stdout.strip()
    except Exception as hata:
        return f"nvidia-smi okunamadi: {hata}"


def _resnet18_cifar10() -> nn.Module:
    """ImageNet ResNet-18'i 32x32 CIFAR-10 girdisine uyarlar."""
    model = resnet18(weights=None, num_classes=10)
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    return model


def _model_olustur(ad: str) -> nn.Module:
    if ad == "cnn_v1":
        return CNNv1("relu")
    if ad == "resnet18":
        return _resnet18_cifar10()
    raise ValueError(f"Bilinmeyen model: {ad}")


def _tek_model_benchmark(
    model_adi: str,
    train_loader,
    val_loader,
    cihaz: torch.device,
    warmup_epoch: int,
    olcum_epoch: int,
    seed: int,
) -> dict:
    sabit_seed_ayarla(seed)
    model = _model_olustur(model_adi).to(cihaz)
    kayip_fonksiyonu = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.0,
        weight_decay=0.0,
    )

    if cihaz.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cihaz)

    print(f"\n--- {model_adi} benchmark ---", flush=True)

    for epoch in range(1, warmup_epoch + 1):
        if cihaz.type == "cuda":
            torch.cuda.synchronize()
        baslangic = time.perf_counter()
        sonuc = bir_epoch_egit(
            model,
            train_loader,
            kayip_fonksiyonu,
            optimizer,
            cihaz,
        )
        if cihaz.type == "cuda":
            torch.cuda.synchronize()
        sure = time.perf_counter() - baslangic
        print(
            f"Warmup {epoch}/{warmup_epoch}: "
            f"{sure:.2f} sn | train_acc={sonuc['accuracy']:.4f}",
            flush=True,
        )

    train_sureleri: list[float] = []
    val_sureleri: list[float] = []
    son_train = None
    son_val = None

    for epoch in range(1, olcum_epoch + 1):
        if cihaz.type == "cuda":
            torch.cuda.synchronize()
        baslangic = time.perf_counter()

        son_train = bir_epoch_egit(
            model,
            train_loader,
            kayip_fonksiyonu,
            optimizer,
            cihaz,
        )

        if cihaz.type == "cuda":
            torch.cuda.synchronize()
        train_suresi = time.perf_counter() - baslangic

        if cihaz.type == "cuda":
            torch.cuda.synchronize()
        val_baslangic = time.perf_counter()

        son_val = dogrula(
            model,
            val_loader,
            kayip_fonksiyonu,
            cihaz,
        )

        if cihaz.type == "cuda":
            torch.cuda.synchronize()
        val_suresi = time.perf_counter() - val_baslangic

        train_sureleri.append(train_suresi)
        val_sureleri.append(val_suresi)

        print(
            f"Olcum {epoch}/{olcum_epoch}: "
            f"train={train_suresi:.2f} sn | "
            f"val={val_suresi:.2f} sn | "
            f"train_acc={son_train['accuracy']:.4f} | "
            f"val_acc={son_val['accuracy']:.4f}",
            flush=True,
        )

    toplam_epoch_sureleri = [
        a + b for a, b in zip(train_sureleri, val_sureleri)
    ]

    if cihaz.type == "cuda":
        tepe_bellek_mb = torch.cuda.max_memory_allocated(cihaz) / (1024**2)
    else:
        tepe_bellek_mb = None

    return {
        "model": model_adi,
        "warmup_epoch": warmup_epoch,
        "olcum_epoch": olcum_epoch,
        "train_seconds": train_sureleri,
        "validation_seconds": val_sureleri,
        "total_seconds_per_epoch": toplam_epoch_sureleri,
        "mean_train_seconds": mean(train_sureleri),
        "mean_validation_seconds": mean(val_sureleri),
        "mean_total_seconds_per_epoch": mean(toplam_epoch_sureleri),
        "std_total_seconds_per_epoch": (
            pstdev(toplam_epoch_sureleri)
            if len(toplam_epoch_sureleri) > 1
            else 0.0
        ),
        "peak_gpu_memory_mb": tepe_bellek_mb,
        "last_train_accuracy": son_train["accuracy"] if son_train else None,
        "last_validation_accuracy": son_val["accuracy"] if son_val else None,
    }


def _saat(epoch_sayisi: int, saniye_epoch: float) -> float:
    return epoch_sayisi * saniye_epoch / 3600.0


def ana(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-epoch", type=int, default=1)
    parser.add_argument("--olcum-epoch", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--veri-klasoru",
        default=None,
        help=(
            "Colab/Drive veri kökü. Verilirse TEZ_VERI_KLASORU olarak "
            "ayarlanır ve CIFAR-10 internetten indirilmez."
        ),
    )
    parser.add_argument(
        "--cikti",
        default="sonuclar/benchmark_colab_t4/benchmark_t4.json",
    )
    args = parser.parse_args(argv)

    if args.veri_klasoru:
        os.environ["TEZ_VERI_KLASORU"] = str(Path(args.veri_klasoru))

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA kullanilabilir degil. Colab Runtime > Change runtime type "
            "bolumunden T4 GPU secilmelidir."
        )

    cihaz = torch.device("cuda")
    gpu_adi = torch.cuda.get_device_name(0)

    if "T4" not in gpu_adi.upper():
        print(
            f"UYARI: Beklenen NVIDIA T4 yerine '{gpu_adi}' algilandi. "
            "Benchmark kaydedilecek ancak T4 sonucu olarak yorumlanmamali."
        )

    print("=" * 62, flush=True)
    print("COLAB T4 TEZ BENCHMARK", flush=True)
    print("=" * 62, flush=True)
    print("GPU:", gpu_adi, flush=True)
    print("PyTorch:", torch.__version__, flush=True)
    print("CUDA:", torch.version.cuda, flush=True)
    print("cuDNN:", torch.backends.cudnn.version(), flush=True)
    print("nvidia-smi:", _nvidia_smi(), flush=True)

    etkin_veri_koku = os.environ.get("TEZ_VERI_KLASORU", "veriler")
    print(f"Veri kaynağı: {etkin_veri_koku}", flush=True)
    print("\n[1/3] CIFAR-10 veri seti hazırlanıyor...", flush=True)
    veri_baslangic = time.perf_counter()

    train_loader, val_loader = cifar10_yukleyicileri(
        veri_klasoru="veriler",
        split_dosyasi="ayarlar/cifar10_split.json",
        batch_size=args.batch_size,
        split_seed=2026,
        loader_seed=args.seed,
        num_workers=args.num_workers,
    )

    veri_suresi = time.perf_counter() - veri_baslangic
    print(
        f"[1/3] CIFAR-10 hazır. Süre: {veri_suresi:.2f} sn | "
        f"train batch={len(train_loader)} | val batch={len(val_loader)}",
        flush=True,
    )

    print("\n[2/3] CNN-v1 benchmark başlıyor...", flush=True)
    sonuclar = []
    for model_adi in ("cnn_v1", "resnet18"):
        if model_adi == "resnet18":
            print("\n[3/3] ResNet-18 benchmark başlıyor...", flush=True)
        sonuclar.append(
            _tek_model_benchmark(
                model_adi=model_adi,
                train_loader=train_loader,
                val_loader=val_loader,
                cihaz=cihaz,
                warmup_epoch=args.warmup_epoch,
                olcum_epoch=args.olcum_epoch,
                seed=args.seed,
            )
        )

    cnn = next(x for x in sonuclar if x["model"] == "cnn_v1")
    resnet = next(x for x in sonuclar if x["model"] == "resnet18")

    cnn_saniye = cnn["mean_total_seconds_per_epoch"]
    resnet_saniye = resnet["mean_total_seconds_per_epoch"]

    tahminler = {
        "lr_pilotu_60x30_gpu_saat": _saat(60 * 30, cnn_saniye),
        "wd_pilotu_45x30_gpu_saat": _saat(45 * 30, cnn_saniye),
        "cifar10_ana_150x100_gpu_saat": _saat(150 * 100, cnn_saniye),
        "cifar10_ana_300x100_gpu_saat": _saat(300 * 100, cnn_saniye),
        "resnet18_120x100_gpu_saat": _saat(120 * 100, resnet_saniye),
        "not": (
            "MNIST, Animals-10 ve 5 Flower Types bu benchmark ile "
            "tahmin edilmemistir. Veri seti bazli benchmark yapilmadan "
            "1.035 kosunun toplam suresi kesinlestirilmeyecektir."
        ),
    }

    cikti = {
        "amac": "Tez ana deneyleri oncesi Colab T4 kapasite benchmarki",
        "bilimsel_sonuc_degil": True,
        "ortam": {
            "git_commit": os.environ.get("TEZ_GIT_COMMIT", "unknown"),
            "gpu": gpu_adi,
            "nvidia_smi": _nvidia_smi(),
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "seed": args.seed,
        },
        "benchmark": sonuclar,
        "kaba_kapasite_tahminleri": tahminler,
    }

    cikti_yolu = Path(args.cikti)
    cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
    cikti_yolu.write_text(
        json.dumps(cikti, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 62)
    print("BENCHMARK OZETI")
    print("=" * 62)
    print(
        f"CNN-v1 ortalama toplam epoch: "
        f"{cnn_saniye:.2f} ± {cnn['std_total_seconds_per_epoch']:.2f} sn"
    )
    print(
        f"ResNet-18 ortalama toplam epoch: "
        f"{resnet_saniye:.2f} ± {resnet['std_total_seconds_per_epoch']:.2f} sn"
    )
    print(
        f"CNN-v1 tepe GPU bellek: "
        f"{cnn['peak_gpu_memory_mb']:.0f} MB"
    )
    print(
        f"ResNet-18 tepe GPU bellek: "
        f"{resnet['peak_gpu_memory_mb']:.0f} MB"
    )
    print("\nKaba kapasite tahmini:")
    for anahtar, deger in tahminler.items():
        if isinstance(deger, float):
            print(f"- {anahtar}: {deger:.2f} GPU-saat")
    print(f"\nJSON cikti: {cikti_yolu}")
    print(
        "\nNOT: Bu degerler performans sonucu degildir. "
        "Yalniz hesaplama butcesi icindir."
    )


if __name__ == "__main__":
    ana()
