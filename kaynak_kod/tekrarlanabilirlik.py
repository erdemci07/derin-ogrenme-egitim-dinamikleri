"""Rastgelelik kaynaklarını kontrol eden yardımcılar.

Bu dosyanın amacı, aynı seed ile yapılan deneylerin mümkün olduğunca
aynı koşullarda tekrar üretilebilmesini sağlamaktır.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def sabit_seed_ayarla(seed: int) -> None:
    """Python, NumPy ve PyTorch için ortak seed ayarlarını uygular."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Ana kontrollü deneylerde tekrar üretilebilirliği önceliklendiriyoruz.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataloader_generator(seed: int) -> torch.Generator:
    """DataLoader sıralamasını kontrol etmek için generator döndürür."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
