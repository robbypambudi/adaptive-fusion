"""Seed global agar run dapat direproduksi."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Set seed untuk ``random``, NumPy, dan PyTorch (CPU & CUDA).

    ``deterministic=True`` meminta PyTorch memakai algoritma deterministik
    (bisa sedikit lebih lambat; operasi yang tidak punya versi deterministik
    hanya memunculkan peringatan).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002  (RNG global yang dipakai banyak library)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
