"""Kontrak dasar untuk semua modul fusion.

Setiap metode fusion baru WAJIB:
1. Mewarisi ``FusionModule``.
2. Didaftarkan dengan ``@register_fusion("nama")``.
3. Lolos ``tests/test_fusion_contract.py`` (otomatis diuji untuk semua entri registry).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping

import torch
from torch import nn


class FusionModule(nn.Module, ABC):
    """Menggabungkan representasi dari beberapa sumber/modalitas menjadi satu tensor.

    Kontrak:
    - Input: ``dict`` nama_sumber -> Tensor ``[batch, input_dims[nama_sumber]]``.
    - Output: Tensor ``[batch, out_dim]``.
    - Output tidak bergantung pada urutan key di dict input.
    - Pada mode ``eval()``, tiap sampel diproses independen dari sampel lain di batch.
    - Gradien mengalir ke semua input.
    """

    def __init__(self, input_dims: Mapping[str, int], out_dim: int) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims tidak boleh kosong")
        if out_dim <= 0:
            raise ValueError(f"out_dim harus positif, dapat {out_dim}")
        # Urutan sumber dikunci (diurutkan) agar hasil tidak bergantung urutan dict.
        self.input_dims = {name: input_dims[name] for name in sorted(input_dims)}
        self.out_dim = out_dim

    @property
    def sources(self) -> list[str]:
        return list(self.input_dims)

    def check_inputs(self, inputs: Mapping[str, torch.Tensor]) -> None:
        """Validasi nama sumber dan dimensi; panggil di awal ``forward``."""
        missing = set(self.input_dims) - set(inputs)
        extra = set(inputs) - set(self.input_dims)
        if missing or extra:
            raise KeyError(f"sumber tidak cocok: kurang={sorted(missing)}, lebih={sorted(extra)}")
        batch_sizes = {t.shape[0] for t in inputs.values()}
        if len(batch_sizes) != 1:
            raise ValueError(f"ukuran batch berbeda antar sumber: {batch_sizes}")
        for name, dim in self.input_dims.items():
            tensor = inputs[name]
            if tensor.ndim != 2 or tensor.shape[1] != dim:
                raise ValueError(f"'{name}' harus [batch, {dim}], dapat {tuple(tensor.shape)}")

    @abstractmethod
    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor: ...


FUSION_REGISTRY: dict[str, type[FusionModule]] = {}


def register_fusion(name: str) -> Callable[[type[FusionModule]], type[FusionModule]]:
    def decorator(cls: type[FusionModule]) -> type[FusionModule]:
        if name in FUSION_REGISTRY:
            raise ValueError(f"fusion '{name}' sudah terdaftar")
        if not issubclass(cls, FusionModule):
            raise TypeError(f"{cls.__name__} harus mewarisi FusionModule")
        FUSION_REGISTRY[name] = cls
        return cls

    return decorator


def build_fusion(
    name: str, input_dims: Mapping[str, int], out_dim: int, **kwargs: object
) -> FusionModule:
    if name not in FUSION_REGISTRY:
        raise KeyError(f"fusion '{name}' tidak dikenal; tersedia: {sorted(FUSION_REGISTRY)}")
    return FUSION_REGISTRY[name](input_dims, out_dim, **kwargs)
