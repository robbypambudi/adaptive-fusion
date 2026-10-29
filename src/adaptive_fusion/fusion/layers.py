"""Blok yang dipakai bersama oleh metode fusion versi notebook."""

from __future__ import annotations

from collections.abc import Mapping

from torch import nn


def projection(in_dim: int, out_dim: int) -> nn.Sequential:
    """``Linear → LayerNorm → GELU``: projection cabang (pre-fusion) dan post-fusion di notebook.

    Lebar < 2 ditolak: LayerNorm atas satu koordinat selalu menghasilkan 0, sehingga output tidak
    bergantung pada input dan gradiennya nol (REV-008).
    """
    if out_dim < 2:
        raise ValueError(f"lebar projection harus >= 2 (LayerNorm), dapat {out_dim}")
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())


def branch_projections(input_dims: Mapping[str, int], d_branch: int) -> nn.ModuleDict:
    return nn.ModuleDict({name: projection(dim, d_branch) for name, dim in input_dims.items()})
