"""Baseline fusion sebagai pembanding wajib (versi notebook Fase 1, keputusan Q1).

Setiap sumber diproyeksikan ke ``d_branch`` (Linear → LayerNorm → GELU), digabung, lalu
diproyeksikan ke ``out_dim`` (Linear → LayerNorm → GELU). Lihat ``docs/konteks-notebook.md``.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch

from adaptive_fusion.fusion.base import FusionModule, register_fusion
from adaptive_fusion.fusion.layers import branch_projections, projection


@register_fusion("concat")
class ConcatFusion(FusionModule):
    """Gabungkan vektor cabang (urutan sumber terurut) lalu post-projection."""

    def __init__(
        self, input_dims: Mapping[str, int], out_dim: int, d_branch: int | None = None
    ) -> None:
        super().__init__(input_dims, out_dim)
        d_branch = d_branch or out_dim
        self.branches = branch_projections(self.input_dims, d_branch)
        self.post = projection(d_branch * len(self.input_dims), out_dim)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self.check_inputs(inputs)
        return self.post(torch.cat([self.branches[n](inputs[n]) for n in self.sources], dim=-1))


@register_fusion("mean")
class MeanFusion(FusionModule):
    """Rata-rata elemen vektor cabang (bobot sama) lalu post-projection."""

    def __init__(
        self, input_dims: Mapping[str, int], out_dim: int, d_branch: int | None = None
    ) -> None:
        super().__init__(input_dims, out_dim)
        d_branch = d_branch or out_dim
        self.branches = branch_projections(self.input_dims, d_branch)
        self.post = projection(d_branch, out_dim)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self.check_inputs(inputs)
        projected = [self.branches[n](inputs[n]) for n in self.sources]
        return self.post(torch.stack(projected, dim=0).mean(dim=0))
