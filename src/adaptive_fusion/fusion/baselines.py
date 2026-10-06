"""Baseline fusion sederhana sebagai pembanding wajib.

Metode fusion baru harus dibandingkan setidaknya dengan baseline ini
(lihat bagian "Baseline wajib" di docs/research-charter.md).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from adaptive_fusion.fusion.base import FusionModule, register_fusion


@register_fusion("concat")
class ConcatFusion(FusionModule):
    """Early/intermediate fusion: gabungkan semua fitur lalu proyeksikan linear."""

    def __init__(self, input_dims: Mapping[str, int], out_dim: int) -> None:
        super().__init__(input_dims, out_dim)
        self.proj = nn.Linear(sum(self.input_dims.values()), out_dim)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self.check_inputs(inputs)
        return self.proj(torch.cat([inputs[name] for name in self.sources], dim=-1))


@register_fusion("mean")
class MeanFusion(FusionModule):
    """Proyeksikan tiap sumber ke ``out_dim`` lalu rata-ratakan dengan bobot sama."""

    def __init__(self, input_dims: Mapping[str, int], out_dim: int) -> None:
        super().__init__(input_dims, out_dim)
        self.projs = nn.ModuleDict(
            {name: nn.Linear(dim, out_dim) for name, dim in self.input_dims.items()}
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self.check_inputs(inputs)
        projected = [self.projs[name](inputs[name]) for name in self.sources]
        return torch.stack(projected, dim=0).mean(dim=0)
