"""Model lengkap: fusion (termasuk projection cabang) → head MLP klasifikasi."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from adaptive_fusion.fusion import build_fusion


class FusionClassifier(nn.Module):
    """Head identik untuk semua metode: ``d_fusion → hidden → GELU → Dropout → n_classes``."""

    def __init__(
        self,
        method: str,
        input_dims: Mapping[str, int],
        d_fusion: int,
        hidden: int,
        n_classes: int = 2,
        dropout: float = 0.2,
        **fusion_kwargs: Any,
    ) -> None:
        super().__init__()
        self.method = method
        self.fusion = build_fusion(method, input_dims, d_fusion, **fusion_kwargs)
        self.head = nn.Sequential(
            nn.Linear(d_fusion, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.head(self.fusion(inputs))
