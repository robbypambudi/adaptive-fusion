"""Cross-attention fusion versi notebook: satu sumber jadi query, sisanya key/value."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from adaptive_fusion.fusion.base import FusionModule, register_fusion
from adaptive_fusion.fusion.layers import branch_projections, projection


@register_fusion("cross_attn")
class CrossAttentionFusion(FusionModule):
    """Query = ``query_source`` (default: sumber pertama hasil urut, = ``image`` di data Amazon).

    Tiap sumber satu token, jadi attention menjadi weighted sum dari token key/value
    (keterbatasan yang sudah dicatat di notebook).
    """

    def __init__(
        self,
        input_dims: Mapping[str, int],
        out_dim: int,
        d_branch: int | None = None,
        n_heads: int = 4,
        query_source: str | None = None,
    ) -> None:
        super().__init__(input_dims, out_dim)
        if len(self.input_dims) < 3:
            # Dengan 1 key/value, bobot softmax selalu 1: output tidak bergantung pada query dan
            # gradien ke query nol (melanggar kontrak FusionModule). Lihat REV-002.
            raise ValueError("cross_attn butuh minimal 3 sumber (1 query + >= 2 key/value)")
        d_branch = d_branch or out_dim
        if d_branch % n_heads != 0:
            raise ValueError(f"d_branch ({d_branch}) harus habis dibagi n_heads ({n_heads})")
        self.query_source = query_source or self.sources[0]
        if self.query_source not in self.input_dims:
            raise KeyError(f"query_source '{self.query_source}' bukan salah satu sumber")
        self.kv_sources = [name for name in self.sources if name != self.query_source]
        self.branches = branch_projections(self.input_dims, d_branch)
        self.attn = nn.MultiheadAttention(d_branch, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_branch)
        self.post = projection(d_branch, out_dim)

    def forward(self, inputs: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self.check_inputs(inputs)
        query = self.branches[self.query_source](inputs[self.query_source]).unsqueeze(1)
        kv = torch.stack([self.branches[n](inputs[n]) for n in self.kv_sources], dim=1)
        out, _ = self.attn(query, kv, kv, need_weights=False)
        return self.post(self.norm(out.squeeze(1)))
