"""Parity T2: fusion repo = kelas notebook (Bagian 7) bila bobotnya sama, pada fixture sintetis.

Kelas referensi di bawah disalin dari ``adaptive_fusion new.ipynb`` (sel 40).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from adaptive_fusion.models import FusionClassifier

D_IN, D_BRANCH, D_FUSION, HIDDEN = 12, 8, 8, 6


class NbBranchProjection(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, x):
        return self.proj(x)


class NbMeanFusion(nn.Module):
    def __init__(self, d_branch, d_fusion):
        super().__init__()
        self.post = nn.Sequential(nn.Linear(d_branch, d_fusion), nn.LayerNorm(d_fusion), nn.GELU())

    def forward(self, branches):
        return self.post(torch.stack(branches, dim=0).mean(dim=0))


class NbConcatFusion(nn.Module):
    def __init__(self, d_branch, d_fusion, n_branches=3):
        super().__init__()
        self.post = nn.Sequential(
            nn.Linear(d_branch * n_branches, d_fusion), nn.LayerNorm(d_fusion), nn.GELU()
        )

    def forward(self, branches):
        return self.post(torch.cat(branches, dim=-1))


class NbCrossAttentionFusion(nn.Module):
    def __init__(self, d_branch, d_fusion, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_branch, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_branch)
        self.post = nn.Sequential(nn.Linear(d_branch, d_fusion), nn.LayerNorm(d_fusion), nn.GELU())

    def forward(self, branches):
        img, title, text = branches
        query = img.unsqueeze(1)
        kv = torch.stack([title, text], dim=1)
        out, _ = self.attn(query, kv, kv, need_weights=False)
        return self.post(self.norm(out.squeeze(1)))


class NbFusionClassifier(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.p_img = NbBranchProjection(D_IN, D_BRANCH)
        self.p_title = NbBranchProjection(D_IN, D_BRANCH)
        self.p_text = NbBranchProjection(D_IN, D_BRANCH)
        self.fusion = {
            "mean": lambda: NbMeanFusion(D_BRANCH, D_FUSION),
            "concat": lambda: NbConcatFusion(D_BRANCH, D_FUSION),
            "cross_attn": lambda: NbCrossAttentionFusion(D_BRANCH, D_FUSION),
        }[method]()
        self.head = nn.Sequential(
            nn.Linear(D_FUSION, HIDDEN), nn.GELU(), nn.Dropout(0.2), nn.Linear(HIDDEN, 2)
        )

    def forward(self, img, title, text):
        return self.head(self.fusion((self.p_img(img), self.p_title(title), self.p_text(text))))


def copy_notebook_weights(nb: NbFusionClassifier, ours: FusionClassifier, method: str) -> None:
    branches = {"image": nb.p_img, "title": nb.p_title, "text": nb.p_text}
    with torch.no_grad():
        for name, src in branches.items():
            ours.fusion.branches[name].load_state_dict(src.proj.state_dict())
        ours.head.load_state_dict(nb.head.state_dict())
        post_state = nb.fusion.post.state_dict()
        if method == "concat":
            # Notebook: cat(image, title, text). Repo: urutan terurut (image, text, title).
            w = post_state["0.weight"]
            blocks = dict(zip(["image", "title", "text"], w.split(D_BRANCH, dim=1), strict=True))
            post_state["0.weight"] = torch.cat([blocks[n] for n in ours.fusion.sources], dim=1)
        ours.fusion.post.load_state_dict(post_state)
        if method == "cross_attn":
            ours.fusion.attn.load_state_dict(nb.fusion.attn.state_dict())
            ours.fusion.norm.load_state_dict(nb.fusion.norm.state_dict())


@pytest.mark.parametrize("method", ["mean", "concat", "cross_attn"])
def test_same_weights_same_output(method):
    torch.manual_seed(0)
    nb = NbFusionClassifier(method).eval()
    dims = {"image": D_IN, "title": D_IN, "text": D_IN}
    extra = {"n_heads": 4} if method == "cross_attn" else {}
    ours = FusionClassifier(method, dims, D_FUSION, HIDDEN, d_branch=D_BRANCH, **extra).eval()
    copy_notebook_weights(nb, ours, method)
    x = {k: torch.randn(5, D_IN) for k in dims}
    torch.testing.assert_close(ours(x), nb(x["image"], x["title"], x["text"]), rtol=1e-5, atol=1e-6)


def test_cross_attn_query_defaults_to_image():
    dims = {"image": 4, "title": 4, "text": 4}
    model = FusionClassifier("cross_attn", dims, 8, 4, d_branch=8, n_heads=4)
    assert model.fusion.query_source == "image"
    assert model.fusion.kv_sources == ["text", "title"]


def test_cross_attn_rejects_bad_config():
    with pytest.raises(ValueError):
        FusionClassifier("cross_attn", {"image": 4}, 8, 4, d_branch=8)
    # 2 sumber: 1 key/value → softmax selalu 1, query tidak berpengaruh (REV-002).
    with pytest.raises(ValueError, match="3 sumber"):
        FusionClassifier("cross_attn", {"image": 4, "text": 4}, 8, 4, d_branch=8)
    with pytest.raises(ValueError):
        FusionClassifier("cross_attn", {"image": 4, "text": 4}, 8, 4, d_branch=6, n_heads=4)


@pytest.mark.parametrize("method", ["mean", "concat", "cross_attn"])
def test_degenerate_widths_rejected(method):
    """Lebar 1 membuat LayerNorm menghapus seluruh informasi input (REV-008)."""
    from adaptive_fusion.fusion import build_fusion

    dims = {"image": 4, "text": 4, "title": 4}
    extra = {"n_heads": 1} if method == "cross_attn" else {}
    with pytest.raises(ValueError, match=">= 2"):
        build_fusion(method, dims, 1, d_branch=4, **extra)
    with pytest.raises(ValueError, match=">= 2"):
        build_fusion(method, dims, 4, d_branch=1, **extra)
