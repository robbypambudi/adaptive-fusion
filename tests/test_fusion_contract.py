"""Uji kontrak yang WAJIB dilewati setiap metode fusion di registry.

Metode baru otomatis ikut diuji begitu didaftarkan dengan ``@register_fusion``.
Jangan melemahkan tes ini agar metode baru lolos; perbaiki metodenya.
"""

from __future__ import annotations

import pytest
import torch

from adaptive_fusion.fusion import FUSION_REGISTRY, build_fusion

INPUT_DIMS = {"modal_a": 8, "modal_b": 5, "modal_c": 3}
OUT_DIM = 4
BATCH = 6


@pytest.fixture(params=sorted(FUSION_REGISTRY))
def fusion(request: pytest.FixtureRequest):
    torch.manual_seed(0)
    return build_fusion(request.param, INPUT_DIMS, OUT_DIM)


def make_inputs(batch: int = BATCH, requires_grad: bool = False) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(1)
    return {
        name: torch.randn(batch, dim, generator=gen, requires_grad=requires_grad)
        for name, dim in INPUT_DIMS.items()
    }


def test_registry_has_baselines():
    assert {"concat", "mean"} <= set(FUSION_REGISTRY)


def test_output_shape_and_finite(fusion):
    out = fusion(make_inputs())
    assert out.shape == (BATCH, OUT_DIM)
    assert torch.isfinite(out).all()


def test_batch_of_one(fusion):
    fusion.eval()
    assert fusion(make_inputs(batch=1)).shape == (1, OUT_DIM)


def test_independent_of_dict_order(fusion):
    fusion.eval()
    inputs = make_inputs()
    reversed_inputs = dict(reversed(list(inputs.items())))
    torch.testing.assert_close(fusion(inputs), fusion(reversed_inputs))


def test_samples_independent_in_eval(fusion):
    """Sampel ke-i tidak boleh dipengaruhi sampel lain dalam batch (cegah kebocoran)."""
    fusion.eval()
    inputs = make_inputs()
    full = fusion(inputs)
    for i in range(BATCH):
        single = fusion({k: v[i : i + 1] for k, v in inputs.items()})
        torch.testing.assert_close(full[i : i + 1], single, rtol=1e-5, atol=1e-6)


def test_deterministic_in_eval(fusion):
    fusion.eval()
    inputs = make_inputs()
    torch.testing.assert_close(fusion(inputs), fusion(inputs))


def test_gradient_reaches_every_source(fusion):
    inputs = make_inputs(requires_grad=True)
    fusion(inputs).sum().backward()
    for name, tensor in inputs.items():
        assert tensor.grad is not None, f"tidak ada gradien ke '{name}'"
        assert tensor.grad.abs().sum() > 0, f"gradien ke '{name}' nol semua"


def test_rejects_missing_source(fusion):
    inputs = make_inputs()
    inputs.pop("modal_b")
    with pytest.raises(KeyError):
        fusion(inputs)


def test_rejects_wrong_dim(fusion):
    inputs = make_inputs()
    inputs["modal_a"] = torch.randn(BATCH, INPUT_DIMS["modal_a"] + 1)
    with pytest.raises(ValueError):
        fusion(inputs)
