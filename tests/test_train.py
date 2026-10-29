"""Tes loop training fusion (T2)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from adaptive_fusion.train import TrainConfig, predict_proba, train_fusion_model

CPU = torch.device("cpu")
SMALL = TrainConfig(
    d_branch=8, d_fusion=8, hidden=8, batch_size=16, epochs=40, lr=1e-2, patience=40
)


def learnable_data(n: int = 80):
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n)
    features = {
        "image": (rng.normal(size=(n, 6)) + 2 * y[:, None]).astype(np.float32),
        "text": rng.normal(size=(n, 4)).astype(np.float32),
        "title": rng.normal(size=(n, 5)).astype(np.float32),
    }
    return features, y.astype(np.int64)


@pytest.mark.parametrize("method", ["mean", "concat", "cross_attn"])
def test_training_learns_and_records_provenance(method):
    features, y = learnable_data()
    opt_idx, sel_idx = np.arange(0, 60), np.arange(60, 80)
    tm = train_fusion_model(method, features, y, opt_idx, sel_idx, SMALL, seed=0, device=CPU)
    probs = predict_proba(tm.model, features, sel_idx, CPU)
    assert probs.shape == (20, 2)
    np.testing.assert_allclose(probs.sum(1), 1.0, rtol=1e-5)
    assert (probs.argmax(1) == y[sel_idx]).mean() > 0.7
    np.testing.assert_array_equal(tm.optimizer_indices, opt_idx)
    np.testing.assert_array_equal(tm.selection_indices, sel_idx)
    assert 1 <= tm.best_epoch <= tm.epochs_run


def test_training_ignores_rows_outside_given_indices():
    """Baris yang tidak ada di indeks optimizer/seleksi (mis. test) boleh berisi NaN."""
    features, y = learnable_data()
    for arr in features.values():
        arr[70:] = np.nan
    tm = train_fusion_model("concat", features, y, np.arange(50), np.arange(50, 70), SMALL, 0, CPU)
    assert np.isfinite(tm.best_selection_loss)


def test_overlapping_indices_rejected():
    features, y = learnable_data()
    with pytest.raises(ValueError):
        train_fusion_model("mean", features, y, np.arange(50), np.arange(40, 60), SMALL, 0, CPU)


def test_same_seed_same_model():
    features, y = learnable_data()
    args = ("mean", features, y, np.arange(60), np.arange(60, 80), SMALL, 3, CPU)
    a, b = train_fusion_model(*args), train_fusion_model(*args)
    for (ka, va), (_, vb) in zip(
        a.model.state_dict().items(), b.model.state_dict().items(), strict=True
    ):
        torch.testing.assert_close(va, vb, msg=ka)
