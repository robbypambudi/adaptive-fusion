"""Tes komponen router (T4–T6): label, GMM, MLP, seleksi, permutasi, metrik."""

from __future__ import annotations

import numpy as np
import pytest

from adaptive_fusion.router.gmm import GMMConfig, GMMFeaturizer, stack_modalities
from adaptive_fusion.router.labels import router_labels
from adaptive_fusion.router.metrics import mcnemar_exact, select_per_sample
from adaptive_fusion.router.mlp import RouterConfig, class_weights, predict_routes, train_router
from adaptive_fusion.router.selection import (
    gmm_grid,
    independent_expectation,
    permutation_null,
    permutation_summary,
    run_selection,
)


def test_router_labels_argmax_with_tie_order():
    conf = np.array([[0.9, 0.95, 0.1], [0.8, 0.8, 0.8], [0.2, 0.3, 0.3], [0.1, 0.2, 0.99]])
    np.testing.assert_array_equal(router_labels(conf), [1, 0, 1, 2])
    with pytest.raises(ValueError):
        router_labels(np.array([[np.nan, 0.1, 0.2]]))


def test_gmm_posteriors_sum_to_one_and_fit_on_train_only():
    rng = np.random.default_rng(0)
    x_train = np.concatenate([rng.normal(-3, 1, (40, 10)), rng.normal(3, 1, (40, 10))])
    fz = GMMFeaturizer(GMMConfig(pca_dim=4, covariance_type="diag", n_components=2), seed=0).fit(
        x_train
    )
    np.testing.assert_allclose(fz.scaler_.mean_, x_train.mean(0))
    post = fz.transform(rng.normal(0, 5, (15, 10)))
    assert post.shape == (15, 2)
    np.testing.assert_allclose(post.sum(1), 1.0, rtol=1e-5)
    with pytest.raises(ValueError):
        GMMFeaturizer(GMMConfig(4, "diag", 2), seed=0).fit(np.full((5, 10), np.nan))


def test_stack_modalities_sorted_order():
    feats = {"title": np.ones((3, 1)), "image": np.zeros((3, 2))}
    np.testing.assert_array_equal(stack_modalities(feats, np.array([0, 2]))[0], [0, 0, 1])


def test_gmm_grid_limits_full_covariance():
    grid = gmm_grid(
        {
            "pca_dims": [16, 64],
            "covariance_types": ["diag", "full"],
            "n_components": [2, 3],
            "full_max_pca_dim": 32,
        }
    )
    assert GMMConfig(64, "full", 2) not in grid
    assert GMMConfig(16, "full", 3) in grid
    assert len(grid) == 6


def test_router_can_overfit_small_batch():
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(4), size=12).astype(np.float32)
    y = p.argmax(1) % 3
    cfg = RouterConfig(hidden=64, dropout=0.0, lr=0.05, max_epochs=500, patience=500)
    model = train_router(p, y, p, y, cfg, seed=0, n_methods=3)
    assert (predict_routes(model, p) == y).mean() == 1.0


def test_class_weights_balanced():
    w = class_weights(np.array([0, 0, 0, 1]), 3, "balanced").numpy()
    np.testing.assert_allclose(w, [4 / 9, 4 / 3, 0.0])
    assert class_weights(np.array([0, 1]), 3, "none") is None


def selection_fixture():
    """Posterior yang memisahkan dua kelompok; metode 1 benar di kelompok A, metode 0 di B."""
    rng = np.random.default_rng(1)
    group_tr, group_va = rng.integers(0, 2, 60), rng.integers(0, 2, 40)

    def post(g, k):
        p = np.full((len(g), k), 0.05 / max(k - 1, 1), dtype=np.float32)
        p[np.arange(len(g)), g % k] = 0.95 if k > 1 else 1.0
        return p

    y_val = rng.integers(0, 2, 40)
    pred_val = np.stack(
        [
            np.where(group_va == 0, y_val, 1 - y_val),  # metode 0 benar di B
            np.where(group_va == 1, y_val, 1 - y_val),  # metode 1 benar di A
            1 - y_val,
        ]
    )
    y_router_train = np.where(group_tr == 1, 1, 0)
    y_router_val = np.where(group_va == 1, 1, 0)
    configs = [GMMConfig(4, "diag", 1), GMMConfig(4, "diag", 2), GMMConfig(4, "diag", 3)]
    posteriors = {
        g: (post(group_tr, g.n_components), post(group_va, g.n_components)) for g in configs
    }
    return posteriors, configs, y_router_train, y_router_val, pred_val, y_val


def test_selection_prefers_best_val_dynamic_then_smaller_k():
    posteriors, configs, yr_tr, yr_va, pred_val, y_val = selection_fixture()
    routers = [RouterConfig(hidden=16, dropout=0.0, lr=0.05, max_epochs=300, patience=50)]
    res = run_selection(posteriors, configs, routers, yr_tr, yr_va, pred_val, y_val, seed=0)
    assert res.val_dynamic_acc == 1.0
    assert res.gmm.n_components == 2  # K=3 juga sempurna, tapi K=2 lebih kecil
    assert len(res.table) == 3
    k1 = next(row for row in res.table if row["gmm"]["n_components"] == 1)
    assert sum(1 for c in k1["val_routing_counts"] if c > 0) == 1  # K=1 → router konstan


def test_permutation_null_repeats_full_selection(monkeypatch):
    """Setiap permutasi melatih ulang router untuk SELURUH grid GMM × router (GMM-006)."""
    from adaptive_fusion.router import selection

    posteriors, configs, yr_tr, yr_va, pred_val, y_val = selection_fixture()
    routers = [
        RouterConfig(hidden=8, dropout=0.0, lr=0.05, max_epochs=30, patience=10),
        RouterConfig(hidden=4, dropout=0.0, lr=0.05, max_epochs=30, patience=10),
    ]
    calls: list[np.ndarray] = []
    real_train = selection.train_router

    def counting_train(p_train, y_train, *args, **kwargs):
        calls.append(y_train.copy())
        return real_train(p_train, y_train, *args, **kwargs)

    monkeypatch.setattr(selection, "train_router", counting_train)
    null = permutation_null(posteriors, configs, routers, yr_tr, yr_va, pred_val, y_val, 0, 3)
    assert len(null) == 3
    assert len(calls) == 3 * len(configs) * len(routers)
    per_perm = len(configs) * len(routers)
    for r in range(3):
        labels = calls[r * per_perm : (r + 1) * per_perm]
        assert all(np.array_equal(labels[0], lab) for lab in labels)  # label sama dalam 1 permutasi
        assert sorted(labels[0].tolist()) == sorted(yr_tr.tolist())  # permutasi dari label asli
    summary = permutation_summary(1.0, [r["val_dynamic_acc"] for r in null])
    assert summary["n_permutations"] == 3


def test_k1_router_is_constant_and_equals_one_global_method():
    posteriors, configs, yr_tr, yr_va, pred_val, y_val = selection_fixture()
    k1 = [c for c in configs if c.n_components == 1]
    routers = [RouterConfig(hidden=8, dropout=0.0, lr=0.05, max_epochs=50, patience=10)]
    res = run_selection(posteriors, k1, routers, yr_tr, yr_va, pred_val, y_val, seed=0)
    routes = predict_routes(res.model, posteriors[k1[0]][1])
    assert len(set(routes.tolist())) == 1
    assert res.val_dynamic_correct == int((pred_val[routes[0]] == y_val).sum())


def test_independent_expectation():
    y = np.array([0, 0, 0, 1])  # p = [.75, .25, 0]
    routes = np.array([0, 1, 1, 1])  # q = [.25, .75, 0]
    assert independent_expectation(y, routes, 3) == pytest.approx(0.75 * 0.25 + 0.25 * 0.75)


def test_select_per_sample_and_mcnemar():
    values = np.array([[1, 2, 3], [10, 20, 30]])
    np.testing.assert_array_equal(select_per_sample(values, np.array([1, 0, 1])), [10, 2, 30])
    a = np.array([True] * 8 + [False] * 2)
    b = np.array([False] * 6 + [True] * 4)
    res = mcnemar_exact(a, b)
    assert (res["b"], res["c"]) == (6, 2)
    assert res["p_value"] == pytest.approx(2 * (1 + 8 + 28) / 256)
    assert mcnemar_exact(a, a)["p_value"] == 1.0
