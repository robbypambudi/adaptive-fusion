"""Seleksi K/PCA/covariance/router di val (Q4) dan kontrol permutasi dengan seleksi berulang."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np

from adaptive_fusion.router.gmm import GMMConfig
from adaptive_fusion.router.metrics import select_per_sample
from adaptive_fusion.router.mlp import RouterConfig, RouterMLP, predict_routes, train_router

# posterior (train, val) per konfigurasi GMM; tidak bergantung label → di-cache antar permutasi.
PosteriorCache = Mapping[GMMConfig, tuple[np.ndarray, np.ndarray]]


def gmm_grid(cfg: Mapping[str, Any]) -> list[GMMConfig]:
    grid = [
        GMMConfig(pca_dim=int(p), covariance_type=str(c), n_components=int(k))
        for p, c, k in product(cfg["pca_dims"], cfg["covariance_types"], cfg["n_components"])
        if not (c == "full" and int(p) > int(cfg["full_max_pca_dim"]))
    ]
    return sorted(grid)


def router_grid(cfg: Mapping[str, Any]) -> list[RouterConfig]:
    fixed = {k: cfg[k] for k in ("dropout", "lr", "weight_decay", "max_epochs", "patience")}
    return [
        RouterConfig(hidden=int(h), class_weight=str(w), **fixed)
        for h, w in product(cfg["hidden"], cfg["class_weight"])
    ]


@dataclass
class SelectionResult:
    gmm: GMMConfig
    router: RouterConfig
    model: RouterMLP
    val_dynamic_correct: int
    val_router_correct: int
    n_val: int
    table: list[dict[str, Any]]

    @property
    def val_dynamic_acc(self) -> float:
        return self.val_dynamic_correct / self.n_val

    @property
    def val_router_acc(self) -> float:
        return self.val_router_correct / self.n_val


def run_selection(
    posteriors: PosteriorCache,
    gmm_configs: Sequence[GMMConfig],
    router_configs: Sequence[RouterConfig],
    y_router_train: np.ndarray,
    y_router_val: np.ndarray,
    pred_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    keep_table: bool = True,
) -> SelectionResult:
    """Pilih kandidat dengan akurasi Dynamic val tertinggi; seri → K terkecil, lalu urutan grid.

    ``pred_val`` = prediksi kelas model Full ``[n_methods, n_val]``.
    """
    n_methods = pred_val.shape[0]
    best: SelectionResult | None = None
    table: list[dict[str, Any]] = []
    for g in gmm_configs:
        p_train, p_val = posteriors[g]
        for r in router_configs:
            model = train_router(p_train, y_router_train, p_val, y_router_val, r, seed, n_methods)
            routes = predict_routes(model, p_val)
            dyn_correct = int((select_per_sample(pred_val, routes) == y_val).sum())
            router_correct = int((routes == y_router_val).sum())
            if keep_table:
                table.append(
                    {
                        "gmm": vars(g),
                        "router": {"hidden": r.hidden, "class_weight": r.class_weight},
                        "val_dynamic_acc": dyn_correct / len(y_val),
                        "val_router_acc": router_correct / len(y_val),
                        "val_routing_counts": np.bincount(routes, minlength=n_methods).tolist(),
                    }
                )
            better = best is None or (
                dyn_correct > best.val_dynamic_correct
                or (
                    dyn_correct == best.val_dynamic_correct
                    and g.n_components < best.gmm.n_components
                )
            )
            if better:
                best = SelectionResult(g, r, model, dyn_correct, router_correct, len(y_val), [])
    assert best is not None
    best.table = table
    return best


def permutation_null(
    posteriors: PosteriorCache,
    gmm_configs: Sequence[GMMConfig],
    router_configs: Sequence[RouterConfig],
    y_router_train: np.ndarray,
    y_router_val: np.ndarray,
    pred_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    n_permutations: int,
) -> list[dict[str, float]]:
    """Ulangi seluruh prosedur seleksi dengan label router train diacak (spec §8, GMM-006)."""
    null = []
    for r in range(n_permutations):
        rng = np.random.default_rng([seed, r])
        shuffled = rng.permutation(y_router_train)
        res = run_selection(
            posteriors,
            gmm_configs,
            router_configs,
            shuffled,
            y_router_val,
            pred_val,
            y_val,
            seed,
            keep_table=False,
        )
        null.append({"val_dynamic_acc": res.val_dynamic_acc, "val_router_acc": res.val_router_acc})
    return null


def permutation_summary(observed: float, null_scores: Sequence[float]) -> dict[str, Any]:
    scores = np.asarray(null_scores, dtype=np.float64)
    if scores.size == 0:
        return {"n_permutations": 0, "p_value": None, "null_p95": None, "exceeds_p95": None}
    p95 = float(np.percentile(scores, 95))
    return {
        "n_permutations": int(scores.size),
        "null_scores": scores.tolist(),
        "null_mean": float(scores.mean()),
        "null_p95": p95,
        "observed": float(observed),
        "exceeds_p95": bool(observed > p95),
        "p_value": float((1 + (scores >= observed).sum()) / (scores.size + 1)),
    }


def independent_expectation(y_true: np.ndarray, routes: np.ndarray, n_methods: int) -> float:
    """Akurasi router acak-independen: Σ_j p_j · q_j (bukan akurasi mayoritas)."""
    p = np.bincount(y_true, minlength=n_methods) / len(y_true)
    q = np.bincount(routes, minlength=n_methods) / len(routes)
    return float((p * q).sum())
