"""Metrik klasifikasi, prediksi Dynamic, dan uji McNemar exact."""

from __future__ import annotations

from math import comb
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, prob_pos: np.ndarray
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "n_correct": int((y_true == y_pred).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": None,
    }
    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, prob_pos))
    return metrics


def select_per_sample(values: np.ndarray, routes: np.ndarray) -> np.ndarray:
    """``values[routes[i], i]`` untuk ``values`` berbentuk ``[n_methods, n]``."""
    return values[routes, np.arange(values.shape[1])]


def mcnemar_exact(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, Any]:
    """Uji binomial exact dua sisi pada pasangan yang tidak sepakat (b: A benar & B salah)."""
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0}
    tail = sum(comb(n, k) for k in range(min(b, c) + 1)) / 2**n
    return {"b": b, "c": c, "p_value": float(min(1.0, 2 * tail))}
