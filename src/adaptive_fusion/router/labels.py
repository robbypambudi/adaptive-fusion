"""Label router (keputusan D3): metode dengan P(kelas benar) tertinggi per sampel."""

from __future__ import annotations

import numpy as np

# Urutan ini juga urutan tie-break (sama dengan notebook): mean → concat → cross_attn.
FUSION_METHODS: tuple[str, ...] = ("mean", "concat", "cross_attn")


def true_class_confidence(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """``probs[i, y[i]]`` untuk ``probs`` berbentuk ``[n, n_classes]``."""
    return probs[np.arange(len(y)), y]


def router_labels(confidence: np.ndarray) -> np.ndarray:
    """``argmax`` per baris dari ``[n, n_methods]``; seri dimenangkan metode yang lebih awal."""
    if confidence.ndim != 2 or not np.isfinite(confidence).all():
        raise ValueError("confidence harus [n, n_methods] dan finite")
    return np.argmax(confidence, axis=1).astype(np.int64)
