"""Fitur router: StandardScaler → PCA → GaussianMixture, semuanya di-fit di train (D2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, order=True)
class GMMConfig:
    pca_dim: int
    covariance_type: str
    n_components: int


def stack_modalities(features: dict[str, np.ndarray], idx: np.ndarray) -> np.ndarray:
    """Gabungkan embedding semua modalitas (urutan nama terurut) untuk sampel ``idx``."""
    return np.concatenate([features[name][idx] for name in sorted(features)], axis=1).astype(
        np.float64
    )


class GMMFeaturizer:
    """Scaler wajib: norma embedding image (DINOv2 ≈ 46) jauh lebih besar dari text (BGE = 1)."""

    def __init__(
        self,
        cfg: GMMConfig,
        seed: int,
        n_init: int = 1,
        max_iter: int = 200,
        reg_covar: float = 1e-6,
    ) -> None:
        self.cfg = cfg
        self.seed = seed
        self.n_init = n_init
        self.max_iter = max_iter
        self.reg_covar = reg_covar

    def fit(self, x_train: np.ndarray) -> GMMFeaturizer:
        if not np.isfinite(x_train).all():
            raise ValueError("fitur train mengandung NaN/inf")
        n, d = x_train.shape
        self.scaler_ = StandardScaler().fit(x_train)
        z = self.scaler_.transform(x_train)
        self.pca_ = PCA(n_components=min(self.cfg.pca_dim, n - 1, d), random_state=self.seed).fit(z)
        self.gmm_ = GaussianMixture(
            n_components=self.cfg.n_components,
            covariance_type=self.cfg.covariance_type,
            n_init=self.n_init,
            max_iter=self.max_iter,
            reg_covar=self.reg_covar,
            random_state=self.seed,
        ).fit(self.pca_.transform(z))
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        """Posterior ``P(k | x)`` berbentuk ``[n, K]`` (tiap baris berjumlah 1)."""
        z = self.pca_.transform(self.scaler_.transform(x))
        return self.gmm_.predict_proba(z).astype(np.float32)
