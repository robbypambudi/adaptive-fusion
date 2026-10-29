"""MLP router: posterior GMM ``[K]`` → logit per metode fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class RouterConfig:
    hidden: int = 32
    class_weight: str = "none"  # "none" | "balanced"
    dropout: float = 0.1
    lr: float = 1e-2
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RouterMLP(nn.Module):
    def __init__(self, n_inputs: int, hidden: int, n_methods: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_inputs, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_methods),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def class_weights(y: np.ndarray, n_classes: int, mode: str) -> torch.Tensor | None:
    """``balanced`` = n / (n_classes · count); kelas tanpa sampel diberi bobot 0."""
    if mode == "none":
        return None
    if mode != "balanced":
        raise ValueError(f"class_weight tidak dikenal: {mode}")
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    weights = np.where(counts > 0, len(y) / (n_classes * np.maximum(counts, 1)), 0.0)
    return torch.as_tensor(weights, dtype=torch.float32)


def train_router(
    p_train: np.ndarray,
    y_train: np.ndarray,
    p_val: np.ndarray,
    y_val: np.ndarray,
    cfg: RouterConfig,
    seed: int,
    n_methods: int,
) -> RouterMLP:
    """Full-batch AdamW di CPU; early stopping by loss val (tanpa bobot), ambil state terbaik."""
    torch.manual_seed(seed)
    model = RouterMLP(p_train.shape[1], cfg.hidden, n_methods, cfg.dropout)
    x_tr, y_tr = torch.as_tensor(p_train), torch.as_tensor(y_train)
    x_va, y_va = torch.as_tensor(p_val), torch.as_tensor(y_val)
    train_loss = nn.CrossEntropyLoss(weight=class_weights(y_train, n_methods, cfg.class_weight))
    val_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best, best_state, wait = float("inf"), None, 0
    for _ in range(cfg.max_epochs):
        model.train()
        optimizer.zero_grad()
        train_loss(model(x_tr), y_tr).backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            loss = val_loss(model(x_va), y_va).item()
        if loss < best:
            best, wait = loss, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model


@torch.no_grad()
def predict_routes(model: RouterMLP, posteriors: np.ndarray) -> np.ndarray:
    model.eval()
    return model(torch.as_tensor(posteriors)).argmax(dim=1).numpy().astype(np.int64)
