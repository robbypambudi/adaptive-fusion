"""Training satu model fusion dengan seleksi checkpoint pada indeks yang ditentukan pemanggil.

Pemanggil menentukan ``optimizer_idx`` (untuk gradien) dan ``selection_idx`` (untuk memilih
checkpoint). Keduanya wajib disjoint dan dicatat sebagai provenance (spec §5).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import numpy as np
import torch
from torch import nn

from adaptive_fusion.models import FusionClassifier
from adaptive_fusion.utils.seed import seed_everything


@dataclass(frozen=True)
class TrainConfig:
    d_branch: int = 512
    d_fusion: int = 512
    hidden: int = 256
    dropout: float = 0.2
    n_heads: int = 4
    batch_size: int = 128
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 300

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise KeyError(f"key training tidak dikenal: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def fusion_kwargs(self, method: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"d_branch": self.d_branch}
        if method == "cross_attn":
            kwargs["n_heads"] = self.n_heads
        return kwargs


@dataclass
class TrainedModel:
    model: FusionClassifier
    best_selection_loss: float
    best_epoch: int
    epochs_run: int
    optimizer_indices: np.ndarray
    selection_indices: np.ndarray


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(
    method: str, input_dims: dict[str, int], cfg: TrainConfig, n_classes: int = 2
) -> FusionClassifier:
    return FusionClassifier(
        method,
        input_dims,
        d_fusion=cfg.d_fusion,
        hidden=cfg.hidden,
        n_classes=n_classes,
        dropout=cfg.dropout,
        **cfg.fusion_kwargs(method),
    )


def to_tensors(
    features: dict[str, np.ndarray], idx: np.ndarray, device: torch.device
) -> dict[str, torch.Tensor]:
    return {name: torch.as_tensor(arr[idx], device=device) for name, arr in features.items()}


@torch.no_grad()
def _mean_loss(
    model: nn.Module, x: dict[str, torch.Tensor], y: torch.Tensor, criterion: nn.Module
) -> float:
    model.eval()
    return criterion(model(x), y).item()


def train_fusion_model(
    method: str,
    features: dict[str, np.ndarray],
    y: np.ndarray,
    optimizer_idx: np.ndarray,
    selection_idx: np.ndarray,
    cfg: TrainConfig,
    seed: int,
    device: torch.device,
) -> TrainedModel:
    """AdamW + CrossEntropy seperti notebook; simpan state dengan loss seleksi terendah."""
    optimizer_idx = np.asarray(optimizer_idx, dtype=np.int64)
    selection_idx = np.asarray(selection_idx, dtype=np.int64)
    if np.intersect1d(optimizer_idx, selection_idx).size:
        raise ValueError("indeks optimizer dan indeks seleksi checkpoint beririsan")

    seed_everything(seed)
    input_dims = {name: arr.shape[1] for name, arr in features.items()}
    model = build_model(method, input_dims, cfg).to(device)  # biner (head 2 kelas)

    x_opt = to_tensors(features, optimizer_idx, device)
    y_opt = torch.as_tensor(y[optimizer_idx], device=device)
    x_sel = to_tensors(features, selection_idx, device)
    y_sel = torch.as_tensor(y[selection_idx], device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)

    best_loss, best_epoch, best_state, wait = float("inf"), -1, None, 0
    epochs_run = 0
    for epoch in range(cfg.epochs):
        model.train()
        order = torch.randperm(len(optimizer_idx), generator=generator).to(device)
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            optimizer.zero_grad()
            loss = criterion(model({k: v[batch] for k, v in x_opt.items()}), y_opt[batch])
            loss.backward()
            optimizer.step()
        epochs_run = epoch + 1

        sel_loss = _mean_loss(model, x_sel, y_sel, criterion)
        if sel_loss < best_loss:
            best_loss, best_epoch, wait = sel_loss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return TrainedModel(
        model=model,
        best_selection_loss=best_loss,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        optimizer_indices=optimizer_idx,
        selection_indices=selection_idx,
    )


@torch.no_grad()
def predict_proba(
    model: nn.Module,
    features: dict[str, np.ndarray],
    idx: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    """Probabilitas softmax ``[len(idx), n_classes]`` dalam ``eval()`` + ``no_grad()``."""
    model.eval()
    idx = np.asarray(idx, dtype=np.int64)
    out = []
    for start in range(0, len(idx), batch_size):
        x = to_tensors(features, idx[start : start + batch_size], device)
        out.append(torch.softmax(model(x), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 2), dtype=np.float32)
