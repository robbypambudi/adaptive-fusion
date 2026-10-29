"""Pipeline EXP-001/EXP-002: Global Fusion vs Dynamic Fusion (GMM + MLP router).

Spesifikasi: ``docs/spec-gmm-router.md``. Tahap (masing-masing satu ``run_id`` immutable, §4a):

- ``full``    (T3): model Full per metode, latih di train, checkpoint by val. Tanpa test.
- ``oof``     (T4): confidence OOF 5-fold di train (inner-val per fold) + confidence val model Full.
- ``select``  (T5/T6): GMM + MLP router, seleksi di val, kontrol permutasi. Tanpa test.
- ``test``    (T7): satu-satunya tahap yang membaca indeks test.
- ``summary``: rekap mean ± std antar seed.

Tahap T3–T6 hanya menerima ``DevSplit`` (train + val), jadi tidak bisa menyentuh test.

Pakai dari terminal::

    python -m adaptive_fusion.experiments.gmm_router dev --config configs/experiments/...
    python -m adaptive_fusion.experiments.gmm_router test --config ...   # setelah inspeksi lolos
    python -m adaptive_fusion.experiments.gmm_router summary --config ...
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import functools
import json
import pickle
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split

from adaptive_fusion.data.amazon import DataConfig, DevSplit, MultimodalData, Split, prepare
from adaptive_fusion.router.gmm import GMMConfig, GMMFeaturizer, stack_modalities
from adaptive_fusion.router.labels import FUSION_METHODS, router_labels, true_class_confidence
from adaptive_fusion.router.metrics import classification_metrics, mcnemar_exact, select_per_sample
from adaptive_fusion.router.mlp import RouterConfig, RouterMLP, predict_routes
from adaptive_fusion.router.selection import (
    gmm_grid,
    independent_expectation,
    permutation_null,
    permutation_summary,
    router_grid,
    run_selection,
)
from adaptive_fusion.train import (
    TrainConfig,
    build_model,
    predict_proba,
    resolve_device,
    train_fusion_model,
)
from adaptive_fusion.utils.artifacts import (
    ArtifactError,
    make_run_id,
    manifest_path,
    read_manifest,
    sha256_json,
    verify_manifest,
    write_manifest,
)
from adaptive_fusion.utils.seed import seed_everything

STAGES = ("full", "oof", "select", "test")
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = {"full": None, "oof": "full", "select": "oof", "test": "select"}
SELECTION_RULE = "max val_dynamic_acc; seri -> K terkecil -> urutan grid"


@dataclass(frozen=True)
class ExperimentConfig:
    methods: tuple[str, ...]
    seeds: tuple[int, ...]
    device: str
    training: TrainConfig
    oof: dict[str, Any]
    gmm: dict[str, Any]
    router: dict[str, Any]
    permutation: dict[str, Any]
    paths: dict[str, str]
    evaluation: dict[str, Any]
    data_config: str | None = None

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ExperimentConfig:
        methods = tuple(raw["methods"])
        if methods != FUSION_METHODS:
            raise ValueError(
                f"methods harus {FUSION_METHODS} (urutan = tie-break D3), dapat {methods}"
            )
        return cls(
            methods=methods,
            seeds=tuple(int(s) for s in raw["seeds"]),
            device=str(raw.get("device", "auto")),
            training=TrainConfig.from_dict(raw["training"]),
            oof=dict(raw["oof"]),
            gmm=dict(raw["gmm"]),
            router=dict(raw["router"]),
            permutation=dict(raw["permutation"]),
            paths=dict(raw["paths"]),
            evaluation=dict(raw["evaluation"]),
            data_config=raw.get("data_config"),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def stage_config(self, stage: str) -> dict[str, Any]:
        """Bagian config yang menentukan hasil tiap tahap (masuk ke ``config_sha256``)."""
        training = {"methods": list(self.methods), "training": self.training.to_dict()}
        if stage == "full":
            return training
        if stage == "oof":
            return {**training, "oof": self.oof}
        if stage == "select":
            return {
                "gmm": self.gmm,
                "router": self.router,
                "permutation": self.permutation,
                "selection_rule": SELECTION_RULE,
            }
        if stage == "test":
            return {"evaluation": self.evaluation}
        raise KeyError(stage)


@functools.cache
def code_fingerprint() -> str:
    """SHA256 seluruh sumber paket: perubahan kode menghasilkan ``run_id`` baru (review #1)."""
    files = sorted(PACKAGE_ROOT.rglob("*.py"))
    return sha256_json({f.relative_to(PACKAGE_ROOT).as_posix(): f.read_text() for f in files})


@dataclass
class Context:
    cfg: ExperimentConfig
    data: MultimodalData
    split: Split
    split_sha256: str
    base: Path
    notebook_reference: dict[str, Any] | None = None

    @property
    def dev(self) -> DevSplit:
        return self.split.dev()

    @property
    def device(self) -> torch.device:
        return resolve_device(self.cfg.device)

    def path(self, kind: str, *parts: str) -> Path:
        return self.base / self.cfg.paths[kind] / self.data.dataset_id / Path(*parts)

    def identity(self, stage: str, seed: int) -> dict[str, Any]:
        upstream_stage = UPSTREAM[stage]
        upstream = {}
        upstream_run_id = None
        if upstream_stage is not None:
            parent = self.identity(upstream_stage, seed)
            upstream = {**parent["upstream"], upstream_stage: parent["run_id"]}
            upstream_run_id = parent["run_id"]
        config_sha = sha256_json(
            {
                "config": self.cfg.stage_config(stage),
                "code": code_fingerprint(),
                "env": {"torch": torch.__version__, "device": self.device.type},
            }
        )
        return {
            "stage": stage,
            "dataset_id": self.data.dataset_id,
            "source_fingerprint": self.data.source_fingerprint,
            "split_sha256": self.split_sha256,
            "config_sha256": config_sha,
            "seed": seed,
            "run_id": make_run_id(
                source_fingerprint=self.data.source_fingerprint,
                split_sha256=self.split_sha256,
                config_sha256=config_sha,
                upstream_run_id=upstream_run_id,
                seed=seed,
            ),
            "upstream": upstream,
        }

    def run_dir(self, identity: dict[str, Any]) -> Path:
        return self.path("results", identity["run_id"])


# ---------------------------------------------------------------------------- infrastruktur tahap


def _run_stage(
    ctx: Context,
    identity: dict[str, Any],
    output_dirs: Sequence[Path],
    produce: Callable[[], tuple[dict[str, Path], dict[str, Path]]],
) -> dict[str, Any]:
    """Jalankan tahap sekali. Jika manifest sudah ada: verifikasi lalu pakai ulang (idempoten).

    Run dikunci eksklusif (REV-003): proses lain yang memakai ``run_id`` sama gagal dengan jelas,
    jadi dua pemanggilan bersamaan tidak bisa menulis run yang sama atau mengevaluasi test dua kali.
    """
    run_dir = ctx.run_dir(identity)
    with _run_lock(run_dir):
        existing = read_manifest(run_dir)
        if existing is not None:
            verify_manifest(existing, ctx.base, identity)
            print(
                f"[{identity['stage']}] seed {identity['seed']}: "
                f"run {identity['run_id']} sudah ada, dipakai ulang"
            )
            return existing
        for d in [run_dir, *output_dirs]:
            if d.exists() and any(d.iterdir()):
                raise ArtifactError(
                    f"{d} berisi file tanpa manifest (run sebelumnya tidak selesai?). "
                    "Hapus manual lalu ulangi."
                )
        start = time.time()
        files, inputs = produce()
        manifest = write_manifest(run_dir, identity, files, inputs, ctx.base)
        print(
            f"[{identity['stage']}] seed {identity['seed']}: run {identity['run_id']} selesai "
            f"({time.time() - start:.1f}s)"
        )
        return manifest


@contextlib.contextmanager
def _run_lock(run_dir: Path):
    """Kunci eksklusif non-blocking per run (file ``.<run_id>.lock`` di samping direktori run).

    File kunci tidak pernah dihapus: menghapusnya memungkinkan dua proses mengunci inode berbeda
    untuk run yang sama (REV-009). Hanya flock-nya yang dilepas.
    """
    lock_path = run_dir.parent / f".{run_dir.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as err:
            raise ArtifactError(f"run {run_dir.name} sedang dikerjakan proses lain") from err
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _load_upstream(ctx: Context, stage: str, seed: int) -> dict[str, Any]:
    identity = ctx.identity(stage, seed)
    manifest = read_manifest(ctx.run_dir(identity))
    if manifest is None:
        raise ArtifactError(
            f"tahap '{stage}' (run {identity['run_id']}, seed {seed}) "
            "belum dijalankan dengan config ini"
        )
    verify_manifest(manifest, ctx.base, identity)
    return manifest


def load_run(ctx: Context, stage: str, run_id: str) -> dict[str, Any]:
    """Muat run dari ``run_id`` eksplisit untuk pembaca (rekap/laporan), REV-004.

    Identitas tidak dihitung ulang dari kode/device/torch saat ini; yang diverifikasi: tahap,
    dataset, fingerprint, dan split cocok dengan data, lalu hash semua file keluaran & input.
    """
    manifest = read_manifest(ctx.path("results", run_id))
    if manifest is None:
        raise ArtifactError(f"run {run_id} tidak ditemukan")
    identity = manifest["identity"]
    expected = {
        "stage": stage,
        "dataset_id": ctx.data.dataset_id,
        "source_fingerprint": ctx.data.source_fingerprint,
        "split_sha256": ctx.split_sha256,
    }
    for key, value in expected.items():
        if identity.get(key) != value:
            raise ArtifactError(
                f"run {run_id}: '{key}' = {identity.get(key)!r}, diharapkan {value!r}"
            )
    verify_manifest(manifest, ctx.base, identity)
    return manifest


def _checkpoint_inputs(ctx: Context, full_manifest: dict[str, Any]) -> dict[str, Path]:
    return {
        f"checkpoint_{m}": manifest_path(full_manifest, "files", f"checkpoint_{m}", ctx.base)
        for m in ctx.cfg.methods
    }


def _same_checkpoints(a: dict[str, Any], b: dict[str, Any], methods: Sequence[str]) -> None:
    """Pastikan dua manifest merujuk ke checkpoint Full dengan hash yang sama."""
    for m in methods:
        key = f"checkpoint_{m}"
        ha = (a["files"].get(key) or a["inputs"].get(key))["sha256"]
        hb = (b["files"].get(key) or b["inputs"].get(key))["sha256"]
        if ha != hb:
            raise ArtifactError(f"{key}: hash berbeda antar tahap ({ha[:12]} vs {hb[:12]})")


def _load_full_model(ctx: Context, path: Path, method: str) -> torch.nn.Module:
    model = build_model(method, ctx.data.input_dims, ctx.cfg.training).to(ctx.device)
    model.load_state_dict(torch.load(path, map_location=ctx.device, weights_only=True))
    model.eval()
    return model


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


# ------------------------------------------------------------------------------------- T3: full


def stage_full(ctx: Context, seed: int) -> dict[str, Any]:
    identity = ctx.identity("full", seed)
    dev = ctx.dev
    ckpt_dir = ctx.path("checkpoints", identity["run_id"], "full")
    val_dir = ctx.run_dir(identity) / "full_val"

    def produce() -> tuple[dict[str, Path], dict[str, Path]]:
        files: dict[str, Path] = {}
        reference = (ctx.notebook_reference or {}).get("best_val_loss", {})
        for method in ctx.cfg.methods:
            tm = train_fusion_model(
                method,
                ctx.data.features,
                ctx.data.y,
                dev.train,
                dev.val,
                ctx.cfg.training,
                seed,
                ctx.device,
            )
            ckpt = ckpt_dir / f"{method}_seed{seed}.pt"
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(tm.model.state_dict(), ckpt)
            probs = predict_proba(tm.model, ctx.data.features, dev.val, ctx.device)
            y_val = ctx.data.y[dev.val]
            payload = {
                **{k: identity[k] for k in ("dataset_id", "run_id", "seed")},
                "method": method,
                "best_val_loss": tm.best_selection_loss,
                "best_epoch": tm.best_epoch,
                "epochs_run": tm.epochs_run,
                "val": classification_metrics(y_val, probs.argmax(1), probs[:, 1]),
                # Pembanding kewajaran, bukan syarat lulus (GMM-001).
                "notebook_reference_best_val_loss": reference.get(method),
                "n_optimizer": len(tm.optimizer_indices),
                "n_selection": len(tm.selection_indices),
            }
            files[f"checkpoint_{method}"] = ckpt
            files[f"val_{method}"] = _write_json(val_dir / f"{method}_seed{seed}.json", payload)
            ref = f" (notebook {reference[method]:.4f})" if method in reference else ""
            print(
                f"  {method:10s} best val loss {tm.best_selection_loss:.4f}{ref} "
                f"@epoch {tm.best_epoch}, "
                f"val acc {payload['val']['accuracy']:.4f}"
            )
        return files, {}

    return _run_stage(ctx, identity, [ckpt_dir], produce)


# -------------------------------------------------------------------------------------- T4: oof


def compute_oof(ctx: Context, seed: int) -> dict[str, np.ndarray]:
    """Confidence OOF di train. Fold f: optimasi di inner-train, checkpoint di inner-val (dari
    bagian latih fold f), prediksi di fold f. Val global & test tidak dipakai (GMM-002)."""
    dev = ctx.dev
    y = ctx.data.y
    train_idx = dev.train
    n_methods = len(ctx.cfg.methods)
    conf_oof = np.full((len(train_idx), n_methods), np.nan)
    fold_of = np.full(len(train_idx), -1, dtype=np.int64)
    provenance: dict[str, np.ndarray] = {}

    skf = StratifiedKFold(n_splits=int(ctx.cfg.oof["n_folds"]), shuffle=True, random_state=seed)
    for fold, (fit_pos, hold_pos) in enumerate(skf.split(train_idx, y[train_idx])):
        fit_idx, hold_idx = train_idx[fit_pos], train_idx[hold_pos]
        opt_idx, sel_idx = train_test_split(
            fit_idx,
            test_size=float(ctx.cfg.oof["inner_val_size"]),
            random_state=seed,
            stratify=y[fit_idx],
        )
        opt_idx, sel_idx = np.sort(opt_idx), np.sort(sel_idx)
        for j, method in enumerate(ctx.cfg.methods):
            tm = train_fusion_model(
                method, ctx.data.features, y, opt_idx, sel_idx, ctx.cfg.training, seed, ctx.device
            )
            probs = predict_proba(tm.model, ctx.data.features, hold_idx, ctx.device)
            conf_oof[hold_pos, j] = true_class_confidence(probs, y[hold_idx])
        fold_of[hold_pos] = fold
        provenance[f"fold{fold}_optimizer_idx"] = opt_idx
        provenance[f"fold{fold}_selection_idx"] = sel_idx
        provenance[f"fold{fold}_predict_idx"] = np.sort(hold_idx)
        print(
            f"  fold {fold}: optimizer {len(opt_idx)}, seleksi {len(sel_idx)}, "
            f"prediksi {len(hold_idx)}"
        )

    return {"conf_oof": conf_oof, "fold_of": fold_of, **provenance}


def check_oof_provenance(arrays: dict[str, np.ndarray], dev: DevSplit) -> None:
    """Tiap indeks train diprediksi tepat sekali; per fold optimizer/seleksi/prediksi disjoint
    dan tidak menyentuh val (atau apa pun di luar train)."""
    train = set(dev.train.tolist())
    predicted: list[int] = []
    n_folds = int(arrays["fold_of"].max()) + 1
    for f in range(n_folds):
        opt = set(arrays[f"fold{f}_optimizer_idx"].tolist())
        sel = set(arrays[f"fold{f}_selection_idx"].tolist())
        pred = set(arrays[f"fold{f}_predict_idx"].tolist())
        if opt & sel or opt & pred or sel & pred:
            raise ArtifactError(f"fold {f}: indeks optimizer/seleksi/prediksi beririsan")
        if not (opt | sel | pred) <= train:
            raise ArtifactError(f"fold {f}: memakai indeks di luar train")
        predicted.extend(pred)
    if sorted(predicted) != sorted(train):
        raise ArtifactError("OOF tidak memprediksi setiap indeks train tepat sekali")
    if not np.isfinite(arrays["conf_oof"]).all():
        raise ArtifactError("conf_oof mengandung NaN")


def stage_oof(ctx: Context, seed: int) -> dict[str, Any]:
    identity = ctx.identity("oof", seed)
    full = _load_upstream(ctx, "full", seed)
    out_dir = ctx.path("processed", identity["run_id"], "router", f"seed{seed}")
    dev = ctx.dev

    def produce() -> tuple[dict[str, Path], dict[str, Path]]:
        seed_everything(seed)
        arrays = compute_oof(ctx, seed)
        check_oof_provenance(arrays, dev)
        y_val = ctx.data.y[dev.val]
        n_methods = len(ctx.cfg.methods)
        conf_val = np.zeros((len(dev.val), n_methods))
        pred_val = np.zeros((n_methods, len(dev.val)), dtype=np.int64)
        prob_pos_val = np.zeros((n_methods, len(dev.val)))
        for j, method in enumerate(ctx.cfg.methods):
            path = manifest_path(full, "files", f"checkpoint_{method}", ctx.base)
            probs = predict_proba(
                _load_full_model(ctx, path, method), ctx.data.features, dev.val, ctx.device
            )
            conf_val[:, j] = true_class_confidence(probs, y_val)
            pred_val[j] = probs.argmax(1)
            prob_pos_val[j] = probs[:, 1]
        y_router_train = router_labels(arrays["conf_oof"])
        y_router_val = router_labels(conf_val)
        out_dir.mkdir(parents=True, exist_ok=True)
        npz = out_dir / "confidences.npz"
        np.savez(
            npz,
            train_idx=dev.train,
            val_idx=dev.val,
            y_router_train=y_router_train,
            conf_val=conf_val,
            pred_val=pred_val,
            prob_pos_val=prob_pos_val,
            y_router_val=y_router_val,
            methods=np.array(ctx.cfg.methods),
            **arrays,
        )
        counts = np.bincount(y_router_train, minlength=n_methods).tolist()
        print(f"  label router train {dict(zip(ctx.cfg.methods, counts, strict=True))}")
        return {"confidences": npz}, _checkpoint_inputs(ctx, full)

    return _run_stage(ctx, identity, [out_dir], produce)


def load_confidences(ctx: Context, oof_manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    path = manifest_path(oof_manifest, "files", "confidences", ctx.base)
    with np.load(path) as npz:
        arrays = {k: npz[k] for k in npz.files}
    if not (
        np.array_equal(arrays["train_idx"], ctx.dev.train)
        and np.array_equal(arrays["val_idx"], ctx.dev.val)
    ):
        raise ArtifactError("indeks train/val di confidences.npz tidak cocok dengan split")
    check_oof_provenance(arrays, ctx.dev)
    return arrays


# --------------------------------------------------------------------------------- T5/T6: select


def stage_select(ctx: Context, seed: int) -> dict[str, Any]:
    identity = ctx.identity("select", seed)
    oof = _load_upstream(ctx, "oof", seed)
    full = _load_upstream(ctx, "full", seed)
    _same_checkpoints(oof, full, ctx.cfg.methods)
    model_dir = ctx.path("processed", identity["run_id"], "router_model", f"seed{seed}")

    def produce() -> tuple[dict[str, Path], dict[str, Path]]:
        seed_everything(seed)
        # MLP kecil: overhead thread lebih besar dari komputasinya. Dikembalikan setelah tahap ini.
        n_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            return _produce_select(ctx, seed, identity, oof, full, model_dir)
        finally:
            torch.set_num_threads(n_threads)

    return _run_stage(ctx, identity, [model_dir], produce)


def _produce_select(
    ctx: Context,
    seed: int,
    identity: dict[str, Any],
    oof: dict[str, Any],
    full: dict[str, Any],
    model_dir: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Badan tahap select; dipisah agar jumlah thread torch bisa dikembalikan di ``finally``."""
    dev = ctx.dev
    run_dir = ctx.run_dir(identity)
    arrays = load_confidences(ctx, oof)
    y_val = ctx.data.y[dev.val]
    x_train = stack_modalities(ctx.data.features, dev.train)
    x_val = stack_modalities(ctx.data.features, dev.val)

    gmm_configs = gmm_grid(ctx.cfg.gmm)
    router_configs = router_grid(ctx.cfg.router)
    gmm_opts = {k: ctx.cfg.gmm[k] for k in ("n_init", "max_iter", "reg_covar")}
    featurizers: dict[GMMConfig, GMMFeaturizer] = {}
    posteriors: dict[GMMConfig, tuple[np.ndarray, np.ndarray]] = {}
    for g in gmm_configs:
        fz = GMMFeaturizer(g, seed, **gmm_opts).fit(x_train)
        featurizers[g] = fz
        posteriors[g] = (fz.transform(x_train), fz.transform(x_val))

    common = dict(
        posteriors=posteriors,
        gmm_configs=gmm_configs,
        router_configs=router_configs,
        y_router_val=arrays["y_router_val"],
        pred_val=arrays["pred_val"],
        y_val=y_val,
        seed=seed,
    )
    sel = run_selection(y_router_train=arrays["y_router_train"], **common)
    null = permutation_null(
        y_router_train=arrays["y_router_train"],
        n_permutations=int(ctx.cfg.permutation["n_permutations"]),
        **common,
    )

    n_methods = len(ctx.cfg.methods)
    routes_val = predict_routes(sel.model, posteriors[sel.gmm][1])
    majority = int(np.bincount(arrays["y_router_train"], minlength=n_methods).argmax())
    global_val_acc = {
        m: float((arrays["pred_val"][j] == y_val).mean()) for j, m in enumerate(ctx.cfg.methods)
    }
    result = {
        **{k: identity[k] for k in ("dataset_id", "run_id", "seed", "upstream")},
        "selected": {
            "gmm": vars(sel.gmm),
            "router": sel.router.to_dict(),
            "rule": SELECTION_RULE,
        },
        "val": {
            "n": len(y_val),
            "dynamic_acc": sel.val_dynamic_acc,
            "router_acc": sel.val_router_acc,
            "global_acc": global_val_acc,
            "majority_router": {
                "method": ctx.cfg.methods[majority],
                "dynamic_acc": global_val_acc[ctx.cfg.methods[majority]],
                "router_acc": float((arrays["y_router_val"] == majority).mean()),
            },
            "router_acc_independent_expectation": independent_expectation(
                arrays["y_router_val"], routes_val, n_methods
            ),
            "routing_counts": dict(
                zip(
                    ctx.cfg.methods,
                    np.bincount(routes_val, minlength=n_methods).tolist(),
                    strict=True,
                )
            ),
        },
        "router_label_counts": {
            "train_oof": np.bincount(arrays["y_router_train"], minlength=n_methods).tolist(),
            "val": np.bincount(arrays["y_router_val"], minlength=n_methods).tolist(),
        },
        "permutation": {
            "dynamic_acc": permutation_summary(
                sel.val_dynamic_acc, [r["val_dynamic_acc"] for r in null]
            ),
            "router_acc": permutation_summary(
                sel.val_router_acc, [r["val_router_acc"] for r in null]
            ),
            "note": "Setiap permutasi mengulang seluruh seleksi (GMM-006). "
            "Diagnostik, bukan tes lulus.",
        },
        "candidates": sel.table,
    }

    model_dir.mkdir(parents=True, exist_ok=True)
    featurizer_path = model_dir / "featurizer.pkl"
    with open(featurizer_path, "wb") as f:
        pickle.dump(featurizers[sel.gmm], f)
    router_path = model_dir / "router.pt"
    torch.save(sel.model.state_dict(), router_path)
    result_path = _write_json(run_dir / "router_selection" / f"seed{seed}.json", result)
    perm = result["permutation"]["dynamic_acc"]
    print(
        f"  terpilih {vars(sel.gmm)} {sel.router.hidden}/{sel.router.class_weight}: "
        f"val Dynamic {sel.val_dynamic_acc:.4f}, router {sel.val_router_acc:.4f}; "
        f"global val {global_val_acc}; permutasi p={perm['p_value']}"
    )
    inputs = {
        "confidences": manifest_path(oof, "files", "confidences", ctx.base),
        **_checkpoint_inputs(ctx, full),
    }
    return {
        "selection": result_path,
        "featurizer": featurizer_path,
        "router": router_path,
    }, inputs


# ------------------------------------------------------------------------------------- T7: test


def stage_test(ctx: Context, seed: int) -> dict[str, Any]:
    """Satu-satunya tahap yang membaca ``ctx.split.test``. Semua angka test dihitung di sini."""
    identity = ctx.identity("test", seed)
    full = _load_upstream(ctx, "full", seed)
    oof = _load_upstream(ctx, "oof", seed)
    select = _load_upstream(ctx, "select", seed)
    _same_checkpoints(full, oof, ctx.cfg.methods)
    _same_checkpoints(full, select, ctx.cfg.methods)
    run_dir = ctx.run_dir(identity)

    def produce() -> tuple[dict[str, Path], dict[str, Path]]:
        seed_everything(seed)
        methods = ctx.cfg.methods
        n_methods = len(methods)
        test_idx = ctx.split.test
        y_test = ctx.data.y[test_idx]
        arrays = load_confidences(ctx, oof)
        selection = json.loads(manifest_path(select, "files", "selection", ctx.base).read_text())

        preds = np.zeros((n_methods, len(test_idx)), dtype=np.int64)
        prob_pos = np.zeros((n_methods, len(test_idx)))
        conf_test = np.zeros((len(test_idx), n_methods))
        for j, m in enumerate(methods):
            path = manifest_path(full, "files", f"checkpoint_{m}", ctx.base)
            probs = predict_proba(
                _load_full_model(ctx, path, m), ctx.data.features, test_idx, ctx.device
            )
            preds[j], prob_pos[j] = probs.argmax(1), probs[:, 1]
            conf_test[:, j] = true_class_confidence(probs, y_test)

        with open(manifest_path(select, "files", "featurizer", ctx.base), "rb") as f:
            featurizer: GMMFeaturizer = pickle.load(f)  # artefak sendiri, hash diverifikasi
        router_cfg = RouterConfig(**selection["selected"]["router"])
        router = RouterMLP(
            featurizer.cfg.n_components, router_cfg.hidden, n_methods, router_cfg.dropout
        )
        router.load_state_dict(
            torch.load(manifest_path(select, "files", "router", ctx.base), weights_only=True)
        )
        routes = predict_routes(
            router, featurizer.transform(stack_modalities(ctx.data.features, test_idx))
        )

        correct = preds == y_test  # [n_methods, n_test]
        dyn_pred = select_per_sample(preds, routes)
        dyn_correct = dyn_pred == y_test

        # Global terbaik dipilih dengan akurasi val (seri → urutan metode), bukan test.
        val_acc = selection["val"]["global_acc"]
        best_j = max(range(n_methods), key=lambda j: (val_acc[methods[j]], -j))
        majority = int(np.bincount(arrays["y_router_train"], minlength=n_methods).argmax())
        rng = np.random.default_rng([seed, int(ctx.cfg.evaluation["random_router_stream"])])
        random_routes = rng.integers(0, n_methods, size=len(test_idx))

        def routed(r: np.ndarray) -> dict[str, Any]:
            pred = select_per_sample(preds, r)
            return classification_metrics(y_test, pred, select_per_sample(prob_pos, r))

        y_router_test = router_labels(conf_test)
        result = {
            **{k: identity[k] for k in ("dataset_id", "run_id", "seed", "upstream")},
            "source_fingerprint": ctx.data.source_fingerprint,
            "split_sha256": ctx.split_sha256,
            "config_sha256": {s: sha256_json(ctx.cfg.stage_config(s)) for s in STAGES},
            "code_fingerprint": code_fingerprint(),
            "environment": {"torch": torch.__version__, "device": ctx.device.type},
            "checkpoints": {m: full["files"][f"checkpoint_{m}"] for m in methods},
            "selected": selection["selected"],
            "val": selection["val"],
            "test": {
                "global": {
                    m: classification_metrics(y_test, preds[j], prob_pos[j])
                    for j, m in enumerate(methods)
                },
                "dynamic": classification_metrics(
                    y_test, dyn_pred, select_per_sample(prob_pos, routes)
                ),
                "majority_router": {
                    "method": methods[majority],
                    **routed(np.full(len(test_idx), majority)),
                },
                "random_router": routed(random_routes),
                "oracle": {
                    "accuracy": float(correct.any(axis=0).mean()),
                    "n_correct": int(correct.any(axis=0).sum()),
                    "note": "batas atas; memakai label test, hanya referensi",
                },
                "router_acc": float((routes == y_router_test).mean()),
                "routing_counts": dict(
                    zip(methods, np.bincount(routes, minlength=n_methods).tolist(), strict=True)
                ),
                "router_label_counts": dict(
                    zip(
                        methods,
                        np.bincount(y_router_test, minlength=n_methods).tolist(),
                        strict=True,
                    )
                ),
                "changed_vs_best_global": {
                    "predictions": int((dyn_pred != preds[best_j]).sum()),
                    "routes": int((routes != best_j).sum()),
                },
                "mcnemar_vs_best_global": {
                    "best_global_by_val": methods[best_j],
                    **mcnemar_exact(dyn_correct, correct[best_j]),
                },
            },
        }
        path = _write_json(run_dir / "gmm_router" / f"seed{seed}.json", result)
        t = result["test"]
        print(
            f"  test: Dynamic {t['dynamic']['accuracy']:.4f} | "
            + " ".join(f"{m} {t['global'][m]['accuracy']:.4f}" for m in methods)
            + f" | oracle {t['oracle']['accuracy']:.4f}"
        )
        inputs = {
            "confidences": manifest_path(oof, "files", "confidences", ctx.base),
            "selection": manifest_path(select, "files", "selection", ctx.base),
            "featurizer": manifest_path(select, "files", "featurizer", ctx.base),
            "router": manifest_path(select, "files", "router", ctx.base),
            **_checkpoint_inputs(ctx, full),
        }
        return {"result": path}, inputs

    return _run_stage(ctx, identity, [], produce)


# ----------------------------------------------------------------------------------- summary


ROWS = (
    "dynamic",
    "global_mean",
    "global_concat",
    "global_cross_attn",
    "majority_router",
    "random_router",
    "oracle",
)


def summarize(
    ctx: Context, seeds: Sequence[int], run_ids: Sequence[str] | None = None
) -> dict[str, Any]:
    """Rekap mean ± std antar seed dari file hasil T7 yang terverifikasi.

    ``run_ids`` (opsional): run T7 eksplisit; dipakai saat kode/device pembaca berbeda dari
    produsen (REV-004). Tanpa itu, run dicari dari identitas config & kode saat ini.
    """
    if run_ids:
        manifests = [load_run(ctx, "test", rid) for rid in run_ids]
        seeds = [m["identity"]["seed"] for m in manifests]
    else:
        manifests = [_load_upstream(ctx, "test", seed) for seed in seeds]
    if len(set(seeds)) != len(seeds):
        raise ArtifactError(f"seed ganda dalam rekap: {seeds}")
    results = []
    result_files: dict[int, dict[str, str]] = {}
    protocol = None
    for seed, manifest in zip(seeds, manifests, strict=True):
        result_files[seed] = manifest["files"]["result"]
        res = json.loads(manifest_path(manifest, "files", "result", ctx.base).read_text())
        identity = manifest["identity"]
        current = (
            identity["dataset_id"],
            identity["source_fingerprint"],
            identity["split_sha256"],
            json.dumps(res["config_sha256"], sort_keys=True),
            res["code_fingerprint"],
            json.dumps(res["environment"], sort_keys=True),
        )
        if protocol is None:
            protocol = current
        elif current != protocol:
            raise ArtifactError(f"seed {seed}: dataset/split/protokol berbeda dari seed lain")
        results.append(res)

    def row_values(res: dict[str, Any], row: str, metric: str) -> float | None:
        t = res["test"]
        if row == "best_global_by_val":
            best = t["mcnemar_vs_best_global"]["best_global_by_val"]
            return t["global"][best].get(metric)
        if row == "router":
            if metric == "router_acc":
                return t["router_acc"]
            return t["changed_vs_best_global"].get(metric.removeprefix("changed_"))
        if row.startswith("global_"):
            return t["global"][row.removeprefix("global_")].get(metric)
        return t[row].get(metric)

    table: dict[str, dict[str, Any]] = {}
    metrics = ("accuracy", "precision", "recall", "f1", "roc_auc")
    router_metrics = ("router_acc", "changed_predictions", "changed_routes")
    for row in (*ROWS, "best_global_by_val", "router"):
        table[row] = {}
        for metric in router_metrics if row == "router" else metrics:
            values = [row_values(r, row, metric) for r in results]
            if any(v is None for v in values):
                continue
            arr = np.asarray(values, dtype=np.float64)
            table[row][metric] = {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "per_seed": arr.tolist(),
            }
    diffs = np.asarray(
        [
            r["test"]["dynamic"]["accuracy"]
            - r["test"]["global"][r["test"]["mcnemar_vs_best_global"]["best_global_by_val"]][
                "accuracy"
            ]
            for r in results
        ]
    )
    summary = {
        "dataset_id": ctx.data.dataset_id,
        # Pembanding utama charter: Dynamic vs Global terbaik yang dipilih di val (REV-001).
        "best_global_by_val": {
            r["seed"]: r["test"]["mcnemar_vs_best_global"]["best_global_by_val"] for r in results
        },
        "dynamic_minus_best_global_accuracy": {
            "mean": float(diffs.mean()),
            "std": float(diffs.std(ddof=1)) if len(diffs) > 1 else 0.0,
            "per_seed": diffs.tolist(),
        },
        "seeds": list(seeds),
        "run_ids": {r["seed"]: r["run_id"] for r in results},
        # Hash file hasil per seed: angka laporan dapat ditelusuri ke file yang terverifikasi.
        "result_files": result_files,
        "code_fingerprint": results[0]["code_fingerprint"],
        "table": table,
        "mcnemar_vs_best_global": {r["seed"]: r["test"]["mcnemar_vs_best_global"] for r in results},
        "routing_counts": {r["seed"]: r["test"]["routing_counts"] for r in results},
    }
    path = _write_json(ctx.path("results", "gmm_router_summary.json"), summary)
    print(f"rekap -> {path}")
    print("| baris | accuracy (mean ± std) |\n|---|---|")
    for row in (*ROWS, "best_global_by_val"):
        acc = table[row].get("accuracy")
        if acc:
            print(f"| {row} | {acc['mean']:.4f} ± {acc['std']:.4f} |")
    return summary


# --------------------------------------------------------------------------------------- CLI


STAGE_FUNCS = {"full": stage_full, "oof": stage_oof, "select": stage_select, "test": stage_test}


def build_context(config_path: Path, base: Path | None = None) -> Context:
    base = base or Path.cwd()
    cfg = ExperimentConfig.from_yaml(config_path)
    if cfg.data_config is None:
        raise KeyError("config eksperimen butuh 'data_config'")
    data_cfg = DataConfig.from_yaml(base / cfg.data_config, base)
    data, split, split_sha = prepare(data_cfg)
    return Context(cfg, data, split, split_sha, base, data_cfg.notebook_reference)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline GMM-based fusion router")
    parser.add_argument("command", choices=[*STAGES, "dev", "summary"])
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/gmm_router.yaml"))
    parser.add_argument("--seed", type=int, action="append", help="default: semua seed di config")
    parser.add_argument(
        "--run-id", action="append", help="summary: run T7 eksplisit (default: dari identitas)"
    )
    args = parser.parse_args()

    ctx = build_context(args.config)
    seeds = args.seed or list(ctx.cfg.seeds)
    print(f"{ctx.data.dataset_id}: device {ctx.device}, seeds {seeds}")
    if args.command == "summary":
        summarize(ctx, seeds, args.run_id)
        return
    stages = ["full", "oof", "select"] if args.command == "dev" else [args.command]
    for seed in seeds:
        for stage in stages:
            STAGE_FUNCS[stage](ctx, seed)


if __name__ == "__main__":
    main()
