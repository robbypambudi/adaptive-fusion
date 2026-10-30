"""Tes end-to-end pipeline GMM router (T3–T7) pada data sintetis kecil."""

from __future__ import annotations

import fcntl
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from adaptive_fusion.data.amazon import DevSplit, MultimodalData, Split, make_split, save_split
from adaptive_fusion.experiments.gmm_router import (
    Context,
    ExperimentConfig,
    _run_lock,
    load_confidences,
    stage_full,
    stage_oof,
    stage_select,
    stage_test,
    summarize,
)
from adaptive_fusion.utils.artifacts import ArtifactError, manifest_path, read_manifest

TINY = {
    "methods": ["mean", "concat", "cross_attn"],
    "seeds": [1, 2],
    "device": "cpu",
    "paths": {"processed": "processed", "results": "results", "checkpoints": "checkpoints"},
    "training": {
        "d_branch": 8,
        "d_fusion": 8,
        "hidden": 8,
        "dropout": 0.0,
        "n_heads": 4,
        "batch_size": 32,
        "epochs": 6,
        "lr": 0.01,
        "weight_decay": 0.0,
        "patience": 6,
    },
    "oof": {"n_folds": 3, "inner_val_size": 0.25},
    "gmm": {
        "pca_dims": [3],
        "n_components": [1, 2],
        "covariance_types": ["diag"],
        "full_max_pca_dim": 3,
        "n_init": 1,
        "max_iter": 50,
        "reg_covar": 1.0e-4,
    },
    "router": {
        "hidden": [8],
        "class_weight": ["none"],
        "dropout": 0.0,
        "lr": 0.05,
        "weight_decay": 0.0,
        "max_epochs": 30,
        "patience": 10,
    },
    "permutation": {"n_permutations": 2},
    "evaluation": {"random_router_stream": 7},
}


def synthetic_data(n: int = 120) -> MultimodalData:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, n).astype(np.int64)
    features = {
        "image": (rng.normal(size=(n, 5)) + y[:, None]).astype(np.float32),
        "text": rng.normal(size=(n, 4)).astype(np.float32),
        "title": (rng.normal(size=(n, 3)) - y[:, None]).astype(np.float32),
    }
    return MultimodalData("synthetic", features, y, np.arange(n) * 2, "fp-synthetic")


def make_context(tmp_path: Path, data: MultimodalData, cfg: dict | None = None) -> Context:
    split = make_split(data.y, seed=42, test_size=0.2, val_size=0.2)
    split_sha = save_split(
        split, data, {"seed": 42}, tmp_path / "processed" / data.dataset_id / "split.json"
    )
    return Context(ExperimentConfig.from_dict(cfg or TINY), data, split, split_sha, tmp_path)


class NoTestSplit:
    """Split tanpa akses test: membaca ``.test`` langsung melempar error."""

    def __init__(self, split: Split) -> None:
        self._dev = split.dev()

    def dev(self) -> DevSplit:
        return self._dev

    @property
    def test(self) -> np.ndarray:
        raise AssertionError("tahap dev membaca indeks test")


def hide_test_rows(ctx: Context) -> Context:
    """Fitur test = NaN, label test = -1, dan ``split.test`` melempar error.

    Tahap dev yang menyentuh test (fitur, label, atau indeks) akan gagal atau menghasilkan NaN.
    """
    features = {k: v.copy() for k, v in ctx.data.features.items()}
    for arr in features.values():
        arr[ctx.split.test] = np.nan
    y = ctx.data.y.copy()
    y[ctx.split.test] = -1
    return replace(
        ctx, data=replace(ctx.data, features=features, y=y), split=NoTestSplit(ctx.split)
    )


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("pipeline")
    torch.use_deterministic_algorithms(False)
    full_ctx = make_context(tmp_path, synthetic_data())
    dev_ctx = hide_test_rows(full_ctx)
    for seed in TINY["seeds"]:
        stage_full(dev_ctx, seed)
        stage_oof(dev_ctx, seed)
        stage_select(dev_ctx, seed)
    for seed in TINY["seeds"]:
        stage_test(full_ctx, seed)
    summary = summarize(full_ctx, TINY["seeds"])
    return full_ctx, dev_ctx, summary


def test_dev_stages_never_read_test_rows(pipeline):
    full_ctx, dev_ctx, _ = pipeline
    for seed in TINY["seeds"]:
        oof = read_manifest(dev_ctx.run_dir(dev_ctx.identity("oof", seed)))
        arrays = load_confidences(dev_ctx, oof)
        assert np.isfinite(arrays["conf_oof"]).all()
        assert np.isfinite(arrays["conf_val"]).all()
        assert not np.intersect1d(arrays["train_idx"], full_ctx.split.test).size


def test_oof_provenance_disjoint_and_complete(pipeline):
    ctx, _, _ = pipeline
    oof = read_manifest(ctx.run_dir(ctx.identity("oof", 1)))
    arrays = load_confidences(ctx, oof)  # memanggil check_oof_provenance
    predicted = np.concatenate([arrays[f"fold{f}_predict_idx"] for f in range(3)])
    assert sorted(predicted.tolist()) == sorted(ctx.split.train.tolist())
    for f in range(3):
        sel = set(arrays[f"fold{f}_selection_idx"].tolist())
        assert not sel & set(ctx.split.val.tolist())
        assert not sel & set(arrays[f"fold{f}_predict_idx"].tolist())


def test_test_result_schema_and_consistency(pipeline):
    ctx, _, summary = pipeline
    manifest = read_manifest(ctx.run_dir(ctx.identity("test", 1)))
    res = json.loads(manifest_path(manifest, "files", "result", ctx.base).read_text())
    t = res["test"]
    assert set(t["global"]) == {"mean", "concat", "cross_attn"}
    assert t["oracle"]["accuracy"] >= max(g["accuracy"] for g in t["global"].values())
    assert t["oracle"]["accuracy"] >= t["dynamic"]["accuracy"]
    assert sum(t["routing_counts"].values()) == len(ctx.split.test)
    assert res["checkpoints"]["mean"]["sha256"]
    assert set(res["upstream"]) == {"full", "oof", "select"}
    # K=1 ada di grid; jika terpilih, Dynamic = satu metode Global persis.
    if res["selected"]["gmm"]["n_components"] == 1:
        chosen = [m for m, c in t["routing_counts"].items() if c]
        assert len(chosen) == 1
        assert t["dynamic"]["accuracy"] == t["global"][chosen[0]]["accuracy"]
    assert summary["table"]["dynamic"]["accuracy"]["per_seed"][0] == t["dynamic"]["accuracy"]


def test_rerun_reuses_and_tampering_is_rejected(pipeline):
    ctx, _, _ = pipeline
    before = read_manifest(ctx.run_dir(ctx.identity("full", 1)))
    stage_full(ctx, 1)  # idempoten
    assert read_manifest(ctx.run_dir(ctx.identity("full", 1))) == before
    ckpt = manifest_path(before, "files", "checkpoint_mean", ctx.base)
    original = ckpt.read_bytes()
    ckpt.write_bytes(original + b"x")
    try:
        with pytest.raises(ArtifactError):
            stage_test(ctx, 1)
        with pytest.raises(ArtifactError):
            summarize(ctx, [1])
    finally:
        ckpt.write_bytes(original)


def test_changed_training_config_gets_new_run_and_refuses_stale_upstream(pipeline):
    ctx, _, _ = pipeline
    changed = json.loads(json.dumps(TINY))
    changed["training"]["epochs"] = 7
    ctx2 = replace(ctx, cfg=ExperimentConfig.from_dict(changed))
    assert ctx2.identity("full", 1)["run_id"] != ctx.identity("full", 1)["run_id"]
    assert ctx2.identity("select", 1)["run_id"] != ctx.identity("select", 1)["run_id"]
    with pytest.raises(ArtifactError, match="belum dijalankan"):
        stage_oof(ctx2, 1)


def test_methods_order_enforced():
    bad = json.loads(json.dumps(TINY))
    bad["methods"] = ["concat", "mean", "cross_attn"]
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict(bad)


def test_summary_records_result_hashes_and_router_rows(pipeline):
    ctx, _, summary = pipeline
    for seed in TINY["seeds"]:
        manifest = read_manifest(ctx.run_dir(ctx.identity("test", seed)))
        assert summary["result_files"][seed] == manifest["files"]["result"]
    assert set(summary["table"]["dynamic"]) >= {"accuracy", "precision", "recall", "f1"}
    assert set(summary["table"]["router"]) == {
        "router_acc",
        "changed_predictions",
        "changed_routes",
    }


def test_code_change_gives_new_run_id(pipeline, monkeypatch):
    from adaptive_fusion.experiments import gmm_router

    ctx, _, _ = pipeline
    before = ctx.identity("full", 1)["run_id"]
    monkeypatch.setattr(gmm_router, "code_fingerprint", lambda: "kode-lain")
    assert ctx.identity("full", 1)["run_id"] != before


def test_summary_has_best_global_by_val_and_paired_difference(pipeline):
    ctx, _, summary = pipeline
    for seed in TINY["seeds"]:
        manifest = read_manifest(ctx.run_dir(ctx.identity("test", seed)))
        res = json.loads(manifest_path(manifest, "files", "result", ctx.base).read_text())
        best = res["test"]["mcnemar_vs_best_global"]["best_global_by_val"]
        assert summary["best_global_by_val"][seed] == best
        i = TINY["seeds"].index(seed)
        best_acc = res["test"]["global"][best]["accuracy"]
        assert summary["table"]["best_global_by_val"]["accuracy"]["per_seed"][i] == best_acc
        diff = summary["dynamic_minus_best_global_accuracy"]["per_seed"][i]
        assert diff == pytest.approx(res["test"]["dynamic"]["accuracy"] - best_acc)


def test_reader_with_other_code_uses_explicit_run_ids(pipeline, monkeypatch):
    from adaptive_fusion.experiments import gmm_router

    ctx, _, summary = pipeline
    run_ids = [ctx.identity("test", s)["run_id"] for s in TINY["seeds"]]
    full_run = ctx.identity("full", 1)["run_id"]
    monkeypatch.setattr(gmm_router, "code_fingerprint", lambda: "kode-pembaca-lain")
    with pytest.raises(ArtifactError, match="belum dijalankan"):
        summarize(ctx, TINY["seeds"])
    again = summarize(ctx, [], run_ids)
    assert again["table"] == summary["table"]
    with pytest.raises(ArtifactError, match="stage"):
        gmm_router.load_run(ctx, "test", full_run)  # run 'full' bukan run 'test'


def test_run_lock_blocks_concurrent_stage(pipeline):
    ctx, _, _ = pipeline
    identity = ctx.identity("full", 1)
    with _run_lock(ctx.run_dir(identity)), pytest.raises(ArtifactError, match="proses lain"):
        stage_full(ctx, 1)
    stage_full(ctx, 1)  # setelah kunci dilepas: dipakai ulang seperti biasa
    assert not list(ctx.run_dir(identity).glob(".*.tmp"))


def test_lock_file_kept_so_paused_opener_cannot_lock_stale_inode(pipeline):
    """REV-009: B membuka file kunci lalu tertunda; A melepas; C mengunci; B harus gagal."""
    ctx, _, _ = pipeline
    run_dir = ctx.run_dir(ctx.identity("full", 2))
    lock_path = run_dir.parent / f".{run_dir.name}.lock"
    with _run_lock(run_dir):  # A
        pass
    assert lock_path.exists()  # file kunci tidak dihapus
    # B membuka file kunci lalu "tertunda"; C memegang kunci; B harus gagal mengunci.
    with (
        open(lock_path, "w") as paused_b,
        _run_lock(run_dir),
        pytest.raises(BlockingIOError),
    ):
        fcntl.flock(paused_b, fcntl.LOCK_EX | fcntl.LOCK_NB)


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_report_rejects_duplicate_seeds_and_mixed_protocols(pipeline):
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    report = load_script("laporan_sementara")
    ctx, _, _ = pipeline
    rid = ctx.identity("select", 1)["run_id"]
    rows = report.load_results(ctx, [], [rid, ctx.identity("select", 2)["run_id"]])
    assert [r["seed"] for r in rows] == [1, 2]
    assert [r["n_folds"] for r in rows] == [TINY["oof"]["n_folds"]] * 2
    with pytest.raises(ValueError, match="ganda"):
        report.load_results(ctx, [], [rid, rid])


def test_report_rejects_mixed_oof_protocols(pipeline):
    """REV-006: hash select & full tidak mencakup config OOF; laporan harus memeriksa run oof."""
    pytest.importorskip("reportlab")
    pytest.importorskip("matplotlib")
    report = load_script("laporan_sementara")
    ctx, dev_ctx, _ = pipeline
    changed = json.loads(json.dumps(TINY))
    changed["oof"]["n_folds"] = 2
    dev2 = replace(dev_ctx, cfg=ExperimentConfig.from_dict(changed))
    assert dev2.identity("full", 2)["run_id"] == ctx.identity("full", 2)["run_id"]
    stage_oof(dev2, 2)
    stage_select(dev2, 2)
    rid1 = ctx.identity("select", 1)["run_id"]
    rid2 = dev2.identity("select", 2)["run_id"]
    assert dev2.identity("select", 2)["config_sha256"] == ctx.identity("select", 1)["config_sha256"]
    with pytest.raises(ValueError, match="'oof'"):
        report.load_results(ctx, [], [rid1, rid2])


def test_feature_builder_resumes_from_finished_outputs(tmp_path, monkeypatch):
    """REV-007: semua keluaran sudah ada → tidak mengunduh/encode ulang dan tidak gagal."""
    pytest.importorskip("requests")
    builder = load_script("build_full_features")
    out = tmp_path / "full"
    out.mkdir()
    builder.write_atomic(
        out / "appliances.csv", lambda p: p.write_text(",rating,title,text\n0,5.0,a,b\n")
    )
    builder.write_atomic(out / "first_image_urls.json", lambda p: p.write_text('["u"]'))
    builder.write_atomic(out / "download_log.json", lambda p: p.write_text('{"ok": 1}'))
    for name in ("image", "title", "text"):
        builder.write_atomic(
            out / f"appliances_{name}_emb.npy",
            lambda p: builder.save_npy(p, np.zeros((1, 768), dtype=np.float32)),
        )

    def boom(*args, **kwargs):
        raise AssertionError("tahap yang sudah selesai dijalankan ulang")

    for fn in ("read_image_reviews", "download_images", "encode_images", "encode_texts"):
        monkeypatch.setattr(builder, fn, boom)
    monkeypatch.setattr(sys, "argv", ["x", "--zip", "unused.zip", "--out", str(out)])
    builder.main()
    assert not list(out.glob(".*.tmp"))


def test_feature_builder_resume_keeps_text_inputs_identical(tmp_path, monkeypatch):
    """REV-010: build yang terputus sebelum encode teks lalu dilanjutkan memberi input encoder
    yang sama persis dengan build tanpa jeda (teks 'NA'/'N/A'/'None'/'007' tidak berubah)."""
    pytest.importorskip("requests")
    import zipfile

    builder = load_script("build_full_features")
    reviews = [
        {"rating": 5, "title": "NA", "text": "N/A", "timestamp": 0},
        {"rating": 1, "title": "None", "text": "", "timestamp": 1},
        {"rating": 4, "title": "007", "text": "null", "timestamp": 2},
        {"rating": 2, "title": "", "text": 'line\nbreak, "quoted"', "timestamp": 3},
    ]
    zip_path = tmp_path / "src.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        lines = [
            json.dumps({**r, "images": [{"small_image_url": f"u{i}"}]})
            for i, r in enumerate(reviews)
        ]
        zf.writestr("raw/Appliances.jsonl", "\n".join(lines) + "\n")

    def run(out: Path, fail_text: bool) -> dict[str, list[str]]:
        seen: dict[str, list[str]] = {}

        def encode_texts(texts, device):
            if fail_text:
                raise KeyboardInterrupt
            seen[f"call{len(seen)}"] = list(texts)
            return np.zeros((len(texts), 768), dtype=np.float32)

        monkeypatch.setattr(builder, "download_images", lambda urls, d, w: {"ok": len(urls)})
        monkeypatch.setattr(
            builder, "encode_images", lambda n, d, dev, b: np.zeros((n, 768), np.float32)
        )
        monkeypatch.setattr(builder, "encode_texts", encode_texts)
        monkeypatch.setattr(sys, "argv", ["x", "--zip", str(zip_path), "--out", str(out)])
        builder.main()
        return seen

    fresh = run(tmp_path / "fresh", fail_text=False)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path / "resumed", fail_text=True)
    resumed = run(tmp_path / "resumed", fail_text=False)
    assert fresh == resumed
    assert fresh["call0"] == [r["title"] for r in reviews]
    assert fresh["call1"] == [r["text"] for r in reviews]
