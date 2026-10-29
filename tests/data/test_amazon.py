"""Tes loader & split Amazon (T1): data sintetis + data nyata jika sudah diekstrak."""

from __future__ import annotations

import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from adaptive_fusion.data.amazon import (
    DataConfig,
    extract_from_zip,
    load_amazon,
    load_split,
    make_split,
    prepare,
)
from adaptive_fusion.utils.artifacts import ArtifactError

REPO = Path(__file__).resolve().parents[2]
REAL_CONFIG = REPO / "configs/data/appliances_zip1000.yaml"


def write_synthetic_raw(raw: Path, n: int = 60, nan_rows: tuple[int, ...] = (3, 10, 41)) -> None:
    rng = np.random.default_rng(0)
    raw.mkdir(parents=True)
    ratings = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 5.0], n // 6)
    pd.DataFrame({"rating": ratings, "title": "t", "text": "x"}).to_csv(raw / "subset.csv")
    for name, dim in {"image": 6, "title": 4, "text": 5}.items():
        arr = rng.normal(size=(n, dim)).astype(np.float32)
        if name == "image":
            arr[list(nan_rows)] = np.nan
        np.save(raw / f"{name}.npy", arr)


def synthetic_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        dataset_id="synthetic",
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        subset_csv="subset.csv",
        embeddings={"image": "image.npy", "title": "title.npy", "text": "text.npy"},
    )


def test_load_filters_invalid_and_maps_source_rows(tmp_path):
    write_synthetic_raw(tmp_path / "raw")
    data = load_amazon(synthetic_config(tmp_path))
    assert data.n == 57
    assert set(data.source_row.tolist()) == set(range(60)) - {3, 10, 41}
    assert all(np.isfinite(v).all() for v in data.features.values())
    assert data.input_dims == {"image": 6, "text": 5, "title": 4}
    ratings = np.tile([1.0, 2.0, 3.0, 4.0, 5.0, 5.0], 10)
    np.testing.assert_array_equal(data.y, (ratings[data.source_row] >= 4).astype(np.int64))


def test_split_disjoint_stratified_and_matches_recipe(tmp_path):
    write_synthetic_raw(tmp_path / "raw")
    cfg = synthetic_config(tmp_path)
    data, split, sha = prepare(cfg)
    parts = [split.train, split.val, split.test]
    assert sum(len(p) for p in parts) == data.n
    assert len(np.unique(np.concatenate(parts))) == data.n
    expected = make_split(data.y, seed=42, test_size=0.2, val_size=0.2)
    np.testing.assert_array_equal(split.test, expected.test)
    # run kedua memverifikasi file yang sama
    _, _, sha2 = prepare(cfg)
    assert sha == sha2


def test_split_rejects_changed_source(tmp_path):
    write_synthetic_raw(tmp_path / "raw")
    cfg = synthetic_config(tmp_path)
    prepare(cfg)
    arr = np.load(cfg.raw_dir / "text.npy")
    arr[0, 0] += 1.0
    np.save(cfg.raw_dir / "text.npy", arr)
    with pytest.raises(ArtifactError):
        load_split(cfg.split_path, load_amazon(cfg))


@pytest.mark.skipif(
    not (REPO / "data/raw/amazon_review_2023/appliances.csv").exists(),
    reason="data zip belum diekstrak (make data)",
)
def test_real_zip_run_matches_notebook(tmp_path):
    cfg = DataConfig.from_yaml(REAL_CONFIG, REPO)
    data = load_amazon(cfg)
    split = make_split(data.y, cfg.split_seed, cfg.test_size, cfg.val_size)
    assert data.n == 484
    assert [len(split.train), len(split.val), len(split.test)] == [309, 78, 97]
    assert np.bincount(data.y[split.train]).tolist() == [64, 245]
    assert np.bincount(data.y[split.val]).tolist() == [16, 62]
    assert np.bincount(data.y[split.test]).tolist() == [20, 77]
    # Cek identitas sampel saja (indeks sumber), isi CSV per-sampel tidak dipakai.
    persample = pd.read_csv(cfg.raw_dir / cfg.notebook_persample_csv)
    assert (persample["test_pos"].to_numpy() == np.arange(len(persample))).all()
    np.testing.assert_array_equal(data.source_row[split.test], persample["subset_idx"].to_numpy())


def test_fingerprint_tracks_modality_mapping_and_label_rule(tmp_path):
    write_synthetic_raw(tmp_path / "raw")
    cfg = synthetic_config(tmp_path)
    base = load_amazon(cfg).source_fingerprint
    swapped = replace(cfg, embeddings={**cfg.embeddings, "title": "text.npy", "text": "title.npy"})
    assert load_amazon(swapped).source_fingerprint != base
    assert load_amazon(replace(cfg, label_min_rating=3.0)).source_fingerprint != base


def test_extract_from_zip_checks_duplicates_and_existing_content(tmp_path):
    write_synthetic_raw(tmp_path / "src")
    cfg = synthetic_config(tmp_path)
    zpath = tmp_path / "a.zip"
    with zipfile.ZipFile(zpath, "w") as zf:
        for f in (tmp_path / "src").iterdir():
            zf.write(f, f"folder/{f.name}")
    assert len(extract_from_zip(zpath, cfg)) == 4
    assert extract_from_zip(zpath, cfg) == []  # sudah ada & identik
    (cfg.raw_dir / "subset.csv").write_text("diubah")
    with pytest.raises(ArtifactError):
        extract_from_zip(zpath, cfg)
    with zipfile.ZipFile(zpath, "a") as zf:
        zf.write(tmp_path / "src" / "image.npy", "lain/image.npy")
    with pytest.raises(ValueError, match="ganda"):
        extract_from_zip(zpath, cfg)
