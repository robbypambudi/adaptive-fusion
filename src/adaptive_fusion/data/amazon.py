"""Amazon Reviews 2023 (Appliances): embedding cache dari notebook Fase 1 → dataset + split.

Dua ruang indeks (spec §4):
- *indeks terfilter* ``0..n_valid-1``: posisi setelah mask valid, dipakai untuk akses fitur & split;
- *indeks sumber* ``source_row``: nomor baris di ``appliances.csv`` (kolom ``subset_idx`` notebook).

Pakai dari terminal::

    python -m adaptive_fusion.data.amazon --config configs/data/appliances_zip1000.yaml \
        [--extract-zip datasets/amazon_review_2023-*.zip]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from adaptive_fusion.utils.artifacts import ArtifactError, sha256_file, sha256_json


@dataclass(frozen=True)
class DataConfig:
    dataset_id: str
    raw_dir: Path
    processed_dir: Path
    subset_csv: str
    embeddings: dict[str, str]
    label_min_rating: float = 4.0
    split_seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    notebook_persample_csv: str | None = None
    notebook_reference: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path, base: Path | None = None) -> DataConfig:
        base = base or Path.cwd()
        raw = yaml.safe_load(Path(path).read_text())
        split = raw.get("split", {})
        return cls(
            dataset_id=raw["dataset_id"],
            raw_dir=base / raw["raw_dir"],
            processed_dir=base / raw["processed_dir"],
            subset_csv=raw["subset_csv"],
            embeddings=dict(raw["embeddings"]),
            label_min_rating=float(raw.get("label_min_rating", 4.0)),
            split_seed=int(split.get("seed", 42)),
            test_size=float(split.get("test_size", 0.2)),
            val_size=float(split.get("val_size", 0.2)),
            notebook_persample_csv=raw.get("notebook_persample_csv"),
            notebook_reference=raw.get("notebook_reference", {}),
        )

    @property
    def split_path(self) -> Path:
        return self.processed_dir / self.dataset_id / "split.json"

    def split_params(self) -> dict[str, Any]:
        return {"seed": self.split_seed, "test_size": self.test_size, "val_size": self.val_size}


@dataclass(frozen=True)
class MultimodalData:
    dataset_id: str
    features: dict[str, np.ndarray]  # nama modalitas -> [n_valid, dim] float32
    y: np.ndarray  # [n_valid] int64
    source_row: np.ndarray  # [n_valid] int64, indeks baris di CSV sumber
    source_fingerprint: str

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def input_dims(self) -> dict[str, int]:
        return {name: arr.shape[1] for name, arr in self.features.items()}


@dataclass(frozen=True)
class DevSplit:
    """Train + val saja. Tahap T3–T6 hanya menerima objek ini, jadi tidak bisa menyentuh test."""

    train: np.ndarray
    val: np.ndarray


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray

    def dev(self) -> DevSplit:
        return DevSplit(train=self.train, val=self.val)


def load_amazon(cfg: DataConfig) -> MultimodalData:
    """Baca CSV subset + embedding cache, buang baris yang embedding-nya tidak finite."""
    csv_path = cfg.raw_dir / cfg.subset_csv
    subset = pd.read_csv(csv_path, index_col=0)
    labels = (subset["rating"].to_numpy() >= cfg.label_min_rating).astype(np.int64)

    embeddings: dict[str, np.ndarray] = {}
    # Fingerprint dikunci per modalitas + aturan label: menukar file antar modalitas atau
    # mengubah ambang rating menghasilkan fingerprint (dan run_id) baru.
    hashes: dict[str, object] = {
        "subset_csv": sha256_file(csv_path),
        "label_min_rating": cfg.label_min_rating,
    }
    for name in sorted(cfg.embeddings):
        path = cfg.raw_dir / cfg.embeddings[name]
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[0] != len(subset):
            raise ValueError(
                f"embedding '{name}' berbentuk {arr.shape}, CSV punya {len(subset)} baris"
            )
        embeddings[name] = arr
        hashes[f"embedding:{name}"] = sha256_file(path)

    valid = np.all([np.isfinite(arr).all(axis=1) for arr in embeddings.values()], axis=0)
    source_row = np.flatnonzero(valid).astype(np.int64)
    return MultimodalData(
        dataset_id=cfg.dataset_id,
        features={k: v[valid].astype(np.float32) for k, v in embeddings.items()},
        y=labels[valid],
        source_row=source_row,
        source_fingerprint=sha256_json(hashes),
    )


def make_split(y: np.ndarray, seed: int, test_size: float, val_size: float) -> Split:
    """Reproduksi persis urutan pemanggilan ``train_test_split`` di notebook (stratified)."""
    idx = np.arange(len(y))
    idx_train, idx_test, y_train, _ = train_test_split(
        idx, y, test_size=test_size, random_state=seed, stratify=y
    )
    idx_train, idx_val, _, _ = train_test_split(
        idx_train, y_train, test_size=val_size, random_state=seed, stratify=y_train
    )
    return Split(
        train=np.asarray(idx_train, dtype=np.int64),
        val=np.asarray(idx_val, dtype=np.int64),
        test=np.asarray(idx_test, dtype=np.int64),
    )


def save_split(split: Split, data: MultimodalData, params: dict[str, Any], path: Path) -> str:
    """Simpan split + pemetaan source_row + fingerprint. Mengembalikan SHA256 file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": data.dataset_id,
        "source_fingerprint": data.source_fingerprint,
        "params": params,
        "n_valid": data.n,
        "source_row": data.source_row.tolist(),
        "train": split.train.tolist(),
        "val": split.val.tolist(),
        "test": split.test.tolist(),
    }
    path.write_text(json.dumps(payload))
    return sha256_file(path)


def load_split(path: Path, data: MultimodalData) -> tuple[Split, str]:
    """Baca split.json dan tolak jika dataset/fingerprint/pemetaan tidak cocok dengan data."""
    payload = json.loads(path.read_text())
    if payload["dataset_id"] != data.dataset_id:
        raise ArtifactError(
            f"split untuk dataset '{payload['dataset_id']}', bukan '{data.dataset_id}'"
        )
    if payload["source_fingerprint"] != data.source_fingerprint:
        raise ArtifactError("fingerprint sumber berubah sejak split dibuat")
    if payload["source_row"] != data.source_row.tolist():
        raise ArtifactError("pemetaan valid -> source_row tidak cocok")
    split = Split(
        train=np.asarray(payload["train"], dtype=np.int64),
        val=np.asarray(payload["val"], dtype=np.int64),
        test=np.asarray(payload["test"], dtype=np.int64),
    )
    check_split(split, data.n)
    return split, sha256_file(path)


def check_split(split: Split, n: int) -> None:
    parts = [split.train, split.val, split.test]
    joined = np.concatenate(parts)
    if len(np.unique(joined)) != len(joined):
        raise ValueError("ada indeks yang muncul di lebih dari satu split")
    if len(joined) != n or joined.min() < 0 or joined.max() >= n:
        raise ValueError("split tidak menutup tepat semua sampel valid")


def prepare(cfg: DataConfig) -> tuple[MultimodalData, Split, str]:
    """Muat data dan split. Split dibuat sekali; run berikutnya memverifikasi file yang ada."""
    data = load_amazon(cfg)
    expected = make_split(data.y, **_split_kwargs(cfg))
    if cfg.split_path.exists():
        split, split_sha = load_split(cfg.split_path, data)
        for part in ("train", "val", "test"):
            if not np.array_equal(getattr(split, part), getattr(expected, part)):
                raise ArtifactError(
                    f"split.json bagian '{part}' berbeda dari resep split di config"
                )
    else:
        check_split(expected, data.n)
        split, split_sha = expected, save_split(expected, data, cfg.split_params(), cfg.split_path)
    return data, split, split_sha


def _split_kwargs(cfg: DataConfig) -> dict[str, Any]:
    return {"seed": cfg.split_seed, "test_size": cfg.test_size, "val_size": cfg.val_size}


def extract_from_zip(zip_path: Path, cfg: DataConfig) -> list[Path]:
    """Ekstrak file yang dibutuhkan dari zip notebook ke raw_dir.

    File yang sudah ada tidak ditimpa, tetapi isinya harus identik dengan versi di zip.
    """
    wanted = {cfg.subset_csv, *cfg.embeddings.values()}
    if cfg.notebook_persample_csv:
        wanted.add(cfg.notebook_persample_csv)
    cfg.raw_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(zip_path) as zf:
        members: dict[str, str] = {}
        for m in zf.namelist():
            name = Path(m).name
            if m.endswith("/") or name not in wanted:
                continue
            if name in members:
                raise ValueError(f"nama file ganda di zip: {members[name]} dan {m}")
            members[name] = m
        missing = wanted - set(members)
        if missing:
            raise FileNotFoundError(f"tidak ada di zip: {sorted(missing)}")
        for name in sorted(wanted):
            target = cfg.raw_dir / name
            if target.exists():
                digest = hashlib.sha256(zf.read(members[name])).hexdigest()
                if digest != sha256_file(target):
                    raise ArtifactError(f"{target} sudah ada tetapi isinya berbeda dari zip")
                continue
            with zf.open(members[name]) as src, open(target, "wb") as dst:
                dst.write(src.read())
            written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--extract-zip", type=Path, default=None)
    args = parser.parse_args()
    cfg = DataConfig.from_yaml(args.config)
    if args.extract_zip:
        for path in extract_from_zip(args.extract_zip, cfg):
            print(f"diekstrak: {path}")
    data, split, split_sha = prepare(cfg)
    print(f"{cfg.dataset_id}: {data.n} sampel valid, fingerprint {data.source_fingerprint[:12]}")
    for part in ("train", "val", "test"):
        idx = getattr(split, part)
        print(f"  {part:5s} n={len(idx):4d} kelas={np.bincount(data.y[idx], minlength=2).tolist()}")
    print(f"  split.json sha256 {split_sha[:12]} -> {cfg.split_path}")


if __name__ == "__main__":
    main()
