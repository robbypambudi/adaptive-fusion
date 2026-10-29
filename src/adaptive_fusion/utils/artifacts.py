"""Identitas run, manifest, dan verifikasi hash artefak (spec §4a).

Setiap tahap pipeline menulis ke direktori ``<run_id>`` yang immutable. ``run_id`` diturunkan dari
fingerprint data, hash split, hash config tahap, ``run_id`` upstream, dan seed, sehingga perubahan
config atau upstream selalu menghasilkan direktori baru. Manifest mencatat SHA256 setiap file
keluaran dan setiap file upstream yang dibaca; konsumen menghitung ulang hash dan menolak artefak
yang tidak cocok.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"
IDENTITY_KEYS = (
    "stage",
    "dataset_id",
    "source_fingerprint",
    "split_sha256",
    "config_sha256",
    "seed",
    "run_id",
    "upstream",
)


class ArtifactError(RuntimeError):
    """Artefak hilang, berubah, atau tidak cocok dengan identitas yang diharapkan."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def make_run_id(
    *,
    source_fingerprint: str,
    split_sha256: str,
    config_sha256: str,
    upstream_run_id: str | None,
    seed: int,
) -> str:
    return sha256_json(
        {
            "source_fingerprint": source_fingerprint,
            "split_sha256": split_sha256,
            "config_sha256": config_sha256,
            "upstream_run_id": upstream_run_id,
            "seed": seed,
        }
    )[:12]


def _relative(path: Path, base: Path) -> str:
    return Path(path).resolve().relative_to(base.resolve()).as_posix()


def write_manifest(
    run_dir: Path,
    identity: Mapping[str, Any],
    files: Mapping[str, Path],
    inputs: Mapping[str, Path],
    base: Path,
) -> dict[str, Any]:
    """Tulis manifest (terakhir, setelah semua file keluaran selesai ditulis)."""
    manifest = {
        "identity": dict(identity),
        "files": {
            k: {"path": _relative(p, base), "sha256": sha256_file(p)} for k, p in files.items()
        },
        "inputs": {
            k: {"path": _relative(p, base), "sha256": sha256_file(p)} for k, p in inputs.items()
        },
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    # Atomik: manifest yang terbaca selalu utuh (tulis ke file sementara lalu rename).
    tmp = run_dir / f".{MANIFEST_NAME}.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, run_dir / MANIFEST_NAME)
    return manifest


def read_manifest(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / MANIFEST_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text())


def verify_manifest(
    manifest: Mapping[str, Any], base: Path, expected_identity: Mapping[str, Any]
) -> None:
    """Cocokkan identitas lalu hitung ulang hash semua file keluaran dan input upstream."""
    identity = manifest["identity"]
    for key in IDENTITY_KEYS:
        if identity.get(key) != expected_identity.get(key):
            raise ArtifactError(
                f"identitas '{key}' tidak cocok: manifest={identity.get(key)!r}, "
                f"diharapkan={expected_identity.get(key)!r}"
            )
    for section in ("files", "inputs"):
        for name, record in manifest[section].items():
            path = base / record["path"]
            if not path.exists():
                raise ArtifactError(f"{section}.{name} hilang: {path}")
            actual = sha256_file(path)
            if actual != record["sha256"]:
                raise ArtifactError(
                    f"{section}.{name} berubah sejak manifest ditulis: {path} "
                    f"({actual[:12]} != {record['sha256'][:12]})"
                )


def manifest_path(manifest: Mapping[str, Any], section: str, name: str, base: Path) -> Path:
    return base / manifest[section][name]["path"]
