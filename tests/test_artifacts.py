"""Tes identitas run & manifest (spec §4a)."""

from __future__ import annotations

import pytest

from adaptive_fusion.utils.artifacts import (
    ArtifactError,
    make_run_id,
    read_manifest,
    verify_manifest,
    write_manifest,
)

BASE_ID = dict(
    source_fingerprint="fp", split_sha256="sp", config_sha256="cf", upstream_run_id=None, seed=1
)


def test_run_id_changes_with_every_part():
    base = make_run_id(**BASE_ID)
    for key, value in [
        ("source_fingerprint", "fp2"),
        ("split_sha256", "sp2"),
        ("config_sha256", "cf2"),
        ("upstream_run_id", "abc"),
        ("seed", 2),
    ]:
        assert make_run_id(**{**BASE_ID, key: value}) != base


def make_identity(**overrides):
    identity = {
        "stage": "full",
        "dataset_id": "d",
        "source_fingerprint": "fp",
        "split_sha256": "sp",
        "config_sha256": "cf",
        "seed": 1,
        "run_id": "r",
        "upstream": {},
    }
    return {**identity, **overrides}


def test_manifest_detects_tampering_and_identity_mismatch(tmp_path):
    out = tmp_path / "ckpt.bin"
    out.write_bytes(b"model")
    upstream = tmp_path / "up.bin"
    upstream.write_bytes(b"input")
    identity = make_identity()
    write_manifest(tmp_path / "run", identity, {"ckpt": out}, {"up": upstream}, tmp_path)
    manifest = read_manifest(tmp_path / "run")
    verify_manifest(manifest, tmp_path, identity)

    with pytest.raises(ArtifactError, match="config_sha256"):
        verify_manifest(manifest, tmp_path, make_identity(config_sha256="other"))
    upstream.write_bytes(b"changed")
    with pytest.raises(ArtifactError, match="inputs.up"):
        verify_manifest(manifest, tmp_path, identity)
    upstream.write_bytes(b"input")
    out.unlink()
    with pytest.raises(ArtifactError, match="hilang"):
        verify_manifest(manifest, tmp_path, identity)
