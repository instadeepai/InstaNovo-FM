"""Tests for spectral rescue pair discovery."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl

from scripts.downstream.discover_spectral_rescue_pairs import (
    _project_pair_candidates,
    build_manifest,
    discover_pairs,
)


def test_project_pair_candidates_picks_eligible_variant():
    sequences = ["PEPA"] * 120 + ["PEPA[UNIMOD:35]"] * 60 + ["ZZZZZ"] * 250
    unmodified = ["PEPA"] * 180 + ["ZZZZZ"] * 250

    eligible, diagnostics = _project_pair_candidates(
        "PXD1",
        sequences,
        unmodified,
        min_base_spectra=101,
        min_modified_spectra=50,
        min_negative_candidates=200,
    )

    assert len(eligible) == 1
    assert eligible[0]["base_sequence"] == "PEPA"
    assert eligible[0]["modified_sequence"] == "PEPA[UNIMOD:35]"
    assert diagnostics[0]["eligible"] is True


def test_build_manifest_one_pair_per_project_shape():
    pairs = [
        {
            "pair_id": "pxd_a",
            "project_id": "PXD1",
            "base_sequence": "PEPA",
            "modified_sequence": "PEPA[UNIMOD:35]",
            "notes": "auto",
        }
    ]
    manifest = build_manifest(
        pairs,
        project_order=["PXD1", "PXD2"],
        source_glob="/tmp/*.parquet",
        stats_json=None,
        min_base_spectra=101,
        min_modified_spectra=50,
        min_negative_candidates=200,
    )
    assert manifest["scope"] == "multi_base"
    assert manifest["discovery"]["projects"] == ["PXD1", "PXD2"]
    assert len(manifest["pairs"]) == 1


def _write_discovery_shard(path: Path, rows: list[dict[str, str]]) -> None:
    pl.DataFrame(rows).write_parquet(path)


def test_discover_pairs_collects_one_project_at_a_time(tmp_path: Path):
    shard_a = tmp_path / "a_valid.parquet"
    shard_b = tmp_path / "b_valid.parquet"
    rows = []
    for i in range(120):
        rows.append(
            {
                "usi": f"mzspec:PXD1:run:scan:{i}:PEPA/2",
                "sequence": "PEPA",
                "unmodified_peptide": "PEPA",
            }
        )
    for i in range(60):
        rows.append(
            {
                "usi": f"mzspec:PXD1:run:scan:{120 + i}:PEPAOX/2",
                "sequence": "PEPA[UNIMOD:35]",
                "unmodified_peptide": "PEPA",
            }
        )
    for i in range(250):
        rows.append(
            {
                "usi": f"mzspec:PXD1:run:scan:{180 + i}:ZZZZZ/2",
                "sequence": "ZZZZZ",
                "unmodified_peptide": "ZZZZZ",
            }
        )
    _write_discovery_shard(shard_a, rows)
    _write_discovery_shard(
        shard_b,
        [
            {
                "usi": "mzspec:PXD2:run:scan:1:PEPB/2",
                "sequence": "PEPB",
                "unmodified_peptide": "PEPB",
            }
        ],
    )

    pairs, diagnostics = discover_pairs(
        [shard_a, shard_b],
        project_order=["PXD1", "PXD2"],
        min_base_spectra=101,
        min_modified_spectra=50,
        min_negative_candidates=200,
        max_pairs=10,
        max_variants_per_backbone=1,
        one_pair_per_project=True,
    )

    assert len(pairs) == 1
    assert pairs[0]["project_id"] == "PXD1"
    assert pairs[0]["base_sequence"] == "PEPA"
    assert not any(row.get("eligible") for row in diagnostics if row.get("project_id") == "PXD2")


def test_cli_exits_when_no_shards_match(tmp_path: Path):
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/downstream/discover_spectral_rescue_pairs.py"),
            "--projects",
            "PXD000001",
            "--source_glob",
            str(tmp_path / "missing*.parquet"),
        ],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    assert result.returncode != 0
    assert "No parquet files matched" in result.stderr + result.stdout
