"""Tests for enforce_nulls helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from scripts.preprocessing.enforce_nulls import enforce_nulls


def test_enforce_nulls_replaces_unknown_in_collision_energy(tmp_path: Path) -> None:
    """Placeholder Unknown in collision_energy becomes null."""
    path = tmp_path / "unknown.parquet"
    pl.DataFrame(
        {
            "collision_energy": ["Unknown", "30.0"],
            "frag_type": ["HCD", "HCD"],
        }
    ).write_parquet(path)

    affected = enforce_nulls(str(tmp_path))

    assert affected == [str(path)]
    assert pl.read_parquet(path)["collision_energy"].to_list() == [None, "30.0"]


def test_enforce_nulls_skips_clean_file(tmp_path: Path) -> None:
    """Files without Unknown placeholders are left alone."""
    path = tmp_path / "clean.parquet"
    pl.DataFrame(
        {
            "collision_energy": ["30.0", "35.0"],
            "frag_type": ["HCD", "HCD"],
        }
    ).write_parquet(path)

    affected = enforce_nulls(str(tmp_path))

    assert affected == []
    assert pl.read_parquet(path)["collision_energy"].to_list() == ["30.0", "35.0"]
