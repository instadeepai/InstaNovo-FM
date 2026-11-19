"""Tests for verify_intensity_max_normalisation helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from scripts.verification.verify_intensity_max_normalisation import (
    normalise_intensity_row,
    verify_file,
)


def test_normalise_intensity_row_scales_to_one() -> None:
    """Peak intensities are scaled so the maximum is 1.0."""
    intensities, scale = normalise_intensity_row([2.0, 4.0], 1.0)

    assert intensities == [0.5, 1.0]
    assert scale == 4.0


def test_verify_file_flags_non_unit_max(tmp_path: Path) -> None:
    """Rows whose intensity max is not 1.0 are counted as bad."""
    path = tmp_path / "sample.parquet"
    pl.DataFrame(
        {
            "intensity_array": [[2.0, 4.0]],
            "scale_factor": [1.0],
        }
    ).write_parquet(path)

    stats = verify_file(path)

    assert stats.rows == 1
    assert stats.bad_max == 1
    assert stats.ok == 0
