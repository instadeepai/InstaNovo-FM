"""Tests for add_acquisition_column mismatch overwrite."""

from pathlib import Path

import polars as pl

from scripts.preprocessing.add_acquisition_column import (
    _process_data_file_with_acquisition,
)


def test_mismatch_overwrites_acquisition(tmp_path: Path) -> None:
    """Stale acquisition values are overwritten from search data."""
    path = tmp_path / "sample.parquet"
    pl.DataFrame({"scan": ["1"], "acquisition": ["DIA"]}).write_parquet(path)

    outcome = _process_data_file_with_acquisition(
        str(path), "PXD000001", "sample", "DDA", dry_run=False, verbose=False
    )
    assert outcome == "updated"
    assert pl.read_parquet(path)["acquisition"].to_list() == ["DDA"]


def test_matching_acquisition_is_skipped(tmp_path: Path) -> None:
    """Matching acquisition values are left alone."""
    path = tmp_path / "sample.parquet"
    pl.DataFrame({"scan": ["1"], "acquisition": ["DDA"]}).write_parquet(path)

    outcome = _process_data_file_with_acquisition(
        str(path), "PXD000001", "sample", "DDA", dry_run=False, verbose=False
    )
    assert outcome == "already_has_column"
