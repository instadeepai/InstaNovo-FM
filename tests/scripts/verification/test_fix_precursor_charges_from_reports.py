"""Tests for fix_precursor_charges_from_reports helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from scripts.verification.fix_precursor_charges_from_reports import (
    _filter_zero_charge_rows,
    load_csv_if_exists,
)


def test_load_csv_if_exists_missing_returns_empty_schema(tmp_path: Path) -> None:
    """A missing report path yields an empty frame with the expected schema."""
    result = load_csv_if_exists(tmp_path / "missing.csv")

    assert result.height == 0
    assert result.columns == [
        "filename",
        "project",
        "error_type",
        "num_error_rows",
        "total_rows",
    ]


def test_filter_zero_charge_rows_removes_zeros() -> None:
    """DDA rows with precursor charge 0 are dropped."""
    df = pl.DataFrame({"precursor_charge": [0, 2, 3], "scan": ["a", "b", "c"]})

    filtered = _filter_zero_charge_rows(df, Path("sample.parquet"))

    assert filtered["precursor_charge"].to_list() == [2, 3]
    assert filtered["scan"].to_list() == ["b", "c"]
