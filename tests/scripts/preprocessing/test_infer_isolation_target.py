"""Tests for infer_isolation_target helpers."""

from __future__ import annotations

import polars as pl

from scripts.preprocessing.infer_isolation_target import (
    infer_it_from_header,
    infer_it_from_precursor_mz,
)


def test_infer_it_from_header_parses_at_suffix() -> None:
    """Missing isolation targets are filled from header mz@ce text."""
    ldf = pl.LazyFrame(
        {
            "header": ["MS2 scan 1234.5@30.0"],
            "isolation_target": [None],
            "precursor_mz": [999.0],
        }
    )

    result = infer_it_from_header(ldf).collect()

    assert result["isolation_target"].to_list() == [1234.5]


def test_infer_it_from_precursor_mz_when_header_fails() -> None:
    """Precursor m/z fills missing isolation targets as a fallback."""
    ldf = pl.LazyFrame(
        {
            "header": ["MS2 scan without at"],
            "isolation_target": [None],
            "precursor_mz": [5678.9],
        }
    )

    result = infer_it_from_precursor_mz(ldf).collect()

    assert result["isolation_target"].to_list() == [5678.9]
