"""Tests for create_subsets empty hold-back handling."""

from __future__ import annotations

import polars as pl

from scripts.splitting.create_subsets import _filter_out_glyco_sequences


def test_hold_back_all_modified_rows_yields_empty_frame() -> None:
    """All-[IN:…] files become empty after the hold-back filter."""
    df = pl.DataFrame(
        {
            "sequence": ["PEPTIDE[IN:1]", "A[IN:2]BC"],
            "hyperscore": [1.0, 2.0],
        }
    )
    filtered = _filter_out_glyco_sequences(df, enabled=True)
    assert filtered.height == 0
