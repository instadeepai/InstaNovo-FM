"""Tests for apply_carbamido_from_calc_mz_report helpers."""

from __future__ import annotations

import polars as pl

from scripts.verification.apply_carbamido_from_calc_mz_report import (
    select_projects_for_carb,
)


def test_select_projects_for_carb_picks_eligible_only() -> None:
    """Only projects with as-is < 100% and after-carb unmodified-C == 100% qualify."""
    report = pl.DataFrame(
        {
            "project": ["PXD_ELIGIBLE", "PXD_ALREADY_OK", "PXD_CARB_FAILS"],
            "calc_mz_match_rate_as_is_pct": [80.0, 100.0, 50.0],
            "calc_mz_match_rate_after_carb_unmod_c_rows_pct": [100.0, 100.0, 90.0],
            "calc_mz_matches_after_carb_unmod_c_rows": [10, 5, 3],
        }
    )

    assert select_projects_for_carb(report) == ["PXD_ELIGIBLE"]
