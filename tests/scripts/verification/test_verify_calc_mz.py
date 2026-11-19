"""Tests for verify_calc_mz helpers."""

from __future__ import annotations

import polars as pl
import pytest

from scripts.verification.verify_calc_mz import (
    assert_no_tmt_quant_and_itraq_modifications_conflict,
    calculate_ppm_error,
    carbamidomethylate_cysteines,
)


def test_carbamidomethylate_cysteines_only_unmodified() -> None:
    """Bare cysteines gain UNIMOD:4; already labelled cysteines stay put."""
    assert carbamidomethylate_cysteines("ACDE") == "AC[UNIMOD:4]DE"
    assert carbamidomethylate_cysteines("AC[UNIMOD:4]DE") == "AC[UNIMOD:4]DE"


def test_calculate_ppm_error_zero_at_match() -> None:
    """Identical calc and reference m/z yield zero ppm error."""
    assert calculate_ppm_error(500.0, 500.0) == 0.0


def test_assert_no_tmt_quant_and_itraq_conflict_raises() -> None:
    """Non-DIA rows with TMT quant and iTRAQ modifications are fatal."""
    df = pl.DataFrame(
        {
            "project": ["PXD000001"],
            "file path": ["/data/PXD000001/sample.mzML"],
            "acquisition": ["DDA"],
            "quant": ["TMT"],
            "modifications": ["iTRAQ 4-plex"],
        }
    )
    with pytest.raises(ValueError, match="TMT quant with iTRAQ"):
        assert_no_tmt_quant_and_itraq_modifications_conflict(df)
