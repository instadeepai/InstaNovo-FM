"""Tests for verify_precursor_charges_and_acq_type."""

from pathlib import Path
import polars as pl
import pytest

def test_run_verification_creates_output_dir_for_dda_only_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DDA-only verification findings should still create the report directory."""
    from scripts.verification import verify_precursor_charges_and_acq_type as verifier

    incorrect_dia_files = pl.DataFrame(
        schema={
            "filename": pl.String,
            "project": pl.String,
            "error_type": pl.String,
            "num_error_rows": pl.Int64,
            "total_rows": pl.Int64,
        }
    )
    incorrect_dda_files = pl.DataFrame(
        {
            "filename": ["sample.parquet"],
            "project": ["PXD000001"],
            "error_type": ["zero_precursor_charge"],
            "num_error_rows": [1],
            "total_rows": [10],
        }
    )
    project_level_summary = pl.DataFrame(
        {
            "project": ["PXD000001"],
            "acquisition": ["DDA"],
            "error_type": ["zero_precursor_charge"],
            "files_with_this_error": [1],
            "files_with_100_percent_error": [0],
            "total_files_in_project_acq": [1],
            "all_files_in_project_affected": [True],
        }
    )

    def fake_analyze_precursor_charges(
        input_dir: str, search_data_path: str, aws_profile: str | None = None
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        return incorrect_dia_files, incorrect_dda_files, project_level_summary

    monkeypatch.setattr(
        verifier, "analyze_precursor_charges", fake_analyze_precursor_charges
    )

    output_dir = tmp_path / "logs" / "verify_precursor_charges_and_acq_type"

    verifier.run_verification(
        input_dir="input",
        search_data_path="search_data.xlsx",
        output_dir=str(output_dir),
        aws_profile=None,
    )

    assert not (output_dir / "incorrect_dia_files.csv").exists()
    assert (output_dir / "incorrect_dda_files.csv").exists()
    assert (output_dir / "project_summary.csv").exists()
    assert len(pl.read_csv(output_dir / "incorrect_dda_files.csv")) == 1
