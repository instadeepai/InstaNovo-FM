"""Tests for verify_precursor_charges_and_acq_type."""

from pathlib import Path

import polars as pl
import pytest

from scripts.verification.verify_precursor_charges_and_acq_type import (
    FILE_PRECURSOR_CHARGE_ERROR_SCHEMA,
    FilePrecursorChargeErrors,
    check_if_all_files_in_project_have_errors,
    errors_to_dataframe,
)


def test_run_verification_creates_output_dir_for_dda_only_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DDA-only verification findings should still create the report directory."""
    from scripts.verification import verify_precursor_charges_and_acq_type as verifier

    incorrect_dia_files = pl.DataFrame(schema=FILE_PRECURSOR_CHARGE_ERROR_SCHEMA)
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


def test_errors_to_dataframe_empty_has_schema() -> None:
    """Empty error lists must keep columns for later concat."""
    df = errors_to_dataframe([])
    assert df.height == 0
    assert df.schema == FILE_PRECURSOR_CHARGE_ERROR_SCHEMA


def test_check_project_errors_with_dda_only() -> None:
    """DDA-only errors must concat with an empty DIA frame."""
    incorrect_dia = errors_to_dataframe([])
    incorrect_dda = errors_to_dataframe(
        [
            FilePrecursorChargeErrors(
                "sample", "PXD000001", "zero_precursor_charge", 1, 10
            )
        ]
    )
    search_data = pl.DataFrame(
        {
            "project": ["PXD000001"],
            "filename": ["sample"],
            "acquisition": ["DDA"],
        }
    )

    summary = check_if_all_files_in_project_have_errors(
        ["PXD000001/sample.parquet"],
        incorrect_dia,
        incorrect_dda,
        search_data,
    )
    assert summary.height == 1
    assert summary["acquisition"][0] == "DDA"


def test_check_project_errors_with_dia_only() -> None:
    """DIA-only errors must concat with an empty DDA frame."""
    incorrect_dia = errors_to_dataframe(
        [
            FilePrecursorChargeErrors(
                "sample", "PXD000001", "non_zero_precursor_charge", 2, 10
            )
        ]
    )
    incorrect_dda = errors_to_dataframe([])
    search_data = pl.DataFrame(
        {
            "project": ["PXD000001"],
            "filename": ["sample"],
            "acquisition": ["DIA"],
        }
    )

    summary = check_if_all_files_in_project_have_errors(
        ["PXD000001/sample.parquet"],
        incorrect_dia,
        incorrect_dda,
        search_data,
    )
    assert summary.height == 1
    assert summary["acquisition"][0] == "DIA"


def test_check_project_errors_with_no_errors() -> None:
    """Both empty sides must concat and yield an empty project summary."""
    summary = check_if_all_files_in_project_have_errors(
        ["PXD000001/sample.parquet"],
        errors_to_dataframe([]),
        errors_to_dataframe([]),
        pl.DataFrame(
            {
                "project": ["PXD000001"],
                "filename": ["sample"],
                "acquisition": ["DDA"],
            }
        ),
    )
    assert summary.height == 0
