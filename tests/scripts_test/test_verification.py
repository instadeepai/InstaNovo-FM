"""Comprehensive test suite for verification scripts.

This module provides comprehensive tests for verification scripts
to ensure they work correctly with realistic data.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Generator
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


class TestDataIntegrityVerification:
    """Test suite for data integrity verification."""

    def test_data_integrity_verification(self) -> None:
        """Test verification of data integrity."""
        # Read original data
        original_df = pl.read_parquet(self.data_dir / "original.parquet")
        shuffled_df = pl.read_parquet(self.data_dir / "shuffled.parquet")
        corrupted_df = pl.read_parquet(self.data_dir / "corrupted.parquet")

        # Test that shuffled data has same structure
        assert original_df.shape[1] == shuffled_df.shape[1]
        assert set(original_df.columns) == set(shuffled_df.columns)

        # Test that corrupted data has different structure
        assert original_df.shape[0] != corrupted_df.shape[0]
        assert original_df.shape[0] == 100
        assert corrupted_df.shape[0] == 90

    def test_row_count_verification(self) -> None:
        """Test verification of row counts."""
        # Read all data files
        original_df = pl.read_parquet(self.data_dir / "original.parquet")
        shuffled_df = pl.read_parquet(self.data_dir / "shuffled.parquet")
        corrupted_df = pl.read_parquet(self.data_dir / "corrupted.parquet")

        # Verify row counts
        assert len(original_df) == 100
        assert len(shuffled_df) == 100
        assert len(corrupted_df) == 90

        # Test row count preservation
        assert len(original_df) == len(shuffled_df)  # Should be equal
        assert len(original_df) != len(corrupted_df)  # Should be different

    def test_column_integrity_verification(self) -> None:
        """Test verification of column integrity."""
        # Read data files
        original_df = pl.read_parquet(self.data_dir / "original.parquet")
        shuffled_df = pl.read_parquet(self.data_dir / "shuffled.parquet")

        # Verify column structure
        original_columns = set(original_df.columns)
        shuffled_columns = set(shuffled_df.columns)

        assert original_columns == shuffled_columns

        # Verify column data types
        for col in original_df.columns:
            assert original_df[col].dtype == shuffled_df[col].dtype

    def test_data_content_verification(self) -> None:
        """Test verification of data content."""
        # Read data files
        original_df = pl.read_parquet(self.data_dir / "original.parquet")
        shuffled_df = pl.read_parquet(self.data_dir / "shuffled.parquet")

        # Sort both dataframes by sequence to compare content
        original_sorted = original_df.sort("sequence")
        shuffled_sorted = shuffled_df.sort("sequence")

        # Verify that content is preserved (ignoring ID differences)
        for col in original_df.columns:
            if col != "id":  # Ignore ID differences
                assert original_sorted[col].to_list() == shuffled_sorted[col].to_list()

    def test_file_format_verification(self) -> None:
        """Test verification of file formats."""
        # Test parquet file format
        original_df = pl.read_parquet(self.data_dir / "original.parquet")
        assert isinstance(original_df, pl.DataFrame)

        # Test that we can write and read back
        test_file = self.output_dir / "test_format.parquet"
        original_df.write_parquet(test_file)

        read_df = pl.read_parquet(test_file)
        assert original_df.equals(read_df)

    def test_error_detection(self) -> None:
        """Test detection of data errors."""
        # Read data files
        original_df = pl.read_parquet(self.data_dir / "original.parquet")
        corrupted_df = pl.read_parquet(self.data_dir / "corrupted.parquet")

        # Test error detection
        errors = []

        # Check row count
        if len(original_df) != len(corrupted_df):
            errors.append(
                f"Row count mismatch: {len(original_df)} vs {len(corrupted_df)}"
            )

        # Check column structure
        if set(original_df.columns) != set(corrupted_df.columns):
            errors.append("Column structure mismatch")

        # Verify errors were detected
        assert len(errors) > 0
        assert "Row count mismatch" in errors[0]

    @pytest.fixture(autouse=True)
    def _setup_integrity_environment(self) -> Generator[None, None, None]:
        """Set up test environment for data integrity verification."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir) / "data"
        self.output_dir = Path(self.test_dir) / "outputs"

        # Create test directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create test data
        self._create_integrity_test_data()

        yield

        # Cleanup
        shutil.rmtree(self.test_dir)

    def _create_integrity_test_data(self) -> None:
        """Create test data for integrity verification."""
        # Create original data
        original_data: dict[str, list] = {
            "id": list(range(100)),
            "sequence": [f"ATCG{i % 100}" for i in range(100)],
            "quality": [i % 50 for i in range(100)],
            "metadata": [f"sample_{i % 10}" for i in range(100)],
            "mz": [[100.0 + i, 200.0 + i, 300.0 + i] for i in range(100)],
            "intensity": [[100.0 + i, 200.0 + i, 300.0 + i] for i in range(100)],
        }

        original_df = pl.DataFrame(original_data)
        original_df.write_parquet(self.data_dir / "original.parquet")

        # Create shuffled data (same content, different order)
        shuffled_data = original_data.copy()
        shuffled_data["id"] = list(
            range(100, 200)
        )  # Different IDs to simulate shuffling
        shuffled_df = pl.DataFrame(shuffled_data)
        shuffled_df.write_parquet(self.data_dir / "shuffled.parquet")

        # Create corrupted data (missing some rows)
        corrupted_data: dict[str, list] = {
            k: v[:90] for k, v in original_data.items()
        }  # Only 90 rows
        corrupted_df = pl.DataFrame(corrupted_data)
        corrupted_df.write_parquet(self.data_dir / "corrupted.parquet")
