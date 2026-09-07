"""Comprehensive test suite for verification scripts.

This module provides comprehensive tests for verification scripts
to ensure they work correctly with realistic data.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Generator
import polars as pl
import pandas as pd
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


class TestVerificationScripts:
    """Test suite for verification scripts."""

    def test_normalise_sequence(self) -> None:
        """Test sequence normalization (L to I conversion)."""
        from scripts.verify_preexisting_splits import normalise_sequence

        # Test L to I conversion
        assert normalise_sequence("PEPTIDEK") == "PEPTIDEK"
        assert normalise_sequence("PEPTIDEL") == "PEPTIDEI"
        assert normalise_sequence("PEPTIDELK") == "PEPTIDEIK"
        assert normalise_sequence("PEPTIDELR") == "PEPTIDEIR"

        # Test multiple L conversions
        assert normalise_sequence("PEPTIDELKL") == "PEPTIDEIKI"

        # Test no L sequences
        assert normalise_sequence("PEPTIDEK") == "PEPTIDEK"
        assert normalise_sequence("PEPTIDER") == "PEPTIDER"

    def test_load_split_files(self) -> None:
        """Test loading split files."""
        from scripts.verify_preexisting_splits import load_split_files

        split_files = [
            str(self.splits_dir / "identity_splits_phospho.csv"),
            str(self.splits_dir / "identity_splits_proteome_tools.csv"),
            str(self.splits_dir / "massivekb_splits.csv"),
        ]

        split_dfs = load_split_files(split_files)

        assert len(split_dfs) == 3
        assert all("sequence" in df.columns for df in split_dfs.values())
        assert all("split" in df.columns for df in split_dfs.values())
        assert all("normalised_sequence" in df.columns for df in split_dfs.values())

        # Check that normalization was applied
        for df in split_dfs.values():
            assert len(df) == 5  # Each file has 5 peptides

    def test_find_overlapping_peptides(self) -> None:
        """Test finding overlapping peptides across files."""
        from scripts.verify_preexisting_splits import (
            load_split_files,
            find_overlapping_peptides,
        )

        split_files = [
            str(self.splits_dir / "identity_splits_phospho.csv"),
            str(self.splits_dir / "identity_splits_proteome_tools.csv"),
            str(self.splits_dir / "massivekb_splits.csv"),
        ]

        split_dfs = load_split_files(split_files)
        overlapping_peptides = find_overlapping_peptides(split_dfs)

        # Should find overlapping peptides since all files have the same peptides
        assert len(overlapping_peptides) > 0

        # Check that overlapping peptides have multiple assignments
        for _peptide, assignments in overlapping_peptides.items():
            assert len(assignments) > 1

    def test_get_all_files(self) -> None:
        """Test getting all files with overlapping peptides."""
        from scripts.verify_preexisting_splits import (
            load_split_files,
            find_overlapping_peptides,
            get_all_files,
        )

        split_files = [
            str(self.splits_dir / "identity_splits_phospho.csv"),
            str(self.splits_dir / "identity_splits_proteome_tools.csv"),
            str(self.splits_dir / "massivekb_splits.csv"),
        ]

        split_dfs = load_split_files(split_files)
        overlapping_peptides = find_overlapping_peptides(split_dfs)
        all_files = get_all_files(overlapping_peptides)

        assert len(all_files) == 3
        assert all(f.endswith(".csv") for f in all_files)

    def test_group_assignments(self) -> None:
        """Test grouping peptide assignments by file."""
        from scripts.verify_preexisting_splits import (
            load_split_files,
            find_overlapping_peptides,
            group_assignments,
        )

        split_files = [
            str(self.splits_dir / "identity_splits_phospho.csv"),
            str(self.splits_dir / "identity_splits_proteome_tools.csv"),
            str(self.splits_dir / "massivekb_splits.csv"),
        ]

        split_dfs = load_split_files(split_files)
        overlapping_peptides = find_overlapping_peptides(split_dfs)
        file_assignments, peptide_assignments = group_assignments(overlapping_peptides)

        assert len(file_assignments) == 3
        assert len(peptide_assignments) > 0

        # Check that each file has assignments
        for _file_path, assignments in file_assignments.items():
            assert len(assignments) > 0

    def test_report_within_file_clashes(self) -> None:
        """Test reporting within-file clashes."""
        from scripts.verify_preexisting_splits import (
            load_split_files,
            find_overlapping_peptides,
            group_assignments,
            report_within_file_clashes,
        )

        split_files = [
            str(self.splits_dir / "identity_splits_phospho.csv"),
            str(self.splits_dir / "identity_splits_proteome_tools.csv"),
            str(self.splits_dir / "massivekb_splits.csv"),
        ]

        split_dfs = load_split_files(split_files)
        overlapping_peptides = find_overlapping_peptides(split_dfs)
        file_assignments, _ = group_assignments(overlapping_peptides)

        # Test reporting (our test data intentionally has clashes to test detection)
        has_clashes = report_within_file_clashes(file_assignments)
        assert has_clashes  # Our test data has within-file clashes to test detection

    def test_create_conflict_row(self) -> None:
        """Test creating conflict row for CSV output."""
        from scripts.verify_preexisting_splits import (
            load_split_files,
            find_overlapping_peptides,
            get_all_files,
            create_conflict_row,
        )

        split_files = [
            str(self.splits_dir / "identity_splits_phospho.csv"),
            str(self.splits_dir / "identity_splits_proteome_tools.csv"),
            str(self.splits_dir / "massivekb_splits.csv"),
        ]

        split_dfs = load_split_files(split_files)
        overlapping_peptides = find_overlapping_peptides(split_dfs)
        all_files = get_all_files(overlapping_peptides)

        # Test creating conflict row
        for peptide, assignments in overlapping_peptides.items():
            row = create_conflict_row(peptide, assignments, all_files)

            assert "normalised_sequence" in row
            assert row["normalised_sequence"] == peptide

            # Check that all files are represented
            for file_path in all_files:
                assert file_path in row

    def test_report_cross_file_conflicts(self) -> None:
        """Test reporting cross-file conflicts."""
        from scripts.verify_preexisting_splits import (
            load_split_files,
            find_overlapping_peptides,
            group_assignments,
            report_cross_file_conflicts,
        )

        split_files = [
            str(self.splits_dir / "identity_splits_phospho.csv"),
            str(self.splits_dir / "identity_splits_proteome_tools.csv"),
            str(self.splits_dir / "massivekb_splits.csv"),
        ]

        split_dfs = load_split_files(split_files)
        overlapping_peptides = find_overlapping_peptides(split_dfs)
        _, peptide_assignments = group_assignments(overlapping_peptides)
        all_files = list(split_files)

        # Test reporting cross-file conflicts
        csv_rows = report_cross_file_conflicts(peptide_assignments, all_files)

        # Should have conflicts since peptides are assigned to different splits
        assert len(csv_rows) > 0

        # Check that each row has the required structure
        for row in csv_rows:
            assert "normalised_sequence" in row
            for file_path in all_files:
                assert file_path in row

    def test_save_conflicts_to_csv(self) -> None:
        """Test saving conflicts to CSV file."""
        from scripts.verify_preexisting_splits import save_conflicts_to_csv

        # Create the output_files directory that the script expects
        output_files_dir = Path("output_files")
        output_files_dir.mkdir(exist_ok=True)

        # Create test conflict data
        csv_rows = [
            {
                "normalised_sequence": "PEPTIDEK",
                "file1.csv": "train (PEPTIDEK)",
                "file2.csv": "valid (PEPTIDEK)",
                "file3.csv": "test (PEPTIDEK)",
            },
            {
                "normalised_sequence": "PEPTIDER",
                "file1.csv": "valid (PEPTIDER)",
                "file2.csv": "train (PEPTIDER)",
                "file3.csv": "valid (PEPTIDER)",
            },
        ]

        # Save conflicts
        save_conflicts_to_csv(csv_rows)

        # Check that the file was created
        assert (output_files_dir / "split_conflicts.csv").exists()

        # Cleanup
        shutil.rmtree(output_files_dir)

    def test_analyse_overlaps(self) -> None:
        """Test complete overlap analysis."""
        from scripts.verify_preexisting_splits import (
            load_split_files,
            find_overlapping_peptides,
            analyse_overlaps,
        )

        # Create the output_files directory that the script expects
        output_files_dir = Path("output_files")
        output_files_dir.mkdir(exist_ok=True)

        split_files = [
            str(self.splits_dir / "identity_splits_phospho.csv"),
            str(self.splits_dir / "identity_splits_proteome_tools.csv"),
            str(self.splits_dir / "massivekb_splits.csv"),
        ]

        split_dfs = load_split_files(split_files)
        overlapping_peptides = find_overlapping_peptides(split_dfs)

        # Test complete analysis
        analyse_overlaps(overlapping_peptides)

        # Check that the file was created
        assert (output_files_dir / "split_conflicts.csv").exists()

        # Cleanup
        shutil.rmtree(output_files_dir)

    def test_complete_verification_workflow(self) -> None:
        """Test complete verification workflow."""
        from scripts.verify_preexisting_splits import (
            load_split_files,
            find_overlapping_peptides,
            analyse_overlaps,
        )

        # Create the output_files directory that the script expects
        output_files_dir = Path("output_files")
        output_files_dir.mkdir(exist_ok=True)

        # Define split files (use the correct names that are actually created)
        split_files = [
            str(self.splits_dir / "identity_splits_phospho.csv"),
            str(self.splits_dir / "identity_splits_proteome_tools.csv"),
            str(self.splits_dir / "massivekb_splits.csv"),
        ]

        # Load split files
        split_dfs = load_split_files(split_files)
        assert len(split_dfs) == 3

        # Find overlapping peptides
        overlapping_peptides = find_overlapping_peptides(split_dfs)
        assert len(overlapping_peptides) > 0

        # Analyse overlaps
        analyse_overlaps(overlapping_peptides)

        # Check that the file was created
        assert (output_files_dir / "split_conflicts.csv").exists()

        # Cleanup
        shutil.rmtree(output_files_dir)

    @pytest.fixture(autouse=True)
    def _setup_test_environment(self) -> Generator[None, None, None]:
        """Set up test environment with temporary directories and test data."""
        self.test_dir = tempfile.mkdtemp()
        self.splits_dir = Path(self.test_dir) / "splits"
        self.output_dir = Path(self.test_dir) / "outputs"

        # Create test directories
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create test data
        self._create_test_splits()

        yield

        # Cleanup
        shutil.rmtree(self.test_dir)

    def _create_test_splits(self) -> None:
        """Create realistic test split data."""
        # Create multiple split files with overlapping peptides
        split_configs = {
            "identity_splits_phospho.csv": {
                "peptides": [
                    "PEPTIDEK",
                    "PEPTIDER",
                    "PEPTIDEK",
                    "PEPTIDER",
                    "PEPTIDEK",
                ],
                "modified_peptides": [
                    "PEPTIDE[142]K",
                    "PEPTIDE[3562]R",
                    "PEPTIDE[2346]K",
                    "PEPTIDE[3923]R",
                    "PEPTIDE[2449]K",
                ],
                "splits": ["train", "valid", "train", "test", "valid"],
            },
            "identity_splits_proteome_tools.csv": {
                "peptides": [
                    "PEPTIDEK",
                    "PEPTIDER",
                    "PEPTIDEK",
                    "PEPTIDER",
                    "PEPTIDEK",
                ],
                "modified_peptides": [
                    "PEPTIDE[142]K",
                    "PEPTIDE[3562]R",
                    "PEPTIDE[2346]K",
                    "PEPTIDE[3923]R",
                    "PEPTIDE[2449]K",
                ],
                "splits": ["valid", "train", "test", "train", "valid"],
            },
            "massivekb_splits.csv": {
                "peptides": [
                    "PEPTIDEK",
                    "PEPTIDER",
                    "PEPTIDEK",
                    "PEPTIDER",
                    "PEPTIDEK",
                ],
                "modified_peptides": [
                    "PEPTIDE[142]K",
                    "PEPTIDE[3562]R",
                    "PEPTIDE[2346]K",
                    "PEPTIDE[3923]R",
                    "PEPTIDE[2449]K",
                ],
                "splits": ["test", "valid", "train", "valid", "train"],
            },
        }

        for filename, config in split_configs.items():
            data = {
                "sequence": config["peptides"],
                "modified_peptide": config["modified_peptides"],
                "split": config["splits"],
            }

            df = pd.DataFrame(data)
            df.to_csv(self.splits_dir / filename, index=False)


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


class TestVerificationIntegration:
    """Integration tests for verification workflows."""

    def test_verification_with_different_scenarios(self) -> None:
        """Test verification with different data scenarios."""
        # Test scenario 1: No overlaps
        no_overlap_data = {
            "sequence": ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"],
            "modified_peptide": ["PEPTIDE1", "PEPTIDE2", "PEPTIDE3"],
            "split": ["train", "valid", "test"],
        }

        df1 = pd.DataFrame(no_overlap_data)
        df1.to_csv(self.splits_dir / "no_overlap.csv", index=False)

        # Test scenario 2: Complete overlaps
        complete_overlap_data = {
            "sequence": ["PEPTIDEK", "PEPTIDEK", "PEPTIDEK"],
            "modified_peptide": ["PEPTIDE[142]K", "PEPTIDE[142]K", "PEPTIDE[142]K"],
            "split": ["train", "valid", "test"],
        }

        df2 = pd.DataFrame(complete_overlap_data)
        df2.to_csv(self.splits_dir / "complete_overlap.csv", index=False)

        # Test both scenarios
        from scripts.verify_preexisting_splits import (
            load_split_files,
            find_overlapping_peptides,
        )

        # Test no overlap scenario
        no_overlap_files = [str(self.splits_dir / "no_overlap.csv")]
        no_overlap_dfs = load_split_files(no_overlap_files)
        no_overlap_peptides = find_overlapping_peptides(no_overlap_dfs)
        assert len(no_overlap_peptides) == 0

        # Test complete overlap scenario
        complete_overlap_files = [str(self.splits_dir / "complete_overlap.csv")]
        complete_overlap_dfs = load_split_files(complete_overlap_files)
        complete_overlap_peptides = find_overlapping_peptides(complete_overlap_dfs)
        assert len(complete_overlap_peptides) > 0

    def test_verification_error_handling(self) -> None:
        """Test error handling in verification process."""
        # Test with non-existent file
        from scripts.verify_preexisting_splits import load_split_files

        non_existent_files = ["/non/existent/file.csv"]
        split_dfs = load_split_files(non_existent_files)
        assert len(split_dfs) == 0

        # Test with malformed CSV
        malformed_file = self.splits_dir / "malformed.csv"
        with open(malformed_file, "w") as f:
            f.write("invalid,csv,content\n")
            f.write("missing,required,columns\n")

        malformed_files = [str(malformed_file)]
        split_dfs = load_split_files(malformed_files)
        assert len(split_dfs) == 0  # Should handle malformed files gracefully

    def test_verification_performance(self) -> None:
        """Test verification performance with large datasets."""
        # Create large dataset
        large_data = {
            "sequence": [f"PEPTIDE{i}" for i in range(1000)],
            "modified_peptide": [f"PEPTIDE[142]{i}" for i in range(1000)],
            "split": [
                "train" if i % 3 == 0 else "valid" if i % 3 == 1 else "test"
                for i in range(1000)
            ],
        }

        df = pd.DataFrame(large_data)
        df.to_csv(self.splits_dir / "large_dataset.csv", index=False)

        # Test performance
        import time

        start_time = time.time()

        from scripts.verify_preexisting_splits import (
            load_split_files,
            find_overlapping_peptides,
        )

        large_files = [str(self.splits_dir / "large_dataset.csv")]
        split_dfs = load_split_files(large_files)
        overlapping_peptides = find_overlapping_peptides(split_dfs)

        end_time = time.time()
        processing_time = end_time - start_time

        # Verify performance is reasonable
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert len(split_dfs) == 1
        assert len(overlapping_peptides) == 0  # No overlaps in single file

    @pytest.fixture(autouse=True)
    def _setup_integration_environment(self) -> Generator[None, None, None]:
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.splits_dir = Path(self.test_dir) / "splits"
        self.output_dir = Path(self.test_dir) / "outputs"

        # Create test directories
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create realistic test data
        self._create_integration_test_data()

        yield

        # Cleanup
        shutil.rmtree(self.test_dir)

    def _create_integration_test_data(self) -> None:
        """Create realistic test data for integration testing."""
        # Create multiple split files with various scenarios
        scenarios = [
            {
                "filename": "phospho_splits.csv",
                "peptides": [
                    "PEPTIDEK",
                    "PEPTIDER",
                    "PEPTIDEK",
                    "PEPTIDER",
                    "PEPTIDEK",
                ],
                "modified_peptides": [
                    "PEPTIDE[142]K",
                    "PEPTIDE[3562]R",
                    "PEPTIDE[2346]K",
                    "PEPTIDE[3923]R",
                    "PEPTIDE[2449]K",
                ],
                "splits": ["train", "valid", "train", "test", "valid"],
            },
            {
                "filename": "proteome_tools_splits.csv",
                "peptides": [
                    "PEPTIDEK",
                    "PEPTIDER",
                    "PEPTIDEK",
                    "PEPTIDER",
                    "PEPTIDEK",
                ],
                "modified_peptides": [
                    "PEPTIDE[142]K",
                    "PEPTIDE[3562]R",
                    "PEPTIDE[2346]K",
                    "PEPTIDE[3923]R",
                    "PEPTIDE[2449]K",
                ],
                "splits": ["valid", "train", "test", "train", "valid"],
            },
            {
                "filename": "massivekb_splits.csv",
                "peptides": [
                    "PEPTIDEK",
                    "PEPTIDER",
                    "PEPTIDEK",
                    "PEPTIDER",
                    "PEPTIDEK",
                ],
                "modified_peptides": [
                    "PEPTIDE[142]K",
                    "PEPTIDE[3562]R",
                    "PEPTIDE[2346]K",
                    "PEPTIDE[3923]R",
                    "PEPTIDE[2449]K",
                ],
                "splits": ["test", "valid", "train", "valid", "train"],
            },
        ]

        for scenario in scenarios:
            data = {
                "sequence": scenario["peptides"],
                "modified_peptide": scenario["modified_peptides"],
                "split": scenario["splits"],
            }

            df = pd.DataFrame(data)
            df.to_csv(self.splits_dir / str(scenario["filename"]), index=False)
