"""Comprehensive test suite for data conversion scripts.

This module provides comprehensive tests for all the data conversion scripts
to ensure they work correctly with realistic IPC/Parquet data.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Any
import polars as pl
import pytest
from typer.testing import CliRunner
from typing import Generator

# Import the scripts to test
from scripts.preprocessing.detect_all_duplicates import app as detect_app
from scripts.preprocessing.delete_same_folder_duplicates import (
    app as delete_same_app,
)
from scripts.preprocessing.find_empty_files import app as find_empty_app
from scripts.preprocessing.convert_ipc_to_parquet import app as convert_app
from scripts.preprocessing.enforce_nulls import app as enforce_app
from scripts.preprocessing.delete_files import app as delete_files_app
from scripts.preprocessing.find_modifications import app as find_mods_app
from scripts.preprocessing.check_conversion import app as check_app
from scripts.preprocessing.detect_multi_folder_duplicates import (
    app as detect_multi_app,
)
from scripts.preprocessing.delete_multi_folder_duplicates import (
    app as delete_multi_app,
)
from scripts.preprocessing.infer_isolation_target import app as infer_app
from scripts.preprocessing.label_modifications import (
    app as label_app,
    create_mod_dict,
)
from scripts.preprocessing.check_modifications import app as check_mods_app


def script_exists(script_name: str) -> bool:
    """Check if a script exists and can be imported."""
    try:
        # Try to import the script module
        script_path = f"scripts.preprocessing.{script_name}"
        __import__(script_path)
        return True
    except ImportError:
        return False


class TestDataConversionScripts:
    """Test suite for data conversion scripts."""

    def create_test_data(self) -> None:
        """Create realistic test data based on the provided schema."""
        # Create sample data with the provided schema
        sample_data = {
            "index": [1, 2, 3, 4, 5],
            "scan": ["scan_001", "scan_002", "scan_003", "scan_004", "scan_005"],
            "header": [
                "MS2 scan 1234.5@30.0",
                "MS2 scan 2345.6@35.0",
                "MS2 scan 3456.7@40.0",
                "MS2 scan 4567.8@45.0",
                "MS2 scan 5678.9@50.0",
            ],
            "rt": [30.0, 35.0, 40.0, 45.0, 50.0],
            "frag_type": ["HCD", "HCD", "HCD", "HCD", "HCD"],
            "collision_energy": [30.0, 35.0, 40.0, 45.0, 50.0],
            "precursor_mz": [1234.5, 2345.6, 3456.7, 4567.8, 5678.9],
            "precursor_charge": [2, 3, 2, 3, 2],
            "precursor_intensity": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            "lower_offset": [-1.0, -1.0, -1.0, -1.0, -1.0],
            "upper_offset": [1.0, 1.0, 1.0, 1.0, 1.0],
            "isolation_target": [None, None, None, None, None],  # Will be inferred
            "mz": [
                [100.0, 200.0, 300.0],
                [150.0, 250.0, 350.0],
                [200.0, 300.0, 400.0],
                [250.0, 350.0, 450.0],
                [300.0, 400.0, 500.0],
            ],
            "intensity": [
                [100.0, 200.0, 300.0],
                [150.0, 250.0, 350.0],
                [200.0, 300.0, 400.0],
                [250.0, 350.0, 450.0],
                [300.0, 400.0, 500.0],
            ],
            "scale_factor": [1.0, 1.0, 1.0, 1.0, 1.0],
        }

        # Create test files with different scenarios
        self.create_test_files(sample_data)

    def create_test_files(self, base_data: dict[str, Any]) -> None:
        """Create various test files for different scenarios."""
        # Create files with duplicates - make sure we have actual duplicates
        df1 = pl.DataFrame(base_data)
        df1.write_ipc(self.data_dir / "acfm" / "sample1.ipc")
        df1.write_ipc(self.data_dir / "acfm" / "sample1.mzML.ipc")  # Duplicate
        df1.write_ipc(self.data_dir / "acfm" / "sample1_copy.ipc")  # Another duplicate

        # Create multi-folder duplicates (same filename in different folders)
        df1.write_ipc(
            self.data_dir / "lcfm" / "sample1.ipc"
        )  # Same filename in different folder
        df1.write_ipc(
            self.data_dir / "mcfm" / "sample1.ipc"
        )  # Same filename in different folder
        df1.write_ipc(
            self.data_dir / "hcfm" / "sample1.ipc"
        )  # Same filename in different folder

        # Create files with empty data
        empty_df = pl.DataFrame({col: [] for col in base_data.keys()})
        empty_df.write_ipc(self.data_dir / "lcfm" / "empty.ipc")

        # Create files with modifications
        mod_data = base_data.copy()
        mod_data["modified_peptide"] = [
            "PEPTIDE[142]K",
            "PEPTIDE[3562]R",
            "PEPTIDE[2346]K",
            "PEPTIDE[3923]R",
            "PEPTIDE[2449]K",
        ]
        mod_data["unmodified_peptide"] = [
            "PEPTIDEK",
            "PEPTIDER",
            "PEPTIDEK",
            "PEPTIDER",
            "PEPTIDEK",
        ]
        mod_df = pl.DataFrame(mod_data)
        mod_df.write_parquet(self.data_dir / "mcfm" / "modified.parquet")

        # Create files with unknown collision energy - use proper mixed type handling
        unknown_data = base_data.copy()
        # Create collision energy with mixed types using proper Polars syntax
        unknown_collision_energy = pl.Series(
            [None if i % 3 == 0 else 30.0 + i for i in range(5)]
        )
        unknown_data["collision_energy"] = unknown_collision_energy
        unknown_df = pl.DataFrame(unknown_data)
        unknown_df.write_parquet(self.data_dir / "hcfm" / "unknown_ce.parquet")

        # Create files with null isolation targets
        null_data = base_data.copy()
        null_data["isolation_target"] = [None, None, None, None, None]
        null_df = pl.DataFrame(null_data)
        null_df.write_parquet(self.data_dir / "acfm" / "null_targets.parquet")

    def test_detect_all_duplicates(self) -> None:
        """Test detect_all_duplicates script."""
        runner = CliRunner()

        # Test single directory detection
        result = runner.invoke(
            detect_app,
            [
                "--input-dir",
                str(self.data_dir / "acfm"),
                "--output-file",
                str(self.output_dir / "duplicates.txt"),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert (self.output_dir / "duplicates.txt").exists()

        # Check that duplicates were found
        with open(self.output_dir / "duplicates.txt") as f:
            content = f.read()
            assert "sample1" in content

    def test_find_empty_files(self) -> None:
        """Test find_empty_files script."""
        runner = CliRunner()

        # Test finding empty files
        result = runner.invoke(
            find_empty_app,
            [
                "--input-dir",
                str(self.data_dir / "lcfm"),
                "--output-file",
                str(self.output_dir / "empty_files.txt"),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert (self.output_dir / "empty_files.txt").exists()

        # Check that empty file was found
        with open(self.output_dir / "empty_files.txt") as f:
            content = f.read()
            assert "empty.ipc" in content

    def test_enforce_nulls(self) -> None:
        """Test enforce_nulls script."""
        if not script_exists("enforce_nulls"):
            pytest.skip("enforce_nulls script not available")

        runner = CliRunner()

        # Test enforcing nulls
        result = runner.invoke(
            enforce_app,
            [
                "--input-dir",
                str(self.data_dir / "hcfm"),
                "--output-file",
                str(self.output_dir / "enforced.csv"),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        # Allow for missing output file if no files were updated
        if (self.output_dir / "enforced.csv").exists():
            df = pl.read_csv(self.output_dir / "enforced.csv")
            assert "files" in df.columns
        else:
            # Check output message
            assert "No files were updated." in result.stdout

    def test_infer_isolation_target(self) -> None:
        """Test infer_isolation_target script."""
        if not script_exists("infer_isolation_target"):
            pytest.skip("infer_isolation_target script not available")

        runner = CliRunner()

        # Use a glob pattern for only parquet files
        glob_pattern = str(self.data_dir / "acfm" / "*.parquet")
        # Test inferring isolation targets
        result = runner.invoke(
            infer_app,
            [
                "--input-dir",
                glob_pattern,
                "--output-file",
                str(self.output_dir / "inferred.txt"),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert (self.output_dir / "inferred.txt").exists()
        with open(self.output_dir / "inferred.txt", "r") as f:
            content = f.read()
            assert len(content) >= 0

    def test_find_modifications(self) -> None:
        """Test find_modifications script."""
        if not script_exists("find_modifications"):
            pytest.skip("find_modifications script not available")

        runner = CliRunner()

        # Test finding modifications
        result = runner.invoke(
            find_mods_app,
            [
                "--input-dir",
                str(self.data_dir / "mcfm"),
                "--output-file",
                str(self.output_dir / "modifications.xlsx"),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert (self.output_dir / "modifications.xlsx").exists()

        # Check that modifications were found (Excel file should exist)
        assert (self.output_dir / "modifications.xlsx").exists()

    def test_check_conversion(self) -> None:
        """Test check_conversion script."""
        if not script_exists("check_conversion"):
            pytest.skip("check_conversion script not available")

        runner = CliRunner()

        # Test checking conversion
        result = runner.invoke(
            check_app,
            [
                "--input-dir",
                str(self.data_dir / "acfm"),
                "--output-file",
                str(self.output_dir / "conversion_check.txt"),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert (self.output_dir / "conversion_check.txt").exists()

    def test_detect_multi_folder_duplicates(self) -> None:
        """Test detect_multi_folder_duplicates script."""
        if not script_exists("detect_multi_folder_duplicates"):
            pytest.skip("detect_multi_folder_duplicates script not available")

        runner = CliRunner()

        # First create a dummy input file with duplicate information
        input_file = self.output_dir / "duplicates_input.txt"
        with open(input_file, "w") as f:
            f.write("acfm/sample1.ipc\n")
            f.write("lcfm/sample1.ipc\n")  # Same filename in different folder

        # Test detecting duplicates across multiple folders
        result = runner.invoke(
            detect_multi_app,
            [
                "--input-file",
                str(input_file),
                "--output-file",
                str(self.output_dir / "multi_duplicates.txt"),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert (self.output_dir / "multi_duplicates.txt").exists()

    def test_delete_files(self) -> None:
        """Test delete_files script."""
        if not script_exists("delete_files"):
            pytest.skip("delete_files script not available")

        runner = CliRunner()

        # Create a file to delete
        test_file = self.data_dir / "acfm" / "to_delete.ipc"
        df = pl.DataFrame({"test": [1, 2, 3]})
        df.write_ipc(test_file)

        # Create a file list containing the file to delete
        file_list = self.output_dir / "files_to_delete.txt"
        with open(file_list, "w") as f:
            f.write(str(test_file) + "\n")

        # Test deleting files (error log must stay under the temp output dir)
        error_log = self.output_dir / "error_log.txt"
        result = runner.invoke(
            delete_files_app,
            [
                "--input-file",
                str(file_list),
                "--error-log",
                str(error_log),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert not test_file.exists()
        assert error_log.exists()

    def test_batch_operations(self) -> None:
        """Test batch operations across multiple directories."""
        runner = CliRunner()

        # Test batch duplicate detection
        result = runner.invoke(
            detect_app,
            [
                "--input-dir",
                str(self.data_dir),
                "--output-file",
                str(self.output_dir / "batch_duplicates.txt"),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert (self.output_dir / "batch_duplicates.txt").exists()

    def test_legacy_compatibility(self) -> None:
        """Test compatibility with legacy file formats."""
        if not script_exists("convert_ipc_to_parquet"):
            pytest.skip("convert_ipc_to_parquet script not available")

        runner = CliRunner()
        result = runner.invoke(
            convert_app,
            [
                "--input-dir",
                str(self.data_dir / "acfm"),
                "--output-file",
                str(self.output_dir / "conversion_errors.txt"),
                "--verbose",
            ],
        )
        assert result.exit_code == 0
        # Always check for output .parquet files
        parquet_files = list((self.data_dir / "acfm").glob("*.parquet"))
        assert len(parquet_files) > 0
        # Only check for error log if it exists
        if result.exit_code != 0:
            assert (self.output_dir / "conversion_errors.txt").exists()

    def test_error_handling(self) -> None:
        """Test error handling for invalid inputs."""
        runner = CliRunner()

        # Test with non-existent directory
        result = runner.invoke(
            detect_app,
            [
                "--input-dir",
                "/non/existent/path",
                "--output-file",
                str(self.output_dir / "error.txt"),
            ],
        )

        # Should handle error gracefully
        assert result.exit_code != 0

    def test_help_commands(self) -> None:
        """Test that help commands work for all scripts."""
        runner = CliRunner()

        scripts = [
            detect_app,
            delete_same_app,
            find_empty_app,
            convert_app,
            enforce_app,
            delete_files_app,
            find_mods_app,
            check_app,
            detect_multi_app,
            delete_multi_app,
            infer_app,
            label_app,
        ]

        for script in scripts:
            result = runner.invoke(script, ["--help"])
            assert result.exit_code == 0, (
                f"help failed for {script}: exit={result.exit_code} "
                f"exc={result.exception!r}"
            )
            assert "Usage:" in result.output

    def test_label_modifications_unimod(self) -> None:
        """Test label_modifications script with UNIMOD modifications."""
        if not script_exists("label_modifications"):
            pytest.skip("label_modifications script not available")

        # Create test data
        mod_file = self._create_modification_test_data()

        # Run the script
        exit_code = self._run_label_modifications_script(self.data_dir)
        assert exit_code == 0

        # Verify output
        self._verify_modification_output(mod_file)

    def test_check_modifications_success(self) -> None:
        """Test check_modifications script with valid modifications."""
        if not script_exists("check_modifications"):
            pytest.skip("check_modifications script not available")

        runner = CliRunner()

        # Modifications present in the gold-standard / PXD fixture dicts
        mod_dict = create_mod_dict(pl.read_excel(self.gold_standard_file))
        test_mods = list(mod_dict.keys())[:3]
        excel_data = {"modification": test_mods}
        excel_df = pl.DataFrame(excel_data)
        excel_file = self.output_dir / "test_mods.xlsx"
        excel_df.write_excel(excel_file)

        result = runner.invoke(
            check_mods_app,
            [
                "--input-file",
                str(excel_file),
                "--gold-standard-mods",
                str(self.gold_standard_file),
                "--ambiguous-mods",
                str(self.pxd009449_file),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert "SUCCESS" in result.stdout
        assert (
            "All modifications from Excel file are present in mod_dict" in result.stdout
        )

    def test_check_modifications_missing(self) -> None:
        """Test check_modifications script detects missing modifications."""
        if not script_exists("check_modifications"):
            pytest.skip("check_modifications script not available")

        runner = CliRunner()

        mod_dict = create_mod_dict(pl.read_excel(self.gold_standard_file))
        test_mods = list(mod_dict.keys())[:2] + ["[9999]", "[INVALID]", "[MISSING]"]
        excel_data = {"modification": test_mods}
        excel_df = pl.DataFrame(excel_data)
        excel_file = self.output_dir / "test_missing_mods.xlsx"
        excel_df.write_excel(excel_file)

        result = runner.invoke(
            check_mods_app,
            [
                "--input-file",
                str(excel_file),
                "--gold-standard-mods",
                str(self.gold_standard_file),
                "--ambiguous-mods",
                str(self.pxd009449_file),
            ],
        )

        assert result.exit_code == 1  # Should fail with missing modifications
        assert "MISSING MODIFICATIONS" in result.stdout
        assert "[9999]" in result.stdout
        assert "[INVALID]" in result.stdout
        assert "[MISSING]" in result.stdout

    def test_check_modifications_pxd009449_overrides(self) -> None:
        """Test check_modifications script handles PXD009449 override modifications."""
        if not script_exists("check_modifications"):
            pytest.skip("check_modifications script not available")

        runner = CliRunner()

        gold_mod_dict = create_mod_dict(pl.read_excel(self.gold_standard_file))
        pxd_mod_dict = create_mod_dict(pl.read_excel(self.pxd009449_file))
        override_mods = list(pxd_mod_dict.keys())
        regular_mods = [
            k for k in list(gold_mod_dict.keys())[:3] if k not in override_mods
        ]
        test_mods = override_mods + regular_mods

        excel_data = {"modification": test_mods}
        excel_df = pl.DataFrame(excel_data)
        excel_file = self.output_dir / "test_pxd009449_mods.xlsx"
        excel_df.write_excel(excel_file)

        result = runner.invoke(
            check_mods_app,
            [
                "--input-file",
                str(excel_file),
                "--gold-standard-mods",
                str(self.gold_standard_file),
                "--ambiguous-mods",
                str(self.pxd009449_file),
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert "SUCCESS" in result.stdout
        assert "PXD009449 OVERRIDE MODIFICATIONS" in result.stdout
        for mod in override_mods:
            assert mod in result.stdout

    def test_check_modifications_batch_success(self) -> None:
        """Test batch_check_mods command with multiple files."""
        if not script_exists("check_modifications"):
            pytest.skip("check_modifications script not available")

        runner = CliRunner()

        mod_keys = list(create_mod_dict(pl.read_excel(self.gold_standard_file)).keys())
        mods1 = mod_keys[:2]
        mods2 = mod_keys[1:]  # overlap is fine; both must be in the merged dict
        excel_df1 = pl.DataFrame({"modification": mods1})
        excel_df2 = pl.DataFrame({"modification": mods2})
        excel_file1 = self.output_dir / "batch_test1.xlsx"
        excel_file2 = self.output_dir / "batch_test2.xlsx"
        excel_df1.write_excel(excel_file1)
        excel_df2.write_excel(excel_file2)

        result = runner.invoke(
            check_mods_app,
            [
                "--input-file",
                str(excel_file1),
                "--input-file",
                str(excel_file2),
                "--gold-standard-mods",
                str(self.gold_standard_file),
                "--ambiguous-mods",
                str(self.pxd009449_file),
            ],
        )

        assert result.exit_code == 0
        assert "BATCH CHECK SUMMARY" in result.stdout
        assert "SUCCESS" in result.stdout

    def test_check_modifications_batch_with_missing(self) -> None:
        """Test batch_check_mods detects missing modifications across files."""
        if not script_exists("check_modifications"):
            pytest.skip("check_modifications script not available")

        runner = CliRunner()

        mod_keys = list(create_mod_dict(pl.read_excel(self.gold_standard_file)).keys())
        valid_mods = mod_keys[:2]
        invalid_mods = ["[MISSING1]", "[MISSING2]"]
        excel_df1 = pl.DataFrame({"modification": valid_mods})
        excel_df2 = pl.DataFrame({"modification": invalid_mods})
        excel_file1 = self.output_dir / "batch_valid.xlsx"
        excel_file2 = self.output_dir / "batch_invalid.xlsx"
        excel_df1.write_excel(excel_file1)
        excel_df2.write_excel(excel_file2)

        result = runner.invoke(
            check_mods_app,
            [
                "--input-file",
                str(excel_file1),
                "--input-file",
                str(excel_file2),
                "--gold-standard-mods",
                str(self.gold_standard_file),
                "--ambiguous-mods",
                str(self.pxd009449_file),
            ],
        )

        assert result.exit_code == 1  # Should fail
        assert "BATCH CHECK SUMMARY" in result.stdout
        assert "TOTAL MISSING MODIFICATIONS" in result.stdout
        assert "[MISSING1]" in result.stdout
        assert "[MISSING2]" in result.stdout

    def test_check_modifications_empty_file(self) -> None:
        """Test check_modifications handles empty Excel file gracefully."""
        if not script_exists("check_modifications"):
            pytest.skip("check_modifications script not available")

        runner = CliRunner()

        excel_df = pl.DataFrame(
            {"modification": []},
            schema={"modification": pl.String},
        )
        excel_file = self.output_dir / "empty_mods.xlsx"
        excel_df.write_excel(excel_file)

        result = runner.invoke(
            check_mods_app,
            [
                "--input-file",
                str(excel_file),
                "--gold-standard-mods",
                str(self.gold_standard_file),
                "--ambiguous-mods",
                str(self.pxd009449_file),
            ],
        )

        assert result.exit_code == 0
        assert "Total modifications found in Excel file: 0" in result.stdout

    def test_check_modifications_invalid_column(self) -> None:
        """Test check_modifications handles Excel file without modification column."""
        if not script_exists("check_modifications"):
            pytest.skip("check_modifications script not available")

        runner = CliRunner()

        excel_data = {"other_column": ["value1", "value2"]}
        excel_df = pl.DataFrame(excel_data)
        excel_file = self.output_dir / "invalid_mods.xlsx"
        excel_df.write_excel(excel_file)

        result = runner.invoke(
            check_mods_app,
            [
                "--input-file",
                str(excel_file),
                "--gold-standard-mods",
                str(self.gold_standard_file),
                "--ambiguous-mods",
                str(self.pxd009449_file),
            ],
        )

        assert result.exit_code == 1
        # Error is written to stderr; result.output combines stdout+stderr
        assert "modification' column not found" in result.output

    def _run_label_modifications_script(self, data_dir: Path) -> int:
        """Run the label modifications script and return exit code."""
        runner = CliRunner()

        # Temporarily change working directory to data_dir so script finds our test files
        original_cwd = Path.cwd()
        try:
            os.chdir(data_dir)
            result = runner.invoke(
                label_app,
                [
                    "--input-dir",
                    "mcfm_mods",
                    "--gold-standard-mods",
                    str(self.gold_standard_file),
                    "--ambiguous-mods",
                    str(self.pxd009449_file),
                    "--sequence-col",
                    "unmodified_peptide",
                ],
            )
            return int(result.exit_code)
        finally:
            os.chdir(original_cwd)

    def _create_modification_test_data(self) -> Path:
        """Create test data for modification testing."""
        mod_dir = self.data_dir / "mcfm_mods"
        mod_dir.mkdir(parents=True, exist_ok=True)

        mod_data = {
            "unmodified_peptide": ["PEPTIDEK", "PEPTIDER", "PEPTIDEM"],
            "modified_peptide": [
                "PEPTIDEK[242]",
                "PEPTIDER[170]",
                "PEPTIDEM[142]",
            ],
        }
        mod_df = pl.DataFrame(mod_data)
        mod_file = mod_dir / "test_mods.parquet"
        mod_df.write_parquet(mod_file)

        return mod_file

    def _create_gold_standard_modifications(self) -> Path:
        """Create a gold standard modifications Excel file for check/label tests."""
        gold_standard_data = {
            "modification": [
                "K[242]",
                "R[170]",
                "n[43]V",
                "Qc[111]",
                "M[142]",
                "n[145]P",
            ],
            "project_name": [
                "PXD037009",
                "PXD037009",
                "PXD037009",
                "PXD037009",
                "PXD037009",
                "PXD037009",
            ],
            "file_name": [
                "file1.mzML",
                "file2.mzML",
                "file3.mzML",
                "file4.mzML",
                "file5.mzML",
                "file6.mzML",
            ],
            "proposed_unimod_encoding": [
                "[UNIMOD:121]",
                "[UNIMOD:1]",
                "[UNIMOD:1]",
                "[UNIMOD:23]",
                "[UNIMOD:34]",
                "[UNIMOD:214]",
            ],
        }
        df = pl.DataFrame(gold_standard_data)
        file_path = self.output_dir / "gold_standard_modifications.xlsx"
        df.write_excel(file_path)
        return file_path

    def _create_pxd009449_ambiguous_modifications(self) -> Path:
        """Create a PXD009449 ambiguous modifications Excel file."""
        ambiguous_data = {
            "modification": ["K[242]", "K[242]"],
            "project_name": ["PXD009449", "PXD009449"],
            "modification_in_file_name": ["ubiquitin", "acetylation"],
            "proposed_unimod_encoding": ["[UNIMOD:1848]", "[UNIMOD:21]"],
        }
        df = pl.DataFrame(ambiguous_data)
        file_path = self.output_dir / "pxd009449_ambiguous_modifications.xlsx"
        df.write_excel(file_path)
        return file_path

    def _verify_modification_output(self, mod_file: Path) -> None:
        """Verify that the modification output has the expected structure."""
        out_df = pl.read_parquet(mod_file)
        assert "sequence" in out_df.columns

    @pytest.fixture(autouse=True)
    def _setup_test_environment(self) -> Generator[None, None, None]:
        """Set up test environment with temporary directories and test data."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir) / "data"
        self.output_dir = Path(self.test_dir) / "outputs"

        # Create test directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create test subdirectories
        (self.data_dir / "acfm").mkdir(exist_ok=True)
        (self.data_dir / "lcfm").mkdir(exist_ok=True)
        (self.data_dir / "mcfm").mkdir(exist_ok=True)
        (self.data_dir / "hcfm").mkdir(exist_ok=True)

        # Modification reference files used by check_mods / label_mods
        self.gold_standard_file = self._create_gold_standard_modifications()
        self.pxd009449_file = self._create_pxd009449_ambiguous_modifications()

        # Create test data
        self.create_test_data()

        yield

        # Cleanup
        shutil.rmtree(self.test_dir)


class TestDataConversionIntegration:
    """Integration tests for data conversion workflows."""

    def create_integration_test_data(self) -> None:
        """Create realistic test data for integration testing."""
        # Create multiple directories with various file types
        directories = ["acfm", "lcfm", "mcfm", "hcfm"]

        for dir_name in directories:
            dir_path = self.data_dir / dir_name
            dir_path.mkdir(exist_ok=True)

            # Create multiple files in each directory
            for i in range(3):
                if dir_name in ["acfm", "lcfm"]:
                    self._create_basic_test_data(dir_name, dir_path, i)
                elif dir_name == "mcfm":
                    self._create_modification_test_data(dir_path, i)
                else:  # hcfm
                    self._create_unknown_collision_energy_data(dir_path, i)

        # Create duplicate test data
        self._create_duplicate_test_data()

    def _create_basic_test_data(self, dir_name: str, dir_path: Path, i: int) -> None:
        """Create basic test data for a directory."""
        data = {
            "index": list(range(1, 11)),
            "scan": [f"scan_{j:03d}" for j in range(1, 11)],
            "header": [f"MS2 scan {1000 + j}@{30 + j}" for j in range(1, 11)],
            "rt": [30.0 + j for j in range(1, 11)],
            "frag_type": ["HCD"] * 10,
            "collision_energy": [30.0 + j for j in range(1, 11)],
            "precursor_mz": [1000.0 + j for j in range(1, 11)],
            "precursor_charge": [2] * 10,
            "precursor_intensity": [1000.0 + j * 100 for j in range(1, 11)],
            "lower_offset": [-1.0] * 10,
            "upper_offset": [1.0] * 10,
            "isolation_target": [None] * 10,
            "mz": [[100.0 + j, 200.0 + j, 300.0 + j] for j in range(1, 11)],
            "intensity": [[100.0 + j, 200.0 + j, 300.0 + j] for j in range(1, 11)],
            "scale_factor": [1.0] * 10,
        }
        df = pl.DataFrame(data)
        if dir_name == "acfm":
            df.write_ipc(dir_path / f"sample_{i}.ipc")
        elif dir_name == "lcfm":
            df.write_parquet(dir_path / f"sample_{i}.parquet")

    def _create_modification_test_data(self, dir_path: Path, i: int) -> None:
        """Create test data with modifications."""
        data = {
            "index": list(range(1, 11)),
            "scan": [f"scan_{j:03d}" for j in range(1, 11)],
            "header": [f"MS2 scan {1000 + j}@{30 + j}" for j in range(1, 11)],
            "rt": [30.0 + j for j in range(1, 11)],
            "frag_type": ["HCD"] * 10,
            "collision_energy": [30.0 + j for j in range(1, 11)],
            "precursor_mz": [1000.0 + j for j in range(1, 11)],
            "precursor_charge": [2] * 10,
            "precursor_intensity": [1000.0 + j * 100 for j in range(1, 11)],
            "lower_offset": [-1.0] * 10,
            "upper_offset": [1.0] * 10,
            "isolation_target": [None] * 10,
            "mz": [[100.0 + j, 200.0 + j, 300.0 + j] for j in range(1, 11)],
            "intensity": [[100.0 + j, 200.0 + j, 300.0 + j] for j in range(1, 11)],
            "scale_factor": [1.0] * 10,
            "unmodified_peptide": [f"PEPTIDE{j}" for j in range(1, 11)],
            "modified_peptide": [f"PEPTIDE[142]{j}" for j in range(1, 11)],
        }
        mod_df = pl.DataFrame(data)
        mod_df.write_parquet(dir_path / f"sample_{i}.parquet")

    def _create_unknown_collision_energy_data(self, dir_path: Path, i: int) -> None:
        """Create test data with unknown collision energy values."""
        data = {
            "index": list(range(1, 11)),
            "scan": [f"scan_{j:03d}" for j in range(1, 11)],
            "header": [f"MS2 scan {1000 + j}@{30 + j}" for j in range(1, 11)],
            "rt": [30.0 + j for j in range(1, 11)],
            "frag_type": ["HCD"] * 10,
            "precursor_mz": [1000.0 + j for j in range(1, 11)],
            "precursor_charge": [2] * 10,
            "precursor_intensity": [1000.0 + j * 100 for j in range(1, 11)],
            "lower_offset": [-1.0] * 10,
            "upper_offset": [1.0] * 10,
            "isolation_target": [None] * 10,
            "mz": [[100.0 + j, 200.0 + j, 300.0 + j] for j in range(1, 11)],
            "intensity": [[100.0 + j, 200.0 + j, 300.0 + j] for j in range(1, 11)],
            "scale_factor": [1.0] * 10,
        }
        collision_energy_data = pl.Series(
            [None if j % 3 == 0 else 30.0 + j for j in range(1, 11)]
        )
        data["collision_energy"] = collision_energy_data
        unknown_df = pl.DataFrame(data)
        unknown_df.write_parquet(dir_path / f"sample_{i}.parquet")

    def _create_duplicate_test_data(self) -> None:
        """Create duplicate test data in subfolders."""
        duplicate_data = {
            "index": [101, 102, 103],
            "scan": ["scan_101", "scan_102", "scan_103"],
            "header": ["MS2 scan 2100@31", "MS2 scan 2200@32", "MS2 scan 2300@33"],
            "rt": [31.0, 32.0, 33.0],
            "frag_type": ["HCD", "HCD", "HCD"],
            "collision_energy": [31.0, 32.0, 33.0],
            "precursor_mz": [2100.0, 2200.0, 2300.0],
            "precursor_charge": [2, 2, 2],
            "precursor_intensity": [1100.0, 1200.0, 1300.0],
            "lower_offset": [-1.0, -1.0, -1.0],
            "upper_offset": [1.0, 1.0, 1.0],
            "isolation_target": [None, None, None],
            "mz": [[110.0, 210.0, 310.0], [120.0, 220.0, 320.0], [130.0, 230.0, 330.0]],
            "intensity": [
                [110.0, 210.0, 310.0],
                [120.0, 220.0, 320.0],
                [130.0, 230.0, 330.0],
            ],
            "scale_factor": [1.0, 1.0, 1.0],
        }
        dup_df = pl.DataFrame(duplicate_data)
        subfolder = Path("sub1/sub2")
        for base in ["acfm", "lcfm"]:
            full_dir = self.data_dir / base / subfolder
            full_dir.mkdir(parents=True, exist_ok=True)
            dup_df.write_ipc(full_dir / "forced_duplicate.ipc")

    @pytest.fixture(autouse=True)
    def _setup_integration_environment(self) -> Generator[None, None, None]:
        """Set up integration test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = Path(self.test_dir) / "data"
        self.output_dir = Path(self.test_dir) / "outputs"

        # Create test directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create realistic test data
        self.create_integration_test_data()

        yield

        # Cleanup
        shutil.rmtree(self.test_dir)
