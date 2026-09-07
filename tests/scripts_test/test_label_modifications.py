"""Comprehensive test suite for label_modifications script.

This module provides comprehensive tests for the label_modifications script
to ensure it correctly labels modifications in peptide sequences.
"""

import tempfile
import shutil
from pathlib import Path
from typing import Generator
import polars as pl
import pytest
from typer.testing import CliRunner

from scripts.preprocessing.label_modifications import (
    app as label_app,
    create_mod_dict,
    create_n_term_mod_dict,
    create_c_term_mod_dict,
    create_file_specific_mod_dicts,
    replace_modifications,
    read_gold_standard_modifications,
    read_pxd009449_ambiguous_modifications,
)


class TestLabelModifications:
    """Test suite for label_modifications script."""

    def test_create_mod_dict(self) -> None:
        """Test creation of residue modification dictionary."""
        df = read_gold_standard_modifications(str(self.gold_standard_file))
        mod_dict = create_mod_dict(df)

        # Should extract amino acid and append to UNIMOD encoding
        assert "K[242]" in mod_dict
        assert mod_dict["K[242]"] == "K[UNIMOD:121]"
        assert "M[142]" in mod_dict
        assert mod_dict["M[142]"] == "M[UNIMOD:34]"

        # Should not include N-terminal or C-terminal modifications
        assert "n[43]V" not in mod_dict
        assert "Qc[111]" not in mod_dict

    def test_create_n_term_mod_dict(self) -> None:
        """Test creation of N-terminal modification dictionary."""
        df = read_gold_standard_modifications(str(self.gold_standard_file))
        n_term_dict = create_n_term_mod_dict(df)

        # Should extract trailing amino acid and append with hyphen
        assert "n[43]V" in n_term_dict
        assert n_term_dict["n[43]V"] == "[UNIMOD:1]-V"
        assert "n[145]P" in n_term_dict
        assert n_term_dict["n[145]P"] == "[UNIMOD:214]-P"

        # Should not include residue or C-terminal modifications
        assert "K[242]" not in n_term_dict
        assert "Qc[111]" not in n_term_dict

    def test_create_c_term_mod_dict(self) -> None:
        """Test creation of C-terminal modification dictionary."""
        df = read_gold_standard_modifications(str(self.gold_standard_file))
        c_term_dict = create_c_term_mod_dict(df)

        # Should extract leading amino acid and prepend with hyphen
        assert "Qc[111]" in c_term_dict
        assert c_term_dict["Qc[111]"] == "Q-[UNIMOD:23]"

        # Should not include residue or N-terminal modifications
        assert "K[242]" not in c_term_dict
        assert "n[43]V" not in c_term_dict

    def test_create_mod_dict_multiple_mods_error(self) -> None:
        """Test that multiple modifications on single amino acid raises error."""
        # Create data with multiple modifications
        data = {
            "modification": ["K[242][170]"],
            "project_name": ["TEST"],
            "file_name": ["test.mzML"],
            "proposed_unimod_encoding": ["[UNIMOD:1]"],
        }
        df = pl.DataFrame(data)

        with pytest.raises(
            ValueError, match="multiple modifications on a single amino acid"
        ):
            create_mod_dict(df)

    def test_create_c_term_mod_dict_complex_error(self) -> None:
        """Test that C-terminal modifications with preceding residue mods raise error."""
        # Create data with complex C-terminal modification
        data = {
            "modification": ["K[170]c[123]"],
            "project_name": ["TEST"],
            "file_name": ["test.mzML"],
            "proposed_unimod_encoding": ["[UNIMOD:1]"],
        }
        df = pl.DataFrame(data)

        with pytest.raises(ValueError, match="preceding residue modifications"):
            create_c_term_mod_dict(df)

    def test_replace_modifications(self) -> None:
        """Test modification replacement in peptide sequences."""
        mod_dict = {"K[242]": "K[UNIMOD:121]", "M[142]": "M[UNIMOD:34]"}
        n_term_dict = {"n[43]V": "[UNIMOD:1]-V"}
        c_term_dict = {"Qc[111]": "Q-[UNIMOD:23]"}

        # Test residue modification
        result = replace_modifications(
            "PEPTIDEK[242]", mod_dict, n_term_dict, c_term_dict
        )
        assert result == "PEPTIDEK[UNIMOD:121]"

        # Test N-terminal modification
        result = replace_modifications(
            "n[43]VPEPTIDE", mod_dict, n_term_dict, c_term_dict
        )
        assert result == "[UNIMOD:1]-VPEPTIDE"

        # Test C-terminal modification
        result = replace_modifications(
            "PEPTIDEQc[111]", mod_dict, n_term_dict, c_term_dict
        )
        assert result == "PEPTIDEQ-[UNIMOD:23]"

        # Test multiple modifications
        result = replace_modifications(
            "n[43]VPEPTIDEK[242]M[142]Qc[111]",
            mod_dict,
            n_term_dict,
            c_term_dict,
        )
        assert result == "[UNIMOD:1]-VPEPTIDEK[UNIMOD:121]M[UNIMOD:34]Q-[UNIMOD:23]"

    def test_create_file_specific_mod_dicts(self) -> None:
        """Test file-specific dictionary creation for PXD009449 with one-to-many mapping."""
        df = read_pxd009449_ambiguous_modifications(str(self.pxd009449_file))

        # Test with ubiquitin filename - K[242] should map to K[UNIMOD:1848]
        file_name = "ubiquitin_sample.mzML"
        mod_dict, n_term_dict, c_term_dict = create_file_specific_mod_dicts(
            file_name, df
        )

        # Should find K[242] with ubiquitin-specific encoding
        assert "K[242]" in mod_dict
        assert mod_dict["K[242]"] == "K[UNIMOD:1848]"
        assert n_term_dict == {}
        assert c_term_dict == {}

        # Test with acetylation filename - K[242] should map to K[UNIMOD:21] (different from ubiquitin!)
        file_name = "acetylation_sample.mzML"
        mod_dict, n_term_dict, c_term_dict = create_file_specific_mod_dicts(
            file_name, df
        )

        # Same modification, different encoding based on filename
        assert "K[242]" in mod_dict
        assert mod_dict["K[242]"] == "K[UNIMOD:21]"  # Different from ubiquitin case
        assert n_term_dict == {}
        assert c_term_dict == {}

        # Test with non-matching filename
        file_name = "other_sample.mzML"
        mod_dict, n_term_dict, c_term_dict = create_file_specific_mod_dicts(
            file_name, df
        )

        # Should return empty dicts when no filename matches
        assert mod_dict == {}
        assert n_term_dict == {}
        assert c_term_dict == {}

    def test_label_modifications_full_workflow(self) -> None:
        """Test full workflow of label_modifications script."""
        # Create test parquet file with modifications
        test_data = {
            "peptide": ["PEPTIDEK", "PEPTIDER", "VPEPTIDE"],
            "modified_peptide": [
                "PEPTIDEK[242]",
                "PEPTIDER[170]",
                "n[43]VPEPTIDE",
            ],
        }
        df = pl.DataFrame(test_data)
        parquet_file = self.data_dir / "lcfm_splits" / "test_mods.parquet"
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_file)

        # Run the script
        runner = CliRunner()
        result = runner.invoke(
            label_app,
            [
                "label-mods",
                str(self.data_dir / "lcfm_splits"),
                str(self.gold_standard_file),
                str(self.pxd009449_file),
                "--sequence-col",
                "peptide",
                "--modified-sequence-col",
                "modified_peptide",
            ],
        )

        assert result.exit_code == 0

        # Verify output
        output_df = pl.read_parquet(parquet_file)
        assert "sequence" in output_df.columns
        assert "unmodified_peptide" in output_df.columns

        # Check that modifications were replaced
        sequences = output_df["sequence"].to_list()
        assert (
            "K[UNIMOD:121]" in sequences[0]
            or "R[UNIMOD:1]" in sequences[1]
            or "[UNIMOD:1]-V" in sequences[2]
        )

    def test_label_modifications_pxd009449_file_specific(self) -> None:
        """Test PXD009449 file-specific modification labeling with one-to-many mapping."""
        # Test 1: File with "ubiquitin" in name - K[242] should map to [UNIMOD:1848]K
        test_data = {
            "peptide": ["PEPTIDEK"],
            "modified_peptide": ["PEPTIDEK[242]"],
        }
        df = pl.DataFrame(test_data)
        parquet_file = self.data_dir / "PXD009449" / "ubiquitin_sample.parquet"
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_file)

        runner = CliRunner()
        result = runner.invoke(
            label_app,
            [
                "label-mods",
                str(self.data_dir / "PXD009449"),
                str(self.gold_standard_file),
                str(self.pxd009449_file),
                "--sequence-col",
                "peptide",
                "--modified-sequence-col",
                "modified_peptide",
            ],
        )

        assert result.exit_code == 0

        # Verify output - should use ubiquitin-specific encoding
        output_df = pl.read_parquet(parquet_file)
        sequences = output_df["sequence"].to_list()
        # Should use K[UNIMOD:1848] from PXD009449 file, not K[UNIMOD:121] from gold standard
        assert "K[UNIMOD:1848]" in sequences[0]

        # Test 2: Same modification in file with "acetylation" in name - should map differently
        # Remove the first parquet file to avoid reprocessing it (it no longer has the peptide column)
        parquet_file.unlink()

        test_data2 = {
            "peptide": ["PEPTIDEK"],
            "modified_peptide": ["PEPTIDEK[242]"],
        }
        df2 = pl.DataFrame(test_data2)
        parquet_file2 = self.data_dir / "PXD009449" / "acetylation_sample.parquet"
        df2.write_parquet(parquet_file2)

        result2 = runner.invoke(
            label_app,
            [
                "label-mods",
                str(self.data_dir / "PXD009449"),
                str(self.gold_standard_file),
                str(self.pxd009449_file),
                "--sequence-col",
                "peptide",
                "--modified-sequence-col",
                "modified_peptide",
            ],
        )

        assert result2.exit_code == 0

        # Verify output - should use acetylation-specific encoding (different from ubiquitin!)
        output_df2 = pl.read_parquet(parquet_file2)
        sequences2 = output_df2["sequence"].to_list()
        # Same modification K[242], but different encoding based on filename
        assert "K[UNIMOD:21]" in sequences2[0]  # Different from ubiquitin case
        assert (
            "K[UNIMOD:1848]" not in sequences2[0]
        )  # Should NOT use ubiquitin encoding

    def test_one_to_many_mapping_same_modification_different_files(self) -> None:
        """Test that the same modification maps to different UNIMOD encodings based on filename."""
        df = read_pxd009449_ambiguous_modifications(str(self.pxd009449_file))

        # The same modification K[242] appears twice with different encodings
        # based on modification_in_file_name
        ubiquitin_dict, _, _ = create_file_specific_mod_dicts("ubiquitin_file.mzML", df)
        acetylation_dict, _, _ = create_file_specific_mod_dicts(
            "acetylation_file.mzML", df
        )

        # Both should have K[242], but with different encodings
        assert "K[242]" in ubiquitin_dict
        assert "K[242]" in acetylation_dict
        assert ubiquitin_dict["K[242]"] == "K[UNIMOD:1848]"
        assert acetylation_dict["K[242]"] == "K[UNIMOD:21]"
        # Verify they are different
        assert ubiquitin_dict["K[242]"] != acetylation_dict["K[242]"]

    def test_label_modifications_no_matching_file(self) -> None:
        """Test PXD009449 file without matching modification_in_file_name."""
        # Create test parquet file in PXD009449 directory with non-matching filename
        test_data = {
            "peptide": ["PEPTIDEK"],
            "modified_peptide": ["PEPTIDEK[242]"],
        }
        df = pl.DataFrame(test_data)
        parquet_file = self.data_dir / "PXD009449" / "other_sample.parquet"
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_file)

        # Run the script
        runner = CliRunner()
        result = runner.invoke(
            label_app,
            [
                "label-mods",
                str(self.data_dir / "PXD009449"),
                str(self.gold_standard_file),
                str(self.pxd009449_file),
                "--sequence-col",
                "peptide",
                "--modified-sequence-col",
                "modified_peptide",
            ],
        )

        assert result.exit_code == 0

        # Verify output - should use gold standard modification
        output_df = pl.read_parquet(parquet_file)
        sequences = output_df["sequence"].to_list()
        # Should use K[UNIMOD:121] from gold standard since filename doesn't match
        assert "K[UNIMOD:121]" in sequences[0]

    def test_label_modifications_drop_old_modifications(self) -> None:
        """Test label_modifications with drop_old_modifications flag."""
        test_data = {
            "peptide": ["PEPTIDEK"],
            "modified_peptide": ["PEPTIDEK[242]"],
        }
        df = pl.DataFrame(test_data)
        parquet_file = self.data_dir / "lcfm_splits" / "test_drop.parquet"
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_file)

        # Run the script with drop flag
        runner = CliRunner()
        result = runner.invoke(
            label_app,
            [
                "label-mods",
                str(self.data_dir / "lcfm_splits"),
                str(self.gold_standard_file),
                str(self.pxd009449_file),
                "--sequence-col",
                "peptide",
                "--modified-sequence-col",
                "modified_peptide",
                "--drop-old-modifications",
            ],
        )

        assert result.exit_code == 0

        # Verify that modified_peptide column was dropped
        output_df = pl.read_parquet(parquet_file)
        assert "modified_peptide" not in output_df.columns
        assert "sequence" in output_df.columns

    def test_label_modifications_empty_modifications(self) -> None:
        """Test label_modifications with empty modification dictionaries."""
        # Create data with no modifications
        empty_data: dict[str, list[str]] = {
            "modification": [],
            "project_name": [],
            "file_name": [],
            "proposed_unimod_encoding": [],
        }
        df = pl.DataFrame(empty_data)
        empty_file = self.output_dir / "empty_modifications.xlsx"
        df.write_excel(empty_file)

        # Should return empty dictionaries
        mod_dict = create_mod_dict(df)
        n_term_dict = create_n_term_mod_dict(df)
        c_term_dict = create_c_term_mod_dict(df)

        assert mod_dict == {}
        assert n_term_dict == {}
        assert c_term_dict == {}

    def test_label_modifications_mixed_terminal_and_residue(self) -> None:
        """Test label_modifications with mixed terminal and residue modifications."""
        test_data = {
            "peptide": ["PEPTIDEKQ"],
            "modified_peptide": ["n[43]VPEPTIDEK[242]Qc[111]"],
        }
        df = pl.DataFrame(test_data)
        parquet_file = self.data_dir / "lcfm_splits" / "test_mixed.parquet"
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(parquet_file)

        # Run the script
        runner = CliRunner()
        result = runner.invoke(
            label_app,
            [
                "label-mods",
                str(self.data_dir / "lcfm_splits"),
                str(self.gold_standard_file),
                str(self.pxd009449_file),
                "--sequence-col",
                "peptide",
                "--modified-sequence-col",
                "modified_peptide",
            ],
        )

        assert result.exit_code == 0

        # Verify all modifications were replaced
        output_df = pl.read_parquet(parquet_file)
        sequences = output_df["sequence"].to_list()
        sequence = sequences[0]
        assert "[UNIMOD:1]-V" in sequence  # N-terminal
        assert "K[UNIMOD:121]" in sequence  # Residue
        assert "Q-[UNIMOD:23]" in sequence  # C-terminal

    def _create_gold_standard_modifications(self) -> Path:
        """Create a realistic gold standard modifications Excel file."""
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
        """Create a realistic PXD009449 ambiguous modifications Excel file."""
        ambiguous_data = {
            "modification": [
                "K[242]",
                "K[242]",
            ],
            "project_name": ["PXD009449", "PXD009449"],
            "modification_in_file_name": [
                "ubiquitin",
                "acetylation",
            ],
            "proposed_unimod_encoding": [
                "[UNIMOD:1848]",
                "[UNIMOD:21]",
            ],
        }
        df = pl.DataFrame(ambiguous_data)
        file_path = self.output_dir / "pxd009449_ambiguous_modifications.xlsx"
        df.write_excel(file_path)
        return file_path

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
        (self.data_dir / "lcfm_splits").mkdir(exist_ok=True)
        (self.data_dir / "PXD009449").mkdir(exist_ok=True)

        # Create test modification files
        self.gold_standard_file = self._create_gold_standard_modifications()
        self.pxd009449_file = self._create_pxd009449_ambiguous_modifications()

        yield

        # Cleanup
        shutil.rmtree(self.test_dir)
