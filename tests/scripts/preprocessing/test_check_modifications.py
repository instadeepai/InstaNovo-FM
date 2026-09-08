"""Tests for check_modifications."""

import polars as pl
from typer.testing import CliRunner

from scripts.preprocessing.check_modifications import app as check_mods_app
from scripts.preprocessing.label_modifications import create_mod_dict


def test_check_modifications_success(preprocessing_env) -> None:
    """Test check_modifications script with valid modifications."""
    runner = CliRunner()
    mod_dict = create_mod_dict(pl.read_excel(preprocessing_env.gold_standard_file))
    test_mods = list(mod_dict.keys())[:3]
    excel_file = preprocessing_env.output_dir / "test_mods.xlsx"
    pl.DataFrame({"modification": test_mods}).write_excel(excel_file)

    result = runner.invoke(
        check_mods_app,
        [
            "--input-file",
            str(excel_file),
            "--gold-standard-mods",
            str(preprocessing_env.gold_standard_file),
            "--ambiguous-mods",
            str(preprocessing_env.pxd009449_file),
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert "SUCCESS" in result.output
    assert "All modifications from Excel file are present in mod_dict" in result.output


def test_check_modifications_missing(preprocessing_env) -> None:
    """Test check_modifications script detects missing modifications."""
    runner = CliRunner()
    mod_dict = create_mod_dict(pl.read_excel(preprocessing_env.gold_standard_file))
    test_mods = list(mod_dict.keys())[:2] + ["[9999]", "[INVALID]", "[MISSING]"]
    excel_file = preprocessing_env.output_dir / "test_missing_mods.xlsx"
    pl.DataFrame({"modification": test_mods}).write_excel(excel_file)

    result = runner.invoke(
        check_mods_app,
        [
            "--input-file",
            str(excel_file),
            "--gold-standard-mods",
            str(preprocessing_env.gold_standard_file),
            "--ambiguous-mods",
            str(preprocessing_env.pxd009449_file),
        ],
    )

    assert result.exit_code == 1
    assert "MISSING MODIFICATIONS" in result.output
    assert "[9999]" in result.output
    assert "[INVALID]" in result.output
    assert "[MISSING]" in result.output


def test_check_modifications_pxd009449_overrides(
    preprocessing_env,
) -> None:
    """Test check_modifications script handles PXD009449 override modifications."""
    runner = CliRunner()
    gold_mod_dict = create_mod_dict(pl.read_excel(preprocessing_env.gold_standard_file))
    pxd_mod_dict = create_mod_dict(pl.read_excel(preprocessing_env.pxd009449_file))
    override_mods = list(pxd_mod_dict.keys())
    regular_mods = [k for k in list(gold_mod_dict.keys())[:3] if k not in override_mods]
    test_mods = override_mods + regular_mods
    excel_file = preprocessing_env.output_dir / "test_pxd009449_mods.xlsx"
    pl.DataFrame({"modification": test_mods}).write_excel(excel_file)

    result = runner.invoke(
        check_mods_app,
        [
            "--input-file",
            str(excel_file),
            "--gold-standard-mods",
            str(preprocessing_env.gold_standard_file),
            "--ambiguous-mods",
            str(preprocessing_env.pxd009449_file),
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert "SUCCESS" in result.output
    assert "PXD009449 OVERRIDE MODIFICATIONS" in result.output
    for mod in override_mods:
        assert mod in result.output


def test_check_modifications_batch_success(preprocessing_env) -> None:
    """Test batch_check_mods command with multiple files."""
    runner = CliRunner()
    mod_keys = list(
        create_mod_dict(pl.read_excel(preprocessing_env.gold_standard_file)).keys()
    )
    excel_file1 = preprocessing_env.output_dir / "batch_test1.xlsx"
    excel_file2 = preprocessing_env.output_dir / "batch_test2.xlsx"
    pl.DataFrame({"modification": mod_keys[:2]}).write_excel(excel_file1)
    pl.DataFrame({"modification": mod_keys[1:]}).write_excel(excel_file2)

    result = runner.invoke(
        check_mods_app,
        [
            "--input-file",
            str(excel_file1),
            "--input-file",
            str(excel_file2),
            "--gold-standard-mods",
            str(preprocessing_env.gold_standard_file),
            "--ambiguous-mods",
            str(preprocessing_env.pxd009449_file),
        ],
    )

    assert result.exit_code == 0
    assert "BATCH CHECK SUMMARY" in result.output
    assert "SUCCESS" in result.output


def test_check_modifications_batch_with_missing(
    preprocessing_env,
) -> None:
    """Test batch_check_mods detects missing modifications across files."""
    runner = CliRunner()
    mod_keys = list(
        create_mod_dict(pl.read_excel(preprocessing_env.gold_standard_file)).keys()
    )
    excel_file1 = preprocessing_env.output_dir / "batch_valid.xlsx"
    excel_file2 = preprocessing_env.output_dir / "batch_invalid.xlsx"
    pl.DataFrame({"modification": mod_keys[:2]}).write_excel(excel_file1)
    pl.DataFrame({"modification": ["[MISSING1]", "[MISSING2]"]}).write_excel(
        excel_file2
    )

    result = runner.invoke(
        check_mods_app,
        [
            "--input-file",
            str(excel_file1),
            "--input-file",
            str(excel_file2),
            "--gold-standard-mods",
            str(preprocessing_env.gold_standard_file),
            "--ambiguous-mods",
            str(preprocessing_env.pxd009449_file),
        ],
    )

    assert result.exit_code == 1
    assert "BATCH CHECK SUMMARY" in result.output
    assert "TOTAL MISSING MODIFICATIONS" in result.output
    assert "[MISSING1]" in result.output
    assert "[MISSING2]" in result.output


def test_check_modifications_empty_file(preprocessing_env) -> None:
    """Test check_modifications handles empty Excel file gracefully."""
    runner = CliRunner()
    excel_file = preprocessing_env.output_dir / "empty_mods.xlsx"
    pl.DataFrame(
        {"modification": []},
        schema={"modification": pl.String},
    ).write_excel(excel_file)

    result = runner.invoke(
        check_mods_app,
        [
            "--input-file",
            str(excel_file),
            "--gold-standard-mods",
            str(preprocessing_env.gold_standard_file),
            "--ambiguous-mods",
            str(preprocessing_env.pxd009449_file),
        ],
    )

    assert result.exit_code == 0
    assert "Total modifications found in Excel file: 0" in result.output


def test_check_modifications_invalid_column(preprocessing_env) -> None:
    """Test check_modifications handles Excel file without modification column."""
    runner = CliRunner()
    excel_file = preprocessing_env.output_dir / "invalid_mods.xlsx"
    pl.DataFrame({"other_column": ["value1", "value2"]}).write_excel(excel_file)

    result = runner.invoke(
        check_mods_app,
        [
            "--input-file",
            str(excel_file),
            "--gold-standard-mods",
            str(preprocessing_env.gold_standard_file),
            "--ambiguous-mods",
            str(preprocessing_env.pxd009449_file),
        ],
    )

    assert result.exit_code == 1
    assert "modification' column not found" in result.output
