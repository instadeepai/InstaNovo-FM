"""Tests for find_modifications helpers."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from scripts.preprocessing.find_modifications import extract_modifications

# Same pattern used by find_modifications().
_MOD_PATTERN = re.compile(
    r"(?:"
    r"[A-Z](?:\[[0-9]+\])+|"
    r"[a-z](?:\[[0-9]+\])+[A-Z]|"
    r"[A-Z](?:\[[0-9]+\])*c(?:\[[0-9]+\])+"
    r")"
)


def test_extract_modifications_finds_bracket_tokens(tmp_path: Path) -> None:
    """Residue-bracket forms are extracted from modified_peptide."""
    path = tmp_path / "proj" / "sample.parquet"
    path.parent.mkdir()
    pl.DataFrame(
        {
            "modified_peptide": ["PEPTIDE[142]K"],
            "scan": ["scan_001"],
            "header": ["MS2 scan 1234.5@30.0"],
        }
    ).write_parquet(path)

    result = extract_modifications(str(path), _MOD_PATTERN)

    assert result is not None
    assert result["modification"].to_list() == ["E[142]"]


def test_extract_modifications_finds_n_terminal_tokens(tmp_path: Path) -> None:
    """N-terminal lowercase+bracket forms are extracted from modified_peptide."""
    path = tmp_path / "proj" / "sample.parquet"
    path.parent.mkdir()
    pl.DataFrame(
        {
            "modified_peptide": ["n[43]VPEPTIDE"],
            "scan": ["scan_001"],
            "header": ["MS2 scan 1234.5@30.0"],
        }
    ).write_parquet(path)

    result = extract_modifications(str(path), _MOD_PATTERN)

    assert result is not None
    assert result["modification"].to_list() == ["n[43]V"]


def test_extract_modifications_finds_double_mods(tmp_path: Path) -> None:
    """Stacked brackets on one residue are kept as a single modification token."""
    path = tmp_path / "proj" / "sample.parquet"
    path.parent.mkdir()
    pl.DataFrame(
        {
            "modified_peptide": ["PEPTIDEK[123][456]"],
            "scan": ["scan_001"],
            "header": ["MS2 scan 1234.5@30.0"],
        }
    ).write_parquet(path)

    result = extract_modifications(str(path), _MOD_PATTERN)

    assert result is not None
    assert result["modification"].to_list() == ["K[123][456]"]


def test_extract_modifications_returns_none_when_no_brackets(tmp_path: Path) -> None:
    """Unmodified peptides yield no inventory rows."""
    path = tmp_path / "proj" / "sample.parquet"
    path.parent.mkdir()
    pl.DataFrame(
        {
            "modified_peptide": ["PEPTIDEK"],
            "scan": ["scan_001"],
            "header": ["MS2 scan 1234.5@30.0"],
        }
    ).write_parquet(path)

    assert extract_modifications(str(path), _MOD_PATTERN) is None
