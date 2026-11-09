from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from scripts.preprocessing.frag_type_from_header import corrected_frag_type, fix_file, header_frag_type


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("FTMS + c NSI d Full ms2 667.3215@hcd30.00 [100.0000-1345.0000]", "HCD"),
        ("ITMS + c NSI r d Full ms2 350.8658@hcd28.00 [100.0000-1063.0000]", "HCD"),
        ("ITMS + c NSI r d Full ms2 350.8658@cid35.00 [95.0000-1063.0000]", "CID"),
        ("FTMS + p NSI d Full ms2 600.3000@etd50.00 [120.0000-2000.0000]", "ETD"),
        ("FTMS + p NSI d Full ms2 600.3000@etd50.00@hcd25.00 [120.0000-2000.0000]", "EThcD"),
        ("FTMS + p NSI d Full ms2 600.3000@etd50.00@cid35.00 [120.0000-2000.0000]", "ETciD"),
        ("FTMS + p NSI d Full ms2 600.3000@uvpd30.00 [120.0000-2000.0000]", "UVPD"),
        ("FTMS + p NSI Full ms [350.0000-1800.0000]", None),
        (None, None),
    ],
)
def test_header_frag_type(header: str | None, expected: str | None) -> None:
    """Every activation method in the filter string counts, in order."""
    frame = pl.DataFrame({"header": [header]}, schema={"header": pl.String})
    assert frame.select(header_frag_type(pl.col("header")))[0, 0] == expected


def test_corrected_frag_type_keeps_values_that_agree() -> None:
    """HCID and HCD both mean HCD; a missing or uninformative header keeps the stored value."""
    frame = pl.DataFrame(
        {
            "frag_type": ["HCID", "HCD", "ETD", "CID", "ETD", "HCID"],
            "header": ["x@hcd30", "x@hcd30", "x@etd50@hcd25", "x@etd50@cid35", None, "x Full ms2 500"],
        }
    )
    assert frame.select(corrected_frag_type(pl.col("frag_type"), pl.col("header")))["frag_type"].to_list() == [
        "HCID",
        "HCD",
        "EThcD",
        "ETciD",
        "ETD",
        "HCID",
    ]


def _write(path: Path) -> None:
    pl.DataFrame(
        {"frag_type": ["ETD", "ETD", "HCID"], "header": ["x@etd50@hcd25", "x@etd50", "x@hcd30"], "sequence": ["PEPTIDEK"] * 3}
    ).write_parquet(path)


def test_fix_file_rewrites_only_disagreeing_rows(tmp_path: Path) -> None:
    """The file is rewritten in place, other columns untouched, and the change is reported."""
    path = tmp_path / "run.parquet"
    _write(path)
    changes = fix_file(str(path), dry_run=False)
    assert changes.rows() == [("ETD", "EThcD", 1)]
    frame = pl.read_parquet(path)
    assert frame["frag_type"].to_list() == ["EThcD", "ETD", "HCID"]
    assert frame["sequence"].to_list() == ["PEPTIDEK"] * 3


def test_dry_run_leaves_the_file_alone(tmp_path: Path) -> None:
    """A dry run reports the same change and writes nothing."""
    path = tmp_path / "run.parquet"
    _write(path)
    before = path.read_bytes()
    assert fix_file(str(path), dry_run=True).rows() == [("ETD", "EThcD", 1)]
    assert path.read_bytes() == before


def test_files_without_a_header_are_skipped(tmp_path: Path) -> None:
    """Sources converted without scan headers have nothing to correct."""
    path = tmp_path / "mgf.parquet"
    pl.DataFrame({"frag_type": ["HCD"], "sequence": ["PEPTIDEK"]}).write_parquet(path)
    assert fix_file(str(path), dry_run=False).height == 0
