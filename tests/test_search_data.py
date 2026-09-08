"""Guards on data/search_data.xlsx.

Two things are checked:

1. The file carries no internal filesystem paths. The upstream copy has a
   ``file path`` column holding Windows UNC paths under an internal host.  Our
   copy retains that column only as a raw-file-name lookup key: each value is
   the basename of the upstream path, with no directory information.

2. The analysis-relevant content still matches the Figshare original. Run with
   ``--figshare-search-data=<path>`` (or set FIGSHARE_SEARCH_DATA) pointing at
   the downloaded original; the comparison is skipped when it is unavailable.
   Content is compared column-wise on the columns the figure notebooks read,
   not byte-wise, because re-saving an .xlsx changes the bytes but not the data.
"""

from __future__ import annotations

import ntpath
import os
import pathlib
import re

import pandas as pd
import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
LOCAL = REPO / "data" / "search_data.xlsx"
SHEET = "search_data"

# Columns the notebooks actually read (see notebooks/figure_1.ipynb).
USED_COLUMNS = [
    "project",
    "acquisition",
    "detector",
    "fragmentation",
    "instrument",
    "enzyme",
    "modifications",
    "organism",
]

RAW_FILENAME_COLUMN = "file path"

PATH_PATTERNS = [
    re.compile(r"\\\\[A-Za-z0-9._-]+\\"),  # \\host\share
    re.compile(r"^[A-Za-z]:\\"),  # C:\ or O:\ mapped drive
    re.compile(r"ait-pdfs", re.I),  # the specific internal host
]


@pytest.fixture(scope="module")
def local() -> pd.DataFrame:
    assert LOCAL.is_file(), f"missing {LOCAL.relative_to(REPO)}"
    return pd.read_excel(LOCAL, sheet_name=SHEET)


def test_expected_columns_present(local: pd.DataFrame) -> None:
    missing = [c for c in USED_COLUMNS if c not in local.columns]
    assert not missing, f"columns the notebooks read are missing: {missing}"


def test_raw_filename_column_is_present_and_sanitised(local: pd.DataFrame) -> None:
    assert RAW_FILENAME_COLUMN in local.columns
    filenames = local[RAW_FILENAME_COLUMN].dropna().astype(str)
    assert not filenames.empty
    assert (
        filenames.map(ntpath.basename).eq(filenames).all()
    ), "file path must contain only raw file names, never directory paths"
    assert not filenames.str.contains(r"[\\\\/]", regex=True).any()


def test_no_internal_paths_in_any_cell(local: pd.DataFrame) -> None:
    text = local.astype(str)
    offenders: dict[str, int] = {}
    for col in text.columns:
        n = sum(int(text[col].str.contains(p, regex=True, na=False).sum()) for p in PATH_PATTERNS)
        if n:
            offenders[col] = n
    assert not offenders, f"cells containing internal paths, by column: {offenders}"


def _figshare_path(request: pytest.FixtureRequest) -> pathlib.Path | None:
    raw = request.config.getoption("--figshare-search-data") or os.environ.get(
        "FIGSHARE_SEARCH_DATA"
    )
    if not raw:
        return None
    p = pathlib.Path(raw).expanduser()
    return p if p.is_file() else None


def test_matches_figshare_original(request: pytest.FixtureRequest, local: pd.DataFrame) -> None:
    """Our sanitised copy must agree with the Figshare original where it overlaps."""
    original = _figshare_path(request)
    if original is None:
        pytest.skip(
            "Figshare original not provided. Pass --figshare-search-data=<path> or set "
            "FIGSHARE_SEARCH_DATA. The published file is at "
            "https://figshare.com/ndownloader/files/67963378 (returns HTTP 202 while the "
            "deposit is unpublished)."
        )

    other = pd.read_excel(original, sheet_name=SHEET)
    assert len(other) == len(
        local
    ), f"row count differs: Figshare {len(other):,} vs local {len(local):,}"

    shared = [c for c in USED_COLUMNS if c in other.columns]
    assert shared, f"none of {USED_COLUMNS} present in the Figshare file"

    a = local[shared].sort_values(shared).reset_index(drop=True).astype(str)
    b = other[shared].sort_values(shared).reset_index(drop=True).astype(str)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)

    assert set(other.columns) == set(local.columns)
    expected_filenames = other[RAW_FILENAME_COLUMN].map(ntpath.basename)
    pd.testing.assert_series_equal(
        local[RAW_FILENAME_COLUMN], expected_filenames, check_names=False
    )
