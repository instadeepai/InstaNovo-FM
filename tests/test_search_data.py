"""Guards on data/search_data.xlsx.

Two things are checked:

1. The file carries no internal filesystem paths. The upstream copy has a
   ``file path`` column holding ~25.8k Windows UNC paths under an internal
   host, which exposes an internal hostname, a team's folder tree and a
   colleague's name. This repository is public, so our copy is a deliberate
   sanitised derivative: the column is dropped. It is therefore *not*
   byte-identical to the Figshare original, by design.

2. The analysis-relevant content still matches the Figshare original. Run with
   ``--figshare-search-data=<path>`` (or set FIGSHARE_SEARCH_DATA) pointing at
   the downloaded original; the comparison is skipped when it is unavailable.
   Content is compared column-wise on the columns the figure notebooks read,
   not byte-wise, because re-saving an .xlsx changes the bytes but not the data.
"""

from __future__ import annotations

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

# Dropped from our copy because it leaks internal infrastructure.
FORBIDDEN_COLUMNS = ["file path"]

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


def test_no_forbidden_columns(local: pd.DataFrame) -> None:
    present = [c for c in FORBIDDEN_COLUMNS if c in local.columns]
    assert not present, (
        f"{present} must not be committed: it holds internal filesystem paths "
        "and this repository is public"
    )


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

    # The only intended difference is the removal of the path column.
    extra = sorted(set(other.columns) - set(local.columns))
    assert extra == FORBIDDEN_COLUMNS, (
        f"unexpected columns present upstream but not locally: "
        f"{[c for c in extra if c not in FORBIDDEN_COLUMNS]}"
    )
