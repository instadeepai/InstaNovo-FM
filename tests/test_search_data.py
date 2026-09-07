"""Guards on data/search_data.xlsx.

Two things are checked:

1. The file carries no internal filesystem paths. Upstream, the ``file path``
   column held ~25.8k Windows UNC paths under an internal host, exposing an
   internal hostname, a team's folder tree and a colleague's name. This
   repository is public, so our copy is a deliberate sanitised derivative: the
   column is reduced to ``<accession>/<filename>``. That is exactly what the
   foundation model's ``_extract_lookup_key`` consumes (parent folder plus
   filename) and every parent is a public repository accession, so behaviour is
   preserved while the internal tree is gone. The file is therefore *not*
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

# Every column is retained; `file path` is kept but reduced to two segments.
FORBIDDEN_COLUMNS: list[str] = []

# `file path` must stay a bare <accession>/<filename>, never an absolute path.
MAX_PATH_SEGMENTS = 2

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


def test_file_path_is_relative_and_shallow(local: pd.DataFrame) -> None:
    """`file path` must stay <accession>/<filename>, not an absolute path."""
    if "file path" not in local.columns:
        pytest.skip("no 'file path' column")
    fp = local["file path"].astype(str)
    too_deep = fp[fp.str.count("/") >= MAX_PATH_SEGMENTS]
    assert too_deep.empty, (
        f"{len(too_deep)} paths have more than {MAX_PATH_SEGMENTS} segments, "
        f"e.g. {too_deep.iloc[0]!r} -- the directory tree must not be committed"
    )
    rooted = fp[fp.str.contains(r"^([A-Za-z]:|/|\\\\)", regex=True)]
    assert rooted.empty, f"{len(rooted)} absolute paths, e.g. {rooted.iloc[0]!r}"


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

    # Column sets should now agree; only cell contents were sanitised.
    extra = sorted(set(other.columns) - set(local.columns))
    assert not extra, f"columns present upstream but not locally: {extra}"


# Table S1 in the manuscript declares the corpus. The search-data table is
# restricted to those accessions: before filtering it carried rows for 104
# projects, 12 of them considered during assembly but not part of the declared
# corpus, so anyone counting distinct projects in the released table got 104
# where the manuscript says 92.
TABLE_S1 = REPO / "assets" / "table_s1_accessions.txt"


@pytest.fixture(scope="module")
def declared() -> set[str]:
    assert TABLE_S1.is_file(), f"missing {TABLE_S1.relative_to(REPO)}"
    return {line.strip() for line in TABLE_S1.read_text().splitlines() if line.strip()}


def test_projects_match_the_declared_corpus(local: pd.DataFrame, declared: set[str]) -> None:
    present = set(local["project"].astype(str))
    extra = sorted(present - declared)
    assert not extra, (
        f"{len(extra)} project(s) are not in Table S1: {extra[:8]}. The released table "
        "must not imply a larger corpus than the manuscript declares."
    )


def test_every_declared_project_has_rows(local: pd.DataFrame, declared: set[str]) -> None:
    present = set(local["project"].astype(str))
    missing = sorted(declared - present)
    assert not missing, f"{len(missing)} Table S1 project(s) have no rows: {missing[:8]}"
