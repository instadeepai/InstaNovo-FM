r"""Verify per-spectrum max intensity normalisation in parquet/IPC files.

Run after conversion/preprocessing so training spectra have max intensity 1.0
and the original maximum is retained in ``scale_factor``.

Each non-empty ``intensity_array`` must have maximum ``1.0``.
Empty peak lists and all-zero spectra are not allowed.

If ``scale_factor`` is present, each row must carry a numeric value (finite float).
The script does not validate the scale beyond that. If ``scale_factor`` is absent
and ``max(intensity_array) == 1.0``, those rows are counted separately as missing
scale metadata (not a verification failure).

With ``--fix``, non-conforming rows are max-normalised. ``scale_factor`` is set to
``max(intensity_array) * scale_factor`` when the column exists (null/NaN treated
as ``1.0``), or **created** as ``Float32`` when the column was missing.

CLI::

    python scripts/verification/verify_intensity_max_normalisation.py --help
    python scripts/verification/verify_intensity_max_normalisation.py \
        --input path/to/file.parquet
    python scripts/verification/verify_intensity_max_normalisation.py \
        --input-dir <data-root>/lcfm/ --project PXD009449
    python scripts/verification/verify_intensity_max_normalisation.py \
        --input path/to/file.parquet --fix --dry-run
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, cast

import numpy as np
import polars as pl
import typer

from scripts.verification.verify_calc_mz import (
    find_parquet_files_in_project,
    find_project_folders,
)

app = typer.Typer(
    help="Verify intensity max-normalisation and optional fix-in-place",
    no_args_is_help=True,
    add_completion=False,
)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INPUT_OPTION = typer.Option(
    None,
    "--input",
    help="Single .parquet or .ipc file",
)
INPUT_DIR_OPTION = typer.Option(
    None,
    "--input-dir",
    "-i",
    help="Root directory with project subfolders containing parquet files",
)
PROJECTS_OPTION = typer.Option(
    [],
    "--project",
    "-p",
    help="Only these project folder names (repeat); used with --input-dir",
)
FIX_OPTION = typer.Option(
    False,
    "--fix",
    help="Re-normalise rows in place; add scale_factor column if missing",
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    "-n",
    help="With --fix, log changes only; do not write files",
)
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Debug logging")


@dataclass
class FileVerifyStats:
    """Per-file intensity-check counts so a tree can be summarised without rewriting."""

    path: str
    rows: int = 0
    ok: int = 0
    bad_max: int = 0
    bad_scale_not_numeric: int = 0
    rows_max_ok_missing_scale_column: int = 0
    sample_bad_row_indices: list[int] = field(default_factory=list)


def _safe_scale_factor(v: object) -> float:
    """Treat missing/NaN scale as 1.0 when reconstructing max * scale during a fix."""
    if v is None:
        return 1.0
    try:
        x = float(cast(Any, v))
    except (TypeError, ValueError):
        return 1.0
    if math.isnan(x):
        return 1.0
    return x


def _is_numeric_scale_factor(sf: Any) -> bool:
    """Require a finite number when the scale_factor column exists."""
    if sf is None:
        return False
    try:
        x = float(cast(Any, sf))
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def normalise_intensity_row(
    intensity: object,
    scale_factor: object,
) -> tuple[list[float], float]:
    """Max-normalise one spectrum while keeping raw intensity recoverable as i * scale.

    Args:
        intensity: Peak intensities for the row.
        scale_factor: Existing scale, or missing/NaN treated as 1.0.

    Returns:
        New intensity list (max 1.0 when peaks are positive) and updated scale_factor.
    """
    if intensity is None:
        return [], _safe_scale_factor(scale_factor)

    arr = np.asarray(intensity, dtype=np.float64)
    if arr.size == 0:
        return [], float(_safe_scale_factor(scale_factor))

    sf = _safe_scale_factor(scale_factor)
    mx = float(arr.max())
    if mx <= 0.0:
        return arr.astype(np.float32).tolist(), 0.0

    new_int = (arr / mx).astype(np.float32)
    return new_int.tolist(), mx * sf


def _read_df(path: Path) -> pl.DataFrame:
    """Load parquet or IPC so both labelled formats can be checked."""
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)
    return pl.read_ipc(path)


def _atomic_write(df: pl.DataFrame, file_path: Path) -> None:
    """Replace parquet/IPC only after a full write so a crash cannot leave a truncated file."""
    temp_fd, temp_path_str = tempfile.mkstemp(
        suffix=file_path.suffix, dir=file_path.parent
    )
    os.close(temp_fd)
    temp_path = Path(temp_path_str)
    try:
        if file_path.suffix.lower() == ".parquet":
            df.write_parquet(temp_path)
        else:
            df.write_ipc(temp_path)
        os.replace(temp_path, file_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _record_bad_sample(stats: FileVerifyStats, idx: int, max_sample: int) -> None:
    """Keep a few failing row indices for verbose logs without dumping the whole file."""
    if len(stats.sample_bad_row_indices) < max_sample:
        stats.sample_bad_row_indices.append(idx)


def _classify_intensity_row(
    peak_max: object,
    sf: object,
    has_sf_col: bool,
) -> Literal["bad_max", "bad_scale", "missing_sf_column", "ok"]:
    """Separate true max-norm failures from missing scale metadata (not a verification failure)."""
    if peak_max is None:
        return "bad_max"
    m = float(cast(Any, peak_max))
    if math.isnan(m) or not math.isfinite(m) or m <= 0.0:
        return "bad_max"
    if m != 1.0:
        return "bad_max"
    if not has_sf_col:
        return "missing_sf_column"
    if not _is_numeric_scale_factor(sf):
        return "bad_scale"
    return "ok"


def verify_file(path: Path, max_sample: int = 5) -> FileVerifyStats:
    """Scan a file for max-norm and scale_factor issues without modifying it.

    Args:
        path: Parquet or IPC path.
        max_sample: Cap on failing row indices kept for verbose logging.

    Returns:
        Per-file counts, including missing-scale-column rows that are not failures.
    """
    df = _read_df(path)
    if "intensity_array" not in df.columns:
        logger.warning("Skipping %s: missing intensity_array", path)
        return FileVerifyStats(path=str(path))

    stats = FileVerifyStats(path=str(path))
    stats.rows = len(df)
    has_sf_col = "scale_factor" in df.columns

    cols: list[pl.Expr] = [
        pl.int_range(0, pl.len()).alias("_row_idx"),
        pl.col("intensity_array").list.max().alias("_imax"),
    ]
    if has_sf_col:
        cols.append(pl.col("scale_factor"))
    else:
        cols.append(pl.lit(None).alias("scale_factor"))

    work = df.select(cols)

    for row in work.iter_rows(named=True):
        idx = row["_row_idx"]
        outcome = _classify_intensity_row(row["_imax"], row["scale_factor"], has_sf_col)
        if outcome == "bad_max":
            stats.bad_max += 1
            _record_bad_sample(stats, idx, max_sample)
        elif outcome == "bad_scale":
            stats.bad_scale_not_numeric += 1
            _record_bad_sample(stats, idx, max_sample)
        elif outcome == "missing_sf_column":
            stats.rows_max_ok_missing_scale_column += 1

    stats.ok = stats.rows - stats.bad_max - stats.bad_scale_not_numeric
    return stats


def apply_fix_to_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    """Max-normalise intensities and write/create scale_factor so raw peaks stay recoverable.

    Args:
        df: Table that must contain ``intensity_array``.

    Returns:
        A copy with updated intensities and scale_factor.

    Raises:
        ValueError: When ``intensity_array`` is missing.
    """
    if "intensity_array" not in df.columns:
        raise ValueError("DataFrame must contain intensity_array")

    has_sf = "scale_factor" in df.columns
    intensities = df["intensity_array"].to_list()
    scales: list[Any] = (
        df["scale_factor"].to_list() if has_sf else [None] * len(intensities)
    )

    new_i: list[list[float]] = []
    new_s: list[float] = []
    for it, sc in zip(intensities, scales):
        ni, ns = normalise_intensity_row(it, sc)
        new_i.append(ni)
        new_s.append(ns)

    int_dtype = df.schema["intensity_array"]
    out = df.with_columns(
        pl.Series("intensity_array", new_i).cast(int_dtype),
    )
    sf_series = pl.Series("scale_factor", new_s, dtype=pl.Float32)
    if has_sf:
        sf_dtype = df.schema["scale_factor"]
        out = out.with_columns(sf_series.cast(sf_dtype))
    else:
        out = out.with_columns(sf_series)
    return out


def collect_parquet_paths(input_dir: str, projects: list[str]) -> list[Path]:
    """Enumerate project parquets for a directory-wide check or fix.

    Args:
        input_dir: Root with project subfolders.
        projects: Restrict to these folder names; all projects when empty.

    Returns:
        Parquet paths to verify.
    """
    project_names = find_project_folders(input_dir) if not projects else projects
    out: list[Path] = []
    for project in project_names:
        proj_path = os.path.join(input_dir, project)
        if not os.path.isdir(proj_path):
            logger.warning("Not a directory: %s", proj_path)
            continue
        for rel in find_parquet_files_in_project(input_dir, project):
            out.append(Path(rel))
    return out


def _print_stats(stats: FileVerifyStats, verbose: bool) -> None:
    """Log per-file counts so operators can see bad_max vs missing scale metadata."""
    logger.info(
        "%s: rows=%d ok=%d bad_max=%d bad_scale_not_numeric=%d "
        "max_ok_missing_scale_column=%d",
        stats.path,
        stats.rows,
        stats.ok,
        stats.bad_max,
        stats.bad_scale_not_numeric,
        stats.rows_max_ok_missing_scale_column,
    )
    if verbose and stats.sample_bad_row_indices:
        logger.info(
            "  sample row indices with issues: %s", stats.sample_bad_row_indices
        )


def _resolve_paths(
    input_path: Optional[str], input_dir: Optional[str], project: list[str]
) -> list[Path]:
    """Require exactly one of --input or --input-dir so a single file and a tree are not mixed."""
    if bool(input_path) == bool(input_dir):
        raise typer.BadParameter("Provide exactly one of --input or --input-dir")
    if input_path:
        return [Path(input_path)]
    assert input_dir is not None
    return collect_parquet_paths(input_dir, list(project))


def _maybe_fix_file(path: Path, dry_run: bool, verbose: bool) -> bool:
    """Rewrite a failing file, then re-verify so silent incomplete fixes are caught."""
    df = _read_df(path)
    if "intensity_array" not in df.columns:
        return False
    new_df = apply_fix_to_dataframe(df)
    if dry_run:
        logger.info("Dry-run: would rewrite %s (%d rows)", path, len(new_df))
        return False
    _atomic_write(new_df, path)
    logger.info("Wrote %s", path)
    stats2 = verify_file(path)
    if stats2.bad_max or stats2.bad_scale_not_numeric:
        logger.warning("Post-fix verification still reports issues for %s", path)
        _print_stats(stats2, verbose)
        return True
    return False


@app.command()
def main(
    input_path: Optional[str] = INPUT_OPTION,
    input_dir: Optional[str] = INPUT_DIR_OPTION,
    project: list[str] = PROJECTS_OPTION,
    fix: bool = FIX_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Check that spectra are max-normalised (and optionally rewrite intensities plus scale_factor).

    Args:
        input_path: Single .parquet or .ipc file.
        input_dir: Root directory with project subfolders containing parquet files.
        project: Restrict to these project folder names (with --input-dir).
        fix: Re-normalise rows in place; add scale_factor if missing.
        dry_run: With --fix, log changes only; do not write files.
        verbose: Enable debug logging and sample failing row indices.
    """
    paths = _resolve_paths(input_path, input_dir, project)

    if not paths:
        logger.warning("No files to process")
        raise typer.Exit(code=1)

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    total_bad = 0
    post_fix_issues = 0
    for p in paths:
        if not p.is_file():
            logger.warning("Not a file: %s", p)
            continue

        stats = verify_file(p)
        _print_stats(stats, verbose)

        bad = stats.bad_max + stats.bad_scale_not_numeric
        total_bad += bad

        if fix and bad > 0 and _maybe_fix_file(p, dry_run, verbose):
            post_fix_issues += 1

    if post_fix_issues:
        raise typer.Exit(code=3)
    if total_bad > 0 and not (fix and not dry_run):
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()
