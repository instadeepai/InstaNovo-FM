r"""Apply implicit carbamidomethylation using verify_calc_mz CSV reports.

PURPOSE:
========
Reads the per-project CSV produced by verify_calc_mz.py and, for projects where
the as-is match rate is below 100% but carbamidomethylation brings plain-AA
peptides with bare cysteine to 100% match vs peptide_calc_mz, rewrites all
parquet files under that project: bare C -> C[UNIMOD:4].

Rows with zero or null precursor_charge are included in the rewrite (they are
skipped by verify_calc_mz scoring but should follow the same convention when the
project gate passes).

USAGE:
======
python scripts/verification/apply_carbamido_from_calc_mz_report.py \
    --input-dir <data-root>/lcfm/ \
    --verification-csv calc_mz_verification.csv

python scripts/verification/apply_carbamido_from_calc_mz_report.py \
    --input-dir <data-root>/lcfm/ \
    --verification-csv calc_mz_verification.csv \
    --dry-run
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

import polars as pl
import typer

from scripts.verification.verify_calc_mz import (
    carbamidomethylate_cysteines,
    find_parquet_files_in_project,
)

app = typer.Typer(help="Apply implicit carbamidomethylation from verify_calc_mz report")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INPUT_DIR_OPTION = typer.Option(
    ...,
    "--input-dir",
    "-i",
    help="Input directory containing parquet files organized by project subfolders",
)
VERIFICATION_CSV_OPTION = typer.Option(
    ...,
    "--verification-csv",
    "-v",
    help="CSV report from verify_calc_mz.py",
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    "-n",
    help="Log actions without modifying files",
)
VERBOSE_OPTION = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Enable verbose logging",
)


@dataclass
class ApplyCarbStats:
    """Counters for apply-carb run."""

    projects_selected: int = 0
    projects_missing_dir: int = 0
    files_processed: int = 0
    files_skipped_no_sequence: int = 0
    rows_sequence_changed: int = 0
    rows_low_or_null_charge_changed: int = 0


def load_verification_report(verification_csv: str | Path) -> pl.DataFrame:
    """Load verify_calc_mz CSV."""
    return pl.read_csv(verification_csv)


def select_projects_for_carb(report: pl.DataFrame) -> List[str]:
    """Project names that pass the implicit-CAM gate."""
    required = (
        "project",
        "calc_mz_match_rate_as_is_pct",
        "calc_mz_match_rate_after_carb_unmod_c_rows_pct",
        "calc_mz_matches_after_carb_unmod_c_rows",
    )
    missing = [c for c in required if c not in report.columns]
    if missing:
        raise ValueError(f"Verification CSV missing columns: {missing}")

    filtered = report.filter(
        (pl.col("calc_mz_match_rate_as_is_pct") < 100)
        & (pl.col("calc_mz_match_rate_after_carb_unmod_c_rows_pct") == 100)
        & (pl.col("calc_mz_matches_after_carb_unmod_c_rows") > 0)
    )
    return cast(List[str], filtered["project"].to_list())


def _carb_sequence(seq: Optional[str]) -> Optional[str]:
    if seq is None:
        return None
    return cast(str, carbamidomethylate_cysteines(seq))


def _atomic_write_parquet(df: pl.DataFrame, file_path: Path) -> None:
    temp_fd, temp_path_str = tempfile.mkstemp(suffix=".parquet", dir=file_path.parent)
    os.close(temp_fd)
    temp_path = Path(temp_path_str)
    try:
        df.write_parquet(temp_path)
        os.replace(temp_path, file_path)
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def _low_or_null_charge_mask(df: pl.DataFrame) -> Optional[pl.Series]:
    """Mask rows with null or non-positive precursor_charge, if column exists and is numeric."""
    if "precursor_charge" not in df.columns:
        return None
    charges = df["precursor_charge"]
    if charges.dtype not in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ):
        return None
    return charges.is_null() | (charges <= 0)


def process_parquet_file(
    file_path: Path,
    dry_run: bool,
    stats: ApplyCarbStats,
) -> None:
    """Carbamidomethylate sequences in one parquet file."""
    df = pl.read_parquet(file_path)
    if "sequence" not in df.columns:
        logger.debug("Skipping %s: no sequence column", file_path)
        stats.files_skipped_no_sequence += 1
        return

    original_sequence = df["sequence"]
    new_sequence = original_sequence.map_elements(
        _carb_sequence, return_dtype=pl.String
    )
    changed = original_sequence.is_not_null() & (
        ~original_sequence.eq_missing(new_sequence)
    )
    n_changed = int(changed.sum())

    low_mask = _low_or_null_charge_mask(df)
    n_low_charge_changed = 0
    if low_mask is not None and n_changed > 0:
        n_low_charge_changed = int((changed & low_mask).sum())

    stats.files_processed += 1
    stats.rows_sequence_changed += n_changed
    stats.rows_low_or_null_charge_changed += n_low_charge_changed

    if n_changed == 0:
        logger.debug("No sequence changes: %s", file_path)
        return

    if dry_run:
        logger.info(
            "[DRY-RUN] Would update %s (%d rows, %d with precursor_charge null or <= 0)",
            file_path,
            n_changed,
            n_low_charge_changed,
        )
        return

    out = df.with_columns(new_sequence.alias("sequence"))
    _atomic_write_parquet(out, file_path)
    logger.info(
        "Updated %s (%d rows, %d with precursor_charge null or <= 0)",
        file_path,
        n_changed,
        n_low_charge_changed,
    )


def run_apply_carbamidomethylation(
    input_dir: str | Path,
    verification_csv: str | Path,
    dry_run: bool = False,
) -> ApplyCarbStats:
    """Select projects from report and apply carbamidomethylation to their parquets."""
    input_path = Path(input_dir)
    report = load_verification_report(verification_csv)
    projects = select_projects_for_carb(report)
    stats = ApplyCarbStats()
    stats.projects_selected = len(projects)

    if not projects:
        logger.info("No projects pass the implicit carbamidomethylation gate.")
        return stats

    logger.info(
        "Applying carbamidomethylation to %d project(s): %s",
        len(projects),
        ", ".join(sorted(projects)),
    )

    for project in projects:
        project_dir = input_path / project
        if not project_dir.is_dir():
            logger.warning("Project folder missing, skipping: %s", project_dir)
            stats.projects_missing_dir += 1
            continue

        parquet_files = find_parquet_files_in_project(str(input_path), project)
        for fp_str in parquet_files:
            process_parquet_file(Path(fp_str), dry_run=dry_run, stats=stats)

    logger.info(
        "Done: files_processed=%d skipped_no_sequence=%d rows_changed=%d "
        "low_or_null_charge_changed=%d",
        stats.files_processed,
        stats.files_skipped_no_sequence,
        stats.rows_sequence_changed,
        stats.rows_low_or_null_charge_changed,
    )
    return stats


@app.command()
def main(
    input_dir: Path = INPUT_DIR_OPTION,
    verification_csv: Path = VERIFICATION_CSV_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Apply C[UNIMOD:4] to sequences for projects flagged by verify_calc_mz."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_apply_carbamidomethylation(
        input_dir=input_dir,
        verification_csv=verification_csv,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
