r"""Apply implicit carbamidomethylation using verify_calc_mz CSV reports.

Run after ``verify_calc_mz.py`` when the report shows that unmodified cysteines
match ``peptide_calc_mz`` only after adding carbamidomethylation (``C[UNIMOD:4]``).

Reads the per-project CSV produced by verify_calc_mz.py and, for projects where
the as-is match rate is below 100% but carbamidomethylation brings plain-amino-acid
peptides with unmodified cysteine to 100% match vs peptide_calc_mz, rewrites all
parquet files under that project so unmodified cysteine becomes ``C[UNIMOD:4]``.

Rows with zero or null precursor_charge are included in the rewrite (they are
skipped by verify_calc_mz scoring but should follow the same convention when the
project gate passes).

CLI::

    uv run python -m scripts.verification.apply_carbamido_from_calc_mz_report --help
    uv run python -m scripts.verification.apply_carbamido_from_calc_mz_report \
        --input-dir <data-root>/lcfm/ \
        --verification-csv calc_mz_verification.csv
    uv run python -m scripts.verification.apply_carbamido_from_calc_mz_report \
        --input-dir <data-root>/lcfm/ \
        --verification-csv calc_mz_verification.csv \
        --dry-run
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, cast

import polars as pl
import typer

from scripts.verification.verify_calc_mz import (
    carbamidomethylate_cysteines,
    find_parquet_files_in_project,
)

from scripts.logging_setup import configure_script_logging
from scripts.preprocessing.parquet_io import atomic_write_parquet

app = typer.Typer(
    help="Apply implicit carbamidomethylation from verify_calc_mz report",
    no_args_is_help=True,
    add_completion=False,
)

logger = logging.getLogger(__name__)

INPUT_DIR_OPTION = typer.Option(
    ...,
    "--input-dir",
    "-i",
    help="Input directory containing parquet files organised by project subfolders",
)
VERIFICATION_CSV_OPTION = typer.Option(
    ...,
    "--verification-csv",
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
    """Track how many projects and rows received carbamidomethylation so a rewrite can be audited."""

    projects_selected: int = 0
    projects_missing_dir: int = 0
    files_processed: int = 0
    files_skipped_no_sequence: int = 0
    rows_sequence_changed: int = 0
    rows_low_or_null_charge_changed: int = 0


def load_verification_report(verification_csv: str | Path) -> pl.DataFrame:
    """Load the verify_calc_mz CSV that decides which projects get implicit carbamidomethylation.

    Args:
        verification_csv: Path to the per-project calc-mz report.

    Returns:
        The report needed by the implicit-carbamidomethylation project gate.
    """
    return pl.read_csv(verification_csv)


def select_projects_for_carb(report: pl.DataFrame) -> List[str]:
    """Choose projects where as-is calc-mz fails but unmodified-cysteine rows match after carbamidomethylation.

    The gate is as-is rate below 100%, after-carbamidomethylation rate on
    unmodified-cysteine rows equal to 100%, and at least one such match.

    Args:
        report: verify_calc_mz CSV with the match-rate columns this gate requires.

    Returns:
        Project folder names that should receive carbamidomethylation rewrites.

    Raises:
        ValueError: When the CSV is missing columns required to apply the gate.
    """
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
    """Leave null sequences untouched while carbamidomethylating unmodified cysteines on real peptides."""
    if seq is None:
        return None
    return cast(str, carbamidomethylate_cysteines(seq))


def _low_or_null_charge_mask(df: pl.DataFrame) -> Optional[pl.Series]:
    """Count carbamidomethylation rewrites on DIA-like (null/<=0 charge) rows that verify_calc_mz does not score."""
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
    """Rewrite unmodified cysteines to ``C[UNIMOD:4]`` in one parquet when the project gate passed.

    Args:
        file_path: Parquet whose ``sequence`` column should be carbamidomethylated.
        dry_run: Log intended changes without writing.
        stats: Run counters updated in place, including low/null-charge rewrites.
    """
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
    atomic_write_parquet(out, file_path)
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
    """Apply carbamidomethylation only to projects that pass the verify_calc_mz implicit-carbamidomethylation gate.

    Args:
        input_dir: Root with per-project parquet subfolders.
        verification_csv: Report from ``verify_calc_mz.py``.
        dry_run: Preview rewrites without modifying files.

    Returns:
        Counters for selected projects, files processed, and sequences changed.
    """
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
    """Carbamidomethylate sequences in projects the calc-mz report flags as missing explicit cysteine modification.

    Args:
        input_dir: Root directory containing parquet files organised by project subfolders.
        verification_csv: CSV report from verify_calc_mz.py.
        dry_run: Log actions without modifying files.
        verbose: Enable verbose logging.
    """
    configure_script_logging(verbose=verbose)

    run_apply_carbamidomethylation(
        input_dir=input_dir,
        verification_csv=verification_csv,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
