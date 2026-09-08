r"""Manually carbamidomethylate bare cysteines in parquet sequences for named projects.

Use this when you already know which project folders need C[UNIMOD:4] and do
not want to wait on a verify_calc_mz CSV.

Rewrites the ``sequence`` column the same way as
``apply_carbamido_from_calc_mz_report.py`` (bare ``C`` -> ``C[UNIMOD:4]``), but
you choose which project subfolders under ``--input-dir`` to process — no
verify_calc_mz CSV. All parquet files under each project are processed, including
DIA parquets where ``precursor_charge`` is zero or null (every row is rewritten
when the sequence contains bare cysteines).

CLI::

    python scripts/verification/apply_carbamido_manual_projects.py --help
    python scripts/verification/apply_carbamido_manual_projects.py \
        --input-dir <data-root>/lcfm/ \
        --project my_dataset_a --project my_dataset_b
    python scripts/verification/apply_carbamido_manual_projects.py \
        -i <data-root>/lcfm/ -p proj1 -p proj2 --dry-run
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from scripts.verification.apply_carbamido_from_calc_mz_report import (
    ApplyCarbStats,
    process_parquet_file,
)
from scripts.verification.verify_calc_mz import find_parquet_files_in_project

app = typer.Typer(
    help="Carbamidomethylate sequences for explicitly listed project folders",
    no_args_is_help=True,
    add_completion=False,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INPUT_DIR_OPTION = typer.Option(
    ...,
    "--input-dir",
    "-i",
    help="Root directory containing parquet files under project subfolders",
)
PROJECTS_OPTION = typer.Option(
    [],
    "--project",
    "-p",
    help="Project folder name under input-dir (repeat for multiple)",
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    "-n",
    help="Log actions without writing files",
)
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Debug logging")


def run_manual(
    input_dir: Path,
    projects: list[str],
    dry_run: bool,
) -> ApplyCarbStats:
    """Carbamidomethylate unmodified cysteines in caller-chosen project folders, including DIA parquets.

    Args:
        input_dir: Root with per-project parquet subfolders.
        projects: Project folder names to rewrite (no CSV gate).
        dry_run: Preview writes without modifying files.

    Returns:
        Counters shared with the report-driven CAM script.
    """
    stats = ApplyCarbStats()
    stats.projects_selected = len(projects)

    logger.info(
        "Applying carbamidomethylation to %d project(s): %s",
        len(projects),
        ", ".join(sorted(projects)),
    )

    for project in projects:
        project_dir = input_dir / project
        if not project_dir.is_dir():
            logger.warning("Project folder missing, skipping: %s", project_dir)
            stats.projects_missing_dir += 1
            continue

        parquet_files = find_parquet_files_in_project(str(input_dir), project)
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
    projects: list[str] = PROJECTS_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Apply C[UNIMOD:4] to sequences for listed projects without requiring a calc-mz report.

    Args:
        input_dir: Root directory containing parquet files under project subfolders.
        projects: Project folder names under input-dir (repeat for multiple).
        dry_run: Log actions without writing files.
        verbose: Enable debug logging.
    """
    if not projects:
        raise typer.BadParameter("Pass at least one --project / -p.")

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_manual(input_dir=input_dir, projects=projects, dry_run=dry_run)


if __name__ == "__main__":
    app()
