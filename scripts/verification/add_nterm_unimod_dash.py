r"""Insert ``-`` between leading ``[UNIMOD:<id>]`` tags and the first residue.

Rewrites ``sequence`` in parquet files so N-terminal forms like
``[UNIMOD:737]PEPTIDE`` become ``[UNIMOD:737]-PEPTIDE``. Sequences that already
have the dash are unchanged. Internal modifications (e.g. ``PEC[UNIMOD:4]TIDE``)
are not affected.

USAGE:
======
python scripts/verification/add_nterm_unimod_dash.py \\
    --input-dir <data-root>/lcfm/

python scripts/verification/add_nterm_unimod_dash.py \\
    -i <data-root>/lcfm/ -p my_project --dry-run
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import polars as pl
import typer

from scripts.verification.verify_calc_mz import (
    find_parquet_files_in_project,
    find_project_folders,
)

app = typer.Typer(
    help="Add '-' after leading [UNIMOD:*] tags before the first amino acid"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_NTERM_UNIMOD_THEN_AA = re.compile(r"^((?:\[UNIMOD:\d+\])+)([A-Z])")

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
    help="Limit to these project folder names (repeat); omit for all projects",
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    "-n",
    help="Log actions without writing files",
)
VERBOSE_OPTION = typer.Option(False, "--verbose", help="Debug logging")


@dataclass
class NtermDashStats:
    """Counters for add-nterm-unimod-dash run."""

    projects_selected: int = 0
    projects_missing_filter: int = 0
    files_processed: int = 0
    files_skipped_no_sequence: int = 0
    rows_sequence_changed: int = 0


def add_nterm_unimod_dash(seq: Optional[str]) -> Optional[str]:
    """If ``sequence`` starts with ``[UNIMOD:digits]`` and then a capital AA, insert ``-``."""
    if seq is None:
        return None
    if not isinstance(seq, str):
        seq = str(seq)
    return _NTERM_UNIMOD_THEN_AA.sub(r"\1-\2", seq, count=1)


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


def process_parquet_file(
    file_path: Path,
    dry_run: bool,
    stats: NtermDashStats,
) -> None:
    """Process one parquet file, updating ``sequence`` in-place."""
    df = pl.read_parquet(file_path)
    if "sequence" not in df.columns:
        logger.debug("Skipping %s: no sequence column", file_path)
        stats.files_skipped_no_sequence += 1
        return

    original_sequence = df["sequence"]
    new_sequence = original_sequence.map_elements(
        add_nterm_unimod_dash, return_dtype=pl.String
    )
    changed = original_sequence.is_not_null() & (
        ~original_sequence.eq_missing(new_sequence)
    )
    n_changed = int(changed.sum())

    stats.files_processed += 1
    stats.rows_sequence_changed += n_changed

    if n_changed == 0:
        logger.debug("No sequence changes: %s", file_path)
        return

    if dry_run:
        logger.info(
            "[DRY-RUN] Would update %s (%d rows)",
            file_path,
            n_changed,
        )
        return

    out = df.with_columns(new_sequence.alias("sequence"))
    _atomic_write_parquet(out, file_path)
    logger.info("Updated %s (%d rows)", file_path, n_changed)


def run_add_nterm_unimod_dash(
    input_dir: Path,
    projects_filter: Optional[List[str]],
    dry_run: bool,
) -> NtermDashStats:
    """Process all parquet files in the input directory, updating ``sequence`` in-place."""
    root = str(input_dir)
    all_projects = find_project_folders(root)
    stats = NtermDashStats()

    if projects_filter:
        want = set(projects_filter)
        projects = [p for p in all_projects if p in want]
        missing = want - set(all_projects)
        for p in sorted(missing):
            logger.warning("Requested project not found or no top-level parquet: %s", p)
            stats.projects_missing_filter += 1
    else:
        projects = all_projects

    stats.projects_selected = len(projects)

    if not projects:
        logger.info("No projects to process.")
        return stats

    logger.info(
        "Processing %d project(s): %s",
        len(projects),
        ", ".join(sorted(projects)),
    )

    for project in projects:
        project_dir = input_dir / project
        if not project_dir.is_dir():
            logger.warning("Project folder missing, skipping: %s", project_dir)
            continue

        for fp_str in find_parquet_files_in_project(root, project):
            process_parquet_file(Path(fp_str), dry_run=dry_run, stats=stats)

    logger.info(
        "Done: files_processed=%d skipped_no_sequence=%d rows_changed=%d",
        stats.files_processed,
        stats.files_skipped_no_sequence,
        stats.rows_sequence_changed,
    )
    return stats


@app.command()
def main(
    input_dir: Path = INPUT_DIR_OPTION,
    projects: List[str] = PROJECTS_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Rewrite sequences: leading [UNIMOD:*] + bare first AA -> insert '-'."""
    if not input_dir.is_dir():
        raise typer.BadParameter(f"Not a directory: {input_dir}")

    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    run_add_nterm_unimod_dash(
        input_dir=input_dir,
        projects_filter=projects if projects else None,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
