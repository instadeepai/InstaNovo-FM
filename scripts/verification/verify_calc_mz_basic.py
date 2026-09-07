r"""Minimal calc_mz check: sequence as-is vs peptide_calc_mz, per-project match rate.

Rows with null or non-positive ``precursor_charge`` are excluded (typical DIA;
same gate as ``verify_calc_mz``), so they do not count toward ``total_rows``.
Null sequence, charge, or ``peptide_calc_mz`` are excluded. Sequences containing
``IN:`` are excluded (same
as verify_calc_mz). Unknown tokenizer tokens are skipped for scoring (same
denominator convention as the full verifier: processed = total_rows − skipped).

Usage:
    uv run python scripts/verification/verify_calc_mz_basic.py \\
        --input-dir <data-root>/lcfm/ \\
        --output-csv calc_mz_basic.csv \\
        --tolerance 10
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import List

import polars as pl
import typer

from instanovo.utils.residues import ResidueSet
from scripts.verification.verify_calc_mz import (
    calculate_mz,
    calculate_ppm_error,
    create_residue_set,
    find_parquet_files_in_project,
    find_project_folders,
)

app = typer.Typer()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INPUT_DIR_OPTION = typer.Option(
    "<data-root>/lcfm/",
    "--input-dir",
    "-i",
    help="Directory with project subfolders containing parquet files",
)
RESIDUE_MASSES_FILE_OPTION = typer.Option(
    "mod_dicts/residue_masses.yaml",
    "--residue-masses-file",
    "-r",
    help="Residue masses YAML",
)
TOLERANCE_OPTION = typer.Option(
    10,
    "--tolerance",
    "-t",
    help="PPM tolerance for m/z matching",
)
OUTPUT_CSV_OPTION = typer.Option(
    ...,
    "--output-csv",
    "-o",
    help="Output CSV path",
)
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Debug logging")


def format_time(seconds: float) -> str:
    """Format a time in seconds as a string."""
    return str(timedelta(seconds=int(seconds)))


@dataclass
class ProjectStats:
    """Project statistics.

    Args:
        project: Project name.
        total_rows: Total rows.
        skipped_unknown_tokens: Skipped unknown tokens.
        calc_mz_matches_as_is: Calculated m/z matches as-is.
    """

    project: str
    total_rows: int = 0
    skipped_unknown_tokens: int = 0
    calc_mz_matches_as_is: int = 0


def _process_file(
    file_path: str, residue_set: ResidueSet, tolerance: float, stats: ProjectStats
) -> None:
    schema = pl.scan_parquet(file_path).collect_schema()
    required = ["sequence", "precursor_charge", "peptide_calc_mz"]
    if not all(c in schema for c in required):
        logger.debug("Skipping %s: missing columns", file_path)
        return

    df = (
        pl.scan_parquet(file_path)
        .select(required)
        .filter(
            pl.col("sequence").is_not_null()
            & pl.col("precursor_charge").is_not_null()
            & pl.col("peptide_calc_mz").is_not_null()
            & (pl.col("precursor_charge") > 0)
            & ~pl.col("sequence").str.contains("IN:", literal=True)
        )
        .collect()
    )
    stats.total_rows += len(df)

    for row in df.iter_rows(named=True):
        seq = row["sequence"]
        charge = row["precursor_charge"]
        ref_mz = row["peptide_calc_mz"]
        our = calculate_mz(seq, charge, residue_set)
        if our is None:
            stats.skipped_unknown_tokens += 1
            continue
        if calculate_ppm_error(our, ref_mz) <= tolerance:
            stats.calc_mz_matches_as_is += 1


def analyze_project(
    input_dir: str, project: str, residue_set: ResidueSet, tolerance: float
) -> ProjectStats:
    """Analyze a project."""
    stats = ProjectStats(project=project)
    for path in find_parquet_files_in_project(input_dir, project):
        try:
            _process_file(path, residue_set, tolerance, stats)
        except Exception as e:
            logger.warning("Error processing %s: %s", path, e)
    return stats


def _row(stats: ProjectStats) -> dict:
    """Create a row from project statistics."""
    processed = stats.total_rows - stats.skipped_unknown_tokens
    rate = (stats.calc_mz_matches_as_is / processed * 100) if processed > 0 else 0.0
    return {
        "project": stats.project,
        "total_rows": stats.total_rows,
        "skipped_unknown_tokens": stats.skipped_unknown_tokens,
        "processed_rows": processed,
        "calc_mz_matches_as_is": stats.calc_mz_matches_as_is,
        "calc_mz_match_rate_as_is_pct": round(rate, 2),
    }


@app.command()
def main(
    input_dir: str = INPUT_DIR_OPTION,
    residue_masses_file: str = RESIDUE_MASSES_FILE_OPTION,
    tolerance: float = TOLERANCE_OPTION,
    output_csv: str = OUTPUT_CSV_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Main function."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.isdir(input_dir):
        raise typer.BadParameter(f"Not a directory: {input_dir}")

    logger.info("Loading residue set from %s", residue_masses_file)
    residue_set = create_residue_set(residue_masses_file)

    projects = find_project_folders(input_dir)
    logger.info("Found %d projects under %s", len(projects), input_dir)
    if not projects:
        return

    t0 = time.time()
    results: List[ProjectStats] = []
    for i, project in enumerate(projects):
        if (i + 1) % 10 == 0:
            logger.info(
                "Processed %d/%d projects (%s)",
                i + 1,
                len(projects),
                format_time(time.time() - t0),
            )
        results.append(analyze_project(input_dir, project, residue_set, tolerance))

    out = [_row(s) for s in results]
    pl.DataFrame(out).sort("project").write_csv(output_csv)
    logger.info("Wrote %s", output_csv)

    proc_total = sum(r.total_rows - r.skipped_unknown_tokens for r in results)
    match_total = sum(r.calc_mz_matches_as_is for r in results)
    if proc_total > 0:
        logger.info(
            "Overall as-is match rate: %.2f%% (%d / %d)",
            match_total / proc_total * 100,
            match_total,
            proc_total,
        )


if __name__ == "__main__":
    app()
