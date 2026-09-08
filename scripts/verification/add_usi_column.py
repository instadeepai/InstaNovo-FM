r"""Add a Universal Spectrum Identifier (USI) column to labelled parquet files.

Each row gets a string compatible with the PSI USI specification
(https://www.psidev.info/usi), built with :class:`pyteomics.usi.USI`.
Run after labelled parquets exist so downstream tools can cite spectra by
dataset, file, and scan.

Sequences that contain internal tokens such as ``[IN:…]`` are not strict ProForma;
they are copied into the interpretation segment as-is and may not resolve in
public PROXI services until converted to UNIMOD-style ProForma.

CLI::

    uv run python -m scripts.verification.add_usi_column --help
    uv run python -m scripts.verification.add_usi_column \
        --input-dir <data-root>/lcfm/
    uv run python -m scripts.verification.add_usi_column \
        -i <data-root>/lcfm/ --project PXD009449 --dry-run
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

import polars as pl
import typer
from pyteomics.usi import USI

from scripts.verification.verify_calc_mz import (
    find_parquet_files_in_project,
    find_project_folders,
)
from scripts.preprocessing.parquet_io import search_data_lookup_key

from scripts.logging_setup import configure_script_logging

app = typer.Typer(
    help="Add USI column to labelled parquet datasets",
    no_args_is_help=True,
    add_completion=False,
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
    help="Only process these project folder names (repeat for multiple)",
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    "-n",
    help="Log actions without modifying files",
)
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Debug logging")
OVERWRITE_OPTION = typer.Option(
    True,
    "--overwrite/--no-overwrite",
    help="Replace existing usi column (default: overwrite)",
)
SCAN_TYPE_OPTION = typer.Option(
    "scan",
    "--scan-type",
    help="USI scan identifier type: scan, index, nativeId, or trace",
)

_PXD_RE = re.compile(r"(PXD\d{6,})")
_MSV_RE = re.compile(r"(MSV\d{6,})")
_SCAN_NUM_RE = re.compile(r"scan=(\d+)", re.IGNORECASE)


def extract_pxd_or_msv_accession(filepath: str | None) -> Optional[str]:
    """Fill the USI collection field from a ProteomeXchange or MassIVE accession in the path.

    Args:
        filepath: Path that should contain a ``PXD######`` or ``MSV######`` token.

    Returns:
        The first matching accession, or None when the locator cannot be formed.
    """
    if filepath is None:
        return None
    m = _PXD_RE.search(str(filepath))
    if m:
        return m.group(1)
    m = _MSV_RE.search(str(filepath))
    if m:
        return m.group(1)
    return None


def normalize_scan_identifier(scan: Any) -> Optional[str]:
    """Turn vendor scan text into the numeric scan id required by a USI.

    Accepts plain integers/strings of digits, or vendor text such as
    ``controllerType=0 controllerNumber=1 scan=1321`` (Thermo-style).

    Args:
        scan: Value from the parquet ``scan`` column.

    Returns:
        A numeric scan string, or None when the row cannot be located.
    """
    if scan is None:
        return None
    s = str(scan).strip()
    if not s:
        return None
    if s.isdigit():
        return str(int(s))
    m = _SCAN_NUM_RE.search(s)
    if m:
        return m.group(1)
    return None


def _interpretation_from_sequence_and_charge(
    sequence: str | None, precursor_charge: Any
) -> Optional[str]:
    """Omit charge on DIA (null/0) so the USI interpretation is not ``/0``."""
    if sequence is None:
        return None
    if precursor_charge is None or precursor_charge == 0:
        return sequence
    try:
        z = int(precursor_charge)
    except (TypeError, ValueError):
        return f"{sequence}/{precursor_charge}"
    return f"{sequence}/{z}"


def build_usi_string(
    filepath: str | None,
    scan: Any,
    sequence: str | None,
    precursor_charge: Any,
    scan_identifier_type: str = "scan",
) -> Optional[str]:
    """Assemble one PSI USI, or skip the row when locator fields are incomplete.

    The USI ``datafile`` field uses the same canonical experiment basename as the
    parquet ``experiment_name`` column (shard and ``.mzml`` suffixes stripped).

    Args:
        filepath: Path used for collection accession and experiment stem.
        scan: Scan locator from the parquet row.
        sequence: Peptide string for the interpretation segment, if labelled.
        precursor_charge: Charge for DDA interpretation; omitted when null or 0 (DIA).
        scan_identifier_type: USI scan type (``scan``, ``index``, ``nativeId``, or ``trace``).

    Returns:
        A USI string, or None so the parquet can still be written with a null ``usi``.
    """
    if filepath is None:
        return None
    filepath = str(filepath)
    pxd_or_msv = extract_pxd_or_msv_accession(filepath)
    if pxd_or_msv is None:
        return None

    try:
        datafile = search_data_lookup_key(filepath)
    except Exception:
        return None
    if not datafile:
        return None
    if ":" in datafile:
        logger.warning(
            "Skipping USI: datafile basename contains ':' (needs PSI encoding): %r",
            datafile,
        )
        return None

    scan_id = normalize_scan_identifier(scan)
    if scan_id is None:
        return None

    seq_str: str | None = str(sequence) if sequence is not None else None
    interpretation = _interpretation_from_sequence_and_charge(seq_str, precursor_charge)
    usi = USI(
        protocol="mzspec",
        dataset=pxd_or_msv,
        datafile=datafile,
        scan_identifier_type=scan_identifier_type,
        scan_identifier=scan_id,
        interpretation=interpretation,
    )
    return str(usi)


@dataclass
class AddUsiStats:
    """Summarise how many files and rows received a USI so a run can be audited."""

    files_processed: int = 0
    files_skipped_missing_cols: int = 0
    files_skipped_existing_usi: int = 0
    rows_written: int = 0
    projects_missing_dir: int = 0


def _atomic_write_parquet(df: pl.DataFrame, file_path: Path) -> None:
    """Replace the parquet only after a full write so a crash cannot leave a truncated file."""
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


def _add_usi_series(
    df: pl.DataFrame, default_filepath: str, scan_identifier_type: str
) -> pl.Series:
    """Prefer per-row ``filepath`` when present so USIs follow the original mzML, not the parquet path."""
    if "scan" not in df.columns:
        raise ValueError("Cannot add USI column: missing scan column")

    use_row_filepath = "filepath" in df.columns
    values: list[Optional[str]] = []
    for row in df.iter_rows(named=True):
        filepath = row.get("filepath") if use_row_filepath else default_filepath
        if filepath is None:
            filepath = default_filepath
        values.append(
            build_usi_string(
                str(filepath),
                row.get("scan"),
                row.get("sequence"),
                row.get("precursor_charge"),
                scan_identifier_type=scan_identifier_type,
            )
        )
    return pl.Series("usi", values, dtype=pl.String)


def process_parquet_file(
    file_path: Path,
    scan_identifier_type: str,
    overwrite: bool,
    dry_run: bool,
    stats: AddUsiStats,
) -> None:
    """Write a ``usi`` column into one parquet, skipping files that cannot support it.

    Args:
        file_path: Parquet to update.
        scan_identifier_type: USI scan identifier type passed through to each row.
        overwrite: Replace an existing ``usi`` column when True.
        dry_run: Log the intended write without modifying the file.
        stats: Run counters updated in place for the final summary.
    """
    schema = pl.scan_parquet(str(file_path)).collect_schema()
    if "scan" not in schema:
        logger.info("Skipping %s: missing scan", file_path)
        stats.files_skipped_missing_cols += 1
        return

    if "usi" in schema and not overwrite:
        logger.info("Skipping %s: usi exists and --no-overwrite", file_path)
        stats.files_skipped_existing_usi += 1
        return

    df = pl.read_parquet(file_path)
    usi_col = _add_usi_series(df, str(file_path), scan_identifier_type)
    out = df.with_columns(usi_col.alias("usi"))
    stats.files_processed += 1
    stats.rows_written += len(out)

    if dry_run:
        logger.info("[DRY-RUN] Would write usi for %s (%d rows)", file_path, len(out))
        return

    _atomic_write_parquet(out, file_path)
    logger.info("Wrote usi column: %s (%d rows)", file_path, len(out))


def run_add_usi(
    input_dir: Path,
    projects: Optional[List[str]],
    scan_identifier_type: str,
    overwrite: bool,
    dry_run: bool,
) -> AddUsiStats:
    """Walk project folders so every labelled parquet can receive a USI.

    Args:
        input_dir: Root whose project subfolders contain parquet files.
        projects: Restrict processing to these folder names; all projects when None.
        scan_identifier_type: USI scan identifier type for every row.
        overwrite: Replace existing ``usi`` columns when True.
        dry_run: Preview writes without changing files.

    Returns:
        Counters for processed files, skips, and rows written.
    """
    stats = AddUsiStats()
    input_str = str(input_dir)

    if projects:
        to_process = list(projects)
    else:
        to_process = find_project_folders(input_str)

    for project in to_process:
        project_dir = input_dir / project
        if not project_dir.is_dir():
            logger.warning("Project folder missing, skipping: %s", project_dir)
            stats.projects_missing_dir += 1
            continue

        for fp_str in find_parquet_files_in_project(input_str, project):
            process_parquet_file(
                Path(fp_str),
                scan_identifier_type=scan_identifier_type,
                overwrite=overwrite,
                dry_run=dry_run,
                stats=stats,
            )

    logger.info(
        "Done: processed=%d skipped_missing_cols=%d skipped_existing_usi=%d "
        "rows=%d missing_projects=%d",
        stats.files_processed,
        stats.files_skipped_missing_cols,
        stats.files_skipped_existing_usi,
        stats.rows_written,
        stats.projects_missing_dir,
    )
    return stats


@app.command()
def main(
    input_dir: Path = INPUT_DIR_OPTION,
    projects: list[str] = PROJECTS_OPTION,
    scan_type: str = SCAN_TYPE_OPTION,
    overwrite: bool = OVERWRITE_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Add a PSI USI column to labelled parquets so spectra can be cited by dataset, file, and scan.

    Args:
        input_dir: Root directory containing parquet files under project subfolders.
        projects: Optional project folder names to restrict the run.
        scan_type: USI scan identifier type: scan, index, nativeId, or trace.
        overwrite: Replace an existing ``usi`` column (default: overwrite).
        dry_run: Log actions without modifying files.
        verbose: Enable debug logging.
    """
    if scan_type not in ("scan", "index", "nativeId", "trace"):
        raise typer.BadParameter(
            "scan_type must be one of: scan, index, nativeId, trace"
        )

    configure_script_logging(verbose=verbose)

    run_add_usi(
        input_dir=input_dir,
        projects=projects if projects else None,
        scan_identifier_type=scan_type,
        overwrite=overwrite,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
