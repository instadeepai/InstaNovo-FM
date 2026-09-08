"""Replace placeholder strings such as ``Unknown`` with null in metadata columns.

Some source files store missing ``collision_energy`` or ``frag_type`` as the
text ``Unknown``. Downstream schemas treat that as a real value, so this script
rewrites it to null before alignment or training.

CLI::

    python scripts/preprocessing/enforce_nulls.py --help
    python scripts/preprocessing/enforce_nulls.py --input-dir <data-root>/lcfm --output-file enforced_nulls.csv
    python scripts/preprocessing/enforce_nulls.py --input-dir <data-root>/lcfm --input-dir <data-root>/hcfm --output-file enforced_nulls.csv

Use ``python script.py --help`` for flags.
"""

from __future__ import annotations

import glob
import logging
from pathlib import Path
from typing import Annotated, List, Optional

import polars as pl
import typer
from tqdm import tqdm

from scripts.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Enforce null values in parquet files",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_COLUMNS = ["collision_energy", "frag_type"]


def find_files(input_dir: str, file_pattern: str) -> list[str]:
    """List parquet files so each dataset can be scanned for placeholder metadata.

    Args:
        input_dir: Root directory for the search.
        file_pattern: Recursive glob selecting candidate files.

    Returns:
        Paths matching the requested pattern.
    """
    search_pattern = Path(input_dir) / file_pattern
    matched_files = glob.glob(str(search_pattern), recursive=True)
    return matched_files


def _columns_with_old_value(
    ldf: pl.LazyFrame,
    column_names: List[str],
    old_value: str,
    schema_column_names: set[str],
) -> List[str]:
    """Skip files that do not contain the placeholder string, so unchanged parquet is not rewritten."""
    columns_to_update: List[str] = []
    for col in column_names:
        if col not in schema_column_names:
            continue
        uniques = pl.Series(ldf.select(col).unique().collect()).to_list()
        if old_value in uniques:
            columns_to_update.append(col)
    return columns_to_update


def _process_parquet_file(
    file: str,
    column_names: List[str],
    old_value: str,
    new_value: Optional[str],
    verbose: bool,
) -> bool:
    """Isolate per-file failures so one corrupt Parquet does not stop the batch."""
    if verbose:
        logger.debug(f"Processing file: {file}")
    try:
        ldf = pl.scan_parquet(file)
        schema = ldf.collect_schema()
        schema_names = set(schema.names())
        columns_to_update = _columns_with_old_value(
            ldf, column_names, old_value, schema_names
        )
        if not columns_to_update:
            return False
        df = ldf.with_columns(
            [pl.col(col).replace(old_value, new_value) for col in columns_to_update]
        ).collect()
        df.write_parquet(file)
        if verbose:
            logger.debug(
                f"Updated file: {file} (columns: {', '.join(columns_to_update)})"
            )
        return True
    except Exception as e:
        logger.error(f"Error processing {file}: {e}")
        return False


def _write_affected_files_list(output_path: str, files_with_unknown: List[str]) -> None:
    """Record changed files so normalisation remains auditable."""
    output_file_path = Path(output_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"files": files_with_unknown}).write_csv(output_path)
    logger.info(f"Updated {len(files_with_unknown)} files. List saved to {output_path}")


def enforce_nulls(
    input_dir: str,
    output_path: str | None = None,
    column_names: Optional[List[str]] = None,
    old_value: str = "Unknown",
    new_value: Optional[str] = None,
    verbose: bool = False,
) -> List[str]:
    """Rewrite placeholder metadata to null so missing values are stored as null, not text.

    Args:
        input_dir: Directory containing Parquet files to update.
        output_path: Optional CSV destination listing modified files.
        column_names: Metadata columns to inspect.
        old_value: Placeholder string to replace, typically ``Unknown``.
        new_value: Replacement value, normally null.
        verbose: Whether to print processing details.

    Returns:
        Paths of files that were updated.
    """
    if column_names is None:
        column_names = DEFAULT_COLUMNS

    logger.debug(f"Processing directory: {input_dir}")
    logger.debug(f"Columns: {', '.join(column_names)}")
    logger.debug(f"Replacing '{old_value}' with {new_value}")
    if output_path:
        logger.debug(f"Output file: {output_path}")

    matched_files = find_files(input_dir=input_dir, file_pattern="**/*.parquet")

    logger.debug(f"Found {len(matched_files)} parquet files to process")

    files_with_unknown: List[str] = []
    for file in tqdm(matched_files, unit="file"):
        if _process_parquet_file(file, column_names, old_value, new_value, verbose):
            files_with_unknown.append(file)

    if output_path is not None:
        if files_with_unknown:
            _write_affected_files_list(output_path, files_with_unknown)
        else:
            logger.info("No files were updated.")

    return files_with_unknown


@app.command()
def main(
    input_dir: Annotated[
        List[Path],
        typer.Option(
            "--input-dir",
            "-i",
            help="Directory containing parquet files (repeatable)",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output-file",
            "-o",
            help="CSV of affected file paths",
        ),
    ] = Path("enforced_nulls.csv"),
    column: Annotated[
        Optional[List[str]],
        typer.Option(
            "--column",
            "-c",
            help="Column name(s) to process (repeatable; default collision_energy, frag_type)",
        ),
    ] = None,
    old_value: Annotated[
        str,
        typer.Option("--old-value", help="Value to replace"),
    ] = "Unknown",
    new_value: Annotated[
        Optional[str],
        typer.Option("--new-value", help="New value (omit for null)"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Replace placeholder metadata with null in parquet files."""
    configure_script_logging(verbose=verbose)

    all_updated: List[str] = []
    for directory in input_dir:
        if not directory.exists():
            logger.warning(f"Directory '{directory}' does not exist, skipping...")
            continue
        logger.info(f"Processing directory: {directory}")
        all_updated.extend(
            enforce_nulls(
                input_dir=str(directory),
                output_path=None,
                column_names=column,
                old_value=old_value,
                new_value=new_value,
                verbose=verbose,
            )
        )

    if all_updated:
        _write_affected_files_list(str(output_file), all_updated)
    else:
        logger.info("No files were updated.")


if __name__ == "__main__":
    app()
