"""Replace placeholder strings such as ``Unknown`` with null in metadata columns.

Some source files store missing ``collision_energy`` or ``frag_type`` as the
text ``Unknown``. Downstream schemas treat that as a real value, so this script
rewrites it to null before alignment or training. Batch mode writes a separate
affected-file report for each dataset.

CLI::

    python scripts/preprocessing/enforce_nulls.py --help
    python scripts/preprocessing/enforce_nulls.py enforce <data-root>/lcfm
    python scripts/preprocessing/enforce_nulls.py batch-enforce <data-root>/lcfm <data-root>/hcfm

Use ``python script.py command --help`` for flags.
"""

import polars as pl
from tqdm import tqdm
import logging
from typing import List, Optional
from pathlib import Path
import glob
import typer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Enforce null values in parquet files")

# Default columns to process
DEFAULT_COLUMNS = ["collision_energy", "frag_type"]

# Module-level constants to avoid B008 errors
INPUT_DIR_ARG = typer.Argument(..., help="Input directory to process")
OUTPUT_FILE_OPTION = typer.Option(
    "enforced_nulls.csv",
    "--output",
    "-o",
    help="Output file for affected files list",
)
COLUMNS_OPTION = typer.Option(
    None,
    "--column",
    "-c",
    help="Column name(s) to process (can be specified multiple times). "
    "Defaults to: collision_energy, frag_type",
)
OLD_VALUE_OPTION = typer.Option("Unknown", "--old-value", help="Value to replace")
NEW_VALUE_OPTION = typer.Option(None, "--new-value", help="New value (None for null)")
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
INPUT_DIRS_ARG = typer.Argument(..., help="Input directories to process")
OUTPUT_DIR_OPTION = typer.Option(
    "output_files", "--output-dir", "-o", help="Output directory for results"
)
PREFIX_OPTION = typer.Option(
    "unknown_nulls", "--prefix", "-p", help="Prefix for output files"
)


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
        logger.info("Processing file: %s", file)
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
            typer.echo(
                f"Updated file: {file} (columns: {', '.join(columns_to_update)})"
            )
        return True
    except Exception as e:
        typer.echo(f"Error processing {file}: {e}", err=True)
        return False


def _write_affected_files_list(output_path: str, files_with_unknown: List[str]) -> None:
    """Record changed files so normalisation remains auditable."""
    output_file_path = Path(output_path)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"files": files_with_unknown}).write_csv(output_path)
    typer.echo(f"Updated {len(files_with_unknown)} files. List saved to {output_path}")


def enforce_nulls(
    input_dir: str,
    output_path: str,
    column_names: Optional[List[str]] = None,
    old_value: str = "Unknown",
    new_value: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Rewrite placeholder metadata to null so missing values are stored as null, not text.

    Args:
        input_dir: Directory containing Parquet files to update.
        output_path: CSV destination listing modified files.
        column_names: Metadata columns to inspect.
        old_value: Placeholder string to replace, typically ``Unknown``.
        new_value: Replacement value, normally null.
        verbose: Whether to print processing details.
    """
    if column_names is None:
        column_names = DEFAULT_COLUMNS

    if verbose:
        typer.echo(f"Processing directory: {input_dir}")
        typer.echo(f"Columns: {', '.join(column_names)}")
        typer.echo(f"Replacing '{old_value}' with {new_value}")
        typer.echo(f"Output file: {output_path}")

    matched_files = find_files(input_dir=input_dir, file_pattern="**/*.parquet")

    if verbose:
        typer.echo(f"Found {len(matched_files)} parquet files to process")

    files_with_unknown: List[str] = []
    for file in tqdm(matched_files, unit="file"):
        if _process_parquet_file(file, column_names, old_value, new_value, verbose):
            files_with_unknown.append(file)

    if files_with_unknown:
        _write_affected_files_list(output_path, files_with_unknown)
    else:
        typer.echo("No files were updated.")


@app.command()
def enforce(
    input_dir: str = INPUT_DIR_ARG,
    output_file: str = OUTPUT_FILE_OPTION,
    columns: Optional[List[str]] = COLUMNS_OPTION,
    old_value: str = OLD_VALUE_OPTION,
    new_value: Optional[str] = NEW_VALUE_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Replace placeholder metadata with null in one dataset before downstream use.

    Args:
        input_dir: Directory containing Parquet files.
        output_file: CSV destination listing modified files.
        columns: Metadata columns to inspect.
        old_value: Placeholder string to replace, typically ``Unknown``.
        new_value: Replacement value, normally null.
        verbose: Whether to print processing details.
    """
    enforce_nulls(
        input_dir=input_dir,
        output_path=output_file,
        column_names=columns,
        old_value=old_value,
        new_value=new_value,
        verbose=verbose,
    )


@app.command()
def batch_enforce(
    input_dirs: List[str] = INPUT_DIRS_ARG,
    output_dir: str = OUTPUT_DIR_OPTION,
    columns: Optional[List[str]] = COLUMNS_OPTION,
    old_value: str = OLD_VALUE_OPTION,
    new_value: Optional[str] = NEW_VALUE_OPTION,
    prefix: str = PREFIX_OPTION,
) -> None:
    """Normalise several datasets while keeping separate change reports.

    Args:
        input_dirs: Dataset directories to process.
        output_dir: Directory that receives change reports.
        columns: Metadata columns to inspect.
        old_value: Placeholder string to replace, typically ``Unknown``.
        new_value: Replacement value, normally null.
        prefix: Prefix used for report filenames.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for input_dir in input_dirs:
        output_file = output_path / f"{prefix}_{input_dir}.csv"
        typer.echo(f"Processing directory: {input_dir}")
        enforce_nulls(
            input_dir=input_dir,
            output_path=str(output_file),
            column_names=columns,
            old_value=old_value,
            new_value=new_value,
        )


def main() -> None:
    """Preserve backwards-compatible normalisation of historical hardcoded paths."""
    # Legacy behaviour for backwards compatibility
    input_dirs = ["lcfm", "hcfm", "mcfm"]
    output_dir = "output_files"

    for input_dir in input_dirs:
        output_file = f"{output_dir}/unknown_ce_{input_dir}.csv"
        typer.echo(f"Processing directory: {input_dir}")
        enforce_nulls(input_dir, output_file)


if __name__ == "__main__":
    app()
