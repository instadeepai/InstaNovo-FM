"""Fill missing isolation targets in Parquet spectra from available metadata.

Run this after conversion when ``isolation_target`` is absent or NaN. The script
prefers header-derived values, falls back to precursor m/z, and records files it
cannot infer; batch mode processes several glob patterns.

CLI::

    python scripts/preprocessing/infer_isolation_target.py --help
    python scripts/preprocessing/infer_isolation_target.py infer-targets "<data-root>/acfm/**/*.parquet"
    python scripts/preprocessing/infer_isolation_target.py batch-infer "<data-root>/acfm/**/*.parquet" "<data-root>/lcfm/**/*.parquet"

Use ``python script.py command --help`` for flags.
"""

import polars as pl
import glob
import os
import re
from tqdm import tqdm
from typing import Tuple
from pathlib import Path
import typer

from scripts.preprocessing.parquet_io import nan_string_to_null_expr

app = typer.Typer(help="Infer isolation target values in parquet files")

# Module-level constants to avoid B008 errors
SOURCE_DIR_ARG = typer.Argument(
    ..., help="Source directory pattern to search for parquet files"
)
LOG_FILE_OPTION = typer.Option(
    "modified_files.txt", "--log", "-l", help="File to log modified files"
)
ERROR_LOG_OPTION = typer.Option(
    "error_files.txt", "--error-log", "-e", help="File to log error files"
)
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
SOURCE_DIRS_ARG = typer.Argument(..., help="Source directory patterns to search")
LOG_DIR_OPTION = typer.Option(
    "outputs", "--log-dir", "-l", help="Directory for log files"
)
PREFIX_OPTION = typer.Option(
    "modified_files", "--prefix", "-p", help="Prefix for log files"
)


def _isolation_target_missing() -> pl.Expr:
    """Treat every known missing-value encoding consistently during inference."""
    return pl.col("isolation_target").is_null() | pl.col("isolation_target").cast(
        pl.String, strict=False
    ).str.to_lowercase().eq("nan")


def check_for_empty_it(ldf: pl.LazyFrame) -> bool:
    """Skip expensive rewrites when every isolation target is already usable.

    Args:
        ldf: Spectra table to inspect lazily.

    Returns:
        Whether at least one isolation target needs inference.
    """
    return bool(ldf.select(_isolation_target_missing().any()).collect().item())


def extract_file_name(file_path: str) -> Tuple[str, str]:
    """Build the project and experiment key needed for metadata matching.

    Args:
        file_path: Parquet path organised beneath its project directory.

    Returns:
        Project name and normalised experiment basename.
    """
    filename = os.path.basename(file_path)
    project = os.path.basename(os.path.dirname(file_path))
    # Strip anything after a full stop
    filename = filename.split(".")[0]
    # Strip sharding, if it occurs
    shard_pattern = re.compile(r".+_\d{4}-\d{4}$")
    if shard_pattern.match(filename):
        filename = filename[:-10]
    return project, filename


def check_search_data_value(
    file_path: str,
    search_data: pl.DataFrame,
    column: str,
    show_duplicate_warnings: bool = False,
) -> str:
    """Require an unambiguous metadata match before trusting a search-data value.

    Args:
        file_path: Data file whose metadata is needed.
        search_data: Table containing file paths and metadata.
        column: Metadata field to retrieve.
        show_duplicate_warnings: Whether to report equivalent duplicate rows.

    Returns:
        The unique matching metadata value.

    Raises:
        ValueError: If no row matches or matching rows disagree.
    """
    project, filename = extract_file_name(file_path)
    search_path = (
        project + "/" + filename + r"\."
    )  # Add the full stop make sure we only match to a full file name

    # Filter rows that match the current search path
    search_data_match = search_data.filter(
        search_data["file path"].str.contains(search_path)
    )

    # Error handling for no or multiple matches
    match_count = len(search_data_match)
    if match_count == 0:
        raise ValueError(f"No matches found in search_data for phrase: {search_path}")
    elif match_count > 1:
        # Get unique values for the column in the matched rows
        unique_column_values = (
            search_data_match.select(column).unique()[column].to_list()
        )
        if len(unique_column_values) > 1:
            # If there are differing column entries, raise an error
            raise ValueError(
                f"Conflicting column entries found in search_data for file name: {search_path}\n"
                f"Conflicting values: {unique_column_values}\n"
                f"Matched rows:\n{search_data_match.select(['file path', column]).to_pandas().to_string(index=False)}"
            )
        elif show_duplicate_warnings is True:
            # If duplicates have the same column entry, count once and report
            typer.echo(
                f"Duplicate matches found in search_data for file name: {search_path}, "
                f"but they have the same column value: {unique_column_values[0]}. Counting once."
            )

    value: str = search_data_match[column][0]

    return value


def infer_it_from_precursor_mz(ldf: pl.LazyFrame) -> pl.LazyFrame:
    """Use precursor m/z when it is the only reliable isolation proxy available.

    Args:
        ldf: Spectra table with missing isolation targets.

    Returns:
        Lazy table with missing targets filled from precursor m/z.
    """
    # Replace Null values in isolation_target with precursor_mz values
    ldf = ldf.with_columns(
        pl.when(_isolation_target_missing())
        .then(pl.col("precursor_mz"))
        .otherwise(pl.col("isolation_target"))
        .alias("isolation_target")
    )

    return ldf


def infer_it_from_header(ldf: pl.LazyFrame) -> pl.LazyFrame:
    """Prefer instrument header values when reconstructing isolation targets.

    Args:
        ldf: Spectra table with parseable instrument headers.

    Returns:
        Lazy table with missing targets filled from headers.
    """
    # Regex pattern to extract the isolation target value
    isolation_target_pattern = r"\b\d+\.\d+@"

    # Extract float value with @
    inferred_column = (
        pl.col("header")
        .str.extract(isolation_target_pattern, group_index=0)
        .str.head(-1)
        .cast(pl.Float64)
    )

    # Replace Null values in isolation_target with inferred values
    ldf = ldf.with_columns(
        pl.when(_isolation_target_missing())
        .then(inferred_column)
        .otherwise(pl.col("isolation_target"))
        .alias("isolation_target")
    )

    return ldf


def process_single_file(file_path: str, verbose: bool = False) -> tuple[bool, bool]:
    """Isolate inference decisions so a batch can log modified and unusable files.

    Args:
        file_path: Parquet file to inspect and potentially rewrite.
        verbose: Whether to print file-level progress.

    Returns:
        Flags indicating whether the file changed and whether inference failed.
    """
    if verbose:
        typer.echo(f"Processing file: {file_path}")

    query = pl.scan_parquet(file_path)

    # Check if there are empty isolation_target entries, if not then continue to next file.
    has_empty_it = check_for_empty_it(query)
    if not has_empty_it:
        return False, False

    # Copy the isolation_target column and rename the old column to isolation_target_old.
    if "isolation_target_old" not in query.collect_schema():
        query = query.rename({"isolation_target": "isolation_target_old"})
        query = query.with_columns(
            nan_string_to_null_expr("isolation_target_old")
            .cast(pl.Float64, strict=False)
            .alias("isolation_target")
        )

    # Infer empty isolation target values from header if it is not empty.
    if query.select("header").null_count().collect().item() == 0:
        query = infer_it_from_header(query)
    # Infer empty isolation target values from precursor_mz if it is not empty.
    elif query.select("precursor_mz").null_count().collect().item() == 0:
        query = infer_it_from_precursor_mz(query)
    else:  # Write filename to an error file and skip writing a new file.
        return False, True

    # Write the new dataframe to the file.
    df = query.collect()
    df.write_parquet(file_path)

    return True, False


def write_log_files(
    modified_files: list, error_files: list, log_file: str, error_log_file: str
) -> None:
    """Persist inference outcomes so changed and unresolved files remain auditable.

    Args:
        modified_files: Paths rewritten successfully.
        error_files: Paths whose targets could not be inferred.
        log_file: Destination for modified paths.
        error_log_file: Destination for unresolved paths.
    """
    # Write modified filenames to a text file
    if modified_files:
        # Ensure output directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        with open(log_file, "w") as f:
            for filename in modified_files:
                f.write(filename + "\n")
        typer.echo(
            f"A list of {len(modified_files)} modified files is saved at {log_file}."
        )

    # Write error filenames to a text file
    if error_files:
        # Ensure output directory exists
        error_path = Path(error_log_file)
        error_path.parent.mkdir(parents=True, exist_ok=True)

        with open(error_log_file, "w") as f:
            for filename in error_files:
                f.write(filename + "\n")
        typer.echo(
            f"A list of {len(error_files)} files that could not be inferred is at {error_log_file}."
        )


def infer_isolation_target(
    source_dir: str,
    log_file: str = "modified_files.txt",
    error_log_file: str = "error_files.txt",
    verbose: bool = False,
) -> None:
    """Repair missing isolation metadata before spectra enter later pipeline stages.

    Args:
        source_dir: Glob pattern selecting Parquet files.
        log_file: Destination for modified paths.
        error_log_file: Destination for unresolved paths.
        verbose: Whether to print processing details.
    """
    if verbose:
        typer.echo(f"Processing directory pattern: {source_dir}")
        typer.echo(f"Log file: {log_file}")
        typer.echo(f"Error log file: {error_log_file}")

    modified_files = []
    error_files = []

    for file_path in tqdm(glob.glob(source_dir), unit="file"):
        was_modified, had_error = process_single_file(file_path, verbose=verbose)

        if was_modified:
            modified_files.append(file_path)
        elif had_error:
            error_files.append(file_path)

    write_log_files(modified_files, error_files, log_file, error_log_file)


@app.command()
def infer_targets(
    source_dir: str = SOURCE_DIR_ARG,
    log_file: str = LOG_FILE_OPTION,
    error_log: str = ERROR_LOG_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Repair missing isolation targets for one selected dataset pattern.

    Args:
        source_dir: Glob pattern selecting Parquet files.
        log_file: Destination for modified paths.
        error_log: Destination for unresolved paths.
        verbose: Whether to print processing details.
    """
    infer_isolation_target(
        source_dir=source_dir,
        log_file=log_file,
        error_log_file=error_log,
        verbose=verbose,
    )


@app.command()
def batch_infer(
    source_dirs: list[str] = SOURCE_DIRS_ARG,
    log_dir: str = LOG_DIR_OPTION,
    prefix: str = PREFIX_OPTION,
) -> None:
    """Repair several dataset patterns while keeping separate outcome logs.

    Args:
        source_dirs: Glob patterns selecting Parquet files.
        log_dir: Directory that receives outcome logs.
        prefix: Prefix used for modified-file logs.
    """
    output_path = Path(log_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for source_dir in source_dirs:
        log_file = output_path / f"{prefix}_{Path(source_dir).name}.txt"
        error_log = output_path / f"error_{prefix}_{Path(source_dir).name}.txt"
        typer.echo(f"Processing directory pattern: {source_dir}")
        infer_isolation_target(
            source_dir=source_dir,
            log_file=str(log_file),
            error_log_file=str(error_log),
        )


def main() -> None:
    """Preserve backwards-compatible inference over historical hardcoded patterns."""
    # Legacy behaviour for backwards compatibility
    source_dirs = [
        "<data-root>/acfm/**/*.parquet",
        "<data-root>/lcfm/**/*.parquet",
        "<data-root>/hcfm/**/*.parquet",
        "<data-root>/mcfm/**/*.parquet",
    ]
    for source_dir in source_dirs:
        infer_isolation_target(
            source_dir=source_dir,
            log_file="modified_files_new.txt",
        )


if __name__ == "__main__":
    app()
