"""Fill missing isolation targets in Parquet spectra from available metadata.

Run this after conversion when ``isolation_target`` is absent or NaN. The script
prefers header-derived values, falls back to precursor m/z, and records files it
cannot infer.

CLI::

    python scripts/preprocessing/infer_isolation_target.py --help
    python scripts/preprocessing/infer_isolation_target.py --input-dir "<data-root>/acfm/**/*.parquet" --output-file modified_files.txt
    python scripts/preprocessing/infer_isolation_target.py --input-dir "<data-root>/acfm/**/*.parquet" --input-dir "<data-root>/lcfm/**/*.parquet" --output-file modified_files.txt

Use ``python script.py --help`` for flags.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path
from typing import Annotated, List, Tuple

import polars as pl
import typer
from tqdm import tqdm

from scripts.logging_setup import configure_script_logging
from scripts.preprocessing.parquet_io import nan_string_to_null_expr

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Infer isolation target values in parquet files",
    no_args_is_help=True,
    add_completion=False,
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
            logger.warning(
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
        logger.debug(f"Processing file: {file_path}")

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
        logger.info(
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
        logger.info(
            f"A list of {len(error_files)} files that could not be inferred is at {error_log_file}."
        )


def infer_isolation_target(
    source_dir: str,
    log_file: str | None = "modified_files.txt",
    error_log_file: str | None = "error_files.txt",
    verbose: bool = False,
) -> tuple[list[str], list[str]]:
    """Repair missing isolation metadata before spectra enter later pipeline stages.

    Args:
        source_dir: Glob pattern selecting Parquet files.
        log_file: Destination for modified paths, or None to skip writing.
        error_log_file: Destination for unresolved paths, or None to skip writing.
        verbose: Whether to print processing details.

    Returns:
        ``(modified_files, error_files)`` path lists.
    """
    logger.debug(f"Processing directory pattern: {source_dir}")
    if log_file:
        logger.debug(f"Log file: {log_file}")
    if error_log_file:
        logger.debug(f"Error log file: {error_log_file}")

    modified_files: list[str] = []
    error_files: list[str] = []

    for file_path in tqdm(glob.glob(source_dir), unit="file"):
        was_modified, had_error = process_single_file(file_path, verbose=verbose)

        if was_modified:
            modified_files.append(file_path)
        elif had_error:
            error_files.append(file_path)

    if log_file is not None and error_log_file is not None:
        write_log_files(modified_files, error_files, log_file, error_log_file)

    return modified_files, error_files


@app.command()
def main(
    input_dir: Annotated[
        List[str],
        typer.Option(
            "--input-dir",
            "-i",
            help="Glob pattern selecting parquet files (repeatable)",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output-file",
            "-o",
            help="Log of files whose isolation_target was inferred",
        ),
    ] = Path("modified_files.txt"),
    error_log: Annotated[
        Path,
        typer.Option("--error-log", help="Log of files that could not be inferred"),
    ] = Path("error_files.txt"),
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Repair missing isolation targets for selected parquet globs."""
    configure_script_logging(verbose=verbose)

    all_modified: list[str] = []
    all_errors: list[str] = []

    for pattern in input_dir:
        logger.info(f"Processing directory pattern: {pattern}")
        modified, errors = infer_isolation_target(
            source_dir=pattern,
            log_file=None,
            error_log_file=None,
            verbose=verbose,
        )
        all_modified.extend(modified)
        all_errors.extend(errors)

    write_log_files(all_modified, all_errors, str(output_file), str(error_log))


if __name__ == "__main__":
    app()
