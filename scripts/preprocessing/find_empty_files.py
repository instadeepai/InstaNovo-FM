"""Find empty IPC files before they cause failures in later preprocessing.

Run this validation after collecting IPC data and before conversion; reports can
also be generated for several dataset directories in one invocation.

CLI::

    python scripts/preprocessing/find_empty_files.py --help
    python scripts/preprocessing/find_empty_files.py find-empty <data-root>/acfm
    python scripts/preprocessing/find_empty_files.py batch-find-empty <data-root>/acfm <data-root>/lcfm

Use ``python script.py command --help`` for flags.
"""

import polars as pl
from tqdm import tqdm
import logging
from typing import List
from pathlib import Path
import glob
import typer
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Find empty or small files in local directories")

# Module-level constants to avoid B008 errors
SOURCE_DIR_ARG = typer.Argument(..., help="Source directory to search")
OUTPUT_FILE_OPTION = typer.Option(
    "empty_files.txt", "--output", "-o", help="Output file for empty files"
)
FILE_PATTERN_OPTION = typer.Option(
    "**/*.ipc", "--pattern", "-p", help="File pattern to match"
)
MIN_SIZE_OPTION = typer.Option(0, "--min-size", "-s", help="Minimum file size in bytes")
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
DIRECTORIES_ARG = typer.Argument(..., help="Directories to check for empty files")
OUTPUT_DIR_OPTION = typer.Option(
    "output_files", "--output-dir", "-o", help="Output directory for results"
)
PREFIX_OPTION = typer.Option(
    "small_files", "--prefix", "-p", help="Prefix for output files"
)


def find_files(input_dir: str, file_pattern: str) -> list[str]:
    """Provide the candidate files that an emptiness check should inspect.

    Args:
        input_dir: Root directory for the search.
        file_pattern: Recursive glob selecting candidate files.

    Returns:
        Paths matching the requested pattern.
    """
    search_pattern = Path(input_dir) / file_pattern
    matched_files = glob.glob(str(search_pattern), recursive=True)
    return matched_files


def check_if_empty(file_path: str, flagged_files: list[str]) -> list[str]:
    """Accumulate empty IPC paths for a report without interrupting the scan.

    Args:
        file_path: IPC file to inspect.
        flagged_files: Existing collection of empty file paths.

    Returns:
        The collection, including the input path when its IPC table is empty.
    """
    lf = pl.scan_ipc(file_path)
    if lf.first().collect().is_empty():  # an empty file
        flagged_files.append(file_path)
    return flagged_files


def flag_small_files_in_dir(
    source_dir: str,
    output_file: str,
    file_pattern: str = "**/*.ipc",
    min_size_bytes: int = 0,
    verbose: bool = False,
) -> None:
    """Create an actionable report of empty IPC files before conversion.

    Args:
        source_dir: Directory tree containing IPC files.
        output_file: Destination for paths that require removal or replacement.
        file_pattern: Glob selecting files to inspect.
        min_size_bytes: Requested size threshold retained for CLI compatibility.
        verbose: Whether to print scan details.
    """
    # Check if directory exists
    if not os.path.exists(source_dir):
        typer.echo(f"Error: Directory '{source_dir}' does not exist", err=True)
        raise typer.Exit(1)

    if verbose:
        typer.echo(f"Searching for files in: {source_dir}")
        typer.echo(f"File pattern: {file_pattern}")
        typer.echo(f"Minimum size: {min_size_bytes} bytes")
        typer.echo(f"Output file: {output_file}")

    matched_files = find_files(
        input_dir=source_dir,
        file_pattern=file_pattern,
    )

    if verbose:
        typer.echo(f"Found {len(matched_files)} files to check")

    flagged_files: list[str] = []

    for file in tqdm(matched_files, unit="file"):
        if verbose:
            logger.info(f"Processing file: {file}")
        try:
            flagged_files = check_if_empty(file, flagged_files)
        except Exception as e:
            typer.echo(f"Error checking file {file}: {e}", err=True)

    # Report flagged files
    if flagged_files:
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            f.writelines(file_path + "\n" for file_path in flagged_files)
        typer.echo(
            f"Flagged {len(flagged_files)} files have been saved to {output_file}"
        )
    else:
        typer.echo("No empty files found.")


@app.command()
def find_empty(
    source_dir: str = SOURCE_DIR_ARG,
    output_file: str = OUTPUT_FILE_OPTION,
    file_pattern: str = FILE_PATTERN_OPTION,
    min_size: int = MIN_SIZE_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Report unusable IPC files in one dataset directory before conversion.

    Args:
        source_dir: Directory tree containing IPC files.
        output_file: Destination for flagged paths.
        file_pattern: Glob selecting IPC files.
        min_size: Requested size threshold retained for CLI compatibility.
        verbose: Whether to print scan details.
    """
    flag_small_files_in_dir(
        source_dir=source_dir,
        output_file=output_file,
        file_pattern=file_pattern,
        min_size_bytes=min_size,
        verbose=verbose,
    )


@app.command()
def batch_find_empty(
    directories: List[str] = DIRECTORIES_ARG,
    output_dir: str = OUTPUT_DIR_OPTION,
    file_pattern: str = FILE_PATTERN_OPTION,
    prefix: str = PREFIX_OPTION,
) -> None:
    """Generate separate empty-file reports for several dataset directories.

    Args:
        directories: Dataset directories to inspect.
        output_dir: Directory that receives the reports.
        file_pattern: Glob selecting IPC files.
        prefix: Prefix used for each report filename.
    """
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for directory in directories:
        output_file = output_path / f"{prefix}_{Path(directory).name}.txt"
        typer.echo(f"Checking empty files in: {directory}")
        flag_small_files_in_dir(
            source_dir=directory,
            output_file=str(output_file),
            file_pattern=file_pattern,
        )


def main() -> None:
    """Preserve backwards-compatible scans of the historical hardcoded paths."""
    # Legacy behaviour for backwards compatibility
    directories = ["hcfm", "mcfm", "lcfm", "acfm"]
    output_dir = "output_files"

    for directory in directories:
        output_file = f"{output_dir}/small_files_{directory}.txt"
        typer.echo(f"Checking empty files in: {directory}")
        flag_small_files_in_dir(directory, output_file)


if __name__ == "__main__":
    app()
