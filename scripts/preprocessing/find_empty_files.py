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
    """Find files in a local directory that match a specified pattern."""
    search_pattern = Path(input_dir) / file_pattern
    matched_files = glob.glob(str(search_pattern), recursive=True)
    return matched_files


def check_if_empty(file_path: str, flagged_files: list[str]) -> list[str]:
    """Check if the file is empty."""
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
    """Flags IPC files smaller than a specified size threshold in a directory recursively.

    Args:
        source_dir (str): Path to the source directory to search for files.
        output_file (str): Path to the output file to save flagged file details.
        file_pattern (str): Glob pattern for file matching
        min_size_bytes (int): Minimum file size in bytes
        verbose (bool): Enable verbose output
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
    """Find empty or small files in local directories."""
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
    """Find empty files in multiple directories."""
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
    """Entry point for the script to find empty ipc files."""
    # Legacy behavior for backward compatibility
    directories = ["hcfm", "mcfm", "lcfm", "acfm"]
    output_dir = "output_files"

    for directory in directories:
        output_file = f"{output_dir}/small_files_{directory}.txt"
        typer.echo(f"Checking empty files in: {directory}")
        flag_small_files_in_dir(directory, output_file)


if __name__ == "__main__":
    app()
