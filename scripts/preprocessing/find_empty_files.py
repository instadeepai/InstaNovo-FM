"""Find empty IPC files before they cause failures in later preprocessing.

Run this validation after collecting IPC data and before conversion.

CLI::

    python scripts/preprocessing/find_empty_files.py --help
    python scripts/preprocessing/find_empty_files.py --input-dir <data-root>/acfm --output-file empty_files.txt
    python scripts/preprocessing/find_empty_files.py --input-dir <data-root>/acfm --input-dir <data-root>/lcfm --output-file empty_files.txt

Use ``python script.py --help`` for flags.
"""

from __future__ import annotations

import glob
import logging
import os
from pathlib import Path
from typing import Annotated, List

import polars as pl
import typer
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Find empty or small files in local directories",
    no_args_is_help=True,
    add_completion=False,
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
    if lf.first().collect().is_empty():
        flagged_files.append(file_path)
    return flagged_files


def flag_small_files_in_dir(
    source_dir: str,
    file_pattern: str = "**/*.ipc",
    min_size_bytes: int = 0,
    verbose: bool = False,
) -> list[str]:
    """Scan one directory and return paths of empty IPC files.

    Args:
        source_dir: Directory tree containing IPC files.
        file_pattern: Glob selecting files to inspect.
        min_size_bytes: Requested size threshold retained for CLI compatibility.
        verbose: Whether to print scan details.

    Returns:
        Paths of empty IPC files under ``source_dir``.
    """
    if not os.path.exists(source_dir):
        typer.echo(f"Error: Directory '{source_dir}' does not exist", err=True)
        raise typer.Exit(1)

    if verbose:
        typer.echo(f"Searching for files in: {source_dir}")
        typer.echo(f"File pattern: {file_pattern}")
        typer.echo(f"Minimum size: {min_size_bytes} bytes")

    matched_files = find_files(input_dir=source_dir, file_pattern=file_pattern)

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

    return flagged_files


@app.command()
def main(
    input_dir: Annotated[
        List[Path],
        typer.Option(
            "--input-dir",
            "-i",
            help="Dataset tree to search (repeatable)",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output-file",
            "-o",
            help="Output file for empty file paths",
        ),
    ] = Path("empty_files.txt"),
    pattern: Annotated[
        str,
        typer.Option("--pattern", help="File pattern to match"),
    ] = "**/*.ipc",
    min_size: Annotated[
        int,
        typer.Option("--min-size", help="Minimum file size in bytes (CLI compatibility)"),
    ] = 0,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Report unusable IPC files before conversion."""
    if verbose:
        typer.echo(f"Output file: {output_file}")

    all_flagged: list[str] = []
    for directory in input_dir:
        if not directory.exists():
            typer.echo(
                f"Warning: Directory '{directory}' does not exist, skipping...",
                err=True,
            )
            continue
        typer.echo(f"Checking empty files in: {directory}")
        all_flagged.extend(
            flag_small_files_in_dir(
                source_dir=str(directory),
                file_pattern=pattern,
                min_size_bytes=min_size,
                verbose=verbose,
            )
        )

    if all_flagged:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            f.writelines(path + "\n" for path in all_flagged)
        typer.echo(f"Flagged {len(all_flagged)} files have been saved to {output_file}")
    else:
        typer.echo("No empty files found.")


if __name__ == "__main__":
    app()
