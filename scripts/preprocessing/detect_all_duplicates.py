"""Find IPC files that share an experiment basename within dataset trees.

Run this before the same-folder and multi-folder duplicate classifiers so later
cleanup stages have a complete candidate report.

CLI::

    uv run python -m scripts.preprocessing.detect_all_duplicates --help
    uv run python -m scripts.preprocessing.detect_all_duplicates --input-dir <data-root>/acfm --output-file duplicates.txt
    uv run python -m scripts.preprocessing.detect_all_duplicates --input-dir <data-root>/acfm --input-dir <data-root>/lcfm --output-file duplicates.txt

Run from the repository root; see ``scripts/README.md`` for the ``uv run python -m`` invocation.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated, Dict, List, Optional

import typer

from scripts.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Detect duplicate files in directory structure",
    no_args_is_help=True,
    add_completion=False,
)


def find_duplicate_files(
    source_dir: str,
    output_file: str = "duplicate_files.txt",
    extensions: Optional[List[str]] = None,
) -> None:
    """Create the candidate report needed for safe duplicate classification.

    Files sharing a basename across configured extensions are treated as
    candidates (default ``.ipc`` / ``.mzML.ipc``).

    Args:
        source_dir: Directory tree to inspect.
        output_file: Destination for grouped duplicate paths.
        extensions: Filename suffixes to include.
    """
    file_dict = group_files_by_base_name(source_dir, extensions=extensions)
    duplicates = identify_duplicates(file_dict)
    save_duplicates(output_file, duplicates)


def group_files_by_base_name(
    source_dir: str,
    extensions: Optional[List[str]] = None,
) -> Dict[str, List[str]]:
    """Preserve every candidate path so duplicate groups can be classified later.

    Args:
        source_dir: Directory tree containing candidate files.
        extensions: Filename suffixes to include; defaults to ``.ipc`` / ``.mzML.ipc``.

    Returns:
        Mapping from experiment basename to matching paths.
    """
    if extensions is None:
        extensions = [".ipc", ".mzML.ipc"]

    file_dict: Dict[str, List[str]] = {}
    for root, _, files in os.walk(source_dir):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                base_name = file.split(".")[0]
                if base_name not in file_dict:
                    file_dict[base_name] = []
                file_dict[base_name].append(os.path.join(root, file))
    return file_dict


def identify_duplicates(file_dict: Dict[str, List[str]]) -> List[List[str]]:
    """Discard singleton experiments so reports contain only cleanup candidates.

    Args:
        file_dict: Experiment basenames mapped to candidate paths.

    Returns:
        Path groups containing more than one file.
    """
    return [file_paths for file_paths in file_dict.values() if len(file_paths) > 1]


def save_duplicates(output_file: str, duplicates: List[List[str]]) -> None:
    """Persist duplicate groups for the folder-aware cleanup stages.

    Args:
        output_file: Destination for the duplicate report.
        duplicates: Path groups to record.
    """
    if duplicates:
        with open(output_file, "w") as f:
            for duplicate_group in duplicates:
                f.write("\n".join(duplicate_group) + "\n\n")
        logger.info(f"Duplicate files have been saved to {output_file}")
    else:
        logger.info("No duplicates found.")


@app.command()
def main(
    input_dir: Annotated[
        List[Path],
        typer.Option(
            "--input-dir",
            "-i",
            help="Dataset tree to search for duplicates (repeatable)",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output-file",
            "-o",
            help="Output file for duplicate groups",
        ),
    ] = Path("duplicate_files.txt"),
    extensions: Annotated[
        Optional[List[str]],
        typer.Option(
            "--extensions",
            "-e",
            help="File extensions to include (default: .ipc .mzML.ipc)",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Generate a duplicate candidate report for one or more dataset trees."""
    configure_script_logging(verbose=verbose)

    if extensions is None:
        extensions = [".ipc", ".mzML.ipc"]

    logger.debug(f"Input dirs: {input_dir}")
    logger.debug(f"Output file: {output_file}")
    logger.debug(f"Extensions: {extensions}")

    valid_dirs = [d for d in input_dir if d.exists()]
    for missing in set(input_dir) - set(valid_dirs):
        logger.warning(f"Directory '{missing}' does not exist, skipping...")

    if not valid_dirs:
        typer.echo("Error: no valid input directories", err=True)
        raise typer.Exit(1)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Multiple trees: write one combined report (paths retain their folders).
    all_duplicates: List[List[str]] = []
    for directory in valid_dirs:
        logger.debug(f"Searching for duplicates in: {directory}")
        file_dict = group_files_by_base_name(str(directory), extensions=extensions)
        all_duplicates.extend(identify_duplicates(file_dict))

    save_duplicates(str(output_file), all_duplicates)

    logger.info("Duplicate detection completed successfully!")


if __name__ == "__main__":
    app()
