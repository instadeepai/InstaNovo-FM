"""Separate cross-folder duplicates that require manual ownership decisions.

Run this on output from ``detect_all_duplicates.py`` so ambiguous copies in
different project folders can be reviewed before any deletion.

CLI::

    uv run python -m scripts.preprocessing.detect_multi_folder_duplicates --help
    uv run python -m scripts.preprocessing.detect_multi_folder_duplicates --input-file preprocessing/outputs/duplicate_files_acfm.txt --output-file multi_folder_duplicates.txt
    uv run python -m scripts.preprocessing.detect_multi_folder_duplicates --input-file report_a.txt --input-file report_b.txt --output-file multi_folder_duplicates.txt

Run from the repository root; see ``scripts/README.md`` for the ``uv run python -m`` invocation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Annotated, List

import typer

from scripts.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Detect multi-folder duplicate files",
    no_args_is_help=True,
    add_completion=False,
)


def find_duplicate_files(
    input_file: str, output_file: str = "multi_folder_duplicates.txt"
) -> dict:
    """Separate ambiguous cross-folder copies for manual resolution.

    This consumes ``detect_all_duplicates.py`` output because only one folder
    entry can represent the intended dataset record.

    Args:
        input_file: Duplicate report to classify by parent folder.
        output_file: Destination for cross-folder duplicate groups.

    Returns:
        Mapping of basename to folders for cross-folder duplicates.

    Raises:
        typer.Exit: If the input report does not exist.
    """
    if not Path(input_file).exists():
        typer.echo(f"Error: Input file '{input_file}' does not exist", err=True)
        raise typer.Exit(1)

    file_map = defaultdict(list)

    with open(input_file, "r") as infile:
        for line in infile:
            line = line.strip()
            if line:
                folder, file_name = line.rsplit("/", 1)
                base_name = file_name.split(".", 1)[0]
                file_map[base_name].append(folder)

    duplicates = {
        file_name: folders
        for file_name, folders in file_map.items()
        if len(folders) > 1
    }

    if duplicates:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as outfile:
            for file_name, folders in duplicates.items():
                outfile.write(f"File: {file_name}\n")
                outfile.write("Folders:\n")
                for folder in folders:
                    outfile.write(f"  - {folder}\n")
                outfile.write("\n")
        logger.info(f"Duplicate detection report written to {output_file}")
        logger.info(f"Found {len(duplicates)} files with multi-folder duplicates")
    else:
        logger.info(f"No multi-folder duplicates found in {input_file}.")

    return duplicates


@app.command()
def main(
    input_file: Annotated[
        List[Path],
        typer.Option(
            "--input-file",
            help="Duplicate report from detect_all_duplicates (repeatable)",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output-file",
            "-o",
            help="Output file for multi-folder duplicate groups",
        ),
    ] = Path("multi_folder_duplicates.txt"),
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Produce a review list of duplicates that cross folder boundaries."""
    configure_script_logging(verbose=verbose)

    logger.debug(f"Input files: {input_file}")
    logger.debug(f"Output file: {output_file}")

    valid_files = [f for f in input_file if f.exists()]
    for missing in set(input_file) - set(valid_files):
        logger.warning(f"Input file '{missing}' does not exist, skipping...")

    if not valid_files:
        typer.echo("Error: no valid input files", err=True)
        raise typer.Exit(1)

    # Merge classifications from all reports into one output.
    merged: dict = {}
    for path in valid_files:
        logger.debug(f"Processing: {path}")
        # Write through a temp merge: collect then write once.
        if not path.exists():
            continue
        file_map = defaultdict(list)
        with open(path, "r") as infile:
            for line in infile:
                line = line.strip()
                if line:
                    folder, file_name = line.rsplit("/", 1)
                    base_name = file_name.split(".", 1)[0]
                    file_map[base_name].append(folder)
        for base_name, folders in file_map.items():
            if len(folders) > 1:
                existing = merged.setdefault(base_name, [])
                for folder in folders:
                    if folder not in existing:
                        existing.append(folder)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if merged:
        with open(output_file, "w") as outfile:
            for file_name, folders in merged.items():
                outfile.write(f"File: {file_name}\n")
                outfile.write("Folders:\n")
                for folder in folders:
                    outfile.write(f"  - {folder}\n")
                outfile.write("\n")
        logger.info(f"Duplicate detection report written to {output_file}")
        logger.info(f"Found {len(merged)} files with multi-folder duplicates")
    else:
        logger.info("No multi-folder duplicates found.")


if __name__ == "__main__":
    app()
