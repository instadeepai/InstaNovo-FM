"""Separate cross-folder duplicates that require manual ownership decisions.

Run this on output from ``detect_all_duplicates.py`` so ambiguous copies in
different project folders can be reviewed before any deletion.

CLI::

    python scripts/preprocessing/detect_multi_folder_duplicates.py --help
    python scripts/preprocessing/detect_multi_folder_duplicates.py detect-duplicates preprocessing/outputs/duplicate_files_acfm.txt
    python scripts/preprocessing/detect_multi_folder_duplicates.py batch-detect preprocessing/outputs/duplicate_files_acfm.txt preprocessing/outputs/duplicate_files_lcfm.txt

Use ``python script.py command --help`` for flags.
"""

from collections import defaultdict
from pathlib import Path
import typer

app = typer.Typer(help="Detect multi-folder duplicate files")

# Module-level constants to avoid B008 errors
INPUT_FILE_ARG = typer.Argument(..., help="Input file containing duplicate information")
OUTPUT_FILE_OPTION = typer.Option(
    "multi_folder_duplicates.txt",
    "--output",
    "-o",
    help="Output file for multi-folder duplicates",
)
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
INPUT_FILES_ARG = typer.Argument(
    ..., help="Input files containing duplicate information"
)
OUTPUT_DIR_OPTION = typer.Option(
    "outputs", "--output-dir", "-o", help="Output directory for results"
)
PREFIX_OPTION = typer.Option(
    "multi_folder_duplicates", "--prefix", "-p", help="Prefix for output files"
)


def find_duplicate_files(
    input_file: str, output_file: str = "multi_folder_duplicates.txt"
) -> None:
    """Separate ambiguous cross-folder copies for manual resolution.

    This consumes ``detect_all_duplicates.py`` output because only one folder
    entry can represent the intended dataset record.

    Args:
        input_file: Duplicate report to classify by parent folder.
        output_file: Destination for cross-folder duplicate groups.

    Raises:
        typer.Exit: If the input report does not exist.
    """
    if not Path(input_file).exists():
        typer.echo(f"Error: Input file '{input_file}' does not exist", err=True)
        raise typer.Exit(1)

    # Dictionary to store file names and their corresponding folders
    file_map = defaultdict(list)

    # Read the input file and organise data
    with open(input_file, "r") as infile:
        for line in infile:
            line = line.strip()
            if line:  # Skip empty lines
                folder, file_name = line.rsplit("/", 1)
                base_name = file_name.split(".", 1)[0]  # Extract name before first dot
                file_map[base_name].append(folder)

    # Find duplicates
    duplicates = {
        file_name: folders
        for file_name, folders in file_map.items()
        if len(folders) > 1
    }

    if duplicates:
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write duplicates to the output file
        with open(output_file, "w") as outfile:
            for file_name, folders in duplicates.items():
                outfile.write(f"File: {file_name}\n")
                outfile.write("Folders:\n")
                for folder in folders:
                    outfile.write(f"  - {folder}\n")
                outfile.write("\n")
        typer.echo(f"Duplicate detection report written to {output_file}")
        typer.echo(f"Found {len(duplicates)} files with multi-folder duplicates")
    else:
        typer.echo(f"No multi-folder duplicates found in {input_file}.")


@app.command()
def detect_duplicates(
    input_file: str = INPUT_FILE_ARG,
    output_file: str = OUTPUT_FILE_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Produce a review list before deleting duplicates across folder boundaries.

    Args:
        input_file: Duplicate report produced by the all-duplicates detector.
        output_file: Destination for cross-folder groups.
        verbose: Whether to print selected paths.
    """
    if verbose:
        typer.echo(f"Input file: {input_file}")
        typer.echo(f"Output file: {output_file}")

    find_duplicate_files(input_file, output_file)


@app.command()
def batch_detect(
    input_files: list[str] = INPUT_FILES_ARG,
    output_dir: str = OUTPUT_DIR_OPTION,
    prefix: str = PREFIX_OPTION,
) -> None:
    """Classify several duplicate reports for a batch of datasets.

    Args:
        input_files: Duplicate reports to classify.
        output_dir: Directory that receives classified reports.
        prefix: Prefix used for each output filename.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for input_file in input_files:
        if not Path(input_file).exists():
            typer.echo(
                f"Warning: Input file '{input_file}' does not exist, skipping...",
                err=True,
            )
            continue

        output_file = output_path / f"{prefix}_{Path(input_file).stem}.txt"
        typer.echo(f"Processing: {input_file}")
        find_duplicate_files(input_file, str(output_file))


def main() -> None:
    """Preserve backwards-compatible classification of the hardcoded ACFM report."""
    # Legacy behaviour for backwards compatibility
    input_file = "preprocessing/outputs/duplicate_files_acfm.txt"
    output_file = "preprocessing/outputs/multi_folder_duplicates_acfm.txt"

    if Path(input_file).exists():
        find_duplicate_files(input_file, output_file)
    else:
        typer.echo(f"Warning: Input file '{input_file}' does not exist", err=True)


if __name__ == "__main__":
    app()
