"""Find IPC files that share an experiment basename within dataset trees.

Run this before the same-folder and multi-folder duplicate classifiers so later
cleanup stages have a complete candidate report.

CLI::

    python scripts/preprocessing/detect_all_duplicates.py --help
    python scripts/preprocessing/detect_all_duplicates.py detect-duplicates <data-root>/acfm
    python scripts/preprocessing/detect_all_duplicates.py batch-detect <data-root>/acfm <data-root>/lcfm

Use ``python script.py command --help`` for flags.
"""

import os
from typing import List, Dict
from pathlib import Path
import typer

app = typer.Typer(help="Detect duplicate files in directory structure")

# Module-level constants to avoid B008 errors
SOURCE_DIR_ARG = typer.Argument(..., help="Source directory to search for duplicates")
OUTPUT_FILE_OPTION = typer.Option(
    "duplicate_files.txt", "--output", "-o", help="Output file for duplicates"
)
EXTENSIONS_OPTION = typer.Option(
    [".ipc", ".mzML.ipc"], "--extensions", "-e", help="File extensions to check"
)
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
DIRECTORIES_ARG = typer.Argument(..., help="Directories to check for duplicates")
OUTPUT_DIR_OPTION = typer.Option(
    "outputs", "--output-dir", "-o", help="Output directory for results"
)
PREFIX_OPTION = typer.Option(
    "duplicate_files", "--prefix", "-p", help="Prefix for output files"
)


def find_duplicate_files(
    source_dir: str, output_file: str = "duplicate_files.txt"
) -> None:
    """Create the candidate report needed for safe duplicate classification.

    Files sharing a basename across ``.ipc`` and ``.mzML.ipc`` variants are
    treated as candidates.

    Args:
        source_dir: Directory tree to inspect.
        output_file: Destination for grouped duplicate paths.
    """
    file_dict = group_files_by_base_name(source_dir)
    duplicates = identify_duplicates(file_dict)
    save_duplicates(output_file, duplicates)


def group_files_by_base_name(source_dir: str) -> Dict[str, List[str]]:
    """Preserve every candidate path so duplicate groups can be classified later.

    Args:
        source_dir: Directory tree containing IPC variants.

    Returns:
        Mapping from experiment basename to matching paths.
    """
    file_dict: Dict[str, List[str]] = {}
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".ipc") or file.endswith(".mzML.ipc"):
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
        typer.echo(f"Duplicate files have been saved to {output_file}")
    else:
        typer.echo("No duplicates found.")


@app.command()
def detect_duplicates(
    source_dir: str = SOURCE_DIR_ARG,
    output_file: str = OUTPUT_FILE_OPTION,
    extensions: List[str] = EXTENSIONS_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Generate a complete duplicate candidate report for one dataset.

    Args:
        source_dir: Directory tree to inspect.
        output_file: Destination for duplicate groups.
        extensions: Requested extensions retained for CLI compatibility.
        verbose: Whether to print scan details.
    """
    if verbose:
        typer.echo(f"Searching for duplicates in: {source_dir}")
        typer.echo(f"Output file: {output_file}")
        typer.echo(f"Extensions: {extensions}")

    if not os.path.exists(source_dir):
        typer.echo(f"Error: Source directory '{source_dir}' does not exist", err=True)
        raise typer.Exit(1)

    # Ensure output directory exists
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    find_duplicate_files(source_dir, output_file)

    if verbose:
        typer.echo("Duplicate detection completed successfully!")


@app.command()
def batch_detect(
    directories: List[str] = DIRECTORIES_ARG,
    output_dir: str = OUTPUT_DIR_OPTION,
    prefix: str = PREFIX_OPTION,
) -> None:
    """Generate separate duplicate reports for several datasets.

    Args:
        directories: Dataset trees to inspect.
        output_dir: Directory that receives the reports.
        prefix: Prefix used for each report filename.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for directory in directories:
        if not os.path.exists(directory):
            typer.echo(
                f"Warning: Directory '{directory}' does not exist, skipping...",
                err=True,
            )
            continue

        output_file = output_path / f"{prefix}_{Path(directory).name}.txt"
        typer.echo(f"Checking duplicates in: {directory}")
        find_duplicate_files(directory, str(output_file))


def main() -> None:
    """Preserve backwards-compatible scans of the historical hardcoded paths."""
    # Legacy behaviour for backwards compatibility
    directories = ["<data-root>/acfm", "<data-root>/lcfm"]
    output_dir = "preprocessing/outputs"

    for directory in directories:
        if os.path.exists(directory):
            output_file = (
                f"{output_dir}/duplicate_files_{os.path.basename(directory)}.txt"
            )
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            find_duplicate_files(directory, output_file)
        else:
            typer.echo(f"Warning: Directory '{directory}' does not exist", err=True)


if __name__ == "__main__":
    app()
