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
    """Finds and records duplicate files in the source directory based on base filenames.

    Duplicates are defined as having the same base name with different extensions (e.g., .ipc, .mzml.ipc).

    Parameters:
        source_dir (str): The directory to search for duplicate files.
        output_file (str): The file where duplicates will be saved.
    """
    file_dict = group_files_by_base_name(source_dir)
    duplicates = identify_duplicates(file_dict)
    save_duplicates(output_file, duplicates)


def group_files_by_base_name(source_dir: str) -> Dict[str, List[str]]:
    """Groups files by their base names in the given directory."""
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
    """Identifies duplicate files from the grouped files dictionary."""
    return [file_paths for file_paths in file_dict.values() if len(file_paths) > 1]


def save_duplicates(output_file: str, duplicates: List[List[str]]) -> None:
    """Saves the list of duplicate files to a text file."""
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
    """Detect duplicate files in a directory structure."""
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
    """Detect duplicates in multiple directories."""
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
    """Entry point for the script to detect duplicate named files in a folder system."""
    # Legacy behavior for backward compatibility
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
