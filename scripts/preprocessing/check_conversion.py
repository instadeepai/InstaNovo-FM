import os
from tqdm import tqdm
import re
from typing import List, Tuple
from pathlib import Path
import typer

app = typer.Typer(help="Check IPC to Parquet conversion completeness")

# Module-level constants to avoid B008 errors
SOURCE_DIR_ARG = typer.Argument(..., help="Source directory to check")
OUTPUT_FILE_OPTION = typer.Option(
    "missing_files.txt", "--output", "-o", help="Output file for missing files"
)
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
DIRECTORIES_ARG = typer.Argument(..., help="Directories to check")
OUTPUT_DIR_OPTION = typer.Option(
    "outputs", "--output-dir", "-o", help="Output directory for results"
)
PREFIX_OPTION = typer.Option(
    "missing_files", "--prefix", "-p", help="Prefix for output files"
)


def check_missing_parquet_variants(
    source_dir: str,
    output_file: str = "missing_files.txt",
) -> List[Tuple[str, str]]:
    """Checks for IPC files without corresponding Parquet files or shards.

    Parameters:
        source_dir (str): Directory to search for files.
        output_file (str): File to save the list of IPC files with missing outputs.

    Returns:
        List[Tuple[str, str]]: List of IPC files and reasons for missing outputs.
    """
    ipc_files = find_ipc_files(source_dir)
    missing_files = find_missing_parquet_files(ipc_files)
    save_missing_files(output_file, missing_files)
    return missing_files


def find_ipc_files(source_dir: str) -> List[str]:
    """Finds all .ipc files in the given directory."""
    ipc_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".ipc"):
                ipc_files.append(os.path.join(root, file))
    return ipc_files


def find_missing_parquet_files(ipc_files: List[str]) -> List[Tuple[str, str]]:
    """Identifies IPC files without corresponding Parquet files or with incomplete shards."""
    shard_pattern = re.compile(r".+_\d{4}-\d{4}.parquet$")
    missing_files = []

    with tqdm(total=len(ipc_files), desc="Checking files", unit="file") as pbar:
        for ipc_path in ipc_files:
            ipc_dir = os.path.dirname(ipc_path)
            ipc_basename = os.path.basename(ipc_path).replace(".ipc", "")
            matches = collect_matching_files(ipc_dir, ipc_basename, ".parquet")

            if not matches:
                missing_files.append((ipc_path, "No matching Parquet files found"))
            else:
                if is_shard_incomplete(matches, shard_pattern):
                    missing_files.append((ipc_path, "Missing or inconsistent shards"))

            pbar.update(1)

    return missing_files


def collect_matching_files(directory: str, basename: str, extension: str) -> List[str]:
    """Collects files in a directory matching the given basename and extension."""
    return [
        file
        for file in os.listdir(directory)
        if file.endswith(extension) and file.startswith(basename)
    ]


def is_shard_incomplete(matches: List[str], shard_pattern: re.Pattern) -> bool:
    """Checks if shard files are incomplete or inconsistent."""
    shard_files = [f for f in matches if shard_pattern.match(f)]
    if shard_files:
        shard_numbers = [int(f.split("-")[-1].split(".")[0]) for f in shard_files]
        if len(set(shard_numbers)) > 1:
            return True
        expected_shards = max(shard_numbers)
        if len(shard_files) != expected_shards:
            return True
    return False


def save_missing_files(output_file: str, missing_files: List[Tuple[str, str]]) -> None:
    """Saves missing file details to the specified output file."""
    if missing_files:
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for file, reason in missing_files:
                f.write(f"{file}: {reason}\n")
        typer.echo(f"Missing files have been saved to {output_file}")
    else:
        typer.echo("No missing files found.")


@app.command()
def check_conversion(
    source_dir: str = SOURCE_DIR_ARG,
    output_file: str = OUTPUT_FILE_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Check IPC to Parquet conversion completeness."""
    if verbose:
        typer.echo(f"Checking directory: {source_dir}")
        typer.echo(f"Output file: {output_file}")

    if not os.path.exists(source_dir):
        typer.echo(f"Error: Source directory '{source_dir}' does not exist", err=True)
        raise typer.Exit(1)

    missing = check_missing_parquet_variants(source_dir, output_file=output_file)

    if missing:
        typer.echo(f"Found {len(missing)} files with missing Parquet variants:")
        for file, reason in missing:
            typer.echo(f"  {file}: {reason}")
    else:
        typer.echo("All IPC files have complete Parquet variants.")


@app.command()
def batch_check(
    directories: List[str] = DIRECTORIES_ARG,
    output_dir: str = OUTPUT_DIR_OPTION,
    prefix: str = PREFIX_OPTION,
) -> None:
    """Check conversion completeness in multiple directories."""
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
        typer.echo(f"Checking directory: {directory}")
        check_missing_parquet_variants(directory, str(output_file))


def main() -> None:
    """Entry point for the script to check all IPC to Parquet file conversion."""
    # Legacy behavior for backward compatibility
    directories_to_check = [
        "<data-root>/acfm",
        "<data-root>/lcfm",
        "<data-root>/mcfm",
        "<data-root>/hcfm",
    ]

    for directory in directories_to_check:
        if os.path.exists(directory):
            typer.echo(f"Checking directory: {directory}")
            output_file = (
                f"preprocessing/outputs/missing_files_{os.path.basename(directory)}.txt"
            )
            missing = check_missing_parquet_variants(directory, output_file=output_file)

            if missing:
                typer.echo(
                    f"Found {len(missing)} files with missing Parquet variants in {directory}"
                )
            else:
                typer.echo(
                    f"All IPC files in {directory} have complete Parquet variants."
                )
        else:
            typer.echo(f"Warning: Directory '{directory}' does not exist", err=True)


if __name__ == "__main__":
    app()
