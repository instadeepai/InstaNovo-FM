"""Identify IPC inputs whose Parquet conversion is missing or incomplete.

Run this after IPC conversion to find absent outputs or inconsistent shard sets
before downstream preprocessing consumes them.

CLI::

    python scripts/preprocessing/check_conversion.py --help
    python scripts/preprocessing/check_conversion.py check-conversion <data-root>/acfm
    python scripts/preprocessing/check_conversion.py batch-check <data-root>/acfm <data-root>/lcfm

Use ``python script.py command --help`` for flags.
"""

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
    """Reveal conversion gaps so callers can rerun only incomplete IPC inputs.

    Args:
        source_dir: Directory whose IPC inputs and Parquet outputs should be compared.
        output_file: Report destination for missing or inconsistent conversions.

    Returns:
        IPC paths paired with reasons that their Parquet output is incomplete.
    """
    ipc_files = find_ipc_files(source_dir)
    missing_files = find_missing_parquet_files(ipc_files)
    save_missing_files(output_file, missing_files)
    return missing_files


def find_ipc_files(source_dir: str) -> List[str]:
    """Supply the complete IPC input set needed for conversion verification.

    Args:
        source_dir: Directory tree to search for IPC files.

    Returns:
        IPC paths found beneath the source directory.
    """
    ipc_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if file.endswith(".ipc"):
                ipc_files.append(os.path.join(root, file))
    return ipc_files


def find_missing_parquet_files(ipc_files: List[str]) -> List[Tuple[str, str]]:
    """Isolate IPC inputs that need conversion or shard repair.

    Args:
        ipc_files: IPC paths whose neighboring Parquet outputs should be checked.

    Returns:
        IPC paths paired with explanations of the detected conversion gap.
    """
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
    """Gather candidate outputs so one IPC input can be checked efficiently.

    Args:
        directory: Directory containing the expected outputs.
        basename: Filename prefix shared by the input and its outputs.
        extension: Output suffix to require.

    Returns:
        Matching filenames in the directory.
    """
    return [
        file
        for file in os.listdir(directory)
        if file.endswith(extension) and file.startswith(basename)
    ]


def is_shard_incomplete(matches: List[str], shard_pattern: re.Pattern) -> bool:
    """Detect shard-number inconsistencies that make a conversion unsafe to consume.

    Args:
        matches: Candidate Parquet filenames for one IPC input.
        shard_pattern: Pattern distinguishing sharded outputs from whole files.

    Returns:
        Whether the shard set has inconsistent totals or missing members.
    """
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
    """Persist conversion gaps so they can drive a targeted rerun.

    Args:
        output_file: Destination for the human-readable report.
        missing_files: IPC paths and reasons to record.
    """
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
    """Verify one directory before allowing its converted data downstream.

    Args:
        source_dir: Directory containing IPC inputs and expected Parquet outputs.
        output_file: Report destination for incomplete conversions.
        verbose: Whether to print input and output details.
    """
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
    """Verify several dataset directories in one preprocessing run.

    Args:
        directories: Dataset directories to check.
        output_dir: Directory that receives one report per dataset.
        prefix: Prefix used to distinguish generated reports.
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
        typer.echo(f"Checking directory: {directory}")
        check_missing_parquet_variants(directory, str(output_file))


def main() -> None:
    """Preserve backwards-compatible checks against the historical hardcoded paths."""
    # Legacy behaviour for backwards compatibility
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
