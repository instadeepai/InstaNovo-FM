"""Identify IPC inputs whose Parquet conversion is missing or incomplete.

Run this after IPC conversion to find absent outputs or inconsistent shard sets
before downstream preprocessing consumes them.

CLI::

    python scripts/preprocessing/check_conversion.py --help
    python scripts/preprocessing/check_conversion.py --input-dir <data-root>/acfm --output-file missing_files.txt
    python scripts/preprocessing/check_conversion.py --input-dir <data-root>/acfm --input-dir <data-root>/lcfm --output-file missing_files.txt

Use ``python script.py --help`` for flags.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Annotated, List, Tuple

import typer
from tqdm import tqdm

app = typer.Typer(
    help="Check IPC to Parquet conversion completeness",
    no_args_is_help=True,
    add_completion=False,
)


def check_missing_parquet_variants(
    source_dir: str,
    output_file: str | None = None,
) -> List[Tuple[str, str]]:
    """Reveal conversion gaps so callers can rerun only incomplete IPC inputs.

    Args:
        source_dir: Directory whose IPC inputs and Parquet outputs should be compared.
        output_file: Optional report destination; when None, only returns the list.

    Returns:
        IPC paths paired with reasons that their Parquet output is incomplete.
    """
    ipc_files = find_ipc_files(source_dir)
    missing_files = find_missing_parquet_files(ipc_files)
    if output_file is not None:
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
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for file, reason in missing_files:
                f.write(f"{file}: {reason}\n")
        typer.echo(f"Missing files have been saved to {output_file}")
    else:
        typer.echo("No missing files found.")


@app.command()
def main(
    input_dir: Annotated[
        List[Path],
        typer.Option(
            "--input-dir",
            "-i",
            help="Directory containing IPC inputs and Parquet outputs (repeatable)",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output-file",
            "-o",
            help="Output file for missing conversion paths",
        ),
    ] = Path("missing_files.txt"),
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Verify IPC to Parquet conversion completeness."""
    if verbose:
        typer.echo(f"Input dirs: {input_dir}")
        typer.echo(f"Output file: {output_file}")

    all_missing: List[Tuple[str, str]] = []
    for directory in input_dir:
        if not directory.exists():
            typer.echo(
                f"Warning: Directory '{directory}' does not exist, skipping...",
                err=True,
            )
            continue
        typer.echo(f"Checking directory: {directory}")
        all_missing.extend(check_missing_parquet_variants(str(directory)))

    save_missing_files(str(output_file), all_missing)

    if all_missing:
        typer.echo(f"Found {len(all_missing)} files with missing Parquet variants:")
        for file, reason in all_missing:
            typer.echo(f"  {file}: {reason}")
    else:
        typer.echo("All IPC files have complete Parquet variants.")


if __name__ == "__main__":
    app()
