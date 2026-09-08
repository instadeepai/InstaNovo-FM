"""Inventory unique EncyclopeDIA modification labels in Parquet datasets.

Run this before validating or translating modification mappings so observed
labels and representative spectrum metadata are available in an Excel report.

CLI::

    python scripts/preprocessing/find_modifications.py --help
    python scripts/preprocessing/find_modifications.py --input-dir <data-root>/lcfm --output-file modifications.xlsx
    python scripts/preprocessing/find_modifications.py --input-dir <data-root>/lcfm --input-dir <data-root>/hcfm --output-file modifications.xlsx

Use ``python script.py --help`` for flags.
"""

from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path
from typing import Annotated, List, Union

import polars as pl
import typer
from tqdm import tqdm

from scripts.logging_setup import configure_script_logging

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Find modifications in parquet files",
    no_args_is_help=True,
    add_completion=False,
)


def find_files(input_dir: str, file_pattern: str) -> List[str]:
    """Provide the Parquet files whose modification labels should be inventoried.

    Args:
        input_dir: Root directory for the search.
        file_pattern: Recursive glob selecting files.

    Returns:
        Paths matching the requested pattern.
    """
    search_pattern = Path(input_dir) / file_pattern
    matched_files = glob.glob(str(search_pattern), recursive=True)
    return matched_files


def extract_modifications(
    file: str, mod_pattern: re.Pattern
) -> Union[pl.DataFrame, None]:
    """Retain one evidence row per observed label for mapping review.

    Args:
        file: Parquet file containing modified peptide annotations.
        mod_pattern: Pattern that extracts supported modification forms.

    Returns:
        Unique labels with source metadata, or null when none are present.
    """
    # Extract filename and parent subfolder
    file_name = os.path.basename(file)
    project_name = os.path.basename(os.path.dirname(file))

    # Read the required columns
    lf = (
        pl.scan_parquet(file)
        .select(["modified_peptide", "scan", "header"])
        .drop_nulls(subset=["modified_peptide"])
    )

    # Skip empty files
    if lf.first().collect().is_empty():
        return None

    # Filter rows where `modified_peptide` contains square brackets
    df = lf.filter(pl.col("modified_peptide").str.contains(r"\[|\]")).collect()

    # Skip files with no modifications
    if df.is_empty():
        return None

    current_file_modifications = []

    # Iterate over filtered rows
    for idx, row in enumerate(df.iter_rows(named=True)):
        scan = row["scan"]
        header = row["header"]
        modified_peptide = row["modified_peptide"]

        # Extract modifications inside square brackets
        mods = mod_pattern.findall(modified_peptide)

        for mod in mods:
            current_file_modifications.append(
                {
                    "modification": mod,
                    "modified_peptide": modified_peptide,
                    "project_name": project_name,
                    "file_name": file_name,
                    "source_file_index": idx,
                    "scan": scan,
                    "header": header,
                }
            )

    # Retain only the unique modifications
    current_modifications_df = pl.DataFrame(current_file_modifications).unique(
        subset="modification"
    )

    return current_modifications_df


def find_modifications(
    input_dir: str,
    output_path: str,
    file_pattern: str = "**/*.parquet",
    verbose: bool = False,
) -> None:
    """Create the modification inventory needed to audit translation mappings.

    Args:
        input_dir: Dataset directory to inspect.
        output_path: Excel destination for unique labels and evidence.
        file_pattern: Glob selecting Parquet files.
        verbose: Whether to print scan details.
    """
    logger.debug(f"Processing directory: {input_dir}")
    logger.debug(f"File pattern: {file_pattern}")
    logger.debug(f"Output file: {output_path}")

    # Find files in the local filesystem
    matched_files = find_files(
        input_dir=input_dir,
        file_pattern=file_pattern,
    )

    logger.debug(f"Found {len(matched_files)} files to process")

    # Regex pattern to find modifications in three cases:
    # 1. Uppercase letter followed by modification: A[123]
    # 2. Lowercase letter followed by modification and uppercase letter (N-terminal): n[123]A
    # 3. Uppercase letter (optionally modified) followed by c-terminal modification: Ac[123] or K[170]c[123]
    mod_pattern = re.compile(
        r"(?:"
        r"[A-Z](?:\[[0-9]+\])+|"  # Case 1: residue modifications
        r"[a-z](?:\[[0-9]+\])+[A-Z]|"  # Case 2: N-terminal modifications
        r"[A-Z](?:\[[0-9]+\])*c(?:\[[0-9]+\])+"  # Case 3: C-terminal modifications
        r")"
    )

    # List to store extracted modifications and their metadata
    global_modifications = pl.DataFrame(
        {
            "modification": [],
            "modified_peptide": [],
            "project_name": [],
            "file_name": [],
            "source_file_index": [],
            "scan": [],
            "header": [],
        },
        schema={
            "modification": pl.String,
            "modified_peptide": pl.String,
            "project_name": pl.String,
            "file_name": pl.String,
            "source_file_index": pl.Int64,
            "scan": pl.String,
            "header": pl.String,
        },
    )

    # Process each file
    for file in tqdm(matched_files, unit="file"):
        if verbose:
            logger.debug(f"Processing file: {file}")
        current_modifications_df = extract_modifications(file, mod_pattern)

        # Skip the current file if the returned df is empty
        if current_modifications_df is None:
            continue

        # Merge into global modifications
        global_modifications = pl.concat(
            [global_modifications, current_modifications_df]
        ).unique(subset="modification")

    # Extract the modification number and sort
    global_modifications = (
        global_modifications.with_columns(
            pl.col("modification")
            .str.extract(r"\[(\d+)\]")
            .cast(pl.Int64)
            .alias("mod_number"),
            pl.col("modification").str.extract(r"^([A-Za-z])").alias("mod_letter"),
        )
        .sort(["mod_number", "mod_letter"])
        .drop(["mod_letter"])
    )

    # Ensure output directory exists
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    global_modifications.write_excel(output_path)
    logger.info(f"Modifications saved to {output_path}")


@app.command()
def main(
    input_dir: Annotated[
        List[Path],
        typer.Option(
            "--input-dir",
            "-i",
            help="Dataset directory to inspect (repeatable)",
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            "--output-file",
            "-o",
            help="Excel destination for the modification inventory",
        ),
    ] = Path("modifications.xlsx"),
    pattern: Annotated[
        str,
        typer.Option("--pattern", help="File pattern to match"),
    ] = "**/*.parquet",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Inventory observed modification labels in parquet datasets."""
    configure_script_logging(verbose=verbose)

    # Multiple trees: run find_modifications once per tree into temp frames via
    # sequential calls that overwrite; better to concatenate by calling once on
    # a combined walk. For simplicity, process dirs sequentially and merge by
    # writing the last combined pass — call find_modifications for each and
    # concat Excel is awkward; process all files through one path.
    if len(input_dir) == 1:
        find_modifications(
            input_dir=str(input_dir[0]),
            output_path=str(output_file),
            file_pattern=pattern,
            verbose=verbose,
        )
        return

    # Multiple dirs: inventary each into memory by temporarily writing then
    # merging Excel sheets is heavy; instead concatenate by scanning all dirs
    # through repeated find_modifications into a shared unique set via Excel
    # rewrite. Use a temp approach: collect via find_modifications helpers.
    frames: list[pl.DataFrame] = []
    for directory in input_dir:
        if not directory.exists():
            logger.warning(f"Directory '{directory}' does not exist, skipping...")
            continue
        logger.info(f"Processing directory: {directory}")
        tmp = output_file.with_name(f".tmp_{directory.name}_{output_file.name}")
        find_modifications(
            input_dir=str(directory),
            output_path=str(tmp),
            file_pattern=pattern,
            verbose=verbose,
        )
        if tmp.exists():
            frames.append(pl.read_excel(tmp))
            tmp.unlink()

    if not frames:
        logger.info("No modifications found.")
        return

    merged = pl.concat(frames).unique(subset="modification")
    if "mod_number" not in merged.columns:
        merged = (
            merged.with_columns(
                pl.col("modification")
                .str.extract(r"\[(\d+)\]")
                .cast(pl.Int64)
                .alias("mod_number"),
                pl.col("modification").str.extract(r"^([A-Za-z])").alias("mod_letter"),
            )
            .sort(["mod_number", "mod_letter"])
            .drop(["mod_letter"])
        )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.write_excel(output_file)
    logger.info(f"Modifications saved to {output_file}")


if __name__ == "__main__":
    app()
