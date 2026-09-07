import polars as pl
import os
import re
from tqdm import tqdm
from typing import Union, List
import logging
import glob
from pathlib import Path
import typer

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = typer.Typer(help="Find modifications in parquet files")

# Module-level constants to avoid B008 errors
INPUT_DIR_ARG = typer.Argument(..., help="Input directory to process")
OUTPUT_FILE_OPTION = typer.Option(
    "modifications.xlsx", "--output", "-o", help="Output file for modifications"
)
FILE_PATTERN_OPTION = typer.Option(
    "**/*.parquet", "--pattern", "-p", help="File pattern to match"
)
VERBOSE_OPTION = typer.Option(False, "--verbose", "-v", help="Enable verbose output")
INPUT_DIRS_ARG = typer.Argument(..., help="Input directories to process")
OUTPUT_DIR_OPTION = typer.Option(
    "output_files", "--output-dir", "-o", help="Output directory for results"
)
PREFIX_OPTION = typer.Option(
    "modifications", "--prefix", "-p", help="Prefix for output files"
)


def find_files(input_dir: str, file_pattern: str) -> List[str]:
    """Find files in a local directory that match a specified pattern."""
    search_pattern = Path(input_dir) / file_pattern
    matched_files = glob.glob(str(search_pattern), recursive=True)
    return matched_files


def extract_modifications(
    file: str, mod_pattern: re.Pattern
) -> Union[pl.DataFrame, None]:
    """Extract unique modifications from a single file's DataFrame."""
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
    """Find modifications in parquet files.

    Args:
        input_dir (str): Input directory to process
        output_path (str): Output file path
        file_pattern (str): File pattern to match
        verbose (bool): Enable verbose output
    """
    if verbose:
        typer.echo(f"Processing directory: {input_dir}")
        typer.echo(f"File pattern: {file_pattern}")
        typer.echo(f"Output file: {output_path}")

    # Find files in the local filesystem
    matched_files = find_files(
        input_dir=input_dir,
        file_pattern=file_pattern,
    )

    if verbose:
        typer.echo(f"Found {len(matched_files)} files to process")

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
            logger.info(f"Processing file: {file}")
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
    typer.echo(f"Modifications saved to {output_path}")


@app.command()
def find_mods(
    input_dir: str = INPUT_DIR_ARG,
    output_file: str = OUTPUT_FILE_OPTION,
    file_pattern: str = FILE_PATTERN_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Find modifications in parquet files."""
    find_modifications(
        input_dir=input_dir,
        output_path=output_file,
        file_pattern=file_pattern,
        verbose=verbose,
    )


@app.command()
def batch_find_mods(
    input_dirs: List[str] = INPUT_DIRS_ARG,
    output_dir: str = OUTPUT_DIR_OPTION,
    file_pattern: str = FILE_PATTERN_OPTION,
    prefix: str = PREFIX_OPTION,
) -> None:
    """Find modifications in multiple directories."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for input_dir in input_dirs:
        output_file = output_path / f"{prefix}_{Path(input_dir).name}.xlsx"
        typer.echo(f"Processing directory: {input_dir}")
        find_modifications(
            input_dir=input_dir,
            output_path=str(output_file),
            file_pattern=file_pattern,
        )


def main() -> None:
    """Entry point for collecting unique modifications across globbed files."""
    # Legacy behavior for backward compatibility
    input_dir = "<data-root>/lcfm"  # Local mounted filesystem path
    output_path = "output_files/modifications.xlsx"

    find_modifications(input_dir, output_path)


if __name__ == "__main__":
    app()
