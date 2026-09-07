"""Add acquisition column to parquet files based on search data.

This script reads an Excel search data file containing project, file path,
and acquisition type information, then adds an "acquisition" column to each
matching parquet file with the value ("DIA" or "DDA") from the search data.

USAGE:
======
# Local directory
python scripts/preprocessing/add_acquisition_column.py \
    --input-dir <data-root>/lcfm/ \
    --search-data search_data_with_new_projects.xlsx

# S3 bucket (requires AWS profile)
python scripts/preprocessing/add_acquisition_column.py \
    --input-dir s3://bucket/acfm/ \
    --search-data search_data.xlsx \
    --aws-profile <your-aws-profile>

# Dry run (preview changes without modifying files)
python scripts/preprocessing/add_acquisition_column.py \
    --input-dir <data-root>/lcfm/ \
    --search-data search_data.xlsx \
    --dry-run
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import polars as pl
import typer
from tqdm import tqdm

from scripts.preprocessing.parquet_io import search_data_lookup_key

app = typer.Typer(help="Add acquisition column to parquet files")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

INPUT_DIR_OPTION = typer.Option(
    ...,
    "--input-dir",
    "-i",
    help="Input directory containing parquet files organized by project subfolders",
)
SEARCH_DATA_OPTION = typer.Option(
    "search_data_with_new_projects.xlsx",
    "--search-data",
    "-s",
    help="Path to search data Excel file with project, file path, and acquisition columns",
)
AWS_PROFILE_OPTION = typer.Option(
    None,
    "--aws-profile",
    "-p",
    help="AWS profile name for S3 access",
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    "-n",
    help="Preview changes without modifying files",
)
VERBOSE_OPTION = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Enable verbose output",
)


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 path."""
    return path.startswith("s3://")


def setup_aws_credentials(aws_profile: Optional[str]) -> None:
    """Set up AWS credentials from profile for polars S3 access."""
    if not aws_profile:
        return

    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    aws_dir = os.path.join(repo_root, ".aws")

    if not os.path.isdir(aws_dir):
        aws_dir = os.path.expanduser("~/.aws")

    if os.path.isdir(aws_dir):
        os.environ["AWS_CONFIG_FILE"] = os.path.join(aws_dir, "config")
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = os.path.join(aws_dir, "credentials")
        logger.info(f"Using AWS config from: {aws_dir}")

    os.environ["AWS_PROFILE"] = aws_profile
    logger.info(f"Using AWS profile: {aws_profile}")


def list_s3_data_files(s3_path: str, aws_profile: Optional[str] = None) -> List[str]:
    """List data files (.parquet or .ipc) in an S3 path using AWS CLI."""
    cmd = ["aws", "s3", "ls", s3_path, "--recursive"]
    if aws_profile:
        cmd.extend(["--profile", aws_profile])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data_files = []
        path_without_scheme = s3_path[5:]
        bucket = path_without_scheme.split("/")[0]

        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []

        for line in lines:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 4:
                file_key = " ".join(parts[3:])
                if file_key.endswith(".parquet") or file_key.endswith(".ipc"):
                    data_files.append(f"s3://{bucket}/{file_key}")

        logger.info(f"Found {len(data_files)} data files in {s3_path}")
        return data_files
    except subprocess.CalledProcessError as e:
        logger.error(f"Error listing S3 path {s3_path}: {e.stderr}")
        return []


def find_data_files_in_folder(
    input_dir: str, aws_profile: Optional[str] = None
) -> List[str]:
    """Find all data files (.parquet or .ipc) in a folder (local or S3)."""
    if is_s3_path(input_dir):
        return list_s3_data_files(input_dir, aws_profile)
    else:
        if not os.path.isdir(input_dir):
            return []

        data_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".parquet") or file.endswith(".ipc"):
                    data_files.append(os.path.join(root, file))
        return data_files


def extract_file_name(path_str: str) -> str:
    """Extract a search-data lookup key from a file path.

    Strips compound proteomics extensions (``.mzml.parquet``, etc.), shard
    suffixes, and embedded ``.mzml`` so paths align with Excel ``file path``
    entries. For the value stored in the parquet ``experiment_name`` column or
    USI ``datafile``, use :func:`experiment_name_from_path` instead.
    """
    return search_data_lookup_key(path_str)


def extract_project(path: str) -> str:
    """Extract project as the immediate folder the file is inside."""
    return Path(path).parent.name


def load_acquisitions_from_search_data(
    search_data_path: str,
) -> Dict[Tuple[str, str], str]:
    """Load acquisition types from the search data Excel file.

    Returns:
        Dictionary mapping (project, filename) to acquisition type ("DIA" or "DDA")
    """
    df = pl.read_excel(search_data_path)

    columns = df.columns

    if (
        "project" not in columns
        or "acquisition" not in columns
        or "file path" not in columns
    ):
        raise ValueError(
            f"Excel file must contain 'project', 'acquisition' and 'file path' columns. "
            f"Found columns: {df.columns}"
        )

    acquisition_map: Dict[Tuple[str, str], str] = {}
    conflicts: List[Tuple[str, str, str, str]] = []

    for row in df.iter_rows(named=True):
        project = row["project"]
        file_path = row["file path"]
        acquisition = row["acquisition"]

        if project is None or file_path is None or acquisition is None:
            continue

        filename = extract_file_name(str(file_path))
        key = (str(project), filename)

        if key in acquisition_map and acquisition_map[key] != acquisition:
            conflicts.append(
                (
                    str(project),
                    filename,
                    acquisition_map[key],
                    str(acquisition),
                )
            )
        else:
            acquisition_map[key] = str(acquisition)

    if conflicts:
        conflict_details = "\n".join(
            f"  {proj}/{fname}: {prev} vs {curr}"
            for proj, fname, prev, curr in conflicts
        )
        raise ValueError(
            f"Found {len(conflicts)} conflicting acquisition assignments:\n{conflict_details}"
        )

    logger.info("No conflicting acquisition assignments found.")
    logger.info(
        f"Loaded {len(acquisition_map)} file-acquisition mappings from search data"
    )
    return acquisition_map


def _process_data_file_with_acquisition(
    file_path: str,
    project: str,
    filename: str,
    acquisition: str,
    dry_run: bool,
    verbose: bool,
) -> Literal["updated", "already_has_column", "error"]:
    """Read parquet, add acquisition column if missing; return outcome."""
    try:
        df = pl.read_parquet(file_path)

        if "acquisition" in df.columns:
            if verbose:
                existing_value = df["acquisition"][0] if len(df) > 0 else None
                logger.info(
                    f"File {project}/{filename} already has acquisition column "
                    f"(value: {existing_value})"
                )
            return "already_has_column"

        df = df.with_columns(pl.lit(acquisition).alias("acquisition"))

        if dry_run:
            if verbose:
                logger.info(
                    f"[DRY RUN] Would add acquisition={acquisition} to {project}/{filename}"
                )
        else:
            df.write_parquet(file_path)
            if verbose:
                logger.info(f"Added acquisition={acquisition} to {project}/{filename}")

        return "updated"

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return "error"


def add_acquisition_column(
    input_dir: str,
    search_data_path: str,
    aws_profile: Optional[str] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> None:
    """Add acquisition column to parquet files based on search data.

    Args:
        input_dir: Input directory containing parquet files
        search_data_path: Path to search data Excel file
        aws_profile: AWS profile for S3 access
        dry_run: If True, preview changes without modifying files
        verbose: Enable verbose output
    """
    if is_s3_path(input_dir):
        setup_aws_credentials(aws_profile)

    data_files = find_data_files_in_folder(input_dir, aws_profile)
    logger.info(f"Found {len(data_files)} data files to process")

    if not data_files:
        logger.warning("No data files found")
        return

    acquisition_map = load_acquisitions_from_search_data(search_data_path)

    updated_count = 0
    not_found_count = 0
    already_has_column_count = 0
    error_count = 0

    for file_path in tqdm(data_files, desc="Processing files"):
        filename = extract_file_name(file_path)
        project = extract_project(file_path)

        key = (project, filename)
        acquisition = acquisition_map.get(key)

        if acquisition is None:
            if verbose:
                logger.warning(f"No acquisition found for {project}/{filename}")
            not_found_count += 1
            continue

        outcome = _process_data_file_with_acquisition(
            file_path, project, filename, acquisition, dry_run, verbose
        )
        if outcome == "updated":
            updated_count += 1
        elif outcome == "already_has_column":
            already_has_column_count += 1
        elif outcome == "error":
            error_count += 1

    logger.info("Summary:")
    logger.info(f"  Updated: {updated_count}")
    logger.info(f"  Already has column: {already_has_column_count}")
    logger.info(f"  Not in search data: {not_found_count}")
    logger.info(f"  Errors: {error_count}")

    if dry_run:
        logger.info("  (DRY RUN - no files were actually modified)")


@app.command()
def main(
    input_dir: str = INPUT_DIR_OPTION,
    search_data: str = SEARCH_DATA_OPTION,
    aws_profile: Optional[str] = AWS_PROFILE_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Add acquisition column to parquet files based on search data."""
    add_acquisition_column(
        input_dir=input_dir,
        search_data_path=search_data,
        aws_profile=aws_profile,
        dry_run=dry_run,
        verbose=verbose,
    )


if __name__ == "__main__":
    app()
