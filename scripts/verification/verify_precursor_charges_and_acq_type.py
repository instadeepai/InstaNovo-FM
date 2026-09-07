r"""Verify precursor charge values for DIA and DDA files.

PURPOSE:
========
This script checks the precursor_charge values for files with acquisition
type "DIA" from the search data Excel file. DIA data typically has precursor
charge values of 0 (unknown) since the acquisition method does not isolate
individual precursors.

This script reports any DIA-marked projects that have a non-zero precursor
charge, and any DDA-marked projects that have a zero precursor charge.

Supports both local directories and S3 buckets as input.

USAGE:
======
# Local directory
python scripts/verification/verify_precursor_charge_and_acq_type.py \
    --input-dir <data-root>/lcfm/ \
    --search-data search_data_with_new_projects.xlsx \
    --output-dir lcfm

# S3 bucket
python scripts/verification/verify_precursor_charge_and_acq_type.py \
    --input-dir s3://<your-bucket>/acfm/ \
    --search-data search_data_with_new_projects.xlsx \
    --output-dir acfm \
    --aws-profile <your-aws-profile>
"""

import polars as pl
import os
import logging
from tqdm import tqdm
import subprocess
from pathlib import Path
from datetime import timedelta
from dataclasses import dataclass
from typing import List, Optional, Tuple
import typer

app = typer.Typer()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Module-level constants for CLI options
INPUT_DIR_OPTION = typer.Option(
    "<data-root>/lcfm/",
    "--input-dir",
    "-i",
    help="Input directory containing parquet files organized by project subfolders",
)
SEARCH_DATA_OPTION = typer.Option(
    "search_data_with_new_projects.xlsx",
    "--search-data",
    "-s",
    help="Path to search data Excel file with project and acquisition columns",
)
OUTPUT_DIR_OPTION = typer.Option(
    ...,
    "--output-dir",
    "-o",
    help="Directory to write the output CSV reports to (one for each acquisition type)",
)
AWS_PROFILE_OPTION = typer.Option(
    None,
    "--aws-profile",
    "-p",
    help="AWS profile name for S3 access (read from ~/.aws/)",
)


def is_s3_path(path: str) -> bool:
    """Check if a path is an S3 path."""
    return path.startswith("s3://")


def setup_aws_credentials(aws_profile: Optional[str]) -> None:
    """Set up AWS credentials from profile for polars S3 access."""
    if not aws_profile:
        return

    # Credentials come from the user's own AWS config only. The original looked
    # for .aws/ inside the checkout first, which invites committing secrets.
    aws_dir = os.path.expanduser("~/.aws")

    if os.path.isdir(aws_dir):
        os.environ["AWS_CONFIG_FILE"] = os.path.join(aws_dir, "config")
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = os.path.join(aws_dir, "credentials")
        logger.info(f"Using AWS config from: {aws_dir}")

    os.environ["AWS_PROFILE"] = aws_profile
    logger.info(f"Using AWS profile: {aws_profile}")


def format_time(seconds: float) -> str:
    """Format seconds into a human-readable time string."""
    return str(timedelta(seconds=int(seconds)))


def list_s3_projects(s3_path: str, aws_profile: Optional[str] = None) -> List[str]:
    """List project folders in an S3 bucket path using AWS CLI."""
    cmd = ["aws", "s3", "ls", s3_path]
    if aws_profile:
        cmd.extend(["--profile", aws_profile])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        projects = []
        for line in result.stdout.strip().split("\n"):
            if line.strip() and "PRE " in line:
                # Parse "PRE PXD000561/" format
                folder = line.strip().replace("PRE ", "").rstrip("/")
                if folder:
                    projects.append(folder)
        return sorted(projects)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error listing S3 path {s3_path}: {e.stderr}")
        return []


def list_s3_data_files(s3_path: str, aws_profile: Optional[str] = None) -> List[str]:
    """List data files (.parquet or .ipc) in an S3 path using AWS CLI."""
    cmd = ["aws", "s3", "ls", s3_path, "--recursive"]
    if aws_profile:
        cmd.extend(["--profile", aws_profile])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data_files = []
        # Parse S3 bucket and prefix from path
        # s3://bucket/prefix/ -> bucket, prefix
        path_without_scheme = s3_path[5:]  # Remove "s3://"
        bucket = path_without_scheme.split("/")[0]

        lines = result.stdout.strip().split("\n") if result.stdout.strip() else []
        logger.debug(f"S3 ls {s3_path}: {len(lines)} lines of output")

        for line in lines:
            if not line.strip():
                continue
            # Support both .parquet and .ipc (Arrow IPC) formats
            if line.endswith(".parquet") or line.endswith(".ipc"):
                # Parse "2024-01-01 12:00:00 123456 prefix/file.parquet" format
                parts = line.split()
                if len(parts) >= 4:
                    key = parts[-1]
                    data_files.append(f"s3://{bucket}/{key}")

        if not data_files and lines:
            # Log sample of what we found if no data files
            sample = lines[:3]
            logger.debug(f"No data files found. Sample output: {sample}")

        return data_files
    except subprocess.CalledProcessError as e:
        logger.error(f"Error listing S3 data files in {s3_path}: {e.stderr}")
        return []


def find_data_files_in_folder(
    input_dir: str, aws_profile: Optional[str] = None
) -> List[str]:
    """Find all data files (.parquet or .ipc) in a folder (local or S3)."""
    if is_s3_path(input_dir):
        # S3 path
        return list_s3_data_files(input_dir, aws_profile)
    else:
        # Local path
        if not os.path.isdir(input_dir):
            return []

        data_files = []
        for root, _, files in os.walk(input_dir):
            for file in files:
                # Support both .parquet and .ipc (Arrow IPC) formats
                if file.endswith(".parquet"):
                    data_files.append(os.path.join(root, file))
        return data_files


def _find_aws_dir() -> str:
    """Find the AWS config directory (repo root or home)."""
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    aws_dir = os.path.join(repo_root, ".aws")
    if os.path.isdir(aws_dir):
        return aws_dir
    return os.path.expanduser("~/.aws")


def _read_aws_credentials(aws_dir: str, profile: str, storage_opts: dict) -> None:
    """Read AWS credentials from credentials file into storage_opts."""
    import configparser

    creds_file = os.path.join(aws_dir, "credentials")
    if not os.path.exists(creds_file):
        return

    creds = configparser.ConfigParser()
    creds.read(creds_file)
    if profile not in creds:
        return

    if "aws_access_key_id" in creds[profile]:
        storage_opts["aws_access_key_id"] = creds[profile]["aws_access_key_id"]
    if "aws_secret_access_key" in creds[profile]:
        storage_opts["aws_secret_access_key"] = creds[profile]["aws_secret_access_key"]


def _read_aws_config(aws_dir: str, profile: str, storage_opts: dict) -> None:
    """Read AWS region from config file into storage_opts."""
    import configparser

    config_file = os.path.join(aws_dir, "config")
    if not os.path.exists(config_file):
        return

    config = configparser.ConfigParser()
    config.read(config_file)
    profile_section = f"profile {profile}"
    if profile_section in config and "region" in config[profile_section]:
        storage_opts["aws_region"] = config[profile_section]["region"]


def get_storage_options(aws_profile: Optional[str] = None) -> Optional[dict]:
    """Build storage options for Polars S3 access."""
    if not aws_profile:
        return None

    aws_dir = _find_aws_dir()
    storage_opts: dict = {}

    _read_aws_credentials(aws_dir, aws_profile, storage_opts)
    _read_aws_config(aws_dir, aws_profile, storage_opts)

    return storage_opts if storage_opts else None


def _read_file_lazy(
    file_path: str, storage_options: Optional[dict] = None
) -> pl.LazyFrame:
    """Read a file as a LazyFrame, supporting both parquet and IPC formats."""
    if file_path.endswith(".ipc"):
        return pl.scan_ipc(file_path, storage_options=storage_options)
    else:
        return pl.scan_parquet(file_path, storage_options=storage_options)


def extract_file_name(path_str: str) -> str:
    """Extract a search-data lookup key from a file path."""
    from scripts.preprocessing.parquet_io import search_data_lookup_key

    return str(search_data_lookup_key(path_str))


def extract_project(path: str) -> str:
    """Extract project as the immediate folder the file is inside."""
    return Path(path).parent.name


def check_conflicting_acquistions(dia_df: pl.DataFrame, dda_df: pl.DataFrame) -> None:
    """Check each unique combination of 'project' and 'filename' is only listed once on one of the dataframes.

    If there are duplicate rows, raise a ValueError.
    """
    # Find the intersection
    # We use join to find rows that exist in both
    overlaps = (
        dia_df.select(["project", "filename"])
        .join(
            dda_df.select(["project", "filename"]),
            on=["project", "filename"],
            how="inner",
        )
        .unique()
    )

    if not overlaps.is_empty():
        # Raise an error with a helpful message showing the duplicates
        raise ValueError(
            f"Found {len(overlaps)} duplicate project/filename combinations: \n{overlaps}"
        )

    logger.info("No conflicting acquisition assignments found. Proceeding...")


def load_aquisitions_from_search_data(
    search_data_path: str,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """Load files with acquisition types 'DIA' and 'DDA' from the search data Excel file."""
    df = pl.read_excel(search_data_path)

    columns = df.columns

    # Ensure required columns exist
    if (
        "project" not in columns
        or "acquisition" not in columns
        or "file path" not in columns
    ):
        raise ValueError(
            f"Excel file must contain 'project', 'acquisition' and 'file path' columns. "
            f"Found columns: {df.columns}"
        )

    # Get DIA projects
    dia_projects = (
        df.filter(pl.col("acquisition") == "DIA")
        .select(["project", "file path", "acquisition"])
        .with_columns(
            pl.col("file path")
            .map_elements(extract_file_name, return_dtype=pl.String)
            .alias("filename")
        )
        .drop("file path")
        .unique()
    )

    logger.info(f"Found {len(dia_projects)} DIA files in search data")

    # Get DDA projects
    dda_projects = (
        df.filter(pl.col("acquisition") == "DDA")
        .select(["project", "file path", "acquisition"])
        .with_columns(
            pl.col("file path")
            .map_elements(extract_file_name, return_dtype=pl.String)
            .alias("filename")
        )
        .drop("file path")
        .unique()
    )

    logger.info(f"Found {len(dda_projects)} DIA files in search data")

    check_conflicting_acquistions(dia_df=dia_projects, dda_df=dda_projects)

    return dia_projects, dda_projects


@dataclass
class FilePrecursorChargeErrors:
    """Precursor charge error statistics for a file."""

    filename: str
    project: str
    error_type: str
    num_error_rows: int
    total_rows: int


def _check_dda_file_precursor_charges(
    filename: str,
    project: str,
    precursor_charge_col: pl.DataFrame,
    total_rows: int,
) -> List[FilePrecursorChargeErrors]:
    """Check one DDA file for zero/unknown/null precursor charges. Return list of errors."""
    errors: List[FilePrecursorChargeErrors] = []
    dtype = precursor_charge_col["precursor_charge"].dtype

    if dtype == pl.Int64:
        num_zero_rows = len(
            precursor_charge_col.filter(pl.col("precursor_charge") == 0)
        )
        if num_zero_rows > 0:
            logger.info(
                f"DDA file {filename} in project {project} has {num_zero_rows} zero precursor charges"
            )
            errors.append(
                FilePrecursorChargeErrors(
                    filename,
                    project,
                    "zero_precursor_charge",
                    num_zero_rows,
                    total_rows,
                )
            )
    elif dtype == pl.String:
        num_unknown_rows = len(
            precursor_charge_col.filter(
                pl.col("precursor_charge").str.to_lowercase() == "unknown"
            )
        )
        if num_unknown_rows > 0:
            logger.info(
                f"DDA file {filename} in project {project} has {num_unknown_rows} 'unknown' precursor charges"
            )
            errors.append(
                FilePrecursorChargeErrors(
                    filename,
                    project,
                    "unknown_precursor_charge",
                    num_unknown_rows,
                    total_rows,
                )
            )
    else:
        raise ValueError(
            f"Column 'precursor_charge' is dtype {dtype} for file {filename} in project {project}"
        )

    num_null_rows = len(
        precursor_charge_col.filter(pl.col("precursor_charge").is_null())
    )
    if num_null_rows > 0:
        logger.info(
            f"DDA file {filename} in project {project} has {num_null_rows} null precursor charges"
        )
        errors.append(
            FilePrecursorChargeErrors(
                filename, project, "null_precursor_charge", num_null_rows, total_rows
            )
        )
    return errors


def _check_dia_file_precursor_charges(
    filename: str,
    project: str,
    precursor_charge_col: pl.DataFrame,
    total_rows: int,
) -> List[FilePrecursorChargeErrors]:
    """Check one DIA file for non-zero/unknown/null precursor charges. Return list of errors."""
    errors: List[FilePrecursorChargeErrors] = []
    dtype = precursor_charge_col["precursor_charge"].dtype

    if dtype == pl.Int64:
        num_non_zero_rows = len(
            precursor_charge_col.filter(pl.col("precursor_charge") > 0)
        )
        if num_non_zero_rows > 0:
            logger.info(
                f"DIA file {filename} in project {project} has {num_non_zero_rows} non-zero precursor charges"
            )
            errors.append(
                FilePrecursorChargeErrors(
                    filename,
                    project,
                    "non_zero_precursor_charge",
                    num_non_zero_rows,
                    total_rows,
                )
            )
    elif dtype == pl.String:
        num_unknown_rows = len(
            precursor_charge_col.filter(
                pl.col("precursor_charge").str.to_lowercase() == "unknown"
            )
        )
        if num_unknown_rows > 0:
            logger.info(
                f"DIA file {filename} in project {project} has {num_unknown_rows} 'unknown' precursor charges"
            )
            errors.append(
                FilePrecursorChargeErrors(
                    filename,
                    project,
                    "unknown_precursor_charge",
                    num_unknown_rows,
                    total_rows,
                )
            )

    num_null_rows = len(
        precursor_charge_col.filter(pl.col("precursor_charge").is_null())
    )
    if num_null_rows > 0:
        logger.info(
            f"DIA file {filename} in project {project} has {num_null_rows} null precursor charges"
        )
        errors.append(
            FilePrecursorChargeErrors(
                filename, project, "null_precursor_charge", num_null_rows, total_rows
            )
        )
    return errors


def check_if_all_files_in_project_have_errors(
    data_files: List[str],
    incorrect_dia_files: pl.DataFrame,
    incorrect_dda_files: pl.DataFrame,
    search_data_files: pl.DataFrame,  # The concat'd df of project/filename/acquisition
) -> pl.DataFrame:
    """Check if all files in a project have errors."""
    # Create a master list of expected files
    master_files = pl.DataFrame({"filepath": data_files}).with_columns(
        [
            pl.col("filepath")
            .map_elements(extract_file_name, return_dtype=pl.String)
            .alias("filename"),
            pl.col("filepath")
            .map_elements(extract_project, return_dtype=pl.String)
            .alias("project"),
        ]
    )

    # Join with search_data_files to get the 'acquisition' (DIA/DDA) for every file
    master_files = master_files.join(
        search_data_files.select(["project", "filename", "acquisition"]),
        on=["project", "filename"],
        how="left",
    )

    error_df = pl.concat([incorrect_dia_files, incorrect_dda_files])

    # Early return if no errors found
    if error_df.is_empty():
        logger.info("No precursor charge errors found in any files.")
        return pl.DataFrame(
            schema={
                "project": pl.String,
                "acquisition": pl.String,
                "error_type": pl.String,
                "files_with_this_error": pl.UInt32,
                "files_with_100_percent_error": pl.UInt32,
                "total_files_in_project_acq": pl.UInt32,
                "all_files_in_project_affected": pl.Boolean,
            }
        )

    # 1. First, get the denominator: How many files exist per Project + Acquisition?
    project_denominators = master_files.group_by(["project", "acquisition"]).agg(
        pl.len().alias("total_files_in_project_acq")
    )

    # 2. Join errors with the master list to ensure we know the acquisition for each error
    # (Using the combined error_df from previous step)
    error_registry = error_df.join(
        master_files.select(["project", "filename", "acquisition"]),
        on=["project", "filename"],
        how="left",
    )

    # 3. Calculate Error Statistics per Project/Acquisition/ErrorType
    error_stats = error_registry.group_by(["project", "acquisition", "error_type"]).agg(
        [
            # How many files had THIS specific error?
            pl.len().alias("files_with_this_error"),
            # How many files are 100% corrupted by THIS error?
            (pl.col("num_error_rows") == pl.col("total_rows"))
            .sum()
            .alias("files_with_100_percent_error"),
        ]
    )

    # 4. Join the stats back to the denominators to get the true percentage
    project_summary = error_stats.join(
        project_denominators, on=["project", "acquisition"], how="left"
    ).with_columns(
        # Now the denominator is the TOTAL files in that project/acq,
        # not just the ones that had errors.
        (pl.col("files_with_this_error") == pl.col("total_files_in_project_acq")).alias(
            "all_files_in_project_affected"
        )
    )

    # Filter out cases where no errors occurred at all
    return project_summary


def analyze_precursor_charges(
    input_dir: str, search_data_path: str, aws_profile: Optional[str] = None
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Analyze precursor charge values for a project.

    Check if any files marked as 'DDA' contain zero or empty precursor charges, and
    check if any files marked as 'DIA' contain non-zero precursor charges.
    """
    data_files = find_data_files_in_folder(input_dir, aws_profile)

    logger.debug(f"Found {len(data_files)} files in {input_dir}")

    if not data_files:
        raise ValueError(f"No files found in {input_dir}")

    # Build storage options for S3 access
    storage_options = (
        get_storage_options(aws_profile) if is_s3_path(input_dir) else None
    )

    incorrect_dda_files = []
    incorrect_dia_files = []

    dia_files, dda_files = load_aquisitions_from_search_data(search_data_path)

    # Single df of "project", "filename" and "acquisition"
    search_data_files = pl.concat([dia_files, dda_files], how="vertical")

    for file_path in tqdm(data_files, desc="file"):
        filename = extract_file_name(file_path)
        project = extract_project(file_path)

        acquisition = search_data_files.filter(
            (pl.col("project") == project) & (pl.col("filename") == filename)
        )["acquisition"][0]

        precursor_charge_col = (
            _read_file_lazy(file_path, storage_options)
            .select("precursor_charge")
            .collect()
        )
        total_rows = len(precursor_charge_col)

        if acquisition == "DDA":
            incorrect_dda_files.extend(
                _check_dda_file_precursor_charges(
                    filename, project, precursor_charge_col, total_rows
                )
            )
        else:
            incorrect_dia_files.extend(
                _check_dia_file_precursor_charges(
                    filename, project, precursor_charge_col, total_rows
                )
            )

    incorrect_dia_df = pl.DataFrame(incorrect_dia_files)
    incorrect_dda_df = pl.DataFrame(incorrect_dda_files)

    project_level_summary = check_if_all_files_in_project_have_errors(
        data_files, incorrect_dia_df, incorrect_dda_df, search_data_files
    ).sort(by=["acquisition", "project"])

    return incorrect_dia_df, incorrect_dda_df, project_level_summary


def run_verification(
    input_dir: str, search_data_path: str, output_dir: str, aws_profile: Optional[str]
) -> None:
    """Verify precursor charge values for DIA and DDA files."""
    # Set up AWS credentials if using S3
    if is_s3_path(input_dir):
        setup_aws_credentials(aws_profile)

    incorrect_dia_files, incorrect_dda_files, project_level_summary = (
        analyze_precursor_charges(input_dir, search_data_path, aws_profile)
    )

    has_incorrect_dia_files = len(incorrect_dia_files) > 0
    has_incorrect_dda_files = len(incorrect_dda_files) > 0
    output_path = Path(output_dir)

    if has_incorrect_dia_files or has_incorrect_dda_files:
        output_path.mkdir(parents=True, exist_ok=True)

    if has_incorrect_dia_files:
        logger.info(
            f"Found incorrect DIA files. Saving outputs to {output_dir}/incorrect_dia_files.csv"
        )
        incorrect_dia_files.write_csv(output_path / "incorrect_dia_files.csv")

    if has_incorrect_dda_files:
        logger.info(
            f"Found incorrect DDA files. Saving outputs to {output_dir}/incorrect_dda_files.csv"
        )
        incorrect_dda_files.write_csv(output_path / "incorrect_dda_files.csv")

    if has_incorrect_dia_files or has_incorrect_dda_files:
        logger.info(
            f"Saving project-level statistics to {output_dir}/project_summary.csv"
        )
        project_level_summary.write_csv(output_path / "project_summary.csv")


@app.command()
def main(
    input_dir: str = INPUT_DIR_OPTION,
    search_data: str = SEARCH_DATA_OPTION,
    output_dir: str = OUTPUT_DIR_OPTION,
    aws_profile: Optional[str] = AWS_PROFILE_OPTION,
) -> None:
    """Verify precursor charge values for DIA projects."""
    run_verification(
        input_dir=input_dir,
        search_data_path=search_data,
        output_dir=output_dir,
        aws_profile=aws_profile,
    )


if __name__ == "__main__":
    app()
