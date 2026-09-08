"""Attach DIA or DDA acquisition metadata to converted Parquet files.

Run this when converted files lack the acquisition metadata required downstream.
It matches project and normalised filenames against an Excel search-data table
and supports local trees or S3 listings.

CLI::

    python scripts/preprocessing/add_acquisition_column.py --help
    python scripts/preprocessing/add_acquisition_column.py --input-dir <data-root>/lcfm/
    python scripts/preprocessing/add_acquisition_column.py --input-dir <data-root>/lcfm/ --search-data data/search_data.xlsx
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Tuple

import polars as pl
import typer
from tqdm import tqdm

from scripts.logging_setup import configure_script_logging
from scripts.paths import DEFAULT_SEARCH_DATA
from scripts.preprocessing.parquet_io import search_data_lookup_key

logger = logging.getLogger(__name__)

app = typer.Typer(
    help="Add acquisition column to parquet files",
    no_args_is_help=True,
    add_completion=False,
)


def is_s3_path(path: str) -> bool:
    """Let callers select remote handling without attempting filesystem access.

    Args:
        path: Input location to classify.

    Returns:
        Whether the location uses the S3 URI scheme.
    """
    return path.startswith("s3://")


def setup_aws_credentials(aws_profile: Optional[str]) -> None:
    """Expose the requested AWS profile so Polars and the AWS CLI agree.

    Args:
        aws_profile: Profile name to activate, or null for ambient credentials.
    """
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
    """Discover remote data files without downloading the dataset.

    Args:
        s3_path: Bucket prefix to search recursively.
        aws_profile: Optional AWS profile used for the listing.

    Returns:
        S3 URIs for Parquet and IPC objects found below the prefix.
    """
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
    """Give acquisition enrichment one uniform file list for local or S3 input.

    Args:
        input_dir: Local directory or S3 prefix to inspect.
        aws_profile: Optional AWS profile for S3 access.

    Returns:
        Paths to Parquet and IPC files below the input location.
    """
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
    """Normalise stored filenames so they match search-data rows reliably.

    Strips compound proteomics extensions (``.mzml.parquet``, etc.), shard
    suffixes, and embedded ``.mzml``.

    Args:
        path_str: Data path whose experiment key is needed.

    Returns:
        Canonical key used by the search-data workbook.
    """
    return search_data_lookup_key(path_str)


def extract_project(path: str) -> str:
    """Associate a file with the project dimension used in metadata lookups.

    Args:
        path: Data file path organised beneath its project folder.

    Returns:
        Immediate parent folder name.
    """
    return Path(path).parent.name


def load_acquisitions_from_search_data(
    search_data_path: str,
) -> Dict[Tuple[str, str], str]:
    """Build an unambiguous lookup before any Parquet files are modified.

    Args:
        search_data_path: Excel workbook containing project, raw-filename
            ``file path``, and acquisition.

    Returns:
        Project and filename keys mapped to acquisition type.

    Raises:
        ValueError: If required columns are absent or assignments conflict.
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
    """Keep per-file failures from aborting acquisition enrichment for the dataset."""
    try:
        df = pl.read_parquet(file_path)

        if "acquisition" in df.columns:
            if verbose:
                existing_value = df["acquisition"][0] if len(df) > 0 else None
                logger.debug(
                    f"File {project}/{filename} already has acquisition column "
                    f"(value: {existing_value})"
                )
            return "already_has_column"

        df = df.with_columns(pl.lit(acquisition).alias("acquisition"))

        if dry_run:
            if verbose:
                logger.debug(
                    f"[DRY RUN] Would add acquisition={acquisition} to {project}/{filename}"
                )
        else:
            df.write_parquet(file_path)
            if verbose:
                logger.debug(f"Added acquisition={acquisition} to {project}/{filename}")

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
    """Fill required acquisition metadata from the authoritative search-data table.

    Args:
        input_dir: Local directory or S3 prefix containing data files.
        search_data_path: Excel workbook with acquisition assignments.
        aws_profile: Optional AWS profile for S3 access.
        dry_run: Whether to preview without modifying files.
        verbose: Whether to print per-file details.
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
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            "-i",
            help="Local directory or S3 prefix with project subfolders",
        ),
    ],
    search_data: Annotated[
        Path,
        typer.Option(
            "--search-data",
            help=(
                "Search-data Excel with project, raw-filename file path, "
                "and acquisition columns"
            ),
        ),
    ] = DEFAULT_SEARCH_DATA,
    aws_profile: Annotated[
        Optional[str],
        typer.Option("--aws-profile", help="AWS profile name for S3 access"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", "-n", help="Preview without modifying files"),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose output"),
    ] = False,
) -> None:
    """Enrich converted files with acquisition type."""
    configure_script_logging(verbose=verbose)

    add_acquisition_column(
        input_dir=str(input_dir),
        search_data_path=str(search_data),
        aws_profile=aws_profile,
        dry_run=dry_run,
        verbose=verbose,
    )


if __name__ == "__main__":
    app()
