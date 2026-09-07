r"""Restore quality-filtered spectra missing from ACFM parquet files.

Compares each parquet experiment on a PVC mount against the matching raw IPC
file on S3 (handling ``.mzML`` / ``.mzml`` naming differences). When rows are
present in IPC but absent from parquet, uses ``pl.scan_ipc`` to fetch only the
missing rows from S3, converts them to the parquet schema, and appends each row
to the shard whose existing ``index`` ordering fits (shards may grow beyond
their original size).

S3 scanning reads credentials from environment variables:

- ``AWS_ACCESS_KEY_ID`` (required for S3 IPC)
- ``AWS_SECRET_ACCESS_KEY`` (required for S3 IPC)
- ``AWS_ENDPOINT_URL`` (required for S3 IPC)

USAGE:
======
python scripts/preprocessing/restore_filtered_spectra.py \\
    --parquet-dir <data-root>/acfm \\
    --s3-prefix s3://<your-bucket>/acfm/ \\
    --search-data search_data_with_new_projects.xlsx \\
    --aws-profile <your-aws-profile>

python scripts/preprocessing/restore_filtered_spectra.py \\
    --parquet-dir <data-root>/acfm \\
    --s3-prefix s3://<your-bucket>/acfm/ \\
    --project PXD047873 \\
    --dry-run
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import polars as pl
import typer

from instanovo.utils.data_handler import SpectrumDataFrame
from scripts.preprocessing.add_acquisition_column import (
    extract_file_name,
    load_acquisitions_from_search_data,
)
from scripts.preprocessing.parquet_io import (
    align_dataframe_to_schema,
    atomic_write_parquet,
    build_shard_order,
    experiment_name_from_path,
    group_parquet_filenames,
    shard_path_for_index,
)
from scripts.preprocessing.recombine_acfm_random_splits import ACFM_REFERENCE_DTYPES
from scripts.preprocessing.sync_acfm_raw import list_s3_project_files
from scripts.verification.add_usi_column import build_usi_string

app = typer.Typer(
    help="Restore spectra filtered out of ACFM parquet files from raw S3 IPC"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_PROJECT_RE = re.compile(r"^(PXD|MSV)\d+", re.IGNORECASE)

ACFM_COLUMN_MAPPING = {
    "rt": "retention_time",
    "mz": "mz_array",
    "intensity": "intensity_array",
}

_DEFAULT_PARQUET_DIR = Path("<data-root>/acfm")
# No default: the source bucket is deployment-specific and must be passed
# explicitly, so a stale identifier cannot be baked into a public repo.
_DEFAULT_S3_PREFIX = None

_DEFAULT_SEARCH_DATA = Path("search_data_with_new_projects.xlsx")

PARQUET_DIR_OPTION = typer.Option(
    _DEFAULT_PARQUET_DIR,
    "--parquet-dir",
    "-p",
    help="PVC directory containing flat project parquet folders",
)
S3_PREFIX_OPTION = typer.Option(
    _DEFAULT_S3_PREFIX,
    "--s3-prefix",
    help="S3 prefix containing raw IPC project folders",
)
AWS_PROFILE_OPTION = typer.Option(
    "default",
    "--aws-profile",
    help="AWS CLI profile for S3 access",
)
SEARCH_DATA_OPTION = typer.Option(
    _DEFAULT_SEARCH_DATA,
    "--search-data",
    "-s",
    help="Search metadata Excel for acquisition lookup",
)
PROJECT_OPTION = typer.Option(
    [],
    "--project",
    help="Process only these project IDs (repeatable)",
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    "-n",
    help="Report missing rows without downloading or writing",
)


@dataclass
class RestoreStats:
    """Counters for a restore run."""

    experiments_checked: int = 0
    experiments_with_missing: int = 0
    rows_restored: int = 0
    shards_written: int = 0
    experiments_written: int = 0
    missing_ipc: int = 0
    missing_acquisition: int = 0
    errors: int = 0
    skipped_complete: int = 0
    issues: List[str] = field(default_factory=list)


def _is_project_dir(name: str) -> bool:
    return _PROJECT_RE.match(name) is not None


def discover_projects(parquet_dir: Path) -> List[str]:
    """List project folders under *parquet_dir*."""
    if not parquet_dir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in parquet_dir.iterdir()
        if entry.is_dir() and _is_project_dir(entry.name)
    )


def discover_experiment_groups(project_dir: Path) -> Dict[str, List[Path]]:
    """Group local parquet shard files by merged output filename."""
    filenames = sorted(path.name for path in project_dir.glob("*.parquet"))
    groups = group_parquet_filenames(filenames)
    return {
        output_name: sorted(project_dir / member for member in members)
        for output_name, members in groups.items()
    }


def _experiment_match_key(name: str) -> str:
    """Normalized experiment key for parquet ↔ IPC matching."""
    fake_path = name if "/" in name else f"dummy/{name}"
    return str(extract_file_name(fake_path).casefold())


def find_matching_ipc(ipc_files: List[str], parquet_output_name: str) -> Optional[str]:
    """Return the S3 IPC basename matching *parquet_output_name*, if any."""
    target = _experiment_match_key(parquet_output_name)
    for filename in ipc_files:
        if not filename.lower().endswith(".ipc"):
            continue
        if _experiment_match_key(filename) == target:
            return filename
    return None


def _require_env(name: str) -> str:
    """Return a non-empty environment variable or raise."""
    if name not in os.environ:
        raise OSError(f"Required environment variable {name} is not set.")
    value = os.environ[name].strip()
    if not value:
        raise OSError(f"Required environment variable {name} is empty.")
    return value


def _normalize_endpoint_url(endpoint: str) -> str:
    """Ensure endpoint is a valid http(s) URI for object_store."""
    if endpoint.startswith("s3://"):
        raise ValueError(f"AWS_ENDPOINT_URL must be an HTTP URL, not {endpoint!r}")
    if not endpoint.startswith(("http://", "https://")):
        endpoint = f"https://{endpoint}"
    return endpoint.rstrip("/")


def get_storage_options_from_env() -> dict:
    """Build Polars storage options from required environment variables."""
    endpoint = _normalize_endpoint_url(_require_env("AWS_ENDPOINT_URL"))

    storage_opts: dict = {
        "aws_access_key_id": _require_env("AWS_ACCESS_KEY_ID"),
        "aws_secret_access_key": _require_env("AWS_SECRET_ACCESS_KEY"),
        "aws_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
        "endpoint": endpoint,
        "endpoint_url": endpoint,
        "aws_endpoint_url": endpoint,
        "aws_virtual_hosted_style_request": "false",
    }

    logger.info(
        "S3 config from env: endpoint=%s region=%s",
        endpoint,
        storage_opts["aws_region"],
    )
    return storage_opts


def _ipc_s3_uri(s3_prefix: str, project: str, ipc_filename: str) -> str:
    return f"{s3_prefix.rstrip('/')}/{project}/{ipc_filename}"


_CACHED_STORAGE_OPTIONS: dict | None = None


def _get_storage_options() -> dict:
    global _CACHED_STORAGE_OPTIONS
    if _CACHED_STORAGE_OPTIONS is None:
        _CACHED_STORAGE_OPTIONS = get_storage_options_from_env()
    return _CACHED_STORAGE_OPTIONS


def _scan_ipc(ipc_source: str) -> pl.LazyFrame:
    return pl.scan_ipc(ipc_source, storage_options=_get_storage_options())


def _collect_streaming(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect(streaming=True)


def _count_ipc_rows(lf: pl.LazyFrame) -> int:
    return int(_collect_streaming(lf.select(pl.len())).item())


def _normalise_ipc_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    df = df.rename({k: v for k, v in ACFM_COLUMN_MAPPING.items() if k in df.columns})
    return SpectrumDataFrame._cast_columns(df)


def _count_missing_ipc_rows(
    ipc_source: str,
    existing: Set[int],
) -> tuple[int, int]:
    """Return ``(total_ipc_rows, missing_row_count)`` via lazy scan."""
    lf = _scan_ipc(ipc_source)
    if "index" not in lf.collect_schema():
        raise ValueError(f"IPC missing 'index' column: {ipc_source}")

    total = _count_ipc_rows(lf)
    missing = _collect_streaming(
        lf.filter(~pl.col("index").is_in(list(existing))).select(pl.len())
    ).item()
    return total, missing


def load_missing_ipc_rows(
    ipc_source: str,
    existing: Set[int],
) -> tuple[pl.DataFrame, int]:
    """Scan IPC and collect only rows whose ``index`` is not in *existing*."""
    lf = _scan_ipc(ipc_source)
    if "index" not in lf.collect_schema():
        raise ValueError(f"IPC missing 'index' column: {ipc_source}")

    total = _count_ipc_rows(lf)
    missing_df = _collect_streaming(lf.filter(~pl.col("index").is_in(list(existing))))
    if missing_df.is_empty():
        return missing_df, total
    return _normalise_ipc_dataframe(missing_df), total


def _existing_indices(parquet_paths: List[Path]) -> Set[int]:
    indices: Set[int] = set()
    for path in parquet_paths:
        if "index" not in pl.read_parquet_schema(path):
            raise ValueError(f"Parquet file missing 'index' column: {path}")
        col = pl.read_parquet(path, columns=["index"])["index"]
        indices.update(int(v) for v in col.to_list() if v is not None)
    return indices


def _lookup_acquisition(
    project: str,
    output_path: Path,
    acquisition_map: Dict[Tuple[str, str], str],
) -> Optional[str]:
    """Resolve acquisition from search data for an experiment output path."""
    return acquisition_map.get((project, extract_file_name(str(output_path))))


def _add_usi_column(df: pl.DataFrame, output_path: Path) -> pl.DataFrame:
    if "scan" not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.String).alias("usi"))

    usi_values = [
        build_usi_string(
            str(output_path),
            row.get("scan"),
            sequence=None,
            precursor_charge=row.get("precursor_charge"),
        )
        for row in df.iter_rows(named=True)
    ]
    return df.with_columns(pl.Series("usi", usi_values, dtype=pl.String))


def _prepare_missing_rows(
    missing_df: pl.DataFrame,
    output_path: Path,
    acquisition: Optional[str],
) -> pl.DataFrame:
    experiment_name = experiment_name_from_path(str(output_path))
    missing_df = missing_df.with_columns(
        pl.lit(experiment_name).alias("experiment_name"),
        pl.lit(acquisition).cast(pl.String).alias("acquisition"),
    )
    missing_df = _add_usi_column(missing_df, output_path)
    return align_dataframe_to_schema(missing_df, ACFM_REFERENCE_DTYPES)


def _refresh_metadata(
    df: pl.DataFrame, output_path: Path, acquisition: Optional[str]
) -> pl.DataFrame:
    """Rewrite experiment_name, acquisition, and USI on a full dataframe."""
    experiment_name = experiment_name_from_path(str(output_path))
    df = df.with_columns(
        pl.lit(experiment_name).alias("experiment_name"),
        pl.lit(acquisition).cast(pl.String).alias("acquisition"),
    )
    df = _add_usi_column(df, output_path)
    return align_dataframe_to_schema(df, ACFM_REFERENCE_DTYPES)


def _group_missing_rows_by_shard(
    missing_df: pl.DataFrame,
    parquet_paths: List[Path],
) -> Dict[Path, pl.DataFrame]:
    """Split missing rows into per-shard dataframes by ``index`` order."""
    shard_order = build_shard_order(parquet_paths)
    grouped: DefaultDict[Path, List[dict]] = defaultdict(list)
    for row in missing_df.iter_rows(named=True):
        shard_path = shard_path_for_index(int(row["index"]), shard_order)
        grouped[shard_path].append(row)
    return {path: pl.DataFrame(rows) for path, rows in grouped.items()}


def restore_experiment(
    project: str,
    output_filename: str,
    parquet_paths: List[Path],
    ipc_files: List[str],
    s3_prefix: str,
    acquisition_map: Dict[Tuple[str, str], str],
    dry_run: bool,
    stats: RestoreStats,
) -> None:
    """Restore missing rows for one experiment group."""
    stats.experiments_checked += 1
    ipc_name = find_matching_ipc(ipc_files, output_filename)
    if ipc_name is None:
        stats.missing_ipc += 1
        msg = f"{project}/{output_filename}: no matching IPC on S3"
        stats.issues.append(msg)
        logger.warning(msg)
        return

    acquisition = _lookup_acquisition(project, parquet_paths[0], acquisition_map)
    if acquisition is None:
        stats.missing_acquisition += 1
        logger.warning(
            "No acquisition in search data for %s/%s",
            project,
            experiment_name_from_path(str(parquet_paths[0])),
        )
        if not dry_run:
            return

    existing = _existing_indices(parquet_paths)

    ipc_source = _ipc_s3_uri(s3_prefix, project, ipc_name)

    if dry_run:
        logger.info(
            "[DRY-RUN] Scanning %s for missing rows (matched %s)",
            ipc_source,
            ipc_name,
        )

    try:
        if dry_run:
            total_ipc_rows, missing_count = _count_missing_ipc_rows(
                ipc_source, existing
            )
            missing_df = pl.DataFrame()
        else:
            missing_df, total_ipc_rows = load_missing_ipc_rows(ipc_source, existing)
            missing_count = len(missing_df)
    except OSError as exc:
        stats.errors += 1
        msg = (
            f"{project}/{output_filename}: S3 credentials required for {ipc_source}: "
            f"{exc}"
        )
        stats.issues.append(msg)
        logger.error(msg)
        return
    except Exception as exc:
        stats.errors += 1
        msg = f"{project}/{output_filename}: failed to scan IPC {ipc_name}: {exc}"
        stats.issues.append(msg)
        logger.error(msg)
        return

    if missing_count == 0:
        stats.skipped_complete += 1
        return

    stats.experiments_with_missing += 1
    logger.info(
        "%s/%s: %d missing row(s) of %d IPC rows (matched %s)",
        project,
        output_filename,
        missing_count,
        total_ipc_rows,
        ipc_name,
    )

    if dry_run:
        stats.rows_restored += missing_count
        return

    missing_by_shard = _group_missing_rows_by_shard(missing_df, parquet_paths)

    wrote_any = False
    restored_count = 0
    for shard_path, shard_missing in missing_by_shard.items():
        shard_df = pl.read_parquet(shard_path)
        shard_missing = _prepare_missing_rows(shard_missing, shard_path, acquisition)
        combined = pl.concat(
            [
                align_dataframe_to_schema(shard_df, ACFM_REFERENCE_DTYPES),
                shard_missing,
            ],
            how="vertical_relaxed",
        ).sort("index")
        combined = _refresh_metadata(combined, shard_path, acquisition)

        atomic_write_parquet(combined, shard_path)
        stats.shards_written += 1
        restored_count += len(shard_missing)
        wrote_any = True
        logger.info(
            "Wrote %s with %d rows (%d restored)",
            shard_path,
            len(combined),
            len(shard_missing),
        )

    if wrote_any:
        stats.experiments_written += 1
        stats.rows_restored += restored_count


def run_restore(
    parquet_dir: Path,
    s3_prefix: str,
    search_data_path: Path,
    aws_profile: str | None,
    projects_filter: Optional[List[str]],
    dry_run: bool,
) -> RestoreStats:
    """Run the full restore pipeline."""
    if not parquet_dir.is_dir():
        raise typer.BadParameter(f"Parquet directory does not exist: {parquet_dir}")
    if not search_data_path.is_file():
        raise typer.BadParameter(f"Search data file not found: {search_data_path}")

    acquisition_map = load_acquisitions_from_search_data(str(search_data_path))

    all_projects = discover_projects(parquet_dir)
    if projects_filter:
        unknown = set(projects_filter) - set(all_projects)
        for project in sorted(unknown):
            logger.warning("Requested project not found locally: %s", project)
        projects = [p for p in all_projects if p in projects_filter]
    else:
        projects = all_projects

    stats = RestoreStats()
    logger.info(
        "Restoring filtered spectra: %d project(s), parquet=%s, s3=%s",
        len(projects),
        parquet_dir,
        s3_prefix,
    )

    for project in projects:
        project_dir = parquet_dir / project
        try:
            ipc_files = list_s3_project_files(s3_prefix, project, aws_profile)
        except subprocess.CalledProcessError as exc:
            stats.errors += 1
            msg = f"{project}: failed to list S3 objects: {exc.stderr}"
            stats.issues.append(msg)
            logger.error(msg)
            continue

        experiment_groups = discover_experiment_groups(project_dir)
        for output_filename, parquet_paths in experiment_groups.items():
            restore_experiment(
                project,
                output_filename,
                parquet_paths,
                ipc_files,
                s3_prefix,
                acquisition_map,
                dry_run,
                stats,
            )

    logger.info(
        "Done: checked=%d missing_experiments=%d rows_restored=%d "
        "experiments_written=%d shards_written=%d "
        "missing_ipc=%d missing_acquisition=%d complete=%d errors=%d",
        stats.experiments_checked,
        stats.experiments_with_missing,
        stats.rows_restored,
        stats.experiments_written,
        stats.shards_written,
        stats.missing_ipc,
        stats.missing_acquisition,
        stats.skipped_complete,
        stats.errors,
    )
    return stats


@app.command()
def main(
    parquet_dir: Path = PARQUET_DIR_OPTION,
    s3_prefix: str = S3_PREFIX_OPTION,
    search_data: Path = SEARCH_DATA_OPTION,
    aws_profile: str = AWS_PROFILE_OPTION,
    project: List[str] = PROJECT_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Restore quality-filtered spectra missing from local ACFM parquet files."""
    run_restore(
        parquet_dir=parquet_dir,
        s3_prefix=s3_prefix,
        search_data_path=search_data,
        aws_profile=aws_profile,
        projects_filter=project if project else None,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
