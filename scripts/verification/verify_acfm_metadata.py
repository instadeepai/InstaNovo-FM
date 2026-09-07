r"""Verify ``experiment_name`` and ``acquisition`` columns in ACFM parquet files.

Reports files where ``experiment_name`` is absent/null or ``acquisition`` is
null. IPC-style shard suffixes (``_0000-0017``) and embedded ``.mzml`` are
stripped when deriving the expected experiment name. Sharded files are checked
and corrected in place — they are not merged.

USAGE:
======
python scripts/verification/verify_acfm_metadata.py \\
    --input-dir <data-root>/acfm

python scripts/verification/verify_acfm_metadata.py \\
    --input-dir <data-root>/acfm \\
    --search-data search_data_with_new_projects.xlsx \\
    --fix-metadata
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
import typer

from scripts.preprocessing.add_acquisition_column import (
    extract_file_name,
    load_acquisitions_from_search_data,
)
from scripts.preprocessing.parquet_io import (
    align_dataframe_to_schema,
    atomic_write_parquet,
    experiment_name_from_path,
)
from scripts.preprocessing.recombine_acfm_random_splits import ACFM_REFERENCE_DTYPES
from scripts.verification.add_usi_column import build_usi_string

app = typer.Typer(
    help="Verify experiment_name and acquisition columns in ACFM parquet files"
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_DEFAULT_INPUT_DIR = Path("<data-root>/acfm")
_DEFAULT_SEARCH_DATA = Path("search_data_with_new_projects.xlsx")

INPUT_DIR_OPTION = typer.Option(
    _DEFAULT_INPUT_DIR,
    "--input-dir",
    "-i",
    help="Root directory containing project parquet subfolders",
)
SEARCH_DATA_OPTION = typer.Option(
    _DEFAULT_SEARCH_DATA,
    "--search-data",
    "-s",
    help="Search metadata Excel for acquisition lookup when fixing",
)
PROJECT_OPTION = typer.Option(
    [],
    "--project",
    "-p",
    help="Process only these project IDs (repeatable)",
)
FIX_METADATA_OPTION = typer.Option(
    False,
    "--fix-metadata",
    help="Rewrite experiment_name/acquisition/USI in each file without merging shards",
)
FIX_ACQUISITION_OPTION = typer.Option(
    False,
    "--fix-acquisition",
    help="Alias for --fix-metadata (deprecated name kept for compatibility)",
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    "-n",
    help="Report issues without modifying files",
)


@dataclass
class MetadataIssue:
    """One metadata problem in a parquet file."""

    project: str
    parquet_path: str
    expected_experiment_name: str
    missing_experiment_name: bool
    wrong_experiment_name_rows: int
    null_acquisition_rows: int
    total_rows: int


@dataclass
class VerifyStats:
    """Counters for a metadata verification run."""

    files_checked: int = 0
    files_with_issues: int = 0
    files_fixed: int = 0
    issues: List[MetadataIssue] = field(default_factory=list)


def _is_project_dir(name: str) -> bool:
    return name.startswith("PXD") or name.startswith("MSV")


def discover_projects(input_dir: Path) -> List[str]:
    """Discover projects under the input directory."""
    if not input_dir.is_dir():
        return []
    return sorted(
        entry.name
        for entry in input_dir.iterdir()
        if entry.is_dir() and _is_project_dir(entry.name)
    )


def _check_dataframe(
    df: pl.DataFrame,
    project: str,
    parquet_path: Path,
    expected_experiment_name: str,
) -> Optional[MetadataIssue]:
    missing_experiment_name = "experiment_name" not in df.columns
    null_experiment_rows = 0
    wrong_experiment_name_rows = 0
    if not missing_experiment_name:
        null_experiment_rows = int(df["experiment_name"].null_count())
        wrong_experiment_name_rows = int(
            df.filter(
                pl.col("experiment_name").is_not_null()
                & (pl.col("experiment_name") != expected_experiment_name)
            ).height
        )

    null_acquisition_rows = 0
    if "acquisition" not in df.columns:
        null_acquisition_rows = len(df)
    else:
        null_acquisition_rows = int(df["acquisition"].null_count())

    has_issue = (
        missing_experiment_name
        or null_experiment_rows > 0
        or wrong_experiment_name_rows > 0
        or null_acquisition_rows > 0
    )
    if not has_issue:
        return None

    return MetadataIssue(
        project=project,
        parquet_path=str(parquet_path),
        expected_experiment_name=expected_experiment_name,
        missing_experiment_name=missing_experiment_name or null_experiment_rows > 0,
        wrong_experiment_name_rows=wrong_experiment_name_rows,
        null_acquisition_rows=null_acquisition_rows,
        total_rows=len(df),
    )


def _add_usi_column(df: pl.DataFrame, file_path: Path) -> pl.DataFrame:
    if "scan" not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.String).alias("usi"))

    usi_values = [
        build_usi_string(
            str(file_path),
            row.get("scan"),
            sequence=None,
            precursor_charge=row.get("precursor_charge"),
        )
        for row in df.iter_rows(named=True)
    ]
    return df.with_columns(pl.Series("usi", usi_values, dtype=pl.String))


def _fix_dataframe(
    df: pl.DataFrame,
    file_path: Path,
    project: str,
    experiment_name: str,
    acquisition_map: Dict[Tuple[str, str], str],
) -> pl.DataFrame:
    acquisition = acquisition_map.get((project, extract_file_name(str(file_path))))
    df = df.with_columns(
        pl.lit(experiment_name).alias("experiment_name"),
        pl.lit(acquisition).cast(pl.String).alias("acquisition"),
    )
    df = _add_usi_column(df, file_path)
    return align_dataframe_to_schema(df, ACFM_REFERENCE_DTYPES)


def verify_project(
    input_dir: Path,
    project: str,
    acquisition_map: Optional[Dict[Tuple[str, str], str]],
    fix_metadata: bool,
    dry_run: bool,
    stats: VerifyStats,
) -> None:
    """Verify (and optionally fix) each parquet file under one project."""
    project_dir = input_dir / project
    parquet_paths = sorted(project_dir.glob("*.parquet"))

    for file_path in parquet_paths:
        df = pl.read_parquet(file_path)
        experiment_name = experiment_name_from_path(str(file_path))

        stats.files_checked += 1
        issue = _check_dataframe(df, project, file_path, experiment_name)
        if issue is None:
            continue

        stats.files_with_issues += 1
        stats.issues.append(issue)
        logger.warning(
            "%s/%s: expected experiment_name=%r, missing_experiment_name=%s, "
            "wrong_experiment_name_rows=%d, null_acquisition_rows=%d/%d",
            project,
            file_path.name,
            experiment_name,
            issue.missing_experiment_name,
            issue.wrong_experiment_name_rows,
            issue.null_acquisition_rows,
            issue.total_rows,
        )

        if not fix_metadata or acquisition_map is None:
            continue

        if acquisition_map.get((project, extract_file_name(str(file_path)))) is None:
            logger.warning(
                "No acquisition in search data for %s/%s",
                project,
                experiment_name,
            )
            continue

        fixed = _fix_dataframe(
            df,
            file_path,
            project,
            experiment_name,
            acquisition_map,
        )
        if dry_run:
            logger.info("[DRY-RUN] Would rewrite %s", file_path)
            continue

        atomic_write_parquet(fixed, file_path)
        stats.files_fixed += 1
        logger.info("Fixed metadata in %s", file_path)


def run_verify(
    input_dir: Path,
    search_data_path: Optional[Path],
    projects_filter: Optional[List[str]],
    fix_metadata: bool,
    dry_run: bool,
) -> VerifyStats:
    """Run metadata verification across all projects."""
    if not input_dir.is_dir():
        raise typer.BadParameter(f"Input directory does not exist: {input_dir}")

    if fix_metadata:
        if search_data_path is None or not search_data_path.is_file():
            raise typer.BadParameter(
                f"Search data file required for --fix-metadata: {search_data_path}"
            )
        acquisition_map = load_acquisitions_from_search_data(str(search_data_path))
    else:
        acquisition_map = None

    all_projects = discover_projects(input_dir)
    if projects_filter:
        projects = [p for p in all_projects if p in projects_filter]
    else:
        projects = all_projects

    stats = VerifyStats()
    for project in projects:
        verify_project(
            input_dir,
            project,
            acquisition_map,
            fix_metadata,
            dry_run,
            stats,
        )

    logger.info(
        "Done: checked=%d with_issues=%d fixed=%d",
        stats.files_checked,
        stats.files_with_issues,
        stats.files_fixed,
    )
    return stats


@app.command()
def main(
    input_dir: Path = INPUT_DIR_OPTION,
    search_data: Path = SEARCH_DATA_OPTION,
    project: List[str] = PROJECT_OPTION,
    fix_metadata: bool = FIX_METADATA_OPTION,
    fix_acquisition: bool = FIX_ACQUISITION_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Verify experiment_name and acquisition columns in ACFM parquet files."""
    run_verify(
        input_dir=input_dir,
        search_data_path=search_data if (fix_metadata or fix_acquisition) else None,
        projects_filter=project if project else None,
        fix_metadata=fix_metadata or fix_acquisition,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
