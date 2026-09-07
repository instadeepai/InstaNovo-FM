r"""Recombine ACFM random train/val/test shards into flat project folders.

Dissolves ``acfm_random_splits/{train,val,test}/PXD…/`` into ``acfm/PXD…/``,
merging shards that share the same parquet basename, sorting by ``index``, and
adding ``experiment_name``, ``usi``, and ``acquisition`` columns.

USAGE:
======
python scripts/preprocessing/recombine_acfm_random_splits.py \\
    --input-root <data-root>/acfm_random_splits \\
    --output-root <data-root>/acfm \\
    --search-data search_data_with_new_projects.xlsx
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

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
    group_parquet_filenames,
)
from scripts.splitting.split_unlabelled_data import REFERENCE_SCHEMA
from scripts.verification.add_usi_column import build_usi_string

app = typer.Typer(help="Recombine ACFM random split shards into flat project folders")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_PROJECT_RE = re.compile(r"^(PXD|MSV)\d+", re.IGNORECASE)

ACFM_REFERENCE_DTYPES: Dict[str, pl.DataType] = {
    **REFERENCE_SCHEMA,
    "isolation_target_old": pl.Float64,
}

_DEFAULT_INPUT_ROOT = Path("<data-root>/acfm_random_splits")
_DEFAULT_OUTPUT_ROOT = Path("<data-root>/acfm")
_DEFAULT_SEARCH_DATA = Path("search_data_with_new_projects.xlsx")

INPUT_ROOT_OPTION = typer.Option(
    _DEFAULT_INPUT_ROOT,
    "--input-root",
    "-i",
    help="Root containing train/val/test split folders",
)
OUTPUT_ROOT_OPTION = typer.Option(
    _DEFAULT_OUTPUT_ROOT,
    "--output-root",
    "-o",
    help="Flat project output directory",
)
SEARCH_DATA_OPTION = typer.Option(
    _DEFAULT_SEARCH_DATA,
    "--search-data",
    "-s",
    help="Search metadata Excel for acquisition lookup",
)
SPLITS_OPTION = typer.Option(
    "train,val,test",
    "--splits",
    help="Comma-separated split folder names",
)
PROJECT_OPTION = typer.Option(
    [],
    "--project",
    "-p",
    help="Process only these project IDs (repeatable)",
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    "-n",
    help="Log actions without writing or deleting files",
)


@dataclass
class RecombineStats:
    """Counters for a recombine run."""

    projects_processed: int = 0
    files_recombined: int = 0
    rows_written: int = 0
    shards_deleted: int = 0
    missing_acquisition_files: int = 0
    skipped_no_shards: int = 0
    projects_seen: List[str] = field(default_factory=list)


def _is_project_dir(name: str) -> bool:
    return _PROJECT_RE.match(name) is not None


def discover_projects(input_root: Path, splits: Tuple[str, ...]) -> List[str]:
    """Union project folder names under each split directory."""
    projects: Set[str] = set()
    for split in splits:
        split_dir = input_root / split
        if not split_dir.is_dir():
            continue
        for entry in split_dir.iterdir():
            if entry.is_dir() and _is_project_dir(entry.name):
                projects.add(entry.name)
    return sorted(projects)


def discover_experiment_groups(
    input_root: Path, project: str, splits: Tuple[str, ...]
) -> Dict[str, List[str]]:
    """Group parquet basenames for *project*, merging IPC-style shard suffixes."""
    names: Set[str] = set()
    for split in splits:
        project_dir = input_root / split / project
        if not project_dir.is_dir():
            continue
        for path in project_dir.glob("*.parquet"):
            names.add(path.name)
    return group_parquet_filenames(sorted(names))


def collect_group_paths(
    input_root: Path,
    project: str,
    member_filenames: List[str],
    splits: Tuple[str, ...],
) -> List[Path]:
    """Return existing paths for shard members across splits."""
    paths: List[Path] = []
    for split in splits:
        for filename in member_filenames:
            candidate = input_root / split / project / filename
            if candidate.is_file():
                paths.append(candidate)
    return paths


def _add_usi_column(df: pl.DataFrame, output_path: Path) -> pl.DataFrame:
    """Add spectrum-only USI strings (no sequence interpretation)."""
    filepath_str = str(output_path)
    if "scan" not in df.columns:
        return df.with_columns(pl.lit(None).cast(pl.String).alias("usi"))

    usi_values: List[Optional[str]] = []
    for row in df.iter_rows(named=True):
        usi_values.append(
            build_usi_string(
                filepath_str,
                row.get("scan"),
                sequence=None,
                precursor_charge=row.get("precursor_charge"),
            )
        )
    return df.with_columns(pl.Series("usi", usi_values, dtype=pl.String))


def _add_metadata_columns(
    df: pl.DataFrame,
    output_path: Path,
    project: str,
    acquisition_map: Dict[Tuple[str, str], str],
    stats: RecombineStats,
) -> pl.DataFrame:
    experiment_name = experiment_name_from_path(str(output_path))
    acquisition = acquisition_map.get((project, extract_file_name(str(output_path))))

    if acquisition is None:
        stats.missing_acquisition_files += 1
        logger.info(
            "No acquisition in search data for %s/%s",
            project,
            experiment_name,
        )

    df = df.with_columns(
        pl.lit(experiment_name).alias("experiment_name"),
        pl.lit(acquisition).cast(pl.String).alias("acquisition"),
    )
    return _add_usi_column(df, output_path)


def recombine_single_file(
    input_root: Path,
    output_root: Path,
    project: str,
    output_filename: str,
    member_filenames: List[str],
    splits: Tuple[str, ...],
    acquisition_map: Dict[Tuple[str, str], str],
    dry_run: bool,
    stats: RecombineStats,
) -> None:
    """Merge shards for one experiment, write output, delete sources."""
    shard_paths = collect_group_paths(input_root, project, member_filenames, splits)
    if not shard_paths:
        stats.skipped_no_shards += 1
        return

    frames = [
        align_dataframe_to_schema(pl.read_parquet(p), ACFM_REFERENCE_DTYPES)
        for p in shard_paths
    ]
    combined = pl.concat(frames, how="vertical_relaxed").sort("index")

    output_path = output_root / project / output_filename
    combined = _add_metadata_columns(
        combined, output_path, project, acquisition_map, stats
    )

    row_count = len(combined)
    shard_count = len(shard_paths)

    if dry_run:
        logger.info(
            "[DRY-RUN] %s/%s: %d source file(s), %d rows → would write %s, delete %d sources",
            project,
            output_filename,
            shard_count,
            row_count,
            output_path,
            shard_count,
        )
        return

    atomic_write_parquet(combined, output_path)
    for shard in shard_paths:
        shard.unlink()
        stats.shards_deleted += 1

    stats.files_recombined += 1
    stats.rows_written += row_count
    logger.info(
        "%s/%s: %d source file(s), %d rows → wrote %s, deleted %d sources",
        project,
        output_filename,
        shard_count,
        row_count,
        output_path,
        shard_count,
    )


def remove_empty_project_split_dirs(
    input_root: Path, project: str, splits: Tuple[str, ...], dry_run: bool
) -> None:
    """Remove empty ``{split}/{project}/`` directories after a project completes."""
    for split in splits:
        project_dir = input_root / split / project
        if not project_dir.is_dir():
            continue
        if any(project_dir.iterdir()):
            continue
        if dry_run:
            logger.info("[DRY-RUN] Would remove empty directory %s", project_dir)
        else:
            project_dir.rmdir()
            logger.info("Removed empty directory %s", project_dir)


def remove_empty_split_dirs(
    input_root: Path, splits: Tuple[str, ...], dry_run: bool
) -> None:
    """Remove empty top-level split directories after all projects complete."""
    for split in splits:
        split_dir = input_root / split
        if not split_dir.is_dir():
            continue
        if any(split_dir.iterdir()):
            continue
        if dry_run:
            logger.info("[DRY-RUN] Would remove empty split directory %s", split_dir)
        else:
            try:
                split_dir.rmdir()
                logger.info("Removed empty split directory %s", split_dir)
            except OSError:
                pass


def run_recombine(
    input_root: Path,
    output_root: Path,
    search_data_path: Path,
    splits: Tuple[str, ...],
    projects_filter: Optional[List[str]],
    dry_run: bool,
) -> RecombineStats:
    """Run the full recombine pipeline."""
    if not input_root.is_dir():
        raise typer.BadParameter(f"Input root does not exist: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    acquisition_map = load_acquisitions_from_search_data(str(search_data_path))

    all_projects = discover_projects(input_root, splits)
    if projects_filter:
        unknown = set(projects_filter) - set(all_projects)
        for proj in sorted(unknown):
            logger.warning("Requested project not found under input root: %s", proj)
        projects = [p for p in all_projects if p in projects_filter]
    else:
        projects = all_projects

    stats = RecombineStats()
    stats.projects_seen = projects
    total_projects = len(projects)

    logger.info(
        "Starting recombine: %d projects, input=%s, output=%s",
        total_projects,
        input_root,
        output_root,
    )

    for idx, project in enumerate(projects, start=1):
        logger.info("Processing %s (%d/%d)", project, idx, total_projects)
        experiment_groups = discover_experiment_groups(input_root, project, splits)
        for output_filename, member_filenames in experiment_groups.items():
            recombine_single_file(
                input_root,
                output_root,
                project,
                output_filename,
                member_filenames,
                splits,
                acquisition_map,
                dry_run,
                stats,
            )
        remove_empty_project_split_dirs(input_root, project, splits, dry_run)
        stats.projects_processed += 1
        logger.info(
            "Finished %s: %d files recombined so far",
            project,
            stats.files_recombined,
        )

    remove_empty_split_dirs(input_root, splits, dry_run)

    logger.info(
        "Done: projects=%d files=%d rows=%d shards_deleted=%d "
        "missing_acquisition_files=%d skipped_no_shards=%d",
        stats.projects_processed,
        stats.files_recombined,
        stats.rows_written,
        stats.shards_deleted,
        stats.missing_acquisition_files,
        stats.skipped_no_shards,
    )
    return stats


@app.command()
def main(
    input_root: Path = INPUT_ROOT_OPTION,
    output_root: Path = OUTPUT_ROOT_OPTION,
    search_data: Path = SEARCH_DATA_OPTION,
    splits: str = SPLITS_OPTION,
    project: List[str] = PROJECT_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Recombine random-split ACFM shards into flat project folders."""
    split_tuple = tuple(s.strip() for s in splits.split(",") if s.strip())
    if not split_tuple:
        raise typer.BadParameter("--splits must list at least one folder name")

    if not search_data.is_file():
        raise typer.BadParameter(f"Search data file not found: {search_data}")

    run_recombine(
        input_root=input_root,
        output_root=output_root,
        search_data_path=search_data,
        splits=split_tuple,
        projects_filter=project if project else None,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
