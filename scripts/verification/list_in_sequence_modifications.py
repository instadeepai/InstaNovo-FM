r"""Scan parquet datasets for ``[IN:<integer>]`` tokens in the ``sequence`` column.

Walks ``--input-dir`` the same way as other verification tools: one subfolder per
project, parquets under each. For every row whose sequence contains one or more
matches of ``[IN:123]`` (digits only), counts **rows** per modification string
(a row with two different ``[IN:…]`` tokens increments both).

USAGE:
======
python scripts/verification/list_in_sequence_modifications.py \\
    --input-dir <data-root>/lcfm/

python scripts/verification/list_in_sequence_modifications.py \\
    -i <data-root>/lcfm/ --output-csv in_mod_counts.csv
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Annotated, DefaultDict, Dict, List

import polars as pl
import typer

app = typer.Typer(help="List [IN:<int>] modifications in sequence columns by project")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MOD_PATTERN = re.compile(r"\[IN:(\d+)\]")


def find_project_folders(input_dir: str) -> List[str]:
    """Direct subfolders of ``input_dir`` that contain at least one ``*.parquet`` (any depth)."""
    projects: List[str] = []
    for entry in os.listdir(input_dir):
        project_path = os.path.join(input_dir, entry)
        if not os.path.isdir(project_path):
            continue
        for _, _, files in os.walk(project_path):
            if any(f.endswith(".parquet") for f in files):
                projects.append(entry)
                break
    return sorted(projects)


def find_parquet_files_in_project(input_dir: str, project: str) -> List[str]:
    """Return paths to all ``*.parquet`` files under ``input_dir/project`` (recursive)."""
    paths: List[str] = []
    project_path = os.path.join(input_dir, project)
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.endswith(".parquet"):
                paths.append(os.path.join(root, file))
    return paths


def mods_from_sequence(seq: str | None) -> List[str]:
    """Return each ``[IN:<digits>]`` token in ``seq``; empty if missing or not a string."""
    if seq is None:
        return []
    if not isinstance(seq, str):
        seq = str(seq)
    return [f"[IN:{n}]" for n in MOD_PATTERN.findall(seq)]


def accumulate_file(parquet_path: str, per_mod: DefaultDict[str, int]) -> None:
    """Update ``per_mod`` with row counts for each ``[IN:…]`` token in ``sequence``."""
    try:
        schema = pl.scan_parquet(parquet_path).collect_schema()
    except Exception as e:
        logger.warning("Skip unreadable parquet %s: %s", parquet_path, e)
        return
    if "sequence" not in schema:
        return

    df = pl.read_parquet(parquet_path, columns=["sequence"])
    for cell in df["sequence"].to_list():
        for token in mods_from_sequence(cell):
            per_mod[token] += 1


def run_scan(
    input_dir: Path,
    projects_filter: List[str] | None,
    output_csv: Path | None,
) -> None:
    """Walk parquet files under ``input_dir``, count rows per ``[IN:*]`` token, optional CSV."""
    root = str(input_dir.resolve())
    all_projects = find_project_folders(root)
    if projects_filter:
        want = set(projects_filter)
        projects = [p for p in all_projects if p in want]
        missing = want - set(projects)
        for p in sorted(missing):
            logger.warning("Requested project not found or no parquet: %s", p)
    else:
        projects = all_projects

    rows_out: List[Dict[str, str | int]] = []

    for project in projects:
        per_mod: DefaultDict[str, int] = defaultdict(int)
        files = find_parquet_files_in_project(root, project)
        for fp in files:
            accumulate_file(fp, per_mod)
        if not per_mod:
            continue
        logger.info("Project %s: %d distinct [IN:*] token(s)", project, len(per_mod))
        for mod in sorted(per_mod, key=lambda m: (int(m[4:-1]), m)):
            cnt = per_mod[mod]
            logger.info("  %s  rows=%d", mod, cnt)
            rows_out.append({"project": project, "modification": mod, "rows": cnt})

    if output_csv and rows_out:
        pl.DataFrame(rows_out).write_csv(output_csv)
        logger.info("Wrote %s", output_csv)
    elif output_csv:
        logger.info("No [IN:integer] tokens found; not writing %s", output_csv)


@app.command()
def main(
    input_dir: Annotated[
        Path,
        typer.Option(
            "--input-dir",
            "-i",
            help="Root directory with per-project subfolders containing parquets",
        ),
    ],
    project: Annotated[
        List[str] | None,
        typer.Option(
            "--project",
            "-p",
            help="Limit to these project folder names (repeat)",
        ),
    ] = None,
    output_csv: Annotated[
        Path | None,
        typer.Option(
            "--output-csv",
            "-o",
            help="Optional CSV: project, modification, rows",
        ),
    ] = None,
) -> None:
    """Print projects that have ``[IN:<digits>]`` in ``sequence``, with row counts per token."""
    if not input_dir.is_dir():
        raise typer.BadParameter(f"Not a directory: {input_dir}")
    run_scan(input_dir, project or None, output_csv)


if __name__ == "__main__":
    app()
