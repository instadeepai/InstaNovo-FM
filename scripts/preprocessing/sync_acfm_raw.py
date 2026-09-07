r"""Sync missing ACFM projects from S3 into a local raw directory.

Downloads only projects present on S3 ``acfm/`` that are not already under
``--processed-dir`` or ``--local-raw-dir``.

USAGE:
======
python scripts/preprocessing/sync_acfm_raw.py \\
    --s3-prefix s3://<your-bucket>/acfm/ \\
    --local-raw-dir <data-root>/acfm_raw \\
    --processed-dir <data-root>/acfm \\
    --aws-profile <your-aws-profile>
"""

from __future__ import annotations

import logging
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

import typer

app = typer.Typer(help="Sync missing ACFM projects from S3 to local acfm_raw")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_PROJECT_RE = re.compile(r"^(PXD|MSV)\d+", re.IGNORECASE)

_DEFAULT_LOCAL_RAW_DIR = Path("<data-root>/acfm_raw")
_DEFAULT_PROCESSED_DIR = Path("<data-root>/acfm")

S3_PREFIX_OPTION = typer.Option(
    None,  # required: pass --s3-prefix explicitly
    "--s3-prefix",
    help="S3 prefix containing flat PXD/MSV project folders",
)
LOCAL_RAW_DIR_OPTION = typer.Option(
    _DEFAULT_LOCAL_RAW_DIR,
    "--local-raw-dir",
    help="Local directory for raw downloads",
)
PROCESSED_DIR_OPTION = typer.Option(
    _DEFAULT_PROCESSED_DIR,
    "--processed-dir",
    help="Local directory of already-processed projects",
)
AWS_PROFILE_OPTION = typer.Option(
    "default",
    "--aws-profile",
    "-p",
    help="AWS CLI profile name",
)
DRY_RUN_OPTION = typer.Option(
    False,
    "--dry-run",
    "-n",
    help="Log planned sync commands without running them",
)


def _parse_s3_prefix(s3_prefix: str) -> tuple[str, str]:
    if not s3_prefix.startswith("s3://"):
        raise typer.BadParameter(f"Not an S3 URI: {s3_prefix}")
    without_scheme = s3_prefix[5:]
    parts = without_scheme.split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket, prefix


def list_s3_projects(s3_prefix: str, aws_profile: str | None) -> List[str]:
    """List top-level project folder names under an S3 prefix."""
    bucket, prefix = _parse_s3_prefix(s3_prefix)
    uri = f"s3://{bucket}/{prefix}"
    cmd = ["aws", "s3", "ls", uri]
    if aws_profile:
        cmd.extend(["--profile", aws_profile])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    projects: List[str] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2 or parts[0] != "PRE":
            continue
        name = parts[1].rstrip("/")
        if _PROJECT_RE.match(name):
            projects.append(name)
    return sorted(projects)


def list_local_projects(directory: Path) -> Set[str]:
    """List project folder names under a local directory."""
    if not directory.is_dir():
        return set()
    return {
        entry.name
        for entry in directory.iterdir()
        if entry.is_dir() and _PROJECT_RE.match(entry.name)
    }


def list_s3_project_files(
    s3_prefix: str, project: str, aws_profile: str | None
) -> List[str]:
    """List object basenames under one project prefix on S3."""
    bucket, prefix = _parse_s3_prefix(s3_prefix)
    uri = f"s3://{bucket}/{prefix}{project}/"
    cmd = ["aws", "s3", "ls", uri]
    if aws_profile:
        cmd.extend(["--profile", aws_profile])

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    files: List[str] = []
    for line in result.stdout.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4 or parts[0] == "PRE":
            continue
        files.append(parts[-1])
    return sorted(files)


def sync_project(
    s3_prefix: str,
    local_dir: Path,
    project: str,
    aws_profile: str | None,
    dry_run: bool,
) -> None:
    """Download one project from S3, one object at a time.

    We intentionally avoid ``aws s3 sync`` / ``cp --recursive`` here. On the
    AIchor PVC those commands open many parallel temp files (``*.ipc.<hex>``)
    under the destination and stall with no error. Single-file
    ``aws s3 cp`` calls complete reliably; ``max_concurrent_requests`` only
    limits in-flight S3 tasks, not how many temp files get opened.
    """
    bucket, prefix = _parse_s3_prefix(s3_prefix)
    source_prefix = f"s3://{bucket}/{prefix}{project}/"
    dest_dir = local_dir / project
    dest_dir.mkdir(parents=True, exist_ok=True)

    files = list_s3_project_files(s3_prefix, project, aws_profile)
    logger.info("Project %s: %d file(s) on S3", project, len(files))

    for idx, filename in enumerate(files, start=1):
        source = f"{source_prefix}{filename}"
        dest = dest_dir / filename
        if dest.is_file():
            logger.info("Skipping existing %s (%d/%d)", filename, idx, len(files))
            continue

        cmd = ["aws", "s3", "cp", source, str(dest)]
        if aws_profile:
            cmd.extend(["--profile", aws_profile])

        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        logger.info("=== %s %s (%d/%d) ===", timestamp, filename, idx, len(files))

        if dry_run:
            logger.info("[DRY-RUN] Would run: %s", " ".join(cmd))
            continue

        subprocess.run(cmd, check=True)

    logger.info("Finished sync for %s", project)


@app.command()
def main(
    s3_prefix: str = S3_PREFIX_OPTION,
    local_raw_dir: Path = LOCAL_RAW_DIR_OPTION,
    processed_dir: Path = PROCESSED_DIR_OPTION,
    aws_profile: str = AWS_PROFILE_OPTION,
    dry_run: bool = DRY_RUN_OPTION,
) -> None:
    """Download ACFM projects from S3 that are not yet local."""
    s3_projects = set(list_s3_projects(s3_prefix, aws_profile))
    local_have = list_local_projects(processed_dir) | list_local_projects(local_raw_dir)
    missing = sorted(s3_projects - local_have)

    logger.info(
        "S3 projects=%d local=%d missing=%d",
        len(s3_projects),
        len(local_have),
        len(missing),
    )

    if not missing:
        logger.info("Nothing to sync.")
        return

    local_raw_dir.mkdir(parents=True, exist_ok=True)

    for idx, project in enumerate(missing, start=1):
        logger.info("Syncing project %s (%d/%d)", project, idx, len(missing))
        sync_project(s3_prefix, local_raw_dir, project, aws_profile, dry_run)

    logger.info("Done: %d project(s) synced", len(missing))


if __name__ == "__main__":
    app()
