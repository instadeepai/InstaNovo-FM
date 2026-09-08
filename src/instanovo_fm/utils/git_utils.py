"""Standalone Git/VCS utilities.

No imports of other instanovo modules are allowed in this file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import cast


def get_current_git_branch() -> str | None:
    """Return the current Git branch name, or None if unavailable.

    Checks (in order):
    1. ``VCS_REF_NAME`` env var (set by AIchor in containers)
    2. GitPython (local development)

    Returns:
        Branch name (e.g. "149-mlflow") or None. Slashes in branch names are
        replaced with "-" for use as an MLFlow experiment name.
    """
    # AIchor injects VCS_REF_NAME_SLUG (already slugified) into the container
    vcs_ref = os.environ.get("VCS_REF_NAME_SLUG", "").strip()
    if vcs_ref:
        return vcs_ref

    try:
        from git import InvalidGitRepositoryError, Repo

        repo = Repo(Path(__file__).resolve().parent, search_parent_directories=True)
        if repo.head.is_detached:
            return None
        branch = repo.active_branch.name
        if not branch:
            return None
        return cast(str, branch.replace("/", "-"))
    except (InvalidGitRepositoryError, TypeError, ValueError, OSError):
        return None


def get_user() -> str:
    """Return the name of the user who started the run.

    Checks (in order): VCS_AUTHOR_NAME, VCS_AUTHOR_EMAIL (AIchor),
    git config user.name/email (local), OS USER/USERNAME.
    """
    user = os.environ.get("VCS_AUTHOR_NAME", "").strip() or os.environ.get("VCS_AUTHOR_EMAIL", "").strip()
    if user:
        return user

    try:
        from git import InvalidGitRepositoryError, Repo

        repo = Repo(Path(__file__).resolve().parent, search_parent_directories=True)
        reader = repo.config_reader()
        name: str = reader.get_value("user", "name", default="")
        email: str = reader.get_value("user", "email", default="")
        if name and email:
            return f"{name} <{email}>"
        if name or email:
            return name or email
    except (InvalidGitRepositoryError, Exception):
        pass

    return os.environ.get("USER", "") or os.environ.get("USERNAME", "") or "unknown"


def get_dataset_name_from_path(path: str, prefix: str = "") -> str:
    """Extract a meaningful dataset name from a file path.

    Uses the filename (without extension) or parent folder name to create
    an informative dataset name for tracking.

    Examples:
        - "data/mouse/dataset-mus-musculus-train-0000-0001.parquet" -> "train_mus-musculus"
        - "data/nine-species/train.parquet" -> "train_nine-species"
        - "s3://bucket/phospho/data.parquet" -> "train_phospho"

    Args:
        path: File path or URL to the dataset
        prefix: Optional prefix (e.g., "train", "valid")

    Returns:
        A meaningful dataset name like "train_mus-musculus" or "valid_nine-species"
    """
    import re

    path_str = str(path)

    if isinstance(path, (list, tuple)) and len(path) > 0:
        path_str = str(path[0])

    p = Path(path_str)
    filename = p.stem

    generic_names = {"train", "valid", "test", "data", "dataset", "parquet", "csv", "random"}

    if filename.lower() in generic_names or filename.startswith("part-"):
        parent_name = p.parent.name
        if parent_name and parent_name.lower() not in generic_names:
            name = parent_name
        else:
            grandparent_name = p.parent.parent.name
            name = grandparent_name if grandparent_name else filename
    else:
        name = filename
        name = re.sub(r"-\d{4,}(-\d{4,})*$", "", name)
        name = re.sub(r"^dataset-", "", name)
        name = re.sub(r"-(train|valid|test)$", "", name)

    if not name:
        name = "dataset"

    if prefix:
        return f"{prefix}_{name}"
    return name
