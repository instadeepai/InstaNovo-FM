"""Project-disjoint splitting for linear probe evaluation.

Two entry points:

* ``project_disjoint_split`` — model-aligned mode.  Requires separate
  train/val/test embedding pools (one per model split); probe-train draws
  only from model-train samples, etc.

* ``flat_project_disjoint_split`` — flat-pool mode.  Works from a single
  embedding array (e.g. embeddings extracted by a standalone baseline
  script).  Projects are assigned to exactly one of train/val/test; all
  samples from a project are drawn from the flat pool into their assigned
  split.

Both share the same train-priority assignment and per-project capping logic.
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def project_disjoint_split(
    splits: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    project_key: str = "search_project",
    train_samples: int = 100_000,
    val_samples: int = 10_000,
    test_samples: int = 10_000,
    max_per_project_frac: float = 0.15,
    random_state: int = 42,
    min_project_samples: int = 5,
) -> Dict[str, Any]:
    """Split embeddings by project with no project overlap across splits.

    Each project is assigned to exactly one probe split. Probe-train only
    uses samples from model-train, probe-val from model-val, probe-test
    from model-test. Assignment is train-priority: val and test get just
    enough projects to hit their targets, train gets the rest.

    Args:
        splits: Dict mapping split names to (embeddings, metadata) tuples.
            Expected keys: "train", "val"/"valid", "test".
        project_key: Metadata key containing project identifiers.
        train_samples: Target number of training samples.
        val_samples: Target number of validation samples.
        test_samples: Target number of test samples.
        max_per_project_frac: Maximum fraction of a split's target that any
            single project can contribute (prevents one project dominating).
        random_state: Random seed for reproducibility.
        min_project_samples: Minimum total samples (across all model splits)
            for a project to be included.

    Returns:
        Dict with keys "train", "val", "test" each containing:
            - "embeddings": np.ndarray of shape (N, D)
            - "metadata": dict of np.ndarray
            - "projects": list of project names in this split
        Plus "assignment_info" with summary statistics.

    Raises:
        ValueError: If project_key is missing from all splits' metadata.
    """
    # Normalise split keys: accept "valid" as alias for "val"
    splits = _normalise_split_keys(splits)

    # Validate project_key exists in at least one split
    _validate_project_key(splits, project_key)

    # Build per-project, per-split sample counts and index maps
    project_info, n_dropped = _build_project_index(splits, project_key, min_project_samples)

    if n_dropped > 0:
        total = sum(len(emb) for emb, _ in splits.values())
        pct = 100.0 * n_dropped / total if total > 0 else 0
        logger.warning(
            f"Dropped {n_dropped} samples ({pct:.1f}%) with missing/invalid "
            f"'{project_key}' metadata."
        )

    if len(project_info) == 0:
        raise ValueError(
            f"No projects with >= {min_project_samples} samples found. "
            f"Cannot perform project-disjoint splitting."
        )

    # Train-priority assignment
    assignment = _train_priority_assignment(
        project_info,
        val_target=val_samples,
        test_target=test_samples,
        random_state=random_state,
        max_per_project_frac=max_per_project_frac,
        train_target=train_samples,
    )

    # Build the output splits
    rng = np.random.RandomState(random_state)
    result = {}
    targets = {"train": train_samples, "val": val_samples, "test": test_samples}

    for split_name, target_n in targets.items():
        assigned_projects = assignment[split_name]
        if not assigned_projects:
            logger.warning(f"No projects assigned to probe-{split_name}.")
            result[split_name] = {
                "embeddings": np.empty((0, splits["train"][0].shape[1])),
                "metadata": {},
                "projects": [],
            }
            continue

        # Gather indices for assigned projects from the corresponding model split
        emb, meta = splits[split_name]
        indices = _gather_project_indices(
            meta, project_key, assigned_projects
        )

        if len(indices) == 0:
            logger.warning(
                f"Projects assigned to probe-{split_name} have no samples "
                f"in model-{split_name}."
            )
            result[split_name] = {
                "embeddings": np.empty((0, emb.shape[1])),
                "metadata": {},
                "projects": list(assigned_projects),
            }
            continue

        # Apply project-capped sampling to hit the target
        sampled_indices = _project_capped_sample(
            meta=meta,
            indices=indices,
            project_key=project_key,
            target_n=target_n,
            max_per_project_frac=max_per_project_frac,
            rng=rng,
        )

        # Index metadata arrays by sampled indices; skip non-indexable values
        sampled_meta = {}
        for k, v in meta.items():
            try:
                sampled_meta[k] = v[sampled_indices]
            except (TypeError, IndexError, KeyError):
                logger.debug(f"Skipping non-indexable metadata key '{k}' (type={type(v).__name__})")

        result[split_name] = {
            "embeddings": emb[sampled_indices],
            "metadata": sampled_meta,
            "projects": list(assigned_projects),
        }

    # Summary info
    result["assignment_info"] = {
        "n_projects_total": len(project_info),
        "n_projects_per_split": {
            s: len(assignment[s]) for s in ("train", "val", "test")
        },
        "n_samples_per_split": {
            s: len(result[s]["embeddings"]) for s in ("train", "val", "test")
        },
        "n_samples_dropped_invalid_project": n_dropped,
        "targets": targets,
    }

    # Log summary
    info = result["assignment_info"]
    for s in ("train", "val", "test"):
        actual = info["n_samples_per_split"][s]
        target = targets[s]
        n_proj = info["n_projects_per_split"][s]
        status = "OK" if actual >= target else "BELOW TARGET"
        logger.info(
            f"Probe-{s}: {actual:,} samples from {n_proj} projects "
            f"(target: {target:,}) [{status}]"
        )

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _normalise_split_keys(
    splits: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
) -> Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Normalise split keys: accept 'valid' as alias for 'val'.

    Raises:
        ValueError: If both 'val' and 'valid' are present (ambiguous).
    """
    if "val" in splits and "valid" in splits:
        raise ValueError(
            "Both 'val' and 'valid' keys are present in splits. "
            "Use one or the other — they are aliases for the same split."
        )
    normalised = {}
    for key, value in splits.items():
        norm_key = "val" if key == "valid" else key
        normalised[norm_key] = value
    return normalised


def _validate_project_key(
    splits: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    project_key: str,
) -> None:
    """Check that project_key exists in at least one split's metadata."""
    found = any(project_key in meta for _, meta in splits.values())
    if not found:
        available_keys = set()
        for _, meta in splits.values():
            available_keys.update(meta.keys())
        raise ValueError(
            f"Project key '{project_key}' not found in any split's metadata. "
            f"Available keys: {sorted(available_keys)}"
        )


def _is_valid_project(value: Any) -> bool:
    """Check if a project label is valid (non-null, non-empty)."""
    if value is None:
        return False
    s = str(value).strip().lower()
    return s not in ("", "none", "nan", "unknown", "null")


def _build_project_index(
    splits: Dict[str, Tuple[np.ndarray, Dict[str, np.ndarray]]],
    project_key: str,
    min_project_samples: int,
) -> Tuple[Dict[str, Dict[str, int]], int]:
    """Build per-project, per-split sample counts.

    Uses vectorised numpy operations (np.unique) for large splits.
    Unknown split keys (anything other than "train", "val", "test") are
    silently ignored so callers are not broken by holdout/extra splits.

    Returns:
        project_info: {project_name: {"train": count, "val": count, "test": count}}
        n_dropped: number of samples with invalid project labels
    """
    _known_splits = {"train", "val", "test"}
    project_info: Dict[str, Dict[str, int]] = {}
    n_dropped = 0

    for split_name, (emb, meta) in splits.items():
        if split_name not in _known_splits:
            logger.debug(f"Ignoring unknown split '{split_name}' in project index build.")
            continue

        if project_key not in meta:
            # This split has no project info — all samples dropped
            n_dropped += len(emb)
            continue

        projects = meta[project_key]
        # Vectorised validity check and string normalisation
        str_projects = np.array(
            [str(p).strip() if p is not None else "" for p in projects], dtype=object
        )
        valid_mask = np.array([_is_valid_project(p) for p in projects], dtype=bool)
        n_dropped += int((~valid_mask).sum())

        valid_projs = str_projects[valid_mask]
        if len(valid_projs) == 0:
            continue

        # Count per project using np.unique (O(n log n) vs O(n * k))
        unique_projs, counts = np.unique(valid_projs, return_counts=True)
        for proj, count in zip(unique_projs, counts):
            proj = str(proj)
            if proj not in project_info:
                project_info[proj] = {"train": 0, "val": 0, "test": 0}
            project_info[proj][split_name] += int(count)

    # Filter by min_project_samples (total across all splits)
    filtered = {}
    for proj, counts in project_info.items():
        total = sum(counts.values())
        if total >= min_project_samples:
            filtered[proj] = counts

    n_filtered_out = len(project_info) - len(filtered)
    if n_filtered_out > 0:
        logger.info(
            f"Filtered out {n_filtered_out} projects with < {min_project_samples} "
            f"total samples."
        )

    return filtered, n_dropped


def _train_priority_assignment(
    project_info: Dict[str, Dict[str, int]],
    val_target: int,
    test_target: int,
    random_state: int,
    max_per_project_frac: float = 0.15,
    train_target: int = 0,
) -> Dict[str, list]:
    """Assign projects to splits with train-priority and an escalating per-project cap.

    Each project's contribution toward a split's sample target is capped at
    ``max_per_project_frac * target``.  This prevents a single large project
    from consuming an entire val/test budget, ensuring better cross-project
    diversity.

    The cap is escalated through [max_per_project_frac, 0.5, 1.0] until all
    three targets (val, test, and optionally train) can be satisfied.  Higher
    fracs assign fewer projects to val/test, freeing more for train.  Without
    this escalation, the lowest frac consumes the most projects for val/test,
    leaving train with only "orphan" projects (those with 0 val/test samples)
    that typically have very few train samples — causing train to be severely
    under-filled even when enough total data exists.

    1. Assign enough projects to val to hit val_target (sorted by val count desc).
    2. Assign enough projects to test to hit test_target (sorted by test count desc).
    3. Everything else goes to train.
    """
    rng = np.random.RandomState(random_state)

    # Build the ordered list of fracs to try, always ending at 1.0
    fracs_to_try = sorted(set([max_per_project_frac, 0.5, 1.0]))

    best_assignment: Dict[str, list] = {"train": [], "val": [], "test": []}
    best_val_achieved = 0
    best_test_achieved = 0
    best_train_achieved = 0
    chosen_frac = max_per_project_frac

    for frac in fracs_to_try:
        val_cap = max(1, int(val_target * frac))
        test_cap = max(1, int(test_target * frac))

        attempt: Dict[str, list] = {"train": [], "val": [], "test": []}
        remaining = set(project_info.keys())

        # Step 1: Assign to val
        val_candidates = sorted(
            remaining,
            key=lambda p: project_info[p]["val"],
            reverse=True,
        )
        val_cumulative = 0
        for proj in val_candidates:
            if val_cumulative >= val_target:
                break
            count_in_val = project_info[proj]["val"]
            if count_in_val == 0:
                continue
            attempt["val"].append(proj)
            remaining.discard(proj)
            val_cumulative += min(count_in_val, val_cap)

        # Step 2: Assign to test
        test_candidates = sorted(
            remaining,
            key=lambda p: project_info[p]["test"],
            reverse=True,
        )
        test_cumulative = 0
        for proj in test_candidates:
            if test_cumulative >= test_target:
                break
            count_in_test = project_info[proj]["test"]
            if count_in_test == 0:
                continue
            attempt["test"].append(proj)
            remaining.discard(proj)
            test_cumulative += min(count_in_test, test_cap)

        # Step 3: Everything else goes to train
        attempt["train"] = list(remaining)
        train_cumulative = sum(project_info[p]["train"] for p in attempt["train"])

        best_assignment = attempt
        best_val_achieved = val_cumulative
        best_test_achieved = test_cumulative
        best_train_achieved = train_cumulative
        chosen_frac = frac

        val_test_met = val_cumulative >= val_target and test_cumulative >= test_target
        train_met = train_target <= 0 or train_cumulative >= train_target
        if val_test_met and train_met:
            break  # All targets met — no need to escalate further

    if chosen_frac > max_per_project_frac:
        logger.info(
            f"Escalated max_per_project_frac from {max_per_project_frac} to {chosen_frac} "
            f"to meet sample targets"
        )

    if best_val_achieved < val_target:
        logger.warning(
            f"Could only assign {best_val_achieved:,} val samples "
            f"(target: {val_target:,}). Not enough projects with val data."
        )
    if best_test_achieved < test_target:
        logger.warning(
            f"Could only assign {best_test_achieved:,} test samples "
            f"(target: {test_target:,}). Not enough projects with test data."
        )
    if train_target > 0 and best_train_achieved < train_target:
        logger.warning(
            f"Could only assign {best_train_achieved:,} train samples "
            f"(target: {train_target:,}). Not enough projects with train-only data."
        )

    # Shuffle within each split for consistency
    for split_name in best_assignment:
        rng.shuffle(best_assignment[split_name])

    logger.info(
        f"Project assignment (cap={chosen_frac:.0%}): "
        f"train={len(best_assignment['train'])} projects, "
        f"val={len(best_assignment['val'])} projects, "
        f"test={len(best_assignment['test'])} projects"
    )

    return best_assignment


def _gather_project_indices(
    meta: Dict[str, np.ndarray],
    project_key: str,
    assigned_projects: list,
) -> np.ndarray:
    """Get indices of samples belonging to assigned projects.

    Uses np.isin for vectorised membership testing.
    """
    if project_key not in meta:
        return np.array([], dtype=int)

    projects = meta[project_key]
    # Normalise to string array; invalid projects map to ""
    str_projects = np.array(
        [str(p).strip() if _is_valid_project(p) else "" for p in projects], dtype=object
    )
    assigned_arr = np.array(list(assigned_projects), dtype=object)
    mask = np.isin(str_projects, assigned_arr)
    return np.where(mask)[0]


def flat_project_disjoint_split(
    embeddings: np.ndarray,
    metadata: Dict[str, np.ndarray],
    project_key: str = "search_project",
    train_samples: int = 100_000,
    val_samples: int = 10_000,
    test_samples: int = 10_000,
    max_per_project_frac: float = 0.15,
    random_state: int = 42,
    min_project_samples: int = 5,
) -> Dict[str, Any]:
    """Project-disjoint split from a flat (unsplit) embedding pool.

    Unlike ``project_disjoint_split``, no model-aligned sub-pools are needed.
    Each project is assigned to exactly one of train/val/test; all samples for
    that project are drawn from the single flat pool.  The project assignment
    and per-project capping logic is identical to ``project_disjoint_split``.

    The return format is the same as ``project_disjoint_split`` and is
    directly usable with ``LinearProbeTask.run(..., pre_filtered=True)``.

    Args:
        embeddings: (N, D) array of all embeddings.
        metadata: Dict of metadata arrays, each of length N.
        project_key: Metadata key containing project identifiers.
        train_samples: Target number of training samples.
        val_samples: Target number of validation samples.
        test_samples: Target number of test samples.
        max_per_project_frac: Maximum fraction of a split's target from one project.
        random_state: Random seed for reproducibility.
        min_project_samples: Minimum total samples for a project to be included.

    Returns:
        Dict with keys "train", "val", "test" each containing:
            - "embeddings": np.ndarray (N_split, D)
            - "metadata": dict of np.ndarray
            - "projects": list of project names assigned to this split
        Plus "assignment_info" with summary statistics.

    Raises:
        ValueError: If project_key is missing from metadata or no valid projects found.
    """
    if project_key not in metadata:
        raise ValueError(
            f"Project key '{project_key}' not found in metadata. "
            f"Available keys: {sorted(metadata.keys())}"
        )

    projects = metadata[project_key]
    n = len(embeddings)

    valid_mask = np.array([_is_valid_project(p) for p in projects], dtype=bool)
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        pct = 100.0 * n_dropped / n if n > 0 else 0.0
        logger.warning(
            f"Dropped {n_dropped:,} samples ({pct:.1f}%) with missing/invalid "
            f"'{project_key}' metadata."
        )

    str_projects = np.array(
        [str(p).strip() if p is not None else "" for p in projects], dtype=object
    )
    valid_indices = np.where(valid_mask)[0]
    valid_str_projs = str_projects[valid_mask]

    if len(valid_str_projs) == 0:
        raise ValueError(f"No valid samples with project key '{project_key}' found.")

    unique_projs, counts = np.unique(valid_str_projs, return_counts=True)
    keep_mask = counts >= min_project_samples
    n_filtered = int((~keep_mask).sum())
    if n_filtered > 0:
        logger.info(f"Filtered out {n_filtered} projects with < {min_project_samples} samples.")
    unique_projs = unique_projs[keep_mask]
    counts = counts[keep_mask]

    if len(unique_projs) == 0:
        raise ValueError(
            f"No projects with >= {min_project_samples} samples found. "
            f"Cannot perform project-disjoint splitting."
        )

    # Replicate total counts across all three split slots so
    # _train_priority_assignment can fill val/test from total-count budgets.
    project_info: Dict[str, Dict[str, int]] = {
        str(proj): {"train": int(count), "val": int(count), "test": int(count)}
        for proj, count in zip(unique_projs, counts)
    }

    assignment = _train_priority_assignment(
        project_info,
        val_target=val_samples,
        test_target=test_samples,
        random_state=random_state,
        max_per_project_frac=max_per_project_frac,
    )

    # Build project → flat-pool indices lookup
    valid_proj_set = set(project_info.keys())
    proj_to_indices: Dict[str, list] = {}
    for idx in valid_indices:
        proj = str_projects[idx]
        if proj in valid_proj_set:
            if proj not in proj_to_indices:
                proj_to_indices[proj] = []
            proj_to_indices[proj].append(int(idx))

    rng = np.random.RandomState(random_state)
    result: Dict[str, Any] = {}
    targets = {"train": train_samples, "val": val_samples, "test": test_samples}

    for split_name, target_n in targets.items():
        assigned_projects = assignment[split_name]
        if not assigned_projects:
            logger.warning(f"No projects assigned to probe-{split_name}.")
            result[split_name] = {
                "embeddings": np.empty((0, embeddings.shape[1])),
                "metadata": {},
                "projects": [],
            }
            continue

        indices = np.array(
            [idx for proj in assigned_projects for idx in proj_to_indices.get(proj, [])],
            dtype=int,
        )

        if len(indices) == 0:
            logger.warning(f"No samples found for projects assigned to probe-{split_name}.")
            result[split_name] = {
                "embeddings": np.empty((0, embeddings.shape[1])),
                "metadata": {},
                "projects": list(assigned_projects),
            }
            continue

        sampled_indices = _project_capped_sample(
            meta=metadata,
            indices=indices,
            project_key=project_key,
            target_n=target_n,
            max_per_project_frac=max_per_project_frac,
            rng=rng,
        )

        result[split_name] = {
            "embeddings": embeddings[sampled_indices],
            "metadata": {k: v[sampled_indices] for k, v in metadata.items()},
            "projects": list(assigned_projects),
        }

    result["assignment_info"] = {
        "mode": "flat_project_disjoint",
        "n_projects_total": len(project_info),
        "n_projects_per_split": {s: len(assignment[s]) for s in ("train", "val", "test")},
        "n_samples_per_split": {s: len(result[s]["embeddings"]) for s in ("train", "val", "test")},
        "n_samples_dropped_invalid_project": n_dropped,
        "targets": targets,
    }

    info = result["assignment_info"]
    for s in ("train", "val", "test"):
        actual = info["n_samples_per_split"][s]
        target = targets[s]
        n_proj = info["n_projects_per_split"][s]
        status = "OK" if actual >= target else "BELOW TARGET"
        logger.info(
            f"Flat probe-{s}: {actual:,} samples from {n_proj} projects "
            f"(target: {target:,}) [{status}]"
        )

    return result


def _project_capped_sample(
    meta: Dict[str, np.ndarray],
    indices: np.ndarray,
    project_key: str,
    target_n: int,
    max_per_project_frac: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Sample indices with per-project caps.

    Args:
        meta: Full metadata dict for this model split.
        indices: Valid indices (samples belonging to assigned projects).
        project_key: Metadata key for project.
        target_n: Target number of samples.
        max_per_project_frac: Max fraction of target_n from any single project.
        rng: Random state for sampling.

    Returns:
        Array of sampled indices.
    """
    if len(indices) <= target_n:
        # Not enough data — use everything
        if len(indices) < target_n:
            logger.info(
                f"Pool has {len(indices):,} samples, below target {target_n:,}. "
                f"Using all available."
            )
        return indices

    projects = meta[project_key]
    max_per_project = max(1, int(target_n * max_per_project_frac))

    # Group indices by project
    project_indices: Dict[str, list] = {}
    for idx in indices:
        proj = str(projects[idx]).strip()
        if proj not in project_indices:
            project_indices[proj] = []
        project_indices[proj].append(idx)

    # First pass: cap each project
    capped: Dict[str, np.ndarray] = {}
    total_capped = 0
    for proj, proj_idxs in project_indices.items():
        arr = np.array(proj_idxs)
        if len(arr) > max_per_project:
            rng.shuffle(arr)
            arr = arr[:max_per_project]
        capped[proj] = arr
        total_capped += len(arr)

    if total_capped <= target_n:
        # After capping we have enough or less — use all capped
        return np.concatenate(list(capped.values()))

    # Second pass: proportionally scale down to hit target_n exactly.
    # Shuffle all capped indices and take target_n — avoids both over- and
    # under-shoot caused by fractional rounding in the deficit accumulator.
    all_capped = np.concatenate(list(capped.values()))
    rng.shuffle(all_capped)
    return all_capped[:target_n]
