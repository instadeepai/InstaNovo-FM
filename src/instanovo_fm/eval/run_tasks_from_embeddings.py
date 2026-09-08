"""Run evaluation tasks on pre-generated embeddings saved in the foundation model HDF5 format.

This script loads embeddings produced by any of the baseline extraction scripts
(extract_instanovo_embeddings.py, extract_casanovo_embeddings.py) or by the
foundation model's own embedding_io.generate(), and runs one or more eval tasks
from the TASK_REGISTRY without needing a model or dataloader.

Usage:
    uv run python -m instanovo_fm.eval.run_tasks_from_embeddings \
        --embeddings_dir /path/to/embeddings \
        --tasks embeddingstatisticstask \
        [--tasks duplicateretrievaltask] \
        [--output_dir /path/to/results]   # defaults to --embeddings_dir

    # List available tasks
    uv run python -m instanovo_fm.eval.run_tasks_from_embeddings --list_tasks

Notes:
    - Tasks that require_model=True are skipped (need a live model instance).
    - Tasks that require_multi_split=True (e.g. LinearProbeTask) are handled as follows:
        * use_project_split=True (default): flat project-disjoint splits are built from the
          embeddings file using the 'search_project' metadata field.  The probe runs if that
          field is present; otherwise the task is skipped with a warning.
        * use_project_split=False: legacy random-split mode (no project key needed).
    - Results are written to <output_dir>/<task_name>/task_results.json and task_summary.json.
    - If --output_dir is not given, results land inside --embeddings_dir.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from instanovo.__init__ import console
from instanovo_fm.eval import embedding_io
from instanovo_fm.eval.embed_eval_tasks import TASK_REGISTRY, get_task
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.s3 import S3FileHandler

# Trigger auto-registration of all tasks by importing the package.
import instanovo_fm.eval.embed_eval_tasks  # noqa: F401 — side-effect import
import importlib, pkgutil, instanovo_fm.eval.embed_eval_tasks as _pkg

for _importer, _modname, _ispkg in pkgutil.walk_packages(
    path=_pkg.__path__,
    prefix=_pkg.__name__ + ".",
    onerror=lambda x: None,
):
    try:
        importlib.import_module(_modname)
    except Exception:
        pass

logger = ColorLog(console, __name__).logger


def _upload_to_s3(output_dir: Path) -> None:
    """Upload all files in output_dir to S3 if running on AIchor."""
    if not S3FileHandler._aichor_enabled():
        return
    s3 = S3FileHandler()
    files = [f for f in output_dir.rglob("*") if f.is_file()]
    logger.info(f"Uploading {len(files)} files to S3...")
    for f in files:
        s3_path = S3FileHandler.convert_to_s3_output(str(f))
        s3.upload(str(f), s3_path)
        logger.info(f"Uploaded {f.name} → {s3_path}")
    logger.info("S3 upload complete")


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-serialisable Python types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(i) for i in obj]
    return obj


def _save_task_results(
    task_name: str,
    task_results: Dict[str, Any],
    task_output_dir: Path,
) -> None:
    """Write task_results.json and task_summary.json to task_output_dir."""
    task_output_dir.mkdir(parents=True, exist_ok=True)

    task_summary: Dict[str, Any] = {
        "task_name": task_name,
        "timestamp": datetime.now().isoformat(),
        "execution_time": task_results.get("execution_time", 0),
        "success": "error" not in task_results,
    }

    if "error" in task_results:
        task_summary["error"] = task_results["error"]
    else:
        try:
            task_class = get_task(task_name)
            task_summary.update(task_class().get_loggable_metrics(task_results))
        except Exception:
            pass

        with open(task_output_dir / "task_results.json", "w") as f:
            json.dump(_make_serializable(task_results), f, indent=2)

    with open(task_output_dir / "task_summary.json", "w") as f:
        json.dump(_make_serializable(task_summary), f, indent=2)


def run(
    embeddings_dir: str,
    tasks: List[str],
    output_dir: Optional[str] = None,
    task_configs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Load embeddings and run the requested evaluation tasks.

    Args:
        embeddings_dir: Directory containing embeddings.h5 and index.faiss.
        tasks: List of task names to run (case-insensitive).
        output_dir: Directory to write results. Defaults to embeddings_dir.
        task_configs: Optional dict mapping task name → kwargs passed to the task
            constructor. E.g. {"linearprobetask": {"use_project_split": true,
            "train_samples": 100000}}.  For LinearProbeTask, use_project_split=True
            (default) builds flat project-disjoint splits from the embeddings file;
            use_project_split=False falls back to legacy random splitting.

    Returns:
        Dict mapping task name → task results dict.
    """
    task_configs = task_configs or {}
    out = Path(output_dir) if output_dir else Path(embeddings_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_dir}")
    embeddings, metadata, faiss_index = embedding_io.load(embeddings_dir)
    logger.info(
        f"Loaded {len(embeddings):,} embeddings of dim {embeddings.shape[1]} "
        f"with metadata keys: {list(metadata.keys())}"
    )

    all_results: Dict[str, Any] = {}

    for task_name in tasks:
        # Resolve task class
        try:
            task_class = get_task(task_name)
        except KeyError:
            logger.warning(
                f"Task '{task_name}' not found in registry. "
                f"Available: {sorted(set(TASK_REGISTRY.keys()))}"
            )
            continue

        cfg = task_configs.get(task_name, task_configs.get(task_name.lower(), {}))

        # Skip tasks that need a live model
        if getattr(task_class, "requires_model", False):
            logger.warning(
                f"Skipping '{task_name}': requires_model=True (needs a live model instance)"
            )
            continue

        # Tasks that need multi-split data: build flat project-disjoint splits when
        # use_project_split=True (default) or fall through to legacy random-split mode
        # when use_project_split=False.
        need_flat_splits = False
        if getattr(task_class, "requires_multi_split", False):
            if cfg.get("use_project_split", True) is not False:
                project_key = cfg.get("project_key", "search_project")
                if project_key not in metadata:
                    logger.warning(
                        f"Skipping '{task_name}': use_project_split=True but "
                        f"'{project_key}' not found in metadata. "
                        f"Pass use_project_split=false in task_configs to use random splits."
                    )
                    continue
                need_flat_splits = True

        logger.info(f"Running task: {task_name}")
        task_output_dir = out / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        try:
            flat_splits = None
            if need_flat_splits:
                from instanovo_fm.eval.probe_splitting import flat_project_disjoint_split
                project_key = cfg.get("project_key", "search_project")
                logger.info(f"Building flat project-disjoint splits (project_key='{project_key}')...")
                split_result = flat_project_disjoint_split(
                    embeddings,
                    metadata,
                    project_key=project_key,
                    train_samples=cfg.get("train_samples", 100_000),
                    val_samples=cfg.get("val_samples", 10_000),
                    test_samples=cfg.get("test_samples", 10_000),
                )
                flat_splits = {
                    s: (split_result[s]["embeddings"], split_result[s]["metadata"])
                    for s in ("train", "val", "test")
                }

            task = task_class(output_dir=str(task_output_dir), **cfg)
            if flat_splits is not None:
                task_results = task.run(
                    embeddings, metadata, faiss_index,
                    splits=flat_splits, pre_filtered=True,
                )
            else:
                task_results = task.run(embeddings, metadata, faiss_index)
            task_results["execution_time"] = time.time() - start
            task_results["output_dir"] = str(task_output_dir)
            _save_task_results(task_name, task_results, task_output_dir)
            logger.info(
                f"  [OK] {task_name} completed in {task_results['execution_time']:.2f}s "
                f"— results in {task_output_dir}"
            )
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(f"  [FAIL] {task_name} failed after {elapsed:.2f}s: {exc}")
            task_results = {"error": str(exc), "execution_time": elapsed}
            _save_task_results(task_name, task_results, task_output_dir)

        all_results[task_name] = task_results

    _upload_to_s3(out)
    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run eval tasks on pre-generated embeddings (HDF5 + FAISS format)"
    )
    parser.add_argument(
        "--embeddings_dir", required=False,
        help="Directory containing embeddings.h5 and index.faiss",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=["embeddingstatisticstask"],
        metavar="TASK",
        help=(
            "One or more task names to run (case-insensitive). "
            "Default: embeddingstatisticstask. "
            "Use --list_tasks to see all available tasks."
        ),
    )
    parser.add_argument(
        "--output_dir", default=None,
        help="Directory to write results (default: same as --embeddings_dir)",
    )
    parser.add_argument(
        "--task_configs", default=None,
        help=(
            "JSON string mapping task name → kwargs, e.g. "
            "'{\"linearprobetask\": {\"use_project_split\": true, "
            "\"train_samples\": 100000, \"val_samples\": 10000, \"test_samples\": 10000}}' "
            "(use_project_split=true requires 'search_project' in the embeddings metadata; "
            "use_project_split=false falls back to legacy random splitting)"
        ),
    )
    parser.add_argument(
        "--list_tasks", action="store_true",
        help="Print all registered task names and exit",
    )
    args = parser.parse_args()

    if args.list_tasks:
        unique = sorted(set(TASK_REGISTRY.keys()))
        print("Available tasks:")
        for t in unique:
            print(f"  {t}")
        return

    if not args.embeddings_dir:
        parser.error("--embeddings_dir is required unless --list_tasks is used")

    task_configs = json.loads(args.task_configs) if args.task_configs else {}
    run(
        embeddings_dir=args.embeddings_dir,
        tasks=args.tasks,
        output_dir=args.output_dir,
        task_configs=task_configs,
    )


if __name__ == "__main__":
    main()
