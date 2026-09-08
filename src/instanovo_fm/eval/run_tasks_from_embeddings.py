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
import importlib
import json
import pkgutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Trigger auto-registration of all tasks by importing the package.
import instanovo_fm.eval.embed_eval_tasks  # noqa: F401 — side-effect import
import instanovo_fm.eval.embed_eval_tasks as _pkg
from instanovo.__init__ import console
from instanovo_fm.eval import embedding_io
from instanovo_fm.eval.embed_eval_tasks import TASK_REGISTRY, get_task
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.s3 import S3FileHandler

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


def _load_eval_task_configs(config_name: str) -> Dict[str, Dict[str, Any]]:
    """Load the ``task_configs`` block from ``instanovo/configs/evaluation/<config_name>.yaml``.

    This sources the *canonical* evaluation task configuration — the same file the
    foundation-model eval composes via ``--config-name foundational`` (``evaluation: default``).
    Sourcing it directly guarantees that external embeddings (Casanovo/InstaNovo) are scored
    with byte-identical task settings (k-values, conditional subsets, probe split mode, …),
    rather than a hand-written JSON that can drift from the FM protocol.

    Args:
        config_name: Name of the evaluation config (without ``.yaml``), e.g. ``"default"``.

    Returns:
        Dict mapping task name → constructor kwargs.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    from omegaconf import OmegaConf

    from instanovo_fm.utils.hydra_config import FM_CONFIG_DIR

    cfg_path = FM_CONFIG_DIR / "evaluation" / f"{config_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    task_cfgs = cfg.get("task_configs", {})
    return OmegaConf.to_container(task_cfgs, resolve=True) or {}


def _load_dataset_paths(dataset_name: str) -> Dict[str, Optional[str]]:
    """Load the train/valid/test parquet paths from ``configs/dataset/<dataset_name>.yaml``.

    This makes the dataset config the single source of truth for the baseline-extraction
    scripts (mirroring Hydra ``dataset=<dataset_name>`` in the FM runs), so the parquet
    paths can never silently drift from a hand-copied shell-script default.

    Args:
        dataset_name: Name of the dataset config (without ``.yaml``), e.g. ``"lcfm"`` or
            ``"mcfm_local"``.

    Returns:
        Dict with keys ``train_path`` / ``valid_path`` / ``test_path`` / ``data_path`` (value
        ``None`` for any key the config does not define). ``data_path`` is the single-path
        alternative: when a config sets it (and omits the three split paths), the baseline scripts
        split it into sequence-disjoint train/valid/test partitions on the fly.

    Raises:
        FileNotFoundError: If the dataset config does not exist.
    """
    from omegaconf import OmegaConf

    from instanovo_fm.utils.hydra_config import FM_CONFIG_DIR

    cfg_path = FM_CONFIG_DIR / "dataset" / f"{dataset_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    return {split: cfg.get(split) for split in ("train_path", "valid_path", "test_path", "data_path")}


def _load_foundation_max_mz(config_name: str = "foundation_base") -> float:
    """Load ``max_mz`` from a model config (default the foundation-model base config).

    The baseline extractors feed spectra through the foundation model's
    ``FoundationalDataProcessor`` (which normalises m/z by dividing by ``max_mz``) and then
    denormalise by the same value before the encoder. Sourcing it from
    ``configs/model/<config_name>.yaml`` keeps both ends tied to the single value the
    foundation model uses, with no hardcoded ``2500.0`` to drift — read here in the orchestration
    layer and passed to the extractor, exactly like the dataset paths.

    Args:
        config_name: Name of the model config (without ``.yaml``), e.g. ``"foundation_base"``.

    Returns:
        The ``max_mz`` value as a float.

    Raises:
        FileNotFoundError: If the model config does not exist.
    """
    from omegaconf import OmegaConf

    from instanovo_fm.utils.hydra_config import FM_CONFIG_DIR

    cfg_path = FM_CONFIG_DIR / "model" / f"{config_name}.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    return float(cfg.get("max_mz", 2500.0))


def _merge_task_configs(
    base: Dict[str, Dict[str, Any]],
    override: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Deep-merge ``override`` task configs on top of ``base`` (override wins; lists replace)."""
    from omegaconf import OmegaConf

    merged = OmegaConf.merge(OmegaConf.create(base), OmegaConf.create(override))
    return OmegaConf.to_container(merged, resolve=True) or {}


# String values that represent a missing label after the HDF5 round-trip (which serialises
# Python ``None`` to the literal ``"None"``). Categorical metadata fields used by retrieval /
# IDs are excluded so we never corrupt peptide sequences or USIs.
_MISSING_LABEL_SENTINELS = frozenset({"", "none", "nan", "null"})
_NON_LABEL_KEYS = frozenset({"peptides", "peptide", "sequence", "seq", "usi", "header", "prediction_id", "sample_idx", "source_file"})


def _normalize_missing_labels(meta: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Restore HDF5-serialised missing labels (``"None"``/``""``/...) to Python ``None``.

    The FM evaluator runs the linear probe on in-memory metadata where missing categorical
    values are real ``None`` (the probe converts these to NaN and drops them). The baseline
    round-trips metadata through HDF5, which stores ``None`` as the string ``"None"`` — that
    would otherwise survive as a spurious probe class (e.g. a ``frag_type="None"`` class). This
    converts those sentinels back to ``None`` (object dtype) so categorical probe targets drop
    missing rows exactly like the FM. Sequence / ID fields are left untouched, and the
    conditional-retrieval subsets are unaffected (``None`` matches no frag-type/detector filter,
    same as the ``"None"`` string did).
    """

    def _is_missing(x: Any) -> bool:
        if x is None:
            return True
        if isinstance(x, bytes):
            x = x.decode("utf-8", "ignore")
        return isinstance(x, str) and x.strip().lower() in _MISSING_LABEL_SENTINELS

    for key, val in meta.items():
        if key in _NON_LABEL_KEYS:
            continue
        if isinstance(val, np.ndarray) and val.dtype == object:
            meta[key] = np.array([None if _is_missing(x) else x for x in val], dtype=object)
    return meta


def run(
    embeddings_dir: str,
    tasks: List[str],
    output_dir: Optional[str] = None,
    task_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    eval_config_name: Optional[str] = None,
    probe_split_dirs: Optional[Dict[str, str]] = None,
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
        eval_config_name: Optional name of an ``instanovo/configs/evaluation/<name>.yaml``
            config whose ``task_configs`` block is used as the base configuration (the
            canonical FM-eval settings). When set, ``task_configs`` is deep-merged on top as
            an override layer. Use ``"default"`` to match the foundation-model eval exactly.
        probe_split_dirs: Optional dict mapping split name ("train", "val"/"valid", "test")
            to a directory containing that split's embeddings.h5. When provided, multi-split
            tasks (e.g. LinearProbeTask) run on these splits directly with ``pre_filtered=True``
            — i.e. train on the train split, report on the test split — matching the FM
            evaluator's ``_run_pre_filtered`` protocol. This is the way to make the probe
            comparable to the foundation model (same train/valid/test data).

    Returns:
        Dict mapping task name → task results dict.
    """
    task_configs = task_configs or {}
    if eval_config_name:
        base_configs = _load_eval_task_configs(eval_config_name)
        logger.info(
            f"Sourcing canonical task configs from evaluation/{eval_config_name}.yaml "
            f"({len(base_configs)} task blocks); CLI task_configs override on top."
        )
        task_configs = _merge_task_configs(base_configs, task_configs)
    out = Path(output_dir) if output_dir else Path(embeddings_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    logger.info(f"Loading embeddings from {embeddings_dir}")
    embeddings, metadata, faiss_index = embedding_io.load(embeddings_dir)
    logger.info(f"Loaded {len(embeddings):,} embeddings of dim {embeddings.shape[1]} with metadata keys: {list(metadata.keys())}")
    metadata = _normalize_missing_labels(metadata)

    # Load externally-provided probe splits (train/val/test) if given. These drive the
    # multi-split (pre_filtered) probe protocol, matching the FM evaluator.
    provided_splits: Optional[Dict[str, tuple[np.ndarray, Dict[str, np.ndarray]]]] = None
    if probe_split_dirs:
        provided_splits = {}
        for split_name, split_dir in probe_split_dirs.items():
            norm = "val" if split_name in ("valid", "val") else split_name
            s_emb, s_meta, _ = embedding_io.load(split_dir)
            s_meta = _normalize_missing_labels(s_meta)
            provided_splits[norm] = (s_emb, s_meta)
            logger.info(f"Loaded probe split '{norm}': {len(s_emb):,} embeddings from {split_dir}")

    all_results: Dict[str, Any] = {}

    for task_name in tasks:
        # Resolve task class
        try:
            task_class = get_task(task_name)
        except KeyError:
            logger.warning(f"Task '{task_name}' not found in registry. Available: {sorted(set(TASK_REGISTRY.keys()))}")
            continue

        cfg = task_configs.get(task_name, task_configs.get(task_name.lower(), {}))

        # Skip tasks that need a live model
        if getattr(task_class, "requires_model", False):
            logger.warning(f"Skipping '{task_name}': requires_model=True (needs a live model instance)")
            continue

        # Multi-split tasks (e.g. LinearProbeTask) need train/val/test splits:
        #   * --probe_split_dirs given     → use those splits directly (pre_filtered), matching
        #     the FM evaluator's _run_pre_filtered (train→test) protocol.
        #   * else use_project_split=True  → build flat project-disjoint splits from the
        #     embeddings file (requires 'search_project' in metadata).
        #   * else                         → legacy random split on the single embeddings set.
        requires_multi = getattr(task_class, "requires_multi_split", False)
        build_flat_splits = False
        if requires_multi and provided_splits is None and cfg.get("use_project_split", True) is not False:
            project_key = cfg.get("project_key", "search_project")
            if project_key not in metadata:
                logger.warning(
                    f"Skipping '{task_name}': use_project_split=True but '{project_key}' not in metadata. "
                    f"Pass use_project_split=false (or --probe_split_dirs) instead."
                )
                continue
            build_flat_splits = True

        logger.info(f"Running task: {task_name}")
        task_output_dir = out / task_name
        task_output_dir.mkdir(parents=True, exist_ok=True)

        start = time.time()
        try:
            splits_arg = None
            if requires_multi and provided_splits is not None:
                splits_arg = provided_splits
                sizes = ", ".join(f"{k}={len(v[0]):,}" for k, v in provided_splits.items())
                logger.info(f"Using externally-provided probe splits ({sizes}) — pre_filtered (train→test) mode.")
            elif build_flat_splits:
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
                splits_arg = {s: (split_result[s]["embeddings"], split_result[s]["metadata"]) for s in ("train", "val", "test")}

            task = task_class(output_dir=str(task_output_dir), **cfg)
            if splits_arg is not None:
                task_results = task.run(  # type: ignore[call-arg]
                    embeddings,
                    metadata,
                    faiss_index,
                    splits=splits_arg,
                    pre_filtered=True,
                )
            else:
                task_results = task.run(embeddings, metadata, faiss_index)
            task_results["execution_time"] = time.time() - start
            task_results["output_dir"] = str(task_output_dir)
            _save_task_results(task_name, task_results, task_output_dir)
            logger.info(f"  [OK] {task_name} completed in {task_results['execution_time']:.2f}s — results in {task_output_dir}")
        except Exception as exc:
            elapsed = time.time() - start
            logger.error(f"  [FAIL] {task_name} failed after {elapsed:.2f}s: {exc}")
            task_results = {"error": str(exc), "execution_time": elapsed}
            _save_task_results(task_name, task_results, task_output_dir)

        all_results[task_name] = task_results

    _upload_to_s3(out)
    return all_results


def main() -> None:
    """Main."""
    parser = argparse.ArgumentParser(description="Run eval tasks on pre-generated embeddings (HDF5 + FAISS format)")
    parser.add_argument(
        "--embeddings_dir",
        required=False,
        help="Directory containing embeddings.h5 and index.faiss",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["embeddingstatisticstask"],
        metavar="TASK",
        help=("One or more task names to run (case-insensitive). Default: embeddingstatisticstask. Use --list_tasks to see all available tasks."),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write results (default: same as --embeddings_dir)",
    )
    parser.add_argument(
        "--task_configs",
        default=None,
        help=(
            "JSON string mapping task name → kwargs, e.g. "
            '\'{"linearprobetask": {"use_project_split": true, '
            '"train_samples": 100000, "val_samples": 10000, "test_samples": 10000}}\' '
            "(use_project_split=true requires 'search_project' in the embeddings metadata; "
            "use_project_split=false falls back to legacy random splitting)"
        ),
    )
    parser.add_argument(
        "--eval_config_name",
        default=None,
        help=(
            "Name of an instanovo/configs/evaluation/<name>.yaml config whose task_configs "
            "block is used as the canonical base configuration (e.g. 'default' to match the "
            "foundation-model eval exactly). --task_configs is deep-merged on top as an override."
        ),
    )
    parser.add_argument(
        "--probe_split_dirs",
        default=None,
        help=(
            "Comma-separated split=dir mapping for multi-split tasks, e.g. "
            "'train=out/probe_train,val=out/probe_val,test=out/probe_test'. Each dir holds that "
            "split's embeddings.h5. Makes the linear probe train on the train split and report on "
            "the test split (pre_filtered mode), matching the foundation-model eval."
        ),
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="Print all registered task names and exit",
    )
    args = parser.parse_args()

    if args.list_tasks:
        unique = sorted(set(TASK_REGISTRY.keys()))
        print("Available tasks:")  # noqa: T201
        for t in unique:
            print(f"  {t}")  # noqa: T201
        return

    if not args.embeddings_dir:
        parser.error("--embeddings_dir is required unless --list_tasks is used")

    task_configs = json.loads(args.task_configs) if args.task_configs else {}

    probe_split_dirs = None
    if args.probe_split_dirs:
        probe_split_dirs = {}
        for pair in args.probe_split_dirs.split(","):
            if "=" not in pair:
                parser.error(f"--probe_split_dirs entry '{pair}' must be 'split=dir'")
            name, _, path = pair.partition("=")
            probe_split_dirs[name.strip()] = path.strip()

    run(
        embeddings_dir=args.embeddings_dir,
        tasks=args.tasks,
        output_dir=args.output_dir,
        task_configs=task_configs,
        eval_config_name=args.eval_config_name,
        probe_split_dirs=probe_split_dirs,
    )


if __name__ == "__main__":
    main()
