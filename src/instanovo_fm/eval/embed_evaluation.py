"""Standalone embedding evaluation script for InstaNovo Foundation Model.

This script provides a Hydra-based CLI for comprehensive evaluation of foundation
model embeddings. It can be run in two ways:

1. After training (integrated with train.py via config flag)
2. Standalone script for evaluating existing checkpoints

Usage:
    # Evaluate with default config
    python -m instanovo_fm.eval.embed_evaluation

    # Evaluate specific checkpoint
    python -m instanovo_fm.eval.embed_evaluation evaluation.checkpoint_path=/path/to/model_best.ckpt

    # Run specific tasks
    python -m instanovo_fm.eval.embed_evaluation evaluation.tasks_to_run=[embedding_statistics,duplicate_retrieval]

    # Evaluate on test split
    python -m instanovo_fm.eval.embed_evaluation evaluation.split=test

Example config in foundational.yaml:
    evaluation:
      enabled: True
      split: valid  # "valid" or "test"
      batch_size: 256
      device: auto  # "auto", "cuda", or "cpu"
      output_dir: "./evaluation_results"
      max_samples: 1000  # Maximum number of samples (null = use all data)
      save_embeddings: False  # Save embeddings to disk (default: False for speed/space)
      force_regenerate_embeddings: False  # Regenerate cached embeddings (only if save_embeddings=True)
      tasks_to_run:  # null for all tasks, or list of specific tasks
        - embedding_statistics
        - duplicate_retrieval
        - charge_linear_probe
        - rt_linear_probe
      task_configs:  # Task-specific configurations
        embedding_statistics:
          n_clusters: 50
          compute_pca: True
        duplicate_retrieval:
          k_values: [1, 5, 10, 20]
          max_samples: 1000
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from instanovo.__init__ import console
from instanovo_fm.eval.evaluator import EmbeddingEvaluator
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

# Path to config directory (same as train.py)
CONFIG_PATH = Path(__file__).parent.parent.parent / "configs"


def _checkpoint_id(ckpt_path: str, index: int) -> str:
    """Extract a short identifier from a checkpoint path.

    For S3 paths of the form s3://.../output/<uuid>/... the first 8 hex chars of
    the UUID are used (e.g. '47c30c9e').  Falls back to 'checkpoint_<index>'.
    """
    match = re.search(r"/output/([0-9a-f]{8})", ckpt_path)
    if match:
        return match.group(1)
    return f"checkpoint_{index}"


def run_evaluation(config: DictConfig) -> None:
    """Run embedding evaluation for one or more checkpoints.

    This holds the evaluation logic itself, independent of how the config was
    built, so that both the Hydra entry point (``python -m
    instanovo_fm.eval.embed_evaluation``) and the Typer CLI
    (``python -m instanovo_fm.cli evaluate``) drive the same code path.

    Args:
        config: Configuration loaded from evaluation.yaml (which inherits from foundational.yaml)
    """
    # Check if evaluation is enabled
    eval_config = config.get("evaluation", {})
    if not eval_config.get("enabled", False):
        logger.warning("Evaluation disabled (evaluation.enabled=False)")
        return

    split = eval_config.get("split", "valid")
    force_regenerate = eval_config.get("force_regenerate_embeddings", False)

    checkpoint_paths = eval_config.get("checkpoint_paths", None)

    if checkpoint_paths:
        # Multi-checkpoint mode: share a single timestamped parent directory and
        # run one evaluator per checkpoint in a checkpoint-specific subdirectory.
        base_output_dir = Path(eval_config.get("output_dir", "./evaluation_results"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        shared_dir = base_output_dir / f"eval_{timestamp}"

        logger.info(f"Multi-checkpoint mode: {len(checkpoint_paths)} checkpoints")
        logger.info(f"Shared output directory: {shared_dir}")

        for i, ckpt_path in enumerate(checkpoint_paths):
            ckpt_id = _checkpoint_id(str(ckpt_path), i)
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Checkpoint {i + 1}/{len(checkpoint_paths)}: {ckpt_id}")
            logger.info(f"  Path: {ckpt_path}")
            logger.info(f"{'=' * 60}")

            ckpt_config = OmegaConf.merge(
                config,
                OmegaConf.create({
                    "evaluation": {
                        "checkpoint_path": str(ckpt_path),
                        "output_dir": str(shared_dir / ckpt_id),
                    }
                }),
            )

            try:
                evaluator = EmbeddingEvaluator(ckpt_config)
            except ValueError as e:
                logger.error(f"Failed to initialise evaluator for {ckpt_id}: {e}")
                sys.exit(1)

            try:
                evaluator.evaluate(split=split, force_regenerate=force_regenerate)
                logger.info(f"Checkpoint {ckpt_id} done. Results: {evaluator.output_dir}")
            except Exception as e:
                logger.error(f"Evaluation failed for {ckpt_id}: {e}")
                import traceback
                traceback.print_exc()
                sys.exit(1)

        logger.info(f"\nAll checkpoints evaluated. Results in: {shared_dir}")

    else:
        # Single-checkpoint mode (existing behaviour).
        try:
            evaluator = EmbeddingEvaluator(config)
        except ValueError as e:
            logger.error(f"Failed to initialize evaluator: {e}")
            logger.error("Make sure to specify a checkpoint path:")
            logger.error("  evaluation.checkpoint_path=/path/to/checkpoint.ckpt")
            sys.exit(1)

        try:
            results = evaluator.evaluate(
                split=split,
                force_regenerate=force_regenerate,
            )
            logger.info("Evaluation completed successfully!")
            logger.info(f"Results saved to: {evaluator.output_dir}")

            # Log metrics to MLflow if a run ID file exists alongside the checkpoint
            _log_metrics_to_mlflow(config, evaluator, results)

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


@hydra.main(config_path=str(CONFIG_PATH), version_base=None, config_name="foundational_local")
def main(config: DictConfig) -> None:
    """Hydra entry point for embedding evaluation.

    Args:
        config: Hydra configuration loaded from evaluation.yaml (which inherits from foundational.yaml)
    """
    run_evaluation(config)


def _log_metrics_to_mlflow(
    config: DictConfig,
    evaluator: EmbeddingEvaluator,
    results: dict,
) -> None:
    """Log embedding eval metrics to an existing MLflow run if available.

    Looks for ``mlflow_run_id.txt`` next to the checkpoint. If found, reopens
    that run and logs all embedding metrics with an ``eval/`` prefix so they
    appear alongside training metrics in the MLflow UI.
    """
    if not config.get("mlflow_enabled", True):
        logger.info("MLflow disabled — skipping post-eval metric logging")
        return

    checkpoint_path = evaluator.eval_config.get("checkpoint_path", "")
    if not checkpoint_path:
        return

    # Look for mlflow_run_id.txt in the checkpoint directory
    run_id_path = Path(checkpoint_path).parent / "mlflow_run_id.txt"
    if not run_id_path.exists():
        logger.info("No mlflow_run_id.txt found — skipping MLflow logging")
        return

    run_id = run_id_path.read_text().strip()
    if not run_id:
        return

    try:
        import mlflow

        # Setup tracking URI and credentials (same as trainer)
        tracking_uri = config.get(
            "mlflow_tracking_uri",
            os.environ.get("MLFLOW_TRACKING_URI", ""),
        )
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Setup AIchor credentials if available
        author_slug = os.environ.get("AICHOR_AUTHOR_SLUG")
        if author_slug:
            token = os.environ.get(f"{author_slug}__MLFLOW_TOKEN")
            if token:
                os.environ["MLFLOW_TRACKING_PASSWORD"] = token
                os.environ.setdefault(
                    "MLFLOW_TRACKING_USERNAME", "instadeep-mlflow"
                )

        # Get embedding stats for logging
        embeddings_info = {}
        if hasattr(evaluator, "_last_embeddings_info"):
            embeddings_info = evaluator._last_embeddings_info
        else:
            # Reconstruct minimal info from results
            embeddings_info = {"num_embeddings": 0, "mean_norm": 0.0, "std_norm": 0.0}

        loggable = evaluator.get_metrics_for_logging(results, embeddings_info)

        with mlflow.start_run(run_id=run_id):
            for name, value in loggable.items():
                mlflow.log_metric(f"eval_post/{name}", value)
            mlflow.set_tag("embed_eval", "post_training")

        logger.info(
            f"Logged {len(loggable)} embedding metrics to MLflow run {run_id}"
        )

    except Exception as e:
        logger.warning(f"Failed to log metrics to MLflow: {e}")


if __name__ == "__main__":
    main()
