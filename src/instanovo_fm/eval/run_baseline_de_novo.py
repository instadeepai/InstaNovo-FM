"""Run and score a baseline model on de novo peptide sequencing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import typer
from omegaconf import DictConfig, OmegaConf, open_dict

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


@dataclass(frozen=True)
class BaselineModel:
    """A baseline model's canonical checkpoint and download URL (fallback).

    Attributes:
        default_checkpoint: Canonical local checkpoint path, used when the config has no ``checkpoint``.
        checkpoint_url: Fallback download URL when the local checkpoint is absent.
    """

    default_checkpoint: str
    checkpoint_url: str


MODELS: Dict[str, BaselineModel] = {
    "casanovo": BaselineModel(
        default_checkpoint="checkpoints/casanovo_v5_0_0.ckpt",
        checkpoint_url="https://github.com/Noble-Lab/casanovo/releases/download/v5.0.0/casanovo_v5_0_0.ckpt",
    ),
    "xuanjinovo": BaselineModel(
        default_checkpoint="checkpoints/XuanjiNovo_100M_massnet.ckpt",
        checkpoint_url="https://huggingface.co/Wyattz23/XuanjiNovo/resolve/main/XuanjiNovo_100M_massnet.ckpt",
    ),
}


def _load_model_and_predictor(model: str) -> tuple[Any, Any]:
    """Resolve a baseline model name to its ``(load_fn, predict_dataframe_fn)`` pair.

    Args:
        model: A registered baseline model name.

    Returns:
        Tuple ``(load_fn, predict_dataframe_fn)``.

    Raises:
        NotImplementedError: If the model is registered but has no de novo predictor wired up.
    """
    if model == "casanovo":
        from instanovo_fm.eval.predict_casanovo_de_novo import load_casanovo_model
        from instanovo_fm.eval.predict_casanovo_de_novo import predict_dataframe as casanovo_predict

        return load_casanovo_model, casanovo_predict
    # Only casanovo has an in-repo greedy predictor. XuanjiNovo goes through the
    # upstream model instead, via run_xuanjinovo.py and score_xuanjinovo.py, so
    # it falls through to the error below rather than getting a branch here.
    raise NotImplementedError(f"No de novo predictor wired up for model={model!r}.")


def _resolve_model_and_checkpoint(config: DictConfig) -> tuple[str, str, Optional[str]]:
    """Resolve the model name, checkpoint path and download URL from the config and registry.

    Args:
        config: The composed inference config.

    Returns:
        Tuple ``(model, checkpoint, checkpoint_url)``.

    Raises:
        ValueError: If the resolved model is not a registered baseline.
    """
    model = config.get("model", None)
    if model not in MODELS:
        raise ValueError(f"model must be one of {list(MODELS)}, got {model!r}")
    spec = MODELS[model]
    checkpoint = config.get("checkpoint") or spec.default_checkpoint
    checkpoint_url = config.get("checkpoint_url") or spec.checkpoint_url
    return model, checkpoint, checkpoint_url


def _normalize_data_groups(config: DictConfig, run_name: str) -> List[Any]:
    """Normalise ``config.data_path`` into a list of ``{result_name, input_path, output_path}`` groups.

    Args:
        config: The composed inference config.
        run_name: The run name (used as the single-dataset group name / output naming).

    Returns:
        A list of dict-like groups, each with ``result_name`` / ``input_path`` / ``output_path``.

    Raises:
        ValueError: If ``config.data_path`` is not set.
    """
    data_path = config.get("data_path")
    if data_path is None:
        raise ValueError("config.data_path is required: a parquet path/glob, or a list of {result_name, input_path, output_path}.")
    if OmegaConf.is_list(data_path):
        return list(data_path)
    return [
        {
            "result_name": run_name or Path(str(data_path)).stem,
            "input_path": str(data_path),
            "output_path": config.get("output_path"),
        }
    ]


def _build_result_row(run_name: str, model: str, num_beams: int, per_group_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the aggregated result-file row from per-group scoring metrics.

    Args:
        run_name: Run label.
        model: Model name.
        num_beams: Beam width used.
        per_group_metrics: Per-dataset ``overall`` metrics dicts (from :func:`score_predictions`).

    Returns:
        A single flat row dict for the aggregated results CSV.
    """
    row: Dict[str, Any] = {"run_name": run_name, "model": model, "num_beams": num_beams}
    for name, metrics in per_group_metrics.items():
        for metric_name, value in metrics.items():
            row[f"{name}_{metric_name}"] = value
    return row


def _write_predictions_to_dest(df: Any, dest: Optional[str], local_path: str, s3: Any, *, allow_fallback: bool, label: str) -> str:
    """Write ``df`` to ``dest`` (local or ``s3://``), falling back to the local copy on S3 failure.

    Args:
        df: The DataFrame to write.
        dest: Destination path (local or ``s3://``), or None.
        local_path: The already-written local copy used as the fallback.
        s3: An :class:`S3FileHandler`.
        allow_fallback: If True, suppress S3 errors and use ``local_path``; if False, re-raise.
        label: Log label for the destination (e.g. ``"[yeast] predictions"``).

    Returns:
        The path actually written (``dest`` on success, else ``local_path``).
    """
    if not dest or dest == local_path:
        return local_path
    if dest.startswith("s3://") and s3.s3 is None:
        if not allow_fallback:
            raise RuntimeError(f"{label}: S3 is not configured but dest is {dest} and allow_local_fallback is False")
        logger.warning(f"{label}: S3 not configured; keeping local copy {local_path} instead of {dest}")
        return local_path
    try:
        s3.upload_to_s3_wrapper(df.to_csv, dest, index=False)
        return dest
    except Exception as exc:
        if not allow_fallback:
            raise
        logger.warning(f"{label}: write to {dest} failed ({exc}); falling back to local copy {local_path}")
        return local_path


def run_from_config(config: DictConfig) -> Optional[str]:
    """Run and score a baseline de novo model across the dataset(s) in an inference config.

    Args:
        config: A composed inference config.

    Returns:
        The path of the aggregated results CSV (or None if nothing was scored).

    Raises:
        ValueError: On an unknown model or a missing ``data_path``.
        NotImplementedError: If the resolved model has no de novo decoder yet (only casanovo today).
    """
    import pandas as pd

    from instanovo_fm.eval._extract_embeddings_common import load_spectrum_dataframe
    from instanovo_fm.eval._predict_de_novo_common import MODEL_SCORING, save_predictions, score_predictions
    from instanovo.utils.s3 import S3FileHandler

    model, checkpoint, checkpoint_url = _resolve_model_and_checkpoint(config)
    load_model, predict_dataframe = _load_model_and_predictor(model)

    device = str(config.get("device") or "cuda")
    num_beams = int(config.get("num_beams", 1))
    batch_size = int(config.get("batch_size", 64))
    subset = float(config.get("subset", 1.0))
    max_samples_cfg = config.get("max_samples")
    max_samples = int(max_samples_cfg) if max_samples_cfg is not None else None
    max_charge_cfg = config.get("max_charge")
    max_charge = int(max_charge_cfg) if max_charge_cfg is not None else None
    min_intensity = config.get("min_intensity", 0.01)
    run_name = config.get("run_name") or f"{model}_baseline"
    column_map_cfg = config.get("column_map")
    column_map = OmegaConf.to_container(column_map_cfg) if column_map_cfg else None
    allow_fallback = bool(config.get("allow_local_fallback", True))  # S3 write failures fall back to local

    groups = _normalize_data_groups(config, run_name)
    local_dir = Path(f"de_novo_results/{run_name}/{model}")
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Baseline de novo — {model} over {len(groups)} dataset(s)")
    logger.info(f"  run_name={run_name}  num_beams={num_beams}  subset={subset}  batch_size={batch_size}  device={device}")
    logger.info(f"  checkpoint={checkpoint}\n")

    loaded = load_model(checkpoint, device, checkpoint_url=checkpoint_url, n_beams=num_beams, min_intensity=min_intensity, max_charge=max_charge)
    s3 = S3FileHandler()

    per_group_metrics: Dict[str, Dict[str, Any]] = {}
    for group in groups:
        name = group.get("result_name")
        input_path = group.get("input_path")
        out_path = group.get("output_path")
        logger.info(f"[{name}] loading {input_path}\n")
        df = load_spectrum_dataframe(input_path, max_samples, column_mapping=column_map)
        if subset < 1.0:
            df = df.iloc[: max(1, int(len(df) * subset))]
        pred_df = predict_dataframe(df, loaded, batch_size)

        # TODO only write local if no s3
        local_csv = local_dir / f"{name}.csv"
        save_predictions(pred_df, str(local_csv), s3_handler=s3)
        if out_path:
            dest = _write_predictions_to_dest(
                pred_df, str(out_path), str(local_csv), s3, allow_fallback=allow_fallback, label=f"[{name}] predictions"
            )
            logger.info(f"[{name}] predictions -> {dest}")

        if "sequence" in df.columns:
            results = score_predictions(str(local_csv), output_dir=str(local_dir / name), **MODEL_SCORING.get(model, {}))
            overall = results["overall"]
            per_group_metrics[name] = overall
            logger.info(
                f"[{name}] true: pep_recall={overall.get('peptide_recall', float('nan')):.4f} (n={overall.get('n', 0)})  |  "
                f"filtered: pep_recall={overall.get('peptide_recall_predictable', float('nan')):.4f} (n={overall.get('n_predictable', 0)})"
            )
        else:
            logger.warning(f"[{name}] no 'sequence' targets — predictions written, scoring skipped")

    result_path: Optional[str] = None
    if per_group_metrics:
        results_df = pd.DataFrame([_build_result_row(run_name, model, num_beams, per_group_metrics)])

        # Always keep a local copy in the working dir. # TODO only if no s3
        local_result = local_dir / f"{model}_de_novo_results.csv"
        results_df.to_csv(local_result, index=False)

        # Aggregated results go to a run-scoped path under output_dir/run_name (matching the per-dataset
        # output layout), else the config's result_file_path; with a local fallback when the S3 write
        # fails / S3 is unconfigured.
        output_dir_cfg = config.get("output_dir")
        result_dest = (
            f"{str(output_dir_cfg).rstrip('/')}/{run_name}/{model}_de_novo_results.csv" if output_dir_cfg else config.get("result_file_path")
        )
        result_path = _write_predictions_to_dest(
            results_df, result_dest, str(local_result), s3, allow_fallback=allow_fallback, label="aggregated results"
        )
        logger.info(f"[OK] aggregated {len(per_group_metrics)} dataset(s) -> {result_path}  (local copy: {local_result})")
    else:
        logger.info("Pipeline complete; no per group metrics.")
    return result_path


app = typer.Typer(rich_markup_mode="rich", pretty_exceptions_enable=False)


@app.command()  # TODO convert to cli
def baseline_predict(
    config_name: Optional[str] = typer.Option(None, "--config-name", "-cn", help="Inference config name (default: casanovo)"),
    config_path: Optional[str] = typer.Option(None, "--config-path", "-cp", help="Config directory (default: InstaNovo inference configs)"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Baseline model override (default: config value, else casanovo)"),
    checkpoint: Optional[str] = typer.Option(None, "--checkpoint", help="Checkpoint path/URI override (default: config value, else registry)"),
    device: Optional[str] = typer.Option(None, "--device", help="Torch device override (default: config value, else cuda)"),
    overrides: Optional[List[str]] = typer.Argument(None, hidden=True),
) -> None:
    """Run and score a baseline de novo model from an inference config."""
    from instanovo.constants import DEFAULT_INFERENCE_CONFIG_PATH
    from instanovo.utils.cli_utils import compose_config

    config = compose_config(config_path or DEFAULT_INFERENCE_CONFIG_PATH, config_name or "casanovo", overrides)
    with open_dict(config):
        if model is not None:
            config.model = model
        if checkpoint is not None:
            config.checkpoint = checkpoint
        if device is not None:
            config.device = device
    run_from_config(config)


if __name__ == "__main__":
    app()
