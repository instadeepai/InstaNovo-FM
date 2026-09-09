from typing import List, Optional

import typer
from typing_extensions import Annotated

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

from instanovo_fm.utils.hydra_config import compose_fm_config

logger = ColorLog(console, __name__).logger

cli = typer.Typer(rich_markup_mode="rich", pretty_exceptions_enable=False)


@cli.command("train")
def foundational_train(
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config-path",
            "-cp",
            help="Relative path to config directory.",
        ),
    ] = None,
    config_name: Annotated[
        Optional[str],
        typer.Option(
            "--config-name",
            "-cn",
            help="The name of the config (usually the file name without the .yaml extension).",
        ),
    ] = None,
    overrides: Optional[List[str]] = typer.Argument(None, hidden=True),
) -> None:
    """Train the InstaNovo Foundation Model."""
    logger.info("Initializing InstaNovo Foundation Model training.")

    if config_name is None:
        config_name = "foundational"

    config = compose_fm_config(
        config_name=config_name,
        overrides=overrides,
        config_dir=config_path,
    )

    logger.info("Starting InstaNovo Foundation Model training.")
    from instanovo_fm.trainer.train import FoundationalTrainer

    trainer = FoundationalTrainer(config)
    trainer.train()

    # Save MLflow run ID alongside checkpoint for post-training eval to pick up
    import os

    if trainer.tracker is not None and hasattr(trainer.tracker, "run_id"):
        checkpoint_dir = config.model.get("model_save_folder_path", "./checkpoints")
        run_id_path = os.path.join(checkpoint_dir, "mlflow_run_id.txt")
        with open(run_id_path, "w") as f:
            f.write(trainer.tracker.run_id)
        logger.info(f"Saved MLflow run ID to {run_id_path}")

    trainer.run_post_training_evaluation()


@cli.command("evaluate")
def foundational_evaluate(
    config_path: Annotated[
        Optional[str],
        typer.Option(
            "--config-path",
            "-cp",
            help="Relative path to config directory.",
        ),
    ] = None,
    config_name: Annotated[
        Optional[str],
        typer.Option(
            "--config-name",
            "-cn",
            help="The name of the config (usually the file name without the .yaml extension).",
        ),
    ] = None,
    checkpoint_path: Annotated[
        Optional[str],
        typer.Option(
            "--checkpoint",
            help="Checkpoint to evaluate. Overrides evaluation.checkpoint_path in the config.",
        ),
    ] = None,
    split: Annotated[
        Optional[str],
        typer.Option(
            "--split",
            help="Dataset split to evaluate on (train, valid or test). Overrides evaluation.split.",
        ),
    ] = None,
    overrides: Optional[List[str]] = typer.Argument(None, hidden=True),
) -> None:
    """Evaluate a Foundation Model checkpoint by computing and probing embeddings.

    Runs the embedding evaluation suite (linear probes, retrieval, peak-type
    classification, attribution) against a trained checkpoint. This is the same
    code path as ``python -m instanovo_fm.eval.embed_evaluation``; the options
    below are shorthand for the corresponding Hydra overrides.
    """
    logger.info("Initializing InstaNovo Foundation Model evaluation.")

    if config_name is None:
        config_name = "foundational"

    # Surface the two overrides people reach for most as first-class options,
    # while still allowing arbitrary Hydra overrides as trailing arguments.
    overrides = list(overrides or [])
    overrides.append("evaluation.enabled=True")
    if checkpoint_path is not None:
        overrides.append(f"evaluation.checkpoint_path={checkpoint_path}")
    if split is not None:
        overrides.append(f"evaluation.split={split}")

    config = compose_fm_config(
        config_name=config_name,
        overrides=overrides,
        config_dir=config_path,
    )

    logger.info("Starting InstaNovo Foundation Model evaluation.")
    from instanovo_fm.eval.embed_evaluation import run_evaluation

    run_evaluation(config)


from instanovo_fm.downstream.de_novo_sequencing.cli import cli as _denovo_cli

cli.add_typer(
    _denovo_cli,
    name="denovo",
    help="Downstream de novo sequencing: the FM encoder plus an InstaNovo decoder.",
)


if __name__ == "__main__":
    cli()
