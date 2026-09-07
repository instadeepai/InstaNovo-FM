from pathlib import Path
from typing import List, Optional

import typer
from omegaconf import DictConfig
from typing_extensions import Annotated

from instanovo.__init__ import console
from instanovo.utils.cli_utils import compose_config
from instanovo.constants import DEFAULT_TRAIN_CONFIG_PATH
from instanovo.utils.colorlogging import ColorLog

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

    if config_path is None:
        config_path = DEFAULT_TRAIN_CONFIG_PATH
    if config_name is None:
        config_name = "foundational"

    config = compose_config(
        config_path=config_path,
        config_name=config_name,
        overrides=overrides,
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
