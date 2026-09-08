import glob
import logging
from pathlib import Path
from typing import List, Optional

import typer
from omegaconf import DictConfig
from typing_extensions import Annotated

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

from instanovo_fm.utils.hydra_config import compose_fm_config

logger = ColorLog(console, __name__).logger

logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.rest_utils").setLevel(logging.ERROR)

cli = typer.Typer(rich_markup_mode="rich", pretty_exceptions_enable=False)


@cli.command("train")
def denovo_train(
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
    """Train the InstaNovo Downstream Model for de novo sequencing."""
    logger.info("Initializing InstaNovo downstream model training.")

    if config_name is None:
        config_name = "denovo"

    config = compose_fm_config(
        config_name=config_name,
        overrides=overrides,
        config_dir=config_path,
    )

    logger.info("Starting InstaNovo downstream model training.")
    from instanovo_fm.downstream.de_novo_sequencing.train import DownstreamDeNovoTrainer

    trainer = DownstreamDeNovoTrainer(config)
    trainer.train()


@cli.command("predict")
def denovo_predict(
    data_path: Annotated[
        Optional[str],
        typer.Option(
            "--data-path",
            "-d",
            help="Path to input data file",
        ),
    ] = None,
    output_path: Annotated[
        Optional[Path],
        typer.Option(
            "--output-path",
            "-o",
            help="Path to output file.",
            exists=False,
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    denovo_model: Annotated[
        Optional[str],
        typer.Option(
            "--denovo-model",
            "-i",
            help=("Path to a downstream de novo checkpoint file (.ckpt format)."),
        ),
    ] = None,
    denovo: Annotated[
        Optional[bool],
        typer.Option(
            "--denovo/--evaluation",
            help="Do [i]de novo[/i] predictions or evaluate an annotated file with peptide sequences?",
        ),
    ] = None,
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
) -> DictConfig:
    """Run predictions with InstaNovo downstream de novo model."""
    # Compose config with overrides
    logger.info("Initializing InstaNovo downstream de novo inference.")

    # Defer imports to improve cli performance
    from instanovo.transformer.model import InstaNovo

    if config_name is None:
        # inference/denovo.yaml, not the training denovo.yaml: this command used
        # to compose from the inference config directory.
        config_name = "inference/denovo"

    config = compose_fm_config(
        config_name=config_name,
        overrides=overrides,
        config_dir=config_path,
    )

    # Check config inputs
    if data_path is not None:
        if "*" in data_path or "?" in data_path or "[" in data_path:
            # Glob notation: path/to/data/*.parquet
            if not glob.glob(data_path):
                raise ValueError(f"The data_path '{data_path}' doesn't correspond to any file(s).")
        config.data_path = str(data_path)

    if not config.get("data_path", None) and data_path is None:
        raise ValueError(
            "Expected 'data_path' but found None. Please specify it in the "
            "`config/inference/<your_config>.yaml` configuration file or with the cli flag "
            "`--data-path='path/to/data'`. Allows `.mgf`, `.mzml`, `.mzxml`, a directory, or a "
            "`.parquet` file. Glob notation is supported:  eg.: `--data-path='./experiment/*.mgf'`."
        )

    if denovo is not None:
        # Don't compute metrics in denovo mode
        config.denovo = denovo

    if output_path is not None:
        if output_path.exists():
            logger.info(f"Output path '{output_path}' already exists and will be overwritten.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        config.output_path = str(output_path)
    if config.get("output_path", None) is None and config.get("denovo", False):
        raise ValueError(
            "Expected 'output_path' but found None in denovo mode. Please specify it in the "
            "`config/inference/<your_config>.yaml` configuration file or with the cli flag "
            "`--output-path=path/to/output_file`."
        )

    if denovo_model is not None:
        candidate = Path(denovo_model)
        # A value that contains a path separator or ends with .ckpt is the
        # user passing a checkpoint location; everything else is parsed as a
        # registry/model ID.
        looks_like_path = "/" in denovo_model or denovo_model.endswith(".ckpt")

        if candidate.is_file():
            if candidate.suffix != ".ckpt":
                raise ValueError(f"Checkpoint file '{denovo_model}' should end with extension '.ckpt'.")
        elif denovo_model.startswith("s3://"):
            pass  # remote checkpoint; existence is verified later by S3FileHandler
        elif looks_like_path:
            raise FileNotFoundError(
                f"Checkpoint file '{denovo_model}' was not found. Pass an existing .ckpt path, "
                "an s3:// URL, or one of the supported model IDs: "
                f"""{", ".join(f"'{model_id}'" for model_id in InstaNovo.get_pretrained())}."""
            )
        elif denovo_model not in InstaNovo.get_pretrained():
            raise ValueError(
                f"InstaNovo model ID '{denovo_model}' is not supported. "
                "Currently supported value(s): "
                f"""{", ".join(f"'{model_id}'" for model_id in InstaNovo.get_pretrained())}."""
            )
        config.denovo_model = denovo_model

    if not config.get("denovo_model", None):
        raise ValueError(
            "Expected 'denovo_model' but found None. Please specify it in the "
            "`config/inference/<your_config>.yaml` configuration file or with the cli flag "
            "`instanovo-fm denovo predict --denovo-model=path/to/model.ckpt`."
        )

    logger.info("Initializing InstaNovo downstream de novo inference.")
    from instanovo_fm.downstream.de_novo_sequencing.predict import DownstreamDeNovoPredictor

    predictor = DownstreamDeNovoPredictor(config)
    predictor.predict()


if __name__ == "__main__":
    cli()
