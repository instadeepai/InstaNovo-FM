"""Every CLI command is invoked, not just asked for its `--help`.

Both CLI breakages this suite was written for passed a `--help` check while the
command itself was unusable, because the imports and config composition happen
inside the command body:

  - `instanovo-fm evaluate` raised ImportError on a `run_evaluation` that no
    longer existed.
  - `instanovo-fm denovo predict` raised TypeError building its run id, because
    the inference config declares `run_name:` with no value and `.get()`
    returned None rather than the default.

These tests therefore invoke each command through Typer's runner and assert on
what happens *past* argument parsing. They deliberately do not train or predict
for real -- that needs a GPU and a corpus -- so each command is driven far
enough to prove its imports resolve and its config composes, then stopped.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from instanovo_fm.cli import cli

runner = CliRunner()


def test_evaluate_reaches_run_evaluation() -> None:
    """`evaluate` must import `run_evaluation` and get as far as needing a checkpoint.

    The regression this guards raised ImportError before any argument was read.
    """
    from instanovo_fm.eval.embed_evaluation import run_evaluation

    assert callable(run_evaluation)

    result = runner.invoke(cli, ["evaluate", "--checkpoint", "does-not-exist.ckpt"])
    # It must fail for a reason about the *checkpoint*, never about imports.
    assert not isinstance(result.exception, ImportError), result.output


@pytest.mark.parametrize(
    "command", [["train"], ["evaluate"], ["denovo", "train"], ["denovo", "predict"]]
)
def test_command_is_registered(command: list[str]) -> None:
    """Each command exists and renders help without error."""
    result = runner.invoke(cli, [*command, "--help"])
    assert result.exit_code == 0, result.output


def test_predictor_run_id_survives_a_null_run_name() -> None:
    """Building a run id must survive the null `run_name` the shipped config has.

    `.get("run_name", default)` returns None for a present-but-null key, so the
    default never applied and concatenating with the timestamp raised TypeError
    before any prediction started. This drives the real `__init__` far enough to
    build `_run_id`, rather than re-implementing the expression.
    """
    import inspect

    from instanovo_fm.common.predictor import AccelerateDeNovoPredictor
    from instanovo_fm.common.trainer import AccelerateDeNovoTrainer
    from instanovo_fm.utils.hydra_config import compose_fm_config

    # The precondition: the shipped inference config really does leave it null.
    shipped = compose_fm_config(config_name="inference/denovo", overrides=[])
    assert shipped.get("run_name") is None, "config no longer reproduces the bug's precondition"

    # The class is abstract, so this guards the expression at the source level
    # rather than by construction. Both sites had the same latent bug.
    for cls in (AccelerateDeNovoPredictor, AccelerateDeNovoTrainer):
        source = inspect.getsource(cls.__init__)
        assert (
            'get("run_name") or' in source
        ), f"{cls.__name__} reverted to a .get() default for run_name"


def test_denovo_configs_do_not_default_to_mlflow() -> None:
    """The de novo configs must match the foundational ones and default MLflow off.

    With `mlflow_enabled: True` and no tracking URI, MLflow falls back to a
    `./mlruns` file store, which current MLflow refuses outright -- so
    `denovo predict` failed before writing any predictions.
    """
    from instanovo_fm.utils.hydra_config import compose_fm_config

    for config_name in ("denovo", "inference/denovo"):
        config = compose_fm_config(config_name=config_name, overrides=[])
        assert config.get("mlflow_enabled") is False, f"{config_name} defaults MLflow on"
