from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from instanovo_fm.downstream.de_novo_sequencing.cli import cli

runner = CliRunner()


def test_cli_invokes_trainer_with_defaults() -> None:
    """Test that the CLI uses default config path and name when none are provided."""
    mock_trainer = MagicMock()
    with (
        patch(
            "instanovo_fm.downstream.de_novo_sequencing.cli.compose_fm_config",
            return_value={"config": "ok"},
        ) as compose,
        patch(
            "instanovo_fm.downstream.de_novo_sequencing.train.DownstreamDeNovoTrainer",
            return_value=mock_trainer,
        ) as trainer_cls,
    ):
        result = runner.invoke(cli, ["train"])

    assert result.exit_code == 0, result.output
    compose.assert_called_once()
    kwargs = compose.call_args.kwargs
    assert kwargs["config_dir"] is None
    assert kwargs["config_name"] == "denovo"
    trainer_cls.assert_called_once_with({"config": "ok"})
    mock_trainer.train.assert_called_once()


def test_cli_respects_overrides() -> None:
    """Test that custom config path, name, and Hydra overrides are forwarded correctly."""
    with (
        patch(
            "instanovo_fm.downstream.de_novo_sequencing.cli.compose_fm_config",
            return_value={"config": "ok"},
        ) as compose,
        patch(
            "instanovo_fm.downstream.de_novo_sequencing.train.DownstreamDeNovoTrainer",
        ),
    ):
        result = runner.invoke(cli, ["train", "-cp", "some/path", "-cn", "my_cfg", "key=val"])

    assert result.exit_code == 0, result.output
    kwargs = compose.call_args.kwargs
    assert kwargs["config_dir"] == "some/path"
    assert kwargs["config_name"] == "my_cfg"
    assert kwargs["overrides"] == ["key=val"]
