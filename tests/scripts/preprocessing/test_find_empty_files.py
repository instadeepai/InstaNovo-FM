"""Tests for find_empty_files."""

from typer.testing import CliRunner

from scripts.preprocessing.find_empty_files import app as find_empty_app


def test_find_empty_files(preprocessing_env) -> None:
    """Test find_empty_files script."""
    runner = CliRunner()
    result = runner.invoke(
        find_empty_app,
        [
            "--input-dir",
            str(preprocessing_env.data_dir / "lcfm"),
            "--output-file",
            str(preprocessing_env.output_dir / "empty_files.txt"),
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert (preprocessing_env.output_dir / "empty_files.txt").exists()

    with open(preprocessing_env.output_dir / "empty_files.txt") as f:
        content = f.read()
        assert "empty.ipc" in content
