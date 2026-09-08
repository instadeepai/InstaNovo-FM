"""Tests for detect_all_duplicates."""

from typer.testing import CliRunner

from scripts.preprocessing.detect_all_duplicates import app as detect_app


def test_detect_all_duplicates(preprocessing_env) -> None:
    """Test detect_all_duplicates script."""
    runner = CliRunner()
    result = runner.invoke(
        detect_app,
        [
            "--input-dir",
            str(preprocessing_env.data_dir / "acfm"),
            "--output-file",
            str(preprocessing_env.output_dir / "duplicates.txt"),
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert (preprocessing_env.output_dir / "duplicates.txt").exists()

    with open(preprocessing_env.output_dir / "duplicates.txt") as f:
        content = f.read()
        assert "sample1" in content


def test_error_handling(preprocessing_env) -> None:
    """Test error handling for invalid inputs."""
    runner = CliRunner()
    result = runner.invoke(
        detect_app,
        [
            "--input-dir",
            "/non/existent/path",
            "--output-file",
            str(preprocessing_env.output_dir / "error.txt"),
        ],
    )
    assert result.exit_code != 0
