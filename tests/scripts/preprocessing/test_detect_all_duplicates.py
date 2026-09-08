"""Tests for detect_all_duplicates."""

from pathlib import Path

from typer.testing import CliRunner

from scripts.preprocessing.detect_all_duplicates import (
    app as detect_app,
    group_files_by_base_name,
)


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


def test_extensions_filters_to_parquet_only(tmp_path: Path) -> None:
    """--extensions must control which suffixes are grouped."""
    root = tmp_path / "data"
    root.mkdir()
    (root / "sample.ipc").write_text("a")
    (root / "sample.parquet").write_text("b")
    (root / "other.parquet").write_text("c")
    (root / "other2.parquet").write_text("d")

    by_parquet = group_files_by_base_name(str(root), extensions=[".parquet"])
    assert set(by_parquet) == {"sample", "other", "other2"}
    assert all(p.endswith(".parquet") for paths in by_parquet.values() for p in paths)

    by_default = group_files_by_base_name(str(root))
    assert "sample" in by_default
    assert all(p.endswith(".ipc") for paths in by_default.values() for p in paths)
