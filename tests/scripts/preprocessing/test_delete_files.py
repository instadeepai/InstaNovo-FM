"""Tests for delete_files."""

import polars as pl
from typer.testing import CliRunner

from scripts.preprocessing.delete_files import app as delete_files_app


def test_delete_files(preprocessing_env) -> None:
    """Test delete_files script."""
    runner = CliRunner()
    test_file = preprocessing_env.data_dir / "acfm" / "to_delete.ipc"
    pl.DataFrame({"test": [1, 2, 3]}).write_ipc(test_file)

    file_list = preprocessing_env.output_dir / "files_to_delete.txt"
    with open(file_list, "w") as f:
        f.write(str(test_file) + "\n")

    error_log = preprocessing_env.output_dir / "error_log.txt"
    result = runner.invoke(
        delete_files_app,
        [
            "--input-file",
            str(file_list),
            "--error-log",
            str(error_log),
            "--verbose",
        ],
    )

    assert result.exit_code == 0
    assert not test_file.exists()
    assert error_log.exists()
