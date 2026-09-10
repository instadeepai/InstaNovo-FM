"""Tests for delete_multi_folder_duplicates."""

from pathlib import Path

from scripts.preprocessing.delete_multi_folder_duplicates import parse_files_to_delete


def test_exact_folder_match_resolves_existing_ipc_suffixes(tmp_path: Path) -> None:
    """Exact folder match; resolve .ipc / .mzML.ipc only when they exist."""
    keep = tmp_path / "keep"
    drop = tmp_path / "drop"
    keep.mkdir()
    drop.mkdir()
    (drop / "sample.mzML.ipc").write_text("x")
    (drop / "sample.ipc").write_text("y")
    # Substring trap: target "drop" must not match "keep_drop_extra"
    trap = tmp_path / "keep_drop_extra"
    trap.mkdir()
    (trap / "sample.mzML.ipc").write_text("z")

    report = tmp_path / "multi.txt"
    report.write_text(
        "\n".join(
            [
                "File: sample",
                "Folders:",
                f"  - {keep}",
                f"  - {drop}",
                f"  - {trap}",
                "",
            ]
        )
    )

    to_delete = parse_files_to_delete(str(report), str(drop))
    assert set(to_delete) == {
        str(drop / "sample.mzML.ipc"),
        str(drop / "sample.ipc"),
    }
