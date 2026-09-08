"""Tests for detect_multi_folder_duplicates."""

from pathlib import Path

from scripts.preprocessing.detect_multi_folder_duplicates import (
    classify_multi_folder_duplicates,
    find_duplicate_files,
    parse_duplicate_report,
)


def test_same_folder_ipc_mzml_pair_is_not_multi_folder(tmp_path: Path) -> None:
    """Same-folder .ipc and .mzML.ipc must not be reported as multi-folder."""
    report = tmp_path / "duplicates.txt"
    report.write_text(
        "\n".join(
            [
                "/data/acfm/PXD000001/sample.ipc",
                "/data/acfm/PXD000001/sample.mzML.ipc",
                "",
            ]
        )
    )

    file_map = parse_duplicate_report(report)
    assert file_map == {"sample": ["/data/acfm/PXD000001"]}
    assert classify_multi_folder_duplicates(file_map) == {}

    output = tmp_path / "multi.txt"
    result = find_duplicate_files(str(report), str(output))
    assert result == {}
    assert not output.exists()


def test_distinct_folders_are_multi_folder(tmp_path: Path) -> None:
    """Copies under different folders remain multi-folder candidates."""
    report = tmp_path / "duplicates.txt"
    report.write_text(
        "\n".join(
            [
                "/data/acfm/PXD000001/sample.mzML.ipc",
                "/data/acfm/PXD000002/sample.mzML.ipc",
                "",
            ]
        )
    )

    result = find_duplicate_files(str(report), str(tmp_path / "multi.txt"))
    assert result == {
        "sample": ["/data/acfm/PXD000001", "/data/acfm/PXD000002"],
    }
