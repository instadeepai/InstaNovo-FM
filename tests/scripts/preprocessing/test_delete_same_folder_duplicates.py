"""Tests for delete_same_folder_duplicates."""

from collections import defaultdict
from pathlib import Path

from scripts.preprocessing.delete_same_folder_duplicates import (
    find_duplicates_to_delete,
    parse_input_file,
)


def test_deletes_all_but_first_same_folder_copy() -> None:
    """Three same-folder copies keep the sorted first and delete the rest."""
    file_map = defaultdict(list)
    folder = "/data/acfm/PXD000001"
    file_map["sample"] = [
        (folder, f"{folder}/sample.c.ipc"),
        (folder, f"{folder}/sample.a.ipc"),
        (folder, f"{folder}/sample.b.ipc"),
    ]

    to_delete = find_duplicates_to_delete(file_map)
    assert to_delete == [
        f"{folder}/sample.b.ipc",
        f"{folder}/sample.c.ipc",
    ]


def test_parse_does_not_treat_dotted_names_as_one_experiment(tmp_path: Path) -> None:
    report = tmp_path / "duplicates.txt"
    folder = "/data/acfm/PXD000001"
    report.write_text(
        "\n".join(
            [
                f"{folder}/run.v1.ipc",
                f"{folder}/run.v2.ipc",
                "",
            ]
        )
    )

    file_map = parse_input_file(str(report))
    assert set(file_map) == {"run.v1", "run.v2"}
    assert find_duplicates_to_delete(file_map) == []
