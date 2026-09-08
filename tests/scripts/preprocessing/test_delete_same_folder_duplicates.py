"""Tests for delete_same_folder_duplicates."""

from collections import defaultdict

from scripts.preprocessing.delete_same_folder_duplicates import find_duplicates_to_delete


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
