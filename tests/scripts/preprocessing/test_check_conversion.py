"""Tests for check_conversion helpers."""

from __future__ import annotations

import re
from pathlib import Path

from scripts.preprocessing.check_conversion import (
    find_missing_parquet_files,
    is_shard_incomplete,
)

_SHARD_PATTERN = re.compile(r".+_\d{4}-\d{4}.parquet$")


def test_is_shard_incomplete_detects_gap() -> None:
    """Incomplete when shard count does not match the declared total."""
    incomplete = [
        "sample_0001-0003.parquet",
        "sample_0003-0003.parquet",
    ]
    complete = [
        "sample_0001-0003.parquet",
        "sample_0002-0003.parquet",
        "sample_0003-0003.parquet",
    ]
    assert is_shard_incomplete(incomplete, _SHARD_PATTERN) is True
    assert is_shard_incomplete(complete, _SHARD_PATTERN) is False


def test_find_missing_parquet_files_no_parquet(tmp_path: Path) -> None:
    """IPC with no sibling parquet is reported as missing."""
    ipc = tmp_path / "sample.ipc"
    ipc.write_bytes(b"")

    missing = find_missing_parquet_files([str(ipc)])

    assert missing == [(str(ipc), "No matching Parquet files found")]
