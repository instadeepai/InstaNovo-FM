"""Tests for ACFM IPC shard naming during conversion."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest

from scripts.preprocessing.check_conversion import is_shard_incomplete
from scripts.preprocessing.convert_ipc_to_parquet import convert_ipc_with_metadata

_SHARD_PATTERN = re.compile(r".+_\d{4}-\d{4}.parquet$")


def _minimal_ipc_frame(n_rows: int) -> pl.DataFrame:
    """Build a small IPC frame with columns conversion can cast and enrich."""
    return pl.DataFrame(
        {
            "scan": [str(i) for i in range(n_rows)],
            "precursor_mz": [500.0 + i for i in range(n_rows)],
            "precursor_charge": [2] * n_rows,
            "mz_array": [[100.0, 200.0]] * n_rows,
            "intensity_array": [[1.0, 2.0]] * n_rows,
        }
    )


def _write_ipc(path: Path, n_rows: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _minimal_ipc_frame(n_rows).write_ipc(path)


def _parquet_names(ipc_path: Path) -> list[str]:
    stem = ipc_path.name.replace(".ipc", "")
    return sorted(
        p.name
        for p in ipc_path.parent.iterdir()
        if p.suffix == ".parquet" and p.name.startswith(stem)
    )


def test_exact_multiple_of_max_shard_size_names_match_count(tmp_path: Path) -> None:
    """Exact multiples must encode the true shard count in filenames."""
    project = "PXD000001"
    ipc_path = tmp_path / project / "sample.mzML.ipc"
    _write_ipc(ipc_path, n_rows=4)

    convert_ipc_with_metadata(
        str(ipc_path),
        acquisition_map={(project, "sample"): "DDA"},
        max_shard_size=2,
        add_usi=False,
    )

    names = _parquet_names(ipc_path)
    assert names == [
        "sample.mzML_0000-0002.parquet",
        "sample.mzML_0001-0002.parquet",
    ]
    assert not is_shard_incomplete(names, _SHARD_PATTERN)


def test_remainder_shard_names_match_count(tmp_path: Path) -> None:
    """Non-exact multiples keep ceil(len / max) in both count and names."""
    project = "PXD000001"
    ipc_path = tmp_path / project / "sample.mzML.ipc"
    _write_ipc(ipc_path, n_rows=5)

    convert_ipc_with_metadata(
        str(ipc_path),
        acquisition_map={(project, "sample"): "DDA"},
        max_shard_size=2,
        add_usi=False,
    )

    names = _parquet_names(ipc_path)
    assert names == [
        "sample.mzML_0000-0003.parquet",
        "sample.mzML_0001-0003.parquet",
        "sample.mzML_0002-0003.parquet",
    ]
    assert not is_shard_incomplete(names, _SHARD_PATTERN)


def test_empty_ipc_raises_and_writes_no_parquet(tmp_path: Path) -> None:
    """Empty IPC must not produce Parquet; callers log the error and continue."""
    project = "PXD000001"
    ipc_path = tmp_path / project / "sample.mzML.ipc"
    _write_ipc(ipc_path, n_rows=0)

    with pytest.raises(ValueError, match="IPC file is empty"):
        convert_ipc_with_metadata(
            str(ipc_path),
            acquisition_map={(project, "sample"): "DDA"},
            max_shard_size=2,
            add_usi=False,
        )

    assert _parquet_names(ipc_path) == []
