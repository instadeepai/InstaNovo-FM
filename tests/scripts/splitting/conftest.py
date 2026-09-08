"""Shared fixtures for splitting script tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import pytest


@dataclass
class SplitEnv:
    """Temporary split input/output layout."""

    test_dir: Path
    split_dir: Path
    output_dir: Path


def write_split_files(
    split_dir: Path, files_per_split: int, rows_per_file: int
) -> None:
    """Write train/valid/test parquet shards with sequential ids."""
    for split_type in ("train", "valid", "test"):
        for i in range(files_per_split):
            start = i * rows_per_file
            end = (i + 1) * rows_per_file
            data = {
                "id": list(range(start, end)),
                "sequence": [f"ATCG{j % 100}" for j in range(start, end)],
                "quality": [j % 50 for j in range(start, end)],
                "metadata": [f"sample_{j % 10}" for j in range(start, end)],
                "mz": [[100.0 + j, 200.0 + j, 300.0 + j] for j in range(start, end)],
                "intensity": [
                    [100.0 + j, 200.0 + j, 300.0 + j] for j in range(start, end)
                ],
            }
            pl.DataFrame(data).write_parquet(split_dir / f"{split_type}_{i}.parquet")


def write_integration_split_files(split_dir: Path) -> None:
    """Write uneven train/valid/test shards used by shuffle integration tests."""
    split_configs = {
        "train": {"files": 8, "rows_per_file": 250},
        "valid": {"files": 2, "rows_per_file": 200},
        "test": {"files": 2, "rows_per_file": 150},
    }
    for split_type, config in split_configs.items():
        rows_per_file = config["rows_per_file"]
        for i in range(config["files"]):
            start = i * rows_per_file
            end = (i + 1) * rows_per_file
            data = {
                "id": list(range(start, end)),
                "sequence": [f"ATCG{j % 100}" for j in range(start, end)],
                "quality": [j % 50 for j in range(start, end)],
                "metadata": [f"sample_{j % 10}" for j in range(start, end)],
                "mz": [[100.0 + j, 200.0 + j, 300.0 + j] for j in range(start, end)],
                "intensity": [
                    [100.0 + j, 200.0 + j, 300.0 + j] for j in range(start, end)
                ],
            }
            pl.DataFrame(data).write_parquet(split_dir / f"{split_type}_{i}.parquet")


def _make_env(tmp_path: Path) -> SplitEnv:
    split_dir = tmp_path / "lcfm_splits"
    output_dir = tmp_path / "outputs"
    split_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    return SplitEnv(test_dir=tmp_path, split_dir=split_dir, output_dir=output_dir)


@pytest.fixture
def shuffle_indices_env(tmp_path: Path) -> SplitEnv:
    env = _make_env(tmp_path)
    write_split_files(env.split_dir, files_per_split=3, rows_per_file=100)
    return env


@pytest.fixture
def shuffle_2pass_env(tmp_path: Path) -> SplitEnv:
    env = _make_env(tmp_path)
    (tmp_path / "temp").mkdir()
    write_split_files(env.split_dir, files_per_split=5, rows_per_file=200)
    return env


@pytest.fixture
def shuffle_integration_env(tmp_path: Path) -> SplitEnv:
    env = _make_env(tmp_path)
    write_integration_split_files(env.split_dir)
    return env
