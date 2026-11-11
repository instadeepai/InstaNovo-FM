"""Tests for parquet_io helpers used to join search-data rows to on-disk files."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from scripts.preprocessing.parquet_io import (
    atomic_write_parquet,
    get_storage_options,
    search_data_lookup_key,
    strip_known_data_suffix,
)


def test_lookup_key_excel_mzml_gz() -> None:
    assert (
        search_data_lookup_key("20220127_SILAC_1.mzML.gz") == "20220127_SILAC_1"
    )


def test_lookup_key_posix_parquet_with_shard() -> None:
    path = "/data/lcfm/MSV000090792/20220127_SILAC_1_0000-0002.parquet"
    assert search_data_lookup_key(path) == "20220127_SILAC_1"


def test_lookup_key_mzml_ipc() -> None:
    assert search_data_lookup_key("exp_sample.mzML.ipc") == "exp_sample"


def test_lookup_key_matches_excel_and_parquet() -> None:
    excel = "runA.mzML.gz"
    parquet = "PXD000001/runA_0001-0004.parquet"
    assert search_data_lookup_key(excel) == search_data_lookup_key(parquet)


def test_strip_known_data_suffix_keeps_dotted_experiment_names() -> None:
    assert strip_known_data_suffix("run.v1.ipc") == "run.v1"
    assert strip_known_data_suffix("run.v2.ipc") == "run.v2"
    assert strip_known_data_suffix("/data/acfm/PXD000001/run.v1.parquet") == "run.v1"


def test_strip_known_data_suffix_groups_ipc_with_mzml_ipc() -> None:
    assert strip_known_data_suffix("sample.ipc") == "sample"
    assert strip_known_data_suffix("sample.mzML.ipc") == "sample"


def test_get_storage_options_reads_named_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aws_dir = tmp_path / ".aws"
    aws_dir.mkdir()
    (aws_dir / "credentials").write_text(
        "[pipeline]\naws_access_key_id = AKIATEST\naws_secret_access_key = secret\n"
    )
    (aws_dir / "config").write_text("[profile pipeline]\nregion = eu-west-1\n")
    monkeypatch.setattr(
        "scripts.preprocessing.parquet_io._find_aws_dir", lambda: str(aws_dir)
    )

    assert get_storage_options("pipeline") == {
        "aws_access_key_id": "AKIATEST",
        "aws_secret_access_key": "secret",
        "aws_region": "eu-west-1",
    }
    assert get_storage_options(None) is None
    assert get_storage_options("missing") is None


def test_atomic_write_parquet_s3_requires_storage_options() -> None:
    with pytest.raises(ValueError, match="storage_options"):
        atomic_write_parquet(pl.DataFrame({"a": [1]}), "s3://bucket/key.parquet")


def test_atomic_write_parquet_s3_passes_storage_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    written: list[tuple[object, object]] = []

    def fake_write(
        self: pl.DataFrame, path: object, **kwargs: object
    ) -> None:
        written.append((path, kwargs.get("storage_options")))

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fake_write)
    opts = {
        "aws_access_key_id": "AKIATEST",
        "aws_secret_access_key": "secret",
        "aws_region": "eu-west-1",
    }
    atomic_write_parquet(pl.DataFrame({"a": [1]}), "s3://bucket/key.parquet", opts)
    assert written == [("s3://bucket/key.parquet", opts)]


def test_atomic_write_parquet_s3_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    def fake_write(self: pl.DataFrame, path: object, **_kwargs: object) -> None:
        raise RuntimeError("cloud write unavailable")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", fake_write)
    with pytest.raises(OSError, match="s3://bucket/key.parquet"):
        atomic_write_parquet(
            pl.DataFrame({"a": [1]}),
            "s3://bucket/key.parquet",
            {"aws_access_key_id": "id", "aws_secret_access_key": "secret"},
        )
    assert not list(tmp_path.rglob("*"))
