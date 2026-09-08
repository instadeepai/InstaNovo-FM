"""Tests for parquet_io helpers used to join search-data rows to on-disk files."""

from __future__ import annotations

from scripts.preprocessing.parquet_io import (
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
