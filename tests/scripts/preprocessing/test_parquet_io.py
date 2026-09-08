"""Tests for parquet_io helpers used to join search-data rows to on-disk files."""

from __future__ import annotations

from scripts.preprocessing.parquet_io import search_data_lookup_key


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
