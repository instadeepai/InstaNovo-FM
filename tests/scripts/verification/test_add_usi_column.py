"""Tests for add_usi_column helpers."""

from __future__ import annotations

from scripts.verification.add_usi_column import (
    extract_pxd_or_msv_accession,
    normalize_scan_identifier,
)


def test_extract_pxd_or_msv_accession_from_path() -> None:
    """PXD accessions are pulled from filepath segments."""
    assert (
        extract_pxd_or_msv_accession("/data/PXD009449/sample.parquet") == "PXD009449"
    )


def test_normalize_scan_identifier_thermo_style() -> None:
    """Thermo controller/scan text yields the numeric scan id."""
    assert (
        normalize_scan_identifier(
            "controllerType=0 controllerNumber=1 scan=1321"
        )
        == "1321"
    )
