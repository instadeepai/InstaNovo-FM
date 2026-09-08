"""Shared fixtures for preprocessing script tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl
import pytest


@dataclass
class PreprocessingEnv:
    """Temporary input/output layout used by preprocessing CLI tests."""

    data_dir: Path
    output_dir: Path
    gold_standard_file: Path
    pxd009449_file: Path


def sample_spectrum_data() -> dict[str, Any]:
    """Realistic spectrum rows matching the conversion-script schema."""
    return {
        "index": [1, 2, 3, 4, 5],
        "scan": ["scan_001", "scan_002", "scan_003", "scan_004", "scan_005"],
        "header": [
            "MS2 scan 1234.5@30.0",
            "MS2 scan 2345.6@35.0",
            "MS2 scan 3456.7@40.0",
            "MS2 scan 4567.8@45.0",
            "MS2 scan 5678.9@50.0",
        ],
        "rt": [30.0, 35.0, 40.0, 45.0, 50.0],
        "frag_type": ["HCD", "HCD", "HCD", "HCD", "HCD"],
        "collision_energy": [30.0, 35.0, 40.0, 45.0, 50.0],
        "precursor_mz": [1234.5, 2345.6, 3456.7, 4567.8, 5678.9],
        "precursor_charge": [2, 3, 2, 3, 2],
        "precursor_intensity": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
        "lower_offset": [-1.0, -1.0, -1.0, -1.0, -1.0],
        "upper_offset": [1.0, 1.0, 1.0, 1.0, 1.0],
        "isolation_target": [None, None, None, None, None],
        "mz": [
            [100.0, 200.0, 300.0],
            [150.0, 250.0, 350.0],
            [200.0, 300.0, 400.0],
            [250.0, 350.0, 450.0],
            [300.0, 400.0, 500.0],
        ],
        "intensity": [
            [100.0, 200.0, 300.0],
            [150.0, 250.0, 350.0],
            [200.0, 300.0, 400.0],
            [250.0, 350.0, 450.0],
            [300.0, 400.0, 500.0],
        ],
        "scale_factor": [1.0, 1.0, 1.0, 1.0, 1.0],
    }


def create_gold_standard_modifications(output_dir: Path) -> Path:
    """Write a gold-standard modifications Excel file."""
    gold_standard_data = {
        "modification": [
            "K[242]",
            "R[170]",
            "n[43]V",
            "Qc[111]",
            "M[142]",
            "n[145]P",
        ],
        "project_name": ["PXD037009"] * 6,
        "file_name": [
            "file1.mzML",
            "file2.mzML",
            "file3.mzML",
            "file4.mzML",
            "file5.mzML",
            "file6.mzML",
        ],
        "proposed_unimod_encoding": [
            "[UNIMOD:121]",
            "[UNIMOD:1]",
            "[UNIMOD:1]",
            "[UNIMOD:23]",
            "[UNIMOD:34]",
            "[UNIMOD:214]",
        ],
    }
    file_path = output_dir / "gold_standard_modifications.xlsx"
    pl.DataFrame(gold_standard_data).write_excel(file_path)
    return file_path


def create_pxd009449_ambiguous_modifications(output_dir: Path) -> Path:
    """Write a PXD009449 ambiguous-modifications Excel file."""
    ambiguous_data = {
        "modification": ["K[242]", "K[242]"],
        "project_name": ["PXD009449", "PXD009449"],
        "modification_in_file_name": ["ubiquitin", "acetylation"],
        "proposed_unimod_encoding": ["[UNIMOD:1848]", "[UNIMOD:21]"],
    }
    file_path = output_dir / "pxd009449_ambiguous_modifications.xlsx"
    pl.DataFrame(ambiguous_data).write_excel(file_path)
    return file_path


def _write_sample_files(data_dir: Path, base_data: dict[str, Any]) -> None:
    df1 = pl.DataFrame(base_data)
    df1.write_ipc(data_dir / "acfm" / "sample1.ipc")
    df1.write_ipc(data_dir / "acfm" / "sample1.mzML.ipc")
    df1.write_ipc(data_dir / "acfm" / "sample1_copy.ipc")
    df1.write_ipc(data_dir / "lcfm" / "sample1.ipc")
    df1.write_ipc(data_dir / "mcfm" / "sample1.ipc")
    df1.write_ipc(data_dir / "hcfm" / "sample1.ipc")

    empty_df = pl.DataFrame({col: [] for col in base_data.keys()})
    empty_df.write_ipc(data_dir / "lcfm" / "empty.ipc")

    mod_data = base_data.copy()
    mod_data["modified_peptide"] = [
        "PEPTIDE[142]K",
        "PEPTIDE[3562]R",
        "PEPTIDE[2346]K",
        "PEPTIDE[3923]R",
        "PEPTIDE[2449]K",
    ]
    mod_data["unmodified_peptide"] = [
        "PEPTIDEK",
        "PEPTIDER",
        "PEPTIDEK",
        "PEPTIDER",
        "PEPTIDEK",
    ]
    pl.DataFrame(mod_data).write_parquet(data_dir / "mcfm" / "modified.parquet")

    unknown_data = base_data.copy()
    unknown_data["collision_energy"] = pl.Series(
        [None if i % 3 == 0 else 30.0 + i for i in range(5)]
    )
    pl.DataFrame(unknown_data).write_parquet(data_dir / "hcfm" / "unknown_ce.parquet")

    null_data = base_data.copy()
    null_data["isolation_target"] = [None, None, None, None, None]
    pl.DataFrame(null_data).write_parquet(data_dir / "acfm" / "null_targets.parquet")


@pytest.fixture
def preprocessing_env(tmp_path: Path) -> PreprocessingEnv:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "outputs"
    data_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    for subdir in ("acfm", "lcfm", "mcfm", "hcfm"):
        (data_dir / subdir).mkdir()

    gold_standard_file = create_gold_standard_modifications(output_dir)
    pxd009449_file = create_pxd009449_ambiguous_modifications(output_dir)
    _write_sample_files(data_dir, sample_spectrum_data())

    return PreprocessingEnv(
        data_dir=data_dir,
        output_dir=output_dir,
        gold_standard_file=gold_standard_file,
        pxd009449_file=pxd009449_file,
    )
