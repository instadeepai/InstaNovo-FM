"""Tests for reformulated spectral rescue sampling helpers."""

from __future__ import annotations

import numpy as np

from instanovo_fm.eval.embed_eval_tasks.spectral_rescue_reformulated import (
    SpectralRescueTaskReformulated,
)


def _record(usi: str, distance: int) -> dict:
    return {"usi": usi, "clean_site_edit_distance": distance}


def test_raw_file_from_usi_extracts_raw_file_component():
    usi = "mzspec:PXD047134:HFX_9619_TFU_PD-GBA-P40_00_260619:scan:70865:LEQGQAIDDLMPAQK/3"
    assert SpectralRescueTaskReformulated.raw_file_from_usi(usi) == "HFX_9619_TFU_PD-GBA-P40_00_260619"


def test_dedupe_by_raw_file_caps_candidates_per_file():
    records = [
        {"usi": "mzspec:PXD:run_a:scan:1:PEP/1"},
        {"usi": "mzspec:PXD:run_a:scan:2:PEP/1"},
        {"usi": "mzspec:PXD:run_b:scan:3:PEP/1"},
    ]
    rng = np.random.RandomState(0)
    deduped = SpectralRescueTaskReformulated.dedupe_by_raw_file(records, 1, rng)

    assert len(deduped) == 2
    raw_files = {SpectralRescueTaskReformulated.raw_file_from_usi(record["usi"]) for record in deduped}
    assert raw_files == {"run_a", "run_b"}


def test_sample_records_random_returns_requested_count():
    records = [_record(f"mzspec:PXD:run_{index}:scan:{index}:PEP/1", 10 + index) for index in range(20)]
    rng = np.random.RandomState(1)
    chosen = SpectralRescueTaskReformulated.sample_records(records, 5, rng, "negative library")
    assert len(chosen) == 5
    assert len({record["usi"] for record in chosen}) == 5


def test_sample_records_stratified_spreads_across_distance_bins():
    records = (
        [_record(f"mzspec:PXD:run_a:{index}:PEP/1", 6) for index in range(10)]
        + [_record(f"mzspec:PXD:run_b:{index}:PEP/1", 12) for index in range(10)]
        + [_record(f"mzspec:PXD:run_c:{index}:PEP/1", 18) for index in range(10)]
    )
    rng = np.random.RandomState(2)
    chosen = SpectralRescueTaskReformulated.sample_records(
        records,
        9,
        rng,
        "negative library",
        strategy="stratified_by_edit_distance",
    )

    bins = {
        SpectralRescueTaskReformulated.edit_distance_bin(record["clean_site_edit_distance"])
        for record in chosen
    }
    assert len(chosen) == 9
    assert bins == {"5-9", "10-14", "15-19"}


def test_sampling_profiles_define_demo_and_rigorous_presets():
    demo = SpectralRescueTaskReformulated.SAMPLING_PROFILES["demo"]
    rigorous = SpectralRescueTaskReformulated.SAMPLING_PROFILES["rigorous"]

    assert demo["num_reference_queries"] == 10
    assert demo["negative_sampling"] == "random"
    assert demo["max_queries_per_raw_file"] is None

    assert rigorous["num_reference_queries"] == 50
    assert rigorous["num_negative_library"] == 200
    assert rigorous["negative_sampling"] == "stratified_by_edit_distance"
    assert rigorous["max_queries_per_raw_file"] == 1
    assert rigorous["seeds"] == [42, 43, 44]


def test_stratified_sampling_backfills_when_bin_is_too_small():
    records = (
        [_record(f"mzspec:PXD:run_a:{index}:PEP/1", 6) for index in range(10)]
        + [_record(f"mzspec:PXD:run_b:{index}:PEP/1", 12) for index in range(2)]
        + [_record(f"mzspec:PXD:run_c:{index}:PEP/1", 22) for index in range(10)]
    )
    rng = np.random.RandomState(3)
    chosen = SpectralRescueTaskReformulated.sample_records(
        records,
        6,
        rng,
        "negative library",
        strategy="stratified_by_edit_distance",
    )

    assert len(chosen) == 6
    bins = {
        SpectralRescueTaskReformulated.edit_distance_bin(record["clean_site_edit_distance"])
        for record in chosen
    }
    assert "5-9" in bins
    assert "20-24" in bins
