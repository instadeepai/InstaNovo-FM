"""Tests for the per-project preparation pass.

The pass rewrites ~46,000 files, so the properties worth pinning are the ones whose
failure would be invisible: a silently nulled column, a join key derived in the wrong
direction, or a target schema that has drifted from the pipeline's own.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


prep = _load(REPO / "scripts" / "release" / "prepare_by_project.py", "prep")


def test_target_schema_matches_the_pipeline_reference() -> None:
    """TARGET_SCHEMA is a copy; it must not drift from REFERENCE_SCHEMA.

    The copy exists because the splitting module imports huggingface_hub, which the
    image running the pass does not carry. A copy that silently diverges would write a
    corpus whose schema disagrees with the pipeline that produced the splits.
    """
    src = (REPO / "scripts" / "splitting" / "split_labelled_data.py").read_text()
    start = src.index("REFERENCE_SCHEMA: Dict[str, pl.DataType] = {")
    body = src[start : src.index("}", start)]
    reference = {
        m.group(1): m.group(2) for m in re.finditer(r'"(\w+)":\s*(pl\.[A-Za-z0-9_().]+)', body)
    }
    # normalised_peptide is deliberately dropped by the pass
    expected = {k: v for k, v in reference.items() if k != "normalised_peptide"}
    actual = {k: f"pl.{v!s}" for k, v in prep.TARGET_SCHEMA.items()}
    assert set(expected) == set(actual), (
        f"schema drift: only in pipeline={set(expected) - set(actual)}, "
        f"only in pass={set(actual) - set(expected)}"
    )


def _frame(**overrides) -> pl.DataFrame:
    base = {
        "unmodified_peptide": ["PEPTIDEIK", "AAALLLIII"],
        "sequence": ["PEPTIDEIK", "AAALLLIII"],
        "precursor_charge": [2, 3],
    }
    base.update(overrides)
    return pl.DataFrame(base)


def test_string_collision_energy_is_cast_and_parses_cleanly() -> None:
    """A numeric string becomes a float without being reported as a loss."""
    out, losses = prep.prepare_frame(_frame(collision_energy=["25.0", "30.5"]))
    assert out["collision_energy"].to_list() == [25.0, 30.5]
    assert losses == {}


def test_unparseable_collision_energy_is_reported_not_hidden() -> None:
    """A non-numeric string must be counted, because it becomes null."""
    out, losses = prep.prepare_frame(_frame(collision_energy=["25.0", "NCE 30"]))
    assert out["collision_energy"].to_list() == [25.0, None]
    assert losses == {"collision_energy": 1}, "a silent null is the failure mode"


def test_integer_auc_intensity_is_widened_without_loss() -> None:
    """Int64 -> Float64 is exact for these magnitudes."""
    out, losses = prep.prepare_frame(_frame(auc_intensity=[12345, 67890]))
    assert out["auc_intensity"].to_list() == [12345.0, 67890.0]
    assert losses == {}


def test_absent_retention_is_added_as_null() -> None:
    """A missing column is filled, not treated as a divergence."""
    out, losses = prep.prepare_frame(_frame())
    assert "retention" in out.columns
    assert out["retention"].to_list() == [None, None]
    assert losses == {}


def test_registry_key_collapses_I_to_L_not_the_reverse() -> None:
    """The registry's own peptide column contains no I; the key must match that."""
    out, _ = prep.prepare_frame(_frame())
    keys = out[prep.REGISTRY_KEY].to_list()
    assert keys == ["PEPTLDELK", "AAALLLLLL"]
    assert not any("I" in k for k in keys), "wrong direction: keys would never match"


def test_dropped_columns_are_gone() -> None:
    """normalised_peptide and isolation_target_old must not reach the release."""
    out, _ = prep.prepare_frame(
        _frame(normalised_peptide=[None, None], isolation_target_old=[1.0, 2.0])
    )
    for col in prep.DROP_COLUMNS:
        assert col not in out.columns


def test_output_column_order_is_the_target_schema_then_registry_key() -> None:
    """A stable column order is what lets the files unify into one scan."""
    out, _ = prep.prepare_frame(_frame())
    assert out.columns == [*prep.TARGET_SCHEMA.keys(), prep.REGISTRY_KEY]


def test_files_with_divergent_types_unify_after_the_pass(tmp_path: Path) -> None:
    """The point of the whole exercise: a single scan over mixed files must work."""
    a = _frame(collision_energy=["25.0", "27.0"], auc_intensity=[1, 2])
    b = _frame(collision_energy=[25.0, 27.0], auc_intensity=[1.5, 2.5], retention=[3.0, 4.0])
    # Before the pass these do not unify, for both of the reasons seen in the corpus:
    # collision_energy/auc_intensity clash on dtype, and b carries a column a lacks.
    with pytest.raises((pl.exceptions.SchemaError, pl.exceptions.ShapeError)):
        pl.concat([a, b], how="vertical")
    outs = []
    for i, df in enumerate((a, b)):
        prepared, _ = prep.prepare_frame(df)
        p = tmp_path / f"part{i}.parquet"
        prepared.write_parquet(p)
        outs.append(p)
    scanned = pl.scan_parquet(outs).collect()
    assert scanned.height == 4
    assert scanned.schema["collision_energy"] == pl.Float64
    assert scanned.schema["auc_intensity"] == pl.Float64


def test_float64_intensity_array_narrowing_is_counted_not_silent() -> None:
    """A Float64 intensity_array narrowed to Float32 changes values without nulling them.

    The parse-failure count cannot see this: every element stays non-null. Without a
    separate check the pass would quietly reduce precision on every such file, which is
    the shape of mistake this whole script is arranged to prevent.
    """
    # 0.1 is not representable in binary32, so it moves under narrowing.
    df = _frame(intensity_array=[[0.1, 0.2], [0.3, 0.4]])
    assert df.schema["intensity_array"] == pl.List(pl.Float64)
    out, losses = prep.prepare_frame(df)
    assert out.schema["intensity_array"] == pl.List(pl.Float32)
    assert "intensity_array (precision)" in losses, f"narrowing went unreported: {losses}"
    assert losses["intensity_array (precision)"] == 4


def test_exactly_representable_values_are_not_reported_as_narrowing() -> None:
    """Halves and integers survive Float32 exactly, so they must not be flagged."""
    out, losses = prep.prepare_frame(_frame(intensity_array=[[0.5, 1.0], [2.0, 4.0]]))
    assert out.schema["intensity_array"] == pl.List(pl.Float32)
    assert losses == {}, f"false positive on exact values: {losses}"


def test_mz_array_float32_widening_is_not_flagged() -> None:
    """Widening Float32 to the Float64 target is exact and must pass silently."""
    df = _frame(mz_array=pl.Series([[100.5, 200.25], [300.0, 400.0]], dtype=pl.List(pl.Float32)))
    out, losses = prep.prepare_frame(df)
    assert out.schema["mz_array"] == pl.List(pl.Float64)
    assert losses == {}


def test_float32_values_stored_as_float64_narrow_back_with_no_change() -> None:
    """The case the corpus actually presents, and why Float32 is the right target.

    Intensity is 32-bit at source in vendor and mzML files. Where a file stores
    intensity_array as Float64, that width came from a conversion step, not from the
    instrument. Widening Float32 to Float64 is exact, so narrowing it back recovers the
    original bits and the precision check must report nothing.

    This is what distinguishes spurious width from real precision: if the pass reports
    changed values on those files, the values did not originate as Float32 and the
    narrowing would be a genuine loss worth stopping for.
    """
    original = pl.Series([[0.1, 0.2], [0.3, 0.4]], dtype=pl.List(pl.Float32))
    widened = original.cast(pl.List(pl.Float64))
    assert widened.dtype == pl.List(pl.Float64), "fixture must present as Float64"

    out, losses = prep.prepare_frame(_frame(intensity_array=widened))

    assert out.schema["intensity_array"] == pl.List(pl.Float32)
    assert losses == {}, f"round-tripped Float32 must be reported clean, got {losses}"
    assert out["intensity_array"].to_list() == original.to_list()
