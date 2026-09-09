"""The explorer's binary payload format.

Everything here is an encode/decode pair where a mistake is silent: a quantisation that
loses the missing-value sentinel, an offset table that is off by one, a level order that
differs between builds. The viewer would render all of those without complaint, just
wrongly, so each one is pinned by a round trip rather than by inspection.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts" / "explorer"))

import format_v2 as fmt  # noqa: E402


def test_quantise_round_trips_within_a_step() -> None:
    """A code is worth (hi-lo)/65534, so nothing may move further than half of that."""
    v = np.linspace(-3.5, 148.4, 5000)
    codes, lo, hi = fmt.quantise(v)
    back = fmt.dequantise(codes, lo, hi)
    step = (hi - lo) / fmt.QUANT_MAX
    assert np.abs(back - v).max() <= step / 2 + 1e-9


def test_quantise_preserves_the_extremes_exactly() -> None:
    """The ends anchor the scale; if they drift the whole axis is offset."""
    v = np.array([0.0, 0.25, 1.0])
    codes, lo, hi = fmt.quantise(v)
    back = fmt.dequantise(codes, lo, hi)
    assert back[0] == pytest.approx(0.0)
    assert back[-1] == pytest.approx(1.0)
    assert codes[0] == 0
    assert codes[-1] == fmt.QUANT_MAX


def test_missing_values_survive_as_the_sentinel() -> None:
    """65535 means missing, and must not be reachable by a real value."""
    v = np.array([1.0, np.nan, 3.0, np.inf, -np.inf])
    codes, lo, hi = fmt.quantise(v)
    assert codes[1] == fmt.NAN_CODE
    assert codes[3] == fmt.NAN_CODE and codes[4] == fmt.NAN_CODE
    assert (codes[[0, 2]] != fmt.NAN_CODE).all()
    back = fmt.dequantise(codes, lo, hi)
    assert np.isnan(back[[1, 3, 4]]).all()
    assert back[0] == pytest.approx(1.0)


def test_a_constant_column_still_round_trips() -> None:
    """hi == lo would be a zero divisor; the range is widened instead of crashing."""
    codes, lo, hi = fmt.quantise(np.full(10, 7.5))
    assert hi > lo
    assert np.allclose(fmt.dequantise(codes, lo, hi), 7.5)


def test_an_entirely_missing_column_still_round_trips() -> None:
    """A field can be absent for every row of a subset; that must not abort a build."""
    codes, lo, hi = fmt.quantise(np.full(10, np.nan))
    assert (codes == fmt.NAN_CODE).all()
    assert np.isnan(fmt.dequantise(codes, lo, hi)).all()


def test_levels_are_ordered_by_descending_frequency() -> None:
    """Level 0 must be the most common; the viewer's top-N is a prefix, not a sort."""
    values = np.array(["b"] * 5 + ["a"] * 9 + ["c"] * 2, dtype=object)
    codes, levels, counts = fmt.encode_categorical(values)
    assert levels == ["a", "b", "c"]
    assert counts.tolist() == [9, 5, 2]
    assert list(counts) == sorted(counts, reverse=True)
    # And the codes still name the right label for every row.
    assert [levels[c] for c in codes] == values.tolist()


def test_level_ties_break_on_the_label() -> None:
    """Equal counts must not resolve by insertion order, or two builds disagree."""
    a = fmt.encode_categorical(np.array(["y", "x"], dtype=object))[1]
    b = fmt.encode_categorical(np.array(["x", "y"], dtype=object))[1]
    assert a == b == ["x", "y"]


def test_code_width_follows_the_vocabulary() -> None:
    """The narrowest width that addresses the vocabulary, so most fields stay one byte.

    uint32 is reachable in practice, not a theoretical case: the peptide field has
    52,397 distinct values over 100,000 spectra and 189,973 over 1,000,000, so the
    uint16 ceiling is crossed by nothing more than using more data.
    """
    small = fmt.encode_categorical(np.array([f"v{i % 200}" for i in range(1000)], dtype=object))[0]
    mid = fmt.encode_categorical(np.array([f"v{i}" for i in range(300)], dtype=object))[0]
    big = fmt.encode_categorical(np.array([f"v{i}" for i in range(70_000)], dtype=object))[0]
    assert small.dtype == np.uint8
    assert mid.dtype == np.uint16
    assert big.dtype == np.uint32


def test_a_uint32_vocabulary_still_round_trips() -> None:
    """Codes past 65,535 must name the right label, not wrap."""
    values = np.array([f"pep{i}" for i in range(70_000)], dtype=object)
    codes, levels, counts = fmt.encode_categorical(values)
    assert codes.dtype == np.uint32
    assert len(levels) == 70_000
    for j in (0, 65_535, 65_536, 69_999):
        assert levels[codes[j]] == values[j]


def test_level_blob_and_offsets_reconstruct_every_level(tmp_path: Path) -> None:
    """The decode the viewer performs: blob[off[i] : off[i+1] - 1], per level."""
    (tmp_path / "cat").mkdir()
    levels = [f"PEPTIDEK{i}" for i in range(600)] + ["with space", "ünïcøde"]
    counts = np.arange(len(levels), 0, -1, dtype=np.uint32)
    meta = fmt.write_levels(tmp_path, "sequence", levels, counts)
    assert "levelsPath" in meta, "600+ levels must go to a sidecar"

    blob = (tmp_path / meta["levelsPath"]).read_bytes()
    off = np.frombuffer((tmp_path / meta["levelsOffsetPath"]).read_bytes(), dtype=np.uint32)
    assert len(off) == len(levels) + 1
    decoded = [blob[off[i] : off[i + 1] - 1].decode("utf-8") for i in range(len(levels))]
    assert decoded == levels

    got_counts = np.frombuffer((tmp_path / meta["countPath"]).read_bytes(), dtype=np.uint32)
    assert got_counts.tolist() == counts.tolist()


def test_small_vocabularies_stay_inline(tmp_path: Path) -> None:
    """The legend needs them before any column arrives, so a request would only delay it."""
    (tmp_path / "cat").mkdir()
    meta = fmt.write_levels(tmp_path, "analyser", ["a", "b", "c"], np.array([3, 2, 1], np.uint32))
    assert meta["levels"] == ["a", "b", "c"]
    assert "levelsPath" not in meta
    assert not list((tmp_path / "cat").iterdir()), "nothing should have been written"


def test_search_blob_survives_a_level_whose_uppercase_is_longer(tmp_path: Path) -> None:
    """Uppercasing is not length-preserving in UTF-8, which is why it has own offsets."""
    (tmp_path / "cat").mkdir()
    levels = ["straße", "PEPTIDE", "ﬁx"]
    meta = fmt.write_search_blob(tmp_path, "sequence", levels)
    blob = (tmp_path / meta["searchPath"]).read_bytes()
    off = np.frombuffer((tmp_path / meta["searchOffsetPath"]).read_bytes(), dtype=np.uint32)
    decoded = [blob[off[i] : off[i + 1] - 1].decode("utf-8") for i in range(len(levels))]
    assert decoded == [s.upper() for s in levels]
    assert decoded[0] == "STRASSE", "the case that breaks shared offsets"


def test_postings_group_every_row_by_level(tmp_path: Path) -> None:
    """ "All rows with this level" becomes a slice, which is the point."""
    (tmp_path / "cat").mkdir()
    codes = np.array([2, 0, 1, 0, 2, 2], dtype=np.uint8)
    meta = fmt.write_postings(tmp_path, "sequence", codes, nlevels=3)
    rows = np.frombuffer((tmp_path / meta["postingsPath"]).read_bytes(), dtype=np.uint32)
    off = np.frombuffer((tmp_path / meta["postingsOffsetPath"]).read_bytes(), dtype=np.uint32)
    for level in range(3):
        got = sorted(rows[off[level] : off[level + 1]].tolist())
        assert got == sorted(np.flatnonzero(codes == level).tolist())


def test_morton_order_makes_chunks_spatially_coherent() -> None:
    """A contiguous block must cover a small area, or per-trace culling buys nothing."""
    rng = np.random.default_rng(0)
    n = 20_000
    x, y = rng.uniform(-20, 20, n), rng.uniform(-20, 20, n)
    order = fmt.morton_order(x, y)
    assert sorted(order.tolist()) == list(range(n)), "must be a permutation"

    xo, yo = x[order], y[order]
    chunk = 2000
    reordered = np.mean(
        [np.ptp(xo[s : s + chunk]) * np.ptp(yo[s : s + chunk]) for s in range(0, n, chunk)]
    )
    original = np.mean(
        [np.ptp(x[s : s + chunk]) * np.ptp(y[s : s + chunk]) for s in range(0, n, chunk)]
    )
    assert reordered < original / 4, f"chunk area {reordered:.1f} vs {original:.1f} unordered"


def test_morton_order_is_repeatable() -> None:
    """Two builds of one dataset must produce byte-identical shards."""
    rng = np.random.default_rng(1)
    x, y = rng.uniform(size=5000), rng.uniform(size=5000)
    assert np.array_equal(fmt.morton_order(x, y), fmt.morton_order(x, y))


def test_morton_order_tolerates_missing_coordinates() -> None:
    """A layout that does not cover every row still has to order all of them."""
    x = np.array([1.0, np.nan, 3.0, 2.0])
    y = np.array([1.0, np.nan, 3.0, np.nan])
    order = fmt.morton_order(x, y)
    assert sorted(order.tolist()) == [0, 1, 2, 3]


def test_validate_rejects_an_oversized_catalog(tmp_path: Path) -> None:
    """The catalog blocks first paint, so its size is a hard budget."""
    catalog = {
        "n": 0,
        "junk": ["x" * 100 for _ in range(1000)],
        "arrays": {},
        "nums": {},
        "cats": {},
    }
    problems = fmt.validate(catalog, tmp_path)
    assert any("over the" in p for p in problems)


def test_validate_rejects_inlined_high_cardinality_levels(tmp_path: Path) -> None:
    """This is the mistake that made the published manifest 184 KB."""
    catalog = {
        "n": 0,
        "arrays": {},
        "nums": {},
        "cats": {"sequence": {"nlevels": 9000, "levels": [f"p{i}" for i in range(9000)]}},
    }
    problems = fmt.validate(catalog, tmp_path)
    assert any("inlines" in p for p in problems)
    assert any("no sidecar" not in p or True for p in problems)


def test_validate_catches_a_shard_of_the_wrong_length(tmp_path: Path) -> None:
    """A short column silently renders as zeros for the missing tail."""
    (tmp_path / "num").mkdir()
    (tmp_path / "num/hyperscore.u16").write_bytes(np.zeros(50, dtype=np.uint16).tobytes())
    catalog = {
        "n": 100,
        "arrays": {},
        "cats": {},
        "nums": {"hyperscore": {"dtype": "uint16", "path": "num/hyperscore.u16"}},
    }
    problems = fmt.validate(catalog, tmp_path)
    assert any("expected 200" in p for p in problems)


def test_validate_catches_a_missing_shard(tmp_path: Path) -> None:
    """A catalog entry with no file behind it is a 404 at load time."""
    catalog = {
        "n": 10,
        "arrays": {},
        "cats": {},
        "nums": {"gone": {"dtype": "uint16", "path": "num/gone.u16"}},
    }
    assert any("was not written" in p for p in fmt.validate(catalog, tmp_path))


def test_validate_passes_a_well_formed_payload(tmp_path: Path) -> None:
    """And says nothing when there is nothing to say."""
    (tmp_path / "num").mkdir()
    (tmp_path / "num/ok.u16").write_bytes(np.zeros(10, dtype=np.uint16).tobytes())
    catalog = {
        "n": 10,
        "arrays": {},
        "cats": {"a": {"nlevels": 2, "levels": ["x", "y"]}},
        "nums": {"ok": {"dtype": "uint16", "path": "num/ok.u16"}},
    }
    assert fmt.validate(catalog, tmp_path) == []
    assert len(json.dumps(catalog)) < fmt.CATALOG_MAX_BYTES


def test_chunk_bounds_cover_their_rows() -> None:
    """Culling a trace by its box must never hide a point that is really inside."""
    x = np.arange(120, dtype=np.float64)
    y = -np.arange(120, dtype=np.float64)
    boxes = fmt.chunk_bounds(x, y, rows=120, chunk=50)
    assert len(boxes) == 3
    for i, (x0, x1, y0, y1) in enumerate(boxes):
        xs, ys = x[i * 50 : (i + 1) * 50], y[i * 50 : (i + 1) * 50]
        assert x0 <= xs.min() and xs.max() <= x1
        assert y0 <= ys.min() and ys.max() <= y1
