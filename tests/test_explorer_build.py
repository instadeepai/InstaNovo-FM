"""End-to-end build of an explorer payload.

The unit tests cover the derivation rules and the wire format separately. What is left,
and what these cover, is the wiring: that every declared field actually gets written, that
the row order is applied consistently to *all* columns rather than some of them, and that
a layout covering only part of the pool leaves the rest marked missing instead of at zero.

A row-order bug is the dangerous one here. Applying the permutation to the coordinates but
not to a metadata column would produce a map that looks entirely plausible and is wrong
everywhere, so the coordinate/metadata correspondence is checked per row.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

REPO = Path(__file__).resolve().parent.parent
BUILDER = REPO / "scripts" / "explorer" / "build_explorer_data.py"

sys.path.insert(0, str(REPO / "scripts" / "explorer"))

import format_v2 as fmt  # noqa: E402

N = 3000


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A metadata parquet shaped like the eval harness's output."""
    rng = np.random.default_rng(0)
    out = tmp_path_factory.mktemp("src") / "meta.parquet"
    analysers = rng.choice(["FTMS", "ITMS"], N, p=[0.8, 0.2])
    activations = rng.choice(["hcd28.00", "cid35.00", "etd70.00@hcd25.00"], N, p=[0.9, 0.07, 0.03])
    pl.DataFrame(
        {
            "usi": [f"mzspec:PXD1:f:scan:{i}" for i in range(N)],
            "sample_idx": np.arange(N),
            "scan": [f"controllerType=0 controllerNumber=1 scan={i * 7}" for i in range(N)],
            "header": [
                f"{a} + c NSI d Full ms2 500.0@{v} [{100 + (i % 3) * 10}.0000-2000.0000]"
                for i, (a, v) in enumerate(zip(analysers, activations, strict=False))
            ],
            "sequence": rng.choice(
                ["PEPTIDEK", "ELVISLIVESR", "AC[UNIMOD:4]DEFGHIK", "SAMPLERR"], N
            ),
            "precursor_charge": rng.choice(["2", "3", "0", "4"], N, p=[0.6, 0.3, 0.05, 0.05]),
            "ptm_present": rng.integers(0, 2, N),
            "search_modifications": rng.choice(
                [
                    "default (N-term acetylation, Met oxidation)",
                    "SILAC",
                    "TMT18",
                    "Phosphorylation",
                ],
                N,
            ),
            "search_quant": rng.choice(["precursor", "TMT"], N, p=[0.9, 0.1]),
            "search_project": rng.choice([f"PXD{i:06d}" for i in range(5)], N),
            "experiment_name": rng.choice([f"run_{i}" for i in range(700)], N),
            "protein": rng.choice([f"P{i:05d}" for i in range(900)], N),
            "hyperscore": [f"{v:.3f}" for v in rng.uniform(1.8, 148.4, N)],
            "spectrum_confidence": rng.uniform(0.01, 0.6, N).astype(np.float32),
            "hydrophobicity": rng.uniform(-3.5, 2.4, N).astype(np.float32),
            "retention_time": [f"{v:.3f}" for v in rng.uniform(0.6, 10799.0, N)],
            "precursor_mz": [f"{v:.5f}" for v in rng.uniform(300.0, 1827.0, N)],
            "precursor_mass": rng.uniform(0.0, 7046.0, N).astype(np.float32),
            # A column that is missing for most rows, to exercise the NaN sentinel.
            "delta_mass": [
                ("nan" if i % 3 else f"{v:.4f}") for i, v in enumerate(rng.uniform(-1, 3, N))
            ],
            "umap_x": rng.uniform(-12, 18, N),
            "umap_y": rng.uniform(-9, 14, N),
        }
    ).write_parquet(out)
    return out


@pytest.fixture(scope="module")
def built(source: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """Run the builder as a subprocess, the way it is actually invoked."""
    out = tmp_path_factory.mktemp("built") / "pool"
    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--metadata",
            str(source),
            "--layout",
            "main=umap_x,umap_y",
            "--order-layout",
            "main",
            "--out",
            str(out),
            "--label",
            "fixture",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if result.returncode != 0:
        pytest.fail(f"builder failed:\n{result.stdout}\n{result.stderr}")
    assert "validation passed" in result.stdout
    return out, json.loads((out / "catalog.json").read_text())


def test_every_declared_field_is_written(built: tuple[Path, dict]) -> None:
    """A catalog entry with no shard behind it is a 404 at load time."""
    base, cat = built
    assert len(cat["nums"]) == 14
    assert len(cat["cats"]) == 23
    for group in ("arrays", "nums", "cats"):
        for key, meta in cat[group].items():
            assert (base / meta["path"]).exists(), f"{group}/{key} shard missing"


def test_the_catalog_stays_within_its_budget(built: tuple[Path, dict]) -> None:
    """It is on the blocking path, so nothing in it may scale with rows or levels."""
    base, cat = built
    assert (base / "catalog.json").stat().st_size < fmt.CATALOG_MAX_BYTES


def test_row_order_is_applied_consistently(built: tuple[Path, dict], source: Path) -> None:
    """The dangerous bug: permute the coordinates but not a metadata column.

    Checked per row rather than by distribution, because a mismatched permutation
    preserves every marginal distribution exactly while making the map wrong everywhere.
    """
    base, cat = built
    src = pl.read_parquet(source)
    order = fmt.morton_order(src["umap_x"].to_numpy(), src["umap_y"].to_numpy())

    axes = cat["layouts"]["main"]["axes"]
    x = fmt.dequantise(
        np.frombuffer((base / axes["x"]["path"]).read_bytes(), dtype=np.uint16),
        axes["x"]["lo"],
        axes["x"]["hi"],
    )
    step = (axes["x"]["hi"] - axes["x"]["lo"]) / fmt.QUANT_MAX
    assert np.abs(x - src["umap_x"].to_numpy()[order]).max() <= step / 2 + 1e-9

    # And a categorical must name the label belonging to that same reordered row.
    meta = cat["cats"]["search_project"]
    codes = np.frombuffer(
        (base / meta["path"]).read_bytes(),
        dtype=np.uint8 if meta["dtype"] == "uint8" else np.uint16,
    )
    got = [meta["levels"][c] for c in codes]
    assert got == src["search_project"].to_numpy()[order].tolist()

    # As must an identity column, which the point panel shows next to the peptide.
    scan = np.frombuffer((base / cat["arrays"]["scan"]["path"]).read_bytes(), dtype=np.uint32)
    assert scan.tolist() == (np.arange(N) * 7)[order].tolist()


def test_a_mostly_missing_column_keeps_its_sentinel(built: tuple[Path, dict]) -> None:
    """Two thirds of delta_mass is the string "nan"; those rows must read as missing."""
    base, cat = built
    meta = cat["nums"]["delta_mass"]
    codes = np.frombuffer((base / meta["path"]).read_bytes(), dtype=np.uint16)
    missing = int((codes == fmt.NAN_CODE).sum())
    assert missing == pytest.approx(N * 2 / 3, abs=2), f"{missing} missing of {N}"


def test_levels_are_frequency_ordered_in_the_output(built: tuple[Path, dict]) -> None:
    """Level 0 must be the most common, which is what the viewer's top-N relies on."""
    _, cat = built
    inline = [m for m in cat["cats"].values() if "count" in m]
    assert inline, "the fixture should produce at least one inline level table"
    for meta in inline:
        assert meta["count"] == sorted(meta["count"], reverse=True)


def test_high_cardinality_fields_get_sidecars(built: tuple[Path, dict]) -> None:
    """700 runs and 900 proteins are past the inline limit, so they must be sidecar tables."""
    base, cat = built
    for key in ("experiment_name", "protein"):
        meta = cat["cats"][key]
        assert meta["nlevels"] > fmt.INLINE_LEVELS_MAX
        assert "levelsPath" in meta, f"{key} should not be inline"
        assert (base / meta["countPath"]).exists()
        blob = (base / meta["levelsPath"]).read_bytes()
        off = np.frombuffer((base / meta["levelsOffsetPath"]).read_bytes(), dtype=np.uint32)
        assert len(off) == meta["nlevels"] + 1
        assert blob[off[0] : off[1] - 1].decode()


def test_the_peptide_field_is_searchable(built: tuple[Path, dict]) -> None:
    """Search and highlight both need their sidecars, or the box silently does nothing."""
    base, cat = built
    meta = cat["cats"]["sequence"]
    for key in ("searchPath", "postingsPath", "postingsOffsetPath"):
        assert key in meta, f"sequence has no {key}"
        assert (base / meta[key]).exists()


def test_chunk_bounds_are_declared_per_draw_chunk(built: tuple[Path, dict]) -> None:
    """The viewer culls whole traces by these, so there must be one per chunk."""
    _, cat = built
    expected = (cat["n"] + cat["chunkRows"] - 1) // cat["chunkRows"]
    assert len(cat["layouts"]["main"]["chunkBounds"]) == expected


def test_a_partial_layout_marks_the_rows_it_does_not_cover(source: Path, tmp_path: Path) -> None:
    """Figure 3's 100k inside the 1M pool is this case: uncovered rows read as missing.

    Zero would be a plausible coordinate and would pile those rows at the origin, so the
    distinction between "at 0" and "not in this layout" has to survive the round trip.
    """
    subset = pl.read_parquet(source).head(1000).select(["usi", "umap_x", "umap_y"])
    partial = tmp_path / "partial.parquet"
    subset.write_parquet(partial)

    out = tmp_path / "pool"
    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--metadata",
            str(source),
            "--layouts",
            str(partial),
            "--layout",
            "subset=umap_x,umap_y",
            "--order-layout",
            "subset",
            "--out",
            str(out),
            "--label",
            "partial",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"
    cat = json.loads((out / "catalog.json").read_text())
    assert cat["layouts"]["subset"]["covered"] == 1000
    assert cat["n"] == N

    axes = cat["layouts"]["subset"]["axes"]
    codes = np.frombuffer((out / axes["x"]["path"]).read_bytes(), dtype=np.uint16)
    assert int((codes == fmt.NAN_CODE).sum()) == N - 1000


def test_positional_alignment_refuses_a_length_mismatch(source: Path, tmp_path: Path) -> None:
    """Aligning by position needs equal lengths; guessing would silently shift every row."""
    short = tmp_path / "short.parquet"
    pl.read_parquet(source).head(10).write_parquet(short)
    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--metadata",
            str(short),
            "--layout",
            "main=umap_x,umap_y",
            "--out",
            str(tmp_path / "x"),
            "--label",
            "t",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    # 10 rows aligned against 10 rows is fine; the guard is for a mismatch, so build a real one.
    assert result.returncode == 0, result.stderr

    result = subprocess.run(
        [
            sys.executable,
            str(BUILDER),
            "--metadata",
            str(source),
            "--layouts",
            str(short),
            "--layout",
            "main=nope_x,nope_y",
            "--out",
            str(tmp_path / "y"),
            "--label",
            "t",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert result.returncode != 0
    assert "no input provided these layout columns" in result.stdout + result.stderr
