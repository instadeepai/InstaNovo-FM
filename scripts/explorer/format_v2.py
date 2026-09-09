"""Write the explorer's column-sharded binary payload (format 2).

Format 2 differs from what the published explorer ships in five ways, each of which
exists because the published shape does not survive ten times the row count:

Coordinates are uint16, not float32.
    Halves the download. Over a UMAP extent of ~40 units, 65534 steps is 6e-4 units per
    step, and a 1300 px viewport at full zoom-out is 0.031 units per pixel -- so a
    quantisation step is 1/50 of a pixel. Dequantised once at load, so the viewer's
    coordinate reads are unchanged.

Categorical levels are ordered by descending frequency.
    Level 0 is the most common. This is what removes the per-draw sorts in the legend and
    filter panel: the top-N classes are now a prefix, not the result of sorting 400,000
    entries on every redraw.

Level tables are a binary blob plus an offset array, not JSON.
    At ~400,000 distinct peptides a JSON array is ~10 MB and about a second of
    main-thread parsing before anything renders. A blob is fetched once and decoded
    lazily, per level, only for the few hundred the UI actually shows.

Per-level counts live in a sidecar, not the catalog.
    Nothing whose size scales with the row count or the level count may go in the
    catalog; ``validate`` enforces that. The published manifest is 184 KB precisely
    because it inlines a count array per categorical.

Rows are ordered so each 50,000-row block is spatially coherent, and blocks are the unit
the viewer draws.
    Plotly's scattergl cost is superlinear in points *per trace* and nearly flat in the
    total, so the viewer draws all 1,000,000 points as ~20 traces rather than sampling.
    Coherent blocks additionally give each trace a tight bounding box.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

NAN_CODE = 65535
QUANT_MAX = 65534
CHUNK_ROWS = 50_000
BOOT_ROWS = 150_000
CATALOG_MAX_BYTES = 50_000
# Above this many levels the table is a sidecar rather than inline in the catalog.
INLINE_LEVELS_MAX = 512
# Level separator inside a blob. Cannot occur inside a level, so it is unambiguous.
SEPARATOR = b"\x00"


def quantise(values: np.ndarray, lo: float | None = None, hi: float | None = None) -> tuple[np.ndarray, float, float]:
    """Linearly quantise floats to uint16, reserving 65535 for missing.

    Returns the codes and the ``lo``/``hi`` the viewer needs to invert it. A column that
    is entirely missing, or constant, still has to round-trip, so the degenerate range is
    widened rather than left as a zero divisor.
    """
    v = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(v)
    if lo is None:
        lo = float(v[finite].min()) if finite.any() else 0.0
    if hi is None:
        hi = float(v[finite].max()) if finite.any() else 1.0
    if not (hi > lo):
        hi = lo + 1.0
    codes = np.full(v.shape, NAN_CODE, dtype=np.uint16)
    scaled = (v[finite] - lo) / (hi - lo) * QUANT_MAX
    codes[finite] = np.clip(np.rint(scaled), 0, QUANT_MAX).astype(np.uint16)
    return codes, lo, hi


def dequantise(codes: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Invert `quantise`, exactly as the viewer does. Used by the round-trip checks."""
    out = np.full(codes.shape, np.nan, dtype=np.float64)
    ok = codes != NAN_CODE
    out[ok] = lo + codes[ok].astype(np.float64) * (hi - lo) / QUANT_MAX
    return out


def encode_categorical(values: np.ndarray) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Dictionary-encode to codes ordered by descending frequency.

    Ties break on the label so two builds of the same data cannot disagree. Returns
    uint8 codes when the vocabulary fits, otherwise uint16 -- past 65,536 levels the
    caller has a different problem and gets told so.
    """
    labels = ["" if v is None else str(v) for v in values]
    counts = Counter(labels)
    levels = [lab for lab, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    if len(levels) > 65536:
        raise ValueError(f"{len(levels)} levels exceeds what a uint16 code can address")
    index = {lab: i for i, lab in enumerate(levels)}
    dtype = np.uint8 if len(levels) <= 256 else np.uint16
    codes = np.fromiter((index[lab] for lab in labels), dtype=dtype, count=len(labels))
    level_counts = np.array([counts[lab] for lab in levels], dtype=np.uint32)
    return codes, levels, level_counts


def morton_order(x: np.ndarray, y: np.ndarray, bits: int = 10) -> np.ndarray:
    """Row order that makes every contiguous block spatially coherent.

    Z-order over a 2^bits grid. Ten bits is a 1024x1024 grid, which at 1,000,000 points
    puts about one point per cell -- finer than that only reorders within a cell and
    costs a wider integer.
    """
    side = (1 << bits) - 1

    def grid(v: np.ndarray) -> np.ndarray:
        finite = np.isfinite(v)
        if not finite.any():
            return np.zeros(len(v), dtype=np.uint64)
        lo, hi = float(v[finite].min()), float(v[finite].max())
        if not (hi > lo):
            hi = lo + 1.0
        g = np.zeros(len(v), dtype=np.uint64)
        g[finite] = np.clip(((v[finite] - lo) / (hi - lo) * side).astype(np.int64), 0, side).astype(np.uint64)
        return g

    gx, gy = grid(np.asarray(x, dtype=np.float64)), grid(np.asarray(y, dtype=np.float64))
    key = np.zeros(len(gx), dtype=np.uint64)
    for b in range(bits):
        key |= ((gx >> np.uint64(b)) & np.uint64(1)) << np.uint64(2 * b)
        key |= ((gy >> np.uint64(b)) & np.uint64(1)) << np.uint64(2 * b + 1)
    # Stable, so rows sharing a cell keep their incoming order and the build is repeatable.
    return np.argsort(key, kind="stable").astype(np.uint32)


def write_levels(base: Path, key: str, levels: list[str], counts: np.ndarray) -> dict[str, Any]:
    """Write a level table as a NUL-separated UTF-8 blob plus uint32 offsets, and its counts.

    Small vocabularies stay inline in the catalog: the legend needs them before any
    column has arrived, and a few hundred bytes of JSON is cheaper than a request.
    """
    if len(levels) <= INLINE_LEVELS_MAX:
        return {"levels": levels, "count": counts.tolist()}

    encoded = [s.encode("utf-8") for s in levels]
    blob = SEPARATOR.join(encoded)
    # nlevels+1 offsets, so level i is blob[off[i] : off[i + 1] - 1]. The separator is
    # never parsed -- the offsets delimit exactly -- but NUL cannot occur inside a level,
    # so a mis-sized offset table shows up as a decode error rather than a wrong label.
    lengths = np.array([len(b) for b in encoded], dtype=np.uint32)
    offsets = np.zeros(len(levels) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(lengths + 1)
    (base / f"cat/{key}.levels.bin").write_bytes(blob)
    (base / f"cat/{key}.levels.off.u32").write_bytes(offsets.tobytes())
    (base / f"cat/{key}.count.u32").write_bytes(counts.tobytes())
    return {
        "levelsPath": f"cat/{key}.levels.bin",
        "levelsOffsetPath": f"cat/{key}.levels.off.u32",
        "countPath": f"cat/{key}.count.u32",
        "levelsBytes": len(blob),
    }


def write_search_blob(base: Path, key: str, levels: list[str]) -> dict[str, Any]:
    """An uppercased copy of a level table, for substring search.

    Searching the peptide field otherwise means uppercasing several hundred thousand
    JavaScript strings on every keystroke. Against this blob it is one byte scan.

    It carries its own offsets rather than reusing the level table's: uppercasing is not
    length-preserving in UTF-8 -- a sharp s becomes two bytes -- so shared offsets would
    misalign the moment a level contained one.
    """
    encoded = [s.upper().encode("utf-8") for s in levels]
    blob = SEPARATOR.join(encoded)
    lengths = np.array([len(b) for b in encoded], dtype=np.uint32)
    offsets = np.zeros(len(levels) + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(lengths + 1)
    (base / f"cat/{key}.search.bin").write_bytes(blob)
    (base / f"cat/{key}.search.off.u32").write_bytes(offsets.tobytes())
    return {
        "searchPath": f"cat/{key}.search.bin",
        "searchOffsetPath": f"cat/{key}.search.off.u32",
        "searchBytes": len(blob),
    }


def write_postings(base: Path, key: str, codes: np.ndarray, nlevels: int) -> dict[str, Any]:
    """Rows grouped by level, so "every row with these levels" is a slice concatenation.

    Turns the peptide-highlight scan from O(rows) into O(matches).
    """
    order = np.argsort(codes, kind="stable").astype(np.uint32)
    counts = np.bincount(codes, minlength=nlevels).astype(np.uint32)
    offsets = np.zeros(nlevels + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(counts)
    (base / f"cat/{key}.postings.u32").write_bytes(order.tobytes())
    (base / f"cat/{key}.postings.off.u32").write_bytes(offsets.tobytes())
    return {
        "postingsPath": f"cat/{key}.postings.u32",
        "postingsOffsetPath": f"cat/{key}.postings.off.u32",
    }


def chunk_bounds(x: np.ndarray, y: np.ndarray, rows: int, chunk: int = CHUNK_ROWS) -> list[list[float]]:
    """Bounding box per draw chunk, so the viewer can skip off-screen traces entirely."""
    out = []
    for s in range(0, rows, chunk):
        xs, ys = x[s : s + chunk], y[s : s + chunk]
        fx, fy = np.isfinite(xs), np.isfinite(ys)
        if not (fx.any() and fy.any()):
            out.append([0.0, 0.0, 0.0, 0.0])
            continue
        out.append([
            round(float(xs[fx].min()), 4), round(float(xs[fx].max()), 4),
            round(float(ys[fy].min()), 4), round(float(ys[fy].max()), 4),
        ])
    return out


def validate(catalog: dict[str, Any], base: Path) -> list[str]:
    """Check the invariants that keep first paint fast. Returns the problems found.

    The catalog is on the blocking path -- nothing renders until it parses -- so the rule
    is that nothing in it may scale with the row count or with a level count.
    """
    problems: list[str] = []

    text = json.dumps(catalog, separators=(",", ":"))
    if len(text) > CATALOG_MAX_BYTES:
        problems.append(f"catalog is {len(text):,} bytes, over the {CATALOG_MAX_BYTES:,} budget")

    n = catalog.get("n", 0)
    for key, meta in catalog.get("cats", {}).items():
        inline = meta.get("levels")
        if inline is not None and len(inline) > INLINE_LEVELS_MAX:
            problems.append(f"cat/{key} inlines {len(inline)} levels, over {INLINE_LEVELS_MAX}")
        if meta.get("nlevels", 0) > INLINE_LEVELS_MAX and "levelsPath" not in meta:
            problems.append(f"cat/{key} has {meta['nlevels']} levels but no sidecar")

    for group in ("arrays", "nums", "cats"):
        for key, meta in catalog.get(group, {}).items():
            path = meta.get("path")
            if path is None:
                continue
            f = base / path
            if not f.exists():
                problems.append(f"{group}/{key} names {path}, which was not written")
                continue
            width = {"uint8": 1, "uint16": 2, "uint32": 4, "float32": 4}[meta["dtype"]]
            expected = n * width
            if f.stat().st_size != expected:
                problems.append(
                    f"{group}/{key} is {f.stat().st_size:,} bytes, expected {expected:,} "
                    f"({n:,} rows x {width})"
                )
    return problems
