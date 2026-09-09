#!/usr/bin/env python
"""Build the interactive explorer's data payload from per-spectrum metadata and UMAP layouts.

This is the stage the explorer never had in version control. The published payload was
sharded out of a 17.7 MB prototype HTML whose own generator has no source anywhere, so
regenerating the explorer at a different point count meant reconstructing it. See
``derive.py`` for the eight computed display fields and ``format_v2.py`` for the wire
format.

Metadata comes from either the eval harness's ``embeddings.h5`` or a parquet carrying the
same columns. Layouts come from parquet files joined on ``usi``; a row missing from a
layout gets the missing-value sentinel, which is how one directory can hold a 1M layout
and a 100k one side by side without duplicating a single metadata column.

Two examples. The published Figure 3 point set, straight from the committed parquet:

    build_explorer_data.py \\
        --metadata data/umap_coordinates.parquet \\
        --layout fig100k=umap_x,umap_y \\
        --order-layout fig100k \\
        --out preview/data/pool100k \\
        --label "LCFM test split, Figure 3 point set"

And the 1M pool with both candidate layouts plus Figure 3 masked into it:

    build_explorer_data.py \\
        --metadata embeddings.h5 \\
        --layouts umap_1m_layouts.parquet \\
        --layout native=native_x,native_y,native3_x,native3_y,native3_z \\
        --layout transform=transform_x,transform_y,transform3_x,transform3_y,transform3_z \\
        --layouts published_100k_anchor.parquet \\
        --layout fig100k=umap_x,umap_y \\
        --order-layout native \\
        --out preview/data/pool1m \\
        --label "LCFM test split, held out"
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import derive  # noqa: E402
import format_v2 as fmt  # noqa: E402

FORMAT_VERSION = 2

# Source columns that are copied straight through, per target field.
DIRECT_NUMS = (
    "collision_energy", "spectrum_confidence", "hyperscore", "precursor_mz",
    "precursor_mass", "hydrophobicity", "retention_time", "delta_mass",
    "expectation", "nextscore", "probability", "precursor_intensity",
)
DIRECT_CATS = (
    "search_fragmentation", "frag_type", "search_detector", "search_instrument",
    "acquisition", "search_quant", "modification_class", "modification_types",
    "glyco_class", "search_modifications", "search_enzyme", "search_organism",
    "search_project", "experiment_name", "protein", "sequence",
)
# Raw columns the derivations read, on top of the above.
DERIVATION_INPUTS = ("header", "precursor_charge", "ptm_present")
# Identity columns the point panel shows.
KEY_COLUMNS = ("usi", "sample_idx", "scan")

# Which columns arrive before first paint. Small, and enough to render a coloured map.
BOOT_CATS = ("analyser", "activation", "charge_cat")
CORE_CATS = ("analyser", "activation", "search_fragmentation", "frag_type",
             "search_detector", "search_instrument", "acquisition", "charge_cat")
LAZY_NUMS = ("ms2_low_mz", "precursor_mass", "delta_mass", "expectation",
             "nextscore", "probability", "precursor_intensity")
# High-cardinality fields whose level tables are big enough to defer.
LAZY_CATS = ("sequence", "protein", "experiment_name")
# Only the peptide field gets a search blob and postings; nothing else is searched.
SEARCHABLE = ("sequence",)


def _log(msg: str) -> None:
    print(f"  {msg}", flush=True)  # noqa: T201


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def _decode(values: Any) -> np.ndarray:
    """HDF5 strings come back as bytes; everything downstream wants str."""
    return np.array([v.decode() if isinstance(v, bytes) else v for v in values], dtype=object)


def read_metadata(path: Path, wanted: set[str]) -> tuple[dict[str, np.ndarray], int]:
    """Read only the columns we need, from either an embeddings HDF5 or a parquet."""
    if path.suffix in {".h5", ".hdf5"}:
        import h5py

        cols: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as f:
            n = int(f.attrs["num_embeddings"])
            available = set(f["metadata"])
            for key in sorted(wanted & available):
                cols[key] = _decode(f[f"metadata/{key}"][:])
            missing = wanted - available
        if missing:
            _log(f"absent from the HDF5, will be blank: {sorted(missing)}")
        return cols, n

    import polars as pl

    table = pl.read_parquet(path)
    n = table.height
    present = wanted & set(table.columns)
    if wanted - present:
        _log(f"absent from the parquet, will be blank: {sorted(wanted - present)}")
    return {k: table[k].to_numpy(allow_copy=True).astype(object) for k in sorted(present)}, n


def read_layouts(
    paths: list[Path], specs: dict[str, list[str]], usi: np.ndarray, positional: bool = False
) -> dict[str, dict[str, np.ndarray]]:
    """Align each layout's coordinate columns onto the metadata row order.

    A layout need not cover every row -- Figure 3's 100k inside the 1M pool is the case
    this exists for -- and uncovered rows come back NaN, which quantises to the sentinel.

    ``positional`` is for when the coordinates live in the metadata file itself, where the
    rows already correspond one-to-one. Joining such a file to itself on ``usi`` would be
    lossy: usi is not unique (100,000 rows of Figure 3 carry 99,997 distinct values), so
    first-occurrence matching silently strands the duplicates with no coordinates.
    """
    import polars as pl

    row_of: dict[str, int] = {}
    for i, u in enumerate(usi):
        if u not in row_of:  # first occurrence, matching how the coords tables were built
            row_of[u] = i

    out: dict[str, dict[str, np.ndarray]] = {}
    remaining = dict(specs)
    for path in paths:
        table = pl.read_parquet(path)
        if positional:
            if table.height != len(usi):
                raise SystemExit(
                    f"{path} has {table.height:,} rows but the metadata has {len(usi):,}; "
                    "positional alignment needs them equal"
                )
            rows = np.arange(len(usi))
        else:
            if "usi" not in table.columns:
                raise SystemExit(f"{path} has no usi column to join on")
            rows = np.array([row_of.get(u, -1) for u in table["usi"].to_numpy()])
        keep = rows >= 0
        for name, columns in list(remaining.items()):
            if not set(columns) <= set(table.columns):
                continue
            axes: dict[str, np.ndarray] = {}
            for axis, column in zip(("x", "y", "x3", "y3", "z3"), columns):
                full = np.full(len(usi), np.nan, dtype=np.float64)
                full[rows[keep]] = table[column].to_numpy()[keep]
                axes[axis] = full
            out[name] = axes
            _log(f"layout {name}: {int(keep.sum()):,} of {len(usi):,} rows covered "
                 f"(from {path.name})")
            del remaining[name]
    if remaining:
        detail = "; ".join(f"{k}={','.join(v)}" for k, v in remaining.items())
        raise SystemExit(f"no input provided these layout columns: {detail}")
    return out


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


def build_columns(meta: dict[str, np.ndarray], n: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Assemble the 14 numeric and 23 categorical display columns."""
    blank_str = np.full(n, "", dtype=object)
    blank_num = np.full(n, np.nan, dtype=np.float64)

    def col(name: str) -> np.ndarray:
        return meta.get(name, blank_str)

    nums = {k: derive.to_float(col(k)) if k in meta else blank_num.copy() for k in DIRECT_NUMS}
    cats = {k: np.array([("" if v is None else str(v)) for v in col(k)], dtype=object)
            for k in DIRECT_CATS}

    header = col("header")
    nums["ms2_low_mz"] = np.array([derive.ms2_low_mz(h) for h in header])
    nums["sequence_length"] = np.array([derive.sequence_length(s) for s in col("sequence")])

    cats["analyser"] = np.array([derive.analyser(h) for h in header], dtype=object)
    cats["activation"] = np.array([derive.activation(h) for h in header], dtype=object)
    cats["charge_cat"] = np.array([derive.charge_cat(z) for z in col("precursor_charge")], dtype=object)
    cats["ptm"] = np.array([derive.ptm(v) for v in col("ptm_present")], dtype=object)
    cats["enrichment"] = np.array([derive.enrichment(m) for m in col("search_modifications")], dtype=object)
    cats["label_chem"] = np.array(
        [derive.label_chem(m, q) for m, q in zip(col("search_modifications"), col("search_quant"))],
        dtype=object,
    )
    cats["replicate_peptide"] = derive.replicate_peptide(cats["sequence"])

    assert set(nums) == set(derive.NUM_FIELDS), sorted(set(nums) ^ set(derive.NUM_FIELDS))
    assert set(cats) == set(derive.CAT_FIELDS), sorted(set(cats) ^ set(derive.CAT_FIELDS))
    return nums, cats


def parse_scan(values: np.ndarray) -> np.ndarray:
    """Scan number out of a nativeID like ``controllerType=0 controllerNumber=1 scan=21763``."""
    import re

    pattern = re.compile(r"scan=(\d+)")
    out = np.zeros(len(values), dtype=np.uint32)
    for i, v in enumerate(values):
        if v is None:
            continue
        text = str(v)
        m = pattern.search(text)
        if m:
            out[i] = int(m.group(1))
        else:
            try:
                out[i] = int(float(text))
            except ValueError:
                pass
    return out


def write_payload(
    out: Path,
    nums: dict[str, np.ndarray],
    cats: dict[str, np.ndarray],
    layouts: dict[str, dict[str, np.ndarray]],
    keys: dict[str, np.ndarray],
    order_layout: str,
    label: str,
    provenance: dict[str, Any],
    chunk_rows: int,
    boot_rows: int,
) -> dict[str, Any]:
    """Order the rows, quantise everything and write the shards plus the catalog."""
    n = len(next(iter(cats.values())))
    for sub in ("num", "cat", "layout", "boot"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    anchor = layouts[order_layout]
    order = fmt.morton_order(anchor["x"], anchor["y"])
    _log(f"row order from layout {order_layout!r}: {n:,} rows, "
         f"{(n + chunk_rows - 1) // chunk_rows} draw chunks of {chunk_rows:,}")

    catalog: dict[str, Any] = {
        "format": FORMAT_VERSION,
        "n": n,
        "dataset": label,
        "chunkRows": chunk_rows,
        "bootRows": min(boot_rows, n),
        "orderLayout": order_layout,
        "defaultLayout": order_layout,
        "source": provenance,
        "layouts": {},
        "arrays": {},
        "nums": {},
        "cats": {},
        "numOrder": list(derive.NUM_FIELDS),
        "catOrder": list(derive.CAT_FIELDS),
    }

    # ---- coordinates, per layout ----
    for name, axes in layouts.items():
        entry: dict[str, Any] = {"axes": {}}
        for axis, values in axes.items():
            reordered = values[order]
            codes, lo, hi = fmt.quantise(reordered)
            path = f"layout/{name}.{axis}.u16"
            (out / path).write_bytes(codes.tobytes())
            entry["axes"][axis] = {"path": path, "dtype": "uint16", "lo": lo, "hi": hi,
                                   "bytes": codes.nbytes}
            catalog["arrays"][f"{name}.{axis}"] = {"path": path, "dtype": "uint16"}
        covered = int(np.isfinite(axes["x"]).sum())
        entry["covered"] = covered
        entry["has3d"] = "z3" in axes
        # Two layouts of the same rows are the same space only if one was derived in the
        # other's frame; the viewer keeps the viewport across a switch only when these match.
        entry["spaceId"] = name
        entry["chunkBounds"] = fmt.chunk_bounds(
            axes["x"][order], axes["y"][order], n, chunk_rows
        )
        catalog["layouts"][name] = entry
        _log(f"layout {name}: {covered:,} rows placed, 3-D {'yes' if entry['has3d'] else 'no'}")

    # ---- identity arrays ----
    for name, values in keys.items():
        reordered = values[order].astype(np.uint32)
        path = f"{name}.u32"
        (out / path).write_bytes(reordered.tobytes())
        catalog["arrays"][name] = {"path": path, "dtype": "uint32", "tier": "detail"}

    # ---- numerics ----
    for key in derive.NUM_FIELDS:
        codes, lo, hi = fmt.quantise(nums[key][order])
        path = f"num/{key}.u16"
        (out / path).write_bytes(codes.tobytes())
        catalog["nums"][key] = {
            **derive.NUM_FIELDS[key],
            "path": path, "dtype": "uint16", "lo": lo, "hi": hi,
            "tier": "lazy" if key in LAZY_NUMS else "detail",
            "bytes": codes.nbytes,
        }

    # ---- categoricals ----
    for key in derive.CAT_FIELDS:
        codes, levels, counts = fmt.encode_categorical(cats[key][order])
        dtype = "uint8" if codes.dtype == np.uint8 else "uint16"
        path = f"cat/{key}.{'u8' if dtype == 'uint8' else 'u16'}"
        (out / path).write_bytes(codes.tobytes())
        entry = {
            **derive.CAT_FIELDS[key],
            "path": path, "dtype": dtype, "nlevels": len(levels),
            "tier": "core" if key in CORE_CATS else ("lazy" if key in LAZY_CATS else "detail"),
            "bytes": codes.nbytes,
        }
        entry.update(fmt.write_levels(out, key, levels, counts))
        if key in SEARCHABLE:
            entry.update(fmt.write_search_blob(out, key, levels))
            entry.update(fmt.write_postings(out, key, codes, len(levels)))
        catalog["cats"][key] = entry

    # ---- boot shards ----
    # The first `bootRows` rows of the columns needed for a first paint, duplicated as
    # their own files. A byte range over the full column would be the obvious alternative,
    # but GitHub Pages applies the range to the gzip stream once a browser negotiates
    # compression, which yields undecodable bytes; ~1 MB of duplication avoids the problem
    # rather than working around it.
    boot = catalog["bootRows"]
    boot_files: dict[str, str] = {}
    if boot >= n:
        # Nothing to gain: the shard would be a byte-for-byte copy of the full column, so
        # the viewer should just fetch the column. Only worth duplicating for a real prefix.
        catalog["bootRows"] = n
        catalog["boot"] = boot_files
        _log(f"no boot shards: {n:,} rows is within the {boot:,}-row budget already")
        return catalog
    for axis in ("x", "y"):
        src = out / catalog["layouts"][order_layout]["axes"][axis]["path"]
        data = np.frombuffer(src.read_bytes(), dtype=np.uint16)[:boot]
        path = f"boot/{order_layout}.{axis}.u16"
        (out / path).write_bytes(data.tobytes())
        boot_files[f"{order_layout}.{axis}"] = path
    for key in BOOT_CATS:
        meta_entry = catalog["cats"][key]
        width = 1 if meta_entry["dtype"] == "uint8" else 2
        data = (out / meta_entry["path"]).read_bytes()[: boot * width]
        path = f"boot/{key}.{'u8' if width == 1 else 'u16'}"
        (out / path).write_bytes(data)
        boot_files[key] = path
    catalog["boot"] = boot_files

    return catalog


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _layout_spec(text: str) -> tuple[str, list[str]]:
    name, _, columns = text.partition("=")
    parts = [c.strip() for c in columns.split(",") if c.strip()]
    if not name or len(parts) not in (2, 5):
        raise argparse.ArgumentTypeError(
            f"expected NAME=xcol,ycol or NAME=xcol,ycol,x3col,y3col,z3col, got {text!r}"
        )
    return name, parts


def main(argv: list[str] | None = None) -> int:
    """Build the payload and report what was written."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metadata", type=Path, required=True, help="embeddings.h5 or a parquet")
    ap.add_argument("--layouts", type=Path, action="append", default=[],
                    help="parquet with usi + coordinate columns; repeatable")
    ap.add_argument("--layout", type=_layout_spec, action="append", required=True,
                    help="NAME=xcol,ycol[,x3col,y3col,z3col]; repeatable")
    ap.add_argument("--order-layout", help="layout whose geometry fixes the row order")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--label", default="", help="dataset description shown in the UI")
    ap.add_argument("--chunk-rows", type=int, default=fmt.CHUNK_ROWS)
    ap.add_argument("--boot-rows", type=int, default=fmt.BOOT_ROWS)
    args = ap.parse_args(argv)

    specs = dict(args.layout)
    order_layout = args.order_layout or next(iter(specs))
    if order_layout not in specs:
        ap.error(f"--order-layout {order_layout!r} is not one of {sorted(specs)}")

    t0 = time.time()
    wanted = set(DIRECT_NUMS) | set(DIRECT_CATS) | set(DERIVATION_INPUTS) | set(KEY_COLUMNS)
    # A metadata parquet may carry the coordinates itself, in which case it is also a layout.
    if not args.layouts:
        wanted |= {c for cols in specs.values() for c in cols}

    print(f"reading {args.metadata}", flush=True)  # noqa: T201
    meta, n = read_metadata(args.metadata, wanted)
    _log(f"{n:,} rows, {len(meta)} source columns")

    usi = meta.get("usi")
    if usi is None:
        raise SystemExit("metadata has no usi column, so layouts cannot be joined")

    print("aligning layouts", flush=True)  # noqa: T201
    layouts = read_layouts(args.layouts or [args.metadata], specs, usi,
                           positional=not args.layouts)

    print("deriving display columns", flush=True)  # noqa: T201
    nums, cats = build_columns(meta, n)
    _log(f"{len(nums)} numeric, {len(cats)} categorical")

    keys = {
        "sample_idx": np.array([int(float(v)) if v not in (None, "") else 0
                                for v in meta.get("sample_idx", np.zeros(n))], dtype=np.int64),
        "scan": parse_scan(meta.get("scan", np.zeros(n))),
    }

    provenance = {
        "metadata": args.metadata.name,
        "layouts": [p.name for p in args.layouts],
        "layoutColumns": dict(specs),
        "builtBy": "scripts/explorer/build_explorer_data.py",
    }

    print(f"writing {args.out}", flush=True)  # noqa: T201
    catalog = write_payload(args.out, nums, cats, layouts, keys, order_layout,
                            args.label, provenance, args.chunk_rows, args.boot_rows)

    problems = fmt.validate(catalog, args.out)
    (args.out / "catalog.json").write_text(json.dumps(catalog, separators=(",", ":")))
    size = (args.out / "catalog.json").stat().st_size
    total = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(  # noqa: T201
        f"done in {time.time() - t0:.0f}s: catalog {size:,} B, payload {total / 2**20:.1f} MiB",
        flush=True,
    )
    if problems:
        print("VALIDATION FAILED:", flush=True)  # noqa: T201
        for p in problems:
            print(f"  - {p}", flush=True)  # noqa: T201
        return 1
    print("validation passed", flush=True)  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
