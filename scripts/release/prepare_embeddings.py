#!/usr/bin/env python
"""Turn an eval-harness ``embeddings.h5`` into parquet shards for a HuggingFace dataset.

The harness writes HDF5, which is a poor publication format: the HuggingFace viewer
cannot read it, ``datasets`` cannot stream it, and its gzip-9 object columns take minutes
to decode. Parquet fixes all three, and a fixed-size list column carries the 768-d vector
without the per-row overhead a variable list would add.

What is published and what is not
---------------------------------
The 54 scalar metadata fields go up alongside the embeddings, because an embedding with
no spectrum identity attached cannot be joined to anything. The four per-peak arrays
(``peak_mask``, ``precursors``, ``spectra_mask``, ``targets``) do not: they are ~1.9 GiB
of model plumbing, reconstructible from the corpus, and nobody asking for embeddings
wants them.

Selecting a subset
------------------
``--rows-from`` takes a parquet naming the rows to keep, via a column of row indices into
the HDF5. For the published Figure 3 point set that column is ``sample_idx``, which the
harness assigns as ``arange(batch) + batch_idx * batch_size`` -- a position within *that*
extraction, so it is only meaningful against the HDF5 it came from. It resolves all
100,000 Figure 3 rows exactly, which ``usi`` cannot: usi is not unique in this corpus
(timsTOF spectra whose USI drops the frame collide), so a usi join silently strands them.

Streaming
---------
A million rows of 768 float32 is 2.86 GiB, so shards are read and written one at a time
rather than assembling the whole table in memory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EMBEDDING_DIM_ATTR = "embedding_dim"
# Named exclusions, on top of the structural test below.
EXCLUDE_FIELDS = frozenset({"peak_mask", "precursors", "spectra_mask", "targets"})
DEFAULT_SHARD_ROWS = 125_000  # ~450 MB per shard at 768 float32 plus metadata
# --coords-join value meaning "the HDF5 row position", not a metadata column.
ROW_INDEX_JOIN = "row_index"


def _log(msg: str) -> None:
    print(msg, flush=True)  # noqa: T201


def _decode(values: np.ndarray) -> list[Any]:
    """HDF5 strings arrive as bytes; parquet wants str."""
    return [v.decode("utf-8", "replace") if isinstance(v, bytes) else v for v in values]


def _holds_one_value_per_row(dataset: Any) -> bool:
    """Whether a 1-D column holds one value per row, or a whole spectrum per row.

    Neither the shape nor the dtype distinguishes them. The per-peak arrays are 1-D
    object columns of *strings*, each holding a stringified numpy array --
    ``b'[ 101.07096  101.107475 ...]'``, a couple of kilobytes a row. So a shape test
    admits them and so does a bytes test: between them mz_array and intensity_array
    added 305 MiB of stringified peaks to a 293 MiB embedding table. The array repr is
    the giveaway.

    The data is not truncated, so nothing is lost by dropping it here -- and it is the
    spectra, which the corpus dataset already publishes properly.
    """
    if dataset.shape[0] == 0:
        return True
    probe = dataset[0]
    if isinstance(probe, (np.ndarray, list, tuple)):
        return False
    if isinstance(probe, bytes):
        return not probe.lstrip().startswith(b"[")
    if isinstance(probe, str):
        return not probe.lstrip().startswith("[")
    return True


def scalar_fields(meta: Any) -> list[str]:
    """The metadata columns worth publishing: one value per row, and not named out."""
    return sorted(
        k for k in meta
        if len(meta[k].shape) == 1 and k not in EXCLUDE_FIELDS and _holds_one_value_per_row(meta[k])
    )


def resolve_rows(h5_rows: int, rows_from: Path | None, rows_column: str) -> np.ndarray:
    """Which HDF5 rows to publish, in ascending order."""
    if rows_from is None:
        return np.arange(h5_rows, dtype=np.int64)

    import polars as pl

    table = pl.read_parquet(rows_from, columns=[rows_column])
    rows = table[rows_column].to_numpy().astype(np.int64)
    if len(np.unique(rows)) != len(rows):
        raise SystemExit(
            f"{rows_from}:{rows_column} has {len(rows) - len(np.unique(rows))} repeated "
            "values, so it does not identify rows uniquely"
        )
    if rows.max() >= h5_rows or rows.min() < 0:
        raise SystemExit(
            f"{rows_from}:{rows_column} spans {rows.min():,}..{rows.max():,} but the "
            f"HDF5 has {h5_rows:,} rows -- this selection belongs to a different extraction"
        )
    return np.sort(rows)


def load_coords(
    paths: list[Path], join_column: str, wanted: dict[str, str]
) -> tuple[dict[int, int], dict[str, np.ndarray]]:
    """Named coordinate columns, keyed by the same row index the selection uses.

    Only the columns asked for. Taking every numeric column instead re-published
    fields the metadata already carries plus a dozen figure-specific metrics, which
    turned a 293 MiB embedding table into a 598 MiB one.
    """
    import polars as pl

    index: dict[int, int] = {}
    columns: dict[str, np.ndarray] = {}
    remaining = dict(wanted)
    for path in paths:
        table = pl.read_parquet(path)
        if join_column not in table.columns:
            raise SystemExit(f"{path} has no {join_column} column to join coordinates on")
        keys = table[join_column].to_numpy().astype(np.int64)
        for local, key in enumerate(keys):
            index.setdefault(int(key), local)
        for source, published in list(remaining.items()):
            if source not in table.columns:
                continue
            columns[published] = table[source].to_numpy().astype(np.float32)
            del remaining[source]
    if remaining:
        raise SystemExit(f"no --coords input provided these columns: {sorted(remaining)}")
    return index, columns


def _coords_spec(text: str) -> dict[str, str]:
    """``source[:published],...`` so a build-time name can be published under a clearer one."""
    out: dict[str, str] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        source, _, published = part.partition(":")
        out[source] = published or source
    return out


def main(argv: list[str] | None = None) -> int:  # noqa: PLR0915 -- one linear pipeline
    """Write parquet shards plus a JSON summary of what went into them."""
    import h5py

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", type=Path, required=True, help="the harness's embeddings.h5")
    ap.add_argument("--out", type=Path, required=True, help="output directory for the shards")
    ap.add_argument("--config", required=True, help="dataset config name, e.g. 100k or 1M")
    ap.add_argument("--rows-from", type=Path,
                    help="parquet naming the rows to publish; omit to publish all of them")
    ap.add_argument("--rows-column", default="sample_idx",
                    help="column in --rows-from holding row indices into the HDF5")
    ap.add_argument("--coords", type=Path, action="append", default=[],
                    help="parquet with UMAP coordinates to ship alongside; repeatable")
    ap.add_argument("--coords-join", default="sample_idx",
                    help="column joining --coords to the HDF5 rows")
    ap.add_argument("--coords-columns", type=_coords_spec, default={},
                    help="source[:published],... coordinate columns to ship; required with --coords")
    ap.add_argument("--shard-rows", type=int, default=DEFAULT_SHARD_ROWS)
    args = ap.parse_args(argv)

    out = args.out / args.config
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.embeddings, "r") as f:
        n_h5 = int(f.attrs["num_embeddings"])
        dim = int(f.attrs[EMBEDDING_DIM_ATTR])
        pooling = str(f.attrs.get("embedding_pooling", "unknown"))
        fields = scalar_fields(f["metadata"])
        skipped = sorted(k for k in f["metadata"] if k not in fields)

        rows = resolve_rows(n_h5, args.rows_from, args.rows_column)
        _log(f"{args.config}: publishing {len(rows):,} of {n_h5:,} rows, {dim}-d {pooling}")
        _log(f"  metadata fields: {len(fields)} published, {len(skipped)} skipped ({', '.join(skipped)})")

        coord_index: dict[int, int] = {}
        coord_columns: dict[str, np.ndarray] = {}
        if args.coords:
            if not args.coords_columns:
                raise SystemExit("--coords needs --coords-columns; publishing every numeric "
                                 "column duplicates the metadata and doubles the download")
            coord_index, coord_columns = load_coords(
                args.coords, args.coords_join, args.coords_columns)
            _log(f"  coordinates: {', '.join(sorted(coord_columns))} "
                 f"for {len(coord_index):,} rows")

        # The join key for coordinates. "row_index" means the HDF5 row position itself,
        # which is what a coords table computed over the whole extraction carries; anything
        # else names a metadata column, read here even when it is not published.
        positional_join = args.coords_join == ROW_INDEX_JOIN
        key_values = None
        if coord_index and not positional_join:
            if args.coords_join not in f["metadata"]:
                raise SystemExit(
                    f"--coords-join {args.coords_join!r} is not a metadata column of "
                    f"{args.embeddings.name}; use {ROW_INDEX_JOIN!r} for a positional join"
                )
            key_values = np.asarray(f[f"metadata/{args.coords_join}"][:])

        summary: dict[str, Any] = {
            "config": args.config, "rows": int(len(rows)), "embedding_dim": dim,
            "embedding_pooling": pooling, "source_rows": n_h5,
            "metadata_fields": fields, "skipped_fields": skipped,
            "coordinate_columns": sorted(coord_columns), "shards": [],
        }

        schema = pa.schema(
            [("embedding", pa.list_(pa.float32(), dim))]
            + [(name, pa.string() if f["metadata"][name].dtype == object else
                pa.from_numpy_dtype(f["metadata"][name].dtype)) for name in fields]
            + [(name, pa.float32()) for name in sorted(coord_columns)]
        )

        for shard, start in enumerate(range(0, len(rows), args.shard_rows)):
            block = rows[start : start + args.shard_rows]
            arrays: dict[str, Any] = {}

            flat = f["embeddings"][block].astype(np.float32, copy=False).reshape(-1)
            arrays["embedding"] = pa.FixedSizeListArray.from_arrays(pa.array(flat), dim)
            for name in fields:
                raw = f[f"metadata/{name}"][block]
                arrays[name] = pa.array(_decode(raw)) if raw.dtype == object else pa.array(raw)

            if coord_index:
                keys = block.astype(np.int64) if positional_join \
                    else np.asarray(key_values)[block].astype(np.int64)
                local = np.array([coord_index.get(int(k), -1) for k in keys])
                for name in sorted(coord_columns):
                    src = coord_columns[name]
                    vals = np.full(len(block), np.nan, dtype=np.float32)
                    hit = local >= 0
                    vals[hit] = src[local[hit]]
                    arrays[name] = pa.array(vals)

            table = pa.table({k: arrays[k] for k in schema.names}, schema=schema)
            path = out / f"{args.config}-{shard:05d}.parquet"
            pq.write_table(table, path, compression="zstd", compression_level=6)
            size = path.stat().st_size
            summary["shards"].append({"file": path.name, "rows": len(block), "bytes": size})
            _log(f"  wrote {path.name}: {len(block):,} rows, {size / 2**20:.0f} MiB")

    total = sum(s["bytes"] for s in summary["shards"])
    summary["total_bytes"] = total
    (args.out / f"{args.config}.summary.json").write_text(json.dumps(summary, indent=2))
    _log(f"{args.config}: {len(summary['shards'])} shards, {total / 2**20:.0f} MiB total")
    return 0


if __name__ == "__main__":
    sys.exit(main())
