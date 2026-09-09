#!/usr/bin/env python3
# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Create an offline sequence-rescue dataset.

This is the simpler duplicate-spectrum rescue setup:

* choose query peptide sequences with enough spectra;
* sample query spectra from each sequence;
* sample positive library spectra from other rows with the same sequence;
* sample negative library spectra from sequences outside the selected query set.

The exact query rows are never included in the library.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from instanovo_fm.eval.embed_eval_tasks.spectral_rescue_reformulated import (
    SpectralRescueTaskReformulated,
)
from instanovo.utils.s3 import S3FileHandler


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def _project_from_usi_expr(usi_col: str = "usi") -> pl.Expr:
    return pl.col(usi_col).str.split(":").list.get(1)


def _canonical_sequence_expr(sequence_col: str = "sequence") -> pl.Expr:
    # Keep this expression simple and do full canonicalization in Python for selected rows.
    return pl.col(sequence_col).cast(pl.Utf8)


def _sorted_shards(source_glob: str, max_shards: int | None = None) -> list[Path]:
    paths = [Path(path) for path in sorted(glob.glob(source_glob))]
    if max_shards is not None:
        paths = paths[:max_shards]
    if not paths:
        raise FileNotFoundError(f"No parquet files matched source_glob={source_glob!r}")
    return paths


def _scan_metadata(shard_paths: list[Path], projects: set[str] | None) -> pl.DataFrame:
    scans = [
        pl.scan_parquet(str(path))
        .select(["usi", "sequence", "unmodified_peptide"])
        .with_row_index("rescue_row_in_shard")
        .with_columns(
            rescue_source_shard=pl.lit(str(path)),
            rescue_project_id=_project_from_usi_expr(),
            rescue_sequence_key=_canonical_sequence_expr(),
        )
        for path in shard_paths
    ]
    scan = pl.concat(scans, how="vertical_relaxed").drop_nulls(
        ["usi", "sequence", "unmodified_peptide", "rescue_project_id", "rescue_sequence_key"]
    )
    if projects:
        scan = scan.filter(pl.col("rescue_project_id").is_in(sorted(projects)))
    df = scan.collect()
    canonical = [
        SpectralRescueTaskReformulated._canonicalize_modified_sequence(str(sequence))
        for sequence in df["rescue_sequence_key"].to_list()
    ]
    return df.with_columns(rescue_sequence_key=pl.Series(canonical))


def _sample_records(
    records: list[dict[str, Any]],
    n: int,
    rng: np.random.RandomState,
    *,
    label: str,
    allow_fewer: bool = False,
) -> list[dict[str, Any]]:
    if len(records) < n:
        if allow_fewer and records:
            n = len(records)
        else:
            raise ValueError(f"Need {n} {label} records, found {len(records)}")
    chosen = rng.choice(len(records), size=n, replace=False)
    return [records[int(index)] for index in sorted(chosen.tolist())]


def _dedupe_by_sequence(records: list[dict[str, Any]], rng: np.random.RandomState) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["sequence_key"]), []).append(record)
    deduped = []
    for rows in grouped.values():
        deduped.append(rows[int(rng.choice(len(rows)))])
    return deduped


def _select_sequence_specs(df: pl.DataFrame, args: argparse.Namespace) -> list[dict[str, Any]]:
    counts = (
        df.group_by(["rescue_project_id", "rescue_sequence_key"])
        .agg(spectra=pl.len())
        .filter(pl.col("spectra") >= args.num_query_spectra + args.num_positive_library)
        .sort(["spectra", "rescue_project_id", "rescue_sequence_key"], descending=[True, False, False])
    )

    specs = []
    seen_projects: set[str] = set()
    for row in counts.iter_rows(named=True):
        project_id = str(row["rescue_project_id"])
        sequence = SpectralRescueTaskReformulated._canonicalize_modified_sequence(str(row["rescue_sequence_key"]))
        if args.one_sequence_per_project and project_id in seen_projects:
            continue
        pair_id = f"{project_id}_{sequence[-8:]}".lower()
        pair_id = "".join(char if char.isalnum() or char == "_" else "_" for char in pair_id)
        specs.append(
            {
                "pair_id": pair_id,
                "project_id": project_id,
                "sequence": sequence,
                "available_spectra": int(row["spectra"]),
            }
        )
        seen_projects.add(project_id)
        if len(specs) >= args.max_sequences:
            break
    if not specs:
        raise ValueError("No sequence candidates satisfied the requested query/positive counts")
    return specs


def _records_from_df(df: pl.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in df.iter_rows(named=True):
        sequence_key = SpectralRescueTaskReformulated._canonicalize_modified_sequence(str(row["sequence"]))
        records.append(
            {
                "source_shard": str(row["rescue_source_shard"]),
                "row_in_shard": int(row["rescue_row_in_shard"]),
                "usi": str(row["usi"]),
                "sequence_key": sequence_key,
            }
        )
    return records


def _collect_rows(selected: list[dict[str, Any]], *, seed: int, scope: str, sampling_config: dict[str, Any]) -> pl.DataFrame:
    by_shard: dict[str, list[dict[str, Any]]] = {}
    for record in selected:
        by_shard.setdefault(str(record["source_shard"]), []).append(record)

    frames = []
    for shard, records in by_shard.items():
        wanted = sorted({int(record["row_in_shard"]) for record in records})
        frame = (
            pl.scan_parquet(shard)
            .with_row_index("rescue_row_in_shard")
            .filter(pl.col("rescue_row_in_shard").is_in(wanted))
            .collect()
        )
        role_frame = pl.DataFrame(
            {
                "rescue_row_in_shard": [int(record["row_in_shard"]) for record in records],
                "rescue_role": [record["rescue_role"] for record in records],
                "rescue_pair_id": [record["pair_id"] for record in records],
                "rescue_project_id": [record["project_id"] for record in records],
                "rescue_base_sequence": [record["sequence"] for record in records],
                "rescue_modified_sequence": [record["sequence"] for record in records],
            }
        )
        frame = frame.with_columns(
            rescue_source_shard=pl.lit(shard),
            rescue_scope=pl.lit(scope),
            rescue_selection_seed=pl.lit(seed),
            rescue_sampling_profile=pl.lit("sequence_rescue"),
        ).join(role_frame, on="rescue_row_in_shard", how="left")
        frames.append(frame)

    out = pl.concat(frames, how="vertical_relaxed")
    role_order = {"reference_query": 0, "positive_library": 1, "negative_library": 2}
    return out.with_columns(rescue_role_order=pl.col("rescue_role").replace(role_order).cast(pl.Int64)).sort(
        ["rescue_pair_id", "rescue_role_order", "rescue_source_shard", "rescue_row_in_shard"]
    ).drop("rescue_role_order").with_columns(
        **{f"rescue_sampling_{key}": pl.lit(value) for key, value in sampling_config.items()}
    )


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.RandomState(args.seed)
    shard_paths = _sorted_shards(args.source_glob, args.max_shards)
    projects = {part.strip() for part in args.projects.split(",") if part.strip()} if args.projects else None
    _log(f"loading metadata from {len(shard_paths)} shard(s); projects={sorted(projects) if projects else 'all'}")
    df = _scan_metadata(shard_paths, projects)
    _log(f"loaded metadata rows={df.height}")

    specs = json.loads(Path(args.sequence_manifest).read_text())["queries"] if args.sequence_manifest else _select_sequence_specs(df, args)
    selected_sequences = {str(spec["sequence"]) for spec in specs}
    _log(f"selected {len(specs)} query sequence(s): {', '.join(spec['pair_id'] for spec in specs)}")

    all_selected: list[dict[str, Any]] = []
    summaries = []
    for spec in specs:
        project_id = str(spec["project_id"])
        sequence = SpectralRescueTaskReformulated._canonicalize_modified_sequence(str(spec["sequence"]))
        pair_id = str(spec["pair_id"])
        project_df = df.filter(pl.col("rescue_project_id") == project_id)
        seq_df = project_df.filter(pl.col("rescue_sequence_key") == sequence)
        seq_records = _records_from_df(seq_df)
        query_records = _sample_records(seq_records, args.num_query_spectra, rng, label=f"{pair_id} query")
        query_keys = {(row["source_shard"], row["row_in_shard"]) for row in query_records}
        positive_pool = [row for row in seq_records if (row["source_shard"], row["row_in_shard"]) not in query_keys]
        positive_records = _sample_records(positive_pool, args.num_positive_library, rng, label=f"{pair_id} positive library")

        negative_df = project_df.filter(~pl.col("rescue_sequence_key").is_in(sorted(selected_sequences)))
        negative_pool = _dedupe_by_sequence(_records_from_df(negative_df), rng) if args.one_negative_per_sequence else _records_from_df(negative_df)
        negative_records = _sample_records(
            negative_pool,
            args.num_negative_library,
            rng,
            label=f"{pair_id} negative library",
            allow_fewer=args.allow_fewer_negatives,
        )

        for role, records in (
            ("reference_query", query_records),
            ("positive_library", positive_records),
            ("negative_library", negative_records),
        ):
            for record in records:
                record.update({"rescue_role": role, "pair_id": pair_id, "project_id": project_id, "sequence": sequence})
                all_selected.append(record)

        summaries.append(
            {
                "pair_id": pair_id,
                "project_id": project_id,
                "sequence": sequence,
                "available_sequence_spectra": len(seq_records),
                "num_reference_queries": len(query_records),
                "num_positive_library": len(positive_records),
                "num_negative_library": len(negative_records),
                "negative_pool_sequences": len(negative_pool),
            }
        )
        _log(
            f"{pair_id}: query={len(query_records)}, positive={len(positive_records)}, "
            f"negative={len(negative_records)} from negative_pool_sequences={len(negative_pool)}"
        )

    sampling_config = {
        "num_query_spectra": args.num_query_spectra,
        "num_positive_library": args.num_positive_library,
        "num_negative_library": args.num_negative_library,
        "allow_fewer_negatives": args.allow_fewer_negatives,
        "one_negative_per_sequence": args.one_negative_per_sequence,
    }
    out_df = _collect_rows(all_selected, seed=args.seed, scope="sequence_rescue", sampling_config=sampling_config)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out_df.write_parquet(output)
    pairs_path = output.with_suffix(".rescue_pairs.json")
    pairs_manifest = {
        "scope": "multi_base",
        "pairs": [
            {
                "pair_id": row["pair_id"],
                "project_id": row["project_id"],
                "base_sequence": row["sequence"],
                "modified_sequence": row["sequence"],
                "notes": "sequence_rescue: modified_sequence intentionally equals base_sequence",
            }
            for row in summaries
        ],
    }
    pairs_path.write_text(json.dumps(pairs_manifest, indent=2))
    summary = {
        "output": str(output),
        "rescue_pairs_path": str(pairs_path),
        "source_glob": args.source_glob,
        "seed": args.seed,
        "num_rows": out_df.height,
        "queries": specs,
        "per_query": summaries,
        "sampling_config": sampling_config,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    _log(f"wrote dataset rows={out_df.height} to {output}")
    _log(f"wrote rescue pairs manifest to {pairs_path}")
    _log(f"wrote summary to {summary_path}")

    if args.s3_output_prefix:
        s3 = S3FileHandler(verbose=True)
        if s3.s3 is None:
            raise RuntimeError("S3 upload requested but S3 credentials/environment are not configured")
        prefix = args.s3_output_prefix.rstrip("/")
        s3.upload(str(output), f"{prefix}/{output.name}")
        s3.upload(str(pairs_path), f"{prefix}/{pairs_path.name}")
        s3.upload(str(summary_path), f"{prefix}/{summary_path.name}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_glob", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--projects", default=None)
    parser.add_argument("--sequence_manifest", default=None)
    parser.add_argument("--max_sequences", type=int, default=5)
    parser.add_argument("--num_query_spectra", type=int, default=50)
    parser.add_argument("--num_positive_library", type=int, default=50)
    parser.add_argument("--num_negative_library", type=int, default=200)
    parser.add_argument("--allow_fewer_negatives", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--one_negative_per_sequence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--one_sequence_per_project", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_shards", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--s3_output_prefix", default=None)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(build_dataset(parse_args()), indent=2))


if __name__ == "__main__":
    main()
