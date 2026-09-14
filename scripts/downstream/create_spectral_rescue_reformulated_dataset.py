#!/usr/bin/env python3
# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Create a small offline dataset for the reformulated spectral rescue task.

The output parquet is intended to be embedded by the normal evaluation pipeline.
Each selected spectrum receives a ``rescue_role`` and, for panels, a
``rescue_pair_id``.

Scopes (see ``spectral_rescue_scope.py`` for full definitions):

* ``single_base`` — one base/modified pair; controlled case study / figures.
* ``multi_base`` — many pairs from a JSON manifest; per-pair libraries, panel metrics.

Sampling profiles:

* ``demo`` — small happy-path case study.
* ``rigorous`` — larger sample, raw-file dedupe, stratified negatives, multi-seed.
  Counts are **per pair** in ``multi_base`` mode.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import polars as pl

from instanovo_fm.eval.embed_eval_tasks.spectral_rescue_reformulated import (
    SpectralRescueTaskReformulated,
)
from instanovo_fm.eval.embed_eval_tasks.spectral_rescue_scope import (
    RescuePairSpec,
    resolve_rescue_scope_config,
    scope_guidance_text,
)
from instanovo.utils.s3 import S3FileHandler


def _log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}] {message}", flush=True)


def _sorted_shards(source_glob: str) -> List[Path]:
    paths = [Path(path) for path in sorted(glob.glob(source_glob))]
    if not paths:
        raise FileNotFoundError(f"No parquet files matched source_glob={source_glob!r}")
    return paths


def _project_from_usi_expr(usi_col: str = "usi") -> pl.Expr:
    return pl.col(usi_col).str.split(":").list.get(1)


def _metadata_candidates(
    shard_paths: Iterable[Path],
    *,
    project_id: str,
    base_sequence: str,
    modified_sequence: str,
    min_negative_clean_edit_distance: int,
    log_prefix: str = "",
) -> Dict[str, List[Dict[str, Any]]]:
    base_canon = SpectralRescueTaskReformulated._canonicalize_modified_sequence(base_sequence)
    modified_canon = SpectralRescueTaskReformulated._canonicalize_modified_sequence(modified_sequence)
    base_backbone = SpectralRescueTaskReformulated._backbone_sequence(base_sequence)

    candidates: Dict[str, List[Dict[str, Any]]] = {
        "base": [],
        "modified": [],
        "negative": [],
        "same_backbone_non_base": [],
    }

    shard_list = list(shard_paths)
    for shard_index, shard in enumerate(shard_list, start=1):
        df = (
            pl.scan_parquet(str(shard))
            .select(["usi", "sequence", "unmodified_peptide"])
            .with_row_index("rescue_row_in_shard")
            .with_columns(rescue_project_id=_project_from_usi_expr())
            .filter(pl.col("rescue_project_id") == project_id)
            .collect()
        )

        for row in df.iter_rows(named=True):
            sequence = row["sequence"]
            unmodified = row["unmodified_peptide"]
            modified = SpectralRescueTaskReformulated._canonicalize_modified_sequence(sequence)
            backbone = SpectralRescueTaskReformulated._backbone_sequence(unmodified)
            record = {
                "source_shard": str(shard),
                "row_in_shard": int(row["rescue_row_in_shard"]),
                "usi": row["usi"],
                "sequence": sequence,
                "unmodified_peptide": unmodified,
                "modified_canon": modified,
                "backbone": backbone,
            }

            if modified == base_canon:
                candidates["base"].append(record)
            elif modified == modified_canon:
                candidates["modified"].append(record)
            elif backbone == base_backbone:
                candidates["same_backbone_non_base"].append(record)
            else:
                distance = SpectralRescueTaskReformulated._clean_site_edit_distance(base_canon, sequence)
                if distance >= min_negative_clean_edit_distance:
                    record["clean_site_edit_distance"] = int(distance)
                    candidates["negative"].append(record)

        if shard_index == 1 or shard_index % 25 == 0 or shard_index == len(shard_list):
            _log(
                f"{log_prefix}scanned {shard_index}/{len(shard_list)} shard(s); "
                f"base={len(candidates['base'])}, modified={len(candidates['modified'])}, "
                f"negative={len(candidates['negative'])}, same_backbone_other={len(candidates['same_backbone_non_base'])}"
            )

    return candidates


def _parse_seeds(raw_seeds: Optional[str], fallback_seed: int) -> List[int]:
    if raw_seeds is None:
        return [fallback_seed]
    seeds = [int(part.strip()) for part in raw_seeds.split(",") if part.strip()]
    if not seeds:
        raise ValueError("Expected at least one seed in --seeds")
    return seeds


def _apply_profile(args: argparse.Namespace) -> None:
    if args.profile is None:
        return
    profile = SpectralRescueTaskReformulated.SAMPLING_PROFILES.get(args.profile)
    if profile is None:
        raise ValueError(f"Unknown profile {args.profile!r}")

    for key, value in profile.items():
        if key == "seeds":
            if args.seeds is None:
                args.seeds = ",".join(str(seed) for seed in value)
            continue
        setattr(args, key, value)


def _resolve_output_path(base_output: Path, seed: int, *, multi_seed: bool) -> Path:
    if "{seed}" in base_output.as_posix():
        return Path(str(base_output).format(seed=seed))
    if multi_seed:
        return base_output.with_name(f"{base_output.stem}_seed{seed}{base_output.suffix}")
    return base_output


def _select_records(
    candidates: Dict[str, List[Dict[str, Any]]],
    args: argparse.Namespace,
    rng: np.random.RandomState,
) -> Dict[str, List[Dict[str, Any]]]:
    base_pool = SpectralRescueTaskReformulated.dedupe_by_raw_file(
        candidates["base"],
        args.max_queries_per_raw_file,
        rng,
    )
    modified_pool = SpectralRescueTaskReformulated.dedupe_by_raw_file(
        candidates["modified"],
        args.max_queries_per_raw_file,
        rng,
    )
    negative_pool = SpectralRescueTaskReformulated.dedupe_by_raw_file(
        candidates["negative"],
        args.max_queries_per_raw_file,
        rng,
    )

    reference_queries = SpectralRescueTaskReformulated.sample_records(
        base_pool,
        args.num_reference_queries,
        rng,
        "reference query",
    )
    reference_keys = {(record["source_shard"], record["row_in_shard"]) for record in reference_queries}
    remaining_base = [
        record
        for record in base_pool
        if (record["source_shard"], record["row_in_shard"]) not in reference_keys
    ]
    positive_library = SpectralRescueTaskReformulated.sample_records(
        remaining_base,
        args.num_positive_library,
        rng,
        "positive library",
    )
    modified_queries = SpectralRescueTaskReformulated.sample_records(
        modified_pool,
        args.num_modified_queries,
        rng,
        "modified query",
    )
    negative_library = SpectralRescueTaskReformulated.sample_records(
        negative_pool,
        args.num_negative_library,
        rng,
        "negative library",
        strategy=args.negative_sampling,
    )

    return {
        "reference_query": reference_queries,
        "modified_query": modified_queries,
        "positive_library": positive_library,
        "negative_library": negative_library,
    }


def _s3_output_paths(output_path: Path, summary_path: Path, s3_output_prefix: str) -> Dict[str, str]:
    if not s3_output_prefix.startswith("s3://"):
        raise ValueError(f"--s3_output_prefix must start with s3://, got {s3_output_prefix!r}")

    prefix = s3_output_prefix.rstrip("/")
    return {
        "offline_dataset_parquet": f"{prefix}/{output_path.name}",
        "summary_json": f"{prefix}/{summary_path.name}",
    }


def _upload_outputs_to_s3(output_path: Path, summary_path: Path, s3_uploads: Dict[str, str]) -> None:
    s3 = S3FileHandler(verbose=True)
    if s3.s3 is None:
        raise RuntimeError("S3 upload requested but S3 credentials/environment are not configured")

    s3.upload(str(output_path), s3_uploads["offline_dataset_parquet"])
    s3.upload(str(summary_path), s3_uploads["summary_json"])


def _collect_selected_rows(
    selected: List[Dict[str, Any]],
    *,
    pair_spec: RescuePairSpec,
    seed: int,
    profile: Optional[str],
    scope: str,
    sampling_config: Dict[str, Any],
) -> pl.DataFrame:
    by_shard: Dict[str, List[Dict[str, Any]]] = {}
    for record in selected:
        by_shard.setdefault(record["source_shard"], []).append(record)

    frames: List[pl.DataFrame] = []
    for shard, shard_records in by_shard.items():
        rows_by_idx = {record["row_in_shard"]: record for record in shard_records}
        wanted = sorted(rows_by_idx)
        frame = (
            pl.scan_parquet(shard)
            .with_row_index("rescue_row_in_shard")
            .filter(pl.col("rescue_row_in_shard").is_in(wanted))
            .collect()
        )

        role_frame = pl.DataFrame(
            {
                "rescue_row_in_shard": wanted,
                "rescue_role": [rows_by_idx[idx]["rescue_role"] for idx in wanted],
            }
        )
        frame = frame.with_columns(
            rescue_source_shard=pl.lit(shard),
            rescue_project_id=pl.lit(pair_spec.project_id),
            rescue_pair_id=pl.lit(pair_spec.pair_id),
            rescue_scope=pl.lit(scope),
            rescue_base_sequence=pl.lit(pair_spec.base_sequence),
            rescue_modified_sequence=pl.lit(pair_spec.modified_sequence),
            rescue_selection_seed=pl.lit(seed),
            rescue_sampling_profile=pl.lit(profile or ""),
        )
        frame = frame.join(role_frame, on="rescue_row_in_shard", how="left")
        frames.append(frame)

    if not frames:
        raise ValueError("No selected rows to collect")

    out = pl.concat(frames, how="vertical_relaxed")
    role_order = {
        "reference_query": 0,
        "modified_query": 1,
        "positive_library": 2,
        "negative_library": 3,
    }
    out = out.with_columns(rescue_role_order=pl.col("rescue_role").replace(role_order).cast(pl.Int64))
    sort_cols = ["rescue_pair_id", "rescue_role_order", "rescue_source_shard", "rescue_row_in_shard"]
    present_sort = [col for col in sort_cols if col in out.columns]
    out = out.sort(present_sort).drop("rescue_role_order")
    if sampling_config:
        for key, value in sampling_config.items():
            out = out.with_columns(**{f"rescue_sampling_{key}": pl.lit(value)})
    return out


def build_dataset_for_pair_seed(
    args: argparse.Namespace,
    pair_spec: RescuePairSpec,
    seed: int,
    *,
    scope: str,
) -> tuple[pl.DataFrame, List[int]]:
    shard_paths = _sorted_shards(args.source_glob)
    rng = np.random.RandomState(seed)
    _log(
        f"seed={seed} pair={pair_spec.pair_id}: collecting candidates from "
        f"{len(shard_paths)} shard(s) for project={pair_spec.project_id}, "
        f"base={pair_spec.base_sequence}, modified={pair_spec.modified_sequence}"
    )
    candidates = _metadata_candidates(
        shard_paths,
        project_id=pair_spec.project_id,
        base_sequence=pair_spec.base_sequence,
        modified_sequence=pair_spec.modified_sequence,
        min_negative_clean_edit_distance=args.min_negative_clean_edit_distance,
        log_prefix=f"seed={seed} pair={pair_spec.pair_id}: ",
    )
    _log(
        f"seed={seed} pair={pair_spec.pair_id}: candidate totals "
        f"base={len(candidates['base'])}, modified={len(candidates['modified'])}, "
        f"negative={len(candidates['negative'])}, same_backbone_other={len(candidates['same_backbone_non_base'])}"
    )

    selected_by_role = _select_records(candidates, args, rng)
    _log(
        f"seed={seed} pair={pair_spec.pair_id}: selected "
        + ", ".join(f"{role}={len(records)}" for role, records in selected_by_role.items())
    )
    selected: List[Dict[str, Any]] = []
    for role, records in selected_by_role.items():
        for record in records:
            record["rescue_role"] = role
            selected.append(record)

    sampling_config = {
        "profile": args.profile or "",
        "scope": scope,
        "negative_sampling": args.negative_sampling,
        "max_queries_per_raw_file": args.max_queries_per_raw_file,
        "num_reference_queries": args.num_reference_queries,
        "num_modified_queries": args.num_modified_queries,
        "num_positive_library": args.num_positive_library,
        "num_negative_library": args.num_negative_library,
    }
    frame = _collect_selected_rows(
        selected,
        pair_spec=pair_spec,
        seed=seed,
        profile=args.profile,
        scope=scope,
        sampling_config=sampling_config,
    )
    negative_distances = [
        int(record["clean_site_edit_distance"])
        for record in selected_by_role["negative_library"]
    ]
    _log(f"seed={seed} pair={pair_spec.pair_id}: collected output rows={frame.height}")
    return frame, negative_distances


def build_dataset_for_seed(args: argparse.Namespace, seed: int) -> Dict[str, Any]:
    scope_config = resolve_rescue_scope_config(
        scope=args.scope,
        project_id=args.project_id,
        base_sequence=args.base_sequence,
        rescue_sequence=args.modified_sequence,
        rescue_pairs_path=args.rescue_pairs_path,
    )
    _log(f"seed={seed}: building {scope_config.scope} dataset with {len(scope_config.pairs)} pair(s)")
    pair_frames: List[pl.DataFrame] = []
    negative_distances: List[int] = []
    for pair_index, pair_spec in enumerate(scope_config.pairs, start=1):
        _log(f"seed={seed}: starting pair {pair_index}/{len(scope_config.pairs)} ({pair_spec.pair_id})")
        pair_frame, pair_negative_distances = build_dataset_for_pair_seed(
            args,
            pair_spec,
            seed,
            scope=scope_config.scope,
        )
        pair_frames.append(pair_frame)
        negative_distances.extend(pair_negative_distances)
    out_df = pl.concat(pair_frames, how="vertical_relaxed") if len(pair_frames) > 1 else pair_frames[0]

    seeds = _parse_seeds(args.seeds, args.seed)
    output_path = _resolve_output_path(Path(args.output), seed, multi_seed=len(seeds) > 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"seed={seed}: writing parquet rows={out_df.height} to {output_path}")
    out_df.write_parquet(output_path)

    negative_bins: Dict[str, int] = {}
    for distance in negative_distances:
        bucket = SpectralRescueTaskReformulated.edit_distance_bin(distance)
        negative_bins[bucket] = negative_bins.get(bucket, 0) + 1
    summary = {
        "output": str(output_path),
        "scope": scope_config.scope,
        "scope_guidance": scope_guidance_text(scope_config.scope),
        "source_glob": args.source_glob,
        "pairs": [pair.to_dict() for pair in scope_config.pairs],
        "profile": args.profile,
        "seed": seed,
        "sampling_config": {
            "profile": args.profile or "",
            "scope": scope_config.scope,
            "negative_sampling": args.negative_sampling,
            "max_queries_per_raw_file": args.max_queries_per_raw_file,
            "num_reference_queries": args.num_reference_queries,
            "num_modified_queries": args.num_modified_queries,
            "num_positive_library": args.num_positive_library,
            "num_negative_library": args.num_negative_library,
        },
        "selected_counts": {
            str(role): int(count)
            for role, count in zip(*np.unique(out_df["rescue_role"].to_numpy(), return_counts=True), strict=False)
        },
        "pair_row_counts": out_df.group_by("rescue_pair_id").agg(pl.len().alias("count")).to_dicts()
        if "rescue_pair_id" in out_df.columns
        else [],
        "negative_edit_distance_bins": dict(sorted(negative_bins.items())),
        "num_rows": out_df.height,
    }
    summary_path = output_path.with_suffix(".summary.json")
    if args.s3_output_prefix:
        summary["s3_uploads"] = _s3_output_paths(output_path, summary_path, args.s3_output_prefix)

    summary_path.write_text(json.dumps(summary, indent=2))
    _log(f"seed={seed}: wrote summary to {summary_path}")

    if args.s3_output_prefix:
        _log(f"seed={seed}: uploading parquet and summary to S3")
        _upload_outputs_to_s3(output_path, summary_path, summary["s3_uploads"])
        _log(f"seed={seed}: S3 upload complete")
    return summary


def build_dataset(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = _parse_seeds(args.seeds, args.seed)
    _log(
        f"starting dataset build: scope={args.scope}, profile={args.profile}, "
        f"seeds={seeds}, source_glob={args.source_glob}, output={args.output}"
    )
    summaries = []
    for seed_index, seed in enumerate(seeds, start=1):
        _log(f"starting seed {seed_index}/{len(seeds)}: {seed}")
        summaries.append(build_dataset_for_seed(args, seed))
        _log(f"finished seed {seed_index}/{len(seeds)}: {seed}")
    if len(summaries) == 1:
        return summaries[0]
    return {"seeds": seeds, "datasets": summaries}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=sorted(SpectralRescueTaskReformulated.SAMPLING_PROFILES.keys()),
        default=None,
        help="Apply a named sampling preset (demo or rigorous). Explicit CLI flags override profile defaults.",
    )
    parser.add_argument(
        "--scope",
        choices=["single_base", "multi_base"],
        default="single_base",
        help="single_base: one peptide pair (case study). multi_base: panel from --rescue_pairs_path.",
    )
    parser.add_argument(
        "--rescue_pairs_path",
        default=None,
        help="JSON manifest of peptide pairs (required for multi_base).",
    )
    parser.add_argument("--source_glob", default="<data-root>/lcfm_splits/valid_*.parquet")
    parser.add_argument(
        "--output",
        default="instanovo/foundational/eval/embed_eval_results/spectral_rescue_reformulated/offline_dataset.parquet",
        help="Output parquet path. With multiple seeds, _seed<N> is appended unless {seed} is present.",
    )
    parser.add_argument("--project_id", default="PXD047134")
    parser.add_argument("--base_sequence", default="LEQGQALDDLMPAQK")
    parser.add_argument("--modified_sequence", default="LEQGQALDDLM[UNIMOD:35]PAQK")
    parser.add_argument("--num_reference_queries", type=int, default=10)
    parser.add_argument("--num_modified_queries", type=int, default=10)
    parser.add_argument("--num_positive_library", type=int, default=20)
    parser.add_argument("--num_negative_library", type=int, default=200)
    parser.add_argument("--min_negative_clean_edit_distance", type=int, default=5)
    parser.add_argument(
        "--max_queries_per_raw_file",
        type=int,
        default=None,
        help="If set, cap candidate spectra per raw file before sampling (rigorous profile uses 1).",
    )
    parser.add_argument(
        "--negative_sampling",
        choices=["random", "stratified_by_edit_distance"],
        default="random",
        help="Negative-library sampling strategy.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds. Rigorous profile defaults to 42,43,44.",
    )
    parser.add_argument(
        "--s3_output_prefix",
        default=None,
        help="Optional S3 prefix where the parquet dataset and summary JSON are uploaded.",
    )
    args = parser.parse_args()
    _apply_profile(args)
    return args


def main() -> None:
    summary = build_dataset(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
