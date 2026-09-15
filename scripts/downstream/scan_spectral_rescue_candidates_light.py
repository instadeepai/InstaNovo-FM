#!/usr/bin/env python3
# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Fast approximate scan for spectral-rescue candidate pairs.

This script is a lightweight pre-screening tool. It finds projects and
base/modified peptide pairs with enough spectra for rescue sampling, without
the expensive per-variant edit-distance negative counting used by
``discover_spectral_rescue_pairs.py``.

The approximate negative count is:

    project_total_spectra - spectra_with_same_base_backbone

The rigorous dataset builder still performs exact sampling and should remain
the final gate before evaluation.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import polars as pl

from instanovo_fm.eval.embed_eval_tasks.spectral_rescue_reformulated import (
    SpectralRescueTaskReformulated,
)

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )


def _project_from_usi_expr(usi_col: str = "usi") -> pl.Expr:
    return pl.col(usi_col).str.split(":").list.get(1)


def _raw_file_from_usi_expr(usi_col: str = "usi") -> pl.Expr:
    return pl.col(usi_col).str.split(":").list.get(2)


def _slug_pair_id(project_id: str, base_sequence: str, modified_sequence: str) -> str:
    mod_tag = "ox" if "UNIMOD:35" in modified_sequence else "mod"
    backbone = SpectralRescueTaskReformulated._backbone_sequence(base_sequence)
    tail = backbone[-6:] if len(backbone) >= 6 else backbone
    return re.sub(r"[^a-zA-Z0-9_]+", "_", f"{project_id}_{tail}_{mod_tag}").lower()


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _scan_grouped_counts(shard_paths: list[str], projects: set[str] | None) -> pl.DataFrame:
    scans = [
        pl.scan_parquet(path).select(["usi", "sequence", "unmodified_peptide"])
        for path in shard_paths
    ]
    scan = (
        pl.concat(scans, how="vertical_relaxed")
        .drop_nulls(["usi", "sequence", "unmodified_peptide"])
        .with_columns(
            project_id=_project_from_usi_expr(),
            raw_file=_raw_file_from_usi_expr(),
        )
        .drop_nulls(["project_id", "raw_file"])
    )
    if projects:
        scan = scan.filter(pl.col("project_id").is_in(sorted(projects)))
    return (
        scan.group_by(["project_id", "unmodified_peptide", "sequence"])
        .agg(
            spectra=pl.len(),
            raw_files=pl.col("raw_file").n_unique(),
        )
        .collect()
    )


def scan_candidates(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.monotonic()
    shard_paths = sorted(glob.glob(args.source_glob))
    if args.max_shards is not None:
        shard_paths = shard_paths[: args.max_shards]
    if not shard_paths:
        raise FileNotFoundError(f"No parquet files matched {args.source_glob!r}")

    projects = None
    if args.projects:
        projects = {part.strip() for part in args.projects.split(",") if part.strip()}

    logger.info("Light spectral-rescue candidate scan")
    logger.info("source_glob=%s", args.source_glob)
    logger.info("matched_shards=%d", len(shard_paths))
    logger.info("POLARS_MAX_THREADS=%s", os.environ.get("POLARS_MAX_THREADS", "<unset>"))
    logger.info("polars_thread_pool_size=%s", pl.thread_pool_size())
    logger.info("projects=%s", ",".join(sorted(projects)) if projects else "all")
    logger.info(
        "thresholds: min_base=%d min_modified=%d min_approx_negative=%d",
        args.min_base_spectra,
        args.min_modified_spectra,
        args.min_negative_candidates,
    )

    grouped_started_at = time.monotonic()
    grouped = _scan_grouped_counts(shard_paths, projects)
    logger.info(
        "grouped project/unmodified/sequence rows=%d in %.1fs",
        grouped.height,
        time.monotonic() - grouped_started_at,
    )

    project_totals: dict[str, int] = defaultdict(int)
    project_raw_files: dict[str, int] = defaultdict(int)
    project_base_sequences: dict[str, set[str]] = defaultdict(set)
    project_modified_variants: dict[str, int] = defaultdict(int)
    base_counts: dict[tuple[str, str], int] = defaultdict(int)
    base_raw_file_counts: dict[tuple[str, str], int] = defaultdict(int)
    backbone_counts: dict[tuple[str, str], int] = defaultdict(int)
    backbone_raw_file_counts: dict[tuple[str, str], int] = defaultdict(int)
    variant_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    variant_raw_file_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    display_base: dict[tuple[str, str], str] = {}
    display_mod: dict[tuple[str, str, str], str] = {}
    base_backbone: dict[tuple[str, str], str] = {}

    for row in grouped.iter_rows(named=True):
        project_id = str(row["project_id"])
        sequence = str(row["sequence"])
        unmodified = str(row["unmodified_peptide"])
        count = int(row["spectra"])
        raw_files = int(row["raw_files"])

        seq_canon = SpectralRescueTaskReformulated._canonicalize_modified_sequence(sequence)
        unmod_canon = SpectralRescueTaskReformulated._canonicalize_modified_sequence(unmodified)
        seq_backbone = SpectralRescueTaskReformulated._backbone_sequence(sequence)
        unmod_backbone = SpectralRescueTaskReformulated._backbone_sequence(unmodified)

        project_totals[project_id] += count
        project_raw_files[project_id] += raw_files
        project_base_sequences[project_id].add(unmod_canon)
        backbone_counts[(project_id, unmod_backbone)] += count
        backbone_raw_file_counts[(project_id, unmod_backbone)] += raw_files

        if seq_canon == unmod_canon:
            base_counts[(project_id, unmod_canon)] += count
            base_raw_file_counts[(project_id, unmod_canon)] += raw_files
            display_base.setdefault((project_id, unmod_canon), unmod_canon)
            base_backbone[(project_id, unmod_canon)] = unmod_backbone
        elif seq_backbone == unmod_backbone:
            key = (project_id, unmod_canon, seq_canon)
            variant_counts[key] += count
            variant_raw_file_counts[key] += raw_files
            display_base.setdefault((project_id, unmod_canon), unmod_canon)
            display_mod.setdefault(key, seq_canon)
            base_backbone[(project_id, unmod_canon)] = unmod_backbone
            project_modified_variants[project_id] += 1

    pair_rows: list[dict[str, Any]] = []
    eligible_pair_rows: list[dict[str, Any]] = []
    for (project_id, base_seq, mod_seq), mod_count in variant_counts.items():
        base_count = base_counts.get((project_id, base_seq), 0)
        base_raw_files = base_raw_file_counts.get((project_id, base_seq), 0)
        mod_raw_files = variant_raw_file_counts.get((project_id, base_seq, mod_seq), 0)
        backbone = base_backbone.get(
            (project_id, base_seq),
            SpectralRescueTaskReformulated._backbone_sequence(base_seq),
        )
        same_backbone_count = backbone_counts.get((project_id, backbone), 0)
        same_backbone_raw_files = backbone_raw_file_counts.get((project_id, backbone), 0)
        approx_negative = max(project_totals[project_id] - same_backbone_count, 0)
        approx_negative_raw_files = max(project_raw_files[project_id] - same_backbone_raw_files, 0)
        eligible = (
            base_count >= args.min_base_spectra
            and mod_count >= args.min_modified_spectra
            and approx_negative >= args.min_negative_candidates
            and base_raw_files >= args.min_base_raw_files
            and mod_raw_files >= args.min_modified_raw_files
            and approx_negative_raw_files >= args.min_negative_raw_files
        )
        out = {
            "project_id": project_id,
            "pair_id": _slug_pair_id(project_id, base_seq, mod_seq),
            "base_sequence": display_base.get((project_id, base_seq), base_seq),
            "modified_sequence": display_mod.get((project_id, base_seq, mod_seq), mod_seq),
            "base_spectra": base_count,
            "modified_spectra": mod_count,
            "approx_negative_candidates": approx_negative,
            "base_raw_files": base_raw_files,
            "modified_raw_files": mod_raw_files,
            "approx_negative_raw_files": approx_negative_raw_files,
            "same_backbone_spectra": same_backbone_count,
            "same_backbone_raw_files": same_backbone_raw_files,
            "project_spectra": project_totals[project_id],
            "project_raw_files": project_raw_files[project_id],
            "eligible": eligible,
        }
        pair_rows.append(out)
        if eligible:
            eligible_pair_rows.append(out)

    pair_rows.sort(
        key=lambda row: (
            bool(row["eligible"]),
            int(row["modified_raw_files"]),
            int(row["base_raw_files"]),
            int(row["modified_spectra"]),
            int(row["base_spectra"]),
            int(row["approx_negative_candidates"]),
        ),
        reverse=True,
    )
    eligible_pair_rows.sort(
        key=lambda row: (
            int(row["modified_raw_files"]),
            int(row["base_raw_files"]),
            int(row["modified_spectra"]),
            int(row["base_spectra"]),
            int(row["approx_negative_candidates"]),
        ),
        reverse=True,
    )

    project_rows = []
    for project_id, spectra in project_totals.items():
        eligible_count = sum(1 for row in eligible_pair_rows if row["project_id"] == project_id)
        project_rows.append(
            {
                "project_id": project_id,
                "spectra": spectra,
                "raw_files": project_raw_files[project_id],
                "base_sequence_groups": len(project_base_sequences[project_id]),
                "modified_variant_groups": project_modified_variants[project_id],
                "eligible_pairs": eligible_count,
            }
        )
    project_rows.sort(key=lambda row: (int(row["eligible_pairs"]), int(row["spectra"])), reverse=True)  # type: ignore[call-overload]

    selected_pairs = []
    seen_project: set[str] = set()
    seen_backbone: set[tuple[str, str]] = set()
    for row in eligible_pair_rows:
        if args.one_pair_per_project and row["project_id"] in seen_project:
            continue
        backbone = SpectralRescueTaskReformulated._backbone_sequence(row["base_sequence"])
        backbone_key = (row["project_id"], backbone)
        if backbone_key in seen_backbone:
            continue
        selected_pairs.append(
            {
                "pair_id": row["pair_id"],
                "project_id": row["project_id"],
                "base_sequence": row["base_sequence"],
                "modified_sequence": row["modified_sequence"],
                "notes": (
                    f"light-scan in {row['project_id']}: base={row['base_spectra']}, "
                    f"mod={row['modified_spectra']}, approx_neg={row['approx_negative_candidates']}, "
                    f"base_raw={row['base_raw_files']}, mod_raw={row['modified_raw_files']}, "
                    f"approx_neg_raw={row['approx_negative_raw_files']}"
                ),
            }
        )
        seen_project.add(row["project_id"])
        seen_backbone.add(backbone_key)
        if len(selected_pairs) >= args.max_pairs:
            break

    summary = {
        "source_glob": args.source_glob,
        "matched_shards": len(shard_paths),
        "polars_max_threads": os.environ.get("POLARS_MAX_THREADS"),
        "polars_thread_pool_size": pl.thread_pool_size(),
        "projects_scanned": len(project_rows),
        "candidate_pairs": len(pair_rows),
        "eligible_pairs": len(eligible_pair_rows),
        "selected_pairs": len(selected_pairs),
        "elapsed_seconds": round(time.monotonic() - started_at, 3),
        "thresholds": {
            "min_base_spectra": args.min_base_spectra,
            "min_modified_spectra": args.min_modified_spectra,
            "min_negative_candidates": args.min_negative_candidates,
            "min_base_raw_files": args.min_base_raw_files,
            "min_modified_raw_files": args.min_modified_raw_files,
            "min_negative_raw_files": args.min_negative_raw_files,
        },
    }
    manifest = {
        "scope": "multi_base",
        "description": "Rescue pairs from lightweight grouped-count scan; approximate negatives.",
        "discovery": summary,
        "pairs": selected_pairs,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    project_csv = output_dir / f"{args.output_prefix}_projects.csv"
    pairs_csv = output_dir / f"{args.output_prefix}_pairs.csv"
    eligible_csv = output_dir / f"{args.output_prefix}_eligible_pairs.csv"
    manifest_json = output_dir / f"{args.output_prefix}_pairs.json"
    summary_json = output_dir / f"{args.output_prefix}_summary.json"

    _write_csv(
        project_csv,
        project_rows,
        ["project_id", "spectra", "raw_files", "base_sequence_groups", "modified_variant_groups", "eligible_pairs"],
    )
    pair_fields = [
        "project_id",
        "pair_id",
        "base_sequence",
        "modified_sequence",
        "base_spectra",
        "modified_spectra",
        "approx_negative_candidates",
        "base_raw_files",
        "modified_raw_files",
        "approx_negative_raw_files",
        "same_backbone_spectra",
        "same_backbone_raw_files",
        "project_spectra",
        "project_raw_files",
        "eligible",
    ]
    _write_csv(
        pairs_csv,
        pair_rows,
        pair_fields,
    )
    _write_csv(
        eligible_csv,
        eligible_pair_rows,
        pair_fields,
    )
    manifest_json.write_text(json.dumps(manifest, indent=2))
    summary_json.write_text(json.dumps(summary, indent=2))

    logger.info("Wrote project summary to %s", project_csv)
    logger.info("Wrote pair summary to %s", pairs_csv)
    logger.info("Wrote eligible pairs to %s", eligible_csv)
    logger.info("Wrote selected pair manifest to %s", manifest_json)
    logger.info("Summary: %s", json.dumps(summary, sort_keys=True))
    print(json.dumps(summary, indent=2))
    return {"summary": summary, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_glob", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--output_prefix", default="light_scan")
    parser.add_argument("--projects", default=None, help="Optional comma-separated project IDs to keep.")
    parser.add_argument("--max_shards", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--min_base_spectra", type=int, default=101)
    parser.add_argument("--min_modified_spectra", type=int, default=50)
    parser.add_argument("--min_negative_candidates", type=int, default=200)
    parser.add_argument(
        "--min_base_raw_files",
        type=int,
        default=100,
        help="Need 50 reference queries + 50 positive library after one-per-raw-file dedupe.",
    )
    parser.add_argument("--min_modified_raw_files", type=int, default=50)
    parser.add_argument("--min_negative_raw_files", type=int, default=200)
    parser.add_argument("--max_pairs", type=int, default=10)
    parser.add_argument(
        "--one_pair_per_project",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select at most one pair per project for the output manifest.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    scan_candidates(parse_args())


if __name__ == "__main__":
    main()
