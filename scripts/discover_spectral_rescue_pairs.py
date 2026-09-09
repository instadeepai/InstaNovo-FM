#!/usr/bin/env python3
# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Discover base/modified peptide pairs for multi-base spectral rescue.

Two-step workflow (recommended):

  1. Discovery — specify projects, print manifest to stdout, pick pairs manually:

         uv run python scripts/discover_spectral_rescue_pairs.py \\
           --projects PXD047134,PXD010595,PXD000561 \\
           --source_glob '<data-root>/lcfm_splits/*valid*.parquet'

  2. Multi-base launch — save stdout to a JSON file (or pass inline) and run eval:

         RESCUE_PAIRS_PATH=/tmp/rescue_pairs.json bash scripts/run_spectral_rescue_multi_base.sh

By default this script prints only to stdout (no repo config files written).
Use ``--output`` / ``--diagnostics_output`` only if you want files on disk.

Projects are processed **one at a time**: each project triggers its own parquet
``collect()``, so peak memory is ~one project's spectra, not all projects combined.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def _load_project_candidates(stats_json: Optional[str], tier: str, top_projects: Optional[int]) -> List[str]:
    if not stats_json:
        raise ValueError("Provide --projects or --stats_json to choose which projects to scan.")
    payload = json.loads(Path(stats_json).read_text())
    detail = payload.get("tiers", {}).get(tier, {}).get("projects_detail", {})
    ranked = sorted(
        detail.items(),
        key=lambda item: int(item[1].get("spectra", 0)),
        reverse=True,
    )
    if top_projects is not None:
        ranked = ranked[:top_projects]
    project_ids = [project_id for project_id, _ in ranked]
    logger.info(
        "Loaded %d project(s) from stats (%s tier, top=%s): %s",
        len(project_ids),
        tier,
        top_projects,
        ", ".join(project_ids[:5]) + (" ..." if len(project_ids) > 5 else ""),
    )
    return project_ids


def _slug_pair_id(project_id: str, base_sequence: str, modified_sequence: str) -> str:
    mod_tag = "ox" if "UNIMOD:35" in modified_sequence else "mod"
    backbone = SpectralRescueTaskReformulated._backbone_sequence(base_sequence)
    tail = backbone[-6:] if len(backbone) >= 6 else backbone
    return re.sub(r"[^a-zA-Z0-9_]+", "_", f"{project_id}_{tail}_{mod_tag}").lower()


def _eligibility_reason(
    *,
    base_count: int,
    mod_count: int,
    neg_count: int,
    min_base_spectra: int,
    min_modified_spectra: int,
    min_negative_candidates: int,
) -> str:
    if base_count < min_base_spectra:
        return f"base_spectra {base_count} < {min_base_spectra}"
    if mod_count < min_modified_spectra:
        return f"modified_spectra {mod_count} < {min_modified_spectra}"
    if neg_count < min_negative_candidates:
        return f"negative_candidates {neg_count} < {min_negative_candidates}"
    return "eligible"


def _project_pair_candidates(
    project_id: str,
    sequences: List[str],
    unmodified: List[str],
    *,
    min_base_spectra: int,
    min_modified_spectra: int,
    min_negative_candidates: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (eligible_pair_rows, all_diagnostic_rows) for one project."""
    base_counts: Dict[str, int] = {}
    variant_counts: Dict[Tuple[str, str], int] = {}

    for seq, unmod in zip(sequences, unmodified, strict=False):
        seq_canon = SpectralRescueTaskReformulated._canonicalize_modified_sequence(seq)
        unmod_canon = SpectralRescueTaskReformulated._canonicalize_modified_sequence(unmod)
        if seq_canon == unmod_canon:
            base_counts[unmod_canon] = base_counts.get(unmod_canon, 0) + 1
        elif SpectralRescueTaskReformulated._backbone_sequence(seq) == SpectralRescueTaskReformulated._backbone_sequence(
            unmod
        ):
            key = (unmod_canon, seq_canon)
            variant_counts[key] = variant_counts.get(key, 0) + 1

    diagnostics: List[Dict[str, Any]] = []
    eligible: List[Dict[str, Any]] = []

    for (base_seq, mod_seq), mod_count in sorted(variant_counts.items(), key=lambda kv: -kv[1]):
        base_count = base_counts.get(base_seq, 0)
        base_backbone = SpectralRescueTaskReformulated._backbone_sequence(base_seq)
        neg_count = sum(
            1
            for seq, unmod in zip(sequences, unmodified, strict=False)
            if SpectralRescueTaskReformulated._backbone_sequence(unmod) != base_backbone
            and SpectralRescueTaskReformulated._clean_site_edit_distance(base_seq, seq) >= 5
        )
        row = {
            "project_id": project_id,
            "base_sequence": base_seq,
            "modified_sequence": mod_seq,
            "base_spectra": base_count,
            "modified_spectra": mod_count,
            "negative_candidates": neg_count,
            "eligible": (
                base_count >= min_base_spectra
                and mod_count >= min_modified_spectra
                and neg_count >= min_negative_candidates
            ),
            "reason": _eligibility_reason(
                base_count=base_count,
                mod_count=mod_count,
                neg_count=neg_count,
                min_base_spectra=min_base_spectra,
                min_modified_spectra=min_modified_spectra,
                min_negative_candidates=min_negative_candidates,
            ),
        }
        diagnostics.append(row)
        if not row["eligible"]:
            continue
        eligible.append(
            {
                "pair_id": _slug_pair_id(project_id, base_seq, mod_seq),
                "project_id": project_id,
                "base_sequence": base_seq,
                "modified_sequence": mod_seq,
                "notes": (
                    f"auto-discovered in {project_id}: "
                    f"base={base_count}, mod={mod_count}, neg={neg_count}"
                ),
                "_rank": (mod_count, base_count, neg_count),
            }
        )

    eligible.sort(key=lambda row: row["_rank"], reverse=True)
    for row in eligible:
        row.pop("_rank", None)
    return eligible, diagnostics


def _build_shard_scan(shard_list: List[Path]) -> pl.LazyFrame:
    """Lazy scan over all shards (no rows loaded until ``collect``)."""
    return pl.concat(
        [
            pl.scan_parquet(str(path))
            .select(["usi", "sequence", "unmodified_peptide"])
            for path in shard_list
        ],
        how="vertical_relaxed",
    ).with_columns(project_id=_project_from_usi_expr())


def _collect_project_frame(scan: pl.LazyFrame, project_id: str) -> Optional[pl.DataFrame]:
    """Materialize spectra for a single project, then release scan work for that filter."""
    project_df = scan.filter(pl.col("project_id") == project_id).collect()
    if project_df.is_empty():
        return None
    return project_df


def discover_pairs(
    shard_paths: Iterable[Path],
    *,
    project_order: List[str],
    min_base_spectra: int,
    min_modified_spectra: int,
    min_negative_candidates: int,
    max_pairs: int,
    max_variants_per_backbone: int,
    one_pair_per_project: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return (pair_manifest_entries, diagnostic_rows).

    When ``one_pair_per_project`` is True (default), take the best eligible variant
    per requested project, in ``project_order``.

    Memory strategy: build one lazy scan over shards, then ``collect()`` **one
    project at a time** so peak RAM scales with the largest single project, not
    the sum of all requested projects.
    """
    shard_list = list(shard_paths)
    logger.info(
        "Scanning %d parquet shard(s) for %d project(s) (one project per collect)",
        len(shard_list),
        len(project_order),
    )
    logger.info(
        "Thresholds: min_base=%d min_modified=%d min_negative=%d max_pairs=%d one_pair_per_project=%s",
        min_base_spectra,
        min_modified_spectra,
        min_negative_candidates,
        max_pairs,
        one_pair_per_project,
    )

    scan = _build_shard_scan(shard_list)

    diagnostics: List[Dict[str, Any]] = []
    pair_rows: List[Dict[str, Any]] = []
    projects_with_data = 0

    for index, project_id in enumerate(project_order, start=1):
        logger.info("[%d/%d] Project %s — collecting rows from shards...", index, len(project_order), project_id)
        project_df = _collect_project_frame(scan, project_id)
        if project_df is None:
            reason = "no spectra in scanned shards for this project"
            logger.warning("  skip: %s", reason)
            diagnostics.append(
                {
                    "project_id": project_id,
                    "eligible": False,
                    "reason": reason,
                }
            )
            continue

        projects_with_data += 1
        num_spectra = len(project_df)
        logger.info("  loaded %d spectra for %s", num_spectra, project_id)
        eligible, project_diag = _project_pair_candidates(
            project_id,
            project_df["sequence"].to_list(),
            project_df["unmodified_peptide"].to_list(),
            min_base_spectra=min_base_spectra,
            min_modified_spectra=min_modified_spectra,
            min_negative_candidates=min_negative_candidates,
        )
        del project_df
        diagnostics.extend(project_diag)

        logger.info(
            "  spectra=%d backbone variants=%d eligible_variants=%d",
            num_spectra,
            len(project_diag),
            len(eligible),
        )

        if not eligible:
            if project_diag:
                best = project_diag[0]
                logger.warning(
                    "  no eligible pair; best variant base=%d mod=%d neg=%d (%s)",
                    best["base_spectra"],
                    best["modified_spectra"],
                    best["negative_candidates"],
                    best["reason"],
                )
            else:
                logger.warning("  no eligible pair; no base/modified variant pairs found")
            continue

        variants_by_backbone: Dict[str, int] = {}
        selected_for_project = 0
        for candidate in eligible:
            backbone = SpectralRescueTaskReformulated._backbone_sequence(candidate["base_sequence"])
            if variants_by_backbone.get(backbone, 0) >= max_variants_per_backbone:
                continue
            variants_by_backbone[backbone] = variants_by_backbone.get(backbone, 0) + 1
            pair_rows.append(candidate)
            selected_for_project += 1
            logger.info(
                "  selected pair_id=%s base=%s mod=%s (%s)",
                candidate["pair_id"],
                candidate["base_sequence"][:24] + ("..." if len(candidate["base_sequence"]) > 24 else ""),
                candidate["modified_sequence"][:32] + ("..." if len(candidate["modified_sequence"]) > 32 else ""),
                candidate["notes"],
            )
            if one_pair_per_project:
                break
            if len(pair_rows) >= max_pairs:
                break
        if selected_for_project == 0:
            logger.warning("  eligible variants found but none selected (max_variants_per_backbone=%d)", max_variants_per_backbone)
        if len(pair_rows) >= max_pairs:
            logger.info("Reached max_pairs=%d; stopping project scan", max_pairs)
            break

    logger.info(
        "Discovery finished: %d pair(s) from %d/%d requested project(s) (%d had shard data)",
        len(pair_rows[:max_pairs]),
        len({row["project_id"] for row in pair_rows}),
        len(project_order),
        projects_with_data,
    )
    return pair_rows[:max_pairs], diagnostics


def build_manifest(
    pairs: List[Dict[str, Any]],
    *,
    project_order: List[str],
    source_glob: str,
    stats_json: Optional[str],
    min_base_spectra: int,
    min_modified_spectra: int,
    min_negative_candidates: int,
) -> Dict[str, Any]:
    return {
        "scope": "multi_base",
        "description": "Rescue pairs discovered from LCFM valid shards (one best pair per project).",
        "discovery": {
            "source_glob": source_glob,
            "stats_json": stats_json,
            "projects": project_order,
            "min_base_spectra": min_base_spectra,
            "min_modified_spectra": min_modified_spectra,
            "min_negative_candidates": min_negative_candidates,
        },
        "pairs": pairs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_glob", default="<data-root>/lcfm_splits/*valid*.parquet")
    parser.add_argument(
        "--stats_json",
        default=None,
        help="Optional stats JSON to pick top projects when --projects is not set.",
    )
    parser.add_argument("--tier", default="lcfm", choices=["lcfm", "mcfm", "hcfm"])
    parser.add_argument(
        "--top_projects",
        type=int,
        default=15,
        help="With --stats_json only: scan top-N projects by spectrum count.",
    )
    parser.add_argument(
        "--projects",
        default=None,
        help="Comma-separated PXD list. One best pair is chosen per project (default).",
    )
    parser.add_argument("--min_base_spectra", type=int, default=101, help="Need ref+pos sampling headroom.")
    parser.add_argument("--min_modified_spectra", type=int, default=50)
    parser.add_argument("--min_negative_candidates", type=int, default=200)
    parser.add_argument("--max_pairs", type=int, default=50, help="Cap total pairs when not using one-per-project.")
    parser.add_argument("--max_variants_per_backbone", type=int, default=1)
    parser.add_argument(
        "--all_variants_per_project",
        action="store_true",
        help="Take all eligible variants per project (up to --max_pairs), not just the best one.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the manifest JSON. Default: stdout only.",
    )
    parser.add_argument(
        "--diagnostics_output",
        default=None,
        help="Optional path for near-miss diagnostics JSON.",
    )
    parser.add_argument(
        "--show_diagnostics",
        action="store_true",
        help="Also print diagnostics JSON to stderr.",
    )
    return parser.parse_args()


def main() -> None:
    import glob

    _configure_logging()
    args = parse_args()
    logger.info("Spectral rescue pair discovery")
    logger.info("source_glob=%s", args.source_glob)

    shard_paths = [Path(p) for p in sorted(glob.glob(args.source_glob))]
    if not shard_paths:
        raise FileNotFoundError(f"No parquet files matched {args.source_glob!r}")
    logger.info("Matched %d shard file(s)", len(shard_paths))

    if args.projects:
        project_order = [part.strip() for part in args.projects.split(",") if part.strip()]
        logger.info("Using explicit project list (%d): %s", len(project_order), ", ".join(project_order))
    else:
        logger.info("Ranking projects from stats_json=%s", args.stats_json)
        project_order = _load_project_candidates(args.stats_json, args.tier, args.top_projects)

    if not project_order:
        raise SystemExit("No projects to scan. Set --projects or --stats_json.")

    pairs, diagnostics = discover_pairs(
        shard_paths,
        project_order=project_order,
        min_base_spectra=args.min_base_spectra,
        min_modified_spectra=args.min_modified_spectra,
        min_negative_candidates=args.min_negative_candidates,
        max_pairs=args.max_pairs,
        max_variants_per_backbone=args.max_variants_per_backbone,
        one_pair_per_project=not args.all_variants_per_project,
    )

    manifest = build_manifest(
        pairs,
        project_order=project_order,
        source_glob=args.source_glob,
        stats_json=args.stats_json,
        min_base_spectra=args.min_base_spectra,
        min_modified_spectra=args.min_modified_spectra,
        min_negative_candidates=args.min_negative_candidates,
    )
    manifest_text = json.dumps(manifest, indent=2)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(manifest_text)
        logger.info("Wrote manifest (%d pairs) to %s", len(pairs), output)
    else:
        logger.info("Writing manifest JSON to stdout (%d pairs)", len(pairs))
        print(manifest_text)

    if args.diagnostics_output:
        Path(args.diagnostics_output).write_text(json.dumps(diagnostics, indent=2))
        logger.info("Wrote diagnostics to %s", args.diagnostics_output)

    if args.show_diagnostics:
        print(json.dumps({"diagnostics": diagnostics}, indent=2), file=sys.stderr)

    if not pairs:
        logger.error(
            "No eligible pairs in %d project(s); inspect diagnostics or lower thresholds",
            len(project_order),
        )
        raise SystemExit(
            f"No eligible pairs in {len(project_order)} project(s). "
            "Try --show_diagnostics or lower --min_* thresholds."
        )

    if len(pairs) < 2:
        logger.warning(
            "Found %d pair(s); multi_base eval needs at least 2 — add projects or relax thresholds",
            len(pairs),
        )


if __name__ == "__main__":
    main()
