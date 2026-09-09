# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Stage 2: offline, resumable spectral-evidence scoring (blocks A/B/C).

Decouples the expensive evidence-metrics computation from the fast embedding
retrieval step of the Kostas cross-set annotation transfer pipeline.

Run Stage 1 first -- the normal evaluator run with evidence metrics disabled:

    uv run python -m instanovo_fm.eval.embed_evaluation \\
        --config-name foundational_eval_cross_set_annotation_transfer \\
        evaluation.task_configs.crosssetannotationtransfertask.compute_evidence_metrics=false

That produces ``cross_set_topk_candidates.csv`` within the embedding-generation
time (tens of minutes), instead of blocking for hours on evidence scoring.

Then run this script against that CSV plus the combined Kostas parquet to compute
blocks A/B/C separately, with:
  - Per-(peptide, charge) memoized theoretical-mass computation (relies on the
    ``lru_cache`` on ``proteomics_mcp.core.annotation.aa_sequence_from_proforma``),
    which removes the ~7.24M-call redundancy that caused the multi-hour stall.
  - Incremental checkpointing to a resumable JSONL file -- safe to kill and rerun
    with ``--resume``.
  - Progress logging with a live ETA.

Usage:
    uv run python scripts/compute_cross_set_evidence_metrics.py \\
        --candidates-csv /path/to/cross_set_topk_candidates.csv \\
        --combined-parquet /path/to/PXD074343_kostas_top20.parquet \\
        --output-dir /path/to/crosssetannotationtransfertask \\
        --num-workers 8 --score-blocks A B C --resume
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from instanovo_fm.eval.spectrum_metrics.mcp_scoring import MCP_AVAILABLE  # noqa: E402
from instanovo_fm.eval.spectrum_metrics.worker import (  # noqa: E402
    LibrarySelfWorkItem,
    QueryRankWorkItem,
    SpectrumRecord,
    score_library_self,
    score_query_rank_worker,
)


def log(t0: float, msg: str) -> None:
    print(f"[t={time.time() - t0:8.1f}s] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-csv", required=True, help="cross_set_topk_candidates.csv from a Stage 1 run")
    parser.add_argument("--combined-parquet", required=True, help="Combined Kostas parquet (source of truth for spectra)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--score-blocks", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--tolerance-da", type=float, default=0.05)
    parser.add_argument("--ion-types", default="by")
    parser.add_argument("--max-ion-charge", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=20_000, help="Work items per checkpoint flush")
    parser.add_argument("--resume", action="store_true", help="Skip (query_id, rank) pairs already in the checkpoint file")
    parser.add_argument("--id-key", default="usi")
    parser.add_argument("--peptide-key", default="sequence")
    parser.add_argument("--unmodified-key", default="unmodified_peptide")
    parser.add_argument("--mz-key", default="mz_array")
    parser.add_argument("--intensity-key", default="intensity_array")
    parser.add_argument("--precursor-mz-key", default="precursor_mz")
    parser.add_argument("--precursor-charge-key", default="precursor_charge")
    return parser.parse_args()


def _row_to_record(row: dict, args: argparse.Namespace, *, is_library: bool) -> SpectrumRecord:
    peptide = str(row.get(args.peptide_key) or "") if is_library else ""
    unmodified = str(row.get(args.unmodified_key) or peptide) if is_library else ""
    mz = np.asarray(row[args.mz_key], dtype=float).flatten()
    intensity = np.asarray(row[args.intensity_key], dtype=float).flatten()
    precursor_mz = row.get(args.precursor_mz_key)
    precursor_charge = row.get(args.precursor_charge_key)
    return SpectrumRecord(
        mz=tuple(float(v) for v in mz.tolist()),
        intensity=tuple(float(v) for v in intensity.tolist()),
        precursor_mz=float(precursor_mz) if precursor_mz is not None else None,
        precursor_charge=int(precursor_charge) if precursor_charge is not None else None,
        peptide=peptide,
        unmodified_peptide=unmodified,
    )


def load_records(
    parquet_path: str,
    ids: Set[str],
    args: argparse.Namespace,
    *,
    is_library: bool,
) -> Dict[str, SpectrumRecord]:
    columns = [args.id_key, args.mz_key, args.intensity_key, args.precursor_mz_key, args.precursor_charge_key]
    if is_library:
        columns += [args.peptide_key, args.unmodified_key]
    df = pl.read_parquet(parquet_path, columns=list(dict.fromkeys(columns)))
    df = df.filter(pl.col(args.id_key).is_in(list(ids)))
    return {str(row[args.id_key]): _row_to_record(row, args, is_library=is_library) for row in df.to_dicts()}


def load_checkpoint(checkpoint_path: Path) -> Dict[Tuple[str, int], Dict[str, Any]]:
    done: Dict[Tuple[str, int], Dict[str, Any]] = {}
    if not checkpoint_path.exists():
        return done
    with checkpoint_path.open("r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[(str(row["query_id"]), int(row["rank"]))] = row
    return done


def main() -> None:
    args = parse_args()
    t0 = time.time()
    print(f"MCP_AVAILABLE={MCP_AVAILABLE}")
    if any(b in args.score_blocks for b in ("B", "C")) and not MCP_AVAILABLE:
        raise SystemExit("Blocks B/C require proteomics-mcp. Install with: uv sync --extra proteomics-metrics")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / "evidence_checkpoint.jsonl"

    candidates = pl.read_csv(args.candidates_csv)
    log(t0, f"Loaded {candidates.height} candidate rows from {args.candidates_csv}")

    query_ids = set(candidates.select("query_id").unique().to_series().to_list())
    library_ids = set(candidates.select("library_id").unique().to_series().to_list())
    log(t0, f"Distinct query spectra: {len(query_ids)}, distinct library candidates: {len(library_ids)}")

    library_records = load_records(args.combined_parquet, library_ids, args, is_library=True)
    query_records = load_records(args.combined_parquet, query_ids, args, is_library=False)
    log(t0, f"Loaded {len(library_records)} library / {len(query_records)} query spectrum records")

    blocks = tuple(b.upper() for b in args.score_blocks)
    workers = max(args.num_workers, 1)

    library_self_cache: Dict[str, Dict[str, Any]] = {}
    if "C" in blocks:
        lib_items = [
            LibrarySelfWorkItem(
                library_index=i,
                library=rec,
                tolerance_da=args.tolerance_da,
                ion_types=args.ion_types,
                max_ion_charge=args.max_ion_charge,
            )
            for i, rec in enumerate(library_records.values())
        ]
        lib_id_by_index = dict(enumerate(library_records.keys()))
        t_lib = time.time()
        if workers > 1 and len(lib_items) > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                for future in as_completed([executor.submit(score_library_self, item) for item in lib_items]):
                    idx, metrics = future.result()
                    library_self_cache[lib_id_by_index[idx]] = metrics
        else:
            for item in lib_items:
                idx, metrics = score_library_self(item)
                library_self_cache[lib_id_by_index[idx]] = metrics
        log(t0, f"Block C (library self-consistency): {len(lib_items)} items in {time.time() - t_lib:.1f}s")

    done = load_checkpoint(checkpoint_path) if args.resume else {}
    if done:
        log(t0, f"Resuming: {len(done)} (query_id, rank) pairs already checkpointed, will be skipped")

    rows_to_score: List[dict] = [
        row for row in candidates.select(["query_id", "rank", "library_id"]).to_dicts() if (str(row["query_id"]), int(row["rank"])) not in done
    ]
    log(t0, f"{len(rows_to_score)} of {candidates.height} pairs remain to score")

    checkpoint_handle = checkpoint_path.open("a")
    n_done = len(done)
    n_total = candidates.height
    t_score_start = time.time()

    def flush_batch(batch_results: Iterable[dict]) -> None:
        nonlocal n_done
        for result in batch_results:
            checkpoint_handle.write(json.dumps(result, default=str) + "\n")
            n_done += 1
        checkpoint_handle.flush()
        elapsed = time.time() - t_score_start
        rate = (n_done - len(done)) / elapsed if elapsed > 0 else 0.0
        remaining = n_total - n_done
        eta_s = remaining / rate if rate > 0 else float("inf")
        log(
            t0,
            f"Checkpointed {n_done}/{n_total} ({100 * n_done / n_total:.1f}%) -- rate={rate:.1f} items/s, ETA={eta_s / 3600:.2f}h",
        )

    for start in range(0, len(rows_to_score), args.batch_size):
        chunk = rows_to_score[start : start + args.batch_size]
        work_items = []
        for row in chunk:
            query_id = str(row["query_id"])
            library_id = str(row["library_id"])
            if query_id not in query_records or library_id not in library_records:
                continue
            work_items.append(
                (
                    query_id,
                    int(row["rank"]),
                    QueryRankWorkItem(
                        query_index=0,
                        rank=int(row["rank"]),
                        library_index=0,
                        query=query_records[query_id],
                        library=library_records[library_id],
                        score_blocks=tuple(b for b in blocks if b in ("A", "B")),
                        tolerance_da=args.tolerance_da,
                        ion_types=args.ion_types,
                        max_ion_charge=args.max_ion_charge,
                    ),
                    library_id,
                )
            )

        batch_results = []
        if workers > 1 and len(work_items) > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        score_query_rank_worker,
                        (item, {library_id: library_self_cache[library_id]} if library_id in library_self_cache else None),
                    ): (query_id, rank, library_id)
                    for query_id, rank, item, library_id in work_items
                }
                for future in as_completed(futures):
                    query_id, rank, library_id = futures[future]
                    result = future.result()
                    result["query_id"] = query_id
                    result["library_id"] = library_id
                    batch_results.append(result)
        else:
            for query_id, _rank, item, library_id in work_items:
                cache = {library_id: library_self_cache[library_id]} if library_id in library_self_cache else None
                result = score_query_rank_worker((item, cache))
                result["query_id"] = query_id
                result["library_id"] = library_id
                batch_results.append(result)

        flush_batch(batch_results)

    checkpoint_handle.close()
    log(t0, f"Evidence scoring complete: {n_done}/{n_total} pairs checkpointed at {checkpoint_path}")

    # --- Merge checkpoint back into candidates and write the enriched CSV ---
    evidence_by_key = load_checkpoint(checkpoint_path)
    enriched_rows = []
    for row in candidates.to_dicts():
        key = (str(row["query_id"]), int(row["rank"]))
        metrics = evidence_by_key.get(key, {})
        merged = dict(row)
        for metric_key, metric_value in metrics.items():
            if metric_key in {"query_id", "rank", "library_id"}:
                continue
            merged[metric_key] = metric_value
        enriched_rows.append(merged)

    enriched_path = out_dir / "cross_set_topk_candidates_with_evidence.csv"
    pl.DataFrame(enriched_rows).write_csv(enriched_path)
    log(t0, f"Wrote enriched candidates with evidence to {enriched_path}")

    summary = {
        "candidates_csv": args.candidates_csv,
        "combined_parquet": args.combined_parquet,
        "score_blocks": list(blocks),
        "num_workers": workers,
        "num_pairs_total": n_total,
        "num_pairs_scored": n_done,
        "checkpoint_jsonl": str(checkpoint_path),
        "enriched_candidates_csv": str(enriched_path),
        "elapsed_seconds": time.time() - t0,
    }
    summary_path = out_dir / "evidence_scoring_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    log(t0, f"Done. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
