# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Select the top-K queries from a Stage 1 candidates CSV before running Stage 2.

Complements the existing library-side pruning (``TOP_N_PEPTIDES``) with a
query-side analog: out of the (possibly huge) set of unlabeled ACFM queries,
keep only the top-K "best" ones before paying the expensive blocks A/B/C
evidence-scoring cost on them in Stage 2
(``scripts/compute_cross_set_evidence_metrics.py``).

ACFM queries carry no identification confidence (no hyperscore/probability/
expectation -- there's no peptide ID to be confident about), so "confidence"
here means one of:

  - ``model_confidence`` : the model's own masked-reconstruction confidence
                      per spectrum (conf_group * conf_offset, averaged over
                      valid peaks) -- present as ``query_spectrum_confidence``
                      in the candidates CSV when the Stage 1 run was launched
                      with ``evaluation.compute_spectrum_confidence=true``
                      (default in run_cross_set_annotation_transfer.sh).
                      Identification-free and model-native, rather than a
                      hand-crafted heuristic -- prefer this over ``quality``
                      when the column is available.
  - ``margin``     : post-retrieval embedding margin (top1 vs top2 cosine
                      similarity from Stage 1). High margin = the model is
                      unambiguously pointing at one specific library peptide.
                      This is the closest analog to a search engine's
                      hyperscore-minus-nextscore margin, just computed in
                      embedding space instead of PSM-score space.
  - ``top1_score``  : raw top1 embedding cosine similarity, regardless of
                      ambiguity among the top candidates.
  - ``quality``     : intrinsic, identification-free spectral quality
                      computed directly from the query's own raw peaks
                      (no peptide hypothesis involved at all). Requires
                      --combined-parquet to look up mz/intensity arrays.
                      Fallback for when query_spectrum_confidence isn't
                      available (e.g. rerunning Stage 2 selection against an
                      older Stage 1 output).
  - ``combination`` : quality pre-filter (keep the best
                      --quality-prefilter-frac by model_confidence if
                      available, else the intrinsic quality metric), then
                      rank the survivors by embedding margin. This is the
                      recommended default: cheap junk-spectrum rejection
                      first, then pick the most unambiguous matches among
                      what's left.
  - ``annotation_coverage`` : spectral-evidence pre-filter using
                      proteomics-mcp. For each query's *rank-1* retrieved
                      library peptide only (not all --top-k ranks -- that's
                      the expensive full Stage 2), scores how well that
                      candidate's theoretical b/y-ion spectrum explains the
                      query's own observed peaks (block B, restricted to
                      rank 1). This is a real "does the assigned peptide's
                      annotation actually fit this spectrum" check, not a
                      heuristic, and costs ~1 MCP call per query instead of
                      --top-k calls per query. Requires --combined-parquet
                      and proteomics-mcp installed
                      (uv sync --extra proteomics-metrics).

Usage:
    uv run python scripts/downstream/select_topk_queries.py \\
        --candidates-csv /path/to/cross_set_topk_candidates.csv \\
        --combined-parquet /path/to/PXD074343_kostas_top10_stage1.parquet \\
        --output-csv /path/to/cross_set_topk_candidates_selected.csv \\
        --metric combination --top-k 5000 --quality-prefilter-frac 0.5
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import polars as pl


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for query selection."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-csv", required=True, help="cross_set_topk_candidates.csv from a Stage 1 run")
    parser.add_argument("--output-csv", required=True, help="Filtered candidates CSV, ready for Stage 2")
    parser.add_argument("--combined-parquet", default=None, help="Required for --metric quality|combination")
    parser.add_argument(
        "--metric",
        default="combination",
        choices=[
            "model_confidence",
            "margin",
            "top1_score",
            "quality",
            "combination",
            "annotation_coverage",
        ],
        help="Query-selection criterion (default: combination)",
    )
    parser.add_argument("--top-k", type=int, default=None, help="Number of queries to keep")
    parser.add_argument("--top-frac", type=float, default=None, help="Alternative to --top-k: fraction of queries to keep")
    parser.add_argument(
        "--quality-metric",
        default="top10_intensity_share",
        choices=["top10_intensity_share", "peak_count", "precursor_intensity", "tic"],
        help="Which intrinsic quality metric to use for --metric quality|combination",
    )
    parser.add_argument(
        "--quality-prefilter-frac",
        type=float,
        default=0.5,
        help="For --metric combination: fraction of queries kept by intrinsic quality before ranking by margin",
    )
    parser.add_argument(
        "--annotation-metric",
        default="annotated_intensity_fraction",
        choices=["annotated_intensity_fraction", "matched_ion_fraction", "matched_ion_count"],
        help=(
            "MCP block-B field to rank by for --metric annotation_coverage: fraction of "
            "the query's observed intensity explained by the rank-1 candidate's theoretical "
            "ions (default), fraction of theoretical ions observed, or raw matched-ion count."
        ),
    )
    parser.add_argument("--tolerance-da", type=float, default=0.05, help="Fragment matching tolerance for block B (Da)")
    parser.add_argument("--ion-types", default="by", help="Ion types for theoretical fragment generation")
    parser.add_argument("--max-ion-charge", type=int, default=2, help="Max fragment ion charge for theoretical spectrum")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "Process pool size for --metric annotation_coverage (0/1 = sequential, single "
            "process, which lets proteomics-mcp's per-(peptide,charge) fragment cache be "
            "shared across queries -- usually fast enough given how few distinct library "
            "peptides there are per file)"
        ),
    )
    parser.add_argument("--id-key", default="usi")
    parser.add_argument("--mz-key", default="mz_array")
    parser.add_argument("--intensity-key", default="intensity_array")
    parser.add_argument("--precursor-intensity-key", default="precursor_intensity")
    parser.add_argument("--precursor-mz-key", default="precursor_mz")
    parser.add_argument("--precursor-charge-key", default="precursor_charge")
    return parser.parse_args()


def compute_quality_scores(
    parquet_path: str,
    query_ids: set[str],
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Compute an intrinsic, identification-free quality score per query usi."""
    columns = [args.id_key, args.mz_key, args.intensity_key, args.precursor_intensity_key]
    df = pl.read_parquet(parquet_path, columns=list(dict.fromkeys(columns)))
    df = df.filter(pl.col(args.id_key).is_in(list(query_ids)))

    scores: Dict[str, float] = {}
    for row in df.to_dicts():
        query_id = str(row[args.id_key])
        intensity = np.asarray(row[args.intensity_key], dtype=float).flatten()
        if args.quality_metric == "peak_count":
            scores[query_id] = float(intensity.size)
        elif args.quality_metric == "tic":
            scores[query_id] = float(intensity.sum())
        elif args.quality_metric == "precursor_intensity":
            value = row.get(args.precursor_intensity_key)
            scores[query_id] = float(value) if value is not None else 0.0
        else:  # top10_intensity_share
            total = float(intensity.sum())
            top10 = float(np.sort(intensity)[-10:].sum()) if intensity.size else 0.0
            scores[query_id] = top10 / total if total > 0 else 0.0
    return scores


def _score_rank1_annotation(
    query_id: str,
    peptide: str,
    mz: list,
    intensity: list,
    precursor_mz: Optional[float],
    precursor_charge: Optional[int],
    tolerance_da: float,
    ion_types: str,
    max_ion_charge: int,
    annotation_metric: str,
) -> tuple[str, float]:
    """Score one query's rank-1 candidate via MCP block B (module-level for picklability)."""
    from instanovo_fm.eval.spectrum_metrics.mcp_scoring import score_observed_vs_theoretical

    if not peptide:
        return query_id, float("-inf")
    try:
        result = score_observed_vs_theoretical(
            observed_mz=np.asarray(mz, dtype=float),
            observed_intensity=np.asarray(intensity, dtype=float),
            peptidoform=peptide,
            precursor_mz=precursor_mz,
            precursor_charge=precursor_charge,
            tolerance_da=tolerance_da,
            ion_types=ion_types,
            max_ion_charge=max_ion_charge,
        )
    except Exception:
        return query_id, float("-inf")
    value = result.get(annotation_metric)
    return query_id, float(value) if value is not None else float("-inf")


def compute_annotation_coverage_scores(
    parquet_path: str,
    rank1_rows: list,
    args: argparse.Namespace,
) -> Dict[str, float]:
    """Score each query's already-retrieved rank-1 candidate via MCP block B.

    This is deliberately restricted to rank 1 (not all --top-k library ranks) so the
    cost is one MCP call per query instead of --top-k calls per query -- the same
    per-item cost as the expensive Stage 2 evidence scoring, but ~20x less work.
    """
    from instanovo_fm.eval.spectrum_metrics.mcp_scoring import MCP_AVAILABLE

    if not MCP_AVAILABLE:
        raise SystemExit(
            "--metric annotation_coverage requires proteomics-mcp. Install with: "
            "uv sync --extra proteomics-metrics"
        )

    query_ids = {row["query_id"] for row in rank1_rows}
    columns = [args.id_key, args.mz_key, args.intensity_key, args.precursor_mz_key, args.precursor_charge_key]
    df = pl.read_parquet(parquet_path, columns=list(dict.fromkeys(columns)))
    df = df.filter(pl.col(args.id_key).is_in(list(query_ids)))
    spectra_by_id = {str(row[args.id_key]): row for row in df.to_dicts()}

    call_args = []
    for row in rank1_rows:
        query_id = row["query_id"]
        spectrum = spectra_by_id.get(query_id)
        if spectrum is None:
            continue
        call_args.append(
            (
                query_id,
                str(row.get("library_peptide") or ""),
                list(np.asarray(spectrum[args.mz_key], dtype=float).flatten()),
                list(np.asarray(spectrum[args.intensity_key], dtype=float).flatten()),
                spectrum.get(args.precursor_mz_key),
                spectrum.get(args.precursor_charge_key),
                args.tolerance_da,
                args.ion_types,
                args.max_ion_charge,
                args.annotation_metric,
            )
        )

    scores: Dict[str, float] = {}
    n_total = len(call_args)
    print(f"Scoring rank-1 annotation coverage for {n_total} queries (metric={args.annotation_metric})...")
    if args.num_workers and args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(_score_rank1_annotation, *call) for call in call_args]
            for i, future in enumerate(as_completed(futures), 1):
                query_id, value = future.result()
                scores[query_id] = value
                if i % 2000 == 0 or i == n_total:
                    print(f"  scored {i}/{n_total} queries")
    else:
        for i, call in enumerate(call_args, 1):
            query_id, value = _score_rank1_annotation(*call)
            scores[query_id] = value
            if i % 2000 == 0 or i == n_total:
                print(f"  scored {i}/{n_total} queries")
    return scores


def main() -> None:
    """Filter a Stage 1 candidates CSV down to the top-K queries by the chosen criterion."""
    args = parse_args()
    if args.top_k is None and args.top_frac is None:
        raise SystemExit("Set either --top-k or --top-frac")

    candidates = pl.read_csv(args.candidates_csv)
    query_ids = list(dict.fromkeys(candidates.select("query_id").to_series().to_list()))
    n_queries = len(query_ids)
    top_k = args.top_k if args.top_k is not None else max(1, round(n_queries * args.top_frac))
    print(f"Loaded {candidates.height} candidate rows covering {n_queries} distinct queries")

    has_model_confidence = "query_spectrum_confidence" in candidates.columns
    if args.metric == "model_confidence" and not has_model_confidence:
        raise SystemExit(
            "candidates CSV has no query_spectrum_confidence column -- rerun Stage 1 with "
            "evaluation.compute_spectrum_confidence=true (default in run_cross_set_annotation_transfer.sh), "
            "or use --metric quality|margin|top1_score instead."
        )

    # embedding_margin_1_2 / embedding_score / query_spectrum_confidence are constant across
    # all rank rows of a given query (see _build_candidate_rows), so a first-row lookup suffices.
    agg_exprs = [
        pl.col("embedding_margin_1_2").first().alias("margin"),
        pl.col("embedding_score").max().alias("top1_score"),
    ]
    if has_model_confidence:
        agg_exprs.append(pl.col("query_spectrum_confidence").first().alias("model_confidence"))
    per_query = candidates.group_by("query_id", maintain_order=True).agg(agg_exprs)
    margin_by_query = dict(zip(per_query["query_id"].to_list(), per_query["margin"].to_list()))
    top1_by_query = dict(zip(per_query["query_id"].to_list(), per_query["top1_score"].to_list()))
    model_confidence_by_query = (
        dict(zip(per_query["query_id"].to_list(), per_query["model_confidence"].to_list())) if has_model_confidence else {}
    )

    needs_intrinsic_quality = args.metric == "quality" or (args.metric == "combination" and not has_model_confidence)
    if needs_intrinsic_quality and not args.combined_parquet:
        raise SystemExit(f"--metric {args.metric} (without query_spectrum_confidence) requires --combined-parquet")

    quality_by_query: Dict[str, float] = {}
    if needs_intrinsic_quality:
        quality_by_query = compute_quality_scores(args.combined_parquet, set(query_ids), args)

    annotation_by_query: Dict[str, float] = {}
    if args.metric == "annotation_coverage":
        if not args.combined_parquet:
            raise SystemExit("--metric annotation_coverage requires --combined-parquet")
        rank1_rows = candidates.filter(pl.col("rank") == 1).select(["query_id", "library_peptide"]).to_dicts()
        annotation_by_query = compute_annotation_coverage_scores(args.combined_parquet, rank1_rows, args)

    # Prefer the model's own confidence over the hand-crafted intrinsic quality metric
    # whenever it's available -- see module docstring for the rationale.
    prefilter_by_query = model_confidence_by_query if has_model_confidence else quality_by_query

    if args.metric == "model_confidence":
        ranked = sorted(query_ids, key=lambda q: model_confidence_by_query.get(q) or -np.inf, reverse=True)
        selected = ranked[:top_k]
    elif args.metric == "margin":
        ranked = sorted(query_ids, key=lambda q: margin_by_query.get(q) or -np.inf, reverse=True)
        selected = ranked[:top_k]
    elif args.metric == "top1_score":
        ranked = sorted(query_ids, key=lambda q: top1_by_query.get(q) or -np.inf, reverse=True)
        selected = ranked[:top_k]
    elif args.metric == "quality":
        ranked = sorted(query_ids, key=lambda q: quality_by_query.get(q, -np.inf), reverse=True)
        selected = ranked[:top_k]
    elif args.metric == "annotation_coverage":
        ranked = sorted(query_ids, key=lambda q: annotation_by_query.get(q, -np.inf), reverse=True)
        selected = ranked[:top_k]
    else:  # combination
        prefilter_n = max(top_k, round(n_queries * args.quality_prefilter_frac))
        by_prefilter = sorted(query_ids, key=lambda q: prefilter_by_query.get(q) or -np.inf, reverse=True)
        survivors = by_prefilter[:prefilter_n]
        selected = sorted(survivors, key=lambda q: margin_by_query.get(q) or -np.inf, reverse=True)[:top_k]

    selected_set = set(selected)
    filtered = candidates.filter(pl.col("query_id").is_in(selected_set))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered.write_csv(output_path)

    summary = {
        "candidates_csv": args.candidates_csv,
        "metric": args.metric,
        "used_model_confidence": has_model_confidence and args.metric in ("model_confidence", "combination"),
        "quality_metric": args.quality_metric if needs_intrinsic_quality else None,
        "quality_prefilter_frac": args.quality_prefilter_frac if args.metric == "combination" else None,
        "annotation_metric": args.annotation_metric if args.metric == "annotation_coverage" else None,
        "n_queries_total": n_queries,
        "n_queries_selected": len(selected),
        "n_candidate_rows_total": candidates.height,
        "n_candidate_rows_selected": filtered.height,
        "output_csv": str(output_path),
    }
    summary_path = output_path.with_suffix(".selection_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {filtered.height} rows for {len(selected)} selected queries to {output_path}")


if __name__ == "__main__":
    main()
