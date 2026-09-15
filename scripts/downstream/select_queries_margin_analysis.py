# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Select queries for the Stage-2 margin-vs-evidence analysis (approaches #2 + #3).

The cross-set retrieval assigns every ACFM query a rank-1 library peptide by
embedding cosine similarity. Before paying the expensive block-B evidence cost we
have to pick *which* queries to score. Two complementary, defensible framings are
combined here into a single rank-1 query set so block B only runs once:

  #2 best-case  -- the top-K queries by embedding margin (top1 - top2 cosine).
                   Answers "when the model is most confident, does the transferred
                   annotation actually fit the query spectrum chemically?".
  #3 stratified -- a sample spread across the full margin range (quantile bins).
                   Lets us plot block-B fit *as a function of* margin and test,
                   with independent chemical evidence, whether margin means
                   anything at all in this (anisotropic) embedding space -- rather
                   than assuming a high margin is good.

Only rank-1 rows are kept (one library candidate per query), so downstream block-B
scoring costs ~1 MCP call per query. Output columns are a superset of the input
candidates plus ``margin``, ``margin_bin`` and ``selection_group`` so the plotting
step can slice by group and bin without recomputing anything.

Usage:
    uv run python scripts/downstream/select_queries_margin_analysis.py \\
        --candidates-csv stage2_477_1/cross_set_topk_candidates.csv \\
        --output-csv stage2_477_1/margin_analysis_queries.csv \\
        --top-k 400 --n-bins 10 --per-bin 40
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for margin-analysis query selection."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-csv", required=True, help="cross_set_topk_candidates.csv from a Stage 1 run")
    parser.add_argument("--output-csv", required=True, help="Rank-1 candidates for the selected union, ready for Stage 2")
    parser.add_argument("--top-k", type=int, default=400, help="#2: number of highest-margin queries to score")
    parser.add_argument("--n-bins", type=int, default=10, help="#3: number of quantile bins across the margin range")
    parser.add_argument("--per-bin", type=int, default=40, help="#3: queries sampled per margin bin")
    parser.add_argument("--margin-key", default="embedding_margin_1_2")
    parser.add_argument("--query-key", default="query_id")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Build the union (top-K by margin) + (stratified across margin) rank-1 query set."""
    args = parse_args()
    rng = np.random.RandomState(args.seed)

    candidates = pl.read_csv(args.candidates_csv)
    if args.margin_key not in candidates.columns:
        raise SystemExit(f"candidates CSV has no '{args.margin_key}' column; columns={candidates.columns}")

    # Rank-1 rows only: one library candidate per query, carrying the (per-query
    # constant) margin. Drop queries with a null/NaN margin -- they can't be ranked
    # or binned and would silently distort both the top-K tail and the strata.
    rank1 = (
        candidates.filter(pl.col("rank") == 1)
        .unique(subset=[args.query_key], keep="first")
        .with_columns(pl.col(args.margin_key).cast(pl.Float64).alias("margin"))
        .filter(pl.col("margin").is_not_nan() & pl.col("margin").is_not_null())
    )
    n_queries = rank1.height
    if n_queries == 0:
        raise SystemExit("No rank-1 rows with a usable margin found.")

    margins = rank1["margin"].to_numpy()
    query_ids = rank1[args.query_key].to_list()

    # #2 best-case: highest-margin queries.
    top_k = min(args.top_k, n_queries)
    top_order = np.argsort(-margins, kind="stable")[:top_k]
    top_margin_ids = {query_ids[i] for i in top_order.tolist()}

    # #3 stratified: quantile-edged bins across the full margin range, sample within each.
    n_bins = max(1, args.n_bins)
    quantiles = np.quantile(margins, np.linspace(0.0, 1.0, n_bins + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    bin_index = np.digitize(margins, quantiles[1:-1], right=False)  # 0..n_bins-1

    stratified_ids: set = set()
    bin_by_query = {query_ids[i]: int(bin_index[i]) for i in range(n_queries)}
    for b in range(n_bins):
        members = [query_ids[i] for i in np.where(bin_index == b)[0].tolist()]
        if not members:
            continue
        take = min(args.per_bin, len(members))
        chosen = rng.choice(np.array(members, dtype=object), size=take, replace=False)
        stratified_ids.update(chosen.tolist())

    selected_ids = top_margin_ids | stratified_ids

    def group_label(qid: str) -> str:
        tags = []
        if qid in top_margin_ids:
            tags.append("top_margin")
        if qid in stratified_ids:
            tags.append("stratified")
        return "+".join(tags)

    selected_frame = (
        rank1.filter(pl.col(args.query_key).is_in(list(selected_ids)))
        .with_columns(
            pl.col(args.query_key)
            .map_elements(lambda q: bin_by_query.get(q, -1), return_dtype=pl.Int64)
            .alias("margin_bin"),
            pl.col(args.query_key)
            .map_elements(group_label, return_dtype=pl.Utf8)
            .alias("selection_group"),
        )
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_frame.write_csv(output_path)

    bin_edges = [float(q) for q in quantiles]
    summary = {
        "candidates_csv": args.candidates_csv,
        "output_csv": str(output_path),
        "n_queries_total": n_queries,
        "n_selected": selected_frame.height,
        "n_top_margin": len(top_margin_ids),
        "n_stratified": len(stratified_ids),
        "n_overlap": len(top_margin_ids & stratified_ids),
        "top_k": top_k,
        "n_bins": n_bins,
        "per_bin": args.per_bin,
        "margin_min": float(margins.min()),
        "margin_max": float(margins.max()),
        "margin_median": float(np.median(margins)),
        "bin_edges": bin_edges,
        "seed": args.seed,
    }
    summary_path = output_path.with_suffix(".selection_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    print(f"Wrote {selected_frame.height} rank-1 rows for {len(selected_ids)} queries to {output_path}")


if __name__ == "__main__":
    main()
