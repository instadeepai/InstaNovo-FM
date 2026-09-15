# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Sample a random rank-1 query pool from a Stage-1 candidates CSV.

The pool is scored once with block B (rank 1) and then re-ranked by every candidate
query selector, so all selectors are compared on the *same* queries (a fair
head-to-head; see ``compare_query_selectors.py``). Only rank-1 rows are kept -- one
library candidate per query -- so block-B cost is ~1 MCP call per query.

Usage:
    uv run python scripts/downstream/sample_query_pool.py \\
        --candidates-csv stage2_477_1/cross_set_topk_candidates.csv \\
        --output-csv stage2_477_1/selector_pool.csv --pool-size 3000 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for random pool sampling."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--pool-size", type=int, default=3000)
    parser.add_argument("--query-key", default="query_id")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Write a rank-1 CSV for a uniformly random sample of distinct queries."""
    args = parse_args()
    candidates = pl.read_csv(args.candidates_csv)
    rank1 = candidates.filter(pl.col("rank") == 1).unique(subset=[args.query_key], keep="first")
    n = rank1.height
    pool_size = min(args.pool_size, n)
    sampled = rank1.sample(n=pool_size, seed=args.seed)

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.write_csv(output_path)
    print(f"Sampled {pool_size} of {n} distinct rank-1 queries -> {output_path}")


if __name__ == "__main__":
    main()
