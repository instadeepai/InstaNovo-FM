# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Head-to-head comparison of query selectors by block-B chemical fit.

Given a block-B enriched pool (``compute_cross_set_evidence_metrics.py`` run on the
random pool from ``sample_query_pool.py``), rank the *same* pool by each identification
-free selector and compare the chemical fit of each selector's top-K against a random
baseline. Answers, concretely: "which cheap, spectrum/embedding-only signal picks the
transfers most likely to be chemically real?"

Selectors compared (each ranked high->low, top-K taken):
  - margin              : embedding top1 - top2 cosine (embedding_margin_1_2)
  - top1_score          : raw embedding top1 cosine (embedding_score)
  - model_confidence    : model masked-reconstruction confidence (query_spectrum_confidence)
  - random              : baseline -- whole-pool distribution

Fit = ``q_obs__lib_theo__annotated_intensity_fraction`` (fraction of the query's
observed intensity explained by the rank-1 transferred peptide's b/y ions).

Usage:
    uv run python scripts/compare_query_selectors.py \\
        --enriched-csv stage2_477_1/selector_pool_blockB/cross_set_topk_candidates_with_evidence.csv \\
        --output-dir stage2_477_1/plots --label "PXD074343 477-1" --top-k 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

PALETTE = ["#4E9AC6", "#6DBF91", "#F5A45D", "#C285C7", "#E87878", "#7CCBC8", "#F5D46A", "#AAAAAA"]

# selector label -> (candidate CSV column, higher-is-better)
SELECTORS = {
    "margin": ("embedding_margin_1_2", True),
    "top1_score": ("embedding_score", True),
    "model_confidence": ("query_spectrum_confidence", True),
}
SELECTOR_COLORS = {
    "margin": PALETTE[0],
    "top1_score": PALETTE[1],
    "model_confidence": PALETTE[3],
    "random": PALETTE[7],
}
FIT_METRIC = "q_obs__lib_theo__annotated_intensity_fraction"
FIT_LABEL = "Annotated intensity fraction"


def set_publication_style() -> None:
    """Apply a consistent, publication-ready serif matplotlib style."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino", "Palatino Linotype", "TeX Gyre Pagella", "Book Antiqua", "DejaVu Serif"],
            "font.size": 14,
            "axes.titlesize": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.fontsize": 12,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.5,
        }
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the selector bake-off."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enriched-csv", required=True, help="Block-B enriched pool CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--top-k", type=int, default=300, help="Top-K taken from each selector's ranking")
    parser.add_argument("--query-key", default="query_id")
    return parser.parse_args()


def _rank_top_k(df: pl.DataFrame, column: str, top_k: int) -> np.ndarray:
    """Return the fit values of the top-K queries by ``column`` (descending)."""
    sub = df.filter(pl.col(column).is_not_null() & pl.col(FIT_METRIC).is_not_null())
    sub = sub.sort(column, descending=True).head(top_k)
    return sub[FIT_METRIC].to_numpy().astype(float)


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Rank the pool by each selector and compare top-K chemical fit vs random."""
    args = parse_args()
    set_publication_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(args.enriched_csv)
    if FIT_METRIC not in df.columns:
        raise SystemExit(f"{FIT_METRIC} not in enriched CSV columns: {df.columns}")

    all_fit = df.filter(pl.col(FIT_METRIC).is_not_null())[FIT_METRIC].to_numpy().astype(float)
    random_median = float(np.median(all_fit))

    available = {name: col for name, (col, _) in SELECTORS.items() if col in df.columns}
    top_k = min(args.top_k, df.height)

    box_labels: list[str] = []
    box_data: list[np.ndarray] = []
    box_colors: list[str] = []
    stats: dict[str, dict] = {}

    for name, col in available.items():
        vals = _rank_top_k(df, col, top_k)
        if vals.size == 0:
            continue
        box_labels.append(name)
        box_data.append(vals)
        box_colors.append(SELECTOR_COLORS[name])
        stats[name] = {
            "column": col,
            "n": int(vals.size),
            "median_fit": float(np.median(vals)),
            "mean_fit": float(np.mean(vals)),
            "lift_over_random_median": float(np.median(vals) - random_median),
        }

    # random baseline as an explicit box (whole pool)
    box_labels.append("random")
    box_data.append(all_fit)
    box_colors.append(SELECTOR_COLORS["random"])
    stats["random"] = {
        "column": None,
        "n": int(all_fit.size),
        "median_fit": random_median,
        "mean_fit": float(np.mean(all_fit)),
        "lift_over_random_median": 0.0,
    }

    # --- Plot: top-K fit per selector vs random baseline ---
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    positions = range(len(box_labels))
    bp = ax.boxplot(
        box_data,
        positions=list(positions),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"color":"#E87878", "linewidth":2},
        whiskerprops={"color":"#555555"},
        capprops={"color":"#555555"},
    )
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("#33668A")

    rng = np.random.RandomState(0)
    for i, d in enumerate(box_data):
        if d.size:
            jitter = rng.uniform(-0.16, 0.16, d.size)
            ax.scatter(np.full(d.size, i) + jitter, d, s=7, color="#33668A", alpha=0.25, zorder=3, linewidths=0)

    ax.axhline(random_median, color=SELECTOR_COLORS["random"], linestyle="--", linewidth=1.5, zorder=1)
    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"{lbl}\n(top {top_k})" if lbl != "random" else f"random\n(n={all_fit.size})" for lbl in box_labels])
    ax.set_ylabel(FIT_LABEL)
    ax.set_xlabel("Query selector")
    label = f" — {args.label}" if args.label else ""
    fig.suptitle(f"Which selector picks chemically-valid transfers?{label}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out_dir, "selector_comparison_topk_fit")

    # --- Plot: enrichment curve -- median fit vs K for each selector ---
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.9))
    ks = [k for k in (25, 50, 100, 150, 200, 300, 400, 500, 750, 1000) if k <= df.height]
    for name, col in available.items():
        medians = []
        for k in ks:
            vals = _rank_top_k(df, col, k)
            medians.append(float(np.median(vals)) if vals.size else np.nan)
        ax2.plot(ks, medians, marker="o", markersize=5, linewidth=2, color=SELECTOR_COLORS[name], label=name)
    ax2.axhline(random_median, color=SELECTOR_COLORS["random"], linestyle="--", linewidth=1.5, label="random baseline")
    ax2.set_xlabel("Top-K queries kept (ranked by selector)")
    ax2.set_ylabel(f"Median {FIT_LABEL.lower()}")
    ax2.legend(loc="best")
    fig2.suptitle(f"Selection enrichment vs random{label}", fontsize=13, fontweight="bold")
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig2, out_dir, "selector_enrichment_curve")

    summary = {
        "enriched_csv": args.enriched_csv,
        "top_k": top_k,
        "pool_size": int(all_fit.size),
        "random_median_fit": random_median,
        "selectors": stats,
        "ranking_by_lift": sorted(stats.items(), key=lambda kv: kv[1]["lift_over_random_median"], reverse=True),
    }
    (out_dir / "selector_comparison_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
