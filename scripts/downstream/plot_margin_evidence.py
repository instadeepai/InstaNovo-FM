# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Publication-style plots for the Stage-2 margin-vs-evidence analysis (#2 + #3).

Reads the block-B enriched candidates produced by
``compute_cross_set_evidence_metrics.py`` (run on the rank-1 union selected by
``select_queries_margin_analysis.py``) and produces, per fit metric:

  #2 best-case distribution -- histogram of the block-B chemical-fit score for the
     highest-margin queries. "When the embedding is most confident, how well does
     the transferred annotation actually explain the query spectrum?"

  #3 fit-vs-margin -- box-per-margin-decile plus a Spearman correlation. This is
     the honest test of whether embedding margin carries any real signal: if fit
     rises with margin, margin is a usable confidence proxy (validated by
     independent chemistry); if it's flat or falls, margin is noise / an
     anisotropy artifact.

Block-B fields are prefixed ``q_obs__lib_theo__`` (query observed vs library-peptide
theoretical). ``annotated_intensity_fraction`` is the primary readout (fraction of
the query's observed intensity explained by the assigned peptide's b/y ions).

Usage:
    uv run python scripts/downstream/plot_margin_evidence.py \\
        --enriched-csv stage2_477_1/blockB_rank1/cross_set_topk_candidates_with_evidence.csv \\
        --output-dir stage2_477_1/plots --label "PXD074343 477-1"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import spearmanr

# Nature-Methods-style pastel, colorblind-safe palette (mirrors
# rescue_task_handover/scripts/publication.py, replicated here because that file
# executes data loading on import and can't be imported cleanly).
PALETTE = ["#4E9AC6", "#6DBF91", "#F5A45D", "#C285C7", "#E87878", "#7CCBC8", "#F5D46A", "#AAAAAA"]
C_BLUE, C_GREEN, C_ORANGE, C_CORAL = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[4]

FIT_METRICS = {
    "q_obs__lib_theo__annotated_intensity_fraction": "Annotated intensity fraction",
    "q_obs__lib_theo__matched_ion_fraction": "Matched b/y-ion fraction",
    "q_obs__lib_theo__hyperscore": "Hyperscore",
}


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
    """Parse CLI arguments for the margin-vs-evidence plots."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--enriched-csv", required=True, help="cross_set_topk_candidates_with_evidence.csv (block B)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="", help="Dataset label used in figure captions")
    parser.add_argument("--margin-key", default="margin")
    parser.add_argument("--bin-key", default="margin_bin")
    parser.add_argument("--group-key", default="selection_group")
    return parser.parse_args()


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_best_case_distribution(
    df: pl.DataFrame, metric: str, metric_label: str, args: argparse.Namespace, out_dir: Path
) -> None:
    """#2: distribution of chemical fit for the highest-margin ("best-case") queries."""
    top = df.filter(pl.col(args.group_key).str.contains("top_margin"))
    values = top[metric].to_numpy().astype(float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return
    median = float(np.median(values))

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.hist(
        values,
        bins=30,
        color=C_BLUE,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.5,
        label=f"top-margin queries (n={values.size})",
    )
    ax.axvline(median, color=C_CORAL, linestyle="--", linewidth=2, label=f"median = {median:.3f}")
    ax.set_xlabel(f"{metric_label}\n(query observed vs transferred-peptide theoretical, block B)")
    ax.set_ylabel("Number of queries")
    ax.legend(loc="upper right")
    label = f" — {args.label}" if args.label else ""
    fig.suptitle(f"Best-case chemical fit of transferred annotations{label}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out_dir, f"bestcase_distribution__{metric.split('__')[-1]}")


def plot_fit_vs_margin(
    df: pl.DataFrame, metric: str, metric_label: str, args: argparse.Namespace, out_dir: Path
) -> None:
    """#3: chemical fit as a function of embedding-margin decile, with Spearman r."""
    margin = df[args.margin_key].to_numpy().astype(float)
    value = df[metric].to_numpy().astype(float)
    bins = df[args.bin_key].to_numpy().astype(int)
    ok = ~np.isnan(margin) & ~np.isnan(value)
    if ok.sum() < 10:
        return
    r, p = spearmanr(margin[ok], value[ok])

    unique_bins = sorted({int(b) for b in bins[ok].tolist()})
    box_data = [value[ok & (bins == b)] for b in unique_bins]
    box_data = [d[~np.isnan(d)] for d in box_data]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    bp = ax.boxplot(
        box_data,
        positions=range(len(unique_bins)),
        widths=0.6,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": C_CORAL, "linewidth": 2},
        whiskerprops={"color": "#555555"},
        capprops={"color": "#555555"},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(C_BLUE)
        patch.set_alpha(0.55)
        patch.set_edgecolor("#33668A")

    # Jittered raw points for honesty about spread and n.
    rng = np.random.RandomState(0)
    for i, d in enumerate(box_data):
        if d.size:
            ax.scatter(
                np.full(d.size, i) + rng.uniform(-0.16, 0.16, d.size),
                d,
                s=9,
                color="#33668A",
                alpha=0.35,
                zorder=3,
                linewidths=0,
            )

    ax.set_xticks(range(len(unique_bins)))
    ax.set_xticklabels([f"D{b + 1}" for b in unique_bins])
    ax.set_xlabel("Embedding-margin decile  (D1 = smallest → D10 = largest)")
    ax.set_ylabel(metric_label)
    ax.margins(x=0.02)

    trend = "rises with" if r > 0 else "falls with"
    ax.text(
        0.03,
        0.96,
        f"Spearman r = {r:+.3f}  (p = {p:.1e}, n = {int(ok.sum())})\nfit {trend} margin",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#EEF4FB", "edgecolor": "#AAAAAA"},
    )
    label = f" — {args.label}" if args.label else ""
    fig.suptitle(f"Does embedding margin predict chemical fit?{label}", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save(fig, out_dir, f"fit_vs_margin__{metric.split('__')[-1]}")


def main() -> None:
    """Generate best-case distribution and fit-vs-margin plots for each fit metric."""
    args = parse_args()
    set_publication_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.read_csv(args.enriched_csv)
    present = [m for m in FIT_METRICS if m in df.columns]
    if not present:
        raise SystemExit(f"No block-B fit metrics found in {args.enriched_csv}. Columns: {df.columns}")

    for metric in present:
        label = FIT_METRICS[metric]
        plot_best_case_distribution(df, metric, label, args, out_dir)
        plot_fit_vs_margin(df, metric, label, args, out_dir)
        print(f"Plotted {metric}")

    print(f"Wrote plots to {out_dir}")


if __name__ == "__main__":
    main()
