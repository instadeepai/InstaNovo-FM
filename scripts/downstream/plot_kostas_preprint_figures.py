# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Kostas preprint figures for confidence-filtering / rescue validation (477-1).

Produces figures 1–5 incrementally:
  1. Box plot: observed cosine by matched-fragment-ion count (>=3 vs <3)
  2. Confidence histogram color-coded by high fragment support
  3. Scatter: confidence vs matched-ion count (high embedding-cosine subset)
  4. Hero UMAP over library (requires embeddings npz)
  5. Refined consecutive-ion example spectra (from top_pairs explain output)

Usage:
    uv run python scripts/downstream/plot_kostas_preprint_figures.py --point 1 \\
        --analysis-table stage2_477_1/confidence_match_analysis/rank1_analysis_table.csv \\
        --output-dir stage2_477_1/kostas_preprint_figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from scripts.downstream.explain_top_pairs import save_all_formats, set_publication_style

# Canonical column names (rank-1 analysis table).
COL_CONF = "query_spectrum_confidence"
COL_EMB = "embedding_score"
COL_S_OBS = "q_obs__lib_obs__cosine_similarity"
COL_MIC = "q_obs__lib_theo__matched_ion_count"
COL_MIF = "q_obs__lib_theo__matched_ion_fraction"
COL_CONSEC_Y = "q_obs__lib_theo__consecutive_y_ions"
COL_CONSEC = "q_obs__lib_theo__consecutive_ion_series"
ION_THRESHOLD = 3
HIGH_EMB_QUANTILE = 0.99  # top 1% embedding cosine for point 3


def _load_table(path: Path) -> pl.DataFrame:
    df = pl.read_csv(path)
    if "rank" in df.columns:
        df = df.filter(pl.col("rank") == 1)
    return df



def _descriptive_stats(x: np.ndarray) -> dict[str, float]:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {}
    return {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "median": float(np.median(x)),
        "q1": float(np.percentile(x, 25)),
        "q3": float(np.percentile(x, 75)),
        "iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's delta for x vs y: P(x>y) - P(x<y). Positive => y tends larger."""
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    greater = sum(1 for a in x for b in y if a > b)
    less = sum(1 for a in x for b in y if a < b)
    return float((greater - less) / (x.size * y.size))


def _bootstrap_median_diff(low: np.ndarray, high: np.ndarray, *, n_boot: int = 2000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    low = low[np.isfinite(low)]
    high = high[np.isfinite(high)]
    if low.size < 2 or high.size < 2:
        return {"median_diff_high_minus_low": float("nan"), "bootstrap_ci_low": float("nan"), "bootstrap_ci_high": float("nan")}
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        lb = rng.choice(low, size=low.size, replace=True)
        hb = rng.choice(high, size=high.size, replace=True)
        diffs[i] = float(np.median(hb) - np.median(lb))
    return {
        "median_diff_high_minus_low": float(np.median(high) - np.median(low)),
        "bootstrap_ci_low": float(np.percentile(diffs, 2.5)),
        "bootstrap_ci_high": float(np.percentile(diffs, 97.5)),
    }


def _format_p(p: float) -> str:
    if not np.isfinite(p):
        return "NA"
    return f"{p:.1e}" if p < 0.001 else f"{p:.4f}"


def point1_boxplot_fragment_ions_vs_cosine(df: pl.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Box plot + two-group statistics: observed cosine for >=3 vs <3 matched fragment ions."""
    sub = df.filter(pl.col(COL_MIC).is_not_null() & pl.col(COL_S_OBS).is_not_null())
    low = sub.filter(pl.col(COL_MIC) < ION_THRESHOLD)[COL_S_OBS].to_numpy()
    high = sub.filter(pl.col(COL_MIC) >= ION_THRESHOLD)[COL_S_OBS].to_numpy()

    u_greater_stat, p_mw_greater = stats.mannwhitneyu(high, low, alternative="greater")
    u_two_stat, p_mw_two = stats.mannwhitneyu(high, low, alternative="two-sided")
    t_stat, p_welch = stats.ttest_ind(high, low, equal_var=False)
    cliffs_d = -_cliffs_delta(low, high)  # positive when high group is larger
    boot = _bootstrap_median_diff(low, high)
    desc_low = _descriptive_stats(low)
    desc_high = _descriptive_stats(high)

    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    bp = ax.boxplot(
        [low, high],
        tick_labels=[f"< {ION_THRESHOLD} ions\n(n={low.size:,})", f">= {ION_THRESHOLD} ions\n(n={high.size:,})"],
        patch_artist=True,
        widths=0.55,
        showfliers=False,
    )
    colors = ["#AAAAAA", "#4E9AC6"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_ylabel("Observed cosine similarity\n(query spectrum vs rank-1 library spectrum)")

    whisker_tops = [float(np.max(w.get_ydata())) for w in bp["whiskers"]]
    y_br = max(whisker_tops) + 0.04
    y_top = y_br + 0.14
    ax.plot([1, 1, 2, 2], [y_br - 0.012, y_br, y_br, y_br - 0.012], color="#333333", lw=1.2, clip_on=False)
    sig = "***" if p_mw_two < 0.001 else ("**" if p_mw_two < 0.01 else ("*" if p_mw_two < 0.05 else "ns"))
    ax.text(1.5, y_br + 0.012, sig, ha="center", va="bottom", fontsize=14, fontweight="bold", clip_on=False)
    ax.set_ylim(0, y_top)

    stats_text = (
        f"Δ median = {boot['median_diff_high_minus_low']:.3f}\n"
        f"95% CI [{boot['bootstrap_ci_low']:.3f}, {boot['bootstrap_ci_high']:.3f}]\n"
        f"Mann–Whitney (2-sided)\n"
        f"p = {_format_p(p_mw_two)}\n"
        f"Cliff's δ = {cliffs_d:.3f}"
    )
    fig.tight_layout(rect=[0, 0, 0.74, 1])
    fig.text(
        0.76, 0.94,
        stats_text,
        transform=fig.transFigure,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#EEF4FB", "edgecolor": "#B8CCE4", "linewidth": 0.8},
    )
    save_all_formats(fig, out_dir / "fragment_ion_threshold_observed_cosine_boxplot")
    plt.close(fig)

    pl.DataFrame([
        {"group": f"< {ION_THRESHOLD} matched fragment ions", **desc_low},
        {"group": f">= {ION_THRESHOLD} matched fragment ions", **desc_high},
    ]).write_csv(out_dir / "point1_group_descriptive_stats.csv")

    summary = {
        "point": 1,
        "comparison": f"observed cosine: >= {ION_THRESHOLD} vs < {ION_THRESHOLD} matched fragment ions",
        "ion_threshold": ION_THRESHOLD,
        "group_low": desc_low,
        "group_high": desc_high,
        "tests": {
            "mann_whitney_u_two_sided": {"statistic": float(u_two_stat), "p_value": float(p_mw_two)},
            "mann_whitney_u_greater_high_vs_low": {"statistic": float(u_greater_stat), "p_value": float(p_mw_greater)},
            "welch_t_test": {"statistic": float(t_stat), "p_value": float(p_welch)},
            "cliffs_delta": float(cliffs_d),
            "median_difference_high_minus_low": boot,
        },
        "y_metric": COL_S_OBS,
        "grouping_metric": COL_MIC,
    }
    (out_dir / "point1_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "point1_statistical_report.txt").write_text(
        f"Two-group comparison: observed cosine (query vs rank-1 library)\n"
        f"Grouping: matched fragment-ion count (query vs transferred peptide)\n"
        f"Threshold: {ION_THRESHOLD} ions\n\n"
        f"Group < {ION_THRESHOLD} (n={desc_low['n']}): "
        f"median={desc_low['median']:.4f}, mean={desc_low['mean']:.4f} ± {desc_low['std']:.4f}, "
        f"IQR=[{desc_low['q1']:.4f}, {desc_low['q3']:.4f}]\n"
        f"Group >= {ION_THRESHOLD} (n={desc_high['n']}): "
        f"median={desc_high['median']:.4f}, mean={desc_high['mean']:.4f} ± {desc_high['std']:.4f}, "
        f"IQR=[{desc_high['q1']:.4f}, {desc_high['q3']:.4f}]\n\n"
        f"Median difference (high − low) = {boot['median_diff_high_minus_low']:.4f}\n"
        f"Bootstrap 95% CI = [{boot['bootstrap_ci_low']:.4f}, {boot['bootstrap_ci_high']:.4f}]\n\n"
        f"Mann–Whitney U (two-sided): U={u_two_stat:.0f}, p={_format_p(p_mw_two)}\n"
        f"Mann–Whitney U (high > low):  U={u_greater_stat:.0f}, p={_format_p(p_mw_greater)}\n"
        f"Welch t-test: t={t_stat:.3f}, p={_format_p(p_welch)}\n"
        f"Cliff's delta = {cliffs_d:.4f}\n",
        encoding="utf-8",
    )
    return summary


def _decile_high_support_fraction(sub: pl.DataFrame) -> list[dict[str, Any]]:
    """Fraction of queries with >= ION_THRESHOLD matched ions, per confidence decile."""
    dec = sub.with_columns(pl.col(COL_CONF).qcut(10, labels=[f"D{i}" for i in range(1, 11)]).alias("decile"))
    dec_stats = []
    for d in [f"D{i}" for i in range(1, 11)]:
        g = dec.filter(pl.col("decile") == d)
        if g.height == 0:
            continue
        frac = float((g[COL_MIC].fill_null(0) >= ION_THRESHOLD).mean())
        conf_lo = float(g[COL_CONF].min())
        conf_hi = float(g[COL_CONF].max())
        dec_stats.append({
            "decile": d, "n": g.height, "frac_high_fragment_support": frac,
            "confidence_range": [conf_lo, conf_hi],
        })
    return dec_stats


def point2_confidence_histogram_colored(df: pl.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Histogram of query confidence; high fragment-support queries highlighted."""
    sub = df.filter(pl.col(COL_CONF).is_not_null())
    conf = sub[COL_CONF].to_numpy()
    high_mask = (sub[COL_MIC].fill_null(0) >= ION_THRESHOLD).to_numpy()
    conf_low = conf[~high_mask]
    conf_high = conf[high_mask]

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(conf.min(), conf.max(), 50)
    ax.hist(
        [conf_low, conf_high],
        bins=bins,
        stacked=True,
        color=["#6B6B6B", "#4E9AC6"],
        edgecolor="white",
        linewidth=0.4,
        label=[
            f"< {ION_THRESHOLD} matched ions (n={conf_low.size:,})",
            f">= {ION_THRESHOLD} matched ions (n={conf_high.size:,})",
        ],
    )
    ax.set_xlabel("Query spectrum confidence")
    ax.set_ylabel("Number of queries")
    ax.legend(loc="upper right", framealpha=0.95)
    fig.tight_layout()
    save_all_formats(fig, out_dir / "query_confidence_histogram_by_fragment_support")
    plt.close(fig)

    dec_stats = _decile_high_support_fraction(sub)

    summary = {
        "point": 2,
        "n_total": int(sub.height),
        "n_high_fragment_support": int(high_mask.sum()),
        "frac_high_support": float(high_mask.mean()),
        "decile_high_support_fraction": dec_stats,
    }
    (out_dir / "point2_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def point2b_decile_fraction_panel(df: pl.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Population-corrected companion panel: fraction of high-fragment-support queries per confidence decile.

    The raw histogram (point 2) is dominated by the overall population size per confidence bin, which
    can visually understate how strongly fragment support concentrates at high confidence. This panel
    removes that population effect by normalising within each confidence decile.
    """
    sub = df.filter(pl.col(COL_CONF).is_not_null())
    dec_stats = _decile_high_support_fraction(sub)
    overall_frac = float((sub[COL_MIC].fill_null(0) >= ION_THRESHOLD).mean())

    deciles = [d["decile"] for d in dec_stats]
    fracs = [d["frac_high_fragment_support"] for d in dec_stats]
    ns = [d["n"] for d in dec_stats]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(deciles, fracs, color="#4E9AC6", edgecolor="white", linewidth=0.6)
    for bar, n in zip(bars, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"n={n:,}",
                ha="center", va="bottom", fontsize=8, color="#444444")
    ax.axhline(overall_frac, color="#333333", linestyle="--", linewidth=1.0,
               label=f"Overall = {overall_frac:.1%}")
    ax.set_xlabel("Query confidence decile (D1 = lowest confidence, D10 = highest)")
    ax.set_ylabel(f"Fraction of queries with\n>= {ION_THRESHOLD} matched fragment ions")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", framealpha=0.95)
    fig.tight_layout()
    save_all_formats(fig, out_dir / "confidence_decile_high_fragment_support_fraction")
    plt.close(fig)

    csv_rows = [
        {
            "decile": d["decile"], "n": d["n"], "frac_high_fragment_support": d["frac_high_fragment_support"],
            "confidence_min": d["confidence_range"][0], "confidence_max": d["confidence_range"][1],
        }
        for d in dec_stats
    ]
    pl.DataFrame(csv_rows).write_csv(out_dir / "point2b_decile_fraction.csv")

    summary = {
        "point": "2b",
        "n_total": int(sub.height),
        "overall_frac_high_support": overall_frac,
        "decile_high_support_fraction": dec_stats,
    }
    (out_dir / "point2b_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def point3_scatter_confidence_vs_ions_high_similarity(df: pl.DataFrame, out_dir: Path) -> dict[str, Any]:
    """Scatter confidence vs matched-ion count for high embedding-cosine queries."""
    thresh = float(df[COL_EMB].quantile(HIGH_EMB_QUANTILE))
    sub = df.filter(pl.col(COL_EMB) >= thresh).filter(
        pl.col(COL_CONF).is_not_null() & pl.col(COL_MIC).is_not_null()
    )
    x = sub[COL_CONF].to_numpy()
    y = sub[COL_MIC].to_numpy()
    rho, p = stats.spearmanr(x, y) if x.size > 10 else (float("nan"), float("nan"))

    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.scatter(x, y, s=14, alpha=0.55, color="#4E9AC6", edgecolors="none")
    ax.set_xlabel("Query spectrum confidence")
    ax.set_ylabel("Matched fragment-ion count\n(query vs transferred peptide, rank-1)")
    ax.text(0.03, 0.97, f"embedding cosine >= {thresh:.4f}\n(n={x.size:,})\nSpearman rho = {rho:.3f}",
            transform=ax.transAxes, va="top")
    fig.tight_layout()
    save_all_formats(fig, out_dir / "confidence_vs_matched_ion_count_high_embedding_cosine")
    plt.close(fig)

    summary = {
        "point": 3,
        "embedding_cosine_threshold": thresh,
        "embedding_quantile": HIGH_EMB_QUANTILE,
        "n_queries": int(x.size),
        "spearman_rho": float(rho) if np.isfinite(rho) else None,
        "spearman_p": float(p) if np.isfinite(p) else None,
    }
    (out_dir / "point3_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _similarity_top_query_ids(df: pl.DataFrame, ranks: list[int]) -> tuple[list[str], list[str]]:
    """Return query IDs and labels (e.g. Q3) for 1-based ranks in embedding-similarity order."""
    top_ids = df.sort(COL_EMB, descending=True).head(max(ranks))["query_id"].to_list()
    ids, labels = [], []
    for rank in ranks:
        if 1 <= rank <= len(top_ids):
            ids.append(top_ids[rank - 1])
            labels.append(f"Q{rank}")
    return ids, labels


def point4_hero_umap(
    df: pl.DataFrame,
    out_dir: Path,
    *,
    embeddings_h5: Path | None,
    hero_query_ids: list[str] | None = None,
    hero_labels: list[str] | None = None,
    n_hero: int = 5,
    output_stem: str = "hero_query_umap_over_library",
    summary_name: str | None = None,
) -> dict[str, Any]:
    """Hero-query UMAP: query stars over library embedding map."""
    try:
        import umap
        import h5py
    except ImportError as e:
        return {"point": 4, "error": f"Missing dependency: {e}"}

    if embeddings_h5 is None or not embeddings_h5.is_file():
        return {"point": 4, "error": f"embeddings file not found: {embeddings_h5}"}

    with h5py.File(embeddings_h5, "r") as f:
        emb = np.asarray(f["embeddings"], dtype=np.float32)
        md = f["metadata"]
        ids = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in md["usi"][:]]
        tiers = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in md["search_tier"][:]]
        peptides = [x.decode() if isinstance(x, (bytes, np.bytes_)) else str(x) for x in md["sequence"][:]]

    id_to_idx = {uid: i for i, uid in enumerate(ids)}
    lib_idx = [i for i, t in enumerate(tiers) if t == "lcfm"]
    e_lib = emb[lib_idx]
    lib_peptides = [peptides[i] for i in lib_idx]

    if hero_query_ids is None:
        # Default: similarity top-5 query ids from rank-1 table
        hero_query_ids = (
            df.sort(COL_EMB, descending=True)
            .head(n_hero)["query_id"]
            .to_list()
        )
    hero_rows: list[tuple[int, str, str]] = []
    for qid in hero_query_ids:
        if qid not in id_to_idx:
            continue
        pep = df.filter(pl.col("query_id") == qid)["library_peptide"][0]
        hero_rows.append((id_to_idx[qid], str(pep), qid))
    if not hero_rows:
        return {"point": 4, "error": "no hero queries found in embeddings file"}

    hero_emb = np.stack([emb[i] for i, _, _ in hero_rows])
    combined = np.concatenate([e_lib, hero_emb], axis=0)
    coords = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42).fit_transform(combined)
    lib_xy, q_xy = coords[: len(e_lib)], coords[len(e_lib) :]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(lib_xy[:, 0], lib_xy[:, 1], s=16, c="#AAAAAA", alpha=0.45, label=f"Library (n={len(e_lib)})")
    cmap = plt.get_cmap("tab10", len(hero_rows))
    for i, ((_, pep, _), xy) in enumerate(zip(hero_rows, q_xy)):
        sib = [j for j, lp in enumerate(lib_peptides) if lp == pep]
        if sib:
            ax.scatter(lib_xy[sib, 0], lib_xy[sib, 1], s=80, c=[cmap(i)], edgecolors="k", linewidths=0.4, zorder=3)
        label_prefix = hero_labels[i] if hero_labels and i < len(hero_labels) else f"Q{i + 1}"
        star_size = 360 if len(hero_rows) == 1 else 280
        ax.scatter(
            xy[0], xy[1], s=star_size, marker="*", c=[cmap(i)], edgecolors="k", linewidths=1.1, zorder=5,
            label=f"{label_prefix}: {pep[:20]}",
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    save_all_formats(fig, out_dir / output_stem)
    plt.close(fig)

    summary = {
        "point": 4,
        "output_stem": output_stem,
        "n_library": len(e_lib),
        "n_hero": len(hero_rows),
        "hero_query_ids": [qid for _, _, qid in hero_rows],
        "hero_peptides": [p for _, p, _ in hero_rows],
        "hero_labels": hero_labels or [f"Q{i + 1}" for i in range(len(hero_rows))],
    }
    summary_path = out_dir / (summary_name or "point4_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def point5_refined_examples(df: pl.DataFrame, out_dir: Path, *, n_examples: int = 5) -> dict[str, Any]:
    """Select hero cases with long consecutive-ion ladders and export query IDs for explain_top_pairs."""
    # Score: consecutive y ions, then observed cosine, then embedding
    ranked = (
        df.filter(pl.col(COL_CONSEC_Y).is_not_null())
        .sort([COL_CONSEC_Y, COL_S_OBS, COL_EMB], descending=[True, True, True])
        .head(n_examples)
    )
    examples = ranked.select([
        "query_id", "library_id", "library_peptide", COL_CONF, COL_EMB, COL_S_OBS,
        COL_MIC, COL_CONSEC_Y, COL_CONSEC, "q_obs__lib_theo__residue_evidence_coverage",
    ]).to_dicts()

    # Also include similarity top-5 for contrast
    sim_top = df.sort(COL_EMB, descending=True).head(5).select([
        "query_id", "library_id", "library_peptide", COL_CONF, COL_EMB, COL_S_OBS,
        COL_MIC, COL_CONSEC_Y, COL_CONSEC, "q_obs__lib_theo__residue_evidence_coverage",
    ]).to_dicts()

    examples_df = ranked.select([
        "query_id", "library_id", "library_peptide", COL_CONF, COL_EMB, COL_S_OBS,
        COL_MIC, COL_CONSEC_Y, COL_CONSEC, "q_obs__lib_theo__residue_evidence_coverage",
    ]).with_columns(pl.lit("consecutive_ions").alias("selection_group"))
    examples_df.write_csv(out_dir / "point5_consecutive_ion_candidates.csv")
    (out_dir / "point5_selection.json").write_text(json.dumps({
        "point": 5,
        "consecutive_ion_heroes": examples,
        "similarity_top5": sim_top,
        "note": "Run explain_top_pairs.py on a filtered candidates CSV built from these query_ids for mirror plots.",
    }, indent=2, default=str))

    return {"point": 5, "n_consecutive_heroes": len(examples), "n_similarity_top5": len(sim_top)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--point", type=str, choices=["1", "2", "2b", "3", "4", "5"], required=True)
    ap.add_argument("--analysis-table", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--embeddings-h5", type=Path, default=None, help="For point 4: embeddings.h5 from Stage-1")
    ap.add_argument("--hero-ranks", type=str, default=None, help="For point 4: comma-separated 1-based similarity ranks, e.g. 3")
    ap.add_argument("--hero-query-ids", type=str, default=None, help="For point 4: comma-separated query USIs")
    ap.add_argument("--umap-output-stem", type=str, default=None, help="For point 4: output filename stem (no extension)")
    ap.add_argument("--umap-summary-name", type=str, default=None, help="For point 4: summary JSON filename")
    args = ap.parse_args()

    set_publication_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    df = _load_table(args.analysis_table)

    def _run_point4() -> dict[str, Any]:
        hero_ids: list[str] | None = None
        hero_labels: list[str] | None = None
        output_stem = "hero_query_umap_over_library"
        summary_name = "point4_summary.json"
        if args.hero_ranks:
            ranks = [int(x.strip()) for x in args.hero_ranks.split(",") if x.strip()]
            hero_ids, hero_labels = _similarity_top_query_ids(df, ranks)
            if len(ranks) == 1:
                output_stem = args.umap_output_stem or f"hero_query_umap_q{ranks[0]}"
                summary_name = args.umap_summary_name or f"point4_q{ranks[0]}_summary.json"
        elif args.hero_query_ids:
            hero_ids = [x.strip() for x in args.hero_query_ids.split(",") if x.strip()]
        if args.umap_output_stem:
            output_stem = args.umap_output_stem
        if args.umap_summary_name:
            summary_name = args.umap_summary_name
        return point4_hero_umap(
            df,
            args.output_dir,
            embeddings_h5=args.embeddings_h5,
            hero_query_ids=hero_ids,
            hero_labels=hero_labels,
            output_stem=output_stem,
            summary_name=summary_name,
        )

    handlers = {
        "1": lambda: point1_boxplot_fragment_ions_vs_cosine(df, args.output_dir),
        "2": lambda: point2_confidence_histogram_colored(df, args.output_dir),
        "2b": lambda: point2b_decile_fraction_panel(df, args.output_dir),
        "3": lambda: point3_scatter_confidence_vs_ions_high_similarity(df, args.output_dir),
        "4": _run_point4,
        "5": lambda: point5_refined_examples(df, args.output_dir),
    }
    s = handlers[args.point]()
    print(json.dumps(s, indent=2, default=str))


if __name__ == "__main__":
    main()
