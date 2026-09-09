#!/usr/bin/env python3
# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Generate publication-style spectral rescue plots from saved artifacts."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.metrics import average_precision_score, roc_auc_score


PAIR_ORDER = [
    "leqgqa_mox",
    "kfvevm_mox",
    "pxd047134_mvnhek_mox",
    "pxd047134_tnlvtk_mox",
    "pxd047134_elmntk_mox",
    "pxd047134_weelvk_mox",
    "pxd047134_gtdvak_mox",
    "pxd047134_mdfslr_mox",
    "pxd047134_maalek_mox",
    "pxd047134_ftntmr_mox",
]


def load_shared_colors() -> dict[str, str]:
    """Load the shared metadata palette and map it to rescue plot semantics."""
    path = Path("/home/hjisaac/Downloads/metadata_colors.json")
    if path.is_file():
        payload = json.loads(path.read_text())
        palette = payload.get("palette", [])
        pastel = payload.get("pastel", [])
        return {
            "same_sequence": palette[0] if len(palette) > 0 else "#4E9AC6",
            "success": palette[1] if len(palette) > 1 else "#6DBF91",
            "secondary": palette[2] if len(palette) > 2 else "#F5A45D",
            "failure": palette[4] if len(palette) > 4 else "#E87878",
            "unmatched": palette[7] if len(palette) > 7 else "#AAAAAA",
            "light_background": pastel[0] if len(pastel) > 0 else "#EEF4FB",
            "dark_text": "#1A3F60",
        }
    return {
        "same_sequence": "#4E9AC6",
        "success": "#6DBF91",
        "secondary": "#F5A45D",
        "failure": "#E87878",
        "unmatched": "#AAAAAA",
        "light_background": "#EEF4FB",
        "dark_text": "#1A3F60",
    }


def set_publication_style() -> None:
    """Apply the publication.py style used by prior paper figures."""
    sns.set_theme(style="ticks")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Palatino",
                "Palatino Linotype",
                "TeX Gyre Pagella",
                "Book Antiqua",
                "URW Palladio L",
                "DejaVu Serif",
            ],
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 300,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.fontsize": 13,
            "legend.frameon": False,
            "legend.columnspacing": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.5,
        }
    )


def save_all_formats(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    paths: list[Path] = []
    for ext in ("png", "svg", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    return paths


def pair_label(pair_id: str) -> str:
    return pair_id.replace("pxd047134_", "").replace("_mox", "")


def wrap_peptide_label(sequence: str, width: int = 16) -> str:
    return "\n".join(textwrap.wrap(str(sequence), width=width, break_long_words=True))


def tokenize_peptide(sequence: str) -> list[str]:
    return re.findall(r"\[[^\]]*\]|[A-Z]", str(sequence))


def canonicalize_peptide(sequence: str) -> str:
    return "".join("L" if token == "I" else token for token in tokenize_peptide(sequence))


def levenshtein_distance(left: str, right: str) -> int:
    left_tokens = tokenize_peptide(canonicalize_peptide(left))
    right_tokens = tokenize_peptide(canonicalize_peptide(right))
    if left_tokens == right_tokens:
        return 0
    if not left_tokens:
        return len(right_tokens)
    if not right_tokens:
        return len(left_tokens)
    previous = list(range(len(right_tokens) + 1))
    for i, left_token in enumerate(left_tokens, start=1):
        current = [i] + [0] * len(right_tokens)
        for j, right_token in enumerate(right_tokens, start=1):
            cost = 0 if left_token == right_token else 1
            current[j] = min(current[j - 1] + 1, previous[j] + 1, previous[j - 1] + cost)
        previous = current
    return previous[-1]


def load_pair_query_metrics(task_dir: Path) -> pd.DataFrame:
    pair_metadata = pd.read_csv(task_dir / "rescue_pair_summary.csv").set_index("pair_id")
    rows = []
    for pair_id in PAIR_ORDER:
        path = task_dir / "pairs" / pair_id / "query_metrics.csv"
        df = pd.read_csv(path)
        df["pair_id"] = pair_id
        df["pair_label"] = str(pair_metadata.loc[pair_id, "base_sequence"])
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def load_base_match_unmatch_scores(task_dir: Path) -> pd.DataFrame:
    frames = []
    for pair_id in PAIR_ORDER:
        pair_dir = task_dir / "pairs" / pair_id
        payload = np.load(pair_dir / "rescue_similarity_matrix.npz", allow_pickle=True)
        similarities = payload["similarity_matrix"]
        query_roles = np.asarray(payload["query_roles"], dtype=object)
        library_roles = np.asarray(payload["library_roles"], dtype=object)

        base_rows = np.where(query_roles == "reference_query")[0]
        positive_cols = np.where(library_roles == "positive_library")[0]
        negative_cols = np.where(library_roles == "negative_library")[0]

        frames.append(
            pd.DataFrame(
                {
                    "pair_id": pair_id,
                    "pair_label": pair_label(pair_id),
                    "class": "same_sequence",
                    "score": similarities[np.ix_(base_rows, positive_cols)].ravel(),
                }
            )
        )
        frames.append(
            pd.DataFrame(
                {
                    "pair_id": pair_id,
                    "pair_label": pair_label(pair_id),
                    "class": "unmatched",
                    "score": similarities[np.ix_(base_rows, negative_cols)].ravel(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def load_query_distance_similarity(task_dir: Path) -> pd.DataFrame:
    frames = []
    distance_cache: dict[tuple[str, str], int] = {}
    for pair_id in PAIR_ORDER:
        pair_dir = task_dir / "pairs" / pair_id
        payload = np.load(pair_dir / "rescue_similarity_matrix.npz", allow_pickle=True)
        similarities = payload["similarity_matrix"]
        query_indices = np.asarray(payload["query_indices"], dtype=np.int64)
        query_roles = np.asarray(payload["query_roles"], dtype=object)
        library_indices = np.asarray(payload["library_indices"], dtype=np.int64)
        library_roles = np.asarray(payload["library_roles"], dtype=object)
        selection = pd.read_csv(pair_dir / "rescue_selection.csv")
        rows = selection.set_index("row_index")

        query_rows = np.where(query_roles == "reference_query")[0]
        library_cols = np.arange(len(library_indices))
        records = []
        for query_row in query_rows:
            query_index = int(query_indices[query_row])
            query_backbone = str(rows.loc[query_index, "backbone"])
            for library_col in library_cols:
                library_index = int(library_indices[library_col])
                library_role = str(library_roles[library_col])
                library_backbone = str(rows.loc[library_index, "backbone"])
                key = (query_backbone, library_backbone)
                if key not in distance_cache:
                    distance_cache[key] = levenshtein_distance(query_backbone, library_backbone)
                records.append(
                    {
                        "pair_id": pair_id,
                        "pair_label": pair_label(pair_id),
                        "query_index": query_index,
                        "library_index": library_index,
                        "library_role": library_role,
                        "sequence_distance": distance_cache[key],
                        "similarity": float(similarities[query_row, library_col]),
                        "class": "same_sequence" if library_role == "positive_library" else "unmatched",
                    }
                )
        frames.append(pd.DataFrame(records))
    return pd.concat(frames, ignore_index=True)


def write_metric_summary(metric_df: pd.DataFrame, out_dir: Path) -> None:
    base = metric_df[metric_df["query_role"] == "reference_query"].copy()
    ks = [1, 5, 10, 20, 50, 100]
    rows = []
    for k in ks:
        col = f"recall@{k}"
        rows.append(
            {
                "k": k,
                "mean_recall": float(base[col].mean()),
                "std_recall": float(base[col].std(ddof=0)),
                "n_queries": int(base[col].notna().sum()),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "query_duplicate_retrieval_recall_curve_summary.csv", index=False)

    pair_summary = (
        base.groupby(["pair_id", "pair_label"], sort=False)
        .agg(
            recall_at_1=("recall@1", "mean"),
            recall_at_10=("recall@10", "mean"),
            mean_margin=("margin", "mean"),
            median_margin=("margin", "median"),
            n_queries=("margin", "size"),
        )
        .reset_index()
    )
    pair_summary.to_csv(out_dir / "query_duplicate_retrieval_pair_summary.csv", index=False)

    margin_summary = pd.DataFrame(
        {
            "metric": ["mean", "median", "min", "max", "success_rate_margin_gt_0", "n_queries"],
            "value": [
                float(base["margin"].mean()),
                float(base["margin"].median()),
                float(base["margin"].min()),
                float(base["margin"].max()),
                float((base["margin"] > 0).mean()),
                int(len(base)),
            ],
        }
    )
    margin_summary.to_csv(out_dir / "query_duplicate_retrieval_margin_summary.csv", index=False)


def plot_pair_recall(pair_summary: pd.DataFrame, out_dir: Path) -> None:
    colors = load_shared_colors()
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    y = np.arange(len(pair_summary))
    ax.barh(y, pair_summary["recall_at_1"], color=colors["same_sequence"], height=0.68)
    ax.axvline(pair_summary["recall_at_1"].mean(), color=colors["dark_text"], linestyle="--", linewidth=1.5)
    ax.set_xlim(0.86, 1.03)
    ax.set_xlabel("Query Recall@1")
    ax.set_ylabel("Query peptide")
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_peptide_label(label) for label in pair_summary["pair_label"]])
    ax.tick_params(axis="y", labelsize=8)
    ax.invert_yaxis()
    fig.subplots_adjust(left=0.36, right=0.98, top=0.97, bottom=0.12)
    save_all_formats(fig, out_dir, "query_duplicate_retrieval_recall_at_1_by_query_peptide")
    plt.close(fig)


def plot_pair_margin(pair_summary: pd.DataFrame, out_dir: Path) -> None:
    colors = load_shared_colors()
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    y = np.arange(len(pair_summary))
    ax.barh(y, pair_summary["mean_margin"], color=colors["success"], height=0.68)
    ax.axvline(0, color=colors["failure"], linestyle="--", linewidth=1.2)
    ax.set_xlim(0, float(pair_summary["mean_margin"].max()) * 1.22)
    ax.set_xlabel("Mean retrieval margin")
    ax.set_ylabel("Query peptide")
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_peptide_label(label) for label in pair_summary["pair_label"]])
    ax.tick_params(axis="y", labelsize=8)
    ax.invert_yaxis()
    fig.subplots_adjust(left=0.36, right=0.98, top=0.97, bottom=0.12)
    save_all_formats(fig, out_dir, "query_duplicate_retrieval_margin_by_query_peptide")
    plt.close(fig)


def plot_margin_cdf(metric_df: pd.DataFrame, out_dir: Path) -> None:
    colors = load_shared_colors()
    base = metric_df[metric_df["query_role"] == "reference_query"].copy()
    margins = base["margin"].to_numpy(dtype=float)
    sorted_margins = np.sort(margins)
    cdf = np.arange(1, len(sorted_margins) + 1) / len(sorted_margins)
    success_rate = float((margins > 0).mean() * 100.0)

    fig, ax1 = plt.subplots(figsize=(7.5, 5.0))
    ax1.axvline(0, color=colors["failure"], linestyle="--", linewidth=1.4)
    counts, bins, patches = ax1.hist(margins, bins=28, alpha=0.74, edgecolor="white", linewidth=0.4)
    for patch, left in zip(patches, bins[:-1], strict=False):
        patch.set_facecolor(colors["success"] if left >= 0 else colors["failure"])
    ax1.set_xlabel("Retrieval margin")
    ax1.set_ylabel("Query count")

    ax2 = ax1.twinx()
    ax2.plot(sorted_margins, cdf, color=colors["same_sequence"], linewidth=2.4)
    ax2.set_ylabel("Cumulative probability")
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(False)

    ax1.text(
        0.04,
        0.92,
        f"{success_rate:.1f}% margin > 0",
        transform=ax1.transAxes,
        color=colors["dark_text"],
        bbox={"facecolor": colors["light_background"], "edgecolor": "none", "boxstyle": "round,pad=0.35"},
    )
    fig.tight_layout()
    save_all_formats(fig, out_dir, "query_duplicate_retrieval_margin_distribution_and_cdf")
    plt.close(fig)


def plot_match_unmatch_distribution(scores: pd.DataFrame, out_dir: Path) -> None:
    colors = load_shared_colors()
    match = scores[scores["class"] == "same_sequence"]["score"].to_numpy(dtype=float)
    unmatch = scores[scores["class"] == "unmatched"]["score"].to_numpy(dtype=float)
    y_true = np.r_[np.ones(len(match)), np.zeros(len(unmatch))]
    y_score = np.r_[match, unmatch]
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))

    summary_rows = [
        {"class": name, "n": len(values), "mean": float(np.mean(values)), "median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "p99": float(np.percentile(values, 99))}
        for name, values in (("same_sequence", match), ("unmatched", unmatch))
    ]
    summary_rows.append({"class": "separation", "n": len(y_true), "mean": auroc, "median": auprc, "p95": np.nan, "p99": np.nan})
    pd.DataFrame(summary_rows).to_csv(out_dir / "query_match_vs_unmatched_cosine_distribution_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    sns.kdeplot(unmatch, ax=ax, fill=True, color=colors["unmatched"], alpha=0.35, linewidth=1.6, label=f"Unmatched (n={len(unmatch):,})")
    sns.kdeplot(match, ax=ax, fill=True, color=colors["same_sequence"], alpha=0.35, linewidth=1.6, label=f"Same sequence (n={len(match):,})")
    ax.axvline(np.median(unmatch), color=colors["unmatched"], linestyle="--", linewidth=1.2)
    ax.axvline(np.median(match), color=colors["same_sequence"], linestyle="-", linewidth=1.2)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.legend(loc="upper left")
    ax.text(
        0.05,
        0.18,
        f"AUROC = {auroc:.3f}\nAUPRC = {auprc:.3f}",
        transform=ax.transAxes,
        color=colors["dark_text"],
        bbox={"facecolor": colors["light_background"], "edgecolor": "none", "boxstyle": "round,pad=0.35"},
    )
    fig.tight_layout()
    save_all_formats(fig, out_dir, "query_match_vs_unmatched_cosine_similarity_distribution")
    plt.close(fig)


def plot_threshold_risk(scores: pd.DataFrame, out_dir: Path) -> None:
    colors = load_shared_colors()
    match = scores[scores["class"] == "same_sequence"]["score"].to_numpy(dtype=float)
    unmatch = scores[scores["class"] == "unmatched"]["score"].to_numpy(dtype=float)
    thresholds = np.linspace(0.90, 0.995, 40)
    rows = [
        {
            "threshold": float(threshold),
            "same_sequence_above_threshold": float(np.mean(match >= threshold)),
            "unmatched_above_threshold": float(np.mean(unmatch >= threshold)),
        }
        for threshold in thresholds
    ]
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "query_similarity_threshold_false_rescue_risk_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(df["threshold"], df["same_sequence_above_threshold"], color=colors["same_sequence"], linewidth=2.4, label="Same sequence")
    ax.plot(df["threshold"], df["unmatched_above_threshold"], color=colors["failure"], linewidth=2.4, label="Unmatched")
    ax.set_xlabel("Cosine similarity threshold")
    ax.set_ylabel("Fraction above threshold")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save_all_formats(fig, out_dir, "query_similarity_threshold_false_rescue_risk")
    plt.close(fig)


def plot_distance_similarity_hexbin(distance_scores: pd.DataFrame, out_dir: Path) -> None:
    colors = load_shared_colors()
    distance_scores.to_csv(out_dir / "query_sequence_distance_vs_similarity_values.csv", index=False)
    summary = (
        distance_scores.groupby(["class", "sequence_distance"], as_index=False)
        .agg(n=("similarity", "size"), mean_similarity=("similarity", "mean"), median_similarity=("similarity", "median"))
        .sort_values(["class", "sequence_distance"])
    )
    summary.to_csv(out_dir / "query_sequence_distance_vs_similarity_hexbin_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    hb = ax.hexbin(
        distance_scores["sequence_distance"],
        distance_scores["similarity"],
        gridsize=(28, 32),
        mincnt=1,
        cmap="Blues",
        linewidths=0.0,
    )
    ax.axvline(0, color=colors["success"], linestyle="--", linewidth=1.3)
    ax.set_xlabel("Sequence distance to query")
    ax.set_ylabel("Cosine similarity")
    ax.set_xlim(-0.5, float(distance_scores["sequence_distance"].max()) + 0.5)
    ax.set_ylim(max(0.80, float(distance_scores["similarity"].min()) - 0.01), 1.005)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Query-library pairs")
    fig.tight_layout()
    save_all_formats(fig, out_dir, "query_sequence_distance_vs_similarity_hexbin")
    plt.close(fig)


def plot_retrieval_curve(metric_df: pd.DataFrame, out_dir: Path) -> None:
    colors = load_shared_colors()
    base = metric_df[metric_df["query_role"] == "reference_query"].copy()
    ks = [1, 5, 10, 20, 50, 100]
    summary = []
    for k in ks:
        col = f"recall@{k}"
        summary.append({"k": k, "mean": float(base[col].mean()), "std": float(base[col].std(ddof=0))})
    df = pd.DataFrame(summary)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(df["k"], df["mean"], marker="o", color=colors["same_sequence"], linewidth=2.4)
    ax.fill_between(df["k"], df["mean"] - df["std"], df["mean"] + df["std"], color=colors["same_sequence"], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(0.88, 1.01)
    ax.set_xlabel("k")
    ax.set_ylabel("Query Recall@k")
    fig.tight_layout()
    save_all_formats(fig, out_dir, "query_duplicate_retrieval_recall_curve")
    plt.close(fig)


def _scale_coords(coords: np.ndarray) -> np.ndarray:
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    scale = max(x_max - x_min, y_max - y_min, 1e-6)
    scaled = np.zeros_like(coords)
    scaled[:, 0] = (coords[:, 0] - x_min) / scale * 10.0
    scaled[:, 1] = (coords[:, 1] - y_min) / scale * 10.0
    return scaled


def plot_best_query_rescue_region(metric_df: pd.DataFrame, task_dir: Path, out_dir: Path) -> None:
    colors = load_shared_colors()
    base_metrics = metric_df[metric_df["query_role"] == "reference_query"].copy()
    pair_summary = (
        base_metrics.groupby(["pair_id", "pair_label"], as_index=False)
        .agg(mean_margin=("margin", "mean"), recall_at_1=("recall@1", "mean"))
        .sort_values(["mean_margin", "recall_at_1"], ascending=False)
    )
    best_pair_id = str(pair_summary.iloc[0]["pair_id"])
    best_pair_label = str(pair_summary.iloc[0]["pair_label"])
    best_pair_dir = task_dir / "pairs" / best_pair_id

    pair_metrics = pd.read_csv(best_pair_dir / "query_metrics.csv")
    query_rows = pair_metrics[pair_metrics["query_role"] == "reference_query"].copy()
    best_query = query_rows.sort_values("margin", ascending=False).iloc[0]
    hero_ordinal = int(best_query["query_ordinal"])

    payload = np.load(best_pair_dir / "rescue_similarity_matrix.npz", allow_pickle=True)
    similarities = payload["similarity_matrix"].astype(np.float32)
    query_indices = np.asarray(payload["query_indices"], dtype=np.int64)
    library_indices = np.asarray(payload["library_indices"], dtype=np.int64)
    library_roles = np.asarray(payload["library_roles"], dtype=object)
    selection = pd.read_csv(best_pair_dir / "rescue_selection.csv").set_index("row_index")

    positive_cols = np.where(library_roles == "positive_library")[0].tolist()
    negative_cols = np.where(library_roles == "negative_library")[0].tolist()
    rng = np.random.RandomState(42)
    if len(negative_cols) > 150:
        negative_cols = rng.choice(negative_cols, size=150, replace=False).tolist()

    selected_cols = positive_cols + negative_cols
    profile_matrix = similarities[:, selected_cols].T
    n_neighbors = min(15, profile_matrix.shape[0] - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, random_state=42)
    coords = _scale_coords(reducer.fit_transform(profile_matrix))
    query_coord = _scale_coords(reducer.transform(similarities[hero_ordinal, selected_cols].reshape(1, -1)))[0]

    positive_points = coords[: len(positive_cols)]
    negative_points = coords[len(positive_cols) :]
    all_distances = np.sqrt(np.sum((coords - query_coord) ** 2, axis=1))
    positive_distances = all_distances[: len(positive_cols)]
    zone1_radius = 1.5
    zone2_radius = 2.5
    zone1_recall = float(np.mean(positive_distances <= zone1_radius) * 100.0)
    zone2_recall = float(np.mean(positive_distances <= zone2_radius) * 100.0)

    rows = []
    for local_col, point, distance in zip(selected_cols, coords, all_distances, strict=False):
        library_index = int(library_indices[local_col])
        rows.append(
            {
                "pair_id": best_pair_id,
                "query_peptide": best_pair_label,
                "query_index": int(query_indices[hero_ordinal]),
                "library_index": library_index,
                "library_role": str(library_roles[local_col]),
                "library_sequence": str(selection.loc[library_index, "sequence"]),
                "umap_x": float(point[0]),
                "umap_y": float(point[1]),
                "distance_to_query": float(distance),
                "similarity_to_query": float(similarities[hero_ordinal, local_col]),
            }
        )
    region_values = pd.DataFrame(rows)
    region_values.to_csv(out_dir / "query_best_peptide_rescue_region_values.csv", index=False)
    pd.DataFrame(
        [
            {
                "pair_id": best_pair_id,
                "query_peptide": best_pair_label,
                "query_ordinal": hero_ordinal,
                "query_index": int(query_indices[hero_ordinal]),
                "query_margin": float(best_query["margin"]),
                "query_best_positive_rank": int(best_query["best_positive_rank"]),
                "zone1_radius": zone1_radius,
                "zone1_recall_percent": zone1_recall,
                "zone2_radius": zone2_radius,
                "zone2_recall_percent": zone2_recall,
                "num_positive_library": len(positive_cols),
                "num_unmatched_sampled": len(negative_cols),
            }
        ]
    ).to_csv(out_dir / "query_best_peptide_rescue_region_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.scatter(
        negative_points[:, 0],
        negative_points[:, 1],
        s=26,
        color=colors["unmatched"],
        alpha=0.22,
        linewidths=0,
        label="Unmatched",
    )
    ax.scatter(
        positive_points[:, 0],
        positive_points[:, 1],
        s=42,
        color=colors["same_sequence"],
        alpha=0.72,
        edgecolors="white",
        linewidths=0.35,
        label="Same sequence",
    )
    ax.scatter(
        [query_coord[0]],
        [query_coord[1]],
        s=190,
        marker="*",
        color=colors["secondary"],
        edgecolors="black",
        linewidths=0.8,
        label="Query",
        zorder=5,
    )
    for radius, linestyle, color, label in (
        (zone1_radius, ":", colors["failure"], f"Region 1 ({zone1_recall:.1f}%)"),
        (zone2_radius, "--", colors["same_sequence"], f"Region 2 ({zone2_recall:.1f}%)"),
    ):
        circle = plt.Circle(query_coord, radius, fill=False, linestyle=linestyle, linewidth=1.4, color=color, alpha=0.95)
        ax.add_patch(circle)
        ax.plot([], [], linestyle=linestyle, color=color, linewidth=1.4, label=label)

    ax.set_xlabel("UMAP dimension 1")
    ax.set_ylabel("UMAP dimension 2")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="best")
    fig.tight_layout()
    save_all_formats(fig, out_dir, "query_best_peptide_rescue_region_umap")
    plt.close(fig)

    ranked_full = region_values.sort_values("similarity_to_query", ascending=False).copy()
    ranked_full["true_rank"] = np.arange(1, len(ranked_full) + 1)
    best_same = ranked_full[ranked_full["library_role"] == "positive_library"].iloc[0]
    best_unmatched = ranked_full[ranked_full["library_role"] == "negative_library"].iloc[0]
    ranked = ranked_full.head(30).copy()
    if not (ranked["library_role"] == "negative_library").any():
        ranked = pd.concat([ranked, best_unmatched.to_frame().T], ignore_index=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    ranked["retrieval_class"] = np.where(ranked["library_role"] == "positive_library", "Same sequence", "Unmatched")
    ranked.to_csv(out_dir / "query_best_peptide_top_retrievals_values.csv", index=False)

    margin = float(best_same["similarity_to_query"] - best_unmatched["similarity_to_query"])
    pd.DataFrame(
        [
            {
                "pair_id": best_pair_id,
                "query_peptide": best_pair_label,
                "query_ordinal": hero_ordinal,
                "best_same_sequence_rank": int(best_same["true_rank"]),
                "best_same_sequence_similarity": float(best_same["similarity_to_query"]),
                "best_unmatched_rank": int(best_unmatched["true_rank"]),
                "best_unmatched_similarity": float(best_unmatched["similarity_to_query"]),
                "margin": margin,
            }
        ]
    ).to_csv(out_dir / "query_best_peptide_top_retrievals_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    bar_colors = [
        colors["same_sequence"] if role == "positive_library" else colors["unmatched"]
        for role in ranked["library_role"]
    ]
    ax.bar(ranked["rank"], ranked["similarity_to_query"], color=bar_colors, width=0.82)
    ax.axhline(float(best_same["similarity_to_query"]), color=colors["same_sequence"], linestyle="-", linewidth=1.4)
    ax.axhline(float(best_unmatched["similarity_to_query"]), color=colors["failure"], linestyle="--", linewidth=1.4)
    ax.annotate(
        f"margin = {margin:.4f}",
        xy=(len(ranked) * 0.62, float(best_same["similarity_to_query"])),
        xytext=(len(ranked) * 0.62, float(best_same["similarity_to_query"]) + 0.003),
        color=colors["dark_text"],
        bbox={"facecolor": colors["light_background"], "edgecolor": "none", "boxstyle": "round,pad=0.35"},
    )
    ax.set_xlabel("Retrieved library rank")
    ax.set_ylabel("Cosine similarity to query")
    ax.set_xlim(0.3, len(ranked) + 0.7)
    y_min = max(0.80, float(ranked["similarity_to_query"].min()) - 0.004)
    y_max = min(1.005, float(ranked["similarity_to_query"].max()) + 0.008)
    ax.set_ylim(y_min, y_max)
    same_handle = plt.Line2D([0], [0], color=colors["same_sequence"], lw=6, label="Same sequence")
    unmatched_handle = plt.Line2D([0], [0], color=colors["unmatched"], lw=6, label="Unmatched")
    ax.legend(handles=[same_handle, unmatched_handle], loc="lower left")
    fig.tight_layout()
    save_all_formats(fig, out_dir, "query_best_peptide_top_retrievals_by_similarity")
    plt.close(fig)


def main() -> None:
    artifact_root = Path("spectral_rescue_seed42_full_artifacts")
    task_dir = artifact_root / "spectralrescuetaskreformulated"
    out_dir = Path("spectral_rescue_seed42_publication_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_publication_style()

    metric_df = load_pair_query_metrics(task_dir)
    scores = load_base_match_unmatch_scores(task_dir)
    distance_scores = load_query_distance_similarity(task_dir)

    write_metric_summary(metric_df, out_dir)
    pair_summary = pd.read_csv(out_dir / "query_duplicate_retrieval_pair_summary.csv")
    scores.to_csv(out_dir / "query_match_vs_unmatched_cosine_values.csv", index=False)

    plot_pair_recall(pair_summary, out_dir)
    plot_pair_margin(pair_summary, out_dir)
    plot_margin_cdf(metric_df, out_dir)
    plot_match_unmatch_distribution(scores, out_dir)
    plot_threshold_risk(scores, out_dir)
    plot_distance_similarity_hexbin(distance_scores, out_dir)
    plot_retrieval_curve(metric_df, out_dir)
    plot_best_query_rescue_region(metric_df, task_dir, out_dir)

    manifest = {
        "artifact_root": str(artifact_root),
        "output_dir": str(out_dir),
        "formats": ["png", "svg", "pdf"],
        "style_source": "rescue_task_handover/scripts/publication.py",
        "color_source": "/home/hjisaac/Downloads/metadata_colors.json",
        "rescue_color_mapping": load_shared_colors(),
        "note": "Plot titles are encoded in filenames; plot canvases intentionally omit titles.",
        "files": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    (out_dir / "publication_plot_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote publication plots to {out_dir}")


if __name__ == "__main__":
    main()
