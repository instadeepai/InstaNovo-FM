#!/usr/bin/env python3
"""Generate publication-style plots for sequence-rescue results."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.metrics import average_precision_score, roc_auc_score


def load_shared_colors() -> dict[str, str]:
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


def save_all_formats(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def wrap_label(text: str, width: int = 16) -> str:
    return "\n".join(textwrap.wrap(str(text), width=width, break_long_words=True))


def pair_label(sequence: str) -> str:
    return str(sequence).replace("[UNIMOD:4]", "[Carb]").replace("[UNIMOD:35]", "[Ox]")


def load_artifacts(task_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_summary = pd.read_csv(task_dir / "rescue_pair_summary.csv")
    pair_summary["pair_label"] = pair_summary["base_sequence"].map(pair_label)
    metrics = []
    scores = []
    ranked = []
    for row in pair_summary.itertuples(index=False):
        pair_id = str(row.pair_id)
        pair_dir = task_dir / "pairs" / pair_id
        metric = pd.read_csv(pair_dir / "query_metrics.csv")
        metric["pair_id"] = pair_id
        metric["pair_label"] = row.pair_label
        metrics.append(metric)

        score = pd.read_csv(pair_dir / "all_pair_scores.csv")
        score["pair_id"] = pair_id
        score["pair_label"] = row.pair_label
        score["class"] = np.where(score["library_role"] == "positive_library", "same_sequence", "unmatched")
        scores.append(score)

        ranked_df = pd.read_csv(pair_dir / "ranked_library_by_query.csv")
        ranked_df["pair_id"] = pair_id
        ranked_df["pair_label"] = row.pair_label
        ranked.append(ranked_df)
    return pair_summary, pd.concat(metrics, ignore_index=True), pd.concat(scores, ignore_index=True), pd.concat(ranked, ignore_index=True)


def write_metric_summaries(metric_df: pd.DataFrame, pair_summary: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    base = metric_df[metric_df["query_role"] == "reference_query"].copy()
    ks = [1, 5, 10, 20, 50, 100]
    recall_rows = [
        {
            "k": k,
            "mean_recall": float(base[f"recall@{k}"].mean()),
            "std_recall": float(base[f"recall@{k}"].std(ddof=0)),
            "n_queries": int(base[f"recall@{k}"].notna().sum()),
        }
        for k in ks
    ]
    pd.DataFrame(recall_rows).to_csv(out_dir / "query_duplicate_retrieval_recall_curve_summary.csv", index=False)

    summary = (
        base.groupby(["pair_id", "pair_label"], sort=False)
        .agg(
            recall_at_1=("recall@1", "mean"),
            recall_at_10=("recall@10", "mean"),
            mean_margin=("margin", "mean"),
            median_margin=("margin", "median"),
            n_queries=("margin", "size"),
        )
        .reset_index()
        .merge(pair_summary[["pair_id", "project_id", "base_sequence"]], on="pair_id", how="left")
    )
    summary.to_csv(out_dir / "query_duplicate_retrieval_pair_summary.csv", index=False)
    pd.DataFrame(
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
    ).to_csv(out_dir / "query_duplicate_retrieval_margin_summary.csv", index=False)
    return summary


def plot_pair_recall(pair_summary: pd.DataFrame, out_dir: Path, colors: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    y = np.arange(len(pair_summary))
    ax.barh(y, pair_summary["recall_at_1"], color=colors["same_sequence"], height=0.68)
    ax.axvline(pair_summary["recall_at_1"].mean(), color=colors["dark_text"], linestyle="--", linewidth=1.5)
    ax.set_xlim(max(0.0, float(pair_summary["recall_at_1"].min()) - 0.08), 1.03)
    ax.set_xlabel("Query Recall@1")
    ax.set_ylabel("Query peptide")
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(label) for label in pair_summary["pair_label"]])
    ax.tick_params(axis="y", labelsize=8)
    ax.invert_yaxis()
    fig.subplots_adjust(left=0.38, right=0.98, top=0.97, bottom=0.14)
    save_all_formats(fig, out_dir, "query_duplicate_retrieval_recall_at_1_by_query_peptide")
    plt.close(fig)


def plot_pair_margin(pair_summary: pd.DataFrame, out_dir: Path, colors: dict[str, str]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    y = np.arange(len(pair_summary))
    ax.barh(y, pair_summary["mean_margin"], color=colors["success"], height=0.68)
    ax.axvline(0, color=colors["failure"], linestyle="--", linewidth=1.2)
    ax.set_xlim(min(0, float(pair_summary["mean_margin"].min()) * 1.2), float(pair_summary["mean_margin"].max()) * 1.25)
    ax.set_xlabel("Mean retrieval margin")
    ax.set_ylabel("Query peptide")
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(label) for label in pair_summary["pair_label"]])
    ax.tick_params(axis="y", labelsize=8)
    ax.invert_yaxis()
    fig.subplots_adjust(left=0.38, right=0.98, top=0.97, bottom=0.14)
    save_all_formats(fig, out_dir, "query_duplicate_retrieval_margin_by_query_peptide")
    plt.close(fig)


def plot_margin_cdf(metric_df: pd.DataFrame, out_dir: Path, colors: dict[str, str]) -> None:
    margins = metric_df[metric_df["query_role"] == "reference_query"]["margin"].to_numpy(dtype=float)
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


def plot_match_unmatch_distribution(score_df: pd.DataFrame, out_dir: Path, colors: dict[str, str]) -> None:
    values = score_df[score_df["query_role"] == "reference_query"].copy()
    values = values[["class", "score"]]
    values.to_csv(out_dir / "query_match_vs_unmatched_cosine_values.csv", index=False)
    match = values[values["class"] == "same_sequence"]["score"].to_numpy(dtype=float)
    unmatch = values[values["class"] == "unmatched"]["score"].to_numpy(dtype=float)
    y_true = np.r_[np.ones(len(match)), np.zeros(len(unmatch))]
    y_score = np.r_[match, unmatch]
    auroc = float(roc_auc_score(y_true, y_score))
    auprc = float(average_precision_score(y_true, y_score))
    summary_rows = [
        {
            "class": name,
            "n": len(vals),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "p99": float(np.percentile(vals, 99)),
        }
        for name, vals in (("same_sequence", match), ("unmatched", unmatch))
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


def plot_threshold_risk(score_df: pd.DataFrame, out_dir: Path, colors: dict[str, str]) -> None:
    values = score_df[score_df["query_role"] == "reference_query"].copy()
    match = values[values["class"] == "same_sequence"]["score"].to_numpy(dtype=float)
    unmatch = values[values["class"] == "unmatched"]["score"].to_numpy(dtype=float)
    lower = max(0.0, min(np.percentile(match, 1), np.percentile(unmatch, 90)) - 0.01)
    upper = min(1.0, max(np.percentile(match, 99), np.percentile(unmatch, 99.9)) + 0.005)
    thresholds = np.linspace(lower, upper, 50)
    df = pd.DataFrame(
        [
            {
                "threshold": float(t),
                "same_sequence_above_threshold": float(np.mean(match >= t)),
                "unmatched_above_threshold": float(np.mean(unmatch >= t)),
            }
            for t in thresholds
        ]
    )
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


def plot_distance_similarity_hexbin(score_df: pd.DataFrame, out_dir: Path, colors: dict[str, str]) -> None:
    values = score_df[score_df["query_role"] == "reference_query"].copy()
    values = values.rename(columns={"clean_edit_distance": "sequence_distance", "score": "similarity"})
    values[["pair_id", "pair_label", "query_index", "library_index", "library_role", "sequence_distance", "similarity", "class"]].to_csv(
        out_dir / "query_sequence_distance_vs_similarity_values.csv",
        index=False,
    )
    summary = (
        values.groupby(["class", "sequence_distance"], as_index=False)
        .agg(n=("similarity", "size"), mean_similarity=("similarity", "mean"), median_similarity=("similarity", "median"))
        .sort_values(["class", "sequence_distance"])
    )
    summary.to_csv(out_dir / "query_sequence_distance_vs_similarity_hexbin_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    hb = ax.hexbin(values["sequence_distance"], values["similarity"], gridsize=(28, 32), mincnt=1, cmap="Blues", linewidths=0.0)
    ax.axvline(0, color=colors["success"], linestyle="--", linewidth=1.3)
    ax.set_xlabel("Sequence distance to query")
    ax.set_ylabel("Cosine similarity")
    ax.set_xlim(-0.5, float(values["sequence_distance"].max()) + 0.5)
    ax.set_ylim(max(0.0, float(values["similarity"].min()) - 0.01), min(1.005, float(values["similarity"].max()) + 0.01))
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("Query-library pairs")
    fig.tight_layout()
    save_all_formats(fig, out_dir, "query_sequence_distance_vs_similarity_hexbin")
    plt.close(fig)


def plot_retrieval_curve(metric_df: pd.DataFrame, out_dir: Path, colors: dict[str, str]) -> None:
    base = metric_df[metric_df["query_role"] == "reference_query"].copy()
    ks = [1, 5, 10, 20, 50, 100]
    df = pd.DataFrame([{"k": k, "mean": float(base[f"recall@{k}"].mean()), "std": float(base[f"recall@{k}"].std(ddof=0))} for k in ks])
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(df["k"], df["mean"], marker="o", color=colors["same_sequence"], linewidth=2.4)
    ax.fill_between(df["k"], df["mean"] - df["std"], df["mean"] + df["std"], color=colors["same_sequence"], alpha=0.15)
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(max(0.0, float(df["mean"].min() - df["std"].max()) - 0.04), 1.01)
    ax.set_xlabel("k")
    ax.set_ylabel("Query Recall@k")
    fig.tight_layout()
    save_all_formats(fig, out_dir, "query_duplicate_retrieval_recall_curve")
    plt.close(fig)


def _scale_coords(coords: np.ndarray) -> np.ndarray:
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    scale = max(x_max - x_min, y_max - y_min, 1e-6)
    out = np.zeros_like(coords)
    out[:, 0] = (coords[:, 0] - x_min) / scale * 10.0
    out[:, 1] = (coords[:, 1] - y_min) / scale * 10.0
    return out


def _projection_inputs(task_dir: Path, metric_df: pd.DataFrame, pair_summary: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    best_pair_id = str(pair_summary.sort_values(["mean_margin", "recall_at_1"], ascending=False).iloc[0]["pair_id"])
    pair_dir = task_dir / "pairs" / best_pair_id
    pair_metrics = pd.read_csv(pair_dir / "query_metrics.csv")
    best_query = pair_metrics[pair_metrics["query_role"] == "reference_query"].sort_values("margin", ascending=False).iloc[0]
    hero_ordinal = int(best_query["query_ordinal"])
    payload = np.load(pair_dir / "rescue_similarity_matrix.npz", allow_pickle=True)
    S = payload["similarity_matrix"].astype(np.float32)
    query_indices = np.asarray(payload["query_indices"], dtype=np.int64)
    library_indices = np.asarray(payload["library_indices"], dtype=np.int64)
    library_roles = np.asarray(payload["library_roles"], dtype=object)
    positive_cols = np.where(library_roles == "positive_library")[0].tolist()
    negative_cols = np.where(library_roles == "negative_library")[0].tolist()
    rng = np.random.RandomState(42)
    if len(negative_cols) > 150:
        negative_cols = rng.choice(negative_cols, size=150, replace=False).tolist()
    selected_cols = positive_cols + negative_cols

    embeddings_path = task_dir.parent / "embeddings.h5"
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Expected embeddings file next to task artifacts: {embeddings_path}")
    with h5py.File(embeddings_path, "r") as handle:
        embeddings = handle["embeddings"][:].astype(np.float32)
    query_embedding_indices = np.asarray([query_indices[hero_ordinal]], dtype=np.int64)
    library_embedding_indices = library_indices[np.asarray(selected_cols, dtype=np.int64)]
    point_indices = np.concatenate([query_embedding_indices, library_embedding_indices]).astype(np.int64)
    features = embeddings[point_indices]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-12)

    rows = []
    for _i, embedding_index in enumerate(query_embedding_indices):
        rows.append(
            {
                "kind": "query",
                "role": "Query",
                "ordinal": int(hero_ordinal),
                "embedding_index": int(embedding_index),
                "similarity_to_query": float(1.0),
            }
        )
    for _j, col in enumerate(selected_cols):
        role = "Same sequence" if library_roles[col] == "positive_library" else "Unmatched"
        rows.append(
            {
                "kind": "library",
                "role": role,
                "ordinal": int(col),
                "embedding_index": int(library_indices[col]),
                "similarity_to_query": float(S[hero_ordinal, col]),
            }
        )
    points = pd.DataFrame(rows)
    return points, features


def plot_best_query_projections(task_dir: Path, metric_df: pd.DataFrame, pair_summary: pd.DataFrame, out_dir: Path, colors: dict[str, str]) -> None:
    points, features = _projection_inputs(task_dir, metric_df, pair_summary)
    reducers = {
        "pca": PCA(n_components=2, random_state=42),
        "mds": MDS(n_components=2, random_state=42, normalized_stress="auto", dissimilarity="euclidean"),
        "tsne": TSNE(n_components=2, random_state=42, init="pca", learning_rate="auto", perplexity=min(30, max(5, (len(points) - 1) // 4))),
        "umap": umap.UMAP(n_components=2, n_neighbors=min(15, len(points) - 1), min_dist=0.3, random_state=42),
    }
    for name, reducer in reducers.items():
        coords = _scale_coords(reducer.fit_transform(features))
        output = points.copy()
        output["x"] = coords[:, 0]
        output["y"] = coords[:, 1]
        output.to_csv(out_dir / f"query_best_peptide_projection_{name}_values.csv", index=False)

        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        for role, color, size, alpha, marker in (
            ("Unmatched", colors["unmatched"], 24, 0.22, "o"),
            ("Same sequence", colors["same_sequence"], 42, 0.72, "o"),
            ("Query", colors["secondary"], 190, 1.0, "*"),
        ):
            part = output[output["role"] == role]
            if part.empty:
                continue
            ax.scatter(part["x"], part["y"], s=size, color=color, alpha=alpha, marker=marker, edgecolors="white" if role == "Same sequence" else "none", linewidths=0.35, label=role)
        ax.set_xlabel(f"{name.upper()} dimension 1")
        ax.set_ylabel(f"{name.upper()} dimension 2")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")
        fig.tight_layout()
        save_all_formats(fig, out_dir, f"query_best_peptide_projection_{name}")
        plt.close(fig)


def plot_best_query_top_retrievals(metric_df: pd.DataFrame, ranked_df: pd.DataFrame, pair_summary: pd.DataFrame, out_dir: Path, colors: dict[str, str]) -> None:
    best_pair_id = str(pair_summary.sort_values(["mean_margin", "recall_at_1"], ascending=False).iloc[0]["pair_id"])
    pair_metrics = metric_df[(metric_df["pair_id"] == best_pair_id) & (metric_df["query_role"] == "reference_query")]
    best_query = pair_metrics.sort_values("margin", ascending=False).iloc[0]
    ranked = ranked_df[(ranked_df["pair_id"] == best_pair_id) & (ranked_df["query_ordinal"] == int(best_query["query_ordinal"]))].copy()
    ranked_full = ranked.sort_values("score", ascending=False).copy()
    best_same = ranked_full[ranked_full["library_role"] == "positive_library"].iloc[0]
    best_unmatched = ranked_full[ranked_full["library_role"] == "negative_library"].iloc[0]
    top = ranked_full.head(30).copy()
    if not (top["library_role"] == "negative_library").any():
        top = pd.concat([top, best_unmatched.to_frame().T], ignore_index=True)
    top["plot_rank"] = np.arange(1, len(top) + 1)
    top["retrieval_class"] = np.where(top["library_role"] == "positive_library", "Same sequence", "Unmatched")
    top.to_csv(out_dir / "query_best_peptide_top_retrievals_values.csv", index=False)
    margin = float(best_same["score"] - best_unmatched["score"])
    pd.DataFrame(
        [
            {
                "pair_id": best_pair_id,
                "query_peptide": best_query["pair_label"],
                "query_ordinal": int(best_query["query_ordinal"]),
                "best_same_sequence_rank": int(best_same["rank"]),
                "best_same_sequence_similarity": float(best_same["score"]),
                "best_unmatched_rank": int(best_unmatched["rank"]),
                "best_unmatched_similarity": float(best_unmatched["score"]),
                "margin": margin,
            }
        ]
    ).to_csv(out_dir / "query_best_peptide_top_retrievals_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    bar_colors = [colors["same_sequence"] if role == "positive_library" else colors["unmatched"] for role in top["library_role"]]
    ax.bar(top["plot_rank"], top["score"], color=bar_colors, width=0.82)
    ax.axhline(float(best_same["score"]), color=colors["same_sequence"], linestyle="-", linewidth=1.4)
    ax.axhline(float(best_unmatched["score"]), color=colors["failure"], linestyle="--", linewidth=1.4)
    ax.annotate(
        f"margin = {margin:.4f}",
        xy=(len(top) * 0.62, float(best_same["score"])),
        xytext=(len(top) * 0.62, float(best_same["score"]) + 0.003),
        color=colors["dark_text"],
        bbox={"facecolor": colors["light_background"], "edgecolor": "none", "boxstyle": "round,pad=0.35"},
    )
    ax.set_xlabel("Retrieved library rank")
    ax.set_ylabel("Cosine similarity to query")
    ax.set_xlim(0.3, len(top) + 0.7)
    ax.set_ylim(max(0.0, float(top["score"].min()) - 0.004), min(1.005, float(top["score"].max()) + 0.008))
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=colors["same_sequence"], lw=6, label="Same sequence"),
            plt.Line2D([0], [0], color=colors["unmatched"], lw=6, label="Unmatched"),
        ],
        loc="lower left",
    )
    fig.tight_layout()
    save_all_formats(fig, out_dir, "query_best_peptide_top_retrievals_by_similarity")
    plt.close(fig)


def generate(task_dir: Path, out_dir: Path, split_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = load_shared_colors()
    set_publication_style()
    raw_pair_summary, metric_df, score_df, ranked_df = load_artifacts(task_dir)
    pair_summary = write_metric_summaries(metric_df, raw_pair_summary, out_dir)
    plot_pair_recall(pair_summary, out_dir, colors)
    plot_pair_margin(pair_summary, out_dir, colors)
    plot_margin_cdf(metric_df, out_dir, colors)
    plot_match_unmatch_distribution(score_df, out_dir, colors)
    plot_threshold_risk(score_df, out_dir, colors)
    plot_distance_similarity_hexbin(score_df, out_dir, colors)
    plot_retrieval_curve(metric_df, out_dir, colors)
    plot_best_query_top_retrievals(metric_df, ranked_df, pair_summary, out_dir, colors)
    plot_best_query_projections(task_dir, metric_df, pair_summary, out_dir, colors)
    manifest = {
        "split": split_name,
        "artifact_dir": str(task_dir),
        "output_dir": str(out_dir),
        "formats": ["png", "svg", "pdf"],
        "note": "Plot titles are encoded in filenames; plot canvases intentionally omit titles.",
        "files": sorted(path.name for path in out_dir.iterdir() if path.is_file()),
    }
    (out_dir / "publication_plot_manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-name", required=True)
    args = parser.parse_args()
    generate(args.artifact_dir, args.output_dir, args.split_name)


if __name__ == "__main__":
    main()
