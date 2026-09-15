#!/usr/bin/env python3
# ruff: noqa: T201 - a CLI script: the printed output is the whole point
"""Compare 2D projections for the best spectral rescue query."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import h5py
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import normalize
import umap


PAIR_ID = "pxd047134_mvnhek_mox"
QUERY_PEPTIDE = "ATLEMVNHEK"
QUERY_ORDINAL = 8
MAX_NEGATIVES = 150


def load_shared_colors() -> dict[str, str]:
    path = Path(__file__).resolve().parents[2] / "config" / "metadata_colors.json"
    if path.is_file():
        payload = json.loads(path.read_text())
        palette = payload.get("palette", [])
        return {
            "same_sequence": palette[0] if len(palette) > 0 else "#4E9AC6",
            "success": palette[1] if len(palette) > 1 else "#6DBF91",
            "secondary": palette[2] if len(palette) > 2 else "#F5A45D",
            "failure": palette[4] if len(palette) > 4 else "#E87878",
            "unmatched": palette[7] if len(palette) > 7 else "#AAAAAA",
        }
    return {
        "same_sequence": "#4E9AC6",
        "success": "#6DBF91",
        "secondary": "#F5A45D",
        "failure": "#E87878",
        "unmatched": "#AAAAAA",
    }


def set_publication_style() -> None:
    sns.set_theme(style="ticks")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Palatino", "Palatino Linotype", "TeX Gyre Pagella", "Book Antiqua", "URW Palladio L", "DejaVu Serif"],
            "font.size": 14,
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
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.5,
        }
    )


def save_all_formats(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_dir / f"{stem}.{ext}", dpi=300, bbox_inches="tight")


def load_points() -> tuple[np.ndarray, pd.DataFrame]:
    artifact_root = Path("spectral_rescue_seed42_full_artifacts")
    task_dir = artifact_root / "spectralrescuetaskreformulated"
    pair_dir = task_dir / "pairs" / PAIR_ID

    with h5py.File(artifact_root / "embeddings.h5", "r") as handle:
        embeddings = handle["embeddings"][:]
    embeddings = normalize(embeddings.astype(np.float32), norm="l2")

    payload = np.load(pair_dir / "rescue_similarity_matrix.npz", allow_pickle=True)
    query_indices = np.asarray(payload["query_indices"], dtype=np.int64)
    library_indices = np.asarray(payload["library_indices"], dtype=np.int64)
    library_roles = np.asarray(payload["library_roles"], dtype=object)

    query_index = int(query_indices[QUERY_ORDINAL])
    positive_cols = np.where(library_roles == "positive_library")[0].tolist()
    negative_cols = np.where(library_roles == "negative_library")[0].tolist()
    rng = np.random.RandomState(42)
    if len(negative_cols) > MAX_NEGATIVES:
        negative_cols = rng.choice(negative_cols, size=MAX_NEGATIVES, replace=False).tolist()

    selected_library_cols = positive_cols + negative_cols
    selected_indices = [query_index] + [int(library_indices[col]) for col in selected_library_cols]
    X = embeddings[selected_indices]

    rows = [{"point_id": "query", "role": "query", "embedding_index": query_index}]
    for col in selected_library_cols:
        role = "same_sequence" if library_roles[col] == "positive_library" else "unmatched"
        rows.append({"point_id": f"library_{int(library_indices[col])}", "role": role, "embedding_index": int(library_indices[col])})
    meta = pd.DataFrame(rows)

    query_vec = X[0]
    similarities = X @ query_vec
    meta["cosine_similarity_to_query"] = similarities
    meta["true_rank_by_similarity"] = meta["cosine_similarity_to_query"].rank(method="first", ascending=False).astype(int)
    return X, meta


def project_points(method: str, X: np.ndarray) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=42).fit_transform(X)
    if method == "mds":
        distances = squareform(pdist(X, metric="cosine"))
        return MDS(n_components=2, metric=True, dissimilarity="precomputed", random_state=42, normalized_stress="auto").fit_transform(distances)
    if method == "tsne":
        perplexity = min(30, max(5, (len(X) - 1) // 3))
        return TSNE(n_components=2, metric="cosine", perplexity=perplexity, init="random", random_state=42).fit_transform(X)
    if method == "umap":
        return umap.UMAP(n_neighbors=min(15, len(X) - 1), min_dist=0.3, metric="cosine", random_state=42).fit_transform(X)
    raise ValueError(method)


def plot_projection(method: str, coords: np.ndarray, meta: pd.DataFrame, out_dir: Path) -> dict[str, float | str]:
    colors = load_shared_colors()
    plot_df = meta.copy()
    plot_df["x"] = coords[:, 0]
    plot_df["y"] = coords[:, 1]
    query = plot_df.iloc[0]
    plot_df["projection_distance_to_query"] = np.sqrt((plot_df["x"] - query["x"]) ** 2 + (plot_df["y"] - query["y"]) ** 2)
    non_query = plot_df[plot_df["role"] != "query"].copy()
    nearest = non_query.sort_values("projection_distance_to_query").iloc[0]
    best_true = non_query.sort_values("cosine_similarity_to_query", ascending=False).iloc[0]
    rho = spearmanr(non_query["cosine_similarity_to_query"], -non_query["projection_distance_to_query"]).statistic

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    for role, label, color, size, alpha in [
        ("unmatched", "Unmatched", colors["unmatched"], 28, 0.24),
        ("same_sequence", "Same sequence", colors["same_sequence"], 44, 0.76),
    ]:
        subset = plot_df[plot_df["role"] == role]
        ax.scatter(subset["x"], subset["y"], s=size, color=color, alpha=alpha, edgecolors="white", linewidths=0.35, label=label)
    ax.scatter([query["x"]], [query["y"]], s=190, marker="*", color=colors["secondary"], edgecolors="black", linewidths=0.8, label="Query", zorder=5)
    ax.set_xlabel(f"{method.upper()} dimension 1")
    ax.set_ylabel(f"{method.upper()} dimension 2")
    ax.legend(loc="best")
    fig.tight_layout()
    stem = f"query_best_peptide_projection_{method}"
    save_all_formats(fig, out_dir, stem)
    plt.close(fig)

    plot_df.to_csv(out_dir / f"{stem}_values.csv", index=False)
    return {
        "method": method,
        "nearest_projected_role": str(nearest["role"]),
        "nearest_projected_similarity": float(nearest["cosine_similarity_to_query"]),
        "best_true_role": str(best_true["role"]),
        "best_true_similarity": float(best_true["cosine_similarity_to_query"]),
        "spearman_similarity_vs_negative_projected_distance": float(rho),
    }


def main() -> None:
    out_dir = Path("spectral_rescue_seed42_publication_plots")
    out_dir.mkdir(exist_ok=True)
    set_publication_style()
    X, meta = load_points()
    summaries = []
    for method in ("pca", "mds", "tsne", "umap"):
        coords = project_points(method, X)
        summaries.append(plot_projection(method, coords, meta, out_dir))
    pd.DataFrame(summaries).to_csv(out_dir / "query_best_peptide_projection_comparison_summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
