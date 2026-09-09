"""Publication-quality plots for the reformulated spectral rescue evaluation task."""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import umap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from sklearn.metrics import auc, average_precision_score, precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)

_METADATA_COLORS_PATH = Path(__file__).with_name("metadata_colors.json")


@dataclass(frozen=True)
class RescuePlotColors:
    match: str
    unmatch: str
    accent: str
    green: str
    red: str
    non_match_fill: str
    hist_marginal: str
    zone1: str
    zone2: str
    horizon: str


def load_rescue_plot_colors(path: Optional[Path] = None) -> RescuePlotColors:
    """Load rescue plot colors from metadata_colors.json palette."""
    colors_path = path or _METADATA_COLORS_PATH
    with colors_path.open() as handle:
        payload = json.load(handle)
    palette = payload["palette"]
    return RescuePlotColors(
        match=palette[0],
        unmatch=palette[7],
        accent=palette[2],
        green=palette[1],
        red=palette[4],
        non_match_fill="#E5E5E5",
        hist_marginal="#9A94A8",
        zone1="#993333",
        zone2="#2E78B8",
        horizon="#CC6666",
    )


def _display_sequence(sequence: str, *, max_len: int = 18) -> str:
    text = (
        str(sequence)
        .replace("[UNIMOD:35]", "[Ox]")
        .replace("[UNIMOD:21]", "[Ox]")
        .replace(" ", "")
    )
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


def _set_publication_style() -> None:
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
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.5,
            "xtick.major.width": 1,
            "ytick.major.width": 1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.fontsize": 11,
            "legend.frameon": False,
            "legend.columnspacing": 1.5,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.5,
        }
    )


def _set_neurips_style() -> None:
    sns.set_theme(style="ticks")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Palatino", "DejaVu Serif", "Book Antiqua"],
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "legend.frameon": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.15,
            "grid.color": "#CCCCCC",
            "grid.linewidth": 0.4,
        }
    )


def _scale_coords(coords: np.ndarray) -> np.ndarray:
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    max_range = max(x_max - x_min, y_max - y_min, 1e-6)
    scaled = np.zeros_like(coords)
    scaled[:, 0] = 1.5 + 7.0 * (coords[:, 0] - x_min) / max_range
    scaled[:, 1] = 1.5 + 7.0 * (coords[:, 1] - y_min) / max_range
    return scaled


def _compute_shared_peaks_and_cosine(
    mz1: np.ndarray,
    int1: np.ndarray,
    mz2: np.ndarray,
    int2: np.ndarray,
    *,
    tolerance: float = 0.02,
) -> Tuple[float, float]:
    matched1: List[float] = []
    matched2: List[float] = []
    for index, mz_value in enumerate(mz1):
        diffs = np.abs(mz2 - mz_value)
        if len(diffs) == 0:
            continue
        min_idx = int(np.argmin(diffs))
        if diffs[min_idx] <= tolerance:
            matched1.append(float(int1[index]))
            matched2.append(float(int2[min_idx]))
    if not matched1:
        return 0.0, 0.0
    dot = float(np.sum(np.array(matched1) * np.array(matched2)))
    norm1 = float(np.sqrt(np.sum(int1**2)))
    norm2 = float(np.sqrt(np.sum(int2**2)))
    if norm1 * norm2 <= 0:
        return 0.0, 0.0
    shared_rate = len(matched1) / len(mz1) * 100 if len(mz1) else 0.0
    return shared_rate, dot / (norm1 * norm2)


def _get_peaks(meta: Optional[Dict[str, np.ndarray]], index: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if meta is None:
        return None, None
    mz_key = "mz_array" if "mz_array" in meta else "mz"
    intensity_key = "intensity_array" if "intensity_array" in meta else "intensity"
    if mz_key not in meta or intensity_key not in meta:
        return None, None
    mz = np.asarray(meta[mz_key][index], dtype=np.float64)
    intensity = np.asarray(meta[intensity_key][index], dtype=np.float64)
    if intensity.size and np.max(intensity) > 0:
        intensity = intensity / np.max(intensity)
    return mz, intensity


def _subsample_scores(scores: np.ndarray, max_scores: int, rng: np.random.RandomState) -> np.ndarray:
    if len(scores) <= max_scores:
        return scores
    chosen = rng.choice(len(scores), size=max_scores, replace=False)
    return scores[np.sort(chosen)]


def _collect_match_unmatch_scores(
    S: np.ndarray,
    selection: Dict[str, Any],
    positive_library_role: str,
    *,
    max_pair_scores: Optional[int] = None,
    sample_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    library_roles = np.array(selection["library_roles"], dtype=object)
    match_scores: List[float] = []
    unmatch_scores: List[float] = []
    for query_ordinal in range(S.shape[0]):
        for library_ordinal, role in enumerate(library_roles):
            score = float(S[query_ordinal, library_ordinal])
            if role == positive_library_role:
                match_scores.append(score)
            else:
                unmatch_scores.append(score)
    match_scores = np.asarray(match_scores, dtype=np.float64)
    unmatch_scores = np.asarray(unmatch_scores, dtype=np.float64)
    if max_pair_scores is not None and max_pair_scores > 0:
        rng = np.random.RandomState(sample_seed)
        per_class_cap = max(1, max_pair_scores // 2)
        match_scores = _subsample_scores(match_scores, per_class_cap, rng)
        unmatch_scores = _subsample_scores(unmatch_scores, per_class_cap, rng)
    return match_scores, unmatch_scores


def _write_match_unmatch_csv(path: Path, match_scores: np.ndarray, unmatch_scores: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class", "score"])
        writer.writeheader()
        for score in match_scores:
            writer.writerow({"class": "match", "score": float(score)})
        for score in unmatch_scores:
            writer.writerow({"class": "unmatch", "score": float(score)})


def _write_margin_csv(path: Path, query_metrics: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["margin"])
        writer.writeheader()
        for row in query_metrics:
            margin = row.get("margin")
            if margin is not None and np.isfinite(float(margin)):
                writer.writerow({"margin": float(margin)})


def _pick_hero_query_ordinal(
    S: np.ndarray,
    selection: Dict[str, Any],
    query_metrics: Sequence[Dict[str, Any]],
    *,
    modified_query_role: str,
    positive_library_role: str,
    negative_library_role: str,
    min_negative_clean_edit_distance: int,
    clean_edit_distance_fn: Any,
) -> Optional[int]:
    candidates: List[Tuple[float, int]] = []
    library_roles = np.array(selection["library_roles"], dtype=object)
    sequences = selection["sequences"]
    for metrics in query_metrics:
        if metrics.get("query_role") != modified_query_role:
            continue
        margin = metrics.get("margin")
        if margin is None or not np.isfinite(float(margin)):
            continue
        query_ordinal = int(metrics["query_ordinal"])
        query_idx = int(selection["query_indices"][query_ordinal])
        query_sequence = sequences[query_idx]
        num_positives = int(np.sum(library_roles == positive_library_role))
        num_negatives = 0
        for library_ordinal, role in enumerate(library_roles):
            if role != negative_library_role:
                continue
            library_idx = int(selection["library_indices"][library_ordinal])
            distance = clean_edit_distance_fn(query_sequence, sequences[library_idx])
            if distance >= min_negative_clean_edit_distance:
                num_negatives += 1
        if num_positives >= 2 and num_negatives >= 5:
            candidates.append((float(margin), query_ordinal))

    if not candidates:
        for metrics in query_metrics:
            if metrics.get("query_role") == modified_query_role:
                margin = metrics.get("margin")
                if margin is not None and np.isfinite(float(margin)):
                    candidates.append((float(margin), int(metrics["query_ordinal"])))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def _resolve_rescue_artifact_file(artifact_dir: Path, filename: str) -> Path:
    """Find a rescue task artifact at the seed root or task subdirectory."""
    candidates = (
        artifact_dir / filename,
        artifact_dir / "spectralrescuetaskreformulated" / filename,
    )
    for path in candidates:
        if path.is_file():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing {filename!r}; searched: {searched}")


def load_rescue_artifacts_from_dir(
    artifact_dir: Path,
    *,
    embeddings_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]], Optional[Dict[str, np.ndarray]]]:
    """Load similarity matrix, selection, and query metrics from a prior rescue run.

    Args:
        artifact_dir: Seed output directory (or the task subdirectory) containing
            ``rescue_similarity_matrix.npz``, ``rescue_selection.csv``, and
            ``query_metrics.csv``.
        embeddings_dir: Optional directory with ``embeddings.h5`` for spectrum
            overlay plots (3-point comparison, clean horizon zoom). Defaults to
            *artifact_dir* when that file exists.

    Returns:
        Tuple of (S, selection, query_metrics, meta). *meta* is ``None`` when no
        embeddings file is available.
    """
    artifact_dir = Path(artifact_dir)
    npz_path = _resolve_rescue_artifact_file(artifact_dir, "rescue_similarity_matrix.npz")
    selection_csv = _resolve_rescue_artifact_file(artifact_dir, "rescue_selection.csv")
    metrics_csv = _resolve_rescue_artifact_file(artifact_dir, "query_metrics.csv")

    payload = np.load(npz_path, allow_pickle=True)
    S = payload["similarity_matrix"]
    query_indices = np.asarray(payload["query_indices"], dtype=np.int64)
    query_roles = np.asarray(payload["query_roles"], dtype=object)
    library_indices = np.asarray(payload["library_indices"], dtype=np.int64)
    library_roles = np.asarray(payload["library_roles"], dtype=object)

    row_fields: Dict[str, List[Any]] = {
        "sequences": [],
        "unmodified": [],
        "projects": [],
        "usis": [],
        "source_shards": [],
        "source_rows": [],
    }
    max_row_index = -1
    with selection_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            idx = int(row["row_index"])
            max_row_index = max(max_row_index, idx)
            while len(row_fields["sequences"]) <= idx:
                for values in row_fields.values():
                    values.append(None)
            row_fields["sequences"][idx] = row["sequence"]
            row_fields["unmodified"][idx] = row["unmodified_peptide"]
            row_fields["projects"][idx] = row["project"]
            row_fields["usis"][idx] = row["usi"]
            row_fields["source_shards"][idx] = row["source_shard"]
            row_fields["source_rows"][idx] = row["source_row_in_shard"]

    if max_row_index < 0:
        raise ValueError(f"No rows found in {selection_csv}")

    selection: Dict[str, Any] = {
        "query_indices": query_indices.tolist(),
        "query_roles": query_roles.tolist(),
        "library_indices": library_indices.tolist(),
        "library_roles": library_roles.tolist(),
        "sequences": np.asarray(row_fields["sequences"], dtype=object),
        "unmodified": np.asarray(row_fields["unmodified"], dtype=object),
        "projects": np.asarray(row_fields["projects"], dtype=object),
        "usis": np.asarray(row_fields["usis"], dtype=object),
        "source_shards": np.asarray(row_fields["source_shards"], dtype=object),
        "source_rows": np.asarray(row_fields["source_rows"], dtype=object),
    }

    query_metrics: List[Dict[str, Any]] = []
    float_fields = {
        "margin",
        "best_positive_similarity",
        "mean_positive_similarity",
        "median_positive_similarity",
        "best_negative_similarity",
    }
    int_fields = {"query_index", "query_ordinal", "best_positive_rank"}
    with metrics_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: Dict[str, Any] = dict(row)
            for key in int_fields:
                if key in parsed and parsed[key] not in (None, ""):
                    parsed[key] = int(float(parsed[key]))
            for key in float_fields:
                if key in parsed and parsed[key] not in (None, ""):
                    parsed[key] = float(parsed[key])
            for key, value in list(parsed.items()):
                if key.startswith(("recall@", "prop_recall@", "map@")) and value not in (None, ""):
                    parsed[key] = float(value)
            query_metrics.append(parsed)

    meta: Optional[Dict[str, np.ndarray]] = None
    embed_root = Path(embeddings_dir) if embeddings_dir is not None else artifact_dir
    embeddings_path = embed_root / "embeddings.h5"
    if embeddings_path.is_file():
        from instanovo_fm.eval import embedding_io

        _, meta, _ = embedding_io.load(embed_root)
        logger.info(
            "Loaded embedding metadata from %s (%d rows, keys=%s)",
            embeddings_path,
            len(meta.get("usi", [])),
            list(meta.keys()),
        )

    return S, selection, query_metrics, meta


@dataclass
class _UmapNeighborhood:
    hero_query_ordinal: int
    hero_query_index: int
    hero_query_sequence: str
    selected_cols: List[int]
    coords_scaled: np.ndarray
    query_point: np.ndarray
    replicate_points: np.ndarray
    rescued_points: np.ndarray
    unmatched_points: np.ndarray
    replicate_cols: List[int]
    rescued_cols: List[int]
    unmatched_cols: List[int]
    recall_zone1: float
    recall_zone2: float
    closest_replicate: Tuple[int, np.ndarray, float, float]
    closest_rescued: Tuple[int, np.ndarray, float, float]
    closest_unmatched: Tuple[int, np.ndarray, float, float]
    closest_unmatched_sequence: str
    rescued_sequence: str


def _build_umap_neighborhood(
    S: np.ndarray,
    selection: Dict[str, Any],
    *,
    hero_query_ordinal: int,
    positive_library_role: str,
    negative_library_role: str,
    min_negative_clean_edit_distance: int,
    clean_edit_distance_fn: Any,
    sample_seed: int,
    zone1_radius: float = 1.5,
    zone2_radius: float = 2.5,
    max_background_negatives: int = 150,
) -> Optional[_UmapNeighborhood]:
    library_roles = np.array(selection["library_roles"], dtype=object)
    sequences = selection["sequences"]
    query_idx = int(selection["query_indices"][hero_query_ordinal])
    query_sequence = str(sequences[query_idx])

    positive_cols = [
        col
        for col, role in enumerate(library_roles)
        if role == positive_library_role
    ]
    negative_cols = []
    for col in range(len(library_roles)):
        if library_roles[col] != negative_library_role:
            continue
        library_idx = int(selection["library_indices"][col])
        distance = clean_edit_distance_fn(query_sequence, sequences[library_idx])
        if distance >= min_negative_clean_edit_distance:
            negative_cols.append(col)

    if len(positive_cols) < 2 or len(negative_cols) < 5:
        return None

    rng = np.random.RandomState(sample_seed)
    if len(negative_cols) > max_background_negatives:
        negative_cols = rng.choice(negative_cols, size=max_background_negatives, replace=False).tolist()

    positive_scores = np.array([float(S[hero_query_ordinal, col]) for col in positive_cols], dtype=np.float64)
    best_positive_col = positive_cols[int(np.argmax(positive_scores))]
    replicate_cols = [col for col in positive_cols if col != best_positive_col]
    rescued_cols = [best_positive_col]
    selected_cols = rescued_cols + replicate_cols + negative_cols

    profile_matrix = S[:, selected_cols].T.astype(np.float32)
    n_samples = profile_matrix.shape[0]
    if n_samples < 3:
        return None
    n_neighbors = min(15, n_samples - 1)
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.3, random_state=sample_seed)
    coords = reducer.fit_transform(profile_matrix)
    query_profile = S[hero_query_ordinal, selected_cols].reshape(1, -1).astype(np.float32)
    query_coord = reducer.transform(query_profile)
    coords_scaled = _scale_coords(coords)
    query_point = _scale_coords(query_coord)[0]

    rescued_points = coords_scaled[: len(rescued_cols)]
    replicate_points = coords_scaled[len(rescued_cols) : len(rescued_cols) + len(replicate_cols)]
    unmatched_points = coords_scaled[len(rescued_cols) + len(replicate_cols) :]

    dists = np.sqrt(np.sum((coords_scaled - query_point) ** 2, axis=1))
    rep_dists = dists[len(rescued_cols) : len(rescued_cols) + len(replicate_cols)]
    rescued_dists = dists[: len(rescued_cols)]
    total_matches = len(replicate_cols) + len(rescued_cols)
    recall_zone1 = float((np.sum(rep_dists <= zone1_radius) + np.sum(rescued_dists <= zone1_radius)) / total_matches * 100)
    recall_zone2 = float((np.sum(rep_dists <= zone2_radius) + np.sum(rescued_dists <= zone2_radius)) / total_matches * 100)

    def _closest(cols: List[int], points: np.ndarray, dist_values: np.ndarray) -> Tuple[int, np.ndarray, float, float]:
        if not cols:
            raise ValueError("Expected at least one column for closest-point lookup")
        best_local = int(np.argmin(dist_values))
        col = cols[best_local]
        return col, points[best_local], float(dist_values[best_local]), float(S[hero_query_ordinal, col])

    closest_rescued = _closest(rescued_cols, rescued_points, rescued_dists)
    closest_replicate = _closest(replicate_cols, replicate_points, rep_dists) if replicate_cols else closest_rescued
    unmatched_dists = dists[len(rescued_cols) + len(replicate_cols) :]
    closest_unmatched = _closest(negative_cols, unmatched_points, unmatched_dists)
    closest_unmatched_idx = int(selection["library_indices"][closest_unmatched[0]])

    return _UmapNeighborhood(
        hero_query_ordinal=hero_query_ordinal,
        hero_query_index=query_idx,
        hero_query_sequence=query_sequence,
        selected_cols=selected_cols,
        coords_scaled=coords_scaled,
        query_point=query_point,
        replicate_points=replicate_points,
        rescued_points=rescued_points,
        unmatched_points=unmatched_points,
        replicate_cols=replicate_cols,
        rescued_cols=rescued_cols,
        unmatched_cols=negative_cols,
        recall_zone1=recall_zone1,
        recall_zone2=recall_zone2,
        closest_replicate=closest_replicate,
        closest_rescued=closest_rescued,
        closest_unmatched=closest_unmatched,
        closest_unmatched_sequence=str(sequences[closest_unmatched_idx]),
        rescued_sequence=str(sequences[int(selection["library_indices"][closest_rescued[0]])]),
    )


def _plot_match_unmatch_distribution(
    out_dir: Path,
    match_scores: np.ndarray,
    unmatch_scores: np.ndarray,
    colors: RescuePlotColors,
    plot_dpi: int,
) -> Optional[str]:
    if len(match_scores) == 0 or len(unmatch_scores) == 0:
        return None

    _set_publication_style()
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    sns.kdeplot(
        unmatch_scores,
        ax=ax,
        fill=True,
        color=colors.unmatch,
        label=f"Unmatch ($n$={len(unmatch_scores):,})",
        alpha=0.35,
        linewidth=1.5,
    )
    sns.kdeplot(
        match_scores,
        ax=ax,
        fill=True,
        color=colors.match,
        label=f"Match ($n$={len(match_scores):,})",
        alpha=0.35,
        linewidth=1.5,
    )
    ax.axvline(float(np.mean(unmatch_scores)), color=colors.unmatch, linestyle="--", linewidth=1.5)
    ax.axvline(float(np.mean(match_scores)), color=colors.match, linestyle="-", linewidth=1.5)
    lower = max(0.5, min(float(np.min(unmatch_scores)), float(np.min(match_scores))) - 0.02)
    ax.set_xlim(lower, 1.02)
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Density")
    ax.set_title("Cosine Similarity Distribution: Match vs Unmatch")
    ax.legend(loc="upper left")

    y_true = np.concatenate([np.ones(len(match_scores)), np.zeros(len(unmatch_scores))])
    y_scores = np.concatenate([match_scores, unmatch_scores])
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)

    inset_ax = ax.inset_axes([0.08, 0.42, 0.44, 0.44])
    inset_ax.plot(fpr, tpr, color=colors.match, lw=1.8, label=f"ROC (AUC={roc_auc:.3f})")
    inset_ax.plot(recall, precision, color=colors.accent, lw=1.8, label=f"PR (AP={pr_auc:.3f})")
    inset_ax.plot([0, 1], [0, 1], color="#CCCCCC", linestyle="--", lw=0.8)
    inset_ax.set_xlim(0.0, 1.0)
    inset_ax.set_ylim(0.0, 1.05)
    inset_ax.set_xlabel("FPR / Recall", fontsize=8)
    inset_ax.set_ylabel("TPR / Precision", fontsize=8)
    inset_ax.tick_params(axis="both", labelsize=8)
    inset_ax.legend(loc="lower left", fontsize=7.5, frameon=False)
    inset_ax.spines["top"].set_visible(False)
    inset_ax.spines["right"].set_visible(False)
    fig.tight_layout()

    filename = f"match_unmatch_distribution_auroc_{roc_auc:.3f}_auprc_{pr_auc:.3f}.png"
    save_path = out_dir / filename
    fig.savefig(save_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def _plot_margin_histogram_and_cdf(
    out_dir: Path,
    margins: np.ndarray,
    colors: RescuePlotColors,
    plot_dpi: int,
) -> Optional[str]:
    if len(margins) == 0:
        return None

    _set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax_hist, ax_cdf = axes

    ax_hist.hist(margins, bins=25, color=colors.match, alpha=0.75, edgecolor="none")
    ax_hist.axvline(0.0, color=colors.red, linestyle="--", linewidth=1.5)
    ax_hist.set_xlabel("margin")
    ax_hist.set_ylabel("count")
    ax_hist.set_title("Margin Histogram (best pos - best neg)")

    sorted_margins = np.sort(margins)
    cdf = np.arange(1, len(sorted_margins) + 1) / len(sorted_margins)
    ax_cdf.plot(sorted_margins, cdf, color=colors.match, linewidth=2.2)
    ax_cdf.axvline(0.0, color=colors.red, linestyle="--", linewidth=1.5)
    ax_cdf.set_xlabel("margin")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("Margin CDF")
    fig.tight_layout()

    save_path = out_dir / "margin_distribution_and_cdf.png"
    fig.savefig(save_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def _plot_unified_margin_and_cdf(
    out_dir: Path,
    margins: np.ndarray,
    colors: RescuePlotColors,
    plot_dpi: int,
) -> Optional[str]:
    if len(margins) == 0:
        return None

    _set_publication_style()
    fig, ax1 = plt.subplots(figsize=(7.5, 5.0))
    ax1.axvspan(0, float(np.max(margins)) + 0.02, color=colors.green, alpha=0.08)
    ax1.axvspan(float(np.min(margins)) - 0.02, 0, color=colors.red, alpha=0.08)
    ax1.axvline(0, color=colors.red, linestyle="--", linewidth=1.5)

    _, bins, patches = ax1.hist(margins, bins=25, alpha=0.6, edgecolor="none")
    for patch, left_bin in zip(patches, bins[:-1]):
        patch.set_facecolor(colors.green if left_bin >= 0 else colors.red)
        patch.set_alpha(0.7)

    ax1.set_xlabel("Retrieval Margin (best positive - best negative similarity)")
    ax1.set_ylabel("Query Count", color="#1A3F60")
    ax1.tick_params(axis="y", labelcolor="#1A3F60")

    ax2 = ax1.twinx()
    sorted_margins = np.sort(margins)
    cdf = np.arange(1, len(sorted_margins) + 1) / len(sorted_margins)
    ax2.plot(sorted_margins, cdf, color=colors.match, linewidth=2.5, label="CDF")
    rescue_rate = float(np.mean(margins > 0) * 100)
    idx_zero = int(np.searchsorted(sorted_margins, 0.0))
    cdf_at_zero = float(cdf[idx_zero] if idx_zero < len(cdf) else 1.0)
    ax2.scatter(0, cdf_at_zero, color=colors.accent, s=60, zorder=10)
    ax2.annotate(
        f"Rescue Rate: {rescue_rate:.1f}%",
        xy=(0, cdf_at_zero),
        xytext=(-0.015, max(0.05, cdf_at_zero - 0.25)),
        arrowprops=dict(arrowstyle="->", color=colors.accent, lw=1.5),
        color=colors.accent,
        weight="bold",
        size=11,
    )
    ax2.set_ylabel("Cumulative Probability", color=colors.match)
    ax2.tick_params(axis="y", labelcolor=colors.match)
    ax2.set_ylim(-0.02, 1.05)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color(colors.match)
    ax2.spines["right"].set_linewidth(1.5)
    ax1.legend(
        handles=[
            Patch(facecolor=colors.green, alpha=0.7, label="Positive Margin (Success)"),
            Patch(facecolor=colors.red, alpha=0.7, label="Negative Margin (Fail)"),
            Line2D([0], [0], color=colors.match, lw=2.5, label="CDF"),
        ],
        loc="upper left",
    )
    fig.tight_layout()

    filename = f"unified_margin_and_cdf_rescue_rate_{rescue_rate:.1f}.png"
    save_path = out_dir / filename
    fig.savefig(save_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def _plot_umap_global(
    out_dir: Path,
    neighborhood: _UmapNeighborhood,
    colors: RescuePlotColors,
    plot_dpi: int,
    *,
    zone1_radius: float = 1.5,
    zone2_radius: float = 2.5,
) -> str:
    _set_neurips_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.set_aspect("equal")
    ax.scatter(
        neighborhood.unmatched_points[:, 0],
        neighborhood.unmatched_points[:, 1],
        c=colors.non_match_fill,
        s=12,
        alpha=0.5,
        edgecolors="none",
        zorder=1,
    )
    ax.scatter(
        np.concatenate([neighborhood.replicate_points[:, 0], neighborhood.rescued_points[:, 0]]),
        np.concatenate([neighborhood.replicate_points[:, 1], neighborhood.rescued_points[:, 1]]),
        c=colors.match,
        s=20,
        alpha=0.8,
        zorder=3,
    )
    ax.scatter(
        neighborhood.query_point[0],
        neighborhood.query_point[1],
        c=colors.accent,
        marker="*",
        s=90,
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
    )
    ax.add_patch(
        Circle(
            (neighborhood.query_point[0], neighborhood.query_point[1]),
            zone1_radius,
            color=colors.horizon,
            fill=False,
            linestyle=":",
            linewidth=0.8,
            alpha=0.9,
            zorder=2,
        )
    )
    ax.add_patch(
        Circle(
            (neighborhood.query_point[0], neighborhood.query_point[1]),
            zone2_radius,
            color=colors.zone2,
            fill=False,
            linestyle="--",
            linewidth=0.8,
            alpha=0.9,
            zorder=2,
        )
    )
    bbox_zone = dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.8)
    ax.text(
        neighborhood.query_point[0] - 0.9,
        neighborhood.query_point[1] + zone1_radius - 0.2,
        f"Zone 1 ($r={zone1_radius}$)\nRecall: {neighborhood.recall_zone1:.1f}%",
        color=colors.zone1,
        fontsize=6.5,
        weight="bold",
        bbox=bbox_zone,
        ha="right",
    )
    ax.text(
        neighborhood.query_point[0] + 0.9,
        neighborhood.query_point[1] - zone2_radius + 0.1,
        f"Zone 2 ($r={zone2_radius}$)\nRecall: {neighborhood.recall_zone2:.1f}%",
        color=colors.zone2,
        fontsize=6.5,
        weight="bold",
        bbox=bbox_zone,
        ha="left",
    )
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Global Embedding Neighborhood", fontsize=8.5, weight="bold", pad=8)
    ax.legend(
        [
            ax.scatter([], [], c=colors.accent, marker="*", s=70, edgecolors="black", linewidths=0.5),
            ax.scatter([], [], c=colors.match, s=20),
            ax.scatter([], [], c=colors.non_match_fill, s=12),
            Line2D([0], [0], color=colors.horizon, linestyle=":", lw=1.0),
            Line2D([0], [0], color=colors.zone2, linestyle="--", lw=1.0),
        ],
        [
            "Query Spectrum",
            "Matches ($d=0, 1$)",
            "Unmatched ($d \\geq 5$)",
            f"Zone 1 ($r={zone1_radius}$)",
            f"Zone 2 ($r={zone2_radius}$)",
        ],
        loc="upper right",
        fontsize=6.5,
        frameon=True,
        facecolor="white",
        edgecolor="#E5E5E5",
    )
    fig.tight_layout()
    save_path = out_dir / "reimagined_rescue_neighborhood_umap_global.png"
    fig.savefig(save_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def _plot_umap_zoom(
    out_dir: Path,
    neighborhood: _UmapNeighborhood,
    colors: RescuePlotColors,
    plot_dpi: int,
    *,
    zone2_radius: float = 2.5,
) -> str:
    _set_neurips_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.set_aspect("equal")
    zoom_half_width = zone2_radius + 0.3
    ax.set_xlim(neighborhood.query_point[0] - zoom_half_width, neighborhood.query_point[0] + zoom_half_width)
    ax.set_ylim(neighborhood.query_point[1] - zoom_half_width, neighborhood.query_point[1] + zoom_half_width)
    ax.scatter(
        neighborhood.unmatched_points[:, 0],
        neighborhood.unmatched_points[:, 1],
        c=colors.non_match_fill,
        s=12,
        alpha=0.5,
        edgecolors="none",
        zorder=1,
    )
    ax.scatter(
        neighborhood.rescued_points[:, 0],
        neighborhood.rescued_points[:, 1],
        c=colors.green,
        s=25,
        alpha=0.9,
        zorder=3,
    )
    ax.scatter(
        neighborhood.replicate_points[:, 0],
        neighborhood.replicate_points[:, 1],
        c=colors.match,
        s=25,
        alpha=0.9,
        zorder=3,
    )
    ax.scatter(
        neighborhood.query_point[0],
        neighborhood.query_point[1],
        c=colors.accent,
        marker="*",
        s=110,
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
    )
    ax.add_patch(
        Circle(
            (neighborhood.query_point[0], neighborhood.query_point[1]),
            zone2_radius,
            color=colors.zone2,
            fill=False,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            zorder=2,
        )
    )
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Zoom: Zone 2 (Rescue Horizon)", fontsize=8.5, weight="bold", pad=8)

    labels = [
        f"Query:     {_display_sequence(neighborhood.hero_query_sequence, max_len=19)}",
        f"Replicate: {_display_sequence(neighborhood.rescued_sequence, max_len=19)} (d=0, Cos: {neighborhood.closest_replicate[3]:.3f})",
        f"Rescued:   {_display_sequence(neighborhood.hero_query_sequence, max_len=19)} (d=1, Cos: {neighborhood.closest_rescued[3]:.3f})",
        f"Unmatched: {_display_sequence(neighborhood.closest_unmatched_sequence, max_len=19)} (d>=5, Cos: {neighborhood.closest_unmatched[3]:.3f})",
    ]
    handles = [
        Line2D([0], [0], marker="*", color=colors.accent, ls="", markeredgecolor="black", markeredgewidth=0.5, markersize=8),
        Line2D([0], [0], marker="o", color=colors.match, ls="", markersize=5),
        Line2D([0], [0], marker="o", color=colors.green, ls="", markersize=5),
        Line2D([0], [0], marker="o", color=colors.non_match_fill, ls="", markersize=4),
    ]
    legend = ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#BBBBBB",
        prop={"family": "monospace", "size": 5.2},
        borderpad=0.35,
        labelspacing=0.45,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(0.4)
    fig.tight_layout()
    save_path = out_dir / "reimagined_rescue_neighborhood_umap_zoom.png"
    fig.savefig(save_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def _plot_umap_combined(
    out_dir: Path,
    neighborhood: _UmapNeighborhood,
    colors: RescuePlotColors,
    plot_dpi: int,
    *,
    zone1_radius: float = 1.5,
    zone2_radius: float = 2.5,
) -> str:
    _set_neurips_style()
    fig, (ax_global, ax_zoom) = plt.subplots(1, 2, figsize=(5.5, 2.8))
    bbox_zone = dict(boxstyle="square,pad=0.2", fc="white", ec="none", alpha=0.8)

    ax_global.set_aspect("equal")
    ax_global.scatter(
        neighborhood.unmatched_points[:, 0],
        neighborhood.unmatched_points[:, 1],
        c=colors.non_match_fill,
        s=8,
        alpha=0.5,
        edgecolors="none",
        zorder=1,
    )
    ax_global.scatter(
        np.concatenate([neighborhood.replicate_points[:, 0], neighborhood.rescued_points[:, 0]]),
        np.concatenate([neighborhood.replicate_points[:, 1], neighborhood.rescued_points[:, 1]]),
        c=colors.match,
        s=15,
        alpha=0.8,
        zorder=3,
    )
    ax_global.scatter(
        neighborhood.query_point[0],
        neighborhood.query_point[1],
        c=colors.accent,
        marker="*",
        s=70,
        edgecolors="black",
        linewidths=0.5,
        zorder=4,
    )
    ax_global.add_patch(
        Circle(
            (neighborhood.query_point[0], neighborhood.query_point[1]),
            zone1_radius,
            color=colors.horizon,
            fill=False,
            linestyle=":",
            linewidth=0.8,
            alpha=0.9,
            zorder=2,
        )
    )
    ax_global.add_patch(
        Circle(
            (neighborhood.query_point[0], neighborhood.query_point[1]),
            zone2_radius,
            color=colors.zone2,
            fill=False,
            linestyle="--",
            linewidth=0.8,
            alpha=0.9,
            zorder=2,
        )
    )
    ax_global.text(
        neighborhood.query_point[0] - 0.9,
        neighborhood.query_point[1] + zone1_radius - 0.2,
        f"Zone 1 ($r={zone1_radius}$)\nRecall: {neighborhood.recall_zone1:.1f}%",
        color=colors.zone1,
        fontsize=6.5,
        weight="bold",
        bbox=bbox_zone,
        ha="right",
    )
    ax_global.text(
        neighborhood.query_point[0] + 0.9,
        neighborhood.query_point[1] - zone2_radius + 0.1,
        f"Zone 2 ($r={zone2_radius}$)\nRecall: {neighborhood.recall_zone2:.1f}%",
        color=colors.zone2,
        fontsize=6.5,
        weight="bold",
        bbox=bbox_zone,
        ha="left",
    )
    ax_global.set_xlim(0, 10)
    ax_global.set_ylim(0, 10)
    ax_global.set_xlabel("UMAP Dimension 1")
    ax_global.set_ylabel("UMAP Dimension 2")
    ax_global.set_xticks([])
    ax_global.set_yticks([])
    ax_global.set_title("A. Global Embedding Neighborhood", fontsize=8.0, weight="bold", pad=8)

    zoom_half_width = zone2_radius + 0.3
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xlim(neighborhood.query_point[0] - zoom_half_width, neighborhood.query_point[0] + zoom_half_width)
    ax_zoom.set_ylim(neighborhood.query_point[1] - zoom_half_width, neighborhood.query_point[1] + zoom_half_width)
    ax_zoom.scatter(
        neighborhood.unmatched_points[:, 0],
        neighborhood.unmatched_points[:, 1],
        c=colors.non_match_fill,
        s=12,
        alpha=0.5,
        edgecolors="none",
        zorder=1,
    )
    ax_zoom.scatter(
        neighborhood.rescued_points[:, 0],
        neighborhood.rescued_points[:, 1],
        c=colors.green,
        s=25,
        alpha=0.9,
        zorder=3,
    )
    ax_zoom.scatter(
        neighborhood.replicate_points[:, 0],
        neighborhood.replicate_points[:, 1],
        c=colors.match,
        s=25,
        alpha=0.9,
        zorder=3,
    )
    ax_zoom.scatter(
        neighborhood.query_point[0],
        neighborhood.query_point[1],
        c=colors.accent,
        marker="*",
        s=110,
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
    )
    ax_zoom.add_patch(
        Circle(
            (neighborhood.query_point[0], neighborhood.query_point[1]),
            zone2_radius,
            color=colors.zone2,
            fill=False,
            linestyle="--",
            linewidth=0.8,
            alpha=0.5,
            zorder=2,
        )
    )
    ax_zoom.set_xlabel("UMAP Dimension 1")
    ax_zoom.set_ylabel("UMAP Dimension 2")
    ax_zoom.set_xticks([])
    ax_zoom.set_yticks([])
    ax_zoom.set_title("B. Zoom: Zone 2 (Rescue Horizon)", fontsize=8.0, weight="bold", pad=8)
    labels = [
        f"Query:     {_display_sequence(neighborhood.hero_query_sequence, max_len=19)}",
        f"Replicate: {_display_sequence(neighborhood.rescued_sequence, max_len=19)} (d=0, Cos: {neighborhood.closest_replicate[3]:.3f})",
        f"Rescued:   {_display_sequence(neighborhood.hero_query_sequence, max_len=19)} (d=1, Cos: {neighborhood.closest_rescued[3]:.3f})",
        f"Unmatched: {_display_sequence(neighborhood.closest_unmatched_sequence, max_len=19)} (d>=5, Cos: {neighborhood.closest_unmatched[3]:.3f})",
    ]
    handles = [
        Line2D([0], [0], marker="*", color=colors.accent, ls="", markeredgecolor="black", markeredgewidth=0.5, markersize=8),
        Line2D([0], [0], marker="o", color=colors.match, ls="", markersize=5),
        Line2D([0], [0], marker="o", color=colors.green, ls="", markersize=5),
        Line2D([0], [0], marker="o", color=colors.non_match_fill, ls="", markersize=4),
    ]
    legend = ax_zoom.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#BBBBBB",
        prop={"family": "monospace", "size": 5.2},
        borderpad=0.35,
        labelspacing=0.45,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(0.4)
    fig.tight_layout()
    save_path = out_dir / "reimagined_rescue_neighborhood_umap_combined.png"
    fig.savefig(save_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def _library_global_index(selection: Dict[str, Any], library_ordinal: int) -> int:
    return int(selection["library_indices"][library_ordinal])


def _legend_metric_lines(
    meta: Optional[Dict[str, np.ndarray]],
    query_index: int,
    library_ordinal: int,
    selection: Dict[str, Any],
    embed_cos: float,
) -> str:
    if meta is None:
        return f"\nEmbed Cos: {embed_cos:.3f}"
    library_index = _library_global_index(selection, library_ordinal)
    mz_q, int_q = _get_peaks(meta, query_index)
    mz_l, int_l = _get_peaks(meta, library_index)
    if mz_q is None or mz_l is None:
        return f"\nEmbed Cos: {embed_cos:.3f}"
    shared_rate, raw_cos = _compute_shared_peaks_and_cosine(mz_q, int_q, mz_l, int_l)
    return f"\nEmbed Cos: {embed_cos:.3f} | Raw Cos: {raw_cos:.3f}\nShared Peaks: {shared_rate:.1f}%"


def _raw_metric_suffix(
    meta: Optional[Dict[str, np.ndarray]],
    query_index: int,
    library_ordinal: int,
    selection: Dict[str, Any],
) -> str:
    if meta is None:
        return ""
    library_index = _library_global_index(selection, library_ordinal)
    mz_q, int_q = _get_peaks(meta, query_index)
    mz_l, int_l = _get_peaks(meta, library_index)
    if mz_q is None or mz_l is None:
        return ""
    shared_rate, raw_cos = _compute_shared_peaks_and_cosine(mz_q, int_q, mz_l, int_l)
    return f"\nRaw Cos:   {raw_cos:.3f} | Shared Peaks: {shared_rate:.1f}%"


def _plot_three_point_comparison(
    out_dir: Path,
    neighborhood: _UmapNeighborhood,
    colors: RescuePlotColors,
    plot_dpi: int,
    meta: Optional[Dict[str, np.ndarray]],
    selection: Dict[str, Any],
) -> str:
    _set_neurips_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.set_aspect("equal")

    pt_query = neighborhood.query_point
    _, pt_rescued, _, embed_rescued = neighborhood.closest_rescued
    _, pt_unmatched, _, embed_unmatched = neighborhood.closest_unmatched

    ax.scatter(
        pt_unmatched[0],
        pt_unmatched[1],
        c=colors.non_match_fill,
        s=40,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )
    ax.scatter(
        pt_rescued[0],
        pt_rescued[1],
        c=colors.green,
        s=40,
        edgecolors="black",
        linewidths=0.5,
        zorder=3,
    )
    ax.scatter(
        pt_query[0],
        pt_query[1],
        c=colors.accent,
        marker="*",
        s=130,
        edgecolors="black",
        linewidths=0.6,
        zorder=4,
    )
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Embedding space vs. Raw Spectral space", fontsize=8.5, weight="bold", pad=8)

    rescued_library_ordinal = int(neighborhood.closest_rescued[0])
    unmatched_library_ordinal = int(neighborhood.closest_unmatched[0])
    rescued_metrics = _legend_metric_lines(
        meta,
        neighborhood.hero_query_index,
        rescued_library_ordinal,
        selection,
        embed_rescued,
    )
    unmatched_metrics = _legend_metric_lines(
        meta,
        neighborhood.hero_query_index,
        unmatched_library_ordinal,
        selection,
        embed_unmatched,
    )

    labels = [
        f"Query:     {_display_sequence(neighborhood.hero_query_sequence, max_len=19)}\n(Modified)",
        f"Rescued:   {_display_sequence(neighborhood.rescued_sequence, max_len=19)}{rescued_metrics}",
        f"Unmatched: {_display_sequence(neighborhood.closest_unmatched_sequence, max_len=19)}{unmatched_metrics}",
    ]
    handles = [
        Line2D([0], [0], marker="*", color=colors.accent, ls="", markeredgecolor="black", markeredgewidth=0.5, markersize=8),
        Line2D([0], [0], marker="o", color=colors.green, ls="", markeredgecolor="black", markeredgewidth=0.5, markersize=6),
        Line2D([0], [0], marker="o", color=colors.non_match_fill, ls="", markeredgecolor="black", markeredgewidth=0.5, markersize=6),
    ]
    legend = ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#BBBBBB",
        prop={"family": "monospace", "size": 5.2},
        borderpad=0.4,
        labelspacing=0.6,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(0.4)

    points = np.array([pt_query, pt_rescued, pt_unmatched])
    center = points.mean(axis=0)
    half_width = max(points[:, 0].max() - center[0], points[:, 1].max() - center[1], 0.5) + 1.0
    ax.set_xlim(center[0] - half_width, center[0] + half_width)
    ax.set_ylim(center[1] - half_width, center[1] + half_width)
    fig.tight_layout()
    save_path = out_dir / "rescue_3point_comparison.png"
    fig.savefig(save_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def _plot_clean_horizon_zoom(
    out_dir: Path,
    neighborhood: _UmapNeighborhood,
    S: np.ndarray,
    colors: RescuePlotColors,
    plot_dpi: int,
    meta: Optional[Dict[str, np.ndarray]],
    selection: Dict[str, Any],
) -> str:
    _set_neurips_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.2))
    ax.set_aspect("equal")

    rescued_library_ordinal = int(neighborhood.closest_rescued[0])
    unmatched_library_ordinal = int(neighborhood.closest_unmatched[0])

    dists = np.sqrt(np.sum((neighborhood.coords_scaled - neighborhood.query_point) ** 2, axis=1))
    unmatched_local_idx = (
        len(neighborhood.rescued_cols)
        + len(neighborhood.replicate_cols)
        + neighborhood.unmatched_cols.index(unmatched_library_ordinal)
    )
    horizon_radius = float(dists[unmatched_local_idx])

    zoom_width = horizon_radius + 0.3
    ax.set_xlim(neighborhood.query_point[0] - zoom_width, neighborhood.query_point[0] + zoom_width)
    ax.set_ylim(neighborhood.query_point[1] - zoom_width, neighborhood.query_point[1] + zoom_width)

    ax.scatter(
        neighborhood.unmatched_points[:, 0],
        neighborhood.unmatched_points[:, 1],
        c=colors.non_match_fill,
        s=12,
        alpha=0.5,
        edgecolors="none",
        zorder=1,
    )
    ax.scatter(
        neighborhood.rescued_points[:, 0],
        neighborhood.rescued_points[:, 1],
        c=colors.green,
        s=25,
        alpha=0.9,
        zorder=3,
    )
    ax.scatter(
        neighborhood.replicate_points[:, 0],
        neighborhood.replicate_points[:, 1],
        c=colors.match,
        s=25,
        alpha=0.9,
        zorder=3,
    )
    ax.scatter(
        neighborhood.query_point[0],
        neighborhood.query_point[1],
        c=colors.accent,
        marker="*",
        s=110,
        edgecolors="black",
        linewidths=0.6,
        zorder=5,
    )
    ax.add_patch(
        Circle(
            (neighborhood.query_point[0], neighborhood.query_point[1]),
            horizon_radius,
            color=colors.horizon,
            fill=False,
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=2,
        )
    )
    _, pt_unmatched, _, embed_unmatched = neighborhood.closest_unmatched
    ax.scatter(pt_unmatched[0], pt_unmatched[1], facecolors="none", edgecolors=colors.horizon, s=55, linewidths=1.0, zorder=4)

    embed_rescued = float(S[neighborhood.hero_query_ordinal, rescued_library_ordinal])
    embed_unmatched = float(S[neighborhood.hero_query_ordinal, unmatched_library_ordinal])
    rescued_suffix = _raw_metric_suffix(
        meta,
        neighborhood.hero_query_index,
        rescued_library_ordinal,
        selection,
    )
    unmatched_suffix = _raw_metric_suffix(
        meta,
        neighborhood.hero_query_index,
        unmatched_library_ordinal,
        selection,
    )

    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("False-Positive-Free Horizon Zoom", fontsize=8.5, weight="bold", pad=8)
    labels = [
        f"Query:     {_display_sequence(neighborhood.hero_query_sequence, max_len=19)}",
        f"Rescued:   {_display_sequence(neighborhood.hero_query_sequence, max_len=19)} (d=1, Cos: {embed_rescued:.3f}){rescued_suffix}",
        f"Border FP: {_display_sequence(neighborhood.closest_unmatched_sequence, max_len=19)} (d>=5, Cos: {embed_unmatched:.3f}){unmatched_suffix}",
    ]
    handles = [
        Line2D([0], [0], marker="*", color=colors.accent, ls="", markeredgecolor="black", markeredgewidth=0.5, markersize=8),
        Line2D([0], [0], marker="o", color=colors.green, ls="", markersize=5),
        Line2D([0], [0], marker="o", color=colors.non_match_fill, ls="", markeredgecolor=colors.horizon, markeredgewidth=0.8, markersize=5),
    ]
    legend = ax.legend(
        handles,
        labels,
        loc="upper right",
        frameon=True,
        facecolor="white",
        edgecolor="#BBBBBB",
        prop={"family": "monospace", "size": 5.2},
        borderpad=0.35,
        labelspacing=0.5,
        handletextpad=0.5,
    )
    legend.get_frame().set_linewidth(0.4)
    fig.tight_layout()
    save_path = out_dir / "rescue_clean_horizon_zoom.png"
    fig.savefig(save_path, dpi=plot_dpi, bbox_inches="tight")
    plt.close(fig)
    return str(save_path)


def save_rescue_publication_plots(
    out_dir: Path,
    S: np.ndarray,
    selection: Dict[str, Any],
    query_metrics: Sequence[Dict[str, Any]],
    *,
    positive_library_role: str,
    negative_library_role: str,
    modified_query_role: str,
    min_negative_clean_edit_distance: int,
    clean_edit_distance_fn: Any,
    sample_seed: int,
    plot_dpi: int,
    plot_max_pair_scores: int = 50_000,
    meta: Optional[Dict[str, np.ndarray]] = None,
    include_distribution_plots: bool = True,
    include_umap_plots: bool = True,
    include_spectrum_plots: bool = True,
) -> Dict[str, str]:
    """Generate publication rescue plots and return output paths."""
    colors = load_rescue_plot_colors()
    paths: Dict[str, str] = {}

    match_scores, unmatch_scores = _collect_match_unmatch_scores(
        S,
        selection,
        positive_library_role,
        max_pair_scores=plot_max_pair_scores,
        sample_seed=sample_seed,
    )
    csv_match = out_dir / "match_unmatch_cosine_values.csv"
    _write_match_unmatch_csv(csv_match, match_scores, unmatch_scores)
    paths["match_unmatch_cosine_values_csv"] = str(csv_match)

    margins = np.array(
        [float(row["margin"]) for row in query_metrics if row.get("margin") is not None and np.isfinite(float(row["margin"]))],
        dtype=np.float64,
    )
    csv_margin = out_dir / "per_query_margin_values.csv"
    _write_margin_csv(csv_margin, query_metrics)
    paths["per_query_margin_values_csv"] = str(csv_margin)

    plotters = [
        ("match_unmatch_distribution", lambda: _plot_match_unmatch_distribution(out_dir, match_scores, unmatch_scores, colors, plot_dpi)),
        ("margin_distribution_and_cdf", lambda: _plot_margin_histogram_and_cdf(out_dir, margins, colors, plot_dpi)),
        ("unified_margin_and_cdf", lambda: _plot_unified_margin_and_cdf(out_dir, margins, colors, plot_dpi)),
    ]
    if include_distribution_plots:
        for key, plotter in plotters:
            try:
                plot_path = plotter()
                if plot_path:
                    paths[key] = plot_path
            except Exception:
                logger.exception("Failed to generate rescue plot %s", key)

    if not include_umap_plots and not include_spectrum_plots:
        return paths

    hero_ordinal = _pick_hero_query_ordinal(
        S,
        selection,
        query_metrics,
        modified_query_role=modified_query_role,
        positive_library_role=positive_library_role,
        negative_library_role=negative_library_role,
        min_negative_clean_edit_distance=min_negative_clean_edit_distance,
        clean_edit_distance_fn=clean_edit_distance_fn,
    )
    if hero_ordinal is None:
        logger.warning("Skipping UMAP rescue neighborhood plots: no suitable hero query found")
        return paths

    neighborhood = _build_umap_neighborhood(
        S,
        selection,
        hero_query_ordinal=hero_ordinal,
        positive_library_role=positive_library_role,
        negative_library_role=negative_library_role,
        min_negative_clean_edit_distance=min_negative_clean_edit_distance,
        clean_edit_distance_fn=clean_edit_distance_fn,
        sample_seed=sample_seed,
    )
    if neighborhood is None:
        logger.warning("Skipping UMAP rescue neighborhood plots: insufficient library groups")
        return paths

    umap_plotters: List[Tuple[str, Any]] = []
    if include_umap_plots:
        umap_plotters.extend(
            [
                ("reimagined_rescue_neighborhood_umap_global", lambda: _plot_umap_global(out_dir, neighborhood, colors, plot_dpi)),
                ("reimagined_rescue_neighborhood_umap_zoom", lambda: _plot_umap_zoom(out_dir, neighborhood, colors, plot_dpi)),
                ("reimagined_rescue_neighborhood_umap_combined", lambda: _plot_umap_combined(out_dir, neighborhood, colors, plot_dpi)),
            ]
        )
    if include_spectrum_plots:
        if meta is None:
            logger.warning("Skipping spectrum overlay plots: embedding metadata not available")
        else:
            umap_plotters.extend(
                [
                    ("rescue_3point_comparison", lambda: _plot_three_point_comparison(out_dir, neighborhood, colors, plot_dpi, meta, selection)),
                    (
                        "rescue_clean_horizon_zoom",
                        lambda: _plot_clean_horizon_zoom(out_dir, neighborhood, S, colors, plot_dpi, meta, selection),
                    ),
                ]
            )

    for key, plotter in umap_plotters:
        try:
            paths[key] = plotter()
        except Exception:
            logger.exception("Failed to generate rescue plot %s", key)

    return paths
