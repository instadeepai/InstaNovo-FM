"""Shared helpers for embedding-clustering evaluation tasks.

This module is intentionally *not* an evaluation task (it defines no ``BaseTask``
subclass), so the auto-discovery in ``__init__.py`` imports it without registering
anything.  It is consumed by:

- ``evoc_clustering.py``        — EVōC clustering + enrichment + hierarchical zoom
- ``glass_box_attribution.py``  — Glass Box UMAP feature attribution

Design notes
------------
* EVōC (``evoc.EVoC``) clusters the *learned* 768-d embeddings directly in
  high-dimensional space.  After ``fit_predict`` it exposes:
    - ``labels_``                 : (N,) finest-layer cluster id per point (-1 = noise)
    - ``cluster_layers_``         : list of (N,) label arrays, index 0 = finest,
                                    higher index = coarser
    - ``cluster_tree_``           : dict mapping a node ``(layer, cluster_id)`` to a
                                    list of child ``(layer-1, cluster_id)`` nodes.
                                    The root sits at layer ``len(cluster_layers_)``
                                    (a virtual node whose members are all points).
    - ``duplicates_``             : set of near-duplicate point indices
* Glass Box UMAP attributes onto *input feature columns*, so it is only meaningful on
  interpretable chemical descriptors — never on the opaque 768-d vectors.  The bridge
  between the two is the EVōC cluster label, overlaid on the descriptor embedding.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Default field lists (shared defaults; tasks may override via config)
# ----------------------------------------------------------------------------

# Numeric per-spectrum descriptors used to build the interpretable feature matrix
# and the cluster enrichment heatmap.  These are produced upstream by
# UMAPVisualisationTask._compute_spectral_properties / _compute_annotation_properties.
DEFAULT_NUMERIC_FIELDS: List[str] = [
    "precursor_charge",
    "precursor_mz",
    "collision_energy",
    "sequence_length",
    "hydrophobicity",
    "retention_time",
    "n_peaks",
    "peak_center_of_mass",
    "peak_spread",
    "annotation_ratio",
    "backbone_coverage",
    "signal_intensity_ratio",
    "median_ppm_error",
    "hyperscore",
]

# Categorical fields used for cluster enrichment / purity and as the candidate pool
# for the data-driven zoom-cascade level selector.
DEFAULT_CATEGORICAL_FIELDS: List[str] = [
    "frag_type",
    "search_detector",
    "search_instrument",
    "search_organism",
    "search_enzyme",
    "modification_types",
]


# ----------------------------------------------------------------------------
# EVōC clustering
# ----------------------------------------------------------------------------

class EVoCResult:
    """Container for EVōC outputs aligned to the (subsampled) embedding rows."""

    def __init__(
        self,
        labels: np.ndarray,
        cluster_layers: List[np.ndarray],
        cluster_tree: Dict[Any, List[Any]],
        duplicates: Optional[set] = None,
        persistence_scores: Optional[List[float]] = None,
    ) -> None:
        self.labels = labels
        self.cluster_layers = cluster_layers
        self.cluster_tree = cluster_tree
        self.duplicates = duplicates or set()
        self.persistence_scores = persistence_scores or []

    @property
    def n_clusters(self) -> int:
        return int(len(np.unique(self.labels[self.labels >= 0])))

    @property
    def noise_fraction(self) -> float:
        if len(self.labels) == 0:
            return 0.0
        return float((self.labels < 0).mean())


def run_evoc(
    emb: np.ndarray,
    normalize: bool = True,
    random_state: int = 42,
    **evoc_kwargs: Any,
) -> EVoCResult:
    """Cluster high-dimensional embeddings with EVōC.

    Args:
        emb: (N, D) embedding matrix.
        normalize: L2-normalise rows first so EVōC's Euclidean graph matches the
            cosine geometry used elsewhere in the eval suite.
        random_state: Seed forwarded to ``evoc.EVoC``.
        **evoc_kwargs: Extra keyword arguments forwarded to ``evoc.EVoC`` (e.g.
            ``noise_level``, ``base_min_cluster_size``, ``n_neighbors``, ``min_samples``).

    Returns:
        ``EVoCResult`` with labels aligned to ``emb`` rows.

    Raises:
        ImportError: if the ``evoc`` package is not installed (caller should skip).
    """
    import evoc  # noqa: F401  (raises ImportError -> caller degrades gracefully)

    X = np.ascontiguousarray(emb, dtype=np.float32)
    if normalize:
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = X / norms

    clusterer = evoc.EVoC(random_state=random_state, **evoc_kwargs)
    labels = np.asarray(clusterer.fit_predict(X))

    layers = [np.asarray(layer) for layer in getattr(clusterer, "cluster_layers_", [])]
    if not layers:
        layers = [labels]
    tree = dict(getattr(clusterer, "cluster_tree_", {}) or {})
    duplicates = set(getattr(clusterer, "duplicates_", set()) or set())
    persistence = list(getattr(clusterer, "persistence_scores_", []) or [])

    return EVoCResult(labels, layers, tree, duplicates, persistence)


# ----------------------------------------------------------------------------
# Metadata field extraction
# ----------------------------------------------------------------------------

def _as_1d(data: Any, n: int) -> Optional[np.ndarray]:
    """Coerce a metadata entry to a length-``n`` 1-D array, else None."""
    if data is None:
        return None
    arr = np.asarray(data, dtype=object) if not isinstance(data, np.ndarray) else data
    if arr.ndim > 1:
        if arr.shape[1] == 1:
            arr = arr.reshape(-1)
        else:
            return None
    if len(arr) != n:
        return None
    return arr


def get_numeric_field(meta: Dict[str, Any], field: str, n: int) -> Optional[np.ndarray]:
    """Return a length-``n`` float array for ``field`` (NaN where non-finite), else None."""
    arr = _as_1d(meta.get(field), n)
    if arr is None:
        return None
    try:
        out = np.asarray(arr, dtype=float)
    except (ValueError, TypeError):
        return None
    if not np.issubdtype(out.dtype, np.number) or np.all(~np.isfinite(out)):
        return None
    return out


def get_categorical_field(meta: Dict[str, Any], field: str, n: int) -> Optional[np.ndarray]:
    """Return a length-``n`` string array for ``field`` (None -> "unknown"), else None."""
    arr = _as_1d(meta.get(field), n)
    if arr is None:
        return None
    return np.array(["unknown" if v is None else str(v) for v in arr], dtype=object)


# ----------------------------------------------------------------------------
# Interpretable descriptor matrix (for Glass Box UMAP)
# ----------------------------------------------------------------------------

def build_descriptor_matrix(
    meta: Dict[str, Any],
    numeric_features: Optional[List[str]] = None,
    n: Optional[int] = None,
    max_nan_frac: float = 0.5,
) -> Tuple[np.ndarray, List[str]]:
    """Build a standardised interpretable feature matrix from metadata.

    Drops features missing entirely or with > ``max_nan_frac`` non-finite values,
    mean-imputes remaining NaNs, then standardises each column to mean 0 / std 1
    (Glass Box UMAP expects standardised input).

    Args:
        meta: Metadata dict.
        numeric_features: Candidate numeric feature keys (defaults to
            ``DEFAULT_NUMERIC_FIELDS``).
        n: Expected number of rows. Inferred from the first usable field if None.
        max_nan_frac: Maximum tolerated fraction of non-finite values per feature.

    Returns:
        ``(X, feature_names)`` where ``X`` is (n, F) float32 standardised, and
        ``feature_names`` lists the F retained features in column order. ``F`` may
        be 0 if nothing usable was found.
    """
    fields = numeric_features or DEFAULT_NUMERIC_FIELDS

    # Infer row count
    if n is None:
        for f in fields:
            arr = get_numeric_field(meta, f, len(meta.get(f, [])) if meta.get(f) is not None else 0)
            if arr is not None:
                n = len(arr)
                break
    if not n:
        return np.empty((0, 0), dtype=np.float32), []

    cols: List[np.ndarray] = []
    names: List[str] = []
    for f in fields:
        arr = get_numeric_field(meta, f, n)
        if arr is None:
            continue
        finite = np.isfinite(arr)
        if finite.mean() < (1.0 - max_nan_frac):
            logger.debug("Descriptor '%s' dropped: %.0f%% non-finite", f, 100 * (1 - finite.mean()))
            continue
        col = arr.copy()
        if not finite.all():
            col[~finite] = float(np.nanmean(arr[finite])) if finite.any() else 0.0
        cols.append(col)
        names.append(f)

    if not cols:
        return np.empty((n, 0), dtype=np.float32), []

    X = np.stack(cols, axis=1).astype(np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    X = (X - mean) / std
    return X, names


# ----------------------------------------------------------------------------
# Cluster enrichment
# ----------------------------------------------------------------------------

def compute_cluster_enrichment(
    labels: np.ndarray,
    meta: Dict[str, Any],
    categorical_fields: Optional[List[str]] = None,
    numeric_fields: Optional[List[str]] = None,
    include_noise: bool = False,
) -> Dict[str, Any]:
    """Characterise each cluster against metadata.

    Categorical fields: per-cluster dominant category, purity (its fraction), and
    log2 enrichment of that category vs the global rate.
    Numeric fields: per-cluster mean and standardised effect size (vs global mean/std).

    Returns a nested dict keyed ``categorical`` / ``numeric`` / ``cluster_sizes``.
    """
    categorical_fields = categorical_fields if categorical_fields is not None else DEFAULT_CATEGORICAL_FIELDS
    numeric_fields = numeric_fields if numeric_fields is not None else DEFAULT_NUMERIC_FIELDS
    n = len(labels)

    cluster_ids = [int(c) for c in np.unique(labels) if include_noise or c >= 0]
    cluster_sizes = {c: int((labels == c).sum()) for c in cluster_ids}

    cat_out: Dict[str, Any] = {}
    for field in categorical_fields:
        vals = get_categorical_field(meta, field, n)
        if vals is None:
            continue
        uniq, counts = np.unique(vals, return_counts=True)
        global_rate = {str(u): float(c) / n for u, c in zip(uniq, counts)}
        clusters: Dict[int, Any] = {}
        for c in cluster_ids:
            mask = labels == c
            cvals = vals[mask]
            cu, cc = np.unique(cvals, return_counts=True)
            j = int(np.argmax(cc))
            dominant = str(cu[j])
            purity = float(cc[j]) / float(mask.sum())
            base = global_rate.get(dominant, 1e-9)
            log2_enr = float(np.log2(max(purity, 1e-9) / max(base, 1e-9)))
            clusters[c] = {
                "dominant": dominant,
                "purity": purity,
                "log2_enrichment": log2_enr,
            }
        cat_out[field] = {"global_rate": global_rate, "clusters": clusters}

    num_out: Dict[str, Any] = {}
    for field in numeric_fields:
        arr = get_numeric_field(meta, field, n)
        if arr is None:
            continue
        finite = np.isfinite(arr)
        if not finite.any():
            continue
        g_mean = float(np.nanmean(arr[finite]))
        g_std = float(np.nanstd(arr[finite])) or 1.0
        clusters = {}
        for c in cluster_ids:
            mask = (labels == c) & finite
            if not mask.any():
                continue
            c_mean = float(arr[mask].mean())
            clusters[c] = {
                "mean": c_mean,
                "effect_size": (c_mean - g_mean) / g_std,
            }
        num_out[field] = {"global_mean": g_mean, "global_std": g_std, "clusters": clusters}

    return {"categorical": cat_out, "numeric": num_out, "cluster_sizes": cluster_sizes}


def enrichment_effect_matrix(
    enrichment: Dict[str, Any],
) -> Tuple[np.ndarray, List[int], List[str]]:
    """Flatten the numeric enrichment into a (clusters x features) effect-size matrix.

    Returns ``(matrix, cluster_ids, feature_names)``; ``matrix`` is NaN where a
    cluster/feature pair has no data.
    """
    num = enrichment.get("numeric", {})
    feature_names = list(num.keys())
    cluster_ids = sorted(enrichment.get("cluster_sizes", {}).keys())
    if not feature_names or not cluster_ids:
        return np.empty((0, 0)), cluster_ids, feature_names
    mat = np.full((len(cluster_ids), len(feature_names)), np.nan)
    for j, f in enumerate(feature_names):
        clusters = num[f]["clusters"]
        for i, c in enumerate(cluster_ids):
            if c in clusters:
                mat[i, j] = clusters[c]["effect_size"]
    return mat, cluster_ids, feature_names


# ----------------------------------------------------------------------------
# Data-driven level selection (for the zoom cascade)
# ----------------------------------------------------------------------------

def select_discriminative_field(
    child_assignment: np.ndarray,
    meta_subset: Dict[str, Any],
    candidate_fields: List[str],
) -> Optional[Dict[str, Any]]:
    """Pick the metadata field that best separates a node's children.

    Uses normalized mutual information (NMI) between the child-cluster assignment
    and each candidate field's categorical values, over the node's member points.
    The field with the highest NMI is returned together with its score and the
    dominant value per child — this is what makes the zoom-cascade level labels
    *data-driven* rather than assumed.

    Args:
        child_assignment: (m,) child-cluster id per member point of the node.
        meta_subset: metadata already sliced to the node's member points.
        candidate_fields: categorical field names to choose from.

    Returns:
        ``{"field", "score", "per_child_dominant"}`` or None if nothing usable.
    """
    from sklearn.metrics import normalized_mutual_info_score

    m = len(child_assignment)
    if m == 0 or len(np.unique(child_assignment)) < 2:
        return None

    best: Optional[Dict[str, Any]] = None
    for field in candidate_fields:
        vals = get_categorical_field(meta_subset, field, m)
        if vals is None or len(np.unique(vals)) < 2:
            continue
        score = float(normalized_mutual_info_score(child_assignment, vals))
        if best is None or score > best["score"]:
            per_child = {}
            for ch in np.unique(child_assignment):
                cv = vals[child_assignment == ch]
                cu, cc = np.unique(cv, return_counts=True)
                per_child[int(ch)] = str(cu[int(np.argmax(cc))])
            best = {"field": field, "score": score, "per_child_dominant": per_child}
    return best


# ----------------------------------------------------------------------------
# EVōC hierarchy navigation (for the zoom cascade)
# ----------------------------------------------------------------------------

def find_root(tree: Dict[Any, List[Any]]) -> Optional[Any]:
    """Return the root node of the EVōC cluster tree (the key that is no one's child)."""
    if not tree:
        return None
    children = {c for kids in tree.values() for c in kids}
    roots = [k for k in tree.keys() if k not in children]
    if roots:
        # Prefer the highest layer index as the coarsest root.
        return max(roots, key=lambda nk: nk[0] if isinstance(nk, (tuple, list)) else 0)
    return max(tree.keys(), key=lambda nk: nk[0] if isinstance(nk, (tuple, list)) else 0)


def node_member_mask(node: Any, layers: List[np.ndarray], n: int) -> np.ndarray:
    """Boolean mask of points belonging to ``node = (layer, cluster_id)``.

    The virtual root sits at ``layer == len(layers)``; its members are all points.
    """
    if not isinstance(node, (tuple, list)) or len(node) != 2:
        return np.ones(n, dtype=bool)
    layer, cid = int(node[0]), int(node[1])
    if layer >= len(layers):
        return np.ones(n, dtype=bool)
    return np.asarray(layers[layer]) == cid


def largest_child(node: Any, tree: Dict[Any, List[Any]], layers: List[np.ndarray], n: int) -> Optional[Any]:
    """Return the child of ``node`` with the most member points, or None if a leaf."""
    children = tree.get(node, [])
    if not children:
        return None
    sizes = [(c, int(node_member_mask(c, layers, n).sum())) for c in children]
    sizes = [(c, s) for c, s in sizes if s > 0]
    if not sizes:
        return None
    return max(sizes, key=lambda cs: cs[1])[0]
