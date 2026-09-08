"""EVōC clustering evaluation task.

Clusters the foundation-model spectrum embeddings *directly in their 768-d space*
using EVōC (Embedding Vector Oriented Clustering, Leland McInnes / Tutte Institute),
then characterises and visualises the resulting clusters.

Unlike the qualitative ``UMAPVisualisationTask`` (which colors a 2-D projection by
metadata), this task provides a quantitative clustering backbone:

* per-point cluster assignments + multi-resolution hierarchy (no 2-D distortion),
* cluster-vs-metadata enrichment (which chemical/instrument features define each cluster),
* a multi-resolution layer panel and a *data-driven* hierarchical zoom cascade that
  reveals what organises the embedding from coarse to fine.

Degrades gracefully (returns a ``skipped`` result) if ``evoc`` is not installed.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from instanovo_fm.eval.embed_eval_tasks import BaseTask
from instanovo_fm.eval.embed_eval_tasks import _clustering_common as cc
from instanovo_fm.eval.embed_eval_tasks.umap_visualisation import UMAPVisualisationTask

logger = logging.getLogger(__name__)


class EVoCClusteringTask(BaseTask):
    """Cluster embeddings with EVōC and visualise the hierarchy + enrichment."""

    name = "EVoC Clustering"
    description = (
        "Cluster 768-d embeddings with EVōC; report cluster-vs-metadata enrichment and "
        "render a multi-resolution layer panel and data-driven hierarchical zoom cascade."
    )
    requires_metadata = True
    requires_faiss = False
    requires_model = False

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the input."""
        super().__init__(**kwargs)
        self.max_samples: int = kwargs.get("max_samples", 20000)
        self.random_state: int = kwargs.get("random_state", 42)
        self.output_dir: Optional[str] = kwargs.get("output_dir", None)
        self.save_dir: Optional[str] = kwargs.get("save_dir", None)
        self.normalize: bool = kwargs.get("normalize", True)

        # EVōC hyperparameters (forwarded to evoc.EVoC; defaults match the library)
        self.evoc_params: Dict[str, Any] = kwargs.get("evoc_params", {})

        # Enrichment fields
        self.categorical_fields: List[str] = kwargs.get("categorical_fields", cc.DEFAULT_CATEGORICAL_FIELDS)
        self.numeric_fields: List[str] = kwargs.get("numeric_fields", cc.DEFAULT_NUMERIC_FIELDS)

        # Hierarchical zoom
        self.enable_zoom_cascade: bool = kwargs.get("enable_zoom_cascade", True)
        self.zoom_candidate_fields: List[str] = kwargs.get(
            "zoom_candidate_fields",
            [
                "frag_type",
                "search_detector",
                "precursor_charge",
                "search_instrument",
                "search_organism",
                "modification_types",
            ],
        )
        self.zoom_depth: int = kwargs.get("zoom_depth", 4)
        self.n_resolution_layers: int = kwargs.get("n_resolution_layers", 4)

        # Per-cluster zoom + recolour by peptide/biological properties.
        # For each of the top-N high-level clusters, crop the UMAP to that cluster and
        # recolour by each property — to see whether peptide-intrinsic structure (mass,
        # length, hydrophobicity, modifications) varies *within* a physical cluster.
        self.enable_cluster_zoom_recolor: bool = kwargs.get("enable_cluster_zoom_recolor", True)
        self.cluster_zoom_top_n: int = kwargs.get("cluster_zoom_top_n", 6)
        self.cluster_zoom_fields: List[str] = kwargs.get(
            "cluster_zoom_fields",
            [
                "precursor_mass",
                "sequence_length",
                "hydrophobicity",
                "modification_types",
                "precursor_mz",
                "precursor_charge",
            ],
        )

        # Silhouette is O(n^2); cap the sample used to compute it.
        self.silhouette_max_samples: int = kwargs.get("silhouette_max_samples", 5000)

        # Figure parameters
        self.dpi: int = kwargs.get("dpi", 200)
        self.point_size: int = kwargs.get("point_size", 5)
        self.alpha: float = kwargs.get("alpha", 0.6)

        # Reused plotting/coloring machinery from the UMAP task (composition, no dup).
        self._viz = UMAPVisualisationTask(
            point_size=self.point_size,
            alpha=self.alpha,
            dpi=self.dpi,
            max_categories=kwargs.get("max_categories", 15),
        )

    # ------------------------------------------------------------------
    # Output dir
    # ------------------------------------------------------------------

    def _create_output_directory(self) -> Path:
        if self.save_dir:
            base_dir = Path(self.save_dir)
        elif self.output_dir:
            base_dir = Path(self.output_dir) / "evoc_figs"
        else:
            base_dir = Path("evaluation_results/evoc_figs")
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    # ------------------------------------------------------------------
    # 2-D projection (for figures only; clustering is done in high-D)
    # ------------------------------------------------------------------

    def _project_2d(self, emb: np.ndarray) -> Optional[np.ndarray]:
        try:
            import umap
        except ImportError:
            logger.warning("umap-learn not installed — skipping EVōC 2-D figures")
            return None
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.1,
            metric="cosine",
            random_state=self.random_state,
            low_memory=True,
        )
        return np.asarray(reducer.fit_transform(emb))

    # ------------------------------------------------------------------
    # Quality metrics
    # ------------------------------------------------------------------

    def _cluster_quality(self, emb: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        mask = labels >= 0
        n_clustered = int(mask.sum())
        if n_clustered < 2 or len(np.unique(labels[mask])) < 2:
            return metrics
        try:
            from sklearn.metrics import silhouette_score

            idx = np.where(mask)[0]
            if len(idx) > self.silhouette_max_samples:
                rng = np.random.default_rng(self.random_state)
                idx = rng.choice(idx, self.silhouette_max_samples, replace=False)
            sub_emb = emb[idx]
            sub_lab = labels[idx]
            if len(np.unique(sub_lab)) >= 2:
                metrics["silhouette"] = float(silhouette_score(sub_emb, sub_lab, metric="cosine"))
        except Exception as e:
            logger.warning("Silhouette computation failed: %s", e)
        return metrics

    # ------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------

    def _plot_clusters_2d(self, umap_2d: np.ndarray, labels: np.ndarray, save_dir: Path) -> Optional[str]:
        """Scatter of the 2-D projection colored by finest EVōC cluster id."""
        fig, ax = plt.subplots(figsize=(10, 8))
        noise = labels < 0
        if noise.any():
            ax.scatter(umap_2d[noise, 0], umap_2d[noise, 1], c="lightgray", s=self.point_size, alpha=0.4, edgecolors="none", label="noise")
        uniq = [c for c in np.unique(labels) if c >= 0]
        cmap = plt.get_cmap("tab20").resampled(max(len(uniq), 1))
        for i, c in enumerate(uniq):
            m = labels == c
            ax.scatter(umap_2d[m, 0], umap_2d[m, 1], c=[cmap(i % cmap.N)], s=self.point_size, alpha=self.alpha, edgecolors="none")
        ax.set_title(f"EVōC clusters (n={len(uniq)}, noise={100 * noise.mean():.0f}%)")
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        path = save_dir / "evoc_clusters_umap.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def _plot_enrichment_heatmap(self, enrichment: Dict[str, Any], save_dir: Path) -> Optional[str]:
        mat, cluster_ids, feat_names = cc.enrichment_effect_matrix(enrichment)
        if mat.size == 0:
            return None
        fig, ax = plt.subplots(figsize=(max(8, len(feat_names) * 0.7), max(4, len(cluster_ids) * 0.4)))
        vmax = float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(feat_names)))
        ax.set_xticklabels(feat_names, rotation=45, ha="right", fontsize=8)
        sizes = enrichment.get("cluster_sizes", {})
        ax.set_yticks(range(len(cluster_ids)))
        ax.set_yticklabels([f"C{c} (n={sizes.get(c, 0)})" for c in cluster_ids], fontsize=8)
        ax.set_title("Cluster × feature enrichment (standardised effect size)")
        fig.colorbar(im, ax=ax, label="(cluster mean − global) / global std")
        path = save_dir / "evoc_enrichment_heatmap.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def _plot_layer_panel(self, umap_2d: np.ndarray, layers: List[np.ndarray], save_dir: Path) -> Optional[str]:
        """Multi-resolution panel: the same projection colored by cluster id at each layer."""
        if not layers:
            return None
        # layers[0] = finest .. show coarsest-first up to n_resolution_layers
        chosen = list(reversed(layers))[: self.n_resolution_layers]
        n = len(chosen)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 5.5), squeeze=False)
        for k, lay in enumerate(chosen):
            ax = axes[0][k]
            lay = np.asarray(lay)
            noise = lay < 0
            if noise.any():
                ax.scatter(umap_2d[noise, 0], umap_2d[noise, 1], c="lightgray", s=self.point_size, alpha=0.3, edgecolors="none")
            uniq = [c for c in np.unique(lay) if c >= 0]
            cmap = plt.get_cmap("tab20").resampled(max(len(uniq), 1))
            for i, c in enumerate(uniq):
                m = lay == c
                ax.scatter(umap_2d[m, 0], umap_2d[m, 1], c=[cmap(i % cmap.N)], s=self.point_size, alpha=self.alpha, edgecolors="none")
            layer_idx = len(layers) - 1 - k
            ax.set_title(f"Layer {layer_idx} ({len(uniq)} clusters)")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("EVōC multi-resolution layers (coarse → fine)", fontsize=13)
        fig.tight_layout()
        path = save_dir / "evoc_layer_panel.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def _slice_meta(self, meta: Dict[str, Any], mask: np.ndarray, n: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, np.ndarray) and v.shape[0] == n:
                out[k] = v[mask]
            elif isinstance(v, list) and len(v) == n:
                out[k] = [v[i] for i in range(n) if mask[i]]
        return out

    def _create_zoom_cascade(
        self,
        umap_2d: np.ndarray,
        result: cc.EVoCResult,
        meta: Dict[str, Any],
        save_dir: Path,
    ) -> Optional[Dict[str, Any]]:
        """Data-driven hierarchical zoom: descend EVōC's tree, auto-labelling each level.

        Each panel crops to the current node and is colored by the metadata field that
        best separates that node's children (max NMI). The story (field + dominant value
        + NMI per level) is returned so the coarse→fine narrative is machine-readable.
        """
        from matplotlib.patches import Rectangle

        n = len(result.labels)
        tree, layers = result.cluster_tree, result.cluster_layers
        root = cc.find_root(tree)
        if root is None:
            return None

        # Walk root -> largest child, recording the discriminative field at each step.
        steps: List[Dict[str, Any]] = []
        node = root
        for _ in range(self.zoom_depth):
            children = tree.get(node, [])
            if len(children) < 2:
                break
            members = cc.node_member_mask(node, layers, n)
            child_idx = np.full(n, -1, dtype=int)
            for ci, child in enumerate(children):
                child_idx[cc.node_member_mask(child, layers, n) & members] = ci
            valid = child_idx >= 0
            if int(valid.sum()) < 2 or len(np.unique(child_idx[valid])) < 2:
                break
            sub_meta = self._slice_meta(meta, members, n)
            sel = cc.select_discriminative_field(child_idx[members], sub_meta, self.zoom_candidate_fields)
            nxt = cc.largest_child(node, tree, layers, n)
            steps.append(
                {
                    "node": node,
                    "members": members,
                    "children": children,
                    "selected": sel,
                    "next": nxt,
                }
            )
            if nxt is None:
                break
            node = nxt

        if not steps:
            return None

        n_panels = len(steps)
        fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5.5), squeeze=False)
        story: List[Dict[str, Any]] = []
        for k, step in enumerate(steps):
            ax = axes[0][k]
            members = step["members"]
            sel = step["selected"]
            coords = umap_2d[members]

            # Color members by the auto-selected field (categorical) when available.
            field = sel["field"] if sel else None
            colored = False
            if field is not None:
                sub_meta = self._slice_meta(meta, members, n)
                cd, clabel, is_cat, labels_, bg = self._viz._get_coloring_data(sub_meta, field, field)
                if cd is not None and is_cat and labels_ is not None:
                    self._viz._plot_categorical(ax, coords, cd, labels_, "tab10", bg)
                    colored = True
                elif cd is not None:
                    ax.scatter(coords[:, 0], coords[:, 1], c=cd, cmap="viridis", s=self.point_size, alpha=self.alpha, edgecolors="none")
                    colored = True
            if not colored:
                ax.scatter(coords[:, 0], coords[:, 1], c="steelblue", s=self.point_size, alpha=self.alpha, edgecolors="none")

            # Crop to this node's bounding box with padding.
            if len(coords) > 0:
                xmin, ymin = coords.min(0)
                xmax, ymax = coords.max(0)
                px, py = 0.05 * (xmax - xmin + 1e-6), 0.05 * (ymax - ymin + 1e-6)
                ax.set_xlim(xmin - px, xmax + px)
                ax.set_ylim(ymin - py, ymax + py)

            # Draw a box around the next node we descend into.
            if step["next"] is not None:
                nxt_mask = cc.node_member_mask(step["next"], layers, n) & members
                if nxt_mask.any():
                    nc = umap_2d[nxt_mask]
                    nx0, ny0 = nc.min(0)
                    nx1, ny1 = nc.max(0)
                    ax.add_patch(Rectangle((nx0, ny0), nx1 - nx0, ny1 - ny0, fill=False, edgecolor="red", lw=1.5, ls="--"))

            score = sel["score"] if sel else 0.0
            ftxt = field if field else "(no separating field)"
            ax.set_title(f"Level {k}: split by {ftxt}\n(NMI={score:.2f}, n={int(members.sum())})", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            story.append(
                {
                    "level": k,
                    "node": list(step["node"]) if isinstance(step["node"], (tuple, list)) else step["node"],
                    "n_members": int(members.sum()),
                    "selected_field": field,
                    "nmi": score,
                    "per_child_dominant": sel["per_child_dominant"] if sel else None,
                }
            )

        fig.suptitle("EVōC hierarchical zoom (data-driven level labels)", fontsize=13)
        fig.tight_layout()
        path = save_dir / "evoc_zoom_cascade.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return {"figure": str(path), "story": story}

    def _create_cluster_zoom_recolor(
        self,
        umap_2d: np.ndarray,
        labels: np.ndarray,
        meta: Dict[str, Any],
        save_dir: Path,
    ) -> Optional[str]:
        """Zoom into each top-N high-level cluster and recolour by peptide properties.

        Grid: rows = the largest EVōC clusters (cropped to each cluster's UMAP region),
        columns = peptide/biological properties (precursor mass, length, hydrophobicity,
        modifications, ...). Reveals whether peptide-intrinsic structure varies *within*
        a physical cluster, without any reclustering. No legends (kept compact); each row
        is labelled with the cluster's dominant instrument / fragmentation.
        """
        n = len(labels)
        meta = dict(meta)
        # Derive neutral precursor_mass from m/z and charge if not already present.
        if meta.get("precursor_mass") is None:
            mz, ch = meta.get("precursor_mz"), meta.get("precursor_charge")
            if mz is not None and ch is not None:
                mz = np.asarray(mz, dtype=float)
                ch = np.asarray(ch, dtype=float)
                with np.errstate(invalid="ignore"):
                    meta["precursor_mass"] = mz * ch - ch * 1.007276

        uniq, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(uniq) == 0:
            return None
        top = [int(uniq[i]) for i in np.argsort(counts)[::-1][: self.cluster_zoom_top_n]]
        fields = self.cluster_zoom_fields
        nrows, ncols = len(top), len(fields)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), squeeze=False)

        def _dominant(arr: Optional[np.ndarray]) -> str:
            if arr is None or len(arr) == 0:
                return "?"
            u, c = np.unique(arr, return_counts=True)
            return str(u[int(c.argmax())])

        for r, cid in enumerate(top):
            cmask = labels == cid
            coords = umap_2d[cmask]
            xmin, ymin = coords.min(0)
            xmax, ymax = coords.max(0)
            px, py = 0.05 * (xmax - xmin + 1e-6), 0.05 * (ymax - ymin + 1e-6)
            sub = self._slice_meta(meta, cmask, n)
            m = int(cmask.sum())
            inst = _dominant(cc.get_categorical_field(sub, "search_instrument", m))
            frag = _dominant(cc.get_categorical_field(sub, "frag_type", m))
            row_lab = f"C{cid} (n={m})\n{inst} / {frag}"
            for j, field in enumerate(fields):
                ax = axes[r][j]
                cd, _clabel, is_cat, labels_, bg = self._viz._get_coloring_data(sub, field, field)
                if cd is None:
                    ax.scatter(coords[:, 0], coords[:, 1], c="lightgray", s=3, alpha=0.4, edgecolors="none")
                    ax.text(0.5, 0.5, f"{field}\n(no data)", transform=ax.transAxes, ha="center", va="center", fontsize=7, color="gray")
                elif is_cat and labels_ is not None:
                    self._viz._plot_categorical(ax, coords, cd, labels_, "tab10", bg)
                else:
                    sc = ax.scatter(coords[:, 0], coords[:, 1], c=cd, cmap="viridis", s=4, alpha=0.6, edgecolors="none")
                    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                ax.set_xlim(xmin - px, xmax + px)
                ax.set_ylim(ymin - py, ymax + py)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(field, fontsize=9)
                if j == 0:
                    ax.set_ylabel(row_lab, fontsize=8)

        fig.suptitle("High-level clusters zoomed in, recoloured by peptide properties", fontsize=13)
        fig.tight_layout()
        path = save_dir / "evoc_cluster_zoom_recolor.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, emb: np.ndarray, meta: Dict[str, np.ndarray], faiss_index: Any = None) -> Dict[str, Any]:  # type: ignore[override]  # base class run() signature differs across tasks
        """Run."""
        start = time.time()
        rng = np.random.default_rng(self.random_state)

        # 1. Subsample (reuse UMAP task's helper) and compute derived descriptors.
        emb_s, meta_s = UMAPVisualisationTask._subsample(emb, meta, self.max_samples, rng)
        UMAPVisualisationTask._compute_spectral_properties(meta_s)
        UMAPVisualisationTask._compute_annotation_properties(meta_s)
        logger.info("EVōC: clustering %d / %d embeddings", len(emb_s), len(emb))

        # 2. Cluster in high-D.
        try:
            result = cc.run_evoc(emb_s, normalize=self.normalize, random_state=self.random_state, **self.evoc_params)
        except ImportError:
            logger.warning("evoc not installed — skipping EVoCClusteringTask")
            return {"task_name": self.name, "skipped": True, "reason": "evoc not installed"}

        labels = result.labels
        logger.info("EVōC: %d clusters, noise=%.1f%%, %d layers", result.n_clusters, 100 * result.noise_fraction, len(result.cluster_layers))

        # 3. Enrichment + quality.
        enrichment = cc.compute_cluster_enrichment(labels, meta_s, self.categorical_fields, self.numeric_fields)
        quality = self._cluster_quality(emb_s, labels)

        # 4. Figures.
        save_dir = self._create_output_directory()
        saved_paths: List[str] = []
        zoom_result: Optional[Dict[str, Any]] = None

        heatmap = self._plot_enrichment_heatmap(enrichment, save_dir)
        if heatmap:
            saved_paths.append(heatmap)

        umap_2d = self._project_2d(emb_s)
        if umap_2d is not None:
            clusters_fig = self._plot_clusters_2d(umap_2d, labels, save_dir)
            if clusters_fig:
                saved_paths.append(clusters_fig)
            layer_fig = self._plot_layer_panel(umap_2d, result.cluster_layers, save_dir)
            if layer_fig:
                saved_paths.append(layer_fig)
            if self.enable_zoom_cascade:
                try:
                    zoom_result = self._create_zoom_cascade(umap_2d, result, meta_s, save_dir)
                    if zoom_result and zoom_result.get("figure"):
                        saved_paths.append(zoom_result["figure"])
                except Exception as e:
                    logger.warning("Zoom cascade failed: %s", e)
            if self.enable_cluster_zoom_recolor:
                try:
                    zr = self._create_cluster_zoom_recolor(umap_2d, labels, meta_s, save_dir)
                    if zr:
                        saved_paths.append(zr)
                except Exception as e:
                    logger.warning("Cluster zoom-recolor failed: %s", e)

        # 5. Dump enrichment + zoom story as JSON for downstream analysis.
        try:
            with open(save_dir / "evoc_enrichment.json", "w") as f:
                json.dump(
                    {"enrichment": enrichment, "zoom_story": zoom_result["story"] if zoom_result else None},
                    f,
                    indent=2,
                    default=str,
                )
        except Exception as e:
            logger.warning("Failed to dump enrichment JSON: %s", e)

        # Per-cluster mean purity vs frag_type (a convenient single-number summary).
        mean_purity_frag = None
        frag = enrichment.get("categorical", {}).get("frag_type")
        if frag and frag["clusters"]:
            mean_purity_frag = float(np.mean([c["purity"] for c in frag["clusters"].values()]))

        return {
            "task_name": self.name,
            "skipped": False,
            "num_embeddings": int(len(emb)),
            "num_sampled": int(len(emb_s)),
            "n_clusters": result.n_clusters,
            "noise_fraction": result.noise_fraction,
            "n_layers": len(result.cluster_layers),
            "n_duplicates": len(result.duplicates),
            "cluster_sizes": enrichment.get("cluster_sizes", {}),
            "quality_metrics": quality,
            "enrichment": enrichment,
            "zoom_story": zoom_result["story"] if zoom_result else None,
            "mean_purity_frag_type": mean_purity_frag,
            "save_paths": saved_paths,
            "save_dir": str(save_dir),
            "execution_time": time.time() - start,
            "evoc_params": self.evoc_params,
        }

    # ------------------------------------------------------------------
    # Loggable metrics
    # ------------------------------------------------------------------

    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """Return loggable metrics."""
        if task_results.get("skipped"):
            return {}
        metrics: Dict[str, float] = {
            "n_clusters": float(task_results.get("n_clusters", 0)),
            "noise_fraction": float(task_results.get("noise_fraction", 0.0)),
        }
        q = task_results.get("quality_metrics", {})
        if "silhouette" in q:
            metrics["silhouette"] = float(q["silhouette"])
        if task_results.get("mean_purity_frag_type") is not None:
            metrics["mean_purity_frag_type"] = float(task_results["mean_purity_frag_type"])
        return metrics
