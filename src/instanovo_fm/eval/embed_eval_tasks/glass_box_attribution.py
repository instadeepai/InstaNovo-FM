"""Glass Box UMAP feature-attribution evaluation task.

Glass Box UMAP (Arcadia Science) is a parametric UMAP whose locally-linear encoder
yields *exact per-feature contributions*: ``compute_contributions(X)`` returns an
array of shape ``(N, n_components, n_features)`` that sums over features back to the
2-D embedding coordinates.

Crucial constraint: Glass Box attributes onto the **input feature columns**, so it is
only interpretable when those columns are interpretable.  Fed the opaque 768-d model
vectors it would attribute to meaningless latent dims.  Therefore this task fits Glass
Box on a matrix of **interpretable chemical descriptors** (precursor m/z, charge,
collision energy, n_peaks, hydrophobicity, …) built from metadata.

The learned model enters via the bridge: EVōC clusters the *768-d learned embeddings*,
and those cluster labels are overlaid on the descriptor-space attribution to answer
"which chemical descriptors characterise each model-discovered cluster".

Two figure families are produced:
  1. Attribution-specific: global feature-importance bar, Glass Box embedding colored by
     EVōC cluster, and a per-cluster feature-importance heatmap.
  2. The *same* metadata-colored visualisations as the usual UMAP task, but drawn on the
     Glass Box embedding (saved under ``glassbox_figs/``) — a parallel, directly
     comparable set alongside (not replacing) the standard UMAP outputs.

Degrades gracefully (returns ``skipped``) if ``glass-box-umap`` is not installed.
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


class GlassBoxAttributionTask(BaseTask):
    """Interpretable UMAP attribution over chemical descriptors, bridged to EVōC clusters."""

    name = "Glass Box Attribution"
    description = (
        "Fit Glass Box UMAP on interpretable spectrum descriptors to compute exact "
        "per-feature contributions; overlay EVōC clusters and render the usual "
        "metadata-colored views on the Glass Box embedding."
    )
    requires_metadata = True
    requires_faiss = False
    requires_model = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.max_samples: int = kwargs.get("max_samples", 20000)
        self.random_state: int = kwargs.get("random_state", 42)
        self.output_dir: Optional[str] = kwargs.get("output_dir", None)
        self.save_dir: Optional[str] = kwargs.get("save_dir", None)

        self.numeric_features: List[str] = kwargs.get("numeric_features", cc.DEFAULT_NUMERIC_FIELDS)

        # Glass Box hyperparameters (forwarded to GlassBoxUMAP).
        self.glassbox_params: Dict[str, Any] = kwargs.get(
            "glassbox_params",
            {"n_neighbors": 30, "min_dist": 0.1, "n_components": 2, "epochs": 200},
        )

        # EVōC bridge
        self.normalize_emb: bool = kwargs.get("normalize", True)
        self.evoc_params: Dict[str, Any] = kwargs.get("evoc_params", {})

        # Whether to also render the full metadata-colored set on the Glass Box embedding.
        self.create_metadata_views: bool = kwargs.get("create_metadata_views", True)

        self.dpi: int = kwargs.get("dpi", 200)
        self.point_size: int = kwargs.get("point_size", 5)
        self.alpha: float = kwargs.get("alpha", 0.6)

        # Reused plotting/coloring machinery from the UMAP task (composition, no dup).
        self._viz = UMAPVisualisationTask(
            point_size=self.point_size,
            alpha=self.alpha,
            dpi=self.dpi,
            max_categories=kwargs.get("max_categories", 15),
            create_summary_panel=True,
        )

    def _create_output_directory(self) -> Path:
        if self.save_dir:
            base_dir = Path(self.save_dir)
        elif self.output_dir:
            base_dir = Path(self.output_dir) / "glassbox_figs"
        else:
            base_dir = Path("evaluation_results/glassbox_figs")
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    # ------------------------------------------------------------------
    # Attribution figures
    # ------------------------------------------------------------------

    def _plot_global_importance(self, importance: np.ndarray, names: List[str], save_dir: Path) -> str:
        order = np.argsort(importance)[::-1]
        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(names))))
        ax.barh([names[i] for i in order][::-1], importance[order][::-1], color="steelblue")
        ax.set_xlabel("Mean |contribution| (L2 over UMAP axes)")
        ax.set_title("Glass Box global feature importance")
        path = save_dir / "glassbox_global_importance.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def _plot_embedding_by_cluster(self, emb2d: np.ndarray, labels: Optional[np.ndarray], save_dir: Path) -> str:
        fig, ax = plt.subplots(figsize=(10, 8))
        if labels is None:
            ax.scatter(emb2d[:, 0], emb2d[:, 1], c="steelblue", s=self.point_size,
                       alpha=self.alpha, edgecolors="none")
        else:
            noise = labels < 0
            if noise.any():
                ax.scatter(emb2d[noise, 0], emb2d[noise, 1], c="lightgray",
                           s=self.point_size, alpha=0.4, edgecolors="none")
            uniq = [c for c in np.unique(labels) if c >= 0]
            cmap = plt.cm.get_cmap("tab20", max(len(uniq), 1))
            for i, c in enumerate(uniq):
                m = labels == c
                ax.scatter(emb2d[m, 0], emb2d[m, 1], c=[cmap(i % cmap.N)],
                           s=self.point_size, alpha=self.alpha, edgecolors="none")
        ax.set_title("Glass Box embedding (descriptors) colored by EVōC cluster")
        ax.set_xlabel("GlassBox-1"); ax.set_ylabel("GlassBox-2")
        path = save_dir / "glassbox_embedding_by_evoc_cluster.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def _plot_cluster_importance_heatmap(
        self, per_sample_importance: np.ndarray, labels: np.ndarray, names: List[str], save_dir: Path
    ) -> Optional[str]:
        uniq = [int(c) for c in np.unique(labels) if c >= 0]
        if not uniq:
            return None
        mat = np.stack([per_sample_importance[labels == c].mean(0) for c in uniq], axis=0)
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.7), max(4, len(uniq) * 0.4)))
        im = ax.imshow(mat, aspect="auto", cmap="viridis")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(uniq)))
        ax.set_yticklabels([f"C{c}" for c in uniq], fontsize=8)
        ax.set_title("Per-cluster mean feature contribution (descriptors)")
        fig.colorbar(im, ax=ax, label="mean |contribution|")
        path = save_dir / "glassbox_cluster_importance_heatmap.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return str(path)

    def _render_metadata_views(self, emb2d: np.ndarray, meta: Dict[str, Any], save_dir: Path) -> List[str]:
        """Render the usual UMAP metadata-colored set on the Glass Box embedding."""
        views_dir = save_dir / "metadata_views"
        views_dir.mkdir(parents=True, exist_ok=True)
        paths: List[str] = []
        for cfg in self._viz.visualization_configs:
            p = self._viz._create_single_visualization(emb2d, meta, cfg, views_dir)
            if p:
                paths.append(p)
        panel = self._viz._create_summary_panel(emb2d, meta, views_dir)
        if panel:
            paths.append(panel)
        return paths

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, emb: np.ndarray, meta: Dict[str, np.ndarray], faiss_index: Any = None) -> Dict[str, Any]:
        start = time.time()
        rng = np.random.default_rng(self.random_state)

        emb_s, meta_s = UMAPVisualisationTask._subsample(emb, meta, self.max_samples, rng)
        # Derived descriptors + coloring fields (needed by both the matrix and the views).
        UMAPVisualisationTask._compute_spectral_properties(meta_s)
        UMAPVisualisationTask._compute_annotation_properties(meta_s)
        UMAPVisualisationTask._compute_top_duplicate_peptides(meta_s)

        X, feat_names = cc.build_descriptor_matrix(meta_s, self.numeric_features, n=len(emb_s))
        if X.shape[1] < 2:
            logger.warning("Glass Box: only %d usable descriptors — skipping", X.shape[1])
            return {"task_name": self.name, "skipped": True,
                    "reason": f"insufficient interpretable descriptors ({X.shape[1]})"}

        try:
            from glass_box_umap import GlassBoxUMAP
        except ImportError:
            logger.warning("glass-box-umap not installed — skipping GlassBoxAttributionTask")
            return {"task_name": self.name, "skipped": True, "reason": "glass-box-umap not installed"}

        logger.info("Glass Box: fitting on %d samples x %d descriptors", X.shape[0], X.shape[1])
        params = dict(self.glassbox_params)
        params.setdefault("random_state", self.random_state)
        params.setdefault("quiet", True)
        model = GlassBoxUMAP(**params)
        emb2d = np.asarray(model.fit_transform(X))
        contrib = np.asarray(model.compute_contributions(X))  # (N, n_components, n_features)

        # Reconstruction sanity check: sum over features == embedding.
        recon = contrib.sum(axis=2)
        recon_residual = float(np.abs(recon - emb2d).max())
        logger.info("Glass Box: reconstruction residual = %.2e", recon_residual)

        # Per-sample feature importance (L2 over UMAP axes) and global importance.
        per_sample_importance = np.linalg.norm(contrib, axis=1)  # (N, n_features)
        global_importance = per_sample_importance.mean(axis=0)   # (n_features,)

        # Bridge: cluster the *learned* embeddings with EVōC.
        labels: Optional[np.ndarray] = None
        try:
            evoc_res = cc.run_evoc(
                emb_s, normalize=self.normalize_emb, random_state=self.random_state, **self.evoc_params
            )
            labels = evoc_res.labels
        except ImportError:
            logger.warning("evoc not installed — Glass Box runs without cluster overlay")

        # Figures.
        save_dir = self._create_output_directory()
        saved_paths: List[str] = [
            self._plot_global_importance(global_importance, feat_names, save_dir),
            self._plot_embedding_by_cluster(emb2d, labels, save_dir),
        ]
        cluster_importance: Optional[Dict[str, Any]] = None
        if labels is not None:
            hm = self._plot_cluster_importance_heatmap(per_sample_importance, labels, feat_names, save_dir)
            if hm:
                saved_paths.append(hm)
            uniq = [int(c) for c in np.unique(labels) if c >= 0]
            cluster_importance = {
                int(c): {feat_names[j]: float(per_sample_importance[labels == c].mean(0)[j])
                         for j in range(len(feat_names))}
                for c in uniq
            }

        if self.create_metadata_views:
            saved_paths.extend(self._render_metadata_views(emb2d, meta_s, save_dir))

        # Dump attribution summary.
        top_idx = int(np.argmax(global_importance))
        try:
            with open(save_dir / "glassbox_attribution.json", "w") as f:
                json.dump({
                    "feature_names": feat_names,
                    "global_importance": {feat_names[i]: float(global_importance[i])
                                          for i in range(len(feat_names))},
                    "top_feature": feat_names[top_idx],
                    "reconstruction_residual": recon_residual,
                    "per_cluster_importance": cluster_importance,
                }, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to dump attribution JSON: %s", e)

        return {
            "task_name": self.name,
            "skipped": False,
            "num_embeddings": int(len(emb)),
            "num_sampled": int(len(emb_s)),
            "n_features": int(len(feat_names)),
            "feature_names": feat_names,
            "top_feature": feat_names[top_idx],
            "top_feature_importance": float(global_importance[top_idx]),
            "global_importance": {feat_names[i]: float(global_importance[i]) for i in range(len(feat_names))},
            "reconstruction_residual": recon_residual,
            "per_cluster_importance": cluster_importance,
            "save_paths": saved_paths,
            "save_dir": str(save_dir),
            "execution_time": time.time() - start,
        }

    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        if task_results.get("skipped"):
            return {}
        metrics: Dict[str, float] = {
            "n_features": float(task_results.get("n_features", 0)),
            "reconstruction_residual": float(task_results.get("reconstruction_residual", 0.0)),
        }
        if task_results.get("top_feature_importance") is not None:
            metrics["top_feature_importance"] = float(task_results["top_feature_importance"])
        return metrics
