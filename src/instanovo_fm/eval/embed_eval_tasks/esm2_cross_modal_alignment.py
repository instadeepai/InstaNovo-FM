"""ESM2 Cross-Modal Alignment evaluation task.

Evaluates whether spectrum embedding space preserves neighbourhood structure
that aligns with protein sequence space (ESM2) using parameter-free metrics:

  - **RSA** (Representational Similarity Analysis): Spearman correlation between
    pairwise cosine similarities in spectrum space vs ESM2 space.  The headline
    metric — handles different dimensionalities, no fitting required.
  - **CKA** (Centered Kernel Alignment): Kernel-level representational similarity,
    invariant to orthogonal transforms and isotropic scaling.

Baselines isolate confounds:
  - Shuffled (permuted spectrum embeddings) — RSA should drop to ~0
  - Metadata-only (seq_length + charge + mass pairwise distances) — how much
    of the correlation comes from trivial scalar features?

Includes an ESM2 sanity check: mean pairwise cosine and effective rank to
verify ESM2 embeddings are not degenerate for short tryptic peptides.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from instanovo_fm.eval.embed_eval_tasks import BaseTask

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _clean_sequence(seq: str) -> str:
    """Strip modification annotations, keep only amino acid letters."""
    return "".join(c for c in seq if c.isalpha()).upper()


def _linear_cka(X: np.ndarray, Y: np.ndarray) -> float:  # noqa: N803
    """Linear Centered Kernel Alignment (Kornblith et al., 2019)."""
    X = X - X.mean(axis=0)  # noqa: N806
    Y = Y - Y.mean(axis=0)  # noqa: N806
    hsic_xy = np.sum((X.T @ Y) ** 2)
    hsic_xx = np.sum((X.T @ X) ** 2)
    hsic_yy = np.sum((Y.T @ Y) ** 2)
    denom = np.sqrt(hsic_xx * hsic_yy)
    return float(hsic_xy / denom) if denom > 1e-12 else 0.0


def _pairwise_cosine(emb: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray) -> np.ndarray:
    """Cosine similarity for specified pairs of rows."""
    n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    return (n[idx_i] * n[idx_j]).sum(axis=1)


def _sample_pairs(
    n: int,
    max_pairs: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample unique (i, j) pairs with i < j."""
    total = n * (n - 1) // 2
    if total <= max_pairs:
        return np.triu_indices(n, k=1)  # type: ignore[no-any-return]
    idx_i = rng.randint(0, n, size=int(max_pairs * 1.1))
    idx_j = rng.randint(0, n, size=int(max_pairs * 1.1))
    mask = idx_i < idx_j
    return idx_i[mask][:max_pairs], idx_j[mask][:max_pairs]


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------


class ESM2CrossModalAlignmentTask(BaseTask):
    """Representational similarity between spectrum embeddings and ESM2.

    Uses RSA (pairwise similarity correlation) and CKA as parameter-free
    cross-modal alignment metrics.  No learned projection, no linearity
    assumption.
    """

    name = "ESM2 Cross-Modal Alignment"
    description = "RSA and CKA between spectrum embeddings and ESM2 sequence embeddings"
    requires_metadata = True
    requires_faiss = False

    def __init__(self, **kwargs: Any) -> None:
        """Initialise the input."""
        super().__init__(**kwargs)
        self.max_samples: int = kwargs.get("max_samples", 5_000)
        self.esm2_model: str = kwargs.get("esm2_model", "esm2_t33_650M_UR50D")
        self.esm2_batch_size: int = kwargs.get("esm2_batch_size", 32)
        self.esm2_pooling: str = kwargs.get("esm2_pooling", "mean")
        self.random_state: int = kwargs.get("random_state", 42)
        self.min_seq_len: int = kwargs.get("min_sequence_length", 7)
        self.max_seq_len: int = kwargs.get("max_sequence_length", 50)
        self.max_pairs: int = kwargs.get("max_pairs", 100_000)
        self.device: Optional[str] = kwargs.get("device", None)
        self.create_plots: bool = kwargs.get("create_plots", True)
        self.dpi: int = kwargs.get("dpi", 300)
        self.output_dir: Optional[str] = kwargs.get("output_dir", None)

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------
    def run(
        self,
        emb: np.ndarray,
        meta: Dict[str, np.ndarray],
        faiss_index: Any,
        model: Any = None,
        dataloader: Any = None,
        config: Any = None,
        device: Any = None,
    ) -> Dict[str, Any]:
        """Run."""
        t0 = time.time()

        # --- 1. Sequences ---
        sequences = self._get_sequences(meta)
        if sequences is None:
            return {"error": "No peptide sequences found in metadata"}

        valid_mask = self._build_valid_mask(sequences)
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) < 50:
            return {"error": f"Only {len(valid_indices)} valid sequences (need >= 50)"}

        rng = np.random.RandomState(self.random_state)
        if len(valid_indices) > self.max_samples:
            valid_indices = rng.choice(valid_indices, self.max_samples, replace=False)

        spec_emb = emb[valid_indices]
        seqs = [sequences[i] for i in valid_indices]
        clean_seqs = [_clean_sequence(s) for s in seqs]
        meta_indices = valid_indices
        logger.info(f"ESM2 alignment: {len(clean_seqs)} spectra after filtering")

        # --- 2. ESM2 embeddings (unique sequences only) ---
        try:
            from instanovo_fm.eval.embed_eval_tasks.esm2_embedder import ESM2Embedder
        except ImportError:
            return {"error": "ESM2 not available (pip install fair-esm)"}

        unique_seqs = sorted(set(clean_seqs))
        logger.info(f"Generating ESM2 embeddings for {len(unique_seqs)} unique peptides ({len(clean_seqs)} spectra) with {self.esm2_model}...")
        embedder = ESM2Embedder(model_name=self.esm2_model, device=self.device)
        unique_esm2 = embedder.embed(
            unique_seqs,
            batch_size=self.esm2_batch_size,
            pooling=self.esm2_pooling,
        )

        seq_to_esm2: Dict[str, np.ndarray] = {}
        for seq, vec in zip(unique_seqs, unique_esm2, strict=False):
            if np.linalg.norm(vec) > 1e-8:
                seq_to_esm2[seq] = vec

        keep = [i for i, s in enumerate(clean_seqs) if s in seq_to_esm2]
        if len(keep) < 50:
            return {"error": f"Only {len(keep)} spectra with valid ESM2 embeddings"}

        spec_emb = spec_emb[keep]
        clean_seqs = [clean_seqs[i] for i in keep]
        meta_indices = valid_indices[np.array(keep)]
        esm2_per_spectrum = np.stack([seq_to_esm2[s] for s in clean_seqs])

        n_spectra = len(clean_seqs)
        n_unique = len(set(clean_seqs))
        logger.info(f"Spectra: {n_spectra}, unique peptides: {n_unique}")

        # --- 3. ESM2 sanity check ---
        esm2_sanity = self._esm2_sanity_check(seq_to_esm2)

        # --- 4. Sample pairs ---
        idx_i, idx_j = _sample_pairs(n_spectra, self.max_pairs, rng)
        same_pep = np.array([clean_seqs[a] == clean_seqs[b] for a, b in zip(idx_i, idx_j, strict=False)])
        logger.info(f"Sampled {len(idx_i)} pairs ({same_pep.sum()} same-peptide, {(~same_pep).sum()} different-peptide)")

        # --- 5. Pairwise similarities ---
        sim_spec = _pairwise_cosine(spec_emb, idx_i, idx_j)
        sim_esm2 = _pairwise_cosine(esm2_per_spectrum, idx_i, idx_j)

        # --- 6. RSA: model (all pairs and different-peptide-only) ---
        from scipy.stats import spearmanr

        rsa_all, rsa_all_p = spearmanr(sim_spec, sim_esm2)
        rsa_all, rsa_all_p = float(rsa_all), float(rsa_all_p)

        diff_mask = ~same_pep
        rsa_diff, rsa_diff_p = spearmanr(sim_spec[diff_mask], sim_esm2[diff_mask])
        rsa_diff, rsa_diff_p = float(rsa_diff), float(rsa_diff_p)
        logger.info(f"RSA (model): all_pairs rho={rsa_all:.4f}, diff_peptide_only rho={rsa_diff:.4f} (p={rsa_diff_p:.2e})")

        # --- 7. RSA: shuffled baseline ---
        shuf_order = rng.permutation(n_spectra)
        sim_spec_shuf = _pairwise_cosine(spec_emb[shuf_order], idx_i, idx_j)
        rsa_shuf_all, _ = spearmanr(sim_spec_shuf, sim_esm2)
        rsa_shuf_diff, _ = spearmanr(sim_spec_shuf[diff_mask], sim_esm2[diff_mask])
        rsa_shuf_all, rsa_shuf_diff = float(rsa_shuf_all), float(rsa_shuf_diff)
        logger.info(f"RSA (shuffled): all={rsa_shuf_all:.4f}, diff_only={rsa_shuf_diff:.4f}")

        # --- 8. RSA: metadata baseline ---
        meta_features = self._build_metadata_features(clean_seqs, meta, meta_indices)
        rsa_meta_all = rsa_meta_diff = None
        if meta_features is not None:
            from sklearn.preprocessing import StandardScaler

            mf = StandardScaler().fit_transform(meta_features)
            sim_meta = _pairwise_cosine(mf, idx_i, idx_j)
            rsa_meta_all, _ = spearmanr(sim_meta, sim_esm2)
            rsa_meta_diff, _ = spearmanr(sim_meta[diff_mask], sim_esm2[diff_mask])
            rsa_meta_all, rsa_meta_diff = float(rsa_meta_all), float(rsa_meta_diff)
            logger.info(f"RSA (metadata): all={rsa_meta_all:.4f}, diff_only={rsa_meta_diff:.4f}")

        # --- 9. CKA ---
        cka_model = _linear_cka(spec_emb, esm2_per_spectrum)
        cka_shuf = _linear_cka(spec_emb[shuf_order], esm2_per_spectrum)
        cka_meta = _linear_cka(meta_features, esm2_per_spectrum) if meta_features is not None else None
        logger.info(f"CKA: model={cka_model:.4f}, shuffled={cka_shuf:.4f}, metadata={cka_meta}")

        # --- 10. Figure ---
        figure_path = None
        if self.create_plots and self.output_dir:
            figure_path = self._create_figure(
                sim_spec=sim_spec,
                sim_esm2=sim_esm2,
                same_pep=same_pep,
                diff_mask=diff_mask,
                rsa_diff=rsa_diff,
                rsa_shuf_diff=rsa_shuf_diff,
                rsa_meta_diff=rsa_meta_diff,
                cka_model=cka_model,
                cka_shuf=cka_shuf,
                cka_meta=cka_meta,
                esm2_sanity=esm2_sanity,
            )

        elapsed = time.time() - t0
        logger.info(
            f"ESM2 alignment done in {elapsed:.1f}s -- "
            f"RSA_diff={rsa_diff:.4f}  RSA_all={rsa_all:.4f}  CKA={cka_model:.4f}  "
            f"RSA_meta_diff={rsa_meta_diff}  RSA_shuf_diff={rsa_shuf_diff:.4f}"
        )

        results: Dict[str, Any] = {
            # --- headline (different-peptide pairs only — the stringent metric) ---
            "rsa_diff_peptide_rho": rsa_diff,
            "rsa_diff_peptide_p": rsa_diff_p,
            # --- all-pairs RSA (includes same-peptide inflation) ---
            "rsa_all_pairs_rho": rsa_all,
            "rsa_all_pairs_p": rsa_all_p,
            "cka_score": cka_model,
            # --- baselines (different-peptide only) ---
            "baseline_shuffled_rsa_diff": rsa_shuf_diff,
            "baseline_shuffled_cka": cka_shuf,
            "baseline_metadata_rsa_diff": rsa_meta_diff,
            "baseline_metadata_cka": cka_meta,
            # --- baselines (all pairs, for reference) ---
            "baseline_shuffled_rsa_all": rsa_shuf_all,
            "baseline_metadata_rsa_all": rsa_meta_all,
            # --- ESM2 sanity ---
            **{f"esm2_{k}": v for k, v in esm2_sanity.items()},
            # --- pair stats ---
            "n_pairs": len(idx_i),
            "n_same_peptide_pairs": int(same_pep.sum()),
            "n_diff_peptide_pairs": int((~same_pep).sum()),
            # --- dataset ---
            "n_spectra": n_spectra,
            "n_unique_peptides": n_unique,
            # --- setup ---
            "esm2_model": self.esm2_model,
            "esm2_embedding_dim": esm2_per_spectrum.shape[1],
            "spectrum_embedding_dim": spec_emb.shape[1],
            "execution_time_seconds": elapsed,
        }
        if figure_path:
            results["figure_path"] = str(figure_path)
        return results

    # ------------------------------------------------------------------
    # ESM2 sanity check
    # ------------------------------------------------------------------
    def _esm2_sanity_check(self, seq_to_esm2: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Verify ESM2 embeddings are not degenerate for short peptides."""
        unique_peps = list(seq_to_esm2.keys())
        esm2_mat = np.stack([seq_to_esm2[p] for p in unique_peps])
        norms = np.linalg.norm(esm2_mat, axis=1, keepdims=True)
        esm2_n = esm2_mat / (norms + 1e-12)

        # Pairwise cosine (sample if large)
        rng = np.random.RandomState(42)
        n = len(unique_peps)
        sample = esm2_n[rng.choice(n, min(n, 500), replace=False)] if n > 500 else esm2_n
        sim = sample @ sample.T
        mask = ~np.eye(len(sample), dtype=bool)
        mean_sim = float(sim[mask].mean())
        std_sim = float(sim[mask].std())

        # Effective rank
        _, s, _ = np.linalg.svd(esm2_mat - esm2_mat.mean(axis=0), full_matrices=False)
        p = s / s.sum()
        p = p[p > 1e-12]
        effective_rank = float(np.exp(-np.sum(p * np.log(p))))

        logger.info(f"ESM2 sanity: mean_pairwise_cosine={mean_sim:.3f} (std={std_sim:.3f}), effective_rank={effective_rank:.1f}/{esm2_mat.shape[1]}")
        if mean_sim > 0.95:
            logger.warning(f"ESM2 embeddings near-degenerate (mean cosine={mean_sim:.3f}). Short peptides may not be well-separated in ESM2 space.")

        return {
            "mean_pairwise_cosine": mean_sim,
            "std_pairwise_cosine": std_sim,
            "effective_rank": effective_rank,
            "n_unique_peptides": len(unique_peps),
        }

    # ------------------------------------------------------------------
    # Metadata features
    # ------------------------------------------------------------------
    def _build_metadata_features(
        self,
        clean_seqs: List[str],
        meta: Dict[str, np.ndarray],
        indices: np.ndarray,
    ) -> Optional[np.ndarray]:
        cols: List[np.ndarray] = []
        cols.append(np.array([len(s) for s in clean_seqs], dtype=np.float64))
        for key in ["precursor_charge", "charge"]:
            if key in meta:
                cols.append(np.asarray(meta[key], dtype=np.float64)[indices])
                break
        for key in ["precursor_mass", "precursor_mz"]:
            if key in meta:
                cols.append(np.asarray(meta[key], dtype=np.float64)[indices])
                break
        if not cols:
            return None
        features = np.column_stack(cols)
        nan_mask = np.isnan(features).any(axis=1)
        if nan_mask.all():
            return None
        if nan_mask.any():
            features[nan_mask] = np.nanmean(features, axis=0)
        return features

    # ------------------------------------------------------------------
    # Figure (2x2)
    # ------------------------------------------------------------------
    def _create_figure(
        self,
        sim_spec: np.ndarray,
        sim_esm2: np.ndarray,
        same_pep: np.ndarray,
        diff_mask: np.ndarray,
        rsa_diff: float,
        rsa_shuf_diff: float,
        rsa_meta_diff: Optional[float],
        cka_model: float,
        cka_shuf: float,
        cka_meta: Optional[float],
        esm2_sanity: Dict[str, float],
    ) -> Optional[Path]:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import LogNorm
        except ImportError:
            return None

        out_dir = Path(self.output_dir)  # type: ignore[arg-type]
        out_dir.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), gridspec_kw={"hspace": 0.35, "wspace": 0.35})

        # Use different-peptide pairs only for panels A and B (the honest analysis)
        sim_spec_d = sim_spec[diff_mask]
        sim_esm2_d = sim_esm2[diff_mask]

        # ---- Panel A: RSA hexbin (different-peptide pairs, zoomed to data) ----
        ax = axes[0, 0]
        # Zoom to actual data range
        x_lo = max(np.percentile(sim_esm2_d, 0.5) - 0.02, -0.3)
        x_hi = min(np.percentile(sim_esm2_d, 99.5) + 0.02, 1.05)
        y_lo = max(np.percentile(sim_spec_d, 0.5) - 0.02, -0.3)
        y_hi = min(np.percentile(sim_spec_d, 99.5) + 0.02, 1.05)
        hb = ax.hexbin(
            sim_esm2_d,
            sim_spec_d,
            gridsize=50,
            cmap="Blues",
            mincnt=1,
            norm=LogNorm(),
            extent=(x_lo, x_hi, y_lo, y_hi),
        )
        fig.colorbar(hb, ax=ax, label="Pair count (log)", shrink=0.8)
        ax.set_xlabel("ESM2 pairwise cosine")
        ax.set_ylabel("Spectrum pairwise cosine")
        ax.set_title(
            f"A.  RSA (diff. peptide pairs): $\\rho$ = {rsa_diff:.3f}",
            fontweight="bold",
            loc="left",
        )

        # ---- Panel B: Violin — spectrum similarity by ESM2 similarity bin ----
        ax = axes[0, 1]
        n_bins = 8
        esm2_bins = np.linspace(sim_esm2_d.min() - 0.001, sim_esm2_d.max() + 0.001, n_bins + 1)
        bin_positions = []
        bin_data = []
        for b in range(n_bins):
            m = (sim_esm2_d >= esm2_bins[b]) & (sim_esm2_d < esm2_bins[b + 1])
            if m.sum() >= 10:
                bin_data.append(sim_spec_d[m])
                bin_positions.append((esm2_bins[b] + esm2_bins[b + 1]) / 2)

        if bin_data:
            bin_width = (bin_positions[-1] - bin_positions[0]) / len(bin_positions) * 0.8
            vp = ax.violinplot(
                bin_data,
                positions=bin_positions,
                widths=bin_width,
                showmedians=True,
                showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor("#2171b5")
                body.set_alpha(0.5)
            vp["cmedians"].set_color("#cb181d")

            # Median trend line
            medians = [np.median(d) for d in bin_data]
            ax.plot(bin_positions, medians, "o-", color="#cb181d", linewidth=1.5, markersize=4, zorder=5, label="median")

            # Global median reference
            global_median = np.median(sim_spec_d)
            ax.axhline(global_median, color="#636363", linewidth=1, linestyle=":", label=f"global median ({global_median:.2f})")
            ax.legend(fontsize=8, loc="lower right")

        ax.set_xlabel("ESM2 pairwise cosine (binned)")
        ax.set_ylabel("Spectrum pairwise cosine")
        ax.set_title("B.  Spectrum similarity by ESM2 similarity", fontweight="bold", loc="left")

        # ---- Panel C: ESM2 pairwise cosine distribution ----
        ax = axes[1, 0]
        ax.hist(sim_esm2_d, bins=60, alpha=0.7, color="#bdbdbd", label="diff. peptide pairs", edgecolor="white", linewidth=0.3)
        if same_pep.any():
            same_vals = sim_esm2[same_pep]
            if np.std(same_vals) < 0.01:
                ax.axvline(same_vals.mean(), color="#2171b5", linewidth=2, label=f"same peptide (cosine = {same_vals.mean():.2f})")
            else:
                ax.hist(same_vals, bins=30, alpha=0.8, color="#2171b5", label="same peptide", edgecolor="white", linewidth=0.3)
        ax.set_xlabel("ESM2 pairwise cosine similarity")
        ax.set_ylabel("Pair count")
        ax.set_title("C.  ESM2 embedding structure", fontweight="bold", loc="left")
        ax.legend(fontsize=8)
        ax.text(
            0.03,
            0.95,
            f"mean cosine = {esm2_sanity['mean_pairwise_cosine']:.3f}\neff. rank = {esm2_sanity['effective_rank']:.0f} / 1280",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#cccccc", "alpha": 0.9},
        )

        # ---- Panel D: Baseline comparison ----
        ax = axes[1, 1]
        # Use diff-peptide RSA as the primary metric
        conditions = ["Shuffled"]
        rsa_vals = [rsa_shuf_diff]
        cka_vals = [cka_shuf]
        bar_colors = ["#bdbdbd"]

        if rsa_meta_diff is not None:
            conditions.append("Metadata")
            rsa_vals.append(rsa_meta_diff)
            cka_vals.append(cka_meta)  # type: ignore[arg-type]
            bar_colors.append("#fdae6b")

        conditions.append("Model")
        rsa_vals.append(rsa_diff)
        cka_vals.append(cka_model)
        bar_colors.append("#2171b5")

        x = np.arange(len(conditions))
        width = 0.32

        bars_rsa = ax.bar(x - width / 2, rsa_vals, width, label="RSA $\\rho$ (diff. pep.)", color=bar_colors, edgecolor="white", linewidth=0.5)
        bars_cka = ax.bar(
            x + width / 2, cka_vals, width, label="CKA (all spectra)", color=bar_colors, edgecolor="white", linewidth=0.5, alpha=0.5, hatch="///"
        )

        for bars in [bars_rsa, bars_cka]:
            for bar in bars:
                y = bar.get_height()
                va = "bottom" if y >= 0 else "top"
                offset = 0.003 if y >= 0 else -0.003
                ax.text(bar.get_x() + bar.get_width() / 2, y + offset, f"{y:.3f}", ha="center", va=va, fontsize=7, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylabel("Score")
        ax.set_title("D.  Baseline comparison", fontweight="bold", loc="left")
        ax.legend(fontsize=8, loc="upper left")
        ax.axhline(0, color="black", linewidth=0.5)
        all_vals = rsa_vals + cka_vals
        ymin = min(min(all_vals), 0) - 0.02
        ymax = max(all_vals) + 0.04
        ax.set_ylim(ymin, ymax)

        fig.suptitle(
            f"ESM2 Cross-Modal Alignment  |  "
            f"RSA $\\rho$ = {rsa_diff:.3f} (diff. peptide pairs)  "
            f"CKA = {cka_model:.3f}  |  "
            f"{esm2_sanity['n_unique_peptides']:.0f} unique peptides",
            fontsize=11,
            fontweight="bold",
            y=1.01,
        )

        fig_path = out_dir / "esm2_cross_modal_alignment.png"
        fig.savefig(fig_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Figure saved: {fig_path}")
        return fig_path

    # ------------------------------------------------------------------
    # Loggable metrics
    # ------------------------------------------------------------------
    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """Return loggable metrics."""
        if "error" in task_results:
            return {}
        keys = [
            "rsa_diff_peptide_rho",
            "rsa_all_pairs_rho",
            "cka_score",
            "baseline_shuffled_rsa_diff",
            "baseline_shuffled_cka",
            "baseline_metadata_rsa_diff",
            "baseline_metadata_cka",
        ]
        return {k: task_results[k] for k in keys if task_results.get(k) is not None}

    # ------------------------------------------------------------------
    # Sequence helpers
    # ------------------------------------------------------------------
    def _get_sequences(self, meta: Dict[str, np.ndarray]) -> Optional[List[str]]:
        for key in ["peptides", "sequence", "modified_peptide", "peptide", "seq"]:
            if key in meta:
                raw = meta[key]
                if hasattr(raw, "tolist"):
                    raw = raw.tolist()
                result = []
                for s in raw:
                    if isinstance(s, bytes):
                        result.append(s.decode("utf-8", errors="replace"))
                    elif isinstance(s, str):
                        result.append(s)
                    else:
                        result.append(str(s) if s is not None else "")
                return result
        return None

    def _build_valid_mask(self, sequences: List[str]) -> np.ndarray:
        mask = np.zeros(len(sequences), dtype=bool)
        for i, seq in enumerate(sequences):
            if not seq or not isinstance(seq, str):
                continue
            clean = _clean_sequence(seq)
            if self.min_seq_len <= len(clean) <= self.max_seq_len:
                mask[i] = True
        return mask
