"""Head Analysis Task — peak-to-peak attention pattern analysis.

Analyzes what individual attention heads in the last transformer layer have
learned by examining the **peak×peak** attention submatrix (excluding the
latent/CLS token, metadata tokens, and padding).

Key analyses:
1. Structural enrichment — do heads attend along physically meaningful
   peak-pair relationships (b/y complementarity, ion ladders, isotope
   spacing, neutral losses)?
2. Head pattern categorization — local, structural, global, sparse, mixed
3. Annotated vs unannotated routing — where does attention flow between
   peaks matched to theoretical ions and unmatched peaks?
4. Head similarity — cosine similarity of attention distributions across heads

Two complementary relationship mask strategies:
- **Annotation-based** (ground truth): uses precomputed theoretical matching
  (matched_annotation, parent_annotation, fragment_group_key).
- **m/z-based** (structural patterns): uses raw mass differences to detect
  ion ladders, isotope spacing, neutral losses, and b/y complementarity.
"""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from instanovo.__init__ import console
from instanovo.constants import CARBON_MASS_DELTA, H2O_MASS, PROTON_MASS_AMU
from instanovo_fm.eval.embed_eval_tasks import BaseTask
from instanovo_fm.utils.modifications import clean_peptide_sequence
from instanovo_fm.utils.peak_classification import (
    extract_fragment_position,
    extract_ion_type,
)
from instanovo_fm.utils.ion_visualization import (
    CATEGORY_COLORS,
    TEXT_COLORS,
    categorize_ion,
    format_annotation_display,
)
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

# Physical constants
_NH3_MASS = 17.026549
_CO_MASS = 27.99491  # CO loss: b-ion → a-ion
_H3PO4_MASS = 97.97690  # Phosphoric acid loss (phosphorylation diagnostic)
_SO3_MASS = 79.95682  # Sulfur trioxide loss (sulfation)
_ISOTOPE_TOL = 0.02  # Da tolerance for isotope spacing
_ISOTOPE_SPACING = CARBON_MASS_DELTA  # 1.00335 Da

# Standard amino acid residue masses (monoisotopic)
_RESIDUE_MASSES = np.array([
    57.021464, 71.037114, 87.032028, 97.052764, 99.068414,   # G A S P V
    101.047670, 103.009185, 113.084064, 113.084064, 114.042927,  # T C L I N
    115.026943, 128.058578, 128.094963, 129.042593, 131.040485,  # D Q K E M
    137.058912, 147.068414, 156.101111, 163.063329, 186.079313,  # H F R Y W
], dtype=np.float64)


# ---------------------------------------------------------------------------
# Helper: online statistics
# ---------------------------------------------------------------------------

class _RunningStats:
    """Welford online mean/variance accumulator."""

    __slots__ = ("_sum", "_sum_sq", "_count")

    def __init__(self) -> None:
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0

    def update(self, value: float) -> None:
        self._sum += value
        self._sum_sq += value * value
        self._count += 1

    def finalize(self) -> Dict[str, float]:
        if self._count == 0:
            return {"mean": 0.0, "std": 0.0, "count": 0}
        mean = self._sum / self._count
        var = max(0.0, self._sum_sq / self._count - mean * mean)
        return {"mean": float(mean), "std": float(var ** 0.5), "count": self._count}


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

class HeadAnalysisTask(BaseTask):
    """Peak-to-peak attention pattern analysis per head."""

    name = "headanalysistask"
    description = "Peak-to-peak attention analysis: structural enrichment, pattern categorization, annotation routing"
    requires_metadata = True
    requires_faiss = False
    requires_model = True

    def __init__(
        self,
        output_dir: str = "./head_analysis",
        sample_n: int = 5_000,
        save_figures: bool = True,
        # Mass-based relationship detection
        da_tol: float = 0.3,
        # Pattern categorization
        mz_local_window_da: float = 5.0,
        entropy_diffuse_threshold: float = 0.9,
        sparse_gini_threshold: float = 0.8,
        structural_enrichment_threshold: float = 2.0,
        # Annotation routing
        min_annotated_peaks: int = 5,
        # Hero spectra
        n_hero_spectra: int = 10,
        # Cross-layer
        analyze_all_layers: bool = False,
        # Quality gates
        min_backbone_coverage: float = 0.0,
        min_fragment_groups: int = 0,
        # Output
        dpi: int = 300,
        **kwargs,
    ):
        super().__init__(output_dir=output_dir, **kwargs)
        self.output_dir = Path(output_dir)
        self.sample_n = sample_n
        self.save_figures = save_figures
        self.da_tol = da_tol
        self.mz_local_window_da = mz_local_window_da
        self.entropy_diffuse_threshold = entropy_diffuse_threshold
        self.sparse_gini_threshold = sparse_gini_threshold
        self.structural_enrichment_threshold = structural_enrichment_threshold
        self.min_annotated_peaks = min_annotated_peaks
        self.n_hero_spectra = n_hero_spectra
        self.analyze_all_layers = analyze_all_layers
        self.min_backbone_coverage = min_backbone_coverage
        self.min_fragment_groups = min_fragment_groups
        self.dpi = dpi

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, emb, meta, faiss_index, model=None, dataloader=None, config=None, device=None):
        for name, val in [("model", model), ("dataloader", dataloader), ("config", config), ("device", device)]:
            if val is None:
                return {"task_name": self.name, "error": f"requires {name}", "success": False}
        return self._run_with_model(model, config, dataloader, device, str(self.output_dir), meta)

    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for k in (
            # Enrichment (strong signal: parent-child, isotope, neutral loss)
            "mean_parent_child_enrichment", "mean_isotope_spacing_enrichment",
            "mean_neutral_loss_enrichment", "mean_ion_ladder_enrichment",
            "max_isotope_spacing_enrichment", "max_ion_ladder_enrichment",
            # Charge state variant enrichment
            "mean_charge_state_variant_enrichment", "max_charge_state_variant_enrichment",
            # Head pattern counts
            "n_structural_heads", "n_local_heads",
            # Annotation routing (global property)
            "mean_annotated_to_annotated",
            # Column-sum headline metric
            "attn_intensity_correlation", "annotated_attn_ratio",
            # Distance + entropy
            "mean_attn_distance_da", "mean_head_entropy",
            # Immonium region routing
            "mean_region_frag_to_imm_enrichment",
            "mean_ann_backbone_to_immonium_enrichment",
        ):
            if k in task_results:
                metrics[k] = float(task_results[k])
        for k in ("num_spectra_analyzed", "n_heads", "n_spectra_with_theory"):
            if k in task_results:
                metrics[k] = float(task_results[k])
        return metrics

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _run_with_model(self, model, config, dataloader, device, output_dir, metadata):
        start = time.time()
        metadata = metadata or {}

        # Resolve max_mz for denormalization
        max_mz = 2500.0
        if config is not None:
            if hasattr(config, "model"):
                max_mz = getattr(config.model, "max_mz", max_mz)
            elif isinstance(config, dict) and "model" in config:
                max_mz = config["model"].get("max_mz", max_mz)

        out_dir = Path(output_dir)
        fig_dir = Path(output_dir) / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        if self.save_figures:
            fig_dir.mkdir(parents=True, exist_ok=True)

        effective_n = self.sample_n
        model.eval()
        model = model.to(device)

        n_heads: Optional[int] = None
        accumulators: Optional[Dict] = None
        sim_accum: Optional[Dict] = None
        hero_candidates: List[Dict] = []  # collect all candidates, sample later
        total_spectra = 0
        n_spectra_with_theory = 0
        n_skipped_low_quality = 0
        rng = np.random.RandomState(42)

        # Pre-fetch spectrum quality array for quality gating
        spectrum_quality_arr = metadata.get("spectrum_quality")

        for batch_idx, batch in enumerate(dataloader):
            if effective_n and total_spectra >= effective_n:
                break

            spectra = batch["spectra"].to(device)
            bs = spectra.size(0)
            if effective_n and total_spectra + bs > effective_n:
                bs = effective_n - total_spectra
                spectra = spectra[:bs]

            batch_start = total_spectra

            try:
                with torch.no_grad():
                    out = model.forward_with_attn(spectra)
                attn_weights = out["attn_weights"]
                special_mask = out["special_mask"]
            except Exception as e:
                logger.warning(f"forward_with_attn failed batch {batch_idx}: {e}")
                total_spectra += bs
                continue

            if not attn_weights:
                total_spectra += bs
                continue
            layer_indices = list(range(len(attn_weights))) if self.analyze_all_layers else [len(attn_weights) - 1]

            for layer_idx in layer_indices:
                layer_attn = attn_weights[layer_idx]
                if layer_attn is None:
                    continue

                peak_attn, valid_mask = self._extract_peak_peak_attention(layer_attn, spectra, special_mask)
                B, H, L, _ = peak_attn.shape

                if n_heads is None:
                    n_heads = H
                    accumulators = self._init_accumulators(H)
                    sim_accum = {"dot": np.zeros((H, H)), "norm_sq": np.zeros((H,)), "count": 0}

                mz_da = spectra[:, :, 0].cpu().numpy() * max_mz

                for i in range(B):
                    global_idx = batch_start + i
                    valid_i = valid_mask[i]
                    n_valid = int(valid_i.sum())
                    if n_valid < 3:
                        continue

                    # --- Quality gate ---
                    if (
                        spectrum_quality_arr is not None
                        and global_idx < len(spectrum_quality_arr)
                        and spectrum_quality_arr[global_idx] is not None
                    ):
                        sq = spectrum_quality_arr[global_idx]
                        if isinstance(sq, dict):
                            bc = sq.get("backbone_coverage", 0.0)
                            ng = sq.get("n_fragment_groups", 0)
                            if bc < self.min_backbone_coverage or ng < self.min_fragment_groups:
                                n_skipped_low_quality += 1
                                continue

                    # --- Get metadata ---
                    ann_mask_raw = self._get_meta_array(metadata, "theoretical_match_mask", global_idx)
                    matched_ann_raw = self._get_meta_array(metadata, "matched_annotation", global_idx)
                    parent_ann_raw = self._get_meta_array(metadata, "parent_annotation", global_idx)
                    frag_group_raw = self._get_meta_array(metadata, "fragment_group_key", global_idx)
                    prec_mz = self._get_meta_scalar(metadata, "precursor_mz", global_idx, 0.0)
                    prec_charge = int(self._get_meta_scalar(metadata, "precursor_charge", global_idx, 2))
                    seq = self._get_meta_scalar(metadata, "sequence", global_idx) or self._get_meta_scalar(metadata, "peptides", global_idx)

                    # Fix 9: proper peptide length via clean_peptide_sequence
                    seq_len = 0
                    if isinstance(seq, str) and seq:
                        clean_seq = clean_peptide_sequence(seq)
                        seq_len = len(clean_seq)

                    # Align variable-length metadata arrays to full spectrum length L
                    ann_mask_L = self._align_bool_mask(ann_mask_raw, valid_i, L)
                    matched_ann_L = self._align_object_array(matched_ann_raw, valid_i, L)
                    parent_ann_L = self._align_object_array(parent_ann_raw, valid_i, L)
                    frag_group_L = self._align_object_array(frag_group_raw, valid_i, L)

                    has_theory = matched_ann_L is not None and any(a is not None for a in matched_ann_L)
                    if has_theory:
                        n_spectra_with_theory += 1

                    # --- Build relationship masks ---
                    rel_masks: Dict[str, np.ndarray] = {}
                    ann_mask_keys: List[str] = []  # track which masks are annotation-based

                    if has_theory:
                        ann_masks = self._build_annotation_masks(
                            matched_ann_L, parent_ann_L, frag_group_L, seq_len, valid_i, L
                        )
                        rel_masks.update(ann_masks)
                        ann_mask_keys = list(ann_masks.keys())

                    mz_masks = self._build_mz_masks(mz_da[i], valid_i, prec_mz, prec_charge)
                    rel_masks.update(mz_masks)

                    # --- Column-sum analysis (once per spectrum, all heads) ---
                    intensity_i = spectra[i, :, 1].cpu().numpy()
                    col_sum_stats = self._compute_column_sum_stats(
                        peak_attn[i], ann_mask_L, intensity_i, valid_i
                    )

                    # --- Per-head analysis ---
                    head_vecs: Dict[int, np.ndarray] = {}
                    for h in range(H):
                        ha = peak_attn[i, h]
                        enrichment = self._compute_enrichment(ha, rel_masks, valid_i)
                        category = self._categorize_head_pattern(ha, valid_i, enrichment, mz_da[i])
                        routing = self._compute_annotation_routing(ha, ann_mask_L, valid_i)
                        entropy_stats = self._compute_head_entropy_stats(ha, valid_i)
                        dist_profile = self._compute_distance_profile(ha, mz_da[i], valid_i)
                        immonium_routing = self._compute_immonium_routing(
                            ha, mz_da[i], matched_ann_L, valid_i
                        )

                        self._update_accumulators(
                            accumulators, h, enrichment,
                            category, routing, entropy_stats,
                            dist_profile, col_sum_stats,
                            immonium_routing,
                            has_theory, ann_mask_keys,
                        )
                        head_vecs[h] = ha[np.ix_(valid_i, valid_i)].flatten()

                    self._update_head_similarity(sim_accum, head_vecs)

                    # Collect hero candidates with full metadata for spectrum plots
                    if ann_mask_L is not None and ann_mask_L.sum() >= self.min_annotated_peaks:
                        hero_candidates.append({
                            "peak_attn": peak_attn[i].copy(),
                            "valid_mask": valid_i.copy(),
                            "mz_da": mz_da[i].copy(),
                            "intensity": intensity_i.copy(),
                            "ann_mask": ann_mask_L.copy(),
                            "matched_ann": matched_ann_L,
                            "global_idx": global_idx,
                            "n_valid": n_valid,
                            "n_annotated": int(ann_mask_L.sum()),
                            "sequence": seq if isinstance(seq, str) else "",
                            "charge": prec_charge,
                            "frag_type": self._get_meta_scalar(metadata, "frag_type", global_idx, ""),
                        })

            total_spectra += bs

        # --- Select diverse hero spectra ---
        hero_data = self._select_hero_spectra(hero_candidates, rng)

        # --- Post-processing ---
        if accumulators is None or n_heads is None:
            return {"task_name": self.name, "error": "No attention data collected", "success": False}

        summary = self._finalize_accumulators(accumulators, n_heads)
        sim_matrix = self._finalize_head_similarity(sim_accum) if sim_accum else np.eye(n_heads)

        with open(out_dir / "per_head_metrics.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self._save_csv(summary, out_dir / "per_head_metrics.csv")

        if self.save_figures:
            self._generate_visualizations(summary, sim_matrix, hero_data, fig_dir)

        elapsed = time.time() - start
        if n_skipped_low_quality > 0:
            logger.info(f"Quality gate: skipped {n_skipped_low_quality} spectra "
                        f"(min_backbone_coverage={self.min_backbone_coverage}, "
                        f"min_fragment_groups={self.min_fragment_groups})")
        logger.info(f"Head analysis complete: {total_spectra} spectra ({n_spectra_with_theory} with theory), "
                     f"{n_heads} heads, {elapsed:.1f}s")

        result = {
            "task_name": self.name, "success": True,
            "num_spectra_analyzed": total_spectra,
            "n_spectra_with_theory": n_spectra_with_theory,
            "n_skipped_low_quality": n_skipped_low_quality,
            "n_heads": n_heads, "execution_time": elapsed,
            "per_head_summary": summary,
        }
        # Aggregate across heads
        for rel_key in ("series_adjacent", "by_complement", "parent_child",
                         "ion_ladder", "isotope_spacing", "charge_state_variant"):
            vals = [summary[str(h)].get(f"{rel_key}_enrichment", {}).get("mean", 0.0) for h in range(n_heads)]
            if vals:
                result[f"mean_{rel_key}_enrichment"] = float(np.mean(vals))
                result[f"max_{rel_key}_enrichment"] = float(np.max(vals))
        cats = [summary[str(h)].get("dominant_category", "mixed") for h in range(n_heads)]
        for cat in ("structural", "local", "global", "sparse", "mixed"):
            result[f"n_{cat}_heads"] = float(cats.count(cat))
        aa = [summary[str(h)].get("annotated_to_annotated", {}).get("mean", 0.0) for h in range(n_heads)]
        result["mean_annotated_to_annotated"] = float(np.mean(aa)) if aa else 0.0
        ent = [summary[str(h)].get("row_entropy", {}).get("mean", 0.0) for h in range(n_heads)]
        result["mean_head_entropy"] = float(np.mean(ent)) if ent else 0.0
        # Neutral loss enrichment
        nl = [summary[str(h)].get("neutral_loss_enrichment", {}).get("mean", 0.0) for h in range(n_heads)]
        result["mean_neutral_loss_enrichment"] = float(np.mean(nl)) if nl else 0.0
        # Max enrichment for key relationships
        iso = [summary[str(h)].get("isotope_spacing_enrichment", {}).get("mean", 0.0) for h in range(n_heads)]
        result["max_isotope_spacing_enrichment"] = float(np.max(iso)) if iso else 0.0
        # Distance profiles
        dist = [summary[str(h)].get("attn_mean_distance", {}).get("mean", 0.0) for h in range(n_heads)]
        result["mean_attn_distance_da"] = float(np.mean(dist)) if dist else 0.0
        # Column-sum headline metrics
        ais = [summary[str(h)].get("attn_intensity_spearman", {}).get("mean", 0.0) for h in range(n_heads)]
        result["attn_intensity_correlation"] = float(np.mean(ais)) if ais else 0.0
        aar = [summary[str(h)].get("annotated_attn_ratio", {}).get("mean", 0.0) for h in range(n_heads)]
        result["annotated_attn_ratio"] = float(np.mean(aar)) if aar else 0.0
        # Immonium region routing
        for imm_key in ("region_frag_to_imm", "region_imm_to_frag",
                          "region_frag_to_imm_enrichment",
                          "ann_backbone_to_immonium", "ann_immonium_to_backbone",
                          "ann_backbone_to_immonium_enrichment"):
            vals = [summary[str(h)].get(imm_key, {}).get("mean", 0.0) for h in range(n_heads)]
            result[f"mean_{imm_key}"] = float(np.mean(vals)) if vals else 0.0
            if "enrichment" in imm_key:
                result[f"max_{imm_key}"] = float(np.max(vals)) if vals else 0.0

        return result

    # ------------------------------------------------------------------
    # Attention extraction
    # ------------------------------------------------------------------

    def _extract_peak_peak_attention(
        self, layer_attn: torch.Tensor, spectra: torch.Tensor, special_mask: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract peak×peak attention submatrix.

        Returns:
            peak_attn: (B, H, L, L) float32 — peak-to-peak attention
            valid_mask: (B, L) bool — non-padding peaks
        """
        if layer_attn.dim() == 3:
            layer_attn = layer_attn.unsqueeze(1)
        B, H, T, _ = layer_attn.shape
        L = spectra.size(1)
        num_prepended = T - L

        peak_attn = layer_attn[:, :, num_prepended:, num_prepended:].detach().cpu().numpy().astype(np.float32)
        valid_mask = (spectra[:, :, 1] > 0).cpu().numpy().astype(bool)
        return peak_attn, valid_mask

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_meta_array(metadata: Dict, key: str, idx: int) -> Optional[np.ndarray]:
        if key not in metadata:
            return None
        arr = metadata[key]
        if idx >= len(arr):
            return None
        val = arr[idx]
        if val is None:
            return None
        return np.asarray(val)

    @staticmethod
    def _get_meta_scalar(metadata: Dict, key: str, idx: int, default=None):
        if key not in metadata:
            return default
        arr = metadata[key]
        if idx >= len(arr):
            return default
        val = arr[idx]
        return val if val is not None else default

    @staticmethod
    def _align_bool_mask(mask, valid_mask: np.ndarray, L: int) -> Optional[np.ndarray]:
        """Align a boolean mask (possibly valid-peaks-only) to full spectrum length L."""
        if mask is None or len(mask) == 0:
            return None
        mask = np.asarray(mask, dtype=bool)
        if len(mask) == L:
            return mask
        n_valid = int(valid_mask.sum())
        if len(mask) == n_valid:
            full = np.zeros(L, dtype=bool)
            full[valid_mask] = mask
            return full
        full = np.zeros(L, dtype=bool)
        n = min(len(mask), L)
        full[:n] = mask[:n]
        return full

    @staticmethod
    def _align_object_array(arr, valid_mask: np.ndarray, L: int) -> Optional[List]:
        """Align an object array (annotations, etc.) to full spectrum length L.

        If arr has length == n_valid_peaks, expand into full-length list using valid_mask.
        """
        if arr is None or len(arr) == 0:
            return None
        arr = list(arr)
        if len(arr) == L:
            return arr
        n_valid = int(valid_mask.sum())
        if len(arr) == n_valid:
            full = [None] * L
            valid_indices = np.where(valid_mask)[0]
            for j, vi in enumerate(valid_indices):
                full[vi] = arr[j]
            return full
        # Best effort: pad
        full = [None] * L
        for j in range(min(len(arr), L)):
            full[j] = arr[j]
        return full

    # ------------------------------------------------------------------
    # Annotation-based relationship masks (vectorized)
    # ------------------------------------------------------------------

    def _build_annotation_masks(
        self, matched_ann_L: Optional[List], parent_ann_L: Optional[List],
        frag_group_L: Optional[List], seq_len: int,
        valid_mask: np.ndarray, L: int,
    ) -> Dict[str, np.ndarray]:
        """Build (L, L) boolean masks from precomputed theoretical annotations."""
        masks: Dict[str, np.ndarray] = {}
        ann_list = matched_ann_L if matched_ann_L is not None else [None] * L

        # Parse annotations into arrays for vectorized operations
        ion_type_arr = np.full(L, "", dtype=object)
        position_arr = np.full(L, -1, dtype=np.int32)
        for j in range(L):
            if ann_list[j] is not None and valid_mask[j]:
                ion_type_arr[j] = extract_ion_type(str(ann_list[j]))
                pos = extract_fragment_position(str(ann_list[j]))
                position_arr[j] = pos if pos >= 0 else -1

        has_annotation = (position_arr >= 0) & valid_mask
        ann_idx = np.where(has_annotation)[0]

        # a. Same-series adjacency — vectorized
        adj = np.zeros((L, L), dtype=bool)
        if len(ann_idx) > 1:
            ion_sub = ion_type_arr[ann_idx]
            pos_sub = position_arr[ann_idx]
            # Same ion type matrix
            same_type = ion_sub[:, None] == ion_sub[None, :]
            # |position diff| == 1
            pos_diff = np.abs(pos_sub[:, None].astype(np.int64) - pos_sub[None, :].astype(np.int64))
            adj_sub = same_type & (pos_diff == 1)
            # Exclude precursor ions
            not_precursor = (ion_sub != "p") & (ion_sub != "unknown")
            adj_sub &= not_precursor[:, None] & not_precursor[None, :]
            # Map back to full L×L
            adj[np.ix_(ann_idx, ann_idx)] = adj_sub
        masks["series_adjacent"] = adj

        # b. b/y complementarity — vectorized
        by_comp = np.zeros((L, L), dtype=bool)
        if seq_len > 0 and len(ann_idx) > 1:
            ion_sub = ion_type_arr[ann_idx]
            pos_sub = position_arr[ann_idx]
            is_b = ion_sub == "b"
            is_y = ion_sub == "y"
            # b_p + y_q = seq_len → complementary
            pos_sum = pos_sub[:, None].astype(np.int64) + pos_sub[None, :].astype(np.int64)
            comp_sub = ((is_b[:, None] & is_y[None, :]) | (is_y[:, None] & is_b[None, :])) & (pos_sum == seq_len)
            by_comp[np.ix_(ann_idx, ann_idx)] = comp_sub
        masks["by_complement"] = by_comp

        # c. Parent-child — loss/isotope → parent base ion
        parent_child = np.zeros((L, L), dtype=bool)
        parent_list = parent_ann_L if parent_ann_L is not None else [None] * L
        ann_to_idx: Dict[str, List[int]] = {}
        for j in range(L):
            if ann_list[j] is not None and valid_mask[j]:
                ann_to_idx.setdefault(str(ann_list[j]), []).append(j)
        for j in range(L):
            if parent_list[j] is not None and valid_mask[j]:
                parent_str = str(parent_list[j])
                if parent_str in ann_to_idx:
                    for k in ann_to_idx[parent_str]:
                        if k != j:
                            parent_child[j, k] = parent_child[k, j] = True
        masks["parent_child"] = parent_child

        # d. Fragment group — peaks sharing same fragment_group_key
        frag_grp = np.zeros((L, L), dtype=bool)
        fg_list = frag_group_L if frag_group_L is not None else [None] * L
        grp_to_idx: Dict[str, List[int]] = {}
        for j in range(L):
            if fg_list[j] is not None and valid_mask[j]:
                grp_to_idx.setdefault(str(fg_list[j]), []).append(j)
        for indices in grp_to_idx.values():
            if len(indices) > 1:
                idx_arr = np.array(indices)
                frag_grp[np.ix_(idx_arr, idx_arr)] = True
                np.fill_diagonal(frag_grp[np.ix_(idx_arr, idx_arr)], False)
        masks["fragment_group"] = frag_grp

        return masks

    # ------------------------------------------------------------------
    # m/z-based relationship masks
    # ------------------------------------------------------------------

    def _build_mz_masks(
        self, mz_da: np.ndarray, valid_mask: np.ndarray,
        precursor_mz: float, precursor_charge: int,
    ) -> Dict[str, np.ndarray]:
        """Build (L, L) boolean masks from raw m/z differences."""
        L = len(mz_da)
        tol = self.da_tol
        masks: Dict[str, np.ndarray] = {}

        delta = np.abs(mz_da[:, None] - mz_da[None, :])
        valid_2d = valid_mask[:, None] & valid_mask[None, :]

        # Ion ladder: peak pairs whose m/z differs by a single residue mass.
        # This is a *mass-delta* signal — it flags any pair where |Δm/z|
        # matches one of the 20 AA masses within ``tol``, regardless of
        # whether the two peaks are truly adjacent backbone positions. A
        # 2-residue gap (e.g. G+V = 156.09 Da) coincidentally matches R
        # (156.10 Da) and will register here as well. The annotation-based
        # :meth:`_build_annotation_masks["series_adjacent"]` is the
        # position-strict counterpart (|Δposition| == 1); use that when
        # ground-truth positions are available. Computed at z=1 only —
        # scaling by charge would inflate false positives without adding
        # meaningful ladder signal.
        ladder = np.zeros((L, L), dtype=bool)
        for rm in _RESIDUE_MASSES:
            ladder |= np.abs(delta - rm) < tol
        masks["ion_ladder"] = ladder & valid_2d

        # Isotope spacing (z=1, z=2, z=3)
        iso = np.abs(delta - _ISOTOPE_SPACING) < _ISOTOPE_TOL
        iso |= np.abs(delta - _ISOTOPE_SPACING / 2) < (_ISOTOPE_TOL / 2)
        iso |= np.abs(delta - _ISOTOPE_SPACING / 3) < (_ISOTOPE_TOL / 3)
        masks["isotope_spacing"] = iso & valid_2d

        # Neutral loss offsets (H₂O, NH₃, CO/a-ion, H₃PO₄, SO₃)
        # Check at z=1 (raw delta) and z≥2 (delta = loss_mass/z) to
        # match how isotope_spacing handles charge-dependent spacing.
        _NL_MASSES = (H2O_MASS, _NH3_MASS, _CO_MASS, _H3PO4_MASS, _SO3_MASS)
        nl = np.zeros((L, L), dtype=bool)
        for loss_mass in _NL_MASSES:
            nl |= np.abs(delta - loss_mass) < tol
        for z in range(2, precursor_charge + 1):
            scaled_tol = tol / z
            for loss_mass in _NL_MASSES:
                nl |= np.abs(delta - loss_mass / z) < scaled_tol
        masks["neutral_loss"] = nl & valid_2d

        # b/y complementarity (m/z-based). Neutral-mass identity:
        #   b_i_neutral + y_{L−i}_neutral = M_neutral (the peptide neutral
        #   mass, already including the terminal H2O by convention).
        # Check all (z_b, z_y) where z_b + z_y <= precursor_charge.
        # The dominant mechanism is z_b + z_y = z_prec (charge conservation);
        # z_b + z_y < z_prec occurs when a proton is lost to a neutral
        # fragment (e.g. z=3 precursor → z=1 b + z=1 y + neutral).
        by_mz = np.zeros((L, L), dtype=bool)
        if precursor_mz > 0 and precursor_charge >= 1:
            # M_neutral = m/z·z − z·proton (same as (m/z − p)·z, written
            # literally so the derivation reads left-to-right).
            precursor_mass = (
                precursor_mz * precursor_charge
                - precursor_charge * PROTON_MASS_AMU
            )
            for z_b in range(1, precursor_charge + 1):
                for z_y in range(1, precursor_charge + 1):
                    if z_b + z_y > precursor_charge:
                        continue
                    neutral_b = mz_da * z_b - z_b * PROTON_MASS_AMU
                    neutral_y = mz_da * z_y - z_y * PROTON_MASS_AMU
                    sum_matrix = neutral_b[:, None] + neutral_y[None, :]
                    match = np.abs(sum_matrix - precursor_mass) < tol
                    by_mz |= match | match.T
        masks["by_complement_mz"] = by_mz & valid_2d

        # Charge-state variants: the same fragment observed at two different
        # charge states. From m/z(z_b) = (M + z_b·p)/z_b we can derive:
        #     m/z(z_a) = (m/z(z_b)·z_b − p·(z_b − z_a)) / z_a
        # Cover every ordered pair (z_a, z_b) with 1 ≤ z_a < z_b ≤ z_prec
        # so high-charge precursors (z=3, z=4) get full coverage, not just
        # adjacent-charge comparisons.
        csv = np.zeros((L, L), dtype=bool)
        for z_b in range(2, precursor_charge + 1):
            for z_a in range(1, z_b):
                expected_mz_a = (
                    mz_da[None, :] * z_b - PROTON_MASS_AMU * (z_b - z_a)
                ) / z_a
                csv_pair = np.abs(mz_da[:, None] - expected_mz_a) < tol
                csv |= csv_pair | csv_pair.T
        masks["charge_state_variant"] = csv & valid_2d

        return masks

    # ------------------------------------------------------------------
    # Enrichment computation
    # ------------------------------------------------------------------

    def _compute_enrichment(
        self, head_attn: np.ndarray, rel_masks: Dict[str, np.ndarray], valid_mask: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Compute fraction_of_attention and enrichment_ratio for each relationship."""
        valid_2d = valid_mask[:, None] & valid_mask[None, :]
        A = head_attn * valid_2d
        total_attn = float(A.sum())
        n_total = int(valid_2d.sum())

        result: Dict[str, Dict[str, float]] = {}
        for name, mask in rel_masks.items():
            valid_rel = mask & valid_2d
            n_rel = int(valid_rel.sum())
            attn_on_rel = float(A[valid_rel].sum())
            fraction = attn_on_rel / (total_attn + 1e-12)
            expected = n_rel / (n_total + 1e-12)
            enrichment = fraction / (expected + 1e-12)
            result[name] = {"fraction": fraction, "enrichment_ratio": enrichment, "n_edges": n_rel}
        return result

    # ------------------------------------------------------------------
    # Head pattern categorization
    # ------------------------------------------------------------------

    def _categorize_head_pattern(
        self, head_attn: np.ndarray, valid_mask: np.ndarray,
        enrichment: Dict[str, Dict[str, float]], mz_da: np.ndarray,
    ) -> str:
        L = len(valid_mask)
        valid_2d = valid_mask[:, None] & valid_mask[None, :]
        A = head_attn * valid_2d
        total = float(A.sum()) + 1e-12
        n_valid = int(valid_mask.sum())

        # Gini coefficient (clamped to [0, 1])
        flat = A[valid_2d]
        gini = 0.0
        if len(flat) > 0 and flat.sum() > 1e-12:
            sorted_vals = np.sort(flat)
            n = len(sorted_vals)
            index = np.arange(1, n + 1)
            gini = float(np.clip(
                (2 * (index * sorted_vals).sum() / (n * sorted_vals.sum() + 1e-12)) - (n + 1) / n,
                0.0, 1.0,
            ))

        # Check structural first (enrichment on physical relationships)
        max_enr = max((v["enrichment_ratio"] for v in enrichment.values()), default=0.0)
        if max_enr > self.structural_enrichment_threshold:
            return "structural"

        # Local attention fraction — measured in m/z distance (Daltons)
        # Check before sparse: a head concentrated on nearby peaks is "local" not "sparse"
        local_mask = np.abs(mz_da[:, None] - mz_da[None, :]) <= self.mz_local_window_da
        local_mask = local_mask & valid_2d
        local_frac = float(A[local_mask].sum()) / total
        if local_frac > 0.5:
            return "local"

        # Sparse: high Gini but NOT local or structural
        if gini > self.sparse_gini_threshold:
            return "sparse"

        # Mean normalized row entropy
        entropies = []
        for j in range(L):
            if not valid_mask[j]:
                continue
            row = A[j, valid_mask]
            row_sum = row.sum()
            if row_sum < 1e-12:
                continue
            p = row / row_sum
            H_val = -float(np.sum(p[p > 1e-12] * np.log(p[p > 1e-12] + 1e-12)))
            H_max = np.log(max(n_valid, 2))
            entropies.append(H_val / H_max)

        if entropies and np.mean(entropies) > self.entropy_diffuse_threshold:
            return "global"

        return "mixed"

    # ------------------------------------------------------------------
    # Annotated vs unannotated routing
    # ------------------------------------------------------------------

    def _compute_annotation_routing(
        self, head_attn: np.ndarray, ann_mask: Optional[np.ndarray], valid_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Partition attention into 2x2 matrix: annotated vs unannotated peaks."""
        if ann_mask is None or ann_mask.sum() < self.min_annotated_peaks:
            return {"annotated_to_annotated": 0.0, "annotated_to_unannotated": 0.0,
                    "unannotated_to_annotated": 0.0, "unannotated_to_unannotated": 0.0}

        ann = ann_mask & valid_mask
        unann = (~ann_mask) & valid_mask
        A = head_attn
        total = float(A[np.ix_(valid_mask, valid_mask)].sum()) + 1e-12

        aa = float(A[np.ix_(ann, ann)].sum()) / total
        au = float(A[np.ix_(ann, unann)].sum()) / total
        ua = float(A[np.ix_(unann, ann)].sum()) / total
        uu = float(A[np.ix_(unann, unann)].sum()) / total
        return {"annotated_to_annotated": aa, "annotated_to_unannotated": au,
                "unannotated_to_annotated": ua, "unannotated_to_unannotated": uu}

    # ------------------------------------------------------------------
    # Entropy stats
    # ------------------------------------------------------------------

    def _compute_head_entropy_stats(
        self, head_attn: np.ndarray, valid_mask: np.ndarray,
    ) -> Dict[str, float]:
        n_valid = int(valid_mask.sum())
        if n_valid < 2:
            return {"row_entropy": 0.0, "gini": 0.0}

        entropies = []
        for j in range(len(valid_mask)):
            if not valid_mask[j]:
                continue
            row = head_attn[j, valid_mask]
            s = row.sum()
            if s < 1e-12:
                continue
            p = row / s
            H_val = -float(np.sum(p[p > 1e-12] * np.log(p[p > 1e-12] + 1e-12)))
            entropies.append(H_val)

        flat = head_attn[np.ix_(valid_mask, valid_mask)].flatten()
        gini = 0.0
        if len(flat) > 0 and flat.sum() > 1e-12:
            sorted_vals = np.sort(flat)
            n = len(sorted_vals)
            index = np.arange(1, n + 1)
            gini = float(np.clip(
                (2 * (index * sorted_vals).sum() / (n * sorted_vals.sum())) - (n + 1) / n,
                0.0, 1.0,
            ))

        return {"row_entropy": float(np.mean(entropies)) if entropies else 0.0, "gini": gini}

    # ------------------------------------------------------------------
    # Distance profiles (Change 3)
    # ------------------------------------------------------------------

    def _compute_distance_profile(
        self, head_attn: np.ndarray, mz_da: np.ndarray, valid_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Attention-weighted m/z distance profile and fingerprint."""
        valid_2d = valid_mask[:, None] & valid_mask[None, :]
        A = head_attn * valid_2d
        total = float(A.sum())
        if total < 1e-12:
            return {"attn_mean_distance": 0.0, "dist_bin_0_2": 0.0, "dist_bin_2_25": 0.0,
                    "dist_bin_25_50": 0.0, "dist_bin_50_200": 0.0, "dist_bin_200_500": 0.0, "dist_bin_500_inf": 0.0}

        delta = np.abs(mz_da[:, None] - mz_da[None, :])
        mean_dist = float((A * delta).sum() / total)

        # Distance fingerprint bins
        bins = [(0, 2), (2, 25), (25, 50), (50, 200), (200, 500)]
        bin_names = ["dist_bin_0_2", "dist_bin_2_25", "dist_bin_25_50", "dist_bin_50_200", "dist_bin_200_500"]
        result: Dict[str, float] = {"attn_mean_distance": mean_dist}
        for (lo, hi), name in zip(bins, bin_names):
            mask = (delta >= lo) & (delta < hi) & valid_2d
            result[name] = float(A[mask].sum() / total)
        # Last bin: [500, inf)
        mask_inf = (delta >= 500) & valid_2d
        result["dist_bin_500_inf"] = float(A[mask_inf].sum() / total)
        return result

    # ------------------------------------------------------------------
    # Column-sum / attention received (Change 4)
    # ------------------------------------------------------------------

    def _compute_column_sum_stats(
        self, peak_attn_all_heads: np.ndarray, ann_mask: Optional[np.ndarray],
        intensity: np.ndarray, valid_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Per-peak attention received (column sum), correlated with intensity and annotation status."""
        H = peak_attn_all_heads.shape[0]
        n_valid = int(valid_mask.sum())
        if n_valid < 3:
            return {"attn_intensity_spearman": 0.0, "annotated_attn_ratio": 0.0}

        # Column sums averaged across heads → per-peak attention received
        valid_2d = valid_mask[:, None] & valid_mask[None, :]
        col_sums = np.zeros(len(valid_mask), dtype=np.float32)
        for h in range(H):
            A = peak_attn_all_heads[h] * valid_2d
            col_sums += A.sum(axis=0)
        col_sums /= max(H, 1)

        valid_cols = col_sums[valid_mask]
        valid_intensity = intensity[valid_mask]

        # Spearman: attention received vs intensity
        attn_intensity_spearman = 0.0
        try:
            from scipy.stats import spearmanr
            res = spearmanr(valid_cols, valid_intensity)
            rho = float(res.correlation if hasattr(res, "correlation") else res[0])
            if np.isfinite(rho):
                attn_intensity_spearman = rho
        except Exception:
            pass

        # Annotated vs unannotated attention ratio
        annotated_attn_ratio = 0.0
        if ann_mask is not None and ann_mask.sum() >= self.min_annotated_peaks:
            ann_valid = ann_mask[valid_mask]
            unann_valid = ~ann_valid
            ann_mean = float(valid_cols[ann_valid].mean()) if ann_valid.sum() > 0 else 0.0
            unann_mean = float(valid_cols[unann_valid].mean()) if unann_valid.sum() > 0 else 0.0
            annotated_attn_ratio = ann_mean / (unann_mean + 1e-12)

        return {"attn_intensity_spearman": attn_intensity_spearman, "annotated_attn_ratio": annotated_attn_ratio}

    # ------------------------------------------------------------------
    # Immonium region routing
    # ------------------------------------------------------------------

    _IMMONIUM_MZ_THRESHOLD = 200.0  # Da — peaks below this are in the immonium region

    def _compute_immonium_routing(
        self, head_attn: np.ndarray, mz_da: np.ndarray,
        matched_ann_L: Optional[List], valid_mask: np.ndarray,
    ) -> Dict[str, float]:
        """Measure attention flow between immonium region (<200 Da) and fragment region.

        Returns both m/z-based region routing (always available) and
        annotation-based immonium routing (when annotations identify
        immonium ions).

        The key question (from Notion §9): do encoder heads attend from
        high-m/z ladder regions to low-m/z immonium ions?
        """
        result: Dict[str, float] = {}
        valid_2d = valid_mask[:, None] & valid_mask[None, :]
        A = head_attn * valid_2d
        total = float(A.sum()) + 1e-12

        # --- m/z-based region routing (annotation-free) ---
        imm_region = (mz_da < self._IMMONIUM_MZ_THRESHOLD) & valid_mask
        frag_region = (mz_da >= self._IMMONIUM_MZ_THRESHOLD) & valid_mask
        n_imm = int(imm_region.sum())
        n_frag = int(frag_region.sum())

        if n_imm >= 1 and n_frag >= 1:
            # Attention from fragment region → immonium region (columns = receivers)
            frag_to_imm = float(A[np.ix_(frag_region, imm_region)].sum()) / total
            imm_to_frag = float(A[np.ix_(imm_region, frag_region)].sum()) / total
            imm_to_imm = float(A[np.ix_(imm_region, imm_region)].sum()) / total

            # Enrichment: compare to expected fraction if attention were uniform
            n_valid = int(valid_mask.sum())
            expected_frag_to_imm = (n_frag * n_imm) / (n_valid * n_valid + 1e-12)
            result["region_frag_to_imm"] = frag_to_imm
            result["region_imm_to_frag"] = imm_to_frag
            result["region_imm_to_imm"] = imm_to_imm
            result["region_frag_to_imm_enrichment"] = frag_to_imm / (expected_frag_to_imm + 1e-12)
        else:
            result["region_frag_to_imm"] = 0.0
            result["region_imm_to_frag"] = 0.0
            result["region_imm_to_imm"] = 0.0
            result["region_frag_to_imm_enrichment"] = 0.0

        # --- Annotation-based immonium routing (when available) ---
        # Only include ann_* keys when both immonium and backbone annotations
        # exist, so that spectra without annotations don't dilute the means.
        if matched_ann_L is not None:
            imm_ann = np.array([
                bool(a and isinstance(a, str) and "immonium" in a) and valid_mask[j]
                for j, a in enumerate(matched_ann_L)
            ], dtype=bool)
            backbone_ann = np.array([
                bool(a and isinstance(a, str) and a[0].lower() in "byaxcz"
                     and not a.startswith("custom:")) and valid_mask[j]
                for j, a in enumerate(matched_ann_L)
            ], dtype=bool)
            n_imm_ann = int(imm_ann.sum())
            n_backbone = int(backbone_ann.sum())

            if n_imm_ann >= 1 and n_backbone >= 1:
                backbone_to_imm = float(A[np.ix_(backbone_ann, imm_ann)].sum()) / total
                imm_to_backbone = float(A[np.ix_(imm_ann, backbone_ann)].sum()) / total
                n_valid = int(valid_mask.sum())
                expected = (n_backbone * n_imm_ann) / (n_valid * n_valid + 1e-12)
                result["ann_backbone_to_immonium"] = backbone_to_imm
                result["ann_immonium_to_backbone"] = imm_to_backbone
                result["ann_backbone_to_immonium_enrichment"] = backbone_to_imm / (expected + 1e-12)

        return result

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------

    def _init_accumulators(self, n_heads: int) -> Dict:
        acc: Dict[str, Any] = {}
        rel_keys = ["series_adjacent", "by_complement", "parent_child", "fragment_group",
                     "ion_ladder", "isotope_spacing", "neutral_loss", "by_complement_mz",
                     "charge_state_variant"]
        for h in range(n_heads):
            h_acc: Dict[str, Any] = {}
            for rk in rel_keys:
                h_acc[f"{rk}_fraction"] = _RunningStats()
                h_acc[f"{rk}_enrichment"] = _RunningStats()
            h_acc["category_counts"] = Counter()
            for rk in ("annotated_to_annotated", "annotated_to_unannotated", "unannotated_to_annotated", "unannotated_to_unannotated"):
                h_acc[rk] = _RunningStats()
            h_acc["row_entropy"] = _RunningStats()
            h_acc["gini"] = _RunningStats()
            # Distance profiles
            h_acc["attn_mean_distance"] = _RunningStats()
            for bn in ("dist_bin_0_2", "dist_bin_2_25", "dist_bin_25_50", "dist_bin_50_200", "dist_bin_200_500", "dist_bin_500_inf"):
                h_acc[bn] = _RunningStats()
            # Column-sum — spectrum-level, same value for all heads
            h_acc["attn_intensity_spearman"] = _RunningStats()
            h_acc["annotated_attn_ratio"] = _RunningStats()
            # Immonium region routing
            for ik in ("region_frag_to_imm", "region_imm_to_frag", "region_imm_to_imm",
                        "region_frag_to_imm_enrichment",
                        "ann_backbone_to_immonium", "ann_immonium_to_backbone",
                        "ann_backbone_to_immonium_enrichment"):
                h_acc[ik] = _RunningStats()
            # Separate counters for annotation-based metrics
            h_acc["n_with_theory"] = 0
            acc[str(h)] = h_acc
        return acc

    def _update_accumulators(
        self, acc: Dict, h: int, enrichment: Dict,
        category: str, routing: Dict[str, float], entropy_stats: Dict[str, float],
        dist_profile: Dict[str, float], col_sum_stats: Dict[str, float],
        immonium_routing: Dict[str, float],
        has_theory: bool, ann_mask_keys: List[str],
    ) -> None:
        ha = acc[str(h)]
        for name, vals in enrichment.items():
            fk = f"{name}_fraction"
            ek = f"{name}_enrichment"
            is_ann = name in ann_mask_keys
            if is_ann and not has_theory:
                continue
            if fk in ha:
                ha[fk].update(vals["fraction"])
            if ek in ha:
                ha[ek].update(vals["enrichment_ratio"])
        if has_theory:
            ha["n_with_theory"] += 1
        ha["category_counts"][category] += 1
        for rk, rv in routing.items():
            if rk in ha:
                ha[rk].update(rv)
        if "row_entropy" in entropy_stats:
            ha["row_entropy"].update(entropy_stats["row_entropy"])
        if "gini" in entropy_stats:
            ha["gini"].update(entropy_stats["gini"])
        for dk, dv in dist_profile.items():
            if dk in ha:
                ha[dk].update(dv)
        for sk, sv in col_sum_stats.items():
            if sk in ha:
                ha[sk].update(sv)
        for ik, iv in immonium_routing.items():
            if ik in ha:
                ha[ik].update(iv)

    def _finalize_accumulators(self, acc: Dict, n_heads: int) -> Dict:
        summary: Dict[str, Any] = {}
        for h in range(n_heads):
            ha = acc[str(h)]
            hs: Dict[str, Any] = {}
            for key, val in ha.items():
                if isinstance(val, _RunningStats):
                    hs[key] = val.finalize()
                elif isinstance(val, Counter):
                    hs[key] = dict(val)
                elif key == "n_with_theory":
                    hs[key] = val
            cats = ha["category_counts"]
            hs["dominant_category"] = cats.most_common(1)[0][0] if cats else "mixed"
            summary[str(h)] = hs
        return summary

    # ------------------------------------------------------------------
    # Head similarity (per-spectrum normalized)
    # ------------------------------------------------------------------

    def _update_head_similarity(self, sim_accum: Dict, head_vectors: Dict[int, np.ndarray]) -> None:
        heads = sorted(head_vectors.keys())
        if not heads:
            return
        max_len = max(len(head_vectors[h]) for h in heads)
        vecs = np.zeros((len(heads), max_len), dtype=np.float32)
        for idx, h in enumerate(heads):
            v = head_vectors[h]
            vecs[idx, :len(v)] = v

        # Normalize each head vector for this spectrum (per-spectrum cosine)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        vecs_normed = vecs / norms

        # Accumulate cosine similarity per spectrum
        sim_accum["dot"] += vecs_normed @ vecs_normed.T  # (H, H)
        sim_accum["count"] += 1

    @staticmethod
    def _finalize_head_similarity(sim_accum: Dict) -> np.ndarray:
        count = max(sim_accum["count"], 1)
        return sim_accum["dot"] / count  # mean cosine similarity across spectra

    # ------------------------------------------------------------------
    # Hero spectrum selection
    # ------------------------------------------------------------------

    def _select_hero_spectra(self, candidates: List[Dict], rng: np.random.RandomState) -> List[Dict]:
        """Select diverse hero spectra: stratify by n_valid bins to get variety."""
        n = min(self.n_hero_spectra, len(candidates))
        if n == 0:
            return []
        if len(candidates) <= n:
            return candidates

        # Stratify by n_valid quartiles
        n_valids = np.array([c["n_valid"] for c in candidates])
        quartiles = np.percentile(n_valids, [25, 50, 75])
        bins = np.digitize(n_valids, quartiles)
        per_bin = max(1, n // 4)

        selected_idx: List[int] = []
        for b in range(4):
            bin_idx = np.where(bins == b)[0]
            if len(bin_idx) == 0:
                continue
            pick = rng.choice(bin_idx, size=min(per_bin, len(bin_idx)), replace=False)
            selected_idx.extend(pick.tolist())

        # Fill remaining slots randomly
        remaining = n - len(selected_idx)
        if remaining > 0:
            leftover = [i for i in range(len(candidates)) if i not in selected_idx]
            if leftover:
                extra = rng.choice(leftover, size=min(remaining, len(leftover)), replace=False)
                selected_idx.extend(extra.tolist())

        return [candidates[i] for i in selected_idx[:n]]

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def _save_csv(self, summary: Dict, path: Path) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["head", "metric", "mean", "std", "count"])
            for h_str, metrics in sorted(summary.items(), key=lambda x: int(x[0])):
                for metric_name, stats in metrics.items():
                    if isinstance(stats, dict) and "mean" in stats:
                        writer.writerow([h_str, metric_name, stats["mean"], stats["std"], stats["count"]])
                    elif metric_name == "dominant_category":
                        writer.writerow([h_str, metric_name, stats, "", ""])
                    elif metric_name == "category_counts" and isinstance(stats, dict):
                        for cat, cnt in stats.items():
                            writer.writerow([h_str, f"category_{cat}", cnt, "", ""])
                    elif metric_name == "n_with_theory":
                        writer.writerow([h_str, metric_name, stats, "", ""])

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    def _generate_visualizations(self, summary: Dict, sim_matrix: np.ndarray,
                                  hero_data: List[Dict], fig_dir: Path) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available, skipping visualizations")
            return

        try:
            self._plot_specialization_heatmap(summary, fig_dir, plt)
            self._plot_pattern_categorization(summary, fig_dir, plt)
            self._plot_annotation_routing_matrix(summary, fig_dir, plt)
            self._plot_head_entropy_comparison(summary, fig_dir, plt)
            self._plot_head_similarity_heatmap(sim_matrix, fig_dir, plt)
            self._plot_distance_fingerprint_heatmap(summary, fig_dir, plt)
            self._plot_immonium_routing(summary, fig_dir, plt)
            if hero_data:
                self._plot_hero_spectrum_attention(hero_data, fig_dir, plt)
                self._plot_hero_spectrum_with_attention(hero_data, fig_dir, plt)
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")

    def _plot_specialization_heatmap(self, summary: Dict, fig_dir: Path, plt) -> None:
        """Heads × relationship types heatmap of enrichment ratios."""
        heads = sorted(summary.keys(), key=int)
        rel_keys = ["series_adjacent", "by_complement", "parent_child", "fragment_group",
                     "ion_ladder", "isotope_spacing", "neutral_loss", "by_complement_mz",
                     "charge_state_variant"]
        labels = ["Series\nadjacent", "b/y\ncomplement", "Parent\nchild", "Fragment\ngroup",
                  "Ion\nladder", "Isotope\nspacing", "Neutral\nloss", "b/y comp.\n(m/z)",
                  "Charge\nvariant"]

        H = len(heads)
        M = np.ones((H, len(rel_keys)))
        for i, h in enumerate(heads):
            for j, rk in enumerate(rel_keys):
                val = summary[h].get(f"{rk}_enrichment", {})
                if isinstance(val, dict):
                    M[i, j] = val.get("mean", 1.0)

        fig, ax = plt.subplots(figsize=(12, max(3, 0.5 * H + 1)))
        vmax = max(3.0, np.nanmax(M))
        im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0, vmax=vmax)
        ax.set_yticks(range(H))
        ax.set_yticklabels([f"H{h}" for h in heads])
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        if H <= 16:
            for i in range(H):
                for j in range(len(rel_keys)):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i, j] < vmax * 0.4 else "black")
        plt.colorbar(im, ax=ax, label="Enrichment ratio (1.0 = random)")
        ax.set_title("Structural Enrichment per Head")
        fig.tight_layout()
        fig.savefig(fig_dir / "specialization_heatmap.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_pattern_categorization(self, summary: Dict, fig_dir: Path, plt) -> None:
        cats = [summary[h].get("dominant_category", "mixed") for h in sorted(summary.keys(), key=int)]
        counts = Counter(cats)
        categories = ["structural", "local", "global", "sparse", "mixed"]
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#95a5a6"]

        fig, ax = plt.subplots(figsize=(8, 5))
        vals = [counts.get(c, 0) for c in categories]
        ax.bar(categories, vals, color=colors, edgecolor="black")
        ax.set_ylabel("Number of heads")
        ax.set_title("Head Pattern Categorization")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(vals):
            if v > 0:
                ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / "pattern_categorization.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_annotation_routing_matrix(self, summary: Dict, fig_dir: Path, plt) -> None:
        heads = sorted(summary.keys(), key=int)
        routing_keys = ["annotated_to_annotated", "annotated_to_unannotated", "unannotated_to_annotated", "unannotated_to_unannotated"]
        labels = ["Ann->Ann", "Ann->Unann", "Unann->Ann", "Unann->Unann"]
        H = len(heads)
        M = np.zeros((H, 4))
        for i, h in enumerate(heads):
            for j, rk in enumerate(routing_keys):
                val = summary[h].get(rk, {})
                M[i, j] = val.get("mean", 0.0) if isinstance(val, dict) else 0.0

        fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * H + 1)))
        im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(0.5, M.max()))
        ax.set_yticks(range(H))
        ax.set_yticklabels([f"H{h}" for h in heads])
        ax.set_xticks(range(4))
        ax.set_xticklabels(labels)
        if H <= 16:
            for i in range(H):
                for j in range(4):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="Attention mass fraction")
        ax.set_title("Annotation Routing per Head")
        fig.tight_layout()
        fig.savefig(fig_dir / "annotation_routing_matrix.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_head_entropy_comparison(self, summary: Dict, fig_dir: Path, plt) -> None:
        heads = sorted(summary.keys(), key=int)
        ent_means = [summary[h].get("row_entropy", {}).get("mean", 0.0) for h in heads]
        ent_stds = [summary[h].get("row_entropy", {}).get("std", 0.0) for h in heads]
        gini_means = [summary[h].get("gini", {}).get("mean", 0.0) for h in heads]
        gini_stds = [summary[h].get("gini", {}).get("std", 0.0) for h in heads]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(heads))
        ax1.bar(x, ent_means, yerr=ent_stds, capsize=4, alpha=0.7, edgecolor="black")
        ax1.set_xticks(x)
        ax1.set_xticklabels([f"H{h}" for h in heads])
        ax1.set_ylabel("Mean Row Entropy")
        ax1.set_title("Attention Entropy per Head")
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(x, gini_means, yerr=gini_stds, capsize=4, alpha=0.7, color="#e74c3c", edgecolor="black")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"H{h}" for h in heads])
        ax2.set_ylabel("Gini Coefficient")
        ax2.set_title("Attention Concentration (Gini) per Head")
        ax2.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(fig_dir / "head_entropy_comparison.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_distance_fingerprint_heatmap(self, summary: Dict, fig_dir: Path, plt) -> None:
        """Heads × distance bins heatmap showing where attention operates in m/z space."""
        heads = sorted(summary.keys(), key=int)
        bin_keys = ["dist_bin_0_2", "dist_bin_2_25", "dist_bin_25_50", "dist_bin_50_200", "dist_bin_200_500", "dist_bin_500_inf"]
        bin_labels = ["[0,2)\nIsotope", "[2,25)\nLosses", "[25,50)\nLarge loss", "[50,200)\nResidue", "[200,500)\nMulti-res", "[500+)\nLong-range"]

        H = len(heads)
        M = np.zeros((H, len(bin_keys)))
        for i, h in enumerate(heads):
            for j, bk in enumerate(bin_keys):
                val = summary[h].get(bk, {})
                M[i, j] = val.get("mean", 0.0) if isinstance(val, dict) else 0.0

        fig, ax = plt.subplots(figsize=(10, max(3, 0.5 * H + 1)))
        im = ax.imshow(M, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.5, M.max()))
        ax.set_yticks(range(H))
        ax.set_yticklabels([f"H{h}" for h in heads])
        ax.set_xticks(range(len(bin_labels)))
        ax.set_xticklabels(bin_labels, fontsize=9)
        if H <= 16:
            for i in range(H):
                for j in range(len(bin_keys)):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im, ax=ax, label="Fraction of attention mass")
        ax.set_title("Attention Distance Fingerprint per Head")
        fig.tight_layout()
        fig.savefig(fig_dir / "distance_fingerprint_heatmap.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_immonium_routing(self, summary: Dict, fig_dir: Path, plt) -> None:
        """Per-head immonium attention routing: m/z-region and annotation-based enrichment."""
        heads = sorted(summary.keys(), key=int)
        H = len(heads)

        region_enr = [summary[h].get("region_frag_to_imm_enrichment", {}).get("mean", 0.0) for h in heads]
        ann_enr = [summary[h].get("ann_backbone_to_immonium_enrichment", {}).get("mean", 0.0) for h in heads]
        region_std = [summary[h].get("region_frag_to_imm_enrichment", {}).get("std", 0.0) for h in heads]
        ann_std = [summary[h].get("ann_backbone_to_immonium_enrichment", {}).get("std", 0.0) for h in heads]
        has_ann = any(v > 0 for v in ann_enr)

        # Also pull the raw attention fractions for a stacked view
        frag_to_imm = [summary[h].get("region_frag_to_imm", {}).get("mean", 0.0) for h in heads]
        imm_to_frag = [summary[h].get("region_imm_to_frag", {}).get("mean", 0.0) for h in heads]
        imm_to_imm = [summary[h].get("region_imm_to_imm", {}).get("mean", 0.0) for h in heads]

        fig, axes = plt.subplots(1, 2, figsize=(16, 5), gridspec_kw={"width_ratios": [3, 2]})
        x = np.arange(H)
        w = 0.35

        # --- Left panel: enrichment comparison ---
        ax = axes[0]
        bars1 = ax.bar(x - w / 2, region_enr, w, yerr=region_std, capsize=3,
                        label="m/z region (<200 Da)", color="#3498db", alpha=0.85, edgecolor="black")
        if has_ann:
            bars2 = ax.bar(x + w / 2, ann_enr, w, yerr=ann_std, capsize=3,
                            label="Annotation-based", color="#e74c3c", alpha=0.85, edgecolor="black")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Random baseline")
        ax.set_xticks(x)
        ax.set_xticklabels([f"H{h}" for h in heads])
        ax.set_ylabel("Enrichment ratio")
        ax.set_title("Immonium Region Attention Enrichment per Head")
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        # --- Right panel: attention fraction breakdown ---
        ax2 = axes[1]
        ax2.bar(x, frag_to_imm, label="Frag → Imm", color="#3498db", alpha=0.8)
        ax2.bar(x, imm_to_frag, bottom=frag_to_imm, label="Imm → Frag", color="#2ecc71", alpha=0.8)
        bottoms = [a + b for a, b in zip(frag_to_imm, imm_to_frag)]
        ax2.bar(x, imm_to_imm, bottom=bottoms, label="Imm → Imm", color="#9b59b6", alpha=0.8)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"H{h}" for h in heads])
        ax2.set_ylabel("Fraction of total attention")
        ax2.set_title("Immonium Region Attention Flow")
        ax2.legend(fontsize=9, loc="upper left")
        ax2.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(fig_dir / "immonium_routing.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_head_similarity_heatmap(self, sim_matrix: np.ndarray, fig_dir: Path, plt) -> None:
        n = sim_matrix.shape[0]
        if n == 0:
            return
        try:
            from scipy.cluster.hierarchy import dendrogram, linkage
            from scipy.spatial.distance import squareform
            dist = np.clip(1 - sim_matrix, 0, 2)
            np.fill_diagonal(dist, 0)
            condensed = squareform(dist, checks=False)
            link = linkage(condensed, method="average")
            order = dendrogram(link, no_plot=True)["leaves"]
        except Exception:
            order = list(range(n))

        reordered = sim_matrix[order, :][:, order]
        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(reordered, cmap="RdYlBu_r", vmin=0, vmax=1)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels([f"H{order[i]}" for i in range(n)], rotation=90, fontsize=9)
        ax.set_yticklabels([f"H{order[i]}" for i in range(n)], fontsize=9)
        plt.colorbar(im, ax=ax, label="Cosine Similarity")
        ax.set_title("Head Similarity (Clustered)")
        fig.tight_layout()
        fig.savefig(fig_dir / "head_similarity_heatmap.png", dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_hero_spectrum_attention(self, hero_data: List[Dict], fig_dir: Path, plt) -> None:
        hero_dir = fig_dir / "hero_spectra"
        hero_dir.mkdir(exist_ok=True)
        for idx, hero in enumerate(hero_data[:self.n_hero_spectra]):
            peak_attn = hero["peak_attn"]
            valid = hero["valid_mask"]
            mz = hero["mz_da"]
            ann_mask_hero = hero["ann_mask"]
            H = peak_attn.shape[0]

            n_valid = int(valid.sum())
            if n_valid < 3:
                continue

            ncols = min(4, H)
            nrows = (H + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)

            for h in range(H):
                r, c = divmod(h, ncols)
                ax = axes[r][c]
                sub = peak_attn[h][np.ix_(valid, valid)]
                ax.imshow(sub, cmap="viridis", aspect="auto", interpolation="nearest")
                ax.set_title(f"Head {h}", fontsize=10)

                ann_valid_hero = ann_mask_hero[valid]
                for j in range(n_valid):
                    if ann_valid_hero[j]:
                        ax.axhline(j, color="red", alpha=0.15, linewidth=0.5)
                        ax.axvline(j, color="red", alpha=0.15, linewidth=0.5)

                if n_valid <= 30:
                    tick_labels = [f"{mz[vi]:.0f}" for vi in np.where(valid)[0]]
                    ax.set_xticks(range(n_valid))
                    ax.set_xticklabels(tick_labels, rotation=90, fontsize=5)
                    ax.set_yticks(range(n_valid))
                    ax.set_yticklabels(tick_labels, fontsize=5)

            for h in range(H, nrows * ncols):
                r, c = divmod(h, ncols)
                axes[r][c].set_visible(False)

            fig.suptitle(f"Spectrum {hero.get('global_idx', idx)} — Peak x Peak Attention "
                         f"({n_valid} valid, {hero.get('n_annotated', 0)} annotated)", fontsize=12)
            fig.tight_layout()
            fig.savefig(hero_dir / f"hero_{idx:03d}.png", dpi=self.dpi, bbox_inches="tight")
            plt.close(fig)

    def _plot_hero_spectrum_with_attention(self, hero_data: List[Dict], fig_dir: Path, plt) -> None:
        """Individual spectrum plots with attention overlaid on m/z stems.

        3-panel layout per spectrum (mirrors confidence analysis):
        Panel 1: m/z spectrum with ion-type coloring (ground truth)
        Panel 2: m/z spectrum colored by mean attention received (all heads)
        Panel 3: m/z spectrum colored by per-head attention (small multiples)
        """
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        spec_dir = fig_dir / "individual_spectra"
        spec_dir.mkdir(exist_ok=True)

        for plot_idx, hero in enumerate(hero_data[:self.n_hero_spectra]):
            peak_attn = hero["peak_attn"]  # (H, L, L)
            valid = hero["valid_mask"]     # (L,)
            mz_full = hero["mz_da"]        # (L,)
            intensity_full = hero.get("intensity", np.zeros_like(mz_full))  # (L,)
            ann_mask_hero = hero["ann_mask"]   # (L,)
            ann_list = hero.get("matched_ann") or [None] * len(valid)
            H = peak_attn.shape[0]

            n_valid = int(valid.sum())
            if n_valid < 3:
                continue

            valid_idx = np.where(valid)[0]
            mz = mz_full[valid_idx]
            norm_int = intensity_full[valid_idx]
            ann_valid_hero = ann_mask_hero[valid_idx]

            # Categorize ions and format annotations for valid peaks
            peak_categories = []
            peak_ann_display = []
            for j in range(n_valid):
                full_j = valid_idx[j]
                ann = ann_list[full_j] if full_j < len(ann_list) else None
                if ann is not None:
                    peak_categories.append(categorize_ion(str(ann)))
                    peak_ann_display.append(format_annotation_display(str(ann)))
                else:
                    peak_categories.append("unannotated")
                    peak_ann_display.append("")

            # Compute per-peak attention received (column sum of peak×peak, averaged across heads)
            # Column j sum = how much attention peak j receives from all other peaks
            attn_received_per_head = np.zeros((H, n_valid), dtype=np.float32)
            for h in range(H):
                sub = peak_attn[h][np.ix_(valid, valid)]  # (n_valid, n_valid)
                attn_received_per_head[h] = sub.sum(axis=0)  # column sums
            mean_attn_received = attn_received_per_head.mean(axis=0)  # average across heads

            # Normalize for coloring
            attn_max = mean_attn_received.max()
            attn_norm = mean_attn_received / (attn_max + 1e-12)

            # Build suptitle
            seq_display = hero.get("sequence", "")[:30]
            charge = hero.get("charge", "?")
            frag = hero.get("frag_type", "")
            title_parts = [seq_display, f"z={charge}"]
            if frag:
                title_parts.append(str(frag))
            suptitle = " | ".join(p for p in title_parts if p)

            # File naming (match confidence_signal_analysis convention)
            spec_idx = hero.get("global_idx", plot_idx)
            seq_for_file = hero.get("sequence", "unknown")[:20].replace("/", "_").replace("\\", "_").replace("[", "(").replace("]", ")")
            filename = f"spectrum_{spec_idx:04d}_{seq_for_file}.png"

            # --- Create 3-panel figure ---
            fig, axes = plt.subplots(3, 1, figsize=(20, 18))
            fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.995)

            # Panel 1: Ion-type coloring (reference)
            self._draw_mz_ion_panel(axes[0], mz, norm_int, peak_categories,
                                     peak_ann_display, n_valid, plt)

            # Panel 2: Mean attention received coloring
            self._draw_mz_attention_panel(axes[1], mz, norm_int, attn_norm,
                                           ann_valid_hero, peak_ann_display,
                                           peak_categories, n_valid, plt,
                                           title="Mean Attention Received (all heads)")

            # Panel 3: Per-head small multiples (top row of a grid inside the axis)
            self._draw_per_head_attention_panels(axes[2], mz, norm_int,
                                                  attn_received_per_head, ann_valid_hero,
                                                  n_valid, H, plt)

            # Adjust spacing for panels 1-2 only (panel 3 uses manual inset axes)
            fig.subplots_adjust(top=0.95, hspace=0.35)
            fig.savefig(spec_dir / filename, dpi=150, bbox_inches="tight")
            plt.close(fig)

    # --- Spectrum panel drawing helpers ---

    def _draw_mz_ion_panel(self, ax, mz, norm_int, peak_categories,
                            peak_ann_display, n_valid, plt) -> None:
        """Panel 1: m/z stems colored by ion type."""
        max_int = norm_int.max() if n_valid > 0 else 1.0

        # Draw unannotated first, then annotated on top
        for layer in ("unannotated", "annotated"):
            for i in range(n_valid):
                cat = peak_categories[i]
                if layer == "unannotated" and cat != "unannotated":
                    continue
                if layer == "annotated" and cat == "unannotated":
                    continue
                color, alpha = CATEGORY_COLORS.get(cat, ("#9467bd", 0.8))
                zorder = 2 if cat == "unannotated" else 3
                ax.plot([mz[i], mz[i]], [0, norm_int[i]], color=color,
                        linewidth=1.2, alpha=alpha, zorder=zorder)
                ax.scatter([mz[i]], [norm_int[i]], color=color, s=20,
                           alpha=alpha, zorder=zorder)

        # Labels with anti-collision
        self._add_peak_labels(ax, mz, norm_int, peak_categories,
                               peak_ann_display, n_valid, max_int)

        ax.set_xlabel("m/z", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized Intensity", fontsize=11, fontweight="bold")
        ax.set_title("m/z Spectrum with Ion-Type Coloring", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(50, mz.max() + 50 if n_valid > 0 else 2500)
        ax.set_ylim(0, max_int * 1.45)

    def _draw_mz_attention_panel(self, ax, mz, norm_int, attn_norm, ann_valid_hero,
                                  peak_ann_display, peak_categories, n_valid,
                                  plt, title="Attention Received") -> None:
        """Panel 2: m/z stems colored by attention score (viridis)."""
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        max_int = norm_int.max() if n_valid > 0 else 1.0
        cmap = plt.cm.viridis
        norm = Normalize(vmin=0, vmax=1)

        for i in range(n_valid):
            color = cmap(norm(attn_norm[i]))
            ax.plot([mz[i], mz[i]], [0, norm_int[i]], color=color,
                    linewidth=1.5, alpha=0.85, zorder=2)
            # Red edge for annotated peaks
            edge = "red" if ann_valid_hero[i] else "none"
            ax.scatter([mz[i]], [norm_int[i]], color=color, s=25,
                       edgecolors=edge, linewidths=0.8, alpha=0.85, zorder=3)

        # Labels
        self._add_peak_labels(ax, mz, norm_int, peak_categories,
                               peak_ann_display, n_valid, max_int)

        # Colorbar
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
        cbar.set_label("Attention Received (normalized)", fontsize=10)

        # Legend
        n_ann = int(ann_valid_hero.sum())
        ax.plot([], [], "o", color="gray", markeredgecolor="red", markeredgewidth=1.0,
                markersize=6, label=f"Annotated peaks ({n_ann})", linestyle="None")
        ax.plot([], [], "o", color="gray", markersize=6,
                label=f"Unannotated peaks ({n_valid - n_ann})", linestyle="None")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

        ax.set_xlabel("m/z", fontsize=11, fontweight="bold")
        ax.set_ylabel("Normalized Intensity", fontsize=11, fontweight="bold")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(50, mz.max() + 50 if n_valid > 0 else 2500)
        ax.set_ylim(0, max_int * 1.45)

    def _draw_per_head_attention_panels(self, ax, mz, norm_int,
                                         attn_received_per_head, ann_valid_hero,
                                         n_valid, H, plt) -> None:
        """Panel 3: per-head attention as small-multiples bar charts."""
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable

        # Clear the main axis — we'll create inset axes
        ax.set_visible(False)

        # Create a grid of small axes in the space of the original axis
        bbox = ax.get_position()
        ncols = min(4, H)
        nrows = (H + ncols - 1) // ncols
        pad_x, pad_y = 0.02, 0.03
        w = (bbox.width - pad_x * (ncols - 1)) / ncols
        h = (bbox.height - pad_y * (nrows - 1)) / nrows

        cmap = plt.cm.viridis

        for head_idx in range(H):
            r, c = divmod(head_idx, ncols)
            x0 = bbox.x0 + c * (w + pad_x)
            y0 = bbox.y0 + bbox.height - (r + 1) * h - r * pad_y
            inset = ax.figure.add_axes([x0, y0, w, h])

            attn_h = attn_received_per_head[head_idx]
            attn_max_h = attn_h.max()
            attn_norm_h = attn_h / (attn_max_h + 1e-12)

            norm_obj = Normalize(vmin=0, vmax=1)
            colors = [cmap(norm_obj(v)) for v in attn_norm_h]
            bars = inset.bar(range(n_valid), norm_int, color=colors, width=1.0, linewidth=0)

            # Mark annotated peaks with red ticks on x-axis
            for j in range(n_valid):
                if ann_valid_hero[j]:
                    inset.axvline(j, color="red", alpha=0.2, linewidth=0.5, zorder=1)

            inset.set_title(f"H{head_idx}", fontsize=9, fontweight="bold", pad=2)
            inset.set_xlim(-0.5, n_valid - 0.5)
            inset.set_ylim(0, norm_int.max() * 1.1 if n_valid > 0 else 1)
            inset.tick_params(labelsize=6)
            if r == nrows - 1:
                inset.set_xlabel("Peak idx", fontsize=7)
            else:
                inset.set_xticklabels([])
            if c == 0:
                inset.set_ylabel("Intensity", fontsize=7)
            else:
                inset.set_yticklabels([])

        # Hide unused insets
        # (no action needed — we only create axes for existing heads)

    @staticmethod
    def _add_peak_labels(ax, mz, norm_int, peak_categories, peak_ann_display,
                          n_valid, max_int) -> None:
        """Add ion annotation labels with intensity threshold + m/z anti-collision."""
        min_label_intensity = max_int * 0.03
        min_mz_gap = 20.0
        placed_mz: List[float] = []

        # Annotated peaks sorted by intensity descending
        annotated = [
            (i, mz[i], norm_int[i], peak_ann_display[i])
            for i in range(n_valid)
            if peak_categories[i] != "unannotated" and peak_ann_display[i]
        ]
        annotated.sort(key=lambda t: t[2], reverse=True)

        for idx, m, inten, display in annotated:
            if inten < min_label_intensity:
                continue
            if any(abs(m - p) < min_mz_gap for p in placed_mz):
                continue
            placed_mz.append(m)
            cat = peak_categories[idx]
            text_color = TEXT_COLORS.get(cat, "black")
            ax.annotate(
                display, xy=(m, inten), xytext=(0, 5), textcoords="offset points",
                ha="center", fontsize=7, color=text_color, rotation=90,
                alpha=0.9, fontweight="bold",
            )

        # Top-20 unannotated peaks labeled with m/z
        unannotated_idx = [i for i in range(n_valid) if peak_categories[i] == "unannotated"]
        if unannotated_idx:
            unannotated_arr = np.array(unannotated_idx)
            top_k = min(20, len(unannotated_arr))
            top_unann = unannotated_arr[np.argsort(norm_int[unannotated_arr])[-top_k:]]
            for i in top_unann:
                if norm_int[i] < min_label_intensity:
                    continue
                if any(abs(mz[i] - p) < min_mz_gap for p in placed_mz):
                    continue
                placed_mz.append(mz[i])
                ax.annotate(
                    f"{mz[i]:.1f}", xy=(mz[i], norm_int[i]),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=6, color="#555555",
                    rotation=90, alpha=0.7, fontstyle="italic",
                )


