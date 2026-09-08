"""Integrated Gradients attribution and PA bias dissection for the foundation model.

Two analyses:
1. Masked Peak Prediction Attribution — Integrated Gradients on the actual m/z
   reconstruction prediction to show which visible peaks the model causally uses.
   Operates at the fragment ion group level (base + isotopes + losses), not
   individual peaks, since the model treats these as a single chemical entity.
2. Pairwise Attention Bias Dissection — synthetic probing of the PA module to
   reveal what m/z differences the model has learned as structural priors.
   Only runs when the model was trained with pairwise attention bias.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn as nn

from instanovo.__init__ import console
from instanovo_fm.eval.embed_eval_tasks import BaseTask
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger
from instanovo_fm.eval.embed_eval_tasks.ig_attribution_helper import (  # noqa: E402
    AMINO_ACID_MASSES,
    EXTENDED_CHEMISTRY_CATEGORIES,
    ISOTOPE_SPACING,
    NEUTRAL_LOSS_MASSES,
    PROTON_MASS,
    SHIFT_NULL_DISTANCES,
    STRUCTURAL_CATEGORIES,
    AttributionResult,
    FragmentGroup,
    PredictionInfo,
    PredictionTarget,
    TopKAnalysis,
    _charge_from_group_key,
    _gap_residues_from_sequence,
    _parse_series_info,
    build_fragment_groups,
    compute_topk_attribution_analysis,
    detect_charge_variant_indices,
    expected_mz_at_other_charges,
    extract_prediction,
    plot_attribution_summary,
    plot_hero_spectrum_attribution,
    plot_novel_chemistry_summary,
    plot_pa_per_head_heatmap,
    plot_pa_response_spectrum,
    reduce_attributions_to_peaks,
    sweep_pa_per_head_bias,
    sweep_pa_response,
)
from instanovo_fm.utils.extended_chemistry import (  # noqa: E402
    ExtendedChemistryLibrary,
    build_extended_chemistry_library,
    compute_immonium_related_negative_control,
    match_peak_to_extended_chemistry,
)

try:
    from captum.attr import IntegratedGradients
except ImportError:
    IntegratedGradients = None


class IGAttributionTask(BaseTask):
    """Integrated Gradients attribution + PA bias dissection.

    Produces paper-quality figures demonstrating the model's learned
    understanding of proteomic mass spectra chemistry.

    PA dissection auto-detects whether the model has a pairwise attention bias
    module and only runs if present.
    """

    name = "ig_attribution"
    description = "Integrated Gradients attribution and PA bias analysis"
    requires_metadata = True
    requires_model = True

    def __init__(
        self,
        output_dir: str = "./ig_attribution",
        # Analysis toggles
        enable_prediction_attribution: bool = True,
        enable_pa_dissection: bool = True,
        # Prediction attribution settings
        ig_n_steps: int = 50,
        ig_internal_batch_size: int = 5,
        max_spectra: int = 100,
        max_groups_per_spectrum: int = 5,
        max_hero_plots: int = 20,
        # PA dissection settings
        pa_dmz_min: float = -250.0,
        pa_dmz_max: float = 250.0,
        pa_resolution: float = 0.01,
        # Quality gates
        min_backbone_coverage: float = 0.0,
        min_fragment_groups: int = 0,
        # Visualization
        dpi: int = 300,
        # Novel chemistry discovery probe
        enable_novel_chemistry: bool = True,
        novel_chemistry_max_internal_length: int = 6,
        novel_chemistry_ppm_tol: float = 10.0,
        novel_chemistry_da_tol: float = 0.5,
        novel_chemistry_scramble_trials: int = 5,
        # Reproducibility
        deterministic_mode: bool = True,
        deterministic_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        """Initialise the input."""
        super().__init__(**kwargs)
        self.output_dir = Path(output_dir)
        self.enable_prediction_attribution = enable_prediction_attribution
        self.enable_pa_dissection = enable_pa_dissection
        self.ig_n_steps = ig_n_steps
        self.ig_internal_batch_size = ig_internal_batch_size
        self.max_spectra = max_spectra
        self.max_groups_per_spectrum = max_groups_per_spectrum
        self.max_hero_plots = max_hero_plots
        self.pa_dmz_min = pa_dmz_min
        self.pa_dmz_max = pa_dmz_max
        self.pa_resolution = pa_resolution
        self.min_backbone_coverage = min_backbone_coverage
        self.min_fragment_groups = min_fragment_groups
        self.dpi = dpi
        self.enable_novel_chemistry = enable_novel_chemistry
        self.novel_chemistry_max_internal_length = novel_chemistry_max_internal_length
        self.novel_chemistry_ppm_tol = novel_chemistry_ppm_tol
        self.novel_chemistry_da_tol = novel_chemistry_da_tol
        self.novel_chemistry_scramble_trials = novel_chemistry_scramble_trials
        self.deterministic_mode = deterministic_mode
        self.deterministic_seed = deterministic_seed

    def run(
        self,
        emb: np.ndarray,
        meta: Dict[str, Any],
        faiss_index: Any,
        model: Any = None,
        dataloader: Any = None,
        config: Any = None,
        device: Any = None,
    ) -> Dict[str, Any]:
        """Run XAI v2 analyses."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results: Dict[str, Any] = {"task_name": self.name, "success": False}
        t0 = time.time()

        if model is None:
            results["error"] = "Model is required for XAI v2"
            return results

        if device is None:
            device = next(model.parameters()).device

        # Store model config for hero reports
        self._model_config = config

        # --- Analysis 2: PA Bias Dissection ---
        # Auto-detect: only run if the model has a pairwise_bias module
        has_pa = getattr(model, "pairwise_bias", None) is not None
        if self.enable_pa_dissection and has_pa:
            pa_results = self._run_pa_dissection(model, device)
            results["pa_dissection"] = pa_results
        elif self.enable_pa_dissection and not has_pa:
            logger.info("Skipping PA dissection — model has no pairwise_bias module")

        # --- Analysis 1: Prediction Attribution ---
        if self.enable_prediction_attribution:
            if IntegratedGradients is None:
                results["prediction_attribution"] = {"error": "captum not installed"}
            elif dataloader is None:
                results["prediction_attribution"] = {"error": "dataloader required"}
            else:
                attr_results = self._run_prediction_attribution(model, meta, dataloader, device)
                results["prediction_attribution"] = attr_results

        results["execution_time_s"] = time.time() - t0
        results["success"] = True
        self._save_results(results)
        return results

    # ------------------------------------------------------------------
    # Analysis 2: PA Bias Dissection
    # ------------------------------------------------------------------

    def _run_pa_dissection(self, model: nn.Module, device: torch.device) -> Dict[str, Any]:
        """Run pairwise attention bias dissection analysis."""
        pa_results: Dict[str, Any] = {"success": False}
        pa_module = model.pairwise_bias

        encoder = getattr(model, "encoder", None)
        pw_projection = getattr(encoder, "pw_projection", None) if encoder else None

        # --- Sweep PA response (full pipeline if projection available) ---
        logger.info("Sweeping PA response spectrum...")
        dmz, response = sweep_pa_response(
            pa_module,
            dmz_min=self.pa_dmz_min,
            dmz_max=self.pa_dmz_max,
            resolution=self.pa_resolution,
            device=str(device),
            pw_projection=pw_projection,
        )

        pa_peaks = self._find_response_peaks(dmz, response)
        pa_results["response_peaks"] = pa_peaks
        pa_results["num_freqs"] = pa_module.num_freqs
        pa_results["hidden_dim"] = pa_module.hidden_dim

        plot_pa_response_spectrum(
            dmz,
            response,
            save_path=self.output_dir / "pa_response_spectrum.png",
            dpi=self.dpi,
        )
        pa_results["response_spectrum_path"] = str(self.output_dir / "pa_response_spectrum.png")

        # --- Per-head bias sweep ---
        if pw_projection is not None:
            logger.info("Sweeping per-head PA bias...")
            dmz_h, per_head = sweep_pa_per_head_bias(
                pa_module,
                pw_projection,
                dmz_min=self.pa_dmz_min,
                dmz_max=self.pa_dmz_max,
                resolution=self.pa_resolution,
            )

            n_heads = getattr(model, "n_heads", 8)
            n_layers = getattr(model, "n_layers", 6)

            plot_pa_per_head_heatmap(
                dmz_h,
                per_head,
                n_heads=n_heads,
                n_layers=n_layers,
                save_path=self.output_dir / "pa_per_head_heatmap.png",
                dpi=self.dpi,
            )
            pa_results["per_head_heatmap_path"] = str(self.output_dir / "pa_per_head_heatmap.png")

        # --- Chemical alignment score ---
        pa_results["chemical_alignment"] = self._compute_chemical_alignment(dmz, response)

        # --- Symmetry analysis ---
        # Compare positive vs negative response at key AA masses
        neg_mask = dmz <= 0
        dmz_neg = -dmz[neg_mask][::-1]
        resp_neg = response[neg_mask][::-1]
        asymmetry: dict[str, Any] = {}
        for name, mass in AMINO_ACID_MASSES.items():
            pos_idx = np.argmin(np.abs(dmz[dmz >= 0] - mass))
            neg_idx = np.argmin(np.abs(dmz_neg - mass))
            if pos_idx < len(response[dmz >= 0]) and neg_idx < len(resp_neg):
                r_pos = float(response[dmz >= 0][pos_idx])
                r_neg = float(resp_neg[neg_idx])
                asymmetry[name] = {
                    "positive": r_pos,
                    "negative": r_neg,
                    "ratio": float(r_pos / (r_neg + 1e-8)),
                }
        pa_results["asymmetry"] = asymmetry

        pa_results["success"] = True

        logger.info(f"PA dissection: {len(pa_peaks)} response peaks, alignment score={pa_results['chemical_alignment']['overall_score']:.3f}")
        return pa_results

    def _find_response_peaks(
        self,
        dmz: np.ndarray,
        response: np.ndarray,
        min_prominence: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """Find local maxima in the PA response spectrum."""
        try:
            from scipy.signal import find_peaks
        except ImportError:
            return []

        pos_mask = dmz > 0.5
        dmz_pos = dmz[pos_mask]
        resp_pos = response[pos_mask]
        resp_norm = resp_pos / (resp_pos.max() + 1e-8)

        indices, _ = find_peaks(
            resp_norm,
            prominence=min_prominence,
            distance=int(0.5 / self.pa_resolution),
        )

        peaks: list[Any] = []
        for idx in indices:
            ref = self._closest_chemical_reference(float(dmz_pos[idx]))
            peaks.append(
                {
                    "mz": float(dmz_pos[idx]),
                    "response": float(resp_pos[idx]),
                    "response_normalized": float(resp_norm[idx]),
                    "closest_reference": ref,
                }
            )
        peaks.sort(key=lambda p: p["response"], reverse=True)
        return peaks[:50]

    def _closest_chemical_reference(self, mz: float) -> Dict[str, Any]:
        """Find the closest known chemical mass difference to a given m/z."""
        best: dict[str, Any] = {"type": "none", "name": "", "mass": 0.0, "delta": float("inf")}
        for name, mass in AMINO_ACID_MASSES.items():
            delta = abs(mz - mass)
            if delta < best["delta"]:
                best = {"type": "amino_acid", "name": name, "mass": mass, "delta": delta}
        for name, mass in NEUTRAL_LOSS_MASSES.items():
            delta = abs(mz - mass)
            if delta < best["delta"]:
                best = {"type": "neutral_loss", "name": name, "mass": mass, "delta": delta}
        for z in [1, 2, 3]:
            spacing = ISOTOPE_SPACING / z
            delta = abs(mz - spacing)
            if delta < best["delta"]:
                best = {"type": "isotope", "name": f"13C/z={z}", "mass": spacing, "delta": delta}
        return best

    def _compute_chemical_alignment(self, dmz: np.ndarray, response: np.ndarray) -> Dict[str, Any]:
        """Compute how well PA response aligns with known chemical masses.

        Control: shift all chemical masses by +3.7 Da (arbitrary offset that
        avoids landing on another chemical mass) instead of random sampling,
        giving a principled baseline that preserves the same frequency density.
        """
        pos_mask = dmz > 0.5
        dmz_pos = dmz[pos_mask]
        resp_pos = response[pos_mask]

        chemical_masses = (
            list(AMINO_ACID_MASSES.values()) + list(NEUTRAL_LOSS_MASSES.values()) + [ISOTOPE_SPACING, ISOTOPE_SPACING / 2, ISOTOPE_SPACING / 3]
        )

        def _mean_response_at(masses: List[float]) -> float:
            values: list[Any] = []
            for m in masses:
                window = np.abs(dmz_pos - m) < 0.05
                if window.sum() > 0:
                    values.append(float(resp_pos[window].mean()))
            return np.mean(values) if values else 0.0

        mean_chem = _mean_response_at(chemical_masses)

        # Control: mean response across the full positive spectrum (global baseline)
        mean_global = float(resp_pos.mean())

        # Also compute shifted control (multiple shifts averaged for robustness)
        control_shifts = list(SHIFT_NULL_DISTANCES)
        shifted_responses: list[Any] = []
        for shift in control_shifts:
            shifted_masses = [m + shift for m in chemical_masses if 0.5 < m + shift < dmz_pos.max()]
            if shifted_masses:
                shifted_responses.append(_mean_response_at(shifted_masses))
        mean_shifted = float(np.mean(shifted_responses)) if shifted_responses else mean_global

        score = float(mean_chem / (mean_chem + mean_shifted + 1e-8))
        return {
            "overall_score": score,
            "mean_chemical_response": float(mean_chem),
            "mean_global_response": float(mean_global),
            "mean_shifted_control": float(mean_shifted),
            "enrichment_ratio": float(mean_chem / (mean_shifted + 1e-8)),
            "n_chemical_masses": len(chemical_masses),
        }

    # ------------------------------------------------------------------
    # Analysis 1: Prediction Attribution
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _deterministic_context(self) -> Iterator[None]:
        """Make the IG pass bit-reproducible across identical runs.

        IG attribution is sensitive to GPU non-determinism because it
        back-propagates through the encoder; FlashAttention's backward
        pass uses atomic reductions that produce run-to-run variance in
        per-peak attributions (median ~34% relative diff on Q Exactive
        spectra). This context manager:

        1. Forces SDPA to dispatch to the ``math`` backend (no flash,
           no memory-efficient) for the duration of the IG pass.
        2. Turns on cuDNN determinism and disables cuDNN auto-tuning.
        3. Enables ``torch.use_deterministic_algorithms(True)`` with
           ``warn_only=True`` so any still-non-deterministic op (should
           not happen in this model's forward/backward) surfaces as a
           warning rather than silently corrupting the result.
        4. Sets a fixed torch/cuda/numpy seed for the scope of the pass.
        5. Exports ``CUBLAS_WORKSPACE_CONFIG=:4096:8`` if not already
           set — cuBLAS needs this at process-load time for deterministic
           matmul; setting it late is best-effort but doesn't hurt.

        Original state is restored on exit so the rest of the evaluator
        (embedding generation, non-IG tasks) runs at normal speed.
        """
        if not self.deterministic_mode:
            yield
            return

        # Import here so the context manager doesn't pay import cost
        # when deterministic_mode=False.
        from torch.nn.attention import SDPBackend, sdpa_kernel

        prev_cudnn_det = torch.backends.cudnn.deterministic
        prev_cudnn_bench = torch.backends.cudnn.benchmark
        prev_det_algo = torch.are_deterministic_algorithms_enabled()
        prev_cublas_env = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        prev_cpu_state = torch.random.get_rng_state()
        prev_cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        prev_np_state = np.random.get_state()

        if prev_cublas_env is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(self.deterministic_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.deterministic_seed)
        np.random.seed(self.deterministic_seed)

        try:
            with sdpa_kernel([SDPBackend.MATH]):
                yield
        finally:
            torch.backends.cudnn.deterministic = prev_cudnn_det
            torch.backends.cudnn.benchmark = prev_cudnn_bench
            torch.use_deterministic_algorithms(prev_det_algo, warn_only=True)
            if prev_cublas_env is None:
                os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
            torch.random.set_rng_state(prev_cpu_state)
            if prev_cuda_state is not None:
                torch.cuda.set_rng_state_all(prev_cuda_state)
            np.random.set_state(prev_np_state)

    def _run_prediction_attribution(
        self,
        model: nn.Module,
        meta: Dict[str, Any],
        dataloader: Any,
        device: torch.device,
    ) -> Dict[str, Any]:
        """Run Integrated Gradients attribution on masked fragment ion groups."""
        attr_results: Dict[str, Any] = {"success": False}

        # Get max_mz for denormalization (spectra are stored normalized to [0, 1])
        self._max_mz = getattr(model, "max_mz", 2500.0)
        # Per-group IG-failure counter — surfaced in the result dict so a
        # silent drop doesn't hide degraded coverage in downstream metrics.
        self._n_ig_failures = 0

        model.eval()

        collected = self._collect_annotated_spectra(meta, dataloader, device)
        if not collected:
            attr_results["error"] = "No annotated spectra found"
            return attr_results

        logger.info(f"Prediction attribution: {len(collected)} annotated spectra")

        all_results: List[AttributionResult] = []
        hero_count = 0
        hero_dir = self.output_dir / "hero_spectra"
        hero_dir.mkdir(parents=True, exist_ok=True)

        n_total = len(collected)
        log_interval = max(1, n_total // 5)  # Log at ~20% milestones
        if self.deterministic_mode:
            logger.info(f"IG deterministic mode ON — SDPA=math, cuDNN deterministic, seed={self.deterministic_seed}")
        with self._deterministic_context():
            for si, spec_info in enumerate(collected):
                if si > 0 and si % log_interval == 0:
                    logger.info(f"  {si}/{n_total} spectra processed...")
                spec_results = self._attribute_spectrum(model, spec_info, device)
                all_results.extend(spec_results)

                if hero_count < self.max_hero_plots and spec_results:
                    self._emit_hero(spec_results, spec_info, hero_count, hero_dir)
                    hero_count += 1

        if not all_results:
            attr_results["error"] = "No attribution results computed"
            return attr_results

        # Prediction quality summary
        preds = [r.prediction for r in all_results if r.prediction is not None]
        if preds:
            bin_correct = [p.correct_bin for p in preds]
            group_acc = [p.group_bin_accuracy for p in preds if p.group_bin_accuracy >= 0]
            errors = [p.error_da for p in preds]
            ppms = [p.error_ppm for p in preds]
            attr_results["prediction_quality"] = {
                "bin_accuracy": float(np.mean(bin_correct)),
                "group_bin_accuracy": float(np.mean(group_acc)) if group_acc else 0.0,
                "median_error_da": float(np.median(errors)),
                "median_error_ppm": float(np.median(ppms)),
                "n_predictions": len(preds),
            }
            # Stratify reconstruction quality by masked-group ion type
            # (b / y / precursor / unknown) so backbone b/y bin-accuracy and
            # ppm can be reported separately from the corpus aggregate.
            by_ion: Dict[str, List[Any]] = defaultdict(list)
            for r in all_results:
                if r.prediction is not None:
                    by_ion[r.masked_group.ion_type or "unknown"].append(r.prediction)
            attr_results["prediction_quality"]["by_ion_type"] = {
                itype: {
                    "bin_accuracy": float(np.mean([p.correct_bin for p in ps])),
                    "group_bin_accuracy": (
                        float(np.mean([p.group_bin_accuracy for p in ps if p.group_bin_accuracy >= 0]))
                        if any(p.group_bin_accuracy >= 0 for p in ps)
                        else 0.0
                    ),
                    "median_error_da": float(np.median([p.error_da for p in ps])),
                    "median_error_ppm": float(np.median([p.error_ppm for p in ps])),
                    "n_predictions": len(ps),
                }
                for itype, ps in sorted(by_ion.items())
            }

        # Top-k attribution analysis (aggregated)
        attr_results["topk_analysis"] = self._aggregate_topk(all_results)

        # Summary figure
        plot_attribution_summary(
            all_results,
            save_path=self.output_dir / "ig_attribution_summary.png",
            dpi=self.dpi,
        )

        # IG convergence (completeness axiom)
        deltas = [r.convergence_delta for r in all_results]
        attr_results["convergence"] = {
            "mean_delta": float(np.mean(deltas)),
            "max_delta": float(np.max(deltas)),
        }

        # --- Novel Chemistry Discovery Probe ---
        if self.enable_novel_chemistry and all_results:
            novel_results = self._analyze_novel_chemistry(collected, all_results)
            attr_results["novel_chemistry"] = novel_results

            # Novel chemistry summary figure
            plot_novel_chemistry_summary(
                novel_results,
                save_path=self.output_dir / "novel_chemistry_summary.png",
                dpi=self.dpi,
            )

            # Patch already-saved hero JSON files with novel chemistry data
            if hasattr(self, "_novel_chemistry_per_spectrum") and self._novel_chemistry_per_spectrum:
                for json_path in hero_dir.glob("hero_*.json"):
                    try:
                        with open(json_path) as f:
                            report = json.load(f)
                        # Find spectrum idx from the report
                        spec_idx = report.get("spectrum", {}).get("_spec_idx")
                        if spec_idx is not None and spec_idx in self._novel_chemistry_per_spectrum:
                            report["novel_chemistry"] = self._novel_chemistry_per_spectrum[spec_idx]
                            with open(json_path, "w") as f:
                                json.dump(report, f, indent=2)
                    except (json.JSONDecodeError, OSError):
                        pass

        attr_results["n_spectra"] = len(collected)
        attr_results["n_masked_groups"] = len(all_results)
        attr_results["n_hero_plots"] = hero_count
        attr_results["n_ig_failures"] = self._n_ig_failures
        attr_results["success"] = True

        # --- Summary ---
        if preds:
            logger.info(
                f"  {len(all_results)} groups — bin_acc={np.mean(bin_correct) * 100:.1f}%, "
                f"group_acc={np.mean(group_acc) * 100:.1f}%, "
                f"median_err={np.median(errors):.3f}Da ({np.median(ppms):.1f}ppm)"
            )
        max_delta = float(np.max(deltas))
        if max_delta > 5.0:
            logger.warning(f"  Convergence delta max={max_delta:.2f} > 5.0 — consider increasing ig_n_steps")

        topk = attr_results.get("topk_analysis", {})
        if topk:
            t1 = topk.get("top1_distribution", {})
            ladder = topk.get("ladder_neighbor", {})
            logger.info(
                f"  Top-1: ladder={t1.get('ladder_neighbor', 0):.0%}, "
                f"unannotated={t1.get('unannotated', 0):.0%} | "
                f"Ladder top-5 hit={ladder.get('top5_hit_rate', 0):.0%}"
            )

        return attr_results

    @staticmethod
    def _select_hero_group(
        spec_results: List[AttributionResult],
        hero_count: int,
    ) -> AttributionResult:
        """Alternate hero-selection policy.

        Alternate hero-selection policy — even heroes highlight "best
        ladder" (smallest ladder_neighbor_best_rank), odd heroes highlight
        "most interesting failure" (highest top-1 unannotated fraction).
        The alternation gives a balanced view of where the model uses
        canonical chemistry vs where it falls back on noise/leakage.
        """
        if hero_count % 2 == 0:
            return max(spec_results, key=lambda r: (-r.topk.ladder_neighbor_best_rank if r.topk and r.topk.ladder_neighbor_best_rank > 0 else -999))
        return max(spec_results, key=lambda r: (r.topk.topk_fractions.get("1", {}).get("unannotated", 0) if r.topk else 0))

    def _emit_hero(
        self,
        spec_results: List[AttributionResult],
        spec_info: Dict[str, Any],
        hero_count: int,
        hero_dir: Path,
    ) -> None:
        """Select, render, and persist a hero spectrum for this spectrum."""
        selected = self._select_hero_group(spec_results, hero_count)
        seq = spec_info.get("sequence", "")
        seq_safe = seq.replace("/", "_").replace("\\", "_").replace("[", "(").replace("]", ")")[:30]
        hero_stem = f"hero_{hero_count:03d}_{seq_safe}_{selected.masked_group.group_key}"
        frag = spec_info.get("fragmentation") or "?"
        inst = spec_info.get("instrument") or "?"
        hero_title = (
            f"{seq} | z={spec_info['precursor_charge']} | "
            f"Masked: {selected.masked_group.group_key} "
            f"(m/z={selected.masked_group.base_mz:.2f}) | "
            f"{frag} | {inst}"
        )
        plot_hero_spectrum_attribution(
            mz=spec_info["mz"],
            intensity=spec_info["intensity"],
            result=selected,
            save_path=hero_dir / f"{hero_stem}.png",
            title=hero_title,
            annotations=spec_info.get("annotations"),
            max_mz=self._max_mz,
            dpi=self.dpi,
            per_peak_confidence=spec_info.get("per_peak_confidence"),
        )
        self._save_hero_report(
            hero_dir / f"{hero_stem}.json",
            selected,
            spec_info,
        )

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def _collect_annotated_spectra(
        self,
        meta: Dict[str, Any],
        dataloader: Any,
        device: torch.device,
    ) -> List[Dict[str, Any]]:
        """Collect spectra with annotations from metadata.

        Builds fragment ion groups from matched_annotation and parent_annotation.
        """
        collected: list[Any] = []

        matched_annotations = meta.get("matched_annotation")
        parent_annotations = meta.get("parent_annotation")
        feature_types = meta.get("feature_type")
        spectra_data = meta.get("spectra")
        precursor_mz = meta.get("precursor_mz")
        precursor_charge = meta.get("precursor_charge")
        spectrum_quality = meta.get("spectrum_quality")

        # Search metadata (instrument, fragmentation, etc.)
        search_instrument = meta.get("search_instrument")
        search_fragmentation = meta.get("search_fragmentation")
        search_acquisition = meta.get("search_acquisition")
        meta_collision_energy = meta.get("meta_collision_energy")
        modification_types = meta.get("modification_types")

        # Get sequences (try multiple keys)
        sequences = None
        for key in ["sequence", "peptides", "peptide", "seq"]:
            if key in meta:
                sequences = meta[key]
                break

        # Per-peak confidence (optional, for novel chemistry probe)
        per_peak_confidence = meta.get("per_peak_confidence")
        per_peak_conf_group = meta.get("per_peak_conf_group")
        per_peak_conf_offset = meta.get("per_peak_conf_offset")

        if matched_annotations is None or spectra_data is None:
            logger.warning(
                f"Missing metadata: matched_annotation={'present' if matched_annotations is not None else 'MISSING'}, "
                f"spectra={'present' if spectra_data is not None else 'MISSING'}"
            )
            return []

        n_skipped_low_quality = 0

        def _meta_str(arr: Any, idx: int) -> Any:
            if arr is None or idx >= len(arr):
                return None
            v = arr[idx]
            if v is None:
                return None
            if isinstance(v, (bytes, np.bytes_)):
                return v.decode("utf-8")
            return str(v) if v else None

        def _meta_float(arr: Any, idx: int) -> Any:
            if arr is None or idx >= len(arr):
                return None
            v = arr[idx]
            return float(v) if v is not None else None

        for i in range(min(len(spectra_data), self.max_spectra)):
            spectrum = spectra_data[i]
            if isinstance(spectrum, np.ndarray):
                spectrum = torch.from_numpy(spectrum).float()
            if spectrum.dim() == 1:
                continue

            mz_raw = spectrum[:, 0].numpy() if isinstance(spectrum, torch.Tensor) else spectrum[:, 0]
            intensity = spectrum[:, 1].numpy() if isinstance(spectrum, torch.Tensor) else spectrum[:, 1]
            # Denormalize m/z (spectra stored as normalized [0, 1] in metadata)
            mz = np.array(mz_raw, dtype=np.float64) * self._max_mz

            annotations = matched_annotations[i] if i < len(matched_annotations) else None
            if annotations is None:
                continue
            parents = parent_annotations[i] if parent_annotations is not None and i < len(parent_annotations) else None
            ftypes = feature_types[i] if feature_types is not None and i < len(feature_types) else None

            # Build fragment groups
            groups = build_fragment_groups(
                annotations=list(annotations),
                parent_annotations=list(parents) if parents is not None else None,
                feature_types=list(ftypes) if ftypes is not None else None,
                mz=np.array(mz, dtype=np.float64),
                intensity=np.array(intensity, dtype=np.float64),
            )

            # Filter to backbone groups (b/y ions) for masking targets
            backbone_groups = [g for g in groups if g.ion_type in ("b", "y")]
            if len(backbone_groups) < 2:
                continue

            seq = str(sequences[i]) if sequences is not None and i < len(sequences) else ""

            p_mz = float(precursor_mz[i]) if precursor_mz is not None else 0.0
            p_charge = int(precursor_charge[i]) if precursor_charge is not None else 2
            neutral_mass = p_mz * p_charge - p_charge * PROTON_MASS

            quality = None
            if spectrum_quality is not None and i < len(spectrum_quality):
                q = spectrum_quality[i]
                if isinstance(q, dict):
                    quality = q

            # Quality gate: skip spectra with poor annotation quality
            if quality is not None:
                bc = quality.get("backbone_coverage", 0.0)
                ng = quality.get("n_fragment_groups", 0)
                if bc < self.min_backbone_coverage or ng < self.min_fragment_groups:
                    n_skipped_low_quality += 1
                    continue

            # Extract per-peak confidence (trim to actual peak count)
            n_peaks = len(mz)
            conf_joint = None
            conf_group = None
            conf_offset = None
            if per_peak_confidence is not None and i < len(per_peak_confidence):
                raw = per_peak_confidence[i]
                if raw is not None and hasattr(raw, "__len__") and len(raw) >= n_peaks:
                    conf_joint = np.array(raw[:n_peaks], dtype=np.float64)
            if per_peak_conf_group is not None and i < len(per_peak_conf_group):
                raw = per_peak_conf_group[i]
                if raw is not None and hasattr(raw, "__len__") and len(raw) >= n_peaks:
                    conf_group = np.array(raw[:n_peaks], dtype=np.float64)
            if per_peak_conf_offset is not None and i < len(per_peak_conf_offset):
                raw = per_peak_conf_offset[i]
                if raw is not None and hasattr(raw, "__len__") and len(raw) >= n_peaks:
                    conf_offset = np.array(raw[:n_peaks], dtype=np.float64)

            collected.append(
                {
                    "idx": i,
                    "sequence": seq,
                    "spectrum": spectrum,
                    "mz": mz,
                    "intensity": np.array(intensity, dtype=np.float64),
                    "annotations": list(annotations),
                    "parent_annotations": list(parents) if parents is not None else None,
                    "all_groups": groups,
                    "backbone_groups": backbone_groups,
                    "precursor_mass": neutral_mass,
                    "precursor_charge": p_charge,
                    # Metadata for interpretation
                    "instrument": _meta_str(search_instrument, i),
                    "fragmentation": _meta_str(search_fragmentation, i),
                    "acquisition": _meta_str(search_acquisition, i),
                    "collision_energy": _meta_float(meta_collision_energy, i),
                    "modifications": _meta_str(modification_types, i),
                    "spectrum_quality": quality,
                    # Per-peak confidence (for novel chemistry probe)
                    "per_peak_confidence": conf_joint,
                    "per_peak_conf_group": conf_group,
                    "per_peak_conf_offset": conf_offset,
                }
            )

        if n_skipped_low_quality > 0:
            logger.info(
                f"Quality gate: skipped {n_skipped_low_quality} spectra "
                f"(min_backbone_coverage={self.min_backbone_coverage}, "
                f"min_fragment_groups={self.min_fragment_groups})"
            )

        return collected

    # ------------------------------------------------------------------
    # Shared helpers for novel-chemistry + hero reporting
    # ------------------------------------------------------------------

    @staticmethod
    def _try_auroc(matched: np.ndarray, unmatched: np.ndarray) -> Optional[float]:
        """Compute ROC AUROC for two score populations (matched=1, unmatched=0).

        Returns ``None`` if either population is empty, if sklearn is
        unavailable, or if ``roc_auc_score`` raises (e.g. constant input).
        """
        if len(matched) == 0 or len(unmatched) == 0:
            return None
        try:
            from sklearn.metrics import roc_auc_score

            labels = np.concatenate([np.ones(len(matched)), np.zeros(len(unmatched))])
            scores = np.concatenate([matched, unmatched])
            return float(roc_auc_score(labels, scores))
        except Exception as e:
            logger.debug("roc_auc_score failed: %s", e)
            return None

    def _novel_chem_match_kwargs(self, spec_info: Dict[str, Any]) -> Dict[str, Any]:
        """Return the ``ppm_tol`` / ``da_tol`` pair for :func:`match_peak_to_extended_chemistry` given the spectrum's fragmentation type.

        CID gets a Da tolerance (low-res ion trap); HCD/ETD/UVPD get ppm only.
        """
        frag = (spec_info.get("fragmentation") or "").strip().upper()
        da_tol = self.novel_chemistry_da_tol if frag == "CID" else None
        return {"ppm_tol": self.novel_chemistry_ppm_tol, "da_tol": da_tol}

    def _build_extended_chemistry_library(
        self,
        spec_info: Dict[str, Any],
    ) -> Optional[Any]:
        """Build (and cache on ``spec_info``) the extended-chemistry library for the given spectrum.

        Returns ``None`` when the sequence is empty or too short, or when library construction fails. Shared by :meth:`_build_novel_chem_lookup` and
        :meth:`_analyze_novel_chemistry` so the library is constructed exactly once per spectrum.
        """
        if "novel_chem_library" in spec_info:
            return spec_info["novel_chem_library"]

        sequence = spec_info.get("sequence", "") or ""
        if not sequence:
            spec_info["novel_chem_library"] = None
            return None
        residues = self._parse_residues(sequence)
        if len(residues) < 2:
            spec_info["novel_chem_library"] = None
            return None

        precursor_charge = spec_info.get("precursor_charge", 2)
        prec_mass = spec_info.get("precursor_mass", 0.0)
        prec_mz = (prec_mass + precursor_charge * PROTON_MASS) / precursor_charge if prec_mass > 0 else 0.0
        try:
            library = build_extended_chemistry_library(
                residues=residues,
                precursor_charge=precursor_charge,
                precursor_mz=prec_mz,
                max_internal_length=self.novel_chemistry_max_internal_length,
                residue_mass_fn=self._residue_mass,
            )
        except Exception as e:
            logger.debug(
                "Extended-chemistry library build failed for spec %s: %s",
                spec_info.get("idx"),
                e,
            )
            spec_info["novel_chem_library"] = None
            return None
        spec_info["novel_chem_library"] = library
        return library

    def _build_novel_chem_lookup(
        self,
        spec_info: Dict[str, Any],
    ) -> Tuple[Dict[int, str], Dict[int, List[Dict[str, Any]]]]:
        """Precompute ``peak_idx → extended_chem_type`` for this spectrum.

        Runs the extended-chemistry matcher over every unannotated, positive-
        intensity peak and picks a canonical type per peak. Returns:

        1. ``type_lookup`` — ``peak_idx → single extended-chemistry type``.
           Internal fragments are preferred when a peak matches multiple
           types (the dominant class in typical HCD/CID spectra).
        2. ``full_matches`` — ``peak_idx → full list of match dicts`` (used
           later when writing per-spectrum ``novel_chemistry`` blocks to avoid
           recomputing matches).

        Called once per spectrum before IG, so every masked-group
        classification in that spectrum sees a consistent extended-chemistry
        re-labeling of unannotated peaks.
        """
        type_lookup: Dict[int, str] = {}
        full_matches: Dict[int, List[Dict[str, Any]]] = {}

        library = self._build_extended_chemistry_library(spec_info)
        if library is None or library.total_hypotheses == 0:
            return type_lookup, full_matches

        mz = spec_info["mz"]
        intensity = spec_info["intensity"]
        annotations = spec_info.get("annotations") or []
        n_peaks = len(mz)
        match_kwargs = self._novel_chem_match_kwargs(spec_info)

        # Priority when a peak matches multiple types — internal fragments are
        # by far the most common extended-chemistry signal; everything else
        # goes in declaration order (fragmentation-type specific first).
        type_priority = (
            "internal_fragment",
            "w_ion",
            "d_ion",
            "side_chain_loss",
            "precursor_combined_loss",
            "immonium_related",
        )

        for pi in range(n_peaks):
            if intensity[pi] <= 0:
                continue
            ann = annotations[pi] if pi < len(annotations) else None
            if ann is not None and str(ann) != "None" and str(ann).strip() != "":
                continue  # skip already-annotated peaks
            matches = match_peak_to_extended_chemistry(float(mz[pi]), library, **match_kwargs)
            if not matches:
                continue
            match_dicts = [
                {
                    "type": m.hypothesis_type,
                    "label": m.hypothesis_label,
                    "expected_mz": round(m.expected_mz, 5),
                    "error_da": round(m.error_da, 5),
                    "error_ppm": round(m.error_ppm, 2),
                    "residue": m.residue,
                    "position": m.position,
                    "discriminates_leu_ile": m.discriminates_leu_ile,
                }
                for m in matches
            ]
            full_matches[pi] = match_dicts
            matched_types = {m.hypothesis_type for m in matches}
            for t in type_priority:
                if t in matched_types:
                    type_lookup[pi] = t
                    break
            else:
                # Unknown type not in priority list — take first match
                type_lookup[pi] = matches[0].hypothesis_type

        return type_lookup, full_matches

    # ------------------------------------------------------------------
    # IG attribution (per spectrum / per masked group)
    # ------------------------------------------------------------------

    def _attribute_spectrum(
        self,
        model: nn.Module,
        spec_info: Dict[str, Any],
        device: torch.device,
    ) -> List[AttributionResult]:
        """Run IG attribution for selected fragment groups in a spectrum."""
        results: list[Any] = []

        spectrum = spec_info["spectrum"]
        if isinstance(spectrum, np.ndarray):
            spectrum = torch.from_numpy(spectrum).float()
        spectra = spectrum.unsqueeze(0).to(device)  # (1, L, 2)

        # Padding mask: peaks with m/z=0 AND intensity=0 are padding.
        # This remains correct during IG interpolation because the zero-intensity
        # baseline preserves m/z values — only true padding (m/z=0) sums to 0.
        spectra_mask = (spectra.sum(dim=-1) == 0).to(device)

        # Precompute once per spectrum — novel-chem matching is sequence-based
        # and independent of which group we mask, so feeding the lookup into
        # every masked-group classification keeps categories consistent across
        # groups and avoids rebuilding the extended-chemistry library per IG run.
        if "novel_chem_type_lookup" not in spec_info:
            type_lookup, full_matches = self._build_novel_chem_lookup(spec_info)
            spec_info["novel_chem_type_lookup"] = type_lookup
            spec_info["novel_chem_full_matches"] = full_matches

        backbone_groups = spec_info["backbone_groups"]
        rng = np.random.RandomState(spec_info["idx"])
        multi_peak = [i for i, g in enumerate(backbone_groups) if len(g.peak_indices) > 1]
        single_peak = [i for i, g in enumerate(backbone_groups) if len(g.peak_indices) == 1]

        n_to_mask = min(self.max_groups_per_spectrum, len(backbone_groups))
        if len(multi_peak) >= n_to_mask:
            selected_indices = rng.choice(multi_peak, size=n_to_mask, replace=False)
        else:
            selected_indices = list(multi_peak)
            remaining = n_to_mask - len(multi_peak)
            if single_peak and remaining > 0:
                extra = rng.choice(single_peak, size=min(remaining, len(single_peak)), replace=False)
                selected_indices.extend(extra)
            selected_indices = np.array(selected_indices)

        for gi in selected_indices:
            masked_group = backbone_groups[gi]
            try:
                result = self._ig_single_group(
                    model,
                    spectra,
                    masked_group,
                    spectra_mask,
                    spec_info,
                    device,
                )
                if result is not None:
                    results.append(result)
            except Exception as e:
                # Count and log; don't swap devices mid-loop (the prior
                # GPU→CPU fallback reassigned ``model.to(...)`` on both
                # success and failure paths, which left the model on CPU
                # on partial failures and broke subsequent spectra).
                self._n_ig_failures += 1
                logger.warning(
                    "IG failed for spec %s group %s: %s",
                    spec_info.get("idx"),
                    masked_group.group_key,
                    e,
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return results

    def _ig_single_group(
        self,
        model: nn.Module,
        spectra: torch.Tensor,
        masked_group: FragmentGroup,
        spectra_mask: torch.Tensor,
        spec_info: Dict[str, Any],
        device: torch.device,
    ) -> Optional[AttributionResult]:
        """Run IG attribution for a single masked fragment ion group.

        MLM-masks the entire fragment group (training-aligned) and uses a
        zero-intensity baseline (m/z preserved, intensities zeroed) so IG
        interpolates only along the intensity dimension.
        """
        if masked_group.base_idx is None:
            return None

        base_idx = masked_group.base_idx

        # --- IG: MLM-mask entire group ---
        target = PredictionTarget(
            model=model,
            masked_indices=masked_group.peak_indices,
            spectra_mask=spectra_mask,
        )
        baseline = spectra.clone()
        baseline[:, :, 1] = 0.0  # zero intensity, keep m/z
        ig = IntegratedGradients(target)
        attr_raw, delta = ig.attribute(
            spectra,
            baselines=baseline,
            n_steps=self.ig_n_steps,
            internal_batch_size=self.ig_internal_batch_size,
            return_convergence_delta=True,
        )
        attr = reduce_attributions_to_peaks(attr_raw, reduction="abs_sum")
        attr_np = attr[0].detach().cpu().numpy()

        # --- Extract prediction quality ---
        true_mz_all = {idx: float(spec_info["mz"][idx]) for idx in masked_group.peak_indices}
        prediction = extract_prediction(
            model=model,
            spectra=spectra,
            masked_indices=masked_group.peak_indices,
            base_idx=base_idx,
            spectra_mask=spectra_mask,
            true_mz_da=masked_group.base_mz,
            true_mz_all=true_mz_all,
            annotations=spec_info.get("annotations"),
            max_mz=self._max_mz,
        )

        # Detect charge-state variants of the masked group among visible peaks.
        # Labeled as leakage rather than "unannotated" — the model can trivially
        # derive the singly-charged base m/z from its doubly-charged sibling.
        masked_charge = _charge_from_group_key(masked_group.group_key)
        # Feed the detector every m/z in the masked group — not just the base.
        # A z=2 variant of an isotope or neutral-loss sibling (e.g. y8++-H2O,
        # or y8++[+1] sitting exactly 0.5 Da off the base z=2) would otherwise
        # slip through and show up as "unannotated" high-attribution leakage.
        sibling_mz = [
            float(spec_info["mz"][pi]) for pi in masked_group.peak_indices if pi != masked_group.base_idx and 0 <= pi < len(spec_info["mz"])
        ]
        charge_variant_indices = detect_charge_variant_indices(
            masked_base_mz=masked_group.base_mz,
            masked_charge=masked_charge,
            peak_mz_array=np.asarray(spec_info["mz"]),
            extra_masked_mz=sibling_mz,
        )
        # Don't treat the masked group's own peaks as variants of themselves.
        charge_variant_indices.difference_update(set(masked_group.peak_indices))

        # --- Top-k attribution analysis ---
        topk = compute_topk_attribution_analysis(
            attr_np,
            masked_group=masked_group,
            all_groups=spec_info["all_groups"],
            annotations=spec_info["annotations"],
            mz=spec_info["mz"],
            precursor_mass=spec_info.get("precursor_mass", 0.0),
            charge_variant_indices=charge_variant_indices,
            novel_chem_lookup=spec_info.get("novel_chem_type_lookup"),
        )

        return AttributionResult(
            spectrum_idx=spec_info["idx"],
            masked_group=masked_group,
            attributions=attr_np,
            prediction=prediction,
            convergence_delta=float(delta.abs().mean()),
            topk=topk,
            charge_variant_indices=charge_variant_indices,
        )

    def _aggregate_topk(
        self,
        results: List[AttributionResult],
    ) -> Dict[str, Any]:
        """Aggregate top-k attribution analysis across all masked groups."""
        analyses = [r.topk for r in results if r.topk is not None]
        if not analyses:
            return {}

        top1_cats = [a.top1_category for a in analyses]
        categories = list(STRUCTURAL_CATEGORIES)
        top1_dist = {c: sum(1 for t in top1_cats if t == c) / len(top1_cats) for c in categories}

        k_values = list(analyses[0].topk_fractions.keys()) if analyses else []
        avg_topk: Dict[str, Dict[str, float]] = {}
        for k in k_values:
            avg_topk[k] = {cat: float(np.mean([a.topk_fractions[k].get(cat, 0.0) for a in analyses])) for cat in categories}

        ladder_present = [a.ladder_neighbors_present for a in analyses]
        ladder_in_top5 = [a.ladder_neighbors_in_top5 for a in analyses]
        ladder_best_ranks = [a.ladder_neighbor_best_rank for a in analyses if a.ladder_neighbor_best_rank > 0]
        ladder_top5_hit_rate = sum(1 for a in analyses if a.ladder_neighbors_in_top5 > 0) / len(analyses)

        return {
            "top1_distribution": top1_dist,
            "topk_fractions": avg_topk,
            "ladder_neighbor": {
                "mean_present": float(np.mean(ladder_present)),
                "mean_in_top5": float(np.mean(ladder_in_top5)),
                "top5_hit_rate": float(ladder_top5_hit_rate),
                "median_best_rank": float(np.median(ladder_best_ranks)) if ladder_best_ranks else -1,
                # Ladder-utilisation funnel (persist the fractions used in the figure panel):
                # "ladder present" is a data feature --- fraction of masked groups with >=1
                # ladder neighbour present in the spectrum; the other two are attribution results.
                "fraction_present": float(np.mean([1.0 if p > 0 else 0.0 for p in ladder_present])),
                "fraction_top1_ladder_or_near": float(
                    sum(1 for a in analyses if a.top1_category in ("ladder_neighbor", "near_ladder")) / len(analyses)
                ),
            },
            "concentration": {
                "mean_top5": float(np.mean([a.top5_concentration for a in analyses])),
                "mean_top10": float(np.mean([a.top10_concentration for a in analyses])),
                "mean_gini": float(np.mean([a.gini for a in analyses])),
            },
            "n": len(analyses),
        }

    # ------------------------------------------------------------------
    # Novel Chemistry Discovery Probe
    # ------------------------------------------------------------------

    def _run_null_models(
        self,
        library: Any,
        residues: List[str],
        precursor_charge: int,
        prec_mz: float,
        mz: np.ndarray,
        unannotated_indices: np.ndarray,
        match_kwargs: Dict[str, Any],
        null_shifts: List[float],
        n_scramble_trials: int,
        spec_idx: int,
    ) -> Dict[str, int]:
        """Run the three novel-chemistry null models for one spectrum.

        Returns per-null hit/total counts as a dict so the caller can
        accumulate across the whole collection without sharing mutable state.

        Null 1 — shift m/z by fixed Da offsets (random-mass baseline).
        Null 2 — scramble the peptide sequence and rebuild the library
        (tests sequence-specificity for order-dependent ion types).
        Null 3 — immonium composition null: match against ions from amino
        acids *absent* from the sequence (only runs when both populations
        exist so the comparison is symmetric).
        """
        counts: dict[str, Any] = {
            "shift_hits": 0,
            "shift_total": 0,
            "scramble_hits": 0,
            "scramble_total": 0,
            "immonium_true_hits": 0,
            "immonium_true_total": 0,
            "immonium_null_hits": 0,
            "immonium_null_total": 0,
        }

        # Null 1 — shift
        for pi in unannotated_indices:
            peak_mz = float(mz[pi])
            for shift in null_shifts:
                if match_peak_to_extended_chemistry(peak_mz + shift, library, **match_kwargs):
                    counts["shift_hits"] += 1
                counts["shift_total"] += 1

        # Null 2 — scramble
        rng = np.random.RandomState(spec_idx)
        for _ in range(n_scramble_trials):
            scrambled = list(residues)
            rng.shuffle(scrambled)
            scrambled_lib = build_extended_chemistry_library(
                residues=scrambled,
                precursor_charge=precursor_charge,
                precursor_mz=prec_mz,
                max_internal_length=self.novel_chemistry_max_internal_length,
                residue_mass_fn=self._residue_mass,
            )
            for pi in unannotated_indices:
                if match_peak_to_extended_chemistry(float(mz[pi]), scrambled_lib, **match_kwargs):
                    counts["scramble_hits"] += 1
                counts["scramble_total"] += 1

        # Null 3 — immonium composition (absent-AA control)
        neg_control_ions = compute_immonium_related_negative_control(residues)
        if library.immonium_related and neg_control_ions:
            neg_lib = ExtendedChemistryLibrary(
                sequence="",
                precursor_charge=0,
                immonium_related=neg_control_ions,
            )
            for pi in unannotated_indices:
                peak_mz_val = float(mz[pi])
                true_matches = [
                    m
                    for m in match_peak_to_extended_chemistry(
                        peak_mz_val,
                        library,
                        **match_kwargs,
                    )
                    if m.hypothesis_type == "immonium_related"
                ]
                if true_matches:
                    counts["immonium_true_hits"] += 1
                counts["immonium_true_total"] += 1
                if match_peak_to_extended_chemistry(peak_mz_val, neg_lib, **match_kwargs):
                    counts["immonium_null_hits"] += 1
                counts["immonium_null_total"] += 1

        return counts

    def _analyze_novel_chemistry(
        self,
        collected: List[Dict[str, Any]],
        all_results: List[AttributionResult],
    ) -> Dict[str, Any]:
        """Identify peaks where the model has learned chemistry beyond standard annotations.

        Analyses (all run on ALL unannotated peaks, no pre-filtering):
        1. Extended chemistry matching — match unannotated peaks against d/w-ions,
           internal fragments, and residue-specific side-chain losses.
        2. Unbiased confidence comparison — compare confidence of matched vs unmatched
           unannotated peaks (tests whether the model's confidence tracks extended chemistry).
        3. Sequence-scramble null model — scramble the peptide sequence, rebuild the
           library, and retest (proves sequence-specificity, not just mass coincidence).
        4. Attribution quadrant analysis — correlate IG attribution rank with extended
           chemistry match status (tests whether the model causally uses these peaks).
        """
        logger.info("Running novel chemistry discovery probe...")

        # Build lookup: spectrum_idx -> list of AttributionResult
        results_by_spectrum: Dict[int, List[AttributionResult]] = {}
        for r in all_results:
            results_by_spectrum.setdefault(r.spectrum_idx, []).append(r)

        # Per-spectrum novel chemistry data (for hero reports)
        self._novel_chemistry_per_spectrum: Dict[int, Dict[str, Any]] = {}

        ALL_TYPES = ["d_ion", "w_ion", "internal_fragment", "side_chain_loss", "immonium_related", "precursor_combined_loss"]  # noqa: N806

        # Accumulators — operate on ALL unannotated peaks (no confidence pre-filter)
        n_leu_ile = 0
        by_type: Dict[str, int] = dict.fromkeys(ALL_TYPES, 0)
        by_frag: Dict[str, Dict[str, int]] = {}
        # Unbiased confidence comparison (all unannotated peaks)
        conf_matched_all: List[float] = []
        conf_unmatched_all: List[float] = []
        # Shift-based null
        shift_null_hits = 0
        shift_null_total = 0
        # Scramble null (sequence-order dependent types: internal, side_chain, d, w)
        scramble_null_hits = 0
        scramble_null_total = 0
        # Immonium composition null (test against amino acids NOT in the peptide)
        immonium_true_hits = 0
        immonium_true_total = 0
        immonium_null_hits = 0
        immonium_null_total = 0
        # Attribution quadrant analysis
        attr_matched: List[float] = []
        attr_unmatched: List[float] = []

        null_shifts = list(SHIFT_NULL_DISTANCES)
        n_scramble_trials = self.novel_chemistry_scramble_trials
        has_confidence = False
        total_unannotated = 0
        total_unannotated_deleakage = 0
        total_matched = 0
        total_charge_variant_peaks = 0

        for spec_info in collected:
            spec_idx = spec_info["idx"]
            spec_results = results_by_spectrum.get(spec_idx, [])
            if not spec_results:
                continue

            annotations = spec_info["annotations"]
            mz = spec_info["mz"]
            intensity = spec_info["intensity"]
            n_peaks = len(mz)
            conf_joint = spec_info.get("per_peak_confidence")
            frag_type = (spec_info.get("fragmentation") or "").strip().upper()
            match_kwargs = self._novel_chem_match_kwargs(spec_info)

            # Identify ALL unannotated peaks (no confidence filter)
            unannotated_mask = np.zeros(n_peaks, dtype=bool)
            for pi in range(n_peaks):
                if intensity[pi] <= 0:
                    continue
                ann = annotations[pi] if pi < len(annotations) else None
                if ann is None or str(ann) == "None" or str(ann).strip() == "":
                    unannotated_mask[pi] = True

            unannotated_indices = np.where(unannotated_mask)[0]
            if len(unannotated_indices) == 0:
                continue

            # Union of charge-state variants across every masked group in this
            # spectrum — any unannotated peak that is a charge-variant of any
            # masked group is information leakage, not novel chemistry.
            spec_charge_variants: Set[int] = set()
            for r in spec_results:
                if r.charge_variant_indices:
                    spec_charge_variants.update(r.charge_variant_indices)
            unannotated_charge_variant_indices = [int(pi) for pi in unannotated_indices if int(pi) in spec_charge_variants]
            total_charge_variant_peaks += len(unannotated_charge_variant_indices)

            if conf_joint is not None:
                has_confidence = True

            # Aggregate max |attribution| per visible peak across all masked groups
            peak_attribution = np.zeros(n_peaks, dtype=np.float64)
            for r in spec_results:
                masked_set = set(r.masked_group.peak_indices)
                abs_attr = np.abs(r.attributions)
                for pi in range(min(len(abs_attr), n_peaks)):
                    if pi not in masked_set:
                        peak_attribution[pi] = max(peak_attribution[pi], abs_attr[pi])

            # Extended-chemistry library — built once per spectrum in
            # ``_build_extended_chemistry_library`` and cached on spec_info.
            library = self._build_extended_chemistry_library(spec_info)
            if library is None or library.total_hypotheses == 0:
                continue
            # residues/precursor_charge still needed below for scramble null
            residues = self._parse_residues(spec_info.get("sequence", "") or "")
            precursor_charge = spec_info.get("precursor_charge", 2)
            prec_mass = spec_info.get("precursor_mass", 0.0)
            prec_mz = (prec_mass + precursor_charge * PROTON_MASS) / precursor_charge if prec_mass > 0 else 0.0

            # --- Match ALL unannotated peaks ---
            spec_matched_candidates: List[Dict[str, Any]] = []
            for pi in unannotated_indices:
                pi_int = int(pi)
                peak_mz = float(mz[pi])
                peak_conf = float(conf_joint[pi]) if conf_joint is not None else None
                peak_attr = float(peak_attribution[pi])
                is_charge_variant = pi_int in spec_charge_variants

                matches = match_peak_to_extended_chemistry(peak_mz, library, **match_kwargs)

                total_unannotated += 1
                if not is_charge_variant:
                    total_unannotated_deleakage += 1
                if matches:
                    total_matched += 1
                    if peak_conf is not None:
                        conf_matched_all.append(peak_conf)
                    attr_matched.append(peak_attr)

                    matched_types = set()
                    for m in matches:
                        matched_types.add(m.hypothesis_type)
                        if m.discriminates_leu_ile:
                            n_leu_ile += 1
                    for ht in matched_types:
                        by_type[ht] = by_type.get(ht, 0) + 1
                        frag_key = frag_type or "UNKNOWN"
                        by_frag.setdefault(frag_key, {})
                        by_frag[frag_key][ht] = by_frag[frag_key].get(ht, 0) + 1

                    spec_matched_candidates.append(
                        {
                            "peak_idx": int(pi),
                            "mz": peak_mz,
                            "intensity": float(intensity[pi]),
                            "confidence": peak_conf,
                            "attribution_max_over_groups": peak_attr,
                            "is_charge_variant_of_masked": is_charge_variant,
                            "matches": [
                                {
                                    "type": m.hypothesis_type,
                                    "label": m.hypothesis_label,
                                    "expected_mz": round(m.expected_mz, 5),
                                    "error_da": round(m.error_da, 5),
                                    "error_ppm": round(m.error_ppm, 2),
                                    "residue": m.residue,
                                    "position": m.position,
                                    "discriminates_leu_ile": m.discriminates_leu_ile,
                                }
                                for m in matches
                            ],
                        }
                    )
                else:
                    if peak_conf is not None:
                        conf_unmatched_all.append(peak_conf)
                    attr_unmatched.append(peak_attr)

            # Run all three null models — see :meth:`_run_null_models`.
            null_counts = self._run_null_models(
                library=library,
                residues=residues,
                precursor_charge=precursor_charge,
                prec_mz=prec_mz,
                mz=mz,
                unannotated_indices=unannotated_indices,
                match_kwargs=match_kwargs,
                null_shifts=null_shifts,
                n_scramble_trials=n_scramble_trials,
                spec_idx=spec_idx,
            )
            shift_null_hits += null_counts["shift_hits"]
            shift_null_total += null_counts["shift_total"]
            scramble_null_hits += null_counts["scramble_hits"]
            scramble_null_total += null_counts["scramble_total"]
            immonium_true_hits += null_counts["immonium_true_hits"]
            immonium_true_total += null_counts["immonium_true_total"]
            immonium_null_hits += null_counts["immonium_null_hits"]
            immonium_null_total += null_counts["immonium_null_total"]

            # Store per-spectrum results for hero reports
            if spec_matched_candidates:
                self._novel_chemistry_per_spectrum[spec_idx] = {
                    "n_unannotated": len(unannotated_indices),
                    "n_unannotated_deleakage": int(len(unannotated_indices) - len(unannotated_charge_variant_indices)),
                    "n_charge_variant_peaks": len(unannotated_charge_variant_indices),
                    "n_matched": len(spec_matched_candidates),
                    "candidates": spec_matched_candidates,
                }

        result = self._aggregate_novel_chemistry_results(
            total_unannotated=total_unannotated,
            total_unannotated_deleakage=total_unannotated_deleakage,
            total_charge_variant_peaks=total_charge_variant_peaks,
            total_matched=total_matched,
            by_type=by_type,
            by_frag=by_frag,
            n_leu_ile=n_leu_ile,
            shift_null_hits=shift_null_hits,
            shift_null_total=shift_null_total,
            scramble_null_hits=scramble_null_hits,
            scramble_null_total=scramble_null_total,
            n_scramble_trials=n_scramble_trials,
            null_shifts=null_shifts,
            immonium_true_hits=immonium_true_hits,
            immonium_true_total=immonium_true_total,
            immonium_null_hits=immonium_null_hits,
            immonium_null_total=immonium_null_total,
            conf_matched=conf_matched_all,
            conf_unmatched=conf_unmatched_all,
            attr_matched=attr_matched,
            attr_unmatched=attr_unmatched,
            has_confidence=has_confidence,
        )

        conf_auroc = result["confidence_profile"].get("auroc")
        conf_str = f", conf_auroc={conf_auroc:.4f}" if conf_auroc is not None else ""
        logger.info(
            "  Novel chemistry: %d/%d matched (%.1f%%), scramble enrichment=%.1fx%s",
            total_matched,
            total_unannotated,
            result["overall_hit_rate"] * 100,
            result["null_models"]["scramble"]["enrichment"],
            conf_str,
        )
        return result

    def _aggregate_novel_chemistry_results(
        self,
        *,
        total_unannotated: int,
        total_unannotated_deleakage: int,
        total_charge_variant_peaks: int,
        total_matched: int,
        by_type: Dict[str, int],
        by_frag: Dict[str, Dict[str, int]],
        n_leu_ile: int,
        shift_null_hits: int,
        shift_null_total: int,
        scramble_null_hits: int,
        scramble_null_total: int,
        n_scramble_trials: int,
        null_shifts: List[float],
        immonium_true_hits: int,
        immonium_true_total: int,
        immonium_null_hits: int,
        immonium_null_total: int,
        conf_matched: List[float],
        conf_unmatched: List[float],
        attr_matched: List[float],
        attr_unmatched: List[float],
        has_confidence: bool,
    ) -> Dict[str, Any]:
        """Assemble the novel-chemistry result dict from per-spectrum counts.

        Pure aggregation (no iteration over spectra) — safe to test in
        isolation with synthetic counts.
        """

        def _rate(hits: int, total: int) -> float:
            return hits / total if total > 0 else 0.0

        def _enrichment(real: float, null: float) -> float:
            if null > 0:
                return real / null
            return float("inf") if real > 0 else 0.0

        overall_hit_rate = _rate(total_matched, total_unannotated)
        # De-leakage rate: drop charge-state variants of masked groups from
        # the denominator. A y5++ peak (when y5+ is masked) is information
        # leakage, not novel chemistry, and counting it biases the rate down.
        overall_hit_rate_deleakage = _rate(total_matched, total_unannotated_deleakage)
        shift_null_rate = _rate(shift_null_hits, shift_null_total)
        scramble_null_rate = _rate(scramble_null_hits, scramble_null_total)
        immonium_true_rate = _rate(immonium_true_hits, immonium_true_total)
        immonium_null_rate = _rate(immonium_null_hits, immonium_null_total)
        hit_rates = {k: _rate(v, total_unannotated) for k, v in by_type.items()}

        confidence_profile = self._summarize_score_populations(
            np.asarray(conf_matched, dtype=np.float64),
            np.asarray(conf_unmatched, dtype=np.float64),
            prefix="conf",
        )
        attribution_profile = self._summarize_score_populations(
            np.asarray(attr_matched, dtype=np.float64),
            np.asarray(attr_unmatched, dtype=np.float64),
            prefix="attr",
        )

        return {
            "n_unannotated_peaks": total_unannotated,
            "n_unannotated_peaks_deleakage": total_unannotated_deleakage,
            "n_charge_variant_peaks": total_charge_variant_peaks,
            "n_matched": total_matched,
            "overall_hit_rate": overall_hit_rate,
            "overall_hit_rate_deleakage": overall_hit_rate_deleakage,
            "hit_rates": hit_rates,
            "null_models": {
                "shift": {
                    "hit_rate": shift_null_rate,
                    "enrichment": _enrichment(overall_hit_rate, shift_null_rate),
                    "shifts_da": null_shifts,
                },
                "scramble": {
                    "hit_rate": scramble_null_rate,
                    "enrichment": _enrichment(overall_hit_rate, scramble_null_rate),
                    "n_trials": n_scramble_trials,
                },
                "immonium_composition": {
                    "true_rate": immonium_true_rate,
                    "null_rate": immonium_null_rate,
                    "enrichment": _enrichment(immonium_true_rate, immonium_null_rate),
                },
            },
            "n_leu_ile_discriminators": n_leu_ile,
            "by_fragmentation_type": by_frag,
            "confidence_profile": confidence_profile,
            "attribution_profile": attribution_profile,
            "has_confidence_data": has_confidence,
            "config": {
                "max_internal_length": self.novel_chemistry_max_internal_length,
                "ppm_tol": self.novel_chemistry_ppm_tol,
                "da_tol": self.novel_chemistry_da_tol,
            },
        }

    def _summarize_score_populations(
        self,
        matched: np.ndarray,
        unmatched: np.ndarray,
        prefix: str,
    ) -> Dict[str, Any]:
        """Mean/median/AUROC summary over two score populations (matched=1 vs unmatched=0).

        Returns an empty dict if both are empty.
        """
        if len(matched) == 0 and len(unmatched) == 0:
            return {}
        summary: Dict[str, Any] = {
            f"mean_{prefix}_matched": float(np.mean(matched)) if len(matched) else 0.0,
            f"mean_{prefix}_unmatched": float(np.mean(unmatched)) if len(unmatched) else 0.0,
            f"median_{prefix}_matched": float(np.median(matched)) if len(matched) else 0.0,
            f"median_{prefix}_unmatched": float(np.median(unmatched)) if len(unmatched) else 0.0,
            "n_matched": int(len(matched)),
            "n_unmatched": int(len(unmatched)),
        }
        auroc = self._try_auroc(matched, unmatched)
        if auroc is not None:
            summary["auroc"] = auroc
        return summary

    # ------------------------------------------------------------------
    # Peptide-sequence helpers (shared with novel-chem + hero report)
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_residues(sequence: str) -> List[str]:
        """Parse a peptide sequence into residues, handling UNIMOD modifications.

        Returns a list of residue strings, e.g. ["V", "C[UNIMOD:4]", "E", "D"].
        """
        residues: list[Any] = []
        i = 0
        while i < len(sequence):
            if sequence[i].isupper():
                res = sequence[i]
                # Check for modification bracket
                if i + 1 < len(sequence) and sequence[i + 1] == "[":
                    end = sequence.index("]", i + 1)
                    res += sequence[i + 1 : end + 1]
                    i = end + 1
                else:
                    i += 1
                residues.append(res)
            else:
                i += 1
        return residues

    @staticmethod
    def _residue_mass(residue: str) -> float:
        """Get mass for a residue, including UNIMOD modifications."""
        # Known UNIMOD mass shifts
        unimod_masses: dict[str, Any] = {
            "4": 57.021464,  # Carbamidomethyl (C)
            "35": 15.994915,  # Oxidation (M)
            "1": 42.010565,  # Acetyl (N-term)
            "21": 79.966331,  # Phospho (S/T/Y)
            "259": 8.014199,  # Label:13C(6)15N(2) (K/R)
            "7": 0.984016,  # Deamidated (N/Q)
            "28": -17.026549,  # Glu->pyro-Glu
            "27": -18.010565,  # Gln->pyro-Glu
        }
        aa = residue[0]
        base = AMINO_ACID_MASSES.get(aa, AMINO_ACID_MASSES.get("L/I", 0.0) if aa in ("L", "I") else 0.0)

        # Check for UNIMOD modification
        if "[UNIMOD:" in residue:
            mod_id = residue.split("UNIMOD:")[1].rstrip("]")
            base += unimod_masses.get(mod_id, 0.0)
        elif "[" in residue:
            # Other bracket notation — try to extract number
            pass

        return base

    # ------------------------------------------------------------------
    # Hero report
    # ------------------------------------------------------------------

    def _compute_sequence_context(
        self,
        masked_group: "FragmentGroup",
        sequence: str,
        precursor_mass: float,
    ) -> Optional[Dict[str, Any]]:
        """Compute sequence context for a masked fragment group.

        Returns cleavage site info, expected ladder neighbors, and complementary ion.
        """
        residues = self._parse_residues(sequence)
        L = len(residues)  # noqa: N806
        if L < 2:
            return None

        ion_type = masked_group.ion_type
        group_key = masked_group.group_key
        base_mz = masked_group.base_mz

        # Delegate to the canonical series-info parser (handles "b4+", "y13++",
        # "b6+-H2O", "y10+[+1]" etc. consistently with the classifier).
        parsed = _parse_series_info(group_key)
        if parsed is None:
            return None
        _, position, charge = parsed
        charge_str = "+" * charge

        if ion_type == "b":
            if position < 1 or position > L:
                return None
            spanned = residues[:position]
            cleavage_left = residues[position - 1][0] if position >= 1 else "?"
            cleavage_right = residues[position][0] if position < L else "?"
            cleavage_pos = position  # 1-indexed

            # Ladder neighbors
            left_gap_residue = residues[position - 1][0] if position >= 1 else None
            left_gap_mass = self._residue_mass(residues[position - 1]) if position >= 1 else 0
            right_gap_residue = residues[position][0] if position < L else None
            right_gap_mass = self._residue_mass(residues[position]) if position < L else 0

            ladder_left_ann = f"b{position - 1}{charge_str}" if position > 1 else None
            ladder_left_mz = base_mz - left_gap_mass if position > 1 else None
            ladder_right_ann = f"b{position + 1}{charge_str}" if position < L else None
            ladder_right_mz = base_mz + right_gap_mass if position < L else None

            # Complementary y-ion (always report at z=1 as primary)
            comp_pos = L - position
            comp_ann = f"y{comp_pos}+"
            comp_mz = precursor_mass + 2 * PROTON_MASS - base_mz

        elif ion_type == "y":
            if position < 1 or position > L:
                return None
            spanned = residues[L - position :]
            cleavage_left = residues[L - position - 1][0] if L - position - 1 >= 0 else "?"
            cleavage_right = residues[L - position][0] if L - position < L else "?"
            cleavage_pos = L - position + 1  # 1-indexed

            # For y-ions: y(n-1) removes the N-terminal residue of the fragment
            left_gap_residue = residues[L - position][0] if L - position < L else None
            left_gap_mass = self._residue_mass(residues[L - position]) if L - position < L else 0
            # y(n+1) adds the residue just N-terminal to the fragment
            right_gap_residue = residues[L - position - 1][0] if L - position - 1 >= 0 else None
            right_gap_mass = self._residue_mass(residues[L - position - 1]) if L - position - 1 >= 0 else 0

            ladder_left_ann = f"y{position - 1}{charge_str}" if position > 1 else None
            ladder_left_mz = base_mz - left_gap_mass if position > 1 else None
            ladder_right_ann = f"y{position + 1}{charge_str}" if position < L else None
            ladder_right_mz = base_mz + right_gap_mass if position < L else None

            # Complementary b-ion (always report at z=1 as primary)
            comp_pos = L - position
            comp_ann = f"b{comp_pos}+"
            comp_mz = precursor_mass + 2 * PROTON_MASS - base_mz
        else:
            return None

        context: dict[str, Any] = {
            "peptide_length": L,
            "cleavage_position": cleavage_pos,
            "cleavage_left_residue": cleavage_left,
            "cleavage_right_residue": cleavage_right,
            "residues_spanned": "".join(r[0] for r in spanned),
        }

        if ladder_left_ann and ladder_left_mz:
            context["expected_ladder_left"] = {
                "annotation": ladder_left_ann,
                "expected_mz": round(ladder_left_mz, 4),
                "gap_residue": left_gap_residue,
                "gap_mass": round(left_gap_mass, 4),
            }
        if ladder_right_ann and ladder_right_mz:
            context["expected_ladder_right"] = {
                "annotation": ladder_right_ann,
                "expected_mz": round(ladder_right_mz, 4),
                "gap_residue": right_gap_residue,
                "gap_mass": round(right_gap_mass, 4),
            }
        context["expected_complementary"] = {
            "annotation": comp_ann,
            "expected_mz_z1": round(comp_mz, 4),
            "expected_mz_z2": round((comp_mz + PROTON_MASS) / 2, 4) if comp_mz > 0 else None,
            "expected_mz_z3": round((comp_mz + 2 * PROTON_MASS) / 3, 4) if comp_mz > 0 else None,
        }

        return context

    def _extract_model_config(self) -> Dict[str, Any]:
        """Extract key architectural settings from model config for interpretation."""
        cfg = self._model_config
        if cfg is None:
            return {}

        def _get(obj: Any, *keys, default: Any = None) -> Any:  # type: ignore[no-untyped-def]
            """Safely traverse nested config (works with OmegaConf and dicts)."""
            for key in keys:
                if obj is None:
                    return default
                obj = obj.get(key, None) if hasattr(obj, "get") else None
            return obj if obj is not None else default

        return {
            "dim_model": _get(cfg, "dim_model"),
            "n_heads": _get(cfg, "n_heads"),
            "n_layers": _get(cfg, "n_layers"),
            "peak_encoder_type": _get(cfg, "peak_encoder", "type"),
            "positional_encoding": _get(cfg, "architecture", "positional_encoding", "type"),
            "rope_rotary_pct": _get(cfg, "architecture", "positional_encoding", "config", "rotary_pct"),
            "relative_bias": _get(cfg, "architecture", "relative_bias", "type"),
            "pa_num_freqs": _get(cfg, "architecture", "relative_bias", "config", "pw_num_freqs"),
            "pa_lambda_min": _get(cfg, "architecture", "relative_bias", "config", "lambda_min"),
            "pa_lambda_max": _get(cfg, "architecture", "relative_bias", "config", "lambda_max"),
            "ion_ladder_enabled": _get(cfg, "ion_ladder", "enabled", default=False),
            "meta_token_enabled": _get(cfg, "meta_token", "enabled", default=False),
            "masking_strategy": _get(cfg, "masking", "strategy"),
        }

    @staticmethod
    def _compute_idx_to_mz_rank(
        mz: Sequence[float],
        intensity: Sequence[float],
    ) -> Dict[int, int]:
        """Return ``peak_idx → rank`` in m/z-sorted order, skipping padding."""
        valid = [i for i in range(len(mz)) if intensity[i] > 0]
        sorted_by_mz = sorted(valid, key=lambda i: mz[i])
        return {idx: rank for rank, idx in enumerate(sorted_by_mz)}

    def _build_visible_peaks_report(
        self,
        result: AttributionResult,
        spec_info: Dict[str, Any],
        idx_to_rank: Dict[int, int],
        masked_base_rank: int,
    ) -> List[Dict[str, Any]]:
        """Build the per-peak visible_peaks list for the hero JSON.

        Each entry carries ``attribution_for_this_group`` (IG scalar for the
        masked group under review) plus optional confidence and rank fields.
        """
        masked_set = set(result.masked_group.peak_indices)
        annotations = spec_info.get("annotations", [])
        abs_attr = np.abs(result.attributions)
        conf_arr = spec_info.get("per_peak_confidence")
        n_peaks = len(spec_info["mz"])
        peaks: List[Dict[str, Any]] = []
        for i in range(n_peaks):
            if i in masked_set or spec_info["intensity"][i] <= 0:
                continue
            ann = str(annotations[i]) if i < len(annotations) and annotations[i] else None
            entry: dict[str, Any] = {
                "peak_idx": i,
                "mz": float(spec_info["mz"][i]),
                "intensity": float(spec_info["intensity"][i]),
                "annotation": ann,
                "attribution_for_this_group": float(abs_attr[i]),
            }
            if conf_arr is not None and i < len(conf_arr):
                entry["confidence"] = float(conf_arr[i])
            peaks.append(entry)
        peaks.sort(key=lambda p: p["attribution_for_this_group"], reverse=True)
        for vp in peaks:
            vp["rank_position"] = idx_to_rank.get(vp["peak_idx"], -1)
            vp["rank_distance"] = abs(vp["rank_position"] - masked_base_rank) if masked_base_rank >= 0 and vp["rank_position"] >= 0 else -1
        return peaks

    @staticmethod
    def _build_charge_state_variants_report(
        result: AttributionResult,
        visible_peaks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Structured list of the masked group's charge-state variants.

        Uses :func:`expected_mz_at_other_charges` so this stays in sync with
        :func:`detect_charge_variant_indices`; for each variant peak we pick
        the charge whose expected m/z lies closest.
        """
        gk = result.masked_group.group_key
        base_ion = gk.rstrip("+")
        current_charge = gk.count("+")
        variant_idx_set = result.charge_variant_indices or set()
        if current_charge == 0 or not variant_idx_set:
            return []

        expected_at_charge = expected_mz_at_other_charges(
            result.masked_group.base_mz,
            current_charge,
        )
        vp_by_idx = {vp["peak_idx"]: vp for vp in visible_peaks}
        variants: List[Dict[str, Any]] = []
        for pi in sorted(variant_idx_set):
            vp = vp_by_idx.get(pi)
            if vp is None:
                continue
            best_z, best_exp, best_delta = None, None, float("inf")
            for z, exp_mz in expected_at_charge.items():
                d = abs(vp["mz"] - exp_mz)
                if d < best_delta:
                    best_z, best_exp, best_delta = z, exp_mz, d
            if best_z is None:
                continue
            variants.append(
                {
                    "annotation": vp["annotation"] or f"{base_ion}{'+' * best_z} (inferred)",
                    "charge_state": best_z,
                    "mz": vp["mz"],
                    "expected_mz": round(best_exp, 4),  # type: ignore[arg-type]
                    "attribution_for_this_group": vp["attribution_for_this_group"],
                    "rank_distance": vp.get("rank_distance", -1),
                }
            )
        return variants

    def _build_ranked_groups_report(
        self,
        result: AttributionResult,
        spec_info: Dict[str, Any],
        idx_to_rank: Dict[int, int],
        masked_base_rank: int,
    ) -> List[Dict[str, Any]]:
        """Hero-JSON ``ranked_groups`` list.

        For same-series ladder/near- ladder peaks with parseable annotations, decompose the gap into the actual skipped residues from the peptide
        sequence (authoritative). Falls back to closest-AA mass matching when unparseable.
        """
        topk = result.topk
        if topk is None or not topk.ranked_peaks:
            return []
        sequence_residues = self._parse_residues(spec_info.get("sequence", "") or "")
        masked_series_info = _parse_series_info(result.masked_group.group_key)
        novel_full_matches = spec_info.get("novel_chem_full_matches", {}) or {}
        masked_mz = result.masked_group.base_mz
        out: List[Dict[str, Any]] = []
        for rp in topk.ranked_peaks:
            rp_rank_pos = idx_to_rank.get(rp.peak_idx, -1)
            entry: Dict[str, Any] = {
                "rank": rp.rank,
                "annotation": rp.annotation,
                "mz": rp.mz,
                "category": rp.category,
                "attribution_for_this_group": rp.attribution,
                "is_group": rp.is_group,
                "rank_distance": (abs(rp_rank_pos - masked_base_rank) if rp_rank_pos >= 0 and masked_base_rank >= 0 else -1),
            }
            if rp.category in ("ladder_neighbor", "near_ladder") and rp.mz > 0:
                self._annotate_ladder_gap(
                    entry,
                    rp,
                    masked_mz,
                    masked_series_info,
                    sequence_residues,
                )
            if rp.category in EXTENDED_CHEMISTRY_CATEGORIES:
                matches = novel_full_matches.get(rp.peak_idx)
                if matches:
                    entry["novel_chemistry_matches"] = matches
            out.append(entry)
        return out

    @staticmethod
    def _annotate_ladder_gap(
        entry: Dict[str, Any],
        rp: Any,
        masked_mz: float,
        masked_series_info: Optional[Tuple[str, int, int]],
        sequence_residues: List[str],
    ) -> None:
        """Fill ``gap_da`` / ``gap_residues`` / ``resolved_by`` on a ranked- peak dict.

        Prefers sequence-based decomposition (authoritative) when both endpoints parse; falls back to closest-AA mass matching.
        """
        gap = abs(rp.mz - masked_mz)
        entry["gap_da"] = round(gap, 4)
        peak_series_info = _parse_series_info(rp.annotation)
        if masked_series_info is not None and peak_series_info is not None and masked_series_info[0] == peak_series_info[0] and sequence_residues:
            gap_residues = _gap_residues_from_sequence(
                sequence_residues,
                masked_series_info[0],
                masked_series_info[1],
                peak_series_info[1],
            )
            if gap_residues:
                entry["gap_residues"] = "".join(gap_residues)
                entry["resolved_by"] = "sequence"
                if len(gap_residues) == 1:
                    entry["gap_residue"] = gap_residues[0]
                    entry["gap_residue_mass"] = round(
                        AMINO_ACID_MASSES.get(gap_residues[0], 0.0),
                        5,
                    )
                return
        # Fallback: closest-AA mass match (ambiguous — interpret as a hint).
        closest_aa = min(AMINO_ACID_MASSES.items(), key=lambda x: abs(x[1] - gap))
        if abs(closest_aa[1] - gap) < 0.05:
            entry["gap_residue"] = closest_aa[0]
            entry["gap_residue_mass"] = closest_aa[1]
        entry["resolved_by"] = "mass_only"

    def _save_hero_report(
        self,
        save_path: Path,
        result: AttributionResult,
        spec_info: Dict[str, Any],
    ) -> None:
        """Save a structured JSON report for a hero spectrum case study."""
        pred = result.prediction
        topk = result.topk
        annotations = spec_info.get("annotations", [])
        masked_set = set(result.masked_group.peak_indices)
        n_peaks = len(spec_info["mz"])
        n_annotated = sum(1 for i, a in enumerate(annotations) if a is not None and i not in masked_set and spec_info["intensity"][i] > 0)

        idx_to_rank = self._compute_idx_to_mz_rank(
            spec_info["mz"],
            spec_info["intensity"],
        )
        masked_base_rank = idx_to_rank.get(result.masked_group.base_idx, -1)  # type: ignore[arg-type]

        visible_peaks = self._build_visible_peaks_report(
            result,
            spec_info,
            idx_to_rank,
            masked_base_rank,
        )
        charge_state_variants = self._build_charge_state_variants_report(
            result,
            visible_peaks,
        )
        ranked_with_gaps = self._build_ranked_groups_report(
            result,
            spec_info,
            idx_to_rank,
            masked_base_rank,
        )

        quality = spec_info.get("spectrum_quality")
        quality_dict = (
            {
                "backbone_coverage": quality.get("backbone_coverage"),
                "n_fragment_groups": quality.get("n_fragment_groups"),
                "n_cleavage_sites": quality.get("n_cleavage_sites"),
            }
            if quality and isinstance(quality, dict)
            else {}
        )

        seq_context = self._compute_sequence_context(
            result.masked_group,
            spec_info.get("sequence", ""),
            float(spec_info.get("precursor_mass", 0)),
        )

        report: dict[str, Any] = {
            "spectrum": {
                "_spec_idx": spec_info.get("idx"),
                "sequence": spec_info.get("sequence", ""),
                "precursor_charge": spec_info.get("precursor_charge", 0),
                "precursor_mass": float(spec_info.get("precursor_mass", 0)),
                "n_peaks": n_peaks,
                "n_annotated_visible": n_annotated,
                "instrument": spec_info.get("instrument"),
                "fragmentation": spec_info.get("fragmentation"),
                "acquisition": spec_info.get("acquisition"),
                "collision_energy": spec_info.get("collision_energy"),
                "modifications": spec_info.get("modifications"),
                "quality": quality_dict or None,
            },
            "masked_group": {
                "group_key": result.masked_group.group_key,
                "ion_type": result.masked_group.ion_type,
                "base_mz": result.masked_group.base_mz,
                "base_idx": result.masked_group.base_idx,
                "n_peaks": len(result.masked_group.peak_indices),
                "peak_indices": result.masked_group.peak_indices,
                "peak_annotations": [
                    str(annotations[i]) if i < len(annotations) and annotations[i] else None for i in result.masked_group.peak_indices
                ],
                "peak_mz": [float(spec_info["mz"][i]) for i in result.masked_group.peak_indices],
                "peak_intensity": [float(spec_info["intensity"][i]) for i in result.masked_group.peak_indices],
                "rank_position": masked_base_rank,
            },
            "sequence_context": seq_context,
            "charge_state_variants": charge_state_variants,
            "prediction": {
                "bin_correct": pred.correct_bin if pred else None,
                "group_bin_accuracy": pred.group_bin_accuracy if pred else None,
                "base_predicted_mz": pred.predicted_mz_da if pred else None,
                "base_true_mz": pred.true_mz_da if pred else None,
                "base_error_da": pred.error_da if pred else None,
                "mean_log_prob": pred.confidence if pred else None,
                "per_peak": [
                    {
                        "annotation": pp.annotation,
                        "true_mz": pp.true_mz_da,
                        "predicted_mz": pp.predicted_mz_da,
                        "correct_bin": pp.correct_bin,
                        "log_prob": pp.log_prob,
                        "correct_bin_rank": pp.correct_bin_rank,
                        "predicted_group_bin": pp.predicted_group_bin,
                        "predicted_offset_bin": pp.predicted_offset_bin,
                        "true_group_bin": pp.true_group_bin,
                        "true_offset_bin": pp.true_offset_bin,
                        "group_bin_distance": pp.group_bin_distance,
                        "top5_candidates": [{"group_bin": g, "offset_bin": o, "mz": mz, "log_prob": lp} for g, o, mz, lp in pp.top_candidates],
                    }
                    for pp in (pred.peak_predictions if pred else [])
                ],
            },
            "attribution": {
                "convergence_delta": result.convergence_delta,
                "top1_category": topk.top1_category if topk else None,
                "top1_annotation": topk.top1_annotation if topk else None,
                "ladder_neighbors_present": topk.ladder_neighbors_present if topk else 0,
                "ladder_neighbors_in_top5": topk.ladder_neighbors_in_top5 if topk else 0,
                "ladder_best_rank": topk.ladder_neighbor_best_rank if topk else -1,
                "top5_concentration": topk.top5_concentration if topk else 0,
                "top10_concentration": topk.top10_concentration if topk else 0,
                "gini": topk.gini if topk else 0,
                "ranked_groups": ranked_with_gaps,
            },
            "visible_peaks": visible_peaks,
            "ig_config": {
                "n_steps": self.ig_n_steps,
                "baseline": "zero_intensity",
                "internal_batch_size": self.ig_internal_batch_size,
            },
            "model_config": self._extract_model_config(),
        }

        # Add novel chemistry matches for this hero spectrum
        if hasattr(self, "_novel_chemistry_per_spectrum"):
            spec_idx = spec_info["idx"]
            novel_data = self._novel_chemistry_per_spectrum.get(spec_idx)
            if novel_data:
                report["novel_chemistry"] = novel_data

        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)

    # ------------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------------

    def _save_results(self, results: Dict[str, Any]) -> None:
        """Save results to JSON."""

        def _serialize(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, AttributionResult):

                def _pred_dict(p: PredictionInfo | None) -> Any:
                    if p is None:
                        return None
                    return {
                        "predicted_mz": p.predicted_mz_da,
                        "true_mz": p.true_mz_da,
                        "error_da": p.error_da,
                        "error_ppm": p.error_ppm,
                        "confidence": p.confidence,
                    }

                return {
                    "spectrum_idx": obj.spectrum_idx,
                    "masked_group": obj.masked_group.group_key,
                    "masked_group_mz": obj.masked_group.base_mz,
                    "prediction": _pred_dict(obj.prediction),
                }
            if isinstance(obj, PredictionInfo):
                return {"predicted_mz": obj.predicted_mz_da, "true_mz": obj.true_mz_da, "error_da": obj.error_da, "error_ppm": obj.error_ppm}
            if isinstance(obj, FragmentGroup):
                return {"group_key": obj.group_key, "base_mz": obj.base_mz, "n_peaks": len(obj.peak_indices)}
            if isinstance(obj, TopKAnalysis):
                return {
                    "top1_category": obj.top1_category,
                    "top1_annotation": obj.top1_annotation,
                    "topk_fractions": obj.topk_fractions,
                    "ladder_neighbors_present": obj.ladder_neighbors_present,
                    "ladder_neighbors_in_top5": obj.ladder_neighbors_in_top5,
                    "ladder_neighbor_best_rank": obj.ladder_neighbor_best_rank,
                    "top5_concentration": obj.top5_concentration,
                    "gini": obj.gini,
                }
            return str(obj)

        out_path = self.output_dir / "ig_attribution_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=_serialize)

    def get_loggable_metrics(self, task_results: Dict[str, Any]) -> Dict[str, float]:
        """Extract core metrics for MLflow logging.

        Focused on 7 metrics chosen for cross-ablation comparison:
        - Attribution quality: topk_top1_ladder, topk_ladder_top5_hit_rate
        - PA structural priors: pa_alignment_score, pa_enrichment_ratio
        - Novel chemistry: novel_overall_hit_rate, novel_scramble_enrichment,
          novel_conf_auroc
        """
        metrics: dict[str, Any] = {}

        # PA dissection: has the model learned structural priors?
        pa = task_results.get("pa_dissection", {})
        if pa.get("success"):
            alignment = pa.get("chemical_alignment", {})
            metrics["pa_alignment_score"] = alignment.get("overall_score", 0.0)
            metrics["pa_enrichment_ratio"] = alignment.get("enrichment_ratio", 0.0)

        attr = task_results.get("prediction_attribution", {})
        if attr.get("success"):
            # Attribution quality: does the model attend to chemically meaningful peaks?
            topk = attr.get("topk_analysis", {})
            if topk:
                t1 = topk.get("top1_distribution", {})
                metrics["topk_top1_ladder"] = t1.get("ladder_neighbor", 0.0)
                # New first-class categories — track top-1 fraction so ablations
                # can chart complement use, leakage, and internal-fragment use
                # over training runs.
                metrics["topk_top1_complementary_pair"] = t1.get("complementary_pair", 0.0)
                metrics["topk_top1_charge_variant_leakage"] = t1.get("charge_variant_leakage", 0.0)
                metrics["topk_top1_internal_fragment"] = t1.get("internal_fragment", 0.0)
                metrics["topk_top1_unannotated"] = t1.get("unannotated", 0.0)
                ladder = topk.get("ladder_neighbor", {})
                metrics["topk_ladder_top5_hit_rate"] = ladder.get("top5_hit_rate", 0.0)

            # Novel chemistry: does the model discover chemistry beyond annotations?
            novel = attr.get("novel_chemistry", {})
            if novel:
                metrics["novel_overall_hit_rate"] = novel.get("overall_hit_rate", 0.0)
                # De-leakage-corrected rate excludes charge-state variants of
                # the masked group from the denominator — the more honest
                # number when ranking ablations for "real" chemistry discovery.
                metrics["novel_overall_hit_rate_deleakage"] = novel.get(
                    "overall_hit_rate_deleakage",
                    0.0,
                )
                nulls = novel.get("null_models", {})
                metrics["novel_scramble_enrichment"] = nulls.get("scramble", {}).get("enrichment", 0.0)
                conf_profile = novel.get("confidence_profile", {})
                if conf_profile and "auroc" in conf_profile:
                    metrics["novel_conf_auroc"] = conf_profile["auroc"]

        return metrics
