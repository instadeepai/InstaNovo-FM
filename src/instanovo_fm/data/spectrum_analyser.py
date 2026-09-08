#!/usr/bin/env python
"""
Spectrum Analyzer for Foundation Model Training Data.

Performs comprehensive spectrum-level analysis including:
1. Annotated vs unannotated analysis via theoretical matching
2. Masking strategy effectiveness
3. Data quality metrics for MLM training
4. Batch-level MLM visualization

This module combines spectrum analysis with MLM visualization capabilities.
"""

from __future__ import annotations

import json
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.figure import Figure
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from instanovo.__init__ import console
from instanovo.common import DataProcessor
from instanovo_fm.data import FoundationalDataProcessor
from instanovo_fm.data.binning_analyser import BinningAnalyser
from instanovo_fm.data.theoretical_analyser import TheoreticalAnalyser
from instanovo_fm.data.search_data_manager import create_search_data_manager
from instanovo_fm.trainer.binning import (
    FixedDaBinning,
    FixedPpmBinning,
    AdaptiveBinning,
)
from instanovo_fm.utils.ion_visualization import (
    categorize_ion,
    CATEGORY_COLORS as category_colors,
    TEXT_COLORS as text_colors,
    format_annotation_display,
)
from instanovo_fm.utils.theoretical_spectra import (
    generate_theoretical_spectrum,
    generate_theoretical_spectrum_rustyms,
    match_theoretical_to_experimental,
    match_with_conditional_features,
)
from instanovo_fm.utils.naming import sanitize_filename
from instanovo.utils.residues import ResidueSet
from instanovo.utils.data_handler import SpectrumDataFrame
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

# Maximum number of worker processes to prevent OOM in containers that
# report the host CPU count (e.g. 256 threads) rather than the cgroup limit.
_MAX_WORKERS = 32

# Chunk size for batched multiprocessing: load this many spectra from the SDF
# in the main process, dispatch them to workers, aggregate results, then discard.
_CHUNK_SIZE = 5000

# ---------------------------------------------------------------------------
# Module-level worker state and functions for multiprocessing.Pool
# Using module-level globals avoids pickling the full SpectrumAnalyser per
# spectrum — the initialiser pickles the analyser only once per worker.
# ---------------------------------------------------------------------------
_worker_analyser: Optional["SpectrumAnalyser"] = None
_worker_processor: Optional[FoundationalDataProcessor] = None


def _worker_init(
    analyser: "SpectrumAnalyser",
    proc_cfg: dict,
    theo_cache: dict,
) -> None:
    """Initialise shared state once per worker process."""
    global _worker_analyser, _worker_processor
    import logging

    logging.disable(logging.INFO)
    _worker_analyser = analyser
    _worker_analyser.theoretical_cache = dict(theo_cache)
    # Re-point the nested TheoreticalAnalyser at the worker-local cache so
    # the two stay in sync after pickling (they were the same dict in the
    # main process but diverge after we replace theoretical_cache above).
    if getattr(_worker_analyser, "theoretical_analyser", None) is not None:
        _worker_analyser.theoretical_analyser.theoretical_cache = (
            _worker_analyser.theoretical_cache
        )
    _worker_processor = FoundationalDataProcessor(**proc_cfg)


def _detensorize(obj: Any) -> Any:
    """Recursively convert torch tensors and numpy scalars to plain Python types.

    Multiprocessing uses pickle to send results from workers back to the main
    process.  PyTorch intercepts tensor pickling and uses /dev/shm mmap, which
    exhausts the (typically 64 MB) shared memory in Docker containers.
    Converting tensors to plain Python types before returning avoids this.
    """
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        if t.ndim == 0:
            return t.item()
        return t.numpy().tolist()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _detensorize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        converted = [_detensorize(v) for v in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    return obj


def _worker_analyze(args: Tuple[int, dict]) -> Tuple[int, dict]:
    """Analyse a single spectrum using pre-initialised worker state.

    Args:
        args: Tuple of (global_index, spectrum_data_dict).

    Returns:
        Tuple of (global_index, result_dict) with all tensors converted to
        plain Python types to avoid PyTorch shared-memory pickling.
    """
    global _worker_analyser, _worker_processor
    idx, spectrum_data = args
    try:
        result = _worker_analyser.analyze_single_spectrum(  # type: ignore[union-attr]
            spectrum_data, _worker_processor, include_visualization_data=False  # type: ignore[arg-type]
        )
        return idx, _detensorize(result)
    except Exception as e:
        return idx, {
            "error": str(e),
            "spectrum_stats": {},
            "theoretical_analysis": {},
            "masking_analysis": {},
            "metadata": {"search_project": "unknown", "spectrum_key": "unknown"},
        }


class SpectrumAnalyser:
    """
    Per-spectrum analysis engine and result aggregator.

    Called by DataAnalyzer with an explicit set of active tasks.
    Handles:
    - Per-spectrum loop (theoretical, custom_ions, intensity, masking)
    - Aggregation and visualization for each sub-analyser
    - Binning analysis (batch-level, after aggregation)
    """

    def __init__(
        self,
        config: DictConfig,
        output_dir: Optional[str] = None,
        active_tasks: Optional[Set[str]] = None,
    ):
        """
        Initialize the spectrum analyzer.

        Args:
            config: Hydra configuration
            output_dir: Output directory (defaults to analysis_output/spectra_analysis)
            active_tasks: Set of task names to run. If None, derives from config
                          enable_* booleans for backward compatibility.
        """
        self.config = config

        # Set output directory
        if output_dir is None:
            self.output_dir = Path(os.getcwd()) / "analysis_output" / "spectra_analysis"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")

        # Initialize results storage
        self.analysis_results = {
            "summary": {},
            "theoretical_matching": {},
            "annotation_analysis": {},
            "masking_analysis": {},
        }

        # Model configuration
        self.max_mz = config.model.get("max_mz", 2500.0)
        self.min_mz = config.model.get("min_mz", 50.0)
        self.min_intensity = config.model.get("min_intensity", 0.01)
        self.n_peaks = config.model.get("n_peaks", 200)

        # ------------------------------------------------------------------
        # Read analysis config — supports both new task_configs and legacy flat
        # ------------------------------------------------------------------
        analysis_config = config.get("analysis", {})

        # Resolve active_tasks
        if active_tasks is not None:
            self.active_tasks: Set[str] = set(active_tasks)
        else:
            # Backward compat: derive from enable_* booleans
            self.active_tasks = self._derive_active_tasks_from_legacy(analysis_config)

        # Helper to read a task-specific key with fallback to flat config
        task_configs_raw = analysis_config.get("task_configs", {})
        if hasattr(task_configs_raw, "items") and hasattr(task_configs_raw, "_metadata"):
            task_configs = OmegaConf.to_container(task_configs_raw, resolve=True)
        elif hasattr(task_configs_raw, "items"):
            task_configs = dict(task_configs_raw)
        else:
            task_configs = {}
        self._task_configs = task_configs

        def _tc(task: str, key: str, default: Any = None) -> Any:
            """Read from task_configs[task][key], fallback to analysis_config[key]."""
            tc = self._task_configs.get(task, {})
            if tc and key in tc:
                return tc[key]
            return analysis_config.get(key, default)

        # Convenience booleans (derived from active_tasks)
        self.enable_theoretical = "theoretical" in self.active_tasks
        self.enable_custom_ions = "custom_ions" in self.active_tasks
        self.enable_intensity_analysis = "intensity" in self.active_tasks
        self.enable_masking_analysis = "masking" in self.active_tasks
        self.enable_mass_error_analysis = (
            "binning" in self.active_tasks
            or "theoretical" in self.active_tasks
        )

        # Global settings
        self.max_samples = analysis_config.get("max_samples", 100000)
        self.max_visualizations = analysis_config.get("max_visualizations", 30)
        n_workers_cfg = analysis_config.get("n_workers", None)
        if n_workers_cfg is not None:
            self.n_workers = n_workers_cfg
        else:
            self.n_workers = min(max(1, cpu_count() - 1), _MAX_WORKERS)

        # Theoretical config
        self.ppm_tol = _tc("theoretical", "ppm_tol", 10.0)
        self.keep_modifications = _tc("theoretical", "keep_modifications", True)
        self.min_backbone_coverage = _tc("theoretical", "min_backbone_coverage", 0.15)
        self.min_fragment_groups = _tc("theoretical", "min_fragment_groups", 3)
        self.theoretical_engine = _tc("theoretical", "theoretical_engine", "pyopenms")
        self.use_conditional_annotation = _tc("theoretical", "use_conditional_annotation", True)
        self.add_losses = _tc("theoretical", "add_losses", True)
        self.loss_types = tuple(_tc("theoretical", "loss_types", ["H2O", "NH3"]))
        self.add_isotopes = _tc("theoretical", "add_isotopes", True)
        self.isotope_intensity_threshold = _tc("theoretical", "isotope_intensity_threshold", 0.02)
        self.max_isotope = _tc("theoretical", "max_isotope", 4)
        self.enable_stratified_analysis = _tc("theoretical", "enable_stratified_analysis", True)

        # Custom ion config
        custom_ions_cfg = _tc("custom_ions", "custom_ions", None)
        self.custom_ions = dict(custom_ions_cfg) if custom_ions_cfg is not None else None

        # m/z range boundaries (shared between intensity and binning)
        self.mz_range_boundaries = _tc("intensity", "mz_range_boundaries", {
            "immonium_internal": (0, 200),
            "core_fragment": (200, 800),
            "extended_fragment": (800, 1500),
            "high_mass_fragment": (1500, float('inf')),
        })
        self.mz_range_order = ["immonium_internal", "core_fragment", "extended_fragment", "high_mass_fragment"]
        self.mz_range_thresholds = _tc("theoretical", "mz_range_thresholds", {
            "low": 200, "mid": 800, "high": 1500
        })

        # Binning config
        self.enable_bin_jump_analysis = "binning" in self.active_tasks
        self.min_ion_observations = _tc("binning", "min_ion_observations", 2)
        self.bin_jump_threshold = _tc("binning", "bin_jump_threshold", 0.10)
        self.enable_hybrid_parameter_derivation = _tc("binning", "enable_hybrid_parameter_derivation", True)
        self.target_jump_rate = _tc("binning", "target_jump_rate", 0.05)
        self.mz_window_size = _tc("binning", "mz_window_size", 100.0)
        binning_strategies = _tc("binning", "strategies", None)
        self.binning_strategies = binning_strategies if binning_strategies is not None else analysis_config.get("binning_strategies", None)
        self.enable_collision_analysis = _tc("binning", "enable_intra_collision_analysis", True)
        self.collision_weight = _tc("binning", "intra_collision_weight", 0.4)
        self.boundary_threshold = analysis_config.get("boundary_threshold", 0.4)
        soft_label_thresholds = analysis_config.get("soft_label_recommendation_threshold", {})
        self.soft_label_boundary_threshold = soft_label_thresholds.get("boundary_fraction", 0.3)
        self.soft_label_adjacent_threshold = soft_label_thresholds.get("adjacent_jump_fraction", 0.7)
        self.enable_soft_label_evaluation = analysis_config.get("enable_soft_label_evaluation", True)

        # Setup residue set (needed for processor)
        self.residue_set = ResidueSet(
            residue_masses=config.residues.get("residues"),
            residue_remapping=config.dataset.get("residue_remapping", None),
        )

        # Setup search data manager (needed for metadata extraction)
        self.search_data_manager = create_search_data_manager({
            "use_search_data": config.dataset.get("use_search_data", False),
            "search_data_path": config.dataset.get("search_data_path", None),
            "search_data_filepath_column": config.dataset.get("search_data_filepath_column", "file path"),
            "search_data_spectrum_key": config.dataset.get("search_data_spectrum_key", "filepath"),
        })

        # Check theoretical spectra availability
        self.theoretical_available = self.enable_theoretical
        if self.enable_theoretical:
            try:
                from instanovo_fm.utils.theoretical_spectra import (
                    generate_theoretical_spectrum,
                )
                # Check if rustyms is available for optimized batch processing
                try:
                    import rustyms
                    self.rustyms_available = True
                except ImportError:
                    self.rustyms_available = False
            except ImportError:
                self.theoretical_available = False
                self.rustyms_available = False
                logger.warning(
                    "Theoretical spectra module not available. Skipping theoretical analysis."
                )
        else:
            logger.info("Theoretical analysis disabled by configuration.")
            self.rustyms_available = False

        # Theoretical spectrum cache for optimized batch processing.
        # Key layout (seq, charge, ion_types) is defined in TheoreticalAnalyser.
        self.theoretical_cache: Dict[
            Tuple[str, int, Tuple[str, ...]], Tuple[np.ndarray, List[str]]
        ] = {}

        # Per-spectrum TheoreticalAnalyser — created once and reused across
        # every call to analyze_single_spectrum. It previously was re-created
        # per spectrum (~0.26 ms/init × 200k spectra ≈ 50s of pure overhead
        # in single-process, amortised per worker in parallel mode), and each
        # re-creation wiped local state. The cache dict is shared by reference
        # so that populated entries remain visible to this outer analyser.
        self.theoretical_analyser = TheoreticalAnalyser(
            self.config, self.output_dir / "theoretical_analysis"
        )
        self.theoretical_analyser.theoretical_cache = self.theoretical_cache

        # Initialize intensity analyser
        if self.enable_intensity_analysis:
            from instanovo_fm.data.intensity_analyser import IntensityAnalyser
            self.intensity_analyser = IntensityAnalyser(
                self.config,
                self.output_dir / "intensity_analysis"
            )
        else:
            self.intensity_analyser = None

        # Initialize masking analyser (replaces standalone MaskingGapAnalyser)
        if self.enable_masking_analysis:
            from instanovo_fm.data.masking_analyser import MaskingAnalyser
            self.masking_analyser = MaskingAnalyser(
                self.config, self.output_dir / "masking_analysis"
            )
        else:
            self.masking_analyser = None

    @staticmethod
    def _derive_active_tasks_from_legacy(analysis_config) -> Set[str]:
        """Derive active_tasks set from legacy enable_* config booleans."""
        tasks: Set[str] = set()
        # Map legacy enable_* booleans to task names
        if analysis_config.get("enable_theoretical", True):
            tasks.add("theoretical")
        if analysis_config.get("enable_custom_ions", False):
            tasks.add("custom_ions")
        if analysis_config.get("enable_intensity_analysis", False):
            tasks.add("intensity")
        if analysis_config.get("enable_masking_analysis", True):
            tasks.add("masking")
        if analysis_config.get("enable_bin_jump_analysis", True) or analysis_config.get("enable_mass_error_analysis", True):
            tasks.add("binning")
        return tasks

    def setup_data_processor(self) -> FoundationalDataProcessor:
        """Set up the data processor for consistent preprocessing."""
        masking_config = self.config.model.get('masking', {})

        proc_cfg = {
            'n_peaks': self.config.model.get('n_peaks', 200),
            'min_mz': self.config.model.get('min_mz', 50.0),
            'max_mz': self.config.model.get('max_mz', 2500.0),
            'min_intensity': self.config.model.get('min_intensity', 0.01),
            'mask_portion': masking_config.get('mask_portion', 0.3),
            'remove_precursor_tol': self.config.model.get('remove_precursor_tol', 0.0),
            'use_spectrum_utils': self.config.model.get('use_spectrum_utils', False),
            'normalize_mz': self.config.model.get('normalize_mz', True),
            'peak_ordering': masking_config.get('ordering_strategy', 'sorted'),
            # Required for metadata extraction
            'residue_set': self.residue_set,
            'annotated': True,
            'return_str': True,
            'metadata_columns': self.config.dataset.get('metadata_columns', []),
            'search_data_manager': self.search_data_manager,
        }

        strategy_mapping = {
            'thompson_span_mask': 'thompson_span',
            'thompson': 'thompson',
            'uniform': 'uniform',
            'ladder': 'ladder',
            'fast_ladder': 'fast_ladder',
            'signal_aware_fragment': 'signal_aware_fragment',
        }
        proc_cfg['masking_strategy'] = strategy_mapping.get(
            masking_config.get('strategy', 'thompson_span'), 'thompson_span'
        )

        proc_cfg.update({
            'thompson_alpha': masking_config.get('alpha', 0.5),
            'thompson_beta': masking_config.get('beta', 0.5),
            'thompson_kappa': masking_config.get('kappa', 4.0),
            'thompson_gamma': masking_config.get('gamma', 0.7),
            'span_min': masking_config.get('span_min', 4),
            'span_max': masking_config.get('span_max', 7),
            'span_bidirectional': masking_config.get('bidirectional', True),
            'include_isotopes': masking_config.get('include_isotopes', False),
            'isotope_ppm': masking_config.get('isotope_ppm', 10.0),
            'isotope_da_floor': masking_config.get('isotope_da_floor', 0.015),
            'isotope_max_charge': masking_config.get('isotope_max_charge', 4),
            'isotope_max_order': masking_config.get('isotope_max_order', 3),
            'max_total_mask_ratio': masking_config.get('max_total_mask_ratio', 0.35),
            # Signal-aware masking parameters
            'signal_min_backbone_coverage': masking_config.get('signal_min_backbone_coverage', 0.15),
            'signal_min_fragment_groups': masking_config.get('signal_min_fragment_groups', 3),
            'signal_ppm': masking_config.get('signal_ppm', 20.0),
            'signal_cid_da_tol': masking_config.get('signal_cid_da_tol', 0.2),
            'signal_ion_types': tuple(masking_config.get('signal_ion_types', ['b', 'y'])),
            'signal_num_workers': masking_config.get('signal_num_workers', 4),
        })

        logger.info(f"Data processor config: mask_portion={proc_cfg['mask_portion']}, "
                   f"include_isotopes={proc_cfg['include_isotopes']}, "
                   f"strategy={proc_cfg['masking_strategy']}, "
                   f"metadata_columns={len(proc_cfg['metadata_columns'])}, "
                   f"search_data={'enabled' if self.search_data_manager and self.search_data_manager.is_loaded else 'disabled'}")

        return FoundationalDataProcessor(**proc_cfg)


    def analyze_single_spectrum(
        self,
        spectrum_data: Dict[str, Any],
        processor: FoundationalDataProcessor,
        include_visualization_data: bool = False,
    ) -> Dict[str, Any]:
        """Analyze a single spectrum for annotated vs unannotated characteristics.

        Args:
            spectrum_data: Raw spectrum data
            processor: Data processor instance
            include_visualization_data: If True, store arrays for visualization
        """
        try:
            # 1. Preprocess spectrum
            processed = processor.process_row(spectrum_data)
            if processed is None:
                return {"error": "Failed to process spectrum"}

            batch_result = processor._collate_batch([processed], apply_masking=False)

            spectra = batch_result["spectra"][0]
            spectra_mask = batch_result["spectra_mask"][0]

            valid_peaks = ~spectra_mask
            n_valid_peaks = valid_peaks.sum().item()

            if n_valid_peaks == 0:
                return {"error": "No valid peaks in processed spectrum"}

            mz_values = spectra[:, 0].cpu().numpy() * self.max_mz
            intensity_values = spectra[:, 1].cpu().numpy()

            valid_mz = mz_values[valid_peaks]
            valid_intensity = intensity_values[valid_peaks]

            # Only store visualization data if requested
            spectrum_visualization_data = None
            if include_visualization_data:
                spectrum_visualization_data = {
                    "mz_values": mz_values,
                    "intensity_values": intensity_values,
                    "valid_peaks": valid_peaks.cpu().numpy(),
                    "spectra_mask": spectra_mask.cpu().numpy(),
                    "peak_ordering": processor.peak_ordering,
                }

            # 2. Calculate spectrum statistics
            spectrum_stats = {
                "n_total_peaks": len(spectra),
                "n_valid_peaks": n_valid_peaks,
                "mz_range": (valid_mz.min(), valid_mz.max()),
                "intensity_range": (valid_intensity.min(), valid_intensity.max()),
                "total_intensity": valid_intensity.sum(),
                "mean_intensity": valid_intensity.mean(),
                "median_intensity": np.median(valid_intensity),
                "std_intensity": valid_intensity.std(),
            }

            # 3. Delegate theoretical and masking analysis to the reusable
            #    TheoreticalAnalyser built once in __init__. Its cache is the
            #    same dict as self.theoretical_cache by reference, so updates
            #    from match-and-collect are visible immediately — no merge
            #    step required.
            theo_result = self.theoretical_analyser.analyze_spectrum(
                spectrum_data=spectrum_data,
                valid_mz=valid_mz,
                valid_intensity=valid_intensity,
                mlm_mask=None,
                processor=processor,
                batch_result=batch_result,
                theoretical_cache=None,
            )

            theoretical_analysis = theo_result.get("theoretical_analysis", {})
            masking_analysis = theo_result.get("masking_analysis", {})
            mass_error_data = theo_result.get("mass_error_data", [])
            unmatched_theo_data = theo_result.get("unmatched_theo_data", [])

            # 4. Extract metadata
            sequence = TheoreticalAnalyser._extract_field(spectrum_data, ["sequence", "modified_peptide", "peptide"])
            clean_sequence = theoretical_analysis.get("clean_sequence")

            metadata = {
                "search_project": batch_result.get("search_project", [None])[0] if "search_project" in batch_result else "unknown",
                "spectrum_key": (
                    batch_result.get("usi", [None])[0] if "usi" in batch_result
                    else batch_result.get("filepath", [None])[0] if "filepath" in batch_result
                    else "unknown"
                ),
                "sequence": sequence,
                "frag_type": batch_result.get("frag_type", [None])[0] or "unknown" if "frag_type" in batch_result else "unknown",
                "search_acquisition": batch_result.get("search_acquisition", [None])[0] or "unknown" if "search_acquisition" in batch_result else "unknown",
                "search_detector": batch_result.get("search_detector", [None])[0] or "unknown" if "search_detector" in batch_result else "unknown",
                "search_instrument": batch_result.get("search_instrument", [None])[0] or "unknown" if "search_instrument" in batch_result else "unknown",
                "clean_sequence": clean_sequence,
                "precursor_mz": float(batch_result["precursor_mz"][0]) if "precursor_mz" in batch_result else None,
                "precursor_mass": float(batch_result["precursor_mass"][0]) if "precursor_mass" in batch_result else None,
            }

            # 5. Intensity analysis (if enabled)
            intensity_result = {}
            if self.enable_intensity_analysis and self.intensity_analyser:
                # Extract annotations if available from theoretical analysis
                annotations = None
                if theoretical_analysis:
                    annotations = theoretical_analysis.get("matched_annotations", None)

                intensity_result = self.intensity_analyser.analyze_spectrum(
                    valid_mz=valid_mz,
                    valid_intensity=valid_intensity,
                    metadata={
                        "frag_type": batch_result.get("frag_type", [None])[0]
                                    if "frag_type" in batch_result else "unknown",
                        "search_instrument": batch_result.get("search_instrument", [None])[0]
                                            if "search_instrument" in batch_result else "unknown",
                        "precursor_charge": batch_result.get("precursor_charge", [None])[0]
                                          if "precursor_charge" in batch_result else None,
                    },
                    annotations=annotations,
                )

            # 6. Custom ion detection (if enabled) — before masking for custom ion peak mask
            custom_ion_result = {}
            custom_ion_peak_mask = None
            if self.enable_custom_ions:
                from instanovo_fm.utils.theoretical_spectra import detect_custom_ions
                custom_ion_result = detect_custom_ions(
                    exp_mz=valid_mz,
                    exp_intensity=valid_intensity,
                    custom_ions=self.custom_ions,
                    ppm_tol=self.ppm_tol,
                    return_details=True,
                )
                # Build peak mask: True for peaks matched by custom ions but NOT by fragments
                annotated_mask = np.array(theoretical_analysis.get("annotated_mask", []))
                custom_ion_peak_mask = np.zeros(len(valid_mz), dtype=bool)
                for group_data in custom_ion_result.values():
                    if not group_data.get("found"):
                        continue
                    for mz_val in group_data.get("matched_mz", []):
                        idx_pos = np.searchsorted(valid_mz, mz_val)
                        for ci in (idx_pos - 1, idx_pos):
                            if 0 <= ci < len(valid_mz):
                                if abs(valid_mz[ci] - mz_val) / max(mz_val, 1e-12) * 1e6 < self.ppm_tol * 2:
                                    if len(annotated_mask) > ci and not annotated_mask[ci]:
                                        custom_ion_peak_mask[ci] = True
                                    break

            # 7. Masking strategy comparison analysis (if enabled)
            masking_analysis_result = {}
            if self.enable_masking_analysis and self.masking_analyser:
                sequence = TheoreticalAnalyser._extract_field(spectrum_data, ["sequence", "modified_peptide", "peptide"])
                # Ensure charges are a tensor (metadata_columns may overwrite
                # the tensor from _collate_batch with a plain Python list)
                raw_charges = batch_result.get("precursor_charge", torch.zeros(1))
                if not isinstance(raw_charges, torch.Tensor):
                    raw_charges = torch.tensor(raw_charges)
                masking_analysis_result = self.masking_analyser.analyze_spectrum(
                    spectra=batch_result["spectra"][0:1],
                    spectra_mask=batch_result["spectra_mask"][0:1],
                    precursor_charges=raw_charges[0:1],
                    valid_mz=valid_mz,
                    valid_intensity=valid_intensity,
                    metadata={
                        "frag_type": batch_result.get("frag_type", [None])[0]
                                    if "frag_type" in batch_result else "unknown",
                        "clean_sequence": theoretical_analysis.get("clean_sequence")
                                          if theoretical_analysis else None,
                        "precursor_charge": str(int(raw_charges[0].item())) if raw_charges.numel() > 0 else "unknown",
                        "search_instrument": batch_result.get("search_instrument", [None])[0]
                                            if "search_instrument" in batch_result else "unknown",
                    },
                    theoretical_analysis=theoretical_analysis,
                    matched_annotations=theoretical_analysis.get("theo_annotations"),
                    feature_types=theoretical_analysis.get("feature_types"),
                    parent_annotations=theoretical_analysis.get("parent_annotations"),
                    peptides=[sequence] if sequence else None,
                    custom_ion_peak_mask=custom_ion_peak_mask,
                )

            # 8. Return combined results
            return {
                "spectrum_stats": spectrum_stats,
                "theoretical_analysis": theoretical_analysis,
                "masking_analysis": masking_analysis,
                "visualization_data": spectrum_visualization_data,
                "metadata": metadata,
                "mass_error_data": mass_error_data if self.enable_mass_error_analysis else [],
                "unmatched_theo_data": unmatched_theo_data if self.enable_mass_error_analysis else [],
                "valid_mz": valid_mz.tolist() if self.enable_mass_error_analysis else [],
                "intensity_result": intensity_result,
                "masking_analysis_result": masking_analysis_result,
                "custom_ion_result": custom_ion_result,
                "error": None,
            }

        except Exception as e:
            # Try to extract basic metadata even on error
            try:
                # Attempt minimal processing just to get metadata
                processed = processor.process_row(spectrum_data)
                if processed:
                    batch_result = processor._collate_batch([processed], apply_masking=False)
                    metadata = {
                        "search_project": batch_result.get("search_project", [None])[0] if "search_project" in batch_result else "unknown",
                        "spectrum_key": (
                    batch_result.get("usi", [None])[0] if "usi" in batch_result
                    else batch_result.get("filepath", [None])[0] if "filepath" in batch_result
                    else "unknown"
                ),
                    }
                else:
                    metadata = {"search_project": "unknown", "spectrum_key": "unknown"}
            except:
                metadata = {"search_project": "unknown", "spectrum_key": "unknown"}

            return {
                "error": str(e),
                "spectrum_stats": {},
                "theoretical_analysis": {},
                "masking_analysis": {},
                "metadata": metadata,
            }

    def run_analysis(self, sdf: SpectrumDataFrame) -> Dict[str, Any]:
        """
        Run comprehensive spectrum analysis.

        Args:
            sdf: SpectrumDataFrame with spectra to analyze

        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting spectrum analysis...")

        # Set up data processor configuration (will be recreated in each worker)
        processor = self.setup_data_processor()

        # Build a plain-dict version of the processor config so that workers can
        # construct their own FoundationalDataProcessor without pickling the full
        # SpectrumAnalyser per spectrum.
        proc_cfg = {
            'n_peaks': self.n_peaks,
            'min_mz': self.min_mz,
            'max_mz': self.max_mz,
            'min_intensity': self.min_intensity,
            'mask_portion': processor.mask_portion,
            'remove_precursor_tol': processor.remove_precursor_tol,
            'use_spectrum_utils': processor.use_spectrum_utils,
            'normalize_mz': processor.normalize_mz,
            'peak_ordering': processor.peak_ordering,
            'masking_strategy': processor.masking_strategy,
            'thompson_alpha': processor.thompson_alpha,
            'thompson_beta': processor.thompson_beta,
            'thompson_kappa': processor.thompson_kappa,
            'thompson_gamma': processor.thompson_gamma,
            'span_min': processor.span_min,
            'span_max': processor.span_max,
            'span_bidirectional': processor.span_bidirectional,
            'include_isotopes': processor.include_isotopes,
            'isotope_ppm': processor.isotope_ppm,
            'isotope_da_floor': processor.isotope_da_floor,
            'isotope_max_charge': processor.isotope_max_charge,
            'isotope_max_order': processor.isotope_max_order,
            'max_total_mask_ratio': processor.max_total_mask_ratio,
            'residue_set': self.residue_set,
            'annotated': True,
            'return_str': True,
            'metadata_columns': self.config.dataset.get('metadata_columns', []),
            'search_data_manager': self.search_data_manager,
        }

        # Test with first spectrum
        logger.info("Testing data processing with first spectrum...")
        try:
            test_spectrum = sdf[0]
            test_result = self.analyze_single_spectrum(test_spectrum, processor, include_visualization_data=False)
            if test_result.get("error"):
                logger.error(f"Test spectrum failed: {test_result['error']}")
            else:
                logger.info("Test spectrum processed successfully")
        except Exception as e:
            logger.error(f"Test spectrum processing failed: {e}")

        # Determine number of samples
        total_samples = len(sdf)
        n_samples = total_samples if self.max_samples is None else min(self.max_samples, total_samples)

        logger.info(f"Analyzing {n_samples:,d} out of {total_samples:,d} spectra")
        logger.info(f"Using {self.n_workers} worker processes")

        # ------------------------------------------------------------------
        # Streaming accumulators — collect only lightweight data per spectrum
        # so that we never hold 2.95M full result dicts in memory.
        # ------------------------------------------------------------------
        acc_spectrum_stats: List[Dict] = []            # scalar dicts
        acc_theoretical: List[Dict] = []               # scalar dicts
        acc_masking: List[Dict] = []                   # scalar dicts
        acc_results_for_stratified: List[Dict] = []    # metadata + theo + masking
        acc_mass_error_data: List[Dict] = []           # per-peak error records
        acc_unmatched_theo_data: List[Dict] = []
        acc_spectrum_mz: List[np.ndarray] = []
        acc_frag_type: List[str] = []
        acc_theoretical_enriched: List[Dict] = []      # for Phase 3.5
        acc_intensity: List[Dict] = []                 # for Phase 4
        acc_masking_result: List[Dict] = []            # for Phase 5
        acc_custom_ion: List[Dict] = []                # for Phase 6
        acc_csv_rows: List[Dict] = []                  # for _save_results CSV
        # For individual visualizations we record indices of valid spectra and
        # re-analyze a small number (~max_visualizations) later.
        acc_valid_indices: List[int] = []

        errors = 0

        def _accumulate_result(idx: int, result: Dict[str, Any]) -> None:
            """Accumulate lightweight data from a single spectrum result."""
            nonlocal errors
            if result.get("error"):
                errors += 1
                if errors <= 5:
                    logger.warning(f"Error in spectrum {idx}: {result['error']}")
                return

            # Summary statistics
            acc_spectrum_stats.append(result["spectrum_stats"])
            acc_theoretical.append(result["theoretical_analysis"])
            acc_masking.append(result["masking_analysis"])
            acc_results_for_stratified.append({
                "metadata": result.get("metadata", {}),
                "theoretical_analysis": result["theoretical_analysis"],
                "masking_analysis": result["masking_analysis"],
                "spectrum_stats": result["spectrum_stats"],
            })

            # Phase 3.5: theoretical aggregation. Determine the spectrum's
            # index in acc_theoretical_enriched (== its index in per_spectrum
            # passed to TheoreticalAnalyser.aggregate_results) BEFORE
            # extending mass_error/unmatched accumulators, so we can stamp
            # those records with the same spectrum_idx for later
            # quality-gate filtering. Spectra without a sequence get
            # spectrum_idx = -1 and will fail any quality gate.
            theo = result.get("theoretical_analysis", {})
            spectrum_idx = -1
            if theo.get("sequence_available", False):
                spectrum_idx = len(acc_theoretical_enriched)
                enriched = dict(theo)
                enriched["_metadata"] = result.get("metadata", {})
                enriched["_spectrum_stats"] = result.get("spectrum_stats", {})
                if result.get("valid_mz"):
                    enriched["_valid_mz"] = result["valid_mz"]
                acc_theoretical_enriched.append(enriched)

            # Mass error / binning data — tag each record with spectrum_idx
            if result.get("mass_error_data"):
                for rec in result["mass_error_data"]:
                    rec["spectrum_idx"] = spectrum_idx
                    acc_mass_error_data.append(rec)
            if result.get("unmatched_theo_data"):
                for rec in result["unmatched_theo_data"]:
                    rec["spectrum_idx"] = spectrum_idx
                    acc_unmatched_theo_data.append(rec)
            if result.get("valid_mz"):
                acc_spectrum_mz.append(np.array(result["valid_mz"], dtype=np.float64))
                ft = result.get("metadata", {}).get("frag_type", "unknown")
                acc_frag_type.append(str(ft) if ft else "unknown")

            # Phase 4: intensity
            ir = result.get("intensity_result", {})
            if ir and "error" not in ir:
                acc_intensity.append(ir)

            # Phase 5: masking
            mr = result.get("masking_analysis_result", {})
            if mr:
                acc_masking_result.append(mr)

            # Phase 6: custom ions
            cr = result.get("custom_ion_result", {})
            if cr:
                acc_custom_ion.append(cr)

            # CSV row for _save_results
            if theo.get("sequence_available", False):
                metadata = result.get("metadata", {})
                acc_csv_rows.append({
                    "search_project": metadata.get("search_project", "unknown"),
                    "spectrum_key": metadata.get("spectrum_key", "unknown"),
                    "sequence": metadata.get("sequence", "unknown"),
                    "frag_type": metadata.get("frag_type", "unknown"),
                    "search_acquisition": metadata.get("search_acquisition", "unknown"),
                    "search_detector": metadata.get("search_detector", "unknown"),
                    "search_instrument": metadata.get("search_instrument", "unknown"),
                    "n_valid_peaks": result["spectrum_stats"]["n_valid_peaks"],
                    "precursor_charge": theo["precursor_charge"],
                    "n_theoretical": theo["n_theoretical"],
                    "n_matched": theo["n_matched"],
                    "match_rate": theo["match_rate"],
                    "frac_intensity": theo["frac_intensity"],
                    "median_ppm": theo["median_ppm"],
                    "mean_ppm": theo["mean_ppm"],
                    "annotated_fraction": theo["annotated_fraction"],
                    "unannotated_fraction": theo["unannotated_fraction"],
                    "annotated_intensity_fraction": theo["annotated_intensity_fraction"],
                    "unannotated_intensity_fraction": theo["unannotated_intensity_fraction"],
                })

            # Track index for visualization sampling
            acc_valid_indices.append(idx)

        # ------------------------------------------------------------------
        # Run analysis — chunked dispatch to avoid pre-loading all spectra
        # ------------------------------------------------------------------
        chunk_size = _CHUNK_SIZE

        if self.n_workers > 1:
            logger.info("Launching multiprocessing pool (initialiser-based, chunked dispatch)...")
            with Pool(
                processes=self.n_workers,
                initializer=_worker_init,
                initargs=(self, proc_cfg, self.theoretical_cache),
            ) as pool:
                # Process spectra in chunks: load a chunk from the SDF in the
                # main process, dispatch to workers, accumulate, discard.
                pbar = tqdm(total=n_samples, desc="Analyzing spectra (parallel)")
                for chunk_start in range(0, n_samples, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_samples)
                    # Load chunk from SDF (only this chunk in memory at a time)
                    chunk_args = [
                        (chunk_start + j, sdf[chunk_start + j])
                        for j in range(chunk_end - chunk_start)
                    ]
                    mp_chunksize = max(1, len(chunk_args) // (self.n_workers * 4))
                    for idx, result in pool.imap_unordered(
                        _worker_analyze, chunk_args, chunksize=mp_chunksize
                    ):
                        _accumulate_result(idx, result)
                        pbar.update(1)
                pbar.close()
        else:
            # Single process (for debugging or small datasets)
            for i in tqdm(range(n_samples), desc="Analyzing spectra (single-process)"):
                try:
                    spectrum_data = sdf[i]
                    result = self.analyze_single_spectrum(
                        spectrum_data, processor, include_visualization_data=False
                    )
                    _accumulate_result(i, _detensorize(result))
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        logger.warning(f"Exception analyzing spectrum {i}: {e}")

        if errors > 0:
            logger.warning(f"Analysis completed with {errors} errors out of {n_samples} spectra")
        else:
            logger.info("Analysis completed successfully with no errors")

        # ------------------------------------------------------------------
        # Post-loop: summaries and aggregation using accumulated data
        # ------------------------------------------------------------------
        # Populate analysis_results from accumulators (no per_spectrum list)
        self._calculate_summary_statistics(
            acc_spectrum_stats, acc_theoretical, acc_masking, acc_results_for_stratified,
        )
        self._calculate_mass_error_statistics(
            acc_mass_error_data, acc_unmatched_theo_data, acc_spectrum_mz, acc_frag_type,
        )

        # Generate binning visualizations using BinningAnalyser
        if self.enable_mass_error_analysis and "mass_error_analysis" in self.analysis_results:
            binning_analyser = BinningAnalyser(self.config, self.output_dir / "binning_analysis")
            binning_analyser.results = self.analysis_results["mass_error_analysis"]
            binning_analyser.generate_visualizations()
            binning_analyser.save_results()

        # =====================================================================
        # 3.5. Theoretical Analysis (Comprehensive)
        # =====================================================================
        if self.enable_theoretical:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 3.5: COMPREHENSIVE THEORETICAL ANALYSIS")
            logger.info("=" * 80)

            if acc_theoretical_enriched:
                # Reuse the analyser created in __init__. Its cache is already
                # pointed at self.theoretical_cache; aggregate_results only
                # populates self.results, which is distinct from per-spectrum
                # state, so there is no cross-contamination.
                theoretical_analyser = self.theoretical_analyser

                logger.info(f"Aggregating theoretical data from {len(acc_theoretical_enriched):,d} spectra...")
                theoretical_analysis = theoretical_analyser.aggregate_results(
                    acc_theoretical_enriched,
                    mass_error_data=acc_mass_error_data,
                    unmatched_theo_data=acc_unmatched_theo_data,
                )

                theoretical_analyser.generate_visualizations()
                theoretical_analyser.save_results()
                theoretical_analyser.print_summary()

                self.analysis_results["theoretical_analysis_detailed"] = theoretical_analysis
                logger.info("Comprehensive theoretical analysis complete")
            else:
                logger.warning("No theoretical analysis data available for aggregation")

        # =====================================================================
        # 4. Intensity Analysis
        # =====================================================================
        if self.enable_intensity_analysis and self.intensity_analyser:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 4: INTENSITY DISTRIBUTION ANALYSIS")
            logger.info("=" * 80)

            if acc_intensity:
                logger.info(f"Aggregating intensity data from {len(acc_intensity):,d} spectra...")
                intensity_analysis = self.intensity_analyser.aggregate_results(acc_intensity)
                self.intensity_analyser.generate_visualizations()
                self.intensity_analyser.save_results()
                self.intensity_analyser.print_summary()
                self.analysis_results["intensity_analysis"] = intensity_analysis
                logger.info("Intensity analysis complete")
            else:
                logger.warning("No intensity data available for aggregation")

        # =====================================================================
        # 5. Masking Strategy Comparison Analysis
        # =====================================================================
        if self.enable_masking_analysis and self.masking_analyser:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 5: MASKING STRATEGY COMPARISON ANALYSIS")
            logger.info("=" * 80)

            if acc_masking_result:
                logger.info(f"Aggregating masking data from {len(acc_masking_result):,d} spectra...")
                masking_analysis = self.masking_analyser.aggregate_results(acc_masking_result)
                self.masking_analyser.generate_visualizations()
                self.masking_analyser.generate_individual_spectrum_visualizations(
                    per_spectrum_results=acc_masking_result,
                    max_viz=self.max_visualizations,
                )
                self.masking_analyser.save_results()
                self.masking_analyser.print_summary()
                self.analysis_results["masking_strategy_comparison"] = masking_analysis
                logger.info("Masking strategy comparison analysis complete")
            else:
                logger.warning("No masking analysis data available for aggregation")

        # =====================================================================
        # 6. Custom Ion Detection Aggregation
        # =====================================================================
        if self.enable_custom_ions:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 6: CUSTOM ION DETECTION")
            logger.info("=" * 80)

            if acc_custom_ion:
                from instanovo_fm.utils.theoretical_spectra import DEFAULT_CUSTOM_IONS
                ion_groups = list((self.custom_ions or DEFAULT_CUSTOM_IONS).keys())
                n_spectra = len(acc_custom_ion)

                agg: Dict[str, Dict] = {}
                for group in ion_groups:
                    n_found = sum(
                        1 for r in acc_custom_ion
                        if r.get(group, {}).get("found", False)
                    )
                    agg[group] = {
                        "n_found": n_found,
                        "n_spectra": n_spectra,
                        "hit_rate": n_found / max(n_spectra, 1),
                    }

                sorted_groups = sorted(agg.items(), key=lambda x: x[1]["hit_rate"], reverse=True)
                top_hits = [(g, d) for g, d in sorted_groups if d["hit_rate"] > 0]
                logger.info(f"Custom ion hit rates across {n_spectra:,d} spectra:")
                for group, data in top_hits[:20]:
                    logger.info(
                        f"  {group:<35s} {data['hit_rate']*100:5.1f}%"
                        f"  ({data['n_found']:,d}/{n_spectra:,d})"
                    )
                if len(top_hits) > 20:
                    logger.info(f"  ... and {len(top_hits) - 20} more groups with hits")

                output_path = self.output_dir / "custom_ion_analysis.json"
                with open(output_path, "w") as f:
                    json.dump(
                        {"n_spectra": n_spectra, "ion_hit_rates": dict(sorted_groups)},
                        f, indent=2,
                    )
                logger.info(f"Custom ion analysis saved: {output_path}")

                self.analysis_results["custom_ion_analysis"] = {
                    "n_spectra": n_spectra,
                    "ion_hit_rates": dict(sorted_groups),
                }
                logger.info("Custom ion detection analysis complete")
            else:
                logger.warning("No custom ion detection data available")

        # Generate individual spectrum visualizations (re-analyse a small sample)
        self._generate_individual_visualizations(sdf, processor, acc_valid_indices)

        # Save per-spectrum CSV from accumulated rows
        self._save_results(acc_csv_rows)

        logger.info(f"Spectrum analysis complete. Processed {n_samples:,d} spectra with {errors} errors.")

        # Log match distribution summary
        match_dist = self.analysis_results.get("match_distribution", {})
        if match_dist:
            logger.info(
                f"Match distribution: mean={match_dist.get('mean_matches', 0):.1f}, "
                f"median={match_dist.get('median_matches', 0):.1f} "
                f"(quality gate details in theoretical_analysis/ output)"
            )

        return self.analysis_results

    def _generate_binning_recommendation_report(self):
        """Generate a human-readable comprehensive binning recommendation report."""
        mass_error_analysis = self.analysis_results.get("mass_error_analysis", {})

        if not mass_error_analysis:
            return

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COMPREHENSIVE BINNING STRATEGY RECOMMENDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")

        # Overall recommendation from mass error analysis
        recommendation = mass_error_analysis.get("recommendation", {})
        if recommendation:
            report_lines.append("## Mass Error Analysis Recommendation")
            report_lines.append(f"Strategy: {recommendation.get('strategy', 'N/A')}")
            report_lines.append(f"Reasoning: {recommendation.get('reasoning', 'N/A')}")
            report_lines.append("")

        # Bin-jump analysis results
        bin_jump_analysis = mass_error_analysis.get("bin_jump_analysis")
        if bin_jump_analysis and "summary" in bin_jump_analysis:
            summary = bin_jump_analysis["summary"]
            report_lines.append("## Bin-Jump Rate Analysis (Label Noise)")
            report_lines.append(f"Best Strategy: {summary.get('best_strategy', 'N/A')}")
            report_lines.append(f"Best Mean Jump Rate: {summary.get('best_mean_jump_rate', 0)*100:.2f}%")
            report_lines.append(f"Multi-Observation Ions: {summary.get('n_multi_obs_ions', 0):,d}")
            report_lines.append("")

            report_lines.append("Strategy Rankings (by mean jump rate):")
            for i, (strat, rate) in enumerate(summary.get("strategy_rankings", []), 1):
                report_lines.append(f"  {i}. {strat}: {rate*100:.2f}%")
            report_lines.append("")

        # Hybrid parameter derivation
        if bin_jump_analysis and "hybrid_parameter_derivation" in bin_jump_analysis:
            hybrid = bin_jump_analysis["hybrid_parameter_derivation"]
            if hybrid and "suggested_hybrid" in hybrid and hybrid["suggested_hybrid"]:
                suggested = hybrid["suggested_hybrid"]
                report_lines.append("## Empirically Derived Hybrid Parameters")
                report_lines.append(f"Type: {suggested.get('type', 'N/A')}")
                report_lines.append(f"Base Da: {suggested.get('base_da', 'N/A')}")
                report_lines.append(f"Transition m/z: {suggested.get('transition_mz', 'N/A')}")
                report_lines.append(f"Reasoning: {suggested.get('reasoning', 'N/A')}")
                report_lines.append("")

        # Part D: Extended analysis
        extended_analysis = mass_error_analysis.get("extended_analysis")
        if extended_analysis:
            # Collision analysis
            if "collision_analysis" in extended_analysis:
                collision = extended_analysis["collision_analysis"]
                if "error" not in collision:
                    report_lines.append("## Bin Collision Analysis (Information Loss)")
                    report_lines.append(f"Best Strategy: {collision.get('best_strategy', 'N/A')}")
                    report_lines.append(f"Best Collision Rate: {collision.get('best_collision_rate', 0)*100:.2f}%")
                    report_lines.append(f"Unique Ions Analyzed: {collision.get('n_unique_ions', 0):,d}")
                    report_lines.append("")

            # Stratified analysis
            if "stratified_analysis" in extended_analysis:
                stratified = extended_analysis["stratified_analysis"]
                if "error" not in stratified and "overall_recommendation" in stratified:
                    overall_rec = stratified["overall_recommendation"]
                    report_lines.append("## Stratified Analysis (Per Fragmentation Type)")
                    report_lines.append(f"Best Overall Strategy: {overall_rec.get('best_overall_strategy', 'N/A')}")
                    report_lines.append("")

                    if "per_frag_type_best" in overall_rec:
                        report_lines.append("Per-Fragmentation-Type Recommendations:")
                        for frag_type, best_strat in overall_rec["per_frag_type_best"].items():
                            report_lines.append(f"  {frag_type}: {best_strat}")
                        report_lines.append("")

            # Soft label analysis
            if "soft_label_analysis" in extended_analysis:
                soft_label = extended_analysis["soft_label_analysis"]
                if "error" not in soft_label and "recommendation" in soft_label:
                    rec = soft_label["recommendation"]
                    report_lines.append("## Soft Label Recommendation")
                    report_lines.append(f"Recommend Soft Labels: {rec.get('recommend_soft_labels', False)}")
                    report_lines.append(f"Reasoning: {rec.get('reasoning', 'N/A')}")
                    report_lines.append(f"Max Expected Benefit: {rec.get('max_expected_benefit', 0)*100:.2f}%")
                    report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)

        # Save report
        report_path = self.output_dir / "binning_recommendation_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        logger.info(f"Binning recommendation report saved to: {report_path}")

    def _calculate_summary_statistics(
        self,
        spectrum_stats: List[Dict],
        theoretical_analyses: List[Dict],
        masking_analyses: List[Dict],
        valid_results_for_stratified: List[Dict],
    ) -> None:
        """Calculate summary statistics from pre-accumulated lists.

        Args:
            spectrum_stats: Per-spectrum stats dicts (n_valid_peaks, total_intensity, etc.).
            theoretical_analyses: Per-spectrum theoretical analysis dicts.
            masking_analyses: Per-spectrum masking analysis dicts.
            valid_results_for_stratified: Per-spectrum dicts with metadata + theo + masking + stats
                for stratified grouping.
        """
        if not spectrum_stats:
            logger.warning("No valid results to summarize")
            return

        # Get all theoretical results (with sequence)
        theo_results = [t for t in theoretical_analyses if t.get("sequence_available", False)]

        # Apply quality gate: filter to spectra meeting backbone coverage and
        # fragment group diversity thresholds.  The old min_matched_peaks filter
        # was replaced by this two-criterion gate — spectra that fail are
        # excluded from all downstream signal-quality statistics so that noisy
        # spectra with poor fragmentation do not dilute the reported metrics.
        n_terminal_ions = {"b", "a", "c"}
        c_terminal_ions = {"y", "x", "z"}

        def _passes_quality_gate(t: Dict[str, Any]) -> bool:
            """Check backbone coverage and fragment group count."""
            feature_types = t.get("feature_types")
            annotations = t.get("theo_annotations")
            if not feature_types or not annotations:
                return False
            seq = t.get("clean_sequence", "")
            seq_len = len(seq) if seq else 0
            # Collect base fragment groups: {(ion_type, position)}
            groups: set = set()
            for ft, ann in zip(feature_types, annotations):
                if ft != "base" or not ann:
                    continue
                ion_type = TheoreticalAnalyser._extract_ion_type(ann)
                position = TheoreticalAnalyser._extract_fragment_position(ann)
                if position < 1 or ion_type == "unknown":
                    continue
                groups.add((ion_type, position))
            n_groups = len(groups)
            # Backbone cleavage coverage
            if seq_len > 1:
                max_sites = seq_len - 1
                cleavage_sites: set = set()
                for (ion_type, position) in groups:
                    if ion_type in n_terminal_ions:
                        cleavage_sites.add(position)
                    elif ion_type in c_terminal_ions:
                        cleavage_sites.add(seq_len - position)
                coverage = len(cleavage_sites) / max_sites
            else:
                coverage = 0.0
            return (
                coverage >= self.min_backbone_coverage
                and n_groups >= self.min_fragment_groups
            )

        hq_results = [t for t in theo_results if _passes_quality_gate(t)]

        # Calculate match distribution statistics
        n_matched_list = [t["n_matched"] for t in theo_results]

        # Basic statistics (all valid spectra)
        self.analysis_results["summary"] = {
            "total_spectra_analyzed": len(spectrum_stats),
            "total_spectra_with_sequence": len(theo_results),
            "total_spectra_high_quality": len(hq_results),
            "total_spectra_below_quality_gate": len(theo_results) - len(hq_results),
            "min_backbone_coverage": self.min_backbone_coverage,
            "min_fragment_groups": self.min_fragment_groups,
            "avg_peaks_per_spectrum": np.mean([s["n_valid_peaks"] for s in spectrum_stats]),
            "avg_total_intensity": np.mean([s["total_intensity"] for s in spectrum_stats]),
        }

        # Match distribution statistics (all spectra with sequence, informational)
        if theo_results:
            self.analysis_results["match_distribution"] = {
                "n_matched_peaks": n_matched_list,
                "min_matches": int(np.min(n_matched_list)),
                "max_matches": int(np.max(n_matched_list)),
                "mean_matches": float(np.mean(n_matched_list)),
                "median_matches": float(np.median(n_matched_list)),
                "std_matches": float(np.asarray(n_matched_list, dtype=np.float64).std()),
            }

        # Theoretical matching statistics (high-quality spectra only)
        if hq_results:
            self.analysis_results["theoretical_matching"] = {
                "n_spectra_high_quality": len(hq_results),
                "avg_match_rate": np.mean([t["match_rate"] for t in hq_results]),
                "avg_frac_intensity": np.mean([t["frac_intensity"] for t in hq_results]),
                "avg_median_ppm": np.nanmean([t["median_ppm"] for t in hq_results if not np.isnan(t["median_ppm"])]),
                "avg_mean_ppm": np.nanmean([t["mean_ppm"] for t in hq_results if not np.isnan(t["mean_ppm"])]),
                "avg_annotated_fraction": np.mean([t["annotated_fraction"] for t in hq_results]),
                "avg_unannotated_fraction": np.mean([t["unannotated_fraction"] for t in hq_results]),
                "avg_annotated_intensity_fraction": np.mean([t["annotated_intensity_fraction"] for t in hq_results]),
                "avg_unannotated_intensity_fraction": np.mean([t["unannotated_intensity_fraction"] for t in hq_results]),
            }

            # Add conditional annotation statistics if available
            if self.use_conditional_annotation:
                results_with_breakdown = [t for t in hq_results if "n_base" in t]
                if results_with_breakdown:
                    stats_update = {
                        "avg_n_base": np.mean([t["n_base"] for t in results_with_breakdown]),
                        "avg_n_losses": np.mean([t["n_losses"] for t in results_with_breakdown]),
                        "avg_n_isotopes": np.mean([t["n_isotopes"] for t in results_with_breakdown]),
                        "avg_n_matched": np.mean([t["n_matched"] for t in results_with_breakdown]),
                    }
                    # Add precursor stats if available
                    if any("n_precursor" in t for t in results_with_breakdown):
                        stats_update["avg_n_precursor"] = np.mean([t.get("n_precursor", 0) for t in results_with_breakdown])
                    self.analysis_results["theoretical_matching"].update(stats_update)

            self.analysis_results["annotation_analysis"] = {
                "annotated_fractions": [t["annotated_fraction"] for t in hq_results],
                "unannotated_fractions": [t["unannotated_fraction"] for t in hq_results],
                "annotated_intensity_fractions": [t["annotated_intensity_fraction"] for t in hq_results],
                "unannotated_intensity_fractions": [t["unannotated_intensity_fraction"] for t in hq_results],
                "match_rates": [t["match_rate"] for t in hq_results],
                "frac_intensities": [t["frac_intensity"] for t in hq_results],
            }

            label_counts_list = [
                t.get("annotation_label_counts", {})
                for t in hq_results
                if t.get("annotation_label_counts") is not None
            ]
            label_stats = {}
            if label_counts_list:
                all_labels = sorted({label for counts in label_counts_list for label in counts.keys()})
                for label in all_labels:
                    values = np.array([counts.get(label, 0) for counts in label_counts_list], dtype=np.float64)
                    label_stats[label] = {
                        "mean": float(np.mean(values)),
                        "std": float(values.std()),
                    }
            self.analysis_results["annotation_label_stats"] = label_stats

        def _aggregate_group_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
            if not results:
                return {}
            masking_results = [r.get("masking_analysis", {}) for r in results]
            label_counts_list = [
                r.get("theoretical_analysis", {}).get("annotation_label_counts", {})
                for r in results
                if r.get("theoretical_analysis", {}).get("annotation_label_counts") is not None
            ]

            label_stats = {}
            if label_counts_list:
                all_labels = sorted({label for counts in label_counts_list for label in counts.keys()})
                for label in all_labels:
                    values = np.array([counts.get(label, 0) for counts in label_counts_list], dtype=np.float64)
                    label_stats[label] = {
                        "mean": float(np.mean(values)),
                        "std": float(values.std()),
                    }

            masking_summary = {}
            if masking_results:
                masking_summary = {
                    "avg_annotated_mask_ratio": float(np.mean([m.get("annotated_mask_ratio", 0) for m in masking_results])),
                    "avg_unannotated_mask_ratio": float(np.mean([m.get("unannotated_mask_ratio", 0) for m in masking_results])),
                    "avg_overall_mask_ratio": float(np.mean([m.get("overall_mask_ratio", 0) for m in masking_results])),
                    "avg_annotated_preservation_ratio": float(np.mean([m.get("annotated_preservation_ratio", 0) for m in masking_results])),
                    "avg_annotated_fraction_of_masked_peaks": float(np.mean([m.get("annotated_fraction_of_masked_peaks", 0) for m in masking_results])),
                    "avg_unannotated_fraction_of_masked_peaks": float(np.mean([m.get("unannotated_fraction_of_masked_peaks", 0) for m in masking_results])),
                    "avg_annotated_intensity_fraction_of_masked": float(np.mean([m.get("annotated_intensity_fraction_of_masked", 0) for m in masking_results])),
                    "avg_unannotated_intensity_fraction_of_masked": float(np.mean([m.get("unannotated_intensity_fraction_of_masked", 0) for m in masking_results])),
                    "avg_annotated_intensity_masked_fraction_of_annotated": float(np.mean([
                        m.get("annotated_intensity_masked_fraction_of_annotated", 0) for m in masking_results
                    ])),
                    "avg_annotated_intensity_unmasked_fraction_of_annotated": float(np.mean([
                        m.get("annotated_intensity_unmasked_fraction_of_annotated", 0) for m in masking_results
                    ])),
                    "avg_unannotated_intensity_masked_fraction_of_unannotated": float(np.mean([
                        m.get("unannotated_intensity_masked_fraction_of_unannotated", 0) for m in masking_results
                    ])),
                    "avg_unannotated_intensity_unmasked_fraction_of_unannotated": float(np.mean([
                        m.get("unannotated_intensity_unmasked_fraction_of_unannotated", 0) for m in masking_results
                    ])),
                    "avg_fragment_group_count": float(np.mean([m.get("fragment_group_count", 0) for m in masking_results])),
                    "avg_fragment_group_full_mask_ratio": float(np.mean([m.get("fragment_group_full_mask_ratio", 0) for m in masking_results])),
                    "avg_fragment_group_avg_mask_fraction": float(np.mean([m.get("fragment_group_avg_mask_fraction", 0) for m in masking_results])),
                    "avg_fragment_group_weighted_mask_fraction": float(np.mean([m.get("fragment_group_weighted_mask_fraction", 0) for m in masking_results])),
                }

            return {
                "n_spectra": len(results),
                "masking_summary": masking_summary,
                "annotation_label_stats": label_stats,
            }

        def _build_stratified_summary(
            results: List[Dict[str, Any]],
            key: str,
            min_group_size: int = 20,
            max_groups: int = 10,
            group_small_into_other: bool = True,
        ) -> Dict[str, Any]:
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for result in results:
                metadata = result.get("metadata", {})
                raw_value = metadata.get(key, "unknown")
                group_value = str(raw_value) if raw_value is not None else "unknown"
                grouped.setdefault(group_value, []).append(result)

            items = sorted(grouped.items(), key=lambda kv: len(kv[1]), reverse=True)
            selected = items[:max_groups]
            summary: Dict[str, Any] = {}
            other_group: List[Dict[str, Any]] = []

            for group_value, group_results in selected:
                if len(group_results) < min_group_size:
                    if group_small_into_other:
                        other_group.extend(group_results)
                    continue
                summary[group_value] = _aggregate_group_metrics(group_results)

            if group_small_into_other:
                for _, group_results in items[max_groups:]:
                    other_group.extend(group_results)
                if other_group:
                    summary["OTHER"] = _aggregate_group_metrics(other_group)

            return summary

        # Masking analysis now handled by MaskingAnalyser in Phase 5

        # Stratified analysis (all spectra with sequence)
        seq_indices = [
            i for i, t in enumerate(theoretical_analyses)
            if t.get("sequence_available", False)
        ]
        if seq_indices:
            stratified_results = [valid_results_for_stratified[i] for i in seq_indices]
            self.analysis_results["stratified_summary"] = {
                "frag_type": _build_stratified_summary(stratified_results, "frag_type"),
                "precursor_charge": _build_stratified_summary(stratified_results, "precursor_charge"),
                "search_instrument": _build_stratified_summary(stratified_results, "search_instrument"),
            }

    def _calculate_mass_error_statistics(
        self,
        all_error_data: List[Dict],
        all_unmatched_theo_data: List[Dict],
        all_spectrum_mz: List[np.ndarray],
        per_spectrum_frag_type: List[str],
    ) -> None:
        """Calculate mass error statistics and run binning analysis.

        Args:
            all_error_data: Per-peak mass error records from matched ions.
            all_unmatched_theo_data: Records for theoretical ions with no experimental match.
            all_spectrum_mz: Per-spectrum arrays of observed m/z values.
            per_spectrum_frag_type: Fragmentation type string per spectrum.
        """
        if not self.enable_mass_error_analysis:
            return

        logger.info("Calculating mass error statistics for binning analysis...")

        if not all_error_data:
            logger.warning("No mass error data available for analysis")
            return

        df = pd.DataFrame(all_error_data)

        total_matched = len(df)
        total_unmatched = len(all_unmatched_theo_data)
        total_theoretical = total_matched + total_unmatched
        coverage_stats = {
            "overall_coverage_rate": float(total_matched / max(total_theoretical, 1)),
            "total_theoretical_ions": total_theoretical,
            "total_matched": total_matched,
            "total_unmatched": total_unmatched,
        }

        binning_analyser = BinningAnalyser(self.config, self.output_dir / "binning_analysis")
        binning_results = binning_analyser.analyze(
            df, coverage_stats, all_unmatched_theo_data,
            per_spectrum_mz=all_spectrum_mz,
            per_spectrum_frag_type=per_spectrum_frag_type,
        )

        self.analysis_results["mass_error_analysis"] = binning_results

        logger.info("Mass error analysis complete.")

    def _generate_individual_visualizations(
        self,
        sdf: SpectrumDataFrame,
        processor: FoundationalDataProcessor,
        valid_indices: List[int],
    ) -> None:
        """Generate individual spectrum visualizations.

        Re-analyses a small sample (~max_visualizations) from the SDF with
        ``include_visualization_data=True`` to produce per-spectrum plots.

        Args:
            sdf: Source SpectrumDataFrame.
            processor: Data processor for spectrum analysis.
            valid_indices: Indices of successfully-analysed spectra.
        """
        if self.max_visualizations <= 0 or not valid_indices:
            return

        logger.info(f"Generating up to {self.max_visualizations} individual spectrum visualizations...")

        viz_dir = self.output_dir / "individual_spectra"
        viz_dir.mkdir(parents=True, exist_ok=True)

        n_viz = min(self.max_visualizations, len(valid_indices))
        # Sample evenly across the valid indices
        sample_positions = np.linspace(0, len(valid_indices) - 1, n_viz, dtype=int)
        selected_sdf_indices = [valid_indices[p] for p in sample_positions]

        logger.info(f"Re-analysing {n_viz} spectra for visualisation...")
        for viz_idx, sdf_idx in enumerate(tqdm(selected_sdf_indices, desc="Creating individual visualizations")):
            try:
                spectrum_data = sdf[sdf_idx]
                result = self.analyze_single_spectrum(
                    spectrum_data, processor, include_visualization_data=True
                )
                if result.get("error"):
                    continue

                fig = self._create_individual_spectrum_plot(result)

                metadata = result.get("metadata", {})
                clean_sequence = metadata.get("clean_sequence")
                if clean_sequence:
                    seq_for_filename = sanitize_filename(clean_sequence)[:30]
                else:
                    seq_for_filename = "unknown"

                filename = f"spectrum_{viz_idx:04d}_{seq_for_filename}.png"
                filepath = viz_dir / filename
                fig.savefig(filepath, format='png', dpi=150, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to create visualization for spectrum {viz_idx}: {e}")
                continue

        logger.info(f"Individual visualizations saved to: {viz_dir}")

    def _save_results(self, csv_rows: List[Dict]) -> None:
        """Save analysis summary JSON and per-spectrum CSV.

        Args:
            csv_rows: Pre-accumulated row dicts for the per-spectrum CSV.
        """
        match_dist = self.analysis_results.get("match_distribution", {})
        match_dist_summary = {k: v for k, v in match_dist.items() if k != "n_matched_peaks"}

        summary_only = {
            "summary": self.analysis_results.get("summary", {}),
            "match_distribution": match_dist_summary,
            "theoretical_analysis": "see theoretical_analysis/ directory",
            "masking_analysis": "see masking_analysis/ directory",
        }

        summary_path = self.output_dir / "spectrum_analysis_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_only, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x)
        logger.info(f"Analysis summary saved to: {summary_path}")

        if csv_rows:
            df = pd.DataFrame(csv_rows)
            theo_dir = self.output_dir / "theoretical_analysis"
            theo_dir.mkdir(parents=True, exist_ok=True)
            csv_path = theo_dir / "per_spectrum_results.csv"
            df.to_csv(csv_path, index=False)
            logger.info(f"Per-spectrum results saved to: {csv_path} ({len(csv_rows):,d} spectra)")

    def _create_individual_spectrum_plot(self, result: Dict[str, Any]) -> Figure:
        """
        Create a detailed plot for a single spectrum with theoretical annotation.

        3-Row Layout:
        -------------
        [Row 0, Col 0] Statistics Box
                       - Spectrum and theoretical statistics
        [Row 0, Col 1] Annotation Distribution (percentage bars)
                       - Annotated vs unannotated by peak count and intensity

        [Row 1]        m/z Spectrum with Theoretical Matching (FULL WIDTH, LARGE)
                       - True m/z positions; ion-type colouring; filtered labels

        [Row 2]        Index-Based Spectrum (FULL WIDTH)
                       - Peaks evenly spaced by index — every annotation shown
                       - Same ion-type colour scheme; m/z context on x-ticks

        Note: Masking visualisation now lives in masking_analysis/ output.
        """
        fig = plt.figure(figsize=(20, 18))
        gs = fig.add_gridspec(3, 2, height_ratios=[2, 5, 5], width_ratios=[3, 2], hspace=0.28, wspace=0.4, top=0.97)

        # Build informative title with key metadata
        metadata = result.get("metadata", {})
        spec_stats = result.get("spectrum_stats", {})
        theo_analysis = result.get("theoretical_analysis", {})

        # Core identification
        clean_sequence = metadata.get("clean_sequence", result.get('spectrum_id', 'unknown'))

        # Key proteomics metadata
        precursor_charge = theo_analysis.get("precursor_charge") or spec_stats.get("precursor_charge")
        frag_type = metadata.get("frag_type", "unknown")

        # Build title components
        title_parts = [f"{clean_sequence}"]

        # Add charge state (critical for interpretation)
        if precursor_charge:
            title_parts.append(f"z={precursor_charge}")

        # Add fragmentation type (affects ion types)
        if frag_type and frag_type != "unknown":
            title_parts.append(f"{frag_type}")

        # Combine into header (removed - no longer displayed as suptitle)
        title = " | ".join(title_parts)
        # fig.suptitle(f"Spectrum Analysis: {title}", fontsize=16, fontweight='bold', y=0.995)

        spec_stats = result["spectrum_stats"]
        theo_analysis = result["theoretical_analysis"]
        viz_data = result.get("visualization_data", {})

        # 1. Statistics Box (row 0, col 0)
        ax_stats = fig.add_subplot(gs[0, 0])
        ax_stats.axis('off')

        # Build statistics text
        clean_seq = theo_analysis.get('clean_sequence', '')
        seq_len = len(clean_seq) if clean_seq else 0
        prec_mz = metadata.get('precursor_mz')
        prec_mass = metadata.get('precursor_mass')
        prec_charge = theo_analysis.get('precursor_charge')

        # Peak density (peaks per 100 Da)
        mz_lo, mz_hi = spec_stats['mz_range']
        mz_span = mz_hi - mz_lo
        peak_density = spec_stats['n_valid_peaks'] / max(mz_span, 1.0) * 100

        stats_lines = [
            "Spectrum Statistics:",
            f"  • Peaks: {spec_stats['n_valid_peaks']} valid (of {spec_stats['n_total_peaks']} total)  |  Density: {peak_density:.1f} peaks/100 Da",
            f"  • m/z range: [{mz_lo:.1f}, {mz_hi:.1f}]  |  Total intensity: {spec_stats['total_intensity']:.2f}",
        ]

        # Precursor line — grouped m/z, mass, charge for quick visual reference
        prec_parts = []
        if prec_mz is not None:
            prec_parts.append(f"m/z {prec_mz:.4f}")
        if prec_mass is not None:
            prec_parts.append(f"Mass {prec_mass:.2f} Da")
        if prec_charge is not None:
            prec_parts.append(f"z={prec_charge}")
        if prec_parts:
            stats_lines.append(f"  • Precursor: {'  |  '.join(prec_parts)}")

        stats_lines.extend(["", "Theoretical Matching:"])

        # Sequence with length
        seq_display = clean_seq[:50] if clean_seq else 'N/A'
        if seq_len > 0:
            stats_lines.append(f"  • Sequence: {seq_display} ({seq_len} residues)")
        else:
            stats_lines.append(f"  • Sequence: {seq_display}")

        stats_lines.append(
            f"  • Fragmentation: {metadata.get('frag_type', 'unknown')}  |  Instrument: {metadata.get('search_instrument', 'unknown')}"
        )

        # Matching stats with PPM bias
        n_theo = theo_analysis.get('n_theoretical', 0)
        n_matched = theo_analysis.get('n_matched', 0)
        match_rate = theo_analysis.get('match_rate', 0) * 100
        median_ppm = theo_analysis.get('median_ppm', np.nan)
        mean_ppm = theo_analysis.get('mean_ppm', np.nan)
        ppm_str = f"Median PPM: {median_ppm:.2f}"
        if not np.isnan(mean_ppm):
            ppm_str += f" (bias: {mean_ppm:+.2f})"
        stats_lines.append(f"  • Matched: {n_matched}/{n_theo} theoretical ({match_rate:.1f}%)  |  {ppm_str}")

        # Ion series coverage (b/y base ions vs possible cleavage sites)
        label_counts = theo_analysis.get('annotation_label_counts', {})
        if label_counts and seq_len > 1:
            n_possible = seq_len - 1
            n_b = label_counts.get('b-ion', 0)
            n_y = label_counts.get('y-ion', 0)
            stats_lines.append(
                f"  • Ion coverage: B={n_b}/{n_possible} ({n_b/n_possible*100:.0f}%)  |  Y={n_y}/{n_possible} ({n_y/n_possible*100:.0f}%)"
            )

        # Conditional annotation breakdown
        if 'n_base' in theo_analysis:
            breakdown_parts = [f"Base={theo_analysis.get('n_base', 0)}"]
            if theo_analysis.get('n_precursor', 0) > 0:
                breakdown_parts.append(f"Precursor={theo_analysis.get('n_precursor', 0)}")
            breakdown_parts.extend([
                f"Losses={theo_analysis.get('n_losses', 0)}",
                f"Isotopes={theo_analysis.get('n_isotopes', 0)}"
            ])
            stats_lines.append(f"  • Breakdown: {' | '.join(breakdown_parts)}")

        stats_lines.append(
            f"  • Annotated: {theo_analysis.get('annotated_fraction', 0)*100:.1f}% peaks  |  {theo_analysis.get('frac_intensity', 0)*100:.1f}% intensity"
        )

        # Custom ion detection summary
        custom_ion_data = theo_analysis.get('custom_ion_data', {})
        custom_detection = custom_ion_data.get('detection', {})
        if custom_detection:
            found_ions = [name for name, d in custom_detection.items() if d.get('found')]
            n_custom_lift = custom_ion_data.get('n_custom_explaining_unannotated', 0)
            if found_ions:
                stats_lines.append(
                    f"  • Custom ions: {len(found_ions)} detected  |  {n_custom_lift} unannotated peaks explained"
                )

        stats_text = "\n".join(stats_lines)

        ax_stats.text(0.02, 0.5, stats_text, transform=ax_stats.transAxes, fontsize=10,
                     verticalalignment='center', horizontalalignment='left', fontfamily='monospace',
                     bbox=dict(boxstyle="round,pad=0.8", facecolor="lightgray", alpha=0.9, edgecolor='gray', linewidth=1.5))

        # 1b. Annotation Distribution (row 0, col 1) — percentages so both groups share a common scale
        ax_dist = fig.add_subplot(gs[0, 1])
        if theo_analysis.get('sequence_available', False) and 'annotated_mask' in theo_analysis:
            _ann_mask = np.array(theo_analysis['annotated_mask'])
            _n_ann = int(_ann_mask.sum())
            _n_unann = int((~_ann_mask).sum())
            _total_peaks = _n_ann + _n_unann

            _ann_int = 0.0
            _unann_int = 0.0
            if viz_data:
                _valid_peaks = viz_data['valid_peaks']
                _int_values = viz_data['intensity_values']
                _valid_int = _int_values[_valid_peaks] if _valid_peaks.any() else np.array([])
                if len(_valid_int) == len(_ann_mask):
                    _ann_int = float(_valid_int[_ann_mask].sum())
                    _unann_int = float(_valid_int[~_ann_mask].sum())
            _total_int = _ann_int + _unann_int

            # Convert to percentages so both groups are on the same 0-100% axis
            _peak_ann_pct = _n_ann / _total_peaks * 100 if _total_peaks > 0 else 0.0
            _peak_unann_pct = _n_unann / _total_peaks * 100 if _total_peaks > 0 else 0.0
            _int_ann_pct = _ann_int / _total_int * 100 if _total_int > 0 else 0.0
            _int_unann_pct = _unann_int / _total_int * 100 if _total_int > 0 else 0.0

            _x = np.arange(2)
            _w = 0.35
            ax_dist.bar(_x - _w / 2, [_peak_ann_pct, _int_ann_pct],
                        _w, label='Annotated', alpha=0.85, color='#3498db')
            ax_dist.bar(_x + _w / 2, [_peak_unann_pct, _int_unann_pct],
                        _w, label='Unannotated', alpha=0.85, color='#e74c3c')

            # Percentage labels above each bar
            for _xi, (_ann_pct, _unann_pct) in zip(_x, [(_peak_ann_pct, _peak_unann_pct),
                                                         (_int_ann_pct, _int_unann_pct)]):
                ax_dist.text(_xi - _w / 2, _ann_pct + 1.5, f"{_ann_pct:.0f}%", ha='center', fontsize=9)
                ax_dist.text(_xi + _w / 2, _unann_pct + 1.5, f"{_unann_pct:.0f}%", ha='center', fontsize=9)

            ax_dist.set_xticks(_x)
            ax_dist.set_xticklabels(['Peak Count', 'Total Intensity'])
            ax_dist.set_ylabel('%', fontsize=10)
            ax_dist.set_ylim(0, 115)
            ax_dist.set_title('Annotation Distribution', fontsize=11, fontweight='bold')
            ax_dist.legend(fontsize=9)
            ax_dist.grid(True, alpha=0.3, axis='y')
        else:
            ax_dist.text(0.5, 0.5, "No annotation data available", ha='center', va='center',
                        transform=ax_dist.transAxes, fontsize=12)
            ax_dist.set_title('Annotation Distribution')

        # 2. Spectrum with Theoretical Matching (row 1, spans full width)
        ax_theo = fig.add_subplot(gs[1, :])

        if viz_data and theo_analysis.get('sequence_available', False):
            mz_values = viz_data['mz_values']
            intensity_values = viz_data['intensity_values']
            valid_peaks = viz_data['valid_peaks']

            # Get annotated mask from theoretical analysis
            annotated_mask_full = np.zeros(len(mz_values), dtype=bool)
            if 'annotated_mask' in theo_analysis:
                annotated_mask = np.array(theo_analysis['annotated_mask'])
                annotated_mask_full[valid_peaks] = annotated_mask

            # Get annotations and categorize peaks
            all_annotations = list(theo_analysis.get('theo_annotations', []))

            # Merge custom ion detections into annotations for unannotated peaks
            custom_ion_data = theo_analysis.get('custom_ion_data', {})
            custom_detection = custom_ion_data.get('detection', {})
            if custom_detection:
                valid_mz_arr = mz_values[valid_peaks] if valid_peaks.any() else np.array([])

                def _annotate_custom_peak(mz_val: float, label: str) -> None:
                    """Set annotation for the peak nearest to *mz_val* if unannotated."""
                    if len(valid_mz_arr) == 0:
                        return
                    idx = int(np.argmin(np.abs(valid_mz_arr - mz_val)))
                    if abs(valid_mz_arr[idx] - mz_val) / max(mz_val, 1e-12) * 1e6 < self.ppm_tol * 2:
                        if idx < len(all_annotations) and all_annotations[idx]:
                            return  # already has a fragment annotation
                        if len(all_annotations) <= idx:
                            all_annotations.extend([''] * (idx + 1 - len(all_annotations)))
                        all_annotations[idx] = label

                for group_name, group_data in custom_detection.items():
                    if not group_data.get('found'):
                        continue
                    # Monoisotopic matches
                    for matched_mz_val in group_data.get('matched_mz', []):
                        _annotate_custom_peak(matched_mz_val, f"custom:{group_name}@{matched_mz_val:.4f}")
                    # Isotope matches (from conditional Pass 2)
                    for iso_match in group_data.get('isotope_matches', []):
                        iso_mz = iso_match['matched_mz']
                        iso_num = iso_match['isotope_num']
                        _annotate_custom_peak(iso_mz, f"custom:{group_name}[+{iso_num}]@{iso_mz:.4f}")

            # Categorize peaks by ion type
            # Categorize all valid peaks
            # Create mapping from full-spectrum indices to valid-peaks indices
            valid_peak_indices = np.where(valid_peaks)[0]  # Full-spectrum indices of valid peaks
            peak_categories = {}
            for valid_idx, full_idx in enumerate(valid_peak_indices):
                if valid_idx < len(all_annotations):
                    peak_categories[full_idx] = categorize_ion(all_annotations[valid_idx])
                else:
                    peak_categories[full_idx] = "unannotated"

            # Plot peaks by category (unannotated first, then by category)
            plotted_categories = set()

            # Plot unmatched first (background)
            unannotated_indices = [i for i, cat in peak_categories.items() if cat == "unannotated"]
            if unannotated_indices:
                un_color, un_alpha = category_colors["unannotated"]
                markerline, stemlines, baseline = ax_theo.stem(
                    mz_values[unannotated_indices], intensity_values[unannotated_indices],
                    linefmt='-', markerfmt='o', basefmt=' ', label='Unannotated')
                plt.setp(markerline, color=un_color, alpha=un_alpha, markersize=3)
                plt.setp(stemlines, color=un_color, alpha=un_alpha)
                plotted_categories.add("unannotated")

            # Plot matched peaks by category
            for category in sorted(set(peak_categories.values()) - {"unannotated"}):
                cat_indices = [i for i, cat in peak_categories.items() if cat == category]
                if cat_indices:
                    color, alpha = category_colors.get(category, ("#9467bd", 0.8))
                    markerline, stemlines, baseline = ax_theo.stem(
                        mz_values[cat_indices], intensity_values[cat_indices],
                        linefmt='-', markerfmt='o', basefmt=' ',
                        label=category)
                    plt.setp(markerline, color=color, alpha=alpha, markersize=3)
                    plt.setp(stemlines, color=color, alpha=alpha)
                    plotted_categories.add(category)

            # Add ion annotations (vertical text)
            if all_annotations:
                # Collect non-empty annotations indexed by valid-peaks position
                valid_peak_indices = np.where(valid_peaks)[0]
                annotated_peaks = []
                for valid_idx, full_idx in enumerate(valid_peak_indices):
                    if valid_idx < len(all_annotations) and all_annotations[valid_idx] and all_annotations[valid_idx] != "":
                        annotated_peaks.append((mz_values[full_idx], intensity_values[full_idx], all_annotations[valid_idx]))

                # Filter to peaks above 3% of max intensity, then apply a minimum
                # m/z separation (prioritising strongest peaks) to prevent label collisions.
                max_valid_intensity = intensity_values[valid_peaks].max() if valid_peaks.any() else 1.0
                min_label_intensity = max_valid_intensity * 0.03
                by_intensity = sorted(annotated_peaks, key=lambda t: t[1], reverse=True)
                placed_mz: list[float] = []
                min_mz_gap = 20.0  # minimum m/z distance between labels
                labeled_peaks = []
                for mz, inten, ann in by_intensity:
                    if inten < min_label_intensity:
                        continue
                    if any(abs(mz - p) < min_mz_gap for p in placed_mz):
                        continue
                    labeled_peaks.append((mz, inten, ann))
                    placed_mz.append(mz)
                labeled_peaks.sort(key=lambda t: t[0])  # left-to-right order

                for mz, intensity, ann in labeled_peaks:
                    display_ann = format_annotation_display(ann)

                    # Determine text color from category
                    category = categorize_ion(ann)
                    text_color = text_colors.get(category, "black")

                    ax_theo.annotate(
                        display_ann,
                        xy=(mz, intensity),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=7, color=text_color,
                        rotation=90,  # Vertical text
                        alpha=0.9, fontweight='bold'
                    )

            ax_theo.set_xlabel('m/z', fontsize=11, fontweight='bold')
            ax_theo.set_ylabel('Normalized Intensity', fontsize=11, fontweight='bold')

            # Add annotation mode to title
            annotation_mode = "Conditional" if self.use_conditional_annotation else "Traditional"
            ax_theo.set_title(f'Spectrum with Theoretical Matching (by Ion Type) - {annotation_mode} Mode',
                             fontsize=12, fontweight='bold')

            # Create legend with note about conditional annotation
            legend = ax_theo.legend(loc='upper right', fontsize=9, ncol=2,
                                    markerscale=0.7, borderpad=0.4, labelspacing=0.3,
                                    handlelength=1.2, handletextpad=0.4)
            if self.use_conditional_annotation and ('n_base' in theo_analysis):
                # Add note about conditional annotation
                note_parts = [f"{theo_analysis.get('n_base', 0)} base"]
                if theo_analysis.get('n_precursor', 0) > 0:
                    note_parts.append(f"{theo_analysis.get('n_precursor', 0)} precursor")
                note_parts.extend([
                    f"{theo_analysis.get('n_losses', 0)} losses",
                    f"{theo_analysis.get('n_isotopes', 0)} isotopes"
                ])
                note_text = f"Conditional: {' + '.join(note_parts)}"
                ax_theo.text(0.98, 0.02, note_text, transform=ax_theo.transAxes,
                           fontsize=8, ha='right', va='bottom',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8, edgecolor='gray'))

            ax_theo.grid(True, alpha=0.3)
            ax_theo.set_xlim((self.min_mz, self.max_mz))

            # Extra headroom so vertical annotation labels don't clip at the top
            max_intensity = intensity_values[valid_peaks].max() if valid_peaks.any() else 1.0

            # Precursor m/z indicator — vertical dashed line at the exact
            # precursor m/z position so missed precursor ions are immediately
            # visible against the annotated fragment peaks.
            if prec_mz is not None:
                prec_charge_lbl = f" (z={prec_charge})" if prec_charge else ""
                ax_theo.axvline(
                    x=prec_mz, color='#ff7f0e', linestyle='--', linewidth=1.5,
                    alpha=0.85, zorder=5,
                    label=f"Precursor m/z={prec_mz:.2f}{prec_charge_lbl}",
                )
                ax_theo.text(
                    prec_mz, max_intensity * 1.30,
                    f"prec\n{prec_mz:.2f}{prec_charge_lbl}",
                    ha='center', va='top', fontsize=7, color='#ff7f0e',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              alpha=0.7, edgecolor='#ff7f0e'),
                )

            ax_theo.set_ylim((0, max_intensity * 1.35))
        else:
            ax_theo.text(0.5, 0.5, "No theoretical data available", ha='center', va='center',
                        transform=ax_theo.transAxes, fontsize=12)
            ax_theo.set_title('Spectrum with Theoretical Matching')

        # 3. Index-based spectrum (row 2, spans full width)
        # Peaks are evenly spaced by index so every annotation can be shown without
        # the m/z crowding that affects the m/z panel above.
        ax_idx = fig.add_subplot(gs[2, :])
        if viz_data and theo_analysis.get('sequence_available', False):
            _idx_mz  = viz_data['mz_values']
            _idx_int = viz_data['intensity_values']
            _idx_vp  = viz_data['valid_peaks']

            _valid_mz  = _idx_mz[_idx_vp]
            _valid_int = _idx_int[_idx_vp]
            n_valid    = int(_idx_vp.sum())
            x_pos      = np.arange(n_valid)

            _ann_mask_idx = np.array(theo_analysis.get('annotated_mask',
                                                        np.zeros(n_valid, dtype=bool)))
            _anns_idx     = list(theo_analysis.get('theo_annotations', []))

            # Merge custom ion detections into index annotations (same as m/z panel)
            _idx_all_annotations = list(_anns_idx)  # copy to extend
            custom_ion_data = theo_analysis.get('custom_ion_data', {})
            custom_detection = custom_ion_data.get('detection', {})
            if custom_detection:
                for group_name, det in custom_detection.items():
                    if not det.get('found'):
                        continue
                    for matched_mz_val in det.get('matched_mz', []):
                        idx_match = int(np.argmin(np.abs(_valid_mz - matched_mz_val)))
                        if abs(_valid_mz[idx_match] - matched_mz_val) / max(matched_mz_val, 1e-12) * 1e6 < self.ppm_tol * 2:
                            if idx_match < len(_idx_all_annotations) and _idx_all_annotations[idx_match]:
                                continue  # already has a fragment annotation
                            _idx_all_annotations.extend([''] * (idx_match + 1 - len(_idx_all_annotations)))
                            _idx_all_annotations[idx_match] = f"custom:{group_name}@{matched_mz_val:.4f}"

            # Classify using same categorize_ion() and category_colors as the m/z panel
            _idx_categories: dict[int, str] = {}
            for i in range(n_valid):
                ann = _idx_all_annotations[i] if i < len(_idx_all_annotations) else ""
                _idx_categories[i] = categorize_ion(ann)

            # Group and plot (unannotated first, annotated on top)
            _idx_groups: dict[str, list[int]] = {}
            for i, cat in _idx_categories.items():
                _idx_groups.setdefault(cat, []).append(i)

            # Plot unannotated first
            if "unannotated" in _idx_groups:
                un_color, un_alpha = category_colors["unannotated"]
                arr = np.array(_idx_groups["unannotated"])
                ax_idx.bar(x_pos[arr], _valid_int[arr],
                           color=un_color, alpha=un_alpha,
                           width=1.0, linewidth=0, label="Unannotated")

            # Plot annotated categories — same colour + alpha as the m/z panel
            for cat in sorted(set(_idx_categories.values()) - {"unannotated"}):
                idxs = _idx_groups.get(cat, [])
                if not idxs:
                    continue
                color, alpha = category_colors.get(cat, ("#9467bd", 0.8))
                arr = np.array(idxs)
                ax_idx.bar(x_pos[arr], _valid_int[arr],
                           color=color, alpha=alpha,
                           width=1.0, linewidth=0, label=cat)

            # Label every annotated peak — uniform spacing means no m/z crowding
            for i in range(n_valid):
                ann = _idx_all_annotations[i] if i < len(_idx_all_annotations) else ""
                if ann and _idx_categories.get(i, "unannotated") != "unannotated":
                    display_ann = format_annotation_display(ann)
                    ax_idx.annotate(
                        display_ann, xy=(i, _valid_int[i]), xytext=(0, 4),
                        textcoords='offset points', ha='center', fontsize=7,
                        rotation=90, alpha=0.9, fontweight='bold',
                        color='black',
                    )

            # Label top-20 unannotated peaks with their m/z values so that
            # high-intensity unmatched peaks can be quickly identified.
            unannotated_idxs = _idx_groups.get("unannotated", [])
            if unannotated_idxs:
                unannotated_arr = np.array(unannotated_idxs)
                top_k = min(20, len(unannotated_arr))
                top_unannotated = unannotated_arr[
                    np.argsort(_valid_int[unannotated_arr])[-top_k:]
                ]
                for i in top_unannotated:
                    ax_idx.annotate(
                        f"{_valid_mz[i]:.2f}",
                        xy=(i, _valid_int[i]), xytext=(0, 4),
                        textcoords='offset points', ha='center', fontsize=6,
                        rotation=90, alpha=0.7, fontstyle='italic',
                        color='#555555',
                    )

            # X-tick labels: show m/z values at regular intervals for context
            tick_step = max(1, n_valid // 20)
            tick_pos  = np.arange(0, n_valid, tick_step)
            ax_idx.set_xticks(tick_pos)
            ax_idx.set_xticklabels(
                [f"{_valid_mz[i]:.0f}" for i in tick_pos], fontsize=8, rotation=45, ha='right'
            )
            ax_idx.set_xlim(-1, n_valid)
            _max_idx_int = _valid_int.max() if len(_valid_int) > 0 else 1.0
            ax_idx.set_ylim(0, _max_idx_int * 1.6)  # extra headroom for labels
            ax_idx.set_xlabel('m/z (at peak index)', fontsize=11, fontweight='bold')
            ax_idx.set_ylabel('Normalized Intensity', fontsize=11, fontweight='bold')
            ax_idx.set_title('All Annotations — Index-Based View',
                             fontsize=11, fontweight='bold')
            ax_idx.legend(fontsize=9, loc='upper right', ncol=2,
                          markerscale=0.7, borderpad=0.4, labelspacing=0.3,
                          handlelength=1.2, handletextpad=0.4)
            ax_idx.grid(True, alpha=0.3, axis='y')
        else:
            ax_idx.text(0.5, 0.5, "No theoretical data available", ha='center', va='center',
                        transform=ax_idx.transAxes, fontsize=12)
            ax_idx.set_title('Index-Based Spectrum')

        return fig

    def print_summary(self):
        """Print spectrum-level summary to console.

        Note: Theoretical matching stats, annotation labels, and fragmentation
        analysis are printed by TheoreticalAnalyser.print_summary().
        """
        summary = self.analysis_results["summary"]
        match_dist = self.analysis_results.get("match_distribution", {})

        print("\n" + "=" * 80)
        print("SPECTRUM ANALYSIS SUMMARY")
        print("=" * 80)
        print(f"Total spectra analyzed: {summary.get('total_spectra_analyzed', 0):,d}")
        print(f"Spectra with sequence: {summary.get('total_spectra_with_sequence', 0):,d}")
        print(f"Average peaks per spectrum: {summary.get('avg_peaks_per_spectrum', 0):.1f}")

        if match_dist:
            print(f"\nMatch Distribution (informational):")
            print(f"  Mean matched peaks: {match_dist.get('mean_matches', 0):.1f}")
            print(f"  Median matched peaks: {match_dist.get('median_matches', 0):.1f}")
            print(f"  Min/Max: {match_dist.get('min_matches', 0)}/{match_dist.get('max_matches', 0)}")
            print(f"  (Quality gate details in theoretical_analysis/ output)")

        custom_ion = self.analysis_results.get("custom_ion_analysis")
        if custom_ion:
            ion_rates = custom_ion.get("ion_hit_rates", {})
            top = [(g, d["hit_rate"]) for g, d in ion_rates.items() if d["hit_rate"] > 0]
            top.sort(key=lambda x: x[1], reverse=True)
            print(f"\nCustom Ion Detection (top hits):")
            for group, rate in top[:5]:
                print(f"  {group:<35s} {rate*100:5.1f}%")
            if len(top) > 5:
                print(f"  ... {len(top)} total groups detected  (see custom_ion_analysis.json)")

        print(f"\n(Theoretical analysis: see theoretical_analysis/ directory)")
        print(f"(Masking analysis: see masking_analysis/ directory)")
        print(f"\nResults saved to: {self.output_dir}")
        print("=" * 80)
