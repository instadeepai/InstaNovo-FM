#!/usr/bin/env python
"""
Comprehensive Data Analysis for Foundation Model Training.

This script orchestrates multiple analysis modules:
1. Metadata analysis (dataset columns and search data)
2. Spectrum analysis (signal-to-noise, theoretical matching)
3. Masking analysis (MLM strategy effectiveness)
4. Binning analysis (label noise measurement)

Usage:
    python -m instanovo_fm.scripts.data_analysis --config configs/foundational.yaml
"""

import gc
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import hydra
from omegaconf import DictConfig, OmegaConf

from instanovo.__init__ import console
from instanovo_fm.data import FoundationalDataProcessor
from instanovo_fm.data.search_data_manager import create_search_data_manager
from instanovo_fm.data.metadata_analyser import MetadataAnalyser
from instanovo_fm.data.spectrum_analyser import SpectrumAnalyser
from instanovo.utils.residues import ResidueSet
from instanovo.utils.data_handler import SpectrumDataFrame
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).parent.parent.parent / "configs"

# ============================================================================
# Task dependency graph (fixed, known)
# ============================================================================
TASK_DEPENDENCIES: Dict[str, Set[str]] = {
    "metadata": set(),
    "theoretical": set(),
    "custom_ions": {"theoretical"},
    "intensity": {"theoretical"},
    "masking": {"theoretical", "custom_ions"},
    "binning": {"theoretical"},
}

ALL_TASKS = list(TASK_DEPENDENCIES.keys())
PER_SPECTRUM_TASKS = {"theoretical", "custom_ions", "intensity", "masking"}
BATCH_TASKS = {"metadata", "binning"}

# Fixed execution order within the per-spectrum loop
PER_SPECTRUM_ORDER = ["theoretical", "custom_ions", "intensity", "masking"]

# ============================================================================
# Legacy config key mapping (old flat keys -> new task_configs)
# ============================================================================
_LEGACY_MAP: Dict[str, tuple] = {
    # (old_key, task, new_key)
    "ppm_tol": ("theoretical", "ppm_tol"),
    "theoretical_engine": ("theoretical", "theoretical_engine"),
    "keep_modifications": ("theoretical", "keep_modifications"),
    "use_conditional_annotation": ("theoretical", "use_conditional_annotation"),
    "add_precursor": ("theoretical", "add_precursor"),
    "add_losses": ("theoretical", "add_losses"),
    "loss_types": ("theoretical", "loss_types"),
    "add_isotopes": ("theoretical", "add_isotopes"),
    "isotope_intensity_threshold": ("theoretical", "isotope_intensity_threshold"),
    "max_isotope": ("theoretical", "max_isotope"),
    "min_backbone_coverage": ("theoretical", "min_backbone_coverage"),
    "min_fragment_groups": ("theoretical", "min_fragment_groups"),
    "enable_mass_error_analysis": ("theoretical", "enable_mass_error_analysis"),
    "enable_stratified_analysis": ("theoretical", "enable_stratified_analysis"),
    "enable_modification_analysis": ("theoretical", "enable_modification_analysis"),
    "mz_range_thresholds": ("theoretical", "mz_range_thresholds"),
    "theoretical_analysis": ("theoretical", "theoretical_analysis"),
    # Custom ions
    "custom_ions": ("custom_ions", "custom_ions"),
    # Masking
    "masking_analysis_n_repeats": ("masking", "n_repeats"),
    "masking_strategies": ("masking", "strategies"),
    # Intensity
    "intensity_bins": ("intensity", "intensity_bins"),
    "intensity_log_scale": ("intensity", "intensity_log_scale"),
    "compute_skewness_kurtosis": ("intensity", "compute_skewness_kurtosis"),
    "mz_range_boundaries": ("intensity", "mz_range_boundaries"),
    # Binning
    "min_ion_observations": ("binning", "min_ion_observations"),
    "bin_jump_threshold": ("binning", "bin_jump_threshold"),
    "enable_resolution_information": ("binning", "enable_resolution_information"),
    "resolution_info_window_size": ("binning", "resolution_info_window_size"),
    "resolution_info_min_peaks_per_window": ("binning", "resolution_info_min_peaks_per_window"),
    "enable_group_size_sensitivity": ("binning", "enable_group_size_sensitivity"),
    "group_size_candidates": ("binning", "group_size_candidates"),
    "group_size_max_offset_utilization": ("binning", "group_size_max_offset_utilization"),
    "group_size_min_offset_headroom": ("binning", "group_size_min_offset_headroom"),
    "group_size_max_n_groups": ("binning", "group_size_max_n_groups"),
    "enable_intra_collision_analysis": ("binning", "enable_intra_collision_analysis"),
    "intra_collision_weight": ("binning", "intra_collision_weight"),
    "binning_strategies": ("binning", "strategies"),
}

# Legacy enable_* booleans -> task name
# Legacy enable_* booleans -> (task name, default value)
_LEGACY_ENABLE_MAP: Dict[str, tuple] = {
    "enable_metadata_analysis": ("metadata", False),
    "enable_spectrum_analysis": ("theoretical", True),  # old catch-all for spectrum loop
    "enable_theoretical": ("theoretical", True),
    "enable_custom_ions": ("custom_ions", True),
    "enable_intensity_analysis": ("intensity", False),
    "enable_masking_analysis": ("masking", True),
    "enable_bin_jump_analysis": ("binning", True),
}


class DataAnalyzer:
    """
    Orchestrator for comprehensive data analysis.

    Coordinates:
    - Metadata analysis (MetadataAnalyser)
    - Spectrum analysis (SpectrumAnalyser)
    - Binning analysis (BinningAnalyser, via SpectrumAnalyser)
    - S3/AIChor upload
    """

    def __init__(self, config: DictConfig, output_dir: Optional[str] = None):
        """
        Initialize the data analyzer.

        Args:
            config: Hydra configuration
            output_dir: Output directory (defaults to foundational data_analysis_results)
        """
        self.config = config

        # Set output directory - default to foundational module directory
        if output_dir is None:
            foundational_dir = Path(__file__).parent.parent  # instanovo/foundational/
            self.output_dir = foundational_dir / "data" / "data_analysis_results"
        else:
            self.output_dir = Path(output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup residue set
        self.residue_set = ResidueSet(
            residue_masses=config.residues.get("residues"),
            residue_remapping=config.dataset.get("residue_remapping", None),
        )

        # Setup search data manager
        self.search_data_manager = create_search_data_manager({
            "use_search_data": config.dataset.get("use_search_data", False),
            "search_data_path": config.dataset.get("search_data_path", None),
            "search_data_filepath_column": config.dataset.get("search_data_filepath_column", "file path"),
            "search_data_spectrum_key": config.dataset.get("search_data_spectrum_key", "filepath"),
        })

        # Merge legacy config if needed, then resolve tasks
        self._analysis_config = self._build_analysis_config()
        self._tasks = self._resolve_tasks()

        # Lazy S3 handler
        self._s3 = None

        logger.info(f"Data analyzer initialized. Output directory: {self.output_dir}")
        logger.info(f"Active tasks: {', '.join(self._tasks)}")

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------

    def _build_analysis_config(self) -> Dict[str, Any]:
        """Build the analysis config dict, merging legacy flat keys if present."""
        raw = self.config.get("analysis", {})
        # Convert OmegaConf -> plain dict for easier manipulation
        if hasattr(raw, "items"):
            cfg = OmegaConf.to_container(raw, resolve=True) if hasattr(raw, "_metadata") else dict(raw)
        else:
            cfg = {}

        # If task_configs already exists, this is the new format
        if "task_configs" in cfg:
            return cfg

        # Otherwise, merge legacy flat keys into task_configs
        return self._merge_legacy_config(cfg)

    @staticmethod
    def _merge_legacy_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Map old flat enable_*/parameter keys to the new task_configs structure."""
        task_configs: Dict[str, Dict[str, Any]] = {t: {} for t in ALL_TASKS}

        # 1. Map flat parameter keys to their task section
        for old_key, (task, new_key) in _LEGACY_MAP.items():
            if old_key in cfg:
                task_configs[task][new_key] = cfg[old_key]

        # 2. Derive tasks_to_run from enable_* booleans
        tasks_to_run: Optional[List[str]] = None
        has_enable_keys = any(k in cfg for k in _LEGACY_ENABLE_MAP)
        if has_enable_keys:
            tasks_to_run = []
            for enable_key, (task_name, default) in _LEGACY_ENABLE_MAP.items():
                if cfg.get(enable_key, default):
                    if task_name not in tasks_to_run:
                        tasks_to_run.append(task_name)

        merged = {
            "tasks_to_run": tasks_to_run,
            "max_samples": cfg.get("max_samples", 10_000),
            "max_visualizations": cfg.get("max_visualizations", 30),
            "n_workers": cfg.get("n_workers", None),
            "output_dir": cfg.get("output_dir", None),
            "task_configs": task_configs,
        }
        return merged

    def _resolve_tasks(self) -> List[str]:
        """Read tasks_to_run from config, validate dependencies, auto-include missing."""
        requested = self._analysis_config.get("tasks_to_run", None)

        # null means all tasks
        if requested is None:
            return list(ALL_TASKS)

        requested = list(requested)
        resolved = set(requested)

        # Auto-include missing dependencies
        changed = True
        while changed:
            changed = False
            for task in list(resolved):
                for dep in TASK_DEPENDENCIES.get(task, set()):
                    if dep not in resolved:
                        logger.warning(
                            f"Task '{task}' depends on '{dep}' -- auto-including '{dep}'"
                        )
                        resolved.add(dep)
                        changed = True

        # Return in canonical order
        return [t for t in ALL_TASKS if t in resolved]

    def get_task_config(self, task: str) -> Dict[str, Any]:
        """Get the config dict for a specific task."""
        task_configs = self._analysis_config.get("task_configs", {})
        return dict(task_configs.get(task, {}))

    # ------------------------------------------------------------------
    # S3/AIChor support
    # ------------------------------------------------------------------

    @property
    def s3(self):
        """Lazy S3 handler — only instantiated when actually needed."""
        if self._s3 is None:
            from instanovo.utils.s3 import S3FileHandler
            self._s3 = S3FileHandler()
        return self._s3

    def _upload_results_to_s3(self) -> None:
        """Upload analysis results to S3 if running on AIChor.

        Uses parallel uploads with retry logic and writes an upload
        manifest so that partial failures are easy to diagnose.
        """
        import json
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        from instanovo.utils.s3 import S3FileHandler

        if not S3FileHandler._aichor_enabled():
            return

        logger.info("Uploading analysis results to S3...")

        output_files = [f for f in self.output_dir.rglob("*") if f.is_file()]
        logger.info(f"Found {len(output_files)} files to upload")

        max_retries = 3
        manifest: List[Dict[str, Any]] = []

        def _upload_one(local_file: Path) -> Dict[str, Any]:
            s3_path = S3FileHandler.convert_to_s3_output(str(local_file))
            entry: Dict[str, Any] = {
                "local_path": str(local_file),
                "s3_path": s3_path,
                "status": "failed",
            }
            last_err: Optional[Exception] = None
            for attempt in range(1, max_retries + 1):
                try:
                    self.s3.upload(str(local_file), s3_path)
                    entry["status"] = "success"
                    return entry
                except Exception as e:
                    last_err = e
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)  # exponential backoff
            entry["error"] = str(last_err)
            return entry

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_upload_one, f): f for f in output_files}
            for idx, future in enumerate(as_completed(futures), 1):
                entry = future.result()
                manifest.append(entry)
                name = Path(entry["local_path"]).name
                status = entry["status"]
                logger.info(f"[{idx}/{len(output_files)}] {status}: {name}")

        succeeded = sum(1 for e in manifest if e["status"] == "success")
        failed = len(manifest) - succeeded

        # Write manifest locally (always available even if S3 upload fails)
        manifest_path = self.output_dir / "upload_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        # Best-effort upload of manifest itself
        try:
            s3_manifest = S3FileHandler.convert_to_s3_output(str(manifest_path))
            self.s3.upload(str(manifest_path), s3_manifest)
        except Exception:
            logger.warning("Could not upload upload_manifest.json to S3")

        logger.info(
            f"Upload complete: {succeeded} succeeded, {failed} failed "
            f"out of {len(output_files)} files"
        )
        if failed:
            for entry in manifest:
                if entry["status"] == "failed":
                    logger.warning(
                        f"  FAILED: {Path(entry['local_path']).name} — {entry.get('error', 'unknown')}"
                    )

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def load_dataset(self) -> SpectrumDataFrame:
        """Load the training dataset for analysis."""
        dataset_config = self.config.get("dataset", {})
        train_path = dataset_config.get("train_path")

        if not train_path:
            raise ValueError("No training dataset path specified in config")

        logger.info(f"Loading training dataset from: {train_path}")

        sdf = SpectrumDataFrame.load(
            source=train_path,
            source_type=dataset_config.get("source_type", "default"),
            lazy=dataset_config.get("lazy_loading", True),
            is_annotated=True,
            shuffle=False,
            partition=dataset_config.get("train_partition", None),
            column_mapping=dataset_config.get("column_remapping", None),
            max_shard_size=dataset_config.get("max_shard_size", 100_000),
            verbose=dataset_config.get("verbose_loading", True),
        )

        logger.info(f"Loaded {len(sdf):,d} spectra for analysis")
        return sdf

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_analysis(self):
        """Run comprehensive data analysis."""
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE DATA ANALYSIS")
        logger.info("=" * 80)

        # Load dataset
        sdf = self.load_dataset()

        # Determine number of samples
        max_samples = self._analysis_config.get("max_samples", 100000)

        if max_samples is None:
            n_samples = len(sdf)
            logger.info(f"Will analyze all {n_samples:,d} spectra")
        else:
            n_samples = min(max_samples, len(sdf))
            logger.info(f"Will analyze {n_samples:,d} out of {len(sdf):,d} spectra")

        # Set up data processor
        processor = self._setup_data_processor()

        tasks = self._tasks

        # =================================================================
        # 1. Metadata (independent batch task)
        # =================================================================
        if "metadata" in tasks:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 1: METADATA ANALYSIS")
            logger.info("=" * 80)
            self._run_metadata_analysis(sdf, n_samples, processor)

        # =================================================================
        # 2. Per-spectrum analysis + binning (batch post-aggregation)
        #    SpectrumAnalyser handles both the per-spectrum loop AND the
        #    binning aggregation step, so it needs the full set of resolved
        #    tasks (excluding metadata, which is handled above).
        # =================================================================
        spectrum_tasks = [t for t in tasks if t != "metadata"]
        if spectrum_tasks:
            logger.info("\n" + "=" * 80)
            logger.info("PHASE 2: SPECTRUM & BINNING ANALYSIS")
            logger.info(f"  Active tasks: {', '.join(spectrum_tasks)}")
            logger.info("=" * 80)
            self._run_spectrum_analysis(sdf, spectrum_tasks)

        # =================================================================
        # 4. S3 upload
        # =================================================================
        self._upload_results_to_s3()

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("COMPREHENSIVE DATA ANALYSIS COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nAll results saved to: {self.output_dir}")
        logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _run_metadata_analysis(
        self,
        sdf: SpectrumDataFrame,
        n_samples: int,
        processor: FoundationalDataProcessor,
    ):
        """Run metadata analysis phase."""
        # Collect all metadata (dataset + search data) in single dict
        metadata_columns = processor.metadata_columns
        search_columns = (
            processor.search_data_manager.get_available_columns()
            if processor.search_data_manager and processor.search_data_manager.is_loaded
            else []
        )

        all_columns = list(metadata_columns) + [f"search_{col}" for col in search_columns]

        # Add calculated metadata fields that are always available from data processor
        calculated_fields = ["precursor_mass", "precursor_mz", "precursor_charge"]
        all_columns.extend([f for f in calculated_fields if f not in all_columns])

        metadata_accumulator: Dict[str, List[Any]] = {col: [] for col in all_columns}

        batch_size = 32
        total_batches = (n_samples + batch_size - 1) // batch_size
        log_interval = max(total_batches // 10, 1)  # Log ~10 times

        logger.info(f"Collecting metadata from {n_samples:,d} samples...")
        for batch_idx, batch_start in enumerate(range(0, n_samples, batch_size)):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_data = [sdf[i] for i in range(batch_start, batch_end)]
            processed_batch = [processor.process_row(row) for row in batch_data]
            collated = processor._collate_batch(processed_batch, apply_masking=False)

            for col in all_columns:
                value = collated.get(col)
                if value is not None and hasattr(value, "tolist"):
                    metadata_accumulator[col].extend(value.tolist())
                elif value is not None and isinstance(value, list):
                    metadata_accumulator[col].extend(value)
                else:
                    metadata_accumulator[col].extend([None] * len(processed_batch))

            if (batch_idx + 1) % log_interval == 0:
                logger.info(
                    f"  Metadata collection: {batch_end:,d}/{n_samples:,d} spectra"
                )

            # Free temporaries each batch
            del batch_data, processed_batch, collated

        gc.collect()

        # Run metadata analysis
        if not metadata_accumulator or not any(len(v) > 0 for v in metadata_accumulator.values()):
            logger.info("No metadata to analyze")
            return

        metadata_analyzer = MetadataAnalyser(self.output_dir)
        metadata_analyzer.analyze_metadata(metadata_accumulator, list(metadata_accumulator.keys()))

        # Free the accumulator before visualization (DataFrame is now inside MetadataAnalyser)
        del metadata_accumulator
        gc.collect()

        metadata_analyzer.generate_visualizations()
        metadata_analyzer.save_results()
        metadata_analyzer.print_summary()

    def _run_spectrum_analysis(
        self,
        sdf: SpectrumDataFrame,
        active_tasks: List[str],
    ):
        """Run per-spectrum analysis phase via SpectrumAnalyser."""
        spectrum_analyzer = SpectrumAnalyser(
            self.config,
            output_dir=str(self.output_dir / "spectra_analysis"),
            active_tasks=set(active_tasks),
        )
        spectrum_analyzer.run_analysis(sdf)
        spectrum_analyzer.print_summary()

    # ------------------------------------------------------------------
    # Data processor setup (unchanged)
    # ------------------------------------------------------------------

    def _setup_data_processor(self) -> FoundationalDataProcessor:
        """Set up data processor (same configuration as train.py)."""
        masking_config = self.config.model.get("masking", {})

        return FoundationalDataProcessor(
            n_peaks=self.config.model.get("n_peaks", 200),
            min_mz=self.config.model.get("min_mz", 50.0),
            max_mz=self.config.model.get("max_mz", 2500.0),
            min_intensity=self.config.model.get("min_intensity", 0.01),
            remove_precursor_tol=self.config.model.get("remove_precursor_tol", 0.0),
            use_spectrum_utils=self.config.model.get("use_spectrum_utils", False),
            normalize_mz=self.config.model.get("normalize_mz", True),
            peak_ordering=masking_config.get("ordering_strategy", "sorted"),
            residue_set=self.residue_set,
            annotated=True,
            return_str=True,
            metadata_columns=self.config.dataset.get("metadata_columns", []),
            masking_strategy=masking_config.get("strategy"),
            mask_portion=masking_config.get("mask_portion", 0.30),
            thompson_alpha=masking_config.get("alpha", 0.5),
            thompson_beta=masking_config.get("beta", 0.5),
            thompson_kappa=masking_config.get("kappa", 4.0),
            thompson_gamma=masking_config.get("gamma", 0.7),
            span_min=masking_config.get("span_min", 4),
            span_max=masking_config.get("span_max", 7),
            span_bidirectional=masking_config.get("bidirectional", True),
            include_isotopes=masking_config.get("include_isotopes", False),
            isotope_ppm=masking_config.get("isotope_ppm", 10.0),
            isotope_da_floor=masking_config.get("isotope_da_floor", 0.015),
            isotope_max_charge=masking_config.get("isotope_max_charge", 4),
            isotope_max_order=masking_config.get("isotope_max_order", 3),
            max_total_mask_ratio=masking_config.get("max_total_mask_ratio", 0.35),
            signal_min_backbone_coverage=masking_config.get("signal_min_backbone_coverage", 0.15),
            signal_min_fragment_groups=masking_config.get("signal_min_fragment_groups", 3),
            signal_ppm=masking_config.get("signal_ppm", 20.0),
            signal_cid_da_tol=masking_config.get("signal_cid_da_tol", 0.2),
            signal_ion_types=tuple(masking_config.get("signal_ion_types", ["b", "y"])),
            signal_num_workers=masking_config.get("signal_num_workers", 4),
            search_data_manager=self.search_data_manager,
        )


@hydra.main(config_path=str(CONFIG_PATH), version_base=None, config_name="foundational")
def main(cfg: DictConfig):
    """Main data analysis entry point.

    Usage:
        python -m instanovo_fm.scripts.data_analysis

    Or with config overrides:
        python -m instanovo_fm.scripts.data_analysis \
            analysis.max_samples=1000 \
            analysis.max_visualizations=50
    """
    analyzer = DataAnalyzer(cfg)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
