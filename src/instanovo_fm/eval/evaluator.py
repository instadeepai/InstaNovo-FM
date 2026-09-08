"""EmbeddingEvaluator class for comprehensive evaluation of foundation model embeddings.

This module provides the main orchestration class for:
1. Loading trained foundation models
2. Generating embeddings on validation/test datasets
3. Running evaluation tasks (retrieval, clustering, linear probes, etc.)
4. Aggregating and saving results
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import torch
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo_fm.data import FoundationalDataProcessor
from instanovo_fm.eval import embedding_io
from instanovo_fm.eval.embed_eval_tasks import TASK_REGISTRY, get_task
from instanovo_fm.model import FoundationModel
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.s3 import S3FileHandler

logger = ColorLog(console, __name__).logger


def _collect_numeric_stats(dicts: list) -> dict:
    """Recursively collect mean/std for all numeric leaves across a list of dicts.

    For each numeric leaf value found at the same path in all dicts, computes
    mean and std and stores them alongside the per-seed values.
    """
    if not dicts:
        return {}

    result = {}
    template = dicts[0]

    for key, value in template.items():
        if key in ("per_seed", "seeds", "n_seeds", "execution_time", "output_dir"):
            continue

        if isinstance(value, dict):
            sub_dicts = [d.get(key, {}) for d in dicts if isinstance(d.get(key), dict)]
            if sub_dicts:
                result[key] = _collect_numeric_stats(sub_dicts)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            vals = []
            for d in dicts:
                v = d.get(key)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if v == v:  # Skip NaN (NaN != NaN)
                        vals.append(float(v))
            if vals:
                mean_val = sum(vals) / len(vals)
                if len(vals) > 1:
                    variance = sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)
                    std_val = variance ** 0.5
                else:
                    std_val = 0.0
                result[key] = mean_val
                result[f"{key}_std"] = std_val
                result[f"{key}_per_seed"] = vals
            else:
                result[key] = value
        elif isinstance(value, list):
            result[key] = value
        else:
            result[key] = value

    return result


class EmbeddingEvaluator:
    """Evaluator for foundation model embeddings.

    This class handles the complete evaluation workflow:
    1. Load model from checkpoint
    2. Setup data loaders
    3. Generate and export embeddings
    4. Run evaluation tasks
    5. Aggregate and save results

    Similar to FoundationalTrainer, but focused on evaluation rather than training.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the embedding evaluator.

        Args:
            config: Hydra configuration with evaluation settings.
        """
        self.config = config
        self.eval_config = config.get("evaluation", {})

        # Lazy S3 handler - only instantiated when actually needed
        self._s3: Optional[S3FileHandler] = None

        # Setup checkpoint path from evaluation config (optional - can be provided externally during training)
        self.checkpoint_path = self.eval_config.get("checkpoint_path", None)

        # Setup base output directory
        base_output_dir = Path(self.eval_config.get("output_dir", "./evaluation_results"))

        # Use provided output_dir directly if it's already specific (e.g., step_XXXXXX
        # from training, or eval_YYYYMMDD_HHMMSS/ckpt_id from multi-checkpoint mode).
        # Otherwise create timestamped folder for standalone evaluation.
        output_dir_str = str(base_output_dir)
        if "step_" in output_dir_str or "eval_" in output_dir_str:
            # Already has specific identifier — use as-is
            self.output_dir = base_output_dir
        else:
            # Standalone evaluation - create unique timestamped folder
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = base_output_dir / f"eval_{timestamp}"

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        device_str = self.eval_config.get("device", "auto")
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        logger.info(f"Evaluator: device={self.device}")

        # Lazy initialization (done in evaluate() or set externally during training)
        self.model: Optional[FoundationModel] = None
        self.model_config: Optional[DictConfig] = None
        self.data_processor: Optional[FoundationalDataProcessor] = None
        self.dataloader: Optional[torch.utils.data.DataLoader] = None

    @property
    def s3(self) -> S3FileHandler:
        """Lazy S3 handler - only instantiated when actually needed."""
        if self._s3 is None:
            self._s3 = S3FileHandler()
        return self._s3

    def load_model(self) -> Tuple[FoundationModel, DictConfig]:
        """Load model from checkpoint.

        Returns:
            Tuple of (model, config)

        Raises:
            ValueError: If no checkpoint path is available
        """
        if self.checkpoint_path is None:
            raise ValueError(
                "No checkpoint path available. Either:\n"
                "1. Set 'evaluation.checkpoint_path' in config for standalone evaluation, or\n"
                "2. Set evaluator.model directly when using during training"
            )

        logger.info(f"Loading model from {self.checkpoint_path}")

        # Download checkpoint from S3 if needed (Aichor support)
        checkpoint_path = self.checkpoint_path
        if checkpoint_path.startswith("s3://"):
            logger.info(f"Downloading checkpoint from S3: {checkpoint_path}")
            checkpoint_path = self.s3.get_local_path(checkpoint_path)
            logger.info(f"Downloaded to local path: {checkpoint_path}")

        # Use the model's load method (same as in encoder.py)
        model, model_config = FoundationModel.load(checkpoint_path)

        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()

        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded successfully ({num_params:,} parameters)")

        return model, model_config

    def setup_data_processor(self) -> FoundationalDataProcessor:
        """Setup data processor for the validation/test split.

        Returns:
            Data processor configured for evaluation.
        """
        assert self.model_config is not None, "Model must be loaded first"

        # Setup residue set (same as trainer)
        from instanovo.utils.residues import ResidueSet
        residue_set = ResidueSet(
            residue_masses=self.config.residues.get("residues"),
            residue_remapping=self.config.dataset.get("residue_remapping", None),
        )

        # Check if meta token is enabled (needed for filtering metadata columns)
        meta_token_enabled = self.model_config.get("meta_token", {}).get("enabled", False)

        # Extract metadata columns from dataset config (same as trainer)
        metadata_columns = self.config.dataset.get("metadata_columns", None)

        # Setup search data manager if enabled (same as trainer)
        from instanovo_fm.data.search_data_manager import create_search_data_manager
        search_data_config = {
            "use_search_data": self.config.dataset.get("use_search_data", False),
            "search_data_path": self.config.dataset.get("search_data_path", None),
            "search_data_filepath_column": self.config.dataset.get("search_data_filepath_column", "file path"),
            "search_data_spectrum_key": self.config.dataset.get("search_data_spectrum_key", "filepath"),
        }
        search_data_manager = create_search_data_manager(search_data_config)


        # Get masking config (same as trainer)
        masking_cfg = self.model_config.get("masking", {})

        # Create processor (same as trainer validation processor)
        processor = FoundationalDataProcessor(
            n_peaks=self.model_config.get("n_peaks", 200),
            min_mz=self.model_config.get("min_mz", 50.0),
            max_mz=self.model_config.get("max_mz", 2500.0),
            min_intensity=self.model_config.get("min_intensity", 0.01),
            remove_precursor_tol=0.0,
            use_spectrum_utils=self.model_config.get("use_spectrum_utils", False),
            normalize_mz=self.model_config.get("normalize_mz", True),
            peak_ordering=masking_cfg.get("ordering_strategy", self.model_config.get("peak_ordering", "sorted")),
            residue_set=residue_set,
            annotated=True,  # Include sequences for evaluation
            return_str=True,  # Keep sequences as strings
            metadata_columns=metadata_columns,  # Use metadata columns from config
            # Masking configuration (same as trainer)
            masking_strategy="none", # Search data integration
            search_data_manager=search_data_manager,
            build_metadata=meta_token_enabled
        )

        return processor

    def setup_dataloader(
        self,
        split: str = "valid",
        filter_usis: Optional[Set[str]] = None,
        override_max_samples: Optional[int] = None,
    ) -> torch.utils.data.DataLoader:
        """Setup dataloader for the specified split.

        Args:
            split: Dataset split to use ("train", "valid", or "test")
            filter_usis: Optional set of USIs to keep (project pre-filter).
                Applied to the SpectrumDataFrame before to_dataset(in_memory=True).
            override_max_samples: Override the global max_samples for this split.

        Returns:
            DataLoader for the specified split.
        """
        assert self.data_processor is not None, "Data processor must be setup first"

        # Get dataset configuration
        dataset_config = self.config.get("dataset", {})

        # Determine which split to use
        split_key = f"{split}_path"
        if split_key not in dataset_config:
            raise ValueError(
                f"Dataset split '{split}' not found in config. "
                f"Available keys: {list(dataset_config.keys())}"
            )

        # Create dataset using SpectrumDataFrame
        from instanovo.utils.data_handler import SpectrumDataFrame

        dataset_path = dataset_config[split_key]

        # Load dataset using SpectrumDataFrame.load (same as trainer)
        dataset = SpectrumDataFrame.load(
            source=dataset_path,
            source_type=dataset_config.get("source_type", "default"),
            lazy=dataset_config.get("lazy_loading", True),
            is_annotated=True,  # Include sequences for evaluation
            shuffle=False,  # No shuffling for evaluation
            partition=None,
            column_mapping=dataset_config.get("column_remapping", None),
            max_shard_size=dataset_config.get("max_shard_size", 100_000),
            add_source_file_column=True,  # synthesize source_file to match training schema; set_format requires it when listed in metadata_columns
            add_spectrum_id=dataset_config.get("add_spectrum_id", False),  # synthesize spectrum_id (experiment_name:scan_number) when usi is unavailable
            preshuffle_across_shards=False,
            verbose=dataset_config.get("verbose_loading", True),
        )

        # Column the enrichment/probe splits are keyed on (default usi; datasets whose
        # usi is null, e.g. massivekb_splits, key on the synthesised spectrum_id).
        spectrum_key = dataset_config.get("spectrum_key", "usi")

        # Apply project-disjoint filter before loading into memory.
        # filter_by_value uses Polars is_in (vectorised) instead of map_elements.
        if filter_usis is not None:
            dataset.filter_by_value(spectrum_key, filter_usis)
            logger.info(f"Applied project filter on '{spectrum_key}': {len(dataset):,} samples remain")

        # Determine the sample cap for this split.
        max_samples = override_max_samples if override_max_samples is not None else self.eval_config.get("max_samples", None)

        # Convert SpectrumDataFrame to HuggingFace Dataset by reading one parquet
        # file at a time. collect_chunked applies the per-file filter masks set by
        # filter_by_value and stops as soon as max_samples rows are collected, so
        # only the first few files are ever opened. This avoids the OOM that occurs
        # when sample_subset spreads a sparse mask across thousands of files and
        # collect_chunked then has to open every file to harvest a handful of rows.
        total = len(dataset)
        if max_samples is not None and max_samples < total:
            logger.info(f"Will collect up to {max_samples:,} samples (from {total:,}) via chunked file reads")
        from datasets import Dataset as HFDataset
        random_state = self.eval_config.get("random_state", 42)
        df = dataset.collect_chunked(max_samples=max_samples, seed=random_state)
        dataset = HFDataset.from_pandas(df.to_pandas())

        logger.info(f"Loaded {len(dataset)} samples from dataset")

        # Add prediction_id column to dataset (same as trainer)
        import numpy as np
        from datasets import Value
        dataset = dataset.add_column("prediction_id", np.arange(len(dataset)), feature=Value("int32"))

        # Add prediction_id column to processor (same as trainer)
        self.data_processor.add_metadata_columns(["prediction_id"])

        # Keep non-tensor metadata (strings, lists with Nones) so that
        # embedding_io.generate() receives frag_type, sequence, search_instrument,
        # collision_energy, etc. for downstream evaluation tasks.
        # Safe here because evaluation always uses a single process — the
        # multi-GPU dispatch_batches broadcast that requires pure-tensor batches
        # only applies to training.
        self.data_processor._keep_non_tensor_metadata = True

        # Process dataset using process_dataset which handles format conversion
        # This is the same as trainer validation data processing
        dataset = self.data_processor.process_dataset(dataset, return_format="torch")

        # Create dataloader
        batch_size = self.eval_config.get("batch_size", 256)
        num_workers = self.config.get("num_workers", 4)

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=self.data_processor.collate_fn,
        )

        return dataloader

    def generate_embeddings(
        self,
        force_regenerate: bool = False,
        override_max_samples: Optional[int] = None,
        split: Optional[str] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], Any]:
        """Generate embeddings for the validation/test set.

        This method handles the complete embedding generation workflow:
        1. Check if embeddings should be cached (if save_embeddings=True)
        2. Generate embeddings using the model
        3. Optionally save to disk for future use

        Args:
            force_regenerate: If True, regenerate embeddings even if cached.
            override_max_samples: If set, overrides the global max_samples from
                eval config for this call. Used by multi-split generation to give
                the train split a larger budget (e.g. 100k) than the global
                max_samples (e.g. 10k) used for single-split evaluation tasks.
            split: Name of the split being generated. The cache is keyed by it, so
                that a multi-split run does not read one split's embeddings back
                for another. Omit for single-split evaluation, which caches
                directly in ``output_dir``.

        Returns:
            Tuple of (embeddings, metadata, faiss_index)
        """
        assert self.model is not None, "Model must be loaded first"
        assert self.dataloader is not None, "DataLoader must be setup first"

        # Check if we should save embeddings to disk
        save_embeddings = self.eval_config.get("save_embeddings", False)

        # The cache lives in output_dir for a single-split run, and in a per-split
        # subdirectory otherwise. Without the split in the path, train embeddings
        # would be written first and then loaded back as valid and test.
        cache_dir = self.output_dir if split is None else self.output_dir / f"embeddings_{split}"

        # Check if cached embeddings exist (only relevant when saving is enabled)
        if save_embeddings and not force_regenerate:
            embeddings_path = cache_dir / "embeddings.h5"
            index_path = cache_dir / "index.faiss"

            if embeddings_path.exists() and index_path.exists():
                # Validate cached embeddings match requested pooling strategy
                requested_pooling = self.eval_config.get("embedding_pooling", "cls")
                try:
                    import h5py
                    with h5py.File(embeddings_path, 'r') as f:
                        cached_pooling = f.attrs.get('embedding_pooling', 'unknown')
                    if cached_pooling != 'unknown' and cached_pooling != requested_pooling:
                        logger.warning(
                            f"Cached embeddings use pooling='{cached_pooling}' but "
                            f"config requests '{requested_pooling}'. Regenerating."
                        )
                    else:
                        logger.info(f"Found cached embeddings (pooling={cached_pooling}), loading from disk")
                        return embedding_io.load(cache_dir)
                except Exception:
                    logger.info("Found cached embeddings, loading from disk")
                    return embedding_io.load(cache_dir)

        # Get max_samples limit — per-split override takes precedence over global config
        max_samples = override_max_samples if override_max_samples is not None else self.eval_config.get("max_samples", None)

        # Generate embeddings (with optional saving)
        save_str = ", saving to disk" if save_embeddings else ""
        limit_str = f", max_samples={max_samples}" if max_samples else ""
        logger.info(f"Generating embeddings{save_str}{limit_str}")

        # Check if we should compute confidence scores
        compute_confidence = self.eval_config.get("compute_spectrum_confidence", False)

        # Check if any task requires per-peak confidence, spectra storage, or peak embeddings
        store_per_peak_confidence = False
        store_peak_embeddings = False
        store_pretransformer_embeddings = False
        tasks_to_run = self.eval_config.get("tasks_to_run", None)
        if tasks_to_run:
            # Check which tasks need per-peak data
            for task_name in tasks_to_run:
                if "confidence" in task_name.lower() and "signal" in task_name.lower():
                    store_per_peak_confidence = True
                if "xai" in task_name.lower() or "igattribution" in task_name.lower() or "ig_attribution" in task_name.lower():
                    store_per_peak_confidence = True
                    store_peak_embeddings = True
                if "headanalysis" in task_name.lower():
                    store_per_peak_confidence = True
                if "peaktype" in task_name.lower() or "peak_type" in task_name.lower():
                    store_peak_embeddings = True
                    store_pretransformer_embeddings = True
                    store_per_peak_confidence = True
                if "theoretical" in task_name.lower() and "peak" in task_name.lower():
                    store_per_peak_confidence = True
                if "umap" in task_name.lower():
                    store_per_peak_confidence = True

        # Check if we should generate theoretical spectra
        theoretical_config = self.eval_config.get("theoretical_spectrum", {})
        generate_theoretical = theoretical_config.get("enabled", False)


        embedding_pooling = self.eval_config.get("embedding_pooling", "cls")

        embeddings, metadata, faiss_index = embedding_io.generate(
            model=self.model,
            dataloader=self.dataloader,
            device=self.device,
            batch_size=self.eval_config.get("batch_size", 256),
            show_progress=True,
            save_to=cache_dir if save_embeddings else None,
            max_samples=max_samples,
            compute_confidence=compute_confidence,
            store_per_peak_confidence=store_per_peak_confidence,
            generate_theoretical=generate_theoretical,
            theoretical_config=theoretical_config,
            store_peak_embeddings=store_peak_embeddings,
            store_pretransformer_embeddings=store_pretransformer_embeddings,
            embedding_pooling=embedding_pooling,
            confidence_temperature=self.eval_config.get("confidence_temperature", 1.0),
        )

        return embeddings, metadata, faiss_index

    def compute_sequence_similarity_clustering(
        self,
        metadata: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Compute sequence similarity clustering using MMseqs2 and inject into metadata.

        This method extracts peptide sequences from metadata, performs sequence similarity
        clustering using MMseqs2, and adds the cluster IDs back into metadata as 'seq_cluster_id'.

        The clustering is configurable via evaluation config:
        - enable_sequence_clustering: Whether to run clustering (default: False)
        - sequence_identity_threshold: Identity threshold for MMseqs2 (default: 0.7)
        - sequence_key: Metadata key containing sequences (default: 'sequence' or 'peptides')

        Args:
            metadata: Metadata dictionary to enrich with clustering information

        Returns:
            Updated metadata dictionary with 'seq_cluster_id' field added
        """
        from instanovo_fm.utils.mmseq2 import MMseqs2

        # Check if clustering is enabled
        clustering_config = self.eval_config.get("sequence_clustering", {})
        enable_clustering = clustering_config.get("enable", False)

        if not enable_clustering:
            return metadata

        logger.info("Computing sequence similarity clustering with MMseqs2...")

        # Get configuration parameters
        identity_threshold = clustering_config.get("identity_threshold", 0.7)
        sequence_key = clustering_config.get("sequence_key", None)

        # Try to find sequence data in metadata
        if sequence_key is None:
            # Auto-detect sequence key
            possible_keys = ['sequence', 'peptides', 'peptide', 'seq']
            sequence_key = None
            for key in possible_keys:
                if key in metadata:
                    sequence_key = key
                    break

            if sequence_key is None:
                logger.warning(
                    "No sequence data found in metadata. "
                    f"Looked for keys: {possible_keys}. "
                    "Skipping sequence clustering."
                )
                return metadata

        if sequence_key not in metadata:
            logger.warning(
                f"Sequence key '{sequence_key}' not found in metadata. "
                "Skipping sequence clustering."
            )
            return metadata

        # Extract sequences
        sequences = metadata[sequence_key]

        # Convert to list if numpy array (and decode if bytes)
        if isinstance(sequences, np.ndarray):
            if sequences.dtype.kind == 'S' or sequences.dtype.kind == 'O':
                # Byte strings or objects - need to decode
                sequences_list = []
                for seq in sequences:
                    if isinstance(seq, bytes):
                        sequences_list.append(seq.decode('utf-8'))
                    elif isinstance(seq, str):
                        sequences_list.append(seq)
                    else:
                        sequences_list.append(str(seq))
                sequences = sequences_list
            else:
                sequences = sequences.tolist()

        # Filter out None, empty, or invalid sequences
        valid_sequences = []
        valid_indices = []
        for idx, seq in enumerate(sequences):
            if seq is not None and isinstance(seq, str) and len(seq) > 0:
                # Basic validation: check if it looks like a peptide sequence
                if all(c.isalpha() or c == '(' or c == ')' or c == '[' or c == ']' for c in seq):
                    valid_sequences.append(seq)
                    valid_indices.append(idx)

        if len(valid_sequences) == 0:
            logger.warning("No valid sequences found for clustering. Skipping sequence clustering.")
            return metadata

        logger.info(f"Running MMseqs2 clustering on {len(valid_sequences)} sequences (identity_threshold={identity_threshold})")

        try:
            # Run MMseqs2 clustering
            mmseqs = MMseqs2(
                sequences=valid_sequences,
                identity_threshold=identity_threshold,
                remove_tmp=True,
                remove_output=True,
            )

            cluster_ids = mmseqs.run()

            if cluster_ids is None or len(cluster_ids) == 0:
                logger.warning("MMseqs2 clustering returned no results")
                return metadata

            # Create full cluster ID array (with -1 for invalid sequences)
            full_cluster_ids = np.full(len(sequences), -1, dtype=np.int32)
            for valid_idx, cluster_id in zip(valid_indices, cluster_ids):
                if cluster_id is not None:
                    full_cluster_ids[valid_idx] = cluster_id

            # Add to metadata
            metadata['seq_cluster_id'] = full_cluster_ids

            # Log statistics
            num_clusters = len(set(cluster_ids) - {None})
            num_clustered = np.sum(full_cluster_ids >= 0)
            logger.info(f"Sequence clustering complete:")
            logger.info(f"  - {num_clusters} unique clusters identified")
            logger.info(f"  - {num_clustered}/{len(sequences)} sequences successfully clustered")
            logger.info(f"  - Cluster IDs stored in metadata['seq_cluster_id']")

        except Exception as e:
            logger.error(f"MMseqs2 clustering failed: {e}")
            logger.warning("Continuing without sequence clustering")

        return metadata

    def get_metrics_for_logging(
        self,
        results: Dict[str, Any],
        embeddings_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract metrics from evaluation results for logging to TensorBoard/Neptune.

        This method processes task results and extracts loggable metrics by calling
        each task's get_loggable_metrics() method. It returns a flat dictionary
        with prefixed metric names ready for logging.

        Args:
            results: Dictionary of task results from run_evaluation_tasks()
            embeddings_info: Dictionary of embedding statistics

        Returns:
            Flat dictionary mapping metric names to scalar values.
            Keys are prefixed with "embed/{task_name}/" for organization.

        Example output:
            {
                "embed/num_embeddings": 2500,
                "embed/mean_norm": 1.0023,
                "embed/duplicateretrievaltask/recall@1": 0.85,
                "embed/duplicateretrievaltask/recall@5": 0.92,
                "embed/duplicateretrievaltask/map@10": 0.78,
                "embed/chargelinearprobetask/accuracy": 0.91,
            }
        """
        loggable_metrics = {}

        # Add primary embedding statistics (without task prefix), if available
        if embeddings_info:
            loggable_metrics["embed/num_embeddings"] = float(embeddings_info["num_embeddings"])
            loggable_metrics["embed/mean_norm"] = float(embeddings_info["mean_norm"])
            loggable_metrics["embed/std_norm"] = float(embeddings_info["std_norm"])

        # Process each task's results
        for result_key, task_results in results.items():
            if "error" in task_results:
                continue  # Skip failed tasks

            # Handle strategy-prefixed keys (e.g. "cls/duplicateretrievaltask")
            if "/" in result_key:
                strategy_prefix, task_name = result_key.split("/", 1)
            else:
                strategy_prefix, task_name = None, result_key

            # Get the task class to call its get_loggable_metrics() method
            try:
                task_class = get_task(task_name)
                task_instance = task_class()  # Create temporary instance

                # Extract loggable metrics from task results
                task_metrics = task_instance.get_loggable_metrics(task_results)

                # Build metric key: embed/{strategy}/{task}/{metric} or embed/{task}/{metric}
                # Sanitize: MLflow rejects metric names containing '@'
                for metric_name, metric_value in task_metrics.items():
                    safe_name = metric_name.replace("@", "_at_")
                    if strategy_prefix:
                        prefixed_name = f"embed/{strategy_prefix}/{task_name}/{safe_name}"
                    else:
                        prefixed_name = f"embed/{task_name}/{safe_name}"
                    loggable_metrics[prefixed_name] = float(metric_value)

            except Exception as e:
                logger.warning(f"Failed to extract loggable metrics from {result_key}: {e}")
                continue

        return loggable_metrics

    def _all_tasks_require_multi_split(self, tasks_to_run: list) -> bool:
        """Return True only if every task in the list has requires_multi_split=True."""
        if not tasks_to_run:
            return False
        for task_name in tasks_to_run:
            try:
                task_class = get_task(task_name)
                if not getattr(task_class, 'requires_multi_split', False):
                    return False
            except KeyError:
                return False
        return True

    @staticmethod
    def _extract_project_from_filepath(filepath: str) -> str:
        """Extract project ID from a filepath using regex patterns.

        Looks for standard proteomics repository identifiers (MSV, PXD) as
        directory components in the filepath.  Returns empty string if no
        match is found.
        """
        if not filepath:
            return ""
        normalized = filepath.replace("\\", "/")
        match = re.search(r"/(MSV\d+|PXD\d+)/", normalized)
        return match.group(1) if match else ""

    @staticmethod
    def _extract_project_from_usi(usi: str) -> str:
        """Extract project ID from a USI string.

        USI format: mzspec:PROJECT:filename:scan:index[:peptide]
        Returns the PROJECT component (e.g. 'PXD000001', 'MSV000001').
        Returns empty string if the USI is missing or malformed.
        """
        if not usi:
            return ""
        parts = usi.split(":")
        return parts[1].strip() if len(parts) >= 2 else ""

    def _evaluate_multi_seed(
        self,
        tasks_to_run_list: list,
        random_states: list,
        force_regenerate: bool,
        lp_config: dict,
    ) -> Dict[str, Any]:
        """Run evaluation multiple times with different random seeds for reproducibility.

        For each seed, re-runs project-disjoint pre-filtering and embedding generation
        with a different split assignment, then runs all tasks. Aggregates per-seed
        results into mean ± std statistics.

        Args:
            tasks_to_run_list: List of task names to run.
            random_states: List of random seeds to evaluate.
            force_regenerate: Whether to force-regenerate embeddings.
            lp_config: Linear probe task config dict.

        Returns:
            Aggregated results dict with per-seed and summary statistics.
        """
        logger.info(
            f"Multi-seed evaluation: running {len(random_states)} seeds: {random_states}"
        )

        per_seed_results: Dict[int, Dict[str, Any]] = {}

        for i, seed in enumerate(random_states):
            logger.info(f"\n{'='*60}")
            logger.info(f"Seed {seed} ({i+1}/{len(random_states)})")
            logger.info(f"{'='*60}")

            # Override the global random_state for data loading
            original_rs = self.eval_config.get("random_state", 42)
            from omegaconf import open_dict
            with open_dict(self.eval_config):
                self.eval_config.random_state = seed

            # Also override the task-level random_state
            task_configs = self.eval_config.get("task_configs", {})
            for tn in tasks_to_run_list:
                tc = task_configs.get(tn, {})
                if tc:
                    with open_dict(tc):
                        tc.random_state = seed

            per_split_filters, per_split_max_samples = self._precompute_project_filter(
                tasks_to_run_list, random_state_override=seed,
            )
            multi_splits = self._generate_multi_split_embeddings(
                force_regenerate=True,  # Always regenerate — different splits per seed
                per_split_filters=per_split_filters or None,
                per_split_max_samples=per_split_max_samples,
            )
            seed_results = self.run_evaluation_tasks(
                embeddings=np.zeros((0, 1)),
                metadata={},
                faiss_index=None,
                precomputed_splits=multi_splits,
                pre_filtered=True,
            )
            per_seed_results[seed] = seed_results

            # Restore original random_state
            with open_dict(self.eval_config):
                self.eval_config.random_state = original_rs
            for tn in tasks_to_run_list:
                tc = task_configs.get(tn, {})
                if tc:
                    with open_dict(tc):
                        tc.random_state = original_rs

        # Aggregate results across seeds
        aggregated = self._aggregate_seed_results(per_seed_results)

        # Save aggregated results to task_results.json (overwrites per-seed files)
        for task_name, task_agg in aggregated.items():
            task_output_dir = self.output_dir / task_name
            task_output_dir.mkdir(parents=True, exist_ok=True)
            results_path = task_output_dir / "task_results.json"
            import json as _json
            with open(results_path, "w") as f:
                _json.dump(task_agg, f, indent=2, default=str)
            logger.info(f"Saved aggregated multi-seed results to {results_path}")

        return aggregated

    @staticmethod
    def _aggregate_seed_results(
        per_seed_results: Dict[int, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate per-seed task results into mean ± std statistics.

        Walks the nested result dicts, finds all numeric leaf values that appear
        across seeds, and computes mean/std. Preserves the structure of the first
        seed's results and adds `_mean`, `_std` suffixes for numeric fields.

        Args:
            per_seed_results: {seed -> {task_name -> task_results_dict}}

        Returns:
            Aggregated results dict with per_seed_results and summary.
        """
        seeds = sorted(per_seed_results.keys())
        task_names = list(next(iter(per_seed_results.values())).keys())

        aggregated: Dict[str, Any] = {}

        for task_name in task_names:
            seed_task_results = []
            for seed in seeds:
                if task_name in per_seed_results[seed]:
                    seed_task_results.append(per_seed_results[seed][task_name])

            if not seed_task_results:
                continue

            # Collect numeric values from the "targets" sub-dict
            summary = _collect_numeric_stats(seed_task_results)
            summary["per_seed"] = {
                seed: per_seed_results[seed].get(task_name, {})
                for seed in seeds
            }
            summary["seeds"] = seeds
            summary["n_seeds"] = len(seeds)
            aggregated[task_name] = summary

        return aggregated


    def _precompute_project_filter(
        self,
        tasks_to_run: list,
        random_state_override: int | None = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
        """Lightweight pre-pass to assign projects to probe splits before embedding generation.

        Loads filepaths lazily from each model split, looks up project labels via
        SearchDataManager, runs train-priority project assignment, and returns:
        - per_split_filters: {split_name -> set of filepaths assigned to that probe split}
        - per_split_max_samples: {split_name -> target sample count}

        Falls back to simple per-split size limits (no project filtering) if
        SearchDataManager is unavailable.
        """
        from instanovo_fm.data.search_data_manager import create_search_data_manager
        from instanovo_fm.eval.probe_splitting import _train_priority_assignment
        from instanovo.utils.data_handler import SpectrumDataFrame

        dataset_config = self.config.get("dataset", {})
        task_configs = self.eval_config.get("task_configs", {})

        # Find the first multi-split task config (LinearProbeTask)
        lp_config: Dict[str, Any] = {}
        for task_name in tasks_to_run:
            lp_config = task_configs.get(task_name, {})
            if lp_config:
                break

        train_samples: int = lp_config.get("train_samples", 100_000)
        val_samples: int = lp_config.get("val_samples", 10_000)
        test_samples: int = lp_config.get("test_samples", 10_000)
        min_project_samples: int = lp_config.get("min_project_samples", 5)
        max_per_project_frac: float = lp_config.get("max_per_project_frac", 0.15)
        per_split_max_samples = {"train": train_samples, "valid": val_samples, "test": test_samples}

        # Build SearchDataManager for project lookups
        search_data_config = {
            "use_search_data": dataset_config.get("use_search_data", False),
            "search_data_path": dataset_config.get("search_data_path", None),
            "search_data_filepath_column": dataset_config.get("search_data_filepath_column", "file path"),
            "search_data_spectrum_key": dataset_config.get("search_data_spectrum_key", "filepath"),
        }
        search_manager = create_search_data_manager(search_data_config)

        use_search_manager = search_manager is not None and search_manager.is_loaded
        if not use_search_manager:
            logger.info(
                "SearchDataManager unavailable — will extract project IDs "
                "directly from filepaths using regex patterns."
            )

        project_col = "project"  # Raw column in search_data.csv; becomes 'search_project' in embeddings

        # Step 1: For each split, lazily load USIs and extract project labels directly.
        # USI format: mzspec:PROJECT:filename:... — project is always the second field.
        split_usis: Dict[str, list] = {}

        for split_name in ("train", "valid", "test"):
            split_key = f"{split_name}_path"
            if split_key not in dataset_config:
                logger.warning(f"No {split_key} in dataset config, skipping {split_name}")
                continue

            logger.info(f"Pre-filter pass: scanning USIs for model-{split_name} split...")
            sdf = SpectrumDataFrame.load(
                source=dataset_config[split_key],
                source_type=dataset_config.get("source_type", "default"),
                lazy=True,
                is_annotated=True,
                shuffle=False,
                partition=None,
                add_source_file_column=False,
                preshuffle_across_shards=False,
                verbose=False,
            )
            import polars as pl
            lazy = sdf.to_polars(return_lazy=True)
            usis = lazy.select([pl.col("usi")]).collect()["usi"].to_list()
            split_usis[split_name] = usis
            logger.info(f"  model-{split_name}: {len(usis):,} spectra scanned")

        # Step 2: Extract project per USI and build project_info counts.
        # Project ID is embedded directly in the USI — no SearchDataManager needed.
        project_info: Dict[str, Dict[str, int]] = {}
        split_usi_projects: Dict[str, list] = {}

        for split_name, usis in split_usis.items():
            norm_name = "val" if split_name == "valid" else split_name

            projects = [self._extract_project_from_usi(u) for u in usis]
            n_matched = sum(1 for p in projects if p)
            n_unmatched = len(projects) - n_matched
            logger.info(
                f"  {split_name}: extracted project from USI for "
                f"{n_matched:,} / {len(usis):,} spectra"
                + (f" ({n_unmatched:,} unmatched)" if n_unmatched else "")
            )

            split_usi_projects[split_name] = projects

            for proj in projects:
                if not proj or proj.lower() in ("none", "nan", "unknown", "null"):
                    continue
                if proj not in project_info:
                    project_info[proj] = {"train": 0, "val": 0, "test": 0}
                project_info[proj][norm_name] += 1

        if not project_info:
            logger.warning(
                "No project labels found from SearchDataManager or filepath regex. "
                "Falling back to plain per-split size limits."
            )
            return {}, per_split_max_samples

        # Filter projects below minimum sample threshold
        project_info = {
            proj: counts
            for proj, counts in project_info.items()
            if sum(counts.values()) >= min_project_samples
        }
        logger.info(f"Pre-filter: {len(project_info)} projects with >= {min_project_samples} samples")

        # Step 3: Train-priority project assignment
        assignment = _train_priority_assignment(
            project_info=project_info,
            val_target=val_samples,
            test_target=test_samples,
            random_state=random_state_override if random_state_override is not None else self.eval_config.get("random_state", 42),
            max_per_project_frac=max_per_project_frac,
            train_target=train_samples,
        )

        # Step 4: Build per-split sets of assigned USIs
        per_split_filters: Dict[str, Set[str]] = {}
        for split_name in ("train", "valid", "test"):
            if split_name not in split_usis:
                continue
            probe_name = "val" if split_name == "valid" else split_name
            assigned_projects = set(assignment[probe_name])
            projects = split_usi_projects[split_name]
            usis = split_usis[split_name]
            assigned_usis: Set[str] = {
                u for u, proj in zip(usis, projects)
                if str(proj).strip() in assigned_projects
            }
            per_split_filters[split_name] = assigned_usis
            logger.info(
                f"  model-{split_name}: {len(assigned_usis):,} unique USIs from "
                f"{len(assigned_projects)} assigned projects"
            )

        return per_split_filters, per_split_max_samples

    def _precompute_stratified_filter(
        self,
        lp_config: Dict[str, Any],
        random_state_override: int | None = None,
    ) -> Tuple[Dict[str, Set[str]], Dict[str, int]]:
        """Build per-split USI include-sets enriched for rare modification targets.

        For rare, project-clustered modifications (phospho / glyco) uniform
        file-level sampling can yield too few — or zero — positives in a split.
        This pass instead scans each model split's peptide strings and selects,
        per split, up to ``enrich_pos_per_class`` positive USIs for each requested
        target plus ``enrich_neg_per_split`` negatives (peptides carrying none of
        the targets). The resulting probe split therefore contains ample positives
        for a stable binary detection AUROC.

        Config (under the linear-probe task):
          enrich_targets: list of target fields, e.g. [mod_phospho, mod_glyco]
          enrich_pos_per_class: positives per target per split (default 5000)
          enrich_neg_per_split: negatives per split (default 10000)

        Returns the same (per_split_filters, per_split_max_samples) contract as
        ``_precompute_project_filter`` so the downstream machinery is unchanged.
        """
        import polars as pl
        from instanovo_fm.utils.modifications import (
            PHOSPHO_UNIMOD_IDS,
            GLYCO_UNIMOD_IDS,
            GLYCO_ID_FAMILY,
        )

        dataset_config = self.config.get("dataset", {})
        pos_cap = int(lp_config.get("enrich_pos_per_class", 5000))
        neg_cap = int(lp_config.get("enrich_neg_per_split", 10000))
        enrich_targets = list(lp_config.get("enrich_targets") or [])
        enrich_multiclass = lp_config.get("enrich_multiclass", None)

        # Spectrum key: default `usi`; datasets whose usi is null (e.g. massivekb_splits)
        # key on `spectrum_id` = experiment_name:scan, matching the loader's synthesised
        # column. `_keyed` scans a path exposing that key as `_key` alongside `sequence`.
        spectrum_key = dataset_config.get("spectrum_key", "usi")

        def _keyed(path: str) -> "pl.LazyFrame":
            lf = pl.scan_parquet(path)
            if spectrum_key == "spectrum_id":
                key = (pl.col("experiment_name").cast(pl.Utf8) + ":"
                       + pl.col("scan").cast(pl.Utf8)).alias("_key")
            else:
                key = pl.col(spectrum_key).alias("_key")
            return lf.select([key, pl.col("sequence")])

        def _head_usis_ml(lf: "pl.LazyFrame", mask, cap: int) -> Set[str]:
            df = lf.filter(mask).select("_key").drop_nulls().head(cap).collect()
            return set(df["_key"].to_list())

        # Multiclass enrichment (e.g. glyco subtype): sample up to enrich_pos_per_class
        # USIs per class among the positive spectra only (no negatives), so the
        # downstream multiclass probe trains on class-balanced positives.
        if enrich_multiclass == "glyco_class":
            fam_ids: Dict[str, list] = {}
            for gid, fam in GLYCO_ID_FAMILY.items():
                fam_ids.setdefault(fam, []).append(gid)
            fam_re = {
                fam: rf"(?i)unimod:(?:{'|'.join(str(i) for i in sorted(ids))})[^0-9]"
                for fam, ids in fam_ids.items()
            }
            per_split_filters = {}
            per_split_max_samples = {}
            for split_name in ("train", "valid", "test"):
                key = f"{split_name}_path"
                if key not in dataset_config:
                    continue
                lf = _keyed(dataset_config[key])
                selected: Set[str] = set()
                counts = {}
                for fam, rx in fam_re.items():
                    usis = _head_usis_ml(lf, pl.col("sequence").str.contains(rx), pos_cap)
                    counts[fam] = len(usis)
                    selected |= usis
                per_split_filters[split_name] = selected
                per_split_max_samples[split_name] = len(selected)
                logger.info(
                    f"stratified {split_name} (glyco_class): "
                    + ", ".join(f"{f}={counts[f]:,}" for f in sorted(counts))
                    + f", total_usis={len(selected):,}"
                )
            return per_split_filters, per_split_max_samples

        # Regexes over the raw peptide string. Rust regex (polars) has no
        # lookahead, so a trailing [^0-9] class delimits the UNIMOD id (tokens are
        # always bracketed, e.g. "[UNIMOD:21]"). Residue-mass phospho tokens and a
        # HexNAc textual fallback cover the non-UNIMOD encoding too.
        phos_ids = "|".join(str(i) for i in sorted(PHOSPHO_UNIMOD_IDS))
        glyco_ids = "|".join(str(i) for i in sorted(GLYCO_UNIMOD_IDS))
        target_re = {
            "mod_phospho": rf"(?i)unimod:(?:{phos_ids})[^0-9]|S\[167\]|T\[181\]|Y\[243\]",
            "mod_glyco": rf"(?i)unimod:(?:{glyco_ids})[^0-9]|hexnac",
            "mod_deam_n": r"(?i)N[\[(]unimod:7[\])]|N\[115(\.[0-9]+)?\]",
        }
        for t in enrich_targets:
            if t not in target_re:
                raise ValueError(
                    f"enrich_targets: unsupported target '{t}' "
                    f"(supported: {list(target_re)})"
                )

        def _head_usis(lf: pl.LazyFrame, mask: pl.Expr, cap: int) -> Set[str]:
            df = lf.filter(mask).select("_key").drop_nulls().head(cap).collect()
            return set(df["_key"].to_list())

        # Paired within-backbone mode for the TEST split (Option Y): build the test
        # set from backbones seen BOTH with and without the target modification, so a
        # within-backbone control can isolate the modification signal from sequence
        # priors. Enabled by ``paired_test_target``; train/valid stay standard, so
        # the probe still trains on train-split spectra only.
        paired_test_target = lp_config.get("paired_test_target")
        bb_cap = int(lp_config.get("paired_backbones_cap", 2000))
        per_state_cap = int(lp_config.get("paired_per_state_cap", 10))
        _BB_STRIP = r"\[[^\]]*\]|\([^)]*\)"  # strip [UNIMOD:x] / [115] / (+0.98)
        if paired_test_target and paired_test_target not in target_re:
            raise ValueError(
                f"paired_test_target '{paired_test_target}' unsupported "
                f"(supported: {list(target_re)})"
            )

        def _paired_test_usis(lf: pl.LazyFrame) -> Tuple[Set[str], Set[str]]:
            """USIs (both states) for capped paired backbones + the backbone set."""
            df = lf.with_columns([
                pl.col("sequence").str.replace_all(_BB_STRIP, "").alias("_bb"),
                pl.col("sequence").str.contains(target_re[paired_test_target])
                .cast(pl.Int8).alias("_pos"),
            ]).collect()
            agg = df.group_by("_bb").agg([
                pl.col("_pos").min().alias("_mn"), pl.col("_pos").max().alias("_mx"),
            ])
            paired_bb = (
                agg.filter((pl.col("_mn") == 0) & (pl.col("_mx") == 1))
                .select("_bb").sort("_bb").head(bb_cap)["_bb"].to_list()
            )
            dfp = df.filter(pl.col("_bb").is_in(paired_bb))
            dfp = dfp.with_columns(
                pl.int_range(pl.len()).over(["_bb", "_pos"]).alias("_rk")
            )
            kept = dfp.filter(pl.col("_rk") < per_state_cap)
            return set(kept["_key"].drop_nulls().to_list()), set(paired_bb)

        per_split_filters: Dict[str, Set[str]] = {}
        per_split_max_samples: Dict[str, int] = {}
        test_paired_bb: Set[str] = set()
        for split_name in ("train", "valid", "test"):
            key = f"{split_name}_path"
            if key not in dataset_config:
                continue
            lf = _keyed(dataset_config[key])

            if paired_test_target and split_name == "test":
                selected, test_paired_bb = _paired_test_usis(lf)
                per_split_filters[split_name] = selected
                per_split_max_samples[split_name] = len(selected)
                logger.info(
                    f"stratified test (paired {paired_test_target}): "
                    f"paired_backbones={len(test_paired_bb):,}, "
                    f"total_usis={len(selected):,}"
                )
                continue

            pos_masks = {
                t: pl.col("sequence").str.contains(target_re[t]) for t in enrich_targets
            }

            selected: Set[str] = set()
            counts: Dict[str, int] = {}
            for t in enrich_targets:
                usis = _head_usis(lf, pos_masks[t], pos_cap)
                counts[t] = len(usis)
                selected |= usis

            any_pos = pos_masks[enrich_targets[0]]
            for t in enrich_targets[1:]:
                any_pos = any_pos | pos_masks[t]
            neg_usis = _head_usis(lf, ~any_pos, neg_cap)
            selected |= neg_usis

            per_split_filters[split_name] = selected
            per_split_max_samples[split_name] = len(selected)
            logger.info(
                f"stratified {split_name}: "
                + ", ".join(f"{t}+={counts[t]:,}" for t in enrich_targets)
                + f", neg={len(neg_usis):,}, total_usis={len(selected):,}"
            )

        # Cross-split leakage assertion: no paired-test backbone may appear in train.
        if paired_test_target and test_paired_bb and "train_path" in dataset_config:
            train_hits = (
                pl.scan_parquet(dataset_config["train_path"])
                .select(pl.col("sequence").str.replace_all(_BB_STRIP, "").alias("_bb"))
                .filter(pl.col("_bb").is_in(list(test_paired_bb)))
                .select(pl.len()).collect().item()
            )
            if train_hits > 0:
                raise ValueError(
                    f"Paired-test leakage: {train_hits} train spectra share a backbone "
                    f"with the paired test set; splits must be backbone-disjoint."
                )
            logger.info(
                f"paired-test disjointness OK: 0 of {len(test_paired_bb):,} paired "
                f"test backbones appear in the train split."
            )

        return per_split_filters, per_split_max_samples

    def _generate_multi_split_embeddings(
        self,
        force_regenerate: bool = False,
        precomputed: Dict[str, tuple] | None = None,
        per_split_filters: Dict[str, Set[str]] | None = None,
        per_split_max_samples: Dict[str, int] | None = None,
    ) -> Dict[str, tuple]:
        """Generate embeddings for all three model splits (train, valid, test).

        Used when tasks require multi-split data (e.g., linear probes with
        project-disjoint splits).

        Args:
            force_regenerate: If True, regenerate embeddings even if cached.
            precomputed: Optional dict of already-generated (embeddings, metadata)
                tuples keyed by split name. Matching splits are reused, avoiding
                redundant embedding generation.
            per_split_filters: Optional dict mapping split name to the set of
                filepaths to keep (project pre-filter). Applied before to_dataset.
            per_split_max_samples: Optional dict mapping split name to the max
                number of samples to load for that split.

        Returns:
            Dict mapping split names to (embeddings, metadata) tuples:
            {"train": (emb, meta), "valid": (emb, meta), "test": (emb, meta)}
        """
        multi_splits = dict(precomputed) if precomputed else {}
        original_dataloader = self.dataloader

        try:
            for split_name in ("train", "valid", "test"):
                if split_name in multi_splits:
                    logger.info(f"Reusing precomputed embeddings for model-{split_name} split ({len(multi_splits[split_name][0]):,} samples)")
                    continue
                logger.info(f"Generating embeddings for model-{split_name} split...")
                filter_fps = per_split_filters.get(split_name) if per_split_filters else None
                override_max = per_split_max_samples.get(split_name) if per_split_max_samples else None
                try:
                    self.dataloader = self.setup_dataloader(
                        split=split_name,
                        filter_usis=filter_fps,
                        override_max_samples=override_max,
                    )
                    emb, meta, _ = self.generate_embeddings(
                        force_regenerate=force_regenerate,
                        override_max_samples=override_max,
                        split=split_name,
                    )
                    multi_splits[split_name] = (emb, meta)
                    logger.info(
                        f"  model-{split_name}: {len(emb):,} embeddings generated"
                    )
                except (ValueError, KeyError) as e:
                    logger.warning(
                        f"  Could not generate embeddings for model-{split_name}: {e}"
                    )
        finally:
            # Always restore original dataloader, even on unexpected exceptions
            self.dataloader = original_dataloader

        if len(multi_splits) < 3:
            available = list(multi_splits.keys())
            logger.warning(
                f"Only generated embeddings for {available} splits "
                f"(expected train, valid, test). Multi-split tasks may fail."
            )

        return multi_splits

    def run_evaluation_tasks(
        self,
        embeddings: np.ndarray,
        metadata: Dict[str, np.ndarray],
        faiss_index: Any,
        force_regenerate: bool = False,
        precomputed_splits: Dict[str, tuple] | None = None,
        pre_filtered: bool = False,
        output_subdir: str | None = None,
    ) -> Dict[str, Any]:
        """Run all configured evaluation tasks.

        Args:
            embeddings: Embeddings array (N, D)
            metadata: Metadata dictionary
            faiss_index: FAISS index for similarity search
            force_regenerate: If True, regenerate embeddings for multi-split
                tasks even if cached.
            precomputed_splits: Optional dict of already-generated
                (embeddings, metadata) tuples keyed by split name, to avoid
                redundant embedding generation for multi-split tasks.
            pre_filtered: If True, multi-split embeddings have already been
                project-disjoint filtered; tasks should skip internal splitting.
            output_subdir: Optional subdirectory under output_dir for results
                (e.g. "cls" or "mean_pool" when comparing strategies)

        Returns:
            Dictionary mapping task names to their results.
        """
        # Get list of tasks to run
        tasks_to_run = self.eval_config.get("tasks_to_run", None)

        if tasks_to_run is None:
            # Run all available tasks
            tasks_to_run = list(TASK_REGISTRY.keys())
            logger.info(f"Running all {len(tasks_to_run)} available tasks")
        else:
            logger.info(f"Running {len(tasks_to_run)} configured tasks: {tasks_to_run}")

        # Determine base output directory for this run
        base_output = self.output_dir / output_subdir if output_subdir else self.output_dir

        # Get task-specific configurations
        task_configs = self.eval_config.get("task_configs", {})

        # Check if any task requires multi-split data and generate if needed
        multi_splits = precomputed_splits
        if multi_splits is None:
            for task_name in tasks_to_run:
                try:
                    task_class = get_task(task_name)
                    if getattr(task_class, 'requires_multi_split', False):
                        logger.info(
                            f"Task '{task_name}' requires multi-split data. "
                            f"Generating embeddings for all model splits..."
                        )
                        # Extract per-split sample counts from task config
                        tc = task_configs.get(task_name, {})
                        per_split_max_samples = {
                            "train": tc.get("train_samples", 50_000),
                            "valid": tc.get("val_samples", 5_000),
                            "test": tc.get("test_samples", 5_000),
                        }
                        # Same gate as the multi-split-only path above, so both paths agree.
                        # Previously this call never passed per_split_filters, so adding a
                        # single-split task to a run silently switched the probe from
                        # project-assigned splits to overlapping ones.
                        if tc.get("use_project_split", True):
                            pf, pm = self._precompute_project_filter(list(tasks_to_run))
                            logger.info("Probe splits: PROJECT-ASSIGNED (use_project_split=true)")
                        else:
                            pf, pm = None, per_split_max_samples
                            logger.info(
                                "Probe splits: NOT project-assigned (use_project_split=false) — "
                                "drawn from the dataset's own splits"
                            )
                        multi_splits = self._generate_multi_split_embeddings(
                            force_regenerate=force_regenerate,
                            per_split_filters=pf,
                            per_split_max_samples=pm or per_split_max_samples,
                        )
                        break  # Only need to generate once
                except KeyError:
                    pass  # Task not found, will fail later with proper error


        # Run tasks
        results = {}
        for task_name in tasks_to_run:
            try:
                logger.info(f"Running task: {task_name}")
                start_time = time.time()

                # Create task-specific output directory
                task_output_dir = base_output / task_name
                task_output_dir.mkdir(parents=True, exist_ok=True)

                # Get task class
                task_class = get_task(task_name)

                # Get task-specific config
                task_config = task_configs.get(task_name, {})

                # Instantiate task (pass output_dir separately)
                task = task_class(output_dir=str(task_output_dir), **task_config)

                # Check if task requires model access
                if task.requires_model:
                    # Verify model and dataloader are available
                    if self.model is None:
                        raise ValueError(
                            f"Task '{task_name}' requires model access but model is not loaded. "
                            "Make sure to call load_model() or set evaluator.model before running evaluation."
                        )
                    if self.dataloader is None:
                        raise ValueError(
                            f"Task '{task_name}' requires model access but dataloader is not setup. "
                            "Make sure to call setup_dataloader() or set evaluator.dataloader before running evaluation."
                        )

                    # Pass model, dataloader, config, and device to task
                    task_results = task.run(
                        embeddings,
                        metadata,
                        faiss_index,
                        model=self.model,
                        dataloader=self.dataloader,
                        config=self.model_config,
                        device=self.device
                    )
                elif task.requires_multi_split:
                    # Pass multi-split embeddings to task
                    if multi_splits is None:
                        raise ValueError(
                            f"Task '{task_name}' requires multi-split data but "
                            f"multi-split embeddings were not generated."
                        )
                    task_results = task.run(
                        embeddings,
                        metadata,
                        faiss_index,
                        splits=multi_splits,
                        pre_filtered=pre_filtered,
                    )
                else:
                    # Standard task execution without model access
                    task_results = task.run(embeddings, metadata, faiss_index)

                # Add timing information
                task_results["execution_time"] = time.time() - start_time
                task_results["output_dir"] = str(task_output_dir)

                # Save task results to its own directory
                self._save_task_results(task_name, task_results, task_output_dir)

                results[task_name] = task_results

                logger.info(f"  [OK] {task_name} ({task_results['execution_time']:.1f}s)")

            except Exception as e:
                logger.error(f"  [FAILED] {task_name} failed: {e}")
                import traceback
                traceback.print_exc()
                results[task_name] = {
                    "error": str(e),
                    "success": False,
                }

        if results and all("error" in v for v in results.values()):
            failed_summaries = "; ".join(
                f"{k}: {v['error']}" for k, v in results.items()
            )
            raise RuntimeError(f"All evaluation tasks failed: {failed_summaries}")

        return results

    def _save_task_results(
        self,
        task_name: str,
        task_results: Dict[str, Any],
        task_output_dir: Path,
    ) -> None:
        """Save individual task results to its own directory.

        Args:
            task_name: Name of the task
            task_results: Results from the task
            task_output_dir: Directory to save results to
        """
        # Create task summary
        task_summary = {
            "task_name": task_name,
            "timestamp": datetime.now().isoformat(),
            "execution_time": task_results.get("execution_time", 0),
            "success": task_results.get("error") is None,
        }

        # Add key metrics via each task's get_loggable_metrics()
        if "error" not in task_results:
            try:
                task_class = get_task(task_name)
                task_metrics = task_class().get_loggable_metrics(task_results)
                task_summary.update(task_metrics)
            except Exception:
                pass  # Task not found or no loggable metrics
        else:
            task_summary["error"] = task_results["error"]

        # Save task summary
        summary_path = task_output_dir / "task_summary.json"
        with open(summary_path, "w") as f:
            json.dump(self._make_json_serializable(task_summary), f, indent=2)

        # Save full task results (if not an error)
        if "error" not in task_results:
            full_results_path = task_output_dir / "task_results.json"
            with open(full_results_path, "w") as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = self._make_json_serializable(task_results)
                json.dump(serializable_results, f, indent=2)

    def save_results(
        self,
        results: Dict[str, Any],
        embeddings_info: Dict[str, Any],
    ) -> None:
        """Print evaluation summary to console.

        Args:
            results: Dictionary of task results
            embeddings_info: Information about embeddings (num, dim, etc.)
        """
        # Create results summary for console output
        summary = {
            "checkpoint": self.checkpoint_path,
            "output_dir": str(self.output_dir),
            "device": str(self.device),
            "embeddings": embeddings_info,
            "tasks": {},
        }

        # Add task results (compact summary)
        for task_name, task_results in results.items():
            # Create compact summary for each task
            task_summary = {
                "success": task_results.get("error") is None,
                "execution_time": task_results.get("execution_time", 0),
            }

            # Add key metrics via each task's get_loggable_metrics()
            if "error" not in task_results:
                try:
                    task_class = get_task(task_name.split("/")[-1] if "/" in task_name else task_name)
                    task_metrics = task_class().get_loggable_metrics(task_results)
                    task_summary.update(task_metrics)
                except Exception:
                    pass  # Task not found or no loggable metrics
            else:
                task_summary["error"] = task_results["error"]

            summary["tasks"][task_name] = task_summary

        # Print summary to console
        self._print_summary(summary)

    def upload_results_to_s3(self) -> None:
        """Upload evaluation results to S3 if running on Aichor.

        This method uploads all files in the output directory to S3, similar to
        how training uploads checkpoints and profiling results.
        """
        import logging

        if not S3FileHandler._aichor_enabled():
            return

        logger.info(f"Uploading evaluation results to S3...")

        # Get all files in output directory recursively
        output_files = list(self.output_dir.rglob("*"))

        # Filter to only files (not directories)
        output_files = [f for f in output_files if f.is_file()]

        # Suppress per-file "Uploading ... to ..." messages from s3 module
        s3_logger = logging.getLogger("instanovo.utils.s3")
        prev_s3_level = s3_logger.level
        s3_logger.setLevel(logging.WARNING)

        uploaded_count = 0
        try:
            for local_file in output_files:
                try:
                    # Convert to S3 path
                    s3_path = S3FileHandler.convert_to_s3_output(str(local_file))
                    # Upload file
                    self.s3.upload(str(local_file), s3_path)
                    uploaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to upload {local_file.name}: {e}")
        finally:
            s3_logger.setLevel(prev_s3_level)

        logger.info(f"Successfully uploaded {uploaded_count}/{len(output_files)} files to S3")

    def _make_json_serializable(self, obj: Any) -> Any:
        """Recursively convert numpy arrays and other non-serializable objects to JSON-compatible types.

        Args:
            obj: Object to convert

        Returns:
            JSON-serializable version of the object
        """
        # Handle OmegaConf types
        try:
            from omegaconf import ListConfig, DictConfig, OmegaConf
            if isinstance(obj, (ListConfig, DictConfig)):
                obj = OmegaConf.to_container(obj, resolve=True)
        except ImportError:
            pass

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

    def _save_config_files(self) -> None:
        """Save evaluation and model configuration to output directory.

        This creates two files:
        1. eval_config.yaml - The evaluation configuration used
        2. model_config.yaml - The model configuration from the checkpoint
        """
        from omegaconf import OmegaConf

        assert self.model is not None, "Model must be loaded before saving config files"
        assert self.model_config is not None, "Model config must be loaded before saving config files"

        # Save evaluation configuration
        eval_config_path = self.output_dir / "eval_config.yaml"
        eval_config_dict = {
            "checkpoint_path": str(self.checkpoint_path) if self.checkpoint_path else "provided_externally",
            "evaluation": OmegaConf.to_container(self.eval_config, resolve=True),
            "dataset": OmegaConf.to_container(self.config.get("dataset", {}), resolve=True),
            "num_workers": self.config.get("num_workers", 4),
            "device": str(self.device),
        }

        with open(eval_config_path, "w") as f:
            OmegaConf.save(eval_config_dict, f)


        # Save model configuration
        if self.model_config is not None:
            model_config_path = self.output_dir / "model_config.yaml"

            # Convert to container and add model metadata
            # Note: resolve=False to avoid interpolation errors when config has unresolved references
            model_config_dict = OmegaConf.to_container(self.model_config, resolve=False)

            # Add model summary information
            model_summary = {
                "model_info": {
                    "total_parameters": sum(p.numel() for p in self.model.parameters()),
                    "trainable_parameters": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                    "architecture": {
                        "dim_model": self.model.dim_model,
                        "n_layers": self.model.n_layers,
                        "n_heads": self.model.n_heads,
                        "dim_feedforward": self.model.dim_feedforward,
                        "dropout": self.model.dropout,
                    },
                    "tasks": {
                        "mz_task": self.model.mz_task,
                        "peak_encoder_type": self.model.peak_encoder_type,
                        "use_meta_token": self.model.use_meta_token,
                        "use_flash_attention": self.model.use_flash_attention,
                    },
                    "data_config": {
                        "n_peaks": self.model.n_peaks,
                        "max_mz": self.model.max_mz,
                        "min_mz": self.model.min_mz,
                        "max_charge": self.model.max_charge,
                    }
                },
                "full_config": model_config_dict,
            }

            with open(model_config_path, "w") as f:
                OmegaConf.save(model_summary, f)


    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Print evaluation summary to console.

        Groups results by embedding pooling strategy when prefixed keys
        (e.g. ``cls/task_name``, ``mean_pool/task_name``) are present.

        Args:
            summary: Evaluation summary dictionary
        """
        logger.info("--- EVALUATION SUMMARY ---")

        # Group tasks by strategy prefix
        grouped: Dict[str, Dict[str, Any]] = {}
        for result_key, task_summary in summary["tasks"].items():
            if "/" in result_key:
                strategy, task_name = result_key.split("/", 1)
            else:
                strategy, task_name = "", result_key
            grouped.setdefault(strategy, {})[task_name] = task_summary

        for strategy, tasks in grouped.items():
            if strategy:
                logger.info(f"Pooling: {strategy}")

            for task_name, task_summary in tasks.items():
                if task_summary["success"]:
                    logger.info(f"  [OK] {task_name} ({task_summary['execution_time']:.1f}s)")
                    for k, v in task_summary.items():
                        if k in ("success", "execution_time"):
                            continue
                        v_str = f"{v:.4f}" if isinstance(v, float) else v
                        logger.info(f"       {k}={v_str}")
                else:
                    logger.info(f"  [FAILED] {task_name}: {task_summary.get('error', 'Unknown error')}")

        logger.info(f"Output: {summary['output_dir']}")

    def evaluate(
        self,
        split: str = "valid",
        force_regenerate: bool = False,
    ) -> Dict[str, Any]:
        """Run the complete evaluation workflow.

        When ``embedding_pooling`` is ``"both"``, the evaluation loop runs
        tasks on CLS **and** mean-pool embeddings, tagging each result set
        with a ``cls/`` or ``mean_pool/`` prefix so they can be compared
        side-by-side.

        Args:
            split: Dataset split to evaluate ("valid" or "test")
            force_regenerate: If True, regenerate embeddings even if cached

        Returns:
            Dictionary of evaluation results
        """
        logger.info(f"Starting evaluation on '{split}' split")

        # Suppress HuggingFace datasets progress bars (map operations produce many lines)
        import datasets
        datasets.disable_progress_bars()

        # 1. Load model
        logger.info(f"Loading model from checkpoint: {self.checkpoint_path}")
        self.model, self.model_config = self.load_model()

        # 2. Setup data processor
        self.data_processor = self.setup_data_processor()

        # Determine which tasks will run
        tasks_to_run_list = self.eval_config.get("tasks_to_run", None) or list(TASK_REGISTRY.keys())

        # Fast path: when ALL tasks require multi-split data (e.g. only LinearProbeTask),
        # skip the primary valid embedding generation entirely and let
        # _generate_multi_split_embeddings handle all three splits with the correct
        # per-split sizes and project-disjoint pre-filtering.
        if self._all_tasks_require_multi_split(tasks_to_run_list):
            logger.info(
                "All tasks require multi-split data — skipping primary embedding generation."
            )

            # Every selected task loads train/valid/test itself, so the top-level
            # `split` setting has no effect here. Say so, rather than letting a run
            # configured with split=test quietly report numbers it did not scope.
            requested_split = self.eval_config.get("split", None)
            if requested_split:
                logger.warning(
                    "split=%s is ignored: every selected task loads train/valid/test itself. "
                    "The setting applies only to single-split tasks.",
                    requested_split,
                )

            # Check if multi-seed evaluation is requested
            task_configs = self.eval_config.get("task_configs", {})
            lp_config = {}
            for tn in tasks_to_run_list:
                lp_config = task_configs.get(tn, {})
                if lp_config:
                    break
            random_states = lp_config.get("random_states", None)

            if random_states and len(random_states) > 1:
                # Multi-seed evaluation: run probe with different splits per seed
                all_results = self._evaluate_multi_seed(
                    tasks_to_run_list, random_states, force_regenerate, lp_config,
                )
            elif lp_config.get("enrich_targets") or lp_config.get("enrich_multiclass"):
                # Stratified enrichment for rare modification targets (phospho/glyco/
                # deamidation-N binary, or glyco-subtype multiclass): guarantee enough
                # positives per split/class for a stable probe.
                per_split_filters, per_split_max_samples = self._precompute_stratified_filter(lp_config)
                multi_splits = self._generate_multi_split_embeddings(
                    force_regenerate=force_regenerate,
                    per_split_filters=per_split_filters or None,
                    per_split_max_samples=per_split_max_samples,
                )
                all_results = self.run_evaluation_tasks(
                    embeddings=np.zeros((0, 1)),
                    metadata={},
                    faiss_index=None,
                    precomputed_splits=multi_splits,
                    pre_filtered=True,
                )
            else:
                # `use_project_split` is authoritative: it decides whether probe splits are
                # project-assigned, NOT which tasks happen to be in the run. Before this gate the
                # pre-filter ran unconditionally here, so a probes-only run silently produced
                # project-disjoint splits (e.g. 65/7/7 projects) while the identical config with a
                # single-split task alongside produced overlapping ones (78/70/75) -- different
                # protocols, incomparable numbers, and the flag ignored in both cases.
                if lp_config.get("use_project_split", True):
                    per_split_filters, per_split_max_samples = self._precompute_project_filter(tasks_to_run_list)
                    logger.info("Probe splits: PROJECT-ASSIGNED (use_project_split=true)")
                else:
                    per_split_filters = None
                    per_split_max_samples = {
                        "train": lp_config.get("train_samples", 100_000),
                        "valid": lp_config.get("val_samples", 10_000),
                        "test": lp_config.get("test_samples", 10_000),
                    }
                    logger.info(
                        "Probe splits: NOT project-assigned (use_project_split=false) — drawn from "
                        "the dataset's own splits, so projects may appear in more than one split"
                    )
                multi_splits = self._generate_multi_split_embeddings(
                    force_regenerate=force_regenerate,
                    per_split_filters=per_split_filters or None,
                    per_split_max_samples=per_split_max_samples,
                )
                all_results = self.run_evaluation_tasks(
                    embeddings=np.zeros((0, 1)),
                    metadata={},
                    faiss_index=None,
                    precomputed_splits=multi_splits,
                    pre_filtered=True,
                )

            self._save_config_files()
            self.save_results(all_results, {})
            self.upload_results_to_s3()
            logger.info("Evaluation complete!")
            return all_results

        # 3. Setup dataloader (primary split — used for single-split tasks)
        self.dataloader = self.setup_dataloader(split=split)

        # Determine pooling strategies to evaluate
        requested_pooling = self.eval_config.get("embedding_pooling", "mean_pool")
        from omegaconf import ListConfig
        if isinstance(requested_pooling, (list, tuple, ListConfig)):
            pooling_strategies = list(requested_pooling)
        elif requested_pooling == "both":
            # Backward compatibility
            pooling_strategies = ["cls", "mean_pool"]
        else:
            pooling_strategies = [requested_pooling]

        all_results: Dict[str, Any] = {}
        last_embeddings_info: Dict[str, Any] = {}

        for strategy in pooling_strategies:
            tag = f"{strategy}/" if len(pooling_strategies) > 1 else ""
            if tag:
                logger.info("=" * 80)
                logger.info(f"Evaluating with embedding_pooling='{strategy}'")
                logger.info("=" * 80)

            # Override pooling strategy for this iteration
            from omegaconf import OmegaConf, open_dict
            with open_dict(self.eval_config):
                self.eval_config.embedding_pooling = strategy

            # 4. Generate embeddings (always regenerate for second strategy)
            regen = force_regenerate or (strategy != pooling_strategies[0])
            embeddings, metadata, faiss_index = self.generate_embeddings(
                force_regenerate=regen
            )

            # 5. Get embedding statistics
            embeddings_info = embedding_io.get_embedding_stats(embeddings)
            embeddings_info["embedding_pooling"] = strategy
            logger.info(f"Embeddings ({strategy}): {embeddings_info['num_embeddings']} x {embeddings_info['embedding_dim']}, mean_norm={embeddings_info['mean_norm']:.4f}")
            last_embeddings_info = embeddings_info

            # 6. Optionally compute sequence similarity clustering (once)
            if strategy == pooling_strategies[0]:
                metadata = self.compute_sequence_similarity_clustering(metadata)

            # 7. Run evaluation tasks (write to strategy-specific subdirectory)
            output_subdir = strategy if len(pooling_strategies) > 1 else None
            results = self.run_evaluation_tasks(
                embeddings, metadata, faiss_index, output_subdir=output_subdir,
            )

            # Tag results with strategy prefix when comparing both
            for task_name, task_results in results.items():
                all_results[f"{tag}{task_name}"] = task_results

        # Restore original config value
        with open_dict(self.eval_config):
            self.eval_config.embedding_pooling = requested_pooling

        # 8. Save configuration files
        self._save_config_files()

        # Store for post-eval MLflow logging
        self._last_embeddings_info = last_embeddings_info

        # 9. Save results
        self.save_results(all_results, last_embeddings_info)

        # 10. Upload results to S3 if running on Aichor
        self.upload_results_to_s3()

        logger.info("Evaluation complete!")

        return all_results
