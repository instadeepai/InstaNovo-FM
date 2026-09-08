"""Training script for InstaNovo Foundation Model.

Self-supervised learning on MS/MS spectra through masked m/z reconstruction.
"""

from __future__ import annotations

import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from instanovo.__init__ import console
from instanovo_fm.common import AccelerateDeNovoTrainer, DataProcessor
from instanovo_fm.data import FoundationalDataProcessor
from instanovo_fm.data.search_data_manager import create_search_data_manager
from instanovo_fm.trainer.profiling import create_profiler_from_config, profile_component
from instanovo.inference import Decoder
from instanovo.utils.colorlogging import ColorLog
from instanovo.utils.s3 import S3FileHandler

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs"


class FoundationalTrainer(AccelerateDeNovoTrainer):
    """Trainer for InstaNovo Foundation Model.

    Implements self-supervised learning through masked m/z reconstruction.
    Unlike transformer/diffusion models, this is encoder-only and doesn't
    require peptide annotations during training.

    Key differences from supervised trainers:
        - No decoder setup (encoder-only)
        - No sequence-based metrics during training
        - MSE/MAE loss for masked peak reconstruction
        - Optional evaluation on annotated data
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize the foundational trainer.

        Args:
            config: Hydra configuration with model, dataset, and training params.
        """
        logger.info("Starting FoundationalTrainer initialization...")
        logger.info("Calling parent __init__...")
        super().__init__(config)
        logger.info("Parent __init__ completed")

        # Disable sequence-based metrics for encoder-only model
        # Foundation model doesn't predict peptide sequences
        self.metrics = None

        # Initialize profiler
        self.profiling_output_dir = Path(config.get("model_save_folder_path", "./checkpoints")) / self.run_id / "profiling"
        self.profiler = create_profiler_from_config(config, str(self.profiling_output_dir))
        if self.profiler.enabled and self.accelerator.is_main_process:
            logger.info(f"Training profiler enabled. Output directory: {self.profiling_output_dir}")

        # Set effective batch size for profiler throughput computation
        effective_batch_size = int(self.config.get("train_batch_size", 1)) * int(self.config.get("grad_accumulation", 1))
        self.profiler.set_batch_size(effective_batch_size)

        # Cache offset conditioning settings for fast access in forward()
        mz_head_cfg = self.config.model.get("mz_head", {})
        self._offset_conditioning = mz_head_cfg.get("offset_conditioning", "none")
        self._eval_topk_groups = mz_head_cfg.get("eval_topk_groups", 1)
        # Cache bin group parameters for classification task (to avoid recomputing every step)
        self._cache_bin_params()

        logger.info("FoundationalTrainer initialized for self-supervised learning")

    def _get_compiled_classification_loss(self) -> Any:
        """Return a compiled version of compute_classification_loss if enabled."""
        if not hasattr(self, "_compiled_classification_loss"):
            from instanovo_fm.trainer.losses import compute_classification_loss

            if self.config.get("compile_loss", False):
                logger.info("Compiling classification loss function...")
                self._compiled_classification_loss = torch.compile(compute_classification_loss, mode="default")
            else:
                self._compiled_classification_loss = compute_classification_loss
        return self._compiled_classification_loss

    def _unwrap_model(self) -> Any:
        """Unwrap model from accelerate DDP and/or torch.compile wrappers.

        accelerate's unwrap_model can fail with KeyError('_orig_mod') when
        torch.compile + DDP are combined. This helper handles that case.
        """
        model = self.model
        # Peel DDP wrapper
        if hasattr(model, "module"):
            model = model.module
        # Peel torch.compile wrapper
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    def _cache_bin_params(self) -> None:
        """Cache bin group parameters from model to avoid recomputing every step.

        This method extracts bin parameters from the model once and caches them
        as trainer attributes. For classification tasks, this avoids costly
        reflection and recomputation in the hot path.
        """
        mz_task = self.config.model.get("mz_head", {}).get("task", "classification")

        if mz_task == "classification":
            try:
                # Unwrap model to access buffers
                unwrapped_model = self._unwrap_model()

                # Cache binning strategy object (if available)
                if hasattr(unwrapped_model, "binning_strategy"):
                    self._binning_strategy = unwrapped_model.binning_strategy

                # Cache bin edges (if available for non-uniform strategies)
                if hasattr(unwrapped_model, "bin_edges"):
                    self._bin_edges = unwrapped_model.bin_edges
                else:
                    self._bin_edges = None

                # Read scalar bin parameters from model buffers
                self._mz_group_size = int(unwrapped_model.bin_group_size_tensor.item())  # type: ignore[attr-defined]
                self._mz_n_groups = int(unwrapped_model.n_bin_groups_tensor.item())  # type: ignore[attr-defined]
                self._mz_last_group = int(unwrapped_model.last_group_size_tensor.item())  # type: ignore[attr-defined]

                # Cache bin_size (for uniform binning strategies)
                if hasattr(self, "_binning_strategy") and hasattr(self._binning_strategy, "bin_size"):
                    self._mz_bin_size = self._binning_strategy.bin_size
                elif hasattr(unwrapped_model, "bin_size_tensor"):
                    self._mz_bin_size = float(unwrapped_model.bin_size_tensor.item())  # type: ignore[attr-defined]
                else:
                    # Non-uniform binning (e.g., fixed_ppm) - bin_size is not a scalar
                    self._mz_bin_size = None

                # Log binning configuration
                strategy_info = f", strategy={self._binning_strategy}" if hasattr(self, "_binning_strategy") else ""
                bin_size_info = f", bin_size={self._mz_bin_size}" if self._mz_bin_size is not None else ""
                logger.info(
                    f"Cached bin parameters: "
                    f"group_size={self._mz_group_size}, n_groups={self._mz_n_groups}, "
                    f"last_group_size={self._mz_last_group}{bin_size_info}{strategy_info}"
                )
            except AttributeError as e:
                # Fallback: compute from config if buffers not available
                logger.warning(f"Could not read bin parameters from model buffers: {e}")
                logger.warning("Falling back to config-based computation")

                # Create binning strategy from config
                from instanovo_fm.trainer.binning import create_binning_strategy

                max_mz_val = float(self.config.model.get("max_mz", 2500.0))
                min_mz_val = float(self.config.model.get("min_mz", 0.0))

                self._binning_strategy = create_binning_strategy(self.config.model, min_mz_val, max_mz_val)

                # Cache bin edges (if non-uniform)
                self._bin_edges = self._binning_strategy.bin_edges

                # Cache scalar parameters
                self._mz_group_size = self._binning_strategy.bin_group_size
                self._mz_n_groups = self._binning_strategy.n_groups
                self._mz_last_group = self._binning_strategy.last_group_size

                # Cache bin_size (for uniform binning strategies)
                if hasattr(self._binning_strategy, "bin_size"):
                    self._mz_bin_size = self._binning_strategy.bin_size
                else:
                    # Non-uniform binning (e.g., fixed_ppm) - bin_size is not a scalar
                    self._mz_bin_size = None

        else:
            # Not a classification task - no bin parameters needed
            self._mz_bin_size = None
            self._mz_group_size = None  # type: ignore[assignment]
            self._mz_n_groups = None  # type: ignore[assignment]
            self._mz_last_group = None  # type: ignore[assignment]

    def setup_model(self) -> nn.Module:
        """Setup the foundation model.

        Returns:
            Foundation model (encoder-only transformer).
        """
        from omegaconf import OmegaConf

        from instanovo_fm.model import FoundationModel

        config = self.config.get("model", {})

        # Inject residue masses for IonLadder (if enabled).
        # We convert to a plain dict without resolving interpolations to avoid
        # OmegaConf InterpolationKeyError on tags that reference parent-level keys.
        ion_ladder_cfg = config.get("ion_ladder", {})
        if ion_ladder_cfg.get("enabled", False):
            config = OmegaConf.to_container(config, resolve=False)
            config.setdefault("ion_ladder", {})["residue_masses"] = self.residue_set.residue_masses
            logger.info(f"IonLadder: injected {len(self.residue_set.residue_masses)} residue masses from config")

        # Create model with full config for advanced settings
        model = FoundationModel(
            dim_model=config.get("dim_model", 512),
            n_heads=config.get("n_heads", 8),
            dim_feedforward=config.get("dim_feedforward", 2048),
            n_layers=config.get("n_layers", 6),
            dropout=config.get("dropout", 0.1),
            n_peaks=config.get("n_peaks", 200),
            max_mz=config.get("max_mz", 2500.0),
            min_mz=config.get("min_mz", 0.0),
            max_charge=config.get("max_charge", 10),
            peak_encoder_type=config.get("peak_encoder", {}).get("type", "multiscale")
            if isinstance(config.get("peak_encoder"), dict)
            else config.get("peak_encoder", "multiscale"),
            mz_task=config.get("mz_head", {}).get("task", "regression"),
            use_meta_token=config.get("meta_token", {}).get("enabled", False),
            cfg=config,  # Pass full config for advanced settings
        )

        logger.info(f"FoundationModel created with {sum(p.numel() for p in model.parameters())} parameters")

        # Apply torch.compile if enabled (before accelerator.prepare)
        # Notes:
        #   - PA bias: FlashMHA uses FlexAttention (torch.nn.attention.flex_attention)
        #     with score_mod for fused attention+bias in a flash-like kernel.
        #     The flex_attention call is compiled at module level, so the outer
        #     torch.compile is skipped when PA is active — the graph breaks from
        #     @torch.compiler.disable on PairwiseAttentionBias.forward create a
        #     large continuation graph that triggers Triton autotuner crashes on
        #     certain GPUs (A100, illegal memory access in fused FFN kernels).
        pa_enabled = config.get("architecture", {}).get("relative_bias", {}).get("type", "none") in ("pa", "alibi_pa")
        multi_gpu = self.accelerator.num_processes > 1
        if self.config.get("compile_model", False) and not pa_enabled and not multi_gpu:
            logger.info("Compiling model with torch.compile(mode='default')...")
            model = torch.compile(model, mode="default")
            logger.info("Model compiled successfully")
        elif pa_enabled:
            logger.info("Skipping torch.compile (PA bias uses FlexAttention with its own compilation)")
        elif multi_gpu:
            # torch.compile + DDP is unreliable for this model:
            # - With gradient_checkpointing: backward graph partitioner crashes
            # - Without gradient_checkpointing: NCCL hangs during validation
            #   (compiled graphs diverge across ranks on data-dependent paths)
            logger.info("Skipping torch.compile (torch.compile + DDP causes hangs on this model)")

        return model

    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup the optimizer.

        Uses Adam with configurable learning rate and weight decay.

        Returns:
            Adam optimizer.
        """
        from torch.optim.adamw import AdamW

        return AdamW(
            self.model.parameters(),
            lr=float(self.config.get("learning_rate", 5e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-5)),
            fused=True,
        )

    def setup_decoder(self) -> Decoder:
        """Setup the decoder.

        Foundation model is encoder-only, so no decoder is needed for training.
        Returns None as a placeholder.

        Returns:
            None (encoder-only model).
        """
        # Encoder-only model doesn't need a decoder
        return None  # type: ignore

    def setup_data_processors(self) -> tuple[DataProcessor, DataProcessor]:
        """Setup train and validation data processors.

        Both processors are configured for self-supervised learning (no sequences).
        Validation processor can optionally include sequences for evaluation.
        Metadata columns from dataset config are passed to processors for extraction.

        Returns:
            Tuple of (train_processor, valid_processor).
        """
        # Check if meta token is enabled (needed for filtering metadata columns)
        meta_token_enabled = self.config.model.get("meta_token", {}).get("enabled", False)

        # Extract metadata columns from dataset config
        metadata_columns = self.config.dataset.get("metadata_columns", None)

        # Filter to only required columns for meta token (performance optimization)
        # Most dataset configs list 30+ columns, but we only need 3:
        # - frag_type, collision_energy (from dataset)
        # - spectrum key for search_data lookup (e.g. "usi" or "filepath", from config)
        # Other fields (precursor_mz, precursor_charge, precursor_mass) are already in batch
        # Get masking config (with backwards compatibility)
        masking_cfg = self.config.model.get("masking", {})
        masking_strategy = masking_cfg.get("strategy", self.config.model.get("masking_strategy", "thompson_span"))
        required_metadata = set()
        if meta_token_enabled:
            spectrum_key = self.config.dataset.get("search_data_spectrum_key", "usi")
            required_metadata = {"frag_type", "collision_energy", spectrum_key}
        # Signal-aware masking needs frag_type for CID Da tolerance
        if masking_strategy == "signal_aware_fragment":
            required_metadata.add("frag_type")
        if required_metadata and metadata_columns:
            metadata_columns_filtered = [col for col in metadata_columns if col in required_metadata]
            # Ensure required columns are present even if not in original metadata_columns
            for col in required_metadata:
                if col not in metadata_columns_filtered and metadata_columns:
                    metadata_columns_filtered.append(col)
            logger.info(f"Filtered metadata_columns: {len(metadata_columns)} -> {len(metadata_columns_filtered)} columns")
        elif required_metadata and not metadata_columns:
            # No metadata_columns configured but we need some — create minimal list
            metadata_columns_filtered = list(required_metadata)
            logger.info(f"Auto-adding metadata_columns for required fields: {metadata_columns_filtered}")
        else:
            metadata_columns_filtered = metadata_columns

        # Setup search data manager if enabled
        search_data_config: dict[str, Any] = {
            "use_search_data": self.config.dataset.get("use_search_data", False),
            "search_data_path": self.config.dataset.get("search_data_path", None),
            "search_data_filepath_column": self.config.dataset.get("search_data_filepath_column", "file path"),
            "search_data_spectrum_key": self.config.dataset.get("search_data_spectrum_key", "filepath"),
        }
        search_data_manager = create_search_data_manager(search_data_config)

        if search_data_manager:
            logger.info(f"Search data integration enabled: {search_data_manager.summary()}")
        else:
            logger.info("Search data integration disabled")

        # Signal-aware masking needs peptide sequences during training
        train_needs_sequences = masking_strategy == "signal_aware_fragment"
        if train_needs_sequences:
            logger.info("Signal-aware masking enabled — training processor will extract sequences")

        # Training processor: self-supervised (no sequences, minimal metadata)
        train_processor = FoundationalDataProcessor(
            n_peaks=self.config.model.get("n_peaks", 200),
            min_mz=self.config.model.get("min_mz", 50.0),
            max_mz=self.config.model.get("max_mz", 2500.0),
            min_intensity=self.config.model.get("min_intensity", 0.01),
            remove_precursor_tol=self.config.model.get("remove_precursor_tol", 2.0),
            use_spectrum_utils=self.config.model.get("use_spectrum_utils", False),
            normalize_mz=self.config.model.get("normalize_mz", True),
            peak_ordering=masking_cfg.get("ordering_strategy", self.config.model.get("peak_ordering", "sorted")),
            residue_set=self.residue_set if train_needs_sequences else None,
            annotated=train_needs_sequences,  # Need sequences for signal-aware masking
            return_str=True,
            metadata_columns=metadata_columns_filtered if (meta_token_enabled or required_metadata) else None,
            # Masking configuration (look in masking section first, fall back to root)
            masking_strategy=masking_cfg.get("strategy", self.config.model.get("masking_strategy", "thompson_span")),
            mask_portion=masking_cfg.get("mask_portion", self.config.model.get("mask_portion", 0.30)),
            thompson_alpha=masking_cfg.get("alpha", self.config.model.get("thompson_alpha", 0.5)),
            thompson_beta=masking_cfg.get("beta", self.config.model.get("thompson_beta", 0.5)),
            thompson_kappa=masking_cfg.get("kappa", self.config.model.get("thompson_kappa", 4.0)),
            thompson_gamma=masking_cfg.get("gamma", self.config.model.get("thompson_gamma", 0.7)),
            span_min=masking_cfg.get("span_min", self.config.model.get("span_min", 4)),
            span_max=masking_cfg.get("span_max", self.config.model.get("span_max", 7)),
            span_bidirectional=masking_cfg.get("bidirectional", self.config.model.get("span_bidirectional", True)),
            # Isotope parameters (now under masking section, with backwards compatibility)
            include_isotopes=masking_cfg.get("include_isotopes", self.config.model.get("include_isotopes", True)),
            isotope_ppm=masking_cfg.get("isotope_ppm", self.config.model.get("isotope_ppm", 25.0)),
            isotope_da_floor=masking_cfg.get("isotope_da_floor", self.config.model.get("isotope_da_floor", 0.02)),
            isotope_max_charge=masking_cfg.get("isotope_max_charge", self.config.model.get("isotope_max_charge", 3)),
            isotope_max_order=masking_cfg.get("isotope_max_order", self.config.model.get("isotope_max_order", 2)),
            max_total_mask_ratio=masking_cfg.get("max_total_mask_ratio", self.config.model.get("max_total_mask_ratio", 0.40)),
            # Signal-aware masking parameters
            signal_min_backbone_coverage=masking_cfg.get("signal_min_backbone_coverage", 0.15),
            signal_min_fragment_groups=masking_cfg.get("signal_min_fragment_groups", 3),
            signal_ppm=masking_cfg.get("signal_ppm", 20.0),
            signal_cid_da_tol=masking_cfg.get("signal_cid_da_tol", 0.2),
            signal_ion_types=tuple(masking_cfg.get("signal_ion_types", ["b", "y"])),
            signal_num_workers=masking_cfg.get("signal_num_workers", 4),
            # Search data integration - disabled for training (only for validation)
            search_data_manager=search_data_manager if meta_token_enabled else None,
            # Metadata building - only if meta token enabled
            build_metadata=meta_token_enabled,
        )

        # Validation processor: includes sequences for analysis
        # For validation, we still use full metadata_columns (not filtered) to support analysis
        valid_processor = FoundationalDataProcessor(
            n_peaks=self.config.model.get("n_peaks", 200),
            min_mz=self.config.model.get("min_mz", 50.0),
            max_mz=self.config.model.get("max_mz", 2500.0),
            min_intensity=self.config.model.get("min_intensity", 0.01),
            remove_precursor_tol=self.config.model.get("remove_precursor_tol", 2.0),
            use_spectrum_utils=self.config.model.get("use_spectrum_utils", False),
            normalize_mz=self.config.model.get("normalize_mz", True),
            peak_ordering=masking_cfg.get("ordering_strategy", self.config.model.get("peak_ordering", "sorted")),
            residue_set=self.residue_set,
            annotated=True,  # Include sequences for analysis
            return_str=True,  # Keep sequences as strings
            metadata_columns=metadata_columns,  # Full columns for validation/analysis
            # Masking configuration (same as training)
            masking_strategy=masking_cfg.get("strategy", self.config.model.get("masking_strategy", "thompson_span")),
            mask_portion=masking_cfg.get("mask_portion", self.config.model.get("mask_portion", 0.30)),
            thompson_alpha=masking_cfg.get("alpha", self.config.model.get("thompson_alpha", 0.5)),
            thompson_beta=masking_cfg.get("beta", self.config.model.get("thompson_beta", 0.5)),
            thompson_kappa=masking_cfg.get("kappa", self.config.model.get("thompson_kappa", 4.0)),
            thompson_gamma=masking_cfg.get("gamma", self.config.model.get("thompson_gamma", 0.7)),
            span_min=masking_cfg.get("span_min", self.config.model.get("span_min", 4)),
            span_max=masking_cfg.get("span_max", self.config.model.get("span_max", 7)),
            span_bidirectional=masking_cfg.get("bidirectional", self.config.model.get("span_bidirectional", True)),
            # Isotope parameters (now under masking section, with backwards compatibility)
            include_isotopes=masking_cfg.get("include_isotopes", self.config.model.get("include_isotopes", True)),
            isotope_ppm=masking_cfg.get("isotope_ppm", self.config.model.get("isotope_ppm", 25.0)),
            isotope_da_floor=masking_cfg.get("isotope_da_floor", self.config.model.get("isotope_da_floor", 0.02)),
            isotope_max_charge=masking_cfg.get("isotope_max_charge", self.config.model.get("isotope_max_charge", 3)),
            isotope_max_order=masking_cfg.get("isotope_max_order", self.config.model.get("isotope_max_order", 2)),
            max_total_mask_ratio=masking_cfg.get("max_total_mask_ratio", self.config.model.get("max_total_mask_ratio", 0.40)),
            # Signal-aware masking parameters
            signal_min_backbone_coverage=masking_cfg.get("signal_min_backbone_coverage", 0.15),
            signal_min_fragment_groups=masking_cfg.get("signal_min_fragment_groups", 3),
            signal_ppm=masking_cfg.get("signal_ppm", 20.0),
            signal_cid_da_tol=masking_cfg.get("signal_cid_da_tol", 0.2),
            signal_ion_types=tuple(masking_cfg.get("signal_ion_types", ["b", "y"])),
            signal_num_workers=masking_cfg.get("signal_num_workers", 4),
            # Search data integration
            search_data_manager=search_data_manager,
            # Metadata building - only if meta token enabled
            build_metadata=meta_token_enabled,
        )

        return train_processor, valid_processor

    def save_model(self, is_best_checkpoint: bool = False) -> None:
        """Save model checkpoint.

        Saves model state, config, and training metadata.

        Args:
            is_best_checkpoint: Whether this is the best checkpoint so far.
        """
        if not self.accelerator.is_main_process:
            return

        checkpoint_dir = self.config.get("model_save_folder_path", "./checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Determine checkpoint path
        if self.config.get("keep_model_every_interval", False):
            model_path = os.path.join(checkpoint_dir, f"model_epoch_{self.epoch:02d}_step_{self.global_step + 1}.ckpt")
        else:
            model_path = os.path.join(checkpoint_dir, "model_latest.ckpt")
            if Path(model_path).exists() and Path(model_path).is_file():
                Path(model_path).unlink()

        # Unwrap model from accelerator
        unwrapped_model = self._unwrap_model()

        # Create checkpoint
        checkpoint_state: dict[str, Any] = {
            "state_dict": unwrapped_model.state_dict(),
            "config": OmegaConf.to_container(self.config.model),
            "residues": self.residue_set.residue_masses,
            "epoch": self.epoch,
            "global_step": self.global_step + 1,
        }

        # Save checkpoint
        torch.save(checkpoint_state, model_path)
        logger.info(f"Saved model to {model_path}")

        # Upload to S3 if enabled
        if S3FileHandler._aichor_enabled():
            self.s3.upload(model_path, S3FileHandler.convert_to_s3_output(model_path))

        # Save best checkpoint
        if is_best_checkpoint:
            best_model_path = os.path.join(checkpoint_dir, "model_best.ckpt")
            if Path(best_model_path).exists() and Path(best_model_path).is_file():
                Path(best_model_path).unlink()

            shutil.copy(model_path, best_model_path)
            logger.info(f"Saved best checkpoint to {best_model_path}")

            if S3FileHandler._aichor_enabled():
                self.s3.upload(best_model_path, S3FileHandler.convert_to_s3_output(best_model_path))

    # we don't track sequence-level metrics ⇒ disable Metrics object
    def setup_metrics(self) -> Any:
        """Set up metrics."""
        return None

    def forward(
        self, batch: Any, return_preds: bool = False
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]] | tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass for masked reconstruction loss.

        Computes loss between model predictions and ground truth
        for masked peak positions only. Supports both regression and classification tasks.

        Loss components are automatically logged to TensorBoard/Neptune during training
        under the 'train/' prefix by the base trainer.

        Args:
            batch: Batch dictionary containing:
                - spectra: Original input spectra (for targets)
                - spectra_mask: Padding mask
                - mlm_mask: Boolean mask indicating masked positions
                - meta: Optional metadata dictionary
            return_preds: If True, decode and return predictions (for validation).
                This enables single-pass validation without a second forward call.

        Returns:
            If return_preds=False (training):
                Tuple of (loss, loss_components)
            If return_preds=True (validation):
                Tuple of (loss, loss_components, pred_mz, aux_out) where:
                    - pred_mz: Decoded m/z predictions in Daltons (B, L)
                    - aux_out: Auxiliary outputs dict

            loss_components dict includes:
                * mlm_loss: Main m/z reconstruction loss
                * classification_loss: Task-specific loss (classification) or huber_loss (regression)
                * charge_loss, rt_nll, dmz_loss: Auxiliary task losses
        """
        # Import loss functions
        from instanovo_fm.trainer.losses import (
            compute_auxiliary_losses,
            compute_total_loss,
        )

        # Forward pass through model
        # Model takes spectra and applies masking internally based on peak_mask
        # Note: Data processor creates 'peak_mask', which is the MLM mask
        mlm_mask = batch.get("peak_mask", batch.get("mlm_mask", None))

        # Wrap both model forward and loss computation in autocast for mixed precision
        # This ensures all heavy compute (attention, MLPs, softmax, logsumexp) runs in fp16/bf16
        # Matches validation behavior and prevents fp32 fallback
        with self.accelerator.autocast():
            # Compute target_groups/offsets for teacher forcing (offset conditioning)
            target_groups = None
            target_offsets = None
            if self._offset_conditioning != "none":
                from instanovo_fm.trainer.utils import mz_to_bin_groups

                max_mz = self.config.model.get("max_mz", 2500.0)
                min_mz = self.config.model.get("min_mz", 0.0)
                target_mz_da = batch["spectra"][:, :, 0] * max_mz
                target_groups, target_offsets = mz_to_bin_groups(
                    target_mz_da,
                    self._mz_bin_size,
                    max_mz,
                    self._mz_group_size,
                    min_mz,
                    bin_edges=self._bin_edges,
                )

            preds, aux_out = self.model(
                spectra=batch["spectra"],
                spectra_mask=batch.get("spectra_mask", None),
                mlm_mask=mlm_mask,
                meta=batch.get("meta", None),
                target_groups=target_groups,
                target_offsets=target_offsets,
                bin_edges=self._bin_edges,
            )

            # Extract targets (original spectra)
            targets = batch["spectra"]  # (B, L, 2) [m/z, intensity]
            target_mz = targets[:, :, 0]  # (B, L) [m/z in 0-1 range]

            # Get masks
            padding_mask = batch.get("spectra_mask", None)  # (B, L), True for padding

            # If no MLM mask provided, we can't compute loss
            if mlm_mask is None:
                raise ValueError("peak_mask (MLM mask) must be provided for training")

            # Valid positions: masked but not padded
            if padding_mask is not None:
                valid_mask = mlm_mask & ~padding_mask
            else:
                valid_mask = mlm_mask

            # Get m/z task configuration
            mz_head_cfg = self.config.model.get("mz_head", {})
            mz_task = mz_head_cfg.get("task", "classification")

            # Compute main MLM loss (classification only)
            if mz_task != "classification":
                raise ValueError(f"Only classification task is supported, got mz_task={mz_task!r}")

            # Classification: bin the ground-truth m/z and apply focal loss
            if not isinstance(preds, tuple):
                raise RuntimeError(f"Expected tuple for classification task, got {type(preds)}")

            # Use cached bin parameters (computed once in __init__)
            assert self._mz_group_size is not None, "group_size must be set for classification task"
            assert self._mz_n_groups is not None, "n_groups must be set for classification task"
            assert self._mz_last_group is not None, "last_group_size must be set for classification task"

            group_size = self._mz_group_size
            n_groups = self._mz_n_groups
            last_group_size = self._mz_last_group

            cls_loss_fn = self._get_compiled_classification_loss()
            total_loss, classification_loss, group_loss, offset_loss = cls_loss_fn(
                preds,
                target_mz,
                valid_mask,
                self.config.model,
                group_size,
                n_groups,
                last_group_size,
                bin_edges=self._bin_edges,
            )

            # Loss components for logging
            loss_components: dict[str, Any] = {
                "classification_loss": classification_loss.detach(),
                "mlm_loss": total_loss.detach(),
                "group_ce_loss": group_loss.detach(),
                "offset_ce_loss": offset_loss.detach(),
            }

            # Auxiliary losses (if enabled)
            aux_config = self.config.model.get("auxiliary", {})
            if aux_config.get("enabled", False):
                charge_loss, rt_loss, dmz_loss, ptm_loss, intensity_loss, aux_loss = compute_auxiliary_losses(
                    aux_out, batch, aux_config, self.config.model
                )
            else:
                # Auxiliary tasks are disabled - use zero loss
                device = total_loss.device
                charge_loss = torch.tensor(0.0, device=device)
                rt_loss = torch.tensor(0.0, device=device)
                dmz_loss = torch.tensor(0.0, device=device)
                ptm_loss = torch.tensor(0.0, device=device)
                intensity_loss = torch.tensor(0.0, device=device)
                aux_loss = torch.tensor(0.0, device=device)

            # Add auxiliary losses to loss_components
            loss_components.update(
                {
                    "charge_loss": charge_loss.detach(),
                    "rt_nll": rt_loss.detach(),
                    "dmz_loss": dmz_loss.detach(),
                    "ptm_loss": ptm_loss.detach(),
                    "intensity_loss": intensity_loss.detach(),
                }
            )

            # Combine all losses
            total_loss = compute_total_loss(total_loss, aux_loss)
            loss_components["total_loss"] = total_loss.detach()

            # If requested, decode predictions for validation (single-pass approach)
            if return_preds:
                # Get max_mz for denormalization
                max_mz = self.config.model.get("max_mz", 2500.0)

                if mz_task == "classification":
                    # Classification: decode grouped logits to m/z values
                    from instanovo_fm.trainer.utils import bin_groups_to_mz

                    # Use cached bin parameters
                    assert self._mz_group_size is not None, "group_size must be set for classification task"
                    assert self._mz_n_groups is not None, "n_groups must be set for classification task"
                    assert self._mz_last_group is not None, "last_group_size must be set for classification task"

                    group_size = self._mz_group_size
                    n_groups = self._mz_n_groups
                    last_group_size = self._mz_last_group
                    min_mz = self.config.model.get("min_mz", 0.0)

                    group_logits, offset_logits = preds

                    eval_topk = self._eval_topk_groups
                    use_topk = eval_topk > 1 and self._offset_conditioning != "none" and "x_tokens" in aux_out

                    if use_topk:
                        # Top-K joint decoding: recompute offset logits per candidate group
                        unwrapped = self._unwrap_model()
                        mz_head = unwrapped.prediction_heads.mz_head

                        top_k_probs, top_k_groups = torch.topk(F.softmax(group_logits, dim=-1), eval_topk, dim=-1)  # (B, L, K)

                        x_tok = aux_out["x_tokens"]
                        best_mz = None
                        best_groups = None
                        best_offsets = None
                        best_joint_prob = torch.full(
                            group_logits.shape[:2],
                            -1.0,
                            device=group_logits.device,
                        )

                        for k in range(eval_topk):
                            g_k = top_k_groups[:, :, k]
                            g_prob_k = top_k_probs[:, :, k]

                            # Recompute offset logits conditioned on this group
                            g_embed_k = mz_head.group_embedding(g_k)
                            offset_input_k = torch.cat([x_tok, g_embed_k], dim=-1)
                            olog_k = mz_head.offset_head(offset_input_k)

                            # Mask invalid offsets for last group
                            is_last_k = g_k.eq(n_groups - 1)
                            if is_last_k.any():
                                bi, li = is_last_k.nonzero(as_tuple=True)
                                olog_k[bi, li, last_group_size:] = -float("inf")

                            o_prob_k, o_k = F.softmax(olog_k, dim=-1).max(dim=-1)
                            joint_prob = g_prob_k * o_prob_k
                            mz_k = bin_groups_to_mz(
                                g_k,
                                o_k,
                                self._mz_bin_size,
                                group_size,
                                min_mz,
                                bin_edges=self._bin_edges,
                            )

                            better = joint_prob > best_joint_prob
                            if best_mz is None:
                                best_mz = mz_k
                                best_groups = g_k
                                best_offsets = o_k
                                best_joint_prob = joint_prob
                            else:
                                best_mz = torch.where(better, mz_k, best_mz)
                                best_groups = torch.where(better, g_k, best_groups)
                                best_offsets = torch.where(better, o_k, best_offsets)
                                best_joint_prob = torch.where(better, joint_prob, best_joint_prob)

                        pred_mz_bin = best_mz
                    else:
                        # Greedy decoding (default, also used when conditioning="none")
                        pred_groups = group_logits.argmax(dim=-1)  # (B, L)

                        # Mask invalid offsets for last group before argmax
                        offset_logits_masked = offset_logits.clone()
                        is_last = pred_groups.eq(n_groups - 1)  # (B, L)
                        if is_last.any():
                            b_idx, l_idx = is_last.nonzero(as_tuple=True)
                            offset_logits_masked[b_idx, l_idx, last_group_size:] = -float("inf")

                        pred_offsets = offset_logits_masked.argmax(dim=-1)  # (B, L)

                        pred_mz_bin = bin_groups_to_mz(
                            pred_groups,
                            pred_offsets,
                            self._mz_bin_size,
                            group_size,
                            min_mz,
                            bin_edges=self._bin_edges,
                        )  # (B, L) in Da

                    pred_mz = pred_mz_bin

                    # Compute classifier confidence metrics (conf_group/offset/joint).
                    # Consumed by StreamingMetrics to emit conf_joint_mean + ECE.
                    if mz_head_cfg.get("confidence", {}).get("enabled", True):
                        from instanovo_fm.trainer.losses import compute_classifier_confidence

                        confidence_dict = compute_classifier_confidence(
                            group_logits=group_logits,
                            offset_logits=offset_logits,
                            n_groups=n_groups,
                            last_group_size=last_group_size,
                        )
                        aux_out.update(confidence_dict)

                return total_loss, loss_components, pred_mz, aux_out

        return total_loss, loss_components

    def get_predictions(self, batch: Any) -> tuple[list[str] | list[list[str]], list[str] | list[list[str]]]:
        """Get predictions for validation.

        For foundation model, this could return:
            - Reconstructed spectra for visualization
            - Downstream task predictions (if available)
            - Empty lists (self-supervised learning doesn't need predictions)

        Args:
            batch: Validation batch.

        Returns:
            Tuple of (predictions, targets). Returns empty lists for now.
        """
        # Foundation model doesn't have sequence predictions during training
        # Could be extended for downstream evaluation tasks

        # Option 1: Return empty lists (self-supervised, no predictions needed)
        return [], []

        # Option 2: Return reconstruction quality metrics (when model is ready)
        # with torch.no_grad():
        #     preds = self.model(
        #         spectra=batch["masked_spectra"],
        #         precursors=batch["precursors"],
        #         spectra_mask=batch["spectra_mask"],
        #     )
        #     # Could compute per-sample reconstruction error
        #     # Or return embeddings for analysis
        #     return [], []

    def validate_epoch(self, num_sanity_steps: int | None = None, calculate_metrics: bool = True) -> None:
        """Validate for one epoch.

        Foundation model validation computes m/z reconstruction metrics instead of
        sequence-based metrics. Tracks MAE and median PPM error.

        Metrics logged to MLflow under 'eval/' prefix:
        - Core metrics: loss, mae_daltons, median_ae_ppm
        - Percentage metrics: pct_within_10ppm, pct_within_20ppm, pct_within_01da
        - Auxiliary metrics: charge_accuracy, rt_mae_seconds, dmz_accuracy, intensity_accuracy
        - Representation health: rep/pca_energy_top1, rep/embedding_norm_mean, etc.

        Args:
            num_sanity_steps: If provided, only validate for this many steps (for sanity checks).
            calculate_metrics: If True, compute and log metrics. If False, only run forward pass.
        """
        if self.valid_dataloader is None:
            return

        # Save training RNG state so we can restore it after validation.
        # This prevents the fixed validation seed from altering training randomness.
        _rng_state_cpu = torch.random.get_rng_state()
        _rng_state_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        _rng_state_python = random.getstate()

        # Fix masking RNG to ensure identical validation masks across training runs.
        # Without this, Thompson/span masking uses the global PyTorch RNG whose state
        # depends on how many random ops occurred during training, making validation
        # metrics non-comparable between ablation runs.
        # Seeds torch (for Beta/randint in Thompson masking), Python random (for
        # random.shuffle in signal_aware_fragment masking).
        validation_seed = int(self.config.get("validation_seed", 42))
        torch.manual_seed(validation_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(validation_seed)
        random.seed(validation_seed)

        if self.accelerator.is_main_process:
            logger.info(f"[VALIDATION] [Epoch {self.epoch:02d}] Starting validation.")

        valid_epoch_step = 0
        valid_prediction_ids: list[int] = []

        # Foundation model metrics: track loss and m/z reconstruction quality
        from instanovo_fm.trainer.metrics import StreamingMetrics

        metrics_tracker = StreamingMetrics()

        # Cache config values to avoid repeated lookups in hot loop
        max_mz = self.config.model.get("max_mz", 2500.0)
        mz_task = self.config.model.get("mz_head", {}).get("task", "classification")
        aux_config = self.config.model.get("auxiliary", {})
        aux_enabled = aux_config.get("enabled", False)

        try:
            num_batches = len(self.valid_dataloader)
        except TypeError:
            # IterableDataset-backed dataloaders don't support len()
            num_batches = None

        # Cap validation steps to avoid hours-long validation on large streaming datasets
        max_valid_steps = self.config.get("max_valid_steps", None)
        if num_sanity_steps is not None:
            effective_max_steps = num_sanity_steps
        elif max_valid_steps is not None:
            effective_max_steps = max_valid_steps
        else:
            effective_max_steps = None

        timer_steps = effective_max_steps or num_batches or 0
        from instanovo_fm.common.utils import Timer

        valid_timer = Timer(timer_steps)

        for batch_idx, batch in enumerate(self.valid_dataloader):
            if effective_max_steps is not None and batch_idx >= effective_max_steps:
                break

            with torch.inference_mode(), self.accelerator.autocast():
                # Single-pass validation: compute loss AND decode predictions in one forward call
                # This eliminates the redundant second forward pass for metrics
                with profile_component(self.profiler, "val_forward"):
                    loss, loss_components, pred_mz, aux_out = self.forward(batch, return_preds=True)  # type: ignore[misc]

            # Track prediction IDs if available
            if "prediction_id" in batch:
                valid_prediction_ids.extend([x.item() if hasattr(x, "item") else x for x in batch["prediction_id"]])

            # Update metrics tracker
            with profile_component(self.profiler, "val_metrics"):
                metrics_tracker.update_loss(loss.item())

                # Update m/z reconstruction metrics
                targets = batch["spectra"]
                tgt_mz = targets[:, :, 0].float() * max_mz  # (B, L) in Daltons
                padding_mask = batch.get("spectra_mask", None)
                mlm_mask = batch.get("peak_mask", batch.get("mlm_mask", None))

                if padding_mask is not None:
                    valid_mask = mlm_mask & ~padding_mask
                else:
                    valid_mask = mlm_mask

                metrics_tracker.update_mz_metrics(pred_mz, tgt_mz, valid_mask)

                # Track bin-only metrics for delta comparison
                if "pred_mz_bin" in aux_out:
                    metrics_tracker.update_mz_metrics_bin_only(aux_out["pred_mz_bin"], tgt_mz, valid_mask)

                # Update entropy metrics for epistemic uncertainty (classification only)
                # Use cached mz_task from loop initialization
                if mz_task == "classification" and "group_logits" in aux_out and "offset_logits" in aux_out:
                    metrics_tracker.update_entropy_metrics(aux_out["group_logits"], aux_out["offset_logits"], valid_mask)

                # Compute classification targets (shared by confidence + annotated/unannotated)
                group_targets = None
                offset_targets = None
                if mz_task == "classification" and "group_logits" in aux_out:
                    from instanovo_fm.trainer.utils import mz_to_bin_groups

                    assert self._mz_group_size is not None, "group_size must be set for classification task"
                    min_mz = self.config.model.get("min_mz", 0.0)

                    group_targets, offset_targets = mz_to_bin_groups(
                        tgt_mz, self._mz_bin_size, max_mz, self._mz_group_size, min_mz, bin_edges=self._bin_edges
                    )

                # Update bin accuracy (unconditional, streaming counters)
                if group_targets is not None and "group_logits" in aux_out:
                    metrics_tracker.update_bin_accuracy(aux_out["group_logits"], aux_out["offset_logits"], group_targets, offset_targets, valid_mask)

                # Update confidence metrics if available
                if mz_task == "classification" and "conf_group" in aux_out and group_targets is not None:
                    metrics_tracker.update_confidence_metrics(
                        aux_out, aux_out["group_logits"], aux_out["offset_logits"], group_targets, offset_targets, valid_mask
                    )

                # Update auxiliary metrics if enabled
                # Use cached aux_enabled and aux_config from loop initialization
                if aux_enabled and aux_out is not None:
                    metrics_tracker.update_auxiliary_metrics(aux_out, batch, aux_config)

            valid_epoch_step += 1
            valid_timer.step()

            # Log progress
            if (valid_epoch_step + 1) % int(self.config.get("console_logging_steps", 2000)) == 0:
                total_display = effective_max_steps or num_batches
                batch_progress = (
                    f"[Batch {valid_epoch_step:05d}/{total_display:05d}]" if total_display is not None else f"[Batch {valid_epoch_step:05d}]"
                )

                logger.info(
                    f"[VALIDATION] "
                    f"[Epoch {self.epoch:02d}] "
                    f"[Step {self.global_step + 1:06d}] "
                    f"{batch_progress} "
                    f"[{valid_timer.get_time_str()}/{valid_timer.get_total_time_str()}, "
                    f"{valid_timer.get_step_time_rate_str()}]"
                )

        # Synchronize all processes at the end of validation
        self.accelerator.wait_for_everyone()

        if not calculate_metrics:
            # Restore training RNG state before early return
            torch.random.set_rng_state(_rng_state_cpu)
            if _rng_state_cuda is not None:
                torch.cuda.set_rng_state_all(_rng_state_cuda)
            random.setstate(_rng_state_python)
            return

        # Gather streaming metric accumulators across all ranks.
        # With split_batches=True each rank processes a disjoint subset of
        # validation data; this ensures scalar metrics (counts, sums, averages)
        # reflect 100% of the validation set.  No-op when num_processes == 1.
        metrics_tracker.gather_across_ranks(self.accelerator)

        # Gather prediction IDs from all devices if available
        if valid_prediction_ids:
            import numpy as np

            self.log_if_verbose("Gathering prediction IDs from all devices")
            valid_prediction_ids = self.accelerator.gather_for_metrics(valid_prediction_ids)  # type: ignore[assignment]

            # Use valid_prediction_ids to remove duplicates
            _, idx = np.unique(valid_prediction_ids, return_index=True)
            self.log_if_verbose(f"Gathered {len(idx)} unique predictions")

        # Compute final metrics
        metrics = metrics_tracker.compute_final_metrics()

        # Update checkpoint metric for best-model tracking.
        # The base trainer sets this in its validate_epoch, but we override it.
        checkpoint_metric_name = self.config.get("checkpoint_metric", None)
        if checkpoint_metric_name is not None and checkpoint_metric_name in metrics:
            self.last_validation_metric = metrics[checkpoint_metric_name]

        # Log validation metrics to TensorBoard/Neptune
        if self.accelerator.is_main_process:
            # Validation metrics are logged at current step
            validation_step = self.global_step + 1

            if self.tracker is not None:
                # Log all metrics with eval/ prefix for consistency
                for k, v in metrics.items():
                    self.tracker.log_scalar(f"eval/{k}", v, validation_step)

            # Log to console — compact summary for monitoring
            log_msg = f"[VALIDATION] [Epoch {self.epoch:02d}] [Step {self.global_step + 1:06d}] Loss: {metrics['loss']:.4f}"

            if "bin_accuracy" in metrics:
                log_msg += f" | BinAcc: {metrics['bin_accuracy']:.1f}% | Grp: {metrics['group_accuracy']:.1f}% Off: {metrics['offset_accuracy']:.1f}%"

            log_msg += f" | <=20ppm: {metrics['pct_within_20ppm']:.1f}%"

            if "intensity_r2" in metrics:
                log_msg += f" | IntR2: {metrics['intensity_r2']:.3f}"

            log_msg += f" | {metrics['total_spectra']:,d} spec"

            logger.info(log_msg)

        self.accelerator.wait_for_everyone()

        # NOTE: Do NOT call torch.cuda.empty_cache() here.
        # With torch.compile, clearing the CUDA cache between validation and
        # training forces Triton kernel recompilation into a fresh allocator,
        # causing severe memory fragmentation and eventual OOM.

        # Restore training RNG state so training randomness is unaffected by
        # the fixed validation seed.
        torch.random.set_rng_state(_rng_state_cpu)
        if _rng_state_cuda is not None:
            torch.cuda.set_rng_state_all(_rng_state_cuda)
        random.setstate(_rng_state_python)

    def log_training_metrics(self, loss: torch.Tensor, loss_components: dict[str, torch.Tensor], lr: float, step: int) -> None:
        """Log training metrics to TensorBoard/Neptune.

        This method is called during training to log loss components,
        learning rate, and other training metrics.

        Args:
            loss: Total loss value
            loss_components: Dictionary of individual loss components
            lr: Current learning rate
            step: Global training step
        """
        if not self.accelerator.is_main_process or self.tracker is None:
            return

        # Log main loss
        self.tracker.log_scalar("train/loss", loss.item(), step)

        # Log loss components
        for k, v in loss_components.items():
            if isinstance(v, torch.Tensor):
                self.tracker.log_scalar(f"train/{k}", v.item(), step)
            else:
                self.tracker.log_scalar(f"train/{k}", v, step)

    def update_vocab(self, model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Update model vocabulary.

        Foundation model doesn't have vocabulary (no sequence prediction).
        Returns state unchanged.

        Args:
            model_state: Model state dict.

        Returns:
            Unchanged model state.
        """
        # No vocabulary in encoder-only model
        logger.info("Foundation model has no vocabulary to update")
        return model_state

    def train_epoch(self) -> None:
        """Train the model for one epoch with comprehensive profiling support.

        global_step counts optimizer updates, not forward passes. All step-based
        logic (logging, validation, checkpointing) runs only when sync_gradients is
        True (i.e. when a real optimizer update occurred). This means training_steps
        always refers to the number of optimizer updates regardless of grad_accumulation.

        This overrides the base trainer's train_epoch to add detailed profiling
        for backward pass, optimizer steps, scheduler, and dataloader operations
        that are not instrumented in the base class.
        """
        total_loss = 0

        self.model.train()
        self.optimizer.zero_grad()
        self.running_loss = None

        from instanovo_fm.common.utils import Timer

        epoch_timer = Timer()

        print_batch_size = True

        # Manual iteration to support IterableDataset and enable dataloader profiling
        dataloader_iter = iter(self.train_dataloader)  # type: ignore[call-overload]
        batch_count = 0
        step_start = time.perf_counter()

        while True:
            try:
                with profile_component(self.profiler, "data_loading"):
                    batch = next(dataloader_iter)
            except StopIteration:
                break

            if print_batch_size:
                self.log_if_verbose(f"Batch {batch_count} shape: {batch['spectra'].shape[0]}")
                print_batch_size = False

            batch = self.prepare_batch(batch)

            _is_optimizer_step = False
            with self.accelerator.accumulate(self.model):
                with profile_component(self.profiler, "forward"):
                    result = self.forward(batch)
                    if isinstance(result, tuple) and len(result) == 2:
                        loss, loss_components = result
                    else:
                        raise ValueError(f"Unexpected forward() return: {type(result)}")

                with profile_component(self.profiler, "backward"):
                    self.accelerator.backward(loss)

                with profile_component(self.profiler, "optimizer"):
                    _is_optimizer_step = self.accelerator.sync_gradients
                    if _is_optimizer_step:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.get("gradient_clip_val", 10.0))
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad()

                # Advance global_step only on real optimizer updates
                if _is_optimizer_step:
                    self.training_state.step()

            # Record step time (every forward pass for accurate throughput measurement)
            if self.profiler.cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            step_end = time.perf_counter()
            self.profiler.record_step_time(step_end - step_start)
            step_start = step_end

            # Update timer (every forward pass for accurate ETA)
            self.train_timer.step()

            # Update running loss (every forward pass)
            if self.running_loss is None:
                self.running_loss = loss.item()
            else:
                self.running_loss = 0.99 * self.running_loss + 0.01 * loss.item()

            total_loss += loss.item()
            batch_count += 1

            # Step profiler (every forward pass)
            if self.profiler.enabled:
                self.profiler.step()

            # Skip step-based logic on accumulation-only forward passes
            if not _is_optimizer_step:
                continue

            # --- All logic below runs once per optimizer update ---

            # Log progress
            if self.global_step % int(self.config.get("console_logging_steps", 2000)) == 0:
                lr = self.lr_scheduler.get_last_lr()[0]

                logger.info(
                    f"[TRAIN] "
                    f"[Epoch {self.epoch:02d}] "
                    f"[Step {self.global_step:06d}/{self.total_steps:06d}] "
                    f"[{self.train_timer.get_time_str()}/{self.train_timer.get_eta_str()}, "
                    f"{self.train_timer.get_step_time_rate_str()}]: "
                    f"train_loss_raw={loss.item():.4f}, "
                    f"running_loss={self.running_loss:.4f}, LR={lr:.6f}"
                )

            # Log to MLflow
            if (
                self.accelerator.is_main_process
                and self.tracker is not None
                and self.global_step % int(self.config.get("metrics_logging_steps", 500)) == 0
            ):
                lr = self.lr_scheduler.get_last_lr()[0]
                self.tracker.log_scalar("train/loss_raw", loss.item(), self.global_step)
                if self.running_loss is not None:
                    self.tracker.log_scalar("train/loss_smooth", self.running_loss, self.global_step)
                # Core loss components always logged
                always_log = {"classification_loss", "mlm_loss", "total_loss", "group_ce_loss", "offset_ce_loss"}
                for k, v in loss_components.items():
                    if k == "loss":
                        continue
                    val = v.item()
                    # Skip zero-valued non-core loss components
                    if val == 0.0 and k not in always_log:
                        continue
                    self.tracker.log_scalar(f"train/{k}", val, self.global_step)
                self.tracker.log_scalar("optim/lr", lr, self.global_step)
                self.tracker.log_scalar("optim/epoch", self.epoch, self.global_step)

            # Validation
            if self.global_step % self.steps_per_validation == 0:
                self.model.eval()
                self.validate_epoch()

                # Check if we should run embedding evaluation
                embed_eval_config = self.config.get("embedding_evaluation", {})
                embed_eval_interval = embed_eval_config.get("interval", None)

                if embed_eval_interval and self.global_step % embed_eval_interval == 0:
                    logger.info("Running embedding evaluation at specified interval...")
                    self.run_embedding_evaluation()
                    # run_embedding_evaluation() returns immediately on non-main ranks.
                    # Barrier here prevents rank 1 from advancing to save_accelerator_state()
                    # (which calls collectives) while rank 0 is still in the evaluator.
                    self.accelerator.wait_for_everyone()

                logger.info("Validation complete, resuming training...")
                self.model.train()

            # Save checkpoint
            if self.global_step % self.steps_per_checkpoint == 0:
                is_best_checkpoint = self.check_if_best_checkpoint()
                self.save_model(is_best_checkpoint)
                # save_model() returns immediately on non-main ranks. Without this
                # barrier rank 1 races to the next training collective while rank 0
                # is still uploading the checkpoint to S3, causing an NCCL timeout.
                self.accelerator.wait_for_everyone()
                if self.config.get("save_accelerator_state", False):
                    self.save_accelerator_state(is_best_checkpoint)

            # Update finetuning scheduler (if enabled)
            if self.finetune_scheduler is not None:
                self.finetune_scheduler.step(self.global_step)

            # Check if training is complete
            if self.global_step >= self.total_steps:
                break

        # Epoch complete - synchronize all processes
        self.accelerator.wait_for_everyone()

        epoch_timer.step()

        # Gather losses from all devices for logging
        gathered_losses = self.accelerator.gather_for_metrics(torch.tensor(total_loss, device=self.accelerator.device))
        gathered_num_batches = self.accelerator.gather_for_metrics(torch.tensor(batch_count, device=self.accelerator.device))

        if self.accelerator.is_main_process and self.tracker is not None:
            # Sum the losses and batch counts from all devices
            # Convert to tensor if needed (gather_for_metrics may return list/dict in some accelerate versions)
            if not isinstance(gathered_losses, torch.Tensor):
                gathered_losses = torch.tensor(gathered_losses, device=self.accelerator.device)
            if not isinstance(gathered_num_batches, torch.Tensor):
                gathered_num_batches = torch.tensor(gathered_num_batches, device=self.accelerator.device)

            total_loss_all_devices = gathered_losses.sum().item()
            total_batches_all_devices = gathered_num_batches.sum().item()
            avg_loss = total_loss_all_devices / total_batches_all_devices

            self.tracker.log_scalar("eval/train_loss", avg_loss, self.epoch)

            logger.info(f"[TRAIN] [Epoch {self.epoch:02d}] Epoch complete, total time {epoch_timer.get_time_str()}")

    def run_embedding_evaluation(self) -> None:
        """Run comprehensive embedding evaluation using EmbeddingEvaluator.

        This method leverages the existing EmbeddingEvaluator infrastructure
        while reusing the already-loaded validation dataloader for efficiency.

        Results are saved to a timestamped subdirectory and key metrics are logged
        to TensorBoard/Neptune under the 'eval/embed/' prefix.
        """
        if not self.accelerator.is_main_process:
            return

        embed_eval_config = self.config.get("embedding_evaluation", {})
        if not embed_eval_config.get("enabled", False):
            return

        if self.valid_dataloader is None:
            logger.warning("Validation dataloader not available, skipping embedding evaluation")
            return

        logger.info("=" * 80)
        logger.info("Running comprehensive embedding evaluation...")
        logger.info("=" * 80)

        try:
            from instanovo_fm.eval import embedding_io
            from instanovo_fm.eval.evaluator import EmbeddingEvaluator

            # Create evaluator config with step-specific output directory
            eval_config_dict: dict[str, Any] = {
                "evaluation": OmegaConf.to_container(self.config.evaluation, resolve=True),
                "dataset": OmegaConf.to_container(self.config.dataset, resolve=True),
                "residues": OmegaConf.to_container(self.config.residues, resolve=True),
                "num_workers": self.config.get("num_workers", 4),
                "model_save_folder_path": self.config.get("model_save_folder_path", "./checkpoints"),
            }

            # Override output directory to organize by training step
            base_output_dir = Path(eval_config_dict["evaluation"].get("output_dir", "./evaluation_results"))
            step_output_dir = base_output_dir / f"step_{self.global_step + 1:06d}"
            eval_config_dict["evaluation"]["output_dir"] = str(step_output_dir)

            # Override tasks list from embedding_evaluation section (keeps training-time eval minimal)
            embed_tasks = embed_eval_config.get("tasks_to_run", None)
            if embed_tasks is not None:
                eval_config_dict["evaluation"]["tasks_to_run"] = list(embed_tasks)

            evaluator_config = OmegaConf.create(eval_config_dict)

            # Create evaluator
            evaluator = EmbeddingEvaluator(evaluator_config)

            # Use the in-training model directly (no checkpoint loading)
            unwrapped_model = self._unwrap_model()
            unwrapped_model.eval()
            evaluator.model = unwrapped_model
            evaluator.model_config = self.config.model
            n_params = sum(p.numel() for p in unwrapped_model.parameters())
            logger.info(f"Using in-training model at step {self.global_step + 1} ({n_params:,} parameters)")

            # Build a fresh dataloader with a smaller batch size to avoid OOM.
            # The training validation dataloader uses predict_batch_size (e.g. 1024)
            # which is too large for models with O(B*L*L) intermediate tensors
            # (e.g., PA pairwise bias). We reuse the dataset and collate_fn from
            # the existing dataloader to ensure correct preprocessing.
            eval_batch_size = embed_eval_config.get("batch_size", 32)
            eval_dataloader = torch.utils.data.DataLoader(
                self.valid_dataloader.dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=self.valid_dataloader.collate_fn,
                num_workers=self.config.get("num_workers", 4),
                pin_memory=False,
            )
            evaluator.dataloader = eval_dataloader

            try:
                n_batches = len(eval_dataloader)
                logger.info(f"Eval dataloader: batch_size={eval_batch_size}, {n_batches} batches")
            except TypeError:
                logger.info("Using existing validation dataloader (streaming, unknown length)")

            # Determine pooling strategies to evaluate
            requested_pooling = evaluator.eval_config.get("embedding_pooling", "cls")
            if requested_pooling == "both":
                pooling_strategies = ["cls", "mean_pool"]
            else:
                pooling_strategies = [requested_pooling]

            all_results: dict[str, Any] = {}
            last_embeddings_info: dict[str, Any] = {}

            for strategy in pooling_strategies:
                tag = f"{strategy}/" if len(pooling_strategies) > 1 else ""
                if tag:
                    logger.info("-" * 40)
                    logger.info(f"Embedding evaluation: pooling={strategy}")
                    logger.info("-" * 40)

                # Override pooling strategy for this iteration
                from omegaconf import open_dict

                with open_dict(evaluator.eval_config):
                    evaluator.eval_config.embedding_pooling = strategy

                # Generate embeddings
                embeddings, metadata, faiss_index = evaluator.generate_embeddings(force_regenerate=True)

                # Get embedding statistics
                embeddings_info = embedding_io.get_embedding_stats(embeddings)
                embeddings_info["embedding_pooling"] = strategy
                logger.info(f"Embeddings ({strategy}): {embeddings_info['num_embeddings']} × {embeddings_info['embedding_dim']}")
                last_embeddings_info = embeddings_info

                # Run evaluation tasks with strategy-specific output subdirectory
                output_subdir = strategy if len(pooling_strategies) > 1 else None
                results = evaluator.run_evaluation_tasks(
                    embeddings,
                    metadata,
                    faiss_index,
                    output_subdir=output_subdir,
                )

                for task_name, task_results in results.items():
                    all_results[f"{tag}{task_name}"] = task_results

            # Restore original config value
            with open_dict(evaluator.eval_config):
                evaluator.eval_config.embedding_pooling = requested_pooling

            # Save configuration files
            evaluator._save_config_files()

            # Log metrics to TensorBoard/Neptune
            self._log_embedding_metrics(evaluator, all_results, last_embeddings_info)

            # Print summary
            evaluator.save_results(all_results, last_embeddings_info)

            logger.info("=" * 80)
            logger.info(f"Results saved to: {step_output_dir}")
            logger.info("=" * 80)

            # Restore model to train mode
            self.model.train()

        except Exception as e:
            logger.error(f"Embedding evaluation failed: {e}")
            logger.error("Training will continue...")
            import traceback

            traceback.print_exc()

            # Ensure model is back in train mode
            self.model.train()
        finally:
            # Clean up evaluator references to free CPU memory.
            # IMPORTANT: Do NOT call torch.cuda.empty_cache() here.
            # With torch.compile, clearing the CUDA cache mid-training forces
            # recompilation of Triton kernels into a fresh allocator, causing
            # severe memory fragmentation (5 GB → 85 GB → OOM). Let the CUDA
            # allocator keep its existing blocks for stable memory reuse.
            import gc

            if "evaluator" in dir():
                del evaluator
            gc.collect()

    def run_post_training_evaluation(self) -> None:
        """Run full embedding evaluation on the best checkpoint after training completes.

        This is separate from the periodic training-time evaluation
        (``run_embedding_evaluation``) and is intended to run once at the end
        of a training job.  It uses the standalone ``EmbeddingEvaluator.evaluate()``
        path, which supports multi-split tasks such as ``LinearProbeTask``.
        """
        if not self.accelerator.is_main_process:
            return

        post_eval_config = self.config.get("post_training_evaluation", {})
        if not post_eval_config.get("enabled", False):
            return

        checkpoint_dir = self.config.get("model_save_folder_path", "./checkpoints")
        best_checkpoint = os.path.join(checkpoint_dir, "model_best.ckpt")

        if not Path(best_checkpoint).exists():
            logger.warning(f"Best checkpoint not found at {best_checkpoint}, skipping post-training evaluation")
            return

        logger.info("Running post-training full evaluation on best checkpoint...")

        # Capture MLflow run ID before cleanup (cleanup ends the MLflow run)
        mlflow_run_id = None
        if self.tracker is not None and hasattr(self.tracker, "run_id"):
            mlflow_run_id = self.tracker.run_id
        training_step = self.global_step + 1

        # Free training model, optimizer, and dataloader workers to reclaim
        # GPU and CPU memory before loading the evaluator's own model copy.
        self.cleanup()
        del self.model, self.optimizer
        if hasattr(self, "lr_scheduler_obj"):
            del self.lr_scheduler_obj
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            from instanovo_fm.eval.evaluator import EmbeddingEvaluator

            eval_config_dict = {
                "evaluation": OmegaConf.to_container(self.config.evaluation, resolve=True),
                "dataset": OmegaConf.to_container(self.config.dataset, resolve=True),
                "residues": OmegaConf.to_container(self.config.residues, resolve=True),
                "num_workers": 0,  # Avoid /dev/shm exhaustion in Docker containers
                "model_save_folder_path": checkpoint_dir,
            }

            # Point at the best checkpoint saved during this training run;
            # remove checkpoint_paths (multi-checkpoint list from default.yaml)
            # so only the best model from this experiment is evaluated.
            eval_config_dict["evaluation"]["checkpoint_path"] = best_checkpoint
            eval_config_dict["evaluation"].pop("checkpoint_paths", None)

            # Override task list from post_training_evaluation config
            tasks_to_run = post_eval_config.get("tasks_to_run", None)
            if tasks_to_run is not None:
                eval_config_dict["evaluation"]["tasks_to_run"] = list(tasks_to_run)

            # Save results under a dedicated post_training subdirectory
            base_output_dir = Path(eval_config_dict["evaluation"].get("output_dir", "./evaluation_results"))
            eval_config_dict["evaluation"]["output_dir"] = str(base_output_dir / "post_training")

            evaluator_config = OmegaConf.create(eval_config_dict)
            evaluator = EmbeddingEvaluator(evaluator_config)
            results = evaluator.evaluate()

            # Log post-training metrics to MLflow under eval_post/ prefix.
            if mlflow_run_id and results:
                try:
                    import mlflow

                    embeddings_info = getattr(evaluator, "_last_embeddings_info", {}) or {}
                    loggable = evaluator.get_metrics_for_logging(results, embeddings_info)
                    logger.info(
                        f"MLflow post-eval: run_id={mlflow_run_id}, "
                        f"{len(loggable)} metrics to log, "
                        f"active_run={'yes' if mlflow.active_run() else 'no'}"
                    )
                    # The run may still be active (accelerate doesn't always end it).
                    # End any active run first, then reopen by ID to log metrics.
                    try:
                        mlflow.end_run()
                    except Exception:
                        pass
                    mlflow.set_tracking_uri(self.config.get("mlflow_tracking_uri", ""))
                    with mlflow.start_run(run_id=mlflow_run_id):
                        for name, value in loggable.items():
                            mlflow.log_metric(f"eval_post/{name}", value, step=training_step)
                    logger.info(f"Logged {len(loggable)} post-training metrics to MLflow run {mlflow_run_id}")
                except Exception as e:
                    logger.warning(f"Failed to log post-training metrics to MLflow: {e}")

            logger.info("Post-training evaluation complete!")

        except Exception as e:
            logger.error(f"Post-training evaluation failed: {e}")
            import traceback

            traceback.print_exc()
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _log_embedding_metrics(self, evaluator: Any, results: dict[str, Any], embeddings_info: dict[str, Any]) -> None:
        """Log embedding evaluation metrics to TensorBoard/Neptune.

        This method uses the evaluator's get_metrics_for_logging() to extract
        loggable metrics from task results. Each task defines its own metrics
        via get_loggable_metrics(), ensuring clean separation of concerns.

        Args:
            evaluator: EmbeddingEvaluator instance
            results: Dictionary of task results from evaluator
            embeddings_info: Dictionary of embedding statistics
        """
        if self.tracker is None:
            return

        validation_step = self.global_step + 1

        # Get all loggable metrics from evaluator (delegates to task.get_loggable_metrics())
        loggable_metrics = evaluator.get_metrics_for_logging(results, embeddings_info)

        # Log all metrics to MLflow
        for metric_name, metric_value in loggable_metrics.items():
            self.tracker.log_scalar(f"eval/{metric_name}", metric_value, validation_step)

    def cleanup(self) -> None:
        """Explicitly shut down dataloaders and accelerator to prevent hang on exit.

        With persistent_workers=True and pin_memory=True, Python's garbage
        collector cannot reliably clean up DataLoader worker processes and the
        pin-memory thread during interpreter shutdown.  Calling this method
        after training ensures workers are joined *before* the process exits.
        """
        logger.info("Cleaning up training resources...")

        # end_training() notifies experiment trackers and cleans up accelerator state
        self.accelerator.end_training()

        # free_memory() releases accelerator references and calls gc.collect()
        self.accelerator.free_memory()

        # Drop our own references so DataLoader.__del__ can run now
        # (while worker processes are still reachable)
        self.train_dataloader = None  # type: ignore[assignment]
        self.valid_dataloader = None  # type: ignore[assignment]

        import gc

        gc.collect()

        logger.info("Cleanup complete.")

    def train(self) -> None:
        """Train the model with profiling support."""
        # Start profiling session
        if self.profiler.enabled and self.accelerator.is_main_process:
            self.profiler.start_profiling()

        # Call parent train method
        super().train()

        # Stop profiling session and save results
        if self.profiler.enabled and self.accelerator.is_main_process:
            self.profiler.stop_profiling()

            # Upload profiling results to S3 if enabled
            if S3FileHandler._aichor_enabled():
                logger.info("Uploading profiling results to S3...")
                profiling_files = ["profiling_summary.json", "component_timing_detailed.json", "profiling_report.txt"]

                for filename in profiling_files:
                    local_file = self.profiling_output_dir / filename
                    if local_file.exists():
                        s3_path = S3FileHandler.convert_to_s3_output(str(local_file))
                        self.s3.upload(str(local_file), s3_path)
                        logger.info(f"Uploaded {filename} to {s3_path}")

                # Also upload torch profiler traces if they exist
                torch_traces = list(self.profiling_output_dir.glob("torch_trace_step_*.json"))
                torch_stats = list(self.profiling_output_dir.glob("torch_stats_step_*.txt"))

                for trace_file in torch_traces:
                    s3_path = S3FileHandler.convert_to_s3_output(str(trace_file))
                    self.s3.upload(str(trace_file), s3_path)

                for stats_file in torch_stats:
                    s3_path = S3FileHandler.convert_to_s3_output(str(stats_file))
                    self.s3.upload(str(stats_file), s3_path)

                if torch_traces or torch_stats:
                    logger.info(f"Uploaded {len(torch_traces)} trace files and {len(torch_stats)} stats files")

            # Print detailed profiling report to console
            report_file = self.profiling_output_dir / "profiling_report.txt"
            if report_file.exists():
                logger.info("=" * 80)
                logger.info("DETAILED PROFILING REPORT")
                logger.info("=" * 80)
                with open(report_file, "r") as f:
                    for line in f:
                        logger.info(line.rstrip())
                logger.info("=" * 80)

            # Log profiling summary
            timing_summary = self.profiler.get_timing_summary()
            if timing_summary:
                logger.info("Training profiling summary:")
                for component, stats in sorted(timing_summary.items(), key=lambda x: x[1]["total_s"], reverse=True):
                    logger.info(f"  {component}: {stats['mean_ms']:.2f}ms avg, {stats['total_s']:.2f}s total, {stats['count']} calls")


@hydra.main(config_path=str(CONFIG_PATH), version_base=None, config_name="foundational")
def main(config: DictConfig) -> None:
    """Main training entry point.

    Args:
        config: Hydra configuration loaded from foundational.yaml
    """
    logger.info("Initializing InstaNovo Foundation Model training")
    logger.info("Self-supervised learning via masked m/z reconstruction")

    try:
        trainer = FoundationalTrainer(config)
        trainer.train()

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)

        # Save MLflow run ID alongside checkpoint for post-training eval to pick up
        if trainer.tracker is not None and hasattr(trainer.tracker, "run_id"):
            checkpoint_dir = config.model.get("model_save_folder_path", "./checkpoints")
            run_id_path = os.path.join(checkpoint_dir, "mlflow_run_id.txt")
            with open(run_id_path, "w") as f:
                f.write(trainer.tracker.run_id)
            logger.info(f"Saved MLflow run ID to {run_id_path}")

        trainer.run_post_training_evaluation()

    except NotImplementedError as e:
        logger.error(f"Training failed: {e}")
        logger.error("Please implement the missing components before training")
        raise
    finally:
        # Explicit cleanup prevents DataLoader worker/pin_memory thread hang on exit
        if "trainer" in locals():
            trainer.cleanup()


if __name__ == "__main__":
    main()
