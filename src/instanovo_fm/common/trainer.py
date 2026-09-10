from __future__ import annotations

import datetime
import logging
import os
import shutil
import sys
from abc import ABCMeta, abstractmethod
from collections import Counter
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn as nn
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, DistributedDataParallelKwargs, InitProcessGroupKwargs, broadcast_object_list
from datasets import Dataset, Value
from datasets.utils.logging import disable_progress_bar
from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

from instanovo.__init__ import console, set_rank
from instanovo_fm.common.dataset import DataProcessor
from instanovo_fm.common.scheduler import CosineWarmupHoldScheduler, CosineWarmupScheduler, FinetuneScheduler, WarmupScheduler
from instanovo_fm.common.tracking import MLFlowTracker, create_tracker
from instanovo_fm.common.utils import (
    Timer,
    TrainingState,
    _get_filepath_mapping,
)
from instanovo.constants import ANNOTATION_ERROR, SHUFFLE_BUFFER_SIZE
from instanovo.inference import Decoder
from instanovo.utils.colorlogging import ColorLog
from instanovo_fm.utils.spectrum_dataframe import SpectrumDataFrame
from instanovo_fm.utils.git_utils import get_dataset_name_from_path, get_user
from instanovo.utils.metrics import Metrics
from instanovo.utils.residues import ResidueSet
from instanovo.utils.s3 import S3FileHandler

load_dotenv()

# Automatic rank logger
logger = ColorLog(console, __name__).logger


class AccelerateDeNovoTrainer(metaclass=ABCMeta):
    """Trainer class that uses the Accelerate library."""

    @property
    def run_id(self) -> str:
        """Get the run ID.

        Returns:
            str: The run ID
        """
        return str(self._run_id)

    @property
    def s3(self) -> S3FileHandler:
        """Get the S3 file handler.

        Returns:
            S3FileHandler: The S3 file handler
        """
        return self._s3

    @property
    def global_step(self) -> int:
        """Get the current global training step.

        This represents the total number of training steps across all epochs.

        Returns:
            int: The current global step number
        """
        return int(self._training_state.global_step)

    @property
    def epoch(self) -> int:
        """Get the current training epoch.

        This represents the current epoch number in the training process.

        Returns:
            int: The current epoch number
        """
        return int(self._training_state.epoch)

    @property
    def training_state(self) -> TrainingState:
        """Get the training state."""
        return self._training_state

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        self.config = config
        self.enable_verbose_logging = self.config.get("enable_verbose_logging", True)
        if not self.config.get("enable_verbose_accelerate", True):
            logging.getLogger("accelerate").setLevel(logging.WARNING)

        # Hide progress bar from HF datasets
        disable_progress_bar()

        # Training state
        # Keeps track of the global step and epoch
        # Used for accelerate training state checkpointing
        self._training_state = TrainingState()

        # `or` rather than a .get() default, for the same reason as the predictor:
        # a config that declares `run_name:` with no value yields None, not the
        # default. Latent here only because the foundational configs set a value.
        self._run_id = (self.config.get("run_name") or "instanovo") + datetime.datetime.now().strftime("_%y_%m_%d_%H_%M")

        self.accelerator = self.setup_accelerator()

        self.log_if_verbose("Verbose logging enabled")

        if self.accelerator.is_main_process:
            logger.info(f"Config:\n{OmegaConf.to_yaml(self.config)}")

        self.residue_set = ResidueSet(
            residue_masses=self.config.residues.get("residues"),
            residue_remapping=self.config.dataset.get("residue_remapping", None),
        )
        logger.info(f"Vocab: {self.residue_set.index_to_residue}")

        # Initialise S3 file handler
        self._s3: S3FileHandler = S3FileHandler(verbose=self.config.get("enable_verbose_s3", True))

        try:
            self.train_dataset, self.valid_dataset, train_size, valid_size = self.load_datasets()
        except Exception:
            logger.exception("FATAL: load_datasets() failed")
            raise

        logger.info(f"Data loaded from {train_size:,} training samples and {valid_size:,} validation samples (unfiltered values)")

        # Store dataset sizes for later logging (after tracker is initialized)
        self._train_size = train_size
        self._valid_size = valid_size

        self.train_dataloader, self.valid_dataloader = self.build_dataloaders(self.train_dataset, self.valid_dataset)
        logger.info("Data loaders built")

        # Print sample batch
        self.print_sample_batch()

        logger.info("Setting up model...")
        self.model = self.setup_model()

        if self.accelerator.is_main_process:
            logger.info(f"Model has {sum(p.numel() for p in self.model.parameters()):,d} parameters")

        self.optimizer = self.setup_optimizer()
        self.lr_scheduler = self.setup_scheduler()

        self.decoder = self.setup_decoder()
        self.metrics = self.setup_metrics()

        # Optionally load a model state for fine-tuning
        # Note: will be overwritten by the accelerator state if resuming
        if self.config.get("resume_checkpoint_path", None) is not None:
            self.load_model_state()

        # Prepare for accelerated training
        (
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
            self.valid_dataloader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            self.train_dataloader,
            self.valid_dataloader,
        )
        # Make sure the training state is checkpointed
        self.accelerator.register_for_checkpointing(self._training_state)

        # Optionally load states if resuming a training run
        if self.config.get("resume_accelerator_state", None):
            # Resuming from an existing run
            self.load_accelerator_state()

        # Setup experiment tracking
        self.setup_tracking()

        # Log datasets to experiment tracker if enabled (after tracker is initialized)
        if self.config.get("mlflow_log_datasets", True):
            self._log_datasets_to_tracker(self._train_size, self._valid_size)

        # Training control variables
        self.running_loss = None

        self.total_steps = self.config.get("training_steps", 2_500_000)

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Setup finetuning scheduler
        if self.config.get("finetune", None):
            self.finetune_scheduler: FinetuneScheduler | None = FinetuneScheduler(
                dict(unwrapped_model.named_parameters()),
                self.config.get("finetune"),
            )
        else:
            self.finetune_scheduler = None

        self.steps_per_validation = self.config.get("validation_interval", 100_000)
        self.steps_per_checkpoint = self.config.get("checkpoint_interval", 100_000)

        # Print training control variables
        if self.accelerator.is_main_process:
            steps_per_epoch = train_size // self.config["train_batch_size"]
            logger.info("Training setup complete.")
            logger.info(f" - Steps per validation: {self.steps_per_validation:,d} ")
            logger.info(f" - Steps per checkpoint: {self.steps_per_checkpoint:,d} ")
            logger.info(f" - Total training steps: {self.total_steps:,d}")
            logger.info("Estimating steps per epoch based on unfiltered training set size:")
            logger.info(f" - Estimated steps per epoch: {steps_per_epoch:,d}")
            logger.info(f" - Estimated total epochs: {self.total_steps / steps_per_epoch:.1f}")

            if self.total_steps < steps_per_epoch:
                logger.warning("Total steps is less than estimated steps per epoch, this may result in less than one epoch during training")

        if self.global_step > 0:
            logger.info(f"Training will resume from epoch {self.epoch}, global_step {self.global_step}")

        self.last_validation_metric = None
        self.best_checkpoint_metric = None

        # Final sync after setup
        self.accelerator.wait_for_everyone()

    @abstractmethod
    def setup_model(self) -> nn.Module:
        """Setup the model."""
        ...

    @abstractmethod
    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup the optimizer."""
        ...

    @abstractmethod
    def setup_decoder(self) -> Decoder:
        """Setup the decoder."""
        ...

    @abstractmethod
    def setup_data_processors(self) -> tuple[DataProcessor, DataProcessor]:
        """Setup the data processor."""
        ...

    @abstractmethod
    def save_model(self, is_best_checkpoint: bool = False) -> None:
        """Save the model."""
        ...

    @abstractmethod
    def forward(self, batch: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass for the model to calculate loss."""
        ...

    @abstractmethod
    def get_predictions(self, batch: Any) -> tuple[list[str] | list[list[str]], list[str] | list[list[str]]]:
        """Get the predictions for a batch."""
        ...

    @staticmethod
    def convert_interval_to_steps(interval: float | int, steps_per_epoch: int) -> int:
        """Convert an interval to steps.

        Args:
            interval (float | int): The interval to convert.
            steps_per_epoch (int): The number of steps per epoch.

        Returns:
            int: The number of steps.
        """
        if isinstance(interval, float):
            return int(interval * steps_per_epoch)
        else:
            raise ValueError(f"Invalid interval: {interval}")

    def log_if_verbose(self, message: str, level: str = "info") -> None:
        """Log a message if verbose logging is enabled."""
        if self.enable_verbose_logging:
            if level == "info":
                logger.info(message)
            elif level == "warning":
                logger.warning(message)
            elif level == "error":
                logger.error(message)
            elif level == "debug":
                logger.debug(message)
            else:
                raise ValueError(f"Invalid level: {level}")

    def setup_metrics(self) -> Metrics:
        """Setup the metrics."""
        return Metrics(self.residue_set, self.config.get("max_isotope_error", 1))

    def setup_accelerator(self) -> Accelerator:
        """Setup the accelerator."""
        timeout = timedelta(seconds=self.config.get("timeout", 3600))

        # Enable find_unused_parameters for DDP when using frozen parameters
        ddp_kwargs = DistributedDataParallelKwargs(
            find_unused_parameters=True,
        )

        accelerator = Accelerator(
            cpu=torch.backends.mps.is_available(),
            mixed_precision="fp16" if torch.cuda.is_available() and not self.config.get("force_cpu", False) else "no",
            gradient_accumulation_steps=self.config.get("grad_accumulation", 1),
            dataloader_config=DataLoaderConfiguration(
                split_batches=True,
                # dispatch_batches=False: each rank iterates its own shard via
                # DistributedSampler instead of rank-0 broadcasting.  The default
                # (True) broadcasts the full batch, which fails when metadata
                # columns contain non-tensor types (str, float) that NCCL cannot
                # handle.
                dispatch_batches=True,
            ),
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timeout), ddp_kwargs],
        )

        device = accelerator.device  # Important, this forces ranks to choose a device.

        if accelerator.num_processes > 1:
            set_rank(accelerator.local_process_index)

        if accelerator.is_main_process:
            logger.info(f"Python version: {sys.version}")
            logger.info(f"Torch version: {torch.__version__}")
            logger.info(f"CUDA version: {torch.version.cuda}")
            logger.info(f"Training with {accelerator.num_processes} devices")
            logger.info(f"Per-device batch size: {self.config['train_batch_size']}")
            logger.info(f"Gradient accumulation steps: {self.config['grad_accumulation']}")
            effective_batch_size = self.config["train_batch_size"] * accelerator.num_processes * self.config["grad_accumulation"]
            logger.info(f"Effective batch size: {effective_batch_size}")

        logger.info(f"Using device: {device}")

        return accelerator

    def build_dataloaders(self, train_dataset: Dataset, valid_dataset: Dataset) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """Setup the dataloaders."""
        train_processor, valid_processor = self.setup_data_processors()

        valid_processor.add_metadata_columns(["prediction_id"])
        if self.using_validation_groups:
            valid_processor.add_metadata_columns(["validation_group"])

        if self.config.get("use_shuffle_buffer", True):
            buffer_size = self.config.get("shuffle_buffer_size", SHUFFLE_BUFFER_SIZE)
            train_dataset = train_dataset.shuffle(buffer_size=buffer_size, seed=42)

        train_dataset = train_dataset.map(
            train_processor.process_row,
        )
        valid_dataset = valid_processor.process_dataset(valid_dataset)

        pin_memory = self.config.get("pin_memory", False)
        if self.accelerator.device == torch.device("cpu") or self.config.get("mps", False):
            pin_memory = False

        # torch rejects prefetch_factor unless num_workers > 0, so only pass it when
        # workers are actually spawned. Without this, num_workers=0 -- the natural
        # choice for a single-process or CPU run -- raises ValueError before training
        # starts, because the configs set prefetch_factor unconditionally.
        num_workers = self.config.get("num_workers", 8)
        prefetch_factor = self.config.get("prefetch_factor", None) if num_workers > 0 else None

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config["train_batch_size"] * self.accelerator.num_processes,
            collate_fn=train_processor.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            drop_last=True,
        )
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.config["predict_batch_size"] * self.accelerator.num_processes,
            collate_fn=valid_processor.collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor,
            drop_last=False,
        )
        return train_dataloader, valid_dataloader

    def setup_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Setup the learning rate scheduler.

        Returns:
            torch.optim.lr_scheduler.LRScheduler: The learning rate scheduler
        """
        # Note: if split_batches is False, the scheduler will be called num_processes times
        # in each optimizer step. Therefore, we need to scale the scheduler steps by num_processes.
        # Default is split_batches is True
        #
        # global_step counts optimizer updates (not forward passes), so max_iters = training_steps
        # is always correct regardless of grad_accumulation. The scheduler is only stepped when
        # sync_gradients=True (see train_epoch), so it naturally advances once per optimizer update.

        if self.config.get("lr_scheduler", "warmup") == "warmup":
            warmup_steps = self.config.get("warmup_iters", 1000)  # * num_processes
            return WarmupScheduler(self.optimizer, warmup_steps)
        elif self.config.get("lr_scheduler", None) == "cosine":
            # train_dataloader is already scaled by num_processes
            max_iters = self.config.get("training_steps", 2_500_000)
            warmup_steps = self.config.get("warmup_iters", 1000)  # * num_processes
            return CosineWarmupScheduler(self.optimizer, warmup_steps, max_iters)
        elif self.config.get("lr_scheduler", None) == "cosine_warmup_hold":
            max_iters = self.config.get("training_steps", 2_500_000)

            # Support fractional specification (fraction of total steps)
            warmup_frac = self.config.get("warmup_frac", None)
            lr_hold_frac = self.config.get("lr_hold_frac", None)

            # Fall back to absolute step specification
            warmup_steps = self.config.get("warmup_iters", 0)
            hold_steps = self.config.get("lr_hold_steps", 0)

            # Convert fractions to steps if provided
            if warmup_frac is not None:
                warmup_steps = int(warmup_frac * max_iters)
            if lr_hold_frac is not None:
                hold_steps = int(lr_hold_frac * max_iters)

            min_lr_factor = self.config.get("min_lr_factor", 0.0)

            return CosineWarmupHoldScheduler(
                self.optimizer,
                max_iters=max_iters,
                warmup_steps=warmup_steps,
                hold_steps=hold_steps,
                min_lr_factor=min_lr_factor,
            )
        else:
            raise ValueError(f"Unknown lr_scheduler type '{self.config.get('lr_scheduler', None)}'")

    def setup_tracking(self) -> None:
        """Setup MLFlow experiment tracking and log run metadata."""
        if not self.accelerator.is_main_process or not self.config.get("mlflow_enabled", True):
            self.tracker: MLFlowTracker | None = None
            return

        self.tracker = create_tracker(
            config=self.config,
            run_name=self.run_id,
        )

        self.tracker.log_hparams({k: v for k, v in self.config.items() if isinstance(v, (int, float, str))}, {})
        self.tracker.log_param("started_by", get_user())
        self.tracker.log_text("configs/hydra_config_resolved.yaml", OmegaConf.to_yaml(self.config, resolve=True))

        manifest_path = Path("manifest.yaml")
        if manifest_path.exists():
            self.tracker.log_text("configs/manifest.yaml", manifest_path.read_text(encoding="utf-8"))

        if S3FileHandler._aichor_enabled():
            self._log_aichor_metadata()

    def _log_aichor_metadata(self, commit_id_length: int = 7) -> None:
        """Log AIchor-specific metadata to the tracker.

        Called when AICHOR_LOGS_PATH is set; other VCS/AIchor env vars are
        treated as best-effort and missing fields are skipped rather than
        crashing the run.
        """
        assert self.tracker is not None

        commit_msg = os.environ.get("VCS_COMMIT_MESSAGE")
        commit_sha = os.environ.get("VCS_SHA")
        if commit_msg is not None and commit_sha:
            git_commit_msg = commit_msg.removeprefix("exp:").removeprefix(" ")
            commit_short_hash = commit_sha[:commit_id_length]
            self.tracker.log_param("git.commit_message", f"{git_commit_msg} ({commit_short_hash})")
        else:
            logger.debug("Skipping git.commit_message: VCS_COMMIT_MESSAGE or VCS_SHA not set")

        project_id = os.environ.get("AICHOR_PROJECT_ID")
        experiment_id = os.environ.get("AICHOR_EXPERIMENT_ID")
        if project_id and experiment_id:
            url = f"https://web.aichor.ai/projects/{project_id}/experiments/{experiment_id}?view=logs"
            self.tracker.log_param("aichor.experiment_url", url)
        else:
            logger.debug("Skipping aichor.experiment_url: AICHOR_PROJECT_ID or AICHOR_EXPERIMENT_ID not set")

    def _get_dataset_config_name(self) -> str | None:
        """Get the dataset config name from Hydra's config composition.

        When using `dataset=massivekb`, this returns "massivekb".

        Tries multiple methods:
        1. Explicit "name" field in dataset config
        2. Hydra's runtime config (if available)
        3. Returns None to fall back to path-based naming

        Returns:
            The dataset config name or None if not available
        """
        try:
            dataset_config = self.config.get("dataset", {})

            # Method 1: Check for explicit name field in dataset config
            if isinstance(dataset_config, DictConfig):
                for key in ["name", "_name"]:
                    if key in dataset_config:
                        name = dataset_config[key]
                        if isinstance(name, str) and name:
                            return name

            # Method 2: Try to get from Hydra's runtime config
            try:
                from hydra.core.global_hydra import GlobalHydra

                gh = GlobalHydra.instance()
                if gh.is_initialized():
                    hydra_cfg = gh.config_loader().get_cfg()
                    if hasattr(hydra_cfg, "runtime") and hasattr(hydra_cfg.runtime, "choices"):
                        choices = hydra_cfg.runtime.choices
                        if "dataset" in choices:
                            return str(choices["dataset"])
            except Exception:
                pass

            # Method 3: Try HydraConfig singleton (available during Hydra run)
            try:
                from hydra.core.hydra_config import HydraConfig

                if HydraConfig.initialized():
                    choices = HydraConfig.get().runtime.choices
                    if "dataset" in choices:
                        return str(choices["dataset"])
            except Exception:
                pass

        except Exception:
            pass

        return None

    def _log_datasets_to_tracker(self, train_size: int, valid_size: int) -> None:
        """Log dataset metadata to the experiment tracker.

        Training datasets (streaming IterableDataset) are logged as metadata
        only (name, source path, sample count). Validation datasets (in-memory
        HuggingFace Dataset) are logged via ``mlflow.data.from_huggingface``
        for full schema and digest tracking.
        """
        if not self.accelerator.is_main_process or self.tracker is None:
            return

        dataset_config = self.config.get("dataset", {})
        dataset_config_name = self._get_dataset_config_name()

        try:
            train_path = dataset_config.get("train_path", "unknown")
            train_name = f"train_{dataset_config_name}" if dataset_config_name else get_dataset_name_from_path(train_path, prefix="train")

            self.tracker.log_dataset(
                context="training",
                name=train_name,
                source=str(train_path),
                num_samples=train_size,
            )
            logger.info(f"Logged training dataset '{train_name}' to tracker ({train_size:,} samples)")

            valid_path = dataset_config.get("valid_path", train_path)
            if isinstance(valid_path, (dict, DictConfig)):
                valid_name = "valid_grouped"
                valid_source = "grouped"
            else:
                valid_name = f"valid_{dataset_config_name}" if dataset_config_name else get_dataset_name_from_path(valid_path, prefix="valid")
                valid_source = str(valid_path)

            self.tracker.log_hf_dataset(
                data=self.valid_dataset,
                context="validation",
                name=valid_name,
                source=valid_source,
                num_samples=valid_size,
            )
            logger.info(f"Logged validation dataset '{valid_name}' to tracker ({valid_size:,} samples)")

            if isinstance(valid_path, (dict, DictConfig)):
                valid_group_params = {f"dataset.validation.group.{group_name}": str(group_path) for group_name, group_path in valid_path.items()}
                self.tracker.log_params(valid_group_params)
                logger.info(f"Logged {len(valid_path)} validation dataset group paths to tracker")

        except Exception as e:
            logger.warning(f"Failed to log datasets to tracker: {e}")

    def load_datasets(self) -> tuple[Dataset, Dataset, int, int]:
        """Load the training and validation datasets.

        Returns:
            tuple[SpectrumDataFrame, SpectrumDataFrame]:
                The training and validation datasets
        """
        validation_group_mapping = None
        dataset_config = self.config.get("dataset", {})
        try:
            logger.info("Loading training dataset...")
            train_sdf = SpectrumDataFrame.load(
                source=dataset_config.get("train_path"),
                source_type=dataset_config.get("source_type", "default"),
                lazy=dataset_config.get("lazy_loading", True),
                is_annotated=True,
                shuffle=True,
                partition=dataset_config.get("train_partition", None),
                column_mapping=dataset_config.get("column_remapping", None),
                max_shard_size=dataset_config.get("max_shard_size", 100_000),
                preshuffle_across_shards=dataset_config.get("preshuffle_shards", False),
                verbose=dataset_config.get("verbose_loading", True),
            )

            valid_path = dataset_config.get("valid_path", None)
            if valid_path is not None:
                if OmegaConf.is_dict(valid_path):
                    logger.info("Found grouped validation datasets.")
                    validation_group_mapping = _get_filepath_mapping(valid_path)
                    _valid_path = list(valid_path.values())
                else:
                    _valid_path = valid_path
            else:
                _valid_path = dataset_config.get("train_path")

            logger.info("Loading validation dataset...")
            valid_sdf = SpectrumDataFrame.load(
                _valid_path,
                lazy=dataset_config.get("lazy_loading", True),
                is_annotated=True,
                shuffle=False,
                partition=dataset_config.get("valid_partition", None),
                column_mapping=dataset_config.get("column_remapping", None),
                max_shard_size=dataset_config.get("max_shard_size", 100_000),
                add_source_file_column=True,  # used to track validation groups
                verbose=dataset_config.get("verbose_loading", True),
            )
        except ValueError as e:
            # More descriptive error message in predict mode.
            if str(e) == ANNOTATION_ERROR:
                raise ValueError("The sequence column is missing annotations, are you trying to run de novo prediction? Add the --denovo flag") from e
            else:
                raise

        if dataset_config.get("valid_path", None) is None:
            raise NotImplementedError("Automatic validation dataset splitting is not supported.")

        train_ds = train_sdf.to_dataset(force_unified_schema=True)
        valid_ds = valid_sdf.to_dataset(in_memory=True)

        # # Sample subsets if needed
        valid_subset = self.config.get("valid_subset", 1.0)
        if valid_subset < 1.0:
            valid_ds = valid_ds.train_test_split(test_size=valid_subset, seed=42)["test"]

        # Check residues
        if self.config.get("perform_data_checks", True):
            logger.info(f"Checking for unknown residues in {len(train_sdf) + len(valid_sdf):,d} rows.")
            supported_residues = set(self.residue_set.residue_masses.keys()) | set(self.residue_set.residue_remapping.keys())

            data_residues = set()
            data_residues.update(train_sdf.get_vocabulary(self.residue_set.tokenize))
            data_residues.update(valid_sdf.get_vocabulary(self.residue_set.tokenize))

            filter_unsupported_residues = len(data_residues - supported_residues) > 0
            if filter_unsupported_residues:
                logger.warning(f"Found {len(data_residues - supported_residues):,d} unsupported residues! These rows will be dropped.")
                self.log_if_verbose(f"New residues found: \n{data_residues - supported_residues}")
                self.log_if_verbose(f"Residues supported: \n{supported_residues}")

            logger.info("Checking charge values...")
            filter_train_charge = not train_sdf.check_values(1, self.config.model.max_charge, "precursor_charge")
            filter_valid_charge = not valid_sdf.check_values(1, self.config.model.max_charge, "precursor_charge")
            if filter_train_charge:
                logger.warning("Found charge values out of range in training set. These rows will be dropped.")
            if filter_valid_charge:
                logger.warning("Found charge values out of range in validation set. These rows will be dropped.")

            if filter_unsupported_residues or filter_train_charge:
                train_ds = train_ds.filter(
                    lambda row, _fur=filter_unsupported_residues, _ftc=filter_train_charge, _sr=supported_residues: (
                        (not _fur or all(r in _sr for r in self.residue_set.tokenize(row["sequence"])))
                        and (not _ftc or (0 < row["precursor_charge"] <= self.config.model.max_charge))
                    )
                )

            if filter_unsupported_residues or filter_valid_charge:
                valid_ds = valid_ds.filter(
                    lambda row, _fur=filter_unsupported_residues, _fvc=filter_valid_charge, _sr=supported_residues: (
                        (not _fur or all(r in _sr for r in self.residue_set.tokenize(row["sequence"])))
                        and (not _fvc or (0 < row["precursor_charge"] <= self.config.model.max_charge))
                    )
                )

        # Create validation groups
        # Initialize validation groups if needed
        if validation_group_mapping is not None:
            logger.info("Computing validation groups.")
            validation_groups = [validation_group_mapping.get(row.get("source_file"), "no_group") for row in valid_ds]

            # Encode group strings as int codes so the column tensorizes cleanly
            # under accelerate's dispatch_batches=True. Decoded back to strings
            # in validate_epoch after gather_for_metrics.
            unique_groups = sorted(set(validation_groups))
            self.validation_group_to_idx: dict[str, int] = {g: i for i, g in enumerate(unique_groups)}
            self.validation_idx_to_group: dict[int, str] = {i: g for g, i in self.validation_group_to_idx.items()}
            validation_group_codes = [self.validation_group_to_idx[g] for g in validation_groups]
            valid_ds = valid_ds.add_column("validation_group", validation_group_codes, feature=Value("int32"))

            logger.info("Sequences per validation group:")
            group_counts = Counter(validation_groups)
            for group, count in group_counts.items():
                logger.info(f" - {group}: {count:,d}")

            self.using_validation_groups = True
        else:
            self.using_validation_groups = False

        # Force add a unique prediction_id column
        # This will be used to order predictions and remove duplicates
        valid_ds = valid_ds.add_column("prediction_id", np.arange(len(valid_ds)), feature=Value("int32"))

        # Keep track of the train_sdf directory so it isn't garbage collected
        self._train_sdf = train_sdf

        return train_ds, valid_ds, len(train_sdf), len(valid_sdf)

    def print_sample_batch(self) -> None:
        """Print a sample batch of the training data."""
        if self.accelerator.is_main_process:
            # sample_batch = next(iter(self.train_dataloader))
            sample_batch = next(iter(self.train_dataloader))
            logger.info("Sample batch:")
            for key, value in sample_batch.items():
                if isinstance(value, torch.Tensor):
                    value_shape = value.shape
                    value_type = value.dtype
                else:
                    value_shape = len(value)
                    value_type = type(value)

                logger.info(f" - {key}: {value_type}, {value_shape}")

    def get_model_input_example(self) -> np.ndarray | None:
        """Get a sample input example for model signature inference.

        Returns a numpy array (spectra tensor) from the validation dataloader
        that can be used as an input_example for MLFlow model logging. This
        enables automatic model signature inference.

        Note: Only returns the primary 'spectra' input as a numpy array since
        MLFlow's PyTorch flavor works best with simple numpy array inputs.

        See: https://www.mlflow.org/docs/latest/ml/model/signatures/

        Returns:
            Numpy array of spectra, or None if unavailable
        """
        try:
            # Get a batch from validation dataloader
            sample_batch = next(iter(self.valid_dataloader))

            # Return spectra as numpy array (primary input tensor)
            if "spectra" in sample_batch:
                spectra = sample_batch["spectra"]
                if isinstance(spectra, torch.Tensor):
                    # Take first sample only, move to CPU, convert to numpy
                    return spectra[:1].cpu().numpy()
                elif isinstance(spectra, np.ndarray):
                    return spectra[:1]

            return None
        except Exception as e:
            logger.debug(f"Could not get model input example: {e}")
            return None

    def _log_checkpoint_artifact(self, local_path: str, artifact_path: str) -> None:
        """Log a checkpoint file or directory to MLflow when enabled."""
        if self.tracker is None or not self.config.get("mlflow_log_checkpoints", True):
            return
        self.tracker.log_artifact(local_path, artifact_path=artifact_path)

    def save_accelerator_state(self, is_best_checkpoint: bool = False) -> None:
        """Save the accelerator state."""
        checkpoint_dir = self.config.get("model_save_folder_path", "./checkpoints")

        if self.config.get("keep_accelerator_every_interval", False):
            checkpoint_path = os.path.join(
                checkpoint_dir,
                "accelerator_state",
                f"epoch_{self.epoch}_step_{self.global_step + 1}",
            )
        else:
            checkpoint_path = os.path.join(checkpoint_dir, "accelerator_state", "latest")
            if self.accelerator.is_main_process and Path(checkpoint_path).exists() and Path(checkpoint_path).is_dir():
                shutil.rmtree(checkpoint_path)

        if self.accelerator.is_main_process:
            os.makedirs(checkpoint_path, exist_ok=True)

        self.accelerator.save_state(checkpoint_path)

        logger.info(f"Saved accelerator state to {checkpoint_path}")

        if self.accelerator.is_main_process and S3FileHandler._aichor_enabled():
            for file in os.listdir(checkpoint_path):
                self.s3.upload(
                    os.path.join(checkpoint_path, file),
                    S3FileHandler.convert_to_s3_output(os.path.join(checkpoint_path, file)),
                )

        self._log_checkpoint_artifact(checkpoint_path, artifact_path=f"checkpoints/accelerator_state/{os.path.basename(checkpoint_path)}")

        # Save best checkpoint and upload to S3
        if is_best_checkpoint and self.accelerator.is_main_process:
            best_checkpoint_path = os.path.join(checkpoint_dir, "accelerator_state", "best")
            if Path(best_checkpoint_path).exists() and Path(best_checkpoint_path).is_dir():
                shutil.rmtree(best_checkpoint_path)

            os.makedirs(best_checkpoint_path, exist_ok=True)

            for file in os.listdir(checkpoint_path):
                full_file = os.path.join(checkpoint_path, file)
                best_file = os.path.join(best_checkpoint_path, file)
                shutil.copy(full_file, best_file)
                if S3FileHandler._aichor_enabled():
                    self.s3.upload(
                        full_file,
                        S3FileHandler.convert_to_s3_output(best_file),
                    )

            self._log_checkpoint_artifact(best_checkpoint_path, artifact_path="checkpoints/accelerator_state/best")

    def check_if_best_checkpoint(self) -> bool:
        """Check if the last validation metric is the best metric."""
        if self.config.get("checkpoint_metric", None) is None:
            return False

        if self.best_checkpoint_metric is None:
            self.best_checkpoint_metric = self.last_validation_metric
            return True

        if self.config.get("checkpoint_metric_mode", "min") == "min":
            is_best = self.last_validation_metric <= self.best_checkpoint_metric
        elif self.config.get("checkpoint_metric_mode", "min") == "max":
            is_best = self.last_validation_metric >= self.best_checkpoint_metric
        else:
            raise ValueError(f"Unknown checkpoint metric mode: {self.config.get('checkpoint_metric_mode', 'min')}")

        if is_best:
            self.best_checkpoint_metric = self.last_validation_metric

        return is_best

    def load_accelerator_state(self) -> None:
        """Load the accelerator state."""
        checkpoint_path = self.config.get("resume_accelerator_state", None)
        if checkpoint_path is None:
            return

        if not os.path.isdir(checkpoint_path) and not checkpoint_path.startswith("s3://"):
            raise ValueError(f"Accelerator state should be a directory of state files, got {checkpoint_path}")

        if S3FileHandler._aichor_enabled() and checkpoint_path.startswith("s3://"):
            # raise NotImplementedError("Loading accelerator state from S3 is not implemented.")

            if self.accelerator.is_main_process:
                local_path = os.path.join(self.s3.temp_dir.name, "accelerator_state")
                os.makedirs(local_path, exist_ok=True)
                logger.info(f"Downloading checkpoint files from {checkpoint_path} to {local_path}")

                # Download all files from the checkpoint folder
                checkpoint_files = self.s3.listdir(checkpoint_path)
                logger.info(f"Found {len(checkpoint_files)} files")
                for file in checkpoint_files:
                    if file.endswith("/"):  # Skip subdirectories
                        continue
                    local_file = os.path.join(local_path, os.path.basename(file))
                    self.s3.download(f"s3://{file}", local_file)
            else:
                local_path = None

            checkpoint_path = broadcast_object_list([local_path])[0]
            logger.info(f"Received checkpoint path: {checkpoint_path}")

            assert checkpoint_path is not None, "Failed to broadcast accelerator state across ranks"

        # Add safe globals
        torch.serialization.add_safe_globals(
            [
                np._core.multiarray.scalar,
                np.dtypes.Float64DType,
            ]
        )

        self.accelerator.load_state(checkpoint_path)
        logger.info(f"Loaded accelerator state from {checkpoint_path}")

    def load_model_state(self) -> None:
        """Load the model state."""
        checkpoint_path = self.config.get("resume_checkpoint_path", None)
        if checkpoint_path is None:
            return

        if os.path.isdir(checkpoint_path) and not checkpoint_path.startswith("s3://"):
            raise ValueError(f"Checkpoint path should be a file, got {checkpoint_path}")

        if self.accelerator.is_main_process:
            logger.info(f"Resuming model state from {checkpoint_path}")
            local_path = self.s3.get_local_path(checkpoint_path)
        else:
            local_path = None

        local_path = broadcast_object_list([local_path])[0]

        assert local_path is not None, "Failed to broadcast model state across ranks"

        # TODO: Switch to model.load(), implement model schema
        model_data = torch.load(local_path, weights_only=False, map_location="cpu")
        # TODO: Remove, only use state_dict
        if "model" in model_data:
            model_state = model_data["model"]
        else:
            model_state = model_data["state_dict"]
            # Remove `model.` if present
            model_state = {k[6:] if k.startswith("model.") else k: v for k, v in model_state.items()}

        # Check residues
        if "residues" in model_data:
            model_residues = dict(model_data["residues"])
        else:
            # Legacy format
            model_residues = dict(model_data["config"]["residues"])

        current_residues = self.config.residues.get("residues")
        if model_residues != current_residues:
            logger.warning(
                f"Checkpoint residues do not match current residues.\nCheckpoint residues: {model_residues}\nCurrent residues: {current_residues}"
            )
            logger.warning("Updating model state to match current residues.")
            model_state = self.update_vocab(model_state)

        model_state = self.update_model_state(model_state, model_data["config"])

        old_model_keys = set(self.model.state_dict().keys())
        new_model_keys = set(model_state.keys())
        if old_model_keys != new_model_keys:
            logger.warning("Model keys do not match.")
            logger.warning(f"Missing keys: {old_model_keys - new_model_keys}")
            logger.warning(f"Extra keys: {new_model_keys - old_model_keys}")

        self.model.load_state_dict(model_state, strict=False)
        logger.info(f"Loaded model state from {local_path}")

    def update_model_state(self, model_state: dict[str, torch.Tensor], model_config: DictConfig) -> dict[str, torch.Tensor]:
        """Update the model state."""
        return model_state

    def update_vocab(self, model_state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Update the vocabulary of the model."""
        # This should call `self._update_vocab` based on model implementation.
        raise NotImplementedError("Updating vocabulary is not implemented for the base trainer.")

    def _update_vocab(
        self,
        model_state: dict[str, torch.Tensor],
        target_layers: list[str],
        resolution: str = "delete",
    ) -> dict[str, torch.Tensor]:
        """Update the target heads of the model."""
        target_vocab_size = len(self.residue_set)
        current_model_state = self.model.state_dict()
        hidden_size = self.config.model.get("dim_model", 768)

        for layer in target_layers:
            if layer not in current_model_state:
                logger.warning(f"Layer {layer} not found in current model state.")
                continue
            tmp = torch.normal(
                mean=0,
                std=1.0 / np.sqrt(hidden_size),
                size=current_model_state[layer].shape,
                dtype=current_model_state[layer].dtype,
            )
            if "bias" in layer:
                # initialise bias to zeros
                tmp = torch.zeros_like(tmp)

            if resolution == "delete":
                del model_state[layer]
            elif resolution == "random":
                model_state[layer] = tmp
            elif resolution == "partial":
                tmp[:target_vocab_size] = model_state[layer][: min(tmp.shape[0], target_vocab_size)]
                model_state[layer] = tmp
            else:
                raise ValueError(f"Unknown residue_conflict_resolution type '{resolution}'")
        return model_state

    def train(self) -> None:
        """Train the model."""
        num_sanity_steps = self.config.get("num_sanity_val_steps", 0)
        if num_sanity_steps > 0:
            logger.info(f"Running sanity validation for {num_sanity_steps} steps...")
            self.validate_epoch(num_sanity_steps=num_sanity_steps, calculate_metrics=False)
            logger.info("Sanity validation complete.")

        if self.config.get("validate_before_training", False):
            logger.info("Running pre-validation...")
            self.validate_epoch()
            logger.info("Pre-validation complete.")

        self.train_timer = Timer(self.total_steps)
        is_first_epoch = True
        logger.info("Starting training...")
        while self.global_step < self.total_steps:
            self.train_epoch()
            self.training_state.step_epoch()
            if self.accelerator.is_main_process and is_first_epoch:
                is_first_epoch = False
                logger.info("First epoch complete:")
                logger.info(f"- Actual steps per epoch: {self.global_step}")
                logger.info(f"- Actual total epochs: {self.total_steps / self.global_step:.1f}")

        logger.info("Training complete.")

    def prepare_batch(self, batch: Iterable[Any]) -> Any:
        """Prepare a batch for training.

        Manually move tensors to accelerator.device since we do not
        prepare our dataloaders with the accelerator.

        Uses non_blocking=True for async H2D transfers to overlap with compute.

        Args:
            batch (Iterable[Any]): The batch to prepare.

        Returns:
            Any: The prepared batch
        """
        if isinstance(batch, dict):
            return {k: v.to(self.accelerator.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        elif isinstance(batch, (list, tuple)):
            return [v.to(self.accelerator.device, non_blocking=True) if isinstance(v, torch.Tensor) else v for v in batch]
        else:
            raise ValueError(f"Unsupported batch type: {type(batch)}")

    def train_epoch(self) -> None:
        """Train the model for one epoch.

        global_step counts optimizer updates, not forward passes. All step-based
        logic (logging, validation, checkpointing) runs only when sync_gradients is
        True (i.e. when a real optimizer update occurred). This means training_steps
        always refers to the number of optimizer updates regardless of grad_accumulation.
        """
        total_loss = 0

        self.model.train()
        self.optimizer.zero_grad()
        self.running_loss = None

        epoch_timer = Timer()

        print_batch_size = True
        for batch_count, batch in enumerate(self.train_dataloader):
            if print_batch_size:
                # Confirm batch size during debugging
                self.log_if_verbose(f"Batch {batch_count} shape: {batch['spectra'].shape[0]}")
                print_batch_size = False

            _is_optimizer_step = False
            with self.accelerator.accumulate(self.model):
                # Forward pass
                loss, loss_components = self.forward(batch)

                # Backward pass
                self.accelerator.backward(loss)

                # Update weights
                _is_optimizer_step = self.accelerator.sync_gradients
                if _is_optimizer_step:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.get("gradient_clip_val", 10.0))
                self.optimizer.step()

                self.lr_scheduler.step()
                self.optimizer.zero_grad()

                # Advance global_step only on real optimizer updates
                if _is_optimizer_step:
                    self.training_state.step()

            if _is_optimizer_step:
                self.train_timer.step()

            # Update running loss (every forward pass)
            if self.running_loss is None:
                self.running_loss = loss.item()
            else:
                self.running_loss = 0.99 * self.running_loss + 0.01 * loss.item()

            total_loss += loss.item()

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

            # Log to experiment tracker
            if (
                self.accelerator.is_main_process
                and self.tracker is not None
                and (self.global_step + 1) % int(self.config.get("metrics_logging_steps", 500)) == 0
            ):
                lr = self.lr_scheduler.get_last_lr()[0]
                self.tracker.log_scalar("train/loss_raw", loss.item(), self.global_step + 1)
                if self.running_loss is not None:
                    self.tracker.log_scalar("train/loss_smooth", self.running_loss, self.global_step + 1)
                for k, v in loss_components.items():
                    if k == "loss":
                        continue
                    self.tracker.log_scalar(f"train/{k}", v.item(), self.global_step + 1)
                self.tracker.log_scalar("optim/lr", lr, self.global_step + 1)
                self.tracker.log_scalar("optim/epoch", self.epoch, self.global_step + 1)
                samples_seen = (self.global_step + 1) * self.config["train_batch_size"] * self.accelerator.num_processes
                self.tracker.log_scalar("train/samples_seen", samples_seen, self.global_step + 1)

            if (self.global_step > 0) and (self.global_step % self.steps_per_validation == 0):
                self.model.eval()
                self.validate_epoch()
                logger.info("Validation complete, resuming training...")
                self.model.train()

            if self.global_step % self.steps_per_checkpoint == 0:
                is_best_checkpoint = self.check_if_best_checkpoint()
                self.save_model(is_best_checkpoint)
                if self.config.get("save_accelerator_state", False):
                    self.save_accelerator_state(is_best_checkpoint)

            # Update finetuning scheduler
            if self.finetune_scheduler is not None:
                self.finetune_scheduler.step(self.global_step)

            if self.global_step >= self.total_steps:
                break

        # Epoch complete
        self.accelerator.wait_for_everyone()

        epoch_timer.step()

        # Gather losses from all devices
        gathered_losses = self.accelerator.gather_for_metrics(torch.tensor(total_loss, device=self.accelerator.device))
        gathered_num_batches = self.accelerator.gather_for_metrics(torch.tensor(batch_count, device=self.accelerator.device))

        if self.accelerator.is_main_process and self.tracker is not None:
            # Sum the losses and batch counts from all devices
            total_loss_all_devices = gathered_losses.sum().item()
            total_batches_all_devices = gathered_num_batches.sum().item()
            avg_loss = total_loss_all_devices / total_batches_all_devices

            self.tracker.log_scalar("eval/train_loss", avg_loss, self.epoch)

            logger.info(f"[TRAIN] [Epoch {self.epoch:02d}] Epoch complete, total time {epoch_timer.get_time_str()}")

    def validate_epoch(self, num_sanity_steps: int | None = None, calculate_metrics: bool = True) -> None:
        """Validate for one epoch."""
        if self.valid_dataloader is None:
            return

        if self.accelerator.is_main_process:
            logger.info(f"[VALIDATION] [Epoch {self.epoch:02d}] Starting validation.")

        valid_epoch_step = 0
        valid_predictions: List[List[str] | str] = []
        valid_targets: List[List[str] | str] = []
        valid_groups: List[str] = []
        valid_prediction_ids: List[int] = []

        valid_metrics: Dict[str, List[float]] = {x: [] for x in ["valid_loss", "aa_er", "aa_prec", "aa_recall", "pep_recall", "pep_prec"]}

        try:
            num_batches = len(self.valid_dataloader)
        except TypeError:
            num_batches = 0

        valid_timer = Timer(num_batches)

        for batch_idx, batch in enumerate(self.valid_dataloader):
            if num_sanity_steps is not None and batch_idx >= num_sanity_steps:
                break

            with torch.no_grad(), self.accelerator.autocast():
                # Loss calculation
                loss, _ = self.forward(batch)
                # Get actual predictions
                y, targets = self.get_predictions(batch)

            valid_predictions.extend(y)
            valid_targets.extend(targets)
            valid_prediction_ids.extend([x.item() if hasattr(x, "item") else x for x in batch["prediction_id"]])

            # Store validation groups if available (as int codes; decoded after gather)
            if self.using_validation_groups:
                valid_groups.extend([x.item() if hasattr(x, "item") else x for x in batch["validation_group"]])

            # Update metrics
            if self.metrics is not None:
                aa_prec, aa_recall, pep_recall, pep_prec = self.metrics.compute_precision_recall(targets, y)
                aa_er = self.metrics.compute_aa_er(targets, y)

                valid_metrics["valid_loss"].append(loss.item())
                valid_metrics["aa_er"].append(aa_er)
                valid_metrics["aa_prec"].append(aa_prec)
                valid_metrics["aa_recall"].append(aa_recall)
                valid_metrics["pep_recall"].append(pep_recall)
                valid_metrics["pep_prec"].append(pep_prec)

            valid_epoch_step += 1

            valid_timer.step()

            # Log progress
            if (valid_epoch_step + 1) % int(self.config.get("console_logging_steps", 2000)) == 0:
                epoch_step = valid_epoch_step % num_batches

                logger.info(
                    f"[VALIDATION] "
                    f"[Epoch {self.epoch:02d}] "
                    f"[Step {self.global_step + 1:06d}] "
                    f"[Batch {epoch_step:05d}/{num_batches:05d}] "
                    f"[{valid_timer.get_time_str()}/{valid_timer.get_total_time_str()}, "
                    f"{valid_timer.get_step_time_rate_str()}]"
                )

        # Synchronize all processes at the end of validation
        # This ensures all ranks wait for the slowest rank to finish
        self.accelerator.wait_for_everyone()

        if not calculate_metrics:
            return

        # Gather predictions from all devices
        if valid_predictions:
            self.log_if_verbose("Gathering predictions from all devices")
            valid_predictions = self.accelerator.gather_for_metrics(valid_predictions)
            valid_targets = self.accelerator.gather_for_metrics(valid_targets)
            valid_prediction_ids = self.accelerator.gather_for_metrics(valid_prediction_ids)

            # Use valid_prediction_ids to remove duplicates
            # Find the indices of the first occurrence of each unique prediction_id
            _, idx = np.unique(valid_prediction_ids, return_index=True)
            valid_predictions = [valid_predictions[i] for i in idx]
            valid_targets = [valid_targets[i] for i in idx]

            self.log_if_verbose(f"Gathered {len(valid_predictions)} predictions")

            # Gather validation groups if available, then decode int codes back to strings
            if self.using_validation_groups:
                valid_groups = self.accelerator.gather_for_metrics(valid_groups)
                valid_groups = [valid_groups[i] for i in idx]
                valid_groups = [self.validation_idx_to_group[int(g)] for g in valid_groups]

            # Gather valid_metrics from all devices
            for metric, values in valid_metrics.items():
                valid_metrics[metric] = self.accelerator.gather_for_metrics(values)

        # Keep validation metrics for checkpointing
        checkpoint_metric = self.config.get("checkpoint_metric", None)
        if checkpoint_metric is not None:
            self.last_validation_metric = np.mean(valid_metrics[checkpoint_metric])

        # Log validation metrics
        if self.accelerator.is_main_process and self.metrics is not None:
            # Validation metrics are logged by epoch
            validation_step = self.global_step + 1

            if self.tracker is not None:
                for k, v in valid_metrics.items():
                    self.tracker.log_scalar(f"eval/{k}", np.mean(v), validation_step)

            logger.info(
                f"[VALIDATION] [Epoch {self.epoch:02d}] "
                f"[Step {self.global_step + 1:06d}] "
                f"train_loss={self.running_loss if self.running_loss else 0:.5f}, "
                f"valid_loss={np.mean(valid_metrics['valid_loss']):.5f}"
            )
            logger.info(f"[VALIDATION] [Epoch {self.epoch:02d}] [Step {self.global_step + 1:06d}] Metrics:")
            for metric in ["aa_er", "aa_prec", "aa_recall", "pep_recall"]:
                val = np.mean(valid_metrics[metric])
                logger.info(f"[VALIDATION] [Epoch {self.epoch:02d}] [Step {self.global_step + 1:06d}] - {metric:11s}{val:.3f}")

            if self.tracker is not None and self.config.get("mlflow_log_evaluations", True):
                overall_eval_df = pd.DataFrame(
                    {
                        "target": valid_targets,
                        "prediction": valid_predictions,
                    }
                )
                self.tracker.log_mlflow_evaluation(
                    data=overall_eval_df,
                    targets_col="target",
                    predictions_col="prediction",
                    dataset_name=f"validation_step_{validation_step}",
                    context="train_validation",
                    step=validation_step,
                    peptide_metrics=self.metrics,
                    log_artifacts=self.config.get("mlflow_eval_log_artifacts", True),
                )

            # Validation group logging
            if self.using_validation_groups and valid_groups and self.tracker is not None:
                preds = pl.Series(valid_predictions)
                targs = pl.Series(valid_targets)
                groups = pl.Series(valid_groups)

                assert len(preds) == len(groups)
                assert len(targs) == len(groups)

                for group in groups.unique():
                    idx = groups == group
                    logger.info(f"Computing group {group} with {idx.sum()} samples")
                    if idx.sum() > 0:  # Only compute metrics if we have samples for this group
                        aa_prec, aa_recall, pep_recall, _ = self.metrics.compute_precision_recall(targs.filter(idx), preds.filter(idx))
                        aa_er = self.metrics.compute_aa_er(targs.filter(idx), preds.filter(idx))
                        self.tracker.log_scalar(f"eval/{group}_aa_er", aa_er, validation_step)
                        self.tracker.log_scalar(f"eval/{group}_aa_prec", aa_prec, validation_step)
                        self.tracker.log_scalar(f"eval/{group}_aa_recall", aa_recall, validation_step)
                        self.tracker.log_scalar(f"eval/{group}_pep_recall", pep_recall, validation_step)

                        if self.config.get("mlflow_log_evaluations", True):
                            group_preds = preds.filter(idx).to_list()
                            group_targs = targs.filter(idx).to_list()
                            eval_df = pd.DataFrame(
                                {
                                    "target": group_targs,
                                    "prediction": group_preds,
                                }
                            )
                            self.tracker.log_mlflow_evaluation(
                                data=eval_df,
                                targets_col="target",
                                predictions_col="prediction",
                                dataset_name=f"validation_{group}_step_{validation_step}",
                                context=f"train_validation_{group}",
                                step=validation_step,
                                peptide_metrics=self.metrics,
                                log_artifacts=self.config.get("mlflow_eval_log_artifacts", True),
                            )

        self.accelerator.wait_for_everyone()
