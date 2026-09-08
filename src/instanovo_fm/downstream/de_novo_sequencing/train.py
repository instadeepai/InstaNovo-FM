from pathlib import Path

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo_fm.common import DataProcessor
from instanovo_fm.data.search_data_manager import create_search_data_manager
from instanovo_fm.downstream.de_novo_sequencing.data import DownstreamDeNovoDataProcessor
from instanovo_fm.downstream.de_novo_sequencing.model import DownstreamDeNovo
from instanovo_fm.baselines import TransformerTrainer
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs"


class DownstreamDeNovoTrainer(TransformerTrainer):
    """Trainer for the downstream de novo sequencing task."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup_model(self) -> nn.Module:
        """Setup the model."""
        config = self.config.get("model", {})
        model = DownstreamDeNovo(
            residue_set=self.residue_set,
            dim_model=config["dim_model"],
            n_head=config["n_head"],
            dim_feedforward=config["dim_feedforward"],
            encoder_layers=config.get("encoder_layers", config.get("n_layers", 9)),
            decoder_layers=config.get("decoder_layers", config.get("n_layers", 9)),
            dropout=config["dropout"],
            use_flash_attention=config.get("use_flash_attention", False),  # this may be overwritten by pretrained model
            conv_peak_encoder=config.get("conv_peak_encoder", False),
            peak_encoder_type=config.get("peak_encoder", {}).get("type", "multiscale"),
            max_mz=config.get("max_mz", 2500.0),
            min_mz=config.get("min_mz", 0.0),
            max_charge=config.get("max_charge", 10),
            cfg=config,
            use_meta_token=config.get("use_meta_token", True),
        )

        # If using a pretrained model, load the checkpoint and replace the relevant parameters
        if self.config.get("use_pretrained_model", False):
            logger.info("Using pretrained encoder.")
            foundation_model_path = self.s3.get_local_path(self.config.foundational_checkpoint)

            model.load_encoder_from_foundation(foundation_model_path)

            if not self.config.get("finetune", None):
                logger.warning(
                    "use_pretrained_model=True but no finetune schedule configured. "
                    "Pretrained encoder parameters will not be frozen and will receive "
                    "gradient updates from step 0. Set finetune.unfreeze_schedule to enable gradual unfreezing."
                )

            logger.info(f"Pairwise bias: {'enabled' if model.pairwise_bias is not None else 'disabled'}")
            logger.info(f"Ion ladder: {'enabled' if model.ion_ladder is not None else 'disabled'}")

            frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)
            trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
            logger.info(f"Model verification: {frozen_count} frozen, {trainable_count} trainable")
            logger.info(f"Encoder verification: {sum(1 for p in model.encoder.parameters() if p.requires_grad)} trainable")
            logger.info(f"Peak encoder verification: {sum(1 for p in model.peak_encoder.parameters() if p.requires_grad)} trainable")

        else:
            logger.info("Training from scratch.")

        return model

    def setup_data_processors(self) -> tuple[DataProcessor, DataProcessor]:
        """Setup train and validation data processors.

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
        # - filepath (for search_data lookup)
        # Other fields (precursor_mz, precursor_charge, precursor_mass) are already in batch
        if meta_token_enabled and metadata_columns:
            required_metadata = {"frag_type", "collision_energy", "filepath"}
            metadata_columns_filtered = [col for col in metadata_columns if col in required_metadata]
            logger.info(f"Filtered metadata_columns for meta token: {len(metadata_columns)} -> {len(metadata_columns_filtered)} columns")
        else:
            metadata_columns_filtered = metadata_columns

        # Setup search data manager if enabled
        search_data_config = {
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

        # Training processor
        train_processor = DownstreamDeNovoDataProcessor(
            residue_set=self.residue_set,
            n_peaks=self.config.model.get("n_peaks", 200),
            min_mz=self.config.model.get("min_mz", 50.0),
            max_mz=self.config.model.get("max_mz", 2500.0),
            min_intensity=self.config.model.get("min_intensity", 1e-6),
            remove_precursor_tol=self.config.model.get("remove_precursor_tol", 2.0),
            use_spectrum_utils=self.config.model.get("use_spectrum_utils", False),
            normalize_mz=self.config.model.get("normalize_mz", True),
            peak_ordering=self.config.model.get("peak_ordering", "sorted"),
            annotated=True,
            return_str=False,
            metadata_columns=metadata_columns_filtered if meta_token_enabled else None,
            search_data_manager=search_data_manager if meta_token_enabled else None,
            build_metadata=meta_token_enabled,
        )

        # Validation processor: uses full metadata columns for analysis
        valid_processor = DownstreamDeNovoDataProcessor(
            residue_set=self.residue_set,
            n_peaks=self.config.model.get("n_peaks", 200),
            min_mz=self.config.model.get("min_mz", 50.0),
            max_mz=self.config.model.get("max_mz", 2500.0),
            min_intensity=self.config.model.get("min_intensity", 1e-6),
            remove_precursor_tol=self.config.model.get("remove_precursor_tol", 2.0),
            use_spectrum_utils=self.config.model.get("use_spectrum_utils", False),
            normalize_mz=self.config.model.get("normalize_mz", True),
            peak_ordering=self.config.model.get("peak_ordering", "sorted"),
            annotated=True,
            return_str=False,
            metadata_columns=metadata_columns,
            search_data_manager=search_data_manager,
            build_metadata=meta_token_enabled,
        )

        return train_processor, valid_processor


@hydra.main(config_path=str(CONFIG_PATH), version_base=None, config_name="denovo")
def main(config: DictConfig) -> None:
    """Train the model."""
    logger.info("Initializing training.")
    trainer = DownstreamDeNovoTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
