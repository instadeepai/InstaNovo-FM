from pathlib import Path
from typing import Tuple

import torch.nn as nn
from omegaconf import DictConfig

from instanovo.__init__ import console
from instanovo_fm.common import DataProcessor
from instanovo_fm.downstream.de_novo_sequencing.data import DownstreamDeNovoDataProcessor
from instanovo_fm.downstream.de_novo_sequencing.model import DownstreamDeNovo
from instanovo_fm.baselines.transformer_predict import TransformerPredictor
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "configs"


class DownstreamDeNovoPredictor(TransformerPredictor):
    """Predictor for the InstaNovo model."""

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        super().__init__(config)

    def load_model(self) -> Tuple[nn.Module, DictConfig]:
        """Setup the model."""
        model_path = self.config.get("denovo_model")
        logger.info(f"Loading InstaNovo downstream de novo model {model_path}")
        model_path = self.s3.get_local_path(model_path)
        assert model_path is not None
        model, model_config = DownstreamDeNovo.load(
            model_path, override_config={"peak_embedding_dtype": "float32"} if self.config.get("mps", False) else None
        )

        return model, model_config

    def setup_data_processor(self) -> DataProcessor:
        """Setup the data processor."""
        processor = DownstreamDeNovoDataProcessor(
            residue_set=self.residue_set,
            n_peaks=self.model_config.get("n_peaks", 200),
            min_mz=self.model_config.get("min_mz", 50.0),
            max_mz=self.model_config.get("max_mz", 2500.0),
            min_intensity=self.model_config.get("min_intensity", 1e-6),
            remove_precursor_tol=self.model_config.get("remove_precursor_tol", 2.0),
            use_spectrum_utils=self.model_config.get("use_spectrum_utils", False),
            normalize_mz=self.model_config.get("normalize_mz", True),
            peak_ordering=self.model_config.get("peak_ordering", "sorted"),
            annotated=True,
            return_str=False,
            metadata_columns=None,
            search_data_manager=None,
            build_metadata=False,
        )

        return processor
