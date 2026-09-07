from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
from instanovo.utils.residues import ResidueSet
from omegaconf import OmegaConf

from instanovo_fm.downstream.de_novo_sequencing.data import DownstreamDeNovoDataProcessor
from instanovo_fm.downstream.de_novo_sequencing.model import DownstreamDeNovo
from instanovo_fm.downstream.de_novo_sequencing.train import DownstreamDeNovoTrainer


def _make_stub_trainer(
    denovo_residue_set: ResidueSet,
    tiny_denovo_model: DownstreamDeNovo,
    config_overrides: dict | None = None,
) -> DownstreamDeNovoTrainer:
    """Construct a trainer without going through TransformerTrainer.__init__."""
    config = OmegaConf.create(
        {
            "model": {
                "dim_model": tiny_denovo_model.dim_model,
                "n_head": 2,
                "dim_feedforward": 32,
                "encoder_layers": 1,
                "decoder_layers": 1,
                "dropout": 0.0,
                "max_charge": 5,
                "use_flash_attention": False,
                "conv_peak_encoder": False,
                "use_meta_token": False,
                "n_peaks": 10,
                "min_mz": 50.0,
                "max_mz": 1000.0,
            },
            "dataset": {"metadata_columns": None},
            "mps": False,
        }
    )
    if config_overrides:
        config = OmegaConf.merge(config, OmegaConf.create(config_overrides))

    with patch("instanovo_fm.baselines.transformer_train.TransformerTrainer.__init__", return_value=None):
        trainer = DownstreamDeNovoTrainer(config)
    trainer.config = config
    trainer.residue_set = denovo_residue_set
    trainer.model = tiny_denovo_model
    trainer.device = torch.device("cpu")
    trainer._s3 = MagicMock()
    return trainer


def test_setup_model_from_scratch_returns_denovo_module(denovo_residue_set: ResidueSet, tiny_denovo_model: DownstreamDeNovo) -> None:
    """Test that setup_model creates a DownstreamDeNovo with the correct vocab size."""
    trainer = _make_stub_trainer(denovo_residue_set, tiny_denovo_model)
    model = trainer.setup_model()
    assert isinstance(model, DownstreamDeNovo)
    assert model.vocab_size == len(denovo_residue_set)


def test_setup_model_with_pretrained_copies_encoder(denovo_residue_set: ResidueSet, tiny_denovo_model: DownstreamDeNovo) -> None:
    """Test that setup_model loads encoder weights from a pretrained foundation model."""
    trainer = _make_stub_trainer(
        denovo_residue_set,
        tiny_denovo_model,
        config_overrides={"use_pretrained_model": True, "foundational_checkpoint": "s3://bucket/ckpt"},
    )
    trainer.s3.get_local_path.return_value = "/local/ckpt"

    with patch.object(
        DownstreamDeNovo,
        "load_encoder_from_foundation",
    ) as mock_load_encoder:
        model = trainer.setup_model()

    trainer.s3.get_local_path.assert_called_once_with("s3://bucket/ckpt")
    mock_load_encoder.assert_called_once_with("/local/ckpt")
    assert isinstance(model, DownstreamDeNovo)
    assert model.decoder


def test_setup_data_processors_returns_pair(denovo_residue_set: ResidueSet, tiny_denovo_model: DownstreamDeNovo) -> None:
    """Test that setup_data_processors returns train and valid processors with correct config."""
    trainer = _make_stub_trainer(denovo_residue_set, tiny_denovo_model)
    with patch(
        "instanovo_fm.downstream.de_novo_sequencing.train.create_search_data_manager",
        return_value=None,
    ):
        train_proc, valid_proc = trainer.setup_data_processors()
    assert isinstance(train_proc, DownstreamDeNovoDataProcessor)
    assert isinstance(valid_proc, DownstreamDeNovoDataProcessor)
    assert train_proc.residue_set is denovo_residue_set
    assert valid_proc.residue_set is denovo_residue_set
    assert train_proc.n_peaks == 10
    assert valid_proc.n_peaks == 10
