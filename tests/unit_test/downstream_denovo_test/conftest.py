from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from instanovo.utils.residues import ResidueSet
from omegaconf import DictConfig, OmegaConf

from instanovo_fm.downstream.de_novo_sequencing.data import DownstreamDeNovoDataProcessor
from instanovo_fm.downstream.de_novo_sequencing.model import DownstreamDeNovo


@pytest.fixture()
def _reset_seed() -> None:
    """Reset torch/numpy/random seeds so tests are deterministic."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


@pytest.fixture(scope="session")
def denovo_residue_set() -> ResidueSet:
    """Small ResidueSet matching tests/configs/residues/unit_test.yaml (A-E)."""
    return ResidueSet(
        residue_masses={"A": 10.5, "B": 20.75, "C": 15.68, "D": 18.25, "E": 12.33},
        residue_remapping={},
    )


@pytest.fixture(scope="session")
def tiny_denovo_config() -> DictConfig:
    """Tiny CPU-friendly config for the DownstreamDeNovo model."""
    return OmegaConf.create(
        {
            "dim_model": 16,
            "n_head": 2,
            "dim_feedforward": 32,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "dropout": 0.0,
            "use_flash_attention": False,
            "conv_peak_encoder": False,
            "use_meta_token": False,
        }
    )


@pytest.fixture(scope="session")
def matched_denovo_model_factory(
    denovo_residue_set: ResidueSet,
) -> Callable[..., DownstreamDeNovo]:
    """Factory: build a tiny DownstreamDeNovo with a foundation-matched config.

    Parameterized over attention backend ("math" or "flash") so callers can
    construct both variants in a single test. Mirrors the architecture
    contract used by load_encoder_from_foundation: multiscale peak encoder,
    no positional encoding, no relative bias, ion_ladder off — the
    minimum surface that matches a foundation checkpoint.
    """

    def _factory(backend: str = "math", dim_model: int = 16) -> DownstreamDeNovo:
        use_flash = backend == "flash"
        cfg: dict[str, Any] = {
            "dim_model": dim_model,
            "n_head": 2,
            "dim_feedforward": 32,
            "encoder_layers": 1,
            "decoder_layers": 1,
            "dropout": 0.0,
            "use_flash_attention": use_flash,
            "conv_peak_encoder": False,
            "use_meta_token": False,
            "max_mz": 2500.0,
            "min_mz": 0.0,
            "peak_encoder": {"type": "multiscale", "config": {"dropout": 0.0}},
            "ion_ladder": {"enabled": False},
            "architecture": {
                "positional_encoding": {"type": "none"},
                "relative_bias": {"type": "none"},
                "attention": {"backend": backend},
            },
        }
        torch.manual_seed(42)
        model = DownstreamDeNovo(
            residue_set=denovo_residue_set,
            dim_model=cfg["dim_model"],
            n_head=cfg["n_head"],
            dim_feedforward=cfg["dim_feedforward"],
            encoder_layers=cfg["encoder_layers"],
            decoder_layers=cfg["decoder_layers"],
            dropout=cfg["dropout"],
            use_flash_attention=cfg["use_flash_attention"],
            conv_peak_encoder=cfg["conv_peak_encoder"],
            peak_encoder_type=cfg["peak_encoder"]["type"],
            max_mz=cfg["max_mz"],
            min_mz=cfg["min_mz"],
            use_meta_token=cfg["use_meta_token"],
            cfg=cfg,
        )
        model.eval()
        return model

    return _factory


@pytest.fixture(scope="session")
def tiny_denovo_model(
    denovo_residue_set: ResidueSet,
    tiny_denovo_config: DictConfig,
) -> DownstreamDeNovo:
    """Construct a tiny DownstreamDeNovo model on CPU in eval mode."""
    model = DownstreamDeNovo(
        residue_set=denovo_residue_set,
        dim_model=tiny_denovo_config.dim_model,
        n_head=tiny_denovo_config.n_head,
        dim_feedforward=tiny_denovo_config.dim_feedforward,
        encoder_layers=tiny_denovo_config.encoder_layers,
        decoder_layers=tiny_denovo_config.decoder_layers,
        dropout=tiny_denovo_config.dropout,
        use_flash_attention=tiny_denovo_config.use_flash_attention,
        conv_peak_encoder=tiny_denovo_config.conv_peak_encoder,
        use_meta_token=tiny_denovo_config.use_meta_token,
        cfg=tiny_denovo_config,
    )
    model.eval()
    return model


@pytest.fixture()
def tiny_denovo_processor(
    denovo_residue_set: ResidueSet,
    _reset_seed: None,
) -> DownstreamDeNovoDataProcessor:
    """Processor configured for deterministic, CPU-only, no-masking tests."""
    return DownstreamDeNovoDataProcessor(
        residue_set=denovo_residue_set,
        n_peaks=10,
        min_mz=50.0,
        max_mz=1000.0,
        min_intensity=0.01,
        remove_precursor_tol=0.0,
        use_spectrum_utils=False,
        normalize_mz=False,
        annotated=True,
        return_str=False,
        reverse_peptide=True,
        add_eos=True,
    )


@pytest.fixture()
def tiny_spectrum_batch(_reset_seed: None) -> dict[str, torch.Tensor]:
    """Small synthetic batch for forward-pass tests.

    Shapes follow the contract enforced by `_collate_batch`:
      spectra      (B, n_peaks, 2)
      precursors   (B, 3) [mass, charge, mz]
      spectra_mask (B, n_peaks)
      peptides     (B, L)
    """
    batch_size = 2
    n_peaks = 10
    spectra = torch.rand(batch_size, n_peaks, 2)
    precursors = torch.tensor(
        [[500.0, 2.0, 251.0], [800.0, 3.0, 267.5]],
        dtype=torch.float32,
    )
    spectra_mask = torch.zeros(batch_size, n_peaks, dtype=torch.bool)
    peptides = torch.tensor(
        [[3, 4, 5, 2], [3, 5, 2, 0]],  # ends with EOS=2, padded with PAD=0
        dtype=torch.long,
    )
    peptides_mask = peptides == 0
    return {
        "spectra": spectra,
        "precursors": precursors,
        "spectra_mask": spectra_mask,
        "peptides": peptides,
        "peptides_mask": peptides_mask,
    }
