"""Tests for mean-pooled peak token embeddings.

Mean pooling over non-padding peak tokens provides an alternative to CLS-based
embeddings. For masked reconstruction models where CLS has no direct training
signal, mean pooling often yields more informative representations.
"""

import pytest
import torch
from omegaconf import OmegaConf

from instanovo_fm.model.encoder import FoundationModel


@pytest.fixture
def base_config():
    """Minimal model config for testing."""
    return {
        "dim_model": 64,
        "n_heads": 4,
        "dim_feedforward": 128,
        "n_layers": 2,
        "dropout": 0.0,
        "n_peaks": 20,
        "max_mz": 2500.0,
        "min_mz": 50.0,
        "max_charge": 10,
        "peak_encoder": {"type": "multiscale"},
        "mz_head": {
            "task": "classification",
            "binning": {"strategy": "fixed_da", "bin_size": 0.1},
            "bin_group_size": 100,
            "w_group": 0.5,
            "w_offset": 0.5,
        },
        "meta_token": {"enabled": False},
        "auxiliary": {"enabled": False},
        "architecture": {
            "positional_encoding": {"type": "none"},
            "relative_bias": {"type": "none"},
            "attention": {"backend": "math"},
        },
    }


def _model_kwargs(cfg):
    """Extract FoundationModel constructor kwargs from config dict."""
    return dict(
        dim_model=cfg["dim_model"],
        n_heads=cfg["n_heads"],
        dim_feedforward=cfg["dim_feedforward"],
        n_layers=cfg["n_layers"],
        dropout=cfg["dropout"],
        n_peaks=cfg["n_peaks"],
        max_mz=cfg["max_mz"],
        min_mz=cfg["min_mz"],
        max_charge=cfg["max_charge"],
        peak_encoder_type=cfg.get("peak_encoder", {}).get("type", "multiscale"),
        mz_task=cfg.get("mz_head", {}).get("task", "classification"),
        use_meta_token=cfg.get("meta_token", {}).get("enabled", False),
    )


@pytest.fixture
def model(base_config):
    """Create a small foundation model."""
    return FoundationModel(cfg=OmegaConf.create(base_config), **_model_kwargs(base_config))


@pytest.fixture
def sample_spectra():
    """Create sample spectra with some padding."""
    B, L = 4, 20
    spectra = torch.zeros(B, L, 2)
    for i in range(B):
        n_real = 10 + i * 2  # Variable number of real peaks (10, 12, 14, 16)
        mz_values = torch.sort(torch.rand(n_real) * 0.9 + 0.05)[0]
        intensities = torch.rand(n_real) * 0.8 + 0.1
        spectra[i, :n_real, 0] = mz_values
        spectra[i, :n_real, 1] = intensities
    return spectra


class TestMeanPoolOutputShape:
    """Test output shape and normalization."""

    def test_output_shape(self, model, sample_spectra):
        """Output is (B, D), L2-normalized."""
        embeddings = model.encode_mean_pooled(sample_spectra)
        assert embeddings.shape == (sample_spectra.shape[0], 64)

    def test_l2_normalized(self, model, sample_spectra):
        """Output vectors have unit L2 norm."""
        embeddings = model.encode_mean_pooled(sample_spectra)
        norms = torch.norm(embeddings, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_output_on_cpu(self, model, sample_spectra):
        """Output is on CPU regardless of model device."""
        embeddings = model.encode_mean_pooled(sample_spectra)
        assert embeddings.device == torch.device("cpu")


class TestMeanPoolPadding:
    """Test that padding is correctly excluded."""

    def test_padding_exclusion(self, model):
        """Padded peaks don't contribute to mean."""
        B, L, D = 2, 20, 64

        # Create two spectra: one with 5 real peaks, one with 10
        spectra = torch.zeros(B, L, 2)
        for i, n_real in enumerate([5, 10]):
            mz_values = torch.sort(torch.rand(n_real) * 0.9 + 0.05)[0]
            intensities = torch.rand(n_real) * 0.8 + 0.1
            spectra[i, :n_real, 0] = mz_values
            spectra[i, :n_real, 1] = intensities

        embeddings = model.encode_mean_pooled(spectra)

        # Both should produce valid embeddings (no NaN)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()

        # Embeddings should differ (different number of peaks = different content)
        assert not torch.allclose(embeddings[0], embeddings[1], atol=1e-3)

    def test_all_padding_edge_case(self, model):
        """All-padding input returns valid tensor (no NaN/inf)."""
        B, L = 2, 20
        spectra = torch.zeros(B, L, 2)  # All zeros = all padding

        embeddings = model.encode_mean_pooled(spectra)

        assert embeddings.shape == (B, 64)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()


class TestMeanPoolVsCLS:
    """Test that mean pool differs from CLS."""

    def test_differs_from_cls(self, model, sample_spectra):
        """encode_mean_pooled() output != encode() output."""
        cls_embeddings = model.encode(sample_spectra)
        mean_pool_embeddings = model.encode_mean_pooled(sample_spectra)

        # Should not be identical (different pooling strategies)
        assert not torch.allclose(cls_embeddings, mean_pool_embeddings, atol=1e-3), (
            "Mean pool embeddings should differ from CLS embeddings"
        )


class TestMeanPoolDeterminism:
    """Test reproducibility."""

    def test_deterministic(self, model, sample_spectra):
        """Same input -> same output across calls."""
        emb1 = model.encode_mean_pooled(sample_spectra)
        emb2 = model.encode_mean_pooled(sample_spectra)
        assert torch.allclose(emb1, emb2, atol=1e-6)

    def test_restores_training_mode(self, base_config, sample_spectra):
        """Model returns to training mode after encode_mean_pooled() if it was training."""
        model = FoundationModel(cfg=OmegaConf.create(base_config), **_model_kwargs(base_config))
        model.train()
        assert model.training

        _ = model.encode_mean_pooled(sample_spectra)
        assert model.training, "Model should be back in training mode"

    def test_stays_in_eval_mode(self, base_config, sample_spectra):
        """Model stays in eval mode after encode_mean_pooled() if it was eval."""
        model = FoundationModel(cfg=OmegaConf.create(base_config), **_model_kwargs(base_config))
        model.eval()
        assert not model.training

        _ = model.encode_mean_pooled(sample_spectra)
        assert not model.training, "Model should remain in eval mode"
