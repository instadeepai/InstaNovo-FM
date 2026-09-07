"""Unit tests for confidence-weighted peak pooling (encode_mean_pooled with pooling='confidence')."""

import pytest
import torch

from instanovo_fm.model import FoundationModel


def _make_classification_config():
    """Build a minimal classification model config."""
    return {
        "dim_model": 64,
        "n_heads": 4,
        "dim_feedforward": 128,
        "n_layers": 2,
        "dropout": 0.0,
        "n_peaks": 20,
        "max_mz": 2500.0,
        "min_mz": 50.0,
        "peak_encoder": {"type": "linear"},
        "mz_head": {
            "task": "classification",
            "binning": {
                "strategy": "fixed_da",
                "bin_size": 0.5,
            },
            "bin_group_size": 50,
            "offset_conditioning": "none",
            "w_group": 0.5,
            "w_offset": 0.5,
        },
        "meta_token": {"enabled": False},
        "architecture": {
            "positional_encoding": {"type": "none", "config": {}},
            "relative_bias": {"type": "none", "config": {}},
            "attention": {"backend": "flash", "config": {}},
        },
    }


@pytest.fixture
def classification_model():
    cfg = _make_classification_config()
    model = FoundationModel(
        dim_model=64,
        n_heads=4,
        dim_feedforward=128,
        n_layers=2,
        dropout=0.0,
        n_peaks=20,
        peak_encoder_type="linear",
        mz_task="classification",
        use_meta_token=False,
        cfg=cfg,
    )
    return model


@pytest.fixture
def sample_spectra():
    """Create sample spectra with some padding."""
    B, L = 3, 20
    spectra = torch.rand(B, L, 2)
    spectra[:, :, 0] *= 0.8  # m/z in [0, 0.8]
    spectra[:, -5:] = 0.0  # Last 5 peaks are padding
    return spectra


class TestConfidencePooling:

    def test_output_shape(self, classification_model, sample_spectra):
        """Output should be (B, D) and L2-normalized."""
        emb = classification_model.encode_mean_pooled(sample_spectra, pooling="confidence")
        B = sample_spectra.shape[0]
        D = classification_model.dim_model
        assert emb.shape == (B, D)

    def test_l2_normalized(self, classification_model, sample_spectra):
        """Embeddings should be L2-normalized."""
        emb = classification_model.encode_mean_pooled(sample_spectra, pooling="confidence")
        norms = torch.norm(emb, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_differs_from_mean_pool_with_temperature(self, classification_model, sample_spectra):
        """With extreme temperature, confidence pooling should differ from mean pooling.

        At random init, confidences are nearly uniform so T=1 ≈ mean pooling.
        Very low temperature forces focus on the highest-confidence peak.
        """
        emb_conf = classification_model.encode_mean_pooled(
            sample_spectra, pooling="confidence", confidence_temperature=0.01,
        )
        emb_mean = classification_model.encode_mean_pooled(sample_spectra)
        diff = (emb_conf - emb_mean).abs().max()
        assert diff > 1e-6, f"Expected divergence with T=0.01, got max diff = {diff}"

    def test_temperature_effect(self, classification_model, sample_spectra):
        """Extreme temperature should produce different embeddings than T=1."""
        emb_t1 = classification_model.encode_mean_pooled(
            sample_spectra, pooling="confidence", confidence_temperature=1.0,
        )
        emb_t001 = classification_model.encode_mean_pooled(
            sample_spectra, pooling="confidence", confidence_temperature=0.01,
        )
        diff = (emb_t1 - emb_t001).abs().max()
        assert diff > 1e-6, f"Temperature had no effect: max diff = {diff}"

    def test_deterministic(self, classification_model, sample_spectra):
        """Same input should produce same output."""
        emb1 = classification_model.encode_mean_pooled(sample_spectra, pooling="confidence")
        emb2 = classification_model.encode_mean_pooled(sample_spectra, pooling="confidence")
        assert torch.allclose(emb1, emb2, atol=1e-6)

    def test_handles_all_padding(self, classification_model):
        """Should handle spectra that are all padding (edge case)."""
        B, L = 2, 20
        spectra = torch.zeros(B, L, 2)
        emb = classification_model.encode_mean_pooled(spectra, pooling="confidence")
        assert emb.shape == (B, 64)
        assert not torch.isnan(emb).any()

    def test_regression_fallback(self, sample_spectra):
        """Regression model should fall back to mean pooling."""
        cfg = _make_classification_config()
        cfg["mz_head"]["task"] = "regression"
        model = FoundationModel(
            dim_model=64,
            n_heads=4,
            dim_feedforward=128,
            n_layers=2,
            dropout=0.0,
            n_peaks=20,
            peak_encoder_type="linear",
            mz_task="regression",
            use_meta_token=False,
            cfg=cfg,
        )
        emb = model.encode_mean_pooled(sample_spectra, pooling="confidence")
        emb_mean = model.encode_mean_pooled(sample_spectra)
        assert torch.allclose(emb, emb_mean, atol=1e-6)

    def test_default_pooling_unchanged(self, classification_model, sample_spectra):
        """Default pooling='mean_pool' should behave identically to the old API."""
        emb_default = classification_model.encode_mean_pooled(sample_spectra)
        emb_explicit = classification_model.encode_mean_pooled(sample_spectra, pooling="mean_pool")
        assert torch.allclose(emb_default, emb_explicit, atol=1e-6)
