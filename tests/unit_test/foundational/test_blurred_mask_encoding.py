"""Tests for Gaussian-blurred mask encoding.

Verifies that when masking.blur_sigma_da > 0, masked peaks receive a
Gaussian-blurred version of their m/z encoding (via peak encoder) plus
the learned mask_token bias, instead of the generic uniform [MASK] token.
"""

import pytest
import torch

from instanovo_fm.model.encoder import FoundationModel


@pytest.fixture
def base_config():
    """Minimal model config with blur disabled (default)."""
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
        "masking": {"blur_sigma_da": 0.0},
        "architecture": {
            "positional_encoding": {"type": "none"},
            "relative_bias": {"type": "none"},
            "attention": {"backend": "math"},
        },
    }


@pytest.fixture
def blur_config(base_config):
    """Config with blur enabled (sigma=25 Da)."""
    base_config["masking"] = {"blur_sigma_da": 25.0}
    return base_config


@pytest.fixture
def sample_spectra():
    """Create sample spectra with distinct m/z values."""
    B, L = 4, 20
    spectra = torch.zeros(B, L, 2)
    for i in range(B):
        n_real = 15
        mz_values = torch.sort(torch.rand(n_real) * 0.9 + 0.05)[0]
        intensities = torch.rand(n_real) * 0.8 + 0.1
        spectra[i, :n_real, 0] = mz_values
        spectra[i, :n_real, 1] = intensities
    return spectra


@pytest.fixture
def mlm_mask():
    """Create MLM mask masking ~30% of first 15 peaks."""
    B, L = 4, 20
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i in range(B):
        indices = torch.randperm(15)[:5]
        mask[i, indices] = True
    return mask


class TestBlurConfig:
    """Test config reading and attribute setting."""

    def test_blur_disabled_by_default(self, base_config):
        model = FoundationModel(cfg=base_config, dim_model=64, n_heads=4, n_peaks=20)
        assert model.blur_sigma_da == 0.0

    def test_blur_config_from_dict(self, blur_config):
        model = FoundationModel(cfg=blur_config, dim_model=64, n_heads=4, n_peaks=20)
        assert model.blur_sigma_da == 25.0

    def test_blur_config_missing_masking_section(self, base_config):
        """Missing masking section defaults to 0.0."""
        del base_config["masking"]
        model = FoundationModel(cfg=base_config, dim_model=64, n_heads=4, n_peaks=20)
        assert model.blur_sigma_da == 0.0


class TestBlurMasking:
    """Test the _apply_mlm_mask method with blur."""

    def test_blur_disabled_is_identical(self, base_config, sample_spectra, mlm_mask):
        """sigma=0 produces same result whether or not spectra is passed."""
        model = FoundationModel(cfg=base_config, dim_model=64, n_heads=4, n_peaks=20)
        model.eval()
        with torch.no_grad():
            x = model._embed_peaks(sample_spectra)
            x1 = model._apply_mlm_mask(x.clone(), mlm_mask)
            x2 = model._apply_mlm_mask(x.clone(), mlm_mask, spectra=sample_spectra)
        assert torch.allclose(x1, x2), "blur_sigma_da=0 should produce identical results"

    def test_blur_differentiates_mask_tokens(self, blur_config, sample_spectra, mlm_mask):
        """With blur, different masked peaks get different embeddings."""
        model = FoundationModel(cfg=blur_config, dim_model=64, n_heads=4, n_peaks=20)
        model.eval()
        with torch.no_grad():
            x = model._embed_peaks(sample_spectra)
            x_masked = model._apply_mlm_mask(x.clone(), mlm_mask, spectra=sample_spectra)

        # Extract embeddings at two different masked positions in batch 0
        masked_indices = mlm_mask[0].nonzero(as_tuple=True)[0]
        assert len(masked_indices) >= 2, "Need at least 2 masked positions"
        emb_a = x_masked[0, masked_indices[0]]
        emb_b = x_masked[0, masked_indices[1]]
        assert not torch.allclose(emb_a, emb_b, atol=1e-4), (
            "Different masked peaks should have different blurred embeddings"
        )

    def test_blur_is_stochastic(self, blur_config, sample_spectra, mlm_mask):
        """Two calls with same input produce different masked embeddings."""
        model = FoundationModel(cfg=blur_config, dim_model=64, n_heads=4, n_peaks=20)
        model.eval()
        with torch.no_grad():
            x = model._embed_peaks(sample_spectra)
            x1 = model._apply_mlm_mask(x.clone(), mlm_mask, spectra=sample_spectra)
            x2 = model._apply_mlm_mask(x.clone(), mlm_mask, spectra=sample_spectra)

        # At masked positions, embeddings should differ (different noise samples)
        masked = mlm_mask.unsqueeze(-1).expand_as(x1)
        diff = (x1[masked] - x2[masked]).abs().max().item()
        assert diff > 1e-6, "Blurred mask should be stochastic across calls"

    def test_blur_preserves_unmasked(self, blur_config, sample_spectra, mlm_mask):
        """Unmasked positions are unchanged by blur."""
        model = FoundationModel(cfg=blur_config, dim_model=64, n_heads=4, n_peaks=20)
        model.eval()
        with torch.no_grad():
            x = model._embed_peaks(sample_spectra)
            x_orig = x.clone()
            x_masked = model._apply_mlm_mask(x, mlm_mask, spectra=sample_spectra)

        # Unmasked positions should be identical
        unmasked = ~mlm_mask.unsqueeze(-1).expand_as(x_orig)
        assert torch.allclose(x_masked[unmasked], x_orig[unmasked]), (
            "Unmasked positions must be unchanged"
        )

    def test_blur_output_shape(self, blur_config, sample_spectra, mlm_mask):
        """Output shape is (B, L, D) regardless of blur."""
        model = FoundationModel(cfg=blur_config, dim_model=64, n_heads=4, n_peaks=20)
        model.eval()
        with torch.no_grad():
            x = model._embed_peaks(sample_spectra)
            x_masked = model._apply_mlm_mask(x, mlm_mask, spectra=sample_spectra)
        assert x_masked.shape == x.shape

    def test_blur_with_3channel_spectra(self, blur_config, mlm_mask):
        """Works with (B, L, 3) spectra [m/z, intensity, charge]."""
        model = FoundationModel(cfg=blur_config, dim_model=64, n_heads=4, n_peaks=20)
        model.eval()
        B, L = 4, 20
        spectra_3ch = torch.zeros(B, L, 3)
        for i in range(B):
            spectra_3ch[i, :15, 0] = torch.sort(torch.rand(15) * 0.9 + 0.05)[0]
            spectra_3ch[i, :15, 1] = torch.rand(15) * 0.8 + 0.1
            spectra_3ch[i, :15, 2] = 2.0  # charge
        with torch.no_grad():
            x = model._embed_peaks(spectra_3ch[..., :2])  # peak encoder expects 2ch
            x_masked = model._apply_mlm_mask(x, mlm_mask, spectra=spectra_3ch)
        assert x_masked.shape == (B, L, 64)

    def test_blur_clamps_to_valid_range(self, base_config, mlm_mask):
        """Very large sigma doesn't produce out-of-range values."""
        base_config["masking"] = {"blur_sigma_da": 500.0}
        model = FoundationModel(cfg=base_config, dim_model=64, n_heads=4, n_peaks=20)
        model.eval()
        B, L = 4, 20
        spectra = torch.zeros(B, L, 2)
        spectra[:, :15, 0] = torch.sort(torch.rand(B, 15) * 0.9 + 0.05, dim=-1)[0]
        spectra[:, :15, 1] = torch.rand(B, 15) * 0.8 + 0.1
        with torch.no_grad():
            x = model._embed_peaks(spectra)
            # Should not raise or produce NaN
            x_masked = model._apply_mlm_mask(x, mlm_mask, spectra=spectra)
        assert not torch.isnan(x_masked).any(), "No NaN values allowed"
        assert not torch.isinf(x_masked).any(), "No Inf values allowed"


class TestBlurGradients:
    """Test gradient flow through blurred mask encoding."""

    def test_mask_token_receives_gradients(self, blur_config, sample_spectra, mlm_mask):
        """mask_token should receive gradients through the blur path."""
        model = FoundationModel(cfg=blur_config, dim_model=64, n_heads=4, n_peaks=20)
        model.train()
        x = model._embed_peaks(sample_spectra)
        x_masked = model._apply_mlm_mask(x, mlm_mask, spectra=sample_spectra)
        # Compute a simple loss at masked positions
        loss = x_masked[mlm_mask].sum()
        loss.backward()
        assert model.mask_token.grad is not None, "mask_token must receive gradients"
        assert model.mask_token.grad.abs().sum() > 0, "mask_token gradient must be nonzero"

    def test_peak_encoder_receives_gradients(self, blur_config, sample_spectra, mlm_mask):
        """Peak encoder should receive gradients through the blur path."""
        model = FoundationModel(cfg=blur_config, dim_model=64, n_heads=4, n_peaks=20)
        model.train()
        x = model._embed_peaks(sample_spectra)
        x_masked = model._apply_mlm_mask(x, mlm_mask, spectra=sample_spectra)
        loss = x_masked[mlm_mask].sum()
        loss.backward()
        # Check that peak encoder parameters received gradients
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.peak_encoder.parameters()
        )
        assert has_grad, "Peak encoder must receive gradients through blur path"


class TestBlurFullForward:
    """Test full model forward + backward with blur enabled."""

    def test_full_forward_backward(self, blur_config, sample_spectra, mlm_mask):
        """Full model forward + backward pass with blur."""
        model = FoundationModel(
            cfg=blur_config, dim_model=64, n_heads=4, n_peaks=20, mz_task="regression"
        )
        model.train()
        predictions, aux = model(sample_spectra, mlm_mask=mlm_mask)
        # Regression returns (B, L, 1)
        loss = predictions[mlm_mask].sum()
        loss.backward()
        # Verify gradient flow
        assert model.mask_token.grad is not None

    def test_full_forward_no_mask(self, blur_config, sample_spectra):
        """Forward without mlm_mask should work (blur is no-op)."""
        model = FoundationModel(
            cfg=blur_config, dim_model=64, n_heads=4, n_peaks=20, mz_task="regression"
        )
        model.eval()
        with torch.no_grad():
            predictions, aux = model(sample_spectra)
        assert predictions.shape[0] == sample_spectra.shape[0]


class TestVisibleIntensity:
    """Test mask_intensity=false (keep intensity visible at masked positions)."""

    def test_visible_intensity_preserves_intensity(self, sample_spectra, mlm_mask):
        """With mask_intensity=false, different intensities produce different embeddings."""
        config_vi = {
            "masking": {"blur_sigma_da": 25.0, "mask_intensity": False},
            "architecture": {
                "positional_encoding": {"type": "none"},
                "relative_bias": {"type": "none"},
                "attention": {"backend": "math"},
            },
        }
        model = FoundationModel(cfg=config_vi, dim_model=64, n_heads=4, n_peaks=20)
        model.eval()

        # Create two spectra with different intensities at same m/z
        spec_high = sample_spectra.clone()
        spec_low = sample_spectra.clone()
        spec_high[:, :, 1] = 0.9  # high intensity
        spec_low[:, :, 1] = 0.1   # low intensity

        with torch.no_grad():
            x_high = model._embed_peaks(spec_high)
            x_low = model._embed_peaks(spec_low)
            masked_high = model._apply_mlm_mask(x_high.clone(), mlm_mask, spectra=spec_high)
            masked_low = model._apply_mlm_mask(x_low.clone(), mlm_mask, spectra=spec_low)

        # Masked positions should differ (different intensities)
        masked_idx = mlm_mask[0].nonzero(as_tuple=True)[0][0].item()
        assert not torch.allclose(
            masked_high[0, masked_idx], masked_low[0, masked_idx], atol=1e-4
        ), "Visible intensity should produce different embeddings for different intensities"

    def test_default_intensity_masked(self, base_config, sample_spectra, mlm_mask):
        """With mask_intensity=true (default), intensity is zeroed."""
        config = {**base_config, "masking": {"blur_sigma_da": 25.0, "mask_intensity": True}}
        model = FoundationModel(cfg=config, dim_model=64, n_heads=4, n_peaks=20)
        model.eval()

        spec_high = sample_spectra.clone()
        spec_high[:, :, 1] = 0.9
        spec_low = sample_spectra.clone()
        spec_low[:, :, 1] = 0.1

        torch.manual_seed(42)
        with torch.no_grad():
            x_high = model._embed_peaks(spec_high)
            masked_high = model._apply_mlm_mask(x_high.clone(), mlm_mask, spectra=spec_high)

        torch.manual_seed(42)
        with torch.no_grad():
            x_low = model._embed_peaks(spec_low)
            masked_low = model._apply_mlm_mask(x_low.clone(), mlm_mask, spectra=spec_low)

        # With same noise seed and masked intensity, masked embeddings should be identical
        masked_idx = mlm_mask[0].nonzero(as_tuple=True)[0][0].item()
        assert torch.allclose(
            masked_high[0, masked_idx], masked_low[0, masked_idx], atol=1e-4
        ), "Masked intensity should produce identical embeddings regardless of intensity"

    def test_visible_intensity_requires_blur(self, base_config):
        """mask_intensity=false without blur should fall back to true."""
        config = {**base_config, "masking": {"blur_sigma_da": 0.0, "mask_intensity": False}}
        model = FoundationModel(cfg=config, dim_model=64, n_heads=4, n_peaks=20)
        assert model.mask_intensity is True, "Should fall back when blur disabled"


class TestBlurMaskedPA:
    """Test blur_masked_pa (blurred m/z for PA at masked positions)."""

    def _make_pa_config(self, blur_masked_pa=False, blur_sigma=25.0):
        return {
            "masking": {"blur_sigma_da": blur_sigma},
            "architecture": {
                "positional_encoding": {"type": "none"},
                "relative_bias": {
                    "type": "pa",
                    "config": {
                        "pw_num_freqs": 8, "pw_hidden_dim": 16,
                        "per_layer_pw": True,
                        "lambda_min": 0.001, "lambda_max": 10000.0,
                        "blur_masked_pa": blur_masked_pa,
                    },
                },
                "attention": {"backend": "math"},
            },
        }

    def test_blur_pa_nonzero_at_masked(self, sample_spectra, mlm_mask):
        """With blur_masked_pa=true, PA features at masked positions are non-zero."""
        config = self._make_pa_config(blur_masked_pa=True)
        model = FoundationModel(cfg=config, dim_model=64, n_heads=4, n_peaks=20)
        assert model.blur_masked_pa
        model.eval()
        with torch.no_grad():
            _, pairwise_feats = model._compute_attn_bias(sample_spectra, mlm_mask)
        masked_idx = mlm_mask[0].nonzero(as_tuple=True)[0][0].item()
        row = pairwise_feats[0, masked_idx]
        assert row.abs().sum() > 0, "Blurred PA should be non-zero at masked positions"

    def test_default_pa_zeroed_at_masked(self, sample_spectra, mlm_mask):
        """With blur_masked_pa=false (default), PA at masked positions is zero."""
        config = self._make_pa_config(blur_masked_pa=False)
        model = FoundationModel(cfg=config, dim_model=64, n_heads=4, n_peaks=20)
        assert not model.blur_masked_pa
        model.eval()
        with torch.no_grad():
            _, pairwise_feats = model._compute_attn_bias(sample_spectra, mlm_mask)
        masked_idx = mlm_mask[0].nonzero(as_tuple=True)[0][0].item()
        row = pairwise_feats[0, masked_idx]
        assert (row == 0).all(), "Default PA should be zero at masked positions"

    def test_blur_pa_requires_pa_enabled(self, base_config):
        """blur_masked_pa=true without PA enabled should fall back."""
        config = {
            **base_config,
            "masking": {"blur_sigma_da": 25.0},
        }
        config["architecture"]["relative_bias"] = {
            "type": "none",
            "config": {"blur_masked_pa": True},
        }
        model = FoundationModel(cfg=config, dim_model=64, n_heads=4, n_peaks=20)
        assert not model.blur_masked_pa, "Should fall back when PA not enabled"

    def test_blur_pa_requires_blur(self):
        """blur_masked_pa=true without blur should fall back."""
        config = {
            "masking": {"blur_sigma_da": 0.0},
            "architecture": {
                "positional_encoding": {"type": "none"},
                "relative_bias": {
                    "type": "pa",
                    "config": {
                        "pw_num_freqs": 8, "pw_hidden_dim": 16,
                        "per_layer_pw": True,
                        "lambda_min": 0.001, "lambda_max": 10000.0,
                        "blur_masked_pa": True,
                    },
                },
                "attention": {"backend": "math"},
            },
        }
        model = FoundationModel(cfg=config, dim_model=64, n_heads=4, n_peaks=20)
        assert not model.blur_masked_pa, "Should fall back when blur disabled"


class TestPAMaskToMask:
    """Test pa_mask_to_mask (exact PA between masked peaks, zero cross-pairs)."""

    def _make_pa_config(self, pa_mask_to_mask=False):
        return {
            "architecture": {
                "positional_encoding": {"type": "none"},
                "relative_bias": {
                    "type": "pa",
                    "config": {
                        "pw_num_freqs": 8, "pw_hidden_dim": 16,
                        "per_layer_pw": True,
                        "lambda_min": 0.001, "lambda_max": 10000.0,
                        "pa_mask_to_mask": pa_mask_to_mask,
                    },
                },
                "attention": {"backend": "math"},
            },
        }

    def test_pa_m2m_keeps_masked_pairs(self, sample_spectra, mlm_mask):
        """PA between two masked positions should be non-zero."""
        config = self._make_pa_config(pa_mask_to_mask=True)
        model = FoundationModel(cfg=config, dim_model=64, n_heads=4, n_peaks=20)
        assert model.pa_mask_to_mask
        model.eval()
        with torch.no_grad():
            _, pairwise_feats = model._compute_attn_bias(sample_spectra, mlm_mask)
        # Find two masked positions in batch 0
        masked_idx = mlm_mask[0].nonzero(as_tuple=True)[0]
        assert len(masked_idx) >= 2
        i, j = masked_idx[0].item(), masked_idx[1].item()
        pa_val = pairwise_feats[0, i, j]
        assert pa_val.abs().sum() > 0, "Mask-to-mask PA should be non-zero"

    def test_pa_m2m_zeros_cross_pairs(self, sample_spectra, mlm_mask):
        """PA between masked and visible positions should be zero."""
        config = self._make_pa_config(pa_mask_to_mask=True)
        model = FoundationModel(cfg=config, dim_model=64, n_heads=4, n_peaks=20)
        model.eval()
        with torch.no_grad():
            _, pairwise_feats = model._compute_attn_bias(sample_spectra, mlm_mask)
        # Find one masked and one visible (non-masked, non-padded) position
        masked_idx = mlm_mask[0].nonzero(as_tuple=True)[0][0].item()
        visible_mask = ~mlm_mask[0] & (sample_spectra[0, :, 0] > 0)
        visible_idx = visible_mask.nonzero(as_tuple=True)[0][0].item()
        pa_cross = pairwise_feats[0, masked_idx, visible_idx]
        assert (pa_cross == 0).all(), "Mask-to-visible PA should be zeroed"

    def test_pa_m2m_keeps_visible_pairs(self, sample_spectra, mlm_mask):
        """PA between two visible positions should be non-zero (unchanged)."""
        config = self._make_pa_config(pa_mask_to_mask=True)
        model = FoundationModel(cfg=config, dim_model=64, n_heads=4, n_peaks=20)
        model.eval()
        with torch.no_grad():
            _, pairwise_feats = model._compute_attn_bias(sample_spectra, mlm_mask)
        # Find two visible positions
        visible_mask = ~mlm_mask[0] & (sample_spectra[0, :, 0] > 0)
        visible_idx = visible_mask.nonzero(as_tuple=True)[0]
        assert len(visible_idx) >= 2
        i, j = visible_idx[0].item(), visible_idx[1].item()
        pa_val = pairwise_feats[0, i, j]
        assert pa_val.abs().sum() > 0, "Visible-to-visible PA should be non-zero"

    def test_pa_m2m_requires_pa(self, base_config):
        """pa_mask_to_mask=true without PA enabled should fall back."""
        config = {
            **base_config,
            "architecture": {
                **base_config.get("architecture", {}),
                "relative_bias": {
                    "type": "none",
                    "config": {"pa_mask_to_mask": True},
                },
                "attention": {"backend": "math"},
            },
        }
        model = FoundationModel(cfg=config, dim_model=64, n_heads=4, n_peaks=20)
        assert not model.pa_mask_to_mask, "Should fall back when PA not enabled"
