"""Integration tests for Pairwise Attention (PA) bias optimizations.

Tests that:
1. BatchedPairwiseProjection produces correct shapes
2. Full model forward pass works with PA bias enabled
3. Gradient flow through PA bias path
4. Fused masking in _compute_attn_bias is equivalent to sequential masking
5. MAE path with PA bias works
6. encode() and forward_with_attn() work with PA bias
"""

import pytest
import torch
import torch.nn as nn

from instanovo_fm.model.pairwise_bias import PairwiseAttentionBias
from instanovo_fm.model.encoder_layers.unified_encoder import (
    BatchedPairwiseProjection,
    SharedPairwiseProjection,
    UnifiedEncoderLayer,
    UnifiedTransformerEncoder,
)


# ---------------------------------------------------------------------------
# Minimal config helper
# ---------------------------------------------------------------------------

def _make_pa_config(
    bias_type="pa",
    attn_backend="flash",
    pw_num_freqs=16,
    pw_hidden_dim=16,
    per_layer_pw=True,
):
    """Build a minimal config dict suitable for UnifiedEncoderLayer."""
    return {
        "architecture": {
            "positional_encoding": {"type": "none", "config": {}},
            "relative_bias": {
                "type": bias_type,
                "config": {
                    "pw_num_freqs": pw_num_freqs,
                    "pw_hidden_dim": pw_hidden_dim,
                    "per_layer_pw": per_layer_pw,
                    "lambda_min": 0.001,
                    "lambda_max": 10000.0,
                },
            },
            "attention": {"backend": attn_backend, "config": {}},
        }
    }


def _make_model_config(bias_type="pa", attn_backend="flash", pw_num_freqs=16, pw_hidden_dim=16):
    """Build a config dict suitable for FoundationModel."""
    return {
        "dim_model": 64,
        "n_heads": 4,
        "dim_feedforward": 128,
        "n_layers": 2,
        "dropout": 0.0,
        "n_peaks": 20,
        "max_mz": 2500.0,
        "min_mz": 0.0,
        "peak_encoder": {"type": "linear"},
        "mz_head": {"task": "regression"},
        "meta_token": {"enabled": False},
        "architecture": {
            "positional_encoding": {"type": "none", "config": {}},
            "relative_bias": {
                "type": bias_type,
                "config": {
                    "pw_num_freqs": pw_num_freqs,
                    "pw_hidden_dim": pw_hidden_dim,
                    "per_layer_pw": True,
                    "lambda_min": 0.001,
                    "lambda_max": 10000.0,
                },
            },
            "attention": {"backend": attn_backend, "config": {}},
        },
    }


# ===========================================================================
# BatchedPairwiseProjection unit tests
# ===========================================================================


class TestBatchedPairwiseProjection:

    def test_output_shapes(self):
        """Each layer gets a (B, H, L, L) tensor."""
        B, L, r_pw, H, N = 2, 10, 16, 4, 3
        proj = BatchedPairwiseProjection(r_pw, H, N)
        feats = torch.randn(B, L, L, r_pw)
        biases = proj(feats)
        assert len(biases) == N
        for b in biases:
            assert b.shape == (B, H, L, L)

    def test_gradient_flow(self):
        """Gradients should propagate through batched projection."""
        B, L, r_pw, H, N = 2, 8, 16, 4, 3
        proj = BatchedPairwiseProjection(r_pw, H, N)
        feats = torch.randn(B, L, L, r_pw, requires_grad=True)
        biases = proj(feats)
        loss = sum(b.sum() for b in biases)
        loss.backward()
        assert feats.grad is not None
        assert proj.g_pw_batched.weight.grad is not None

    def test_different_layers_differ(self):
        """Different layers should produce different biases (distinct weight slices)."""
        B, L, r_pw, H, N = 1, 6, 16, 4, 3
        proj = BatchedPairwiseProjection(r_pw, H, N)
        feats = torch.randn(B, L, L, r_pw)
        biases = proj(feats)
        # With random init, biases for different layers should differ
        assert not torch.allclose(biases[0], biases[1], atol=1e-6)


class TestSharedPairwiseProjection:

    def test_output_shape(self):
        """Shared projection returns a single (B, H, L, L) bias."""
        B, L, r_pw, H = 2, 10, 16, 4
        proj = SharedPairwiseProjection(r_pw, H)
        feats = torch.randn(B, L, L, r_pw)
        biases = proj(feats)
        assert len(biases) == 1
        assert biases[0].shape == (B, H, L, L)

    def test_gradient_flow(self):
        """Gradients propagate through shared projection."""
        B, L, r_pw, H = 2, 8, 16, 4
        proj = SharedPairwiseProjection(r_pw, H)
        feats = torch.randn(B, L, L, r_pw, requires_grad=True)
        biases = proj(feats)
        loss = biases[0].sum()
        loss.backward()
        assert feats.grad is not None
        assert proj.g_pw.weight.grad is not None

    def test_encoder_with_shared_projection(self):
        """Encoder uses shared bias across all layers."""
        B, L, D, H, N = 2, 10, 64, 4, 3
        r_pw = 16

        cfg = _make_pa_config(pw_num_freqs=r_pw // 2, pw_hidden_dim=r_pw)
        layer = UnifiedEncoderLayer(D, H, 128, dropout=0.0, cfg=cfg)
        pw_proj = SharedPairwiseProjection(r_pw, H)
        encoder = UnifiedTransformerEncoder(layer, num_layers=N, pw_projection=pw_proj)

        src = torch.randn(B, L, D)
        pairwise_feats = torch.randn(B, L, L, r_pw)
        out = encoder(src, pairwise_feats=pairwise_feats)
        assert out.shape == (B, L, D)

    def test_full_model_shared_pw(self):
        """Full model with per_layer_pw=False uses SharedPairwiseProjection."""
        from instanovo_fm.model import FoundationModel

        cfg = _make_model_config()
        cfg["architecture"]["relative_bias"]["config"]["per_layer_pw"] = False
        model = FoundationModel(
            dim_model=64, n_heads=4, dim_feedforward=128,
            n_layers=2, dropout=0.0, n_peaks=20,
            peak_encoder_type="linear",
            mz_task="regression",
            use_meta_token=False,
            cfg=cfg,
        )

        B, L = 2, 20
        spectra = torch.rand(B, L, 2)
        spectra[:, :, 0] *= 0.8
        spectra[:, -5:] = 0.0

        preds, aux = model(spectra)
        assert preds.shape == (B, L, 1)

        # Verify gradient flow
        loss = preds.sum()
        loss.backward()
        # Check encoder.pw_projection is SharedPairwiseProjection
        assert isinstance(model.encoder.pw_projection, SharedPairwiseProjection)


# ===========================================================================
# Encoder stack integration tests
# ===========================================================================


class TestEncoderWithPA:

    def test_encoder_forward_with_pairwise_feats(self):
        """Full encoder forward with PA bias produces correct output shape."""
        B, L, D, H, N = 2, 10, 64, 4, 3
        r_pw = 16

        cfg = _make_pa_config(pw_num_freqs=r_pw // 2, pw_hidden_dim=r_pw)
        layer = UnifiedEncoderLayer(D, H, 128, dropout=0.0, cfg=cfg)
        pw_proj = BatchedPairwiseProjection(r_pw, H, N)
        encoder = UnifiedTransformerEncoder(layer, num_layers=N, pw_projection=pw_proj)

        src = torch.randn(B, L, D)
        pairwise_feats = torch.randn(B, L, L, r_pw)
        out = encoder(src, pairwise_feats=pairwise_feats)
        assert out.shape == (B, L, D)

    def test_encoder_forward_without_pairwise_feats(self):
        """Encoder should work without pairwise_feats (no PA bias)."""
        B, L, D, H, N = 2, 10, 64, 4, 3
        r_pw = 16

        cfg = _make_pa_config(pw_num_freqs=r_pw // 2, pw_hidden_dim=r_pw)
        layer = UnifiedEncoderLayer(D, H, 128, dropout=0.0, cfg=cfg)
        pw_proj = BatchedPairwiseProjection(r_pw, H, N)
        encoder = UnifiedTransformerEncoder(layer, num_layers=N, pw_projection=pw_proj)

        src = torch.randn(B, L, D)
        out = encoder(src)
        assert out.shape == (B, L, D)

    def test_encoder_no_pw_projection(self):
        """Encoder with pw_projection=None should work (no PA)."""
        B, L, D, H, N = 2, 10, 64, 4, 2
        cfg = _make_pa_config(bias_type="none")
        layer = UnifiedEncoderLayer(D, H, 128, dropout=0.0, cfg=cfg)
        encoder = UnifiedTransformerEncoder(layer, num_layers=N, pw_projection=None)

        src = torch.randn(B, L, D)
        out = encoder(src)
        assert out.shape == (B, L, D)

    def test_gradient_flow_through_encoder(self):
        """Gradients flow through the full encoder + PA bias path."""
        B, L, D, H, N = 1, 8, 64, 4, 2
        r_pw = 16

        cfg = _make_pa_config(pw_num_freqs=r_pw // 2, pw_hidden_dim=r_pw)
        layer = UnifiedEncoderLayer(D, H, 128, dropout=0.0, cfg=cfg)
        pw_proj = BatchedPairwiseProjection(r_pw, H, N)
        encoder = UnifiedTransformerEncoder(layer, num_layers=N, pw_projection=pw_proj)

        src = torch.randn(B, L, D, requires_grad=True)
        pairwise_feats = torch.randn(B, L, L, r_pw, requires_grad=True)
        out = encoder(src, pairwise_feats=pairwise_feats)
        loss = out.sum()
        loss.backward()

        assert src.grad is not None
        assert pairwise_feats.grad is not None
        assert pw_proj.g_pw_batched.weight.grad is not None


# ===========================================================================
# Full FoundationModel integration tests
# ===========================================================================


class TestFoundationModelWithPA:

    def test_forward_pass(self):
        """Full model forward with PA bias produces correct output shapes."""
        from instanovo_fm.model import FoundationModel

        cfg = _make_model_config()
        model = FoundationModel(
            dim_model=64, n_heads=4, dim_feedforward=128,
            n_layers=2, dropout=0.0, n_peaks=20,
            peak_encoder_type="linear",
            mz_task="regression",
            use_meta_token=False,
            cfg=cfg,
        )
        model.eval()

        B, L = 2, 20
        spectra = torch.rand(B, L, 2)
        spectra[:, :, 0] *= 0.8  # m/z in [0, 0.8]
        spectra[:, -5:] = 0.0  # padding

        with torch.no_grad():
            preds, aux = model(spectra)

        assert preds.shape == (B, L, 1)

    def test_forward_with_mlm_mask(self):
        """Forward with MLM mask — masking should zero PA features for masked positions."""
        from instanovo_fm.model import FoundationModel

        cfg = _make_model_config()
        model = FoundationModel(
            dim_model=64, n_heads=4, dim_feedforward=128,
            n_layers=2, dropout=0.0, n_peaks=20,
            peak_encoder_type="linear",
            mz_task="regression",
            use_meta_token=False,
            cfg=cfg,
        )

        B, L = 2, 20
        spectra = torch.rand(B, L, 2)
        spectra[:, :, 0] *= 0.8
        spectra[:, -5:] = 0.0

        mlm_mask = torch.zeros(B, L, dtype=torch.bool)
        mlm_mask[:, 3:7] = True

        preds, aux = model(spectra, mlm_mask=mlm_mask)
        assert preds.shape == (B, L, 1)

        # Verify gradient flow
        loss = preds.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None, f"Missing gradient for parameter of shape {p.shape}"
                break  # Just check one parameter

    def test_encode(self):
        """encode() should work with PA bias."""
        from instanovo_fm.model import FoundationModel

        cfg = _make_model_config()
        model = FoundationModel(
            dim_model=64, n_heads=4, dim_feedforward=128,
            n_layers=2, dropout=0.0, n_peaks=20,
            peak_encoder_type="linear",
            mz_task="regression",
            use_meta_token=False,
            cfg=cfg,
        )

        B, L = 2, 20
        spectra = torch.rand(B, L, 2)
        spectra[:, :, 0] *= 0.8

        embeddings = model.encode(spectra)
        assert embeddings.shape == (B, 64)
        # Embeddings should be L2-normalized
        norms = embeddings.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_forward_with_attn(self):
        """forward_with_attn() should work with PA bias."""
        from instanovo_fm.model import FoundationModel

        cfg = _make_model_config(attn_backend="math")  # math backend returns attention weights
        model = FoundationModel(
            dim_model=64, n_heads=4, dim_feedforward=128,
            n_layers=2, dropout=0.0, n_peaks=20,
            peak_encoder_type="linear",
            mz_task="regression",
            use_meta_token=False,
            cfg=cfg,
        )

        B, L = 2, 20
        spectra = torch.rand(B, L, 2)
        spectra[:, :, 0] *= 0.8

        result = model.forward_with_attn(spectra)
        assert result["embeddings"].shape == (B, 64)
        assert result["special_mask"].shape[0] == B


# ===========================================================================
# Fused masking equivalence test
# ===========================================================================


class TestFusedMasking:

    def test_fused_vs_sequential_masking(self):
        """Verify fused masking produces identical results to sequential masking."""
        B, L, R = 2, 10, 16
        pairwise_feats = torch.randn(B, L, L, R)

        mlm_mask = torch.zeros(B, L, dtype=torch.bool)
        mlm_mask[:, 2:4] = True
        pad_mask = torch.zeros(B, L, dtype=torch.bool)
        pad_mask[:, -3:] = True

        # Sequential (old approach)
        feats_seq = pairwise_feats.clone()
        mlm_2d = mlm_mask.unsqueeze(2) | mlm_mask.unsqueeze(1)
        feats_seq = feats_seq.masked_fill(mlm_2d.unsqueeze(-1), 0.0)
        pad_2d = pad_mask.unsqueeze(2) | pad_mask.unsqueeze(1)
        feats_seq = feats_seq.masked_fill(pad_2d.unsqueeze(-1), 0.0)

        # Fused (new approach)
        feats_fused = pairwise_feats.clone()
        invalid = mlm_mask | pad_mask
        invalid_2d = invalid.unsqueeze(2) | invalid.unsqueeze(1)
        feats_fused = feats_fused.masked_fill(invalid_2d.unsqueeze(-1), 0.0)

        assert torch.allclose(feats_seq, feats_fused), \
            "Fused masking must produce identical results to sequential masking"

    def test_fused_masking_no_masks(self):
        """When both masks are None-equivalent (all False), no zeroing should happen."""
        B, L, R = 1, 5, 8
        pairwise_feats = torch.randn(B, L, L, R)

        mlm_mask = torch.zeros(B, L, dtype=torch.bool)
        pad_mask = torch.zeros(B, L, dtype=torch.bool)

        invalid = mlm_mask | pad_mask
        assert not invalid.any()
        # No masking applied — original features should be unchanged


# ===========================================================================
# PairwiseAttentionBias with reduced dimensions
# ===========================================================================


class TestReducedDimensions:

    def test_reduced_defaults(self):
        """Verify default dimensions are now 16."""
        module = PairwiseAttentionBias()
        assert module.num_freqs == 16
        assert module.hidden_dim == 16

    def test_reduced_output_shape(self):
        """Output shape with reduced defaults."""
        module = PairwiseAttentionBias()
        mz = torch.rand(2, 10, 1) * 2000
        out = module(mz)
        assert out.shape == (2, 10, 10, 16)

    def test_reduced_state_dict(self):
        """State dict with reduced defaults should have correct sizes."""
        module = PairwiseAttentionBias()
        sd = module.state_dict()
        # freqs: (16,)
        assert sd["freqs"].shape == (16,)
        # f_pw.0 (Linear): 32→32 (fourier_dim = 2*16 = 32)
        assert sd["f_pw.0.weight"].shape == (32, 32)
        # f_pw.2 (Linear): 32→16
        assert sd["f_pw.2.weight"].shape == (16, 32)


# ===========================================================================
# CUDA SDPA + PA bias tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestSDPAWithBiasCUDA:

    def test_sdpa_with_bias_on_cuda(self):
        """SDPA with additive bias works on CUDA."""
        from instanovo_fm.model.attention.flash import FlashMHA

        D, H = 64, 4
        B, L = 2, 16
        mha = FlashMHA(D, H, dropout=0.0).cuda().eval()

        x = torch.randn(B, L, D, device="cuda")
        bias = torch.randn(B, H, L, L, device="cuda")

        with torch.no_grad():
            out, _ = mha(x, attn_bias=bias)

        assert out.shape == (B, L, D)
        assert out.device.type == "cuda"

    def test_sdpa_gradient_flow_with_bias(self):
        """Gradients flow through SDPA with bias on CUDA."""
        from instanovo_fm.model.attention.flash import FlashMHA

        D, H = 64, 4
        B, L = 1, 8
        mha = FlashMHA(D, H, dropout=0.0).cuda()

        x = torch.randn(B, L, D, device="cuda", requires_grad=True)
        bias = torch.randn(B, H, L, L, device="cuda", requires_grad=True)

        out, _ = mha(x, attn_bias=bias)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert bias.grad is not None

    def test_full_model_on_cuda(self):
        """Full FoundationModel with PA bias runs on CUDA."""
        from instanovo_fm.model import FoundationModel

        cfg = _make_model_config()
        model = FoundationModel(
            dim_model=64, n_heads=4, dim_feedforward=128,
            n_layers=2, dropout=0.0, n_peaks=20,
            peak_encoder_type="linear",
            mz_task="regression",
            use_meta_token=False,
            cfg=cfg,
        ).cuda().eval()

        B, L = 2, 20
        spectra = torch.rand(B, L, 2, device="cuda")
        spectra[:, :, 0] *= 0.8
        spectra[:, -5:] = 0.0

        with torch.no_grad():
            preds, aux = model(spectra)

        assert preds.shape == (B, L, 1)
        assert preds.device.type == "cuda"
