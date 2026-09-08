from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from instanovo.constants import MASS_SCALE
from instanovo.transformer.layers import PositionalEncoding
from instanovo.utils.residues import ResidueSet
from omegaconf import DictConfig, OmegaConf

from instanovo_fm.downstream.de_novo_sequencing.model import DownstreamDeNovo
from instanovo_fm.model.embeddings import MultiScalePeakEmbedding


def test_model_init(tiny_denovo_model: DownstreamDeNovo, denovo_residue_set: ResidueSet) -> None:
    """Verify model attributes and sub-modules are correctly initialised."""
    model = tiny_denovo_model
    assert model.vocab_size == len(denovo_residue_set)
    assert model.dim_model == 16
    assert isinstance(model.latent_token, nn.Parameter)
    assert isinstance(model.peak_encoder, MultiScalePeakEmbedding)
    assert isinstance(model.encoder, nn.TransformerEncoder)
    assert isinstance(model.aa_embed, nn.Embedding)
    assert isinstance(model.aa_pos_embed, PositionalEncoding)
    assert isinstance(model.decoder, nn.TransformerDecoder)
    assert isinstance(model.head, nn.Linear)
    assert len(model.encoder.layers) == 1
    assert len(model.decoder.layers) == 1


def test_forward_shape(tiny_denovo_model: DownstreamDeNovo, tiny_spectrum_batch: dict) -> None:
    """Check that forward output has the expected shape with add_bos."""
    logits = tiny_denovo_model.forward(
        x=tiny_spectrum_batch["spectra"],
        p=tiny_spectrum_batch["precursors"],
        y=tiny_spectrum_batch["peptides"],
        x_mask=tiny_spectrum_batch["spectra_mask"],
        y_mask=tiny_spectrum_batch["peptides_mask"],
        add_bos=True,
    )
    batch_size, seq_len = tiny_spectrum_batch["peptides"].shape
    # add_bos prepends one SOS token → output length seq_len+1.
    assert logits.shape == torch.Size([batch_size, seq_len + 1, tiny_denovo_model.vocab_size])
    assert torch.isfinite(logits).all()


def test_init_returns_memory_and_logprobs(tiny_denovo_model: DownstreamDeNovo, tiny_spectrum_batch: dict) -> None:
    """Verify init returns correctly shaped memory, mask, and log-probabilities."""
    (memory, memory_mask), logprobs = tiny_denovo_model.init(
        spectra=tiny_spectrum_batch["spectra"],
        precursors=tiny_spectrum_batch["precursors"],
        spectra_mask=tiny_spectrum_batch["spectra_mask"],
    )
    batch_size = tiny_spectrum_batch["spectra"].shape[0]
    # Latent token is prepended, so memory has n_peaks + 1 positions.
    assert memory.shape[0] == batch_size
    assert memory.shape[2] == tiny_denovo_model.dim_model
    assert memory.shape[1] == tiny_spectrum_batch["spectra"].shape[1] + 1
    assert memory_mask.shape == torch.Size([batch_size, memory.shape[1]])
    assert logprobs.shape == torch.Size([batch_size, tiny_denovo_model.vocab_size])
    # log-softmax → exp sums to 1 per row.
    assert torch.allclose(logprobs.exp().sum(-1), torch.ones(batch_size), atol=1e-5)


def test_score_candidates_shape(tiny_denovo_model: DownstreamDeNovo, tiny_spectrum_batch: dict) -> None:
    """Check score_candidates output shape matches batch and vocab size."""
    (memory, memory_mask), _ = tiny_denovo_model.init(
        spectra=tiny_spectrum_batch["spectra"],
        precursors=tiny_spectrum_batch["precursors"],
        spectra_mask=tiny_spectrum_batch["spectra_mask"],
    )
    out = tiny_denovo_model.score_candidates(
        sequences=tiny_spectrum_batch["peptides"],
        precursor_mass_charge=tiny_spectrum_batch["precursors"],
        spectra=memory,
        spectra_mask=memory_mask,
    )
    batch_size = tiny_spectrum_batch["peptides"].shape[0]
    assert out.shape == torch.Size([batch_size, tiny_denovo_model.vocab_size])
    assert torch.isfinite(out).all()


def test_get_residue_masses(tiny_denovo_model: DownstreamDeNovo) -> None:
    """Verify residue masses have correct shape and special tokens have zero mass."""
    masses = tiny_denovo_model.get_residue_masses(MASS_SCALE)
    assert masses.shape == torch.Size([tiny_denovo_model.vocab_size])
    # Special tokens (PAD, SOS, EOS) have zero mass.
    assert masses[0].item() == 0
    assert masses[1].item() == 0
    assert masses[2].item() == 0
    # Real residue slots are positive.
    assert (masses[3:] > 0).all()


def test_eos_and_pad_indices(tiny_denovo_model: DownstreamDeNovo) -> None:
    """Verify EOS and PAD token indices."""
    assert tiny_denovo_model.get_eos_index() == 2
    assert tiny_denovo_model.get_empty_index() == 0


def test_decode_reverses_sequence(tiny_denovo_model: DownstreamDeNovo) -> None:
    """Verify decode reverses the predicted sequence to reading order."""
    # Residue indices for A, B, C (after PAD=0, SOS=1, EOS=2).
    r2i = tiny_denovo_model.residue_set.residue_to_index
    seq = torch.tensor([r2i["A"], r2i["B"], r2i["C"]], dtype=torch.long)
    decoded = tiny_denovo_model.decode(seq)
    # decode() reverses during output so model's right-to-left prediction reads forward.
    assert decoded == ["C", "B", "A"]


def test_batch_idx_to_aa(tiny_denovo_model: DownstreamDeNovo) -> None:
    """Verify batch index-to-amino-acid conversion."""
    r2i = tiny_denovo_model.residue_set.residue_to_index
    idx = torch.tensor(
        [
            [r2i["A"], r2i["B"], tiny_denovo_model.residue_set.EOS_INDEX],
            [r2i["C"], r2i["D"], tiny_denovo_model.residue_set.EOS_INDEX],
        ],
        dtype=torch.long,
    )
    decoded = tiny_denovo_model.batch_idx_to_aa(idx, reverse=False)
    assert decoded == [["A", "B"], ["C", "D"]]


def test_causal_mask_upper_triangular() -> None:
    """Verify causal mask blocks upper-triangular positions."""
    mask = DownstreamDeNovo._get_causal_mask(4)
    # Boolean mask: True where we *block* attention (upper triangle above diag).
    expected = torch.tensor(
        [
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ]
    )
    assert torch.equal(mask, expected)


def test_load_builds_model_from_mocked_checkpoint(
    tiny_denovo_config: DictConfig,
    denovo_residue_set: ResidueSet,
) -> None:
    """load() should read config from the checkpoint and construct a matching model.

    We patch torch.load to avoid disk I/O and patch load_state_dict to avoid having
    to materialise a matching state dict for the default meta-token layout.
    """
    ckpt_config = OmegaConf.create(
        {
            "dim_model": tiny_denovo_config.dim_model,
            "n_head": tiny_denovo_config.n_head,
            "dim_feedforward": tiny_denovo_config.dim_feedforward,
            "n_layers": tiny_denovo_config.encoder_layers,
            "dropout": tiny_denovo_config.dropout,
            "use_flash_attention": False,
            "conv_peak_encoder": False,
            "residues": dict(denovo_residue_set.residue_masses),
            "residue_remapping": {},
        }
    )
    ckpt = {"config": ckpt_config, "state_dict": {}}
    with (
        patch("instanovo_fm.downstream.de_novo_sequencing.model.torch.load", return_value=ckpt),
        patch("instanovo_fm.downstream.de_novo_sequencing.model._whitelist_torch_omegaconf", return_value=None),
        patch.object(DownstreamDeNovo, "load_state_dict", return_value=None),
    ):
        loaded, cfg = DownstreamDeNovo.load(path="/fake/path.ckpt", update_residues_to_unimod=False)
    assert isinstance(loaded, DownstreamDeNovo)
    assert loaded.dim_model == tiny_denovo_config.dim_model
    assert cfg.dim_model == tiny_denovo_config.dim_model


def _matched_cfg(dim_model: int = 16) -> dict[str, Any]:
    """Config that matches the foundation checkpoint shape.

    Multiscale peak encoder, no PE, no relative bias, math attention backend,
    ion_ladder off — the same configuration the foundation was trained with.
    """
    return {
        "dim_model": dim_model,
        "n_head": 2,
        "n_heads": 2,
        "dim_feedforward": 32,
        "encoder_layers": 1,
        "n_layers": 1,
        "dropout": 0.0,
        "use_flash_attention": False,
        "conv_peak_encoder": False,
        "use_meta_token": False,
        "max_mz": 2500.0,
        "min_mz": 0.0,
        "normalize_mz": True,
        "peak_encoder": {"type": "multiscale", "config": {"dropout": 0.0}},
        "ion_ladder": {"enabled": False},
        "architecture": {
            "positional_encoding": {"type": "none"},
            "relative_bias": {"type": "none"},
            "attention": {"backend": "math"},
        },
    }


def _build_downstream(cfg_dict: dict[str, Any], residue_set: ResidueSet) -> DownstreamDeNovo:
    """Construct a DownstreamDeNovo from a matched-config dict."""
    return DownstreamDeNovo(
        residue_set=residue_set,
        dim_model=cfg_dict["dim_model"],
        n_head=cfg_dict["n_head"],
        dim_feedforward=cfg_dict["dim_feedforward"],
        encoder_layers=cfg_dict["encoder_layers"],
        decoder_layers=cfg_dict["encoder_layers"],
        dropout=cfg_dict["dropout"],
        use_flash_attention=cfg_dict["use_flash_attention"],
        conv_peak_encoder=cfg_dict["conv_peak_encoder"],
        peak_encoder_type=cfg_dict["peak_encoder"]["type"],
        max_mz=cfg_dict["max_mz"],
        min_mz=cfg_dict["min_mz"],
        use_meta_token=cfg_dict["use_meta_token"],
        cfg=cfg_dict,
    )


def test_load_encoder_from_foundation_transfers_encoder_weights(
    denovo_residue_set: ResidueSet,
) -> None:
    """Matched config transfers peak_encoder, latent_token and encoder weights."""
    cfg = _matched_cfg()
    torch.manual_seed(0)
    foundation = _build_downstream(cfg, denovo_residue_set).eval()
    torch.manual_seed(1)
    downstream = _build_downstream(cfg, denovo_residue_set).eval()
    assert not torch.equal(downstream.peak_encoder.mlp[0].weight, foundation.peak_encoder.mlp[0].weight)
    with patch(
        "instanovo_fm.model.encoder.FoundationModel.load",
        return_value=(foundation, cfg),
    ):
        downstream.load_encoder_from_foundation("/fake/path.ckpt")
    assert torch.equal(downstream.peak_encoder.mlp[0].weight, foundation.peak_encoder.mlp[0].weight)
    assert torch.equal(downstream.latent_token, foundation.latent_token)
    assert torch.equal(
        downstream.encoder.layers[0].self_attn.qkv.weight,
        foundation.encoder.layers[0].self_attn.qkv.weight,
    )


def test_load_encoder_from_foundation_raises_on_peak_encoder_mismatch(
    denovo_residue_set: ResidueSet,
) -> None:
    """Peak encoder type drift surfaces as a structural state-dict mismatch."""
    fm_cfg = _matched_cfg()
    foundation = _build_downstream(fm_cfg, denovo_residue_set).eval()
    ds_cfg = _matched_cfg()
    ds_cfg["peak_encoder"] = {"type": "dual", "config": {"num_rbf": 64, "dropout": 0.0}}
    downstream = _build_downstream(ds_cfg, denovo_residue_set).eval()
    with patch(
        "instanovo_fm.model.encoder.FoundationModel.load",
        return_value=(foundation, fm_cfg),
    ):
        with pytest.raises(RuntimeError, match=r"'peak_encoder\.\*' submodule but no foundation key matches"):
            downstream.load_encoder_from_foundation("/fake/path.ckpt")


def test_load_encoder_from_foundation_raises_on_ion_ladder_mismatch(
    denovo_residue_set: ResidueSet,
) -> None:
    """Downstream ion_ladder enabled with foundation lacking it raises structurally."""
    fm_cfg = _matched_cfg()
    foundation = _build_downstream(fm_cfg, denovo_residue_set).eval()
    ds_cfg = _matched_cfg()
    ds_cfg["ion_ladder"] = {
        "enabled": True,
        "window_k": 2,
        "sigma": 0.35,
        "charge_states": [1, 2],
        "residue_masses": dict(denovo_residue_set.residue_masses),
        "neutral_losses": {"enabled": False},
    }
    downstream = _build_downstream(ds_cfg, denovo_residue_set).eval()
    with patch(
        "instanovo_fm.model.encoder.FoundationModel.load",
        return_value=(foundation, fm_cfg),
    ):
        with pytest.raises(RuntimeError, match=r"'ion_ladder\.\*' submodule but no foundation key matches"):
            downstream.load_encoder_from_foundation("/fake/path.ckpt")
