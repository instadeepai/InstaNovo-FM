"""End-to-end smoke test: the ported model builds from the shipped config and runs.

Deliberately small (2 layers) so it is cheap enough for CI, but it instantiates
from configs/model/foundation_base.yaml rather than from hand-written kwargs, so
a config/code mismatch introduced by the port would fail here.
"""

from __future__ import annotations

import pathlib

import pytest

torch = pytest.importorskip("torch")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf

CONFIG = (
    pathlib.Path(__file__).resolve().parents[1]
    / "src"
    / "instanovo_fm"
    / "configs"
    / "model"
    / "foundation_base.yaml"
)


@pytest.fixture(scope="module")
def model():
    from instanovo_fm.model.encoder import FoundationModel

    cfg = OmegaConf.load(CONFIG)
    cfg.n_layers = 2  # keep CI quick; depth is not what this test covers
    return FoundationModel(
        dim_model=cfg.dim_model,
        n_heads=cfg.n_heads,
        dim_feedforward=cfg.dim_feedforward,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        n_peaks=cfg.n_peaks,
        cfg=cfg,
    )


def test_config_is_shipped() -> None:
    assert CONFIG.is_file(), f"missing {CONFIG}"
    cfg = OmegaConf.load(CONFIG)
    # the modular architecture block is mandatory -- the encoder factory raises
    # without it, which is how a truncated config port would show up
    assert "architecture" in cfg, "foundation_base.yaml lost its 'architecture' section"
    for key in ("positional_encoding", "relative_bias", "attention"):
        assert key in cfg.architecture, f"architecture.{key} missing"


def test_instantiates(model) -> None:
    n = sum(p.numel() for p in model.parameters())
    assert n > 1_000_000, f"suspiciously small model: {n} parameters"


def test_forward_pass_shapes(model) -> None:
    batch, peaks = 2, 32
    mz = torch.rand(batch, peaks) * 2000 + 50
    intensity = torch.rand(batch, peaks)
    spectra = torch.stack([mz, intensity], dim=-1)
    spectra_mask = torch.zeros(batch, peaks, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        out = model(spectra, spectra_mask=spectra_mask)

    assert out is not None
    tensors = (
        list(out.values())
        if isinstance(out, dict)
        else list(out)
        if isinstance(out, tuple)
        else [out]
    )
    tensors = [t for t in tensors if torch.is_tensor(t)]
    assert tensors, f"no tensor in the output: {type(out)}"
    assert all(t.shape[0] == batch for t in tensors), "batch dimension not preserved"
    assert all(torch.isfinite(t).all() for t in tensors), "non-finite values in the output"
