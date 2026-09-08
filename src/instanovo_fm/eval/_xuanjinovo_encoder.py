"""Vendored, self-contained copy of the XuanjiNovo spectrum encoder.

These classes (:class:`MassEncoder`, :class:`PeakEncoder`, :class:`SpectrumEncoder`) are
copied — with type annotations and docstrings added, but numerically unchanged — from the
MassNet-DDA / XuanjiNovo project so the pretrained encoder can be loaded without importing
the upstream package (which pins ``torch==2.1.0`` / ``pytorch-lightning==1.8.6`` / ``pydantic<2``
plus ``cupy`` and a C++ ``ctcdecode`` extension, all of which conflict with this environment).

Only the spectrum-encoder path is reproduced — the peptide decoder and its dependencies
(``einops``, ``PeptideMass``, ``listify``) are intentionally omitted.

Source: https://github.com/guomics-lab/MassNet-DDA
    XuanjiNovo/components/encoders.py, XuanjiNovo/components/transformers.py
Copyright 2024 PHOENIX center. Licensed under the Apache License, Version 2.0
    (http://www.apache.org/licenses/LICENSE-2.0).
"""

from __future__ import annotations

import re
import warnings
from typing import Any

import numpy as np
import torch

from instanovo.__init__ import console
from instanovo.utils.colorlogging import ColorLog

logger = ColorLog(console, __name__).logger


class MassEncoder(torch.nn.Module):
    """Encode mass / m/z values using sine and cosine waves."""

    sin_term: torch.Tensor
    cos_term: torch.Tensor

    def __init__(self, dim_model: int, min_wavelength: float = 0.001, max_wavelength: float = 10000) -> None:
        """Initialise the encoder.

        Args:
            dim_model: The number of features to output.
            min_wavelength: The minimum wavelength to use.
            max_wavelength: The maximum wavelength to use.
        """
        super().__init__()

        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin

        if min_wavelength:
            base = min_wavelength / (2 * np.pi)
            scale = max_wavelength / min_wavelength
        else:
            base = 1
            scale = max_wavelength / (2 * np.pi)

        sin_term = base * scale ** (torch.arange(0, n_sin).float() / (n_sin - 1))
        cos_term = base * scale ** (torch.arange(0, n_cos).float() / (n_cos - 1))

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode m/z values.

        Args:
            x: The masses to embed.

        Returns:
            The encoded features.
        """
        sin_mz = torch.sin(x / self.sin_term)
        cos_mz = torch.cos(x / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)


class PeakEncoder(MassEncoder):
    """Encode m/z values and intensities in a mass spectrum using sine and cosine waves."""

    def __init__(
        self,
        dim_model: int,
        dim_intensity: int | None = None,
        min_wavelength: float = 0.001,
        max_wavelength: float = 10000,
    ) -> None:
        """Initialise the encoder.

        Args:
            dim_model: The number of features to output.
            dim_intensity: The number of features to use for intensity. The remaining features
                encode the m/z values. If ``None``, intensity is encoded into all ``dim_model``
                features via a linear layer and added to the m/z encoding.
            min_wavelength: The minimum wavelength to use.
            max_wavelength: The maximum wavelength to use.
        """
        self.dim_intensity = dim_intensity
        self.dim_mz = dim_model
        if self.dim_intensity is not None:
            self.dim_mz -= self.dim_intensity

        super().__init__(
            dim_model=self.dim_mz,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

        self.int_encoder: torch.nn.Module
        if dim_intensity is None:
            self.int_encoder = torch.nn.Linear(1, dim_model, bias=False)
        else:
            self.int_encoder = MassEncoder(
                dim_model=dim_intensity,
                min_wavelength=0,
                max_wavelength=1,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode m/z values and intensities.

        Intensities are expected to fall within the interval ``[0, 1]``.

        Args:
            x: The spectra to embed, shape ``(n_spectra, n_peaks, 2)`` — m/z/intensity pairs.

        Returns:
            The encoded features, shape ``(n_spectra, n_peaks, dim_model)``.
        """
        m_over_z = x[:, :, [0]]
        encoded = super().forward(m_over_z)
        intensity = self.int_encoder(x[:, :, [1]])
        if self.dim_intensity is None:
            return encoded + intensity

        return torch.cat([encoded, intensity], dim=2)


class SpectrumEncoder(torch.nn.Module):
    """A Transformer encoder for input mass spectra."""

    def __init__(
        self,
        dim_model: int = 128,
        n_head: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0.0,
        peak_encoder: bool = True,
        dim_intensity: int | None = None,
    ) -> None:
        """Initialise the encoder.

        Args:
            dim_model: The latent dimensionality to represent peaks in the mass spectrum.
            n_head: The number of attention heads in each layer. ``dim_model`` must be
                divisible by ``n_head``.
            dim_feedforward: The dimensionality of the fully connected layers.
            n_layers: The number of Transformer layers.
            dropout: The dropout probability for all layers.
            peak_encoder: Use sinusoidal positional encodings for the m/z values of each peak.
            dim_intensity: The number of features to use for encoding peak intensity.
        """
        super().__init__()

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, dim_model))

        self.peak_encoder: torch.nn.Module
        if peak_encoder:
            self.peak_encoder = PeakEncoder(dim_model, dim_intensity=dim_intensity)
        else:
            self.peak_encoder = torch.nn.Linear(2, dim_model)

        layer = torch.nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )
        self.mass_encoder = MassEncoder(dim_model)
        self.charge_encoder = torch.nn.Embedding(10, dim_model)

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

    def forward(self, spectra: torch.Tensor, precursors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed a batch of mass spectra.

        Args:
            spectra: The spectra to embed, shape ``(n_spectra, n_peaks, 2)`` — zero-padded
                m/z/intensity pairs.
            precursors: Precursor information, shape ``(n_spectra, 3)`` —
                ``[neutral_mass, charge, m/z]``.

        Returns:
            A tuple ``(latent, mem_mask)`` where ``latent`` has shape
            ``(n_spectra, n_peaks + 1, dim_model)`` (a prepended precursor token followed by the
            peak tokens) and ``mem_mask`` is the padding mask (``True`` = padding).
        """
        masses = self.mass_encoder(precursors[:, None, [0]])  # (bz, 1, dim)
        charges = self.charge_encoder(precursors[:, 1].int() - 1)  # (bz, dim)
        precursor_tokens = masses + charges[:, None, :]  # (bz, 1, dim)

        zeros = ~spectra.sum(dim=2).bool()
        precursor_mask = torch.tensor([[False]] * spectra.shape[0]).type_as(zeros)
        mask = torch.cat([precursor_mask, zeros], dim=1)

        peaks = self.peak_encoder(spectra)
        peaks = torch.cat([precursor_tokens, peaks], dim=1)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage")
            output = self.transformer_encoder(peaks, src_key_padding_mask=mask)
        return output, mask

    @property
    def device(self) -> torch.device:
        """The current device of the model."""
        return next(self.parameters()).device


def _infer_encoder_config(encoder_sd: dict[str, torch.Tensor], hparams: dict[str, Any]) -> dict[str, Any]:
    """Infer ``SpectrumEncoder`` constructor arguments from a checkpoint's encoder weights.

    Dimensions are read from tensor shapes (ground truth); ``n_head`` is read from the
    checkpoint hyper-parameters with a fallback of 8 (it is not encoded in the weights).

    Args:
        encoder_sd: The encoder ``state_dict`` (keys with the ``encoder.`` prefix stripped).
        hparams: The checkpoint's ``hyper_parameters`` dict (may be empty).

    Returns:
        Keyword arguments for :class:`SpectrumEncoder`.

    Raises:
        ValueError: If the weights are missing expected keys or ``dim_model`` is not divisible
            by the resolved ``n_head``.
    """
    if "charge_encoder.weight" not in encoder_sd:
        raise ValueError("Encoder state_dict missing 'charge_encoder.weight'; not a XuanjiNovo SpectrumEncoder checkpoint.")
    dim_model = int(encoder_sd["charge_encoder.weight"].shape[1])

    layer_indices = {int(m.group(1)) for k in encoder_sd if (m := re.match(r"transformer_encoder\.layers\.(\d+)\.", k))}
    if not layer_indices:
        raise ValueError("Encoder state_dict has no 'transformer_encoder.layers.*' weights.")
    n_layers = max(layer_indices) + 1

    dim_feedforward = int(encoder_sd["transformer_encoder.layers.0.linear1.weight"].shape[0])

    # Peak encoder variant: sinusoidal PeakEncoder (has sin_term) vs a plain Linear(2, dim_model).
    use_peak_encoder = "peak_encoder.sin_term" in encoder_sd
    dim_intensity: int | None = None
    if use_peak_encoder:
        int_w = encoder_sd.get("peak_encoder.int_encoder.weight")
        if int_w is not None and int_w.shape[1] == 1:
            dim_intensity = None  # additive Linear(1, dim_model) intensity encoder
        else:
            sin = encoder_sd.get("peak_encoder.int_encoder.sin_term")
            cos = encoder_sd.get("peak_encoder.int_encoder.cos_term")
            if sin is not None and cos is not None:
                dim_intensity = int(sin.shape[0] + cos.shape[0])

    n_head = int(hparams.get("n_head", 8))
    if dim_model % n_head != 0:
        raise ValueError(
            f"Resolved n_head={n_head} does not divide dim_model={dim_model}. "
            "Pass an explicit n_head (the checkpoint hyper_parameters did not record a compatible value)."
        )

    return {
        "dim_model": dim_model,
        "n_head": n_head,
        "dim_feedforward": dim_feedforward,
        "n_layers": n_layers,
        "dropout": 0.0,  # inference only — dropout is inert under eval()
        "peak_encoder": use_peak_encoder,
        "dim_intensity": dim_intensity,
    }


def load_xuanjinovo_encoder(
    checkpoint_path: str,
    device: torch.device,
    n_head: int | None = None,
) -> tuple[SpectrumEncoder, dict[str, Any]]:
    """Load the XuanjiNovo ``SpectrumEncoder`` from a pretrained Lightning checkpoint.

    The full ``Spec2Pep`` model is never instantiated — only the ``encoder.*`` weights are
    extracted and loaded into the vendored :class:`SpectrumEncoder`. Constructor dimensions are
    inferred from the weight shapes and the load is ``strict`` so any mismatch fails loudly.

    Args:
        checkpoint_path: Path to the ``.ckpt`` file.
        device: Device to move the encoder to.
        n_head: Optional override for the attention head count (otherwise read from the
            checkpoint hyper-parameters, falling back to 8).

    Returns:
        A tuple ``(encoder, config)`` — the eval-mode encoder on ``device`` and the inferred config.

    Raises:
        ValueError: If no ``encoder.*`` weights are found in the checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    hparams = ckpt.get("hyper_parameters", {}) if isinstance(ckpt, dict) else {}
    print(hparams)

    prefix = "encoder."
    encoder_sd = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
    if not encoder_sd:
        raise ValueError(f"No 'encoder.*' weights found in checkpoint: {checkpoint_path}")

    config = _infer_encoder_config(encoder_sd, hparams)
    # {'dim_model': 400, 'n_head': 8, 'dim_feedforward': 1024, 'n_layers': 9, 'dropout': 0.0, 'peak_encoder': True, 'dim_intensity': None}

    if n_head is not None:
        config["n_head"] = n_head
        if config["dim_model"] % n_head != 0:
            raise ValueError(f"n_head={n_head} does not divide dim_model={config['dim_model']}.")

    logger.info(
        f"XuanjiNovo encoder config: dim_model={config['dim_model']}, n_layers={config['n_layers']}, "
        f"n_head={config['n_head']}, dim_feedforward={config['dim_feedforward']}, dim_intensity={config['dim_intensity']}"
    )

    encoder = SpectrumEncoder(**config)
    encoder.load_state_dict(encoder_sd, strict=True)
    # for name, param in encoder.named_parameters():
    #     print(name)
    encoder = encoder.to(device)
    encoder.eval()
    return encoder, config
