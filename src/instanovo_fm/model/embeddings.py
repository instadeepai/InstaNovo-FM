from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor
from math import ceil

from instanovo.__init__ import console
from instanovo.types import Spectrum, SpectrumEmbedding, SpectrumMask
from instanovo.utils.colorlogging import ColorLog

class MultiScalePeakEmbedding(nn.Module):
    """Multi-scale sinusoidal embedding based on Voronov et. al."""

    def __init__(self, h_size: int, dropout: float = 0) -> None:
        super().__init__()
        self.h_size = h_size

        self.mlp = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_size, h_size),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.Linear(h_size + 1, h_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_size, h_size),
            nn.Dropout(dropout),
        )

        # Learnable frequencies that adapt to input scale during training.
        # Initial periods span 2.5–25 Da (logspace -3 to -2), providing good
        # amino acid discrimination for normalized m/z input.
        freqs = 2 * np.pi / torch.logspace(-3, -2, int(h_size / 2), dtype=torch.float32)
        self.register_parameter("freqs", nn.Parameter(freqs, requires_grad=True))

    # @torch.autocast("cuda", dtype=torch.float32)
    def forward(self, spectra: Float[Spectrum, " batch"]) -> Float[SpectrumEmbedding, " batch"]:
        """Encode peaks."""
        mz_values, intensities = spectra[:, :, [0]], spectra[:, :, [1]]
        x = self.encode_mass(mz_values)
        x = self.mlp(x)
        x = torch.cat([x, intensities], dim=2)
        return self.head(x)

    def encode_mass(self, x: Float[Tensor, " batch"]) -> Float[Tensor, "batch embedding"]:
        """Encode mz."""
        x = self.freqs[None, None, :] * x
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=2)
        return x


class FourierFeatures(nn.Module):
    def __init__(self, strategy, x_min, x_max, trainable=True, funcs='both', sigma=10, num_freqs=512):

        assert strategy in {'random', 'voronov_et_al', 'lin_float_int'}
        assert funcs in {'both', 'sin', 'cos'}
        assert 0 < x_min < x_max

        super().__init__()
        self.funcs = funcs
        self.strategy = strategy
        self.trainable = trainable
        self.num_freqs = num_freqs

        if strategy == 'random':
            # Store on CPU initially - will be moved to correct device during first forward
            self.b = torch.randn(num_freqs, dtype=torch.float32) * sigma
        elif self.strategy == 'voronov_et_al':
            self.b = torch.tensor(
                [1 / (x_min * (x_max / x_min) ** (2 * i / (num_freqs - 1))) for i in range(num_freqs)],
                dtype=torch.float32
            )
        elif self.strategy == 'lin_float_int':
            self.b = torch.tensor(
                [1 / (x_min * i) for i in range(2, ceil(1 / x_min), 2)] +
                [1 / (1 * i) for i in range(2, ceil(x_max), 1)],
                dtype=torch.float32
            )
        self.b = self.b.unsqueeze(0)

        self.b = nn.Parameter(self.b, requires_grad=self.trainable)
        self.register_parameter('fourier_frequencies', self.b)
        
        # Track if frequencies have been frozen
        self._frequencies_frozen = False

    def forward(self, x):
        # Force float32 for the matmul to prevent autocast downcast to float16/bfloat16
        # which causes NaN (float16 overflow) or high RMSE (bfloat16 precision loss).
        x = x.float()
        with torch.amp.autocast('cuda', enabled=False):
            x = 2 * torch.pi * x @ self.b.float()
        if self.funcs == 'both':
            x = torch.cat((torch.cos(x), torch.sin(x)), dim=-1)
        elif self.funcs == 'cos':
            x = torch.cos(x)
        elif self.funcs == 'sin':
            x = torch.sin(x)
        return x
    
    def freeze_frequencies(self):
        """Freeze the Fourier frequencies to prevent further training."""
        if not self._frequencies_frozen:
            self.b.requires_grad_(False)
            self._frequencies_frozen = True

    def num_features(self):
        return self.b.shape[1] if self.funcs != 'both' else 2 * self.b.shape[1]


class FourierPeakEmbedding(nn.Module):
    """
    DreaMS-style peak encoder:
        Fourier(m/z)  →  MLP  →  concat(norm-I)  →  MLP
    Output dim == h_size, just like MultiScalePeakEmbedding.
    """

    def __init__(self, h_size: int, dropout: float = 0,
                 num_freqs: int | None = None, strategy: str = "voronov_et_al",
                 trainable_fourier: bool = True,
                 x_min: float = 0.001, x_max: float = 1.5):
        super().__init__()
        if num_freqs is None:
            num_freqs = h_size // 4          # keeps Fourier+I <= h_size
        self.ff_mz  = FourierFeatures(strategy, x_min, x_max,
                                      trainable=trainable_fourier, num_freqs=num_freqs)
        d_ff = self.ff_mz.num_features()              # ~ num_freqs*2
        self.mlp1 = nn.Sequential(
            nn.Linear(d_ff, h_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_size//2, h_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # +1 for intensity
        self.mlp2 = nn.Sequential(
            nn.Linear(h_size//2 + 1, h_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_size, h_size),
            nn.Dropout(dropout),
        )

    def forward(self, spectra):          # spectra: (B,L,2) [m/z, I]
        mz, intens = spectra[..., 0:1], spectra[..., 1:2]
        x = self.ff_mz(mz)
        x = self.mlp1(x)
        x = torch.cat([x, intens], dim=-1)
        return self.mlp2(x)


class LinearPeakEmbedding(nn.Module):
    """Minimal baseline: linear projection of raw (m/z, intensity) to token embedding.

    No frequency expansion, no RBF, no structured encoding — just a two-layer MLP
    on the 2D input. Serves as a control to test whether structured m/z encodings
    provide any benefit over a learned projection.
    """

    def __init__(self, h_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, h_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_size, h_size),
            nn.Dropout(dropout),
        )

    def forward(
        self, spectra: Float[Spectrum, " batch"]
    ) -> Float[SpectrumEmbedding, " batch"]:
        return self.mlp(spectra)


class RBFPeakEmbedding(nn.Module):
    """Radial Basis Function peak embedding — soft binning of m/z values.

    Places ``num_rbf`` Gaussian kernels uniformly between ``min_mz`` and
    ``max_mz``.  Each m/z value is mapped to its RBF activations, concatenated
    with intensity, and projected through an MLP to produce a token embedding.
    """

    def __init__(
        self,
        h_size: int,
        dropout: float = 0.0,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        num_rbf: int = 1024,
        normalize_mz: bool = True,
    ) -> None:
        super().__init__()
        self.h_size = h_size
        self.num_rbf = num_rbf

        # Place centers in the actual input domain
        lo = min_mz / max_mz if normalize_mz else min_mz  # 0.02 when normalized
        hi = 1.0 if normalize_mz else max_mz               # 1.0 when normalized
        centers = torch.linspace(lo, hi, num_rbf)
        self.register_buffer("centers", centers)
        self.sigma = max((hi - lo) / num_rbf, 1e-6)

        self.mlp = nn.Sequential(
            nn.Linear(num_rbf + 1, h_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h_size, h_size),
            nn.Dropout(dropout),
        )

    def forward(
        self, spectra: Float[Spectrum, " batch"]
    ) -> Float[SpectrumEmbedding, " batch"]:
        """Encode peaks via RBF expansion + intensity."""
        mz = spectra[..., 0:1]          # (B, L, 1)
        intensities = spectra[..., 1:2]  # (B, L, 1)
        rbf = self._rbf_expand(mz)       # (B, L, num_rbf)
        x = torch.cat([rbf, intensities], dim=-1)  # (B, L, num_rbf+1)
        return self.mlp(x)

    def encode_mass(
        self, x: Float[Tensor, " batch"]
    ) -> Float[Tensor, "batch embedding"]:
        """RBF expansion for precursor mass (API parity with MultiScale)."""
        return self._rbf_expand(x)

    def _rbf_expand(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Gaussian RBF activations: exp(-(x - c)^2 / (2*sigma^2))."""
        return torch.exp(-((x - self.centers) ** 2) / (2 * self.sigma**2))


class DualPeakEmbedding(nn.Module):
    """Dual peak embedding: sinusoidal + RBF with learnable ReZero gate.

    Combines MultiScalePeakEmbedding (learnable sinusoidal frequency encoding)
    with RBFPeakEmbedding (Gaussian soft binning).  The sinusoidal branch
    captures relative mass differences well, while RBF captures absolute mass
    position.  A zero-initialised gate (ReZero; Bachlechner et al. 2020) blends
    them, so the model starts from the sinusoidal baseline and learns the
    optimal mixing ratio during training.
    """

    def __init__(
        self,
        h_size: int,
        dropout: float = 0.0,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        num_rbf: int = 2048,
        normalize_mz: bool = True,
    ) -> None:
        super().__init__()
        self.sin_encoder = MultiScalePeakEmbedding(h_size, dropout=dropout)
        self.rbf_encoder = RBFPeakEmbedding(
            h_size,
            dropout=dropout,
            min_mz=min_mz,
            max_mz=max_mz,
            num_rbf=num_rbf,
            normalize_mz=normalize_mz,
        )
        self.rbf_gate = nn.Parameter(torch.zeros(1))  # ReZero

    def forward(
        self, spectra: Float[Spectrum, " batch"]
    ) -> Float[SpectrumEmbedding, " batch"]:
        """Encode peaks with dual sinusoidal + RBF embedding."""
        x_sin = self.sin_encoder(spectra)
        x_rbf = self.rbf_encoder(spectra)
        return x_sin + torch.sigmoid(self.rbf_gate) * x_rbf


# ───────────────────────────────────────────────────────────────────
#  META-TOKEN EMBED (import from new clean implementation)
# ───────────────────────────────────────────────────────────────────

# Import the new clean MetaTokenEmbed from meta_token module
from instanovo_fm.model.meta_token import MetaTokenEmbed

__all__ = [
    "MultiScalePeakEmbedding",
    "FourierFeatures",
    "FourierPeakEmbedding",
    "LinearPeakEmbedding",
    "RBFPeakEmbedding",
    "DualPeakEmbedding",
    "MetaTokenEmbed",
]
