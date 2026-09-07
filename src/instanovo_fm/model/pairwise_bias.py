import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class PairwiseAttentionBias(nn.Module):
    """Pairwise Attention (PA) bias from signed Δm/z differences.

    Computes Fourier-encoded pairwise m/z differences and projects them
    through a small MLP (f_pw).  The per-layer projection g^(ℓ) lives in
    each encoder layer, not here.

    Reference: Lapin et al. (PMC12235698) — Pairwise Attention for
    proteomic MS2 spectra.
    """

    def __init__(
        self,
        num_freqs: int = 16,
        hidden_dim: int = 16,
        lambda_min: float = 0.001,
        lambda_max: float = 10000.0,
    ):
        super().__init__()
        self.num_freqs = num_freqs
        self.hidden_dim = hidden_dim

        # Log-spaced angular frequencies: 2π/λ_max … 2π/λ_min
        freqs = torch.exp(
            torch.linspace(
                math.log(2 * math.pi / lambda_max),
                math.log(2 * math.pi / lambda_min),
                num_freqs,
            )
        )
        self.register_buffer("freqs", freqs)  # (num_freqs,)

        # f_pw MLP: r'_pw → r'_pw → r_pw  (Eq. 10 in Lapin et al.)
        # Hidden dimension equals the Fourier input dim (2*num_freqs = r'_pw),
        # not the output dim. This matches the paper's parameter count:
        # (r'_pw + r_pw)(2*r_pw + 1) = (128 + 64)(129) ≈ 24.8k params.
        fourier_dim = 2 * num_freqs  # r'_pw
        self.f_pw = nn.Sequential(
            nn.Linear(fourier_dim, fourier_dim),
            nn.SiLU(),
            nn.Linear(fourier_dim, hidden_dim),
        )

    def _fourier_encode(self, dmz: torch.Tensor) -> torch.Tensor:
        """Fourier encode signed Δm/z.

        Uses torch.cat on sin/cos. For large num_freqs (64) with B=512,
        L=200 this creates ~5 GB intermediates per op. The caller
        (forward) processes batches in chunks to bound peak memory.

        Args:
            dmz: (B, L, L, 1) signed mass differences.

        Returns:
            (B, L, L, 2*num_freqs) sin/cos features.
        """
        # dmz * freqs → (B, L, L, num_freqs)
        x = dmz * self.freqs
        return torch.cat([x.sin(), x.cos()], dim=-1)

    # Maximum batch-chunk size for the L² Fourier + MLP computation.
    # With num_freqs=64, each sample produces (L, L, 128) bf16 ≈ 10 MB
    # for L=200. Chunks of 64 keep peak memory at ~640 MB for intermediates.
    _CHUNK_SIZE = 64

    @torch.compiler.disable
    def forward(self, mz: torch.Tensor) -> torch.Tensor:
        """Compute pairwise attention features.

        Excluded from torch.compile: the L² f_pw MLP triggers a Triton
        autotuner illegal-memory-access on certain GPUs (A100, B=512, L=200).

        For large num_freqs (64) the Fourier intermediates can OOM on H100
        at B=512. We process in batch-chunks to bound peak memory while
        keeping full-batch gradient accumulation correct.

        Args:
            mz: Peak m/z values, shape (B, L, 1).

        Returns:
            pairwise_feats: (B, L, L, hidden_dim).
        """
        B = mz.shape[0]
        # Wrap each chunk in gradient checkpointing during training.
        # Without it, all 8 chunks' MLP activations accumulate in GPU memory
        # for backward (~1.83 GB × 8 = 14.6 GB). With checkpointing, only one
        # chunk's activations are live at a time during the backward pass.
        run_chunk = (
            (lambda mz_c: checkpoint(self._forward_chunk, mz_c, use_reentrant=False))
            if self.training
            else self._forward_chunk
        )
        if B <= self._CHUNK_SIZE:
            return run_chunk(mz)

        chunks = []
        for i in range(0, B, self._CHUNK_SIZE):
            chunks.append(run_chunk(mz[i : i + self._CHUNK_SIZE]))
        return torch.cat(chunks, dim=0)

    def _forward_chunk(self, mz: torch.Tensor) -> torch.Tensor:
        """Process a single batch-chunk through Fourier encode + f_pw."""
        dmz = mz.unsqueeze(2) - mz.unsqueeze(1)   # (C, L, L, 1)
        fourier = self._fourier_encode(dmz)         # (C, L, L, 2*num_freqs)
        del dmz
        return self.f_pw(fourier)                    # (C, L, L, hidden_dim)
