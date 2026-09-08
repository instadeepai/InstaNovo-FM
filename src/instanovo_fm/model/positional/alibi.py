"""
Attention with Linear Biases (ALiBi) implementation.

This module provides the ALiBi positional encoding method which adds learnable
positional biases to attention scores.
"""

import math
import torch
import torch.nn as nn


class ALiBi(nn.Module):
    """Attention with Linear Biases (ALiBi) implementation."""

    def __init__(self, num_heads: int, max_seq_len: int = 2048):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Learnable slopes for each head - use default dtype to match model
        slopes = torch.tensor(
            self._get_slopes(num_heads),
            dtype=torch.get_default_dtype()
        )
        self.register_buffer('slopes', slopes)

        # Pre-compute bias matrix — symmetric absolute distance, negated for decay
        # -|i - j| ensures symmetric attention decay for bidirectional encoder
        bias = -(torch.arange(max_seq_len).unsqueeze(1) - torch.arange(max_seq_len).unsqueeze(0)).abs()
        bias = bias.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        self.register_buffer('bias', bias.to(slopes.dtype), persistent=False)

    def _get_slopes(self, num_heads: int) -> list:
        """Get the slopes for ALiBi."""
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(num_heads).is_integer():
            return get_slopes_power_of_2(num_heads)
        else:
            closest_power_of_2 = 2**math.floor(math.log2(num_heads))
            return get_slopes_power_of_2(closest_power_of_2) + self._get_slopes(2*closest_power_of_2)[0::2][:num_heads-closest_power_of_2]

    def get_bias(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get ALiBi bias tensor for the given sequence length.

        Args:
            seq_len: Sequence length
            device: Target device
            dtype: Target dtype

        Returns:
            ALiBi bias tensor of shape (1, H, L, L)
        """
        if seq_len > self.max_seq_len:
            # Extend bias matrix if needed
            bias = -(torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1) - torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0)).abs()
            bias = bias.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
        else:
            bias = self.bias[:, :, :seq_len, :seq_len].to(device=device, dtype=dtype)

        # Add ALiBi bias: slopes * bias
        # slopes: (H,) -> (1, H, 1, 1)
        # bias: (1, 1, L, L)
        # result: (1, H, L, L)
        alibi_bias = self.slopes.to(device=device, dtype=dtype).unsqueeze(-1).unsqueeze(-1) * bias

        return alibi_bias

    def forward(self, attention_scores: torch.Tensor) -> torch.Tensor:
        """Add ALiBi bias to attention scores.

        Args:
            attention_scores: Tensor of shape (B, H, L, L) where B=batch, H=heads, L=sequence_length

        Returns:
            Tensor of shape (B, H, L, L) with ALiBi bias added
        """
        seq_len = attention_scores.size(-1)
        alibi_bias = self.get_bias(seq_len, attention_scores.device, attention_scores.dtype)

        return attention_scores + alibi_bias
