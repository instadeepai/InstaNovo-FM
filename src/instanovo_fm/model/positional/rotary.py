import torch
import torch.nn as nn


class SimpleRotaryEmbedding(nn.Module):
    """Optimized RoPE implementation with configurable rotary percentage.

    Cos/sin embeddings are precomputed in __init__ and registered as buffers.
    This avoids mutating module state during forward(), which is required for
    compatibility with torch.compile().
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0, rotary_pct: float = 1.0) -> None:
        """Initialise the input."""
        super().__init__()
        self.dim = dim
        self.base = base
        self.rotary_pct = rotary_pct
        self.rotary_dim = int(dim * rotary_pct)
        if self.rotary_dim % 2 != 0:
            self.rotary_dim -= 1
        if self.rotary_dim > 0:
            inv_freq = 1.0 / (base ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            # Precompute cos/sin for max_seq_len as registered buffers.
            # This ensures no module state mutation during forward(), which is
            # required for torch.compile() with CUDA graphs.
            t = torch.arange(max_seq_len, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self.register_buffer("_cos_cached", emb.cos(), persistent=False)
            self.register_buffer("_sin_cached", emb.sin(), persistent=False)
        else:
            self.register_buffer("inv_freq", torch.empty(0), persistent=False)
            self.register_buffer("_cos_cached", torch.empty(0), persistent=False)
            self.register_buffer("_sin_cached", torch.empty(0), persistent=False)

    def clear_cache(self) -> None:
        """Recompute cos/sin embeddings.

        This should be called before gradient-based XAI methods if needed.
        """
        if self.rotary_dim > 0:
            max_seq_len = self._cos_cached.size(0)  # type: ignore[has-type]
            t = torch.arange(max_seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos().to(self._cos_cached.dtype)  # type: ignore[has-type]
            self._sin_cached = emb.sin().to(self._sin_cached.dtype)  # type: ignore[has-type]

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary pos emb."""
        if self.rotary_dim == 0:
            return q, k
        seq_len = q.size(-2)
        q_rot = q[..., : self.rotary_dim]
        q_pass = q[..., self.rotary_dim :]
        k_rot = k[..., : self.rotary_dim]
        k_pass = k[..., self.rotary_dim :]
        cos = self._cos_cached[:seq_len].to(q.dtype).unsqueeze(0).unsqueeze(0)
        sin = self._sin_cached[:seq_len].to(q.dtype).unsqueeze(0).unsqueeze(0)
        q_rot_embed = (q_rot * cos) + (self.rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (self.rotate_half(k_rot) * sin)
        q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)
        return q_embed, k_embed

    def apply_rotary_pos_emb_with_positions(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RoPE using explicit (possibly non-contiguous) position indices.

        Used for self-attention between mask queries in the MAE decoder, where
        queries occupy scattered positions in the original sequence.

        Args:
            q: (B, H, M, d_head)
            k: (B, H, M, d_head)
            positions: (B, M) integer indices into the precomputed cos/sin buffers

        Returns:
            Tuple of rotated (q, k), same shape as input.
        """
        if self.rotary_dim == 0:
            return q, k

        q_rot = q[..., : self.rotary_dim]
        q_pass = q[..., self.rotary_dim :]
        k_rot = k[..., : self.rotary_dim]
        k_pass = k[..., self.rotary_dim :]

        # Advanced indexing: _cos_cached is (max_seq_len, rotary_dim)
        # positions is (B, M) → cos/sin are (B, M, rotary_dim)
        cos = self._cos_cached[positions].to(q.dtype).unsqueeze(1)  # (B, 1, M, rotary_dim)
        sin = self._sin_cached[positions].to(q.dtype).unsqueeze(1)  # (B, 1, M, rotary_dim)

        q_rot_embed = (q_rot * cos) + (self.rotate_half(q_rot) * sin)
        k_rot_embed = (k_rot * cos) + (self.rotate_half(k_rot) * sin)
        q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
        k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)
        return q_embed, k_embed
