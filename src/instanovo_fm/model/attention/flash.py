import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FlashMHA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, rotary_emb: nn.Module | None = None):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.h = num_heads
        self.d = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.o = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.rotary_emb = rotary_emb
        self._inv_sqrt_d = 1.0 / math.sqrt(self.d)

        # Add attributes that nn.TransformerEncoder expects
        self.batch_first = True
        self._qkv_same_embed_dim = True
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        attn_bias=None,                  # Additive attention bias (PA, ALiBi, etc.)
        key_padding_mask=None,
        is_causal: bool = False,
        return_attn_weights: bool = False,
        **_
    ) -> tuple[torch.Tensor, None | torch.Tensor]:
        B, L, _ = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.d).transpose(1, 3)  # B,H,L,d
        q, k, v = qkv.unbind(dim=2)

        # Apply RoPE to q/k if provided
        if self.rotary_emb is not None and hasattr(self.rotary_emb, "rotary_pct") and self.rotary_emb.rotary_pct > 0:
            q, k = self.rotary_emb.apply_rotary_pos_emb(q, k)

        dropout_p = self.dropout if self.training else 0.0

        # Manual path: only when attention weights are explicitly requested
        # (e.g. for visualization). This materializes the full attention matrix.
        if return_attn_weights:
            dots = torch.matmul(q, k.transpose(-2, -1))  # (B, H, L, L)
            dots = dots * self._inv_sqrt_d
            if attn_bias is not None:
                dots = dots + attn_bias
            if attn_mask is not None:
                if attn_mask.dtype == torch.bool:
                    dots.masked_fill_(attn_mask, float('-inf'))
                else:
                    dots = dots + attn_mask
            if key_padding_mask is not None:
                dots.masked_fill_(key_padding_mask[:, None, None, :], float('-inf'))
            if is_causal:
                causal_mask = torch.triu(torch.ones(L, L, device=dots.device, dtype=torch.bool), diagonal=1)
                dots.masked_fill_(causal_mask, float('-inf'))
            att = torch.softmax(dots, -1)
            att = torch.dropout(att, dropout_p, self.training)
            y = torch.matmul(att, v).transpose(1, 2).reshape(B, L, -1)
            return self.o(y), att

        # ── SDPA path ──
        # On NVIDIA hardware the kernel PyTorch selects here is closed source
        # (cuDNN or cuBLAS), so this call is the only place proprietary code enters
        # the forward pass. Nothing in this file is a vendored CUDA kernel and this
        # project does not depend on the flash-attn package -- "flash" names the
        # backend PyTorch may pick, not a bundled implementation. See
        # THIRD_PARTY_NOTICES.md.
        #
        # Let PyTorch choose the best backend:
        #   - No bias → flash kernel (fastest)
        #   - With bias → memory-efficient kernel via attn_mask
        #
        # Note: torch.compile is disabled when PA bias is active (see
        # train.py), so the Triton codegen bug with float attn_mask does
        # not apply. SDPA's memory-efficient kernel works reliably in
        # eager mode with gradient checkpointing.
        #
        # Post-scale convention (matches AF2, AF3, ALiBi, T5, SDPA):
        #   softmax(QK^T / sqrt(d) + bias)
        # SDPA applies scaling internally, so we pass bias directly as attn_mask.
        sdpa_mask = None
        if attn_bias is not None:
            sdpa_mask = attn_bias

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                float_mask = torch.zeros_like(q[:, :, :1, :1].expand(-1, -1, L, L), dtype=q.dtype)
                float_mask.masked_fill_(attn_mask, float('-inf'))
                attn_mask = float_mask
            sdpa_mask = sdpa_mask + attn_mask if sdpa_mask is not None else attn_mask

        if key_padding_mask is not None:
            pad_bias = torch.zeros(B, 1, 1, L, device=q.device, dtype=q.dtype)
            pad_bias.masked_fill_(key_padding_mask[:, None, None, :], float('-inf'))
            sdpa_mask = sdpa_mask + pad_bias if sdpa_mask is not None else pad_bias

        if sdpa_mask is not None:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=sdpa_mask,
                is_causal=False,
                dropout_p=dropout_p,
            )
        else:
            # No bias — pure flash kernel, fastest path
            y = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=is_causal,
                dropout_p=dropout_p,
            )
        y = y.transpose(1, 2).reshape(B, L, -1)
        return self.o(y), None
