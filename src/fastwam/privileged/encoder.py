"""`E_psi` -- the privileged-information encoder.

Maps the causal window of privileged features to the token group that the
teacher's gated adapters cross-attend to:

    c : [B, K, 28]  ──►  E_psi  ──►  phi : [B, 8, 512]

Teacher-side only. Trained in Stage B alongside the adapters, then frozen, then
discarded -- nothing here runs at deployment (`PROPOSED_METHOD.md` §3).

Two design points:

- **Causal convolutions.** The whole window already sits before the action chunk,
  so causality is not strictly required for correctness here. It is enforced
  anyway so that extending `K`, or later running this in a streaming fashion,
  cannot silently leak information forward. A non-causal conv would make that
  failure invisible.
- **Learned query pooling rather than temporal pooling.** The `N_phi` output
  tokens come from learned queries attending over the convolution output, so the
  output shape is independent of `K` and each token is free to specialize
  (contact onset vs. steady load vs. incipient slip). Average-pooling to a fixed
  length would instead smear the `dF/dt` transient at contact onset, which is
  exactly the signal that carries the jam signature.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import C_V0_DIM

__all__ = ["CausalConv1d", "PrivilegedEncoder"]


class CausalConv1d(nn.Module):
    """Conv1d over time that cannot see the future: left-padded, no right pad."""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"`kernel_size` must be >= 1, got {kernel_size}")
        if dilation < 1:
            raise ValueError(f"`dilation` must be >= 1, got {dilation}")
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """`x`: [B, C, T] -> [B, C_out, T]. Output length always matches input."""
        if x.ndim != 3:
            raise ValueError(f"`x` must be [B, C, T], got shape {tuple(x.shape)}")
        return self.conv(F.pad(x, (self.pad, 0)))


class PrivilegedEncoder(nn.Module):
    """`E_psi`: `c [B, K, c_dim]` -> `phi [B, num_tokens, out_dim]`."""

    def __init__(
        self,
        c_dim: int = C_V0_DIM,
        hidden_dims: tuple[int, ...] = (256, 512),
        out_dim: int = 512,
        num_tokens: int = 8,
        num_heads: int = 8,
        kernel_size: int = 3,
    ):
        super().__init__()
        if c_dim < 1:
            raise ValueError(f"`c_dim` must be >= 1, got {c_dim}")
        if num_tokens < 1:
            raise ValueError(f"`num_tokens` must be >= 1, got {num_tokens}")
        if out_dim % num_heads != 0:
            raise ValueError(
                f"`out_dim` ({out_dim}) must be divisible by `num_heads` ({num_heads})"
            )

        self.c_dim = int(c_dim)
        self.out_dim = int(out_dim)
        self.num_tokens = int(num_tokens)
        self.num_heads = int(num_heads)
        self.head_dim = out_dim // num_heads

        dims = (c_dim,) + tuple(hidden_dims) + (out_dim,)
        convs: list[nn.Module] = []
        for i in range(len(dims) - 1):
            convs.append(CausalConv1d(dims[i], dims[i + 1], kernel_size=kernel_size))
            convs.append(nn.GELU(approximate="tanh"))
        self.convs = nn.Sequential(*convs)
        self.norm_seq = nn.LayerNorm(out_dim)

        # Learned queries pool a variable-length window into a fixed token group.
        self.queries = nn.Parameter(torch.randn(num_tokens, out_dim) * 0.02)
        self.to_q = nn.Linear(out_dim, out_dim, bias=False)
        self.to_k = nn.Linear(out_dim, out_dim, bias=False)
        self.to_v = nn.Linear(out_dim, out_dim, bias=False)
        self.to_out = nn.Linear(out_dim, out_dim, bias=False)
        self.norm_out = nn.LayerNorm(out_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """`c`: [B, K, c_dim] -> `phi`: [B, num_tokens, out_dim]."""
        if c.ndim != 3:
            raise ValueError(f"`c` must be [B, K, c_dim], got shape {tuple(c.shape)}")
        if c.shape[2] != self.c_dim:
            raise ValueError(
                f"`c` last dim must be {self.c_dim}, got {c.shape[2]}"
            )
        if c.shape[1] < 1:
            raise ValueError("`c` must contain at least one timestep")

        batch = c.shape[0]
        seq = self.convs(c.transpose(1, 2)).transpose(1, 2)  # [B, K, out_dim]
        seq = self.norm_seq(seq)

        q = self._split_heads(self.to_q(self.queries.unsqueeze(0).expand(batch, -1, -1)))
        k = self._split_heads(self.to_k(seq))
        v = self._split_heads(self.to_v(seq))

        attn = F.scaled_dot_product_attention(q, k, v)  # [B, H, num_tokens, head_dim]
        attn = attn.transpose(1, 2).reshape(batch, self.num_tokens, self.out_dim)
        return self.norm_out(self.to_out(attn))
