"""Gated cross-attention adapter for injecting privileged tokens `phi`.

Teacher-side only. Lets a frozen expert read `phi` without touching the shared
MoT softmax, and without changing its output at initialization.

Why not append `phi` as a third token group in MoT's shared attention: that
attention concatenates every expert's q/k/v and runs ONE softmax
(`mot.py:519-521`). Adding keys grows the softmax denominator, so the output over
the ORIGINAL tokens is scaled down -- even with `V_phi = 0`:

    out_i  ->  out_i * (old_denom / new_denom)  <  out_i

`alpha = 0` would then fail to reproduce the base model and the preservation
guarantee would be fiction. This module runs in parallel to that softmax instead.

    x ──►[ frozen block ]──► x' ──┐
                                  ├──► x' + tanh(alpha) * XAttn(x', phi)
    phi ─────────────────────────►┘

`tanh(0) = 0`, so at init the residual add contributes exactly zero and the
output is bit-identical to the frozen model (see `tests/test_phi_adapter.py`).

Note on the gradient at init: because the gate multiplies the whole branch, the
inner projections receive zero gradient while `alpha == 0`; only `alpha` itself
moves at first. That is the intended Flamingo-style behaviour -- `alpha` lifts
off zero immediately and the branch then trains normally. Do NOT additionally
zero-init `up`, which would leave the branch permanently dead.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GatedCrossAttnAdapter"]


class GatedCrossAttnAdapter(nn.Module):
    """Cross-attention from expert tokens (queries) to `phi` (keys/values).

    Args:
        dim: Expert hidden dim `D` (3072 video / 1024 action).
        phi_dim: Width of `phi` tokens.
        bottleneck: Internal attention width, kept small to bound parameters.
        num_heads: Attention heads over `bottleneck`.
    """

    def __init__(
        self,
        dim: int,
        phi_dim: int = 512,
        bottleneck: int = 512,
        num_heads: int = 8,
        alpha_init: float = 0.0,
        zero_init_up: bool = False,
    ):
        super().__init__()
        if dim < 1 or phi_dim < 1 or bottleneck < 1:
            raise ValueError(
                f"`dim`/`phi_dim`/`bottleneck` must be >= 1, got {dim}/{phi_dim}/{bottleneck}"
            )
        if bottleneck % num_heads != 0:
            raise ValueError(
                f"`bottleneck` ({bottleneck}) must be divisible by `num_heads` ({num_heads})"
            )

        self.dim = int(dim)
        self.phi_dim = int(phi_dim)
        self.bottleneck = int(bottleneck)
        self.num_heads = int(num_heads)
        self.head_dim = bottleneck // num_heads

        self.norm = nn.LayerNorm(dim)
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.to_k = nn.Linear(phi_dim, bottleneck, bias=False)
        self.to_v = nn.Linear(phi_dim, bottleneck, bias=False)
        self.to_out = nn.Linear(bottleneck, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, dim, bias=False)

        # Two ways to start the branch at exactly zero, with different dynamics:
        #
        # alpha_init=0 (Flamingo): tanh(0)=0 kills the residual, but it ALSO kills
        #   the gradient to the inner projections, since d L/d W_inner ~ tanh(alpha).
        #   Only alpha moves at first, pushed by a `delta` built from random weights
        #   -- so the branch can fail to bootstrap and sit near zero indefinitely.
        #
        # zero_init_up=True (ControlNet / AdaLN-zero): `up` starts at zero so
        #   delta = 0 and the output is preserved, but d L/d W_up ~ inner != 0, so
        #   the branch learns from step 1. Pair it with alpha_init > 0.
        #
        # Do NOT set alpha_init=0 AND zero_init_up=True: then nothing has gradient
        # and the branch is permanently dead.
        if alpha_init == 0.0 and zero_init_up:
            raise ValueError(
                "alpha_init=0 with zero_init_up=True leaves the adapter permanently "
                "dead: tanh(alpha)=0 zeroes the gradient to `up`, and delta=0 zeroes "
                "the gradient to `alpha`. Choose one zero, not both."
            )
        self.alpha = nn.Parameter(torch.full((1,), float(alpha_init)))
        if zero_init_up:
            nn.init.zeros_(self.up.weight)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def gate(self) -> torch.Tensor:
        """Current gate value `tanh(alpha)`. Exactly 0 at init."""
        return torch.tanh(self.alpha)

    def forward(self, x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """`x`: [B, S, dim], `phi`: [B, N, phi_dim] -> [B, S, dim]."""
        if x.ndim != 3:
            raise ValueError(f"`x` must be [B, S, D], got shape {tuple(x.shape)}")
        if x.shape[2] != self.dim:
            raise ValueError(f"`x` last dim must be {self.dim}, got {x.shape[2]}")
        if phi.ndim != 3:
            raise ValueError(f"`phi` must be [B, N, phi_dim], got shape {tuple(phi.shape)}")
        if phi.shape[2] != self.phi_dim:
            raise ValueError(f"`phi` last dim must be {self.phi_dim}, got {phi.shape[2]}")
        if phi.shape[0] != x.shape[0]:
            raise ValueError(
                f"batch mismatch: `x` has {x.shape[0]}, `phi` has {phi.shape[0]}"
            )

        gate = self.gate()
        phi = phi.to(dtype=x.dtype)

        q = self._split_heads(self.down(self.norm(x)))
        k = self._split_heads(self.to_k(phi))
        v = self._split_heads(self.to_v(phi))

        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(x.shape[0], x.shape[1], self.bottleneck)
        delta = self.up(self.to_out(attn))
        return x + gate * delta
