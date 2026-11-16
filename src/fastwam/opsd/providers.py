"""Sources of the privileged token block `phi` that the teacher reads.

A provider is any module mapping a batch to `phi: [B, N, phi_dim]`. The teacher
is otherwise identical to the student, so everything the teacher knows and the
student does not flows through this one tensor.

Two are shipped:

`PrivilegedFeatureProvider` -- the design target. Wraps `PrivilegedEncoder` over
the 28-D physical feature vector `c` (contact, friction-cone margin, slip, force,
object motion). Needs `c` in the batch, which means running RoboTwin collection
with the privileged hook enabled first.

`FutureLatentProvider` -- trainable immediately, because the ground-truth future
video latents are ALREADY in every training sample: at training time the model is
given the future it must imagine at inference. This is the GATE 1a oracle. It
answers "can a privileged teacher beat the student at all?" without collecting
anything. If it cannot, no amount of contact features will help and the whole
approach is dead -- so run this one first.

Note the two differ in the transfer sense from PROPOSED_METHOD.md: future latents
are NEVER inferable from student inputs, so distilling them risks teaching a
shortcut. That is fine for a diagnostic and wrong for the final method. `c` is
chosen precisely because it is RETROSPECTIVELY inferable.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..privileged.encoder import PrivilegedEncoder
from ..privileged.schema import C_V0_DIM

__all__ = ["PrivilegedFeatureProvider", "FutureLatentProvider", "build_phi_provider"]


def _lookup(key: str, sample, inputs):
    """Fetch `key` from the processed inputs first, then the raw batch.

    Some phi sources live in the raw dataloader batch (`privileged_c`) and some
    only exist after `FastWAM.build_inputs` has run the VAE (`input_latents`).
    Checking both means a provider never forces the caller to guess which dict
    to hand it, and `build_inputs` is still only ever called once per step.
    """
    if inputs is not None and key in inputs and inputs[key] is not None:
        return inputs[key]
    if sample is not None and key in sample:
        return sample[key]
    return None


class PrivilegedFeatureProvider(nn.Module):
    """`c: [B, K, 28]` -> `phi: [B, num_tokens, out_dim]`."""

    batch_key = "privileged_c"

    def __init__(
        self,
        c_dim: int = C_V0_DIM,
        out_dim: int = 512,
        num_tokens: int = 8,
        num_heads: int = 8,
    ):
        super().__init__()
        self.encoder = PrivilegedEncoder(
            c_dim=c_dim, out_dim=out_dim, num_tokens=num_tokens, num_heads=num_heads
        )
        self.out_dim = int(out_dim)
        self.num_tokens = int(num_tokens)

    def forward(self, sample: dict, inputs: dict | None = None) -> torch.Tensor:
        c = _lookup(self.batch_key, sample, inputs)
        if c is None:
            raise KeyError(
                f"`{self.batch_key}` missing from the batch. Collect privileged features "
                "first (see fastwam.privileged.robotwin_hook), or use "
                "`phi_provider=future_latent` for the GATE 1a oracle teacher."
            )
        # Training batches arrive on the GPU via `accelerator.prepare`, but the
        # evaluation scripts build a plain DataLoader, and `build_inputs` only moves
        # the keys it knows (video, action, ...). Place c with the encoder either way.
        param = next(self.encoder.parameters())
        return self.encoder(c.to(device=param.device, dtype=param.dtype))


class FutureLatentProvider(nn.Module):
    """Ground-truth future video latents -> `phi`.

    Latents arrive as `[B, C, T, H, W]`. Spatial mean-pooling per frame keeps the
    temporal axis -- which is what carries "what happens next" -- and discards
    spatial layout the action expert can already see in frame 0. Learned queries
    then pool `T` frames to a fixed `num_tokens`, so `phi` shape is independent of
    horizon.
    """

    batch_key = "input_latents"

    def __init__(
        self,
        latent_channels: int,
        out_dim: int = 512,
        num_tokens: int = 8,
        num_heads: int = 8,
        skip_first_frame: bool = True,
    ):
        super().__init__()
        if out_dim % num_heads != 0:
            raise ValueError(f"`out_dim` ({out_dim}) must be divisible by `num_heads` ({num_heads})")
        self.out_dim = int(out_dim)
        self.num_tokens = int(num_tokens)
        self.num_heads = int(num_heads)
        self.head_dim = out_dim // num_heads
        # Frame 0 is the current observation, which the student already has. Keeping
        # it would let the teacher "win" on information the student is not missing.
        self.skip_first_frame = bool(skip_first_frame)

        self.proj = nn.Linear(int(latent_channels), out_dim)
        self.norm_seq = nn.LayerNorm(out_dim)
        self.queries = nn.Parameter(torch.randn(num_tokens, out_dim) * 0.02)
        self.to_q = nn.Linear(out_dim, out_dim, bias=False)
        self.to_k = nn.Linear(out_dim, out_dim, bias=False)
        self.to_v = nn.Linear(out_dim, out_dim, bias=False)
        self.to_out = nn.Linear(out_dim, out_dim, bias=False)
        self.norm_out = nn.LayerNorm(out_dim)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        return x.view(b, s, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, sample: dict, inputs: dict | None = None) -> torch.Tensor:
        latents = _lookup(self.batch_key, sample, inputs)
        if latents is None:
            raise KeyError(
                f"`{self.batch_key}` not found. It is produced by "
                "`FastWAM.build_inputs`, not present in the raw dataloader batch, "
                "so the engine must pass `inputs=`."
            )
        if latents.ndim != 5:
            raise ValueError(f"expected latents [B, C, T, H, W], got {tuple(latents.shape)}")

        x = latents
        if self.skip_first_frame and x.shape[2] > 1:
            x = x[:, :, 1:]

        seq = x.mean(dim=(3, 4)).transpose(1, 2)  # [B, T, C]
        seq = self.norm_seq(self.proj(seq.to(dtype=self.proj.weight.dtype)))

        batch_size = seq.shape[0]
        q = self._split_heads(self.to_q(self.queries.unsqueeze(0).expand(batch_size, -1, -1)))
        k = self._split_heads(self.to_k(seq))
        v = self._split_heads(self.to_v(seq))
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(batch_size, self.num_tokens, self.out_dim)
        return self.norm_out(self.to_out(attn))


def build_phi_provider(
    kind: str,
    out_dim: int = 512,
    num_tokens: int = 8,
    num_heads: int = 8,
    latent_channels: Optional[int] = None,
    c_dim: int = C_V0_DIM,
) -> nn.Module:
    """Factory. `kind` is `"privileged_c"` or `"future_latent"`."""
    if kind == "privileged_c":
        return PrivilegedFeatureProvider(
            c_dim=c_dim, out_dim=out_dim, num_tokens=num_tokens, num_heads=num_heads
        )
    if kind == "future_latent":
        if latent_channels is None:
            raise ValueError("`latent_channels` is required for kind='future_latent'")
        return FutureLatentProvider(
            latent_channels=latent_channels,
            out_dim=out_dim,
            num_tokens=num_tokens,
            num_heads=num_heads,
        )
    raise ValueError(f"unknown phi provider {kind!r}, expected 'privileged_c' or 'future_latent'")
