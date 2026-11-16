"""Which sigma points to distill at, and how to weight them.

The student integrates its action ODE over the schedule built by
`WanContinuousFlowMatchScheduler.build_inference_schedule`. Discretization error
at step `i` scales with `delta_i^2`, so that is the natural importance weight
over evaluation points.

For the shipped setting (`shift=5`, 10 steps) that weighting is extremely
concentrated:

    sigma   1.0000 0.9783 0.9524 0.9211 0.8824 0.8333 0.7692 0.6818 0.5556 0.3571
    weight  0.0024 0.0033 0.0049 0.0075 0.0120 0.0205 0.0381 0.0795 0.1962 0.6357

The last point alone carries 63.6% and the last three carry 91.1%. Effective
sample size of the full weighting is 2.22 -- so Monte-Carlo sampling a sigma per
batch would spend most of its draws on points that barely matter and add variance
for nothing. Enumerating the top-3 is both cheaper and lower variance.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["SigmaSelection", "select_sigma_points", "effective_sample_size"]


class SigmaSelection:
    """Chosen evaluation points on the student's ODE trajectory.

    Attributes:
        indices: `[K]` long tensor, indices into the inference schedule.
        sigmas: `[K]` sigma value at each chosen point (data at 0, noise at 1).
        weights: `[K]` normalized importance weights, summing to 1.
        covered_mass: fraction of the full `delta^2` mass these K points carry.
    """

    def __init__(
        self,
        indices: torch.Tensor,
        sigmas: torch.Tensor,
        weights: torch.Tensor,
        covered_mass: float,
    ):
        self.indices = indices
        self.sigmas = sigmas
        self.weights = weights
        self.covered_mass = float(covered_mass)

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        pairs = ", ".join(
            f"sigma={s:.4f}:w={w:.4f}"
            for s, w in zip(self.sigmas.tolist(), self.weights.tolist())
        )
        return f"SigmaSelection(K={len(self)}, mass={self.covered_mass:.4f}, {pairs})"


def effective_sample_size(weights: torch.Tensor) -> float:
    """`1 / sum(w^2)` for normalized `w`. 1.0 = all mass on one point."""
    w = weights.to(dtype=torch.float64)
    total = w.sum()
    if float(total) <= 0.0:
        raise ValueError("`weights` must have positive sum")
    w = w / total
    return float(1.0 / (w * w).sum())


def select_sigma_points(
    deltas: torch.Tensor,
    timesteps: torch.Tensor,
    num_points: int,
    num_train_timesteps: int = 1000,
    renormalize: bool = True,
) -> SigmaSelection:
    """Pick the `num_points` highest-`delta^2` points from an inference schedule.

    Args:
        deltas: `[N]` per-step sigma increments (negative; noise -> data).
        timesteps: `[N]` matching timesteps, i.e. `sigma * num_train_timesteps`.
        num_points: how many points to enumerate. `num_points >= N` keeps all.
        num_train_timesteps: divisor turning timesteps back into sigma.
        renormalize: if True the returned weights sum to 1 over the CHOSEN points.
            If False they stay on the original scale, so dropping points lowers
            the total loss magnitude rather than reweighting the survivors.

    Returns:
        A `SigmaSelection`, ordered by descending weight.
    """
    if deltas.ndim != 1:
        raise ValueError(f"`deltas` must be 1-D, got shape {tuple(deltas.shape)}")
    if timesteps.shape != deltas.shape:
        raise ValueError(
            f"`timesteps` {tuple(timesteps.shape)} must match `deltas` {tuple(deltas.shape)}"
        )
    if num_points < 1:
        raise ValueError(f"`num_points` must be >= 1, got {num_points}")
    if num_train_timesteps <= 0:
        raise ValueError(f"`num_train_timesteps` must be positive, got {num_train_timesteps}")

    full = (deltas.to(dtype=torch.float64) ** 2)
    total = full.sum()
    if float(total) <= 0.0:
        raise ValueError("`deltas` are all zero; the schedule has no extent")

    k = min(int(num_points), int(deltas.numel()))
    top = torch.topk(full, k)
    indices = top.indices
    covered = float(top.values.sum() / total)

    weights = top.values / (top.values.sum() if renormalize else total)
    sigmas = timesteps.to(dtype=torch.float64)[indices] / float(num_train_timesteps)

    return SigmaSelection(
        indices=indices.to(dtype=torch.long),
        sigmas=sigmas.to(dtype=torch.float32),
        weights=weights.to(dtype=torch.float32),
        covered_mass=covered,
    )
