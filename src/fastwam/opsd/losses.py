"""The OPSD distillation objective.

Both experts predict a velocity `v = d y / d sigma` under this repo's inverted
flow convention (`sigma = 0` is data, `sigma = 1` is noise,
`y_sigma = (1 - sigma) y + sigma eps`, `target = eps - y`).

Why weighted L2 on velocity is the right objective, and not a KL over
distributions: there are no distributions here to take a KL of -- both models
emit a single vector per point. But the probability-flow ODE has a score
    s = -((1 - sigma) v + y_sigma) / sigma
so a velocity gap maps to a score gap by a pure scale factor:
    ||s_T - s_S||^2 = ((1 - sigma) / sigma)^2 ||v_T - v_S||^2
and by Girsanov the path-KL between the two diffusions is the sigma-integral of
that score gap. So weighted velocity L2 IS the KL, up to the weighting -- and
`F.kl_div` on these tensors would be meaningless.

`weighting="kl"` applies `((1 - sigma) / sigma)^2` and gives the true path-KL.
`weighting="uniform"` drops it, which is the usual practical choice because the
KL weight up-weights the low-sigma end where the velocity field is already easy.
On the shipped schedule sigma never goes below 0.357, so the KL weight stays
bounded (max 3.24) and either choice is numerically safe.
"""

from __future__ import annotations

from typing import Optional

import torch

__all__ = ["action_mse_loss", "energy_score_loss", "flow_distillation_loss", "implied_action", "score_scale"]

_VALID_WEIGHTINGS = ("uniform", "kl")


def score_scale(sigma: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """`((1 - sigma) / sigma)^2`, the velocity->score conversion factor."""
    sigma = sigma.to(dtype=torch.float32).clamp_min(eps)
    return ((1.0 - sigma) / sigma) ** 2


def flow_distillation_loss(
    v_student: torch.Tensor,
    v_teacher: torch.Tensor,
    sigma: torch.Tensor,
    point_weights: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
    weighting: str = "uniform",
) -> torch.Tensor:
    """Velocity-field distillation loss at points on the student's trajectory.

    Args:
        v_student: `[K, B, T, D]` student velocity, requires grad.
        v_teacher: `[K, B, T, D]` teacher velocity, treated as a constant target.
        sigma: `[K]` sigma of each evaluation point.
        point_weights: `[K]` importance weights over points; uniform if None.
        valid_mask: `[B, T]` True where the action step is padding (same
            convention as `action_is_pad`); those steps are excluded.
        weighting: `"uniform"` or `"kl"`.

    Returns:
        Scalar loss.
    """
    if weighting not in _VALID_WEIGHTINGS:
        raise ValueError(f"`weighting` must be one of {_VALID_WEIGHTINGS}, got {weighting!r}")
    if v_student.shape != v_teacher.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(v_student.shape)} vs teacher {tuple(v_teacher.shape)}"
        )
    if v_student.ndim != 4:
        raise ValueError(f"expected [K, B, T, D], got {tuple(v_student.shape)}")

    k, b, t, _ = v_student.shape
    if sigma.numel() != k:
        raise ValueError(f"`sigma` must have {k} entries, got {sigma.numel()}")

    # The teacher is a fixed target. Detaching here rather than relying on the
    # caller means a stray grad-enabled teacher pass cannot silently train the
    # teacher through the distillation loss.
    diff = v_student.float() - v_teacher.float().detach()
    per_step = diff.pow(2).mean(dim=3)  # [K, B, T]

    if valid_mask is not None:
        if valid_mask.shape != (b, t):
            raise ValueError(
                f"`valid_mask` must be [B, T] = {(b, t)}, got {tuple(valid_mask.shape)}"
            )
        keep = (~valid_mask).to(device=per_step.device, dtype=per_step.dtype)
        denom = keep.sum(dim=1).clamp(min=1.0)  # [B]
        per_sample = (per_step * keep.unsqueeze(0)).sum(dim=2) / denom.unsqueeze(0)
    else:
        per_sample = per_step.mean(dim=2)  # [K, B]

    per_point = per_sample.mean(dim=1)  # [K]

    if weighting == "kl":
        per_point = per_point * score_scale(sigma).to(per_point.device)

    if point_weights is None:
        return per_point.mean()

    if point_weights.numel() != k:
        raise ValueError(f"`point_weights` must have {k} entries, got {point_weights.numel()}")
    w = point_weights.to(device=per_point.device, dtype=per_point.dtype)
    return (per_point * w).sum()


def implied_action(state: torch.Tensor, velocity: torch.Tensor, sigma: float | torch.Tensor) -> torch.Tensor:
    """The clean action a point on the trajectory is heading to.

    Under this repo's convention `y_sigma = (1 - sigma) y + sigma eps` and
    `v = eps - y`, so `y = y_sigma - sigma v` exactly. One velocity evaluation
    therefore yields a differentiable sample of the model's action distribution,
    without backpropagating through ten sequential ODE steps.
    """
    if torch.is_tensor(sigma):
        sigma = sigma.to(device=state.device, dtype=state.dtype)
    return state - sigma * velocity


def energy_score_loss(
    student_actions: torch.Tensor,
    teacher_action: torch.Tensor,
    point_weights: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Energy score between the student's action samples and ONE teacher sample.

        ES = (1/M) sum_i ||a_i - a_T||  -  (1/2M^2) sum_ij ||a_i - a_j||

    A strictly proper scoring rule: its expected minimum over targets drawn from
    p(a | obs) is attained only when the student's distribution IS p(a | obs).
    Two properties make it the right substitute for the token-level divergence
    that on-policy self-distillation uses on discrete vocabularies.

    First, it needs only ONE target sample per observation, which is all a
    demonstration frame offers -- each frame carries a single realized value of
    whatever the teacher conditions on.

    Second, and the reason this exists at all: squared error against that single
    target has its minimum at the conditional MEAN. Where the privileged signal
    is not inferable from the student's inputs, the teacher's behaviour is
    multi-valued given what the student sees, and that mean can be an action
    correct under no branch of it -- the average of "go left" and "go right".
    The repulsion term is what stops the collapse; without it this reduces to
    chasing the mean again. See `scripts/toy_energy_distillation.py`, where the
    squared-error student scores 0.000 on a bimodal task and this one 0.917.

    Args:
        student_actions: `[M, K, B, T, D]`, M samples per point, requires grad.
        teacher_action: `[K, B, T, D]`, one target per point, treated as constant.
        point_weights: `[K]` importance weights over points; uniform if None.
        valid_mask: `[B, T]` True where the action step is padding.

    Returns:
        Scalar loss.
    """
    if student_actions.ndim != 5:
        raise ValueError(f"expected student [M, K, B, T, D], got {tuple(student_actions.shape)}")
    if teacher_action.ndim != 4:
        raise ValueError(f"expected teacher [K, B, T, D], got {tuple(teacher_action.shape)}")
    if student_actions.shape[1:] != teacher_action.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(student_actions.shape)} vs teacher "
            f"{tuple(teacher_action.shape)}"
        )
    m, k, b, t, _ = student_actions.shape
    if m < 2:
        raise ValueError(f"the energy score needs at least 2 student samples, got {m}")

    student = student_actions.float()
    teacher = teacher_action.float().detach()

    if valid_mask is not None:
        if valid_mask.shape != (b, t):
            raise ValueError(f"`valid_mask` must be [B, T] = {(b, t)}, got {tuple(valid_mask.shape)}")
        keep = (~valid_mask).to(device=student.device, dtype=student.dtype)[None, None, :, :, None]
        student = student * keep
        teacher = teacher * keep[0]

    # Norms over the whole action chunk: the score lives in chunk space, so a
    # per-step average would measure something else.
    def chunk_norm(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(start_dim=-2).norm(dim=-1)

    accuracy = chunk_norm(student - teacher[None]).mean(dim=0)  # [K, B]
    pairwise = chunk_norm(student[:, None] - student[None, :])  # [M, M, K, B]
    diversity = pairwise.mean(dim=(0, 1))  # [K, B]
    per_point = (accuracy - 0.5 * diversity).mean(dim=1)  # [K]

    if point_weights is None:
        return per_point.mean()
    if point_weights.numel() != k:
        raise ValueError(f"`point_weights` must have {k} entries, got {point_weights.numel()}")
    w = point_weights.to(device=per_point.device, dtype=per_point.dtype)
    return (per_point * w).sum()


def action_mse_loss(
    student_actions: torch.Tensor,
    teacher_action: torch.Tensor,
    point_weights: Optional[torch.Tensor] = None,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Squared error between the student's implied actions and ONE teacher sample.

    The action-space counterpart of velocity matching, and the ablation of the energy
    score's objective: same student samples, same teacher sample, same sigma points,
    but a squared error instead of a proper scoring rule. Its minimizer over targets
    drawn from the teacher is the teacher's conditional MEAN action, exactly like
    velocity matching -- only the space differs. Averaging over the M student samples
    lowers the variance and does not change the minimizer.

    Args:
        student_actions: `[M, K, B, T, D]` implied actions, requires grad.
        teacher_action: `[K, B, T, D]`, one target per point, treated as constant.
        point_weights: `[K]` importance weights over points; uniform if None.
        valid_mask: `[B, T]` True where the action step is padding.

    Returns:
        Scalar loss: per-element mean squared error, as in velocity matching.
    """
    if student_actions.ndim != 5:
        raise ValueError(f"expected student [M, K, B, T, D], got {tuple(student_actions.shape)}")
    if student_actions.shape[1:] != teacher_action.shape:
        raise ValueError(
            f"shape mismatch: student {tuple(student_actions.shape)} vs teacher {tuple(teacher_action.shape)}"
        )
    _, k, b, t, _ = student_actions.shape
    err = (student_actions.float() - teacher_action.float().detach()[None]) ** 2  # [M, K, B, T, D]
    err = err.mean(dim=-1)  # [M, K, B, T]
    if valid_mask is not None:
        if valid_mask.shape != (b, t):
            raise ValueError(f"`valid_mask` must be [B, T] = {(b, t)}, got {tuple(valid_mask.shape)}")
        keep = (~valid_mask).to(device=err.device, dtype=err.dtype)[None, None]
        per_point = (err * keep).sum(dim=(0, 2, 3)) / (keep.sum() * err.shape[0]).clamp(min=1.0)
    else:
        per_point = err.mean(dim=(0, 2, 3))  # [K]
    if point_weights is None:
        return per_point.mean()
    if point_weights.numel() != k:
        raise ValueError(f"`point_weights` must have {k} entries, got {point_weights.numel()}")
    w = point_weights.to(device=per_point.device, dtype=per_point.dtype)
    return (per_point * w).sum()
