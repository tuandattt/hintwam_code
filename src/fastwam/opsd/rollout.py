"""Record the student's own ODE trajectory, so the teacher is evaluated on it.

This is the "on-policy" half of OPSD, and it is on-policy in the sampler sense
rather than the DAgger sense: no simulator is involved. The student integrates
its action flow from noise to data exactly as it would at inference, and we keep
the state it actually visits at every schedule point. The teacher then scores
those states.

Why this matters: training the student on states drawn from the FORWARD process
(`add_noise` on ground-truth actions) matches a distribution the student never
sees at inference. Its own trajectory drifts, and the drift compounds over steps.
Evaluating on the student's own states is what removes that exposure bias.

The trajectory is recorded under `no_grad`. Gradients flow only through the
re-evaluation at the selected points -- backpropagating through 10 sequential ODE
steps would be both enormous and unnecessary, since the target is the velocity
field at a point, not the endpoint.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

__all__ = ["StudentTrajectory", "record_student_trajectory"]


class StudentTrajectory:
    """States visited by the student's sampler.

    Attributes:
        states: list of `[B, T, D]`, the state AT each schedule sigma, i.e.
            before that step's update is applied.
        timesteps: `[N]` schedule timesteps (`sigma * num_train_timesteps`).
        deltas: `[N]` sigma increments, negative under this convention.
        final: `[B, T, D]` state after the last step (the sampled action).
    """

    def __init__(self, states, timesteps, deltas, final):
        self.states = states
        self.timesteps = timesteps
        self.deltas = deltas
        self.final = final

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        """Stack the selected states into `[K, B, T, D]`."""
        return torch.stack([self.states[int(i)] for i in indices], dim=0)


@torch.no_grad()
def record_student_trajectory(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    scheduler,
    initial_noise: torch.Tensor,
    num_inference_steps: int,
    shift_override: Optional[float] = None,
) -> StudentTrajectory:
    """Integrate the student's action ODE, keeping every visited state.

    Args:
        velocity_fn: `(latents_action, timestep) -> velocity`, same signature the
            sampler uses. Called `num_inference_steps` times.
        scheduler: a `WanContinuousFlowMatchScheduler`.
        initial_noise: `[B, T, D]` starting point, sigma = 1.
        num_inference_steps: schedule length; use the deployed value so the
            recorded states match what inference actually visits.
        shift_override: optional sigma-schedule shift override.

    Returns:
        A `StudentTrajectory`.
    """
    if initial_noise.ndim != 3:
        raise ValueError(f"`initial_noise` must be [B, T, D], got {tuple(initial_noise.shape)}")

    timesteps, deltas = scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps,
        device=initial_noise.device,
        dtype=initial_noise.dtype,
        shift_override=shift_override,
    )

    states = []
    x = initial_noise
    for step_t, step_delta in zip(timesteps, deltas):
        # Record BEFORE stepping: this is the state at this sigma, which is the
        # point the teacher must be evaluated at.
        states.append(x)
        timestep = step_t.reshape(1).expand(x.shape[0]).to(dtype=x.dtype, device=x.device)
        velocity = velocity_fn(x, timestep)
        x = scheduler.step(velocity, step_delta, x)

    return StudentTrajectory(states=states, timesteps=timesteps, deltas=deltas, final=x)
