"""Pure-numpy computation of the privileged feature vector `c` v0.

This module has **no simulator dependency**. It consumes plain dataclass records
(`ContactPoint` / `ContactSet` / `PrivilegedStep`) and produces the 28-D vector
laid out in `schema.py`. The simulator-facing code that fills those records lives
in `probe.py`.

The split exists so that the feature definition — the part that is easy to get
subtly wrong, and that must stay stable across a whole data-generation run — is
unit-testable without SAPIEN, MuJoCo, or a GPU.

Frame conventions
-----------------
All record inputs are **world frame**. Direction-valued outputs (`slip_vel_*`,
`force_dir_*`) are rotated into the **end-effector frame** so they are invariant
to where the arm happens to be pointing. Magnitude-valued outputs are
frame-independent and left in world units.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .schema import (
    ARM_DIM,
    ARM_SLICES,
    ARM_TAGS,
    C_V0_DIM,
    CONE_MARGIN_CLIP,
    CONTACT_FORCE_EPS,
    DEFAULT_FRICTION_COEFF,
    GLOBAL_DIM,
    GLOBAL_SLICE,
)

__all__ = [
    "ContactPoint",
    "ContactSet",
    "PrivilegedStep",
    "compute_arm_block",
    "compute_global_block",
    "compute_c_v0",
    "stack_window",
]


def _as_vec3(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (3,):
        raise ValueError(f"`{name}` must be a 3-vector, got shape {np.shape(value)}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"`{name}` contains non-finite values: {arr}")
    return arr


def _unit(vec: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError(f"`{name}` must have non-zero norm, got {vec}")
    return vec / norm


@dataclass
class ContactPoint:
    """One solver contact point involving the gripper. All fields world-frame.

    `impulse` is the solver impulse accumulated over one physics step; force is
    recovered as `impulse / dt`. Storing impulse rather than force keeps the
    record faithful to what the solver actually reports.

    `rel_vel` is the velocity of the *other* body relative to the gripper at this
    contact point. Its tangential component is the slip.
    """

    position: np.ndarray
    normal: np.ndarray
    impulse: np.ndarray
    rel_vel: np.ndarray
    separation: float = 0.0

    def __post_init__(self) -> None:
        self.position = _as_vec3(self.position, "ContactPoint.position")
        self.normal = _unit(_as_vec3(self.normal, "ContactPoint.normal"), "ContactPoint.normal")
        self.impulse = _as_vec3(self.impulse, "ContactPoint.impulse")
        self.rel_vel = _as_vec3(self.rel_vel, "ContactPoint.rel_vel")
        self.separation = float(self.separation)


@dataclass
class ContactSet:
    """Every contact involving one arm's gripper, plus the frame to express them in.

    `touching_*` come from link-level contact queries and are kept independent of
    `points` on purpose: a contact can register as touching while carrying an
    impulse below `CONTACT_FORCE_EPS`, and that distinction is informative.
    """

    ee_rot_world_to_ee: np.ndarray
    points: list[ContactPoint] = field(default_factory=list)
    touching_finger_l: bool = False
    touching_finger_r: bool = False
    touching_palm: bool = False
    friction_coeff: float = DEFAULT_FRICTION_COEFF

    def __post_init__(self) -> None:
        rot = np.asarray(self.ee_rot_world_to_ee, dtype=np.float64)
        if rot.shape != (3, 3):
            raise ValueError(
                f"`ContactSet.ee_rot_world_to_ee` must be 3x3, got shape {rot.shape}"
            )
        if not np.allclose(rot @ rot.T, np.eye(3), atol=1e-4):
            raise ValueError(
                "`ContactSet.ee_rot_world_to_ee` must be orthonormal; "
                f"max deviation {np.abs(rot @ rot.T - np.eye(3)).max():.3e}"
            )
        self.ee_rot_world_to_ee = rot
        self.friction_coeff = float(self.friction_coeff)
        if self.friction_coeff <= 0.0:
            raise ValueError(
                f"`ContactSet.friction_coeff` must be > 0, got {self.friction_coeff}"
            )


@dataclass
class PrivilegedStep:
    """One logged simulator step: both arms plus the target-object global block."""

    left: ContactSet
    right: ContactSet
    obj_lin_vel: np.ndarray
    obj_ang_vel: np.ndarray
    dt: float

    def __post_init__(self) -> None:
        self.obj_lin_vel = _as_vec3(self.obj_lin_vel, "PrivilegedStep.obj_lin_vel")
        self.obj_ang_vel = _as_vec3(self.obj_ang_vel, "PrivilegedStep.obj_ang_vel")
        self.dt = float(self.dt)
        if self.dt <= 0.0:
            raise ValueError(f"`PrivilegedStep.dt` must be > 0, got {self.dt}")


def compute_arm_block(contacts: ContactSet, dt: float) -> np.ndarray:
    """Compute the 11-D per-arm block. See `schema.ARM_FEATURE_NAMES` for order."""
    if dt <= 0.0:
        raise ValueError(f"`dt` must be > 0, got {dt}")

    out = np.zeros(ARM_DIM, dtype=np.float32)
    out[0] = float(contacts.touching_finger_l)
    out[1] = float(contacts.touching_finger_r)
    out[2] = float(contacts.touching_palm)

    if not contacts.points:
        return out

    # Per-point force decomposition. Normals differ between points, so the normal
    # / tangential split must happen per point and be aggregated afterwards --
    # decomposing a summed force against a single averaged normal is wrong once
    # the gripper touches an object from two sides.
    forces = np.stack([p.impulse for p in contacts.points], axis=0) / dt  # [P, 3]
    normals = np.stack([p.normal for p in contacts.points], axis=0)       # [P, 3]
    rel_vels = np.stack([p.rel_vel for p in contacts.points], axis=0)     # [P, 3]

    normal_scalar = np.einsum("pi,pi->p", forces, normals)                # [P]
    force_normal = normal_scalar[:, None] * normals                       # [P, 3]
    force_tangent = forces - force_normal                                 # [P, 3]

    per_point_mag = np.linalg.norm(forces, axis=1)                        # [P]
    total_mag = float(per_point_mag.sum())
    if total_mag < CONTACT_FORCE_EPS:
        # Registered as touching but carrying no measurable load: leave every
        # force-derived dimension at exactly zero rather than amplifying noise.
        return out

    weights = per_point_mag / total_mag                                   # [P]

    # Cone margin: force-weighted mean of per-point ‖F_t‖ / (μ‖F_n‖).
    # 1.0 is the slip threshold; clipped because beyond ~2 the value is solver noise.
    denom = np.maximum(contacts.friction_coeff * np.abs(normal_scalar), CONTACT_FORCE_EPS)
    per_point_margin = np.linalg.norm(force_tangent, axis=1) / denom
    out[3] = float(np.clip(np.dot(weights, per_point_margin), 0.0, CONE_MARGIN_CLIP))

    # Slip: tangential component of the relative velocity, force-weighted, in EE frame.
    rel_normal_scalar = np.einsum("pi,pi->p", rel_vels, normals)
    slip_world = rel_vels - rel_normal_scalar[:, None] * normals
    slip_mean_world = np.einsum("p,pi->i", weights, slip_world)
    out[4:7] = (contacts.ee_rot_world_to_ee @ slip_mean_world).astype(np.float32)

    # Net force: log magnitude (frame-free) + unit direction in EE frame.
    net_force_world = forces.sum(axis=0)
    net_mag = float(np.linalg.norm(net_force_world))
    out[7] = float(np.log1p(net_mag))
    if net_mag >= CONTACT_FORCE_EPS:
        dir_world = net_force_world / net_mag
        out[8:11] = (contacts.ee_rot_world_to_ee @ dir_world).astype(np.float32)

    return out


def compute_global_block(step: PrivilegedStep) -> np.ndarray:
    """Compute the 6-D global block: target-object linear and angular velocity."""
    out = np.zeros(GLOBAL_DIM, dtype=np.float32)
    out[0:3] = step.obj_lin_vel.astype(np.float32)
    out[3:6] = step.obj_ang_vel.astype(np.float32)
    return out


def compute_c_v0(step: PrivilegedStep) -> np.ndarray:
    """Compute the flattened 28-D `c` v0 vector for one simulator step.

    Layout: `[left(11), right(11), global(6)]`. Use `schema.ARM_SLICES` /
    `schema.GLOBAL_SLICE` to recover the blocks rather than hard-coding offsets.
    """
    out = np.zeros(C_V0_DIM, dtype=np.float32)
    for tag in ARM_TAGS:
        out[ARM_SLICES[tag]] = compute_arm_block(getattr(step, tag), step.dt)
    out[GLOBAL_SLICE] = compute_global_block(step)
    if not np.all(np.isfinite(out)):
        bad = [int(i) for i in np.flatnonzero(~np.isfinite(out))]
        raise ValueError(f"`c` v0 contains non-finite values at indices {bad}")
    return out


def stack_window(steps: list[PrivilegedStep]) -> np.ndarray:
    """Stack a causal window of steps into `[K, 28]`, oldest first.

    `K` is chosen at load time, not at generation time: log at the simulator's
    control rate and decimate here. See `PROPOSED_METHOD.md` §4.3.
    """
    if not steps:
        raise ValueError("`steps` must contain at least one PrivilegedStep")
    return np.stack([compute_c_v0(s) for s in steps], axis=0)
