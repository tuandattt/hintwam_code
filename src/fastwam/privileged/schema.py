"""Canonical layout for the privileged feature vector `c`.

`c` v0 is 28-D: two 11-D per-arm blocks followed by a 6-D global block.
See `PROPOSED_METHOD.md` §4.1.

Two deliberate properties, both of which remove an otherwise-irreversible
generation-time decision (`PROPOSED_METHOD.md` §4.3):

1. Features are stored per-arm under named blocks, never as a pre-flattened
   vector. Whether the encoder concatenates both arms (28-D) or runs per-arm
   with shared weights (11-D + 6-D) therefore stays a model-side choice and
   does not require regenerating data.
2. Every dimension is tagged with how it should be normalized, so dims that
   are bounded by construction (binaries, unit vectors, clipped ratios) are
   left alone instead of being rescaled by dataset statistics.
"""

from __future__ import annotations

from typing import Literal

# --- per-arm block -----------------------------------------------------------
# Order is load-bearing: `features.compute_arm_block` writes into these slots.
ARM_FEATURE_NAMES: tuple[str, ...] = (
    "contact_finger_l",  # 0  binary
    "contact_finger_r",  # 1  binary
    "contact_palm",      # 2  binary
    "cone_margin",       # 3  ‖F_t‖ / (μ‖F_n‖), clipped to [0, CONE_MARGIN_CLIP]
    "slip_vel_x",        # 4  tangential relative velocity at contact, EE frame
    "slip_vel_y",        # 5
    "slip_vel_z",        # 6
    "force_log_mag",     # 7  log1p(‖F‖)
    "force_dir_x",       # 8  unit vector, EE frame (zero when no contact)
    "force_dir_y",       # 9
    "force_dir_z",       # 10
)

# --- global block ------------------------------------------------------------
GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "obj_lin_vel_x",  # 0  target object linear velocity, world frame
    "obj_lin_vel_y",  # 1
    "obj_lin_vel_z",  # 2
    "obj_ang_vel_x",  # 3  target object angular velocity, world frame
    "obj_ang_vel_y",  # 4
    "obj_ang_vel_z",  # 5
)

ARM_DIM: int = len(ARM_FEATURE_NAMES)          # 11
GLOBAL_DIM: int = len(GLOBAL_FEATURE_NAMES)    # 6
ARM_TAGS: tuple[str, ...] = ("left", "right")
C_V0_DIM: int = ARM_DIM * len(ARM_TAGS) + GLOBAL_DIM  # 28

# Slices into the flattened 28-D vector.
ARM_SLICES: dict[str, slice] = {
    tag: slice(i * ARM_DIM, (i + 1) * ARM_DIM) for i, tag in enumerate(ARM_TAGS)
}
GLOBAL_SLICE: slice = slice(ARM_DIM * len(ARM_TAGS), C_V0_DIM)

NormalizeMode = Literal["none", "std"]

# "none": bounded by construction — normalizing would only inject dataset noise.
# "std":  unbounded physical quantity — needs per-dimension dataset statistics.
ARM_NORMALIZE: tuple[NormalizeMode, ...] = (
    "none",  # contact_finger_l
    "none",  # contact_finger_r
    "none",  # contact_palm
    "none",  # cone_margin   (clipped)
    "std",   # slip_vel_x
    "std",   # slip_vel_y
    "std",   # slip_vel_z
    "std",   # force_log_mag
    "none",  # force_dir_x   (unit)
    "none",  # force_dir_y
    "none",  # force_dir_z
)
GLOBAL_NORMALIZE: tuple[NormalizeMode, ...] = ("std",) * GLOBAL_DIM

# --- feature groups, for the GATE 4 ablation screen --------------------------
# Maps a group name to its indices *within a per-arm block*. `PROPOSED_METHOD.md`
# §10.2 ablates `c` by group; keeping the grouping here means the ablation is a
# config change rather than a code change.
ARM_FEATURE_GROUPS: dict[str, tuple[int, ...]] = {
    "contact": (0, 1, 2),
    "cone_margin": (3,),
    "slip": (4, 5, 6),
    "force": (7, 8, 9, 10),
}
GLOBAL_FEATURE_GROUPS: dict[str, tuple[int, ...]] = {
    "obj_velocity": tuple(range(GLOBAL_DIM)),
}

# --- constants used by the feature computation -------------------------------
CONE_MARGIN_CLIP: float = 2.0
"""Cone margin is clipped here. 1.0 is the slip threshold; values above ~2 carry
no extra information and would otherwise be dominated by solver noise."""

DEFAULT_FRICTION_COEFF: float = 0.5
"""Fallback μ when the simulator does not expose the contact material. Always
log the μ actually used — see `ContactSet.friction_coeff`."""

CONTACT_FORCE_EPS: float = 1e-6
"""Below this force magnitude a contact is treated as absent, so that direction
and cone margin stay exactly zero rather than amplifying numerical noise."""

CONTACT_SEPARATION_TOL: float = 1e-3
"""Metres. Contact points further apart than this are not touching. PhysX reports
speculative contacts anywhere inside the contact offset (2 cm on RoboTwin), and on
the first lift_pot pilot the median recorded separation was 5-8 mm: without this
gate a pot 1.5 cm from the gripper read as held. True contacts sat at ~-0.2 mm."""


def flat_feature_names() -> tuple[str, ...]:
    """Names for the flattened 28-D vector, in index order."""
    names: list[str] = []
    for tag in ARM_TAGS:
        names.extend(f"{tag}_{n}" for n in ARM_FEATURE_NAMES)
    names.extend(GLOBAL_FEATURE_NAMES)
    return tuple(names)


def flat_normalize_modes() -> tuple[NormalizeMode, ...]:
    """Normalization mode per dimension of the flattened 28-D vector."""
    modes: list[NormalizeMode] = []
    for _ in ARM_TAGS:
        modes.extend(ARM_NORMALIZE)
    modes.extend(GLOBAL_NORMALIZE)
    return tuple(modes)


def flat_group_indices() -> dict[str, tuple[int, ...]]:
    """Group name -> indices into the flattened 28-D vector (both arms pooled)."""
    groups: dict[str, tuple[int, ...]] = {}
    for name, local in ARM_FEATURE_GROUPS.items():
        idx: list[int] = []
        for tag in ARM_TAGS:
            base = ARM_SLICES[tag].start
            idx.extend(base + i for i in local)
        groups[name] = tuple(idx)
    for name, local in GLOBAL_FEATURE_GROUPS.items():
        base = GLOBAL_SLICE.start
        groups[name] = tuple(base + i for i in local)
    return groups


def _validate_layout() -> None:
    if len(ARM_NORMALIZE) != ARM_DIM:
        raise ValueError(
            f"`ARM_NORMALIZE` must have {ARM_DIM} entries, got {len(ARM_NORMALIZE)}"
        )
    if len(GLOBAL_NORMALIZE) != GLOBAL_DIM:
        raise ValueError(
            f"`GLOBAL_NORMALIZE` must have {GLOBAL_DIM} entries, got {len(GLOBAL_NORMALIZE)}"
        )
    covered = sorted(i for idx in ARM_FEATURE_GROUPS.values() for i in idx)
    if covered != list(range(ARM_DIM)):
        raise ValueError(
            "`ARM_FEATURE_GROUPS` must partition the per-arm block exactly once, "
            f"got coverage {covered}"
        )
    covered_g = sorted(i for idx in GLOBAL_FEATURE_GROUPS.values() for i in idx)
    if covered_g != list(range(GLOBAL_DIM)):
        raise ValueError(
            "`GLOBAL_FEATURE_GROUPS` must partition the global block exactly once, "
            f"got coverage {covered_g}"
        )


_validate_layout()
