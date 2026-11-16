"""Privileged features for on-policy self-distillation.

`c` is read only by the **teacher**, only during training. Nothing in this package
runs at deployment — see `PROPOSED_METHOD.md` §3.

Layers, in dependency order:

- `schema`   — the canonical 28-D layout, names, ablation groups, constants.
                Pure Python, no dependencies.
- `features` — pure-numpy computation of `c` v0 from plain records.
                Unit-tested without a simulator (`tests/test_privileged_v0.py`).
- `probe`    — the SAPIEN-facing adapter that fills those records from a live
                RoboTwin scene. Requires `sapien`; import it lazily.
"""

from .features import (
    ContactPoint,
    ContactSet,
    PrivilegedStep,
    compute_arm_block,
    compute_c_v0,
    compute_global_block,
    stack_window,
)
from .schema import (
    ARM_DIM,
    ARM_FEATURE_NAMES,
    ARM_SLICES,
    ARM_TAGS,
    C_V0_DIM,
    GLOBAL_DIM,
    GLOBAL_FEATURE_NAMES,
    GLOBAL_SLICE,
    flat_feature_names,
    flat_group_indices,
    flat_normalize_modes,
)

__all__ = [
    "ARM_DIM",
    "ARM_FEATURE_NAMES",
    "ARM_SLICES",
    "ARM_TAGS",
    "C_V0_DIM",
    "ContactPoint",
    "ContactSet",
    "GLOBAL_DIM",
    "GLOBAL_FEATURE_NAMES",
    "GLOBAL_SLICE",
    "PrivilegedStep",
    "compute_arm_block",
    "compute_c_v0",
    "compute_global_block",
    "flat_feature_names",
    "flat_group_indices",
    "flat_normalize_modes",
    "stack_window",
]
