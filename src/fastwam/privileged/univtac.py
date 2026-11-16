"""UniVTAC's GelSight tactile as a privileged feature vector.

UniVTAC episodes carry, per finger, a 240x320 depth map, a 1200-marker flow
field, and the sensor pose. That is far too wide to hand a teacher directly and
far more than the signal needs: what distinguishes a seated connector from a
jammed one is how hard, where, and in which direction the gel is being pushed.

This reduces each finger to 11 scalars, mirroring `schema.py`'s role for
RoboTwin's `c` v0, so the same encoder, provider and evaluation scripts work
unchanged.

Layout (22 = 2 fingers x 11), left finger first:

    0  contact_area      fraction of the gel pressed beyond the threshold
    1  depth_mean        mean indentation over contacting pixels
    2  depth_max         peak indentation
    3  centroid_x        where on the gel the contact sits
    4  centroid_y
    5  spread            how concentrated it is
    6  shear_x           mean marker displacement
    7  shear_y
    8  shear_mag_mean    magnitude, which survives cancelling directions
    9  shear_mag_max
    10 twist             curl of the marker field about its centroid

The depth map is a DISTANCE to the gel surface, resting at a constant value
(34.0 in the released data) wherever nothing touches it. Thresholding it
directly marks every pixel as contact; the indentation against a per-pixel
baseline is what carries information.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "FINGERS",
    "align_episode",
    "PER_FINGER",
    "UNIVTAC_C_DIM",
    "feature_names",
    "flat_group_indices",
    "tactile_features",
]

FINGERS = ("left_gsmini", "right_gsmini")
PER_FINGER = 11
UNIVTAC_C_DIM = PER_FINGER * len(FINGERS)

# Rest distance of an untouched gel pixel, in the released data's units. Only a
# fallback: the per-pixel median is preferred, being robust to a sensor whose
# rest value differs.
GEL_REST_DISTANCE = 34.0
# Indentation counted as contact. Chosen from the released data, where an
# untouched pixel is flat at the rest value and a pressed one moves by ~0.5.
DEFAULT_DEPTH_THRESHOLD = 0.05

_FEATURE_SUFFIXES = (
    "contact_area", "depth_mean", "depth_max", "centroid_x", "centroid_y",
    "spread", "shear_x", "shear_y", "shear_mag_mean", "shear_mag_max", "twist",
)

# Group -> offsets inside one finger's block. Used by the ablations and by the
# inferability screen, which reports transferability per physical quantity.
FEATURE_GROUPS = {
    "contact": (0,),
    "depth": (1, 2),
    "shape": (3, 4, 5),
    "shear": (6, 7, 8, 9),
    "twist": (10,),
}


def feature_names() -> list[str]:
    return [f"{finger}_{suffix}" for finger in FINGERS for suffix in _FEATURE_SUFFIXES]


def flat_group_indices() -> dict[str, tuple[int, ...]]:
    """Group name -> indices into the flattened 22-D vector (both fingers pooled)."""
    return {
        name: tuple(f * PER_FINGER + o for f in range(len(FINGERS)) for o in offsets)
        for name, offsets in FEATURE_GROUPS.items()
    }


def tactile_features(
    depth: np.ndarray,
    marker: np.ndarray,
    depth_thresh: float = DEFAULT_DEPTH_THRESHOLD,
) -> np.ndarray:
    """`depth` `[T, H, W]` and `marker` `[T, 2, M, 2]` -> `[T, 11]` for one finger."""
    if depth.ndim != 3:
        raise ValueError(f"`depth` must be [T, H, W], got {depth.shape}")
    if marker.ndim != 4 or marker.shape[1] != 2 or marker.shape[3] != 2:
        raise ValueError(f"`marker` must be [T, 2, M, 2], got {marker.shape}")
    if len(depth) != len(marker):
        raise ValueError(f"frame count mismatch: depth {len(depth)}, marker {len(marker)}")

    t, h, w = depth.shape
    d = depth.reshape(t, -1).astype(np.float64)
    # Per-pixel rest distance. The median is the untouched value for any pixel
    # that is untouched most of the episode; a pixel pressed in every frame
    # keeps whatever offset it has, which costs a constant, not the signal.
    d = np.median(d, axis=0, keepdims=True) - d  # positive = pressed in
    mask = d > depth_thresh
    contact = d * mask

    count = mask.sum(axis=1)
    area = mask.mean(axis=1)
    depth_mean = np.divide(
        contact.sum(axis=1), np.maximum(count, 1),
        out=np.zeros(t, dtype=np.float64), where=count > 0,
    )
    depth_max = contact.max(axis=1)

    ys, xs = np.mgrid[0:h, 0:w]
    xs = (xs.ravel() / w).astype(np.float64)
    ys = (ys.ravel() / h).astype(np.float64)
    total = np.maximum(contact.sum(axis=1), 1e-12)
    cx = (contact * xs).sum(axis=1) / total
    cy = (contact * ys).sum(axis=1) / total
    spread = np.sqrt(
        (contact * ((xs - cx[:, None]) ** 2 + (ys - cy[:, None]) ** 2)).sum(axis=1) / total
    )
    # No contact means no location: leave those frames at zero rather than
    # reporting the centre of the gel, which a teacher would read as a contact.
    touched = count > 0
    cx, cy, spread = cx * touched, cy * touched, spread * touched

    flow = marker[:, 1].astype(np.float64) - marker[:, 0].astype(np.float64)
    fx, fy = flow[..., 0], flow[..., 1]
    mag = np.hypot(fx, fy)
    px, py = marker[:, 1, :, 0].astype(np.float64), marker[:, 1, :, 1].astype(np.float64)
    rx = px - px.mean(axis=1, keepdims=True)
    ry = py - py.mean(axis=1, keepdims=True)
    twist = (rx * fy - ry * fx).mean(axis=1)

    out = np.stack(
        [area, depth_mean, depth_max, cx, cy, spread,
         fx.mean(axis=1), fy.mean(axis=1), mag.mean(axis=1), mag.max(axis=1), twist],
        axis=1,
    )
    return np.nan_to_num(out).astype(np.float32)


def episode_tactile(handle, depth_thresh: float = DEFAULT_DEPTH_THRESHOLD) -> np.ndarray:
    """Both fingers of an open UniVTAC HDF5 file -> `[T, 22]`."""
    blocks = []
    for finger in FINGERS:
        group = f"tactile/{finger}"
        if group not in handle:
            raise KeyError(f"`{group}` missing; this is not a UniVTAC benchmark episode")
        blocks.append(
            tactile_features(
                handle[f"{group}/depth"][:], handle[f"{group}/marker"][:], depth_thresh
            )
        )
    return np.concatenate(blocks, axis=1)


def align_episode(joints: np.ndarray, tactile: np.ndarray) -> dict[str, np.ndarray]:
    """`action[t] = state[t + 1]`; a T-frame episode yields T - 1 frames.

    UniVTAC stores no action stream, only the realized joint trajectory, so the
    action is the next observed state -- the convention the released RoboTwin
    LeRobot data uses, which lets one dataset config load both.
    """
    if len(joints) != len(tactile):
        raise ValueError(f"length mismatch: joints {len(joints)}, tactile {len(tactile)}")
    if len(joints) < 2:
        raise ValueError("an episode needs at least 2 frames to define one action")
    if tactile.shape[1] != UNIVTAC_C_DIM:
        raise ValueError(f"tactile must have {UNIVTAC_C_DIM} columns, got {tactile.shape[1]}")
    return {
        "state": joints[:-1].astype(np.float32),
        "action": joints[1:].astype(np.float32),
        "c": tactile[:-1].astype(np.float32),
    }
