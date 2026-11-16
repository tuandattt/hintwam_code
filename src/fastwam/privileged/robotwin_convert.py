"""Pure-numpy pieces of the RoboTwin HDF5 -> LeRobot v2.1 conversion.

Kept apart from `scripts/convert_robotwin_privileged.py` so the decisions that
silently corrupt training if wrong -- frame alignment, per-frame instructions,
stats shapes, camera channel order -- are testable without h5py, PyAV or a
generated episode.

Conventions established against the released RoboTwin 2.0 LeRobot dataset:

- `action[t] == state[t + 1]` exactly (max abs difference 0.00000), while
  `action[t] - state[t]` reaches 0.105. So a T-frame RoboTwin episode becomes
  T - 1 LeRobot frames, and the last recorded frame only supplies an action.
- Frame spacing already matches (`save_freq: 15`): released `lift_pot`
  episodes have 107-116 frames and 0.0141 mean joint motion per frame, the
  privileged pilot 119 frames and 0.0134. The `fps: 50` in `info.json` is a label.
- Cameras are 640x480 (`Large_D435`); `D435` renders 320x240.
"""

from __future__ import annotations

import numpy as np

from .schema import C_V0_DIM, flat_feature_names

__all__ = [
    "CAMERA_MAP",
    "PRIVILEGED_KEY",
    "align_episode",
    "frame_task_indices",
    "image_stats",
    "numeric_stats",
    "pick_channel_order",
    "privileged_feature",
]

CAMERA_MAP: dict[str, str] = {
    "observation.images.cam_high": "head_camera",
    "observation.images.cam_left_wrist": "left_camera",
    "observation.images.cam_right_wrist": "right_camera",
}
"""LeRobot video key -> RoboTwin HDF5 camera group under `observation/`."""

PRIVILEGED_KEY = "observation.privileged_c"


def align_episode(joint_vector: np.ndarray, c: np.ndarray) -> dict[str, np.ndarray]:
    """RoboTwin per-frame arrays -> LeRobot state / action / c, `T - 1` frames each.

    Args:
        joint_vector: `[T, 14]` `joint_action/vector` from the episode HDF5.
        c: `[T, 28]` `privileged/c`, recorded at the same steps as the joints.

    Returns:
        `state[t] = joint_vector[t]`, `action[t] = joint_vector[t + 1]` and
        `c[t] = c[t]` for `t in [0, T - 2]`. Frame `t`'s video frame is also
        index `t`, so callers slice cameras with `[:T - 1]`.
    """
    joint_vector = np.asarray(joint_vector)
    c = np.asarray(c)
    if joint_vector.ndim != 2:
        raise ValueError(f"`joint_vector` must be [T, D], got shape {joint_vector.shape}")
    if c.ndim != 2 or c.shape[1] != C_V0_DIM:
        raise ValueError(f"`c` must be [T, {C_V0_DIM}], got shape {c.shape}")
    if len(joint_vector) != len(c):
        raise ValueError(
            f"joints have {len(joint_vector)} frames but c has {len(c)}; they are recorded "
            "at the same simulator steps, so a mismatch means a corrupted episode"
        )
    if len(joint_vector) < 2:
        raise ValueError(f"an episode needs at least 2 frames to form an action, got {len(joint_vector)}")
    return {
        "state": joint_vector[:-1].astype(np.float32),
        "action": joint_vector[1:].astype(np.float32),
        "c": c[:-1].astype(np.float32),
    }


def frame_task_indices(num_frames: int, num_instructions: int, rng: np.random.Generator) -> np.ndarray:
    """Per-frame index into an episode's instruction list.

    The loader reads one instruction per sample from the frame's `task_index`.
    Drawing it independently per frame gives every sample start a varied
    instruction, the same diversity the released multi-instruction episodes have.
    """
    if num_frames < 1:
        raise ValueError(f"`num_frames` must be >= 1, got {num_frames}")
    if num_instructions < 1:
        raise ValueError("an episode needs at least one instruction")
    return rng.integers(0, num_instructions, size=num_frames, dtype=np.int64)


def numeric_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    """LeRobot v2.1 per-episode stats for a `[N]` or `[N, D]` column."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2 or len(arr) == 0:
        raise ValueError(f"expected a non-empty [N] or [N, D] array, got shape {arr.shape}")
    return {
        "min": arr.min(axis=0),
        "max": arr.max(axis=0),
        "mean": arr.mean(axis=0),
        "std": arr.std(axis=0),
        # `aggregate_stats` requires count to be shape (1,).
        "count": np.array([len(arr)], dtype=np.int64),
    }


def image_stats(frames: np.ndarray, max_samples: int = 32) -> dict[str, np.ndarray]:
    """Per-channel stats of uint8 `[N, H, W, 3]` frames, in [0, 1], shaped `(3, 1, 1)`.

    The loader's `_assert_type_and_shape` rejects any other shape for image keys.
    Frames are subsampled uniformly: the stats only need to be representative.
    """
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected [N, H, W, 3] frames, got shape {frames.shape}")
    idx = np.linspace(0, len(frames) - 1, num=min(max_samples, len(frames))).round().astype(int)
    x = frames[idx].astype(np.float64) / 255.0  # [n, H, W, 3]
    per_channel = x.transpose(3, 0, 1, 2).reshape(3, -1)  # [3, n*H*W]
    shape = (3, 1, 1)
    return {
        "min": per_channel.min(axis=1).reshape(shape),
        "max": per_channel.max(axis=1).reshape(shape),
        "mean": per_channel.mean(axis=1).reshape(shape),
        "std": per_channel.std(axis=1).reshape(shape),
        "count": np.array([len(idx)], dtype=np.int64),
    }


def pick_channel_order(decoded: np.ndarray, reference_rgb: np.ndarray) -> tuple[str, float, float]:
    """Decide whether JPEG-decoded frames need their R and B channels swapped.

    RoboTwin stores camera frames with `cv2.imencode`, which assumes BGR, and also
    writes an MP4 of the head camera straight from the raw RGB arrays. Comparing a
    decoded frame against that MP4's frame in both channel orders settles the
    question empirically instead of by reasoning about OpenCV conventions.

    Returns `(order, error_as_decoded, error_swapped)`, `order` in
    `{"as_decoded", "swapped"}`, errors as mean absolute pixel difference.
    """
    decoded = np.asarray(decoded, dtype=np.float64)
    reference_rgb = np.asarray(reference_rgb, dtype=np.float64)
    if decoded.shape != reference_rgb.shape or decoded.shape[-1] != 3:
        raise ValueError(
            f"frames must share an [H, W, 3] shape, got {decoded.shape} and {reference_rgb.shape}"
        )
    err_as = float(np.abs(decoded - reference_rgb).mean())
    err_swap = float(np.abs(decoded[..., ::-1] - reference_rgb).mean())
    return ("as_decoded" if err_as <= err_swap else "swapped"), err_as, err_swap


def privileged_feature() -> dict:
    """`info.json` feature entry for `c`."""
    return {"dtype": "float32", "shape": (C_V0_DIM,), "names": [list(flat_feature_names())]}
