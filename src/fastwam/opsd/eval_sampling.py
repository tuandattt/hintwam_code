"""Which validation frames the paired evaluation scripts score.

Reading the validation set in order covers only its first few episodes: 200
batches of 4 is 800 frames, about 5 episodes at ~170 frames each. Consecutive
frames are nearly identical, so those batches are not independent samples and
the paired t-statistic comes out inflated, while most held-out episodes -- and
possibly whole tasks -- are never scored.

Spreading the same number of frames evenly over the whole validation set fixes
both, and stays deterministic, so separate runs (baseline vs. checkpoint) still
score identical frames and remain pairable.

`contact_only` narrows further, to frames where a contact flag is actually set.
About 44% of the RoboTwin privileged frames have an all-zero c -- the gripper has
not reached the object, or the whole K-frame window is padding. On those frames a
c-conditioned teacher and a c-blind student solve an identical problem, so they
can only dilute any difference. Scoring the contact frames instead measures the
same effect with roughly twice the concentration, at the cost of no longer being
comparable with an all-frames run: the mode is part of the `sampling` label the
comparison scripts refuse to cross.
"""

from __future__ import annotations

import numpy as np
import torch

from ..privileged.schema import flat_group_indices

__all__ = [
    "SAMPLING_CONTIGUOUS",
    "SAMPLING_SPREAD",
    "contact_frame_indices",
    "eval_subset",
    "spread_indices",
]

SAMPLING_SPREAD = "spread"
SAMPLING_CONTIGUOUS = "contiguous"
PRIVILEGED_COLUMN = "observation.privileged_c"


def spread_indices(num_items: int, num_samples: int) -> list[int]:
    """`num_samples` distinct indices spread evenly over `range(num_items)`."""
    if num_items < 1:
        raise ValueError("the validation set is empty")
    if num_samples < 1:
        raise ValueError(f"`num_samples` must be >= 1, got {num_samples}")
    if num_samples >= num_items:
        return list(range(num_items))
    return np.unique(np.linspace(0, num_items - 1, num=num_samples).round().astype(np.int64)).tolist()


def spread_over(candidates: list[int], num_samples: int) -> list[int]:
    """`num_samples` of `candidates`, spread evenly over that list."""
    picks = spread_indices(len(candidates), num_samples)
    return [candidates[i] for i in picks]


def _column_from_dataset(dataset, column: str) -> np.ndarray:
    """The per-frame `column` for every frame, in dataset index order.

    Read from the parquet-backed `hf_dataset` of each sub-dataset rather than
    through `__getitem__`, which would decode video for all ~6 k frames. The
    concatenation order matches `MultiLeRobotDataset.__getitem__`, which walks
    `_datasets` in the same order and offsets by `num_frames`.
    """
    # `RobotVideoDataset` wraps `BaseLerobotDataset` one level down and indexes
    # it 1:1, so the frame order is shared and either level can be handed in.
    multi = getattr(dataset, "multi_dataset", None)
    if multi is None:
        inner = getattr(dataset, "lerobot_dataset", None)
        multi = getattr(inner, "multi_dataset", None)
    if multi is None or not getattr(multi, "_datasets", None):
        raise TypeError(
            "contact filtering needs a LeRobot-backed dataset with `multi_dataset`; "
            f"got {type(dataset).__name__}"
        )
    chunks = []
    for sub in multi._datasets:
        hf = sub.hf_dataset
        if column not in hf.column_names:
            raise KeyError(
                f"`{column}` is not in this dataset ({sub.repo_id}); contact filtering "
                "needs the privileged dataset produced by scripts/convert_robotwin_privileged.py"
            )
        values = hf.select_columns([column]).to_pandas()[column].to_numpy()
        chunks.append(np.stack([np.asarray(v, dtype=np.float32) for v in values]))
    return np.concatenate(chunks, axis=0)


def contact_frame_indices(dataset, column: str = PRIVILEGED_COLUMN, threshold: float = 0.5) -> list[int]:
    """Indices of frames whose c has at least one contact flag set.

    The flags are binary, so any reasonable threshold gives the same answer; the
    0.5 is there because c is stored as float32.
    """
    c = _column_from_dataset(dataset, column)
    contact_dims = list(flat_group_indices()["contact"])
    if c.ndim != 2 or c.shape[1] <= max(contact_dims):
        raise ValueError(f"expected c of shape [N, 28], got {c.shape}")
    hits = (c[:, contact_dims] > threshold).any(axis=1)
    idx = np.flatnonzero(hits).tolist()
    if not idx:
        raise ValueError(
            "no frame in this split has a contact flag set -- the probe may have "
            "failed during collection, or this is the wrong dataset"
        )
    return idx


def eval_subset(
    dataset,
    num_samples: int,
    contiguous: bool = False,
    contact_only: bool = False,
    privileged_column: str = PRIVILEGED_COLUMN,
):
    """Return `(dataset_to_score, sampling_mode)`.

    `contiguous=True` reproduces the old behaviour (the first `num_samples`
    frames), only for comparing against results measured that way. `contact_only`
    restricts to frames where c is non-degenerate; it composes with the spread so
    the chosen frames still cover the whole split.
    """
    if contiguous:
        if contact_only:
            raise ValueError("`contact_only` and `contiguous` are mutually exclusive")
        return dataset, SAMPLING_CONTIGUOUS
    if contact_only:
        candidates = contact_frame_indices(dataset, column=privileged_column)
        picked = spread_over(candidates, num_samples)
        print(f"contact frames: {len(candidates)} of {len(dataset)} | scoring {len(picked)}")
        return torch.utils.data.Subset(dataset, picked), f"{SAMPLING_SPREAD}+contact"
    return torch.utils.data.Subset(dataset, spread_indices(len(dataset), num_samples)), SAMPLING_SPREAD
