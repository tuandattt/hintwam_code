"""Integration hook for RoboTwin data generation.

All logic lives here so the patch to the vendored `third_party/RoboTwin` stays
three lines. Insert into `envs/_base_task.py::get_obs`, following the existing
`self.data_type.get(...)` pattern used by every other block there:

    # privileged features (fastwam)
    if self.data_type.get("privileged", False):
        from fastwam.privileged.robotwin_hook import collect_privileged
        pkl_dic["privileged"] = collect_privileged(self)

Then enable `privileged: true` in the task's `data_type` config.

What gets written per step
-------------------------
- ``c``: float32[28], the flattened `c` v0 vector (`schema.flat_feature_names()`)
- ``dt``: float32, physics timestep the forces were computed with
- ``target``: bytes, name of the tracked object (``b""`` when none)
- ``raw_points``: float32[2, P, 13], contact points per arm (`RAW_POINT_FIELDS`),
  NaN-padded to ``P = max_raw_points_per_arm``
- ``raw_num_points``: int32[2], true contact count per arm before truncation
- ``raw_friction``: float32[2], friction coefficient used per arm

Every value is a fixed-shape, fixed-dtype array. That is load-bearing: RoboTwin's
`pkl2hdf5` stacks each key across steps with `np.array` and hands the result to
h5py, which cannot store object arrays (ragged lists, dicts, None) or numpy
unicode strings. The earlier dict-of-lists layout crashed that conversion at the
end of every episode, after which `collect_data.sh` deletes the `.cache` pickles.

`raw` is deliberate insurance. `c` v0's derived features depend on choices that
may want revising later -- the friction coefficient, the cone-margin clip, the
contact threshold. Keeping the raw contacts means those can be recomputed
offline instead of forcing a full regeneration pass, which is the single most
expensive mistake available in this pipeline (`PROPOSED_METHOD.md` §4.3).
Storage is negligible next to the video frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .features import compute_c_v0
from .probe import (
    GripperLinkRoles,
    PrivilegedProbe,
    dump_link_names,
    infer_roles_from_names,
    link_map_from_robot,
    robot_body_keys,
)
from .schema import ARM_TAGS, DEFAULT_FRICTION_COEFF

logger = logging.getLogger(__name__)

_PROBE_ATTR = "_fastwam_privileged_probe"
_CONFIG_ATTR = "fastwam_privileged_config"
_TARGET_ATTR = "fastwam_target_actor"

MAX_RAW_POINTS_PER_ARM = 4
"""Contact points kept per arm in the `raw` dump. Bounds record size; the solver
occasionally reports dozens of near-duplicate points on a single mesh contact."""

RAW_POINT_FIELDS: tuple[str, ...] = (
    "position_x", "position_y", "position_z",
    "normal_x", "normal_y", "normal_z",
    "impulse_x", "impulse_y", "impulse_z",
    "rel_vel_x", "rel_vel_y", "rel_vel_z",
    "separation",
)
"""Column layout of the last axis of ``raw_points``."""


@dataclass
class PrivilegedHookConfig:
    """Generation-time configuration for the privileged channel.

    `roles=None` runs the bootstrap heuristic. That is fine for a pilot run, but
    pin an explicit mapping before generating at scale -- see
    `probe.VERIFY_ON_FIRST_RUN` item 2.
    """

    dt: float
    roles: GripperLinkRoles | None = None
    friction_fallback: float = DEFAULT_FRICTION_COEFF
    force_on_gripper_sign: float = 1.0
    max_raw_points_per_arm: int = MAX_RAW_POINTS_PER_ARM
    store_raw: bool = True

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"`dt` must be > 0, got {self.dt}")
        if self.max_raw_points_per_arm < 0:
            raise ValueError(
                f"`max_raw_points_per_arm` must be >= 0, got {self.max_raw_points_per_arm}"
            )


@dataclass
class _RawArmDump:
    points: list[dict] = field(default_factory=list)
    friction_coeff: float = DEFAULT_FRICTION_COEFF
    num_points_total: int = 0


def _resolve_config(task) -> PrivilegedHookConfig:
    config = getattr(task, _CONFIG_ATTR, None)
    if isinstance(config, PrivilegedHookConfig):
        return config
    if isinstance(config, dict):
        return PrivilegedHookConfig(**config)
    dt = _infer_dt(task)
    logger.warning(
        "no `%s` on the task; defaulting to dt=%.6g with heuristic link roles. "
        "Pin an explicit PrivilegedHookConfig before generating at scale.",
        _CONFIG_ATTR,
        dt,
    )
    return PrivilegedHookConfig(dt=dt)


def _infer_dt(task) -> float:
    """Best-effort physics timestep. Wrong dt rescales every force linearly."""
    for getter in (
        lambda: float(task.scene.get_timestep()),
        lambda: float(task.scene.timestep),
        lambda: float(task.frame_dt),
    ):
        try:
            dt = getter()
        except Exception:  # noqa: BLE001 - probing several optional APIs
            continue
        if dt > 0.0:
            return dt
    raise ValueError(
        "could not infer the physics timestep from the task; set "
        f"`task.{_CONFIG_ATTR} = PrivilegedHookConfig(dt=...)` explicitly. "
        "An incorrect dt rescales every logged force."
    )


def _get_probe(task) -> PrivilegedProbe:
    probe = getattr(task, _PROBE_ATTR, None)
    # RoboTwin builds a fresh scene every episode but reuses the task object, so a
    # probe cached across episodes would hold body identities from a dead scene.
    if probe is not None and getattr(task, _PROBE_ATTR + "_scene", None) is getattr(task, "scene", None):
        return probe

    config = _resolve_config(task)
    probe = PrivilegedProbe(
        roles=config.roles or GripperLinkRoles(),
        dt=config.dt,
        friction_fallback=config.friction_fallback,
        force_on_gripper_sign=config.force_on_gripper_sign,
    )
    # Explicit roles win; otherwise RoboTwin's own gripper definition; the name
    # heuristic is the last resort, since RoboTwin link names are generic.
    robot = getattr(task, "robot", None)
    if config.roles is not None:
        probe.bind(task)
        source = "config roles"
    elif robot is not None:
        probe.bind_link_map(link_map_from_robot(robot))
        source = "robot gripper definition"
    else:
        probe.roles = infer_roles_from_names(dump_link_names(task))
        probe.bind(task)
        source = "name heuristic"

    if robot is not None:
        probe.ignore_bodies(robot_body_keys(robot))

    setattr(task, _PROBE_ATTR, probe)
    setattr(task, _PROBE_ATTR + "_config", config)
    setattr(task, _PROBE_ATTR + "_scene", getattr(task, "scene", None))
    # print, not logger: RoboTwin never configures logging, so INFO never shows,
    # and this mapping is exactly what VERIFY_ON_FIRST_RUN item 2 asks to check.
    # Once per run, not per episode: the mapping is the same every scene.
    mapping = ", ".join(f"{n}={a}/{r}" for n, (a, r) in sorted(probe.flat_link_map().items()))
    if getattr(task, _PROBE_ATTR + "_printed", None) != mapping:
        print(
            f"[privileged] probe bound via {source}: dt={config.dt:.6g}, "
            f"{len(probe.flat_link_map())} gripper links, "
            f"{len(probe.ignored_bodies)} robot bodies excluded, "
            f"force_sign={config.force_on_gripper_sign:+.0f}",
            flush=True,
        )
        print(f"[privileged] links: {mapping}", flush=True)
        setattr(task, _PROBE_ATTR + "_printed", mapping)
    return probe


def _raw_dump(step, max_points: int) -> dict:
    out: dict[str, dict] = {}
    for arm in ARM_TAGS:
        contacts = getattr(step, arm)
        # Keep the highest-impulse points: those dominate the aggregate anyway.
        ordered = sorted(
            contacts.points, key=lambda p: float(np.linalg.norm(p.impulse)), reverse=True
        )
        out[arm] = {
            "friction_coeff": float(contacts.friction_coeff),
            "num_points_total": len(contacts.points),
            "points": [
                {
                    "position": p.position.astype(np.float32),
                    "normal": p.normal.astype(np.float32),
                    "impulse": p.impulse.astype(np.float32),
                    "rel_vel": p.rel_vel.astype(np.float32),
                    "separation": float(p.separation),
                }
                for p in ordered[:max_points]
            ],
        }
    return out


def _pack_raw(raw: dict, max_points: int) -> dict[str, np.ndarray]:
    """Fixed-shape arrays from `_raw_dump`'s dict form, so HDF5 can store them."""
    points = np.full((len(ARM_TAGS), max_points, len(RAW_POINT_FIELDS)), np.nan, dtype=np.float32)
    num_points = np.zeros(len(ARM_TAGS), dtype=np.int32)
    friction = np.zeros(len(ARM_TAGS), dtype=np.float32)
    for a, arm in enumerate(ARM_TAGS):
        dump = raw[arm]
        num_points[a] = int(dump["num_points_total"])
        friction[a] = float(dump["friction_coeff"])
        for i, pt in enumerate(dump["points"][:max_points]):
            points[a, i, 0:3] = pt["position"]
            points[a, i, 3:6] = pt["normal"]
            points[a, i, 6:9] = pt["impulse"]
            points[a, i, 9:12] = pt["rel_vel"]
            points[a, i, 12] = pt["separation"]
    return {"raw_points": points, "raw_num_points": num_points, "raw_friction": friction}


def _build_record(step, target_name, config: PrivilegedHookConfig) -> dict:
    """One step's record. Every leaf is fixed-shape so `pkl2hdf5` can stack it."""
    record: dict = {
        "c": np.asarray(compute_c_v0(step), dtype=np.float32),
        "dt": np.float32(step.dt),
        "target": np.bytes_((target_name or "").encode("utf-8")),
    }
    if config.store_raw:
        raw = _raw_dump(step, config.max_raw_points_per_arm)
        record.update(_pack_raw(raw, config.max_raw_points_per_arm))
    return record


def collect_privileged(task, target_actor=None) -> dict:
    """Sample the privileged channel for the current simulator state.

    Returns the fixed-shape record described in the module docstring.

    Safe to call at the simulator control rate; `K` is chosen at load time by
    decimating, not at generation time (`PROPOSED_METHOD.md` §4.3).
    """
    probe = _get_probe(task)
    config = getattr(task, _PROBE_ATTR + "_config")
    if target_actor is None:
        target_actor = getattr(task, _TARGET_ATTR, None)

    step = probe.sample(task, target_actor=target_actor)
    return _build_record(step, probe.last_target_name, config)
