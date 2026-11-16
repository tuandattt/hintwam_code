"""SAPIEN-facing probe: live RoboTwin scene -> `features.PrivilegedStep` records.

Split from `features.py` so the feature *definition* stays testable without a
simulator. Everything here needs a running scene and must be validated on the
training box; see `VERIFY_ON_FIRST_RUN` below.

Gripper link names are **embodiment-specific and not vendored** in this repo --
they come from the RoboTwin asset configs (`fix_gripper_name`, `gripper_name`),
which are downloaded separately. So the link -> role mapping is configuration,
not a constant. Bootstrap it once per embodiment:

    from fastwam.privileged.probe import dump_link_names, infer_roles_from_names
    names = dump_link_names(task)          # prints every rigid-body link name
    print(infer_roles_from_names(names))   # heuristic first guess -- CHECK IT

then pin the result in your generation config.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np

from .features import ContactPoint, ContactSet, PrivilegedStep
from .schema import ARM_TAGS, CONTACT_SEPARATION_TOL, DEFAULT_FRICTION_COEFF

# Stdlib logging rather than `fastwam.utils.logging_config.get_logger`: that
# wrapper only adds rank-awareness (irrelevant for single-process data
# generation) and importing it drags in torch.distributed and imageio via
# `fastwam.utils.__init__`, which would make this module -- and the pure
# resolver logic below -- untestable without the full training stack.
logger = logging.getLogger(__name__)

VERIFY_ON_FIRST_RUN = """
Three things cannot be verified without a live scene. Check them once, on the
first pilot episode, before generating at scale (PROPOSED_METHOD.md Phase 2.4):

1. FORCE SIGN. Rest the gripper on the table with no motion. `force_dir` in the
   EE frame must point *away* from the table (the reaction force on the
   gripper). If it points into the table, flip `force_on_gripper_sign`.
2. LINK ROLES. `infer_roles_from_names` is a keyword heuristic. Print the
   resolved mapping and confirm each finger/palm link is where you expect, and
   that left/right are not swapped.
3. SLIP SIGN AND SCALE. Slide a grasped object out of the gripper. `slip_vel`
   magnitude must rise before the object leaves, and `cone_margin` must cross
   1.0 at the moment it starts moving.

`cone_margin` is sign-robust by construction (the denominator uses |F_n|), so a
wrong force sign shows up in `force_dir` only.
"""

ROLES: tuple[str, ...] = ("finger_l", "finger_r", "palm")

# Word-boundary that also treats `_` as a separator. Plain `\b` does not work
# here because `_` is a word character, so `finger_l\b` fails on `finger_l_link`.
_END = r"(?![a-z0-9])"

# Role patterns are matched against the name with the arm token REMOVED. On a
# bimanual robot "left"/"right" names the arm, so a pattern like `left_?finger`
# would misfile `left_finger_r_link` as the left finger of the gripper.
_ROLE_PATTERNS: dict[str, tuple[str, ...]] = {
    "finger_l": (rf"finger_?(?:l|1|a){_END}", rf"(?:l|1)_?finger{_END}"),
    "finger_r": (rf"finger_?(?:r|2|b){_END}", rf"(?:r|2)_?finger{_END}"),
    "palm": (r"palm", r"hand_?base", r"gripper_?base", r"tcp", r"eef?_?link"),
}
_ARM_PATTERNS: dict[str, tuple[str, ...]] = {
    "left": (r"^left", r"_left(?:_|$)", rf"^l_{_END}", rf"^fl{_END}", rf"arm_?l{_END}"),
    "right": (r"^right", r"_right(?:_|$)", rf"^r_{_END}", rf"^fr{_END}", rf"arm_?r{_END}"),
}


@dataclass
class GripperLinkRoles:
    """Maps simulator link names to (arm, role). Pure data -- no simulator needed.

    `patterns[arm][role]` is a list of regexes matched case-insensitively against
    link names. A link matching more than one role is an error rather than a
    silent first-match, because a mis-assigned finger corrupts the contact flags
    for every episode generated afterwards.
    """

    patterns: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for arm in self.patterns:
            if arm not in ARM_TAGS:
                raise ValueError(f"unknown arm tag {arm!r}, expected one of {ARM_TAGS}")
            for role in self.patterns[arm]:
                if role not in ROLES:
                    raise ValueError(f"unknown role {role!r}, expected one of {ROLES}")

    def resolve(self, link_names: list[str]) -> dict[str, dict[str, set[str]]]:
        """Return `{arm: {role: {link_name, ...}}}` for the given link names."""
        out: dict[str, dict[str, set[str]]] = {
            arm: {role: set() for role in ROLES} for arm in ARM_TAGS
        }
        for name in link_names:
            hits: list[tuple[str, str]] = []
            for arm, roles in self.patterns.items():
                for role, pats in roles.items():
                    if any(re.search(p, name, re.IGNORECASE) for p in pats):
                        hits.append((arm, role))
            if not hits:
                continue
            unique = set(hits)
            if len(unique) > 1:
                raise ValueError(
                    f"link {name!r} matches multiple (arm, role) pairs {sorted(unique)}; "
                    "tighten the patterns -- an ambiguous finger corrupts every episode"
                )
            arm, role = hits[0]
            out[arm][role].add(name)
        return out

    def flat_link_to_role(self, link_names: list[str]) -> dict[str, tuple[str, str]]:
        """Return `{link_name: (arm, role)}`, the form the probe indexes by."""
        resolved = self.resolve(link_names)
        flat: dict[str, tuple[str, str]] = {}
        for arm, roles in resolved.items():
            for role, names in roles.items():
                for name in names:
                    flat[name] = (arm, role)
        return flat


def _match_arm(name: str) -> tuple[str, str] | None:
    """Return `(arm, name_with_arm_token_removed)`, or None if no arm matched."""
    for arm, pats in _ARM_PATTERNS.items():
        for pat in pats:
            m = re.search(pat, name, re.IGNORECASE)
            if m:
                return arm, name[: m.start()] + name[m.end() :]
    return None


def infer_roles_from_names(link_names: list[str]) -> GripperLinkRoles:
    """Bootstrap a `GripperLinkRoles` from observed link names. **Verify the result.**

    Keyword heuristic only. Its output is a starting point to be checked and
    pinned into config, never something to generate data against unreviewed.

    The arm token is stripped before role matching, so `left_finger_r_link`
    resolves to `(left, finger_r)` rather than colliding with the `finger_l`
    patterns.
    """
    patterns: dict[str, dict[str, tuple[str, ...]]] = {arm: {} for arm in ARM_TAGS}
    for name in link_names:
        arm_match = _match_arm(name)
        if arm_match is None:
            continue
        arm_hit, stripped = arm_match
        role_hit = next(
            (r for r, pats in _ROLE_PATTERNS.items()
             if any(re.search(p, stripped, re.IGNORECASE) for p in pats)),
            None,
        )
        if role_hit is None:
            continue
        # Pin the exact observed name, so `resolve` is an exact match rather than
        # inheriting the heuristic's fuzziness.
        existing = patterns[arm_hit].get(role_hit, ())
        pinned = rf"^{re.escape(name)}$"
        patterns[arm_hit][role_hit] = tuple(sorted(set(existing) | {pinned}))

    roles = GripperLinkRoles(patterns=patterns)
    missing = [
        f"{arm}/{role}"
        for arm in ARM_TAGS
        for role in ROLES
        if not patterns.get(arm, {}).get(role)
    ]
    if missing:
        logger.warning(
            "infer_roles_from_names could not assign: %s. "
            "Fill these in manually from the dumped link names.",
            ", ".join(missing),
        )
    return roles


def _entity_key(entity) -> tuple:
    """Identity of a scene entity, valid within one scene.

    Prefers SAPIEN's numeric ids over names: object meshes often use generic link
    names (`link_1`) that can coincide with a robot link's name.
    """
    for attr in ("per_scene_id", "global_id"):
        value = getattr(entity, attr, None)
        if value is not None:
            return (attr, int(value))
    return ("name", entity.name)


def robot_body_keys(robot) -> frozenset:
    """Entity keys of every link of the robot's articulation(s).

    Covers arm links, wrist links and wrist-mounted cameras. A gripper link touching
    any of these is the robot touching itself, not a grasp: on aloha-agilex the
    palm link sits against the wrist and camera links for the whole episode, which
    made every contact flag read 1 from step 0 and picked `left_camera` as the
    manipulated object.
    """
    keys: set = set()
    articulations: list = []
    for attr in ("left_entity", "right_entity"):
        art = getattr(robot, attr, None)
        if art is None or any(art is seen for seen in articulations):
            continue  # dual-arm embodiments share one articulation
        articulations.append(art)
        for link in art.get_links():
            keys.add(_entity_key(link.entity))
    return frozenset(keys)


def link_map_from_robot(robot) -> dict[str, tuple[str, str]]:
    """`{link_name: (arm, role)}` from RoboTwin's own gripper definition.

    Preferred over the name heuristic: RoboTwin embodiments use generic link
    names (`fl_link7`) that no keyword can classify, but `envs/robot/robot.py`
    already records which links form each gripper, and RoboTwin's grasp check
    matches contact entity names against exactly these (`Robot.gripper_name`).

    - `finger_l` / `finger_r`: child links of the gripper joints, in embodiment
      config order (`gripper_name.base`, then its `mimic`). That order, not the
      physical side, decides which is "l"; the features only need the two fingers
      kept distinct and consistent across episodes, which config order guarantees.
    - `palm`: the embodiment's `fix_gripper_name` links, or the parent link of the
      base finger joint when the config lists none.

    Duck-typed on the `Robot` attributes, so it is testable without SAPIEN.
    """
    link_map: dict[str, tuple[str, str]] = {}
    for arm in ARM_TAGS:
        joints = getattr(robot, f"{arm}_gripper", None) or []
        joints = [j[0] for j in joints if j is not None and j[0] is not None]
        if not joints:
            raise ValueError(f"robot has no {arm} gripper joints; cannot resolve gripper links")
        moving = list(dict.fromkeys(j.child_link.get_name() for j in joints))
        if len(moving) != 2:
            raise ValueError(
                f"{arm} gripper has {len(moving)} moving links {moving}; the finger_l / "
                "finger_r split needs exactly 2. Pin `PrivilegedHookConfig.roles` by "
                "hand for this embodiment."
            )
        palm = list(getattr(robot, f"{arm}_fix_gripper_name", None) or [])
        if not palm:
            palm = [joints[0].parent_link.get_name()]

        for name, role in [(moving[0], "finger_l"), (moving[1], "finger_r")] + [(n, "palm") for n in palm]:
            previous = link_map.get(name)
            if previous is not None and previous != (arm, role):
                raise ValueError(
                    f"link {name!r} resolves to both {previous} and {(arm, role)}; "
                    "an ambiguous gripper link corrupts every episode"
                )
            link_map[name] = (arm, role)
    return link_map


# --------------------------------------------------------------------------- #
# simulator-facing
# --------------------------------------------------------------------------- #

def _require_sapien():
    try:
        import sapien.physx as physx  # noqa: PLC0415 - optional heavy dependency
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(
            "fastwam.privileged.probe requires `sapien`. It is only needed on the "
            "data-generation box; the feature core in `features.py` does not."
        ) from exc
    return physx


def dump_link_names(task) -> list[str]:
    """List every rigid-body link name in the scene. Diagnostic for role mapping."""
    physx = _require_sapien()
    names: list[str] = []
    for entity in task.scene.entities:
        for comp in entity.components:
            if isinstance(comp, physx.PhysxRigidBaseComponent):
                names.append(entity.name)
                break
    names = sorted(set(names))
    logger.info("scene has %d rigid-body links:\n  %s", len(names), "\n  ".join(names))
    return names


def _velocity_at_point(component, point: np.ndarray) -> np.ndarray:
    """World-frame velocity of a rigid body at `point`. Zero for static bodies.

    Uses `v + omega x (p - com)`. SAPIEN reports `angular_velocity` about the
    centre of mass, so the CoM is the correct reference; falls back to the link
    origin when `cmass_local_pose` is unavailable.
    """
    lin = getattr(component, "linear_velocity", None)
    if lin is None:
        return np.zeros(3)  # static body
    lin = np.asarray(lin, dtype=np.float64).reshape(3)
    ang = np.asarray(getattr(component, "angular_velocity", np.zeros(3)), np.float64).reshape(3)
    try:
        pose = component.get_pose()
        cmass_local = getattr(component, "cmass_local_pose", None)
        com = np.asarray((pose * cmass_local).p if cmass_local is not None else pose.p, np.float64)
    except Exception:  # noqa: BLE001 - geometry access varies by component type
        com = point
    return lin + np.cross(ang, point - com)


def _dynamic_friction(component, fallback: float) -> float:
    try:
        for shape in component.get_collision_shapes():
            material = getattr(shape, "physical_material", None)
            if material is not None:
                mu = float(material.dynamic_friction)
                if mu > 0.0:
                    return mu
    except Exception:  # noqa: BLE001 - not all components expose shapes
        pass
    return fallback


@dataclass
class PrivilegedProbe:
    """Samples one `PrivilegedStep` from a live RoboTwin task.

    Construct once per episode; call `sample()` at the simulator control rate and
    decimate downstream, so `K` stays a dataloader parameter (PROPOSED_METHOD.md §4.3).
    """

    roles: GripperLinkRoles
    dt: float
    friction_fallback: float = DEFAULT_FRICTION_COEFF
    force_on_gripper_sign: float = 1.0
    """Flip to -1.0 if the empirical check in `VERIFY_ON_FIRST_RUN` item 1 fails."""

    _link_map: dict[str, tuple[str, str]] | None = None
    ignored_bodies: frozenset = frozenset()
    """`_entity_key`s of bodies whose contact with a gripper is not a grasp -- the
    robot's own links. Set per scene with `ignore_bodies`."""
    last_target_name: str | None = None
    """Entity whose velocity filled the global block on the last `sample()`.
    Log it: if it flips between objects mid-episode, the global block is noise."""

    def __post_init__(self) -> None:
        if self.dt <= 0.0:
            raise ValueError(f"`dt` must be > 0, got {self.dt}")
        if self.force_on_gripper_sign not in (1.0, -1.0):
            raise ValueError(
                f"`force_on_gripper_sign` must be +1.0 or -1.0, got {self.force_on_gripper_sign}"
            )

    def bind(self, task) -> "PrivilegedProbe":
        """Resolve link names against this scene. Call once per episode."""
        self._link_map = self.roles.flat_link_to_role(dump_link_names(task))
        if not self._link_map:
            raise ValueError(
                "no scene link matched any gripper role -- the role patterns do not fit "
                "this embodiment. Run `infer_roles_from_names(dump_link_names(task))`."
            )
        return self

    def bind_link_map(self, link_map: dict[str, tuple[str, str]]) -> "PrivilegedProbe":
        """Bind a pre-resolved link map directly, bypassing scene inspection."""
        if not link_map:
            raise ValueError("`link_map` must be non-empty")
        self._link_map = dict(link_map)
        return self

    def ignore_bodies(self, keys) -> "PrivilegedProbe":
        """Exclude contacts between a gripper and these bodies (see `robot_body_keys`)."""
        self.ignored_bodies = frozenset(keys)
        return self

    def flat_link_map(self) -> dict[str, tuple[str, str]]:
        """The resolved `{link_name: (arm, role)}` map. Requires `bind` first."""
        if self._link_map is None:
            raise ValueError("call `bind(task)` before `flat_link_map()`")
        return dict(self._link_map)

    def sample(self, task, target_actor=None) -> PrivilegedStep:
        """Build one `PrivilegedStep`.

        `target_actor` is the manipulated object for the global block. Pass None to
        auto-resolve it as the body carrying the largest contact impulse against a
        gripper. Auto-resolution is task-agnostic -- RoboTwin tasks each name their
        own object (`self.pot`, ...) with no shared registry -- and it matches what
        the feature is for: is the thing I am holding actually moving?
        """
        if self._link_map is None:
            raise ValueError("call `bind(task)` before `sample()`")

        per_arm: dict[str, dict] = {
            arm: {"points": [], "roles": set(), "mu": None} for arm in ARM_TAGS
        }
        # entity name -> (total |impulse|, component), for auto-resolving the target.
        candidates: dict[str, tuple[float, object]] = {}

        for contact in task.scene.get_contacts():
            body_a, body_b = contact.bodies[0], contact.bodies[1]
            name_a, name_b = body_a.entity.name, body_b.entity.name
            hit_a, hit_b = self._link_map.get(name_a), self._link_map.get(name_b)
            if (hit_a is None) == (hit_b is None):
                continue  # neither is a gripper link, or gripper self-contact
            if hit_a is not None:
                arm, role, gripper, other, flip = *hit_a, body_a, body_b, 1.0
            else:
                arm, role, gripper, other, flip = *hit_b, body_b, body_a, -1.0
            if self.ignored_bodies and _entity_key(other.entity) in self.ignored_bodies:
                continue  # gripper against the robot's own wrist, arm or camera
            touching = [
                pt for pt in contact.points
                if float(getattr(pt, "separation", 0.0)) <= CONTACT_SEPARATION_TOL
            ]
            if not touching:
                continue  # speculative contact inside the contact offset, not a touch

            bucket = per_arm[arm]
            bucket["roles"].add(role)
            if bucket["mu"] is None:
                bucket["mu"] = _dynamic_friction(other, self.friction_fallback)

            for pt in touching:
                position = np.asarray(pt.position, dtype=np.float64).reshape(3)
                # Orient so the normal points away from the gripper and the force
                # is the force acting *on* the gripper.
                normal = np.asarray(pt.normal, dtype=np.float64).reshape(3) * flip
                if float(np.linalg.norm(normal)) <= 0.0:
                    continue
                impulse = (
                    np.asarray(pt.impulse, dtype=np.float64).reshape(3)
                    * flip
                    * self.force_on_gripper_sign
                )
                rel_vel = _velocity_at_point(other, position) - _velocity_at_point(
                    gripper, position
                )
                bucket["points"].append(
                    ContactPoint(
                        position=position,
                        normal=normal,
                        impulse=impulse,
                        rel_vel=rel_vel,
                        separation=float(getattr(pt, "separation", 0.0)),
                    )
                )
                other_name = other.entity.name
                prev = candidates.get(other_name, (0.0, other))
                candidates[other_name] = (
                    prev[0] + float(np.linalg.norm(impulse)),
                    other,
                )

        contact_sets: dict[str, ContactSet] = {}
        for arm in ARM_TAGS:
            bucket = per_arm[arm]
            contact_sets[arm] = ContactSet(
                ee_rot_world_to_ee=self._ee_rot_world_to_ee(task, arm),
                points=bucket["points"],
                touching_finger_l="finger_l" in bucket["roles"],
                touching_finger_r="finger_r" in bucket["roles"],
                touching_palm="palm" in bucket["roles"],
                friction_coeff=bucket["mu"] or self.friction_fallback,
            )

        if target_actor is None and candidates:
            # Prefer bodies that can move. Static bodies (the table) have no
            # `linear_velocity`; on grab_roller the table out-pushed the roller in
            # more frames than the roller itself, zeroing the object-velocity block
            # while the object was being handled.
            dynamic = {k: v for k, v in candidates.items() if getattr(v[1], "linear_velocity", None) is not None}
            pool = dynamic or candidates
            self.last_target_name = max(pool.items(), key=lambda kv: kv[1][0])[0]
            target_component = candidates[self.last_target_name][1]
            lin, ang = self._component_velocity(target_component)
        elif target_actor is not None:
            self.last_target_name = getattr(
                getattr(target_actor, "actor", target_actor), "name", "<explicit>"
            )
            lin, ang = self._actor_velocity(target_actor)
        else:
            # Nothing in contact: the object velocity block is genuinely unknown,
            # and zero is the honest encoding of "not touching anything".
            self.last_target_name = None
            lin, ang = np.zeros(3), np.zeros(3)

        return PrivilegedStep(
            left=contact_sets["left"],
            right=contact_sets["right"],
            obj_lin_vel=lin,
            obj_ang_vel=ang,
            dt=self.dt,
        )

    @staticmethod
    def _component_velocity(component) -> tuple[np.ndarray, np.ndarray]:
        """Linear/angular velocity of a rigid-body component; zeros if static."""
        lin = getattr(component, "linear_velocity", None)
        if lin is None:
            return np.zeros(3), np.zeros(3)
        return (
            np.asarray(lin, dtype=np.float64).reshape(3),
            np.asarray(
                getattr(component, "angular_velocity", np.zeros(3)), dtype=np.float64
            ).reshape(3),
        )

    @staticmethod
    def _ee_rot_world_to_ee(task, arm: str) -> np.ndarray:
        """World -> EE rotation, i.e. the transpose of the EE's world orientation."""
        import transforms3d as t3d  # noqa: PLC0415 - only needed with a live scene

        pose = task.get_arm_pose(arm)  # [x, y, z, qw, qx, qy, qz]
        quat = np.asarray(pose, dtype=np.float64).reshape(-1)[3:7]
        return t3d.quaternions.quat2mat(quat).T

    @staticmethod
    def _actor_velocity(target_actor) -> tuple[np.ndarray, np.ndarray]:
        physx = _require_sapien()
        entity = getattr(target_actor, "actor", target_actor)
        for comp in entity.components:
            if isinstance(comp, physx.PhysxRigidDynamicComponent):
                return (
                    np.asarray(comp.linear_velocity, dtype=np.float64).reshape(3),
                    np.asarray(comp.angular_velocity, dtype=np.float64).reshape(3),
                )
        return np.zeros(3), np.zeros(3)  # static or articulated target
