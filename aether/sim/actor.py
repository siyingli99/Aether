"""Actor / Scene primitives.

Coordinate conventions follow ``aether/envs/driving_env.py``:

* ``c2w`` poses are 4x4 numpy arrays in **DriveStudio (DS) world frame**
  (z-up).  Same convention :class:`ActorReplay` and the renderer expect.
* For BEV/IDM bookkeeping we additionally cache (x_bev, y_bev, yaw_bev) in
  the **HUGSIM convention** (x = right, y = forward, z = up), because
  ``info["obj_boxes"]`` and the BEV renderer already use it.

The primary contract is ``Scene.pose_at(frame_idx) -> list[dict]`` returning
items shaped like::

    {"class": str, "instance_id": int, "size": [w, l, h],
     "c2w": np.ndarray(4,4), "actor_id": int, "velo": float, "yaw": float}

so the existing :class:`AetherDrivingEnv` can drop a :class:`Scene` directly
into ``self.actors`` with no further modifications.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as SCR


# DS-world axes per dataset (drivestudio convention):
#   pandaset : X = lateral_right, Y = forward, Z = up
#   nuscenes : X = right,         Y = down,    Z = forward (FRONT-cam OpenCV)
# We project actor poses to a canonical BEV  (x_bev = forward, y_bev = left)
# matching ``_ego_box_envframe`` (box[0]=fwd, box[1]=left).

_DS_DATASET: str = "pandaset"


def set_ds_dataset(name: str) -> None:
    """Configure DS-world axis convention used by ActorState BEV roundtrip.

    Called by ``AetherDrivingEnv.__init__`` once per env so that pose_at()
    can faithfully round-trip nuscenes c2w through (x_bev, y_bev, yaw_bev).
    """
    global _DS_DATASET
    if name not in ("pandaset", "nuscenes"):
        raise ValueError(f"Unknown DS dataset {name!r}")
    _DS_DATASET = name


def _ds_axes(dataset: str | None = None):
    """Return (fwd_axis, lat_axis, up_axis, up_sign) for the DS-world frame."""
    ds = dataset or _DS_DATASET
    if ds == "nuscenes":
        return 2, 0, 1, -1.0   # fwd=Z, lat=X(right+), up=-Y (Y is down)
    return 1, 0, 2, +1.0       # pandaset: fwd=Y, lat=X(right+), up=+Z


def _bev_from_c2w(c2w: np.ndarray) -> Tuple[float, float, float]:
    """Project a DS-world c2w to canonical BEV (x_fwd, y_left, yaw_around_up)."""
    fwd_ax, lat_ax, up_ax, up_sign = _ds_axes()
    t = np.asarray(c2w, dtype=np.float32)[:3, 3]
    R = np.asarray(c2w, dtype=np.float32)[:3, :3]
    # Build env-frame rotation (x=fwd, y=left, z=up) from DS rotation.
    P = np.zeros((3, 3), dtype=np.float32)
    P[0, fwd_ax] = 1.0       # env x = DS fwd
    P[1, lat_ax] = -1.0      # env y = -DS lat (= left, since DS lat is right+)
    P[2, up_ax]  = float(up_sign)
    R_env = (P @ R @ P.T).astype(np.float32)
    yaw = float(SCR.from_matrix(R_env).as_euler("ZYX")[0])
    return float(t[fwd_ax]), -float(t[lat_ax]), yaw


def _c2w_from_bev(x_bev: float, y_bev: float, yaw_bev: float,
                  z_ds: float = 0.0) -> np.ndarray:
    """Inverse of :func:`_bev_from_c2w` — build DS-world c2w from BEV state.

    ``z_ds`` is the raw DS-world height (axis index depends on dataset);
    callers should pass through whatever was stored in ``ActorState.z_ds``.
    """
    fwd_ax, lat_ax, up_ax, up_sign = _ds_axes()
    # Env rotation around up-axis (z), then map back to DS rotation.
    R_env = SCR.from_euler("ZYX", [float(yaw_bev), 0.0, 0.0]).as_matrix().astype(np.float32)
    P = np.zeros((3, 3), dtype=np.float32)
    P[0, fwd_ax] = 1.0
    P[1, lat_ax] = -1.0
    P[2, up_ax]  = float(up_sign)
    R_ds = (P.T @ R_env @ P).astype(np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R_ds
    c2w[fwd_ax, 3] = float(x_bev)
    c2w[lat_ax, 3] = -float(y_bev)
    c2w[up_ax,  3] = float(z_ds)   # raw DS-axis height (callers preserve original)
    return c2w


@dataclass
class ActorState:
    x: float = 0.0          # HUGSIM-BEV x (right+)
    y: float = 0.0          # HUGSIM-BEV y (forward+)
    yaw: float = 0.0        # HUGSIM-BEV yaw (rad)
    velo: float = 0.0       # forward speed (m/s)
    z_ds: float = 0.0       # DS world height (y-axis in DS)

    def as_c2w(self) -> np.ndarray:
        return _c2w_from_bev(self.x, self.y, self.yaw, z_ds=self.z_ds)

    @classmethod
    def from_c2w(cls, c2w: np.ndarray, velo: float = 0.0) -> "ActorState":
        x, y, yaw = _bev_from_c2w(c2w)
        _, _, up_ax, _ = _ds_axes()
        return cls(x=x, y=y, yaw=yaw, velo=velo, z_ds=float(c2w[up_ax, 3]))


@dataclass
class Actor:
    """A single dynamic agent in a scene.

    ``policy`` decides the next state each frame; ``state`` is the current
    HUGSIM-BEV bookkeeping; ``c2w_cache`` is regenerated on demand.
    """
    actor_id: int
    class_name: str
    instance_id: int
    size: Tuple[float, float, float] = (2.0, 4.5, 1.5)  # (w, l, h)
    policy: Optional[Any] = None  # ActorPolicy; forward decl to avoid cycle
    state: ActorState = field(default_factory=ActorState)
    asset_path: Optional[str] = None
    label: str = "vehicle"
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "actor_id":    int(self.actor_id),
            "class":       self.class_name,
            "instance_id": int(self.instance_id),
            "size":        list(self.size),
            "c2w":         self.state.as_c2w(),
            "velo":        float(self.state.velo),
            "yaw":         float(self.state.yaw),
            "x":           float(self.state.x),
            "y":           float(self.state.y),
            "label":       self.label,
            "asset_path":  self.asset_path,
        }


class Scene:
    """A registry of actors plus an ego reference.

    Compatible with ``AetherDrivingEnv`` because ``pose_at(frame)`` returns
    list[dict] with at minimum ``{class, instance_id, size, c2w}``.
    """

    def __init__(self, actors: Optional[Sequence[Actor]] = None):
        self.actors: List[Actor] = list(actors or [])
        self._next_id: int = 1 + max((a.actor_id for a in self.actors), default=-1)
        # Lightweight ego mirror (set externally by the driving env each step).
        self.ego: Dict[str, float] = {"x": 0.0, "y": 0.0, "yaw": 0.0, "velo": 0.0}
        # Per-frame state log keyed by actor_id; populated as policies advance.
        self.history: Dict[int, Dict[int, ActorState]] = {}

    # ----------------- editing API ----------------- #
    def add(self, actor: Actor) -> Actor:
        if actor.actor_id < 0:
            actor.actor_id = self._next_id
        self._next_id = max(self._next_id, actor.actor_id + 1)
        self.actors.append(actor)
        return actor

    def remove(self, actor_id: int) -> bool:
        n0 = len(self.actors)
        self.actors = [a for a in self.actors if a.actor_id != actor_id]
        return len(self.actors) != n0

    def get(self, actor_id: int) -> Optional[Actor]:
        for a in self.actors:
            if a.actor_id == actor_id:
                return a
        return None

    def replace_asset(self, actor_id: int, *,
                      class_name: Optional[str] = None,
                      instance_id: Optional[int] = None,
                      asset_path: Optional[str] = None,
                      size: Optional[Tuple[float, float, float]] = None) -> bool:
        a = self.get(actor_id)
        if a is None:
            return False
        if class_name is not None:    a.class_name = class_name
        if instance_id is not None:   a.instance_id = int(instance_id)
        if asset_path is not None:    a.asset_path = asset_path
        if size is not None:          a.size = tuple(size)
        return True

    def update_ego(self, *, x: float, y: float, yaw: float, velo: float) -> None:
        self.ego.update(x=float(x), y=float(y), yaw=float(yaw), velo=float(velo))

    # ----------------- runtime hook (driving_env compatible) ----------------- #
    def pose_at(self, frame_idx: int) -> List[Dict[str, Any]]:
        """Step every active actor and return renderer-ready dicts."""
        out: List[Dict[str, Any]] = []
        for actor in self.actors:
            if not actor.enabled or actor.policy is None:
                continue
            new_state = actor.policy.step(actor=actor, frame_idx=int(frame_idx),
                                          scene=self)
            if new_state is None:
                continue
            actor.state = new_state
            self.history.setdefault(actor.actor_id, {})[int(frame_idx)] = new_state
            out.append(actor.to_dict())
        return out

    # ----------------- helpers ----------------- #
    def actor_ids(self) -> List[int]:
        return [a.actor_id for a in self.actors]

    def serialize(self) -> Dict[str, Any]:
        """JSON-friendly dump for editor / scenario yaml export."""
        return {
            "actors": [
                {
                    "actor_id":    a.actor_id,
                    "class":       a.class_name,
                    "instance_id": a.instance_id,
                    "size":        list(a.size),
                    "asset_path":  a.asset_path,
                    "label":       a.label,
                    "enabled":     a.enabled,
                    "policy":      (a.policy.serialize() if a.policy is not None else None),
                    "state":       {
                        "x":     a.state.x, "y": a.state.y, "yaw": a.state.yaw,
                        "velo":  a.state.velo, "z_ds": a.state.z_ds,
                    },
                } for a in self.actors
            ],
        }


# ----- helpers reused by adapters and tests ----- #
def scene_from_legacy_actor_replay(actor_replay) -> Scene:
    """Build a :class:`Scene` from an existing ``ActorReplay`` so callers can
    swap the env's ``actors`` field with no behavioural change."""
    from .policy import LogReplayPolicy
    scene = Scene()
    for entry in actor_replay.entries:
        traj_pairs = sorted(entry["trajectory"].items())
        traj = {int(f): np.asarray(c, dtype=np.float32) for f, c in traj_pairs}
        actor = Actor(
            actor_id=-1,
            class_name=entry["class"],
            instance_id=int(entry["instance_id"]),
            size=tuple(entry.get("size", [2.0, 4.5, 1.5])),
            policy=LogReplayPolicy(traj),
            label="vehicle",
        )
        scene.add(actor)
    return scene
