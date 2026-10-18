"""Actor policies: log-replay, IDM-following, manual override.

Each policy implements ``step(actor, frame_idx, scene) -> ActorState | None``.
Returning ``None`` means "actor is not active at this frame" (the renderer
will skip it).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import math
import numpy as np

from .actor import Actor, ActorState, _ds_axes


class ActorPolicy(ABC):
    name: str = "base"

    @abstractmethod
    def step(self, *, actor: Actor, frame_idx: int, scene) -> Optional[ActorState]:
        ...

    def serialize(self) -> Dict[str, Any]:
        return {"type": self.name}


class LogReplayPolicy(ActorPolicy):
    """Replays a recorded trajectory keyed by frame index.

    Behaviourally equivalent to the legacy ``ActorReplay`` per-entry path.
    """
    name = "log_replay"

    def __init__(self, trajectory: Dict[int, np.ndarray], frame_dt: float = 0.1):
        self.trajectory: Dict[int, np.ndarray] = {
            int(k): np.asarray(v, dtype=np.float32) for k, v in trajectory.items()
        }
        self._frames = sorted(self.trajectory.keys())
        self.frame_dt = float(frame_dt)

    def step(self, *, actor: Actor, frame_idx: int, scene) -> Optional[ActorState]:
        c2w = self.trajectory.get(int(frame_idx))
        if c2w is None:
            return None
        velo = float(actor.state.velo)
        if frame_idx - 1 in self.trajectory:
            fwd_ax, lat_ax, _, _ = _ds_axes()
            prev = self.trajectory[frame_idx - 1]
            d_fwd = float(c2w[fwd_ax, 3] - prev[fwd_ax, 3])
            d_lat = float(c2w[lat_ax, 3] - prev[lat_ax, 3])
            velo = math.hypot(d_fwd, d_lat) / self.frame_dt
        return ActorState.from_c2w(c2w, velo=velo)

    def serialize(self) -> Dict[str, Any]:
        return {"type": self.name, "n_frames": len(self.trajectory)}


class ManualOverridePolicy(ActorPolicy):
    """Per-frame keyframes from the BEV editor.

    keyframes maps frame_idx -> {"x", "y", "yaw", "velo", "z_ds"?}.  Frames
    between keyframes are linearly interpolated; outside the recorded range
    we hold the closest endpoint.  Returning None requires no keyframes at all.
    """
    name = "manual"

    def __init__(self, keyframes: Dict[int, Dict[str, float]]):
        self.keyframes = {int(k): {kk: float(vv) for kk, vv in v.items()}
                          for k, v in keyframes.items()}
        self._sorted = sorted(self.keyframes.keys())

    def step(self, *, actor: Actor, frame_idx: int, scene) -> Optional[ActorState]:
        if not self._sorted:
            return None
        f = int(frame_idx)
        if f in self.keyframes:
            kf = self.keyframes[f]
            return ActorState(x=kf.get("x", 0.0), y=kf.get("y", 0.0),
                              yaw=kf.get("yaw", 0.0), velo=kf.get("velo", 0.0),
                              z_ds=kf.get("z_ds", actor.state.z_ds))
        prev = next((k for k in reversed(self._sorted) if k <= f), None)
        nxt  = next((k for k in self._sorted if k > f), None)
        if prev is None and nxt is None:
            return None
        if prev is None:
            kf = self.keyframes[nxt]
            return ActorState(x=kf["x"], y=kf["y"], yaw=kf["yaw"], velo=kf["velo"])
        if nxt is None:
            kf = self.keyframes[prev]
            return ActorState(x=kf["x"], y=kf["y"], yaw=kf["yaw"], velo=kf["velo"])
        a, b = self.keyframes[prev], self.keyframes[nxt]
        t = (f - prev) / max(nxt - prev, 1)
        def _lerp(k):
            return (1 - t) * a.get(k, 0.0) + t * b.get(k, 0.0)
        ay, by = a.get("yaw", 0.0), b.get("yaw", 0.0)
        d = (by - ay + math.pi) % (2 * math.pi) - math.pi
        yaw = ay + t * d
        return ActorState(x=_lerp("x"), y=_lerp("y"), yaw=yaw, velo=_lerp("velo"))

    def serialize(self) -> Dict[str, Any]:
        return {"type": self.name, "keyframes": self.keyframes}


@dataclass
class IDMParams:
    v0: float = 12.0
    T: float = 1.5
    s0: float = 2.0
    a_max: float = 1.5
    b_comf: float = 2.0
    delta: float = 4.0


class IDMFollowerPolicy(ActorPolicy):
    """Intelligent Driver Model: react to the closest forward agent.

    Scans every other actor + ego, picks the one with smallest signed forward
    distance within ``lateral_window``, and applies the standard IDM accel
    update at simulator dt.
    """
    name = "idm"

    def __init__(self, params: Optional[IDMParams] = None,
                 dt: float = 0.1, lateral_window: float = 2.5):
        self.p = params or IDMParams()
        self.dt = float(dt)
        self.lateral_window = float(lateral_window)
        self._initialized: Dict[int, bool] = {}

    def reset(self) -> None:
        self._initialized.clear()

    def _init_state(self, actor: Actor) -> ActorState:
        s = actor.state
        if s.velo == 0.0:
            s = ActorState(x=s.x, y=s.y, yaw=s.yaw,
                           velo=self.p.v0 * 0.5, z_ds=s.z_ds)
        self._initialized[actor.actor_id] = True
        return s

    def _leader_gap(self, actor: Actor, scene):
        s = actor.state
        cy, sy = math.cos(s.yaw), math.sin(s.yaw)
        candidates = []
        for other in scene.actors:
            if other.actor_id == actor.actor_id or not other.enabled:
                continue
            candidates.append((other.state.x, other.state.y, other.state.velo))
        e = scene.ego
        candidates.append((e["x"], e["y"], e["velo"]))
        best = None
        for ox, oy, ov in candidates:
            dx, dy = ox - s.x, oy - s.y
            fwd = dx * cy + dy * sy
            lat = -dx * sy + dy * cy
            if fwd <= 0.5 or abs(lat) > self.lateral_window:
                continue
            if best is None or fwd < best[0]:
                best = (fwd, ov)
        return best

    def step(self, *, actor: Actor, frame_idx: int, scene) -> Optional[ActorState]:
        s = actor.state if self._initialized.get(actor.actor_id) else self._init_state(actor)
        v = max(s.velo, 0.0)
        leader = self._leader_gap(actor, scene)
        a_free = self.p.a_max * (1 - (v / max(self.p.v0, 1e-3)) ** self.p.delta)
        if leader is None:
            a = a_free
        else:
            gap, v_lead = leader
            dv = v - max(v_lead, 0.0)
            s_star = self.p.s0 + max(0.0, v * self.p.T +
                                     v * dv / (2 * math.sqrt(self.p.a_max * self.p.b_comf)))
            a = a_free - self.p.a_max * (s_star / max(gap, 0.1)) ** 2
        v_new = max(v + a * self.dt, 0.0)
        x_new = s.x + v_new * math.cos(s.yaw) * self.dt
        y_new = s.y + v_new * math.sin(s.yaw) * self.dt
        return ActorState(x=x_new, y=y_new, yaw=s.yaw, velo=v_new, z_ds=s.z_ds)

    def serialize(self) -> Dict[str, Any]:
        return {"type": self.name, "params": self.p.__dict__,
                "dt": self.dt, "lateral_window": self.lateral_window}


def policy_from_dict(d: Dict[str, Any], default_dt: float = 0.1) -> ActorPolicy:
    """Reconstruct a policy from its ``serialize()`` dict.

    ``default_dt`` is the time between two policy updates; the driving env
    passes the closed-loop step (``kinematic.dt``) when the YAML omits ``dt``.
    """
    t = d.get("type")
    if t == "manual":
        return ManualOverridePolicy(d.get("keyframes", {}))
    if t == "idm":
        p = IDMParams(**d.get("params", {}))
        return IDMFollowerPolicy(params=p, dt=d.get("dt", default_dt),
                                 lateral_window=d.get("lateral_window", 2.5))
    if t == "log_replay":
        raise ValueError("log_replay must be reconstructed with explicit trajectory")
    raise ValueError("Unknown policy type: %r" % t)
