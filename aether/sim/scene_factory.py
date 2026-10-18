"""Build :class:`aether.sim.Scene` from a scenario YAML actor list.

Scenario YAML schema (additions, all optional)::

    scenario:
      actor_backend: scene        # selects Scene over legacy ActorReplay
      traffic_log: traffic/<scene>.npz    # optional sidecar with baked
                                          # per-actor trajectories.  Keys:
                                          #   c2w/<class>/<inst>: (T,4,4)
                                          #   frames/<class>/<inst>: (T,)
                                          #   size/<class>/<inst>: (3,) wlh
      actors:
        - class: RigidNodes
          instance_id: 12
          size: [1.85, 4.5, 1.6]
          state:                  # optional initial BEV pose
            x: 12.0
            y: 0.5
            yaw: 0.0
            velo: 5.0
            z_ds: 0.0
          policy:
            type: log_replay      # or "idm" | "manual"
            # If type == log_replay and no inline trajectory is given,
            # the bakeed trajectory is fetched from `traffic_log`
            # under f"{class}/{instance_id}".
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .actor import Actor, ActorState, Scene
from .policy import (
    ActorPolicy,
    IDMFollowerPolicy,
    IDMParams,
    LogReplayPolicy,
    ManualOverridePolicy,
    policy_from_dict,
)


def _state_from_dict(d: Dict[str, Any]) -> ActorState:
    if not d:
        return ActorState()
    return ActorState(
        x=float(d.get("x", 0.0)),
        y=float(d.get("y", 0.0)),
        yaw=float(d.get("yaw", 0.0)),
        velo=float(d.get("velo", 0.0)),
        z_ds=float(d.get("z_ds", 0.0)),
    )


def _load_traffic_log(path: Optional[str | Path]) -> Dict[str, Dict[int, np.ndarray]]:
    """Returns {f"{class}/{inst}": {frame_idx: c2w_4x4}}."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"traffic_log not found: {p}")
    data = np.load(str(p))
    out: Dict[str, Dict[int, np.ndarray]] = {}
    for key in data.files:
        if not key.startswith("c2w/"):
            continue
        actor_key = key[len("c2w/"):]
        c2w = data[key]                                # (T, 4, 4)
        frames = data[f"frames/{actor_key}"].tolist()  # (T,)
        out[actor_key] = {int(f): np.asarray(c2w[i], dtype=np.float32)
                          for i, f in enumerate(frames)}
    return out


def _actor_from_dict(d: Dict[str, Any],
                     traffic_log: Dict[str, Dict[int, np.ndarray]],
                     sim_dt: float = 0.1,
                     frame_dt: float = 0.1,
                     ) -> Optional[Actor]:
    sz = d.get("size", [1.85, 4.5, 1.6])
    if len(sz) != 3:
        raise ValueError(f"actor.size must have 3 elements (w, l, h); got {sz!r}")
    class_name = str(d.get("class", "RigidNodes"))
    instance_id = int(d.get("instance_id", 0))

    policy_spec = d.get("policy")
    policy: Optional[ActorPolicy] = None
    if policy_spec is not None:
        ptype = str(policy_spec.get("type", "")).lower()
        if ptype == "log_replay":
            key = f"{class_name}/{instance_id}"
            traj = traffic_log.get(key)
            if traj is None:
                # No baked trajectory found -> skip this actor (better than crashing)
                return None
            policy = LogReplayPolicy(traj, frame_dt=frame_dt)
        else:
            policy = policy_from_dict(policy_spec, default_dt=sim_dt)
    return Actor(
        actor_id=int(d.get("actor_id", -1)),
        class_name=class_name,
        instance_id=instance_id,
        size=tuple(float(v) for v in sz),   # type: ignore[arg-type]
        policy=policy,
        state=_state_from_dict(d.get("state") or {}),
        asset_path=d.get("asset_path"),
        label=str(d.get("label", "vehicle")),
        enabled=bool(d.get("enabled", True)),
    )


def scene_from_scenario_actors(actors: Iterable[Dict[str, Any]],
                               traffic_log: Optional[str | Path] = None,
                               sim_dt: float = 0.1,
                               frame_dt: float = 0.1,
                               ) -> Scene:
    """Build a :class:`Scene` from a YAML ``actors`` list.

    ``traffic_log`` (optional) is the .npz path produced by
    ``tools/extract_traffic_from_ckpt.py`` and supplies LogReplay
    trajectories for actors whose ``policy.type == "log_replay"``.
    ``sim_dt`` is the closed-loop step used by stateful policies (IDM) and
    ``frame_dt`` the time between two recorded DriveStudio frames.
    """
    tl = _load_traffic_log(traffic_log)
    sc = Scene()
    for spec in (actors or []):
        a = _actor_from_dict(spec, tl, sim_dt=sim_dt, frame_dt=frame_dt)
        if a is not None:
            sc.add(a)
    return sc
