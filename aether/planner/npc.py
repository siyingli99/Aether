"""Replay-based NPC actor planner.

Reads `scenario.actors`: a list of {class, instance_id, trajectory:[{frame, c2w}]}
and exposes `pose_at(frame)` lookups.  Each entry's c2w is in DriveStudio
world frame (z-up), same convention as the loaded checkpoint, so the renderer
can install it directly via `set_actor_pose`.

In v0.1 we only support pure replay (no IDM / unicycle).  Adding scripted
controllers later means swapping the entry's trajectory generator.
"""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np


class ActorReplay:
    def __init__(self, actors_cfg):
        # actors_cfg: list[dict]
        self.entries: List[dict] = []
        for a in (actors_cfg or []):
            traj = {}
            for t in a.get("trajectory", []):
                traj[int(t["frame"])] = np.asarray(t["c2w"], dtype=np.float32)
            self.entries.append({
                "class":       a["class"],
                "instance_id": int(a["instance_id"]),
                "size":        list(a.get("size", [2.0, 4.0, 1.5])),
                "trajectory":  traj,
            })

    def pose_at(self, frame_idx: int) -> List[dict]:
        """Return [{class, instance_id, size, c2w}] for actors active at frame."""
        out: List[dict] = []
        for e in self.entries:
            c2w = e["trajectory"].get(int(frame_idx))
            if c2w is None:
                continue
            out.append({
                "class":       e["class"],
                "instance_id": e["instance_id"],
                "size":        e["size"],
                "c2w":         c2w,
            })
        return out
