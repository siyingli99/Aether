"""DatasetAdapter abstraction.

A *DatasetAdapter* turns a raw recording (sequence of camera frames + ego
poses + per-frame actor boxes) into the AETHER scenario YAML schema consumed
by ``configs/scenarios/*.yaml`` and ``aether.sim.scene_factory``.

The two concrete adapters shipped here cover the data we already have on
disk:

- ``NuScenesAdapter`` — reads the drivestudio-converted nuScenes layout
  (``Datasets/NuScenes/nusc_kitti/<scene>/``) used by the existing
  ``configs/scenarios/nuscenes_*.yaml`` checkpoints.
- ``PandaSetAdapter`` — reads the drivestudio-converted PandaSet layout
  (``Datasets/PandaSet/processed/<seq>/``).

Other datasets (Waymo, KITTI-360, ZOD, ...) are abstract: ``register`` them
with a custom subclass.
"""
from __future__ import annotations

import abc
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class FrameRecord:
    timestamp: float
    ego_pose: List[float]              # [x, y, z, qx, qy, qz, qw] world frame
    cam_paths: Dict[str, str] = field(default_factory=dict)
    actors: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DatasetRecord:
    scene_id: str
    fps: float
    frames: List[FrameRecord]
    extras: Dict[str, Any] = field(default_factory=dict)


class DatasetAdapter(abc.ABC):
    """Abstract dataset reader.

    Subclasses must implement :meth:`load` and :meth:`to_scenario`.
    """

    name: str = "abstract"

    @abc.abstractmethod
    def load(self, root: str, scene_id: str) -> DatasetRecord: ...

    def to_scenario(self, rec: DatasetRecord, *,
                    checkpoint: Optional[str] = None,
                    actor_backend: str = "logreplay",
                    ) -> Dict[str, Any]:
        """Default conversion: empty actors, ego start at first frame."""
        if not rec.frames:
            raise ValueError("empty record")
        ego0 = rec.frames[0].ego_pose
        scenario = {
            "scene_name": f"{self.name}_{rec.scene_id}_aether",
            "mode": "easy",
            "checkpoint": checkpoint or
                f"${{aether:data/checkpoints/{self.name}_outputs}}/"
                f"scene_recon_full/{rec.scene_id}/checkpoint_final.pth",
            "start_ab": [float(ego0[0]), float(ego0[1])],
            "start_euler": [0.0, 0.0, 0.0],
            "start_velo": 1.0,
            "start_steer": 0.0,
            "ego_whl": [1.85, 1.5, 4.5],
            "actors": [],
            "actor_backend": actor_backend,
            "termination": {
                "fg_collision": True,
                "fg_collision_min_overlap": 0.10,
                "max_steps": max(50, len(rec.frames)),
                "bg_collision": True,
                "bg_threshold": 50,
            },
        }
        return scenario


# ---------------------------------------------------------------------------
# NuScenes (drivestudio-processed) -- real implementation
# ---------------------------------------------------------------------------

class NuScenesAdapter(DatasetAdapter):
    name = "nuscenes"

    def load(self, root: str, scene_id: str) -> DatasetRecord:
        # Expected layout: <root>/<scene_idx>/{ego_pose, intrinsics, images/, ...}
        scene_dir = Path(root) / scene_id
        if not scene_dir.exists():
            # try drivestudio conventional name
            alt = Path(root) / f"{int(scene_id):03d}"
            if alt.exists():
                scene_dir = alt
        if not scene_dir.exists():
            raise FileNotFoundError(f"nuscenes scene dir not found: {scene_dir}")

        ego_pose_dir = scene_dir / "ego_pose"
        frames: List[FrameRecord] = []
        if ego_pose_dir.exists():
            files = sorted(p for p in ego_pose_dir.iterdir() if p.suffix == ".txt")
            for i, f in enumerate(files):
                try:
                    arr = [float(x) for x in f.read_text().split()]
                except Exception:
                    continue
                # drivestudio dumps 4x4 transform; keep translation + identity quat
                if len(arr) >= 12:
                    t = arr[3], arr[7], arr[11]
                else:
                    t = (arr[0], arr[1], arr[2] if len(arr) > 2 else 0.0)
                frames.append(FrameRecord(
                    timestamp=i / 10.0,
                    ego_pose=[t[0], t[1], t[2], 0.0, 0.0, 0.0, 1.0],
                ))
        return DatasetRecord(scene_id=str(scene_id), fps=10.0, frames=frames)


# ---------------------------------------------------------------------------
# PandaSet (drivestudio-processed) -- real implementation
# ---------------------------------------------------------------------------

class PandaSetAdapter(DatasetAdapter):
    name = "pandaset"

    def load(self, root: str, scene_id: str) -> DatasetRecord:
        seq_dir = Path(root) / scene_id
        if not seq_dir.exists():
            raise FileNotFoundError(f"pandaset seq not found: {seq_dir}")
        ego_dir = seq_dir / "ego_pose"
        frames: List[FrameRecord] = []
        if ego_dir.exists():
            files = sorted(p for p in ego_dir.iterdir() if p.suffix == ".json")
            for i, f in enumerate(files):
                try:
                    j = json.loads(f.read_text())
                except Exception:
                    continue
                pos = j.get("position", {})
                hd = j.get("heading", {})
                ego = [
                    float(pos.get("x", 0.0)),
                    float(pos.get("y", 0.0)),
                    float(pos.get("z", 0.0)),
                    float(hd.get("x", 0.0)),
                    float(hd.get("y", 0.0)),
                    float(hd.get("z", 0.0)),
                    float(hd.get("w", 1.0)),
                ]
                frames.append(FrameRecord(timestamp=i / 10.0, ego_pose=ego))
        return DatasetRecord(scene_id=str(scene_id), fps=10.0, frames=frames)


# ---------------------------------------------------------------------------
# Abstract placeholders for the rest of the family
# ---------------------------------------------------------------------------

class WaymoAdapter(DatasetAdapter):
    """Abstract — implement when Waymo TFRecord support lands."""

    name = "waymo"

    def load(self, root: str, scene_id: str) -> DatasetRecord:
        raise NotImplementedError("WaymoAdapter is abstract; subclass to implement")


class KittiAdapter(DatasetAdapter):
    name = "kitti"

    def load(self, root: str, scene_id: str) -> DatasetRecord:
        raise NotImplementedError("KittiAdapter is abstract; subclass to implement")


REGISTRY: Dict[str, type] = {
    "nuscenes": NuScenesAdapter,
    "pandaset": PandaSetAdapter,
    "waymo": WaymoAdapter,
    "kitti": KittiAdapter,
}


def make(name: str, **kw) -> DatasetAdapter:
    if name not in REGISTRY:
        raise KeyError(f"unknown dataset '{name}', have {list(REGISTRY)}")
    return REGISTRY[name](**kw)
