"""Scene bundles: everything one reconstructed scene needs, in one folder.

::

    data/scenes/<scene-id>/
      scene.yaml            manifest (dataset, frame period, vehicle, sources)
      rig.yaml              camera intrinsics and sensor mounts
      recon/checkpoint.pth  DriveStudio reconstruction (+ recon/config.yaml)
      log.npz               logged anchor-sensor track and frame-0 camera poses
      actors.npz            recorded trajectories of reconstructed actors
      map/hdmap.json        HD map (lanes, drivable area) in the road frame
      map/network.net.xml   traffic network for SUMO/LimSim (optional)
      map/routes.rou.xml    background traffic demand (optional)
      cache/                derived data (ground model), safe to delete

``scene.yaml``::

    id: nuscenes-0383
    dataset: nuscenes            # selects the scene-frame convention
    frame_dt: 0.1                # seconds between reconstruction frames
    vehicle: renault_zoe         # configs/vehicles/<name>.yaml
    anchor: CAM_FRONT            # sensor whose frame-0 pose is the scene origin
    source: {dataset: nuScenes, scene: scene-0383, location: singapore-onenorth}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..geometry.frames import DatasetFrame, dataset_frame
from ..paths import SCENES_ROOT
from .rig import Rig, load_rig


@dataclass(frozen=True)
class SceneLog:
    """Logged sensor poses in the scene frame (anchored at frame 0)."""

    anchor: np.ndarray              # (T, 4, 4) anchor-sensor pose per frame
    cameras_t0: dict[str, np.ndarray]  # camera -> (4, 4) pose at frame 0

    @property
    def num_frames(self) -> int:
        return int(self.anchor.shape[0])


# Bundle ids corrected after release: v0.2 demo "nuscenes-0088" is trainval
# index 88, i.e. nuScenes scene-0106.
RENAMED = {"nuscenes-0088": "nuscenes-0106"}


@dataclass(frozen=True)
class SceneBundle:
    root: Path
    manifest: dict[str, Any] = field(repr=False)

    # -------------------------------------------------------------- loading
    @classmethod
    def load(cls, scene: str | Path) -> "SceneBundle":
        scene = RENAMED.get(str(scene), scene)
        path = Path(str(scene)).expanduser()
        if not (path / "scene.yaml").is_file():
            path = SCENES_ROOT / str(scene)
        manifest_path = path / "scene.yaml"
        if not manifest_path.is_file():
            known = ", ".join(list_scenes()) or "none"
            raise FileNotFoundError(f"scene {scene!r} not found (looked in {SCENES_ROOT}); known: {known}")
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        for key in ("id", "dataset", "frame_dt"):
            if key not in manifest:
                raise ValueError(f"{manifest_path}: missing '{key}'")
        return cls(path.resolve(), manifest)

    # ----------------------------------------------------------- properties
    @property
    def id(self) -> str:
        return str(self.manifest["id"])

    @property
    def dataset(self) -> str:
        return str(self.manifest["dataset"]).lower()

    @property
    def frame(self) -> DatasetFrame:
        return dataset_frame(self.dataset)

    @property
    def frame_dt(self) -> float:
        return float(self.manifest["frame_dt"])

    @property
    def vehicle(self) -> str:
        return str(self.manifest.get("vehicle", "sedan"))

    @property
    def anchor_sensor(self) -> str:
        return str(self.manifest.get("anchor", "CAM_FRONT"))

    @property
    def anchor_mount(self) -> np.ndarray:
        """vehicle_from_anchor: from the rig, or ``anchor_mount`` in scene.yaml."""
        if "anchor_mount" in self.manifest:
            mount = np.asarray(self.manifest["anchor_mount"], dtype=np.float64)
            if self.anchor_sensor == "EGO" and "anchor_axes" not in self.manifest:
                # Bundles imported before 2026-10-19 stored an identity rotation,
                # i.e. treated PandaSet's right-forward-up ego frame as
                # forward-left-up (the vehicle then drove sideways).
                mount = mount.copy()
                mount[:3, :3] = self.frame.road_from_scene
            return mount
        if self.anchor_sensor in self.rig.cameras:
            return self.rig.cameras[self.anchor_sensor].mount
        return self.rig.lidar_mount(self.anchor_sensor)

    @property
    def source(self) -> dict[str, Any]:
        return dict(self.manifest.get("source") or {})

    def path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    @property
    def checkpoint(self) -> Path:
        return self.path("recon", "checkpoint.pth")

    @property
    def hdmap_path(self) -> Path | None:
        path = self.path("map", "hdmap.json")
        return path if path.is_file() else None

    @property
    def network(self) -> tuple[Path, Path] | None:
        net, routes = self.path("map", "network.net.xml"), self.path("map", "routes.rou.xml")
        return (net, routes) if net.is_file() and routes.is_file() else None

    @property
    def actors_path(self) -> Path | None:
        path = self.path("actors.npz")
        return path if path.is_file() else None

    @property
    def cache_dir(self) -> Path:
        path = self.path("cache")
        path.mkdir(exist_ok=True)
        return path

    @cached_property
    def rig(self) -> Rig:
        return load_rig(self.path("rig.yaml"))

    @cached_property
    def log(self) -> SceneLog:
        data = np.load(self.path("log.npz"))
        names = [str(n) for n in data["camera_names"]]
        return SceneLog(
            anchor=data["anchor"].astype(np.float64),
            cameras_t0={n: data["cameras_t0"][i].astype(np.float64) for i, n in enumerate(names)},
        )

    @cached_property
    def hdmap(self):
        from .hdmap import HDMap
        return HDMap.load(self.hdmap_path) if self.hdmap_path else None

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id, "dataset": self.dataset, "frame_dt": self.frame_dt,
            "vehicle": self.vehicle, "frames": self.log.num_frames,
            "cameras": self.rig.camera_names, "has_map": self.hdmap_path is not None,
            "has_network": self.network is not None,
            "has_actors": self.actors_path is not None, "source": self.source,
            "name": self.manifest.get("name", self.id),
        }


def list_scenes(root: Path | None = None) -> list[str]:
    root = root or SCENES_ROOT
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "scene.yaml").is_file())
