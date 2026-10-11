"""Discover Studio scenes, topology, reconstructed actors and assets."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else REPO_ROOT / value).resolve()


class StudioCatalog:
    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = _resolve(config_path or "configs/studio/catalog.yaml")
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.data = yaml.safe_load(handle) or {}
        self._scenes = {entry["id"]: entry for entry in self.data.get("scenes", [])}
        self._assets = {entry["id"]: entry for entry in self.data.get("assets", [])}
        self._controllers = {entry["id"]: entry for entry in self.data.get("controllers", [])}
        self._adapters = {entry["id"]: entry for entry in self.data.get("adapters", [])}

    def scene(self, scene_id: str) -> dict[str, Any]:
        if scene_id not in self._scenes:
            raise KeyError(f"unknown scene: {scene_id}")
        entry = dict(self._scenes[scene_id])
        for key in ("scenario", "limsim_scenario", "camera", "traffic_log", "map", "net", "route", "sumocfg"):
            if entry.get(key):
                entry[key] = str(_resolve(entry[key]))
        entry["has_map"] = bool(entry.get("map") and Path(entry["map"]).is_file())
        entry["has_limsim"] = bool(
            entry.get("net") and Path(entry["net"]).is_file()
            and entry.get("route") and Path(entry["route"]).is_file()
        )
        return entry

    def asset(self, asset_id: str) -> dict[str, Any]:
        if asset_id not in self._assets:
            raise KeyError(f"unknown asset: {asset_id}")
        entry = dict(self._assets[asset_id])
        entry["path"] = str(_resolve(entry["path"]))
        return entry

    def controller(self, controller_id: str) -> dict[str, Any]:
        if controller_id not in self._controllers:
            raise KeyError(f"unknown controller: {controller_id}")
        return dict(self._controllers[controller_id])

    @staticmethod
    def _exists(entry: dict[str, Any], fields: tuple[str, ...]) -> bool:
        return all(not entry.get(field) or Path(entry[field]).exists() for field in fields)

    def list_scenes(self) -> list[dict[str, Any]]:
        result = []
        for scene_id in self._scenes:
            entry = self.scene(scene_id)
            required = ("scenario", "camera", "traffic_log")
            entry["available"] = self._exists(entry, required)
            entry["has_map"] = bool(entry.get("map") and Path(entry["map"]).is_file())
            entry["has_limsim"] = bool(
                entry.get("net") and Path(entry["net"]).is_file()
                and entry.get("route") and Path(entry["route"]).is_file()
            )
            result.append(entry)
        return result

    def list_assets(self) -> list[dict[str, Any]]:
        result = []
        for asset_id in self._assets:
            entry = self.asset(asset_id)
            root = Path(entry["path"])
            metadata_path = root / "metadata.json"
            size_path = root / "wlh.json"
            metadata: dict[str, Any] = {}
            if metadata_path.is_file():
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            elif size_path.is_file():
                width, length, height = json.loads(size_path.read_text(encoding="utf-8"))
                metadata["size_lwh"] = [length, width, height]
            entry["available"] = (root / "asset.pt").is_file() or (root / "gs.pth").is_file()
            entry["metadata"] = metadata
            result.append(entry)
        return result

    def list_controllers(self) -> list[dict[str, Any]]:
        return [dict(entry) for entry in self._controllers.values()]

    def list_adapters(self) -> list[dict[str, Any]]:
        result = []
        for entry in self._adapters.values():
            item = dict(entry)
            item["config"] = str(_resolve(item["config"]))
            item["available"] = Path(item["config"]).is_file()
            result.append(item)
        return result

    def topology(self, scene_id: str) -> dict[str, Any]:
        scene = self.scene(scene_id)
        map_path = Path(scene["map"])
        if not map_path.is_file():
            return {"lanes": [], "route": [], "bounds": None}
        payload = json.loads(map_path.read_text(encoding="utf-8"))
        lanes = [
            {
                "id": lane.get("token", f"lane-{index}"),
                "kind": lane.get("kind", "lane"),
                "width": float(lane.get("lane_width_m", 3.5)),
                "points": lane.get("centerline", []),
                "incoming": lane.get("incoming", []),
                "outgoing": lane.get("outgoing", []),
            }
            for index, lane in enumerate(payload.get("map_lanes", []))
        ]
        route = payload.get("traffic_link", {}).get("centerline", [])
        points = [point for lane in lanes for point in lane["points"]] + list(route)
        bounds = None
        if points:
            coords = np.asarray(points, dtype=np.float64)
            bounds = {
                "min_x": float(coords[:, 0].min()), "max_x": float(coords[:, 0].max()),
                "min_y": float(coords[:, 1].min()), "max_y": float(coords[:, 1].max()),
            }
        return {
            "axes": payload.get("axes", {"x": "forward", "y": "left", "z": "up"}),
            "lanes": lanes,
            "route": route,
            "bounds": bounds,
            "location": payload.get("location"),
        }

    def actors(self, scene_id: str) -> list[dict[str, Any]]:
        scene = self.scene(scene_id)
        archive = np.load(scene["traffic_log"], allow_pickle=False)
        actors: list[dict[str, Any]] = []
        for key in sorted(name for name in archive.files if name.startswith("c2w/")):
            _, node_class, instance_text = key.split("/", 2)
            instance_id = int(instance_text)
            matrices = archive[key]
            frames = archive[f"frames/{node_class}/{instance_id}"].astype(int)
            size = archive[f"size/{node_class}/{instance_id}"].astype(float).tolist()
            trajectory = []
            for frame, matrix in zip(frames, matrices):
                # nuScenes DS: +z forward, +x right, therefore road y=-DS x.
                x = float(matrix[2, 3])
                y = -float(matrix[0, 3])
                trajectory.append({"frame": int(frame), "x": x, "y": y})
            for index in range(len(trajectory)):
                if len(trajectory) == 1:
                    yaw = 0.0
                else:
                    lo = max(0, index - 1)
                    hi = min(len(trajectory) - 1, index + 1)
                    dx = trajectory[hi]["x"] - trajectory[lo]["x"]
                    dy = trajectory[hi]["y"] - trajectory[lo]["y"]
                    yaw = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-6 else 0.0
                trajectory[index]["yaw"] = yaw
            label = "vehicle" if node_class == "RigidNodes" else (
                "pedestrian" if node_class == "SMPLNodes" else "deformable"
            )
            actors.append({
                "id": f"{node_class}-{instance_id}",
                "source": "scene", "node_class": node_class,
                "instance_id": instance_id, "label": label,
                "size": size, "first_frame": int(frames.min()),
                "last_frame": int(frames.max()), "trajectory": trajectory,
            })
        return actors

    def bootstrap_experiment(self, scene_id: str) -> dict[str, Any]:
        actors = []
        for actor in self.actors(scene_id):
            actors.append({
                "id": actor["id"], "source": "scene", "operation": "keep",
                "node_class": actor["node_class"], "instance_id": actor["instance_id"],
                "label": actor["label"], "size": actor["size"],
                "controller": {"kind": "recorded", "trajectory": [], "params": {}},
            })
        return {
            "schema_version": "aether.experiment.v1", "scene_id": scene_id,
            "name": f"{scene_id} experiment", "ego": {"start": {}, "speed": 1.0},
            "actors": actors, "algorithm": {"adapter": "uniad", "cuda": "0"},
            "runtime": {"max_steps": 39, "seed": 0, "difix": {"enabled": True, "cuda": "0"}},
        }
