"""Sensor rig: camera intrinsics and sensor mounts in the vehicle frame.

``rig.yaml`` format::

    vehicle_frame: rear_axle_ground_flu     # x fwd, y left, z up
    cameras:
      CAM_FRONT:
        index: 0                  # camera index inside the reconstruction
        size: [800, 450]          # width, height of rendered images
        K: [fx, fy, cx, cy]       # pinhole intrinsics at ``size``
        mount: 4x4                # vehicle_from_camera (OpenCV camera axes)
    lidar:
      LIDAR_TOP:
        mount: 4x4                # vehicle_from_lidar

:func:`load_rig` also reads the older Euler-angle camera YAMLs
(``extrinsics.v2c_rot`` / ``intrinsics.fovx``) used before AETHER 0.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

CAMERA_ORDER = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK")


@dataclass(frozen=True)
class Camera:
    name: str
    index: int
    width: int
    height: int
    K: np.ndarray          # 3x3
    mount: np.ndarray      # vehicle_from_camera 4x4

    def resized(self, width: int, height: int) -> "Camera":
        """Same camera rendered or resampled at another resolution."""
        sx, sy = width / self.width, height / self.height
        K = self.K.copy()
        K[0, :] *= sx
        K[1, :] *= sy
        return Camera(self.name, self.index, int(width), int(height), K, self.mount)

    @property
    def fov(self) -> tuple[float, float]:
        return (2 * math.atan(self.width / (2 * self.K[0, 0])),
                2 * math.atan(self.height / (2 * self.K[1, 1])))


@dataclass(frozen=True)
class Rig:
    cameras: dict[str, Camera]
    lidars: dict[str, np.ndarray] = field(default_factory=dict)  # name -> vehicle_from_lidar

    @property
    def camera_names(self) -> list[str]:
        return list(self.cameras)

    def lidar_mount(self, name: str = "LIDAR_TOP") -> np.ndarray:
        if name in self.lidars:
            return self.lidars[name]
        if self.lidars:
            return next(iter(self.lidars.values()))
        raise KeyError("rig defines no LiDAR")

    def to_dict(self) -> dict:
        return {
            "vehicle_frame": "rear_axle_ground_flu",
            "cameras": {
                cam.name: {
                    "index": cam.index, "size": [cam.width, cam.height],
                    "K": [float(cam.K[0, 0]), float(cam.K[1, 1]), float(cam.K[0, 2]), float(cam.K[1, 2])],
                    "mount": np.round(cam.mount, 9).tolist(),
                } for cam in self.cameras.values()
            },
            "lidar": {name: {"mount": np.round(m, 9).tolist()} for name, m in self.lidars.items()},
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8")


def _euler(rot_deg, trans) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_euler("XYZ", rot_deg, degrees=True).as_matrix()
    matrix[:3, 3] = trans
    return matrix


def _from_legacy(data: dict) -> Rig:
    cameras: dict[str, Camera] = {}
    lidar_mounts = []
    names = [n for n in CAMERA_ORDER if n in data["cams"]] + \
        [n for n in data["cams"] if n not in CAMERA_ORDER]
    for name in names:
        cam = data["cams"][name]
        ext, intr = cam["extrinsics"], cam["intrinsics"]
        v2c = _euler(ext["v2c_rot"], ext["v2c_trans"])
        l2c = _euler(ext.get("l2c_rot", ext["v2c_rot"]), ext.get("l2c_trans", ext["v2c_trans"]))
        mount = np.linalg.inv(v2c)
        lidar_mounts.append(mount @ l2c)
        W, H = int(intr["W"]), int(intr["H"])
        fx = W / (2 * math.tan(math.radians(float(intr["fovx"])) / 2))
        fy = H / (2 * math.tan(math.radians(float(intr["fovy"])) / 2))
        K = np.array([[fx, 0, float(intr["cx"])], [0, fy, float(intr["cy"])], [0, 0, 1.0]])
        index = CAMERA_ORDER.index(name) if name in CAMERA_ORDER else len(cameras)
        index = int(cam.get("index", index))     # camera id inside the reconstruction
        cameras[name] = Camera(name, index, W, H, K, mount)
    lidars = {"LIDAR_TOP": lidar_mounts[0]} if lidar_mounts else {}
    return Rig(cameras, lidars)


def load_rig(path: str | Path) -> Rig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "cams" in data:
        return _from_legacy(data)
    cameras = {}
    for name, cam in data["cameras"].items():
        fx, fy, cx, cy = (float(v) for v in cam["K"])
        W, H = (int(v) for v in cam["size"])
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])
        cameras[name] = Camera(name, int(cam["index"]), W, H, K,
                               np.asarray(cam["mount"], dtype=np.float64))
    lidars = {name: np.asarray(entry["mount"], dtype=np.float64)
              for name, entry in (data.get("lidar") or {}).items()}
    return Rig(cameras, lidars)
