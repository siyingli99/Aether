"""Explicit transforms between traffic road-local and DriveStudio frames."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation as SCR


def _matrix4(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation must be right-handed")
    return matrix


@dataclass(frozen=True)
class FrameTransform:
    """Map homogeneous poses from traffic road-local into DS world."""

    traffic_to_ds: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "traffic_to_ds",
            _matrix4(self.traffic_to_ds, "traffic_to_ds"),
        )

    @property
    def ds_to_traffic(self) -> np.ndarray:
        return np.linalg.inv(self.traffic_to_ds)

    def pose_to_ds(self, pose_traffic: np.ndarray) -> np.ndarray:
        return (
            self.traffic_to_ds @ _matrix4(pose_traffic, "traffic_pose")
        ).astype(np.float32)

    def point_to_ds(self, point_traffic: np.ndarray) -> np.ndarray:
        point = np.asarray(point_traffic, dtype=np.float64).reshape(3)
        return (
            self.traffic_to_ds[:3, :3] @ point + self.traffic_to_ds[:3, 3]
        ).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {"traffic_to_ds": self.traffic_to_ds.tolist()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FrameTransform":
        matrix = value.get("traffic_to_ds")
        if matrix is None:
            raise KeyError("traffic_to_ds")
        return cls(traffic_to_ds=matrix)

    @classmethod
    def load(cls, path: str | Path) -> "FrameTransform":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def vehicle_pose(x: float, y: float, yaw: float, z: float = 0.0) -> np.ndarray:
    """Object pose in traffic-local axes (x forward, y left, z up)."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = SCR.from_euler("z", float(yaw)).as_matrix()
    pose[:3, 3] = [float(x), float(y), float(z)]
    return pose


def nuscenes_scene_local_to_ds(ground_ds_y: float = 0.0) -> FrameTransform:
    """nuScenes road-local axes to FRONT-camera-anchored DriveStudio axes.

    Traffic local: x=forward, y=left, z=up.
    nuScenes DS: x=right, y=down, z=forward.
    Both origins are the first FRONT-camera/ego frame for this MVP.
    """
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
        dtype=np.float64,
    )
    matrix[1, 3] = float(ground_ds_y)
    return FrameTransform(matrix)
