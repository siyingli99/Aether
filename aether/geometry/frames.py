"""Coordinate frames used by AETHER — the single source of truth.

road frame
    The 2D/2.5D map frame of a scene, shared by the HD map, traffic
    simulators, Studio, trajectories and metrics.  ``x`` points along the
    initial ego heading, ``y`` to the left, ``z`` up; yaw is counter-clockwise.
    The origin is the scene anchor sensor at the first frame (CAM_FRONT for
    nuScenes, the ego pose for PandaSet) projected onto the road surface.

vehicle frame
    Rigidly attached to a vehicle: ``x`` forward, ``y`` left, ``z`` up, with
    the origin at the centre of the rear axle on the ground (the nuScenes ego
    convention).  Sensor mounts and planner outputs are expressed in it.

scene frame
    The reconstruction world of the renderer (DriveStudio).  Its axes depend
    on how the dataset loader anchored the scene; :data:`DATASET_FRAMES` maps
    them to the road frame.  Nothing outside ``aether.render`` and
    ``aether.scene`` should need to know them.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DatasetFrame:
    """How a dataset's scene frame relates to the road frame."""

    name: str
    # Rotation taking road-frame vectors into scene-frame vectors.
    scene_from_road: np.ndarray
    # Index of the scene axis pointing up and its sign (+1 = up, -1 = down).
    up_axis: int
    up_sign: float

    @property
    def road_from_scene(self) -> np.ndarray:
        return self.scene_from_road.T


def _frame(name: str, rows: list[list[float]]) -> DatasetFrame:
    rotation = np.asarray(rows, dtype=np.float64)
    if not np.isclose(np.linalg.det(rotation), 1.0):
        raise ValueError(f"{name}: scene axes must be right-handed")
    up = rotation @ np.array([0.0, 0.0, 1.0])
    axis = int(np.argmax(np.abs(up)))
    return DatasetFrame(name, rotation, axis, float(np.sign(up[axis])))


DATASET_FRAMES: dict[str, DatasetFrame] = {
    # Anchored on the FRONT camera (OpenCV): x right, y down, z forward.
    "nuscenes": _frame("nuscenes", [[0, -1, 0], [0, 0, -1], [1, 0, 0]]),
    # Anchored on the ego pose: x right, y forward, z up.
    "pandaset": _frame("pandaset", [[0, -1, 0], [1, 0, 0], [0, 0, 1]]),
    # Anchored on the ego pose, native FLU axes.
    "nuplan": _frame("nuplan", [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
}


def dataset_frame(name: str) -> DatasetFrame:
    key = str(name).lower()
    if key not in DATASET_FRAMES:
        raise KeyError(f"unknown dataset frame {name!r}; known: {sorted(DATASET_FRAMES)}")
    return DATASET_FRAMES[key]


def wrap_angle(angle):
    """Wrap an angle (scalar or array) to [-pi, pi)."""
    wrapped = (np.asarray(angle, dtype=np.float64) + math.pi) % (2.0 * math.pi) - math.pi
    return float(wrapped) if wrapped.ndim == 0 else wrapped


def yaw_of(rotation: np.ndarray) -> float:
    """CCW heading of a rotation's +x axis in the road (x, y) plane."""
    rotation = np.asarray(rotation)
    return math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))


def rotation_zyx(yaw: float, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def pose2d(x: float, y: float, yaw: float, z: float = 0.0,
           pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """4x4 pose from a planar position and heading (plus optional tilt)."""
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation_zyx(yaw, pitch, roll)
    pose[:3, 3] = (float(x), float(y), float(z))
    return pose


def pose_from_matrix(pose: np.ndarray) -> tuple[float, float, float]:
    """(x, y, yaw) of a 4x4 road-frame pose."""
    pose = np.asarray(pose)
    return float(pose[0, 3]), float(pose[1, 3]), yaw_of(pose[:3, :3])


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:3, :3].T + matrix[:3, 3]
