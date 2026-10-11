"""Coordinate boundaries for Studio's canonical road-local frame."""
from __future__ import annotations

import math
import numpy as np
from scipy.spatial.transform import Rotation as Rotation

from .schemas import RoadPose


ROAD_TO_NUSCENES_DS = np.asarray(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=np.float64,
)


def wrap_yaw(yaw: float) -> float:
    return (float(yaw) + math.pi) % (2.0 * math.pi) - math.pi


def road_pose_to_nuscenes_ds(pose: RoadPose, *, ground_ds_y: float = 0.0) -> np.ndarray:
    """Convert x-forward/y-left/yaw-left into the nuScenes DriveStudio frame."""
    road = np.eye(4, dtype=np.float64)
    road[:3, :3] = Rotation.from_euler("z", pose.yaw).as_matrix()
    road[:3, 3] = [pose.x, pose.y, pose.z]
    align = np.eye(4, dtype=np.float64)
    align[:3, :3] = ROAD_TO_NUSCENES_DS
    align[1, 3] = float(ground_ds_y)
    return (align @ road).astype(np.float32)


def nuscenes_ds_to_road_pose(matrix: np.ndarray, *, ground_ds_y: float = 0.0) -> RoadPose:
    ds = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    align = np.eye(4, dtype=np.float64)
    align[:3, :3] = ROAD_TO_NUSCENES_DS
    align[1, 3] = float(ground_ds_y)
    road = np.linalg.inv(align) @ ds
    yaw = Rotation.from_matrix(road[:3, :3]).as_euler("ZYX")[0]
    return RoadPose(x=road[0, 3], y=road[1, 3], z=road[2, 3], yaw=wrap_yaw(yaw))


def road_pose_to_bicycle(pose: RoadPose) -> tuple[list[float], list[float]]:
    """Convert Studio pose to BicycleEgo ``start_ab``/``start_euler``.

    BicycleEgo stores lateral-right in ``vab[0]`` and clockwise/right-turn
    yaw in ``vr[1]``. This sign boundary is where the historical left/right
    inversion occurred, so it is intentionally explicit and tested.
    """
    start_ab = [-float(pose.y), float(pose.x)]
    start_euler = [0.0, -math.degrees(float(pose.yaw)), 0.0]
    return start_ab, start_euler


def bicycle_to_road_pose(start_ab: list[float], start_euler: list[float]) -> RoadPose:
    return RoadPose(
        x=float(start_ab[1]),
        y=-float(start_ab[0]),
        yaw=wrap_yaw(-math.radians(float(start_euler[1]))),
    )


def e2e_plan_to_road(plan: np.ndarray) -> np.ndarray:
    """Convert adapter plan ``[lateral-right, forward]`` to ``[x,y]`` road points."""
    points = np.asarray(plan, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("E2E plan must have shape [N, >=2]")
    return np.stack((points[:, 1], -points[:, 0]), axis=1)


def road_plan_to_e2e(plan: np.ndarray) -> np.ndarray:
    points = np.asarray(plan, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("road plan must have shape [N, >=2]")
    return np.stack((-points[:, 1], points[:, 0]), axis=1)
