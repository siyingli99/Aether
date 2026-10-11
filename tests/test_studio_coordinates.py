import math

import numpy as np

from aether.studio.coordinates import (
    bicycle_to_road_pose,
    e2e_plan_to_road,
    nuscenes_ds_to_road_pose,
    road_plan_to_e2e,
    road_pose_to_bicycle,
    road_pose_to_nuscenes_ds,
)
from aether.studio.schemas import RoadPose


def test_nuscenes_roundtrip() -> None:
    pose = RoadPose(x=12.5, y=3.0, z=0.2, yaw=0.4)
    actual = nuscenes_ds_to_road_pose(road_pose_to_nuscenes_ds(pose))
    assert np.allclose([actual.x, actual.y, actual.z, actual.yaw], [12.5, 3.0, 0.2, 0.4])


def test_left_turn_stays_left_at_bicycle_boundary() -> None:
    pose = RoadPose(x=5.0, y=2.5, yaw=math.radians(25.0))
    start_ab, start_euler = road_pose_to_bicycle(pose)
    assert start_ab == [-2.5, 5.0]
    assert start_euler[1] == -25.0
    actual = bicycle_to_road_pose(start_ab, start_euler)
    assert np.allclose([actual.x, actual.y, actual.yaw], [pose.x, pose.y, pose.yaw])


def test_e2e_plan_right_positive_to_road_left_positive() -> None:
    e2e = np.asarray([[2.0, 8.0], [-1.0, 10.0]], dtype=np.float32)
    road = e2e_plan_to_road(e2e)
    assert np.allclose(road, [[8.0, -2.0], [10.0, 1.0]])
    assert np.allclose(road_plan_to_e2e(road), e2e)
