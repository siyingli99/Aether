"""Geometry checks for AetherDrivingEnv that run without a GPU or checkpoint.

The env is assembled with ``object.__new__`` and only the attributes the
methods under test read, so these tests pin down the coordinate conventions
(ground under the ego box, actor heading, frame time base) in isolation.
"""
import math
import unittest
from types import SimpleNamespace

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from aether.envs.driving_env import AetherDrivingEnv

# nuScenes DriveStudio world: x=right, y=down, z=forward.
ROAD_TO_DS = np.asarray([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def _nuscenes_env(points=None, legacy=False, ground_ds_y=2.0):
    env = object.__new__(AetherDrivingEnv)
    env.legacy_compat = legacy
    env.data_frame_dt = 0.1
    env._ds_dataset = "nuscenes"
    env._ds_up_axis, env._ds_up_sign = 1, -1.0
    env._ds_lat_axis, env._ds_fwd_axis = 0, 2
    env.ground_query_radius = 4.0
    env.ground_quantile = 0.10
    env.ground_cell_size = 2.0
    env.term_follow_ground = False
    env._ground_ref = ground_ds_y
    env._ground_cache = {}
    env.whl = np.asarray([1.85, 1.5, 4.5], dtype=np.float32)
    env.ego = SimpleNamespace(vab=np.zeros(2, np.float32), vr=np.zeros(3, np.float32),
                              dt=0.25, velo=0.0)
    if points is None:
        # Flat road 2 m below the front camera (DS_y = +2).
        xs, zs = np.meshgrid(np.linspace(-10, 10, 41), np.linspace(-10, 30, 81))
        points = np.stack([xs.ravel(), np.full(xs.size, ground_ds_y), zs.ravel()], axis=1)
    env._bg_points_np = points.astype(np.float32)
    env._bg_bev_tree_all = cKDTree(env._bg_points_np[:, [0, 2]])
    env.timestamp = 0.0
    env._max_frame = 195
    env.bg_min_neighbors, env.bg_neighbor_radius = 8, 0.3
    env.carve_logged_route = True
    env._logged_bev = np.stack([np.zeros(31), np.linspace(0, 30, 31)], axis=1)
    return env


class GroundAndEgoBoxTest(unittest.TestCase):
    def test_ego_box_sits_on_local_ground(self):
        env = _nuscenes_env()
        corners = env._ego_box_8corners_ds()
        # Base corners at the road (DS_y = 2), roof 1.5 m above (DS_y = 0.5).
        np.testing.assert_allclose(corners[[0, 1, 2, 3], 1], 2.0, atol=1e-5)
        np.testing.assert_allclose(corners[[4, 5, 6, 7], 1], 0.5, atol=1e-5)

    def test_legacy_box_stays_at_camera_height(self):
        env = _nuscenes_env(legacy=True)
        corners = env._ego_box_8corners_ds()
        np.testing.assert_allclose(corners[[0, 1, 2, 3], 1], 0.0, atol=1e-5)

    def test_tree_ground_query_matches_brute_force(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform([-20, 0.5, -20], [20, 3.0, 40], size=(20000, 3)).astype(np.float32)
        env = _nuscenes_env(points=pts)
        brute = _nuscenes_env(points=pts)
        brute._bg_bev_tree_all = None
        for query in [(0.0, 0.0), (3.2, 11.5), (-7.0, 25.0), (100.0, 100.0)]:
            self.assertAlmostEqual(env._ground_height_at(*query),
                                   brute._ground_height_at(*query), places=5)

    def test_obstacle_band_follows_slope(self):
        # Road rising 1 m every 10 m forward, plus a 1 m tall kerb wall at
        # x = 4 m.  A single global ground plane misclassifies the far wall.
        xs, zs = np.meshgrid(np.linspace(-6, 6, 61), np.linspace(0, 60, 301))
        road_y = 2.0 - 0.1 * zs.ravel()
        road = np.stack([xs.ravel(), road_y, zs.ravel()], axis=1)
        wall_z = np.linspace(0, 60, 301)
        wall = np.stack([np.full_like(wall_z, 4.0), 2.0 - 0.1 * wall_z - 1.0, wall_z], axis=1)
        pts = np.concatenate([road, wall]).astype(np.float32)
        env = _nuscenes_env(points=pts)
        above = env._height_above_local_ground(pts)
        # Within one 2 m cell the road rises up to 0.2 m.
        np.testing.assert_allclose(above[: len(road)], 0.0, atol=0.25)
        np.testing.assert_allclose(above[len(road):], 1.0, atol=0.25)
        # The global-plane estimate puts the far end of the wall ~6 m up,
        # outside the 0.3-2.5 m obstacle band.
        global_ground = np.quantile(pts[:, 1], 0.90)
        self.assertGreater(float(global_ground - pts[-1, 1]), 5.0)


    def test_floaters_below_road_do_not_lift_the_road_into_the_band(self):
        rng = np.random.default_rng(1)
        xs, zs = np.meshgrid(np.linspace(-6, 6, 61), np.linspace(0, 20, 101))
        road = np.stack([xs.ravel(), np.full(xs.size, 2.0), zs.ravel()], axis=1)
        road[:, 1] += rng.normal(0.0, 0.02, len(road))
        # 15 % of splats float 0.3-0.8 m under the surface (DS_y is down).
        floaters = road[rng.random(len(road)) < 0.15].copy()
        floaters[:, 1] += rng.uniform(0.3, 0.8, len(floaters))
        pts = np.concatenate([road, floaters]).astype(np.float32)
        env = _nuscenes_env(points=pts)
        above = env._height_above_local_ground(pts)
        road_above = above[: len(road)]
        self.assertLess(float(np.mean(road_above > 0.3)), 0.001)
        # The ego box base uses the same robust ground.
        env.ego.vab[:] = [0.0, 10.0]
        corners = env._ego_box_8corners_ds()
        self.assertAlmostEqual(float(corners[0, 1]), 2.0, delta=0.05)


class ObstacleFilterTest(unittest.TestCase):
    def test_isolated_floaters_and_logged_corridor_are_removed(self):
        rng = np.random.default_rng(2)
        # Dense 1 m wall 4 m to the right of the logged route ...
        wall = np.stack([4.0 + rng.normal(0, 0.05, 3000),
                         2.0 - rng.uniform(0.3, 1.3, 3000),
                         rng.uniform(0, 30, 3000)], axis=1)
        # ... a dense clump (debris) sitting in the logged corridor ...
        clump = np.stack([rng.normal(0.0, 0.1, 400), 2.0 - rng.uniform(0.4, 0.9, 400),
                          rng.normal(15.0, 0.1, 400)], axis=1)
        # ... and isolated floaters 3 m left of the route.
        floaters = np.stack([rng.uniform(-3.5, -2.5, 60), 2.0 - rng.uniform(0.4, 1.4, 60),
                             rng.uniform(0, 30, 60)], axis=1)
        pts = np.concatenate([wall, clump, floaters]).astype(np.float32)
        env = _nuscenes_env(points=pts)
        above = 2.0 - pts[:, 1]
        band = (above > 0.3) & (above < 2.5)
        keep = env._filter_obstacles(pts, above, band)
        self.assertGreater(keep[: len(wall)].mean(), 0.95)
        self.assertFalse(keep[len(wall): len(wall) + len(clump)].any())
        self.assertLess(keep[len(wall) + len(clump):].mean(), 0.05)


class ActorBoxTest(unittest.TestCase):
    def _env_with_actor(self, yaw, legacy=False):
        env = _nuscenes_env(legacy=legacy)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = ROAD_TO_DS @ Rotation.from_euler("z", yaw).as_matrix()
        pose[:3, 3] = ROAD_TO_DS @ np.asarray([12.0, -3.0, 0.75])
        env.actors = SimpleNamespace(pose_at=lambda frame: [
            {"class": "RigidNodes", "instance_id": 0, "size": [1.9, 4.0, 1.5], "c2w": pose}
        ])
        return env

    def test_reconstructed_actor_box_keeps_heading(self):
        for yaw in (0.0, 0.3, -1.2, 2.9):
            box = self._env_with_actor(yaw)._objs_list_envframe()[0]
            self.assertAlmostEqual(box[0], 12.0, places=4)
            self.assertAlmostEqual(box[1], -3.0, places=4)
            self.assertAlmostEqual(math.remainder(box[6] - yaw, 2 * math.pi), 0.0, places=5)

    def test_legacy_actor_box_reproduces_old_rotation(self):
        box = self._env_with_actor(0.0, legacy=True)._objs_list_envframe()[0]
        self.assertAlmostEqual(abs(box[6]), math.pi / 2, places=4)


class TimeBaseTest(unittest.TestCase):
    def test_frame_index_uses_dataset_rate(self):
        env = _nuscenes_env()
        env.timestamp = 0.75          # three 0.25 s steps
        self.assertEqual(env.frame_idx, 8)   # round(7.5) at 10 Hz
        env.timestamp = 100.0
        self.assertEqual(env.data_frame, 195)

    def test_legacy_frame_index_counts_steps(self):
        env = _nuscenes_env(legacy=True)
        env.timestamp = 0.75
        self.assertEqual(env.frame_idx, 3)


if __name__ == "__main__":
    unittest.main()
