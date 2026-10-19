"""GPU-free checks of the ground model, obstacle filtering and violation events."""
import math
import unittest

import numpy as np

from aether.geometry.boxes import Box2D
from aether.scene.ground import GroundModel, ground_layer
from aether.sim.collision import CollisionChecker, CollisionConfig
from aether.vehicle import BicycleModel, VehicleState, load_profile


def _road(rng, slope=0.0, floaters=0.0):
    xs, ys = np.meshgrid(np.linspace(-5, 60, 261), np.linspace(-8, 8, 65))
    z = slope * xs.ravel() + rng.normal(0, 0.02, xs.size)
    road = np.column_stack([xs.ravel(), ys.ravel(), z])
    if floaters:
        sub = road[rng.random(len(road)) < floaters].copy()
        sub[:, 2] -= rng.uniform(0.3, 0.8, len(sub))          # floaters under the road
        road = np.concatenate([road, sub])
    return road


def _route(slope=0.0):
    x = np.linspace(0, 50, 101)
    return np.column_stack([x, np.zeros_like(x), slope * x])


class GroundModelTest(unittest.TestCase):
    def test_dense_layer_ignores_floaters_below(self):
        heights = np.sort(np.concatenate([np.full(200, 1.0), np.linspace(0.2, 0.6, 30)]))
        self.assertAlmostEqual(ground_layer(heights), 1.0, places=3)

    def test_follows_a_slope_and_resists_floaters(self):
        rng = np.random.default_rng(0)
        ground = GroundModel.build(_road(rng, slope=0.05, floaters=0.15), _route(0.05))
        for x in (0.0, 20.0, 45.0):
            for y in (0.0, 5.0):
                self.assertAlmostEqual(ground.height(x, y), 0.05 * x, delta=0.08)

    def test_attitude_on_slope(self):
        rng = np.random.default_rng(1)
        ground = GroundModel.build(_road(rng, slope=0.05), _route(0.05))
        z, pitch, roll = ground.attitude(10.0, 0.0, 0.0, wheelbase=2.6, track=1.6)
        self.assertAlmostEqual(z, 0.5, delta=0.05)
        self.assertAlmostEqual(pitch, -math.atan(0.05), delta=0.02)   # nose up on an uphill
        self.assertAlmostEqual(roll, 0.0, delta=0.02)


class CollisionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        road = _road(rng)
        # A dense 1 m wall at y = 4 m, a dense clump on the logged route at x = 30 m
        # and isolated floaters at y = -3 m.
        wall = np.column_stack([rng.uniform(0, 50, 4000), 4.0 + rng.normal(0, 0.05, 4000),
                                rng.uniform(0.3, 1.3, 4000)])
        clump = np.column_stack([30 + rng.normal(0, 0.1, 400), rng.normal(0, 0.1, 400),
                                 rng.uniform(0.4, 0.9, 400)])
        floaters = np.column_stack([rng.uniform(0, 50, 60), rng.uniform(-3.5, -2.5, 60),
                                    rng.uniform(0.4, 1.4, 60)])
        self.n_wall = len(wall)
        splats = np.concatenate([road, wall, clump, floaters])
        self.ground = GroundModel.build(road, _route())
        self.profile = load_profile("sedan")
        logged = [Box2D(x, 0.0, 0.0, self.profile.length, self.profile.width)
                  for x in np.arange(0, 50, 0.5)]
        self.checker = CollisionChecker(splats, self.ground, self.profile, None, logged,
                                        CollisionConfig())

    def box(self, x, y, yaw=0.0):
        return Box2D(x, y, yaw, self.profile.length, self.profile.width)

    def test_obstacles_keep_wall_drop_floaters_and_corridor(self):
        self.assertGreater(self.checker.static_count(self.box(20, 3.5)), 50)
        self.assertEqual(self.checker.static_count(self.box(30, 0.0)), 0)    # carved corridor
        self.assertLess(self.checker.static_count(self.box(20, -3.0)), 5)    # floaters

    def test_static_and_vehicle_events(self):
        events = self.checker.check(1.0, 4, self.box(20, 0), self.box(20, 3.5), 5.0, {}, {})
        self.assertIn("static_collision", [e.type for e in events])
        other = {"npc": (self.box(25, 0), 0.0)}
        events = self.checker.check(1.0, 4, self.box(18, 0), self.box(22, 0), 8.0,
                                    {"npc": (self.box(25, 0), 0.0)}, other)
        hits = [e for e in events if e.type == "vehicle_collision"]
        self.assertEqual(len(hits), 1)
        self.assertTrue(hits[0].detail["at_fault"])
        self.assertEqual(hits[0].detail["side"], "front")

    def test_rear_end_by_other_is_not_at_fault(self):
        events = self.checker.check(1.0, 4, self.box(20, 0), self.box(20, 0), 0.0,
                                    {"npc": (self.box(12, 0), 6.0)}, {"npc": (self.box(16, 0), 6.0)})
        hit = [e for e in events if e.type == "vehicle_collision"][0]
        self.assertFalse(hit.detail["at_fault"])


class BicycleTest(unittest.TestCase):
    def test_braking_stops_instead_of_reversing(self):
        model = BicycleModel(load_profile("sedan"), 0.25)
        state = VehicleState(v=1.0)
        for _ in range(10):
            state = model.step(state, -3.0, 0.0)
        self.assertEqual(state.v, 0.0)
        self.assertGreaterEqual(state.x, 0.0)

    def test_allow_reverse(self):
        model = BicycleModel(load_profile("sedan"), 0.25, allow_reverse=True)
        state = VehicleState(v=1.0)
        for _ in range(10):
            state = model.step(state, -3.0, 0.0)
        self.assertLess(state.v, 0.0)

    def test_left_steer_turns_left(self):
        model = BicycleModel(load_profile("sedan"), 0.25)
        state = VehicleState(v=5.0)
        for _ in range(8):
            state = model.step(state, 0.0, 0.5)
        self.assertGreater(state.yaw, 0.0)
        self.assertGreater(state.y, 0.0)

    def test_lateral_acceleration_is_limited_at_speed(self):
        profile = load_profile("chrysler_pacifica")
        model = BicycleModel(profile, 0.25)
        state = VehicleState(v=16.0)
        for _ in range(8):                         # full right steer for 2 s
            state = model.step(state, 0.0, -0.7)
        lat = state.v * abs(model.yaw_rate(state))
        self.assertLessEqual(lat, profile.max_lat_accel + 1e-6)
        slow = VehicleState(v=2.0)
        for _ in range(8):                         # low speed: full lock allowed
            slow = model.step(slow, 0.0, -0.7)
        self.assertAlmostEqual(slow.steer, -profile.max_steer, places=6)


if __name__ == "__main__":
    unittest.main()
