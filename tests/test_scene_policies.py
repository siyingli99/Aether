"""Scene/actor regressions: per-frame idempotence, heading and time base."""
import math
import unittest
from types import SimpleNamespace

import numpy as np

from aether.kinematic.bicycle import BicycleEgo
from aether.sim.actor import (
    Actor, ActorState, Scene, _bev_from_c2w, _c2w_from_bev, set_ds_dataset,
)
from aether.sim.policy import IDMFollowerPolicy, IDMParams, LogReplayPolicy
from aether.sim.scene_factory import scene_from_scenario_actors


class HeadingRoundTripTest(unittest.TestCase):
    def tearDown(self):
        set_ds_dataset("pandaset")

    def test_round_trip_both_datasets(self):
        for dataset in ("nuscenes", "pandaset"):
            set_ds_dataset(dataset)
            for yaw in (0.0, 0.4, -2.0, 3.0):
                c2w = _c2w_from_bev(7.0, -2.5, yaw, z_ds=1.2)
                self.assertAlmostEqual(float(np.linalg.det(c2w[:3, :3])), 1.0, places=5)
                x, y, back = _bev_from_c2w(c2w)
                self.assertAlmostEqual(x, 7.0, places=5)
                self.assertAlmostEqual(y, -2.5, places=5)
                self.assertAlmostEqual(math.remainder(back - yaw, 2 * math.pi), 0.0, places=5)

    def test_nuscenes_heading_is_object_x_axis(self):
        set_ds_dataset("nuscenes")
        # Object +x pointing along DS +z (forward) and +y-up = DS -y.
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.asarray([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float32)
        self.assertAlmostEqual(_bev_from_c2w(c2w)[2], 0.0, places=6)


class SceneIdempotenceTest(unittest.TestCase):
    def setUp(self):
        set_ds_dataset("nuscenes")

    def tearDown(self):
        set_ds_dataset("pandaset")

    def _scene(self, dt=0.25):
        policy = IDMFollowerPolicy(params=IDMParams(v0=10.0, a_max=0.0), dt=dt)
        actor = Actor(actor_id=-1, class_name="RigidNodes", instance_id=0,
                      policy=policy, state=ActorState(x=0.0, y=0.0, yaw=0.0, velo=8.0))
        scene = Scene()
        scene.add(actor)
        scene.ego.update(x=-100.0, y=50.0, yaw=0.0, velo=0.0)
        return scene, actor

    def test_policy_advances_once_per_frame(self):
        scene, actor = self._scene()
        for _ in range(5):          # render + collision + info all query frame 1
            scene.pose_at(1)
        self.assertAlmostEqual(actor.state.x, 8.0 * 0.25, places=6)
        scene.pose_at(2)
        self.assertAlmostEqual(actor.state.x, 2 * 8.0 * 0.25, places=6)

    def test_reset_restores_initial_state(self):
        scene, actor = self._scene()
        scene.pose_at(1)
        scene.pose_at(2)
        scene.reset()
        self.assertEqual(actor.state.x, 0.0)
        scene.pose_at(1)
        self.assertAlmostEqual(actor.state.x, 8.0 * 0.25, places=6)

    def test_factory_uses_simulation_step_for_idm(self):
        scene = scene_from_scenario_actors(
            [{"class": "RigidNodes", "instance_id": 3, "policy": {"type": "idm"}}],
            sim_dt=0.25,
        )
        self.assertEqual(scene.actors[0].policy.dt, 0.25)

    def test_log_replay_speed_uses_frame_period(self):
        traj = {}
        for frame in range(3):
            c2w = _c2w_from_bev(1.5 * frame, 0.0, 0.0)
            traj[frame] = c2w
        policy = LogReplayPolicy(traj, frame_dt=0.1)
        actor = Actor(actor_id=1, class_name="RigidNodes", instance_id=0, policy=policy)
        state = policy.step(actor=actor, frame_idx=2, scene=None)
        self.assertAlmostEqual(state.velo, 15.0, places=4)


class BicycleReverseTest(unittest.TestCase):
    def _ego(self, **kinematic):
        cfg = SimpleNamespace(dt=0.25, Lr=1.35, Lf=1.35, fric=0.2, min_steer=15,
                              max_steer=15, min_acc=-2, max_acc=2, **kinematic)
        scenario = SimpleNamespace(start_ab=[0, 0], start_euler=[0, 0, 0], start_velo=1.0)
        return BicycleEgo(cfg, scenario)

    def test_braking_stops_instead_of_reversing(self):
        ego = self._ego()
        for _ in range(10):
            ego.step(-2.0, 0.0)
        self.assertEqual(ego.velo, 0.0)
        self.assertGreaterEqual(float(ego.vab[1]), 0.0)

    def test_allow_reverse_keeps_hugsim_behaviour(self):
        ego = self._ego(allow_reverse=True)
        for _ in range(10):
            ego.step(-2.0, 0.0)
        self.assertLess(ego.velo, 0.0)


if __name__ == "__main__":
    unittest.main()
