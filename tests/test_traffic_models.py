"""Traffic models, slot recycling in TrafficManager, and the planner FIFO transport."""
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from aether.agents import ipc
from aether.geometry.frames import pose2d
from aether.render.external_assets import GaussianVehicleAsset
from aether.scene.hdmap import HDMap
from aether.traffic.builtin import IDMLanes, ScriptedVehicles
from aether.traffic.manager import TrafficManager, build_model
from aether.traffic.model import AgentState, TrafficModel, TrafficScene


def _straight_map(tmp: Path) -> HDMap:
    lanes = [{"id": "L0", "centerline": [[x, 0.0] for x in range(0, 201, 5)], "width": 3.5},
             {"id": "L1", "centerline": [[x, 3.5] for x in range(0, 201, 5)], "width": 3.5}]
    path = tmp / "map.json"
    path.write_text(json.dumps({"schema": "aether.hdmap.v2", "lanes": lanes, "local_to_global": np.eye(4).tolist()}))
    return HDMap.load(path)


class IDMTest(unittest.TestCase):
    def test_follower_keeps_gap_behind_slow_ego(self):
        with tempfile.TemporaryDirectory() as tmp:
            hdmap = _straight_map(Path(tmp))
        model = IDMLanes(vehicles=[{"id": "f", "lane": "L0", "s": 5.0, "speed": 10.0}], desired_speed=12.0)
        ego = AgentState("ego", 40.0, 0.0, 0.0, 2.0, 4.0, 1.8, 1.5)
        agents = model.reset(TrafficScene("t", 0.25, ego, hdmap))
        for k in range(1, 81):
            ego = AgentState("ego", 40.0 + 2.0 * 0.25 * k, 0.0, 0.0, 2.0, 4.0, 1.8, 1.5)
            agents = model.step(0.25 * k, 0.25, ego, agents)
        gap = ego.x - agents[0].x - 0.5 * (ego.length + agents[0].length)
        self.assertGreater(gap, 2.0)
        self.assertLess(abs(agents[0].speed - 2.0), 0.5)


class ScriptedTest(unittest.TestCase):
    def test_constant_velocity_plugin(self):
        model = ScriptedVehicles([{"id": "c", "start": {"x": 0, "y": 0, "yaw": 0.5},
                                   "controller": {"type": "plugin", "class": "aether.traffic.builtin:ConstantVelocity",
                                                  "params": {"speed_mps": 4.0}}}])
        ego = AgentState("ego", 0, 0, 0, 0)
        model.reset(TrafficScene("t", 0.25, ego))
        state = model.step(2.0, 0.25, ego, [])[0]
        self.assertAlmostEqual(state.x, 8 * np.cos(0.5), places=6)
        self.assertAlmostEqual(state.speed, 4.0)


class _Script(TrafficModel):
    """Vehicles appear and leave on a fixed schedule."""
    SCHEDULE = {0.0: ["a", "b"], 1.0: ["b"], 2.0: ["b", "c"]}

    def reset(self, scene):
        return self._at(0.0)

    def step(self, t, dt, ego, vehicles):
        return self._at(t)

    def _at(self, t):
        return [AgentState(v, 10.0 + i, 0.0, 0.0, 1.0) for i, v in enumerate(self.SCHEDULE[round(t, 3)])]


def _fake_renderer(frames=4):
    n = 3
    rigid = SimpleNamespace(
        _means=torch.nn.Parameter(torch.zeros(n, 3)), _features_dc=torch.nn.Parameter(torch.zeros(n, 3)),
        _features_rest=torch.nn.Parameter(torch.zeros(n, 15, 3)), _opacities=torch.nn.Parameter(torch.zeros(n, 1)),
        _scales=torch.nn.Parameter(torch.zeros(n, 3)), _quats=torch.nn.Parameter(torch.zeros(n, 4)),
        point_ids=torch.tensor([[0], [0], [1]]),
        instances_trans=torch.nn.Parameter(torch.zeros(frames, 2, 3)),
        instances_quats=torch.nn.Parameter(torch.zeros(frames, 2, 4)),
        instances_fv=torch.ones(frames, 2, dtype=torch.bool), instances_size=torch.ones(2, 3))
    return SimpleNamespace(trainer=SimpleNamespace(models={"RigidNodes": rigid}))


class ManagerTest(unittest.TestCase):
    def test_slots_recycle_and_originals_hidden(self):
        renderer = _fake_renderer()
        asset = GaussianVehicleAsset(torch.zeros(2, 3), torch.zeros(2, 3), torch.zeros(2, 15, 3), torch.zeros(2, 1),
                                     torch.zeros(2, 3), torch.tensor([[1.0, 0, 0, 0]] * 2), (4.4, 1.8, 1.5), "test")
        geometry = SimpleNamespace(to_scene=lambda p: p, to_road=lambda p: p,
                                   object_pose=lambda x, y, yaw, l, w, h: pose2d(x, y, yaw, 0.5 * h))
        bundle = SimpleNamespace(id="t", hdmap=None, network=None, actors_path=None, frame_dt=0.1)
        manager = TrafficManager(renderer, geometry, bundle,
                                 {"hide_originals": True, "max_vehicles": 2, "model": {"class": f"{__name__}:_Script"}},
                                 dt=1.0, assets=[asset])
        self.assertFalse(renderer.trainer.models["RigidNodes"].instances_fv[:, :2].any())
        ego = AgentState("ego", 0, 0, 0, 0)
        manager.reset(ego, 0)
        manager.advance(1.0, ego, 1)
        manager.advance(2.0, ego, 2)
        slots = {a.state.id: a.binding for a in manager.agents()}
        self.assertEqual(slots, {"b": "asset:3", "c": "asset:2"})
        self.assertEqual(manager.agents()[0].state.length, 4.4)        # asset size wins
        manager.assert_hidden()

    def test_set_keep_and_pose_override_of_recorded_actor(self):
        import tempfile
        renderer = _fake_renderer()
        frames = np.arange(4)
        c2w = np.stack([pose2d(5.0 + i, 2.0, 0.0) for i in frames])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "actors.npz")
            np.savez(path, **{"c2w/RigidNodes/1": c2w, "frames/RigidNodes/1": frames,
                              "size/RigidNodes/1": np.array([1.8, 4.4, 1.5])})
            geometry = SimpleNamespace(to_scene=lambda p: p, to_road=lambda p: p)
            bundle = SimpleNamespace(id="t", hdmap=None, network=None, actors_path=path, frame_dt=0.1)
            manager = TrafficManager(renderer, geometry, bundle, {"hide_originals": True}, dt=0.1)
        fv = renderer.trainer.models["RigidNodes"].instances_fv
        self.assertFalse(fv.any())
        manager.set_keep(["RigidNodes/1"])
        self.assertTrue(fv[:, 1].all() and not fv[:, 0].any())
        manager.pose_overrides[("RigidNodes", 1)] = lambda t, pose: pose @ pose2d(0.0, -1.0, 0.3)
        manager.reset(AgentState("ego", 0, 0, 0, 0), 2)
        agent = manager.agents()[0]
        self.assertAlmostEqual(agent.state.x, 7.0)
        self.assertAlmostEqual(agent.state.y, 1.0)
        self.assertAlmostEqual(agent.state.yaw, 0.3)
        (cls, idx, pose), = manager.render_overrides()
        self.assertEqual((cls, idx), ("RigidNodes", 1))
        self.assertAlmostEqual(float(pose[1, 3]), 1.0)
        manager.set_keep(["*"])
        self.assertEqual(manager.keep, {("RigidNodes", 1)})
        manager.set_keep(["RigidNodes/*", "SMPLNodes/2"])
        self.assertEqual(manager.keep, {("RigidNodes", 1), ("SMPLNodes", 2)})

    def test_build_model_passes_inline_parameters(self):
        model = build_model({"type": "idm", "count": 2, "desired_speed": 7.0})
        self.assertEqual((model.count, model.v0), (2, 7.0))


class FifoTest(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "pipe")
        os.mkfifo(self.path)

    def test_round_trip_large_payload(self):
        payload = os.urandom(3_000_000)

        def writer():
            time.sleep(0.2)
            with open(self.path, "wb") as handle:
                handle.write(payload)

        threading.Thread(target=writer).start()
        self.assertEqual(ipc.receive(self.path, timeout_s=5), payload)
        received = []

        def reader():
            time.sleep(0.2)
            with open(self.path, "rb") as handle:
                received.append(handle.read())

        thread = threading.Thread(target=reader)
        thread.start()
        ipc.send(self.path, payload, timeout_s=5)
        thread.join()
        self.assertEqual(received[0], payload)

    def test_dead_planner_fails_fast(self):
        proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
        proc.wait()
        with self.assertRaisesRegex(RuntimeError, "exited with code 3"):
            ipc.receive(self.path, proc=proc, timeout_s=10)
        with self.assertRaisesRegex(RuntimeError, "exited with code 3"):
            ipc.send(self.path, b"x", proc=proc, timeout_s=10)

    def test_silent_planner_times_out(self):
        with self.assertRaises(TimeoutError):
            ipc.receive(self.path, timeout_s=0.6, poll_s=0.1)


if __name__ == "__main__":
    unittest.main()
