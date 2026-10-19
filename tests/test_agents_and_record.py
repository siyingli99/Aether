"""GPU-free tests: plan tracking, tuple pipe payload, trajectory log, BEV, metrics."""
import math
import unittest

import numpy as np

from aether.agents import Plan
from aether.agents.base import Observation
from aether.agents.observation import tuple_payload
from aether.agents.tracker import TrajectoryTracker, rear_axle_reference
from aether.eval.metrics import score_episode
from aether.record.bev import BEVRenderer
from aether.record.interp import interpolate_state
from aether.record.trajectory import TrajectoryRecorder
from aether.sim.route import Route
from aether.vehicle import load_profile


def _info(t, step, x, speed=5.0, agents=()):
    return {"time": t, "step": step, "frame": step, "command": 2, "route_progress": 0.0,
            "events": [],
            "ego": {"x": x, "y": 0.0, "yaw": 0.0, "speed": speed, "accel": 0.0, "steer": 0.0,
                    "steer_rate": 0.0, "yaw_rate": 0.0, "rear_axle": [x - 1.4, 0.0],
                    "length": 4.0, "width": 1.8, "height": 1.5},
            "agents": list(agents)}


class PlanTest(unittest.TestCase):
    def test_rfu_and_flu_agree(self):
        a = Plan(np.array([[0.5, 2.0], [1.0, 4.0]]), axes="rfu")
        b = Plan(np.array([[2.0, -0.5], [4.0, -1.0]]), axes="flu")
        np.testing.assert_allclose(a.forward_left(), b.forward_left())

    def test_lidar_plan_moves_back_to_rear_axle(self):
        ref = rear_axle_reference(Plan(np.array([[0, 2.0], [0, 4.0]]), frame="lidar", axes="rfu"),
                                  np.array([0.986, 0.0]))
        np.testing.assert_allclose(ref[:, 0], [0.0, 2.0, 4.0], atol=1e-9)

    def test_tracker_accelerates_toward_far_waypoints(self):
        tracker = TrajectoryTracker(load_profile("renault_zoe"))
        far = Plan(np.column_stack([np.arange(1, 7) * 4.0, np.zeros(6)]), axes="flu")
        accel, steer_rate = tracker.control(far, speed=2.0, steer=0.0)
        self.assertGreater(accel, 0.5)
        self.assertAlmostEqual(steer_rate, 0.0, places=3)
        left = Plan(np.column_stack([np.arange(1, 7) * 2.0, 0.08 * (np.arange(1, 7) * 2.0) ** 2]))
        _, steer_left = tracker.control(left, speed=4.0, steer=0.0)
        self.assertGreater(steer_left, 0.0)          # left turn = positive steer rate


class LegacyPayloadTest(unittest.TestCase):
    def test_conventions(self):
        mount = np.eye(4)
        mount[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]   # OpenCV camera looking forward
        mount[:3, 3] = [1.7, 0.0, 1.5]
        obs = Observation(time=1.0, step=4, images={"CAM_FRONT": np.zeros((450, 800, 3), np.uint8)},
                          cameras={"CAM_FRONT": {"K": np.array([[600, 0, 400], [0, 600, 225], [0, 0, 1.0]]),
                                                 "width": 800, "height": 450, "mount": mount}},
                          ego={**_info(0, 0, 3.0)["ego"], "yaw": 0.2, "steer": 0.1, "steer_rate": 0.05,
                               "rear_axle": [3.0, 1.0]},
                          command=1, info={"agents": []})
        payload, info = tuple_payload(obs, np.eye(4))
        self.assertEqual(info["ego_pos"], [-1.0, 0.0, 3.0])
        self.assertAlmostEqual(info["ego_rot"][1], -0.2)
        self.assertAlmostEqual(info["ego_steer"], -0.1)
        intr = info["cam_params"]["CAM_FRONT"]["intrinsic"]
        self.assertAlmostEqual(800 / (2 * math.tan(intr["fovx"] / 2)), 600.0, places=6)
        v2c = info["cam_params"]["CAM_FRONT"]["v2c"]
        point = v2c @ np.array([11.7, 0.0, 1.5, 1.0])         # 10 m ahead of the camera
        np.testing.assert_allclose(point[:3], [0, 0, 10], atol=1e-5)
        self.assertIn("CAM_FRONT", payload["rgb"])


class RecorderTest(unittest.TestCase):
    def test_interpolation_is_smooth_and_exact_at_steps(self):
        a = {"x": 0.0, "y": 0.0, "yaw": 0.0, "speed": 4.0}
        b = {"x": 1.0, "y": 0.0, "yaw": 0.0, "speed": 4.0}
        mid = interpolate_state(a, b, 0.25, 0.5)
        self.assertAlmostEqual(mid["x"], 0.5)
        self.assertEqual(interpolate_state(a, b, 0.25, 1.0)["x"], 1.0)

    def test_csv_and_resampling(self):
        import csv
        import tempfile
        rec = TrajectoryRecorder()
        agent = {"id": "7", "source": "model", "x": 10.0, "y": 3.0, "yaw": 0.0, "speed": 3.0,
                 "length": 4.5, "width": 1.8, "height": 1.5, "type": "car"}
        for k in range(5):
            rec.add(_info(0.25 * k, k, 1.25 * k, agents=[{**agent, "x": 10 + 0.75 * k}]))
        with tempfile.TemporaryDirectory() as tmp:
            files = rec.write(tmp, hz=10)
            rows = list(csv.DictReader(open(files["steps"])))
            self.assertEqual(len(rows), 10)
            fine = list(csv.DictReader(open(files["resampled"])))
            self.assertEqual(len({r["time_s"] for r in fine}), 11)        # 0.0 … 1.0 s
            r = [x for x in fine if x["vehicle_id"] == "model:7" and x["time_s"] == "0.5000"][0]
            self.assertAlmostEqual(float(r["x"]), 11.5, places=3)
            self.assertAlmostEqual(float(r["accel"]), 0.0, places=3)
            import xml.etree.ElementTree as ET
            fcd = ET.parse(files["fcd"]).getroot()
            steps = fcd.findall("timestep")
            self.assertEqual(len(steps), 11)
            car = [v for v in steps[5].findall("vehicle") if v.get("id") == "model:7"][0]
            self.assertAlmostEqual(float(car.get("x")), 11.5, places=2)
            self.assertAlmostEqual(float(car.get("angle")), 90.0, places=2)   # heading +x = east


class BEVTest(unittest.TestCase):
    def test_frame_renders(self):
        bev = BEVRenderer(None, np.column_stack([np.linspace(0, 50, 20), np.zeros(20)]), size=(320, 320))
        rec = TrajectoryRecorder()
        rec.add(_info(0.0, 0, 2.0))
        img = bev.frame((2.0, 0.0), rec.sample(0.0), trail=[(0, 0), (1, 0)],
                        plan=np.array([[4.0, 0.0], [6.0, 0.0]]), hud=["t = 0.00 s"], alerts=["x"])
        self.assertEqual(img.shape, (320, 320, 3))
        self.assertGreater(img.std(), 5)


class MetricsTest(unittest.TestCase):
    def test_clean_run_and_collision(self):
        route = Route(np.column_stack([np.linspace(0, 40, 41), np.zeros(41)]), 0.25)
        rec = TrajectoryRecorder()
        for k in range(9):
            rec.add(_info(0.25 * k, k, 1.0 * k, speed=4.0))
        clean = score_episode(rec.steps, [], route, lambda t: route.progress(4 * t, 0))
        self.assertEqual((clean.nc, clean.dac, clean.ttc, clean.comfort), (1.0, 1.0, 1.0, 1.0))
        self.assertAlmostEqual(clean.ep, 1.0, places=2)
        hit = score_episode(rec.steps, [{"type": "vehicle_collision", "time": 1.0, "step": 4,
                                         "other": "model:1", "at_fault": True}],
                            route, lambda t: route.progress(4 * t, 0))
        self.assertEqual(hit.score, 0.0)

    def test_ttc_detects_stopped_car_ahead(self):
        route = Route(np.column_stack([np.linspace(0, 40, 41), np.zeros(41)]), 0.25)
        rec = TrajectoryRecorder()
        stopped = {"id": "1", "source": "model", "x": 9.0, "y": 0.0, "yaw": 0.0, "speed": 0.0,
                   "length": 4.5, "width": 1.8, "height": 1.5, "type": "car"}
        for k in range(3):
            rec.add(_info(0.25 * k, k, 2.0 + 1.0 * k, speed=8.0, agents=[stopped]))
        m = score_episode(rec.steps, [], route, lambda t: 0.1)
        self.assertEqual(m.ttc, 0.0)


if __name__ == "__main__":
    unittest.main()
