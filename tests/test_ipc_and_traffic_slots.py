"""Adapter FIFO timeouts and traffic-slot recycling."""
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from aether.agents.launcher import recv_fifo, send_fifo
from aether.traffic.client import ReplayTrafficClient
from aether.traffic.coordinates import nuscenes_scene_local_to_ds
from aether.traffic.protocol import EgoState, TrafficFrame, VehicleState
from aether.traffic.provider import TrafficActorProvider


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
        self.assertEqual(recv_fifo(self.path, timeout_s=5), payload)

        received = []

        def reader():
            time.sleep(0.2)
            with open(self.path, "rb") as handle:
                received.append(handle.read())

        thread = threading.Thread(target=reader)
        thread.start()
        send_fifo(self.path, payload, timeout_s=5)
        thread.join()
        self.assertEqual(received[0], payload)

    def test_dead_adapter_fails_fast(self):
        proc = subprocess.Popen([sys.executable, "-c", "raise SystemExit(3)"])
        proc.wait()
        with self.assertRaisesRegex(RuntimeError, "exited with code 3"):
            recv_fifo(self.path, proc=proc, timeout_s=10)
        with self.assertRaisesRegex(RuntimeError, "exited with code 3"):
            send_fifo(self.path, b"x", proc=proc, timeout_s=10)

    def test_silent_adapter_times_out(self):
        with self.assertRaises(TimeoutError):
            recv_fifo(self.path, timeout_s=0.6, poll_s=0.1)


class SlotRecyclingTest(unittest.TestCase):
    def test_departed_vehicle_frees_slot(self):
        frames = [
            TrafficFrame(0, 0.0, (VehicleState("a", 5, 0, 0, 1), VehicleState("b", 9, 0, 0, 1))),
            TrafficFrame(1, 1.0, (VehicleState("b", 10, 0, 0, 1),)),
            TrafficFrame(2, 2.0, (VehicleState("b", 11, 0, 0, 1), VehicleState("c", 2, 0, 0, 1))),
        ]
        provider = TrafficActorProvider(
            ReplayTrafficClient(frames), nuscenes_scene_local_to_ds(), [40, 41]
        )
        provider.reset()
        ego = EgoState(0, 0, 0, 0)
        provider.advance(1.0, ego)
        entries = provider.advance(2.0, ego)
        slots = {entry["actor_id"]: entry["instance_id"] for entry in entries}
        self.assertEqual(slots, {"b": 41, "c": 40})

    def test_replay_yaw_takes_short_arc(self):
        frames = [
            TrafficFrame(0, 0.0, (VehicleState("a", 0, 0, 3.0, 1),)),
            TrafficFrame(1, 1.0, (VehicleState("a", 0, 0, -3.0, 1),)),
        ]
        mid = ReplayTrafficClient(frames).tick(0.5, EgoState(0, 0, 0, 0)).vehicles[0]
        self.assertGreater(abs(mid.yaw), 3.0)


if __name__ == "__main__":
    unittest.main()
