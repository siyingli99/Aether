import socket
import threading
import unittest

import numpy as np

from aether.traffic.client import (
    LimSimTrafficClient,
    ReplayTrafficClient,
    resolve_external_host,
)
from aether.traffic.coordinates import nuscenes_scene_local_to_ds, vehicle_pose
from aether.traffic.protocol import (
    EgoState,
    ProtocolError,
    TrafficFrame,
    VehicleState,
    decode_message,
    encode_message,
    response,
)
from aether.traffic.provider import TrafficActorProvider


class ProtocolTest(unittest.TestCase):
    def test_frame_round_trip(self):
        frame = TrafficFrame(7, 0.25, (VehicleState("npc-1", 4, -2, 0.1, 6),))
        message = response("tick", 9, frame=frame.to_dict())
        decoded = decode_message(encode_message(message))
        self.assertEqual(TrafficFrame.from_dict(decoded["frame"]), frame)

    def test_rejects_wrong_version(self):
        with self.assertRaisesRegex(ProtocolError, "version mismatch"):
            decode_message('{"version":"old"}')


class CoordinateTest(unittest.TestCase):
    def test_nuscenes_axis_mapping(self):
        transform = nuscenes_scene_local_to_ds()
        np.testing.assert_allclose(transform.point_to_ds([3, 2, 1]), [-2, -1, 3])
        ds_pose = transform.pose_to_ds(vehicle_pose(3, 2, 0, 1))
        np.testing.assert_allclose(ds_pose[:3, 3], [-2, -1, 3])
        self.assertAlmostEqual(float(np.linalg.det(ds_pose[:3, :3])), 1.0)


class ProviderTest(unittest.TestCase):
    def test_replay_interpolation_is_idempotent_per_timestamp(self):
        frames = [
            TrafficFrame(0, 0.0, (VehicleState("a", 5, 1, 0, 2),)),
            TrafficFrame(1, 1.0, (VehicleState("a", 9, 3, 0, 4),)),
        ]
        provider = TrafficActorProvider(
            ReplayTrafficClient(frames), nuscenes_scene_local_to_ds(), [42]
        )
        provider.reset()
        ego = EgoState(0, 0, 0, 2)
        first = provider.advance(0.5, ego)
        second = provider.advance(0.5, ego)
        self.assertEqual(first[0]["instance_id"], 42)
        np.testing.assert_allclose(first[0]["c2w"][:3, 3], [-2, -0.65, 7])
        np.testing.assert_allclose(first[0]["c2w"], second[0]["c2w"])


class TcpClientTest(unittest.TestCase):
    def test_auto_host_resolves_to_ipv4(self):
        socket.inet_aton(resolve_external_host("auto"))
        self.assertEqual(resolve_external_host("example.test"), "example.test")

    def test_limsim_handshake_reset_and_tick(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]

        def server():
            conn, _ = listener.accept()
            reader = conn.makefile("rb")
            try:
                for command in ("hello", "reset", "tick"):
                    incoming = decode_message(reader.readline())
                    payload = {"ok": True}
                    if command == "hello":
                        payload["server"] = "limsim"
                    else:
                        payload["frame"] = TrafficFrame(
                            incoming["sequence"],
                            float(incoming.get("target_time_s", 0.0)),
                            (),
                        ).to_dict()
                    conn.sendall(encode_message(response(
                        command, incoming["sequence"], **payload
                    )))
            finally:
                reader.close()
                conn.close()
                listener.close()

        thread = threading.Thread(target=server, daemon=True)
        thread.start()
        client = LimSimTrafficClient(port=port, timeout_s=2)
        self.assertEqual(client.reset().time_s, 0.0)
        self.assertEqual(client.tick(0.25, EgoState(0, 0, 0, 1)).time_s, 0.25)
        client.close()
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
