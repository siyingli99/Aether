import socket
import threading
import unittest


from aether.traffic.client import (
    LimSimTrafficClient,
    ReplayTrafficClient,
    resolve_external_host,
)
from aether.traffic.protocol import (
    EgoState,
    ProtocolError,
    TrafficFrame,
    VehicleState,
    decode_message,
    encode_message,
    response,
)


class ProtocolTest(unittest.TestCase):
    def test_frame_round_trip(self):
        frame = TrafficFrame(7, 0.25, (VehicleState("npc-1", 4, -2, 0.1, 6),))
        message = response("tick", 9, frame=frame.to_dict())
        decoded = decode_message(encode_message(message))
        self.assertEqual(TrafficFrame.from_dict(decoded["frame"]), frame)

    def test_rejects_wrong_version(self):
        with self.assertRaisesRegex(ProtocolError, "version mismatch"):
            decode_message('{"version":"old"}')


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


class ReplayClientTest(unittest.TestCase):
    def test_replay_yaw_takes_short_arc(self):
        frames = [
            TrafficFrame(0, 0.0, (VehicleState("a", 0, 0, 3.0, 1),)),
            TrafficFrame(1, 1.0, (VehicleState("a", 0, 0, -3.0, 1),)),
        ]
        mid = ReplayTrafficClient(frames).tick(0.5, EgoState(0, 0, 0, 0)).vehicles[0]
        self.assertGreater(abs(mid.yaw), 3.0)


if __name__ == "__main__":
    unittest.main()
