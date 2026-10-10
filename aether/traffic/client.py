"""Synchronous clients for external traffic-simulator sidecars."""
from __future__ import annotations

import bisect
import json
from pathlib import Path
import socket
import sys
import threading
from typing import Iterable, Protocol

from .protocol import (
    EgoState,
    ProtocolError,
    TrafficFrame,
    VehicleState,
    decode_message,
    encode_message,
    request,
)


class TrafficClient(Protocol):
    def reset(self) -> TrafficFrame: ...
    def tick(self, target_time_s: float, ego: EgoState) -> TrafficFrame: ...
    def close(self) -> None: ...


def resolve_external_host(host: str) -> str:
    """Resolve ``auto`` to localhost, or the Windows host under WSL."""
    value = str(host).strip()
    if value.lower() not in {"auto", "windows"}:
        return value
    if sys.platform == "win32":
        return "127.0.0.1"
    try:
        # /proc/net/route stores IPv4 gateway bytes in little-endian order.
        with open("/proc/net/route", "r", encoding="ascii") as handle:
            for line in handle:
                fields = line.split()
                if len(fields) >= 4 and fields[1] == "00000000":
                    flags = int(fields[3], 16)
                    if flags & 0x2:
                        packed = bytes.fromhex(fields[2])
                        return socket.inet_ntoa(packed[::-1])
    except (OSError, ValueError):
        pass
    return "127.0.0.1"


class SocketTrafficClient:
    """Newline-framed TCP client shared by simulator sidecars."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765,
                 timeout_s: float = 5.0, expected_server: str = "traffic") -> None:
        self.host = resolve_external_host(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.expected_server = str(expected_server)
        self._socket: socket.socket | None = None
        self._reader = None
        self._sequence = 0
        # _connect performs the version handshake through _request, so this
        # must be re-entrant for the first application request.
        self._lock = threading.RLock()

    def _connect(self) -> None:
        if self._socket is not None:
            return
        sock = socket.create_connection((self.host, self.port), self.timeout_s)
        sock.settimeout(self.timeout_s)
        self._socket = sock
        self._reader = sock.makefile("rb")
        reply = self._request("hello", client="aether")
        if reply.get("server") != self.expected_server:
            self.close()
            raise ProtocolError(
                f"peer did not identify itself as {self.expected_server!r}"
            )

    def _request(self, command: str, **payload):
        with self._lock:
            if command != "hello":
                self._connect()
            elif self._socket is None:
                # hello is called only after the socket has been created.
                raise RuntimeError("socket must exist before hello")
            self._sequence += 1
            sequence = self._sequence
            assert self._socket is not None and self._reader is not None
            self._socket.sendall(encode_message(request(command, sequence, **payload)))
            line = self._reader.readline()
            if not line:
                self.close()
                raise ConnectionError(f"{self.expected_server} bridge closed the connection")
            reply = decode_message(line)
            if reply.get("kind") != "response" or reply.get("command") != command:
                raise ProtocolError(f"unexpected traffic response: {reply}")
            if int(reply.get("sequence", -1)) != sequence:
                raise ProtocolError("traffic response sequence mismatch")
            if reply.get("ok", True) is False:
                raise RuntimeError(str(reply.get("error", "traffic bridge request failed")))
            return reply

    def reset(self) -> TrafficFrame:
        reply = self._request("reset")
        return TrafficFrame.from_dict(reply["frame"])

    def tick(self, target_time_s: float, ego: EgoState) -> TrafficFrame:
        reply = self._request(
            "tick", target_time_s=float(target_time_s), ego=ego.to_dict()
        )
        return TrafficFrame.from_dict(reply["frame"])

    def status(self) -> dict:
        """Return bridge diagnostics without advancing simulation time."""
        return dict(self._request("status").get("diagnostics", {}))

    def shutdown(self) -> None:
        """Ask a locally managed sidecar to stop gracefully."""
        self._request("shutdown")

    def close(self) -> None:
        reader, sock = self._reader, self._socket
        self._reader = None
        self._socket = None
        if reader is not None:
            try:
                reader.close()
            except OSError:
                pass
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


class LimSimTrafficClient(SocketTrafficClient):
    """Client for the headless LimSim bridge."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8766,
                 timeout_s: float = 10.0) -> None:
        super().__init__(host, port, timeout_s, expected_server="limsim")


class ReplayTrafficClient:
    """Deterministic, interpolating bridge substitute used by tests/smoke runs."""

    def __init__(self, frames: Iterable[TrafficFrame]) -> None:
        self.frames = sorted(tuple(frames), key=lambda frame: frame.time_s)
        if not self.frames:
            raise ValueError("ReplayTrafficClient requires at least one frame")
        self._times = [frame.time_s for frame in self.frames]

    @classmethod
    def from_json(cls, path: str | Path) -> "ReplayTrafficClient":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw_frames = payload.get("frames", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw_frames, list):
            raise ValueError("traffic replay JSON must contain a frames list")
        return cls(TrafficFrame.from_dict(frame) for frame in raw_frames)

    def reset(self) -> TrafficFrame:
        return self.frames[0]

    @staticmethod
    def _interpolate(a: VehicleState, b: VehicleState, alpha: float) -> VehicleState:
        def lerp(x, y):
            return float(x) + alpha * (float(y) - float(x))
        return VehicleState(
            vehicle_id=a.vehicle_id,
            x=lerp(a.x, b.x), y=lerp(a.y, b.y), z=lerp(a.z, b.z),
            yaw=lerp(a.yaw, b.yaw), speed=lerp(a.speed, b.speed),
            length=lerp(a.length, b.length), width=lerp(a.width, b.width),
            height=lerp(a.height, b.height), vehicle_type=a.vehicle_type,
        )

    def tick(self, target_time_s: float, ego: EgoState) -> TrafficFrame:
        del ego
        target = float(target_time_s)
        right = bisect.bisect_right(self._times, target)
        if right <= 0:
            return self.frames[0]
        if right >= len(self.frames):
            return self.frames[-1]
        left_frame, right_frame = self.frames[right - 1], self.frames[right]
        span = right_frame.time_s - left_frame.time_s
        alpha = 0.0 if span <= 0 else (target - left_frame.time_s) / span
        right_by_id = {vehicle.vehicle_id: vehicle for vehicle in right_frame.vehicles}
        vehicles = tuple(
            self._interpolate(vehicle, right_by_id[vehicle.vehicle_id], alpha)
            if vehicle.vehicle_id in right_by_id else vehicle
            for vehicle in left_frame.vehicles
        )
        return TrafficFrame(
            sequence=right_frame.sequence, time_s=target, vehicles=vehicles
        )

    def close(self) -> None:
        return None
