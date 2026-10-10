"""Versioned JSON protocol shared by AETHER and external traffic simulators.

Coordinates on the wire are always a local, right-handed road frame:
``x=forward, y=left, z=up`` in metres and yaw is counter-clockwise radians.
Each message is one compact UTF-8 JSON object followed by a newline.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping, Sequence


PROTOCOL_VERSION = "aether.traffic.v1"


class ProtocolError(ValueError):
    """Raised when the peer sends an incompatible or malformed message."""


def _finite(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"{field} must be numeric") from exc
    if not math.isfinite(result):
        raise ProtocolError(f"{field} must be finite")
    return result


def _positive(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result <= 0:
        raise ProtocolError(f"{field} must be positive")
    return result


@dataclass(frozen=True)
class EgoState:
    x: float
    y: float
    yaw: float
    speed: float
    length: float = 4.5
    width: float = 1.85
    height: float = 1.5

    def to_dict(self) -> dict[str, Any]:
        return {
            "x": float(self.x), "y": float(self.y), "yaw": float(self.yaw),
            "speed": float(self.speed), "length": float(self.length),
            "width": float(self.width), "height": float(self.height),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EgoState":
        return cls(
            x=_finite(value.get("x"), "ego.x"),
            y=_finite(value.get("y"), "ego.y"),
            yaw=_finite(value.get("yaw"), "ego.yaw"),
            speed=_finite(value.get("speed"), "ego.speed"),
            length=_positive(value.get("length", 4.5), "ego.length"),
            width=_positive(value.get("width", 1.85), "ego.width"),
            height=_positive(value.get("height", 1.5), "ego.height"),
        )


@dataclass(frozen=True)
class VehicleState:
    vehicle_id: str
    x: float
    y: float
    yaw: float
    speed: float
    length: float = 4.1
    width: float = 1.75
    height: float = 1.3
    z: float = 0.0
    vehicle_type: str = "car"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.vehicle_id, "x": float(self.x), "y": float(self.y),
            "z": float(self.z), "yaw": float(self.yaw),
            "speed": float(self.speed), "length": float(self.length),
            "width": float(self.width), "height": float(self.height),
            "type": self.vehicle_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VehicleState":
        vehicle_id = str(value.get("id", "")).strip()
        if not vehicle_id:
            raise ProtocolError("vehicle.id must be non-empty")
        return cls(
            vehicle_id=vehicle_id,
            x=_finite(value.get("x"), f"vehicle[{vehicle_id}].x"),
            y=_finite(value.get("y"), f"vehicle[{vehicle_id}].y"),
            z=_finite(value.get("z", 0.0), f"vehicle[{vehicle_id}].z"),
            yaw=_finite(value.get("yaw"), f"vehicle[{vehicle_id}].yaw"),
            speed=_finite(value.get("speed", 0.0), f"vehicle[{vehicle_id}].speed"),
            length=_positive(value.get("length", 4.1), f"vehicle[{vehicle_id}].length"),
            width=_positive(value.get("width", 1.75), f"vehicle[{vehicle_id}].width"),
            height=_positive(value.get("height", 1.3), f"vehicle[{vehicle_id}].height"),
            vehicle_type=str(value.get("type", "car")),
        )


@dataclass(frozen=True)
class TrafficFrame:
    sequence: int
    time_s: float
    vehicles: tuple[VehicleState, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": int(self.sequence),
            "time_s": float(self.time_s),
            "vehicles": [vehicle.to_dict() for vehicle in self.vehicles],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrafficFrame":
        try:
            sequence = int(value.get("sequence"))
        except (TypeError, ValueError) as exc:
            raise ProtocolError("frame.sequence must be an integer") from exc
        raw_vehicles = value.get("vehicles", [])
        if not isinstance(raw_vehicles, Sequence) or isinstance(raw_vehicles, (str, bytes)):
            raise ProtocolError("frame.vehicles must be a list")
        vehicles = tuple(VehicleState.from_dict(item) for item in raw_vehicles)
        ids = [item.vehicle_id for item in vehicles]
        if len(ids) != len(set(ids)):
            raise ProtocolError("frame contains duplicate vehicle ids")
        return cls(
            sequence=sequence,
            time_s=_finite(value.get("time_s"), "frame.time_s"),
            vehicles=vehicles,
        )


def request(command: str, sequence: int, **payload: Any) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "kind": "request",
        "command": str(command),
        "sequence": int(sequence),
        **payload,
    }


def response(command: str, sequence: int, **payload: Any) -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "kind": "response",
        "command": str(command),
        "sequence": int(sequence),
        **payload,
    }


def encode_message(message: Mapping[str, Any]) -> bytes:
    return (json.dumps(dict(message), ensure_ascii=True, separators=(",", ":")) + "\n").encode("utf-8")


def decode_message(line: bytes | str) -> dict[str, Any]:
    try:
        value = json.loads(line)
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON message") from exc
    if not isinstance(value, dict):
        raise ProtocolError("message must be a JSON object")
    if value.get("version") != PROTOCOL_VERSION:
        raise ProtocolError(
            f"protocol version mismatch: expected {PROTOCOL_VERSION!r}, "
            f"got {value.get('version')!r}"
        )
    return value
