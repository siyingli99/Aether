"""Cache one external traffic update per timestamp and expose renderer actors."""
from __future__ import annotations

import math
from typing import Iterable

from .client import TrafficClient
from .coordinates import FrameTransform, vehicle_pose
from .protocol import EgoState, TrafficFrame


class TrafficActorProvider:
    """Stable traffic vehicle-id to DriveStudio instance-slot adapter."""

    def __init__(self, client: TrafficClient, transform: FrameTransform,
                 instance_ids: Iterable[int]) -> None:
        self.client = client
        self.transform = transform
        self._all_slots = tuple(int(value) for value in instance_ids)
        self._free_slots = list(self._all_slots)
        if not self._all_slots:
            raise ValueError("at least one DriveStudio instance slot is required")
        self._vehicle_slots: dict[str, int] = {}
        self._frame: TrafficFrame | None = None
        self._entries: list[dict] = []

    @property
    def frame(self) -> TrafficFrame | None:
        return self._frame

    def _assign_slot(self, vehicle_id: str) -> int | None:
        if vehicle_id in self._vehicle_slots:
            return self._vehicle_slots[vehicle_id]
        if not self._free_slots:
            return None
        slot = self._free_slots.pop(0)
        self._vehicle_slots[vehicle_id] = slot
        return slot

    def _release_departed(self, frame: TrafficFrame) -> None:
        """Return slots of vehicles that left the simulation to the pool.

        Without this, a long LimSim run exhausts the slots once ``capacity``
        distinct vehicles have spawned and every later vehicle is dropped.
        Freed slots are reused in sorted order so assignments stay
        deterministic.
        """
        present = {vehicle.vehicle_id for vehicle in frame.vehicles}
        departed = [vid for vid in self._vehicle_slots if vid not in present]
        for vehicle_id in departed:
            self._free_slots.append(self._vehicle_slots.pop(vehicle_id))
        if departed:
            self._free_slots.sort()

    def _consume(self, frame: TrafficFrame) -> list[dict]:
        self._release_departed(frame)
        entries: list[dict] = []
        for vehicle in sorted(frame.vehicles, key=lambda item: item.vehicle_id):
            slot = self._assign_slot(vehicle.vehicle_id)
            if slot is None:
                continue
            c2w = self.transform.pose_to_ds(
                # Traffic backends are 2D: z denotes road/tire-contact elevation while a
                # DriveStudio rigid pose is centred on its canonical box.
                vehicle_pose(
                    vehicle.x, vehicle.y, vehicle.yaw,
                    vehicle.z + vehicle.height * 0.5,
                )
            )
            entries.append({
                "class": "RigidNodes",
                "instance_id": slot,
                "actor_id": vehicle.vehicle_id,
                "size": [vehicle.width, vehicle.length, vehicle.height],
                "c2w": c2w,
                "velo": vehicle.speed,
                "yaw": vehicle.yaw,
                "x": vehicle.x,
                "y": vehicle.y,
                "z": vehicle.z,
                "label": vehicle.vehicle_type,
            })
        self._frame = frame
        self._entries = entries
        return entries

    def reset(self, ego: EgoState | None = None) -> list[dict]:
        self._vehicle_slots.clear()
        self._free_slots = list(self._all_slots)
        entries = self._consume(self.client.reset())
        if ego is not None:
            # A live traffic bridge creates the ego/background vehicles on its
            # first simulation callback.  A zero-time tick produces that
            # initial synchronized frame before AETHER renders observation 0.
            entries = self._consume(self.client.tick(0.0, ego))
        return entries

    def advance(self, target_time_s: float, ego: EgoState) -> list[dict]:
        target = float(target_time_s)
        if self._frame is not None:
            if target < self._frame.time_s - 1e-9:
                raise ValueError("traffic time cannot move backwards without reset")
            if math.isclose(target, self._frame.time_s, abs_tol=1e-9):
                return self._entries
        return self._consume(self.client.tick(target, ego))

    def pose_at(self, frame_idx: int) -> list[dict]:
        """Read the cached frame; this method never advances the simulator."""
        del frame_idx
        return list(self._entries)

    def close(self) -> None:
        self.client.close()
