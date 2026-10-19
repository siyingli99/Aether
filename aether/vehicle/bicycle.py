"""Kinematic bicycle model about the rear axle, in the road frame.

State: rear-axle position (x, y), heading ``yaw`` (CCW), speed ``v`` and
front-wheel angle ``steer`` (positive = left).  Inputs: longitudinal
acceleration and steering rate.  With ``clip_inputs`` the steering angle is
also limited by the tyre-friction circle, ``v^2 tan(steer) / L <=
max_lat_accel``: a kinematic model has no slip, so without this limit a
planner could corner at 3 g.  Integration is semi-implicit Euler (speed
and steering first), identical to the controller's prediction model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from .profile import VehicleProfile


@dataclass(frozen=True)
class VehicleState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    v: float = 0.0
    steer: float = 0.0
    accel: float = 0.0
    steer_rate: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.yaw, self.v, self.steer])


class BicycleModel:
    def __init__(self, profile: VehicleProfile, dt: float,
                 clip_inputs: bool = True, allow_reverse: bool = False) -> None:
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.profile = profile
        self.dt = float(dt)
        self.clip_inputs = bool(clip_inputs)
        self.allow_reverse = bool(allow_reverse)

    def yaw_rate(self, state: VehicleState) -> float:
        return state.v * math.tan(state.steer) / self.profile.wheelbase

    def step(self, state: VehicleState, accel: float, steer_rate: float) -> VehicleState:
        p, dt = self.profile, self.dt
        accel, steer_rate = float(accel), float(steer_rate)
        if self.clip_inputs:
            accel = min(max(accel, -p.max_decel), p.max_accel)
            steer_rate = min(max(steer_rate, -p.max_steer_rate), p.max_steer_rate)
        v = state.v + accel * dt
        if not self.allow_reverse:
            v = max(v, 0.0)
        v = min(v, p.max_speed)
        steer = state.steer + steer_rate * dt
        if self.clip_inputs:
            limit = p.max_steer
            if v > 1e-3:
                limit = min(limit, math.atan(p.max_lat_accel * p.wheelbase / (v * v)))
            steer = min(max(steer, -limit), limit)
        x = state.x + v * math.cos(state.yaw) * dt
        y = state.y + v * math.sin(state.yaw) * dt
        yaw = state.yaw + v * math.tan(steer) / p.wheelbase * dt
        return VehicleState(x, y, yaw, v, steer, accel, steer_rate)

    def center(self, state: VehicleState) -> tuple[float, float]:
        """Box centre of the vehicle in the road frame."""
        offset = self.profile.center_offset
        return (state.x + offset * math.cos(state.yaw),
                state.y + offset * math.sin(state.yaw))

    @staticmethod
    def with_speed(state: VehicleState, v: float) -> VehicleState:
        return replace(state, v=float(v))
