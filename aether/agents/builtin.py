"""Reference agents that need no learned model (smoke tests, baselines)."""
from __future__ import annotations

import numpy as np

from .base import Control, EgoAgent, Observation, Plan


class LogReplayAgent(EgoAgent):
    """Follows the logged ego route at the logged speed (expert baseline)."""

    cameras: list[str] = []          # needs no images

    def __init__(self, horizon: int = 6, dt: float = 0.5) -> None:
        self.horizon, self.dt = int(horizon), float(dt)

    def setup(self, context) -> None:
        self.route = np.asarray(context["route_xy"], dtype=float)
        self.frame_dt = float(context["frame_dt"])
        seg = np.linalg.norm(np.diff(self.route, axis=0), axis=1)
        self.s = np.concatenate([[0.0], np.cumsum(seg)])
        self.t = np.arange(len(self.route)) * self.frame_dt

    def act(self, obs: Observation) -> Plan:
        ego = obs.ego
        times = obs.time + self.dt * np.arange(1, self.horizon + 1)
        s_now = np.interp(obs.time, self.t, self.s)
        s_future = np.interp(times, self.t, self.s)
        # Start from the ego's own arc position, keep the logged spacing.
        target = s_future - s_now + self._closest_s(ego["x"], ego["y"])
        xs = np.interp(target, self.s, self.route[:, 0])
        ys = np.interp(target, self.s, self.route[:, 1])
        c, s = np.cos(ego["yaw"]), np.sin(ego["yaw"])
        dx, dy = xs - ego["x"], ys - ego["y"]
        local = np.column_stack([c * dx + s * dy, -s * dx + c * dy])
        return Plan(local, self.dt, frame="center", axes="flu")

    def _closest_s(self, x: float, y: float) -> float:
        d = np.hypot(self.route[:, 0] - x, self.route[:, 1] - y)
        return float(self.s[int(np.argmin(d))])


class ConstantControl(EgoAgent):
    """Applies a fixed command; useful to test kinematics and collisions."""

    cameras: list[str] = []

    def __init__(self, accel: float = 0.0, steer_rate: float = 0.0) -> None:
        self.command = Control(float(accel), float(steer_rate))

    def act(self, obs: Observation) -> Control:
        return self.command
