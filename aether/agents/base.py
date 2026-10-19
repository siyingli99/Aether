"""Ego-agent interface.

An ego agent receives an :class:`Observation` every planning step and returns
either a :class:`Plan` (future waypoints, tracked by AETHER's iLQR controller)
or a :class:`Control` (acceleration and steering rate, applied directly).

Frames
------
``Plan.frame`` names the sensor the waypoints are relative to:

* ``rear_axle`` – vehicle frame (nuScenes ego / nuPlan / NAVSIM convention),
* ``center``    – geometric centre of the ego box,
* ``lidar``     – the LIDAR_TOP mount of the scene rig (UniAD / VAD),
* any camera name in the rig.

``Plan.axes`` is ``"flu"`` (x forward, y left; default) or ``"rfu"``
(x right, y forward – the legacy UniAD/VAD layout).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np


@dataclass
class Plan:
    waypoints: np.ndarray                 # (N, 2) or (N, 3) with heading
    dt: float = 0.5                        # seconds between waypoints (first at +dt)
    frame: str = "rear_axle"
    axes: str = "flu"
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = np.asarray(self.waypoints, dtype=np.float64)
        if points.ndim != 2 or len(points) < 1 or points.shape[1] < 2:
            raise ValueError(f"plan waypoints must be (N, 2+), got {points.shape}")
        if not np.isfinite(points).all():
            raise ValueError("plan waypoints contain non-finite values")
        if self.axes not in ("flu", "rfu"):
            raise ValueError("Plan.axes must be 'flu' or 'rfu'")
        self.waypoints = points

    def forward_left(self) -> np.ndarray:
        """(N, 2) waypoints as (forward, left) in ``self.frame``."""
        xy = self.waypoints[:, :2]
        return xy.copy() if self.axes == "flu" else np.column_stack([xy[:, 1], -xy[:, 0]])


@dataclass
class Control:
    accel: float                           # m/s², longitudinal
    steer_rate: float                      # rad/s, positive = steer left


Action = Union[Plan, Control, None]


@dataclass
class Observation:
    """What an ego agent sees at one planning step.

    ``images`` maps camera name → (H, W, 3) uint8 RGB.  ``cameras`` gives, per
    camera, the intrinsics ``K`` (3×3, for the delivered image size) and the
    ``mount`` (4×4 camera-to-vehicle, OpenCV camera axes, vehicle = rear axle
    FLU).  ``ego`` is in the road frame.  ``info`` is the full environment info
    (agents, events, route progress …).
    """
    time: float
    step: int
    images: dict[str, np.ndarray]
    cameras: dict[str, dict[str, Any]]
    ego: dict[str, float]
    command: int                           # 0 right, 1 left, 2 straight
    info: dict[str, Any]
    lidar: np.ndarray | None = None
    lidar_mount: np.ndarray | None = None


class EgoAgent:
    """In-process ego agent.  Subclass and override :meth:`act`."""

    #: cameras the agent needs (``None`` = all rig cameras)
    cameras: list[str] | None = None
    #: deliver images at this (width, height); intrinsics are rescaled
    image_size: tuple[int, int] | None = None
    needs_lidar: bool = False

    def setup(self, context: dict[str, Any]) -> None:
        """Called once per run with scene/vehicle context."""

    def reset(self) -> None:
        """Called at the start of each episode."""

    def act(self, obs: Observation) -> Action:
        raise NotImplementedError

    def close(self) -> None:
        """Release resources."""
