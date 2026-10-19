"""Logged route of a scene: progress along it and driving commands.

End-to-end planners such as UniAD and VAD condition on a navigation command
(``RIGHT=0``, ``LEFT=1``, ``STRAIGHT=2``).  It is derived from where the
recorded vehicle went next: the lateral offset of the logged route
``lookahead_s`` seconds ahead, measured in the ego's current heading frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

RIGHT, LEFT, STRAIGHT = 0, 1, 2
COMMAND_NAMES = {RIGHT: "right", LEFT: "left", STRAIGHT: "straight"}


@dataclass
class Route:
    xy: np.ndarray            # (T, 2) logged positions (vehicle box centre)
    frame_dt: float
    lookahead_s: float = 3.0
    lateral_threshold: float = 2.0

    def __post_init__(self) -> None:
        segments = np.diff(self.xy, axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        keep = lengths > 1e-6
        self._starts = self.xy[:-1][keep]
        self._segments = segments[keep]
        self._lengths = lengths[keep]
        self._index = np.flatnonzero(keep)
        self._cum = np.concatenate([[0.0], np.cumsum(self._lengths)])
        self.length = float(self._cum[-1])

    def project(self, x: float, y: float) -> tuple[float, int, float]:
        """(arc length, nearest logged frame, lateral distance)."""
        if not len(self._lengths):
            return 0.0, 0, float(np.hypot(x - self.xy[0, 0], y - self.xy[0, 1]))
        p = np.array([x, y])
        t = np.clip(np.sum((p - self._starts) * self._segments, 1) / self._lengths ** 2, 0, 1)
        foot = self._starts + t[:, None] * self._segments
        d = np.linalg.norm(foot - p, axis=1)
        i = int(np.argmin(d))
        return float(self._cum[i] + t[i] * self._lengths[i]), int(self._index[i]), float(d[i])

    def progress(self, x: float, y: float) -> float:
        """Route completion in [0, 1]."""
        s, _, _ = self.project(x, y)
        return float(np.clip(s / max(self.length, 1e-9), 0.0, 1.0))

    def command(self, x: float, y: float, yaw: float) -> int:
        _, frame, _ = self.project(x, y)
        ahead = min(len(self.xy) - 1, frame + max(1, int(round(self.lookahead_s / self.frame_dt))))
        dx, dy = self.xy[ahead] - (x, y)
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy        # left positive
        if lateral >= self.lateral_threshold:
            return LEFT
        if lateral <= -self.lateral_threshold:
            return RIGHT
        return STRAIGHT
