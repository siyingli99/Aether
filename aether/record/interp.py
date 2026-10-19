"""Kinematic interpolation between simulation steps.

Positions use cubic Hermite splines whose end tangents are the velocity
vectors (speed × heading), so the interpolated path is C¹ and consistent with
each vehicle's motion; heading follows the short arc and speed is linear.
"""
from __future__ import annotations

import math

import numpy as np


def hermite(p0, v0, p1, v1, dt: float, u: np.ndarray) -> np.ndarray:
    """Positions at fractions ``u`` ∈ [0, 1] of the interval (vectorised)."""
    u = np.asarray(u, dtype=float)[:, None]
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return (h00 * np.asarray(p0) + h10 * dt * np.asarray(v0)
            + h01 * np.asarray(p1) + h11 * dt * np.asarray(v1))


def interpolate_state(a: dict, b: dict, dt: float, u: float) -> dict:
    """Interpolate two state dicts with keys x, y, yaw, speed (others from ``a``)."""
    if u <= 0.0:
        return dict(a)
    if u >= 1.0:
        return dict(b)
    va = (a["speed"] * math.cos(a["yaw"]), a["speed"] * math.sin(a["yaw"]))
    vb = (b["speed"] * math.cos(b["yaw"]), b["speed"] * math.sin(b["yaw"]))
    jump = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
    if jump > max(5.0, 3.0 * dt * max(a["speed"], b["speed"], 1.0)):
        out = dict(a if u < 0.5 else b)       # teleport (respawn): no smoothing
        return out
    x, y = hermite((a["x"], a["y"]), va, (b["x"], b["y"]), vb, dt, np.array([u]))[0]
    dyaw = math.remainder(b["yaw"] - a["yaw"], 2 * math.pi)
    out = dict(a)
    out.update(x=float(x), y=float(y), yaw=float(math.remainder(a["yaw"] + u * dyaw, 2 * math.pi)),
               speed=float(a["speed"] + u * (b["speed"] - a["speed"])))
    return out
