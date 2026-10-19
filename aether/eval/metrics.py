"""Per-episode closed-loop metrics.

The sub-scores follow the PDM score used by NAVSIM / nuPlan, so numbers are
comparable in spirit with those benchmarks (they are computed on AETHER's
re-simulated scenes, not on the original logs):

====  ================================================================
NC    no at-fault collision: 1, collision with static scene only: 0.5,
      at-fault collision with a vehicle: 0
DAC   drivable-area compliance: 0 if the ego box ever leaves the HD-map
      drivable area by more than the tolerance, else 1
TTC   time-to-collision: 0 if, at any step, constant-velocity projection
      predicts contact with a vehicle in front within 0.9 s, else 1
C     comfort: 1 if longitudinal/lateral acceleration, jerk, yaw rate
      and yaw acceleration stay within the nuPlan bounds, else 0
EP    ego progress: route progress / progress of the logged expert over
      the same time (clipped to [0, 1])
RC    route completion: fraction of the logged route driven
====  ================================================================

``score = NC × DAC × (5·TTC + 2·C + 5·EP) / 12``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any

import numpy as np

from ..geometry.boxes import Box2D, overlap_ratio

COMFORT = {"lon_accel": (-4.05, 2.40), "lat_accel": 4.89, "jerk": 4.13,
           "yaw_rate": 0.95, "yaw_accel": 1.93}
TTC_HORIZON = 0.9


@dataclass
class EpisodeMetrics:
    nc: float
    dac: float
    ttc: float
    comfort: float
    ep: float
    rc: float
    score: float
    duration_s: float
    distance_m: float
    mean_speed: float
    collisions: int
    at_fault_collisions: int
    static_collisions: int
    off_road_steps: int
    min_ttc_s: float | None              # None: no vehicle ahead on a collision course
    terminated_by: str | None
    comfort_detail: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ttc(ego: dict, agents: list[dict], horizon: float = TTC_HORIZON, dt: float = 0.1) -> float:
    """Smallest time to contact with a vehicle ahead (inf if none)."""
    best = math.inf
    c, s = math.cos(ego["yaw"]), math.sin(ego["yaw"])
    for a in agents:
        dx, dy = a["x"] - ego["x"], a["y"] - ego["y"]
        if dx * c + dy * s <= 0:                       # only vehicles in front
            continue
        if math.hypot(dx, dy) > 30:
            continue
        for k in range(int(round(horizon / dt)) + 1):
            t = k * dt
            e = Box2D(ego["x"] + ego["speed"] * c * t, ego["y"] + ego["speed"] * s * t,
                      ego["yaw"], ego["length"], ego["width"])
            o = Box2D(a["x"] + a["speed"] * math.cos(a["yaw"]) * t,
                      a["y"] + a["speed"] * math.sin(a["yaw"]) * t, a["yaw"], a["length"], a["width"])
            if overlap_ratio(e, o) > 0:
                best = min(best, t)
                break
    return best


def _comfort(times: np.ndarray, ego: list[dict]) -> dict[str, float]:
    if len(ego) < 3:
        return {k: 0.0 for k in ("lon_accel_min", "lon_accel_max", "lat_accel", "jerk", "yaw_rate", "yaw_accel")}
    t = np.asarray(times)
    v = np.array([e["speed"] for e in ego])
    yaw = np.unwrap([e["yaw"] for e in ego])
    acc = np.gradient(v, t)
    jerk = np.gradient(acc, t)
    yaw_rate = np.gradient(yaw, t)
    yaw_acc = np.gradient(yaw_rate, t)
    return {"lon_accel_min": float(acc.min()), "lon_accel_max": float(acc.max()),
            "lat_accel": float(np.abs(v * yaw_rate).max()), "jerk": float(np.abs(jerk).max()),
            "yaw_rate": float(np.abs(yaw_rate).max()), "yaw_accel": float(np.abs(yaw_acc).max())}


def score_episode(steps: list[tuple[float, int, dict[str, dict]]], events: list[dict],
                  route, expert_progress_at, terminated_by: str | None = None) -> EpisodeMetrics:
    """``steps`` from :class:`~aether.record.trajectory.TrajectoryRecorder`.

    ``expert_progress_at(t)`` returns the route fraction the logged vehicle
    had covered at time ``t``.
    """
    times = np.array([s[0] for s in steps])
    ego = [s[2]["ego"] for s in steps]
    vehicles = [s[2] for s in steps]

    vehicle_hits = [e for e in events if e["type"] == "vehicle_collision"]
    at_fault = [e for e in vehicle_hits if e.get("at_fault")]
    static_hits = [e for e in events if e["type"] == "static_collision"]
    off_road = [e for e in events if e["type"] == "off_road"]
    nc = 0.0 if at_fault else (0.5 if static_hits else 1.0)
    dac = 0.0 if off_road else 1.0

    min_ttc = math.inf
    for states in vehicles[1:]:
        others = [v for k, v in states.items() if k != "ego" and v.get("type") != "pedestrian"]
        min_ttc = min(min_ttc, _ttc(states["ego"], others))
    ttc = 0.0 if min_ttc < TTC_HORIZON else 1.0

    detail = _comfort(times, ego)
    ok = (COMFORT["lon_accel"][0] <= detail["lon_accel_min"] and detail["lon_accel_max"] <= COMFORT["lon_accel"][1]
          and detail["lat_accel"] <= COMFORT["lat_accel"] and detail["jerk"] <= COMFORT["jerk"]
          and detail["yaw_rate"] <= COMFORT["yaw_rate"] and detail["yaw_accel"] <= COMFORT["yaw_accel"])
    comfort = 1.0 if ok else 0.0

    rc = route.progress(ego[-1]["x"], ego[-1]["y"]) if ego else 0.0
    expert = float(expert_progress_at(float(times[-1]))) if len(times) else 0.0
    ep = 1.0 if expert < 1e-3 else float(np.clip(rc / expert, 0.0, 1.0))
    score = nc * dac * (5 * ttc + 2 * comfort + 5 * ep) / 12.0

    xy = np.array([[e["x"], e["y"]] for e in ego])
    distance = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
    duration = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    return EpisodeMetrics(
        nc=nc, dac=dac, ttc=ttc, comfort=comfort, ep=round(ep, 4), rc=round(rc, 4),
        score=round(score, 4), duration_s=duration, distance_m=round(distance, 3),
        mean_speed=round(distance / duration, 3) if duration > 0 else 0.0,
        collisions=len({e.get("other") for e in vehicle_hits}),
        at_fault_collisions=len({e.get("other") for e in at_fault}),
        static_collisions=len(static_hits), off_road_steps=len({e["step"] for e in off_road}),
        min_ttc_s=round(min_ttc, 3) if math.isfinite(min_ttc) else None,
        terminated_by=terminated_by, comfort_detail={k: round(v, 3) for k, v in detail.items()})
