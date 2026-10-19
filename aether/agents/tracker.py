"""Waypoint tracking: :class:`Plan` → (acceleration, steering rate).

The plan is converted to a rear-axle reference trajectory and tracked with
the ego vehicle's own wheelbase and limits, so a long minivan and a compact
car follow the same plan differently, as they would in reality.

Two trackers (``sim.tracker``):

* ``pure_pursuit`` (default) – decoupled control as in production AV stacks:
  the longitudinal loop tracks the plan's speed profile, the lateral loop
  picks the steering arc that best overlays the plan's *path* over the next
  1.5 s of travel (arc-length matched preview, a pure-pursuit variant).  The ego can only
  turn the way the path turns, also when the planner asks for more
  deceleration than the vehicle can deliver.
* ``ilqr`` – the nuPlan iLQR solver on the time-indexed waypoints.  When a
  plan is infeasible (e.g. a stop requested at 60 km/h) the optimiser can
  trade position error for a swerve to the opposite side; kept for
  comparison with v0.1 results.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

from ..control.ilqr.lqr_solver import ILQRSolver, ILQRSolverParameters, ILQRWarmStartParameters
from ..vehicle import VehicleProfile
from .base import Plan

_MIN_SEGMENT = 0.05      # m; shorter steps keep the previous heading


@lru_cache(maxsize=16)
def _solver(dt: float, wheelbase: float, max_accel: float, max_steer: float,
            max_steer_rate: float) -> ILQRSolver:
    params = ILQRSolverParameters(
        discretization_time=dt,
        state_cost_diagonal_entries=[1.0, 1.0, 10.0, 0.0, 0.0],
        input_cost_diagonal_entries=[1.0, 10.0],
        state_trust_region_entries=[1.0] * 5,
        input_trust_region_entries=[1.0] * 2,
        max_ilqr_iterations=100,
        convergence_threshold=1e-6,
        max_solve_time=None,                  # deterministic: no wall-clock cut-off
        max_acceleration=max_accel,
        max_steering_angle=max_steer,
        max_steering_angle_rate=max_steer_rate,
        min_velocity_linearization=0.01,
        wheelbase=wheelbase,
    )
    warm = ILQRWarmStartParameters(
        k_velocity_error_feedback=0.5,
        k_steering_angle_error_feedback=0.05,
        lookahead_distance_lateral_error=15.0,
        k_lateral_error=0.1,
        jerk_penalty_warm_start_fit=1e-4,
        curvature_rate_penalty_warm_start_fit=1e-2,
    )
    return ILQRSolver(solver_params=params, warm_start_params=warm)


def rear_axle_reference(plan: Plan, origin: np.ndarray) -> np.ndarray:
    """(N+1, 3) rear-axle (x, y, heading) reference, row 0 = current pose.

    ``origin`` is the (forward, left) offset of ``plan.frame`` from the rear
    axle.  Waypoints are the path of that point, so each one is moved back
    to the rear axle along its own heading.
    """
    origin = np.asarray(origin, dtype=float)[:2]
    path = np.vstack([origin, origin + plan.forward_left()])
    headings = np.zeros(len(path))
    if plan.waypoints.shape[1] >= 3 and plan.extra.get("has_heading"):
        headings[1:] = plan.waypoints[:, 2]
    else:
        last = 0.0
        for i in range(1, len(path)):
            d = path[i] - path[i - 1]
            if math.hypot(*d) > _MIN_SEGMENT:
                last = math.atan2(d[1], d[0])
            headings[i] = last
        headings = np.unwrap(headings)
    c, s = np.cos(headings), np.sin(headings)
    rear = path - np.column_stack([c * origin[0] - s * origin[1], s * origin[0] + c * origin[1]])
    rear[0] = 0.0
    headings[0] = 0.0
    return np.column_stack([rear, headings])


TRACKERS = ("pure_pursuit", "ilqr")


class TrajectoryTracker:
    def __init__(self, profile: VehicleProfile, method: str = "pure_pursuit",
                 dt: float = 0.25) -> None:
        if method not in TRACKERS:
            raise ValueError(f"unknown tracker {method!r}; expected one of {TRACKERS}")
        self.profile = profile
        self.method = method
        self.dt = float(dt)                      # control interval (closed-loop step)

    def control(self, plan: Plan, speed: float, steer: float,
                origin: np.ndarray = (0.0, 0.0)) -> tuple[float, float]:
        """Return (accel, steer_rate) for the next control interval."""
        if self.method == "pure_pursuit":
            return self._pure_pursuit(plan, float(speed), float(steer), np.asarray(origin))
        return self._ilqr(plan, speed, steer, origin)

    # ------------------------------------------------------------ pursuit
    def _pure_pursuit(self, plan: Plan, speed: float, steer: float,
                      origin: np.ndarray) -> tuple[float, float]:
        p = self.profile
        ref = rear_axle_reference(plan, origin)
        xy = ref[:, :2]
        seg = np.hypot(*np.diff(xy, axis=0).T)
        # Longitudinal: speed of the plan one second (two waypoints) ahead.
        t_mid = (np.arange(len(seg)) + 0.5) * plan.dt
        v_plan = seg / plan.dt
        horizon = min(1.0, float(t_mid[-1]))
        v_target = float(np.interp(horizon, t_mid, v_plan))
        accel = (v_target - speed) / max(horizon, self.dt)
        accel = min(max(accel, -p.max_decel), p.max_accel)
        # Lateral: arc-length matched preview.  Choose the constant steering
        # angle whose arc best overlays the plan's *path* at 0.5/1.0/1.5 s of
        # travel at the current speed.  The path is compared at equal arc
        # length (not equal time), so a plan that asks for a different speed
        # never turns into a lateral error, and curvature ahead is anticipated.
        s = np.concatenate([[0.0], np.cumsum(seg)])
        heading_end = ref[-1, 2]

        def path_at(arc: np.ndarray) -> np.ndarray:
            if s[-1] < 0.5:                      # (almost) stationary plan: straight ahead
                return np.column_stack([arc, np.zeros_like(arc)])
            inside = np.column_stack([np.interp(arc, s, xy[:, 0]), np.interp(arc, s, xy[:, 1])])
            beyond = np.maximum(arc - s[-1], 0.0)[:, None]
            return inside + beyond * np.array([math.cos(heading_end), math.sin(heading_end)])

        preview = max(speed, 2.0) * np.array([0.5, 1.0, 1.5])
        target = path_at(preview)
        weights = np.array([1.0, 1.0, 0.5])
        limit = p.max_steer
        if speed > 1e-3:
            limit = min(limit, math.atan(p.max_lat_accel * p.wheelbase / (speed * speed)))
        def best(candidates: np.ndarray) -> float:
            k = np.tan(candidates)[:, None] / p.wheelbase
            ks = k * preview[None, :]
            small = np.abs(ks) < 1e-6
            safe_k = np.where(np.abs(k) < 1e-9, 1e-9, k)
            ax = np.where(small, preview[None, :], np.sin(ks) / safe_k)
            ay = np.where(small, 0.5 * k * preview[None, :] ** 2, (1.0 - np.cos(ks)) / safe_k)
            err = (ax - target[None, :, 0]) ** 2 + (ay - target[None, :, 1]) ** 2
            return float(candidates[int(np.argmin(err @ weights))])

        coarse = best(np.linspace(-limit, limit, 61))
        step = 2.0 * limit / 60
        desired = best(np.clip(np.linspace(coarse - step, coarse + step, 41), -limit, limit))
        steer_rate = (desired - steer) / self.dt
        steer_rate = min(max(steer_rate, -p.max_steer_rate), p.max_steer_rate)
        return float(accel), float(steer_rate)

    # --------------------------------------------------------------- iLQR
    def _ilqr(self, plan: Plan, speed: float, steer: float,
              origin: np.ndarray = (0.0, 0.0)) -> tuple[float, float]:
        ref = rear_axle_reference(plan, np.asarray(origin))
        states = np.zeros((len(ref), 5))
        states[:, :3] = ref
        p = self.profile
        solver = _solver(round(float(plan.dt), 6), float(p.wheelbase), float(p.max_accel),
                         float(p.max_steer), float(p.max_steer_rate))
        current = np.array([0.0, 0.0, 0.0, float(speed), float(steer)])
        inputs = solver.solve(current, states)[-1].input_trajectory
        return float(inputs[0, 0]), float(inputs[0, 1])
