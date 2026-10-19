"""Waypoint tracking: :class:`Plan` → (acceleration, steering rate).

The plan is converted to a rear-axle reference trajectory and tracked with
the iLQR solver (nuPlan formulation) using the ego vehicle's own wheelbase
and limits, so a long minivan and a compact car follow the same plan
differently, as they would in reality.
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


class TrajectoryTracker:
    def __init__(self, profile: VehicleProfile) -> None:
        self.profile = profile

    def control(self, plan: Plan, speed: float, steer: float,
                origin: np.ndarray = (0.0, 0.0)) -> tuple[float, float]:
        """Return (accel, steer_rate) for the first control interval."""
        ref = rear_axle_reference(plan, np.asarray(origin))
        states = np.zeros((len(ref), 5))
        states[:, :3] = ref
        p = self.profile
        solver = _solver(round(float(plan.dt), 6), float(p.wheelbase), float(p.max_accel),
                         float(p.max_steer), float(p.max_steer_rate))
        current = np.array([0.0, 0.0, 0.0, float(speed), float(steer)])
        inputs = solver.solve(current, states)[-1].input_trajectory
        return float(inputs[0, 0]), float(inputs[0, 1])
