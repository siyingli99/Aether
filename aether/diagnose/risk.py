"""Stage 1.A — risk function and critical-window extraction.

Implements

    Risk_t = R_phys_t * C_cog_t
    R_phys_t = w1 * P_TTC_t + w2 * PSD(t)
    C_cog_t  = exp(-lambda_cog * H(pi_t))
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import math
import numpy as np


@dataclass
class RiskWeights:
    w_ttc: float = 0.7
    w_psd: float = 0.3
    lambda_cog: float = 0.5
    ttc_horizon: float = 5.0
    psd_horizon: float = 30.0
    plan_dt: float = 0.25


def _bev_distance(ego_box, obj_box):
    return math.hypot(float(ego_box[0]) - float(obj_box[0]),
                      float(ego_box[1]) - float(obj_box[1]))


def _bev_radius(box):
    return 0.5 * math.hypot(float(box[3]), float(box[4]))


def _ttc_one(ego_box, obj_box, ego_velo, ttc_horizon):
    ex, ey = ego_box[0], ego_box[1]
    ox, oy = obj_box[0], obj_box[1]
    yaw = -float(ego_box[6])
    fwd = (math.cos(yaw), math.sin(yaw))
    rel = (ox - ex, oy - ey)
    along = rel[0] * fwd[0] + rel[1] * fwd[1]
    if along <= 0:
        return ttc_horizon
    gap = max(0.0, math.hypot(*rel) - _bev_radius(ego_box) - _bev_radius(obj_box))
    if ego_velo <= 1e-3:
        return ttc_horizon
    return min(ttc_horizon, gap / ego_velo)


def _psd_one(ego_box, obj_box):
    return max(0.0, _bev_distance(ego_box, obj_box)
                    - _bev_radius(ego_box) - _bev_radius(obj_box))


def _trajectory_entropy(plan_traj):
    p = np.asarray(plan_traj, dtype=np.float64)
    if p.ndim != 2 or p.shape[0] < 2:
        return 0.0
    diffs = np.diff(p, axis=0)
    headings = np.arctan2(diffs[:, 0], diffs[:, 1] + 1e-9)
    hist, _ = np.histogram(headings, bins=8, range=(-math.pi, math.pi))
    s = hist.sum()
    if s == 0:
        return 0.0
    pp = hist / s
    pp = pp[pp > 0]
    return float(-np.sum(pp * np.log(pp)))


def compute_risk(step: Dict[str, Any], weights: RiskWeights = RiskWeights()):
    ego_box = step.get("ego_box")
    obj_boxes = step.get("obj_boxes") or []
    ego_velo = float(step.get("ego_velo", 0.0))
    plan_traj = step.get("plan_traj")

    if not obj_boxes or ego_box is None:
        ttc = weights.ttc_horizon
        psd_v = weights.psd_horizon
    else:
        ttc = min(_ttc_one(ego_box, ob, ego_velo, weights.ttc_horizon) for ob in obj_boxes)
        psd_v = min(_psd_one(ego_box, ob) for ob in obj_boxes)

    p_ttc = 1.0 - ttc / max(weights.ttc_horizon, 1e-6)
    p_psd = 1.0 - min(1.0, psd_v / max(weights.psd_horizon, 1e-6))
    r_phys = weights.w_ttc * p_ttc + weights.w_psd * p_psd

    if plan_traj is None:
        c_cog = 1.0
    else:
        h = _trajectory_entropy(np.asarray(plan_traj))
        c_cog = math.exp(-weights.lambda_cog * h)

    risk = float(r_phys * c_cog)
    return risk, {
        "p_ttc": float(p_ttc), "psd": float(p_psd),
        "r_phys": float(r_phys), "c_cog": float(c_cog),
        "ttc_raw": float(ttc),
    }


@dataclass
class CriticalWindow:
    t_critical: int
    start: int
    end: int
    peak_risk: float
    per_step_risk: List[float]


def extract_critical_window(trace, k_back: int = 5, f_fwd: int = 5,
                            weights: RiskWeights = RiskWeights()):
    if len(trace) == 0:
        raise ValueError("Empty trace; cannot extract critical window.")
    risks = [compute_risk(s, weights)[0] for s in trace]
    t_c = int(np.argmax(risks))
    return CriticalWindow(
        t_critical=t_c, start=max(0, t_c - int(k_back)),
        end=min(len(trace) - 1, t_c + int(f_fwd)),
        peak_risk=float(risks[t_c]), per_step_risk=risks,
    )
