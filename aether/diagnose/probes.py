"""Observability probes: extract multi-channel signals from a rollout trace.

The diagnosis engine no longer relies solely on plan_loss. Each step in
the trace is augmented with a probe vector that captures:

  * plan_flicker  -- ||plan_t - plan_{t-1}||_F (planner instability)
  * ttc           -- minimum time-to-collision over actors
  * acc_jerk      -- |acc_t - acc_{t-1}|
  * acc_oscillate -- sign-changes of acc inside a sliding window
  * pred_iou_gap  -- IoU(model predicted boxes, GT obj_boxes)
                    (only available when adapter returns ``agent_states``)
  * plan_dev      -- ||plan_t - plan_t^baseline||_F (vs baseline trace)
  * action_norm   -- |acc| + |steer_rate|

Probes operate on the same trace dict format as risk.py / losses.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


CHANNEL_NAMES: Tuple[str, ...] = (
    "plan_flicker", "ttc", "acc_jerk", "acc_oscillate",
    "pred_iou_gap", "plan_dev", "action_norm",
)


def _bev_radius(box: Sequence[float]) -> float:
    w, l = float(box[3]), float(box[4])
    return 0.5 * math.hypot(w, l)


def _ttc_one(ego_box: Sequence[float], obj_box: Sequence[float],
             ego_velo: float, horizon: float = 5.0) -> float:
    ex, ey = ego_box[0], ego_box[1]
    ox, oy = obj_box[0], obj_box[1]
    yaw = -float(ego_box[6])
    fwd = (math.cos(yaw), math.sin(yaw))
    rel = (ox - ex, oy - ey)
    along = rel[0] * fwd[0] + rel[1] * fwd[1]
    if along <= 0:
        return horizon
    gap = max(0.0, math.hypot(*rel) - _bev_radius(ego_box) - _bev_radius(obj_box))
    if ego_velo <= 1e-3:
        return horizon
    return min(horizon, gap / ego_velo)


def _box_iou_bev(b1: Sequence[float], b2: Sequence[float]) -> float:
    """Axis-aligned BEV IoU approximation (uses w*l footprints, ignores yaw)."""
    x1, y1 = b1[0], b1[1]; w1, l1 = b1[3], b1[4]
    x2, y2 = b2[0], b2[1]; w2, l2 = b2[3], b2[4]
    ix1 = max(x1 - w1 / 2, x2 - w2 / 2)
    iy1 = max(y1 - l1 / 2, y2 - l2 / 2)
    ix2 = min(x1 + w1 / 2, x2 + w2 / 2)
    iy2 = min(y1 + l1 / 2, y2 + l2 / 2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = w1 * l1 + w2 * l2 - inter
    return float(inter / union) if union > 1e-9 else 0.0


def compute_step_probes(trace: Sequence[Dict[str, Any]],
                        baseline_trace: Optional[Sequence[Dict[str, Any]]] = None,
                        ttc_horizon: float = 5.0) -> Dict[str, np.ndarray]:
    """Return per-channel time series of length len(trace)."""
    n = len(trace)
    out = {k: np.zeros(n, dtype=np.float64) for k in CHANNEL_NAMES}
    if n == 0:
        return out

    # plan_flicker
    last_plan = None
    for t, s in enumerate(trace):
        plan = s.get("plan_traj")
        if plan is None:
            out["plan_flicker"][t] = 0.0
        else:
            p = np.asarray(plan, dtype=np.float64)
            if last_plan is None or last_plan.shape != p.shape:
                out["plan_flicker"][t] = 0.0
            else:
                out["plan_flicker"][t] = float(np.linalg.norm(p - last_plan))
            last_plan = p

    # ttc
    for t, s in enumerate(trace):
        ego = s.get("ego_box"); objs = s.get("obj_boxes") or []
        ego_velo = float(s.get("ego_velo", 0.0))
        if ego is None or not objs:
            out["ttc"][t] = ttc_horizon
        else:
            out["ttc"][t] = min(_ttc_one(ego, ob, ego_velo, ttc_horizon) for ob in objs)

    # acc_jerk + acc_oscillate
    accs = np.array([float(s.get("accelerate", 0.0)) for s in trace])
    jerks = np.zeros(n)
    if n >= 2:
        jerks[1:] = np.abs(np.diff(accs))
    out["acc_jerk"] = jerks
    win = 5
    for t in range(n):
        lo = max(0, t - win); hi = min(n, t + 1)
        seg = accs[lo:hi]
        signs = np.sign(seg)
        out["acc_oscillate"][t] = float(np.sum(np.abs(np.diff(signs)) > 0)) if seg.size > 1 else 0.0

    # pred_iou_gap (1 - max IoU over GT, averaged)
    for t, s in enumerate(trace):
        preds = s.get("pred_boxes")
        gts = s.get("obj_boxes") or []
        if preds is None or not gts:
            out["pred_iou_gap"][t] = 0.0  # no signal, neutral
        else:
            gaps = []
            for gt in gts:
                best_iou = max((_box_iou_bev(pb, gt) for pb in preds), default=0.0)
                gaps.append(1.0 - best_iou)
            out["pred_iou_gap"][t] = float(np.mean(gaps)) if gaps else 0.0

    # plan_dev (vs baseline trace, frame-aligned)
    if baseline_trace is not None:
        m = min(n, len(baseline_trace))
        for t in range(m):
            p = trace[t].get("plan_traj")
            p0 = baseline_trace[t].get("plan_traj")
            if p is None or p0 is None:
                continue
            p = np.asarray(p, dtype=np.float64); p0 = np.asarray(p0, dtype=np.float64)
            if p.shape == p0.shape:
                out["plan_dev"][t] = float(np.linalg.norm(p - p0))

    # action_norm
    for t, s in enumerate(trace):
        out["action_norm"][t] = float(abs(s.get("accelerate", 0.0))
                                      + abs(s.get("steer_rate", 0.0)))
    return out


def aggregate_channels(probes: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Reduce per-channel time series into scalar summaries."""
    summary = {}
    for k, v in probes.items():
        if v.size == 0:
            summary[k + "_mean"] = 0.0; summary[k + "_max"] = 0.0
        else:
            summary[k + "_mean"] = float(np.mean(v))
            summary[k + "_max"] = float(np.max(v))
    return summary
