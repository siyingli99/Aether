"""Multi-channel composite plan loss (L_v2).

L_v2 = w_collision * L_collision        (BEV gap penalty)
     + w_offroad   * L_offroad          (lateral plan_traj excess)
     + w_kin       * L_kin              (jerk threshold excess)
     + w_route     * L_route             (route-completion shortfall)
     + w_flicker   * L_flicker          (plan instability)        NEW
     + w_ttc       * L_ttc              (TTC < threshold)         NEW
     + w_hesitate  * L_hesitate         (acceleration oscillation) NEW
     + w_plan_dev  * L_plan_dev         (deviation from baseline) NEW
     + w_pred_gap  * L_pred_gap         (predicted-vs-GT IoU gap) NEW

The signal is *higher when the planner fails worse*. Sub-terms are computed
from the trace plus optional ``baseline_trace`` for plan_dev.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import math
import numpy as np

from .probes import compute_step_probes


@dataclass
class PlanLossWeights:
    w_collision: float = 1.0
    w_offroad: float = 0.3
    w_kin: float = 0.2
    w_route: float = 0.5
    w_flicker: float = 0.4
    w_ttc: float = 0.6
    w_hesitate: float = 0.2
    w_plan_dev: float = 0.5
    w_pred_gap: float = 0.3
    lat_threshold: float = 1.75
    jerk_threshold: float = 4.0
    rc_target: float = 1.0
    ttc_safe: float = 2.5      # TTC below this contributes
    ttc_horizon: float = 5.0


def _collision_term(trace: Sequence[Dict[str, Any]]) -> float:
    gaps: List[float] = []
    for s in trace:
        ego = s.get("ego_box")
        objs = s.get("obj_boxes") or []
        if ego is None or not objs:
            gaps.append(1.0); continue
        ex, ey = float(ego[0]), float(ego[1])
        er = 0.5 * math.hypot(float(ego[3]), float(ego[4]))
        local = []
        for ob in objs:
            ox, oy = float(ob[0]), float(ob[1])
            orad = 0.5 * math.hypot(float(ob[3]), float(ob[4]))
            local.append(max(0.0, math.hypot(ex - ox, ey - oy) - er - orad))
        gaps.append(min(local) if local else 1.0)
    if not gaps:
        return 0.0
    g = np.asarray(gaps, dtype=np.float64)
    return float(np.mean(1.0 - np.clip(g, 0.0, 1.0)))


def _offroad_term(trace, lat_thresh):
    total = 0.0; n = 0
    for s in trace:
        plan = s.get("plan_traj")
        if plan is None: continue
        p = np.asarray(plan, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] < 2: continue
        excess = np.maximum(0.0, np.abs(p[:, 0]) - lat_thresh)
        total += float(excess.mean()) if excess.size else 0.0
        n += 1
    return total / max(n, 1)


def _kin_term(trace, jerk_thresh):
    accs = [float(s.get("accelerate", 0.0)) for s in trace]
    if len(accs) < 2: return 0.0
    jerks = np.abs(np.diff(np.asarray(accs)))
    excess = np.maximum(0.0, jerks - jerk_thresh)
    return float(excess.mean())


def _route_term(trace, rc_target):
    last = trace[-1] if trace else {}
    rc = float(last.get("rc", 0.0))
    return float(max(0.0, rc_target - rc))


def plan_loss(trace: Sequence[Dict[str, Any]],
              weights: PlanLossWeights = PlanLossWeights(),
              baseline_trace: Optional[Sequence[Dict[str, Any]]] = None,
              return_breakdown: bool = False):
    if not trace:
        return ({}, 0.0) if return_breakdown else 0.0
    probes = compute_step_probes(trace, baseline_trace=baseline_trace,
                                 ttc_horizon=weights.ttc_horizon)
    L_coll = _collision_term(trace)
    L_off  = _offroad_term(trace, weights.lat_threshold)
    L_kin  = _kin_term(trace, weights.jerk_threshold)
    L_rou  = _route_term(trace, weights.rc_target)
    L_fli  = float(np.mean(probes["plan_flicker"]))
    L_ttc  = float(np.mean(np.maximum(0.0, weights.ttc_safe - probes["ttc"])))
    L_hes  = float(np.mean(probes["acc_oscillate"]))
    L_dev  = float(np.mean(probes["plan_dev"])) if baseline_trace is not None else 0.0
    L_pgap = float(np.mean(probes["pred_iou_gap"]))

    L = (weights.w_collision * L_coll
         + weights.w_offroad   * L_off
         + weights.w_kin       * L_kin
         + weights.w_route     * L_rou
         + weights.w_flicker   * L_fli
         + weights.w_ttc       * L_ttc
         + weights.w_hesitate  * L_hes
         + weights.w_plan_dev  * L_dev
         + weights.w_pred_gap  * L_pgap)

    if return_breakdown:
        return (
            {
                "L_collision": L_coll, "L_offroad": L_off, "L_kin": L_kin,
                "L_route": L_rou, "L_flicker": L_fli, "L_ttc_neg": L_ttc,
                "L_hesitate": L_hes, "L_plan_dev": L_dev, "L_pred_gap": L_pgap,
            },
            float(L),
        )
    return float(L)
