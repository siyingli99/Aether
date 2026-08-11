"""Stage 2 -- causal target localisation via 3-layer ablation.

Replaces the (high-variance) Monte-Carlo Shapley estimator with three
causal probes per candidate actor:

  L1 Removal       hide actor i from the rendered scene -> Delta plan / risk
  L2 Displacement  shift actor i by +-delta_lon / +-delta_lat -> Delta plan
  L3 Class swap    re-key actor i to the alternative class (RigidNodes <-> SMPLNodes)
                   -- if alternative class has no instances, this layer is skipped
                   and recorded as ``skipped=True`` in the breakdown.

Per-actor causal score:

    Phi_i = a1 * |Delta peak_risk| + a2 * mean ||Delta plan|| + a3 * |Delta TTC|

The top-1 actor becomes the PGD target. The full per-actor breakdown
(L1/L2/L3 + Phi) is exported to the report and visualised as stacked bars.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .probes import compute_step_probes


# rollout(actor_keep, perturbation, max_steps) -> trace
RolloutFn = Callable[..., List[Dict]]


@dataclass
class CausalScores:
    actor_id: int
    actor_class: str
    L1_removal: float
    L2_displacement: float
    L3_class_swap: float
    L3_skipped: bool
    delta_peak_risk: float
    delta_plan_norm: float
    delta_ttc: float
    phi: float
    breakdown: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalLocalisation:
    target_index_in_pool: int
    target_class: str
    target_id: int
    per_actor: List[CausalScores]
    base_trace_summary: Dict[str, float]
    weights: Tuple[float, float, float]


def _trace_summary(trace: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    return compute_step_probes(trace)


def _delta_metrics(trace_pert: Sequence[Dict[str, Any]],
                   trace_base: Sequence[Dict[str, Any]]) -> Tuple[float, float, float]:
    n = min(len(trace_pert), len(trace_base))
    if n == 0:
        return 0.0, 0.0, 0.0
    p_pert = _trace_summary(trace_pert)
    p_base = _trace_summary(trace_base)
    # peak risk proxy = peak of (1/ttc) over window
    rp = float(np.max(1.0 / np.maximum(p_pert["ttc"][:n], 1e-3)))
    rb = float(np.max(1.0 / np.maximum(p_base["ttc"][:n], 1e-3)))
    delta_risk = abs(rp - rb)

    # plan norm delta
    diffs = []
    for t in range(n):
        pa = trace_pert[t].get("plan_traj"); pb = trace_base[t].get("plan_traj")
        if pa is None or pb is None: continue
        a = np.asarray(pa, dtype=np.float64); b = np.asarray(pb, dtype=np.float64)
        if a.shape == b.shape:
            diffs.append(float(np.linalg.norm(a - b)))
    delta_plan = float(np.mean(diffs)) if diffs else 0.0

    # ttc delta (mean)
    delta_ttc = float(np.mean(np.abs(p_pert["ttc"][:n] - p_base["ttc"][:n])))
    return delta_risk, delta_plan, delta_ttc


def causal_target_localisation(
    actor_pool: Sequence[Tuple[str, int]],
    rollout_keep: Callable[[Sequence[int]], List[Dict]],
    rollout_displace: Optional[Callable[[int, np.ndarray], List[Dict]]] = None,
    rollout_swap: Optional[Callable[[int, str], List[Dict]]] = None,
    weights: Tuple[float, float, float] = (1.0, 1.0, 0.5),
    max_steps_window: Optional[int] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> CausalLocalisation:
    """Run the 3-layer ablation and return the highest-scoring actor.

    ``rollout_displace(idx, delta_xyz)`` and ``rollout_swap(idx, new_class)`` are
    optional. When omitted the corresponding layer score is 0 (skipped).
    """
    n = len(actor_pool)
    if n == 0:
        raise ValueError("actor_pool is empty.")

    a1, a2, a3 = weights
    full_keep = list(range(n))
    log = progress_cb or (lambda *_: None)

    log(f"[causal] L0 baseline rollout (keep all {n} actors)")
    base_trace = rollout_keep(full_keep)
    base_probes = _trace_summary(base_trace)

    per: List[CausalScores] = []
    for i, (cls, inst) in enumerate(actor_pool):
        log(f"[causal] actor {i+1}/{n}  ({cls}#{inst})  L1 removal...")
        keep = [k for k in full_keep if k != i]
        trace_l1 = rollout_keep(keep)
        d_risk, d_plan, d_ttc = _delta_metrics(trace_l1, base_trace)
        L1 = float(a1 * d_risk + a2 * d_plan + a3 * d_ttc)

        # L2 displacement
        L2 = 0.0
        L2_break: Dict[str, float] = {}
        if rollout_displace is not None:
            log(f"[causal] actor {i+1}/{n}  L2 displacement (+lat / -lat / +lon / -lon)")
            for tag, dxyz in [
                ("lat+1m", np.array([0.0, 1.0, 0.0])),
                ("lat-1m", np.array([0.0, -1.0, 0.0])),
                ("lon+2m", np.array([2.0, 0.0, 0.0])),
                ("lon-2m", np.array([-2.0, 0.0, 0.0])),
            ]:
                try:
                    tr = rollout_displace(i, dxyz)
                    dr, dp, dt = _delta_metrics(tr, base_trace)
                    score = float(a1 * dr + a2 * dp + a3 * dt)
                    L2_break[tag] = score
                    L2 = max(L2, score)
                except Exception as e:
                    L2_break[tag] = 0.0
                    log(f"  [causal] displacement {tag} skipped: {e}")

        # L3 class swap
        L3 = 0.0; L3_skipped = True
        if rollout_swap is not None:
            other = "SMPLNodes" if cls == "RigidNodes" else "RigidNodes"
            log(f"[causal] actor {i+1}/{n}  L3 class swap ({cls} -> {other})")
            try:
                tr = rollout_swap(i, other)
                if tr is not None:
                    dr, dp, dt = _delta_metrics(tr, base_trace)
                    L3 = float(a1 * dr + a2 * dp + a3 * dt)
                    L3_skipped = False
            except Exception as e:
                log(f"  [causal] L3 skipped: {e}")

        # final phi: weighted aggregate of the three layers
        phi = float(L1 + 0.5 * L2 + 0.3 * L3)
        per.append(CausalScores(
            actor_id=int(inst), actor_class=cls,
            L1_removal=L1, L2_displacement=L2, L3_class_swap=L3,
            L3_skipped=L3_skipped,
            delta_peak_risk=d_risk, delta_plan_norm=d_plan, delta_ttc=d_ttc,
            phi=phi, breakdown={"L2": L2_break},
        ))
        log(f"[causal] actor {i+1}/{n}  phi={phi:.4f} (L1={L1:.3f} L2={L2:.3f} L3={L3:.3f})")

    target_idx = int(np.argmax([p.phi for p in per]))
    cls, inst = actor_pool[target_idx]
    return CausalLocalisation(
        target_index_in_pool=target_idx,
        target_class=str(cls), target_id=int(inst),
        per_actor=per,
        base_trace_summary={
            "n_steps": float(len(base_trace)),
            "mean_ttc": float(np.mean(base_probes["ttc"])) if base_probes["ttc"].size else 0.0,
            "mean_flicker": float(np.mean(base_probes["plan_flicker"])),
        },
        weights=tuple(weights),
    )
