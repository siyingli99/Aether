"""Stage 3 -- Dual-track black-box PGD over the 13-D semantic dictionary.

Same PGD update as v1 (FD-grad + sign-step + projection), but generalised
to the 13-D dictionary and the multi-channel plan_loss. Per-iter FD vector
is recorded for downstream sensitivity normalisation.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .losses import plan_loss, PlanLossWeights
from .semantics import (
    SemanticPerturbation,
    SEMANTIC_VAR_NAMES, SEMANTIC_BOUNDS, SEMANTIC_NEUTRAL,
    TOPO_DIMS, PHOTO_DIMS, DIM, neutral_vec, per_axis_range,
)


RolloutWithZ = Callable[[SemanticPerturbation], List[Dict]]


@dataclass
class PGDConfig:
    iters: int = 3
    step_size: float = 0.25
    fd_eps: float = 0.05
    loss_weights: PlanLossWeights = field(default_factory=PlanLossWeights)


@dataclass
class TrackResult:
    track: str
    z_final: SemanticPerturbation
    grad_final: np.ndarray              # (13,)
    grad_history: List[np.ndarray]
    loss_history: List[float]
    base_loss: float
    breakdown_history: List[Dict[str, float]]


@dataclass
class DualTrackResult:
    topo: TrackResult
    photo: TrackResult


def _fd_gradient(rollout: RolloutWithZ, z: SemanticPerturbation,
                 active: Sequence[int], eps_vec: np.ndarray,
                 weights: PlanLossWeights,
                 baseline_trace) -> np.ndarray:
    g = np.zeros(DIM, dtype=np.float64)
    for i in active:
        if eps_vec[i] <= 0:
            continue
        z_plus = SemanticPerturbation(z=z.z.copy()); z_plus.z[i] += eps_vec[i]
        z_minus = SemanticPerturbation(z=z.z.copy()); z_minus.z[i] -= eps_vec[i]
        l_plus = plan_loss(rollout(z_plus.project()),  weights, baseline_trace=baseline_trace)
        l_minus = plan_loss(rollout(z_minus.project()), weights, baseline_trace=baseline_trace)
        g[i] = (l_plus - l_minus) / (2.0 * eps_vec[i])
    return g


def _run_track(rollout: RolloutWithZ, active: Sequence[int],
               cfg: PGDConfig, base_trace, base_loss: float,
               track_name: str) -> TrackResult:
    rng = per_axis_range()
    eps_vec = cfg.fd_eps * rng
    eta_vec = cfg.step_size * rng

    z = SemanticPerturbation.zero()
    history = [base_loss]
    bk_hist = []
    grads = []
    grad = np.zeros(DIM, dtype=np.float64)

    print(f"  [PGD/{track_name}] base_loss={base_loss:.4f}", flush=True, file=sys.stderr)
    for it in range(int(cfg.iters)):
        print(f"  [PGD/{track_name}] iter {it+1}/{cfg.iters}: FD gradient...",
              flush=True, file=sys.stderr)
        grad = _fd_gradient(rollout, z, active, eps_vec, cfg.loss_weights, base_trace)
        grads.append(grad.copy())
        step = np.zeros(DIM, dtype=np.float64)
        step[list(active)] = eta_vec[list(active)] * np.sign(grad[list(active)])
        z = SemanticPerturbation(z=z.z + step).project()
        bk, L = plan_loss(rollout(z), cfg.loss_weights,
                          baseline_trace=base_trace, return_breakdown=True)
        history.append(L)
        bk_hist.append(bk)
        print(f"  [PGD/{track_name}] iter {it+1}/{cfg.iters}: loss={L:.4f}",
              flush=True, file=sys.stderr)

    return TrackResult(
        track=track_name, z_final=z, grad_final=grad, grad_history=grads,
        loss_history=history, base_loss=base_loss, breakdown_history=bk_hist,
    )


def dual_track_pgd(rollout: RolloutWithZ,
                   cfg: PGDConfig = PGDConfig(),
                   base_trace: Optional[List[Dict]] = None) -> DualTrackResult:
    """Run topo-track then photo-track. ``base_trace`` is used as plan_dev anchor."""
    bt = base_trace if base_trace is not None else rollout(SemanticPerturbation.zero())
    base = plan_loss(bt, cfg.loss_weights, baseline_trace=bt)
    topo = _run_track(rollout, TOPO_DIMS, cfg, bt, base, "topo")
    photo = _run_track(rollout, PHOTO_DIMS, cfg, bt, base, "photo")
    return DualTrackResult(topo=topo, photo=photo)
