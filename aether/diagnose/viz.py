"""Stage 6 -- matplotlib visualisations.

All figures are written to ``output_dir/figs/``. We deliberately avoid
seaborn so the package only depends on matplotlib (already in env).

Generated artefacts:

  * risk_curve.png        per-step risk + critical window highlight
  * causal_bars.png       L1/L2/L3 ablation breakdown per actor + Phi
  * sensitivity_heatmap.png    13-D normalised sensitivity, topo|photo blocks
  * pgd_loss_curves.png   loss vs iteration for both tracks + per-channel
  * plan_overlay.png      baseline plan vs perturbed-final plan over time
  * obs_channels.png      multi-channel probe time series

Each function returns the saved path. Failures degrade gracefully so the
report still gets written.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .semantics import SEMANTIC_VAR_NAMES, TOPO_DIMS, PHOTO_DIMS


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_risk_curve(per_step_risk: Sequence[float], window_start: int,
                    window_end: int, t_critical: int, out_path: Path) -> Path:
    arr = np.asarray(per_step_risk, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(arr, lw=1.6, color="#1f77b4", label="per-step risk")
    if arr.size:
        ax.axvspan(window_start, window_end, color="orange", alpha=0.18,
                   label="critical window")
        ax.axvline(t_critical, color="red", ls="--", lw=1.2,
                   label=f"t*={t_critical}")
    ax.set_xlabel("step"); ax.set_ylabel("risk")
    ax.set_title("Risk timeline & critical window")
    ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return out_path


def plot_causal_bars(per_actor: Sequence[Any], out_path: Path) -> Path:
    n = len(per_actor)
    if n == 0:
        return out_path
    labels = [f"{a.actor_class}#{a.actor_id}" for a in per_actor]
    L1 = [a.L1_removal for a in per_actor]
    L2 = [a.L2_displacement for a in per_actor]
    L3 = [a.L3_class_swap for a in per_actor]
    phi = [a.phi for a in per_actor]
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * n), 4))
    ax.bar(x, L1, label="L1 removal", color="#d62728")
    ax.bar(x, L2, bottom=L1, label="L2 displacement (x0.5)", color="#ff7f0e")
    L3_disp = [0.3 * v for v in L3]
    ax.bar(x, L3_disp, bottom=[a + b for a, b in zip(L1, L2)],
           label="L3 class swap (x0.3)", color="#2ca02c")
    for xi, p in zip(x, phi):
        ax.text(xi, p, f"phi={p:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("score"); ax.set_title("Per-actor causal ablation breakdown")
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return out_path


def plot_sensitivity_heatmap(s_vec: np.ndarray, out_path: Path) -> Path:
    s = np.asarray(s_vec).reshape(1, -1)
    fig, ax = plt.subplots(figsize=(10, 1.8))
    im = ax.imshow(s, aspect="auto", cmap="magma")
    ax.set_yticks([]); ax.set_xticks(range(len(SEMANTIC_VAR_NAMES)))
    ax.set_xticklabels(SEMANTIC_VAR_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_title("Normalised sensitivity S_hat (13-D)")
    for x, name in enumerate(SEMANTIC_VAR_NAMES):
        color = "white" if s[0, x] < s.max() * 0.5 else "black"
        ax.text(x, 0, f"{s[0, x]:.2f}", ha="center", va="center",
                color=color, fontsize=7)
    # block divider between topo (0..5) and photo (6..12)
    ax.axvline(len(TOPO_DIMS) - 0.5, color="cyan", lw=1.5)
    fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return out_path


def plot_pgd_curves(dual: Any, out_path: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, track, color in [(axes[0], dual.topo, "#d62728"),
                             (axes[1], dual.photo, "#1f77b4")]:
        ax.plot(track.loss_history, marker="o", color=color)
        ax.set_title(f"PGD {track.track}: loss")
        ax.set_xlabel("iter"); ax.set_ylabel("L_v2")
        ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return out_path


def plot_plan_overlay(base_trace, pert_trace, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, trace, color, alpha in [
        ("baseline", base_trace, "#1f77b4", 0.5),
        ("PGD-final", pert_trace, "#d62728", 0.5),
    ]:
        for s in trace:
            p = s.get("plan_traj")
            if p is None: continue
            p = np.asarray(p)
            if p.ndim == 2 and p.shape[1] >= 2:
                ax.plot(p[:, 0], p[:, 1], color=color, alpha=alpha, lw=0.9)
        ax.plot([], [], color=color, label=label)
    ax.set_xlabel("lateral (m)"); ax.set_ylabel("longitudinal (m)")
    ax.set_title("Plan trajectories: baseline vs PGD-final")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return out_path


def plot_obs_channels(probes: Dict[str, np.ndarray], out_path: Path) -> Path:
    keys = list(probes.keys())
    n = len(keys)
    if n == 0:
        return out_path
    fig, axes = plt.subplots(n, 1, figsize=(8, 1.4 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, k in zip(axes, keys):
        ax.plot(probes[k], lw=1.2)
        ax.set_ylabel(k, fontsize=8); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("step")
    fig.suptitle("Observability probes (per-step)", fontsize=10)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)
    return out_path


def emit_all(output_dir: Path, *,
             per_step_risk, window_start, window_end, t_critical,
             causal_per_actor, sensitivity_vec, dual,
             base_trace, pert_trace, probes) -> Dict[str, str]:
    fig_dir = _ensure_dir(Path(output_dir) / "figs")
    out: Dict[str, str] = {}
    safe = lambda fn, name, **kw: out.setdefault(name, str(fn(out_path=fig_dir / name, **kw)))
    try:
        safe(plot_risk_curve, "risk_curve.png",
             per_step_risk=per_step_risk, window_start=window_start,
             window_end=window_end, t_critical=t_critical)
    except Exception as e:
        out["risk_curve.png"] = f"failed: {e}"
    try:
        safe(plot_causal_bars, "causal_bars.png", per_actor=causal_per_actor)
    except Exception as e:
        out["causal_bars.png"] = f"failed: {e}"
    try:
        safe(plot_sensitivity_heatmap, "sensitivity_heatmap.png", s_vec=sensitivity_vec)
    except Exception as e:
        out["sensitivity_heatmap.png"] = f"failed: {e}"
    try:
        safe(plot_pgd_curves, "pgd_loss_curves.png", dual=dual)
    except Exception as e:
        out["pgd_loss_curves.png"] = f"failed: {e}"
    try:
        safe(plot_plan_overlay, "plan_overlay.png",
             base_trace=base_trace, pert_trace=pert_trace)
    except Exception as e:
        out["plan_overlay.png"] = f"failed: {e}"
    try:
        safe(plot_obs_channels, "obs_channels.png", probes=probes)
    except Exception as e:
        out["obs_channels.png"] = f"failed: {e}"
    return out
