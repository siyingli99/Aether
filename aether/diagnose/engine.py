"""End-to-end diagnosis orchestrator (v2).

Pipeline:
  Stage 0  Baseline rollout, build observability probes.
  Stage 1  Risk + critical window extraction.
  Stage 2  Causal target localisation (L1/L2/L3 ablations).
  Stage 3  Dual-track PGD over 13-D semantic dictionary.
  Stage 4  Sensitivity normalisation + confidence.
  Stage 5  Visualisations (matplotlib).
  Stage 6  Report assembly + write.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .causal import causal_target_localisation
from .losses import PlanLossWeights, plan_loss
from .pgd import PGDConfig, dual_track_pgd
from .probes import compute_step_probes, aggregate_channels
from .report import build_report, write_report
from .risk import RiskWeights, extract_critical_window
from .rollout import Rollout
from .semantics import SemanticPerturbation
from .sensitivity import compute_normalized_sensitivities
from . import viz as vizmod


@dataclass
class DiagnosisConfig:
    risk_weights: RiskWeights = field(default_factory=RiskWeights)
    plan_weights: PlanLossWeights = field(default_factory=PlanLossWeights)
    pgd: PGDConfig = field(default_factory=PGDConfig)
    k_back: int = 5
    f_fwd: int = 5
    rng_seed: int = 0
    baseline_max_steps: Optional[int] = None
    causal_weights: tuple = (1.0, 1.0, 0.5)
    enable_l2_displacement: bool = True
    enable_l3_class_swap: bool = False
    emit_viz: bool = True


@dataclass
class DiagnosisOutput:
    report: Dict[str, Any]
    report_path: Optional[Path]
    viz_paths: Dict[str, str]


class DiagnosisEngine:
    def __init__(self, rollout: Rollout, tested_model: str,
                 cfg: DiagnosisConfig = DiagnosisConfig()):
        self.rollout = rollout
        self.tested_model = tested_model
        self.cfg = cfg
        self._t0 = time.time()

    def _log(self, msg: str) -> None:
        dt = time.time() - self._t0
        print("[diagnose +%6.2fs] %s" % (dt, msg), flush=True)

    def _baseline_trace(self) -> List[Dict[str, Any]]:
        return self.rollout(actor_keep=None,
                            perturbation=SemanticPerturbation.zero(),
                            max_steps=self.cfg.baseline_max_steps)

    def _build_rollout_keep(self, actor_pool, win_len):
        pool_idx = [self.rollout.actor_pool.index(p) for p in actor_pool]
        def _rk(keep):
            mapped = [pool_idx[i] for i in keep]
            return self.rollout(actor_keep=mapped,
                                perturbation=SemanticPerturbation.zero(),
                                max_steps=win_len)
        return _rk

    def _build_rollout_displace(self, actor_pool, win_len):
        # MockRollout has no displacement entry-point. PipeRollout could
        # support it through the renderer. We only enable when the rollout
        # exposes ``rollout_displace``.
        fn = getattr(self.rollout, "rollout_displace", None)
        if fn is None:
            return None
        def _rd(idx, dxyz):
            return fn(idx, dxyz, max_steps=win_len)
        return _rd

    def _build_rollout_with_z(self, actor_pool, target_idx, win_len):
        full_idx = self.rollout.actor_pool.index(actor_pool[target_idx])
        def _r(z):
            return self.rollout(actor_keep=[full_idx], perturbation=z,
                                max_steps=win_len)
        return _r

    def run(self, output_path=None) -> DiagnosisOutput:
        cfg = self.cfg
        # Stage 0
        self._log("[stage 0] baseline rollout")
        trace0 = self._baseline_trace()
        probes = compute_step_probes(trace0)

        # Stage 1
        self._log("[stage 1] critical-window extraction")
        critical = extract_critical_window(trace0, k_back=cfg.k_back,
                                           f_fwd=cfg.f_fwd,
                                           weights=cfg.risk_weights)
        win_len = critical.end - critical.start + 1
        self._log("[stage 1] window=[%d,%d] t*=%d peak=%.3f"
                  % (critical.start, critical.end, critical.t_critical,
                     critical.peak_risk))

        actor_pool = list(self.rollout.actor_pool)
        if not actor_pool:
            raise RuntimeError("Rollout reports no actor_pool.")
        rollout_keep = self._build_rollout_keep(actor_pool, win_len)
        rollout_disp = (self._build_rollout_displace(actor_pool, win_len)
                        if cfg.enable_l2_displacement else None)

        # Stage 2
        self._log("[stage 2] causal target localisation (L1/L2/L3)")
        causal = causal_target_localisation(
            actor_pool=actor_pool, rollout_keep=rollout_keep,
            rollout_displace=rollout_disp, rollout_swap=None,
            weights=cfg.causal_weights,
            progress_cb=lambda m: self._log("  " + m),
        )
        self._log("[stage 2] target=%s#%d (phi=%.4f)"
                  % (causal.target_class, causal.target_id,
                     causal.per_actor[causal.target_index_in_pool].phi))

        # Stage 3
        rollout_z = self._build_rollout_with_z(actor_pool,
                                               causal.target_index_in_pool,
                                               win_len)
        self._log("[stage 3] dual-track 13-D PGD")
        # use trace0 as the plan_dev anchor
        anchor_window = trace0[critical.start: critical.end + 1] or trace0
        dual = dual_track_pgd(rollout_z, cfg.pgd, base_trace=anchor_window)

        # Stage 4
        self._log("[stage 4] sensitivity normalisation")
        sens = compute_normalized_sensitivities(dual)
        self._log("[stage 4] G_topo=%.3f G_photo=%.3f r_vt=%.3f"
                  % (sens.G_topo, sens.G_photo, sens.r_vt_ratio))

        # Stage 5: viz
        viz_paths: Dict[str, str] = {}
        if cfg.emit_viz and output_path is not None:
            out_dir = Path(output_path).parent
            self._log("[stage 5] writing visualisations to %s/figs" % out_dir)
            try:
                pert_trace = rollout_z(dual.topo.z_final)
            except Exception:
                pert_trace = trace0
            try:
                viz_paths = vizmod.emit_all(
                    out_dir,
                    per_step_risk=critical.per_step_risk,
                    window_start=critical.start, window_end=critical.end,
                    t_critical=critical.t_critical,
                    causal_per_actor=causal.per_actor,
                    sensitivity_vec=sens.s, dual=dual,
                    base_trace=trace0, pert_trace=pert_trace, probes=probes,
                )
            except Exception as e:
                self._log("[stage 5] viz failed: %s" % e)

        # Stage 6: report
        self._log("[stage 6] report assembly")
        probes_summary = aggregate_channels(probes)
        report = build_report(
            scene_id=self.rollout.scene_id, tested_model=self.tested_model,
            critical=critical, causal=causal, sens=sens, dual=dual,
            probes_summary=probes_summary, viz_paths=viz_paths,
        )
        path = write_report(report, output_path) if output_path else None
        self._log("[done] report -> %s" % (path or "<inline>"))
        return DiagnosisOutput(report=report, report_path=path,
                               viz_paths=viz_paths)
