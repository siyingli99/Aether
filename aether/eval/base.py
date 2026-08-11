"""Pluggable metric layer for AETHER.

Defines a small ``Metric`` ABC plus three concrete back-ends:

- ``HUGSimMetric``  -- wraps the existing ``aether.eval.score.hugsim_evaluate``
  driving-score implementation (NC * DAC * (TTC + C) ... * RC).
- ``NavsimPDMSMetric`` -- abstract scaffold mirroring the NAVSIM PDMS formula
  (no-collision * drivable-area * (ttc + comfort)).  Falls back to the HUGSim
  implementation when NAVSIM isn't installed; the formula is identical so the
  numbers match for the common closed-loop case.
- ``Bench2DriveMetric`` -- abstract scaffold mirroring CARLA Leaderboard's
  Driving Score (route_completion * infraction_penalty).  Falls back to a
  light-weight estimator built from the same telemetry.

The intent is that callers iterate ``metrics.run(...)`` and aggregate the
returned ``MetricResult`` objects so we can compare adapters on the same
trajectory recording.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class MetricResult:
    """Single (metric, scenario, adapter) score record."""

    name: str
    score: float
    sub_scores: Dict[str, float] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": float(self.score),
            "sub_scores": {k: float(v) for k, v in self.sub_scores.items()},
            "extras": dict(self.extras),
        }


class Metric(abc.ABC):
    """Abstract closed-loop metric."""

    name: str = "metric"

    @abc.abstractmethod
    def evaluate(self, save_data: Mapping[str, Any],
                 ground_xyz: Optional[Any] = None,
                 scene_xyz: Optional[Any] = None) -> MetricResult:
        """Compute the metric for one closed-loop episode.

        ``save_data`` follows the HUGSIM convention: a dict with keys
        ``ego_states``, ``obj_boxes``, ``timestamp``, ``collision`` etc.
        ``ground_xyz`` / ``scene_xyz`` are optional point clouds used for the
        drivable-area test.
        """


# ---------------------------------------------------------------------------
# Concrete back-ends
# ---------------------------------------------------------------------------


class HUGSimMetric(Metric):
    name = "hugsim"

    def evaluate(self, save_data, ground_xyz=None, scene_xyz=None):
        # Lazy: full hugsim_evaluate needs ground/scene point clouds for the
        # DAC test. When they're missing, fall back to a trajectory-only
        # estimate (RC + collision flag) so the e2e pipeline still produces
        # numbers.
        if ground_xyz is None or scene_xyz is None:
            return self._fallback(save_data)
        try:
            import numpy as np
            from aether.eval.score import hugsim_evaluate
            out = hugsim_evaluate(save_data, np.asarray(ground_xyz),
                                  np.asarray(scene_xyz))
        except Exception as e:
            res = self._fallback(save_data)
            res.extras["error"] = str(e)
            return res

        if isinstance(out, Mapping):
            sub = {k: float(v) for k, v in out.items()
                   if not isinstance(v, (list, dict)) and v is not None}
            ds = float(out.get("hdscore", out.get("pdms", 0.0)))
            return MetricResult(name=self.name, score=ds, sub_scores=sub,
                                extras={"per_step": out.get("details", [])})
        # legacy tuple form
        if isinstance(out, tuple):
            mean_pdms, rc, ds = float(out[0]), float(out[1]), float(out[2])
            averages = out[3] if len(out) > 3 and isinstance(out[3], Mapping) else {}
            score_list = out[4] if len(out) > 4 else []
            sub = {k: float(v) for k, v in (averages or {}).items()}
            sub.setdefault("pdms", mean_pdms)
            sub.setdefault("rc", rc)
            return MetricResult(name=self.name, score=ds, sub_scores=sub,
                                extras={"per_step": list(score_list)})
        return MetricResult(name=self.name, score=float(out))

    def _fallback(self, save_data) -> "MetricResult":
        """Trajectory-only score when no ground/scene PLYs are available."""
        try:
            exp = save_data[0] if isinstance(save_data, list) else save_data
            frames = exp.get("frames", []) if isinstance(exp, Mapping) else []
        except Exception:
            frames = []
        if not frames:
            return MetricResult(name=self.name, score=0.0,
                                extras={"fallback": "no_frames"})
        rc = 0.0
        nc = 1.0
        for f in frames:
            rc = max(rc, float(f.get("rc", 0.0)))
            if f.get("collision", False):
                nc = 0.0
        rc = min(1.0, rc)
        ds = float(rc * nc)
        return MetricResult(
            name=self.name,
            score=ds,
            sub_scores={"pdms": ds, "rc": rc, "nc": nc, "dac": 1.0,
                        "ttc": 1.0, "c": 1.0, "hdscore": ds},
            extras={"fallback": "no_pointclouds", "frames": len(frames)},
        )


class NavsimPDMSMetric(Metric):
    """NAVSIM PDM Score wrapper.

    The official implementation lives in ``navsim.evaluate.pdm_score``; if it
    is not importable (most AETHER setups today), we degrade to the
    HUGSIM-internal PDMS which uses the identical formula so users still get a
    comparable number while developing a real adapter.
    """

    name = "navsim_pdms"

    def __init__(self, weights: Optional[Mapping[str, float]] = None):
        self.weights = dict(weights) if weights else None

    def evaluate(self, save_data, ground_xyz=None, scene_xyz=None):
        try:  # real impl
            from navsim.evaluate.pdm_score import compute_pdm_score  # type: ignore
            score = float(compute_pdm_score(save_data))
            return MetricResult(name=self.name, score=score,
                                sub_scores={"pdms": score})
        except Exception:
            base = HUGSimMetric().evaluate(save_data, ground_xyz, scene_xyz)
            return MetricResult(
                name=self.name,
                score=float(base.sub_scores.get("pdms", base.score)),
                sub_scores=dict(base.sub_scores),
                extras={"fallback": "hugsim_pdms"},
            )


class Bench2DriveMetric(Metric):
    """Bench2Drive (CARLA Leaderboard) Driving Score wrapper.

    Driving Score = route_completion * infraction_penalty.  In the absence of
    an installed leaderboard we approximate ``infraction_penalty`` by
    ``no_collision * drivable_area_compliance`` (HUGSim NC * DAC), which
    matches the "no major infraction" definition for closed-loop replays.
    """

    name = "bench2drive_ds"

    def evaluate(self, save_data, ground_xyz=None, scene_xyz=None):
        base = HUGSimMetric().evaluate(save_data, ground_xyz, scene_xyz)
        rc = float(base.sub_scores.get("rc", 0.0))
        nc = float(base.sub_scores.get("nc", 1.0))
        dac = float(base.sub_scores.get("dac", 1.0))
        infraction_penalty = max(0.0, min(1.0, nc * dac))
        ds = max(0.0, min(1.0, rc * infraction_penalty))
        return MetricResult(
            name=self.name,
            score=ds,
            sub_scores={
                "route_completion": rc,
                "infraction_penalty": infraction_penalty,
                "no_collision": nc,
                "drivable_area_compliance": dac,
            },
            extras={"backend": "fallback_from_hugsim"},
        )


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

REGISTRY: Dict[str, type] = {
    "hugsim": HUGSimMetric,
    "navsim_pdms": NavsimPDMSMetric,
    "bench2drive_ds": Bench2DriveMetric,
}


def make(name: str, **kw) -> Metric:
    if name not in REGISTRY:
        raise KeyError(f"unknown metric '{name}', have {list(REGISTRY)}")
    return REGISTRY[name](**kw)


def evaluate_all(save_data, ground_xyz=None, scene_xyz=None,
                 metric_names: Optional[List[str]] = None
                 ) -> Dict[str, MetricResult]:
    names = metric_names or list(REGISTRY.keys())
    out: Dict[str, MetricResult] = {}
    for n in names:
        try:
            out[n] = make(n).evaluate(save_data, ground_xyz, scene_xyz)
        except Exception as e:
            out[n] = MetricResult(name=n, score=float("nan"),
                                  extras={"error": str(e)})
    return out
