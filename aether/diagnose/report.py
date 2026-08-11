"""Stage 7 -- v2 JSON report assembly + viz dispatch."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .semantics import SEMANTIC_BOUNDS, SEMANTIC_VAR_NAMES, TOPO_DIMS, PHOTO_DIMS


def _diagnosis_label(s) -> str:
    if (s.G_topo + s.G_photo) < 1e-6:
        return "no_observable_pathology"
    if s.r_vt_ratio > 1.5:
        return "kinematic_topology_dominated"
    if s.r_vt_ratio < (1.0 / 1.5):
        return "photometric_dominated"
    return "balanced_dual_track"


def _bound_str(name: str) -> str:
    lo, hi = SEMANTIC_BOUNDS[name]
    return "[%+.3f, %+.3f]" % (lo, hi)


def build_report(*, scene_id: str, tested_model: str,
                 critical, causal, sens, dual,
                 probes_summary: Dict[str, float],
                 viz_paths: Dict[str, str]) -> Dict[str, Any]:
    top3 = sorted(causal.per_actor, key=lambda x: x.phi, reverse=True)[:3]
    return {
        "schema_version": "aether-diag-v2",
        "metadata": {
            "scene_id": str(scene_id),
            "tested_model": str(tested_model),
            "critical_frame_anchor": int(critical.t_critical),
            "critical_window": [int(critical.start), int(critical.end)],
            "peak_risk": float(critical.peak_risk),
            "target_object_id": "%s#%d" % (causal.target_class, causal.target_id),
        },
        "macro_pathology_metrics": {
            "G_topo": float(sens.G_topo),
            "G_photo": float(sens.G_photo),
            "r_vt_ratio": float(sens.r_vt_ratio),
            "topo_norm": float(sens.topo_norm),
            "photo_norm": float(sens.photo_norm),
            "diagnosis_conclusion": _diagnosis_label(sens),
            "confidence": float(sens.confidence),
        },
        "s_named_flat": dict(sens.s_named),
        "normalized_sensitivities_S_hat": {
            "topological": {SEMANTIC_VAR_NAMES[i]: float(sens.s[i])
                            for i in TOPO_DIMS},
            "photometric": {SEMANTIC_VAR_NAMES[i]: float(sens.s[i])
                            for i in PHOTO_DIMS},
        },
        "intervention_bounds": {n: _bound_str(n) for n in SEMANTIC_VAR_NAMES},
        "causal_localisation": {
            "top3": [
                {
                    "rank": i + 1,
                    "actor": "%s#%d" % (a.actor_class, a.actor_id),
                    "phi": float(a.phi),
                    "L1_removal": float(a.L1_removal),
                    "L2_displacement": float(a.L2_displacement),
                    "L3_class_swap": float(a.L3_class_swap),
                    "L3_skipped": bool(a.L3_skipped),
                    "delta_peak_risk": float(a.delta_peak_risk),
                    "delta_plan_norm": float(a.delta_plan_norm),
                    "delta_ttc": float(a.delta_ttc),
                }
                for i, a in enumerate(top3)
            ],
            "ablation_weights": list(causal.weights),
        },
        "pgd": {
            "topo": {
                "loss_history": [float(x) for x in dual.topo.loss_history],
                "base_loss": float(dual.topo.base_loss),
                "z_final": dual.topo.z_final.as_dict(),
                "grad_final": [float(x) for x in dual.topo.grad_final.tolist()],
                "breakdown_history": [{k: float(v) for k, v in bk.items()}
                                       for bk in dual.topo.breakdown_history],
            },
            "photo": {
                "loss_history": [float(x) for x in dual.photo.loss_history],
                "base_loss": float(dual.photo.base_loss),
                "z_final": dual.photo.z_final.as_dict(),
                "grad_final": [float(x) for x in dual.photo.grad_final.tolist()],
                "breakdown_history": [{k: float(v) for k, v in bk.items()}
                                       for bk in dual.photo.breakdown_history],
            },
        },
        "observability_probes": probes_summary,
        "viz_paths": viz_paths,
    }


def write_report(report: Dict[str, Any], path) -> Path:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return p
