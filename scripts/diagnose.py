"""Function: Run semantic-causal diagnosis for a configured closed-loop scene.
Input/Output: Reads scene and model configs; writes a diagnosis JSON report and optional visualizations.
Example:
    python scripts/diagnose.py --scenario configs/scenarios/pandaset_095_aether.yaml \
        --base configs/base.yaml --camera configs/cameras/pandaset_800x450.yaml \
        --kinematic configs/kinematic.yaml --ad uniad --output outputs/pandaset095_diagnosis.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _build_real_rollout(args):
    from aether.config import load_cfg
    from aether.diagnose.rollout import PipeRollout
    cfg = load_cfg(args.scenario, args.base, args.camera, args.kinematic)
    ad_shell = getattr(cfg.base, "%s_path" % args.ad, None)
    if not ad_shell:
        raise ValueError("Missing AD shell path: cfg.base.%s_path" % args.ad)
    if args.output_dir is None:
        out_root = getattr(cfg.base, "output_dir", "outputs")
        scene = str(getattr(cfg.scenario, "scene_name",
                            getattr(cfg.scenario, "scene_id", "scene")))
        args.output_dir = os.path.join(out_root, "diagnose", scene, args.ad, "diag")
    os.makedirs(args.output_dir, exist_ok=True)
    target_class, target_instance = None, None
    if args.target:
        cls, inst = args.target.split("#")
        target_class, target_instance = cls, int(inst)
    return PipeRollout(
        scene_id=str(getattr(cfg.scenario, "scene_name",
                             getattr(cfg.scenario, "scene_id", "scene"))),
        cfg=cfg, output_dir=args.output_dir,
        ad_shell=ad_shell, ad_cuda=str(args.ad_cuda),
        target_class=target_class, target_instance=target_instance,
        max_actors=int(args.max_actors),
    ), str(args.ad)


def main():
    ap = argparse.ArgumentParser(description="AETHER-Diag v2 -- semantic-causal diagnosis")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--scenario", type=str)
    ap.add_argument("--base", type=str)
    ap.add_argument("--camera", type=str)
    ap.add_argument("--kinematic", type=str)
    ap.add_argument("--ad", type=str, default="vla", choices=["uniad", "vad", "ltf", "vla"])
    ap.add_argument("--ad_cuda", type=str, default="0")
    ap.add_argument("--target", type=str, default=None)
    ap.add_argument("--pgd_iters", type=int, default=3)
    ap.add_argument("--pgd_eta", type=float, default=0.25)
    ap.add_argument("--pgd_eps", type=float, default=0.05)
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--max_actors", type=int, default=6)
    ap.add_argument("--baseline_steps", type=int, default=None)
    ap.add_argument("--k_back", type=int, default=5)
    ap.add_argument("--f_fwd", type=int, default=5)
    ap.add_argument("--no_viz", action="store_true",
                    help="Skip matplotlib outputs.")
    ap.add_argument("--no_l2", action="store_true",
                    help="Skip L2 displacement causal layer.")
    args = ap.parse_args()

    from aether.diagnose import (
        DiagnosisEngine, DiagnosisConfig, MockRollout, PGDConfig,
    )

    cfg = DiagnosisConfig(
        pgd=PGDConfig(iters=args.pgd_iters, step_size=args.pgd_eta,
                      fd_eps=args.pgd_eps),
        k_back=args.k_back, f_fwd=args.f_fwd,
        baseline_max_steps=args.baseline_steps,
        emit_viz=not args.no_viz,
        enable_l2_displacement=not args.no_l2,
    )

    if args.mock:
        rollout = MockRollout(); tested = "mock_planner"
    else:
        for k in ("scenario", "base", "camera", "kinematic"):
            if getattr(args, k) is None:
                ap.error("--%s is required when --mock is not set." % k)
        rollout, tested = _build_real_rollout(args)

    try:
        out = DiagnosisEngine(rollout, tested_model=tested, cfg=cfg).run(
            output_path=args.output)
    finally:
        rollout.close()

    macro = out.report["macro_pathology_metrics"]
    print(json.dumps({
        "report_path": str(out.report_path),
        "G_topo": macro["G_topo"],
        "G_photo": macro["G_photo"],
        "r_vt_ratio": macro["r_vt_ratio"],
        "diagnosis_conclusion": macro["diagnosis_conclusion"],
        "confidence": macro["confidence"],
        "target_object_id": out.report["metadata"]["target_object_id"],
        "critical_frame_anchor": out.report["metadata"]["critical_frame_anchor"],
        "viz": list(out.viz_paths.keys()),
    }, indent=2))


if __name__ == "__main__":
    main()
