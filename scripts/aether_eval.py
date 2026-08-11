#!/usr/bin/env python
"""Function: Evaluate a closed-loop rollout with selected driving metrics.
Input/Output: Reads ``data.pkl`` from a rollout directory; prints scores and optionally writes JSON.
Example:
    # Prerequisite: run a closed-loop rollout to create data.pkl.
    python scripts/aether_eval.py aether_results/pandaset/pandaset_095_aether/uniad/easy \
        --metrics hugsim,navsim_pdms --out outputs/pandaset095_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(THIS))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", help="closed-loop output dir (contains data.pkl)")
    ap.add_argument("--metrics", default="hugsim,navsim_pdms,bench2drive_ds",
                    help="comma-separated metric names")
    ap.add_argument("--out", default=None, help="dump JSON here")
    args = ap.parse_args()

    pkl = os.path.join(args.output_dir, "data.pkl")
    if not os.path.isfile(pkl):
        print(f"ERROR: {pkl} not found", file=sys.stderr)
        return 2
    with open(pkl, "rb") as f:
        save_data = pickle.load(f)

    # Optional point clouds for the full HUGSim DAC test.
    ground_xyz = scene_xyz = None
    g_ply = os.path.join(args.output_dir, "ground.ply")
    s_ply = os.path.join(args.output_dir, "scene.ply")
    if os.path.isfile(g_ply) and os.path.isfile(s_ply):
        try:
            import numpy as np
            import open3d as o3d  # type: ignore
            ground_xyz = np.asarray(o3d.io.read_point_cloud(g_ply).points)
            scene_xyz = np.asarray(o3d.io.read_point_cloud(s_ply).points)
            print(f"loaded ground={ground_xyz.shape} scene={scene_xyz.shape}",
                  file=sys.stderr)
        except Exception as e:
            print(f"WARN: failed to load PLYs: {e}", file=sys.stderr)

    from aether.eval.base import evaluate_all

    names = [n.strip() for n in args.metrics.split(",") if n.strip()]
    results = evaluate_all(save_data, ground_xyz=ground_xyz,
                           scene_xyz=scene_xyz, metric_names=names)
    out = {n: r.to_dict() for n, r in results.items()}
    text = json.dumps(out, indent=2, default=float)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"\nwrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
