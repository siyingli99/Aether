"""Function: Render one configured scene frame as a renderer smoke test.
Input/Output: Reads scenario, base, camera, and kinematic configs; writes one RGB image.
Example:
    python scripts/render_smoke_test.py \
        --scenario configs/scenarios/pandaset_095_example.yaml \
        --base configs/base.yaml --camera configs/cameras/pandaset_800x450.yaml \
        --kinematic configs/kinematic.yaml \
        --frame 0 --out outputs/smoke.png
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

import imageio.v2 as iio
import numpy as np

import aether  # noqa: F401
from aether.config import load_cfg
from aether.envs.driving_env import AetherDrivingEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario",  required=True)
    ap.add_argument("--base",      required=True)
    ap.add_argument("--camera",    required=True)
    ap.add_argument("--kinematic", required=True)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--out",   default="smoke.png")
    args = ap.parse_args()

    cfg = load_cfg(args.scenario, args.base, args.camera, args.kinematic)
    env = AetherDrivingEnv(cfg, output_dir="aether_smoke")
    obs, info = env.reset()
    # Optionally fast-forward to a specific renderer frame.
    env.timestamp = args.frame * env.ego.dt
    obs = env._get_obs()

    cams = list(obs["rgb"].keys())
    print(f"[smoke] cameras: {cams}")
    grid = np.concatenate([obs["rgb"][c] for c in cams], axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(args.out, grid)
    print(f"[smoke] wrote {args.out}  shape={grid.shape}")


if __name__ == "__main__":
    main()
