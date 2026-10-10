#!/usr/bin/env python3
"""Validate downloaded AETHER demo data."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aether.config import load_cfg  # noqa: E402
from aether.render.external_assets import GaussianVehicleAsset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["0383", "088"], default="0383")
    args = parser.parse_args()
    scenario = ROOT / "configs" / "scenarios" / f"nuscenes_{args.scene}.yaml"
    cfg = load_cfg(
        str(scenario), str(ROOT / "configs/base.yaml"),
        str(ROOT / "configs/cameras/nuscenes.yaml"),
        str(ROOT / "configs/kinematic.yaml"),
    )
    required = {
        "checkpoint": Path(str(cfg.scenario.checkpoint)),
        "processed scene": ROOT / "data/nuscenes/processed_10Hz/trainval" /
            ("300" if args.scene == "0383" else "088"),
        "LimSim network": Path(str(cfg.scenario.traffic.net_file)),
        "vehicle asset": Path(str(cfg.scenario.traffic.asset_path)),
        "UniAD checkpoint": ROOT / "checkpoints/uniad_base_e2e.pth",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing demo dependencies:\n  " + "\n  ".join(missing))
    asset = GaussianVehicleAsset.load(required["vehicle asset"], max_gaussians=50_000)
    print(f"[aether-check] scene={args.scene} asset_gaussians={asset.num_gaussians}")
    print("[aether-check] demo layout is valid")


if __name__ == "__main__":
    main()
