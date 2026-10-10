#!/usr/bin/env python3
"""Convert one reconstructed 3DRealCar sequence into an AETHER vehicle asset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aether.render.external_assets import GaussianVehicleAsset  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert native 3DRealCar gs.pth/wlh.json files to asset.pt."
    )
    parser.add_argument("--input", required=True, help="Directory containing gs.pth and wlh.json")
    parser.add_argument("--output", required=True, help="Destination asset directory")
    parser.add_argument("--sequence-id", required=True, help="Original 3DRealCar sequence ID")
    parser.add_argument("--max-gaussians", type=int, default=50_000)
    args = parser.parse_args()

    source = Path(args.input).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    asset = GaussianVehicleAsset.load(source, max_gaussians=args.max_gaussians)
    asset = GaussianVehicleAsset(
        means=asset.means,
        features_dc=asset.features_dc,
        features_rest=asset.features_rest,
        opacities=asset.opacities,
        scales=asset.scales,
        quats=asset.quats,
        size_lwh=asset.size_lwh,
        source=f"3DRealCar:{args.sequence_id}",
    )
    asset_path = asset.save(output)
    metadata = {
        "format": "aether.gaussian_vehicle.v1",
        "dataset": "3DRealCar",
        "sequence_id": args.sequence_id,
        "source_url": "https://xiaobiaodu.github.io/3drealcar/",
        "gaussians": asset.num_gaussians,
        "size_lwh_m": list(asset.size_lwh),
        "asset_file": asset_path.name,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
