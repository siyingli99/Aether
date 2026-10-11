"""GPU worker that renders the six-camera scene preview outside the API."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import imageio.v2 as imageio
from PIL import Image, ImageDraw

from aether.config import load_cfg
from aether.envs.driving_env import AetherDrivingEnv
from aether.paths import DRIVESTUDIO_ROOT
from aether.studio.catalog import REPO_ROOT, StudioCatalog

os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")


def render_preview(scene_id: str, output_dir: Path) -> dict:
    catalog = StudioCatalog()
    scene = catalog.scene(scene_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg(
        scene["scenario"], str(REPO_ROOT / "configs/base.yaml"),
        scene["camera"], str(REPO_ROOT / "configs/kinematic.yaml"),
    )
    # A preview renders the trained scene only; it must not depend on a live
    # LimSim bridge or inherit destructive traffic visibility settings.
    if getattr(cfg.scenario, "traffic", None) is not None:
        cfg.scenario.traffic.enabled = False
    cfg.scenario.actor_backend = "replay"
    if not (DRIVESTUDIO_ROOT / "smpl_models" / "SMPL_NEUTRAL.pkl").is_file():
        cfg.scenario.excluded_model_classes = ["SMPLNodes"]
    env = AetherDrivingEnv(cfg, str(output_dir / "runtime"))
    try:
        obs, _ = env.reset()
        paths = {}
        ordered = [
            "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
            "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
        ]
        tiles = []
        for name in ordered:
            if name not in obs["rgb"]:
                continue
            array = obs["rgb"][name]
            path = output_dir / f"{name}.jpg"
            imageio.imwrite(path, array, quality=82)
            paths[name] = path.name
            tile = Image.fromarray(array).resize((400, 225), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(tile)
            draw.rounded_rectangle((10, 9, 142, 33), radius=5, fill=(8, 12, 18, 210))
            draw.text((18, 15), name.replace("CAM_", ""), fill=(238, 244, 252))
            tiles.append(tile)
        if len(tiles) == 6:
            mosaic = Image.new("RGB", (1200, 450), (9, 13, 20))
            for index, tile in enumerate(tiles):
                mosaic.paste(tile, ((index % 3) * 400, (index // 3) * 225))
            mosaic.save(output_dir / "mosaic.jpg", quality=84)
        result = {"scene_id": scene_id, "cameras": paths, "mosaic": "mosaic.jpg"}
        (output_dir / "preview.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render_preview(args.scene, args.output)


if __name__ == "__main__":
    main()
