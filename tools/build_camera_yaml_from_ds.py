"""Function: Build an AETHER camera YAML from preprocessed DriveStudio calibration.
Input/Output: Reads per-camera intrinsics and extrinsics from a processed scene and writes one camera YAML.
Example:
    python tools/build_camera_yaml_from_ds.py --ds-scene-dir data/pandaset/processed/095 \
        --pose-source ego_pose --frame-idx 0 --num-cams 6 \
        --out outputs/pandaset095_camera.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import yaml
from PIL import Image

DS_CAM_ORDER = ["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
                "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_BACK"]


def _mat(M):
    M = np.asarray(M, dtype=np.float64).reshape(4, 4)
    return [[float(x) for x in row] for row in M]


def _intr(p):
    v = np.loadtxt(p).flatten().astype(float)
    return {"fx": float(v[0]), "fy": float(v[1]),
            "cx": float(v[2]), "cy": float(v[3])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-scene-dir", type=Path, required=True)
    ap.add_argument("--pose-source", choices=["lidar_pose", "ego_pose"],
                    default="ego_pose")
    ap.add_argument("--frame-idx", type=int, default=0)
    ap.add_argument("--num-cams", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sd = args.ds_scene_dir
    pose_dir = sd / args.pose_source
    if not pose_dir.exists():
        raise FileNotFoundError(pose_dir)

    fname = f"{args.frame_idx:03d}.txt"
    scene_origin = np.loadtxt(pose_dir / fname).reshape(4, 4)

    cams = {}
    for i in range(min(args.num_cams, len(DS_CAM_ORDER))):
        name = DS_CAM_ORDER[i]
        ext = sd / "extrinsics" / f"{args.frame_idx:03d}_{i}.txt"
        intr = sd / "intrinsics" / f"{i}.txt"
        img = sd / "images" / f"{args.frame_idx:03d}_{i}.jpg"
        if not ext.exists() or not intr.exists():
            raise FileNotFoundError(f"missing {ext} or {intr}")
        c2w = np.loadtxt(ext).reshape(4, 4)
        K = _intr(intr)
        if img.exists():
            with Image.open(img) as im:
                W, H = im.size
        else:
            W = int(round(K["cx"] * 2)); H = int(round(K["cy"] * 2))
        cams[name] = {
            "c2w_ds_at_frame_0": _mat(c2w),
            "intrinsics": {"H": int(H), "W": int(W),
                           **{k: float(v) for k, v in K.items()}},
        }

    doc = {
        "format": "drivestudio",
        "scene_origin_ds": _mat(scene_origin),
        "cam_align": [
            ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"],
            ["CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"],
        ],
        "cams": cams,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False)
    print(f"[ok] wrote {args.out} ({len(cams)} cams)")


if __name__ == "__main__":
    main()
