"""Function: Generate pose-perturbed 3DGS LiDAR and front-RGB training pairs.
Input/Output: Reads registered joint checkpoints, sidecars, and processed scenes; writes NPZ data, images, and manifests.
Example:
    # Prerequisite: register generated joint checkpoints and sidecars in the registry.
    python aether/refine/gen_multimodal_pairs.py \
        --registry configs/refine/scene_registry.yaml --scene_ids pandaset095_joint \
        --out outputs/multimodal_refine --stride 2
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from aether.paths import AETHER_ROOT, resolve_data_root
from aether.render.ds_backend import DriveStudioRenderer
from aether.render.lidar_closed_loop import ClosedLoopLidar
from aether.refine.distill_multimodal import make_world_delta
from aether.refine.pair_utils import (
    build_lidar_source,
    latest_lidar_sidecar,
    pack_lidar_rgb,
    project_gt_frame,
    save_pair_npz,
)
from aether.render.lidar_head import LidarHeadConfig


def _rgb_np(t):
    return (t.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)


def _load_registry(path: Path):
    with open(path) as f:
        return yaml.safe_load(f)["scenes"]


def _perturbations():
    """Pose perturbations calibrated to closed-loop ego drift (not i.i.d. noise)."""
    return [
        ("nominal", 0.0, 0.0, 0.0),
        ("lat02", 0.0, 0.2, 0.0),
        ("lat04", 0.0, 0.4, 0.0),
        ("lat08", 0.0, 0.8, 0.0),
        ("lat12", 0.0, 1.2, 0.0),
        ("yaw03", 3.0, 0.0, 0.0),
        ("yaw05", 5.0, 0.0, 0.0),
        ("yaw08", 8.0, 0.0, 0.0),
        ("yaw05_lat04", 5.0, 0.4, 0.0),
        ("yaw08_lat08", 8.0, 0.8, 0.0),
        ("fwd10", 0.0, 0.0, 1.0),
        ("fwd20_lat04", 2.0, 0.4, 2.0),
        ("yaw10_lat12", 10.0, 1.2, 0.0),
    ]


def process_registry_scene(entry: dict, out_root: Path, stride: int, limit: int,
                         img_hw: tuple[int, int], device: str):
    sid = entry["id"]
    jdir = AETHER_ROOT / entry["joint_dir"]
    ckpt = jdir / "checkpoint_final.pth"
    sidecar = latest_lidar_sidecar(jdir)
    config = jdir / "config.yaml"
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    scene_out = out_root / "render_corpus" / sid
    npz_dir = scene_out / "npz"
    dirs = {k: scene_out / k for k in ("lidar_in", "lidar_ref", "lidar_gt",
                                       "rgb_in", "rgb_ref", "rgb_gt")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    renderer = DriveStudioRenderer(str(ckpt), device=device)
    cl = ClosedLoopLidar(renderer, str(sidecar), device=device)
    far = float(cl.cfg.far_plane)
    head_cfg = LidarHeadConfig()

    cfg = OmegaConf.load(config)
    cfg.data.data_root = str(resolve_data_root(entry["data_root"]))
    cfg.data.scene_idx = int(entry["scene_idx"])
    src, data_path = build_lidar_source(cfg, device)
    lidar_bin = Path(data_path) / "lidar"

    from utils.misc import import_str
    dp = Path(data_path)
    tot = len(os.listdir(dp / ("lidar_pose" if (dp / "lidar_pose").is_dir() else "ego_pose")))
    e = (tot - 1 if cfg.data.end_timestep == -1 else cfg.data.end_timestep) + 1
    cam_obj = import_str(cfg.data.pixel_source.type)(
        cfg.data.dataset, cfg.data.pixel_source, str(dp),
        cfg.data.start_timestep, e, device=device)
    cam_obj.to(device)

    n_frames = src.lidar_to_worlds.shape[0]
    frames = list(range(0, n_frames, max(1, stride)))
    if limit > 0:
        frames = frames[:limit]
    H, W = img_hw

    gt_cache = {f: project_gt_frame(src, f, lidar_bin, head_cfg, device) for f in frames}
    poses = {f: gt_cache[f]["pose"] for f in frames}

    def nearest(f):
        q = poses[f][:2, 3]
        best, bd = f, 1e9
        for j in frames:
            if j == f:
                continue
            d = float(np.linalg.norm(poses[j][:2, 3] - q))
            if d < bd:
                bd, best = d, j
        return best

    manifest = {}
    for f in tqdm(frames, desc=sid, leave=False):
        ref_f = nearest(f)
        _, ref_ci = cam_obj.get_image(ref_f * cam_obj.num_cams + 0)
        ref_cc = ref_ci["camera_to_world"].cpu().numpy().astype(np.float32)
        ref_K = ref_ci["intrinsics"].cpu().numpy().astype(np.float32)
        gt = gt_cache[f]
        ref_gt = gt_cache[ref_f]
        ref_keep = np.clip(ref_gt["gt_mask"], 0, 1)

        for tag, yaw, dy, dx in _perturbations():
            lp = poses[f]
            T = make_world_delta(yaw, dx, dy, 0.0, lidar2world=lp)
            novel_lp = (T @ lp).astype(np.float32)
            _, ci = cam_obj.get_image(f * cam_obj.num_cams + 0)
            base_cc = ci["camera_to_world"].cpu().numpy().astype(np.float32)
            K = ci["intrinsics"].cpu().numpy().astype(np.float32)
            novel_cc = (T @ base_cc).astype(np.float32)

            out = cl.render_sweep(novel_lp, f, apply_overrides=False)
            in_r = out["range"].clamp(0, far).detach().cpu().numpy()
            in_i = out["intensity"].clamp(0, 1).detach().cpu().numpy()
            in_k = (1 - torch.sigmoid(out["raydrop_logit"])).clamp(0, 1).detach().cpu().numpy()

            with torch.no_grad():
                rgb_in = renderer.render(novel_cc, K, H, W, f)["rgb"]
                rgb_ref = renderer.render(ref_cc, ref_K, H, W, ref_f)["rgb"]
                rgb_gt = renderer.render(base_cc, K, H, W, f)["rgb"]

            name = f"{f:04d}_{tag}"
            npz_path = npz_dir / f"{name}.npz"
            save_pair_npz(
                npz_path, in_r, in_i, in_k,
                ref_gt["gt_range"], ref_gt["gt_int"], ref_keep,
                gt["gt_range"], gt["gt_int"], gt["gt_mask"],
                meta={"frame": f, "ref_frame": ref_f, "perturb": tag, "source": "real_render_perturbed"},
            )
            Image.fromarray(pack_lidar_rgb(in_r, in_i, in_k, far)).save(dirs["lidar_in"] / f"{name}.png")
            Image.fromarray(pack_lidar_rgb(ref_gt["gt_range"], ref_gt["gt_int"], ref_keep, far)).save(
                dirs["lidar_ref"] / f"{name}.png")
            Image.fromarray(pack_lidar_rgb(gt["gt_range"], gt["gt_int"], gt["gt_mask"], far)).save(
                dirs["lidar_gt"] / f"{name}.png")
            Image.fromarray(_rgb_np(rgb_in)).save(dirs["rgb_in"] / f"{name}.png")
            Image.fromarray(_rgb_np(rgb_ref)).save(dirs["rgb_ref"] / f"{name}.png")
            Image.fromarray(_rgb_np(rgb_gt)).save(dirs["rgb_gt"] / f"{name}.png")

            manifest[name] = {
                "npz": str(npz_path),
                "lidar_in": str(dirs["lidar_in"] / f"{name}.png"),
                "lidar_ref": str(dirs["lidar_ref"] / f"{name}.png"),
                "lidar_gt": str(dirs["lidar_gt"] / f"{name}.png"),
                "rgb_in": str(dirs["rgb_in"] / f"{name}.png"),
                "rgb_ref": str(dirs["rgb_ref"] / f"{name}.png"),
                "rgb_gt": str(dirs["rgb_gt"] / f"{name}.png"),
                "scene": sid,
                "source": "real_render_perturbed",
                "prompt": "remove degradation",
            }

    with open(scene_out / "pairs.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    return sid, len(manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(AETHER_ROOT / "configs/refine/scene_registry.yaml"))
    ap.add_argument("--out", default=str(AETHER_ROOT / "outputs/multimodal_refine"))
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help=">0: max frames per scene")
    ap.add_argument("--scene_ids", default="", help="comma subset of registry ids")
    ap.add_argument("--img_hw", default="256,448")
    args = ap.parse_args()

    device = "cuda"
    H, W = (int(x) for x in args.img_hw.split(","))
    out_root = Path(args.out)
    scenes = _load_registry(Path(args.registry))
    if args.scene_ids.strip():
        allow = set(args.scene_ids.split(","))
        scenes = [s for s in scenes if s["id"] in allow]

    print(f"[render_corpus] scenes={len(scenes)} stride={args.stride}", flush=True)
    total = 0
    for entry in scenes:
        try:
            sid, n = process_registry_scene(entry, out_root, args.stride, args.limit,
                                            (H, W), device)
            total += n
            print(f"  ✓ {sid}: {n} pairs", flush=True)
        except Exception as e:
            print(f"  ✗ {entry.get('id')}: {e}", flush=True)
    print(f"[render_corpus] DONE total_pairs={total}", flush=True)


if __name__ == "__main__":
    main()
