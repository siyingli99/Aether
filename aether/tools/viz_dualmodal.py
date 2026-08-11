"""Function: Render one RGB/LiDAR comparison panel from a dual-modal checkpoint.
Input/Output: Reads a lidar_phase checkpoint and processed scene; writes one four-panel PNG.
Example:
    # Prerequisite: generate the dual-modal checkpoint with lidar_phase.py.
    python aether/tools/viz_dualmodal.py \
        --ckpt outputs/dualmodal/pandaset095_lidar.pth \
        --processed_dir data/pandaset/processed/095 --frame_idx 40 \
        --out_png outputs/pandaset095_dualmodal.png
"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aether.paths import AETHER_ROOT
sys.path.insert(0, str(AETHER_ROOT))
from aether.render.lidar_head import (
    LidarHead, LidarHeadConfig, build_raster_pts,
)
from aether.train.lidar_phase import load_pandaset_lidar


def _unproject_range_map(range_map, alpha, cfg: LidarHeadConfig, lidar2world, device):
    """Convert (n_el, W) range map to a 3D point cloud in world frame.
    Only cells whose alpha > thr are kept."""
    n_el, W = range_map.shape
    az = torch.linspace(cfg.min_azimuth + cfg.azimuth_resolution * 0.5,
                        cfg.max_azimuth - cfg.azimuth_resolution * 0.5,
                        W, device=device)
    el_step = (cfg.max_elevation - cfg.min_elevation) / n_el
    el = torch.linspace(cfg.min_elevation + el_step * 0.5,
                        cfg.max_elevation - el_step * 0.5,
                        n_el, device=device)
    az_g, el_g = torch.meshgrid(az, el, indexing="xy")  # [n_el, W]
    az_r = torch.deg2rad(az_g); el_r = torch.deg2rad(el_g)
    dx = torch.cos(el_r) * torch.cos(az_r)
    dy = torch.cos(el_r) * torch.sin(az_r)
    dz = torch.sin(el_r)
    dirs = torch.stack([dx, dy, dz], dim=-1)  # [n_el, W, 3]
    pts_sensor = dirs * range_map.unsqueeze(-1)
    pts_world = (lidar2world[:3, :3] @ pts_sensor.reshape(-1, 3).T).T + lidar2world[:3, 3]
    keep = (alpha.reshape(-1) > 0.3) & (range_map.reshape(-1) < cfg.far_plane * 0.99)
    return pts_world[keep], keep


def _project_to_camera(pts_world, K, c2w, H, W):
    w2c = torch.linalg.inv(c2w)
    pts_cam = (w2c[:3, :3] @ pts_world.T).T + w2c[:3, 3]
    z = pts_cam[:, 2]
    in_front = z > 0.1
    pts_cam = pts_cam[in_front]; z = z[in_front]
    px = (K[0, 0] * pts_cam[:, 0] / z + K[0, 2])
    py = (K[1, 1] * pts_cam[:, 1] / z + K[1, 2])
    valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return px[valid].cpu().numpy(), py[valid].cpu().numpy(), z[valid].cpu().numpy()


def visualize(ckpt: Path, processed_dir: Path, frame_idx: int,
              out_png: Path, device: str = "cuda"):
    # ---- 1) load lidar sidecar + RGB ckpt
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    rgb_ckpt = Path(blob["source_ckpt"])
    print(f"[viz] RGB ckpt {rgb_ckpt}; lidar sidecar {ckpt}")
    from aether.render.ds_backend import DriveStudioRenderer
    renderer = DriveStudioRenderer(str(rgb_ckpt), device=device)

    # ---- 2) rebuild LidarHead and load features
    lcfg = LidarHeadConfig(**blob["cfg"])
    head = LidarHead(lcfg).to(device)
    head.load_state_dict({k: v.to(device) for k, v in blob["head_state"].items()})
    bg_int = blob["bg_intensity"].to(device)
    bg_drop = blob["bg_raydrop_logit"].to(device)

    # ---- 3) read camera calibration for front cam at this frame
    extr_dir = processed_dir / "extrinsics"
    intr_dir = processed_dir / "intrinsics"
    # front_camera idx is 0 in pandaset processed
    c2w_v = np.loadtxt(extr_dir / f"{frame_idx:03d}_0.txt")
    # pandaset intrinsics file: 9 floats = fx, fy, cx, cy, then 5 distortion coeffs
    intr_vals = np.loadtxt(intr_dir / "0.txt")
    fx, fy, cx, cy = intr_vals[:4]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    img_path = processed_dir / "images" / f"{frame_idx:03d}_0.jpg"
    rgb_gt = plt.imread(img_path)
    H, W = rgb_gt.shape[:2]
    # rescale K to match: original was 1920x1080
    sx = W / 1920.0; sy = H / 1080.0
    K_s = K.copy()
    K_s[0, :] *= sx; K_s[1, :] *= sy

    # c2w is camera-to-world in scene-relative frame (lidar2world @ v2c^-1)
    # Need: full c2w in world frame for renderer.render
    # extrinsics file already maps vehicle -> camera (or camera -> world?).
    # Let me check by reading a few:
    # PandaSet processed extrinsics is cam_to_vehicle (v2c^-1). To get c2w:
    # c2w = ego2world @ cam2vehicle
    # ego2world (scene-relative) = lidar2world (computed in load_pandaset_lidar)
    ego_start = np.loadtxt(processed_dir / "ego_pose" / "000.txt")
    ego_t = np.loadtxt(processed_dir / "ego_pose" / f"{frame_idx:03d}.txt")
    ego2world = np.linalg.inv(ego_start) @ ego_t
    c2w_world = ego2world @ c2w_v
    print(f"[viz] frame {frame_idx} cam-front c2w_world translation: {c2w_world[:3,3]}")

    # ---- 4) RGB render via drivestudio
    print("[viz] rendering RGB ...")
    out_rgb = renderer.render(
        c2w_4x4=c2w_world, K_3x3=K_s, H=H, W=W,
        frame_idx=frame_idx, cam_idx=0,
    )
    rgb_pred = out_rgb.get("rgb", None)
    if rgb_pred is not None:
        rgb_pred = rgb_pred.detach().cpu().numpy()
        rgb_pred = np.clip(rgb_pred, 0, 1)

    # ---- 5) LiDAR render via head (Background only)
    print("[viz] rendering LiDAR ...")
    trainer = renderer.trainer
    bg = trainer.models["Background"]
    means = bg._means.detach()
    quats = bg.get_quats.detach()
    scales = bg.get_scaling.detach()
    opac = bg.get_opacity.detach().reshape(-1)

    lidar2world_t = torch.from_numpy(ego2world.astype(np.float32)).to(device)
    raster_pts, teb = build_raster_pts(lcfg, device)
    with torch.no_grad():
        lout = head.render(
            means_w=means, quats=quats, scales=scales, opacities=opac,
            intensity=torch.sigmoid(bg_int), raydrop_logit=bg_drop,
            lidar_pose_w=lidar2world_t,
            raster_pts=raster_pts, tile_elevation_boundaries=teb,
        )
    range_map = lout["range"]
    alpha_map = lout["alpha"]
    intensity_map = lout["intensity"]
    pts_pred, _ = _unproject_range_map(range_map, alpha_map, lcfg, lidar2world_t, device)
    print(f"[viz] predicted lidar pts: {pts_pred.shape[0]}")

    # ---- 6) GT lidar
    lidar_frame = load_pandaset_lidar(processed_dir, frame_idx, frame_idx + 1,
                                      only_360=True, device=device)[0]
    pts_gt = lidar_frame["pts_world"]
    print(f"[viz] GT lidar pts: {pts_gt.shape[0]}")

    # ---- 7) reproject to front camera
    c2w_t = torch.from_numpy(c2w_world.astype(np.float32)).to(device)
    K_t = torch.from_numpy(K_s.astype(np.float32)).to(device)
    px_p, py_p, z_p = _project_to_camera(pts_pred, K_t, c2w_t, H, W)
    px_g, py_g, z_g = _project_to_camera(pts_gt, K_t, c2w_t, H, W)

    # ---- 8) figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 11))
    axes[0,0].imshow(rgb_gt); axes[0,0].set_title("GT RGB (front camera)"); axes[0,0].axis("off")
    if rgb_pred is not None:
        axes[0,1].imshow(rgb_pred); axes[0,1].set_title("DriveStudio RGB render"); axes[0,1].axis("off")
    else:
        axes[0,1].text(0.5, 0.5, "RGB render not available", ha="center"); axes[0,1].axis("off")

    axes[1,0].imshow(rgb_gt)
    sc = axes[1,0].scatter(px_g, py_g, s=2, c=z_g, cmap="turbo", vmin=2, vmax=60)
    axes[1,0].set_title(f"GT LiDAR overlay ({len(px_g)} pts) -- intensity = depth")
    axes[1,0].axis("off"); plt.colorbar(sc, ax=axes[1,0], fraction=0.025)

    axes[1,1].imshow(rgb_gt)
    sc = axes[1,1].scatter(px_p, py_p, s=2, c=z_p, cmap="turbo", vmin=2, vmax=60)
    axes[1,1].set_title(f"Pred LiDAR overlay ({len(px_p)} pts) -- LidarHead from BG gaussians")
    axes[1,1].axis("off"); plt.colorbar(sc, ax=axes[1,1], fraction=0.025)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=110)
    print(f"[viz] saved {out_png}")

    # also dump small JSON stats
    stats = dict(
        n_gt_lidar=int(pts_gt.shape[0]),
        n_pred_lidar=int(pts_pred.shape[0]),
        n_gt_overlay=int(len(px_g)),
        n_pred_overlay=int(len(px_p)),
        bg_intensity_mean=float(torch.sigmoid(bg_int).mean()),
        bg_raydrop_pos_frac=float((bg_drop > 0).float().mean()),
    )
    (out_png.with_suffix(".json")).write_text(json.dumps(stats, indent=2))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--processed_dir", required=True, type=Path)
    p.add_argument("--frame_idx", type=int, default=40)
    p.add_argument("--out_png", required=True, type=Path)
    a = p.parse_args()
    visualize(a.ckpt, a.processed_dir, a.frame_idx, a.out_png)
