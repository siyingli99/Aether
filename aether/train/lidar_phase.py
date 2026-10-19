"""Function: Fit a LiDAR head while keeping a Phase-A RGB Gaussian checkpoint frozen.
Input/Output: Reads an RGB checkpoint and processed PandaSet scene; writes a dual-modal checkpoint.
Example:
    # Prerequisite: train the referenced RGB checkpoint and preprocess the scene.
    python aether/train/lidar_phase.py \
        --rgb_ckpt data/scenes/pandaset-095/recon/checkpoint.pth \
        --processed_dir data/pandaset/processed/095 \
        --out_ckpt outputs/train/pandaset-095/lidar.pth --num_iters 1500
"""
from __future__ import annotations
import sys, time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch

from aether.paths import AETHER_ROOT
sys.path.insert(0, str(AETHER_ROOT))
from aether.paths import use_vendored_backends
use_vendored_backends()  # DriveStudio / SplatAD on sys.path
from aether.render.lidar_head import (
    LidarHead, LidarHeadConfig, GaussianLidarFeatures, build_raster_pts,
)


# --------------------------------------------------------------------- #
# Raw PandaSet LiDAR loader (independent of the drivestudio dataset class).
# --------------------------------------------------------------------- #
def load_pandaset_lidar(processed_dir: Path, start: int, end: int,
                        only_360: bool = True, device="cuda"):
    """Returns lists indexed by frame: each frame -> dict of tensors in WORLD
    frame (drivestudio convention: ego at frame `start` is the origin)."""
    ego_start = np.loadtxt(processed_dir / "ego_pose" / f"{start:03d}.txt")
    inv_ego_start = np.linalg.inv(ego_start)
    frames = []
    for t in range(start, end):
        ego_t = np.loadtxt(processed_dir / "ego_pose" / f"{t:03d}.txt")
        lidar2world = inv_ego_start @ ego_t  # (4,4)
        bin_path = processed_dir / "lidar" / f"{t:03d}.bin"
        info = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 5)  # x,y,z,int,laser_id
        if only_360:
            info = info[info[:, 4] == 0]
        pts_sensor = info[:, :3]
        intensity = info[:, 3]
        # transform sensor -> world
        R = lidar2world[:3, :3]; t_v = lidar2world[:3, 3]
        pts_world = pts_sensor @ R.T + t_v
        origin_world = t_v.copy()  # all rays share one origin per frame
        dirs = pts_world - origin_world
        ranges = np.linalg.norm(dirs, axis=1)
        dirs = dirs / np.clip(ranges[:, None], 1e-6, None)
        frames.append(dict(
            lidar2world=torch.from_numpy(lidar2world.astype(np.float32)).to(device),
            origin_world=torch.from_numpy(origin_world.astype(np.float32)).to(device),
            pts_world=torch.from_numpy(pts_world.astype(np.float32)).to(device),
            ranges=torch.from_numpy(ranges.astype(np.float32)).to(device),
            intensity=torch.from_numpy(intensity.astype(np.float32) / 255.0).to(device),
            dirs_world=torch.from_numpy(dirs.astype(np.float32)).to(device),
        ))
    return frames


# --------------------------------------------------------------------- #
# Rasterise real GT into the head's elevation/azimuth grid so we can compare
# against the rendered range map directly.
# --------------------------------------------------------------------- #
def gt_to_spherical_grid(frame: Dict[str, torch.Tensor], cfg: LidarHeadConfig,
                         device="cuda"):
    """Bin real LiDAR points into (n_el, W) grid using min-range per cell.
    Returns range_map [n_el,W], intensity_map [n_el,W], hit_mask [n_el,W]."""
    dirs_lidar = (
        torch.linalg.inv(frame["lidar2world"])[:3, :3] @ frame["dirs_world"].T
    ).T  # back into sensor frame
    az = torch.rad2deg(torch.atan2(dirs_lidar[:, 1], dirs_lidar[:, 0]))   # [-180,180]
    el = torch.rad2deg(torch.atan2(dirs_lidar[:, 2],
                                   torch.linalg.norm(dirs_lidar[:, :2], dim=1)))
    W = int(round((cfg.max_azimuth - cfg.min_azimuth) / cfg.azimuth_resolution))
    n_el = cfg.n_elevation_channels
    # bin
    j = ((az - cfg.min_azimuth) / cfg.azimuth_resolution).long().clamp(0, W - 1)
    i = ((el - cfg.min_elevation) /
         (cfg.max_elevation - cfg.min_elevation) * n_el).long().clamp(0, n_el - 1)
    range_map = torch.full((n_el, W), float("inf"), device=device)
    intensity_map = torch.zeros((n_el, W), device=device)
    flat_idx = i * W + j
    # min-range per cell
    rng = frame["ranges"]
    range_map.view(-1).scatter_reduce_(0, flat_idx, rng, reduce="amin",
                                       include_self=True)
    # for intensity, take the value of the closest point per cell
    # simple heuristic: scatter then mean (good enough for MVP)
    cnt = torch.zeros_like(range_map.view(-1))
    intensity_map.view(-1).scatter_add_(0, flat_idx, frame["intensity"])
    cnt.scatter_add_(0, flat_idx, torch.ones_like(rng))
    intensity_map = (intensity_map / cnt.clamp(min=1).view(n_el, W))
    hit = (range_map.isfinite() & (range_map < cfg.far_plane)).float()
    range_map = torch.where(hit.bool(), range_map,
                            torch.full_like(range_map, cfg.far_plane))
    return range_map, intensity_map, hit


# --------------------------------------------------------------------- #
# Trainer
# --------------------------------------------------------------------- #
def train_lidar_head(
    rgb_ckpt: Path,
    processed_dir: Path,
    out_ckpt: Path,
    num_iters: int = 1500,
    head_cfg_overrides: Dict[str, Any] = None,
    log_freq: int = 50,
    device: str = "cuda",
):
    print(f"[phase2] loading RGB ckpt {rgb_ckpt}")
    # bootstrap drivestudio renderer (it pulls trainer + builds Background, etc)
    from aether.render.ds_backend import DriveStudioRenderer
    renderer = DriveStudioRenderer(str(rgb_ckpt), device=device)
    trainer = renderer.trainer
    if "Background" not in trainer.models:
        raise RuntimeError("Expected Background class in trainer.models")
    bg = trainer.models["Background"]
    N = bg.num_points
    print(f"[phase2] Background gaussians: N={N}")

    # freeze ALL trainer params (geometry stays frozen this phase)
    for p in trainer.parameters() if hasattr(trainer, "parameters") else []:
        p.requires_grad_(False)
    for m in trainer.models.values():
        if hasattr(m, "parameters"):
            for p in m.parameters(): p.requires_grad_(False)

    # build lidar head + per-gaussian feats
    cfg = LidarHeadConfig(**(head_cfg_overrides or {}))
    head = LidarHead(cfg).to(device)
    feats = GaussianLidarFeatures(N, device=device)

    # collect frozen background geometry (one tensor each) in WORLD frame.
    bg_means = bg._means.detach()
    bg_quats = bg.get_quats.detach()
    bg_scales = bg.get_scaling.detach()
    bg_opac = bg.get_opacity.detach().reshape(-1)
    print(f"[phase2] means range x[{bg_means[:,0].min():.1f},{bg_means[:,0].max():.1f}] "
          f"opac~{bg_opac.mean():.3f}")

    # load real lidar
    n_frames = renderer.num_frames
    print(f"[phase2] loading raw lidar for {n_frames} frames from {processed_dir}")
    lidar_frames = load_pandaset_lidar(processed_dir, 0, n_frames,
                                       only_360=True, device=device)
    print(f"[phase2] loaded; first frame has {lidar_frames[0]['ranges'].numel()} pts")

    # precompute raster grid
    raster_pts, teb = build_raster_pts(cfg, device)

    opt = torch.optim.Adam(
        list(feats.parameters()) + list(head.parameters()),
        lr=5e-2,
    )

    t0 = time.time()
    last_total = None
    for step in range(num_iters):
        f = step % n_frames
        frame = lidar_frames[f]
        gt_range, gt_int, hit = gt_to_spherical_grid(frame, cfg, device=device)

        opt.zero_grad()
        out = head.render(
            means_w=bg_means, quats=bg_quats, scales=bg_scales, opacities=bg_opac,
            # Pass intensity logits; LidarHead.render sigmoid-activates once.
            intensity=feats._intensity, raydrop_logit=feats.raydrop_logit,
            lidar_pose_w=frame["lidar2world"],
            raster_pts=raster_pts, tile_elevation_boundaries=teb,
        )
        losses = head.losses(
            out, gt_range, gt_int, hit,
            weights={"depth": 1.0, "intensity": 0.5, "raydrop": 0.5, "los": 0.01},
        )
        total = sum(losses.values())
        total.backward()
        opt.step()

        if step % log_freq == 0 or step == num_iters - 1:
            d = losses.get("lidar_depth", torch.tensor(0.)).item()
            it = losses.get("lidar_intensity", torch.tensor(0.)).item()
            rd = losses.get("lidar_raydrop", torch.tensor(0.)).item()
            ls = losses.get("lidar_los", torch.tensor(0.)).item()
            elap = time.time() - t0
            print(f"[phase2] step={step:5d} frame={f:2d} "
                  f"depth={d:.3f} int={it:.4f} drop={rd:.3f} los={ls:.4f} "
                  f"total={total.item():.3f} ({elap:.1f}s)")
            last_total = total.item()

    # save lidar sidecar (drivestudio's strict loader rejects unknown top-level
    # keys, so we keep the RGB ckpt untouched and write a companion file).
    print(f"[phase2] saving lidar sidecar to {out_ckpt}")
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    blob = dict(
        cfg=asdict(cfg),
        head_state={k: v.cpu() for k, v in head.state_dict().items()},
        bg_intensity=feats._intensity.detach().cpu(),
        bg_raydrop_logit=feats._raydrop_logit.detach().cpu(),
        bg_num_points=N,
        source_ckpt=str(rgb_ckpt),
    )
    torch.save(blob, out_ckpt)
    print(f"[phase2] DONE final_loss={last_total:.3f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--rgb_ckpt", required=True, type=Path)
    p.add_argument("--processed_dir", required=True, type=Path)
    p.add_argument("--out_ckpt", required=True, type=Path)
    p.add_argument("--num_iters", type=int, default=1500)
    a = p.parse_args()
    train_lidar_head(a.rgb_ckpt, a.processed_dir, a.out_ckpt, num_iters=a.num_iters)
