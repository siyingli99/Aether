"""Multimodal closed-loop helpers: real 3DGS LiDAR sweep + refiner enhancement
and per-timestep visualizations (RGB, point-cloud-on-RGB, BEV) before/after.

Used by ``scripts/closed_loop_multimodal.py``. Kept import-light so it can be
reused by offline tools too. Runs in the ``aether`` conda env.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# LiDAR reference bank (nearest training-frame GT sweep, matches refiner train) #
# --------------------------------------------------------------------------- #
class LidarRefBank:
    """Precomputed GT spherical sweeps for the checkpoint's own scene, used as
    the reference branch (x_ref) of the CrossModalLidarRefiner in closed loop.
    Picks the nearest bank entry by ego XY in DS world."""

    def __init__(self, far: float):
        self.far = float(far)
        self._xy: List[np.ndarray] = []
        self._ref: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def add(self, pose_xy: np.ndarray, gt_range, gt_int, gt_mask):
        self._xy.append(np.asarray(pose_xy, np.float64))
        self._ref.append((gt_range.astype(np.float32),
                          gt_int.astype(np.float32),
                          np.clip(gt_mask, 0, 1).astype(np.float32)))

    def __len__(self):
        return len(self._ref)

    def nearest(self, ego_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = np.asarray(ego_xy, np.float64)
        best_i, best_d = 0, 1e18
        for i, xy in enumerate(self._xy):
            d = float(np.sum((xy - q) ** 2))
            if d < best_d:
                best_d, best_i = d, i
        return self._ref[best_i]


def build_ref_bank(renderer, head_cfg, device: str = "cuda") -> LidarRefBank:
    """Project every training frame's GT LiDAR to a spherical sweep bank."""
    from omegaconf import OmegaConf
    from aether.refine.pair_utils import build_lidar_source, project_gt_frame

    cfg = OmegaConf.create({"data": OmegaConf.to_container(renderer.cfg.data, resolve=True)})
    src, data_path = build_lidar_source(cfg, device)
    lidar_bin = Path(data_path) / "lidar"
    bank = LidarRefBank(float(head_cfg.far_plane))
    n = src.lidar_to_worlds.shape[0]
    for f in range(n):
        gt = project_gt_frame(src, f, lidar_bin, head_cfg, device)
        bank.add(gt["pose"][:2, 3], gt["gt_range"], gt["gt_int"], gt["gt_mask"])
    return bank


# --------------------------------------------------------------------------- #
# LiDAR enhancement                                                            #
# --------------------------------------------------------------------------- #
def sweep_arrays(out: Dict[str, torch.Tensor], far: float):
    """Extract (range[m], intensity, keep_prob) numpy arrays from a raw sweep."""
    rng = out["range"].clamp(0, far).detach().cpu().numpy().astype(np.float32)
    inten = out["intensity"].clamp(0, 1).detach().cpu().numpy().astype(np.float32)
    keep = (1 - torch.sigmoid(out["raydrop_logit"])).clamp(0, 1).detach().cpu().numpy().astype(np.float32)
    return rng, inten, keep


def enhance_sweep(enhancer, in_range, in_int, in_keep, ref):
    """Run the unified LiDAR refiner. ``ref`` = (range,int,keep) from the bank."""
    ref_r, ref_i, ref_k = ref
    return enhancer.enhance_lidar_arrays(in_range, in_int, in_keep, ref_r, ref_i, ref_k)


def spherical_to_points(rng, keep, raster_pts, near, far,
                        keep_thresh: float = 0.3) -> np.ndarray:
    """Back-project a (range, keep) spherical map into a lidar-frame point cloud
    (x-fwd, y-left, z-up), (N,3)."""
    pts, _ = spherical_to_obs_cloud(
        rng, None, keep, raster_pts, near, far, keep_thresh=keep_thresh,
    )
    return pts


def spherical_to_obs_cloud(rng, intensity, keep, raster_pts, near, far,
                           keep_thresh: float = 0.3) -> Tuple[np.ndarray, int]:
    """Back-project spherical (range, intensity, keep) into lidar-frame (N,4).

    Columns: x, y, z, intensity.  Intensity defaults to 0 when ``intensity``
    is None.  Returns (cloud, num_valid).
    """
    az = np.deg2rad(raster_pts[..., 0])
    el = np.deg2rad(raster_pts[..., 1])
    valid = (keep > keep_thresh) & (rng > near) & (rng < far)
    r = rng[valid]
    a = az[valid]
    e = el[valid]
    x = r * np.cos(e) * np.cos(a)
    y = r * np.cos(e) * np.sin(a)
    z = r * np.sin(e)
    out = np.empty((r.shape[0], 4), dtype=np.float32)
    out[:, 0] = x
    out[:, 1] = y
    out[:, 2] = z
    if intensity is not None:
        out[:, 3] = intensity[valid]
    else:
        out[:, 3] = 0.0
    return out, int(r.shape[0])


# --------------------------------------------------------------------------- #
# Visualization                                                               #
# --------------------------------------------------------------------------- #
def project_points_on_image(pts_lidar: np.ndarray, l2c: np.ndarray, K_full: np.ndarray,
                            rig_hw: Tuple[int, int], img: np.ndarray) -> np.ndarray:
    """Overlay LiDAR points (colored by depth) on an RGB image.

    ``pts_lidar`` (N,3) in lidar frame; ``l2c`` lidar->camera (OpenCV +z fwd);
    ``K_full`` intrinsics at ``rig_hw``=(H,W); ``img`` HxWx3 uint8 (possibly a
    different resolution -> K is scaled)."""
    Himg, Wimg = img.shape[:2]
    rigH, rigW = rig_hw
    sx, sy = Wimg / float(rigW), Himg / float(rigH)
    fx, fy = K_full[0, 0] * sx, K_full[1, 1] * sy
    cx, cy = K_full[0, 2] * sx, K_full[1, 2] * sy

    if pts_lidar.shape[0] == 0:
        return img.copy()
    R = l2c[:3, :3]; t = l2c[:3, 3]
    cam = pts_lidar @ R.T + t
    z = cam[:, 2]
    front = z > 0.1
    cam = cam[front]; z = z[front]
    if cam.shape[0] == 0:
        return img.copy()
    u = (fx * cam[:, 0] / z + cx).astype(np.int32)
    v = (fy * cam[:, 1] / z + cy).astype(np.int32)
    inb = (u >= 0) & (u < Wimg) & (v >= 0) & (v < Himg)
    u, v, z = u[inb], v[inb], z[inb]

    out = img.copy()
    if u.shape[0] == 0:
        return out
    import matplotlib.cm as cm
    d = np.clip(z / 60.0, 0, 1)
    colors = (cm.turbo(1.0 - d)[:, :3] * 255).astype(np.uint8)
    rad = 1
    for du in range(-rad, rad + 1):
        for dv in range(-rad, rad + 1):
            uu = np.clip(u + du, 0, Wimg - 1)
            vv = np.clip(v + dv, 0, Himg - 1)
            out[vv, uu] = colors
    return out


def bev_scatter(ax, pts: np.ndarray, rng_m: float = 50.0, title: str = "",
                color=None):
    ax.clear()
    if pts.shape[0] > 0:
        c = pts[:, 2] if color is None else color
        ax.scatter(pts[:, 1], pts[:, 0], s=0.6, c=c, cmap="viridis",
                   vmin=-2, vmax=6, linewidths=0)
    ax.plot([0], [0], "r^", markersize=8)
    ax.set_xlim(rng_m, -rng_m)   # y-left -> flip so left is left
    ax.set_ylim(-rng_m, rng_m)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.3)


def make_step_panel(
    rgb_raw: np.ndarray, rgb_enh: np.ndarray,
    proj_raw: np.ndarray, proj_enh: np.ndarray,
    pts_raw: np.ndarray, pts_enh: np.ndarray,
    step: int, timestamp: float, out_path: str, bev_range: float = 50.0,
) -> np.ndarray:
    """2x3 panel: [RGB raw|enh], [proj raw|enh], [BEV raw|enh]. Returns RGB frame."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 9))
    axes[0, 0].imshow(rgb_raw); axes[0, 0].set_title("RGB (raw render)")
    axes[0, 1].imshow(rgb_enh); axes[0, 1].set_title("RGB (Difix enhanced)")
    axes[0, 2].imshow(proj_raw); axes[0, 2].set_title("LiDAR->RGB (raw)")
    for ax in axes[0]:
        ax.axis("off")
    axes[1, 0].imshow(proj_enh); axes[1, 0].set_title("LiDAR->RGB (refined)")
    axes[1, 0].axis("off")
    bev_scatter(axes[1, 1], pts_raw, bev_range, f"BEV raw  (N={pts_raw.shape[0]})")
    bev_scatter(axes[1, 2], pts_enh, bev_range, f"BEV refined  (N={pts_enh.shape[0]})")
    fig.suptitle(f"step {step:04d}  t={timestamp:.2f}s", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    frame = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return frame
