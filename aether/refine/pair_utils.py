"""Shared helpers for LiDAR / multimodal refiner training data."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from aether.paths import AETHER_ROOT, resolve_data_root
from aether.render.lidar_head import LidarHeadConfig
from aether.train.joint_phase import gt_to_spherical_grid


def pack_lidar_rgb(range_m: np.ndarray, inten: np.ndarray, third: np.ndarray,
                   far: float) -> np.ndarray:
    r = np.clip(range_m / far, 0.0, 1.0)
    g = np.clip(inten, 0.0, 1.0)
    b = np.clip(third, 0.0, 1.0)
    return (np.stack([r, g, b], -1) * 255.0).astype(np.uint8)


def scene_folder_name(scene_idx: int) -> str:
    return f"{int(scene_idx):03d}"


def latest_lidar_sidecar(jdir: Path) -> Path:
    """Pick the sidecar with the highest training step (not lexicographic)."""
    import re
    sidecars = list(jdir.glob("lidar_sidecar_step*.pt"))
    if not sidecars:
        raise FileNotFoundError(f"no lidar sidecar in {jdir}")

    def _step(p: Path) -> int:
        m = re.search(r"step(\d+)", p.name)
        return int(m.group(1)) if m else -1

    return max(sidecars, key=_step)


def list_processed_scenes(data_root: Path) -> List[int]:
    out: List[int] = []
    if not data_root.is_dir():
        return out
    for p in sorted(data_root.iterdir()):
        if not p.is_dir():
            continue
        if not (p / "lidar").is_dir():
            continue
        try:
            out.append(int(p.name))
        except ValueError:
            continue
    return out


def build_lidar_source(cfg, device: str = "cuda"):
    from utils.misc import import_str
    dcfg = cfg.data
    data_path = resolve_data_root(dcfg.data_root) / scene_folder_name(int(dcfg.scene_idx))
    if not data_path.is_dir():
        raise FileNotFoundError(data_path)
    pose_dir = "lidar_pose" if (data_path / "lidar_pose").is_dir() else "ego_pose"
    total = len(list((data_path / pose_dir).iterdir()))
    end = (total - 1 if dcfg.end_timestep == -1 else dcfg.end_timestep) + 1
    start = int(dcfg.start_timestep)
    lidar = import_str(dcfg.lidar_source.type)(
        dcfg.lidar_source, str(data_path), start, end, device=device)
    lidar.to(device)
    return lidar, str(data_path)


def read_bin_intensity(bin_path: Path, n_rays: int, device,
                       only_use_360_lidar: bool = False) -> torch.Tensor:
    """Read per-ray intensity from a processed lidar .bin file.

    nuScenes bins are Nx4 (x,y,z,i); PandaSet bins are Nx5 (x,y,z,i,laser_id).
    """
    if not bin_path.exists():
        return torch.zeros(n_rays, device=device)
    raw = np.fromfile(bin_path, dtype=np.float32)
    if raw.size % 5 == 0:
        arr = raw.reshape(-1, 5)
        if only_use_360_lidar:
            arr = arr[arr[:, 4] == 0]
        inten = arr[:, 3].astype(np.float32) / 255.0
    elif raw.size % 4 == 0:
        arr = raw.reshape(-1, 4)
        inten = arr[:, 3].astype(np.float32) / 255.0
    else:
        return torch.zeros(n_rays, device=device)
    t = torch.from_numpy(inten.copy()).float().to(device)
    if t.numel() == n_rays:
        return t
    return torch.zeros(n_rays, device=device)


def project_gt_frame(src, frame: int, lidar_bin_dir: Path, cfg: LidarHeadConfig,
                     device: str = "cuda") -> Dict[str, np.ndarray]:
    lidar2world = src.lidar_to_worlds[frame].to(device)
    rays = src.get_lidar_rays(frame)
    ranges = rays["lidar_ranges"].squeeze(-1).to(device)
    only_360 = bool(getattr(getattr(src, "data_cfg", None), "only_use_360_lidar", False))
    inten = read_bin_intensity(
        lidar_bin_dir / f"{frame:03d}.bin", ranges.numel(), device,
        only_use_360_lidar=only_360,
    )
    gt_range, gt_int, hit = gt_to_spherical_grid(
        rays["lidar_origins"][0].to(device),
        rays["lidar_viewdirs"].to(device),
        ranges, inten, lidar2world, cfg,
    )
    pose = lidar2world.detach().cpu().numpy().astype(np.float32)
    return {
        "gt_range": gt_range.detach().cpu().numpy().astype(np.float32),
        "gt_int": gt_int.detach().cpu().numpy().astype(np.float32),
        "gt_mask": hit.detach().cpu().numpy().astype(np.float32),
        "pose": pose,
    }


def pose_xy(pose: np.ndarray) -> np.ndarray:
    return pose[:2, 3].astype(np.float64)


def nearest_ref_frame(poses: List[np.ndarray], query: int,
                      exclude: Optional[int] = None) -> int:
    q = pose_xy(poses[query])
    best_j, best_d = 0, float("inf")
    for j, p in enumerate(poses):
        if j == query or (exclude is not None and j == exclude):
            continue
        d = float(np.linalg.norm(pose_xy(p) - q))
        if d < best_d:
            best_d, best_j = d, j
    return best_j


def synthetic_degrade_lidar(
    gt_range: np.ndarray,
    gt_int: np.ndarray,
    gt_mask: np.ndarray,
    far: float,
    rng: np.random.Generator,
    severity: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate typical 3DGS LiDAR artifacts on GT spherical maps."""
    hit = gt_mask > 0.5
    in_r = gt_range.copy()
    in_i = gt_int.copy()
    in_k = hit.astype(np.float32)

    if not hit.any():
        return in_r, in_i, in_k

    # range noise + occasional spikes
    sigma = 0.35 * severity
    noise = rng.normal(0.0, sigma, size=in_r.shape).astype(np.float32)
    spike = rng.random(in_r.shape) < (0.008 * severity)
    noise = np.where(spike, rng.uniform(2.0, 8.0, size=in_r.shape), noise)
    in_r = np.where(hit, np.clip(in_r + noise, 0.0, far), in_r)

    # boundary holes (ray-drop errors)
    hole_p = 0.04 * severity
    holes = (rng.random(in_k.shape) < hole_p) & hit
    in_k = np.where(holes, 0.0, in_k)

    # intensity drift
    in_i = np.where(hit, np.clip(in_i + rng.normal(0, 0.12 * severity, in_i.shape), 0, 1), in_i)

    # weak cells far from sensor
    far_cells = hit & (gt_range > 0.55 * far)
    drop_far = rng.random(in_k.shape) < (0.02 * severity)
    in_k = np.where(far_cells & drop_far, in_k * 0.3, in_k)

    return in_r.astype(np.float32), in_i.astype(np.float32), in_k.astype(np.float32)


def save_pair_npz(
    path: Path,
    in_range, in_int, in_keep,
    ref_range, ref_int, ref_keep,
    gt_range, gt_int, gt_mask,
    meta: Optional[dict] = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(
        in_range=in_range, in_int=in_int, in_keep=in_keep,
        ref_range=ref_range, ref_int=ref_int, ref_keep=ref_keep,
        gt_range=gt_range, gt_int=gt_int, gt_mask=gt_mask,
    )
    if meta:
        for k, v in meta.items():
            kw[f"meta_{k}"] = np.array(v)
    np.savez_compressed(path, **kw)
