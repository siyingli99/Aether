"""Mode A LiDAR simulator.

Backprojects depth from each rendered camera into a single ego-frame point
cloud, then applies a coarse spherical FOV/range filter and (optional)
uniform downsampling to ``beams * az_samples`` points.

This is a minimal stand-in until the full SplatAD-style LiDAR path lands.
Returned points are in the ego (= vehicle body, DS world rotation removed)
frame, layout ``(N, 4) float32`` with ``columns = (x, y, z, intensity)``
and ``intensity = NaN`` because we have no reflectance signal yet.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass
class LidarSim:
    beams: int = 32
    az_samples: int = 1024
    hfov_deg: Tuple[float, float] = (-180.0, 180.0)
    vfov_deg: Tuple[float, float] = (-25.0, 5.0)
    range_max: float = 80.0
    range_min: float = 0.5
    # ego -> lidar mount (4x4). Identity = lidar at ego origin.
    mount_e2l: np.ndarray = field(default_factory=lambda: np.eye(4, dtype=np.float32))

    def simulate(self,
                 depths: Dict[str, np.ndarray],
                 cam_K:  Dict[str, np.ndarray],
                 cam_c2w_ds: Dict[str, np.ndarray],
                 ego_c2w_ds: np.ndarray,
                 ) -> np.ndarray:
        """Aggregate per-camera depths into a single ego-frame point cloud.

        Parameters
        ----------
        depths       cam_name -> (H, W) float32 depth (meters), 0 = invalid
        cam_K        cam_name -> (3, 3) intrinsics
        cam_c2w_ds   cam_name -> (4, 4) camera-to-DS-world pose
        ego_c2w_ds   (4, 4) ego-to-DS-world pose
        """
        ego_inv = np.linalg.inv(np.asarray(ego_c2w_ds, dtype=np.float32))
        all_pts: list[np.ndarray] = []
        for name, depth in depths.items():
            if depth is None:
                continue
            K  = np.asarray(cam_K[name], dtype=np.float32)
            c2w = np.asarray(cam_c2w_ds[name], dtype=np.float32)
            H, W = depth.shape
            uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                                 np.arange(H, dtype=np.float32))
            mask = (depth > self.range_min) & (depth < self.range_max) & np.isfinite(depth)
            if not mask.any():
                continue
            d  = depth[mask]
            u  = uu[mask]
            v  = vv[mask]
            # Backproject to camera frame (x right, y down, z forward).
            x = (u - K[0, 2]) * d / K[0, 0]
            y = (v - K[1, 2]) * d / K[1, 1]
            z = d
            pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=1)  # (M,4)
            pts_world = pts_cam @ c2w.T
            pts_ego = pts_world @ ego_inv.T
            all_pts.append(pts_ego[:, :3])

        if not all_pts:
            return np.zeros((0, 4), dtype=np.float32)

        pts = np.concatenate(all_pts, axis=0)
        # ego -> lidar mount
        if not np.allclose(self.mount_e2l, np.eye(4)):
            mnt_inv = np.linalg.inv(self.mount_e2l).astype(np.float32)
            pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), np.float32)], axis=1)
            pts = (pts_h @ mnt_inv.T)[:, :3]

        # Spherical FOV / range filter (lidar frame).
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r = np.linalg.norm(pts, axis=1)
        az = np.degrees(np.arctan2(y, x))
        el = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y) + 1e-9))
        ok = (
            (r > self.range_min) & (r < self.range_max)
            & (az >= self.hfov_deg[0]) & (az <= self.hfov_deg[1])
            & (el >= self.vfov_deg[0]) & (el <= self.vfov_deg[1])
        )
        pts = pts[ok]

        # Uniform random downsample to beams * az_samples.
        target = int(self.beams) * int(self.az_samples)
        if target > 0 and pts.shape[0] > target:
            idx = np.random.default_rng(0).choice(pts.shape[0], target, replace=False)
            pts = pts[idx]

        out = np.empty((pts.shape[0], 4), dtype=np.float32)
        out[:, :3] = pts
        out[:,  3] = np.nan
        return out

    @classmethod
    def from_cfg(cls, cfg) -> "LidarSim":
        """Build from a DictConfig-like node (cfg.scenario.lidar)."""
        kw: dict = {}
        for k in ("beams", "az_samples", "range_max", "range_min"):
            if hasattr(cfg, k):
                kw[k] = type(getattr(cls, k))(getattr(cfg, k))
        for k in ("hfov_deg", "vfov_deg"):
            if hasattr(cfg, k):
                v = list(getattr(cfg, k))
                kw[k] = (float(v[0]), float(v[1]))
        if hasattr(cfg, "mount_e2l"):
            kw["mount_e2l"] = np.asarray(list(cfg.mount_e2l), dtype=np.float32).reshape(4, 4)
        return cls(**kw)
