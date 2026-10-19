"""SplatAD sweep + unified LiDAR refiner -> ``obs['lidar']`` for closed-loop AD.

Used by :mod:`aether.sensors.provider` when a SplatAD LiDAR sidecar exists
for the scene (``lidar.sidecar`` or next to the RGB checkpoint).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from aether.render.lidar_closed_loop import ClosedLoopLidar
from aether.refine.unified_multimodal_enhancer import load_unified_bundle
from aether.tools.multimodal_cl_viz import (
    build_ref_bank,
    sweep_arrays,
    enhance_sweep,
    spherical_to_obs_cloud,
)

DEFAULT_LIDAR_BUNDLE = (
    "outputs/multimodal_refine_v2/unified/unified_multimodal.bundle.pth"
)
# Prefer freeze-geometry sidecars (compatible with original RGB ckpt).
DEFAULT_SIDECAR_CANDIDATES = (
    "lidar_sidecar_frozen.pt",
    "lidar_sidecar_frozen_step10000.pt",
    "lidar_sidecar_step10000.pt",
)


def resolve_lidar_sidecar(checkpoint: str | os.PathLike | None,
                          explicit: Optional[str] = None) -> Optional[str]:
    """Explicit path, else a sidecar next to the RGB checkpoint (frozen first)."""
    if explicit:
        p = os.path.abspath(os.path.expanduser(str(explicit)))
        return p if os.path.isfile(p) else None
    if not checkpoint:
        return None
    ckpt_dir = os.path.dirname(os.path.abspath(str(checkpoint)))
    for name in DEFAULT_SIDECAR_CANDIDATES:
        cand = os.path.join(ckpt_dir, name)
        if os.path.isfile(cand):
            return cand
    return None


def resolve_lidar_bundle(explicit: Optional[str], repo_root: str) -> Optional[str]:
    path = explicit or DEFAULT_LIDAR_BUNDLE
    if not os.path.isabs(path):
        path = os.path.join(repo_root, path)
    return path if os.path.isfile(path) else None


class SplatAdLidarProvider:
    """Render (and optionally refine) a LiDAR sweep at a given sensor pose."""

    def __init__(self, renderer, sidecar_path: str, bundle_path: Optional[str],
                 device: str = "cuda"):
        print(f"[splatad-lidar] sidecar: {sidecar_path}", flush=True)
        print(f"[splatad-lidar] bundle:  {bundle_path or 'disabled (raw full-RD sweep)'}", flush=True)
        self._cl = ClosedLoopLidar(renderer, sidecar_path, device=device)
        self._enhancer = load_unified_bundle(bundle_path, device=device) if bundle_path else None
        self._ref_bank = build_ref_bank(renderer, self._cl.cfg, device=device) if bundle_path else None
        self._far = float(self._cl.cfg.far_plane)
        self._near = float(self._cl.cfg.near_plane)
        self._raster_np = self._cl.raster_pts[0].detach().cpu().numpy()

    def sweep(self, lidar_pose_scene: np.ndarray, frame_idx: int) -> np.ndarray:
        """(N, 4) points in the LiDAR frame for a scene-frame sensor pose."""
        pose = np.asarray(lidar_pose_scene, dtype=np.float32)
        sweep = self._cl.render_sweep(pose, int(frame_idx), apply_overrides=True)
        if self._enhancer is None:
            return self._cl.sweep_to_points(sweep)
        in_r, in_i, in_k = sweep_arrays(sweep, self._far)
        ref = self._ref_bank.nearest(pose[:2, 3])
        try:
            pr, pi, pk = enhance_sweep(self._enhancer, in_r, in_i, in_k, ref)
        except Exception as exc:
            print(f"[splatad-lidar] refine failed ({exc}); using raw sweep", flush=True)
            pr, pi, pk = in_r, in_i, in_k
        pts, _ = spherical_to_obs_cloud(pr, pi, pk, self._raster_np, self._near, self._far)
        return pts
