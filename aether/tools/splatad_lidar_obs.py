"""SplatAD sweep + unified LiDAR refiner -> ``obs['lidar']`` for closed-loop AD.

Used by ``scripts/closed_loop.py`` when ``--lidar_sidecar`` / joint sidecar
auto-resolve is enabled for UniAD multimodal simulation.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch

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


def resolve_lidar_sidecar(cfg, explicit: Optional[str] = None) -> Optional[str]:
    """Resolve LiDAR sidecar for closed-loop.

    Order:
      1) CLI ``--lidar_sidecar``
      2) ``scenario.lidar_sidecar`` in YAML
      3) siblings of RGB checkpoint (prefer frozen*)
    """
    if explicit:
        p = os.path.abspath(explicit)
        return p if os.path.isfile(p) else None

    scen_side = getattr(cfg.scenario, "lidar_sidecar", None)
    if scen_side:
        p = os.path.abspath(str(scen_side))
        if os.path.isfile(p):
            return p
        print(f"[splatad-lidar] WARN: scenario.lidar_sidecar missing: {p}", flush=True)

    ckpt = getattr(cfg.scenario, "checkpoint", None)
    if not ckpt:
        return None
    ckpt_dir = os.path.dirname(os.path.abspath(str(ckpt)))
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
    """Render + refine a LiDAR sweep at the current ego pose."""

    def __init__(self, env, sidecar_path: str, bundle_path: Optional[str],
                 device: str = "cuda"):
        self._env = env
        self._device = device
        print(f"[splatad-lidar] sidecar: {sidecar_path}", flush=True)
        print(f"[splatad-lidar] bundle:  {bundle_path or 'disabled (raw full-RD sweep)'}", flush=True)
        self._cl = ClosedLoopLidar(env.renderer, sidecar_path, device=device)
        self._enhancer = (
            load_unified_bundle(bundle_path, device=device) if bundle_path else None
        )
        self._ref_bank = (
            build_ref_bank(env.renderer, self._cl.cfg, device=device)
            if bundle_path else None
        )
        self._far = float(self._cl.cfg.far_plane)
        self._near = float(self._cl.cfg.near_plane)
        self._raster_np = self._cl.raster_pts[0].detach().cpu().numpy()
        if self._ref_bank is not None:
            print(f"[splatad-lidar] ref bank size = {len(self._ref_bank)}", flush=True)

    def enhance_obs_lidar(self, frame_idx: Optional[int] = None) -> np.ndarray:
        env = self._env
        if frame_idx is None:
            frame_idx = min(env.frame_idx, env._max_frame)
        ego_ds = env._ego_c2w_ds().astype(np.float32)
        sweep = self._cl.render_sweep(
            ego_ds, frame_idx, apply_overrides=True,
        )
        if self._enhancer is None:
            return self._cl.sweep_to_points(sweep)
        in_r, in_i, in_k = sweep_arrays(sweep, self._far)
        ref = self._ref_bank.nearest(ego_ds[:2, 3])
        try:
            pr, pi, pk = enhance_sweep(self._enhancer, in_r, in_i, in_k, ref)
        except Exception as exc:
            print(f"[splatad-lidar] refine failed ({exc}); using raw sweep", flush=True)
            pr, pi, pk = in_r, in_i, in_k
        pts, _ = spherical_to_obs_cloud(
            pr, pi, pk, self._raster_np, self._near, self._far,
        )
        return pts


def maybe_create_provider(env, cfg, sidecar: Optional[str], bundle: Optional[str],
                          repo_root: str, needs_lidar: bool,
                          enabled: bool = True,
                          refine: bool = True) -> Optional[SplatAdLidarProvider]:
    if not enabled or not needs_lidar:
        return None
    sidecar_path = resolve_lidar_sidecar(cfg, sidecar)
    if not sidecar_path:
        return None
    bundle_path = resolve_lidar_bundle(bundle, repo_root) if refine else None
    if refine and not bundle_path:
        print("[splatad-lidar] WARN: bundle not found; using depth-backproj LiDAR",
              flush=True)
        return None
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SplatAdLidarProvider(env, sidecar_path, bundle_path, device=device)
