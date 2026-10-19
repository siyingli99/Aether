"""LiDAR for agents that need it (LTF, GaussianFusion …).

Two sources, chosen automatically:

1. **SplatAD** – a learned LiDAR head (sidecar checkpoint) renders a real
   spinning sweep from the Gaussians, optionally refined by the unified
   multimodal enhancer.  Used when ``lidar.sidecar`` is set or a sidecar
   sits next to the scene checkpoint.
2. **Depth back-projection** – rendered camera depth lifted to 3D and
   resampled; always available, no reflectance.

Points are returned in the scene rig's ``LIDAR_TOP`` frame (nuScenes:
x right, y forward, z up), ``(N, 4) float32 = x, y, z, intensity``.
"""
from __future__ import annotations

import numpy as np

from .lidar import LidarSim


class LidarProvider:
    def __init__(self, env, cfg) -> None:
        self.env = env
        self.splatad = None
        if cfg.splatad:
            from ..tools.splatad_lidar_obs import (resolve_lidar_bundle, resolve_lidar_sidecar,
                                                    SplatAdLidarProvider)
            from ..paths import REPO_ROOT
            sidecar = resolve_lidar_sidecar(env.bundle.checkpoint, cfg.sidecar)
            if sidecar:
                bundle = resolve_lidar_bundle(cfg.bundle, str(REPO_ROOT)) if cfg.refine else None
                self.splatad = SplatAdLidarProvider(env.renderer, sidecar, bundle)
        self.depth = LidarSim()
        self.source = "splatad" if self.splatad is not None else "depth"
        self.needs_depth = self.splatad is None

    def sweep(self, depths: dict[str, np.ndarray] | None = None) -> np.ndarray:
        env = self.env
        lidar_pose = env.sensor_pose("LIDAR_TOP")
        if self.splatad is not None:
            return self.splatad.sweep(lidar_pose, env.frame)
        cams = {n: env.cameras[n].K for n in depths}
        poses = {n: env.camera_pose(n) for n in depths}
        return self.depth.simulate(depths=depths, cam_K=cams, cam_c2w_ds=poses,
                                   ego_c2w_ds=lidar_pose)
