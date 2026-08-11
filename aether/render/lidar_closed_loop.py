"""Closed-loop SplatAD LiDAR renderer.

Renders a full LiDAR sweep from the (dynamic) Gaussian scene during a gym
rollout, replacing the depth-backprojection Mode-A stand-in in
``aether/sensors/lidar.py``.

Key differences from the joint-phase fitter (which used object-LOCAL means and
only the Background class):

* Gathers WORLD-frame gaussians from *every* drivestudio node — Background plus
  the dynamic RigidNodes / DeformableNodes / SMPLNodes — each through its own
  ``get_gaussians()`` so the per-frame instance pose (including any closed-loop
  actor-pose overrides) is applied before the sweep.
* Attaches the per-gaussian LiDAR features (intensity + ray-drop logit) trained
  in the joint phase and stored in the sidecar ``.pt``.
* Back-projects the rendered spherical range image into an ego/lidar-frame
  point cloud ``(N, 4) = (x, y, z, intensity)``.

This module is read-only w.r.t. the renderer state: actor-pose overrides are
applied/restored around the sweep so the trainer is left clean.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from aether.render.lidar_head import LidarHead, LidarHeadConfig, build_raster_pts


class ClosedLoopLidar:
    """LiDAR-head sweep on top of a (frozen) DriveStudioRenderer."""

    def __init__(self, renderer, sidecar_path: str, device: str = "cuda"):
        self.renderer = renderer
        self.device = torch.device(device)

        sc = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        self.cfg = LidarHeadConfig(**sc["cfg"])
        self.head = LidarHead(self.cfg).to(self.device)
        self.head.load_state_dict(sc["head_state"])
        self.head.eval()

        # Per-class sidecar features (raw logits) kept on device.
        self.feat_intensity: Dict[str, torch.Tensor] = {}
        self.feat_raydrop: Dict[str, torch.Tensor] = {}
        for cls, d in sc["feats"].items():
            self.feat_intensity[cls] = d["_intensity"].to(self.device)    # [N,1] logit
            self.feat_raydrop[cls] = d["_raydrop_logit"].to(self.device)  # [N,1] logit

        self.raster_pts, self.teb = build_raster_pts(self.cfg, self.device)
        self._classes = list(sc["feats"].keys())

    # ------------------------------------------------------------------ #
    def _dummy_cam(self, pose_w: torch.Tensor):
        """Minimal camera so dynamic nodes' get_gaussians() can compute SH
        viewdirs (only camtoworlds[:3,3] is read)."""
        from models.gaussians.basics import dataclass_camera
        c2w = torch.as_tensor(pose_w, dtype=torch.float32, device=self.device)
        return dataclass_camera(
            camtoworlds=c2w, camtoworlds_gt=c2w,
            Ks=torch.eye(3, dtype=torch.float32, device=self.device), H=1, W=1,
        )

    @torch.no_grad()
    def _gather_world(self, cam, background_only: bool = False,
                      dynamic_only: bool = False) -> Tuple[torch.Tensor, ...]:
        """Concatenate world-frame geometry + LiDAR features across classes."""
        means_l, quats_l, scales_l, opac_l, int_l, rd_l = [], [], [], [], [], []
        for cls in self._classes:
            if background_only and cls != "Background":
                continue
            if dynamic_only and cls == "Background":
                continue
            model = self.renderer.trainer.models.get(cls)
            if model is None:
                continue

            if hasattr(model, "transform_means"):           # dynamic node
                gs = model.get_gaussians(cam)
                if gs is None:                              # no visible instance
                    continue
                means = gs["_means"]
                quats = gs["_quats"]
                scales = gs["_scales"]
                opac = gs["_opacities"].reshape(-1)
            else:                                            # static Background
                means = model._means
                quats = model.get_quats
                scales = model.get_scaling
                opac = model.get_opacity.reshape(-1)

            n = means.shape[0]
            fi = self.feat_intensity.get(cls)
            fr = self.feat_raydrop.get(cls)
            if fi is None or fr is None:
                continue
            if fi.shape[0] != n:
                # Count drift vs sidecar (e.g. densify mismatch). Align by
                # truncate/pad with neutral values so the sweep never crashes.
                m = min(fi.shape[0], n)
                fi_full = torch.zeros(n, 1, device=self.device)
                fr_full = torch.zeros(n, 1, device=self.device)
                fi_full[:m] = fi[:m]
                fr_full[:m] = fr[:m]
                fi, fr = fi_full, fr_full

            means_l.append(means)
            quats_l.append(quats)
            scales_l.append(scales)
            opac_l.append(opac)
            # Pass intensity LOGITS. LidarHead.render applies sigmoid after
            # rasterization; pre-sigmoiding here double-activates and collapses
            # predicted intensity near 0.5 (I-MAE ≈ 0.45 vs GT ≈ 0.05).
            int_l.append(fi)
            rd_l.append(fr)                    # raw logits

        if not means_l:
            raise RuntimeError("ClosedLoopLidar: no gaussians gathered for sweep")

        return (
            torch.cat(means_l, 0), torch.cat(quats_l, 0), torch.cat(scales_l, 0),
            torch.cat(opac_l, 0), torch.cat(int_l, 0), torch.cat(rd_l, 0),
        )

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def render_sweep(self, lidar_pose_w: np.ndarray, frame_idx: int,
                     background_only: bool = False, dynamic_only: bool = False,
                     apply_overrides: bool = True) -> Dict[str, torch.Tensor]:
        """Render one spherical sweep. ``lidar_pose_w`` is lidar-to-DS-world."""
        snapshot = {}
        if apply_overrides:
            snapshot = self.renderer._apply_overrides(int(frame_idx))
        try:
            self.renderer._set_cur_frame(int(frame_idx))
            cam = self._dummy_cam(lidar_pose_w)
            means, quats, scales, opac, inten, rd = self._gather_world(
                cam, background_only=background_only, dynamic_only=dynamic_only)
            pose = torch.as_tensor(lidar_pose_w, dtype=torch.float32,
                                   device=self.device)
            out = self.head.render(
                means_w=means, quats=quats, scales=scales, opacities=opac,
                intensity=inten, raydrop_logit=rd, lidar_pose_w=pose,
                raster_pts=self.raster_pts, tile_elevation_boundaries=self.teb,
            )
        finally:
            if snapshot:
                self.renderer._restore_overrides(int(frame_idx), snapshot)
        return out

    # ------------------------------------------------------------------ #
    def sweep_to_points(self, out: Dict[str, torch.Tensor],
                        raydrop_thresh: float = 0.3,
                        alpha_thresh: float = 0.1) -> np.ndarray:
        """Back-project a rendered sweep into a lidar-frame point cloud
        ``(N, 4) = (x, y, z, intensity)``.

        x-forward, y-left, z-up (sensor frame implied by the head's spherical
        grid: az measured from +x toward +y, el from xy-plane toward +z).
        """
        range_img = out["range"]                       # [n_el, W]
        raster = out["raster_pts"]                     # [n_el, W, 4]
        az = torch.deg2rad(raster[..., 0])
        el = torch.deg2rad(raster[..., 1])

        prob_keep = 1.0 - torch.sigmoid(out["raydrop_logit"])
        valid = (
            (prob_keep > raydrop_thresh)
            & (out["alpha"] > alpha_thresh)
            & (range_img > self.cfg.near_plane)
            & (range_img < self.cfg.far_plane)
        )

        r = range_img[valid]
        a = az[valid]
        e = el[valid]
        x = r * torch.cos(e) * torch.cos(a)
        y = r * torch.cos(e) * torch.sin(a)
        z = r * torch.sin(e)
        inten = out["intensity"][valid]
        pts = torch.stack([x, y, z, inten], dim=-1)
        return pts.detach().cpu().numpy().astype(np.float32)
