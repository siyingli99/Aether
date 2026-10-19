"""LiDAR head as a standalone, swappable module.

Train-once design: this module owns two per-gaussian features (intensity and
ray-drop logit), exposes a `render()` method that produces a spherical LiDAR
sweep (range, intensity, ray-drop, alpha), and provides loss helpers. The
host reconstruction backbone (currently drivestudio's 3DGS) is responsible
for storing the per-gaussian features and feeding them through `extras` of
the activated Gaussian set.

gsplat collision:
  DriveStudio uses upstream ``gsplat`` 1.3.0 from site-packages. The LiDAR
  rasterization kernel lives in the vendored fork
  ``AETHER/third_party/splatad_gsplat`` (imported as ``splatad_gsplat``) so the
  two do not collide.
"""
from __future__ import annotations

from aether.paths import use_vendored_backends
use_vendored_backends()  # DriveStudio / SplatAD on sys.path

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import aether.paths  # noqa: F401  — puts third_party/splatad_gsplat on sys.path

try:
    from splatad_gsplat.rendering import lidar_rasterization  # type: ignore
except ImportError as e:
    from aether.paths import SPLATAD_GSPLAT_ROOT
    raise ImportError(
        f"LiDAR gsplat fork not importable as `splatad_gsplat` from "
        f"{SPLATAD_GSPLAT_ROOT}. Original error: {e}"
    ) from e



# ----------------------------- config ---------------------------------------


@dataclass
class LidarHeadConfig:
    n_elevation_channels: int = 64
    azimuth_resolution: float = 0.2       # degrees per column
    min_azimuth: float = -180.0
    max_azimuth: float = 180.0
    min_elevation: float = -30.0
    max_elevation: float = 10.0
    tile_width: int = 32
    tile_height: int = 8
    near_plane: float = 0.1
    far_plane: float = 100.0
    rolling_shutter: bool = False         # set True when feeding velocities


# ----------------------------- per-gaussian features -------------------------


class GaussianLidarFeatures(nn.Module):
    """Per-gaussian (intensity, raydrop_logit) parameters.

    The host backbone owns the gaussian centres (means/quats/scales/opacities/
    color). This module owns the LiDAR-only extras. It exposes the two
    parameters as nn.Parameter so the host's optimizer/densify/prune logic can
    grow/shrink them in lockstep with the gaussian count by re-binding the
    parameter tensors directly (Parameter(...)).
    """

    def __init__(self, n: int = 1, device: str | torch.device = "cuda"):
        super().__init__()
        self._intensity = nn.Parameter(torch.zeros(n, 1, device=device))
        self._raydrop_logit = nn.Parameter(torch.zeros(n, 1, device=device))

    @property
    def intensity(self) -> torch.Tensor:           # sigmoid-activated [0,1]
        return torch.sigmoid(self._intensity)

    @property
    def raydrop_logit(self) -> torch.Tensor:       # raw logits, no activation
        return self._raydrop_logit

    @property
    def n(self) -> int:
        return self._intensity.shape[0]

    @torch.no_grad()
    def resize_to(self, n: int) -> None:
        """Reset both params to zeros with the new count (used after a
        host-side load-from-checkpoint that resets gaussian count)."""
        device = self._intensity.device
        self._intensity = nn.Parameter(torch.zeros(n, 1, device=device))
        self._raydrop_logit = nn.Parameter(torch.zeros(n, 1, device=device))


# ----------------------------- raster grid helper ---------------------------


def build_raster_pts(
    cfg: LidarHeadConfig, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the [1,n_el,W,4] (azimuth_deg, elevation_deg, range_init, time)
    raster_pts grid and the [n_el//tile_h + 1] tile elevation boundaries
    consumed by splatad's lidar_rasterization."""
    n_el = cfg.n_elevation_channels
    assert n_el % cfg.tile_height == 0, (
        f"n_elevation_channels={n_el} must be divisible by tile_height={cfg.tile_height}"
    )
    W = int(math.ceil((cfg.max_azimuth - cfg.min_azimuth) / cfg.azimuth_resolution))

    az = torch.linspace(
        cfg.min_azimuth + cfg.azimuth_resolution / 2,
        cfg.max_azimuth - cfg.azimuth_resolution / 2,
        W, device=device,
    )
    el_step = (cfg.max_elevation - cfg.min_elevation) / n_el
    el = torch.linspace(
        cfg.min_elevation + el_step / 2,
        cfg.max_elevation - el_step / 2,
        n_el, device=device,
    )
    grid = torch.stack(torch.meshgrid(el, az, indexing="ij"), dim=-1)  # (n_el,W,2): (el,az)
    grid = grid[..., [1, 0]]                                           # -> (az,el)
    r0 = torch.full_like(grid[..., :1], 10.0)                          # init range
    t = torch.zeros_like(grid[..., :1])                                # time
    raster_pts = torch.cat([grid, r0, t], dim=-1).unsqueeze(0)         # (1,n_el,W,4)

    teb = torch.linspace(cfg.min_elevation, cfg.max_elevation,
                         n_el // cfg.tile_height + 1, device=device)
    return raster_pts, teb


# ----------------------------- main head ------------------------------------


class LidarHead(nn.Module):
    """Differentiable LiDAR renderer on top of an existing gaussian field."""

    def __init__(self, cfg: Optional[LidarHeadConfig] = None):
        super().__init__()
        self.cfg = cfg or LidarHeadConfig()
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def render(
        self,
        means_w: torch.Tensor,                    # [N,3] world (or ego if matched)
        quats: torch.Tensor,                      # [N,4] (w,x,y,z), normalised
        scales: torch.Tensor,                     # [N,3] activated (positive)
        opacities: torch.Tensor,                  # [N] in (0,1)
        intensity: torch.Tensor,                  # [N,1] in [0,1]
        raydrop_logit: torch.Tensor,              # [N,1] raw logits
        lidar_pose_w: torch.Tensor,               # [4,4] lidar-to-world
        velocities: Optional[torch.Tensor] = None,        # [N,3] world frame
        linear_velocity: Optional[torch.Tensor] = None,   # [3]
        angular_velocity: Optional[torch.Tensor] = None,  # [3]
        rolling_shutter_time: Optional[torch.Tensor] = None,
        raster_pts: Optional[torch.Tensor] = None,
        tile_elevation_boundaries: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        device = means_w.device

        if raster_pts is None or tile_elevation_boundaries is None:
            raster_pts, tile_elevation_boundaries = build_raster_pts(cfg, device)

        # features: [intensity, raydrop_logit] per gaussian (D=2)
        feats = torch.cat([intensity, raydrop_logit], dim=-1).unsqueeze(0).contiguous()  # [1,N,2]
        viewmat = torch.linalg.inv(lidar_pose_w).unsqueeze(0)  # [1,4,4]

        out_feat, alphas, alpha_until_pts, meta = lidar_rasterization(
            means=means_w, quats=quats, scales=scales, opacities=opacities,
            lidar_features=feats, velocities=velocities,
            linear_velocity=linear_velocity, angular_velocity=angular_velocity,
            rolling_shutter_time=rolling_shutter_time,
            viewmats=viewmat, raster_pts=raster_pts,
            tile_elevation_boundaries=tile_elevation_boundaries,
            min_azimuth=cfg.min_azimuth, max_azimuth=cfg.max_azimuth,
            min_elevation=cfg.min_elevation, max_elevation=cfg.max_elevation,
            n_elevation_channels=cfg.n_elevation_channels,
            azimuth_resolution=cfg.azimuth_resolution,
            tile_width=cfg.tile_width, tile_height=cfg.tile_height,
            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
        )
        # out_feat: [1, n_el, W, D+1]  last channel = expected range
        out_feat = out_feat[0]
        alphas = alphas[0, ..., 0]
        intensity_img = torch.sigmoid(out_feat[..., 0])
        raydrop_logit_img = out_feat[..., 1]            # raw logits
        range_img = out_feat[..., -1]
        return {
            "range": range_img,                          # [n_el, W]
            "intensity": intensity_img,                  # [n_el, W]
            "raydrop_logit": raydrop_logit_img,          # [n_el, W]
            "alpha": alphas,                             # [n_el, W]
            "alpha_until_pts": alpha_until_pts[0, ..., 0] if alpha_until_pts is not None else None,
            "raster_pts": raster_pts[0],                 # [n_el, W, 4]
            "meta": meta,
        }

    # -------- loss helpers --------

    def losses(
        self,
        pred: Dict[str, torch.Tensor],
        gt_range: torch.Tensor,             # [n_el, W]
        gt_intensity: torch.Tensor,         # [n_el, W] in [0,1]
        hit_mask: torch.Tensor,             # [n_el, W] {0,1}
        weights: Dict[str, float] | None = None,
        depth_quantile: float = 0.95,
    ) -> Dict[str, torch.Tensor]:
        """Compute the four SplatAD LiDAR losses; returns a dict of weighted
        scalar losses. `weights` controls each term (depth/intensity/raydrop/los).
        """
        w = {"depth": 1.0, "intensity": 1.0, "raydrop": 1.0, "los": 1.0}
        if weights:
            w.update(weights)

        out: Dict[str, torch.Tensor] = {}
        hit = hit_mask > 0.5

        # depth L1 with top-quantile clipping (SplatAD style)
        if hit.any():
            d_err = (pred["range"] - gt_range).abs()[hit]
            if 0 < depth_quantile < 1 and d_err.numel() > 0:
                q = torch.quantile(d_err.detach(), depth_quantile)
                d_err = d_err[d_err <= q]
            out["lidar_depth"] = w["depth"] * d_err.mean()

            # intensity MSE on hits only
            out["lidar_intensity"] = w["intensity"] * F.mse_loss(
                pred["intensity"][hit], gt_intensity[hit]
            )

        # ray-drop BCE on the entire image (target 0 if hit, 1 if no return)
        rd_target = (~hit).float()
        out["lidar_raydrop"] = w["raydrop"] * self.bce(
            pred["raydrop_logit"], rd_target
        ).mean()

        # line-of-sight: penalise accumulated alpha *before* the GT hit (i.e.
        # the gaussians should be transparent up to the true surface).
        if pred.get("alpha_until_pts") is not None and hit.any():
            out["lidar_los"] = w["los"] * pred["alpha_until_pts"][hit].mean()

        return out
