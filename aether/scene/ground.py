"""Ground height model of a reconstructed scene.

AETHER does not need a separately trained ground Gaussian layer.  The road
surface is estimated once per scene from two complementary sources:

1. **Logged trajectory** — the recording vehicle stood on the road at every
   logged frame, so its rear-axle poses give the *shape* of the surface along
   the driven corridor (the elevation profile; flat for nuScenes, whose ego
   poses are planar).  They are immune to reconstruction floaters.
2. **Gaussian splats** — per 2 m cell, the lowest *dense* layer of
   high-opacity splats (:func:`ground_layer`) gives the surface away from the
   corridor and the absolute level of the reconstructed road.

The corridor profile is shifted by the median splat-minus-profile offset,
cells within ``corridor`` metres of the route use the profile, cells further
away blend towards their splat estimate, which is only allowed to deviate
from the corridor level by a slope-limited margin (3 %, at most 0.6 m): far
from the driven corridor the reconstruction often has no road splats at all
(only foliage or building bases), and a wrong surface there makes traffic
vehicles pitch wildly.

Road surfaces are smooth, but single cells are not: kerbs, planting strips,
low vegetation and parked-car undersides lift the dense layer locally.  Cell
estimates are therefore (a) restricted to the HD-map drivable area when the
scene has one and (b) replaced by a robust local plane fit over a
``SMOOTH_RADIUS`` neighbourhood, dropping cells that stick out by more than
``OUTLIER`` metres.  Vehicles placed off the logged corridor (traffic, an
ego that leaves the recorded path) then ride on a surface without bumps.  The result is a 0.5 m
height field queried with bilinear interpolation, cached in
``<scene>/cache/ground.npz``.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

CELL = 2.0           # splat statistics cell (m)
RESOLUTION = 0.5     # output height-field resolution (m)
CORRIDOR = 4.0       # distance (m) within which the logged profile is used
BLEND = 8.0          # distance (m) over which splat estimates take over
MAX_SLOPE = 0.03     # tolerated deviation slope off the corridor (3 %)
MAX_DEVIATION = 0.6  # m, cap on the deviation from the corridor level
SMOOTH_RADIUS = 12.0 # m, neighbourhood of the robust local plane fit
OUTLIER = 0.15       # m, cells further from the local plane are discarded
MAX_GRADE = 0.10     # steepest grade / cross-fall a vehicle is placed on
VERSION = 6          # bump to invalidate cached ground models


def ground_layer(heights: np.ndarray, window: float = 0.25, min_share: float = 0.5) -> float:
    """Surface height of one cell from its *sorted* point heights.

    Splat reconstructions leave sparse floaters under the road, so a low
    quantile sits below the true surface.  Take the lowest ``window``-thick
    layer holding at least ``min_share`` of the densest layer's points and
    return its median.
    """
    ends = np.searchsorted(heights, heights + window, side="right")
    counts = ends - np.arange(len(heights))
    start = int(np.flatnonzero(counts >= min_share * counts.max())[0])
    return float(np.median(heights[start:ends[start]]))


def _cell_ground(points: np.ndarray, min_points: int = 8):
    cells = np.floor(points[:, :2] / CELL).astype(np.int64)
    keys, inverse = np.unique(cells, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    order = np.lexsort((points[:, 2], inverse))
    counts = np.bincount(inverse, minlength=len(keys))
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    z = points[order, 2]
    ground = np.full(len(keys), np.nan)
    for index in np.flatnonzero(counts >= min_points):
        ground[index] = ground_layer(z[starts[index]:starts[index] + counts[index]])
    dense = np.isfinite(ground)
    return (keys[dense] + 0.5) * CELL, ground[dense]


def _smooth_cells(centers: np.ndarray, z: np.ndarray, radius: float = SMOOTH_RADIUS,
                  outlier: float = OUTLIER, iterations: int = 4) -> np.ndarray:
    """Robust local plane fit through the *lower envelope* of cell heights.

    Off the driven corridor many cells hold no road splats at all, only tree
    canopies or building bases several metres up.  The fit starts from a low
    neighbourhood quantile and keeps only cells within ``outlier`` above (and
    0.5 m below) the current surface, so it settles on the road layer and
    ignores kerbs, vegetation and canopies.
    """
    from scipy.spatial import cKDTree
    if len(centers) < 4:
        return z.copy()
    tree = cKDTree(centers)
    neighbours = [np.asarray(n) for n in tree.query_ball_point(centers, radius)]
    fitted = np.array([np.quantile(z[n], 0.2) for n in neighbours])
    for _ in range(iterations):
        keep = (z - fitted <= outlier) & (z - fitted >= -0.5)
        new = fitted.copy()
        for i, idx in enumerate(neighbours):
            sel = idx[keep[idx]]
            if len(sel) < 4:
                continue
            d = centers[sel] - centers[i]
            w = np.exp(-0.5 * (np.hypot(d[:, 0], d[:, 1]) / (0.5 * radius)) ** 2)
            A = np.column_stack([np.ones(len(sel)), d]) * w[:, None]
            coef = np.linalg.lstsq(A, z[sel] * w, rcond=None)[0]
            new[i] = coef[0]
        fitted = new
    return fitted


@dataclass
class GroundModel:
    origin: np.ndarray        # (2,) road-frame xy of grid cell (0, 0)
    resolution: float
    heights: np.ndarray       # (ny, nx) ground z in the road frame

    # ------------------------------------------------------------- build
    @classmethod
    def build(cls, splats: np.ndarray, route: np.ndarray, margin: float = 20.0,
              drivable=None) -> "GroundModel":
        """``splats``: (N, 3) road-frame points; ``route``: (T, 3) logged
        rear-axle ground points (x, y, z) in the road frame; ``drivable``:
        optional ``fn((M, 2) xy) -> bool mask`` of the HD-map drivable area."""
        from scipy.ndimage import uniform_filter
        from scipy.spatial import cKDTree

        centers, cell_z = _cell_ground(np.asarray(splats, dtype=np.float64))
        if drivable is not None and len(centers):
            road = np.asarray(drivable(centers), dtype=bool)
            if road.sum() >= 8:
                centers, cell_z = centers[road], cell_z[road]
        if len(centers):
            cell_z = _smooth_cells(centers, cell_z)
        route = np.asarray(route, dtype=np.float64)
        route_tree = cKDTree(route[:, :2])
        offset = 0.0
        if len(centers):
            d, idx = route_tree.query(centers)
            near = d < CORRIDOR
            if near.any():
                offset = float(np.median(cell_z[near] - route[idx[near], 2]))
        lo = np.minimum(route[:, :2].min(0), centers.min(0) if len(centers) else route[:, :2].min(0)) - margin
        hi = np.maximum(route[:, :2].max(0), centers.max(0) if len(centers) else route[:, :2].max(0)) + margin
        nx, ny = (np.ceil((hi - lo) / RESOLUTION).astype(int) + 1)
        gx = lo[0] + RESOLUTION * np.arange(nx)
        gy = lo[1] + RESOLUTION * np.arange(ny)
        grid = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)

        d_route, i_route = route_tree.query(grid)
        corridor_z = route[i_route, 2] + offset
        fused = corridor_z.copy()
        if len(centers):
            cell_tree = cKDTree(centers)
            d_cell, i_cell = cell_tree.query(grid)
            splat_z = cell_z[i_cell]
            allowed = np.minimum(0.3 + MAX_SLOPE * np.maximum(d_route - CORRIDOR, 0.0), MAX_DEVIATION)
            weight = np.clip((d_route - CORRIDOR) / BLEND, 0.0, 1.0)
            weight[d_cell > 2.0 * CELL] = 0.0      # no splat support nearby
            # Estimates outside the plausible band are not road (canopies,
            # building bases, floaters): fall back to the corridor level
            # instead of clipping them to the band edge.
            weight[np.abs(splat_z - corridor_z) > allowed] = 0.0
            fused = (1.0 - weight) * corridor_z + weight * splat_z
        heights = uniform_filter(fused.reshape(ny, nx), size=7, mode="nearest")   # 3.5 m box
        return cls(lo.astype(np.float64), RESOLUTION, heights)

    # ------------------------------------------------------------ caching
    def save(self, path: Path) -> None:
        np.savez_compressed(path, origin=self.origin, resolution=self.resolution,
                            heights=self.heights.astype(np.float32))

    @classmethod
    def load(cls, path: Path) -> "GroundModel":
        data = np.load(path)
        return cls(data["origin"].astype(np.float64), float(data["resolution"]),
                   data["heights"].astype(np.float64))

    # ------------------------------------------------------------- query
    def height(self, x, y):
        """Ground z at road-frame (x, y); scalars or arrays, bilinear."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        ny, nx = self.heights.shape
        fx = np.clip((x - self.origin[0]) / self.resolution, 0, nx - 1.000001)
        fy = np.clip((y - self.origin[1]) / self.resolution, 0, ny - 1.000001)
        ix, iy = np.floor(fx).astype(int), np.floor(fy).astype(int)
        tx, ty = fx - ix, fy - iy
        h = self.heights
        z = ((1 - tx) * (1 - ty) * h[iy, ix] + tx * (1 - ty) * h[iy, ix + 1]
             + (1 - tx) * ty * h[iy + 1, ix] + tx * ty * h[iy + 1, ix + 1])
        return float(z) if z.ndim == 0 else z

    def attitude(self, x: float, y: float, yaw: float, wheelbase: float,
                 track: float) -> tuple[float, float, float]:
        """(z, pitch, roll) of a vehicle whose rear axle is at (x, y).

        Heights are sampled under the four wheels; pitch is positive nose
        down (rotation about +y), roll positive right side down.
        """
        c, s = math.cos(yaw), math.sin(yaw)
        half = 0.5 * track
        wheels = np.array([[0.0, half], [0.0, -half], [wheelbase, half], [wheelbase, -half]])
        wx = x + wheels[:, 0] * c - wheels[:, 1] * s
        wy = y + wheels[:, 0] * s + wheels[:, 1] * c
        z = self.height(wx, wy)
        rear, front = 0.5 * (z[0] + z[1]), 0.5 * (z[2] + z[3])
        left, right = 0.5 * (z[0] + z[2]), 0.5 * (z[1] + z[3])
        limit = math.atan(MAX_GRADE)
        pitch = min(max(-math.atan2(front - rear, wheelbase), -limit), limit)
        roll = min(max(math.atan2(left - right, track), -limit), limit)
        return float(rear), pitch, roll
