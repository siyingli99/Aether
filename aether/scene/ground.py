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
from the corridor level by a slope-limited margin.  The result is a 0.5 m
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
MAX_SLOPE = 0.08     # tolerated deviation slope off the corridor (8 %)


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


@dataclass
class GroundModel:
    origin: np.ndarray        # (2,) road-frame xy of grid cell (0, 0)
    resolution: float
    heights: np.ndarray       # (ny, nx) ground z in the road frame

    # ------------------------------------------------------------- build
    @classmethod
    def build(cls, splats: np.ndarray, route: np.ndarray,
              margin: float = 20.0) -> "GroundModel":
        """``splats``: (N, 3) road-frame points; ``route``: (T, 3) logged
        rear-axle ground points (x, y, z) in the road frame."""
        from scipy.ndimage import uniform_filter
        from scipy.spatial import cKDTree

        centers, cell_z = _cell_ground(np.asarray(splats, dtype=np.float64))
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
            allowed = 0.3 + MAX_SLOPE * np.maximum(d_route - CORRIDOR, 0.0)
            splat_z = np.clip(splat_z, corridor_z - allowed, corridor_z + allowed)
            weight = np.clip((d_route - CORRIDOR) / BLEND, 0.0, 1.0)
            weight[d_cell > 2.0 * CELL] = 0.0      # no splat support nearby
            fused = (1.0 - weight) * corridor_z + weight * splat_z
        heights = uniform_filter(fused.reshape(ny, nx), size=3, mode="nearest")
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
        pitch = -math.atan2(front - rear, wheelbase)
        roll = math.atan2(left - right, track)
        return float(rear), pitch, roll
