"""Oriented boxes in the road frame (vehicle footprints, collision volumes)."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Box2D:
    """Oriented rectangle: centre (x, y), heading ``yaw`` (CCW), length along
    the heading and width across it, in metres."""

    x: float
    y: float
    yaw: float
    length: float
    width: float

    def corners(self) -> np.ndarray:
        """Four corners (front-left, front-right, rear-right, rear-left)."""
        return footprint(self.x, self.y, self.yaw, self.length, self.width)

    def polygon(self):
        from shapely.geometry import Polygon
        return Polygon(self.corners())

    def lerp(self, other: "Box2D", t: float) -> "Box2D":
        d_yaw = (other.yaw - self.yaw + math.pi) % (2.0 * math.pi) - math.pi
        return Box2D(
            self.x + t * (other.x - self.x), self.y + t * (other.y - self.y),
            self.yaw + t * d_yaw,
            self.length + t * (other.length - self.length),
            self.width + t * (other.width - self.width),
        )


def footprint(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    half_l, half_w = 0.5 * length, 0.5 * width
    local = np.array([[half_l, half_w], [half_l, -half_w],
                      [-half_l, -half_w], [-half_l, half_w]])
    return local @ np.array([[c, s], [-s, c]]) + (x, y)


def overlap_ratio(a: Box2D, b: Box2D) -> float:
    """Intersection area over the smaller box area (0 when disjoint)."""
    # Cheap rejection before building polygons.
    reach = 0.5 * (math.hypot(a.length, a.width) + math.hypot(b.length, b.width))
    if math.hypot(a.x - b.x, a.y - b.y) > reach:
        return 0.0
    pa, pb = a.polygon(), b.polygon()
    if not pa.intersects(pb):
        return 0.0
    return float(pa.intersection(pb).area / max(min(pa.area, pb.area), 1e-9))


def points_in_box(points: np.ndarray, box: Box2D, z_min: np.ndarray | float,
                  height: float) -> np.ndarray:
    """Mask of road-frame points inside a box standing on ``z_min``."""
    points = np.asarray(points)
    c, s = math.cos(box.yaw), math.sin(box.yaw)
    dx, dy = points[:, 0] - box.x, points[:, 1] - box.y
    along = dx * c + dy * s
    across = -dx * s + dy * c
    dz = points[:, 2] - z_min
    return ((np.abs(along) < 0.5 * box.length) & (np.abs(across) < 0.5 * box.width)
            & (dz > 0.0) & (dz < height))
