"""HD map in the road frame: lanes, drivable area and lane matching.

``map/hdmap.json`` (schema ``aether.hdmap.v2``)::

    {
      "schema": "aether.hdmap.v2",
      "location": "singapore-onenorth",
      "local_to_global": 4x4,            # road frame -> dataset map frame
      "lanes": [{"id", "kind", "centerline": [[x, y], ...],
                 "polygon": [[x, y], ...], "width", "incoming", "outgoing"}],
      "drivable_area": [[[x, y], ...] | {"exterior": [...], "holes": [[...]]}, ...],
      "ego_route": [[x, y], ...]               # logged ego corridor
    }

Version-1 files (``aether.nuscenes_hd_map.v1``: centrelines with a nominal
width) are read too; their lane polygons and drivable area are then the
centrelines buffered by half the lane width.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path

import numpy as np


@dataclass
class Lane:
    id: str
    kind: str
    centerline: np.ndarray            # (N, 2)
    width: float
    polygon: object                   # shapely Polygon
    incoming: list[str] = field(default_factory=list)
    outgoing: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        segments = np.diff(self.centerline, axis=0)
        self._lengths = np.linalg.norm(segments, axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(self._lengths)])
        self._segments = segments

    @property
    def length(self) -> float:
        return float(self._cum[-1])

    def project(self, x: float, y: float) -> tuple[float, float, float]:
        """(s along centreline, signed lateral offset (left +), lane heading)."""
        starts = self.centerline[:-1]
        denom = np.maximum(self._lengths ** 2, 1e-12)
        t = np.clip(((x - starts[:, 0]) * self._segments[:, 0]
                     + (y - starts[:, 1]) * self._segments[:, 1]) / denom, 0.0, 1.0)
        px = starts[:, 0] + t * self._segments[:, 0]
        py = starts[:, 1] + t * self._segments[:, 1]
        dist = np.hypot(x - px, y - py)
        i = int(np.argmin(dist))
        heading = math.atan2(self._segments[i, 1], self._segments[i, 0])
        cross = (math.cos(heading) * (y - py[i]) - math.sin(heading) * (x - px[i]))
        return float(self._cum[i] + t[i] * self._lengths[i]), float(cross), heading


@dataclass
class LaneMatch:
    lane_id: str
    s: float
    d: float
    heading_error: float


class HDMap:
    def __init__(self, lanes: list[Lane], drivable, local_to_global: np.ndarray,
                 location: str = "", ego_route: np.ndarray | None = None) -> None:
        from shapely.strtree import STRtree
        self.lanes = lanes
        self.by_id = {lane.id: lane for lane in lanes}
        self.drivable = drivable
        self.local_to_global = np.asarray(local_to_global, dtype=np.float64)
        self.location = location
        self.ego_route = ego_route if ego_route is not None else np.zeros((0, 2))
        self._tree = STRtree([lane.polygon for lane in lanes]) if lanes else None
        self._boundary = drivable.boundary if drivable is not None else None

    # ---------------------------------------------------------------- load
    @classmethod
    def load(cls, path: str | Path) -> "HDMap":
        from shapely.geometry import LineString, Polygon
        from shapely.ops import unary_union
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        raw_lanes = data.get("lanes", data.get("map_lanes", []))
        lanes = []
        for index, item in enumerate(raw_lanes):
            center = np.asarray(item["centerline"], dtype=np.float64)[:, :2]
            if len(center) < 2:
                continue
            width = float(item.get("width", item.get("lane_width_m", 3.5)))
            if item.get("polygon"):
                polygon = Polygon(np.asarray(item["polygon"], dtype=np.float64)[:, :2]).buffer(0)
            else:
                polygon = LineString(center).buffer(0.5 * width, cap_style=2)
            lanes.append(Lane(
                id=str(item.get("id", item.get("token", f"lane-{index}"))),
                kind=str(item.get("kind", "lane")), centerline=center, width=width,
                polygon=polygon, incoming=list(item.get("incoming", [])),
                outgoing=list(item.get("outgoing", [])),
            ))
        if data.get("drivable_area"):
            parts = []
            for item in data["drivable_area"]:
                if isinstance(item, dict):                     # {exterior, holes}
                    parts.append(Polygon(np.asarray(item["exterior"])[:, :2],
                                         [np.asarray(h)[:, :2] for h in item.get("holes", []) if len(h) >= 3]))
                elif len(item) >= 3:
                    parts.append(Polygon(np.asarray(item)[:, :2]))
            drivable = unary_union([g.buffer(0) for g in parts])
        else:
            drivable = unary_union([lane.polygon for lane in lanes]) if lanes else None
        route = data.get("ego_route") or data.get("traffic_link", {}).get("centerline")
        return cls(lanes, drivable, data.get("local_to_global", np.eye(4)),
                   str(data.get("location", "")),
                   np.asarray(route, dtype=np.float64) if route else None)

    # ------------------------------------------------------------- queries
    def to_global(self, xy: np.ndarray) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, dtype=np.float64))
        homogeneous = np.column_stack([xy[:, :2], np.zeros(len(xy)), np.ones(len(xy))])
        return (homogeneous @ self.local_to_global.T)[:, :2]

    def global_yaw(self, yaw: np.ndarray | float) -> np.ndarray:
        offset = math.atan2(self.local_to_global[1, 0], self.local_to_global[0, 0])
        return (np.asarray(yaw) + offset + math.pi) % (2 * math.pi) - math.pi

    def match(self, x: float, y: float, yaw: float | None = None,
              max_distance: float = 5.0) -> LaneMatch | None:
        """Nearest lane, preferring lanes whose heading agrees with ``yaw``."""
        if self._tree is None:
            return None
        from shapely.geometry import Point
        point = Point(float(x), float(y))
        candidates = self._tree.query(point.buffer(max_distance))
        best, best_cost = None, float("inf")
        for index in np.atleast_1d(candidates):
            lane = self.lanes[int(index)]
            s, d, heading = lane.project(float(x), float(y))
            error = 0.0 if yaw is None else abs((yaw - heading + math.pi) % (2 * math.pi) - math.pi)
            if abs(d) > max_distance:
                continue
            cost = abs(d) / max(0.5 * lane.width, 0.5) + (2.0 * error if yaw is not None else 0.0)
            if cost < best_cost:
                best, best_cost = LaneMatch(lane.id, s, d, error), cost
        return best

    def outside_distance(self, points: np.ndarray) -> np.ndarray:
        """Distance (m) of each point outside the drivable area (0 inside)."""
        if self.drivable is None:
            return np.zeros(len(points))
        from shapely import contains_xy, distance
        points = np.asarray(points, dtype=np.float64)
        inside = contains_xy(self.drivable, points[:, 0], points[:, 1])
        out = np.zeros(len(points))
        if not inside.all():
            from shapely import points as make_points
            out[~inside] = distance(self.drivable, make_points(points[~inside]))
        return out

    def bounds(self) -> tuple[float, float, float, float] | None:
        geometry = self.drivable
        return tuple(geometry.bounds) if geometry is not None and not geometry.is_empty else None
