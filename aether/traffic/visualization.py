"""Deterministic top-down rendering of the SUMO network used by LimSim."""
from __future__ import annotations

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .protocol import EgoState, TrafficFrame


def _shape(value: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in value.split():
        x, y = item.split(",")[:2]
        points.append((float(x), float(y)))
    return points


class LimSimTopDownRenderer:
    """Render the complete imported nuScenes lane graph and live actors."""

    def __init__(self, net_file: str | Path, width: int = 900,
                 height: int = 900, padding: int = 42) -> None:
        self.net_file = Path(net_file).resolve()
        if not self.net_file.is_file():
            raise FileNotFoundError(f"LimSim SUMO network not found: {self.net_file}")
        self.width, self.height, self.padding = int(width), int(height), int(padding)
        directory_label = self.net_file.parent.name.removeprefix("nuscenes_scene")
        self.scene_label = directory_label or self.net_file.stem
        root = ET.parse(self.net_file).getroot()
        self.lanes: list[tuple[str, list[tuple[float, float]], float]] = []
        for edge in root.findall("edge"):
            edge_id = edge.get("id", "")
            if edge_id.startswith(":") or edge.get("function") == "internal":
                continue
            for lane in edge.findall("lane"):
                points = _shape(lane.get("shape", ""))
                if len(points) >= 2:
                    self.lanes.append((edge_id, points, float(lane.get("width", "3.2"))))
        if not self.lanes:
            raise ValueError(f"SUMO network contains no non-internal lanes: {self.net_file}")
        all_points = np.asarray(
            [point for _, points, _ in self.lanes for point in points], dtype=np.float64
        )
        self.minimum = all_points.min(axis=0)
        self.maximum = all_points.max(axis=0)
        span = np.maximum(self.maximum - self.minimum, 1.0)
        self.scale = min(
            (self.width - 2 * self.padding) / span[0],
            (self.height - 2 * self.padding) / span[1],
        )
        content = span * self.scale
        self.origin = np.asarray(
            [(self.width - content[0]) * 0.5, (self.height + content[1]) * 0.5],
            dtype=np.float64,
        )
        self._font = ImageFont.load_default()

    def _screen(self, x: float, y: float) -> tuple[float, float]:
        # SUMO/local-world x is right on screen and y is up.
        return (
            float(self.origin[0] + (x - self.minimum[0]) * self.scale),
            float(self.origin[1] - (y - self.minimum[1]) * self.scale),
        )

    def _vehicle_polygon(self, x: float, y: float, yaw: float,
                         length: float, width: float) -> list[tuple[float, float]]:
        c, s = math.cos(yaw), math.sin(yaw)
        forward = np.asarray([c, s]) * (length * 0.5)
        left = np.asarray([-s, c]) * (width * 0.5)
        center = np.asarray([x, y])
        return [self._screen(*point) for point in (
            center + forward + left,
            center + forward - left,
            center - forward - left,
            center - forward + left,
        )]

    def render(self, frame: TrafficFrame | None, ego: EgoState) -> np.ndarray:
        image = Image.new("RGB", (self.width, self.height), (18, 22, 28))
        draw = ImageDraw.Draw(image)
        # First draw carriageway widths, then all HD-map lane centerlines.
        for _, points, lane_width in self.lanes:
            pixels = [self._screen(x, y) for x, y in points]
            draw.line(
                pixels, fill=(58, 64, 72),
                width=max(2, int(round(lane_width * self.scale))), joint="curve",
            )
        for _, points, _ in self.lanes:
            draw.line(
                [self._screen(x, y) for x, y in points],
                fill=(153, 163, 174), width=1, joint="curve",
            )

        vehicles = () if frame is None else frame.vehicles
        for vehicle in vehicles:
            polygon = self._vehicle_polygon(
                vehicle.x, vehicle.y, vehicle.yaw, vehicle.length, vehicle.width
            )
            draw.polygon(polygon, fill=(242, 145, 45), outline=(255, 216, 133))
        ego_polygon = self._vehicle_polygon(
            ego.x, ego.y, ego.yaw, ego.length, ego.width
        )
        draw.polygon(ego_polygon, fill=(40, 150, 255), outline=(183, 224, 255))

        time_s = 0.0 if frame is None else frame.time_s
        title = (
            f"LimSim 2D | nuScenes scene {self.scene_label} HD map | "
            f"lanes={len(self.lanes)} traffic={len(vehicles)} t={time_s:.2f}s"
        )
        draw.rounded_rectangle((14, 48, min(self.width - 14, 645), 75), radius=7,
                               fill=(0, 0, 0), outline=(80, 91, 104))
        draw.text((24, 56), title, font=self._font, fill=(244, 247, 250))
        draw.text((24, self.height - 28), "blue: ego   orange: external 3DRealCar traffic",
                  font=self._font, fill=(200, 207, 214))
        return np.asarray(image, dtype=np.uint8)
