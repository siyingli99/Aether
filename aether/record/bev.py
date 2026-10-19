"""Bird's-eye-view rendering of a closed-loop run (LimSim-style).

The static map (drivable area, lane borders, centrelines, logged route) is
drawn once into a large raster; each video frame crops a window that
follows the ego and draws the vehicles on top, supersampled 2× for smooth
edges.  The view is north-up in the road frame: +x (initial ego heading)
points up, +y (left) points left, so a traffic engineer reads it like a
plan drawing of the road.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw

from .video import font

PALETTE = {
    "background": (238, 240, 243),
    "drivable": (206, 211, 219),
    "lane_edge": (250, 251, 252),
    "centerline": (170, 178, 190),
    "route": (128, 160, 196),
    "ego": (228, 87, 46),
    "ego_trail": (228, 87, 46),
    "plan": (40, 167, 69),
    "vehicle": (46, 110, 190),
    "recorded": (132, 82, 170),
    "pedestrian": (214, 140, 30),
    "outline": (30, 34, 40),
    "text": (28, 32, 38),
    "alert": (220, 40, 50),
}


def _rect_corners(x: float, y: float, yaw: float, length: float, width: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    dx = np.array([1, 1, -1, -1]) * 0.5 * length
    dy = np.array([1, -1, -1, 1]) * 0.5 * width
    return np.column_stack([x + dx * c - dy * s, y + dx * s + dy * c])


class BEVRenderer:
    def __init__(self, hdmap=None, route_xy: np.ndarray | None = None,
                 size: tuple[int, int] = (900, 900), view_m: float = 70.0,
                 supersample: int = 2, margin_m: float = 60.0) -> None:
        self.size = (int(size[0]), int(size[1]))
        self.ss = int(supersample)
        self.ppm = self.size[0] / float(view_m) * self.ss          # pixels / metre (supersampled)
        self.hdmap, self.route = hdmap, (None if route_xy is None else np.asarray(route_xy))
        self._font = font(13 * self.ss)
        self._small = font(10 * self.ss)
        # Static raster extent: route bbox (or map bounds) + margin.
        if self.route is not None and len(self.route):
            lo, hi = self.route.min(0) - margin_m, self.route.max(0) + margin_m
        elif hdmap is not None and hdmap.bounds() is not None:
            b = hdmap.bounds()
            lo, hi = np.array(b[:2]) - 10, np.array(b[2:]) + 10
        else:
            lo, hi = np.array([-100.0, -100.0]), np.array([100.0, 100.0])
        self.x_max, self.y_max = float(hi[0]), float(hi[1])
        w = int(math.ceil((hi[1] - lo[1]) * self.ppm))
        h = int(math.ceil((hi[0] - lo[0]) * self.ppm))
        self.static = self._draw_static(w, h)

    # ----------------------------------------------------------- mapping
    def _px(self, xy: np.ndarray, origin=(0.0, 0.0)) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        col = (self.y_max - xy[:, 1]) * self.ppm - origin[0]
        row = (self.x_max - xy[:, 0]) * self.ppm - origin[1]
        return np.column_stack([col, row])

    def _poly(self, draw, xy, origin, **kw) -> None:
        pts = [tuple(p) for p in self._px(xy, origin)]
        if len(pts) >= 3:
            draw.polygon(pts, **kw)

    # ------------------------------------------------------------ static
    def _draw_static(self, w: int, h: int) -> Image.Image:
        image = Image.new("RGB", (max(w, 1), max(h, 1)), PALETTE["background"])
        draw = ImageDraw.Draw(image)
        m = self.hdmap
        if m is not None:
            geoms = []
            if m.drivable is not None:
                geoms = list(getattr(m.drivable, "geoms", [m.drivable]))
            for g in geoms:
                self._poly(draw, np.asarray(g.exterior.coords), (0, 0), fill=PALETTE["drivable"])
                for hole in g.interiors:
                    self._poly(draw, np.asarray(hole.coords), (0, 0), fill=PALETTE["background"])
            for lane in m.lanes:
                ring = np.asarray(lane.polygon.exterior.coords)
                draw.line([tuple(p) for p in self._px(ring)], fill=PALETTE["lane_edge"],
                          width=max(1, self.ss))
            for lane in m.lanes:
                pts = self._px(lane.centerline)
                for a, b in zip(pts[:-1:2], pts[1::2]):        # dashed
                    draw.line([tuple(a), tuple(b)], fill=PALETTE["centerline"], width=max(1, self.ss))
        if self.route is not None and len(self.route) > 1:
            draw.line([tuple(p) for p in self._px(self.route)], fill=PALETTE["route"],
                      width=2 * self.ss)
        return image

    # ------------------------------------------------------------- frame
    def frame(self, center: tuple[float, float], vehicles: dict[str, dict],
              trail: Iterable[tuple[float, float]] = (), plan: np.ndarray | None = None,
              hud: list[str] | None = None, alerts: list[str] | None = None,
              collided: set[str] | None = None) -> np.ndarray:
        W, H = self.size[0] * self.ss, self.size[1] * self.ss
        c = self._px(np.array(center))[0]
        # Ego slightly below the centre: more road ahead is visible.
        origin = (c[0] - W / 2, c[1] - H * 0.62)
        image = Image.new("RGB", (W, H), PALETTE["background"])
        image.paste(self.static, (int(round(-origin[0])), int(round(-origin[1]))))
        draw = ImageDraw.Draw(image)
        collided = collided or set()

        trail = np.asarray(list(trail), dtype=float)
        if len(trail) > 1:
            draw.line([tuple(p) for p in self._px(trail, origin)], fill=PALETTE["ego_trail"],
                      width=2 * self.ss)
        if plan is not None and len(plan):
            for p in self._px(plan, origin):
                r = 3 * self.ss
                draw.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=PALETTE["plan"])

        for vid, st in sorted(vehicles.items(), key=lambda kv: kv[0] == "ego"):
            kind = "ego" if vid == "ego" else ("pedestrian" if st.get("type") == "pedestrian"
                                                else "recorded" if st.get("source") == "recorded"
                                                else "vehicle")
            corners = _rect_corners(st["x"], st["y"], st["yaw"], st["length"], st["width"])
            outline = PALETTE["alert"] if vid in collided else PALETTE["outline"]
            self._poly(draw, corners, origin, fill=PALETTE[kind], outline=outline,
                       width=(3 if vid in collided else 1) * self.ss)
            # windscreen mark: heading at a glance
            nose = _rect_corners(st["x"] + 0.3 * st["length"] * math.cos(st["yaw"]),
                                 st["y"] + 0.3 * st["length"] * math.sin(st["yaw"]),
                                 st["yaw"], 0.18 * st["length"], 0.8 * st["width"])
            self._poly(draw, nose, origin, fill=(255, 255, 255))
            if vid != "ego":
                p = self._px(np.array([st["x"], st["y"]]), origin)[0]
                text = f'{vid.split(":")[-1]} {st["speed"]:.1f}'
                draw.text((p[0] + 0.6 * st["width"] * self.ppm, p[1] - 6 * self.ss), text,
                          fill=PALETTE["text"], font=self._small)

        self._scale_bar(draw, W, H)
        if hud:
            self._panel(draw, hud, (12 * self.ss, 12 * self.ss), PALETTE["text"])
        if alerts:
            self._panel(draw, alerts, (12 * self.ss, H - (24 + 20 * len(alerts)) * self.ss),
                        PALETTE["alert"])
            draw.rectangle((0, 0, W - 1, H - 1), outline=PALETTE["alert"], width=6 * self.ss)
        out = image.resize(self.size, Image.Resampling.LANCZOS) if self.ss > 1 else image
        return np.asarray(out)

    def _panel(self, draw, lines: list[str], xy, color) -> None:
        x, y = xy
        widths = [draw.textbbox((0, 0), s, font=self._font)[2] for s in lines]
        line_h = 20 * self.ss
        draw.rounded_rectangle((x, y, x + max(widths) + 20 * self.ss, y + line_h * len(lines) + 10 * self.ss),
                               radius=8 * self.ss, fill=(255, 255, 255), outline=(200, 205, 212))
        for i, s in enumerate(lines):
            draw.text((x + 10 * self.ss, y + 6 * self.ss + i * line_h), s, fill=color, font=self._font)

    def _scale_bar(self, draw, W: int, H: int) -> None:
        length = 10.0 * self.ppm
        x1, y = W - 20 * self.ss, H - 24 * self.ss
        x0 = x1 - length
        draw.line([(x0, y), (x1, y)], fill=PALETTE["text"], width=2 * self.ss)
        for x in (x0, x1):
            draw.line([(x, y - 5 * self.ss), (x, y + 5 * self.ss)], fill=PALETTE["text"], width=2 * self.ss)
        draw.text((x0, y - 22 * self.ss), "10 m", fill=PALETTE["text"], font=self._small)
