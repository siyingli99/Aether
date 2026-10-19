"""Collision and rule-violation checks for the ego vehicle.

Events (``Event.type``):

``vehicle_collision``
    Ego footprint overlaps another agent's footprint (swept between steps so
    fast motion cannot tunnel).  ``at_fault`` is ``False`` when the ego was
    hit from behind or was standing still — the convention of nuPlan/NAVSIM.
``static_collision``
    More than ``static_threshold`` background Gaussians inside the ego box.
    Obstacles are splats 0.3–2.5 m above the ground model that are dense
    (``min_neighbors`` within ``neighbor_radius``) and outside the boxes the
    recording vehicle occupied (it drove there, so they are floaters).
``off_road``
    An ego corner is more than ``off_road_tolerance`` metres outside the HD
    map drivable area.
``off_scene``
    The ego centre has been more than ``off_scene_radius`` from every
    reconstructed splat for ``off_scene_steps`` consecutive steps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

import numpy as np

from ..geometry.boxes import Box2D, overlap_ratio


@dataclass
class Event:
    type: str
    time: float
    step: int
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "time": self.time, "step": self.step, **self.detail}


@dataclass
class CollisionConfig:
    min_overlap: float = 0.0          # vehicle overlap ratio counted as collision
    static_threshold: int = 50
    obstacle_min_height: float = 0.3
    obstacle_max_height: float = 2.5
    min_neighbors: int = 8
    neighbor_radius: float = 0.3
    carve_logged_route: bool = True
    off_road_tolerance: float = 0.3
    off_scene_radius: float = 5.0
    off_scene_steps: int = 4
    opacity_min: float = 0.3


def _filter_obstacles(points: np.ndarray, above: np.ndarray, cfg: CollisionConfig,
                      logged_boxes: list[Box2D], height: float) -> np.ndarray:
    from scipy.spatial import cKDTree
    mask = (above > cfg.obstacle_min_height) & (above < cfg.obstacle_max_height)
    index = np.flatnonzero(mask)
    if cfg.min_neighbors > 0 and len(index):
        candidates = points[index]
        neighbours = cKDTree(candidates).query_ball_point(
            candidates, r=cfg.neighbor_radius, return_length=True) - 1
        mask[index[neighbours < cfg.min_neighbors]] = False
    if cfg.carve_logged_route and logged_boxes:
        index = np.flatnonzero(mask)
        tree = cKDTree(points[index, :2])
        carve = np.zeros(len(index), dtype=bool)
        for box in logged_boxes:
            near = tree.query_ball_point((box.x, box.y), r=0.5 * math.hypot(box.length, box.width))
            if not near:
                continue
            near = np.asarray(near)
            rel = points[index[near], :2] - (box.x, box.y)
            c, s = math.cos(box.yaw), math.sin(box.yaw)
            along, across = rel @ (c, s), rel @ (-s, c)
            inside = ((np.abs(along) < 0.5 * box.length) & (np.abs(across) < 0.5 * box.width)
                      & (above[index[near]] < height))
            carve[near[inside]] = True
        mask[index[carve]] = False
    return mask


class CollisionChecker:
    def __init__(self, splats_road: np.ndarray, ground, profile, hdmap,
                 logged_boxes: list[Box2D], cfg: CollisionConfig | None = None) -> None:
        from scipy.spatial import cKDTree
        self.cfg = cfg or CollisionConfig()
        self.profile = profile
        self.hdmap = hdmap
        above = splats_road[:, 2] - ground.height(splats_road[:, 0], splats_road[:, 1])
        mask = _filter_obstacles(splats_road, above, self.cfg, logged_boxes, profile.height)
        self.obstacles = splats_road[mask, :2]
        self.obstacle_above = above[mask]
        self._obstacle_tree = cKDTree(self.obstacles) if len(self.obstacles) else None
        self._scene_tree = cKDTree(splats_road[:, :2]) if len(splats_road) else None
        self.stats = {"splats": int(len(splats_road)),
                      "obstacles": int(mask.sum())}
        self._off_scene = 0

    def reset(self) -> None:
        self._off_scene = 0

    # ---------------------------------------------------------------- checks
    def static_count(self, box: Box2D) -> int:
        if self._obstacle_tree is None:
            return 0
        near = self._obstacle_tree.query_ball_point((box.x, box.y), r=0.5 * math.hypot(box.length, box.width))
        if not near:
            return 0
        near = np.asarray(near)
        rel = self.obstacles[near] - (box.x, box.y)
        c, s = math.cos(box.yaw), math.sin(box.yaw)
        along, across = rel @ (c, s), rel @ (-s, c)
        inside = ((np.abs(along) < 0.5 * box.length) & (np.abs(across) < 0.5 * box.width)
                  & (self.obstacle_above[near] < self.profile.height))
        return int(inside.sum())

    def off_road_distance(self, box: Box2D) -> float:
        if self.hdmap is None:
            return 0.0
        return float(self.hdmap.outside_distance(box.corners()).max())

    @staticmethod
    def _classify(ego: Box2D, other: Box2D, ego_speed: float, other_speed: float) -> tuple[str, bool]:
        c, s = math.cos(ego.yaw), math.sin(ego.yaw)
        dx, dy = other.x - ego.x, other.y - ego.y
        along = dx * c + dy * s
        across = -dx * s + dy * c
        if along > 0.5 * ego.length * 0.6:
            side = "front"
        elif along < -0.5 * ego.length * 0.6:
            side = "rear"
        else:
            side = "left" if across > 0 else "right"
        at_fault = not (ego_speed < 0.1 or (side == "rear" and other_speed >= ego_speed))
        return side, at_fault

    def check(self, t: float, step: int, ego_prev: Box2D | None, ego_now: Box2D,
              ego_speed: float, agents_prev: dict[str, tuple[Box2D, float]],
              agents_now: dict[str, tuple[Box2D, float]]) -> list[Event]:
        events: list[Event] = []
        # Vehicle collisions, swept in sub-steps of at most 0.5 m.
        for agent_id, (box, speed) in agents_now.items():
            prev = agents_prev.get(agent_id, (box, speed))[0]
            ego0 = ego_prev or ego_now
            travel = max(math.hypot(ego_now.x - ego0.x, ego_now.y - ego0.y),
                         math.hypot(box.x - prev.x, box.y - prev.y))
            n = max(1, min(10, int(math.ceil(travel / 0.5))))
            for k in range(1, n + 1):
                a, b = ego0.lerp(ego_now, k / n), prev.lerp(box, k / n)
                ratio = overlap_ratio(a, b)
                if ratio > self.cfg.min_overlap:
                    side, fault = self._classify(a, b, ego_speed, speed)
                    events.append(Event("vehicle_collision", t, step, {
                        "other": agent_id, "overlap": round(ratio, 3), "side": side,
                        "at_fault": fault, "ego_speed": round(ego_speed, 3),
                        "other_speed": round(float(speed), 3)}))
                    break
        count = self.static_count(ego_now)
        if count > self.cfg.static_threshold:
            events.append(Event("static_collision", t, step, {"points": count}))
        distance = self.off_road_distance(ego_now)
        if distance > self.cfg.off_road_tolerance:
            events.append(Event("off_road", t, step, {"distance": round(distance, 3)}))
        if self._scene_tree is not None:
            d, _ = self._scene_tree.query((ego_now.x, ego_now.y))
            self._off_scene = self._off_scene + 1 if d > self.cfg.off_scene_radius else 0
            if self._off_scene >= self.cfg.off_scene_steps:
                events.append(Event("off_scene", t, step, {"distance": round(float(d), 2)}))
        return events
