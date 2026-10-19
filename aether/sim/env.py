"""Closed-loop driving environment on a reconstructed Gaussian scene.

One step: integrate the ego kinematics → advance background traffic → render
the cameras from the new pose → check collisions and violations.

All public state is in the road frame (see :mod:`aether.geometry.frames`).
The ego reference point is the rear-axle centre on the ground; the camera
mounts are calibrated so frame 0 reproduces the logged camera poses exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..config import Scenario
from ..geometry.boxes import Box2D
from ..paths import ASSETS_ROOT, resolve
from ..scene.bundle import SceneBundle
from ..scene.geometry import SceneGeometry
from ..traffic.model import AgentState
from ..vehicle import BicycleModel, VehicleState, load_profile
from .collision import CollisionChecker, CollisionConfig, Event
from .route import Route


@dataclass
class StepResult:
    info: dict[str, Any]
    terminated: bool
    truncated: bool


def _asset_dir(name: str):
    path = resolve(name)
    if (path / "asset.pt").is_file() or (path / "gs.pth").is_file():
        return path
    return ASSETS_ROOT / "vehicles" / name


class AetherEnv:
    def __init__(self, scenario: Scenario, device: str = "cuda") -> None:
        from ..render.ds_backend import DriveStudioRenderer

        self.scenario = scenario
        self.bundle = SceneBundle.load(scenario.scene)
        vehicle = scenario.ego.vehicle
        self.profile = load_profile(self.bundle.vehicle if vehicle == "auto" else vehicle)
        self.dt = float(scenario.sim.dt)
        self.frame_dt = self.bundle.frame_dt

        exclude = set(scenario.render.exclude_models)
        if scenario.traffic.hide_originals and not any(
                k.startswith("SMPLNodes/") for k in scenario.traffic.keep):
            exclude.add("SMPLNodes")       # also avoids needing the SMPL body model
        self.renderer = DriveStudioRenderer(self.bundle.checkpoint, device=device,
                                            excluded_model_classes=tuple(sorted(exclude)))
        if scenario.render.distill_overlay:
            self._apply_overlay(resolve(scenario.render.distill_overlay))
        self.num_frames = self.renderer.num_frames

        splats = self.renderer.background_points(min_opacity=0.3)
        self.geometry = SceneGeometry(self.bundle, self.profile, splats)
        self.rig = self.bundle.rig
        self.cameras = self.rig.cameras
        self.bicycle = BicycleModel(self.profile, self.dt, allow_reverse=scenario.ego.allow_reverse)

        track = self.geometry.track
        off = self.profile.center_offset
        centers = np.column_stack([track.x + off * np.cos(track.yaw), track.y + off * np.sin(track.yaw)])
        self.route = Route(centers, self.frame_dt)
        logged_boxes = [Box2D(cx, cy, yaw, self.profile.length, self.profile.width)
                        for (cx, cy), yaw in zip(centers, track.yaw)]
        term = scenario.termination
        self.collision = CollisionChecker(
            self.geometry.points_to_road(splats), self.geometry.ground, self.profile,
            self.bundle.hdmap, logged_boxes,
            CollisionConfig(min_overlap=term.min_overlap, static_threshold=term.static_threshold,
                            min_neighbors=term.min_neighbors, neighbor_radius=term.neighbor_radius,
                            carve_logged_route=term.carve_logged_route,
                            off_road_tolerance=term.off_road_tolerance,
                            off_scene_radius=term.off_scene_radius,
                            off_scene_steps=term.off_scene_steps))

        from ..render.external_assets import GaussianVehicleAsset
        assets = [GaussianVehicleAsset.load(_asset_dir(a)) for a in scenario.traffic.assets]
        from ..traffic.manager import TrafficManager
        self.traffic = TrafficManager(self.renderer, self.geometry, self.bundle,
                                      scenario.traffic.model_dump(), self.dt,
                                      seed=scenario.sim.seed, assets=assets)
        self.state = VehicleState()
        self.time = 0.0
        self.step_index = 0
        self.events: list[Event] = []
        self._prev_boxes: dict[str, tuple[Box2D, float]] = {}
        self._prev_ego: Box2D | None = None

    # ------------------------------------------------------------ helpers
    def _apply_overlay(self, path) -> None:
        import torch
        state = torch.load(path, map_location="cuda", weights_only=False)
        self.renderer.trainer.models["Background"].load_state_dict(state)

    @property
    def frame(self) -> int:
        """Reconstruction frame shown at the current time (clamped)."""
        index = int(math.floor(self.time / self.frame_dt + 0.5 + 1e-9))
        return min(index, self.num_frames - 1)

    def start_state(self) -> VehicleState:
        track = self.geometry.track
        start = self.scenario.ego.start
        if start == "logged":
            x, y, yaw = float(track.x[0]), float(track.y[0]), float(track.yaw[0])
        else:
            off = self.profile.center_offset
            yaw = float(start.yaw)
            x, y = start.x - off * math.cos(yaw), start.y - off * math.sin(yaw)
        speed = self.scenario.ego.speed
        if speed == "logged":
            speed = float(self.geometry.track.speed()[0])
        return VehicleState(x, y, yaw, max(0.0, float(speed)))

    def ego_box(self, state: VehicleState | None = None) -> Box2D:
        s = state or self.state
        cx, cy = self.bicycle.center(s)
        return Box2D(cx, cy, s.yaw, self.profile.length, self.profile.width)

    def ego_agent(self) -> AgentState:
        box = self.ego_box()
        return AgentState("ego", box.x, box.y, box.yaw, self.state.v, self.profile.length,
                          self.profile.width, self.profile.height, self.state.accel, "ego")

    def vehicle_pose_road(self) -> np.ndarray:
        s = self.state
        return self.geometry.vehicle_pose(s.x, s.y, s.yaw)

    def camera_pose(self, name: str) -> np.ndarray:
        """Scene-frame camera-to-world of camera ``name`` (render pose)."""
        return self.geometry.to_scene(self.vehicle_pose_road()) @ self.geometry.camera_mounts[name]

    def sensor_pose(self, name: str = "LIDAR_TOP") -> np.ndarray:
        """Scene-frame pose of a camera, the LiDAR, ``ANCHOR`` or ``VEHICLE``."""
        vehicle = self.geometry.to_scene(self.vehicle_pose_road())
        if name == "VEHICLE":
            return vehicle
        if name == "ANCHOR":
            return vehicle @ self.geometry.anchor_mount_effective
        if name in self.geometry.camera_mounts:
            return vehicle @ self.geometry.camera_mounts[name]
        return vehicle @ self.rig.lidar_mount(name)

    # ------------------------------------------------------------- render
    def render(self, cameras: list[str] | None = None, depth: bool = False) -> dict[str, Any]:
        """Render ``cameras`` (default: all) at the current pose and time."""
        import torch

        frame = self.frame
        self.renderer.clear_actor_overrides()
        for cls, index, pose in self.traffic.render_overrides():
            self.renderer.set_actor_pose(cls, index, frame, pose)
        rgb, depths = {}, {}
        for name in cameras or list(self.cameras):
            cam = self.cameras[name]
            out = self.renderer.render(c2w_4x4=self.camera_pose(name), K_3x3=cam.K,
                                       H=cam.height, W=cam.width, frame_idx=frame, cam_idx=cam.index)
            rgb[name] = (out["rgb"].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
            if depth:
                depths[name] = out["depth"][..., 0].float().cpu().numpy()
        return {"rgb": rgb, "depth": depths}

    # ---------------------------------------------------------- lifecycle
    def reset(self) -> dict[str, Any]:
        """Start an episode; returns the info dict (call :meth:`render` for images)."""
        self.state = self.start_state()
        self.time = 0.0
        self.step_index = 0
        self.events = []
        self.collision.reset()
        self.traffic.reset(self.ego_agent(), self.frame)
        self._prev_ego = self.ego_box()
        self._prev_boxes = self._agent_boxes()
        return self.info()

    def _agent_boxes(self) -> dict[str, tuple[Box2D, float]]:
        return {a.state.id if a.source == "recorded" else f"{a.source}:{a.state.id}":
                (Box2D(a.state.x, a.state.y, a.state.yaw, a.state.length, a.state.width), a.state.speed)
                for a in self.traffic.agents()}

    def step(self, accel: float, steer_rate: float) -> StepResult:
        """Advance ``sim.dt`` seconds with the given control (steer_rate: left +)."""
        self.state = self.bicycle.step(self.state, accel, steer_rate)
        self.time = round(self.time + self.dt, 9)
        self.step_index += 1
        self.traffic.advance(self.time, self.ego_agent(), self.frame)
        boxes = self._agent_boxes()
        ego_box = self.ego_box()
        events = self.collision.check(self.time, self.step_index, self._prev_ego, ego_box,
                                      self.state.v, self._prev_boxes, boxes)
        self.events.extend(events)
        self._prev_ego, self._prev_boxes = ego_box, boxes
        term = self.scenario.termination
        terminal = [e for e in events if e.type in term.stop_on and
                    not (term.at_fault_only and e.type == "vehicle_collision" and not e.detail.get("at_fault"))]
        terminated = bool(terminal)
        truncated = (self.step_index >= self.scenario.sim.max_steps
                     or self.time / self.frame_dt >= self.num_frames - 1)
        info = self.info(events)
        info["terminated_by"] = terminal[0].type if terminal else None
        return StepResult(info, terminated, truncated)

    def close(self) -> None:
        self.traffic.close()

    # --------------------------------------------------------------- info
    def info(self, events: list[Event] | None = None) -> dict[str, Any]:
        s = self.state
        box = self.ego_box()
        return {
            "time": self.time, "step": self.step_index, "frame": self.frame,
            "ego": {
                "x": box.x, "y": box.y, "yaw": s.yaw, "speed": s.v, "accel": s.accel,
                "steer": s.steer, "steer_rate": s.steer_rate,
                "yaw_rate": self.bicycle.yaw_rate(s), "rear_axle": [s.x, s.y],
                "length": self.profile.length, "width": self.profile.width,
                "height": self.profile.height,
            },
            "agents": [{**a.state.to_dict(), "source": a.source, "binding": a.binding}
                       for a in self.traffic.agents()],
            "command": self.route.command(box.x, box.y, s.yaw),
            "route_progress": self.route.progress(box.x, box.y),
            "events": [e.to_dict() for e in (events or [])],
        }
