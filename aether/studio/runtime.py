"""Runtime bridge from a compiled Studio experiment to DriveStudio actors."""
from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aether.render.external_assets import ExternalVehiclePool, GaussianVehicleAsset
from aether.traffic.client import LimSimTrafficClient
from aether.traffic.coordinates import nuscenes_scene_local_to_ds
from aether.traffic.provider import TrafficActorProvider

from .coordinates import nuscenes_ds_to_road_pose, road_pose_to_nuscenes_ds
from .schemas import ActorEdit, ExperimentSpec, RoadPose, TrajectoryPoint


class PluginController:
    """Adapter for a registered ``module:Class`` local traffic controller.

    A plugin class may implement ``setup(context)``, ``reset()``,
    ``step(time_s, actor, ego) -> RoadPose | dict`` and ``close()``.
    """

    def __init__(self, reference: str, context: dict[str, Any]) -> None:
        module_name, separator, class_name = reference.partition(":")
        if not separator:
            raise ValueError("plugin reference must be 'python.module:ClassName'")
        cls = getattr(importlib.import_module(module_name), class_name)
        self.controller = cls()
        if hasattr(self.controller, "setup"):
            self.controller.setup(context)

    def reset(self) -> None:
        if hasattr(self.controller, "reset"):
            self.controller.reset()

    def step(self, time_s: float, actor: ActorEdit, ego: Any) -> RoadPose:
        value = self.controller.step(time_s=time_s, actor=actor, ego=ego)
        return value if isinstance(value, RoadPose) else RoadPose.model_validate(value)

    def close(self) -> None:
        if hasattr(self.controller, "close"):
            self.controller.close()


def _interpolate(points: list[TrajectoryPoint], time_s: float) -> RoadPose | None:
    if not points:
        return None
    if time_s <= points[0].t:
        point = points[0]
        return RoadPose(x=point.x, y=point.y, z=point.z, yaw=point.yaw)
    if time_s >= points[-1].t:
        point = points[-1]
        return RoadPose(x=point.x, y=point.y, z=point.z, yaw=point.yaw)
    right = next(index for index, point in enumerate(points) if point.t > time_s)
    first, second = points[right - 1], points[right]
    alpha = (time_s - first.t) / max(second.t - first.t, 1e-9)
    delta_yaw = (second.yaw - first.yaw + math.pi) % (2.0 * math.pi) - math.pi
    return RoadPose(
        x=first.x + alpha * (second.x - first.x),
        y=first.y + alpha * (second.y - first.y),
        z=first.z + alpha * (second.z - first.z),
        yaw=first.yaw + alpha * delta_yaw,
    )


class StudioActorManager:
    """Apply selective scene editing and merge local/LimSim controllers."""

    def __init__(self, renderer: Any, studio_cfg: Any, env: Any) -> None:
        self.renderer = renderer
        self.env = env
        self.dt = float(env.ego.dt)
        self.spec = ExperimentSpec.model_validate_json(
            Path(str(studio_cfg.experiment_path)).read_text(encoding="utf-8")
        )
        self.log = np.load(str(studio_cfg.traffic_log), allow_pickle=False)
        self._original_visibility: dict[str, torch.Tensor] = {}
        for node_class in ("RigidNodes", "DeformableNodes", "SMPLNodes"):
            model = renderer.trainer.models.get(node_class)
            visibility = getattr(model, "instances_fv", None) if model is not None else None
            if visibility is not None:
                self._original_visibility[node_class] = visibility.detach().clone()

        self.external_specs = [
            actor for actor in self.spec.actors
            if actor.operation in {"replace", "add"}
        ]
        self.pool: ExternalVehiclePool | None = None
        self.external_slots: dict[str, int] = {}
        if self.external_specs:
            asset_paths = {str(key): str(value) for key, value in studio_cfg.assets.items()}
            assets = [
                GaussianVehicleAsset.load(asset_paths[actor.asset_id], max_gaussians=50_000)
                for actor in self.external_specs
            ]
            self.pool = ExternalVehiclePool(
                renderer, assets, capacity=len(assets),
                hide_original_rigid=False, hide_original_dynamic=False,
            )
            self.external_slots = {
                actor.id: slot for actor, slot in zip(self.external_specs, self.pool.instance_ids)
            }
        self._asset_index = {actor.id: index for index, actor in enumerate(self.external_specs)}

        self._configure_visibility()
        self.plugins: dict[str, PluginController] = {}
        for actor in self.spec.actors:
            if actor.controller.kind == "plugin":
                reference = str(studio_cfg.plugins[str(actor.controller.plugin)])
                self.plugins[actor.id] = PluginController(
                    reference,
                    {"experiment": self.spec, "actor": actor},
                )

        limsim_specs = [
            actor for actor in self.external_specs if actor.controller.kind == "limsim"
        ]
        self.limsim: TrafficActorProvider | None = None
        if limsim_specs:
            cfg = studio_cfg.limsim
            client = LimSimTrafficClient(
                host=str(cfg.host), port=int(cfg.port), timeout_s=30.0
            )
            slots = [self.external_slots[actor.id] for actor in limsim_specs]
            transform = nuscenes_scene_local_to_ds(
                ground_ds_y=float(getattr(env, "_ground_ref", 0.0))
            )
            self.limsim = TrafficActorProvider(client, transform, slots)

        self._entries: list[dict[str, Any]] = []
        self._entries_key: tuple[int, float] | None = None

    @property
    def frame(self):
        return self.limsim.frame if self.limsim is not None else None

    def _configure_visibility(self) -> None:
        # An experiment is a complete scene description: unspecified originals
        # are removed, then explicit keep operations restore their training mask.
        for node_class, original in self._original_visibility.items():
            model = self.renderer.trainer.models[node_class]
            count = int(original.shape[1])
            with torch.no_grad():
                model.instances_fv[:, :count] = False
        for actor in self.spec.actors:
            if actor.source != "scene" or actor.instance_id is None:
                continue
            if actor.operation != "keep" or actor.controller.kind != "recorded":
                continue
            original = self._original_visibility.get(actor.node_class)
            if original is None or actor.instance_id >= original.shape[1]:
                continue
            with torch.no_grad():
                self.renderer.trainer.models[actor.node_class].instances_fv[:, actor.instance_id] = (
                    original[:, actor.instance_id]
                )

    def removal_audit(self) -> dict[str, int]:
        result = {}
        for node_class, original in self._original_visibility.items():
            current = self.renderer.trainer.models[node_class].instances_fv[:, :original.shape[1]]
            originally_visible = original.any(dim=0)
            currently_visible = current.any(dim=0)
            result[node_class] = int((originally_visible & ~currently_visible).sum().item())
        return result

    def _recorded(self, actor: ActorEdit, frame_idx: int) -> np.ndarray | None:
        if actor.instance_id is None:
            return None
        actor_key = f"{actor.node_class}/{actor.instance_id}"
        frames_key, c2w_key = f"frames/{actor_key}", f"c2w/{actor_key}"
        if frames_key not in self.log or c2w_key not in self.log:
            return None
        frames = self.log[frames_key]
        indices = np.flatnonzero(frames == int(frame_idx))
        return None if indices.size == 0 else self.log[c2w_key][int(indices[0])].astype(np.float32)

    def _pose_for(self, actor: ActorEdit, frame_idx: int, time_s: float) -> np.ndarray | None:
        if actor.controller.kind == "recorded":
            recorded = self._recorded(actor, frame_idx)
            if recorded is not None:
                return recorded
            pose = actor.start
        elif actor.controller.kind == "open_loop":
            pose = _interpolate(actor.controller.trajectory, time_s)
        elif actor.controller.kind == "plugin":
            pose = self.plugins[actor.id].step(time_s, actor, self.env._traffic_ego_state())
        else:
            return None
        if pose is None:
            return None
        height = float((actor.size or (2.0, 4.5, 1.5))[2])
        if actor.id in self.external_slots and self.pool is not None:
            height = float(self.pool.assets[self._asset_index[actor.id]].size_lwh[2])
        centered = RoadPose(x=pose.x, y=pose.y, z=pose.z + height * 0.5, yaw=pose.yaw)
        return road_pose_to_nuscenes_ds(
            centered, ground_ds_y=float(getattr(self.env, "_ground_ref", 0.0))
        )

    def pose_at(self, frame_idx: int) -> list[dict[str, Any]]:
        """Return actor poses for the env's current time.

        The env calls this several times per step (render, collision, info);
        plugin controllers may be stateful, so they are stepped once per
        simulation time and the result is cached.  ``frame_idx`` selects the
        recorded DriveStudio frame; open-loop and plugin controllers use the
        simulation clock.
        """
        time_s = float(getattr(self.env, "timestamp", frame_idx * self.dt))
        key = (int(frame_idx), round(time_s, 9))
        if key == self._entries_key:
            return list(self._entries)
        entries: list[dict[str, Any]] = []
        for actor in self.spec.actors:
            if actor.operation == "remove" or actor.controller.kind == "limsim":
                continue
            matrix = self._pose_for(actor, frame_idx, time_s)
            if matrix is None:
                continue
            node_class = "RigidNodes" if actor.id in self.external_slots else actor.node_class
            instance_id = self.external_slots.get(actor.id, actor.instance_id)
            if instance_id is None:
                continue
            if actor.id in self.external_slots and self.pool is not None:
                length, width, height = self.pool.assets[self._asset_index[actor.id]].size_lwh
                size = [width, length, height]
            else:
                size = list(actor.size or (2.0, 4.5, 1.5))
            road_pose = nuscenes_ds_to_road_pose(
                matrix, ground_ds_y=float(getattr(self.env, "_ground_ref", 0.0))
            )
            entries.append({
                "class": node_class, "instance_id": int(instance_id),
                "actor_id": actor.id, "size": size, "c2w": matrix,
                "x": road_pose.x, "y": road_pose.y, "z": road_pose.z,
                "yaw": road_pose.yaw, "velo": 0.0, "label": actor.label,
            })
        if self.limsim is not None:
            entries.extend(self.limsim.pose_at(frame_idx))
        self._entries = entries
        self._entries_key = key
        return list(entries)

    def _current_frame(self) -> int:
        frame = getattr(self.env, "data_frame", None)
        if frame is None:
            frame = int(round(float(getattr(self.env, "timestamp", 0.0)) / self.dt))
        return int(frame)

    def reset(self, ego=None) -> list[dict[str, Any]]:
        for plugin in self.plugins.values():
            plugin.reset()
        if self.limsim is not None:
            self.limsim.reset(ego)
        self._entries_key = None
        return self.pose_at(self._current_frame())

    def advance(self, target_time_s: float, ego) -> list[dict[str, Any]]:
        if self.limsim is not None:
            self.limsim.advance(target_time_s, ego)
        self._entries_key = None
        return self.pose_at(self._current_frame())

    def close(self) -> None:
        if self.limsim is not None:
            self.limsim.close()
        for plugin in self.plugins.values():
            plugin.close()
