"""Scenario configuration: typed, validated, with ``extends`` inheritance.

A scenario YAML names a scene bundle and says how to run it::

    scene: nuscenes-0383
    sim:     {dt: 0.25, max_steps: 39, seed: 0}
    ego:     {vehicle: auto, start: logged, speed: 1.0}
    traffic: {assets: [3drealcar_2024_06_04_15_34_51], max_vehicles: 3,
              model: {type: limsim, port: 8766}}
    refine:  {enabled: true, required: true}
    termination: {stop_on: [vehicle_collision, static_collision, off_scene]}

``extends: base.yaml`` merges a parent file (recursively, relative to the
child).  Unknown keys are errors, so a typo never silently falls back to a
default.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SimConfig(_Strict):
    dt: float = Field(0.25, gt=0)            # closed-loop step (s)
    max_steps: int = Field(40, ge=1)
    seed: int = 0


class StartPose(_Strict):
    x: float = 0.0                           # vehicle box centre, road frame
    y: float = 0.0
    yaw: float = 0.0


class EgoConfig(_Strict):
    vehicle: str | dict[str, Any] = "auto"   # profile name, dict, or auto
    start: Literal["logged"] | StartPose = "logged"
    speed: Literal["logged"] | float = "logged"   # initial speed (m/s) or the recorded one
    allow_reverse: bool = False


class TrafficConfig(_Strict):
    hide_originals: bool = True
    keep: list[str] = Field(default_factory=list)       # "RigidNodes/3", "RigidNodes/*", "*"
    assets: list[str] = Field(default_factory=list)
    max_vehicles: int = Field(6, ge=0)
    model: dict[str, Any] | None = None
    vehicles: list[dict[str, Any]] = Field(default_factory=list)


class RefineConfig(_Strict):
    enabled: bool = True
    required: bool = False
    model: str = "nvidia/difix"
    max_side: int = Field(1024, ge=256)
    timestep: int = Field(199, ge=0)
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    cuda: str | None = None


class TerminationConfig(_Strict):
    stop_on: list[str] = Field(default_factory=lambda: [
        "vehicle_collision", "static_collision", "off_scene"])
    at_fault_only: bool = False
    min_overlap: float = 0.0
    static_threshold: int = 50
    min_neighbors: int = 8
    neighbor_radius: float = 0.3
    carve_logged_route: bool = True
    off_road_tolerance: float = 0.3
    off_scene_radius: float = 5.0
    off_scene_steps: int = 4

    @field_validator("stop_on")
    @classmethod
    def _known(cls, value):
        known = {"vehicle_collision", "static_collision", "off_road", "off_scene"}
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"unknown termination events {unknown}; choose from {sorted(known)}")
        return value


class RenderConfig(_Strict):
    exclude_models: list[str] = Field(default_factory=list)   # e.g. [SMPLNodes]
    distill_overlay: str | None = None


class LidarConfig(_Strict):
    """LiDAR is produced only when the agent asks for it (``input.lidar``)."""
    splatad: bool = True                     # use a SplatAD sidecar when one exists
    sidecar: str | None = None               # default: next to the scene checkpoint
    bundle: str | None = None                # unified refiner bundle
    refine: bool = True


class OutputConfig(_Strict):
    root: str | None = None                  # default: $AETHER_OUTPUTS/runs
    videos: bool = True
    bev_fps: int = Field(20, ge=1, le=60)
    trajectory_hz: float = Field(10.0, gt=0)
    # Save what the agent received each step (logs/io/step_XXXX.jpg) and the
    # scene-frame render poses (logs/render_poses.json) for offline
    # re-rendering / closed-loop distillation.  Costs disk, off by default.
    io_trace: bool = False


class AgentConfig(_Strict):
    name: str | None = None                  # adapter (uniad, vad, ltf, vla …), replay, or pkg.mod:Class
    cuda: str = "0"
    options: dict[str, Any] = Field(default_factory=dict)   # kwargs for in-process agents


class Scenario(_Strict):
    scene: str
    name: str | None = None
    description: str | None = None
    sim: SimConfig = Field(default_factory=SimConfig)
    ego: EgoConfig = Field(default_factory=EgoConfig)
    traffic: TrafficConfig = Field(default_factory=TrafficConfig)
    refine: RefineConfig = Field(default_factory=RefineConfig)
    termination: TerminationConfig = Field(default_factory=TerminationConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    lidar: LidarConfig = Field(default_factory=LidarConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    source_path: str | None = None


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular 'extends' through {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        parent_path = Path(str(parent))
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        data = _merge(_read(parent_path, seen + (path,)), data)
    return data


def load_scenario(path: str | Path, overrides: dict[str, Any] | None = None) -> Scenario:
    """Read a scenario YAML (following ``extends``) and validate it."""
    path = Path(path)
    data = _read(path)
    if overrides:
        data = _merge(data, overrides)
    data["source_path"] = str(path.resolve())
    return Scenario.model_validate(data)


def scenario_from_dict(data: dict[str, Any]) -> Scenario:
    return Scenario.model_validate(data)
