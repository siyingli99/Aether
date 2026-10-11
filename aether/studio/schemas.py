"""Versioned public data model used by the Studio API and compiler."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RoadPose(BaseModel):
    """Pose in the only coordinate frame exposed by Studio.

    ``x`` points forward from the scene's initial ego pose, ``y`` points left,
    ``z`` points up and positive yaw turns counter-clockwise (towards left).
    Units are metres and radians.
    """

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    yaw: float = 0.0


class TrajectoryPoint(RoadPose):
    t: float = Field(ge=0.0)
    speed: float | None = Field(default=None, ge=0.0)


class ControllerSpec(BaseModel):
    kind: Literal["recorded", "open_loop", "limsim", "plugin"] = "recorded"
    trajectory: list[TrajectoryPoint] = Field(default_factory=list)
    plugin: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_controller(self) -> "ControllerSpec":
        if self.kind == "open_loop" and len(self.trajectory) < 2:
            raise ValueError("open_loop controller requires at least two trajectory points")
        if self.kind == "plugin" and not self.plugin:
            raise ValueError("plugin controller requires a registered plugin name")
        times = [point.t for point in self.trajectory]
        if times != sorted(times) or len(set(times)) != len(times):
            raise ValueError("trajectory point times must be unique and increasing")
        return self


class ActorEdit(BaseModel):
    id: str
    source: Literal["scene", "external"]
    operation: Literal["keep", "remove", "edit", "replace", "add"]
    node_class: Literal["RigidNodes", "DeformableNodes", "SMPLNodes"] = "RigidNodes"
    instance_id: int | None = Field(default=None, ge=0)
    label: str = "vehicle"
    asset_id: str | None = None
    start: RoadPose | None = None
    size: tuple[float, float, float] | None = None
    controller: ControllerSpec = Field(default_factory=ControllerSpec)

    @model_validator(mode="after")
    def validate_actor(self) -> "ActorEdit":
        if self.source == "scene" and self.instance_id is None:
            raise ValueError("scene actors require instance_id")
        if self.operation in {"replace", "add"} and not self.asset_id:
            raise ValueError(f"{self.operation} requires asset_id")
        if self.operation == "add" and self.source != "external":
            raise ValueError("added actors must use source=external")
        if self.node_class != "RigidNodes" and self.operation in {"edit", "replace"}:
            raise ValueError("v1 only supports keep/remove for pedestrians and cyclists")
        return self


class EgoSpec(BaseModel):
    start: RoadPose = Field(default_factory=RoadPose)
    speed: float = Field(default=1.0, ge=0.0)


class AlgorithmSpec(BaseModel):
    adapter: str = "uniad"
    cuda: str = "0"
    config_path: str | None = None


class DifixSpec(BaseModel):
    enabled: bool = True
    cuda: str = "0"
    max_side: int = Field(default=1024, ge=256)
    model: str = "nvidia/difix"
    timestep: int = Field(default=199, ge=0)
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"


class RuntimeSpec(BaseModel):
    max_steps: int = Field(default=39, ge=1, le=4000)
    seed: int = 0
    difix: DifixSpec = Field(default_factory=DifixSpec)


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["aether.experiment.v1"] = "aether.experiment.v1"
    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    name: str = "Untitled experiment"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    scene_id: str
    ego: EgoSpec = Field(default_factory=EgoSpec)
    actors: list[ActorEdit] = Field(default_factory=list)
    algorithm: AlgorithmSpec = Field(default_factory=AlgorithmSpec)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)

    @field_validator("id")
    @classmethod
    def safe_id(cls, value: str) -> str:
        if not value or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for char in value):
            raise ValueError("id may only contain letters, digits, '-' and '_'")
        return value

    @model_validator(mode="after")
    def unique_actor_ids(self) -> "ExperimentSpec":
        actor_ids = [actor.id for actor in self.actors]
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("actor ids must be unique")
        return self
