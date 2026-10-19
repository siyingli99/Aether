"""Traffic model interface — plug a 2D traffic-flow model into AETHER.

A traffic model owns a set of background vehicles and advances all of them
together, exactly like a microscopic traffic simulator.  AETHER calls it
once per closed-loop step with the current ego state and renders whatever it
returns in the reconstructed 3D scene.

Coordinates: road frame of the scene (metres, ``x`` along the initial ego
heading, ``y`` left, yaw counter-clockwise in radians), the same frame as
the HD map in :attr:`TrafficScene.hdmap`.  Positions are box centres.

Minimal model::

    from aether.traffic import AgentState, TrafficModel

    class ConstantSpeed(TrafficModel):
        def reset(self, scene):
            self.vehicles = {"car-1": AgentState("car-1", x=20.0, y=0.0, yaw=0.0, speed=8.0)}
            return list(self.vehicles.values())

        def step(self, t, dt, ego, vehicles):
            return [v.moved(dt) for v in vehicles]

Register it in the scenario::

    traffic:
      model: {class: my_pkg.traffic:ConstantSpeed, params: {}}
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import importlib
import math
from typing import Any, Iterable


@dataclass(frozen=True)
class AgentState:
    """State of one vehicle (or the ego) in the road frame."""

    id: str
    x: float
    y: float
    yaw: float
    speed: float = 0.0
    length: float = 4.5
    width: float = 1.85
    height: float = 1.5
    accel: float = 0.0
    type: str = "car"
    lane_id: str | None = None

    def moved(self, dt: float, accel: float = 0.0, yaw_rate: float = 0.0) -> "AgentState":
        """Constant-turn-rate-and-acceleration update (convenience helper)."""
        speed = max(self.speed + accel * dt, 0.0)
        yaw = self.yaw + yaw_rate * dt
        return replace(self, x=self.x + speed * math.cos(yaw) * dt,
                       y=self.y + speed * math.sin(yaw) * dt,
                       yaw=yaw, speed=speed, accel=accel)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "x": self.x, "y": self.y, "yaw": self.yaw,
                "speed": self.speed, "length": self.length, "width": self.width,
                "height": self.height, "accel": self.accel, "type": self.type,
                "lane_id": self.lane_id}


@dataclass
class TrafficScene:
    """What a traffic model receives on reset."""

    scene_id: str
    dt: float                         # closed-loop step (s)
    ego: AgentState                   # initial ego state
    hdmap: Any = None                 # aether.scene.hdmap.HDMap or None
    network: tuple | None = None      # (net.xml, rou.xml) paths when available
    seed: int = 0
    params: dict[str, Any] = field(default_factory=dict)


class TrafficModel:
    """Base class of background traffic models.

    Override :meth:`reset` and :meth:`step`; :meth:`close` is optional.
    ``params`` from the scenario are passed to ``__init__`` as keywords.
    """

    def __init__(self, **params: Any) -> None:
        self.params = params

    def reset(self, scene: TrafficScene) -> list[AgentState]:
        """Initialise and return the vehicles present at t = 0."""
        raise NotImplementedError

    def step(self, t: float, dt: float, ego: AgentState,
             vehicles: list[AgentState]) -> list[AgentState]:
        """Advance from ``t - dt`` to ``t`` and return the vehicles present at ``t``.

        ``vehicles`` is what the model returned last time.  Vehicles may
        appear and disappear between steps; ids must stay stable.
        """
        raise NotImplementedError

    def close(self) -> None:
        pass


class VehicleController:
    """Per-vehicle controller used by Studio (one instance per vehicle).

    ``step`` returns the vehicle's new road-frame pose as a dict
    ``{x, y, yaw, speed}`` (box centre); vehicles using controllers are driven
    by :class:`aether.traffic.builtin.ScriptedVehicles`.
    """

    def setup(self, context: dict[str, Any]) -> None:
        pass

    def reset(self) -> None:
        pass

    def step(self, time_s: float, actor: Any, ego: AgentState) -> dict[str, float]:
        raise NotImplementedError

    def close(self) -> None:
        pass


def load_class(reference: str):
    """Import ``package.module:ClassName`` or ``path/to/file.py:ClassName``."""
    module_name, separator, class_name = str(reference).rpartition(":")
    if not separator or not module_name:
        raise ValueError(f"class reference must be 'package.module:ClassName', got {reference!r}")
    if module_name.endswith(".py"):
        from importlib.util import module_from_spec, spec_from_file_location
        from pathlib import Path
        path = Path(module_name).expanduser().resolve()
        spec = spec_from_file_location(path.stem, path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return getattr(module, class_name)
    return getattr(importlib.import_module(module_name), class_name)


def as_agents(values: Iterable[Any]) -> list[AgentState]:
    """Accept AgentState objects or plain dicts from user models."""
    agents = []
    for value in values or []:
        if isinstance(value, AgentState):
            agents.append(value)
        elif isinstance(value, dict):
            known = {k: value[k] for k in AgentState.__dataclass_fields__ if k in value}
            known["id"] = str(known["id"])
            agents.append(AgentState(**known))
        else:
            raise TypeError(f"traffic models must return AgentState or dict, got {type(value)!r}")
    ids = [a.id for a in agents]
    if len(ids) != len(set(ids)):
        raise ValueError("traffic model returned duplicate vehicle ids")
    return agents
