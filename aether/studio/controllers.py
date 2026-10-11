"""Reference controllers for the Studio plugin interface."""
from __future__ import annotations

import math
from typing import Any

from .schemas import RoadPose


class ConstantVelocityController:
    """Move an actor straight ahead in its road-frame start direction."""

    def setup(self, context: dict[str, Any]) -> None:
        self.initial = context["actor"].start or RoadPose()
        self.speed = float(context["actor"].controller.params.get("speed_mps", 5.0))

    def reset(self) -> None:
        pass

    def step(self, time_s: float, actor: Any, ego: Any) -> RoadPose:
        distance = self.speed * time_s
        return RoadPose(
            x=self.initial.x + distance * math.cos(self.initial.yaw),
            y=self.initial.y + distance * math.sin(self.initial.yaw),
            z=self.initial.z,
            yaw=self.initial.yaw,
        )

    def close(self) -> None:
        pass
