"""External traffic integration for AETHER's camera-only closed loop.

Imports are lazy so the standalone Windows bridge can use the stdlib-only
protocol module without installing AETHER's NumPy/SciPy rendering stack.
"""

from .protocol import EgoState, TrafficFrame, VehicleState

__all__ = [
    "EgoState",
    "FrameTransform",
    "LimSimTrafficClient",
    "ReplayTrafficClient",
    "TrafficActorProvider",
    "TrafficFrame",
    "VehicleState",
    "nuscenes_scene_local_to_ds",
]


def __getattr__(name):
    if name in {"LimSimTrafficClient", "ReplayTrafficClient"}:
        from .client import LimSimTrafficClient, ReplayTrafficClient
        return {"ReplayTrafficClient": ReplayTrafficClient,
                "LimSimTrafficClient": LimSimTrafficClient}[name]
    if name in {"FrameTransform", "nuscenes_scene_local_to_ds"}:
        from .coordinates import FrameTransform, nuscenes_scene_local_to_ds
        return {"FrameTransform": FrameTransform,
                "nuscenes_scene_local_to_ds": nuscenes_scene_local_to_ds}[name]
    if name == "TrafficActorProvider":
        from .provider import TrafficActorProvider
        return TrafficActorProvider
    raise AttributeError(name)
