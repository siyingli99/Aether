"""Background traffic: model interface, built-in models and the TCP protocol.

Write a traffic model by subclassing :class:`TrafficModel` (whole traffic
flow) or :class:`VehicleController` (one vehicle); see "Plug in a traffic model" in README.md.
The protocol module is stdlib-only so external simulators can vendor it.
"""
from .model import AgentState, TrafficModel, TrafficScene, VehicleController, load_class
from .protocol import EgoState, TrafficFrame, VehicleState

__all__ = ["AgentState", "TrafficModel", "TrafficScene", "VehicleController", "load_class",
           "EgoState", "TrafficFrame", "VehicleState"]
