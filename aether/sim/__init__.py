"""Closed-loop simulation: environment, collision checks, route, episode runner."""
from .collision import CollisionChecker, CollisionConfig, Event
from .route import LEFT, RIGHT, STRAIGHT, Route

__all__ = ["CollisionChecker", "CollisionConfig", "Event", "Route", "LEFT", "RIGHT", "STRAIGHT",
           "AetherEnv", "run_episode"]


def __getattr__(name):            # heavy imports (torch, renderer) only on demand
    if name == "AetherEnv":
        from .env import AetherEnv
        return AetherEnv
    if name == "run_episode":
        from .runner import run_episode
        return run_episode
    raise AttributeError(name)
