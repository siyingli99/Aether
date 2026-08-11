"""AETHER simulation primitives: Actor / Scene / Policy.

Goals
-----
1. Backward compatible with ``aether/planner/npc.py:ActorReplay`` — exposes
   ``pose_at(frame)`` returning the same list of dicts the renderer expects.
2. Reactive: each actor owns an :class:`ActorPolicy` that can be log-replay,
   IDM-following, manual-override, or any user-defined subclass.
3. Editable: ``Scene.add/remove/replace`` for the BEV editor and CLI tools.
4. Introspectable: every actor carries a stable integer ``actor_id`` plus
   per-frame state (pos, heading, velo) for BEV visualization.
"""
from .actor import Actor, ActorState, Scene
from .policy import (
    ActorPolicy,
    LogReplayPolicy,
    IDMFollowerPolicy,
    ManualOverridePolicy,
)

__all__ = [
    "Actor", "ActorState", "Scene",
    "ActorPolicy", "LogReplayPolicy", "IDMFollowerPolicy", "ManualOverridePolicy",
]
