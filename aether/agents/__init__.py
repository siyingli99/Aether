"""Ego agents: in-process (:class:`EgoAgent`) or external processes (adapters)."""
from .base import Action, Control, EgoAgent, Observation, Plan

__all__ = ["Action", "Control", "EgoAgent", "Observation", "Plan", "create_agent"]


def create_agent(spec: str, workdir, cuda: str = "0", **kwargs) -> EgoAgent:
    """``spec``: adapter name/YAML (``uniad``), ``replay``, ``constant`` or
    ``package.module:Class`` for an in-process agent."""
    from pathlib import Path
    if spec in ("replay", "log_replay"):
        from .builtin import LogReplayAgent
        return LogReplayAgent(**kwargs)
    if spec in ("constant", "idle"):
        from .builtin import ConstantControl
        return ConstantControl(**kwargs)
    if ":" in spec and not Path(spec).exists():
        from ..traffic.model import load_class
        return load_class(spec)(**kwargs)
    from .external import ExternalAgent, load_adapter
    return ExternalAgent(load_adapter(spec), Path(workdir) / "agent", cuda)
