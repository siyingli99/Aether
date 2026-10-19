"""Background traffic: sources, render bindings and per-step state.

One :class:`TrafficManager` per closed loop replaces the earlier replay /
policy / LimSim-provider / Studio-runtime actor managers.  Each vehicle
comes from a *source* and is shown through a *render binding*:

=================  ==============================================  ==================
source             vehicles                                        rendered with
=================  ==============================================  ==================
``recorded``       reconstructed actors listed in ``keep``          their own Gaussians,
                                                                    baked trajectory
``vehicles``       scripted vehicles (open loop / plugin / static)  an asset slot, or an
                                                                    original instance
``model``          vehicles owned by a traffic model (LimSim, IDM,  asset slots
                   replay, user ``TrafficModel``)
=================  ==============================================  ==================

:meth:`advance` is called once per step; :meth:`agents` and
:meth:`render_overrides` only read the cached state, so querying them
several times per step never advances any model.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

import math

from ..geometry.frames import pose2d, pose_from_matrix
from .builtin import ExternalSimulator, IDMLanes, ReplayTraffic, ScriptedVehicles
from .model import AgentState, TrafficModel, TrafficScene, as_agents, load_class

DYNAMIC_CLASSES = ("RigidNodes", "DeformableNodes", "SMPLNodes")


@dataclass
class TrafficAgent:
    state: AgentState
    source: str                      # recorded | vehicles | model
    binding: str                     # "asset:<slot>" | "original:<Class>/<id>"
    scene_pose: np.ndarray | None = None   # 4x4 render pose (None = baked)


def build_model(spec: dict[str, Any] | None) -> TrafficModel | None:
    """Model from ``traffic.model``: ``{type: limsim|external|replay|idm}``
    or ``{class: package.module:Class, params: {...}}``."""
    if not spec:
        return None
    spec = dict(spec)
    params = dict(spec.pop("params", {}) or {})
    kind = str(spec.pop("type", "class" if "class" in spec else "")).lower()
    if kind in ("limsim", "external"):
        server = "limsim" if kind == "limsim" else spec.pop("server", "traffic")
        spec.pop("max_vehicles", None)
        return ExternalSimulator(server=server, **spec, **params)
    if kind == "replay":
        return ReplayTraffic(**spec, **params)
    if kind == "idm":
        return IDMLanes(**spec, **params)
    if kind == "class" or "class" in spec:
        model = load_class(spec["class"])(**params)
        if not isinstance(model, TrafficModel) and not hasattr(model, "step"):
            raise TypeError(f"{spec['class']} is not a TrafficModel")
        return model
    raise ValueError(f"unknown traffic model type {kind!r}")


class TrafficManager:
    def __init__(self, renderer, geometry, bundle, cfg: dict[str, Any], dt: float,
                 seed: int = 0, assets: list | None = None) -> None:
        self.renderer, self.geometry, self.bundle = renderer, geometry, bundle
        self.cfg = cfg
        self.dt = float(dt)
        self.seed = int(seed)
        models = renderer.trainer.models
        self._models = models
        self._original_fv = {
            name: models[name].instances_fv.detach().clone()
            for name in DYNAMIC_CLASSES if name in models and hasattr(models[name], "instances_fv")
        }
        # ------------------------------------------------ recorded originals
        self.recorded = self._load_recorded()
        self.keep = self._keys(cfg.get("keep", []))
        # ---------------------------------------------- scripted vehicles
        self.vehicle_specs = list(cfg.get("vehicles", []) or [])
        self.edited_originals = {}
        for spec in self.vehicle_specs:
            render = spec.get("render") or {}
            if render.get("original"):
                self.edited_originals[str(spec["id"])] = self._key(render["original"])
        self.scripted = ScriptedVehicles(self.vehicle_specs) if self.vehicle_specs else None
        self.model = build_model(cfg.get("model"))
        # ----------------------------------------------------- asset pool
        asset_vehicles = [s for s in self.vehicle_specs if not (s.get("render") or {}).get("original")]
        capacity = len(asset_vehicles) + (int(cfg.get("max_vehicles", 6)) if self.model else 0)
        self.pool = None
        self.slot_size: dict[int, tuple[float, float, float]] = {}
        if capacity > 0:
            if not assets:
                raise ValueError("traffic vehicles need at least one asset (traffic.assets)")
            from ..render.external_assets import ExternalVehiclePool
            cycled = [assets[i % len(assets)] for i in range(capacity)]
            self.pool = ExternalVehiclePool(renderer, cycled, capacity=capacity,
                                            hide_original_rigid=False, hide_original_dynamic=False)
            self.slot_size = {slot: a.size_lwh for slot, a in zip(self.pool.instance_ids, cycled)}
        self._free_slots = list(self.pool.instance_ids) if self.pool else []
        self._slot_of: dict[str, int] = {}
        self.hidden = self._apply_visibility()
        # Optional per-actor edit of a *recorded* original's road-frame pose:
        # {(class, index): fn(time_s, road_pose 4x4) -> road_pose 4x4}.  Used
        # for counterfactual perturbations (e.g. AETHER-Diag) without
        # re-driving the actor; collision boxes follow the edited pose.
        self.pose_overrides: dict[tuple[str, int], Any] = {}
        self._agents: list[TrafficAgent] = []
        self._raw: dict[str, list[AgentState]] = {}
        # Per-vehicle low-pass state of height and pitch (suspension-like):
        # the reconstructed ground is noisy off the driven corridor.
        self._settle: dict[str, tuple[float, float, float]] = {}
        self.time = 0.0

    # ------------------------------------------------------------- helpers
    @staticmethod
    def _key(value) -> tuple[str, int]:
        name, _, index = str(value).partition("/")
        return name, int(index)

    def _keys(self, values) -> set[tuple[str, int]]:
        """``["RigidNodes/3", ...]``; ``"*"`` = every recorded actor, ``"RigidNodes/*"`` = one class."""
        keys: set[tuple[str, int]] = set()
        for value in values or []:
            if isinstance(value, tuple):
                keys.add((str(value[0]), int(value[1])))
                continue
            name, _, index = str(value).partition("/")
            if name == "*":
                keys |= set(self.recorded)
            elif index == "*":
                keys |= {k for k in self.recorded if k[0] == name}
            else:
                keys.add((name, int(index)))
        return keys

    def _load_recorded(self) -> dict[tuple[str, int], dict]:
        path = self.bundle.actors_path
        if path is None:
            return {}
        data = np.load(path, allow_pickle=False)
        out = {}
        for key in data.files:
            if not key.startswith("c2w/"):
                continue
            name, _, index = key[4:].partition("/")
            k = (name, int(index))
            size = data.get(f"size/{name}/{index}")
            w, l, h = (size.tolist() if size is not None else (1.85, 4.5, 1.5))
            out[k] = {"frames": data[f"frames/{name}/{index}"].astype(int),
                      "c2w": data[key].astype(np.float64), "size": (float(l), float(w), float(h))}
        return out

    def _apply_visibility(self) -> dict[str, int]:
        """Hide reconstructed dynamic actors except kept/edited ones."""
        hide_all = bool(self.cfg.get("hide_originals", True))
        keep = set(self.keep) | set(self.edited_originals.values())
        hidden = {}
        import torch
        for name, original in self._original_fv.items():
            fv = self._models[name].instances_fv
            count = original.shape[1]
            with torch.no_grad():
                if hide_all:
                    fv[:, :count] = False
                    for cls, idx in keep:
                        if cls == name and idx < count:
                            fv[:, idx] = original[:, idx]
            hidden[name] = int((original.any(0) & ~fv[:, :count].any(0)).sum().item())
        return hidden

    def set_keep(self, keys) -> dict[str, int]:
        """Change which recorded originals are shown (``"RigidNodes/3"`` or
        ``("RigidNodes", 3)``); takes effect at the next reset/advance."""
        self.keep = self._keys(keys)
        self.hidden = self._apply_visibility()
        return self.hidden

    def _assign(self, vid: str) -> int | None:
        if vid in self._slot_of:
            return self._slot_of[vid]
        if not self._free_slots:
            return None
        slot = self._free_slots.pop(0)
        self._slot_of[vid] = slot
        return slot

    def _release(self, present: set[str]) -> None:
        for vid in [v for v in self._slot_of if v not in present]:
            self._free_slots.append(self._slot_of.pop(vid))
        self._free_slots.sort()

    # ------------------------------------------------------------ lifecycle
    def reset(self, ego: AgentState, frame: int) -> None:
        scene = TrafficScene(self.bundle.id, self.dt, ego, self.bundle.hdmap,
                             self.bundle.network, self.seed, {})
        self._slot_of.clear()
        self._free_slots = list(self.pool.instance_ids) if self.pool else []
        self._raw = {}
        self._settle = {}
        if self.scripted:
            self._raw["vehicles"] = as_agents(self.scripted.reset(scene))
        if self.model:
            self._raw["model"] = as_agents(self.model.reset(scene))
        self.time = 0.0
        self._rebuild(frame)

    def advance(self, t: float, ego: AgentState, frame: int) -> None:
        if t < self.time - 1e-9:
            raise ValueError("traffic time cannot move backwards without reset")
        if abs(t - self.time) > 1e-9:
            dt = t - self.time
            if self.scripted:
                self._raw["vehicles"] = as_agents(self.scripted.step(t, dt, ego, self._raw.get("vehicles", [])))
            if self.model:
                self._raw["model"] = as_agents(self.model.step(t, dt, ego, self._raw.get("model", [])))
            self.time = t
        self._rebuild(frame)

    def close(self) -> None:
        for model in (self.scripted, self.model):
            if model is not None:
                model.close()

    # --------------------------------------------------------------- state
    def _rebuild(self, frame: int) -> None:
        agents: list[TrafficAgent] = []
        for key in sorted(self.keep):
            rec = self.recorded.get(key)
            if rec is None:
                continue
            hit = np.flatnonzero(rec["frames"] == frame)
            if hit.size == 0:
                continue
            c2w = rec["c2w"][hit[0]]
            road = self.geometry.to_road(c2w)
            edit = self.pose_overrides.get(key)
            scene_pose = None
            if edit is not None:
                road = np.asarray(edit(self.time, road), dtype=np.float64)
                scene_pose = self.geometry.to_scene(road)
            x, y, yaw = pose_from_matrix(road)
            prev = np.flatnonzero(rec["frames"] == frame - 1)
            speed = 0.0
            if prev.size:
                p = self.geometry.to_road(rec["c2w"][prev[0]])
                speed = float(np.hypot(p[0, 3] - x, p[1, 3] - y) / self.bundle.frame_dt)
            l, w, h = rec["size"]
            state = AgentState(f"{key[0]}/{key[1]}", x, y, yaw, speed, l, w, h,
                               type="car" if key[0] == "RigidNodes" else "pedestrian")
            agents.append(TrafficAgent(state, "recorded", f"original:{key[0]}/{key[1]}", scene_pose))
        present = set()
        for source in ("vehicles", "model"):
            for state in self._raw.get(source, []):
                if state.id in self.edited_originals:
                    cls, idx = self.edited_originals[state.id]
                    pose = self.geometry.to_scene(self._settled(state.id, self.geometry.object_pose(
                        state.x, state.y, state.yaw, state.length, state.width, state.height)))
                    agents.append(TrafficAgent(state, source, f"original:{cls}/{idx}", pose))
                    continue
                vid = f"{source}:{state.id}"
                present.add(vid)
                slot = self._assign(vid)
                if slot is None:
                    continue                      # pool exhausted: not rendered
                l, w, h = self.slot_size.get(slot, (state.length, state.width, state.height))
                shown = replace(state, length=l, width=w, height=h)
                pose = self.geometry.to_scene(self._settled(vid, self.geometry.object_pose(
                    shown.x, shown.y, shown.yaw, l, w, h)))
                agents.append(TrafficAgent(shown, source, f"asset:{slot}", pose))
        self._release(present)
        self._agents = agents

    def _settled(self, key: str, pose: np.ndarray, tau: float = 0.4) -> np.ndarray:
        """Low-pass height and pitch of a placed vehicle (time constant ``tau`` s)."""
        x, y, yaw = pose_from_matrix(pose)
        z, pitch = float(pose[2, 3]), math.asin(max(-1.0, min(1.0, -float(pose[2, 0]))))
        prev = self._settle.get(key)
        if prev is not None and self.time > prev[2]:
            k = 1.0 - math.exp(-(self.time - prev[2]) / tau)
            z, pitch = prev[0] + k * (z - prev[0]), prev[1] + k * (pitch - prev[1])
        elif prev is not None:
            z, pitch = prev[0], prev[1]
        self._settle[key] = (z, pitch, self.time)
        return pose2d(x, y, yaw, z, pitch, 0.0)

    def agents(self) -> list[TrafficAgent]:
        return list(self._agents)

    def render_overrides(self) -> list[tuple[str, int, np.ndarray]]:
        out = []
        for agent in self._agents:
            if agent.scene_pose is None:
                continue
            kind, _, ref = agent.binding.partition(":")
            if kind == "asset":
                out.append(("RigidNodes", int(ref), agent.scene_pose))
            else:
                cls, idx = self._key(ref)
                out.append((cls, idx, agent.scene_pose))
        return out

    def assert_hidden(self) -> None:
        """Fail if a reconstructed actor that should be hidden became visible."""
        if not bool(self.cfg.get("hide_originals", True)):
            return
        allowed = set(self.keep) | set(self.edited_originals.values())
        for name, original in self._original_fv.items():
            fv = self._models[name].instances_fv[:, :original.shape[1]]
            visible = np.flatnonzero(fv.any(0).cpu().numpy())
            stray = [int(i) for i in visible if (name, int(i)) not in allowed]
            if stray:
                raise RuntimeError(f"reconstructed {name} {stray[:5]} visible although hidden")
