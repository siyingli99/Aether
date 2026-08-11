"""Rollout backends used by the diagnosis engine."""
from __future__ import annotations

import math
import os
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .semantics import SemanticPerturbation, perturb_pose, TOPO_DIMS, PHOTO_DIMS


class Rollout(ABC):
    actor_pool: List[Tuple[str, int]]
    scene_id: str

    @abstractmethod
    def __call__(self, actor_keep: Optional[Sequence[int]] = None,
                 perturbation: Optional[SemanticPerturbation] = None,
                 max_steps: Optional[int] = None) -> List[Dict[str, Any]]:
        ...

    def close(self) -> None:
        pass


# --- topo / photo amplification weights -- per-dim relative magnitude ----
_TOPO_W = np.array([0.20, 0.30, 0.50, 0.60, 0.20, 0.40], dtype=np.float64)  # 6 dims
_PHOTO_W = np.array([0.40, 0.40, 0.30, 0.30, 0.50, 0.40, 0.45], dtype=np.float64)  # 7 dims


@dataclass
class MockRollout(Rollout):
    """Synthetic rollout exposing 13-D semantic sensitivity for tests."""
    scene_id: str = "mock_scene"
    n_steps: int = 24
    actor_pool: List[Tuple[str, int]] = field(default_factory=lambda: [
        ("RigidNodes", 7), ("RigidNodes", 11), ("SMPLNodes", 3),
    ])
    danger_idx: int = 0
    photo_sensitivity: float = 1.0
    topo_sensitivity: float = 1.0

    def _make_ego(self, t: int) -> List[float]:
        return [0.0, 8.0 * 0.25 * t, 0.0, 1.6, 4.5, 1.5, 0.0]

    def _make_actor(self, idx: int, t: int) -> List[float]:
        if idx == self.danger_idx:
            y = 8.0 * 0.25 * t + max(0.0, 25.0 - 0.6 * t)
            x = 0.0
        else:
            y = 8.0 * 0.25 * t + 60.0 + 5.0 * idx
            x = 6.0 * (idx - 0.5)
        return [x, y, 0.0, 1.6, 4.5, 1.5, 0.0]

    def _topo_amp(self, z) -> float:
        # all 6 topo dims contribute proportionally via |z_i - neutral|/range
        from .semantics import SEMANTIC_VAR_NAMES, SEMANTIC_NEUTRAL, SEMANTIC_BOUNDS
        amp = 0.0
        for k, i in enumerate(TOPO_DIMS):
            n = SEMANTIC_VAR_NAMES[i]
            lo, hi = SEMANTIC_BOUNDS[n]
            neutral = SEMANTIC_NEUTRAL[n]
            d = abs(float(z.z[i]) - neutral) / max(hi - lo, 1e-9)
            amp += _TOPO_W[k] * d
        return amp * self.topo_sensitivity

    def _photo_amp(self, z) -> float:
        from .semantics import SEMANTIC_VAR_NAMES, SEMANTIC_NEUTRAL, SEMANTIC_BOUNDS
        amp = 0.0
        for k, i in enumerate(PHOTO_DIMS):
            n = SEMANTIC_VAR_NAMES[i]
            lo, hi = SEMANTIC_BOUNDS[n]
            neutral = SEMANTIC_NEUTRAL[n]
            d = abs(float(z.z[i]) - neutral) / max(hi - lo, 1e-9)
            amp += _PHOTO_W[k] * d
        return amp * self.photo_sensitivity

    def __call__(self, actor_keep=None, perturbation=None, max_steps=None):
        keep = list(range(len(self.actor_pool))) if actor_keep is None else list(actor_keep)
        z = perturbation if perturbation is not None else SemanticPerturbation.zero()
        n = self.n_steps if max_steps is None else min(self.n_steps, int(max_steps))

        topo_amp = self._topo_amp(z)
        photo_amp = self._photo_amp(z)

        trace: List[Dict[str, Any]] = []
        for t in range(n):
            ego = self._make_ego(t)
            objs = []
            for i in keep:
                box = self._make_actor(i, t)
                if i == self.danger_idx:
                    # apply some pose-equivalent of topological deltas
                    box[1] += float(z.z[0]) * 0.25       # delta_s_lon
                    box[0] += float(z.z[1]) * 0.25       # delta_d_lat
                    box[6] += float(z.z[2])              # delta_yaw
                    box[3] *= float(z.z[3])              # scale -> width
                    box[4] *= float(z.z[3])              # scale -> length
                objs.append(box)

            danger_present = self.danger_idx in keep
            lateral = 0.0
            if danger_present:
                # baseline lateral wobble + amplification by perturbation
                lateral = 0.4 + topo_amp + photo_amp
            plan = np.array([
                [lateral * 0.2 * (k + 1), 2.0 * (k + 1)] for k in range(6)
            ], dtype=np.float64)

            acc = -0.5 if danger_present else 0.0
            acc += float(z.z[4]) * 0.2     # delta_v_lon -> jerky accel
            acc += float(z.z[5]) * 0.1     # omega_yaw

            rc = (t + 1) / max(self.n_steps, 1)
            trace.append({
                "ego_box": ego, "obj_boxes": objs,
                "ego_velo": 8.0, "accelerate": acc,
                "plan_traj": plan, "rc": rc,
                "timestamp": 0.25 * t,
            })
        return trace


# ----------------------- PipeRollout (real) ------------------------------
@dataclass
class PipeRollout(Rollout):
    scene_id: str
    cfg: Any
    output_dir: str
    ad_shell: str
    ad_cuda: str = "0"
    target_class: Optional[str] = None
    target_instance: Optional[int] = None
    max_actors: int = 6

    def __post_init__(self):
        from aether.envs.driving_env import AetherDrivingEnv
        from aether.agents.launcher import launch, make_pipes
        from aether.control import ForegroundController, TrainedActorAdapter
        os.makedirs(self.output_dir, exist_ok=True)
        self.env = AetherDrivingEnv(self.cfg, self.output_dir)
        self.controller = ForegroundController(self.env.renderer)

        cfg_actors = list(getattr(self.cfg.scenario, "actors", []) or [])
        decl_pool = [(a["class"], int(a["instance_id"])) for a in cfg_actors]
        trained = self.controller.list_instances()
        trained.sort(key=lambda x: x.n_visible_frames, reverse=True)
        merged: List[Tuple[str, int]] = list(decl_pool)
        for inst in trained:
            if inst.key in merged:
                continue
            merged.append(inst.key)
            if len(merged) >= int(self.max_actors):
                break
        self.actor_pool = merged[: int(self.max_actors)]

        if not decl_pool and self.actor_pool:
            self.env.actors = TrainedActorAdapter(
                self.controller, only=list(self.actor_pool), inject_pose=False,
            )

        self._obs_pipe, self._plan_pipe = make_pipes(self.output_dir)
        self._proc = launch(self.ad_shell, self.ad_cuda,
                            os.path.abspath(self.output_dir))

        self._handle = None
        if self.target_class is not None and self.target_instance is not None:
            from .semantics import TargetGaussianHandle
            self._handle = TargetGaussianHandle(
                self.env.renderer, self.target_class, int(self.target_instance),
            )

    def __call__(self, actor_keep=None, perturbation=None, max_steps=None):
        from .pipe_runner import run_one_rollout
        z = perturbation if perturbation is not None else SemanticPerturbation.zero()
        keep_keys = None
        if actor_keep is not None:
            keep_keys = [self.actor_pool[i] for i in actor_keep]
        if hasattr(self.env, "actors") and hasattr(self.env.actors, "set_keep"):
            self.env.actors.set_keep(keep_keys)
        target = (self.target_class, self.target_instance)
        if self._handle is not None:
            with self._handle.apply_photometric(z):
                trace = run_one_rollout(self.env, self._obs_pipe, self._plan_pipe,
                                         z, target, max_steps=max_steps)
        else:
            trace = run_one_rollout(self.env, self._obs_pipe, self._plan_pipe,
                                     z, target, max_steps=max_steps)
        return trace

    def close(self) -> None:
        try:
            with open(self._obs_pipe, "wb") as p:
                p.write(pickle.dumps("Done", protocol=4))
        except Exception:
            pass
