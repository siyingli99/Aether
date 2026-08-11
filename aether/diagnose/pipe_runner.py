"""Single closed-loop rollout used by PipeRollout."""
from __future__ import annotations

import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .semantics import SemanticPerturbation, perturb_pose


def _traj_to_action(plan_traj, info):
    
    from aether.control.ilqr.lqr import plan2control
    plan_traj = np.asarray(plan_traj)
    plan_traj_stats = np.zeros((plan_traj.shape[0] + 1, 5))
    plan_traj_stats[1:, :2] = plan_traj[:, [1, 0]]
    prev_a, prev_b = 0.0, 0.0
    for i, (a, b) in enumerate(plan_traj):
        da = (a - prev_a) if (a - prev_a) != 0 else 1e-6
        plan_traj_stats[i + 1, 2] = np.arctan((b - prev_b) / da)
    curr_stat = np.array([0, 0, 0, info["ego_velo"], info["ego_steer"]])
    return plan2control(plan_traj_stats, curr_stat)


def _maybe_perturb_target_pose(env, target, perturbation, t_step: int):
    cls, inst = target
    if cls is None or inst is None:
        return
    base = None
    for entry in env.actors.pose_at(env.frame_idx):
        if entry["class"] == cls and entry["instance_id"] == int(inst):
            base = entry; break
    if base is None:
        return
    new_c2w = perturb_pose(base["c2w"], perturbation,
                           t=float(t_step) * env.ego.dt, v_lon0=0.0)
    env.renderer.set_actor_pose(cls, int(inst), env.frame_idx, new_c2w)


def _decode_plan_response(raw):
    """Accept legacy ndarray or v2 dict {plan_xy, agent_states?, agent_labels?}."""
    if isinstance(raw, dict):
        plan = raw.get("plan_xy")
        if plan is None:
            plan = raw.get("trajectory")
        return plan, raw
    return raw, None


def run_one_rollout(env, obs_pipe, plan_pipe, perturbation, target,
                    max_steps=None):
    obs, info = env.reset()
    trace = []
    cnt = 0
    done = False
    while not done:
        if max_steps is not None and cnt >= int(max_steps):
            break
        _maybe_perturb_target_pose(env, target, perturbation, cnt)
        with open(obs_pipe, "wb") as p:
            p.write(pickle.dumps((obs, info), protocol=4))
        with open(plan_pipe, "rb") as p:
            raw = pickle.loads(p.read())
        if raw is None:
            break
        plan_traj, response = _decode_plan_response(raw)
        if plan_traj is None:
            break
        acc, steer_rate = _traj_to_action(plan_traj, info)
        obs, _, terminated, truncated, info = env.step(
            {"acc": acc, "steer_rate": steer_rate}
        )
        step = {
            "ego_box": info["ego_box"], "obj_boxes": info["obj_boxes"],
            "ego_velo": float(info.get("ego_velo", 0.0)),
            "accelerate": float(info.get("accelerate", 0.0)),
            "steer_rate": float(steer_rate),
            "plan_traj": np.asarray(plan_traj),
            "rc": float(info.get("rc", 0.0)),
            "timestamp": float(info.get("timestamp", 0.0)),
        }
        if response is not None:
            ag_states = response.get("agent_states")
            ag_labels = response.get("agent_labels")
            if ag_states is not None:
                step["pred_boxes"] = np.asarray(ag_states).tolist()
            if ag_labels is not None:
                step["pred_labels"] = list(ag_labels)
        trace.append(step)
        cnt += 1
        done = terminated or truncated
    return trace
