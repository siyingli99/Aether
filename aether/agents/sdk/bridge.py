"""Planner-side FIFO bridge (runs inside the planner's own environment).

This module depends only on NumPy and the standard library: copy it into
any project, or put the AETHER checkout on ``PYTHONPATH``.

    from aether.agents.sdk import AdapterAgent, serve

    class MyPlanner(AdapterAgent):
        def setup(self):            # load weights once
            ...
        def infer(self, obs, info):
            # tuple protocol: obs["rgb"][cam] (H, W, 3) uint8, info as documented
            # observation protocol: obs is the Observation dict, info is None
            return waypoints          # (N, 2) as declared in the adapter YAML
            # or {"waypoints": ..., "dt": 0.5, "frame": "rear_axle", "axes": "flu"}
            # or {"accel": a, "steer_rate": r}

    if __name__ == "__main__":
        serve(MyPlanner())          # argv: <cuda_id> <workdir>
"""
from __future__ import annotations
import os
import pickle
import sys
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class AdapterAgent(ABC):
    def __init__(self) -> None:
        self.cuda_id: str = "0"
        self.output_dir: str = ""

    def setup(self) -> None:  # noqa: B027 — optional override
        pass

    @abstractmethod
    def infer(self, obs: Dict[str, Any], info: Dict[str, Any]) -> Any: ...

    def teardown(self) -> None:  # noqa: B027
        pass


def _open_fifo_read(path: str, retries: int = 600):
    for _ in range(retries):
        if os.path.exists(path):
            return open(path, "rb")
        time.sleep(0.1)
    raise FileNotFoundError(f"obs pipe never appeared: {path}")


def _open_fifo_write(path: str, retries: int = 600):
    for _ in range(retries):
        if os.path.exists(path):
            return open(path, "wb")
        time.sleep(0.1)
    raise FileNotFoundError(f"plan pipe never appeared: {path}")


def _trajectory(value: Any):
    """Validate an adapter result (waypoints array, plan dict or control dict)."""
    if isinstance(value, dict):
        if "accel" in value:
            return {"accel": float(value["accel"]), "steer_rate": float(value["steer_rate"])}
        if "waypoints" in value:
            return {**value, "waypoints": _trajectory(value["waypoints"])}
        value = value.get("plan_traj")
    plan = np.asarray(value, dtype=np.float32)
    if plan.ndim != 2 or plan.shape[0] < 1 or plan.shape[1] < 2:
        raise ValueError(f"planner must return a non-empty Nx2 path, got {plan.shape}")
    plan = plan[:, :2]
    if not np.isfinite(plan).all():
        raise ValueError("planner path contains non-finite values")
    return plan


def serve(agent: AdapterAgent,
          cuda_id: str | None = None,
          output_dir: str | None = None,
          obs_name: str = "obs_pipe",
          plan_name: str = "plan_pipe",
          pickle_protocol: int = 4) -> None:
    """Run the FIFO read/infer/write loop until the ``Done`` sentinel arrives."""
    if cuda_id is None:
        cuda_id = sys.argv[1] if len(sys.argv) > 1 else "0"
    if output_dir is None:
        output_dir = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    agent.cuda_id = str(cuda_id)
    agent.output_dir = os.path.abspath(output_dir)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    print(f"[aether-adapter] cuda={cuda_id} output={agent.output_dir}",
          flush=True)
    agent.setup()
    obs_path = os.path.join(agent.output_dir, obs_name)
    plan_path = os.path.join(agent.output_dir, plan_name)
    print("[aether-adapter] ready; waiting for obs", flush=True)
    try:
        while True:
            # AETHER opens and closes both FIFOs once per frame. Matching that
            # lifecycle lets the reader use ``read()`` without waiting forever
            # for an EOF from a persistent writer.
            with _open_fifo_read(obs_path) as obs_f:
                payload = pickle.load(obs_f)
            if isinstance(payload, str) and payload == "Done":
                print("[aether-adapter] received Done — exiting", flush=True)
                break
            try:
                if isinstance(payload, dict) and payload.get("type") == "step":
                    obs, info = payload["observation"], None
                elif isinstance(payload, tuple) and len(payload) == 2:
                    obs, info = payload
                else:
                    raise ValueError("unexpected payload from AETHER")
                plan = _trajectory(agent.infer(obs, info))
            except Exception:
                traceback.print_exc()
                plan = None
            with _open_fifo_write(plan_path) as plan_f:
                pickle.dump(plan, plan_f, protocol=pickle_protocol)
    finally:
        agent.teardown()


BaseAgent = AdapterAgent       # old name
