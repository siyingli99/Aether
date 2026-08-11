"""HUGSIM-compatible FIFO bridge for adapter authors.

Spawns a blocking read/infer/write loop:

* Reads pickle obs from ``$output_dir/obs_pipe``.
* Sentinel ``"Done"`` (string) → exit cleanly.
* Calls ``agent.infer(obs)`` and writes the dict back to ``$output_dir/plan_pipe``.

Plan dict must contain ``plan_traj`` of shape ``(T, 2)`` in ego-local frame.
"""
from __future__ import annotations
import os
import pickle
import sys
import time
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    def __init__(self) -> None:
        self.cuda_id: str = "0"
        self.output_dir: str = ""

    def setup(self) -> None:  # noqa: B027 — optional override
        pass

    @abstractmethod
    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]: ...

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


def serve(agent: BaseAgent,
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
    obs_f = _open_fifo_read(obs_path)
    plan_f = _open_fifo_write(plan_path)
    print("[aether-adapter] pipes connected, awaiting obs", flush=True)
    try:
        while True:
            try:
                payload = pickle.load(obs_f)
            except EOFError:
                obs_f.close()
                obs_f = _open_fifo_read(obs_path, retries=200)
                continue
            if isinstance(payload, str) and payload == "Done":
                print("[aether-adapter] received Done — exiting", flush=True)
                break
            try:
                plan = agent.infer(payload)
            except Exception:
                traceback.print_exc()
                plan = {"plan_traj": [[0.0, 0.0]] * 6, "error": True}
            pickle.dump(plan, plan_f, protocol=pickle_protocol)
            plan_f.flush()
    finally:
        try:
            obs_f.close()
        except Exception:
            pass
        try:
            plan_f.close()
        except Exception:
            pass
        agent.teardown()
