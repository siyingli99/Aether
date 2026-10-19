"""External planners running in their own process / conda environment.

An adapter is described by ``configs/agents/<name>.yaml``::

    name: uniad
    title: UniAD
    command: [bash, "${AETHER_ROOT}/third_party/uniad_sim/tools/e2e.sh", "${cuda}", "${workdir}"]
    env: {PYTHONUNBUFFERED: "1"}
    ready: ["ready for"]          # log text printed once the model is loaded
    protocol: tuple              # tuple: (sensors, info) pickle | observation: Observation dict
    input:
      cameras: [CAM_FRONT, ...]
      image_size: null           # [W, H]; null = rendered size (800x450 for nuScenes)
      lidar: false
    output:
      dt: 0.5                    # seconds between waypoints
      frame: lidar               # rear_axle | center | lidar
      axes: rfu                  # rfu = (right, forward); flu = (forward, left)

``${cuda}``, ``${workdir}``, ``${AETHER_ROOT}``, ``${AETHER_EXTERNAL}`` and any
environment variable are substituted.  ``workdir`` is ``<run>/agent``; the
two FIFOs ``obs_pipe`` / ``plan_pipe`` and ``planner.log`` live there.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..paths import AETHER_ROOT, EXTERNAL_ROOT, REPO_ROOT
from . import ipc
from .base import Control, EgoAgent, Observation, Plan
from .observation import tuple_payload

AGENT_DIR = REPO_ROOT / "configs" / "agents"
# Names used by AETHER < 0.2 configs.
_PROTOCOL_ALIASES = {"legacy": "tuple", "aether": "observation"}


@dataclass
class AdapterSpec:
    name: str
    title: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    ready: list[str] = field(default_factory=lambda: ["ready"])
    protocol: str = "tuple"
    cameras: list[str] | None = None
    image_size: tuple[int, int] | None = None
    lidar: bool = False
    dt: float = 0.5
    frame: str = "rear_axle"
    axes: str = "rfu"
    startup_timeout_s: float = 900.0
    step_timeout_s: float = 900.0
    description: str = ""
    source: str | None = None


def _expand(value: str, extra: dict[str, str]) -> str:
    for key, sub in extra.items():
        value = value.replace("${" + key + "}", sub)
    return os.path.expandvars(os.path.expanduser(value))


def list_adapters(directory: Path = AGENT_DIR) -> list[str]:
    return sorted(p.stem for p in Path(directory).glob("*.yaml"))


def load_adapter(name_or_path: str | Path) -> AdapterSpec:
    path = Path(name_or_path)
    if not path.suffix:
        path = AGENT_DIR / f"{name_or_path}.yaml"
    data = yaml.safe_load(path.read_text()) or {}
    inp, out = data.get("input") or {}, data.get("output") or {}
    size = inp.get("image_size")
    return AdapterSpec(
        name=str(data.get("name", path.stem)), title=str(data.get("title", path.stem)),
        command=[str(c) for c in data["command"]],
        env={str(k): str(v) for k, v in (data.get("env") or {}).items()},
        cwd=data.get("cwd"), ready=[str(m) for m in data.get("ready", ["ready"])],
        protocol=_PROTOCOL_ALIASES.get(str(data.get("protocol", "tuple")), str(data.get("protocol", "tuple"))),
        cameras=list(inp["cameras"]) if inp.get("cameras") else None,
        image_size=tuple(int(v) for v in size) if size else None,
        lidar=bool(inp.get("lidar", False)),
        dt=float(out.get("dt", 0.5)), frame=str(out.get("frame", "rear_axle")),
        axes=str(out.get("axes", "rfu")),
        startup_timeout_s=float(data.get("startup_timeout_s", 900)),
        step_timeout_s=float(data.get("step_timeout_s", 900)),
        description=str(data.get("description", "")), source=str(path))


class ExternalAgent(EgoAgent):
    """Runs an adapter process and exchanges one message pair per step."""

    def __init__(self, spec: AdapterSpec, workdir: str | Path, cuda: str = "0") -> None:
        self.spec = spec
        self.workdir = Path(workdir)
        self.cuda = str(cuda)
        self.cameras = spec.cameras
        self.image_size = spec.image_size
        self.needs_lidar = spec.lidar
        self.proc: subprocess.Popen | None = None
        self.log = self.workdir / "planner.log"
        self.lidar_mount = np.eye(4)

    # ------------------------------------------------------------ process
    def setup(self, context: dict[str, Any]) -> None:
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.lidar_mount = np.asarray(context.get("lidar_mount", np.eye(4)), dtype=float)
        self.obs_pipe, self.plan_pipe = ipc.make_pipes(self.workdir)
        subs = {"cuda": self.cuda, "workdir": str(self.workdir.resolve()),
                "AETHER_ROOT": str(AETHER_ROOT), "AETHER_EXTERNAL": str(EXTERNAL_ROOT)}
        cmd = [_expand(c, subs) for c in self.spec.command]
        env = dict(os.environ, AETHER_ROOT=str(AETHER_ROOT), AETHER_EXTERNAL=str(EXTERNAL_ROOT))
        env.update({k: _expand(v, subs) for k, v in self.spec.env.items()})
        cwd = _expand(self.spec.cwd, subs) if self.spec.cwd else None
        with open(self.log, "w") as handle:
            self.proc = subprocess.Popen(cmd, stdout=handle, stderr=subprocess.STDOUT,
                                         cwd=cwd, env=env, start_new_session=True)
        ipc.wait_ready(self.proc, self.log, self.spec.ready, self.spec.startup_timeout_s)

    def _exchange(self, message: Any) -> Any:
        ipc.send(self.obs_pipe, pickle.dumps(message, protocol=4), self.proc, self.log,
                 self.spec.step_timeout_s)
        raw = ipc.receive(self.plan_pipe, self.proc, self.log, self.spec.step_timeout_s)
        return pickle.loads(raw) if raw else None

    # -------------------------------------------------------------- agent
    def act(self, obs: Observation):
        if self.spec.protocol == "tuple":
            reply = self._exchange(tuple_payload(obs, self.lidar_mount))
        else:
            reply = self._exchange({"type": "step", "observation": obs.__dict__})
        return self.decode(reply)

    def decode(self, reply: Any):
        if reply is None:
            return None
        if isinstance(reply, dict):
            if "accel" in reply:
                return Control(float(reply["accel"]), float(reply["steer_rate"]))
            return Plan(np.asarray(reply["waypoints"]), float(reply.get("dt", self.spec.dt)),
                        str(reply.get("frame", self.spec.frame)), str(reply.get("axes", self.spec.axes)))
        return Plan(np.asarray(reply, dtype=float)[:, :2], self.spec.dt, self.spec.frame, self.spec.axes)

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.poll() is None:
                ipc.send(self.obs_pipe, pickle.dumps("Done", protocol=4), self.proc, self.log, 30.0)
                self.proc.wait(timeout=30)
        except Exception:
            pass
        finally:
            if self.proc.poll() is None:
                import signal
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                    self.proc.wait(timeout=10)
                except Exception:
                    self.proc.kill()
            self.proc = None
