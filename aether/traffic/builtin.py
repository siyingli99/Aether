"""Traffic models shipped with AETHER.

* :class:`ExternalSimulator` — any simulator speaking the JSON-over-TCP
  protocol in :mod:`aether.traffic.protocol` (LimSim, SUMO/TraCI bridges,
  TESS NG, your own C++/MATLAB code).
* :class:`ReplayTraffic` — recorded frames from a JSON file.
* :class:`IDMLanes` — Intelligent Driver Model on the HD-map lane graph, a
  self-contained example of a microscopic traffic-flow model.
* :class:`ScriptedVehicles` — per-vehicle open-loop trajectories and
  :class:`~aether.traffic.model.VehicleController` plugins (used by Studio).
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .client import LimSimTrafficClient, ReplayTrafficClient, SocketTrafficClient
from .model import AgentState, TrafficModel, TrafficScene, VehicleController, load_class
from .protocol import EgoState, TrafficFrame


def _ego_message(ego: AgentState) -> EgoState:
    return EgoState(x=ego.x, y=ego.y, yaw=ego.yaw, speed=ego.speed,
                    length=ego.length, width=ego.width, height=ego.height)


def _from_frame(frame: TrafficFrame) -> list[AgentState]:
    return [AgentState(id=v.vehicle_id, x=v.x, y=v.y, yaw=v.yaw, speed=v.speed,
                       length=v.length, width=v.width, height=v.height, type=v.vehicle_type)
            for v in frame.vehicles]


def _port_open(host: str, port: int) -> bool:
    import socket
    try:
        with socket.create_connection((host, port), 0.3):
            return True
    except OSError:
        return False


def _default_limsim_python() -> str:
    import os
    import shutil
    from pathlib import Path
    if os.environ.get("LIMSIM_PYTHON"):
        return os.environ["LIMSIM_PYTHON"]
    for root in ("~/miniconda3", "~/anaconda3", "~/mambaforge", "/opt/conda"):
        candidate = Path(root).expanduser() / "envs" / "limsim" / "bin" / "python"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("python") or "python"


class ExternalSimulator(TrafficModel):
    """Background traffic from an external simulator process (TCP sidecar).

    With ``launch: true`` (default for LimSim) AETHER starts the LimSim bridge
    itself on the scene's SUMO network (``map/network.net.xml`` +
    ``map/routes.rou.xml``) and stops it afterwards.  LimSim is looked up in
    ``$LIMSIM_ROOT`` (default ``$AETHER_EXTERNAL/LimSim``) and run with
    ``$LIMSIM_PYTHON`` (default: the ``limsim`` conda env).  If something
    already listens on ``host:port`` it is used as is.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8766, timeout_s: float = 30.0,
                 server: str = "limsim", launch: bool | None = None,
                 startup_timeout_s: float = 60.0, **params: Any) -> None:
        super().__init__(**params)
        self.host, self.port, self.server = host, int(port), server
        self.launch = (server == "limsim") if launch is None else bool(launch)
        self.startup_timeout_s = float(startup_timeout_s)
        self.process = None
        if server == "limsim":
            self.client = LimSimTrafficClient(host, self.port, float(timeout_s))
        else:
            self.client = SocketTrafficClient(host, self.port, float(timeout_s), expected_server=server)

    def _start_limsim(self, scene: TrafficScene) -> None:
        import os
        import subprocess
        import time
        from pathlib import Path
        from ..paths import AETHER_ROOT, EXTERNAL_ROOT, OUTPUTS_ROOT
        if _port_open(self.host, self.port):
            return
        if scene.network is None:
            raise FileNotFoundError(f"scene {scene.scene_id} has no SUMO network (map/network.net.xml)")
        root = Path(os.environ.get("LIMSIM_ROOT", EXTERNAL_ROOT / "LimSim")).expanduser()
        server = root / "aether_bridge" / "server.py"
        if not server.is_file():
            raise FileNotFoundError(f"LimSim AETHER bridge not found at {server}; set LIMSIM_ROOT")
        net, routes = scene.network
        python = _default_limsim_python()
        log_dir = OUTPUTS_ROOT / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = log_dir / f"limsim-{self.port}.log"
        env = dict(os.environ, AETHER_ROOT=str(AETHER_ROOT), PYTHONNOUSERSITE="1")
        env["PATH"] = f"{Path(python).parent}{os.pathsep}{env.get('PATH', '')}"
        with open(self.log, "w") as handle:
            self.process = subprocess.Popen(
                [python, "-u", str(server), "--host", self.host, "--port", str(self.port),
                 "--net-file", str(net), "--route-file", str(routes)],
                stdout=handle, stderr=subprocess.STDOUT, cwd=str(root), env=env)
        deadline = time.monotonic() + self.startup_timeout_s
        while not _port_open(self.host, self.port):
            if self.process.poll() is not None:
                raise RuntimeError(f"LimSim exited (code {self.process.returncode}); see {self.log}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"LimSim did not open port {self.port}; see {self.log}")
            time.sleep(0.25)

    def reset(self, scene: TrafficScene) -> list[AgentState]:
        if self.launch and self.server == "limsim":
            self._start_limsim(scene)
        self.client.reset()
        # The bridge spawns vehicles on its first callback: a zero-time tick
        # synchronises the initial frame before AETHER renders frame 0.
        return _from_frame(self.client.tick(0.0, _ego_message(scene.ego)))

    def step(self, t, dt, ego, vehicles):
        return _from_frame(self.client.tick(float(t), _ego_message(ego)))

    def close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            try:
                self.client.shutdown()
            except Exception:
                pass
        self.client.close()
        if self.process is not None:
            try:
                self.process.wait(timeout=10)
            except Exception:
                self.process.kill()
            self.process = None


class ReplayTraffic(TrafficModel):
    def __init__(self, file: str, **params: Any) -> None:
        super().__init__(**params)
        self.client = ReplayTrafficClient.from_json(file)

    def reset(self, scene):
        return _from_frame(self.client.tick(0.0, _ego_message(scene.ego)))

    def step(self, t, dt, ego, vehicles):
        return _from_frame(self.client.tick(float(t), _ego_message(ego)))


# --------------------------------------------------------------------------- #
# IDM on the HD-map lane graph
# --------------------------------------------------------------------------- #
@dataclass
class _Path:
    points: np.ndarray        # (N, 2) concatenated centrelines
    cum: np.ndarray           # (N,) arc length
    lanes: list[str]

    def pose(self, s: float) -> tuple[float, float, float]:
        s = float(np.clip(s, 0.0, self.cum[-1]))
        i = int(np.clip(np.searchsorted(self.cum, s) - 1, 0, len(self.points) - 2))
        seg = self.points[i + 1] - self.points[i]
        span = max(self.cum[i + 1] - self.cum[i], 1e-9)
        p = self.points[i] + seg * (s - self.cum[i]) / span
        return float(p[0]), float(p[1]), math.atan2(seg[1], seg[0])

    def project(self, x: float, y: float) -> tuple[float, float]:
        d = np.hypot(self.points[:, 0] - x, self.points[:, 1] - y)
        i = int(np.argmin(d))
        return float(self.cum[i]), float(d[i])


class IDMLanes(TrafficModel):
    """IDM car-following on lane paths of the HD map.

    Parameters (scenario ``traffic.model.params``):

    ``vehicles``: list of ``{id, lane, s, speed, desired_speed}`` (``lane`` is an
        HD-map lane id, ``s`` the arc length on it in metres); or
    ``count``: number of vehicles spawned on random lanes (``seed`` from the run);
    ``desired_speed`` (m/s, default 10), ``time_headway`` (1.5 s),
    ``min_gap`` (2 m), ``max_accel`` (1.5 m/s²), ``comfort_decel`` (2 m/s²),
    ``length``/``width``/``height`` of spawned cars.

    The ego is treated as a potential leader when it is on a vehicle's path.
    """

    def __init__(self, vehicles: list[dict] | None = None, count: int = 0,
                 desired_speed: float = 10.0, time_headway: float = 1.5,
                 min_gap: float = 2.0, max_accel: float = 1.5, comfort_decel: float = 2.0,
                 length: float = 4.5, width: float = 1.85, height: float = 1.5,
                 path_lanes: int = 6, **params: Any) -> None:
        super().__init__(**params)
        self.spec = list(vehicles or [])
        self.count = int(count)
        self.v0, self.T, self.s0 = float(desired_speed), float(time_headway), float(min_gap)
        self.a, self.b = float(max_accel), float(comfort_decel)
        self.size = (float(length), float(width), float(height))
        self.path_lanes = int(path_lanes)

    def _path(self, lane_id: str, rng: np.random.Generator) -> _Path:
        lanes, points = [lane_id], [self.map.by_id[lane_id].centerline]
        current = lane_id
        for _ in range(self.path_lanes - 1):
            nxt = [t for t in self.map.by_id[current].outgoing if t in self.map.by_id]
            if not nxt:
                break
            current = nxt[int(rng.integers(len(nxt)))]
            lanes.append(current)
            points.append(self.map.by_id[current].centerline[1:])
        pts = np.concatenate(points)
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
        return _Path(pts, cum, lanes)

    def reset(self, scene: TrafficScene) -> list[AgentState]:
        if scene.hdmap is None or not scene.hdmap.lanes:
            raise ValueError("IDMLanes needs a scene HD map (map/hdmap.json)")
        self.map = scene.hdmap
        rng = np.random.default_rng(scene.seed)
        self.state: dict[str, dict] = {}
        spec = list(self.spec)
        lanes = [lane.id for lane in self.map.lanes if lane.kind == "lane"] or [l.id for l in self.map.lanes]
        for k in range(self.count):
            lane = lanes[int(rng.integers(len(lanes)))]
            spec.append({"id": f"idm-{k}", "lane": lane,
                         "s": float(rng.uniform(0, self.map.by_id[lane].length)),
                         "speed": float(rng.uniform(0.5, 1.0) * self.v0)})
        for item in spec:
            vid, lane = str(item["id"]), str(item["lane"])
            if lane not in self.map.by_id:
                raise KeyError(f"IDMLanes: unknown lane {lane!r}")
            self.state[vid] = {"path": self._path(lane, rng), "s": float(item.get("s", 0.0)),
                               "v": float(item.get("speed", self.v0)),
                               "v0": float(item.get("desired_speed", self.v0)), "a": 0.0}
        return self._agents()

    def _agents(self) -> list[AgentState]:
        out = []
        for vid, st in self.state.items():
            if st["s"] >= st["path"].cum[-1]:
                continue                                  # left the mapped area
            x, y, yaw = st["path"].pose(st["s"])
            l, w, h = self.size
            out.append(AgentState(vid, x, y, yaw, st["v"], l, w, h, st["a"]))
        return out

    def step(self, t, dt, ego, vehicles):
        active = {a.id for a in vehicles}
        gaps = {}
        for vid, st in self.state.items():
            if vid not in active:
                continue
            best = None
            for other, ot in self.state.items():
                if other == vid or other not in active:
                    continue
                s_o, lat = st["path"].project(*ot["path"].pose(ot["s"])[:2])
                if lat < 1.5 and s_o > st["s"]:
                    gap = s_o - st["s"] - self.size[0]
                    if best is None or gap < best[0]:
                        best = (gap, ot["v"])
            s_e, lat = st["path"].project(ego.x, ego.y)
            if lat < 1.5 and s_e > st["s"]:
                gap = s_e - st["s"] - 0.5 * (self.size[0] + ego.length)
                if best is None or gap < best[0]:
                    best = (gap, ego.speed)
            gaps[vid] = best
        for vid, best in gaps.items():
            st = self.state[vid]
            v = st["v"]
            acc = self.a * (1 - (v / max(st["v0"], 0.1)) ** 4)
            if best is not None:
                gap, v_lead = best
                s_star = self.s0 + max(0.0, v * self.T + v * (v - v_lead) / (2 * math.sqrt(self.a * self.b)))
                acc -= self.a * (s_star / max(gap, 0.1)) ** 2
            acc = max(acc, -9.0)
            st["v"] = max(v + acc * dt, 0.0)
            st["s"] += st["v"] * dt
            st["a"] = acc
        return self._agents()


# --------------------------------------------------------------------------- #
# Per-vehicle scripted control (Studio)
# --------------------------------------------------------------------------- #
def _interpolate(points: list[dict], t: float) -> tuple[float, float, float, float]:
    times = [float(p["t"]) for p in points]
    if t <= times[0]:
        p = points[0]
        return p["x"], p["y"], p.get("yaw", 0.0), 0.0
    if t >= times[-1]:
        p = points[-1]
        return p["x"], p["y"], p.get("yaw", 0.0), 0.0
    j = int(np.searchsorted(times, t, side="right"))
    a, b = points[j - 1], points[j]
    alpha = (t - times[j - 1]) / max(times[j] - times[j - 1], 1e-9)
    dyaw = (b.get("yaw", 0.0) - a.get("yaw", 0.0) + math.pi) % (2 * math.pi) - math.pi
    speed = math.hypot(b["x"] - a["x"], b["y"] - a["y"]) / max(times[j] - times[j - 1], 1e-9)
    return (a["x"] + alpha * (b["x"] - a["x"]), a["y"] + alpha * (b["y"] - a["y"]),
            a.get("yaw", 0.0) + alpha * dyaw, speed)


class ScriptedVehicles(TrafficModel):
    """Vehicles driven by open-loop trajectories or per-vehicle plugins.

    ``vehicles``: list of ``{id, size: [l, w, h], start: {x, y, yaw},
    controller: {type: open_loop, trajectory: [{t, x, y, yaw}, ...]}
    | {type: plugin, class: pkg.mod:Class, params: {...}}
    | {type: static}}``.
    """

    def __init__(self, vehicles: list[dict] | None = None, **params: Any) -> None:
        super().__init__(**params)
        self.spec = list(vehicles or [])

    def reset(self, scene: TrafficScene) -> list[AgentState]:
        self.plugins: dict[str, VehicleController] = {}
        for item in self.spec:
            ctrl = item.get("controller") or {"type": "static"}
            if ctrl.get("type") == "plugin":
                plugin = load_class(ctrl["class"])()
                if hasattr(plugin, "setup"):
                    plugin.setup({"vehicle": item, "params": ctrl.get("params", {}), "scene": scene})
                if hasattr(plugin, "reset"):
                    plugin.reset()
                self.plugins[str(item["id"])] = plugin
        return self._at(0.0, scene.ego)

    def _at(self, t: float, ego: AgentState) -> list[AgentState]:
        out = []
        for item in self.spec:
            vid = str(item["id"])
            l, w, h = (float(v) for v in item.get("size", (4.5, 1.85, 1.5)))
            start = item.get("start") or {}
            ctrl = item.get("controller") or {"type": "static"}
            kind = ctrl.get("type", "static")
            if kind == "open_loop" and ctrl.get("trajectory"):
                x, y, yaw, speed = _interpolate(ctrl["trajectory"], t)
            elif kind == "plugin":
                pose = self.plugins[vid].step(t, item, ego)
                pose = pose if isinstance(pose, dict) else pose.__dict__
                x, y, yaw = float(pose["x"]), float(pose["y"]), float(pose.get("yaw", 0.0))
                speed = float(pose.get("speed", 0.0))
            else:
                x, y, yaw, speed = start.get("x", 0.0), start.get("y", 0.0), start.get("yaw", 0.0), 0.0
            out.append(AgentState(vid, float(x), float(y), float(yaw), float(speed), l, w, h))
        return out

    def step(self, t, dt, ego, vehicles):
        return self._at(float(t), ego)

    def close(self):
        for plugin in getattr(self, "plugins", {}).values():
            if hasattr(plugin, "close"):
                plugin.close()


class ConstantVelocity(VehicleController):
    """Reference per-vehicle plugin: straight ahead at ``speed_mps``."""

    def setup(self, context):
        start = context["vehicle"].get("start") or {}
        self.x0, self.y0 = float(start.get("x", 0.0)), float(start.get("y", 0.0))
        self.yaw = float(start.get("yaw", 0.0))
        self.speed = float((context.get("params") or {}).get("speed_mps", 5.0))

    def step(self, time_s, actor, ego):
        d = self.speed * time_s
        return {"x": self.x0 + d * math.cos(self.yaw), "y": self.y0 + d * math.sin(self.yaw),
                "yaw": self.yaw, "speed": self.speed}
