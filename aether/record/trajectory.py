"""Per-vehicle trajectory log in a traffic-engineering table format.

``trajectories.csv`` – one row per vehicle per simulation step;
``trajectories_<hz>hz.csv`` – the same, resampled on a uniform clock (default
10 Hz, NGSIM/highD-style) with kinematic interpolation.

Columns (SI units, road frame unless stated)::

    time_s, step, vehicle_id, role, type, source,
    x, y, z, yaw, speed, vx, vy, accel, yaw_rate, length, width, height,
    lane_id, s, d, x_global, y_global, yaw_global

``x, y`` is the vehicle *centre* on the ground (x along the initial ego
heading, y to the left).  ``s, d`` are the arc length along and the signed
lateral offset (left +) from the matched lane centreline.  ``*_global`` are
dataset map coordinates (e.g. nuScenes map frame), handy for joining with
the original HD map in SUMO/VISSIM/LimSim.

``trajectories_fcd.xml`` – the resampled table in SUMO's floating-car-data
format (``<fcd-export><timestep><vehicle id x y angle speed .../>``).  Its
x/y are road-frame coordinates, which are also the coordinates of the
scene's ``map/network.net.xml``, so ``sumo-gui -n network.net.xml`` can
replay a run and SUMO/LimSim/traffic-analysis tools read it unchanged.
``angle`` follows SUMO: degrees, 0 = +y (north), clockwise.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from .interp import interpolate_state

COLUMNS = ["time_s", "step", "vehicle_id", "role", "type", "source", "x", "y", "z", "yaw",
           "speed", "vx", "vy", "accel", "yaw_rate", "length", "width", "height",
           "lane_id", "s", "d", "x_global", "y_global", "yaw_global"]


class TrajectoryRecorder:
    def __init__(self, hdmap=None, ground=None) -> None:
        self.hdmap, self.ground = hdmap, ground
        self.steps: list[tuple[float, int, dict[str, dict]]] = []   # (time, step, {id: state})
        self._prev: dict[str, dict] = {}

    # ----------------------------------------------------------- capture
    @staticmethod
    def _states(info: dict[str, Any]) -> dict[str, dict]:
        ego = info["ego"]
        out = {"ego": {"x": ego["x"], "y": ego["y"], "yaw": ego["yaw"], "speed": ego["speed"],
                       "accel": ego["accel"], "yaw_rate": ego["yaw_rate"], "length": ego["length"],
                       "width": ego["width"], "height": ego["height"], "role": "ego",
                       "type": "car", "source": "ego"}}
        for agent in info.get("agents", []):
            vid = agent["id"] if agent["source"] == "recorded" else f'{agent["source"]}:{agent["id"]}'
            out[vid] = {"x": agent["x"], "y": agent["y"], "yaw": agent["yaw"],
                        "speed": agent["speed"], "accel": agent.get("accel", float("nan")),
                        "yaw_rate": float("nan"), "length": agent["length"],
                        "width": agent["width"], "height": agent["height"],
                        "role": "background", "type": agent.get("type", "car"),
                        "source": agent["source"]}
        return out

    def add(self, info: dict[str, Any]) -> None:
        states = self._states(info)
        if self.steps:
            t0, _, prev = self.steps[-1]
            dt = info["time"] - t0
            for vid, st in states.items():
                p = prev.get(vid)
                if vid == "ego" or p is None or dt <= 0:
                    continue
                if math.isnan(st["accel"]):
                    st["accel"] = (st["speed"] - p["speed"]) / dt
                st["yaw_rate"] = math.remainder(st["yaw"] - p["yaw"], 2 * math.pi) / dt
        self.steps.append((float(info["time"]), int(info["step"]), states))

    # ------------------------------------------------------------- write
    def _row(self, t: float, step: int, vid: str, st: dict) -> list:
        x, y, yaw = st["x"], st["y"], st["yaw"]
        z = float(self.ground.height(x, y)) if self.ground is not None else 0.0
        lane_id, s, d = "", float("nan"), float("nan")
        xg = yg = yawg = float("nan")
        if self.hdmap is not None:
            match = self.hdmap.match(x, y, yaw)
            if match is not None:
                lane_id, s, d = match.lane_id, match.s, match.d
            xg, yg = self.hdmap.to_global(np.array([[x, y]]))[0]
            yawg = float(self.hdmap.global_yaw(yaw))
        return [round(t, 4), step, vid, st["role"], st["type"], st["source"], x, y, z, yaw,
                st["speed"], st["speed"] * math.cos(yaw), st["speed"] * math.sin(yaw),
                st["accel"], st["yaw_rate"], st["length"], st["width"], st["height"],
                lane_id, s, d, xg, yg, yawg]

    def _write(self, path: Path, rows) -> None:
        with open(path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for row in rows:
                writer.writerow([f"{v:.4f}" if isinstance(v, float) else v for v in row])

    def write(self, directory: str | Path, hz: float = 10.0) -> dict[str, Path]:
        directory = Path(directory)
        out = {"steps": directory / "trajectories.csv"}
        self._write(out["steps"], (self._row(t, k, vid, st)
                                   for t, k, states in self.steps for vid, st in states.items()))
        if hz and len(self.steps) > 1:
            out["resampled"] = directory / f"trajectories_{int(hz)}hz.csv"
            self._write(out["resampled"], self._resampled(hz))
            out["fcd"] = directory / "trajectories_fcd.xml"
            write_fcd(out["fcd"], self._resampled(hz))
        return out

    def sample(self, t: float) -> dict[str, dict]:
        """States at an arbitrary time (vehicles present at both ends)."""
        times = [s[0] for s in self.steps]
        i = int(np.searchsorted(times, t, side="right")) - 1
        i = min(max(i, 0), len(self.steps) - 1)
        t0, _, a = self.steps[i]
        if i + 1 >= len(self.steps) or t <= t0:
            return a
        t1, _, b = self.steps[i + 1]
        u = (t - t0) / (t1 - t0)
        return {vid: interpolate_state(a[vid], b[vid], t1 - t0, u) if vid in b else a[vid]
                for vid in a if vid in b or u < 0.5}

    def _resampled(self, hz: float):
        end = self.steps[-1][0]
        n = int(math.floor(end * hz + 1e-6))
        for k in range(n + 1):
            t = k / hz
            step = int(np.searchsorted([s[0] for s in self.steps], t + 1e-9, side="right")) - 1
            for vid, st in self.sample(t).items():
                yield self._row(t, self.steps[max(step, 0)][1], vid, st)


def write_fcd(path: str | Path, rows) -> None:
    """SUMO floating-car data from trajectory rows (see :data:`COLUMNS`)."""
    col = {name: i for i, name in enumerate(COLUMNS)}
    from xml.sax.saxutils import quoteattr
    with open(path, "w", encoding="utf-8") as handle:
        handle.write('<?xml version="1.0" encoding="UTF-8"?>\n<!-- AETHER closed-loop run; '
                     'road-frame coordinates = map/network.net.xml -->\n<fcd-export>\n')
        current = None
        for row in rows:
            t = float(row[col["time_s"]])
            if current is None or abs(t - current) > 1e-9:
                if current is not None:
                    handle.write("    </timestep>\n")
                handle.write(f'    <timestep time="{t:.2f}">\n')
                current = t
            yaw = float(row[col["yaw"]])
            angle = (90.0 - math.degrees(yaw)) % 360.0
            lane = row[col["lane_id"]]
            extra = f' lane={quoteattr(str(lane))} pos="{float(row[col["s"]]):.2f}"' if lane else ""
            handle.write(
                f'        <vehicle id={quoteattr(str(row[col["vehicle_id"]]))} '
                f'x="{float(row[col["x"]]):.2f}" y="{float(row[col["y"]]):.2f}" '
                f'z="{float(row[col["z"]]):.2f}" angle="{angle:.2f}" '
                f'type={quoteattr(str(row[col["role"]]))} speed="{float(row[col["speed"]]):.2f}" '
                f'acceleration="{float(row[col["accel"]]):.2f}"{extra}/>\n')
        if current is not None:
            handle.write("    </timestep>\n")
        handle.write("</fcd-export>\n")
