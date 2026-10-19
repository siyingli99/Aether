"""Recompute ``metrics.json`` from a finished run's CSV outputs (no GPU)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from ..scene.bundle import SceneBundle
from ..sim.route import Route
from .metrics import score_episode


def _steps(path: Path):
    by_step: dict[int, tuple[float, dict]] = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            k = int(row["step"])
            t = float(row["time_s"])
            states = by_step.setdefault(k, (t, {}))[1]
            states[row["vehicle_id"]] = {
                "x": float(row["x"]), "y": float(row["y"]), "yaw": float(row["yaw"]),
                "speed": float(row["speed"]), "length": float(row["length"]),
                "width": float(row["width"]), "type": row["type"], "role": row["role"]}
    return [(t, k, states) for k, (t, states) in sorted(by_step.items())]


def rescore(run: str | Path) -> dict:
    run = Path(run)
    meta = json.loads((run / "run.json").read_text())
    steps = _steps(run / "trajectories.csv")
    events = []
    with open(run / "events.csv") as handle:
        for row in csv.DictReader(handle):
            event = {"type": row["type"], "time": float(row["time_s"]), "step": int(row["step"]),
                     "other": row["other_id"] or None, "at_fault": row["at_fault"] == "True"}
            event.update(json.loads(row["detail"] or "{}"))
            events.append(event)
    bundle = SceneBundle.load(meta["scene"])
    # Logged centre track for progress, from the bundle log (no renderer needed).
    from ..scene.geometry import logged_centers
    centers = logged_centers(bundle, meta.get("vehicle"))
    route = Route(centers, bundle.frame_dt)

    def expert(t: float) -> float:
        i = min(t / bundle.frame_dt, len(centers) - 1)
        k = int(np.floor(i))
        p = centers[min(k + 1, len(centers) - 1)] * (i - k) + centers[k] * (1 - (i - k))
        return route.progress(*p)

    metrics = score_episode(steps, events, route, expert, meta.get("terminated_by")).to_dict()
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
