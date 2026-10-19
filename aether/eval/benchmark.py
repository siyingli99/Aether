"""Run a set of scenarios × agents and aggregate their metrics.

``configs/benchmarks/<name>.yaml``::

    name: nuscenes-demo
    scenarios: [configs/scenarios/nuscenes-0383.yaml, configs/scenarios/nuscenes-0106.yaml]
    agents: [uniad, vad]
    seeds: [0]
    overrides: {refine: {enabled: true}}

Output ``outputs/benchmarks/<name>-<time>/``: ``results.csv`` (one row per
run, with the run directory) and ``summary.csv`` / ``summary.json`` (mean
per agent).  A failed run is recorded with ``status=error`` and scores 0.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
from pathlib import Path
import traceback

import yaml

from ..config import load_scenario
from ..paths import OUTPUTS_ROOT, resolve

SCORE_KEYS = ["score", "nc", "dac", "ttc", "comfort", "ep", "rc"]


def run_benchmark(spec_path: str | Path, cuda: str = "0", output_root: str | Path | None = None,
                  agents: list[str] | None = None) -> Path:
    from ..sim.runner import run_episode

    spec = yaml.safe_load(Path(spec_path).read_text())
    name = spec.get("name", Path(spec_path).stem)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = (resolve(output_root) if output_root else OUTPUTS_ROOT / "benchmarks") / f"{name}-{stamp}"
    runs_root = out / "runs"
    runs_root.mkdir(parents=True)
    rows = []
    for agent in agents or spec["agents"]:
        for scenario_path in spec["scenarios"]:
            for seed in spec.get("seeds", [0]):
                overrides = dict(spec.get("overrides") or {})
                overrides["sim"] = {**overrides.get("sim", {}), "seed": seed}
                scenario = load_scenario(resolve(scenario_path), overrides)
                row = {"agent": agent, "scenario": scenario.name or scenario.scene, "seed": seed}
                try:
                    result = run_episode(scenario, agent, output_root=runs_root, cuda=cuda)
                    row.update(status="ok", run=str(result.path.relative_to(out)),
                               terminated_by=result.terminated_by,
                               **{k: result.metrics[k] for k in SCORE_KEYS})
                except Exception as error:
                    traceback.print_exc()
                    row.update(status="error", error=str(error).splitlines()[0][:200],
                               **{k: 0.0 for k in SCORE_KEYS})
                rows.append(row)
                _write_rows(out / "results.csv", rows)
    summary = []
    for agent in dict.fromkeys(r["agent"] for r in rows):
        mine = [r for r in rows if r["agent"] == agent]
        entry = {"agent": agent, "runs": len(mine), "errors": sum(r["status"] != "ok" for r in mine)}
        entry.update({k: round(sum(float(r[k]) for r in mine) / len(mine), 4) for k in SCORE_KEYS})
        summary.append(entry)
    _write_rows(out / "summary.csv", summary)
    (out / "summary.json").write_text(json.dumps({"benchmark": name, "agents": summary}, indent=2))
    print(f"[benchmark] {name}: {out}")
    for entry in summary:
        print("  " + "  ".join(f"{k}={entry[k]}" for k in ["agent", "runs", "errors"] + SCORE_KEYS))
    return out


def _write_rows(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
