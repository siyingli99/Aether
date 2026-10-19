"""FastAPI backend of AETHER Studio.

    GET  /api/health
    GET  /api/catalog                      scenes, agents, assets, vehicle profiles, defaults
    GET  /api/scenes/{id}                  map geometry, logged route, recorded actors
    POST /api/validate                     validate a scenario dict
    POST /api/runs                         start a closed-loop run  {scenario, agent, cuda}
    GET  /api/jobs | /api/jobs/{id}        job status
    POST /api/jobs/{id}/cancel
    WS   /api/jobs/{id}/live               live frames (JPEG, base64), log lines, status
    GET  /api/results | /api/results/{run} finished runs with metrics, events, files
    GET  /files/runs/<run>/<file>          run outputs (videos, CSV)
"""
from __future__ import annotations

import asyncio
import base64
import csv
from dataclasses import asdict
import json
from pathlib import Path

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

from .. import __version__
from ..config import Scenario, scenario_from_dict
from ..paths import ASSETS_ROOT, OUTPUTS_ROOT, REPO_ROOT
from .jobs import JobManager
from .scenes import scene_detail, scene_list

RUNS = OUTPUTS_ROOT / "runs"


def _agents() -> list[dict]:
    from ..agents.external import list_adapters, load_adapter
    out = [{"id": "replay", "title": "Log replay (expert)", "kind": "builtin",
            "description": "Follows the recorded route at the recorded speed. No model needed.",
            "cameras": [], "lidar": False}]
    for name in list_adapters():
        spec = load_adapter(name)
        out.append({"id": name, "title": spec.title, "kind": "adapter", "description": spec.description,
                    "cameras": spec.cameras or [], "lidar": spec.lidar,
                    "plan": {"frame": spec.frame, "axes": spec.axes, "dt": spec.dt}})
    return out


def _assets() -> list[str]:
    root = ASSETS_ROOT / "vehicles"
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def _read_csv(path: Path, limit: int = 1000) -> list[dict]:
    if not path.is_file():
        return []
    with open(path) as handle:
        return [row for _, row in zip(range(limit), csv.DictReader(handle))]


def _run_summary(path: Path) -> dict:
    meta = json.loads((path / "run.json").read_text()) if (path / "run.json").is_file() else {}
    metrics = json.loads((path / "metrics.json").read_text()) if (path / "metrics.json").is_file() else None
    return {"id": path.name, "scene": meta.get("scene"), "agent": meta.get("agent"),
            "started": meta.get("started"), "steps": meta.get("steps"),
            "terminated_by": meta.get("terminated_by"), "wall_s": meta.get("wall_s"),
            "metrics": metrics, "complete": metrics is not None}


def create_app() -> FastAPI:
    from ..vehicle.profile import list_profiles, load_profile

    jobs = JobManager(OUTPUTS_ROOT / "studio" / "jobs")
    RUNS.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="AETHER Studio", version=__version__)
    app.add_middleware(CORSMiddleware, allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
                       allow_methods=["*"], allow_headers=["*"])

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "version": __version__}

    @app.get("/api/catalog")
    def catalog() -> dict:
        return {"scenes": scene_list(), "agents": _agents(), "assets": _assets(),
                "vehicles": [{"name": n, **{k: getattr(load_profile(n), k)
                                            for k in ("length", "width", "height", "wheelbase")}}
                             for n in list_profiles()],
                "defaults": Scenario(scene="").model_dump(mode="json"),
                "frame": {"x": "forward (initial ego heading)", "y": "left", "yaw": "counter-clockwise",
                          "unit": "metre / radian"}}

    @app.get("/api/scenes/{scene_id}")
    def scene(scene_id: str) -> dict:
        try:
            return scene_detail(scene_id)
        except FileNotFoundError as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/api/validate")
    def validate(scenario: dict = Body(...)) -> dict:
        try:
            return {"valid": True, "scenario": scenario_from_dict(scenario).model_dump(mode="json")}
        except ValidationError as error:
            return {"valid": False, "errors": json.loads(error.json())}

    @app.post("/api/runs")
    def start_run(payload: dict = Body(...)) -> dict:
        try:
            scenario = scenario_from_dict(payload["scenario"]).model_dump(mode="json")
        except (KeyError, ValidationError) as error:
            raise HTTPException(422, str(error)) from error
        try:
            job = jobs.start(scenario, str(payload.get("agent") or "replay"), str(payload.get("cuda", "0")))
        except RuntimeError as error:
            raise HTTPException(409, str(error)) from error
        return asdict(job)

    @app.get("/api/jobs")
    def list_jobs() -> list[dict]:
        return [asdict(j) for j in jobs.list()]

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        try:
            return asdict(jobs.get(job_id))
        except KeyError as error:
            raise HTTPException(404, f"unknown job {job_id}") from error

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str) -> dict:
        return asdict(jobs.cancel(job_id))

    @app.websocket("/api/jobs/{job_id}/live")
    async def live(ws: WebSocket, job_id: str) -> None:
        await ws.accept()
        offset, seen = 0, {"bev": 0.0, "camera": 0.0}
        try:
            while True:
                try:
                    job = jobs.get(job_id)
                except KeyError:
                    await ws.send_json({"type": "error", "message": f"unknown job {job_id}"})
                    return
                live_dir = Path(job.dir) / "live"
                for kind in ("bev", "camera"):
                    path = live_dir / f"{kind}.jpg"
                    if path.is_file() and path.stat().st_mtime > seen[kind]:
                        seen[kind] = path.stat().st_mtime
                        state = {}
                        if (live_dir / "state.json").is_file():
                            try:
                                state = json.loads((live_dir / "state.json").read_text())
                            except ValueError:
                                pass
                        await ws.send_json({"type": "frame", "kind": kind, "time": state.get("time"),
                                            "jpeg": base64.b64encode(path.read_bytes()).decode()})
                text, offset = jobs.log(job_id, offset)
                if text:
                    await ws.send_json({"type": "log", "text": text})
                if job.status != "running":
                    await ws.send_json({"type": "status", "job": asdict(job)})
                    return
                await asyncio.sleep(0.1)
        except WebSocketDisconnect:
            return

    @app.get("/api/results")
    def results() -> list[dict]:
        runs = [p for p in RUNS.iterdir() if (p / "run.json").is_file()] if RUNS.is_dir() else []
        return [_run_summary(p) for p in sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)]

    @app.get("/api/results/{run_id}")
    def result(run_id: str) -> dict:
        path = (RUNS / run_id).resolve()
        if not path.is_relative_to(RUNS.resolve()) or not (path / "run.json").is_file():
            raise HTTPException(404, f"unknown run {run_id}")
        files = sorted(str(p.relative_to(path)) for p in path.rglob("*")
                       if p.is_file() and "agent" not in p.relative_to(path).parts[:1])
        return {**_run_summary(path), "run": json.loads((path / "run.json").read_text()),
                "events": _read_csv(path / "events.csv"), "files": files,
                "base_url": f"/files/runs/{run_id}"}

    app.mount("/files/runs", StaticFiles(directory=RUNS), name="runs")
    dist = REPO_ROOT / "web" / "dist"
    if dist.is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")
        root = dist.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def frontend(path: str):
            candidate = (dist / path).resolve()
            if not candidate.is_relative_to(root) or not candidate.is_file():
                candidate = dist / "index.html"
            return FileResponse(candidate)

    return app
