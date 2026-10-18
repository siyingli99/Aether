"""FastAPI application for the local AETHER Studio."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .catalog import REPO_ROOT, StudioCatalog
from .compiler import ExperimentCompiler
from .jobs import JobManager
from .schemas import ExperimentSpec


WORKSPACE = (REPO_ROOT / "studio_workspace").resolve()
WORKSPACE.mkdir(parents=True, exist_ok=True)
catalog = StudioCatalog()
compiler = ExperimentCompiler(catalog, WORKSPACE)
jobs = JobManager(WORKSPACE)
preview_processes: dict[str, subprocess.Popen] = {}

app = FastAPI(title="AETHER Studio", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "version": app.version}


@app.get("/api/catalog")
def get_catalog() -> dict:
    return {
        "scenes": catalog.list_scenes(), "assets": catalog.list_assets(),
        "controllers": catalog.list_controllers(), "adapters": catalog.list_adapters(),
        "coordinates": {
            "x": "forward", "y": "left", "z": "up", "yaw": "CCW/left",
            "distance_unit": "metre", "angle_unit": "radian",
        },
    }


@app.get("/api/scenes/{scene_id}")
def get_scene(scene_id: str) -> dict:
    try:
        return {
            "scene": catalog.scene(scene_id), "topology": catalog.topology(scene_id),
            "actors": catalog.actors(scene_id),
            "experiment": catalog.bootstrap_experiment(scene_id),
            "preview": preview_status(scene_id),
        }
    except KeyError as error:
        raise HTTPException(404, str(error)) from error


def preview_status(scene_id: str) -> dict:
    output = WORKSPACE / "previews" / scene_id
    metadata = output / "preview.json"
    process = preview_processes.get(scene_id)
    if process is not None and process.poll() is None:
        return {"status": "running"}
    if process is not None and process.returncode:
        return {"status": "failed", "returncode": process.returncode}
    if metadata.is_file():
        payload = json.loads(metadata.read_text(encoding="utf-8"))
        payload["status"] = "ready"
        payload["base_url"] = f"/media/previews/{scene_id}"
        return payload
    return {"status": "missing"}


@app.post("/api/scenes/{scene_id}/preview")
def create_preview(scene_id: str, force: bool = False) -> dict:
    catalog.scene(scene_id)
    status = preview_status(scene_id)
    if status["status"] in {"ready", "running"} and not force:
        return status
    output = WORKSPACE / "previews" / scene_id
    output.mkdir(parents=True, exist_ok=True)
    log = (output / "preview.log").open("wb")
    worker_env = os.environ.copy()
    worker_env["PYTHONNOUSERSITE"] = "1"
    process = subprocess.Popen(
        [sys.executable, "-m", "aether.studio.preview_worker", "--scene", scene_id,
         "--output", str(output)],
        cwd=REPO_ROOT, env=worker_env, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log.close()
    preview_processes[scene_id] = process
    return {"status": "running", "pid": process.pid}


@app.get("/api/scenes/{scene_id}/preview")
def get_preview_status(scene_id: str) -> dict:
    catalog.scene(scene_id)
    return preview_status(scene_id)


@app.post("/api/experiments/validate")
def validate_experiment(spec: ExperimentSpec) -> dict:
    scene = catalog.scene(spec.scene_id)
    warnings = []
    if not spec.runtime.difix.enabled:
        warnings.append("Difix is disabled; camera inputs will be sent directly to the adapter")
    if any(actor.controller.kind == "limsim" for actor in spec.actors) and not Path(scene["net"]).is_file():
        raise HTTPException(422, "selected scene has no LimSim network")
    return {"valid": True, "warnings": warnings, "normalized": spec.model_dump(mode="json")}


@app.post("/api/experiments")
def save_experiment(spec: ExperimentSpec) -> dict:
    try:
        return compiler.compile(spec)
    except (KeyError, ValueError, FileNotFoundError) as error:
        raise HTTPException(422, str(error)) from error


@app.post("/api/experiments/{experiment_id}/run")
def run_experiment(experiment_id: str) -> dict:
    manifest = WORKSPACE / "experiments" / experiment_id / "manifest.json"
    if not manifest.is_file():
        raise HTTPException(404, "compile the experiment before running it")
    try:
        return asdict(jobs.start(manifest, experiment_id))
    except RuntimeError as error:
        raise HTTPException(409, str(error)) from error


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [asdict(job) for job in jobs.list()]


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        job = jobs.get(job_id)
    except KeyError as error:
        raise HTTPException(404, f"unknown job: {job_id}") from error
    payload = asdict(job)
    result_path = Path(job.result_path)
    if result_path.is_file():
        payload["result"] = json.loads(result_path.read_text(encoding="utf-8"))
    return payload


@app.websocket("/api/jobs/{job_id}/logs")
async def job_logs(websocket: WebSocket, job_id: str) -> None:
    await websocket.accept()
    offset = 0
    try:
        while True:
            try:
                job = jobs.get(job_id)
            except KeyError:
                await websocket.send_json({"error": f"unknown job: {job_id}"})
                return
            chunk, offset = jobs.log_text(job_id, offset)
            if chunk:
                await websocket.send_json({"text": chunk, "offset": offset})
            if job.status in {"succeeded", "failed"}:
                await websocket.send_json({"status": job.status, "returncode": job.returncode})
                return
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


app.mount("/media", StaticFiles(directory=WORKSPACE), name="media")
WEB_DIST = REPO_ROOT / "web/dist"
if WEB_DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")

    _WEB_DIST_ROOT = WEB_DIST.resolve()

    @app.get("/{path:path}", include_in_schema=False)
    def frontend(path: str):
        candidate = (WEB_DIST / path).resolve()
        # SPA fallback; never serve files outside web/dist (e.g. "../..").
        if not candidate.is_relative_to(_WEB_DIST_ROOT) or not candidate.is_file():
            candidate = WEB_DIST / "index.html"
        return FileResponse(candidate)
