"""Studio jobs: closed-loop runs in a worker subprocess (one GPU job at a time)."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

from ..paths import REPO_ROOT


@dataclass
class Job:
    id: str
    status: str                      # running | succeeded | failed | cancelled
    scene: str
    agent: str
    created: str
    dir: str
    pid: int | None = None
    returncode: int | None = None
    run: str | None = None           # run directory once finished


class JobManager:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._procs: dict[str, subprocess.Popen] = {}

    def _path(self, job_id: str) -> Path:
        return self.root / job_id / "job.state.json"

    def _save(self, job: Job) -> None:
        self._path(job.id).write_text(json.dumps(asdict(job), indent=2))

    def _refresh(self, job: Job) -> Job:
        proc = self._procs.get(job.id)
        if proc is not None and job.status == "running":
            rc = proc.poll()
            if rc is not None:
                job.returncode = rc
                job.status = "succeeded" if rc == 0 else ("cancelled" if rc < 0 else "failed")
                result = Path(job.dir) / "result.json"
                if result.is_file():
                    job.run = json.loads(result.read_text()).get("run")
                self._procs.pop(job.id, None)
                self._save(job)
        elif job.status == "running" and proc is None:
            # Server restarted: the worker is no longer ours to track.
            result = Path(job.dir) / "result.json"
            job.status = "succeeded" if result.is_file() else "failed"
            if result.is_file():
                job.run = json.loads(result.read_text()).get("run")
            self._save(job)
        return job

    def start(self, scenario: dict, agent: str, cuda: str = "0") -> Job:
        if any(self._refresh(j).status == "running" for j in self.list()):
            raise RuntimeError("a run is already in progress (one GPU job at a time)")
        job_id = datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid4().hex[:4]
        directory = self.root / job_id
        directory.mkdir(parents=True)
        spec = {"scenario": scenario, "agent": agent, "cuda": cuda}
        (directory / "job.json").write_text(json.dumps(spec, indent=2))
        env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1")
        with open(directory / "worker.log", "wb") as log:
            proc = subprocess.Popen([sys.executable, "-m", "aether.studio.worker", str(directory / "job.json")],
                                    cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        job = Job(job_id, "running", scenario.get("scene", "?"), agent,
                  datetime.now().isoformat(timespec="seconds"), str(directory), proc.pid)
        self._procs[job_id] = proc
        self._save(job)
        return job

    def get(self, job_id: str) -> Job:
        path = self._path(job_id)
        if not path.is_file():
            raise KeyError(job_id)
        return self._refresh(Job(**json.loads(path.read_text())))

    def list(self) -> list[Job]:
        jobs = [Job(**json.loads(p.read_text())) for p in self.root.glob("*/job.state.json")]
        return sorted((self._refresh(j) for j in jobs), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> Job:
        import signal
        self.get(job_id)                      # KeyError for unknown ids
        proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
        return self.get(job_id)

    def log(self, job_id: str, offset: int = 0) -> tuple[str, int]:
        path = Path(self.get(job_id).dir) / "worker.log"
        if not path.is_file():
            return "", offset
        with open(path, "rb") as handle:
            handle.seek(offset)
            data = handle.read()
            return data.decode("utf-8", errors="replace"), handle.tell()
