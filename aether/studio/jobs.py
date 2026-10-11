"""Durable subprocess job manager for long-running GPU experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import os
import subprocess
import sys
from uuid import uuid4

from .catalog import REPO_ROOT


@dataclass
class Job:
    id: str
    experiment_id: str
    status: str
    command: list[str]
    log_path: str
    result_path: str
    created_at: str
    pid: int | None = None
    returncode: int | None = None


class JobManager:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.jobs_dir = workspace / "jobs"
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, Job] = {}
        self._processes: dict[str, subprocess.Popen] = {}

    def _refresh(self, job: Job) -> Job:
        process = self._processes.get(job.id)
        if process is not None:
            returncode = process.poll()
            if returncode is None:
                job.status = "running"
            else:
                job.returncode = returncode
                job.status = "succeeded" if returncode == 0 else "failed"
                self._processes.pop(job.id, None)
                self._persist(job)
        return job

    def _persist(self, job: Job) -> None:
        (self.jobs_dir / f"{job.id}.json").write_text(
            json.dumps(asdict(job), indent=2), encoding="utf-8"
        )

    def start(self, manifest_path: str | Path, experiment_id: str) -> Job:
        active = [job for job in self._jobs.values() if self._refresh(job).status == "running"]
        if active:
            raise RuntimeError(f"GPU job {active[0].id} is already running")
        job_id = uuid4().hex[:12]
        log_path = self.jobs_dir / f"{job_id}.log"
        command = [
            sys.executable, "-u", str(REPO_ROOT / "scripts/run_studio_experiment.py"),
            "--manifest", str(Path(manifest_path).resolve()),
        ]
        log_handle = log_path.open("wb")
        worker_env = os.environ.copy()
        worker_env["PYTHONNOUSERSITE"] = "1"
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, env=worker_env,
            stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_handle.close()
        job = Job(
            id=job_id, experiment_id=experiment_id, status="running", command=command,
            log_path=str(log_path),
            result_path=str(Path(manifest_path).resolve().parent / "run_result.json"),
            created_at=datetime.now(timezone.utc).isoformat(), pid=process.pid,
        )
        self._jobs[job_id] = job
        self._processes[job_id] = process
        self._persist(job)
        return job

    def get(self, job_id: str) -> Job:
        if job_id not in self._jobs:
            path = self.jobs_dir / f"{job_id}.json"
            if not path.is_file():
                raise KeyError(job_id)
            self._jobs[job_id] = Job(**json.loads(path.read_text(encoding="utf-8")))
        return self._refresh(self._jobs[job_id])

    def list(self) -> list[Job]:
        for path in self.jobs_dir.glob("*.json"):
            if path.stem not in self._jobs:
                self._jobs[path.stem] = Job(**json.loads(path.read_text(encoding="utf-8")))
        return sorted(
            (self._refresh(job) for job in self._jobs.values()),
            key=lambda job: job.created_at, reverse=True,
        )

    def log_text(self, job_id: str, offset: int = 0) -> tuple[str, int]:
        path = Path(self.get(job_id).log_path)
        if not path.is_file():
            return "", offset
        with path.open("rb") as handle:
            handle.seek(max(0, offset))
            data = handle.read()
            return data.decode("utf-8", errors="replace"), handle.tell()
