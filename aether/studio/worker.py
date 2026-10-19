"""Studio run worker: one closed-loop episode with live frames.

``python -m aether.studio.worker <job.json>``; the job file holds the
scenario dict, agent and GPU.  Live state goes to ``<job dir>/live/``:
``bev.jpg``, ``camera.jpg`` (latest frames) and ``state.json``.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import time


def main(job_file: str) -> None:
    job = json.loads(Path(job_file).read_text())
    live = Path(job_file).parent / "live"
    live.mkdir(exist_ok=True)
    from PIL import Image
    from ..config import scenario_from_dict
    from ..sim.runner import run_episode

    last_write = {"bev": 0.0, "camera": 0.0}

    def publish(event: dict) -> None:
        kind = event["kind"]
        now = time.monotonic()
        if kind == "bev" and now - last_write["bev"] < 0.08:      # ≤ ~12 fps to disk
            return
        last_write[kind] = now
        buf = io.BytesIO()
        image = Image.fromarray(event["image"])
        if kind == "camera" and image.width > 1600:
            image = image.resize((1600, int(image.height * 1600 / image.width)))
        image.save(buf, "JPEG", quality=82)
        tmp = live / f".{kind}.jpg"
        tmp.write_bytes(buf.getvalue())
        os.replace(tmp, live / f"{kind}.jpg")
        state = {"kind": kind, "time": event["time"], "updated": time.time()}
        (live / ".state.json").write_text(json.dumps(state))
        os.replace(live / ".state.json", live / "state.json")

    scenario = scenario_from_dict(job["scenario"])
    result = run_episode(scenario, job["agent"], output_root=job.get("output_root"),
                         cuda=str(job.get("cuda", "0")), listener=publish)
    (Path(job_file).parent / "result.json").write_text(json.dumps(
        {"run": str(result.path), "metrics": result.metrics, "terminated_by": result.terminated_by}))


if __name__ == "__main__":
    main(sys.argv[1])
