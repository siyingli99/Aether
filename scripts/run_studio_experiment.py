#!/usr/bin/env python3
"""Execute one compiled Studio manifest, including its LimSim sidecar."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
AETHER_PYTHON = Path(sys.executable)
LIMSIM_ROOT = Path(os.environ.get("LIMSIM_ROOT", str(REPO_ROOT.parent / "LimSim")))
LIMSIM_PYTHON = Path(os.environ.get(
    "LIMSIM_PYTHON", str(AETHER_PYTHON.parents[2] / "limsim" / "bin" / "python")
))


def _wait_port(host: str, port: int, process: subprocess.Popen, timeout: float = 45.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"LimSim bridge exited with {process.returncode}")
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError(f"LimSim bridge did not become ready at {host}:{port}")


def _terminate(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=8)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    limsim = manifest.get("limsim") or {}
    bridge = None
    result_path = manifest_path.parent / "run_result.json"
    started = time.time()
    try:
        if limsim.get("enabled"):
            host, port = str(limsim.get("host", "127.0.0.1")), int(limsim.get("port", 8766))
            bridge_cmd = [
                str(LIMSIM_PYTHON), "-u", str(LIMSIM_ROOT / "aether_bridge/server.py"),
                "--host", host, "--port", str(port),
                "--net-file", str(limsim["net_file"]),
                "--route-file", str(limsim["route_file"]),
            ]
            bridge_env = os.environ.copy()
            bridge_env["AETHER_ROOT"] = str(REPO_ROOT)
            bridge_env["PATH"] = f"{LIMSIM_PYTHON.parent}:{AETHER_PYTHON.parent}:/usr/local/bin:/usr/bin:/bin"
            bridge = subprocess.Popen(
                bridge_cmd, cwd=REPO_ROOT, env=bridge_env,
                start_new_session=True,
            )
            _wait_port(host, port, bridge)
            print(f"[studio] LimSim ready at {host}:{port}", flush=True)

        difix = manifest["difix"]
        command = [
            str(AETHER_PYTHON), "-u", str(REPO_ROOT / "scripts/closed_loop.py"),
            "--scenario", manifest["scenario"],
            "--base", str(REPO_ROOT / "configs/base.yaml"),
            "--camera", manifest["camera"],
            "--kinematic", str(REPO_ROOT / "configs/kinematic.yaml"),
            "--ad", manifest["adapter"], "--ad_cuda", str(manifest["ad_cuda"]),
            "--seed", str(manifest["seed"]),
        ]
        if manifest.get("adapter_config"):
            command += ["--ad_config", manifest["adapter_config"]]
        if difix.get("enabled"):
            command += [
                "--refine", "--refine_cuda", str(difix["cuda"]),
                "--refine_max_side", str(difix["max_side"]),
                "--refine_model", str(difix["model"]),
                "--refine_timestep", str(difix["timestep"]),
                "--refine_dtype", str(difix["dtype"]),
            ]
        else:
            command += ["--no_refine"]
        env = os.environ.copy()
        env.setdefault("AETHER_AD_READY_TIMEOUT", "1800")
        print("[studio] " + " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=REPO_ROOT, env=env)
        status = "succeeded" if completed.returncode == 0 else "failed"
        result = {
            "status": status, "returncode": completed.returncode,
            "started_at": started, "finished_at": time.time(),
            "output_dir": manifest["output_dir"],
        }
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return completed.returncode
    except Exception as error:
        result_path.write_text(json.dumps({
            "status": "failed", "error": f"{type(error).__name__}: {error}",
            "started_at": started, "finished_at": time.time(),
        }, indent=2), encoding="utf-8")
        raise
    finally:
        _terminate(bridge)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.manifest.resolve()))


if __name__ == "__main__":
    main()
