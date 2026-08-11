"""AETHER-side bridge to the vendored Difix3D refiner.

Difix runs as a subprocess of the active AETHER environment. The subprocess
keeps GPU memory and binary image IPC isolated without requiring another conda
environment.

Public API:
    refine_directory(input_dir, output_dir, ...)   # one-shot offline
    DifixDaemon(...)                               # long-lived in-loop
"""
from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from aether.paths import AETHER_ROOT

_DEFAULT_DIFIX_REPO = Path(
    os.environ.get("DIFIX3D_ROOT", AETHER_ROOT / "third_party" / "difix3d")
).expanduser()
_DEFAULT_DIFIX_PYTHON = Path(sys.executable)


class DifixBridgeError(RuntimeError):
    pass


# ===================================================================== #
# One-shot offline path (unchanged: uses scripts/refine_dir.py)
# ===================================================================== #
def refine_directory(
    input_dir: str | os.PathLike,
    output_dir: str | os.PathLike,
    *,
    ref_dir: Optional[str | os.PathLike] = None,
    model_name: str = "nvidia/difix",
    prompt: str = "remove degradation",
    max_side: int = 1024,
    timestep: int = 199,
    seed: int = 42,
    video: bool = True,
    fps: int = 10,
    limit: int = 0,
    difix_python: Optional[str | os.PathLike] = None,
    difix_repo: Optional[str | os.PathLike] = None,
    cuda_visible: Optional[str] = None,
    timeout: Optional[float] = None,
) -> dict:
    py = Path(difix_python or _DEFAULT_DIFIX_PYTHON).resolve()
    repo = Path(difix_repo or _DEFAULT_DIFIX_REPO).resolve()
    if not py.exists():
        raise DifixBridgeError(f"difix python not found: {py}")
    script = repo / "scripts" / "refine_dir.py"
    if not script.exists():
        raise DifixBridgeError(f"difix refine_dir.py not found: {script}")

    in_p  = Path(input_dir).resolve()
    out_p = Path(output_dir).resolve()
    out_p.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(py), str(script),
        "--input_dir",  str(in_p),
        "--output_dir", str(out_p),
        "--model_name", model_name,
        "--prompt",     prompt,
        "--max_side",   str(int(max_side)),
        "--timestep",   str(int(timestep)),
        "--seed",       str(int(seed)),
        "--fps",        str(int(fps)),
    ]
    if ref_dir is not None:
        cmd.extend(["--ref_dir", str(Path(ref_dir).resolve())])
    if video:
        cmd.append("--video")
    if limit > 0:
        cmd.extend(["--limit", str(int(limit))])

    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible)

    r = subprocess.run(cmd, env=env, cwd=str(repo), timeout=timeout)
    if r.returncode != 0:
        raise DifixBridgeError(f"refine_dir.py exited {r.returncode}")

    perf = out_p / "refine_perf.json"
    if not perf.exists():
        return {}
    with open(perf, "r") as f:
        return json.load(f)


# ===================================================================== #
# Long-lived in-loop daemon (binary batched IPC, v2)
# ===================================================================== #
class DifixDaemon:
    """Persistent Difix refiner with batched + bf16 IPC.

    Sends every closed-loop step's N camera images in a single batched
    UNet/VAE forward. Stream is binary over stdin/stdout: a JSON header
    line followed by raw uint8 RGB bytes both ways.

    Example::

        with DifixDaemon(cuda_visible="0", max_side=512, dtype="bf16") as dx:
            obs["rgb"] = dx.refine_obs_rgb(obs["rgb"])
    """

    def __init__(self,
                 *,
                 model_name: str = "nvidia/difix",
                 prompt: str = "remove degradation",
                 max_side: int = 512,
                 timestep: int = 199,
                 seed: int = 42,
                 dtype: str = "bf16",
                 difix_python: Optional[str | os.PathLike] = None,
                 difix_repo:   Optional[str | os.PathLike] = None,
                 cuda_visible: Optional[str] = None,
                 work_root:    Optional[str | os.PathLike] = None) -> None:
        self.model_name = model_name
        self.prompt     = prompt
        self.max_side   = int(max_side)
        self.timestep   = int(timestep)
        self.seed       = int(seed)
        self.dtype      = str(dtype)
        self.py   = Path(difix_python or _DEFAULT_DIFIX_PYTHON).resolve()
        self.repo = Path(difix_repo   or _DEFAULT_DIFIX_REPO).resolve()
        self.cuda_visible = cuda_visible
        self.proc: Optional[subprocess.Popen] = None
        self._step = 0
        self._last_ms = 0.0
        # ``work_root`` is retained only for log files; no per-step disk IO.
        self._work_root = Path(work_root) if work_root else None
        if self._work_root is not None:
            self._work_root.mkdir(parents=True, exist_ok=True)

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "DifixDaemon":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if not self.py.exists():
            raise DifixBridgeError(f"difix python not found: {self.py}")
        script = self.repo / "scripts" / "refine_daemon.py"
        if not script.exists():
            raise DifixBridgeError(f"refine_daemon.py not found: {script}")

        env = os.environ.copy()
        env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
        env["PYTHONUNBUFFERED"] = "1"
        if self.cuda_visible is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible)

        cmd = [str(self.py), "-u", str(script),
               "--model_name", self.model_name,
               "--prompt", self.prompt,
               "--seed", str(self.seed),
               "--dtype", self.dtype]
        print(f"[difix-daemon] starting: {' '.join(cmd)}")
        # Binary mode: text=False (default).  bufsize=0 keeps writes unbuffered.
        self.proc = subprocess.Popen(
            cmd, env=env, cwd=str(self.repo),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=sys.stderr, bufsize=0,
        )
        ready = self._read_header()
        if not ready.get("ok"):
            raise DifixBridgeError(f"daemon failed to start: {ready}")
        if ready.get("event") != "ready":
            raise DifixBridgeError(f"daemon unexpected first event: {ready}")
        print(f"[difix-daemon] ready (model={self.model_name},"
              f" dtype={self.dtype}, max_side={self.max_side})")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self._send_header({"cmd": "exit"})
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        self.proc = None

    # -- low-level IO ----------------------------------------------------
    def _read_exact(self, n: int) -> bytes:
        assert self.proc is not None and self.proc.stdout is not None
        out = self.proc.stdout
        buf = bytearray()
        while len(buf) < n:
            chunk = out.read(n - len(buf))
            if not chunk:
                rc = self.proc.poll()
                raise DifixBridgeError(
                    f"daemon stdout EOF after {len(buf)}/{n} bytes (exit={rc})")
            buf.extend(chunk)
        return bytes(buf)

    def _read_header(self) -> dict:
        assert self.proc is not None and self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            rc = self.proc.poll()
            raise DifixBridgeError(f"daemon stdout closed (exit={rc})")
        try:
            return json.loads(line.decode("utf-8").strip())
        except Exception as e:
            raise DifixBridgeError(f"bad header from daemon: {line!r}: {e!r}")

    def _send_header(self, obj: dict) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.proc.stdin.flush()

    def _send_with_payload(self, obj: dict, payload: bytes) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write((json.dumps(obj) + "\n").encode("utf-8"))
        self.proc.stdin.write(payload)
        self.proc.stdin.flush()

    # -- main API --------------------------------------------------------
    @property
    def last_refine_ms(self) -> float:
        return float(self._last_ms)

    def refine_obs_rgb(self, rgb: dict) -> dict:
        """Refine an obs['rgb'] dict ``{cam_name: HxWx3 uint8}``.

        All cameras MUST share the same (H, W). Returns a new dict.
        """
        if not rgb:
            return {}
        self._step += 1
        names = sorted(rgb.keys())
        # Validate shapes & stack contiguously.
        first = np.asarray(rgb[names[0]])
        if first.dtype != np.uint8:
            first = first.astype(np.uint8) if first.max() > 1.0 \
                else (np.clip(first, 0, 1) * 255).astype(np.uint8)
        H, W = int(first.shape[0]), int(first.shape[1])
        N = len(names)
        batch = np.empty((N, H, W, 3), dtype=np.uint8)
        for i, name in enumerate(names):
            arr = np.asarray(rgb[name])
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8) if arr.max() > 1.0 \
                    else (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            if arr.shape[:2] != (H, W) or arr.shape[2] != 3:
                raise DifixBridgeError(
                    f"camera '{name}' shape {arr.shape} differs from "
                    f"first cam '{names[0]}' ({H},{W},3)")
            batch[i] = arr

        header = {
            "cmd": "refine", "n": N, "H": H, "W": W,
            "max_side": self.max_side, "timestep": self.timestep,
        }
        self._send_with_payload(header, batch.tobytes())

        rep = self._read_header()
        if not rep.get("ok"):
            raise DifixBridgeError(f"daemon error: {rep.get('err')}")
        nbytes = N * H * W * 3
        payload = self._read_exact(nbytes)
        out = np.frombuffer(payload, dtype=np.uint8).reshape(N, H, W, 3)
        self._last_ms = float(rep.get("ms", 0.0))
        return {name: out[i].copy() for i, name in enumerate(names)}


__all__ = ["refine_directory", "DifixBridgeError", "DifixDaemon"]
