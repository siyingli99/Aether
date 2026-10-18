"""Launches an external AD adapter process and exchanges (obs, plan) via two
named FIFOs in `output_dir`.

Compatible with HUGSIM's `sim.utils.launch_ad`: same FIFO names and pickle
protocol, so existing `e2e.sh` / `vad.sh` / `uniad.sh` scripts work as-is.
"""
from __future__ import annotations
import errno
import fcntl
import os
import select
import subprocess
import time


def launch(shell_path: str, cuda_id: str, output_dir: str) -> subprocess.Popen:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "ad_output.txt")
    with open(log_path, "w") as f:
        proc = subprocess.Popen(
            ["bash", shell_path, str(cuda_id), output_dir],
            stdout=f, stderr=f,
        )
    print(f"[ad] launched {shell_path} (pid={proc.pid}) -> {log_path}")
    return proc


# UniAD emits "Ready for recieving" when the adapter is ready.
_DEFAULT_READY_MARKERS = (
    "ready for",
    "[gf] ready",
    "adapter ready",
    "waiting for obs",
)


def wait_adapter_ready(proc: subprocess.Popen,
                       log_path: str,
                       ready_markers: tuple[str, ...] = _DEFAULT_READY_MARKERS,
                       timeout_s: float = 600.0,
                       poll_s: float = 0.5) -> None:
    """Block until the adapter log contains a ready marker or the process exits.

    Matching is case-insensitive so UniAD's ``Ready for recieving`` matches
    ``ready for`` (a bare ``ready`` would false-positive on MMCV warnings).
    """
    markers = tuple(m.lower() for m in ready_markers)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        rc = proc.poll()
        if rc is not None:
            tail = ""
            try:
                with open(log_path, encoding="utf-8", errors="replace") as f:
                    tail = f.read()[-4000:]
            except OSError:
                pass
            raise RuntimeError(
                f"AD adapter exited with code {rc} before becoming ready.\n"
                f"See log: {log_path}\n--- tail ---\n{tail}"
            )
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                text = f.read().lower()
        except OSError:
            text = ""
        if any(m in text for m in markers):
            print(f"[ad] adapter ready (matched in {log_path})", flush=True)
            return
        time.sleep(poll_s)
    tail = ""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            tail = f.read()[-4000:]
    except OSError:
        pass
    raise TimeoutError(
        f"AD adapter not ready after {timeout_s:.0f}s (log: {log_path})\n"
        f"--- tail ---\n{tail}"
    )


def assert_adapter_alive(proc: subprocess.Popen | None, log_path: str | None = None) -> None:
    if proc is None:
        return
    rc = proc.poll()
    if rc is None:
        return
    tail = ""
    if log_path:
        try:
            with open(log_path, encoding="utf-8", errors="replace") as f:
                tail = f.read()[-4000:]
        except OSError:
            pass
    raise RuntimeError(
        f"AD adapter exited with code {rc}.\n"
        + (f"See log: {log_path}\n--- tail ---\n{tail}" if log_path else "")
    )


def check_alive(proc: subprocess.Popen, tolerant: int = 100) -> None:
    for i in range(tolerant):
        rc = proc.poll()
        if rc is not None:
            print(f"[ad] adapter exited with code {rc}.")
            proc.kill()
            return
        if i % 5 == 0:
            print(f"[ad] adapter alive (remaining tolerance {tolerant - i}s).")
        time.sleep(1)
    proc.kill()
    print("[ad] adapter killed after tolerance timeout.")


def make_pipes(output_dir: str):
    obs = os.path.join(output_dir, "obs_pipe")
    plan = os.path.join(output_dir, "plan_pipe")
    for p in (obs, plan):
        if not os.path.exists(p):
            os.mkfifo(p)
    return obs, plan


# --------------------------------------------------------------------------- #
# Timeout-aware FIFO IO
# --------------------------------------------------------------------------- #
# A plain ``open(fifo)`` blocks forever when the peer has died: opening for
# write waits for a reader and opening for read waits for a writer.  These
# helpers poll the adapter process while waiting, so a crashed planner turns
# into an error with its log tail instead of a hung closed loop.

def _log_tail(log_path: str | None, limit: int = 4000) -> str:
    if not log_path:
        return ""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            return handle.read()[-limit:]
    except OSError:
        return ""


def _check_peer(proc: subprocess.Popen | None, log_path: str | None,
                deadline: float, what: str) -> None:
    if proc is not None and proc.poll() is not None:
        raise RuntimeError(
            f"AD adapter exited with code {proc.returncode} while {what}.\n"
            f"See log: {log_path}\n--- tail ---\n{_log_tail(log_path)}"
        )
    if time.monotonic() > deadline:
        raise TimeoutError(
            f"AD adapter did not respond in time while {what}.\n"
            f"See log: {log_path}\n--- tail ---\n{_log_tail(log_path)}"
        )


def send_fifo(path: str, payload: bytes, proc: subprocess.Popen | None = None,
              log_path: str | None = None, timeout_s: float = 600.0,
              poll_s: float = 0.05) -> None:
    """Write ``payload`` to a FIFO once the adapter opens it for reading."""
    deadline = time.monotonic() + float(timeout_s)
    while True:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            break
        except OSError as error:
            if error.errno != errno.ENXIO:  # ENXIO: no reader yet
                raise
        _check_peer(proc, log_path, deadline, f"waiting to send on {path}")
        time.sleep(poll_s)
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            view = view[written:]
    finally:
        os.close(fd)


def recv_fifo(path: str, proc: subprocess.Popen | None = None,
              log_path: str | None = None, timeout_s: float = 600.0,
              poll_s: float = 0.5) -> bytes:
    """Read one complete message (until the writer closes) from a FIFO."""
    deadline = time.monotonic() + float(timeout_s)
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    chunks: list[bytes] = []
    try:
        while True:
            readable, _, _ = select.select([fd], [], [], poll_s)
            if readable:
                try:
                    data = os.read(fd, 1 << 20)
                except BlockingIOError:
                    data = None
                if data:
                    chunks.append(data)
                    continue
                if data == b"":
                    # EOF: the writer connected and closed.
                    return b"".join(chunks)
            _check_peer(proc, log_path, deadline, f"waiting for {path}")
    finally:
        os.close(fd)
