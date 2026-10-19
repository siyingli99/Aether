"""Timeout-aware named-pipe transport for external planner processes.

A plain ``open(fifo)`` blocks forever when the peer has died.  These helpers
poll the planner process while waiting, so a crashed planner becomes an error
carrying its log tail instead of a hung closed loop.  One message per open:
the writer opens, writes, closes; the reader reads until EOF.
"""
from __future__ import annotations

import errno
import fcntl
import os
import select
import subprocess
import time
from pathlib import Path


def log_tail(path: str | Path | None, limit: int = 4000) -> str:
    if not path:
        return ""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")[-limit:]
    except OSError:
        return ""


class PeerError(RuntimeError):
    pass


def check_peer(proc: subprocess.Popen | None, log: str | Path | None,
               deadline: float | None = None, what: str = "") -> None:
    if proc is not None and proc.poll() is not None:
        raise PeerError(f"planner exited with code {proc.returncode} while {what}.\n"
                        f"log: {log}\n--- tail ---\n{log_tail(log)}")
    if deadline is not None and time.monotonic() > deadline:
        raise TimeoutError(f"planner did not respond in time while {what}.\n"
                           f"log: {log}\n--- tail ---\n{log_tail(log)}")


def make_pipes(directory: str | Path, names=("obs_pipe", "plan_pipe")) -> tuple[str, ...]:
    out = []
    for name in names:
        path = os.path.join(directory, name)
        if not os.path.exists(path):
            os.mkfifo(path)
        out.append(path)
    return tuple(out)


def send(path: str, payload: bytes, proc=None, log=None, timeout_s: float = 600.0,
         poll_s: float = 0.05) -> None:
    deadline = time.monotonic() + float(timeout_s)
    while True:
        try:
            fd = os.open(path, os.O_WRONLY | os.O_NONBLOCK)
            break
        except OSError as error:
            if error.errno != errno.ENXIO:          # ENXIO: no reader yet
                raise
        check_peer(proc, log, deadline, f"waiting to send on {path}")
        time.sleep(poll_s)
    try:
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
        view = memoryview(payload)
        while view:
            view = view[os.write(fd, view):]
    finally:
        os.close(fd)


def receive(path: str, proc=None, log=None, timeout_s: float = 600.0,
            poll_s: float = 0.5) -> bytes:
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
                    return b"".join(chunks)
            check_peer(proc, log, deadline, f"waiting for {path}")
    finally:
        os.close(fd)


def wait_ready(proc: subprocess.Popen, log: str | Path, markers: list[str],
               timeout_s: float = 900.0, poll_s: float = 0.5) -> None:
    """Block until the planner log contains one of ``markers`` (case-insensitive)."""
    markers = [m.lower() for m in markers]
    deadline = time.monotonic() + timeout_s
    while True:
        check_peer(proc, log, deadline, "starting up")
        text = log_tail(log, limit=1 << 20).lower()
        if any(m in text for m in markers):
            return
        time.sleep(poll_s)
