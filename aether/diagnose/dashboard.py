"""structlog + rich.Live closed-loop dashboard.

Lightweight wrappers used by ``scripts/closed_loop.py``-style drivers to
emit structured JSON logs (via ``structlog`` if available, ``logging``
otherwise) and render a per-step progress dashboard with ``rich.Live``.

All UI helpers degrade gracefully to plain ``print`` when the optional
dependencies are missing so they're safe to import inside the perf-critical
inner loop.
"""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

try:                              # optional deps
    import structlog              # type: ignore
    _HAS_STRUCTLOG = True
except Exception:                 # pragma: no cover
    structlog = None              # type: ignore
    _HAS_STRUCTLOG = False

try:
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    _HAS_RICH = True
except Exception:                 # pragma: no cover
    Console = None                # type: ignore
    Live = None                   # type: ignore
    Table = None                  # type: ignore
    _HAS_RICH = False


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def get_logger(name: str = "aether"):
    """Return a structlog logger if available, else a stdlib-style shim."""
    if _HAS_STRUCTLOG:
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer(),
            ],
        )
        return structlog.get_logger(name)

    class _Shim:
        def _log(self, level: str, msg: str, **kw):
            payload = {"ts": time.time(), "level": level,
                       "logger": name, "event": msg, **kw}
            print(json.dumps(payload, default=str), file=sys.stderr, flush=True)
        def debug(self, msg, **kw): self._log("debug", msg, **kw)
        def info(self, msg, **kw): self._log("info", msg, **kw)
        def warning(self, msg, **kw): self._log("warning", msg, **kw)
        def error(self, msg, **kw): self._log("error", msg, **kw)
    return _Shim()


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@dataclass
class StepStat:
    step: int = 0
    frame: int = 0
    ego_x: float = 0.0
    ego_y: float = 0.0
    ego_v: float = 0.0
    ad_ms: float = 0.0
    render_ms: float = 0.0
    bev_ms: float = 0.0
    collision: bool = False
    notes: str = ""
    extras: Dict[str, Any] = field(default_factory=dict)


class Dashboard:
    """Per-step rich.Live table; falls back to one-line stderr prints."""

    def __init__(self, scenario: str, adapter: str, *, refresh: float = 4.0):
        self.scenario = scenario
        self.adapter = adapter
        self.refresh_per_second = refresh
        self._stat = StepStat()
        self._t0 = time.time()
        self._live = None
        self._console = None
        if _HAS_RICH:
            self._console = Console(stderr=True)
            self._live = Live(self._render(), console=self._console,
                              refresh_per_second=refresh)

    def __enter__(self):
        if self._live is not None:
            self._live.__enter__()
        return self

    def __exit__(self, *exc):
        if self._live is not None:
            try:
                self._live.__exit__(*exc)
            except Exception:
                pass
        return False

    def update(self, **kw) -> None:
        for k, v in kw.items():
            if hasattr(self._stat, k):
                setattr(self._stat, k, v)
            else:
                self._stat.extras[k] = v
        if self._live is not None:
            self._live.update(self._render())
        else:
            self._fallback()

    def _render(self):
        if not _HAS_RICH:
            return ""
        t = Table(title=f"AETHER  {self.scenario}  /  {self.adapter}")
        t.add_column("metric", style="bold cyan")
        t.add_column("value", style="green")
        s = self._stat
        elapsed = time.time() - self._t0
        rows = [
            ("step", str(s.step)),
            ("frame", str(s.frame)),
            ("ego (x,y)", f"{s.ego_x:7.2f}, {s.ego_y:7.2f}"),
            ("velo m/s", f"{s.ego_v:6.2f}"),
            ("ad ms", f"{s.ad_ms:8.1f}"),
            ("render ms", f"{s.render_ms:8.1f}"),
            ("bev ms", f"{s.bev_ms:8.1f}"),
            ("collision", "[red]YES[/]" if s.collision else "no"),
            ("elapsed s", f"{elapsed:7.1f}"),
            ("notes", s.notes or ""),
        ]
        for k, v in rows:
            t.add_row(k, v)
        return t

    def _fallback(self) -> None:
        s = self._stat
        print(f"[dash] step={s.step} frame={s.frame} v={s.ego_v:.2f} "
              f"ad={s.ad_ms:.0f}ms render={s.render_ms:.0f}ms "
              f"col={s.collision}", file=sys.stderr, flush=True)


@contextmanager
def dashboard(scenario: str, adapter: str, *, refresh: float = 4.0):
    """Context-manager helper: ``with dashboard(...) as d: d.update(...)``."""
    dash = Dashboard(scenario, adapter, refresh=refresh)
    with dash:
        yield dash
