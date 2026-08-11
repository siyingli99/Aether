"""Adapter registry & launcher.

Loads declarative ``configs/adapters/<name>.yaml`` and provides a uniform
interface to launch the adapter sub-process. Backward-compatible with the
legacy ``cfg.base.{name}_path`` shell-launcher field — if the YAML is missing
we fall back to the legacy field.
"""
from __future__ import annotations
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CFG_DIR = os.path.normpath(
    os.path.join(_HERE, "..", "..", "configs", "adapters"))


@dataclass
class AdapterSpec:
    name: str
    display_name: str = ""
    launcher: Dict[str, Any] = field(default_factory=dict)
    ipc: Dict[str, Any] = field(default_factory=dict)
    observation: Dict[str, Any] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    capabilities: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    @property
    def entrypoint(self) -> str:
        return str(self.launcher.get("entrypoint", ""))

    @property
    def conda_env(self) -> Optional[str]:
        v = self.launcher.get("conda_env")
        return str(v) if v else None

    @property
    def cwd(self) -> Optional[str]:
        v = self.launcher.get("cwd")
        return str(v) if v else None

    @property
    def needs_depth(self) -> bool:
        return bool(self.observation.get("needs_depth", False))

    @property
    def cameras(self) -> List[str]:
        v = self.observation.get("cameras", [])
        return [str(c) for c in v] if v else []

    def render_args(self, cuda_id: str, output_dir: str) -> List[str]:
        raw = self.launcher.get("args") or ["${cuda_id}", "${output_dir}"]
        sub = {"cuda_id": str(cuda_id), "output_dir": output_dir}
        out: List[str] = []
        for a in raw:
            s = str(a)
            for k, v in sub.items():
                s = s.replace("${" + k + "}", v)
            out.append(s)
        return out

    def render_env(self) -> Dict[str, str]:
        env = dict(os.environ)
        for k, v in (self.launcher.get("env") or {}).items():
            env[str(k)] = str(v)
        return env


def adapter_dir() -> str:
    return _DEFAULT_CFG_DIR


def list_adapters(cfg_dir: Optional[str] = None) -> List[str]:
    d = cfg_dir or _DEFAULT_CFG_DIR
    if not os.path.isdir(d):
        return []
    return sorted(os.path.splitext(f)[0]
                  for f in os.listdir(d) if f.endswith(".yaml"))


def load_adapter(name: str,
                 cfg_dir: Optional[str] = None,
                 legacy_cfg: Any = None) -> AdapterSpec:
    """Load `<cfg_dir>/<name>.yaml`. Falls back to ``legacy_cfg.{name}_path``."""
    d = cfg_dir or _DEFAULT_CFG_DIR
    path = os.path.join(d, f"{name}.yaml")
    if os.path.isfile(path):
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        launcher = dict(data.get("launcher") or {})
        repo_root = Path(path).resolve().parents[2]
        for key in ("entrypoint", "cwd"):
            value = launcher.get(key)
            if not value:
                continue
            expanded = Path(os.path.expandvars(os.path.expanduser(str(value))))
            if not expanded.is_absolute():
                expanded = repo_root / expanded
            launcher[key] = str(expanded.resolve())
        spec = AdapterSpec(
            name=str(data.get("name", name)),
            display_name=str(data.get("display_name", name)),
            launcher=launcher,
            ipc=dict(data.get("ipc") or {}),
            observation=dict(data.get("observation") or {}),
            output=dict(data.get("output") or {}),
            capabilities=dict(data.get("capabilities") or {}),
            source_path=path,
        )
        return spec
    # Legacy fallback
    if legacy_cfg is not None:
        ep = getattr(legacy_cfg, f"{name}_path", None)
        if ep:
            return AdapterSpec(name=name, display_name=name,
                               launcher={"type": "shell", "entrypoint": ep})
    raise FileNotFoundError(
        f"adapter '{name}' not found in {d} and no legacy *_path fallback")


def launch_adapter(spec: AdapterSpec,
                   cuda_id: str,
                   output_dir: str) -> subprocess.Popen:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "ad_output.txt")
    f = open(log_path, "w")
    args = spec.render_args(cuda_id, output_dir)
    cmd = ["bash", spec.entrypoint, *args]
    proc = subprocess.Popen(
        cmd, stdout=f, stderr=f,
        cwd=spec.cwd, env=spec.render_env(),
    )
    print(f"[ad] launched {spec.name} ({spec.entrypoint}) "
          f"pid={proc.pid} -> {log_path}")
    return proc
