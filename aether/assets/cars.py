"""Optional 3DRealCar asset import for Aether.

Pre-trained HUGSIM/Scaffold-GS cars are not redistributed. Set
``AETHER_CAR_ASSET_ROOT`` to a local 3DRealCar directory after obtaining the
assets under their original terms.

Each directory holds ``gs.pth`` (a Scaffold-GS tuple
``([sh_deg, xyz, fdc, frest, scaling, rotation, opacity, ...], iters)``) and
``wlh.json`` (``[width, length, height]`` in metres).

This module exposes a tiny registry so callers can ask for cars by short
name without hard-coding the full filesystem path.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_ROOT = Path(
    os.environ.get("AETHER_CAR_ASSET_ROOT", "~/.cache/aether/3DRealCar")
).expanduser()


@dataclass
class CarAsset:
    name: str
    gs_path: Path
    wlh: Tuple[float, float, float]   # width, length, height (metres)

    @property
    def width(self) -> float: return float(self.wlh[0])
    @property
    def length(self) -> float: return float(self.wlh[1])
    @property
    def height(self) -> float: return float(self.wlh[2])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "gs_path": str(self.gs_path),
            "wlh": list(self.wlh),
        }

    def load_gaussians(self, *, map_location: str = "cpu"):
        """Load the raw Scaffold-GS tuple. Heavy: ~150-230 MB CPU tensors."""
        import torch  # local import to avoid pulling torch at import time
        return torch.load(self.gs_path, map_location=map_location,
                          weights_only=False)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, CarAsset] = {}


def _build_default_registry(root: Optional[Path] = None) -> Dict[str, CarAsset]:
    root = Path(root) if root else _DEFAULT_ROOT
    entries: Dict[str, CarAsset] = {}
    for sub in ("2024_04_22_10_35_34", "2024_06_04_13_44_39"):
        d = root / sub
        gs = d / "gs.pth"
        wlh_path = d / "wlh.json"
        if not gs.exists() or not wlh_path.exists():
            continue
        try:
            wlh = tuple(json.loads(wlh_path.read_text()))[:3]
        except Exception:
            wlh = (1.85, 4.5, 1.6)
        entries[sub] = CarAsset(name=sub, gs_path=gs,
                                 wlh=(float(wlh[0]), float(wlh[1]), float(wlh[2])))
    return entries


def registry(refresh: bool = False) -> Dict[str, CarAsset]:
    global _REGISTRY
    if refresh or not _REGISTRY:
        _REGISTRY = _build_default_registry()
    return _REGISTRY


def list_cars() -> List[str]:
    return sorted(registry().keys())


def get(name: str) -> CarAsset:
    reg = registry()
    if name not in reg:
        raise KeyError(f"unknown 3DRealCar asset '{name}'; have {list(reg)}")
    return reg[name]


def make_actor_yaml(name: str, *, instance_id: int = 1, x: float = 10.0,
                    y: float = 0.0, yaw: float = 0.0, velo: float = 0.0,
                    policy: str = "logreplay") -> Dict[str, Any]:
    """Compose a scenario ``actors:`` entry from a registered car."""
    car = get(name)
    return {
        "class": "RigidNodes",
        "instance_id": int(instance_id),
        "size": [car.width, car.length, car.height],
        "asset": {"source": "3DRealCar", "name": car.name,
                  "gs_path": str(car.gs_path)},
        "state": {"x": float(x), "y": float(y), "yaw": float(yaw),
                  "velo": float(velo)},
        "policy": {"type": policy},
    }
