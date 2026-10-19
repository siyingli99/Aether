"""Vehicle profiles: geometry and actuation limits of an ego vehicle.

All lengths are metres in the vehicle frame (origin at the rear-axle centre
on the ground, x forward).  ``configs/vehicles/<name>.yaml`` holds one
profile; a scene selects the vehicle that recorded it (nuScenes: Renault
Zoe, PandaSet: Chrysler Pacifica), so the ego box always matches the camera
rig the reconstruction was trained with.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
from pathlib import Path

import yaml

from ..paths import REPO_ROOT

PROFILE_DIR = REPO_ROOT / "configs" / "vehicles"


@dataclass(frozen=True)
class VehicleProfile:
    name: str
    length: float
    width: float
    height: float
    wheelbase: float
    rear_overhang: float
    max_steer: float = math.radians(35.0)    # road-wheel angle, rad
    max_steer_rate: float = 0.7              # rad/s
    max_accel: float = 3.0                   # m/s^2
    max_decel: float = 8.0                   # m/s^2 (positive number)
    max_speed: float = 30.0                  # m/s
    max_lat_accel: float = 8.0               # m/s^2, tyre-friction limit (~0.8 g)

    @property
    def center_offset(self) -> float:
        """Distance from the rear axle to the box centre along +x."""
        return 0.5 * self.length - self.rear_overhang

    @property
    def track(self) -> float:
        return self.width * 0.85

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def list_profiles() -> list[str]:
    return sorted(path.stem for path in PROFILE_DIR.glob("*.yaml"))


def load_profile(value: str | dict | VehicleProfile) -> VehicleProfile:
    """Load a profile by name (``configs/vehicles/<name>.yaml``) or from a dict."""
    if isinstance(value, VehicleProfile):
        return value
    if isinstance(value, dict):
        data = dict(value)
        if "base" in data:
            base = load_profile(data.pop("base")).to_dict()
            base.update(data)
            data = base
    else:
        path = Path(str(value))
        if not path.suffix:
            path = PROFILE_DIR / f"{value}.yaml"
        if not path.is_file():
            raise FileNotFoundError(
                f"vehicle profile {value!r} not found; available: {list_profiles()}")
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data.setdefault("name", path.stem)
    for key in ("max_steer",):
        if key + "_deg" in data:
            data[key] = math.radians(float(data.pop(key + "_deg")))
    known = {f.name for f in fields(VehicleProfile)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown vehicle profile fields: {unknown}")
    profile = VehicleProfile(**{key: (str(v) if key == "name" else float(v)) for key, v in data.items()})
    if profile.wheelbase <= 0 or profile.length <= profile.wheelbase:
        raise ValueError(f"{profile.name}: need 0 < wheelbase < length")
    return profile
