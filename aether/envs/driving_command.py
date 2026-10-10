"""Driving-command providers for closed-loop AD (UniAD / VAD).

UniAD's planning head conditions on a discrete high-level command:
  0 = turn right, 1 = turn left, 2 = go straight.

HUGSIM supplies this from ``ground_param.pkl`` (nearest log pose lookup).
AETHER can load the same pickle or derive commands from the processed
dataset ego trajectory under ``<scene_dir>/ego_pose/``.
"""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
from scipy.spatial.transform import Rotation as SCR


def _load_4x4(path: Path) -> np.ndarray:
    return np.loadtxt(str(path)).astype(np.float64).reshape(4, 4)


def _ds_to_bev_fwd_lat(T_ds: np.ndarray, dataset: str) -> np.ndarray:
    """Map a DS-world ego 4x4 to HUGSIM BEV (forward, lateral_right)."""
    if dataset == "nuscenes":
        # DS: x=right, z=forward -> BEV fwd=z, lat_right=x
        return np.array([float(T_ds[2, 3]), float(T_ds[0, 3])], dtype=np.float64)
    # pandaset DS: x=fwd, y=lat_right
    return np.array([float(T_ds[0, 3]), float(T_ds[1, 3])], dtype=np.float64)


def _commands_from_bev_path(xy_fwd_lat: np.ndarray,
                            planning_steps: int = 12,
                            lat_thresh: float = 2.0) -> np.ndarray:
    """NuScenes / UniAD-style command from future lateral offset in ego frame."""
    n = len(xy_fwd_lat)
    cmds = np.full(n, 2, dtype=np.int32)
    for i in range(n):
        end = min(i + planning_steps, n - 1)
        if end <= i:
            continue
        nxt = min(i + 1, n - 1)
        d = xy_fwd_lat[nxt] - xy_fwd_lat[i]
        if float(np.hypot(d[0], d[1])) < 1e-6:
            continue
        yaw = math.atan2(float(d[1]), float(d[0]))
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = xy_fwd_lat[end] - xy_fwd_lat[i]
        # In (forward, right) coordinates the right-hand normal to the
        # current heading (c, s) is (-s, c).  The opposite sign silently
        # swaps UniAD's right/left commands on curved roads.
        lat_right = -s * dx + c * dy
        if lat_right >= lat_thresh:
            cmds[i] = 0
        elif lat_right <= -lat_thresh:
            cmds[i] = 1
    return cmds


class DrivingCommandProvider(Protocol):
    def command_at(self, vab, vr, frame_idx: int) -> int: ...


class _NearestLookupProvider:
    """Shared nearest-pose lookup in BEV (forward, lateral_right)."""

    def __init__(self, ref_bev: np.ndarray, commands: np.ndarray):
        self.ref_bev = np.asarray(ref_bev, dtype=np.float64)
        self.commands = np.asarray(commands, dtype=np.int32)
        if len(self.ref_bev) != len(self.commands):
            raise ValueError("ref_bev and commands length mismatch")

    def _nearest(self, vab) -> int:
        cur = np.array([float(vab[1]), float(vab[0])], dtype=np.float64)
        d2 = np.sum((self.ref_bev - cur) ** 2, axis=1)
        return int(np.argmin(d2))

    def command_at(self, vab, vr, frame_idx: int) -> int:
        idx = self._nearest(vab)
        return int(self.commands[idx])


class GroundParamCommandProvider(_NearestLookupProvider):
    """Load HUGSIM ``ground_param.pkl``: (cam_poses, height, commands)."""

    def __init__(self, path: Path):
        with open(path, "rb") as f:
            cam_poses, _height, commands = pickle.load(f)
        cam_poses = np.asarray(cam_poses, dtype=np.float64)
        commands = np.asarray(commands, dtype=np.int32)
        ref_bev = cam_poses[:, :3, 3][:, [2, 0]]  # HUGSIM world: fwd=z, lat=x
        super().__init__(ref_bev, commands)
        print(f"[command] loaded ground_param.pkl ({len(commands)} poses) "
              f"from {path}", flush=True)


class EgoPoseCommandProvider(_NearestLookupProvider):
    """Derive commands from the processed logged trajectory in DS world."""

    def __init__(self, scene_dir: Path, inv_anchor: np.ndarray, dataset: str):
        if dataset == "nuscenes":
            pose_dir = scene_dir / "extrinsics"
            paths = sorted(pose_dir.glob("*_0.txt"))
        else:
            pose_dir = scene_dir / "ego_pose"
            paths = sorted(pose_dir.glob("*.txt"))
        if not pose_dir.is_dir():
            raise FileNotFoundError(f"logged pose dir not found: {pose_dir}")
        if not paths:
            raise FileNotFoundError(f"no logged pose files in {pose_dir}")
        bev = []
        for p in paths:
            raw = _load_4x4(p)
            T_ds = (inv_anchor @ raw).astype(np.float64)
            bev.append(_ds_to_bev_fwd_lat(T_ds, dataset))
        ref_bev = np.stack(bev, axis=0)
        commands = _commands_from_bev_path(ref_bev)
        super().__init__(ref_bev, commands)
        print(f"[command] derived from GT trajectory ({len(commands)} frames, "
              f"scene={scene_dir.name})", flush=True)


class FixedCommandProvider:
    def __init__(self, command: int = 2):
        self.command = int(command)

    def command_at(self, vab, vr, frame_idx: int) -> int:
        return self.command


def build_command_provider(
    scenario,
    *,
    scene_dir: Optional[Path] = None,
    inv_anchor: Optional[np.ndarray] = None,
    dataset: str = "pandaset",
) -> DrivingCommandProvider:
    """Pick provider from scenario fields (first match wins)."""
    for attr in ("ground_param", "ground_param_pkl", "lidar_poses_pkl"):
        p = getattr(scenario, attr, None)
        if p:
            path = Path(str(p))
            if not path.is_absolute():
                yaml_dir = getattr(scenario, "_yaml_dir", None)
                if yaml_dir:
                    path = Path(yaml_dir) / path
            if path.is_file():
                return GroundParamCommandProvider(path)

    cmd_mode = str(getattr(scenario, "command_mode", "gt_ego") or "gt_ego").lower()
    if cmd_mode in ("fixed", "straight"):
        return FixedCommandProvider(int(getattr(scenario, "fixed_command", 2)))

    if scene_dir is not None and inv_anchor is not None:
        try:
            return EgoPoseCommandProvider(Path(scene_dir), inv_anchor, dataset)
        except FileNotFoundError as e:
            print(f"[command] GT trajectory unavailable ({e}); "
                  f"falling back to straight", flush=True)

    print("[command] no ground_param / ego_pose; default command=2 (straight)",
          flush=True)
    return FixedCommandProvider(2)
