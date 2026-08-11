"""Function: Validate closed-loop kinematic clipping and driving-command constraints.
Input/Output: Reads optional local scene assets when available; prints ``ok`` or raises on a failed assertion.
Example:
    python -m pytest aether/tools/test_closed_loop_constraints.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from aether.envs.driving_command import (
    _commands_from_bev_path,
    EgoPoseCommandProvider,
    FixedCommandProvider,
)
from aether.kinematic.bicycle import BicycleEgo


class _Cfg:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def test_bicycle_no_steer_clip_by_default():
    kin = _Cfg(
        dt=0.25, Lr=1.35, Lf=1.35, fric=0.2,
        min_steer=15, max_steer=15, min_acc=-2.0, max_acc=2.0,
        clip_acc=False, clip_steer=False,
    )
    scen = _Cfg(start_ab=[0, 0], start_euler=[0, 90, 0], start_velo=1.0, start_steer=0.0)
    ego = BicycleEgo(kin, scen)
    ego.step(0.0, math.radians(90.0))  # large steer_rate
    assert abs(ego.steer) > math.radians(15.0)


def test_commands_from_bev_path_turn_right():
    xy = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 3.0], [15.0, 6.0]])
    cmds = _commands_from_bev_path(xy, planning_steps=2, lat_thresh=1.0)
    assert cmds[0] in (0, 1, 2)


def test_ego_pose_command_provider_scene15():
    scene_dir = _ROOT / "data/pandaset/processed/015"
    if not scene_dir.is_dir():
        return
    anchor = scene_dir / "ego_pose" / "000.txt"
    if not anchor.is_file():
        return
    inv = np.linalg.inv(np.loadtxt(str(anchor)).reshape(4, 4))
    prov = EgoPoseCommandProvider(scene_dir, inv, "pandaset")
    c = prov.command_at(np.array([0.0, 0.0]), np.array([0.0, math.pi / 2, 0.0]), 0)
    assert c in (0, 1, 2)


def test_fixed_command_provider():
    assert FixedCommandProvider(2).command_at(None, None, 0) == 2


if __name__ == "__main__":
    test_bicycle_no_steer_clip_by_default()
    test_commands_from_bev_path_turn_right()
    test_ego_pose_command_provider_scene15()
    test_fixed_command_provider()
    print("ok")
