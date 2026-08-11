"""Closed-loop scoring (vendored hugsim_evaluate, now part of aether)."""
from __future__ import annotations

from aether.eval.score import hugsim_evaluate


def evaluate(save_data, ground_xyz, scene_xyz):
    return hugsim_evaluate(save_data, ground_xyz, scene_xyz)
