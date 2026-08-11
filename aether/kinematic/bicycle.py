"""Kinematic bicycle ego model.  Identical equations to HUGSIM, isolated here
to keep the env class small."""
from __future__ import annotations
from copy import deepcopy
import math
import numpy as np


class BicycleEgo:
    def __init__(self, kinematic_cfg, scenario_cfg):
        # SI units everywhere.
        self.dt = float(kinematic_cfg.dt)
        self.Lr = float(kinematic_cfg.Lr)
        self.Lf = float(kinematic_cfg.Lf)
        self.fric = float(kinematic_cfg.fric)
        self.min_steer = -math.radians(float(kinematic_cfg.min_steer))
        self.max_steer =  math.radians(float(kinematic_cfg.max_steer))
        self.min_acc = float(kinematic_cfg.min_acc)
        self.max_acc = float(kinematic_cfg.max_acc)
        # HUGSIM does not clip acc/steer inside step(); keep clipping opt-in.
        self.clip_acc = bool(getattr(kinematic_cfg, "clip_acc", False))
        self.clip_steer = bool(getattr(kinematic_cfg, "clip_steer", False))

        # Initial state, mirroring HUGSIM convention:
        #  vab = [a, b]  -> ego (x, y) in env frame (x=fwd, y=left)
        #  vr  = [roll, yaw, pitch] in radians (XYZ Euler)
        self.start_vab = np.asarray(scenario_cfg.start_ab, dtype=np.float32)
        self.start_vr  = np.asarray(scenario_cfg.start_euler, dtype=np.float32) * np.pi / 180.0
        self.start_velo  = float(scenario_cfg.start_velo)
        self.start_steer = float(getattr(scenario_cfg, "start_steer", 0.0))
        self.reset()

    def reset(self):
        self.vab   = deepcopy(self.start_vab)
        self.vr    = deepcopy(self.start_vr)
        self.velo  = float(self.start_velo)
        self.steer = float(self.start_steer)
        self.last_acc = 0.0
        self.last_steer_rate = 0.0

    def step(self, acc: float, steer_rate: float):
        acc = float(acc)
        steer_rate = float(steer_rate)
        if self.clip_acc:
            acc = float(np.clip(acc, self.min_acc, self.max_acc))
        self.last_acc, self.last_steer_rate = acc, steer_rate
        L = self.Lr + self.Lf
        self.velo  += acc * self.dt
        self.steer += steer_rate * self.dt
        if self.clip_steer:
            self.steer = float(np.clip(self.steer, self.min_steer, self.max_steer))
        theta = float(self.vr[1])
        self.vab[0] += self.velo * np.sin(theta) * self.dt
        self.vab[1] += self.velo * np.cos(theta) * self.dt
        self.vr[1] = theta + self.velo * np.tan(self.steer) / L * self.dt
