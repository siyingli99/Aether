"""AETHER-Diag v2 -- 13-D semantic perturbation dictionary."""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


SEMANTIC_VAR_NAMES: Tuple[str, ...] = (
    "delta_s_lon", "delta_d_lat", "delta_yaw", "scale",
    "delta_v_lon", "omega_yaw",
    "brightness", "contrast", "saturation", "hue_rotate",
    "opacity_fade", "sh_jitter", "fog",
)
TOPO_DIMS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
PHOTO_DIMS: Tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12)
DIM = 13

SEMANTIC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "delta_s_lon":  (-5.0, 5.0),
    "delta_d_lat":  (-3.0, 3.0),
    "delta_yaw":    (-0.5, 0.5),
    "scale":        (0.7, 1.5),
    "delta_v_lon":  (-5.0, 5.0),
    "omega_yaw":    (-1.0, 1.0),
    "brightness":   (-0.5, 0.5),
    "contrast":     (0.5, 1.5),
    "saturation":   (0.0, 2.0),
    "hue_rotate":   (-math.pi / 2, math.pi / 2),
    "opacity_fade": (0.0, 0.9),
    "sh_jitter":    (0.0, 0.3),
    "fog":          (0.0, 0.8),
}

SEMANTIC_NEUTRAL: Dict[str, float] = {
    "delta_s_lon": 0.0, "delta_d_lat": 0.0, "delta_yaw": 0.0, "scale": 1.0,
    "delta_v_lon": 0.0, "omega_yaw": 0.0,
    "brightness": 0.0, "contrast": 1.0, "saturation": 1.0, "hue_rotate": 0.0,
    "opacity_fade": 0.0, "sh_jitter": 0.0, "fog": 0.0,
}


def neutral_vec() -> np.ndarray:
    return np.array([SEMANTIC_NEUTRAL[n] for n in SEMANTIC_VAR_NAMES], dtype=np.float64)


def per_axis_range() -> np.ndarray:
    rng = np.zeros(DIM, dtype=np.float64)
    for i, name in enumerate(SEMANTIC_VAR_NAMES):
        lo, hi = SEMANTIC_BOUNDS[name]
        rng[i] = hi - lo
    return rng


@dataclass
class SemanticPerturbation:
    z: np.ndarray = field(default_factory=neutral_vec)

    def __post_init__(self) -> None:
        z = np.asarray(self.z, dtype=np.float64).reshape(-1)
        if z.size != DIM:
            raise ValueError(f"SemanticPerturbation expects {DIM} dims, got {z.size}")
        self.z = z

    @classmethod
    def zero(cls) -> "SemanticPerturbation":
        return cls(z=neutral_vec())

    def project(self) -> "SemanticPerturbation":
        z = self.z.copy()
        for i, name in enumerate(SEMANTIC_VAR_NAMES):
            lo, hi = SEMANTIC_BOUNDS[name]
            z[i] = float(np.clip(z[i], lo, hi))
        return SemanticPerturbation(z=z)

    def freeze_topo(self) -> "SemanticPerturbation":
        z = self.z.copy()
        for i in TOPO_DIMS:
            z[i] = SEMANTIC_NEUTRAL[SEMANTIC_VAR_NAMES[i]]
        return SemanticPerturbation(z=z)

    def freeze_photo(self) -> "SemanticPerturbation":
        z = self.z.copy()
        for i in PHOTO_DIMS:
            z[i] = SEMANTIC_NEUTRAL[SEMANTIC_VAR_NAMES[i]]
        return SemanticPerturbation(z=z)

    def is_identity(self, atol: float = 1e-6) -> bool:
        return bool(np.allclose(self.z, neutral_vec(), atol=atol))

    def as_dict(self) -> Dict[str, float]:
        return {n: float(self.z[i]) for i, n in enumerate(SEMANTIC_VAR_NAMES)}


def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=np.float64)


def _yaw_delta_quat(yaw_offset: float) -> np.ndarray:
    half = 0.5 * yaw_offset
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def perturb_pose(c2w_0: np.ndarray, z: SemanticPerturbation, t: float,
                 v_lon0: float = 0.0) -> np.ndarray:
    """Apply topological semantics to a baseline pose; returns new (4,4) c2w."""
    c2w = np.asarray(c2w_0, dtype=np.float64).copy()
    R0 = c2w[:3, :3]
    mu0 = c2w[:3, 3]

    delta_s_lon = float(z.z[0])
    delta_d_lat = float(z.z[1])
    delta_yaw = float(z.z[2])
    scale = float(z.z[3])
    delta_v_lon = float(z.z[4])
    omega_yaw = float(z.z[5])

    local = np.array([
        delta_s_lon + delta_v_lon * t + (scale - 1.0) * 0.5,
        delta_d_lat,
        0.0,
    ], dtype=np.float64)
    mu_t = mu0 + R0 @ local

    yaw_offset = delta_yaw + omega_yaw * t
    if abs(yaw_offset) > 1e-9:
        from scipy.spatial.transform import Rotation as SCR
        q0 = SCR.from_matrix(R0).as_quat()  # (x, y, z, w)
        q0_wxyz = np.array([q0[3], q0[0], q0[1], q0[2]], dtype=np.float64)
        dq = _yaw_delta_quat(yaw_offset)
        qt_wxyz = _quat_mul(dq, q0_wxyz)
        qt_xyzw = np.array([qt_wxyz[1], qt_wxyz[2], qt_wxyz[3], qt_wxyz[0]])
        Rt = SCR.from_quat(qt_xyzw).as_matrix()
    else:
        Rt = R0

    out = np.eye(4, dtype=np.float32)
    out[:3, :3] = Rt.astype(np.float32)
    out[:3, 3] = mu_t.astype(np.float32)
    return out


class TargetGaussianHandle:
    """Locate the target instance's Gaussians and apply photometric edits."""

    def __init__(self, renderer, class_name: str, instance_id: int):
        self.renderer = renderer
        self.class_name = class_name
        self.instance_id = int(instance_id)
        self._model = (
            renderer.trainer.models.get(class_name) if renderer is not None else None
        )
        self._point_idx = None  # lazy

    def _resolve_idx(self):
        if self._point_idx is not None or self._model is None:
            return
        if not hasattr(self._model, "point_ids"):
            return
        ids = self._model.point_ids[..., 0]
        mask = (ids == self.instance_id)
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        self._point_idx = idx

    @contextmanager
    def apply_photometric(self, z: SemanticPerturbation):
        self._resolve_idx()
        m = self._model
        if m is None or self._point_idx is None or self._point_idx.numel() == 0:
            yield
            return
        if z.is_identity():
            yield
            return

        import torch
        idx = self._point_idx.to(m._features_dc.device)

        brightness = float(z.z[6])
        contrast = float(z.z[7])
        saturation = float(z.z[8])
        hue_rotate = float(z.z[9])
        opacity_fade = float(z.z[10])
        sh_jitter = float(z.z[11])
        fog = float(z.z[12])

        orig_dc = m._features_dc[idx].detach().clone()
        orig_op = m._opacities[idx].detach().clone()
        orig_rest = None
        if hasattr(m, "_features_rest") and m._features_rest is not None:
            try:
                orig_rest = m._features_rest[idx].detach().clone()
            except Exception:
                orig_rest = None

        try:
            with torch.no_grad():
                dc = m._features_dc[idx].clone()
                if abs(brightness) > 1e-9:
                    dc = dc + brightness
                if abs(contrast - 1.0) > 1e-9:
                    mean = dc.mean(dim=0, keepdim=True)
                    dc = mean + contrast * (dc - mean)
                if abs(hue_rotate) > 1e-9 and dc.shape[-1] >= 3:
                    rgb = dc[..., :3]
                    c, s = float(math.cos(hue_rotate)), float(math.sin(hue_rotate))
                    inv3 = 1.0 / 3.0
                    sqrt3 = math.sqrt(3.0)
                    r = rgb[..., 0]; g = rgb[..., 1]; b = rgb[..., 2]
                    a11 = c + (1 - c) * inv3
                    a12 = (1 - c) * inv3 - s / sqrt3
                    a13 = (1 - c) * inv3 + s / sqrt3
                    new_r = r * a11 + g * a12 + b * a13
                    new_g = r * a13 + g * a11 + b * a12
                    new_b = r * a12 + g * a13 + b * a11
                    rgb = torch.stack([new_r, new_g, new_b], dim=-1)
                    dc = torch.cat([rgb, dc[..., 3:]], dim=-1) if dc.shape[-1] > 3 else rgb
                if abs(saturation - 1.0) > 1e-9 and dc.shape[-1] >= 3:
                    rgb = dc[..., :3]
                    gray = rgb.mean(dim=-1, keepdim=True)
                    rgb = gray + saturation * (rgb - gray)
                    dc = torch.cat([rgb, dc[..., 3:]], dim=-1) if dc.shape[-1] > 3 else rgb
                if fog > 1e-9:
                    dc = (1.0 - fog) * dc
                m._features_dc[idx] = dc.clamp(-5.0, 5.0)
                if opacity_fade > 1e-9:
                    m._opacities[idx] = m._opacities[idx] * (1.0 - opacity_fade)
                if sh_jitter > 1e-9 and orig_rest is not None:
                    noise = torch.randn_like(m._features_rest[idx]) * sh_jitter
                    m._features_rest[idx] = m._features_rest[idx] + noise
            yield
        finally:
            with torch.no_grad():
                m._features_dc[idx] = orig_dc
                m._opacities[idx] = orig_op
                if orig_rest is not None:
                    m._features_rest[idx] = orig_rest
