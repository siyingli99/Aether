"""Per-instance foreground controller (pose / kinematics / appearance / visibility).

Designed to be the single source of truth for editing trained 3DGS foreground
instances at runtime, without mutating the on-disk checkpoint and without
breaking any existing AETHER closed-loop / rendering flow.

Backed entirely by the existing :class:`DriveStudioRenderer` API (no new
dependencies on drivestudio internals beyond what aether/render already uses).
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------- public dataclasses -------------------------- #

@dataclass
class InstanceInfo:
    class_name: str
    instance_id: int
    size_whl: Tuple[float, float, float]
    n_visible_frames: int

    @property
    def key(self) -> Tuple[str, int]:
        return (self.class_name, int(self.instance_id))


@dataclass
class KinematicPerturbation:
    """Window-level kinematic edit applied as a per-frame pose offset.

    The edit is expressed in the actor's frame: lateral velocity, longitudinal
    acceleration, yaw rate, and a longitudinal-distance scale.
    """
    v_lat: float = 0.0
    a_lon: float = 0.0
    omega_yaw: float = 0.0
    kappa_len: float = 0.0
    v_lon0: float = 0.0
    dt: float = 0.1


@dataclass
class AppearancePerturbation:
    """Instance-level appearance edit (SH0 + opacity)."""
    I_glare: float = 0.0
    dH_camou: float = 0.0
    alpha_fade: float = 0.0


# ----------------------------- adapter for env.actors --------------------- #

class TrainedActorAdapter:
    """ActorReplay-compatible adapter exposing a *subset* of trained instances.

    AetherDrivingEnv reads ``self.actors.pose_at(frame_idx)`` to (a) inject
    actor-pose overrides and (b) build the BEV ``obj_boxes`` list. Trained
    instances normally bypass that loop; wrapping them in this adapter lets
    them surface in ``info["obj_boxes"]`` and become first-class citizens.
    """

    def __init__(self, controller: "ForegroundController",
                 only: Optional[Sequence[Tuple[str, int]]] = None,
                 inject_pose: bool = False):
        self.controller = controller
        self.only = None if only is None else set((c, int(i)) for c, i in only)
        # If inject_pose=False (default) the env will NOT call set_actor_pose;
        # the trained Gaussians render at their native baseline poses. Set to
        # True only when you want to forcibly snap the pose every frame.
        self.inject_pose = bool(inject_pose)

    def pose_at(self, frame_idx: int) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for inst in self.controller.list_instances():
            if self.only is not None and inst.key not in self.only:
                continue
            c2w = self.controller.pose_at(inst.class_name, inst.instance_id, frame_idx)
            if c2w is None:
                continue
            out.append({
                "class":       inst.class_name,
                "instance_id": int(inst.instance_id),
                "size":        list(inst.size_whl),
                "c2w":         c2w if self.inject_pose else c2w,
                "_native":     not self.inject_pose,
            })
        return out


# ----------------------------- main controller ----------------------------- #

class ForegroundController:
    """High-level operations on trained foreground instances."""

    KNOWN_CLASSES = ("RigidNodes", "SMPLNodes", "DeformableNodes")

    def __init__(self, renderer):
        self.renderer = renderer
        self._cached_instances: Optional[List[InstanceInfo]] = None

    # ----------- discovery ----------- #

    def _model(self, class_name: str):
        if self.renderer is None:
            return None
        return self.renderer.trainer.models.get(class_name)

    def list_instances(self, refresh: bool = False) -> List[InstanceInfo]:
        if self._cached_instances is not None and not refresh:
            return self._cached_instances
        out: List[InstanceInfo] = []
        for cls in self.KNOWN_CLASSES:
            m = self._model(cls)
            if m is None or not hasattr(m, "instances_fv"):
                continue
            fv = m.instances_fv  # (F, N) bool
            sizes = getattr(m, "instances_size", None)
            n_inst = fv.shape[1]
            for i in range(n_inst):
                vis = int(fv[:, i].sum().item())
                if vis == 0:
                    continue
                if sizes is not None and sizes.shape[0] > i:
                    s = sizes[i].detach().cpu().numpy().tolist()
                    # drivestudio stores (w, h, l)? actually (w, l, h) varies
                    # by node. We treat the tuple verbatim and document.
                    s = tuple(float(x) for x in s)
                else:
                    s = (1.6, 1.5, 4.0)
                out.append(InstanceInfo(class_name=cls, instance_id=int(i),
                                        size_whl=s, n_visible_frames=vis))
        self._cached_instances = out
        return out

    def pose_at(self, class_name: str, instance_id: int,
                frame_idx: int) -> Optional[np.ndarray]:
        """Return the baseline 4x4 c2w in world frame, or None if instance not visible."""
        m = self._model(class_name)
        if m is None or not hasattr(m, "instances_trans"):
            return None
        F = m.instances_trans.shape[0]
        f = int(np.clip(int(frame_idx), 0, F - 1))
        if not bool(m.instances_fv[f, int(instance_id)].item()):
            # Instance not visible at this frame; fall back to nearest visible.
            fv = m.instances_fv[:, int(instance_id)].detach().cpu().numpy()
            visible = np.flatnonzero(fv)
            if visible.size == 0:
                return None
            f = int(visible[np.argmin(np.abs(visible - f))])
        t = m.instances_trans[f, int(instance_id)].detach().cpu().numpy()
        q_wxyz = m.instances_quats[f, int(instance_id)].detach().cpu().numpy()
        # SMPL stores instances_quats with an extra trailing axis (..., 1, 4).
        q_wxyz = np.asarray(q_wxyz).reshape(-1)[:4]
        from scipy.spatial.transform import Rotation as SCR
        q_xyzw = np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]],
                          dtype=np.float64)
        R = SCR.from_quat(q_xyzw).as_matrix()
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = R.astype(np.float32)
        c2w[:3, 3] = t.astype(np.float32)
        return c2w

    def nearest_at(self, frame_idx: int, ego_xy_world: Tuple[float, float],
                   max_n: int = 8, max_dist: float = 60.0) -> List[InstanceInfo]:
        """Return up to ``max_n`` instances closest to ``ego_xy_world`` (DS-world XY)."""
        cand = []
        for inst in self.list_instances():
            c2w = self.pose_at(inst.class_name, inst.instance_id, frame_idx)
            if c2w is None:
                continue
            dx = float(c2w[0, 3]) - float(ego_xy_world[0])
            dy = float(c2w[1, 3]) - float(ego_xy_world[1])
            d = math.hypot(dx, dy)
            if d > max_dist:
                continue
            cand.append((d, inst))
        cand.sort(key=lambda x: x[0])
        return [c[1] for c in cand[:max_n]]

    # ----------- pose override ----------- #

    def set_pose_override(self, class_name: str, instance_id: int,
                          frame_idx: int, c2w: np.ndarray) -> None:
        self.renderer.set_actor_pose(class_name, int(instance_id),
                                     int(frame_idx), c2w)

    def clear_pose_overrides(self) -> None:
        self.renderer.clear_actor_overrides()

    def apply_kinematic_window(self, class_name: str, instance_id: int,
                               frames: Sequence[int],
                               kin: KinematicPerturbation,
                               base_frame: Optional[int] = None) -> None:
        """Stamp per-frame pose overrides for ``frames`` derived from ``kin``.

        ``base_frame`` (default = first of ``frames``) gives the c2w used as
        anchor for the integration; the offset at frame ``f`` is computed
        with ``t = (f - base_frame) * kin.dt``.
        """
        if not frames:
            return
        bf = int(frames[0]) if base_frame is None else int(base_frame)
        c2w0 = self.pose_at(class_name, instance_id, bf)
        if c2w0 is None:
            return
        from scipy.spatial.transform import Rotation as SCR
        base_rotation = np.asarray(c2w0[:3, :3], dtype=np.float64)
        base_translation = np.asarray(c2w0[:3, 3], dtype=np.float64)
        for f in frames:
            t = (int(f) - bf) * float(kin.dt)
            longitudinal = (
                float(kin.v_lon0) * t + 0.5 * float(kin.a_lon) * t * t
            ) * (1.0 + float(kin.kappa_len))
            lateral = float(kin.v_lat) * t
            c2w_t = np.eye(4, dtype=np.float32)
            c2w_t[:3, :3] = (
                base_rotation @ SCR.from_euler("z", float(kin.omega_yaw) * t).as_matrix()
            ).astype(np.float32)
            c2w_t[:3, 3] = (
                base_translation + base_rotation @ np.asarray(
                    [longitudinal, lateral, 0.0], dtype=np.float64
                )
            ).astype(np.float32)
            self.set_pose_override(class_name, int(instance_id), int(f), c2w_t)

    # ----------- appearance ----------- #

    @contextmanager
    def apply_appearance(self, class_name: str, instance_id: int,
                         appearance: AppearancePerturbation) -> Iterator[None]:
        """Snapshot/edit/restore SH0 + opacity for one instance."""
        m = self._model(class_name)
        if m is None or not hasattr(m, "_features_dc") or not hasattr(m, "point_ids"):
            yield
            return
        import torch
        ids = m.point_ids[..., 0]
        mask = (ids == int(instance_id))
        if not bool(mask.any()):
            yield
            return
        idx = mask.nonzero(as_tuple=False).squeeze(-1)
        I_glare = float(appearance.I_glare)
        dH = float(appearance.dH_camou)
        a_fade = float(appearance.alpha_fade)
        orig_dc = m._features_dc[idx].detach().clone()
        orig_op = m._opacities[idx].detach().clone()
        try:
            with torch.no_grad():
                m._features_dc[idx] = m._features_dc[idx] + I_glare * 0.5
                if abs(dH) > 1e-9 and m._features_dc.shape[-1] >= 3:
                    c, s = float(np.cos(dH)), float(np.sin(dH))
                    rgb = m._features_dc[idx, :3]
                    rotated = torch.stack([
                        c * rgb[:, 0] - s * rgb[:, 1],
                        s * rgb[:, 0] + c * rgb[:, 1],
                        rgb[:, 2],
                    ], dim=-1)
                    m._features_dc[idx, :3] = rotated
                m._features_dc[idx] = m._features_dc[idx].clamp(-5.0, 5.0)
                if a_fade > 0:
                    m._opacities[idx] = m._opacities[idx] * (1.0 - a_fade)
            yield
        finally:
            with torch.no_grad():
                m._features_dc[idx] = orig_dc
                m._opacities[idx] = orig_op

    # ----------- visibility ----------- #

    @contextmanager
    def hide_instances(self, hidden: Sequence[Tuple[str, int]],
                       frames: Optional[Sequence[int]] = None) -> Iterator[None]:
        """Toggle ``instances_fv`` to False for ``hidden`` over ``frames``.

        If ``frames`` is None, hides over the entire timeline. Restores on
        exit so callers are free to nest / reuse the controller.
        """
        import torch
        snapshots: List[Tuple[Any, Any, Any]] = []
        try:
            for cls, inst in hidden:
                m = self._model(cls)
                if m is None or not hasattr(m, "instances_fv"):
                    continue
                fv = m.instances_fv
                if int(inst) >= fv.shape[1]:
                    continue
                if frames is None:
                    orig = fv[:, int(inst)].detach().clone()
                    snapshots.append((m, int(inst), None, orig))
                    with torch.no_grad():
                        fv[:, int(inst)] = False
                else:
                    fr = torch.as_tensor(list(frames), dtype=torch.long, device=fv.device)
                    fr = fr.clamp(0, fv.shape[0] - 1)
                    orig = fv[fr, int(inst)].detach().clone()
                    snapshots.append((m, int(inst), fr, orig))
                    with torch.no_grad():
                        fv[fr, int(inst)] = False
            yield
        finally:
            for entry in snapshots:
                m, inst, fr, orig = entry
                with torch.no_grad():
                    if fr is None:
                        m.instances_fv[:, inst] = orig
                    else:
                        m.instances_fv[fr, inst] = orig
