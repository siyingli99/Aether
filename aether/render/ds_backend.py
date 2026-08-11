"""DriveStudio renderer wrapper for closed-loop inference.

This module rebuilds a DriveStudio `MultiTrainer` from a saved
`checkpoint_final.pth` and exposes a single `render(c2w, K, H, W, frame_idx,
cam_name=...) -> dict` call that is fast enough to use inside a gym step loop.

Key design points
-----------------

1.  We do *not* instantiate a DrivingDataset (which requires raw images and
    LiDAR on disk).  Everything the trainer needs at inference time
    (`num_train_images`, `num_full_images`, `num_timesteps`, `scene_aabb`,
    `test_set_indices`) can be reconstructed from the checkpoint state +
    `config.yaml` next to it.

2.  Per-frame actor poses can be overridden via `set_actor_pose(class_name,
    instance_id, frame_idx, c2w_4x4)`.  This is how the env injects scripted
    NPC trajectories on top of the (otherwise frozen) RigidNodes / SMPLNodes
    learned during training.  When no override is provided, the baked
    trajectory is used (i.e. classic replay).

3.  `image_infos` for novel views uses `img_idx = frame_idx` clamped into
    the original training image-index space (so the AffineTransform MLP
    receives a defined embedding).  Camera-pose perturbation modules
    (`CamPosePerturb`, `CamPose`) are bypassed via `novel_view=True`.

4.  Sky composite + per-camera affine are produced by the trainer's own
    `forward()`, so the output already matches DriveStudio's eval video.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


@dataclass
class CameraSpec:
    """Single ego-mounted camera at inference time."""
    name: str
    H: int
    W: int
    K: np.ndarray              # (3, 3) intrinsics in pixels
    v2c: np.ndarray            # (4, 4) ego/vehicle -> camera (OpenCV convention: +z fwd)


class DriveStudioRenderer:
    """Inference-only wrapper around drivestudio's MultiTrainer."""

    def __init__(self, ckpt_path: str | Path, device: str = "cuda"):
        import os
        from utils.misc import import_str  # drivestudio util
        from aether.paths import DRIVESTUDIO_ROOT

        # drivestudio's SMPL/sky/AABB loaders use paths relative to the
        # drivestudio repo root (e.g. "smpl_models/SMPL_NEUTRAL.pkl"),
        # so we must run with that as cwd while the trainer is being built.
        _prev_cwd = os.getcwd()
        os.chdir(str(DRIVESTUDIO_ROOT))
        ckpt_path = Path(ckpt_path).expanduser().resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(ckpt_path)
        self.ckpt_path = ckpt_path
        self.log_dir = ckpt_path.parent
        self.device = torch.device(device)

        cfg_path = self.log_dir / "config.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"Expected drivestudio training config at {cfg_path}; "
                "AETHER reads scene metadata from it (num_timesteps, aabb, ...)."
            )
        self.cfg: DictConfig = OmegaConf.load(cfg_path)

        # Recover the per-checkpoint metadata that the trainer constructor
        # would normally get from a DrivingDataset.
        meta = self._extract_metadata(ckpt_path)
        self._frame_count = meta["num_timesteps"]
        self._image_count_full = meta["num_full_images"]
        self._image_count_train = meta["num_train_images"]
        self._cams_per_frame = max(1, meta["num_full_images"] // max(meta["num_timesteps"], 1))

        # Build trainer with the same config used during training.
        self.trainer = import_str(self.cfg.trainer.type)(
            **self.cfg.trainer,
            num_timesteps=meta["num_timesteps"],
            model_config=self.cfg.model,
            num_train_images=meta["num_train_images"],
            num_full_images=meta["num_full_images"],
            test_set_indices=meta["test_set_indices"],
            scene_aabb=meta["scene_aabb"],
            device=self.device,
        )
        try:
            self.trainer.resume_from_checkpoint(
                ckpt_path=str(ckpt_path),
                load_only_model=True,
            )
            self.trainer.set_eval()
        finally:
            os.chdir(_prev_cwd)
        # Pose-override registry: {(class_name, instance_id, frame_idx): (4,4) tensor}
        self._actor_overrides: Dict[Tuple[str, int, int], torch.Tensor] = {}

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    def _extract_metadata(self, ckpt_path: Path) -> Dict[str, Any]:
        """Pull num_timesteps / num_*_images / scene_aabb out of the
        checkpoint's stored module states without touching raw data on disk."""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        models = ckpt["models"]

        # Time count: RigidNodes.instances_quats.shape[0] is the per-frame
        # buffer.  Fallback: SMPLNodes / DeformableNodes have the same.
        num_timesteps = None
        for k in ("RigidNodes", "SMPLNodes", "DeformableNodes"):
            if k in models:
                for tn in ("instances_quats", "instances_trans"):
                    t = models[k].get(tn)
                    if t is not None and t.dim() >= 2:
                        num_timesteps = int(t.shape[0])
                        break
            if num_timesteps is not None:
                break

        # Image count: Affine.embedding.weight rows == num_full_images.
        num_full_images = None
        if "Affine" in models and "embedding.weight" in models["Affine"]:
            num_full_images = int(models["Affine"]["embedding.weight"].shape[0])
        if num_full_images is None and num_timesteps is not None:
            # Heuristic fallback: assume 6 cameras (DriveStudio's pandaset default)
            num_full_images = num_timesteps * 6

        if num_timesteps is None:
            num_timesteps = max(1, num_full_images // 6) if num_full_images else 1

        # Trainer constructor wants test_set_indices and scene_aabb.  For
        # inference we use trivially-correct values: no test holdout, AABB
        # from the Background means.
        bg = models.get("Background", {})
        means = bg.get("_means")
        if means is not None and means.numel() > 0:
            mn = means.min(dim=0).values.cpu().numpy().astype(np.float32)
            mx = means.max(dim=0).values.cpu().numpy().astype(np.float32)
        else:
            mn = -np.ones(3, dtype=np.float32) * 50.0
            mx =  np.ones(3, dtype=np.float32) * 50.0
        scene_aabb = torch.from_numpy(np.stack([mn, mx], axis=0))

        return dict(
            num_timesteps=num_timesteps,
            num_full_images=num_full_images,
            num_train_images=num_full_images,   # we treat all images as train (no eval split at inference)
            test_set_indices=[],
            scene_aabb=scene_aabb,
        )

    @property
    def num_frames(self) -> int:
        return self._frame_count

    # ------------------------------------------------------------------ #
    # Actor-pose injection
    # ------------------------------------------------------------------ #
    def set_actor_pose(self, class_name: str, instance_id: int,
                       frame_idx: int, c2w_4x4: np.ndarray) -> None:
        """Override an instance's per-frame pose for the next render call.

        class_name: 'RigidNodes' | 'SMPLNodes' | 'DeformableNodes'
        c2w_4x4: object-to-world 4x4 in DriveStudio world frame (z-up).
        """
        T = torch.as_tensor(c2w_4x4, dtype=torch.float32, device=self.device)
        self._actor_overrides[(class_name, int(instance_id), int(frame_idx))] = T

    def clear_actor_overrides(self) -> None:
        self._actor_overrides.clear()

    def _apply_overrides(self, frame_idx: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Patch instances_trans / instances_quats in-place at `frame_idx`.

        Returns a snapshot (per class) of the original tensor slices so the
        caller can restore them after the render, leaving the trainer state
        clean across steps.
        """
        from scipy.spatial.transform import Rotation as SCR
        snapshot: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for (cls, inst, f), pose in self._actor_overrides.items():
            if f != frame_idx:
                continue
            model = self.trainer.models.get(cls)
            if model is None or not hasattr(model, "instances_trans"):
                continue
            if inst >= model.instances_trans.shape[1]:
                continue
            if cls not in snapshot:
                snapshot[cls] = (
                    model.instances_trans[frame_idx].detach().clone(),
                    model.instances_quats[frame_idx].detach().clone(),
                )
            t = pose[:3, 3]
            R = pose[:3, :3].cpu().numpy()
            q_xyzw = SCR.from_matrix(R).as_quat().astype(np.float32)
            q_wxyz = torch.tensor([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
                                  dtype=torch.float32, device=self.device)
            with torch.no_grad():
                model.instances_trans[frame_idx, inst].copy_(t.to(self.device))
                model.instances_quats[frame_idx, inst].copy_(q_wxyz)
                if hasattr(model, "instances_fv"):
                    model.instances_fv[frame_idx, inst] = True
        return snapshot

    def _restore_overrides(self, frame_idx: int,
                           snapshot: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> None:
        for cls, (orig_t, orig_q) in snapshot.items():
            model = self.trainer.models[cls]
            with torch.no_grad():
                model.instances_trans[frame_idx].copy_(orig_t)
                model.instances_quats[frame_idx].copy_(orig_q)

    # ------------------------------------------------------------------ #
    # Core render
    # ------------------------------------------------------------------ #
    def _set_cur_frame(self, frame_idx: int) -> None:
        for m in self.trainer.models.values():
            if hasattr(m, "set_cur_frame"):
                m.set_cur_frame(int(frame_idx))

    def _make_image_infos(self, H: int, W: int, frame_idx: int,
                          img_idx: int, c2w: torch.Tensor,
                          K: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Build the dict consumed by trainer.forward / Sky / Affine.

        Mirrors `datasets/base/pixel_source.py::get_rays` so the per-pixel
        viewdirs/origins are bit-identical to what the trainer saw at
        training time.
        """
        device = self.device
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing="ij",
        )
        # Per-pixel camera-frame ray dirs, then rotate into world.
        cx, cy = K[0, 2], K[1, 2]
        fx, fy = K[0, 0], K[1, 1]
        cam_dirs = torch.stack([
            (xs - cx + 0.5) / fx,
            (ys - cy + 0.5) / fy,
            torch.ones_like(xs),
        ], dim=-1)  # (H,W,3)
        directions = (cam_dirs.unsqueeze(-2) * c2w[:3, :3]).sum(dim=-1)  # (H,W,3)
        direction_norm = torch.linalg.norm(directions, dim=-1, keepdim=True)
        viewdirs = directions / (direction_norm + 1e-8)
        origins = c2w[:3, 3].expand_as(viewdirs).contiguous()
        normed_time = float(frame_idx) / max(self._frame_count - 1, 1)
        return {
            "img_idx":   torch.full((H, W), img_idx, dtype=torch.long, device=device),
            "frame_idx": torch.full((H, W), int(frame_idx), dtype=torch.long, device=device),
            "normed_time": torch.full((H, W), normed_time, dtype=torch.float32, device=device),
            "pixel_coords": torch.stack(
                [ys / max(H - 1, 1), xs / max(W - 1, 1)], dim=-1
            ),
            "viewdirs": viewdirs,
            "origins":  origins,
            "direction_norm": direction_norm,
        }

    def _make_camera_infos(self, c2w_4x4: np.ndarray, K_3x3: np.ndarray,
                           H: int, W: int) -> Dict[str, torch.Tensor]:
        device = self.device
        c2w = torch.as_tensor(c2w_4x4, dtype=torch.float32, device=device)
        K = torch.as_tensor(K_3x3, dtype=torch.float32, device=device)
        return {
            "camera_to_world": c2w,
            "intrinsics": K,
            "height": torch.tensor([H], dtype=torch.long, device=device),
            "width":  torch.tensor([W], dtype=torch.long, device=device),
        }

    def _resolve_img_idx(self, frame_idx: int, cam_idx: int) -> int:
        """Map (frame, cam) to the affine-embedding row used during training.

        Out-of-range frames are clipped to the nearest valid one (we do not
        synthesize new embeddings for novel frames, so we just reuse the
        nearest training one — same fallback drivestudio's eval uses).
        """
        f = max(0, min(frame_idx, self._frame_count - 1))
        c = max(0, min(cam_idx, self._cams_per_frame - 1))
        return f * self._cams_per_frame + c

    def _render_core(self, c2w_4x4: np.ndarray, K_3x3: np.ndarray, H: int, W: int,
                     frame_idx: int, cam_idx: int = 0,
                     manage_overrides: bool = True) -> Dict[str, torch.Tensor]:
        """Shared render body (NO grad context — caller decides).

        When ``manage_overrides`` is False the caller must apply / restore actor
        pose overrides around one or more renders (needed when accumulating a
        multi-camera loss before ``backward()``).
        """
        snapshot = self._apply_overrides(frame_idx) if manage_overrides else None
        try:
            self._set_cur_frame(frame_idx)
            img_idx = self._resolve_img_idx(frame_idx, cam_idx)
            camera_infos = self._make_camera_infos(c2w_4x4, K_3x3, H, W)
            image_infos = self._make_image_infos(
                H, W, frame_idx, img_idx,
                camera_infos["camera_to_world"], camera_infos["intrinsics"],
            )
            outputs = self.trainer.forward(
                image_infos=image_infos,
                camera_infos=camera_infos,
                novel_view=True,
            )
        finally:
            if manage_overrides and snapshot:
                self._restore_overrides(frame_idx, snapshot)
        return outputs

    @torch.no_grad()
    def render(self, c2w_4x4: np.ndarray, K_3x3: np.ndarray, H: int, W: int,
               frame_idx: int, cam_idx: int = 0,
               manage_overrides: bool = True) -> Dict[str, torch.Tensor]:
        """Render a single view through the drivestudio model (inference, no grad).

        Returns a dict with at least ``rgb`` (H, W, 3) float in [0, 1] and
        ``depth`` (H, W, 1).  Stored on `self.device` (typically cuda).
        """
        return self._render_core(
            c2w_4x4, K_3x3, H, W, frame_idx, cam_idx,
            manage_overrides=manage_overrides,
        )

    def render_grad(self, c2w_4x4: np.ndarray, K_3x3: np.ndarray, H: int, W: int,
                    frame_idx: int, cam_idx: int = 0,
                    manage_overrides: bool = True) -> Dict[str, torch.Tensor]:
        """Differentiable render: gradients flow back to the Gaussian params.

        Used by the multimodal distillation RGB stream so the diffusion
        pseudo-GT actually updates means/scales/opacity/SH appearance.
        """
        return self._render_core(
            c2w_4x4, K_3x3, H, W, frame_idx, cam_idx,
            manage_overrides=manage_overrides,
        )
