"""Unified multimodal enhancer bundle: frozen Difix RGB + trainable LiDAR refiner.

RGB weights are NOT duplicated — the bundle stores ``rgb_model`` (e.g.
``nvidia/difix_ref``) and inference delegates to the existing Difix3D bridge
(``aether.refine.refine_directory`` / ``DifixDaemon``).

Only the LiDAR branch (``CrossModalLidarRefiner``) is trained and saved in the
``.bundle.pth`` artifact.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from aether.refine.lidar_refiner_models import CrossModalLidarRefiner, LidarRefiner


@dataclass
class UnifiedEnhancerConfig:
    rgb_model: str = "nvidia/difix_ref"
    far: float = 100.0
    lidar_arch: str = "cross_modal_v1"
    lidar_ch: int = 64


class UnifiedMultimodalEnhancer:
    def __init__(self, cfg: UnifiedEnhancerConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = torch.device(device)
        if cfg.lidar_arch == "legacy_v0":
            self.lidar = LidarRefiner().to(self.device)
            self._needs_ref = False
            self._keep_is_logit = True
        elif cfg.lidar_arch == "cross_modal_v2":
            from aether.refine.lidar_refiner_models import CrossModalLidarRefinerV2
            self.lidar = CrossModalLidarRefinerV2(ch=cfg.lidar_ch).to(self.device)
            self._needs_ref = True
            self._keep_is_logit = False
        else:
            self.lidar = CrossModalLidarRefiner(ch=cfg.lidar_ch).to(self.device)
            self._needs_ref = True
            self._keep_is_logit = True
        self.lidar.eval()

    @property
    def rgb_model_id(self) -> str:
        return self.cfg.rgb_model

    def load_lidar_weights(self, state: dict) -> None:
        self.lidar.load_state_dict(state)

    @classmethod
    def from_bundle(cls, path: str | Path, device: str = "cuda") -> "UnifiedMultimodalEnhancer":
        ckpt = torch.load(path, map_location=device, weights_only=False)
        cfg = UnifiedEnhancerConfig(
            rgb_model=ckpt.get("rgb_model", "nvidia/difix_ref"),
            far=float(ckpt.get("far", 100.0)),
            lidar_arch=ckpt.get("lidar_arch", "cross_modal_v1"),
            lidar_ch=int(ckpt.get("lidar_ch", 64)),
        )
        obj = cls(cfg, device=device)
        obj.load_lidar_weights(ckpt["lidar_state"])
        return obj

    def save_bundle(self, path: str | Path, extra: Optional[dict] = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "rgb_model": self.cfg.rgb_model,
            "far": self.cfg.far,
            "lidar_arch": self.cfg.lidar_arch,
            "lidar_ch": self.cfg.lidar_ch,
            "lidar_state": self.lidar.state_dict(),
            "config": asdict(self.cfg),
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    def _tensors_from_sweep(self, in_range, in_int, in_keep,
                            ref_range, ref_int, ref_keep) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        far = self.cfg.far
        x_deg = torch.stack([
            in_range / far, in_int, in_keep,
        ], dim=0)[None].float().to(self.device)
        if not self._needs_ref:
            return x_deg, None
        x_ref = torch.stack([
            ref_range / far, ref_int, ref_keep,
        ], dim=0)[None].float().to(self.device)
        return x_deg, x_ref

    @torch.no_grad()
    def enhance_lidar_arrays(
        self,
        in_range: np.ndarray,
        in_int: np.ndarray,
        in_keep: np.ndarray,
        ref_range: np.ndarray,
        ref_int: np.ndarray,
        ref_keep: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_deg, x_ref = self._tensors_from_sweep(
            torch.from_numpy(in_range).float(),
            torch.from_numpy(in_int).float(),
            torch.from_numpy(in_keep).float(),
            torch.from_numpy(ref_range).float(),
            torch.from_numpy(ref_int).float(),
            torch.from_numpy(ref_keep).float(),
        )
        if x_ref is None:
            pred = self.lidar(x_deg)
        else:
            pred = self.lidar(x_deg, x_ref)
        far = self.cfg.far
        pr = (pred[0, 0].cpu().numpy() * far).astype(np.float32)
        pi = pred[0, 1].cpu().numpy().astype(np.float32)
        if getattr(self, "_keep_is_logit", True):
            pk = torch.sigmoid(pred[0, 2]).cpu().numpy().astype(np.float32)
        else:
            pk = pred[0, 2].clamp(0, 1).cpu().numpy().astype(np.float32)
        return pr, pi, pk

    @torch.no_grad()
    def enhance_lidar_render_dict(self, out: dict, ref_range, ref_int, ref_keep) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enhance a ClosedLoopLidar render dict (training / distillation helper)."""
        far = self.cfg.far
        in_r = out["range"].clamp(0, far)
        in_i = out["intensity"].clamp(0, 1)
        in_k = (1 - torch.sigmoid(out["raydrop_logit"])).clamp(0, 1)
        x_deg, x_ref = self._tensors_from_sweep(in_r, in_i, in_k, ref_range, ref_int, ref_keep)
        if x_ref is None:
            p = self.lidar(x_deg)
        else:
            p = self.lidar(x_deg, x_ref)
        pr = (p[:, 0:1] * far)[0, 0]
        pi = p[:, 1:2][0, 0]
        if getattr(self, "_keep_is_logit", True):
            pk = torch.sigmoid(p[:, 2:3])[0, 0]
        else:
            pk = p[:, 2:3].clamp(0, 1)[0, 0]
        return pr.detach(), pi.detach(), pk.detach()


def load_unified_bundle(path: str | Path, device: str = "cuda") -> UnifiedMultimodalEnhancer:
    return UnifiedMultimodalEnhancer.from_bundle(path, device=device)
