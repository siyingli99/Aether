"""Reconstructed dynamic actors (``actors.npz``) extracted from a checkpoint.

Keys per instance ``<Class>/<index>`` (Class = RigidNodes | SMPLNodes |
DeformableNodes)::

    c2w/<Class>/<i>     (Tv, 4, 4)  object-to-scene pose on the frames it is visible
    frames/<Class>/<i>  (Tv,)       reconstruction frame indices
    size/<Class>/<i>    (3,)        width, length, height in metres
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

CLASSES = ("RigidNodes", "DeformableNodes", "SMPLNodes")


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    xyzw = np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)
    return Rotation.from_quat(xyzw.reshape(-1, 4)).as_matrix().reshape(*q.shape[:-1], 3, 3)


def extract_actors(checkpoint: str | Path, min_frames: int = 2,
                   classes: tuple[str, ...] = CLASSES) -> dict[str, np.ndarray]:
    import torch
    models = torch.load(str(checkpoint), map_location="cpu", weights_only=False, mmap=True)["models"]
    out: dict[str, np.ndarray] = {}
    for cls in classes:
        if cls not in models:
            continue
        m = models[cls]
        trans = m["instances_trans"].detach().float().cpu().numpy()
        quats = m["instances_quats"].detach().float().cpu().numpy()
        while quats.ndim > 3:                       # SMPLNodes: (T, N, 1, 4)
            quats = quats.reshape(quats.shape[0], quats.shape[1], -1)[..., :4]
        sizes = m["instances_size"].detach().float().cpu().numpy()
        fv = m.get("instances_fv")
        fv = np.ones(trans.shape[:2], bool) if fv is None else fv.detach().cpu().numpy().astype(bool)
        if sizes.ndim == 3:
            sizes = (sizes * fv[..., None]).sum(0) / np.maximum(fv.sum(0), 1)[:, None]
        rot = _quat_wxyz_to_matrix(quats)
        for i in range(trans.shape[1]):
            frames = np.flatnonzero(fv[:, i]).astype(np.int32)
            if len(frames) < min_frames:
                continue
            pose = np.tile(np.eye(4, dtype=np.float32), (len(frames), 1, 1))
            pose[:, :3, :3] = rot[frames, i]
            pose[:, :3, 3] = trans[frames, i]
            length, width, height = (max(float(v), 0.1) for v in sizes[i, :3])
            out[f"c2w/{cls}/{i}"] = pose
            out[f"frames/{cls}/{i}"] = frames
            out[f"size/{cls}/{i}"] = np.array([width, length, height], np.float32)
    return out
