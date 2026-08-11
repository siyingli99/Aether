"""Camera-rig loader (HUGSIM-style yaml).

Each camera yaml has::

    cam_rect:                 # optional rectification of ego->front-cam
      rot:   [rx, ry, rz]     # XYZ Euler degrees
      trans: [tx, ty, tz]
    cams:
      CAM_FRONT:
        extrinsics:
          v2c_rot:   [rx, ry, rz]
          v2c_trans: [tx, ty, tz]
          l2c_rot:   [rx, ry, rz]   # optional, defaults to v2c_*
          l2c_trans: [tx, ty, tz]
        intrinsics:
          fovx: <deg>  fovy: <deg>
          cx: <px>     cy: <px>
          H: <px>      W: <px>
      ...

Returns ``(rig, rect)`` where ``rig[name]`` carries ``K`` (3x3),
``v2c`` (4x4), ``l2c`` (4x4), ``H``, ``W``, ``intrinsic`` (dict, fov in
radians, with derived ``fx``/``fy``).  The driving env now computes c2w
directly from the trained DriveStudio data and only uses ``K``/``H``/
``W`` from the rig for rendering, but ``v2c``/``l2c``/``intrinsic`` are
kept so AD adapters that consume ``cam_params`` still work.
"""
from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as SCR


def _euler_xyz_to_mat(rot_deg, trans) -> np.ndarray:
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = SCR.from_euler("XYZ", rot_deg, degrees=True).as_matrix()
    M[:3, 3] = np.asarray(trans, dtype=np.float32)
    return M


def fov2focal(fov_rad: float, pixels: int) -> float:
    return pixels / (2.0 * math.tan(fov_rad / 2.0))


def _build_K(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx],
                     [0.0, fy, cy],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def load_camera_rig(camera_cfg) -> Tuple[Dict[str, dict], np.ndarray]:
    rig: Dict[str, dict] = {}
    cams = OmegaConf.to_container(camera_cfg.cams, resolve=True)
    for cam_name, cam in cams.items():
        ext = cam["extrinsics"]
        v2c = _euler_xyz_to_mat(ext["v2c_rot"], ext["v2c_trans"])
        l2c_rot   = ext.get("l2c_rot",   ext["v2c_rot"])
        l2c_trans = ext.get("l2c_trans", ext["v2c_trans"])
        l2c = _euler_xyz_to_mat(l2c_rot, l2c_trans)

        intr = dict(cam["intrinsics"])
        H, W = int(intr["H"]), int(intr["W"])
        intr["H"], intr["W"] = H, W
        intr["fovx"] = math.radians(float(intr["fovx"]))
        intr["fovy"] = math.radians(float(intr["fovy"]))
        intr["cx"] = float(intr["cx"])
        intr["cy"] = float(intr["cy"])
        fx = fov2focal(intr["fovx"], W)
        fy = fov2focal(intr["fovy"], H)
        intr["fx"], intr["fy"] = fx, fy
        K = _build_K(fx, fy, intr["cx"], intr["cy"])
        rig[cam_name] = {
            "K": K, "v2c": v2c, "l2c": l2c,
            "H": H, "W": W, "intrinsic": intr,
        }

    rect = np.eye(4, dtype=np.float32)
    if "cam_rect" in camera_cfg:
        rect[:3, :3] = SCR.from_euler(
            "XYZ", camera_cfg.cam_rect.rot, degrees=True
        ).as_matrix()
        rect[:3, 3] = np.asarray(camera_cfg.cam_rect.trans, dtype=np.float32)
    return rig, rect
