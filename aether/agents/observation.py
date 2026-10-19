"""Build agent observations from the environment state.

Two encodings exist:

* :class:`~aether.agents.base.Observation` – the canonical in-process form;
* :func:`tuple_payload` – AETHER pipe protocol ``tuple``: a pickled
  ``(sensors, info)`` pair, read by the supplied adapter scripts (UniAD_SIM,
  VAD_SIM, NAVSIM LTF, Impromptu-VLA, GaussianFusion).  Its conventions are
  fixed because those adapters depend on them:
  ``ego_pos = [right, 0, forward]``, ``ego_rot = [0, yaw_cw, 0]``, steering
  right-positive, ``cam_params[name] = {intrinsic{fovx,fovy,cx,cy,H,W},
  v2c, l2c}`` (vehicle/LiDAR → OpenCV camera).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from .base import Observation


def _resize(image: np.ndarray, size: tuple[int, int] | None) -> tuple[np.ndarray, float, float]:
    if size is None:
        return image, 1.0, 1.0
    w, h = int(size[0]), int(size[1])
    H, W = image.shape[:2]
    if (W, H) == (w, h):
        return image, 1.0, 1.0
    from PIL import Image
    out = np.asarray(Image.fromarray(image).resize((w, h), Image.BICUBIC))
    return out, w / W, h / H


def build_observation(env, rendered: dict[str, Any], info: dict[str, Any],
                      cameras: list[str] | None = None,
                      image_size: tuple[int, int] | None = None,
                      lidar: np.ndarray | None = None) -> Observation:
    names = cameras or list(rendered["rgb"])
    images, cams = {}, {}
    for name in names:
        image, sx, sy = _resize(rendered["rgb"][name], image_size)
        K = np.array(env.cameras[name].K, dtype=np.float64)
        K[0] *= sx
        K[1] *= sy
        images[name] = image
        cams[name] = {"K": K, "width": image.shape[1], "height": image.shape[0],
                      "mount": np.asarray(env.geometry.camera_mounts[name], dtype=np.float64)}
    ego = info["ego"]
    return Observation(time=info["time"], step=info["step"], images=images, cameras=cams,
                       ego=dict(ego), command=int(info["command"]), info=info, lidar=lidar,
                       lidar_mount=env.rig.lidar_mount() if lidar is not None else None)


def _tuple_box(entry: dict[str, Any]) -> list[float]:
    return [float(entry["x"]), float(entry["y"]), 0.0, float(entry["width"]),
            float(entry["length"]), float(entry["height"]), float(entry["yaw"])]


def tuple_payload(obs: Observation, lidar_mount: np.ndarray) -> tuple[dict, dict]:
    """``(sensors, info)`` for the ``tuple`` pipe protocol (see module docstring)."""
    ego = obs.ego
    x, y = ego["rear_axle"]
    cam_params = {}
    for name, cam in obs.cameras.items():
        K, W, H = cam["K"], cam["width"], cam["height"]
        v2c = np.linalg.inv(cam["mount"])
        cam_params[name] = {
            "intrinsic": {"fovx": 2.0 * math.atan(W / (2.0 * K[0, 0])),
                          "fovy": 2.0 * math.atan(H / (2.0 * K[1, 1])),
                          "fx": float(K[0, 0]), "fy": float(K[1, 1]),
                          "cx": float(K[0, 2]), "cy": float(K[1, 2]), "H": int(H), "W": int(W)},
            "v2c": v2c.astype(np.float32),
            "l2c": (v2c @ lidar_mount).astype(np.float32),
        }
    agents = obs.info.get("agents", [])
    info = {
        "ego_pos": [-float(y), 0.0, float(x)],
        "ego_rot": [0.0, -float(ego["yaw"]), 0.0],
        "ego_velo": float(ego["speed"]),
        "ego_steer": -float(ego["steer"]),
        "accelerate": float(ego["accel"]),
        "steer_rate": -float(ego["steer_rate"]),
        "timestamp": float(obs.time),
        "command": int(obs.command),
        "ego_box": [float(ego["x"]), float(ego["y"]), 0.0, float(ego["width"]),
                    float(ego["length"]), float(ego["height"]), float(ego["yaw"])],
        "obj_boxes": [_tuple_box(a) for a in agents],
        "cam_params": cam_params,
        "aether": obs.info,                 # canonical road-frame info
    }
    payload = {"rgb": obs.images}
    if obs.lidar is not None:
        payload["lidar"] = obs.lidar
    return payload, info
