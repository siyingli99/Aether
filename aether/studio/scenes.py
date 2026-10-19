"""Scene payloads for the Studio map editor (no GPU needed)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ..scene.bundle import SceneBundle, list_scenes
from ..scene.geometry import logged_centers, logged_track
from ..vehicle.profile import load_profile


def _ring(coords, step: float = 0.5) -> list[list[float]]:
    pts = np.asarray(coords, dtype=float)[:, :2]
    if len(pts) > 2:
        keep = [0]
        for i in range(1, len(pts)):
            if np.hypot(*(pts[i] - pts[keep[-1]])) >= step:
                keep.append(i)
        pts = pts[keep]
    return np.round(pts, 2).tolist()


def scene_list() -> list[dict[str, Any]]:
    out = []
    for scene_id in list_scenes():
        try:
            b = SceneBundle.load(scene_id)
        except Exception as error:                 # noqa: BLE001 – shown in the UI
            out.append({"id": scene_id, "error": str(error)})
            continue
        out.append({"id": scene_id, "name": b.manifest.get("name", scene_id), "dataset": b.dataset,
                    "frame_dt": b.frame_dt, "frames": b.manifest.get("frames"),
                    "vehicle": b.vehicle, "has_map": b.hdmap_path is not None,
                    "has_network": b.network is not None, "source": b.source})
    return out


@lru_cache(maxsize=8)
def scene_detail(scene_id: str) -> dict[str, Any]:
    b = SceneBundle.load(scene_id)
    profile = load_profile(b.vehicle)
    track = logged_track(b)
    centers = logged_centers(b)
    payload: dict[str, Any] = {
        "id": b.id, "name": b.manifest.get("name", b.id), "dataset": b.dataset,
        "frame_dt": b.frame_dt, "duration_s": round((len(centers) - 1) * b.frame_dt, 2),
        "vehicle": {"name": profile.name, "length": profile.length, "width": profile.width,
                    "height": profile.height, "wheelbase": profile.wheelbase},
        "route": np.round(centers, 2).tolist(),
        "ego_start": {"x": float(centers[0, 0]), "y": float(centers[0, 1]), "yaw": float(track.yaw[0])},
        "lanes": [], "drivable": [], "recorded": [],
    }
    m = b.hdmap
    if m is not None:
        payload["lanes"] = [{"id": lane.id, "kind": lane.kind, "centerline": _ring(lane.centerline),
                             "polygon": _ring(lane.polygon.exterior.coords), "width": lane.width}
                            for lane in m.lanes]
        if m.drivable is not None:
            geoms = list(getattr(m.drivable, "geoms", [m.drivable]))
            payload["drivable"] = [_ring(g.exterior.coords) for g in geoms]
    payload["recorded"] = recorded_actors(b)
    return payload


def recorded_actors(b: SceneBundle) -> list[dict[str, Any]]:
    """Reconstructed actors at their first visible frame (road frame, 2D)."""
    if b.actors_path is None:
        return []
    from ..geometry.frames import pose_from_matrix
    data = np.load(b.actors_path)
    R = np.eye(4)
    R[:3, :3] = b.frame.scene_from_road
    road_from_scene = np.linalg.inv(R)
    out = []
    for key in data.files:
        if not key.startswith("c2w/"):
            continue
        cls, _, idx = key[4:].partition("/")
        frames = data[f"frames/{cls}/{idx}"]
        c2w = data[key]
        if not len(frames):
            continue
        x, y, yaw = pose_from_matrix(road_from_scene @ c2w[0])
        size = data.get(f"size/{cls}/{idx}")
        w, l, h = size.tolist() if size is not None else (1.85, 4.5, 1.5)
        path = [pose_from_matrix(road_from_scene @ m)[:2] for m in c2w[:: max(1, len(c2w) // 40)]]
        out.append({"ref": f"{cls}/{idx}", "kind": "vehicle" if cls == "RigidNodes" else "pedestrian",
                    "x": round(x, 2), "y": round(y, 2), "yaw": round(yaw, 3),
                    "length": round(float(l), 2), "width": round(float(w), 2), "height": round(float(h), 2),
                    "first_frame": int(frames[0]), "last_frame": int(frames[-1]),
                    "path": np.round(np.asarray(path), 2).tolist()})
    return out
