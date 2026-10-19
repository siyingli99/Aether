"""Road frame, logged ego track, ground model and calibrated camera mounts.

Built once per scene from the bundle and the reconstruction's background
splats (see :mod:`aether.scene.ground`).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from ..geometry.frames import pose2d, yaw_of
from ..vehicle.profile import VehicleProfile
from .bundle import SceneBundle
from .ground import MAX_GRADE, GroundModel


@dataclass
class LoggedTrack:
    """Recorded ego motion in the road frame (rear axle on the ground)."""

    t: np.ndarray        # (T,) seconds
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray

    @property
    def xy(self) -> np.ndarray:
        return np.column_stack([self.x, self.y])

    def speed(self) -> np.ndarray:
        d = np.hypot(np.diff(self.x), np.diff(self.y)) / np.diff(self.t)
        return np.concatenate([d[:1], d]) if len(d) else np.zeros(1)


def logged_track(bundle: SceneBundle) -> LoggedTrack:
    """Rear-axle track in the road frame from the bundle log alone.

    Only x, y, yaw – the ground height needs the splats (see SceneGeometry).
    """
    R = np.eye(4)
    R[:3, :3] = bundle.frame.scene_from_road
    poses = np.einsum("ij,tjk,kl->til", np.linalg.inv(R), bundle.log.anchor,
                      np.linalg.inv(bundle.anchor_mount))
    yaw = np.unwrap([yaw_of(p[:3, :3]) for p in poses])
    return LoggedTrack(np.arange(len(poses)) * bundle.frame_dt, poses[:, 0, 3].copy(),
                       poses[:, 1, 3].copy(), np.asarray(yaw))


def logged_centers(bundle: SceneBundle, vehicle: str | None = None) -> np.ndarray:
    """(T, 2) logged vehicle-box centres (route used for progress / commands)."""
    from ..vehicle.profile import load_profile
    track = logged_track(bundle)
    off = load_profile(vehicle or bundle.vehicle).center_offset
    return np.column_stack([track.x + off * np.cos(track.yaw), track.y + off * np.sin(track.yaw)])



MAX_OBJECT_PITCH = math.atan(MAX_GRADE)


class SceneGeometry:
    def __init__(self, bundle: SceneBundle, profile: VehicleProfile, splats_scene: np.ndarray,
                 cache: bool = True) -> None:
        self.bundle = bundle
        self.profile = profile
        frame = bundle.frame
        log = bundle.log
        R = frame.scene_from_road
        # Provisional road frame: rotation fixed by the dataset convention,
        # origin at the anchor sensor at frame 0.
        provisional = np.eye(4)
        provisional[:3, :3] = R
        prov_inv = np.linalg.inv(provisional)
        vehicle_from_anchor = bundle.anchor_mount
        anchor_from_vehicle = np.linalg.inv(vehicle_from_anchor)
        vehicles = np.einsum("ij,tjk,kl->til", prov_inv, log.anchor, anchor_from_vehicle)
        route = vehicles[:, :3, 3]
        splats_road = (np.asarray(splats_scene, dtype=np.float64) @ R)  # R^T p

        ground = None
        cache_path = bundle.cache_dir / "ground.npz"
        key = self._cache_key(bundle, splats_scene)
        if cache and cache_path.is_file():
            data = np.load(cache_path)
            if str(data.get("key", "")) == key:
                ground = GroundModel(data["origin"], float(data["resolution"]), data["heights"])
        if ground is None:
            hdmap = bundle.hdmap
            drivable = None
            if hdmap is not None and hdmap.drivable is not None:
                drivable = lambda xy: hdmap.outside_distance(xy) < 1.0   # noqa: E731
            ground = GroundModel.build(splats_road, route, drivable=drivable)
            if cache:
                np.savez_compressed(cache_path, origin=ground.origin, resolution=ground.resolution,
                                    heights=ground.heights.astype(np.float32), key=key)
        # Final road frame: origin on the ground below the anchor.
        z0 = ground.height(0.0, 0.0)
        ground.heights = ground.heights - z0
        self.ground = ground
        self.scene_from_road = provisional.copy()
        self.scene_from_road[:3, 3] = R @ np.array([0.0, 0.0, z0])
        self.road_from_scene = np.linalg.inv(self.scene_from_road)

        poses = np.einsum("ij,tjk,kl->til", self.road_from_scene, log.anchor, anchor_from_vehicle)
        yaw = np.array([yaw_of(p[:3, :3]) for p in poses])
        self.track = LoggedTrack(
            t=np.arange(len(poses)) * bundle.frame_dt,
            x=poses[:, 0, 3].copy(), y=poses[:, 1, 3].copy(), yaw=np.unwrap(yaw),
        )
        # Camera mounts calibrated so that the modelled vehicle pose at frame
        # 0 reproduces the logged camera poses exactly.
        vehicle0 = self.vehicle_pose(self.track.x[0], self.track.y[0], self.track.yaw[0])
        scene_from_vehicle0 = self.scene_from_road @ vehicle0
        inv0 = np.linalg.inv(scene_from_vehicle0)
        self.camera_mounts = {name: inv0 @ pose for name, pose in log.cameras_t0.items()}
        self.anchor_mount_effective = inv0 @ log.anchor[0]

    @staticmethod
    def _cache_key(bundle: SceneBundle, splats: np.ndarray) -> str:
        digest = hashlib.sha1()
        digest.update(np.ascontiguousarray(splats[:: max(1, len(splats) // 4096)]).tobytes())
        digest.update(str(bundle.log.anchor.shape).encode())
        digest.update(np.ascontiguousarray(bundle.anchor_mount, dtype=np.float64).tobytes())
        from .ground import VERSION
        digest.update(f"ground-v{VERSION}".encode())
        if bundle.hdmap_path is not None:
            digest.update(str(bundle.hdmap_path.stat().st_mtime_ns).encode())
        return digest.hexdigest()[:16]

    # ---------------------------------------------------------------- poses
    def vehicle_pose(self, x: float, y: float, yaw: float) -> np.ndarray:
        """Road-frame 4x4 pose of a vehicle (rear axle at x, y) resting on the ground."""
        z, pitch, roll = self.ground.attitude(x, y, yaw, self.profile.wheelbase, self.profile.track)
        return pose2d(x, y, yaw, z, pitch, roll)

    def object_pose(self, x: float, y: float, yaw: float, length: float,
                    width: float, height: float) -> np.ndarray:
        """Road-frame pose of an object's box centre, resting on the ground.

        Pitch comes from the ground under the axles, measured over at least
        6 m and limited to ``MAX_OBJECT_PITCH`` (a 10 % grade): reconstructed
        ground is noisy away from the driven corridor, real roads are not.
        """
        c, s = math.cos(yaw), math.sin(yaw)
        half = max(0.5 * length * 0.8, 3.0)
        z_front = self.ground.height(x + half * c, y + half * s)
        z_rear = self.ground.height(x - half * c, y - half * s)
        pitch = -math.atan2(z_front - z_rear, 2 * half)
        pitch = min(max(pitch, -MAX_OBJECT_PITCH), MAX_OBJECT_PITCH)
        z = float(self.ground.height(x, y)) + 0.5 * height
        return pose2d(x, y, yaw, z, pitch, 0.0)

    def to_scene(self, road_pose: np.ndarray) -> np.ndarray:
        return self.scene_from_road @ road_pose

    def to_road(self, scene_pose: np.ndarray) -> np.ndarray:
        return self.road_from_scene @ scene_pose

    def points_to_road(self, points_scene: np.ndarray) -> np.ndarray:
        T = self.road_from_scene
        return np.asarray(points_scene) @ T[:3, :3].T + T[:3, 3]
