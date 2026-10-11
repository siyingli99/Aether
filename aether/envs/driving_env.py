"""Closed-loop driving env on top of the DriveStudio renderer.

Coordinate strategy (v3, DS-native)
-----------------------------------

DriveStudio normalises the scene by ``inv(raw_ego_pose_at_start_timestep)``,
so the trained Gaussians live in a "DS world" where the start-frame ego
sits at the origin with identity rotation.

Pandaset's body frame is empirically ``+y=forward, +x=right, +z=up`` (the
ego trajectory between consecutive frames moves along ego ``+y``), so DS
world also has the road extending along ``+y`` at start.

We:
  * keep the bicycle integrator in HUGSIM convention so HUGSIM AD adapters
    / ego-box layouts continue to work;
  * read the *raw* pandaset ``ego_pose/{t:03d}.txt`` /
    ``extrinsics/{t:03d}_{cam_id}.txt`` files that produced the trained
    gaussians, and pre-compute, per camera::

        c2w_DS_at_start  = inv(raw_ego_start) @ raw_extr_start

    Since ``ego_DS_start = I`` by construction, this matrix is also the
    rigid camera-to-body transform we need at runtime;
  * map the HUGSIM bicycle state to a DS-world ego pose so that the
    HUGSIM start state ``vr=[0, pi/2, 0], vab=(0,0)`` produces identity::

        ego_DS_rot = R_z(yaw_HUGSIM - pi/2)
        ego_DS_pos = [vab[1], vab[0], 0]      # DS x = vab[1], DS y = vab[0]

  * render with ``c2w_DS = ego_DS @ c2w_DS_at_start[cam]``.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict

import gymnasium as gym
import numpy as np
import torch  # noqa: F401
from gymnasium import spaces
from scipy.spatial.transform import Rotation as SCR

from ..kinematic.bicycle import BicycleEgo
from ..planner.npc import ActorReplay
from ..paths import resolve_data_root
from ..sim.scene_factory import scene_from_scenario_actors
from ..sensors import LidarSim
from ..render.camera import load_camera_rig
from ..render.ds_backend import DriveStudioRenderer
from .driving_command import build_command_provider


_PANDA_CAM_ID = {
    "CAM_FRONT":       0,
    "CAM_FRONT_LEFT":  1,
    "CAM_FRONT_RIGHT": 2,
    "CAM_BACK_LEFT":   3,
    "CAM_BACK_RIGHT":  4,
    "CAM_BACK":        5,
}


def _load_4x4(path: Path) -> np.ndarray:
    return np.loadtxt(str(path)).astype(np.float64).reshape(4, 4)


def _ego_pose_ds(vab, vr) -> np.ndarray:
    """Pandaset DS-world ego pose (z-up). Original/tested formula."""
    yaw_h = float(vr[1])
    R = SCR.from_euler("z", yaw_h - math.pi / 2.0).as_matrix().astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[0, 3] = float(vab[1])
    T[1, 3] = float(vab[0])
    T[2, 3] = 0.0
    return T


def _ego_pose_ds_nuscenes(vab, vr) -> np.ndarray:
    """Nuscenes DS-world ego pose.

    DriveStudio's NuScenesPixelSource anchors the world on the FRONT-cam
    OpenCV pose at start frame, so DS-world axes equal the FRONT-cam-at-
    start axes:  x = right, y = down, z = forward.  At t=0 the ego
    coincides with this frame and static_c2w_ds[FRONT] = I, hence
    ego_pose_ds(0) must equal I -> the trained frame-0 view renders
    exactly.

    HUGSIM's BicycleEgo uses (matches sim/hugsim_env/envs/hug_sim.py):
        vab[0] = lateral right (sin(theta) term)
        vab[1] = forward       (cos(theta) term)
        vr[1]  = yaw around HUGSIM up axis; positive turns the heading
                 toward +vab[0] (i.e. right turn).
    Mapping into the FRONT-cam frame:
        right (vab[0]) -> +DS_x
        forward (vab[1]) -> +DS_z
        up           -> -DS_y (DS_y is down, so vertical = 0)
        right turn (yaw_h>0) -> +R_y around DS_y (down axis):
            rotates +z toward +x.
    """
    yaw_h = float(vr[1])
    R = SCR.from_euler("y", yaw_h).as_matrix().astype(np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[0, 3] = float(vab[0])  # lateral right -> +DS_x
    T[1, 3] = 0.0            # ground -> 0 along DS_y (down)
    T[2, 3] = float(vab[1])  # forward       -> +DS_z
    return T


class _SceneActorAdapter:
    """Bridges :class:`aether.sim.Scene` to the legacy ``ActorReplay`` API
    (``pose_at(frame_idx)``) used by ``AetherDrivingEnv``.

    Before each query, mirrors the env's ego pose into ``scene.ego`` so that
    IDM / manual policies can react to the latest ego state.
    """

    def __init__(self, scene, env):
        self.scene = scene
        self.env = env

    def _refresh_ego(self) -> None:
        e = self.env.ego
        # ``BicycleEgo.vab`` is (lateral_right, forward). Project into HUGSIM BEV
        # consistent with ``_ego_box_envframe`` (x = vt2, y = -vt0).
        x_bev = float(e.vab[1])
        y_bev = -float(e.vab[0])
        yaw_bev = float(e.vr[1])
        self.scene.update_ego(x=x_bev, y=y_bev, yaw=yaw_bev,
                              velo=float(e.velo))

    def pose_at(self, frame_idx: int):
        self._refresh_ego()
        return self.scene.pose_at(int(frame_idx))


class AetherDrivingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(self, cfg, output_dir: str):
        super().__init__()
        self.cfg = cfg
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        ckpt = cfg.scenario.checkpoint
        traffic_cfg = getattr(cfg.scenario, "traffic", None)
        hide_original_dynamic = (
            bool(getattr(traffic_cfg, "hide_original_dynamic", False))
            and bool(getattr(traffic_cfg, "enabled", False))
            if traffic_cfg is not None else False
        )
        excluded_models = set(getattr(
            cfg.scenario, "excluded_model_classes", []
        ) or [])
        if hide_original_dynamic:
            excluded_models.add("SMPLNodes")
        self.renderer = DriveStudioRenderer(
            ckpt, device="cuda", excluded_model_classes=tuple(sorted(excluded_models))
        )
        self._apply_distill_overlay()
        self._max_frame = self.renderer.num_frames - 1

        self.cam_rig, _ = load_camera_rig(cfg.camera)
        self._cam_order = list(self.cam_rig.keys())

        self._c2w_static_ds: Dict[str, np.ndarray] = self._load_static_c2w_ds()
        self._init_route_reference()

        self._command_provider = build_command_provider(
            cfg.scenario,
            scene_dir=getattr(self, "_ds_scene_dir", None),
            inv_anchor=getattr(self, "_ds_inv_anchor", None),
            dataset=getattr(self, "_ds_dataset", "pandaset"),
        )

        self.ego = BicycleEgo(cfg.kinematic, cfg.scenario)
        whl = getattr(cfg.scenario, "ego_whl", [1.6, 1.5, 3.0])
        self.whl = np.array(whl, dtype=np.float32)

        self.actors = ActorReplay(getattr(cfg.scenario, "actors", []))

        # Optional Scene-backed actor manager (IDM / manual / log-replay).
        # Activated by `scenario.actor_backend: scene` (or "policy"/"idm").
        self._scene = None
        _backend = str(getattr(cfg.scenario, "actor_backend", "")).lower()
        if _backend in {"scene", "policy", "idm"}:
            _tl = getattr(cfg.scenario, "traffic_log", None)
            if _tl:
                _tl_path = Path(str(_tl))
                if not _tl_path.is_absolute():
                    # resolve relative to scenario YAML dir (passed via cfg)
                    _scen_dir = getattr(cfg.scenario, "_yaml_dir", None) \
                        or getattr(cfg, "_scenario_dir", None)
                    if _scen_dir:
                        _tl_path = Path(_scen_dir) / _tl_path
                    else:
                        # fall back to repo-root traffic_log
                        _tl_path = Path(__file__).resolve().parents[2] / _tl_path
                _tl = str(_tl_path)
            self._scene = scene_from_scenario_actors(
                getattr(cfg.scenario, "actors", []) or [],
                traffic_log=_tl,
            )
            self.actors = _SceneActorAdapter(self._scene, self)

        # Observation channel toggle: skip depth GPU->CPU copy when adapters
        # only consume RGB.  Defaults to camera-only.
        _obs_cfg = getattr(cfg.base, "obs", None) if hasattr(cfg, "base") else None
        self.obs_cameras_only = bool(getattr(_obs_cfg, "cameras_only", True)) if _obs_cfg is not None else True
        # Mode A LiDAR (depth-backproj). Enabled via scenario.lidar.enabled.
        self.lidar = None
        _lidar_cfg = getattr(cfg.scenario, "lidar", None)
        if _lidar_cfg is not None and bool(getattr(_lidar_cfg, "enabled", False)):
            self.lidar = LidarSim.from_cfg(_lidar_cfg)
            self.obs_cameras_only = False  # need depth for backprojection

        self.action_space = spaces.Dict({
            "steer_rate": spaces.Box(self.ego.min_steer, self.ego.max_steer, dtype=float),
            "acc":        spaces.Box(self.ego.min_acc,   self.ego.max_acc,   dtype=float),
        })
        _space = {
            "rgb": spaces.Dict({
                name: spaces.Box(0, 255, shape=(c["H"], c["W"], 3), dtype=np.uint8)
                for name, c in self.cam_rig.items()
            }),
        }
        if not self.obs_cameras_only:
            _space["depth"] = spaces.Dict({
                name: spaces.Box(0, 1000.0, shape=(c["H"], c["W"]), dtype=np.float32)
                for name, c in self.cam_rig.items()
            })
        self.observation_space = spaces.Dict(_space)

        term = getattr(cfg, "termination", None) or getattr(cfg.scenario, "termination", None)

        def _t(name, default):
            return float(getattr(term, name, default)) if term is not None else default

        self.max_steps = int(_t("max_steps", 400))
        self.fg_collision_min_overlap = _t("fg_collision_min_overlap", 0.10)
        self.term_fg_collision = bool(getattr(term, "fg_collision", True)) if term is not None else True

        self.timestamp = 0.0
        self._step_count = 0
        # Background gaussian centers for collision and ground-height queries.
        self._init_collision_assets(term)
        self._traffic_provider = None
        self._external_vehicle_pool = None
        self._traffic_visualizer = None
        self._dynamic_removal_audit = {}
        self._studio_manager = None
        if _backend == "studio":
            from ..studio.runtime import StudioActorManager
            self._studio_manager = StudioActorManager(
                self.renderer, cfg.scenario.studio, self
            )
            self.actors = self._studio_manager
            self._traffic_provider = self._studio_manager
            self._external_vehicle_pool = self._studio_manager.pool
            self._dynamic_removal_audit = self._studio_manager.removal_audit()
            net_file = getattr(cfg.scenario.studio.limsim, "net_file", None)
            if net_file and bool(getattr(cfg.scenario.studio.limsim, "enabled", False)):
                from ..traffic.visualization import LimSimTopDownRenderer
                self._traffic_visualizer = LimSimTopDownRenderer(Path(str(net_file)))
        self._init_traffic_backend()

    def _scenario_path(self, value: str | Path) -> Path:
        path = Path(str(value)).expanduser()
        if not path.is_absolute():
            path = Path(str(self.cfg.scenario._yaml_dir)) / path
        return path.resolve()

    def _init_traffic_backend(self) -> None:
        """Attach a traffic source and reusable 3DRealCar Gaussian pool."""
        traffic = getattr(self.cfg.scenario, "traffic", None)
        if traffic is None or not bool(getattr(traffic, "enabled", False)):
            return
        from ..render.external_assets import ExternalVehiclePool, GaussianVehicleAsset
        from ..traffic.client import LimSimTrafficClient, ReplayTrafficClient
        from ..traffic.coordinates import FrameTransform, nuscenes_scene_local_to_ds
        from ..traffic.provider import TrafficActorProvider

        asset_path = self._scenario_path(str(traffic.asset_path))
        asset = GaussianVehicleAsset.load(
            asset_path,
            max_gaussians=int(getattr(traffic, "max_gaussians", 50_000)),
        )
        pool = ExternalVehiclePool(
            self.renderer,
            asset,
            capacity=int(getattr(traffic, "max_vehicles", 6)),
            hide_original_rigid=bool(getattr(traffic, "hide_original_rigid", True)),
            hide_original_dynamic=bool(getattr(
                traffic, "hide_original_dynamic",
                getattr(traffic, "hide_original_rigid", True),
            )),
        )

        backend = str(getattr(traffic, "backend", "limsim")).lower()
        if backend == "limsim":
            client = LimSimTrafficClient(
                host=str(getattr(traffic, "host", "127.0.0.1")),
                port=int(getattr(traffic, "port", 8766)),
                timeout_s=float(getattr(traffic, "timeout_s", 10.0)),
            )
        elif backend == "replay":
            client = ReplayTrafficClient.from_json(
                self._scenario_path(str(traffic.replay_path))
            )
        else:
            raise ValueError(f"unsupported traffic backend: {backend!r}")

        alignment = getattr(traffic, "alignment_path", None)
        if alignment:
            transform = FrameTransform.load(self._scenario_path(str(alignment)))
        elif self._ds_dataset == "nuscenes":
            transform = nuscenes_scene_local_to_ds(
                ground_ds_y=float(getattr(self, "_ground_ref", 0.0))
            )
        else:
            raise ValueError("automatic traffic alignment is currently implemented for nuScenes only")

        self._external_vehicle_pool = pool
        self._traffic_provider = TrafficActorProvider(client, transform, pool.instance_ids)
        self.actors = self._traffic_provider
        self._dynamic_removal_audit = pool.assert_original_dynamic_hidden()
        net_file = getattr(traffic, "net_file", None)
        if net_file:
            from ..traffic.visualization import LimSimTopDownRenderer
            self._traffic_visualizer = LimSimTopDownRenderer(
                self._scenario_path(str(net_file))
            )
        print(
            f"[traffic] backend={backend}, asset={asset_path.name}, "
            f"gaussians/vehicle={asset.num_gaussians}, slots={len(pool.instance_ids)}",
            flush=True,
        )
        print(
            f"[traffic] original dynamic foreground hidden: "
            f"{self._dynamic_removal_audit}", flush=True,
        )

    def render_traffic_2d(self) -> np.ndarray | None:
        """Return the synchronized full-map LimSim view for video composition."""
        if self._traffic_visualizer is None or self._traffic_provider is None:
            return None
        return self._traffic_visualizer.render(
            self._traffic_provider.frame, self._traffic_ego_state()
        )

    def _traffic_ego_state(self):
        from ..traffic.protocol import EgoState
        return EgoState(
            x=float(self.ego.vab[1]),
            y=-float(self.ego.vab[0]),
            # BicycleEgo positive yaw turns toward lateral-right; traffic yaw is
            # CCW toward local-left.
            yaw=-float(self.ego.vr[1]),
            speed=float(self.ego.velo),
            length=float(self.whl[2]),
            width=float(self.whl[0]),
            height=float(self.whl[1]),
        )

    def _apply_distill_overlay(self) -> None:
        """Apply a formal reverse-distilled Background state when configured."""
        overlay = getattr(self.cfg.scenario, "distill_overlay", None)
        if not overlay:
            return
        path = Path(str(overlay))
        if not path.is_file():
            raise FileNotFoundError(f"configured distill_overlay missing: {path}")
        state = torch.load(path, map_location="cuda", weights_only=False)
        background = self.renderer.trainer.models.get("Background")
        if background is None:
            raise RuntimeError("distill_overlay requires a Background model")
        background.load_state_dict(state)
        print(f"[aether] applied formal distill overlay: {path}", flush=True)

    def _init_route_reference(self) -> None:
        """Build the logged route centerline used only for route-progress scoring."""
        if self._ds_dataset == "nuscenes":
            # DriveStudio's nuScenes preprocessing stores the moving camera
            # poses in ``extrinsics`` and does not create ``ego_pose``.  The
            # FRONT camera (id 0) is also the world anchor used by its pixel
            # source, so its per-frame poses give the logged route in exactly
            # the same DS coordinate frame as the renderer.
            route_source = self._ds_scene_dir / "extrinsics"
            paths = sorted(route_source.glob("*_0.txt"))
        else:
            route_source = self._ds_scene_dir / "ego_pose"
            paths = sorted(route_source.glob("*.txt"))
        points = []
        for path in paths:
            pose_ds = self._ds_inv_anchor @ _load_4x4(path)
            points.append(
                [
                    float(pose_ds[self._ds_fwd_axis, 3]),
                    float(-pose_ds[self._ds_lat_axis, 3]),
                ]
            )
        route = np.asarray(points, dtype=np.float64)
        if len(route) < 2:
            raise RuntimeError(
                f"route requires at least two logged poses: {route_source}"
            )
        segments = route[1:] - route[:-1]
        lengths = np.linalg.norm(segments, axis=1)
        keep = lengths > 1e-6
        self._route_starts = route[:-1][keep]
        self._route_segments = segments[keep]
        self._route_lengths = lengths[keep]
        self._route_cumulative = np.concatenate(
            [[0.0], np.cumsum(self._route_lengths)]
        )
        self._route_total = float(self._route_lengths.sum())

    def _route_progress(self) -> float:
        """Project ego center onto the logged route and return [0, 1] progress."""
        point = np.asarray(
            [float(self.ego.vab[1]), float(-self.ego.vab[0])],
            dtype=np.float64,
        )
        denom = np.maximum(self._route_lengths ** 2, 1e-12)
        t = np.sum((point - self._route_starts) * self._route_segments, axis=1) / denom
        t = np.clip(t, 0.0, 1.0)
        projected = self._route_starts + t[:, None] * self._route_segments
        distances = np.linalg.norm(projected - point, axis=1)
        index = int(np.argmin(distances))
        along = self._route_cumulative[index] + t[index] * self._route_lengths[index]
        return float(np.clip(along / max(self._route_total, 1e-9), 0.0, 1.0))

    def _init_collision_assets(self, term):
        """Build cached high-opacity Background gaussian point cloud.
        Splits into:
          * self._bg_points_np: full high-opacity point set (used for ground).
          * self._bg_points: obstacle-only subset (band-filtered around the
            global ground reference) on cuda for fast bg-collision tensor ops.
          * self._scene_bev_tree: KDTree over obstacle-band BEV positions, used
            to detect when the ego has driven outside the reconstructed scene.
        """
        import torch
        # Termination knobs (read early so they're available to band filter).
        self.term_bg_collision = bool(getattr(term, "bg_collision", True)) if term is not None else True
        self.bg_collision_threshold = int(getattr(term, "bg_threshold", 50)) if term is not None else 100
        self.ground_query_radius = float(getattr(term, "ground_query_radius", 4.0)) if term is not None else 4.0
        self.ground_quantile = float(getattr(term, "ground_quantile", 0.10)) if term is not None else 0.10
        self.term_follow_ground = bool(getattr(term, "follow_ground", True)) if term is not None else True
        # Obstacle band: keep BG points between [min_h, max_h] metres above
        # the global ground reference; both inputs and walls live in this band
        # for typical urban scenes.
        self.bg_obstacle_min_h = float(getattr(term, "bg_obstacle_min_h", 0.30)) if term is not None else 0.30
        self.bg_obstacle_max_h = float(getattr(term, "bg_obstacle_max_h", 2.50)) if term is not None else 2.50
        self.bg_opacity_min    = float(getattr(term, "bg_opacity_min", 0.30))    if term is not None else 0.30
        # Off-scene termination: if the ego BEV centre is further than
        # `off_scene_radius` metres from every obstacle-band BG point for
        # `off_scene_streak` consecutive steps, treat it as a bg collision
        # (the car drove out of the reconstructed area, where nothing exists
        # to collide with).
        self.term_off_scene = bool(getattr(term, "off_scene", True)) if term is not None else True
        self.off_scene_radius = float(getattr(term, "off_scene_radius", 5.0)) if term is not None else 5.0
        self.off_scene_streak = int(getattr(term, "off_scene_streak", 4)) if term is not None else 4
        self._off_scene_count = 0
        try:
            bg = self.renderer.trainer.models.get("Background")
            if bg is None:
                self._bg_points = None
                self._bg_points_np = None
                self._scene_bev_tree = None
            else:
                xyz = bg._means.detach()
                op = bg.get_opacity.squeeze(-1).detach()
                mask = op > self.bg_opacity_min
                pts_all = xyz[mask].contiguous()
                pts_all_np = pts_all.detach().cpu().numpy()
                lat_axis = self._ds_lat_axis
                fwd_axis = self._ds_fwd_axis
                up_axis  = self._ds_up_axis
                # Global ground reference along the up axis, derived from the
                # bulk of high-opacity points (a robust quantile is sturdier
                # than the per-point KNN trick, which classified wall tops as
                # obstacles and wall mid-heights as ground).
                ys = pts_all_np[:, up_axis]
                if self._ds_up_sign < 0:
                    # nuscenes: up = -y, ground = high y
                    ground_y = float(np.quantile(ys, 0.90))
                    above = ground_y - ys
                else:
                    ground_y = float(np.quantile(ys, 0.10))
                    above = ys - ground_y
                obstacle_mask = (above > self.bg_obstacle_min_h) & (above < self.bg_obstacle_max_h)
                pts_obs_np = pts_all_np[obstacle_mask]
                pts_obs = pts_all[torch.from_numpy(obstacle_mask).to(pts_all.device)].contiguous()
                self._bg_points = pts_obs       # obstacle-only, for collision
                self._bg_points_np = pts_all_np  # full, for ground query
                # BEV KDTree over the obstacle band -> used to detect when
                # ego drives outside the reconstructed scene.
                try:
                    from scipy.spatial import cKDTree
                    if len(pts_obs_np) > 0:
                        self._scene_bev_tree = cKDTree(pts_obs_np[:, [lat_axis, fwd_axis]])
                    else:
                        self._scene_bev_tree = None
                except Exception as _e:
                    print(f"[bg_collision] BEV tree unavailable: {_e}")
                    self._scene_bev_tree = None
                print(f"[bg_collision] op>{self.bg_opacity_min:.2f} points={len(pts_all_np)}, "
                      f"obstacles({self.bg_obstacle_min_h:.1f}-{self.bg_obstacle_max_h:.1f}m above ground)={len(pts_obs_np)}")
        except Exception as e:
            print(f"[bg_collision] init failed: {e}")
            self._bg_points = None
            self._bg_points_np = None
            self._scene_bev_tree = None
        # Reference ground at start; use only deltas so the camera keeps its
        # training-time mount height (do NOT bury the camera by adding the
        # absolute ground_y, which on nuscenes is ~+2.08m DS_y).
        gh0 = self._ground_height_at(0.0, 0.0)
        self._ground_ref = float(gh0) if gh0 is not None else 0.0
        if gh0 is not None:
            print(f"[ground] reference height (DS up-axis) = {self._ground_ref:.3f}")

    def _ground_height_at(self, vab0: float, vab1: float):
        """Return DS-world height (signed along DS up axis) of ground at the
        ego (vab0, vab1).  Computed as a low quantile of nearby Background
        gaussians' coordinate along the DS up axis.

        For nuscenes (DS_y is down) the returned value is the DS_y of the
        ground (positive number, since cam-front origin sits above ground).
        For pandaset (DS_z is up) the returned value is the ground's DS_z.
        Returns None if no points available.
        """
        if self._bg_points_np is None or len(self._bg_points_np) == 0:
            return None
        pts = self._bg_points_np
        # In DS world the (lateral, forward) of ego maps to specific DS axes.
        # nuscenes: lat = vab0 -> DS_x, fwd = vab1 -> DS_z.
        # pandaset: lat = vab0 -> DS_y, fwd = vab1 -> DS_x.
        lat_axis = self._ds_lat_axis
        fwd_axis = self._ds_fwd_axis
        up_axis  = self._ds_up_axis
        # Note: for pandaset _ego_pose_ds places ego at DS=(vab[1], vab[0], 0)
        # so x=vab[1]=fwd_in_DS, y=vab[0]=lat_in_DS (already accounted for above).
        d2 = (pts[:, lat_axis] - vab0) ** 2 + (pts[:, fwd_axis] - vab1) ** 2
        r2 = self.ground_query_radius ** 2
        m = d2 < r2
        if m.sum() < 8:
            # Fall back to nearest 32 points if the radius is too small.
            k = min(32, len(pts))
            idx = np.argpartition(d2, k - 1)[:k]
            ys = pts[idx, up_axis]
        else:
            ys = pts[m, up_axis]
        # For nuscenes (DS_y down): "ground" = high-y -> upper quantile.
        # For pandaset (DS_z up):   "ground" = low-z  -> lower quantile.
        if self._ds_up_sign < 0:
            # ground = high-y: take 1-quantile
            return float(np.quantile(ys, 1.0 - self.ground_quantile))
        else:
            return float(np.quantile(ys, self.ground_quantile))

    def _ego_box_8corners_ds(self) -> np.ndarray:
        """Return ego 3D box (8 corners) in DS world coordinates.
        Corner order: 0..7 with corner 1 = +length, corner 2 = +width,
        corner 5 = +height (matches HUGSIM bg_collision_det parallelepiped).
        """
        T = self._ego_c2w_ds()
        R = T[:3, :3]
        t = T[:3, 3]
        w, h, l = float(self.whl[0]), float(self.whl[1]), float(self.whl[2])
        # Body-frame axis indices per dataset:
        #   nuscenes: length=+body_z, width=+body_x, up=-body_y.
        #   pandaset: length=+body_y, width=+body_x, up=+body_z.
        if self._ds_dataset == "nuscenes":
            ax_l = R[:, 2]      # +z body
            ax_w = R[:, 0]      # +x body
            ax_h = -R[:, 1]     # -y body  (up)
        else:
            ax_l = R[:, 1]      # +y body  (forward)
            ax_w = R[:, 0]      # +x body
            ax_h = R[:, 2]      # +z body
        # Origin at ground under ego center.  Corners encode +/- l/2, +/- w/2, [0, h].
        # HUGSIM's parallelepiped uses: O, A=+L, B=+W, C(idx 5)=+H.
        O = t - 0.5 * l * ax_l - 0.5 * w * ax_w
        A = O + l * ax_l
        B = O + w * ax_w
        C = O + h * ax_h
        # Build all 8 corners; index 5 must be O+h*ax_h to match HUGSIM convention.
        c0 = O
        c1 = A
        c2 = B
        c3 = O + l * ax_l + w * ax_w
        c4 = O + h * ax_h + l * ax_l
        c5 = C
        c6 = O + h * ax_h + w * ax_w
        c7 = O + h * ax_h + l * ax_l + w * ax_w
        return np.stack([c0, c1, c2, c3, c4, c5, c6, c7], axis=0).astype(np.float32)

    def _bg_collision(self) -> bool:
        """Count high-opacity Background gaussians inside the ego 3D box.
        Mirrors HUGSIM's `bg_collision_det` parallelepiped test.
        """
        if self._bg_points is None:
            return False
        import torch as _torch
        verts = _torch.from_numpy(self._ego_box_8corners_ds()).to(self._bg_points.device)
        O, A, B, C = verts[0], verts[1], verts[2], verts[5]
        OA, OB, OC = A - O, B - O, C - O
        pts = self._bg_points
        POA = pts @ OA
        POB = pts @ OB
        POC = pts @ OC
        m = ((_torch.dot(O, OA) < POA) & (POA < _torch.dot(A, OA)) &
             (_torch.dot(O, OB) < POB) & (POB < _torch.dot(B, OB)) &
             (_torch.dot(O, OC) < POC) & (POC < _torch.dot(C, OC)))
        n = int(m.sum().item())
        if n > self.bg_collision_threshold:
            return True
        return False

    def _off_scene_check(self) -> bool:
        """True if the ego BEV centre has been further than `off_scene_radius`
        metres from every obstacle-band BG point for `off_scene_streak`
        consecutive steps. Catches the common failure mode where the planner
        drifts off the reconstructed road into empty space, where there are
        no gaussians for `_bg_collision` to register a hit against.
        """
        if self._scene_bev_tree is None:
            return False
        verts = self._ego_box_8corners_ds()
        center = verts.mean(axis=0)
        bev = np.array([center[self._ds_lat_axis], center[self._ds_fwd_axis]],
                       dtype=np.float32)
        d, _ = self._scene_bev_tree.query(bev, k=1)
        if float(d) > self.off_scene_radius:
            self._off_scene_count += 1
        else:
            self._off_scene_count = 0
        return self._off_scene_count >= self.off_scene_streak

    def _load_static_c2w_ds(self) -> Dict[str, np.ndarray]:
        """Per-camera c2w in the trained DS world (with frame-0 anchor
        normalisation applied), matching drivestudio's source loaders:

        * pandaset_sourceloader.PandaCameraData: anchor =
          ``ego_pose/{start_ts:03d}.txt``.
        * nuscenes_sourceloader.NuScenesCameraData: anchor =
          ``extrinsics/{start_ts:03d}_0.txt`` (FRONT camera at start
          frame).  Lidar-pose is NOT used as the world anchor.

        Using the wrong anchor for nuscenes gives every camera a
        ~lidar->front_cam offset (visible as ~90 deg yaw error and
        left/right cameras swapped).
        """
        ds_cfg = self.renderer.cfg
        try:
            data_root = str(ds_cfg.data.data_root)
            scene_idx = str(ds_cfg.data.scene_idx)
            start_ts  = int(getattr(ds_cfg.data, "start_timestep", 0))
            ds_dataset = str(getattr(ds_cfg.data, "dataset", "pandaset")).lower()
        except Exception as e:
            raise RuntimeError(
                f"DriveStudio config missing data.{{data_root,scene_idx,dataset}}: {e}"
            ) from e

        data_root_p = Path(data_root)
        if not data_root_p.is_absolute():
            data_root_p = resolve_data_root(data_root_p)
        scn_override = (getattr(self.cfg.scenario, "ds_scene_dir", None)
                        or getattr(self.cfg.scenario, "pandaset_scene_dir", None))
        if scn_override:
            scene_dir = Path(scn_override)
        else:
            # drivestudio stores scene_idx as int but processed dirs are
            # zero-padded ("003"); fall back to that form.
            scene_dir = data_root_p / scene_idx
            if not scene_dir.exists() and scene_idx.isdigit():
                scene_dir = data_root_p / f"{int(scene_idx):03d}"
        if not scene_dir.exists():
            raise FileNotFoundError(
                f"DS processed scene dir not found: {scene_dir}. "
                "Set scenario.ds_scene_dir in the scenario yaml.")

        # Stash for _ego_c2w_ds dispatch (different axis conventions per
        # dataset world anchor).
        self._ds_dataset = ds_dataset
        self._ds_up_axis = 1 if ds_dataset == "nuscenes" else 2  # index of DS-world up
        self._ds_up_sign = -1.0 if ds_dataset == "nuscenes" else +1.0  # +1 = up
        self._ds_lat_axis = 0 if ds_dataset == "nuscenes" else 1
        self._ds_fwd_axis = 2 if ds_dataset == "nuscenes" else 0
        # Ensure ActorState BEV round-trip respects the dataset axes.
        from aether.sim.actor import set_ds_dataset as _set_ds
        _set_ds(ds_dataset)

        # Per-dataset world anchor (matches drivestudio's PixelSource).
        if ds_dataset == "nuscenes":
            anchor_path = scene_dir / "extrinsics" / f"{start_ts:03d}_0.txt"
        else:
            anchor_path = scene_dir / "ego_pose" / f"{start_ts:03d}.txt"
        if not anchor_path.exists():
            raise FileNotFoundError(
                f"DS world anchor not found at {anchor_path} (dataset={ds_dataset}).")
        inv_anchor = np.linalg.inv(_load_4x4(anchor_path))
        self._ds_scene_dir = scene_dir
        self._ds_inv_anchor = inv_anchor

        out: Dict[str, np.ndarray] = {}
        for cam_name in self._cam_order:
            if cam_name not in _PANDA_CAM_ID:
                raise KeyError(
                    f"Unknown camera name {cam_name!r}; expected one of "
                    f"{sorted(_PANDA_CAM_ID)}.")
            cid = _PANDA_CAM_ID[cam_name]
            extr_path = scene_dir / "extrinsics" / f"{start_ts:03d}_{cid}.txt"
            raw_extr  = _load_4x4(extr_path)
            c2w_ds_0  = (inv_anchor @ raw_extr).astype(np.float32)
            out[cam_name] = c2w_ds_0
        return out

    @property
    def frame_idx(self) -> int:
        return int(round(self.timestamp / self.ego.dt))

    def _ego_c2w_ds(self) -> np.ndarray:
        if getattr(self, "_ds_dataset", "pandaset") == "nuscenes":
            T = _ego_pose_ds_nuscenes(self.ego.vab, self.ego.vr)
        else:
            T = _ego_pose_ds(self.ego.vab, self.ego.vr)
        # Vertical offset: follow terrain SLOPE relative to the start frame.
        # _ego_pose_ds(_nuscenes) places the camera at its training-time mount
        # height (DS_y=0 for nuscenes / DS_z=0 for pandaset), so we must add
        # ONLY the DELTA between current ground and start-frame ground.  Adding
        # the absolute height would bury the camera (~2m underground on
        # nuscenes), producing the foggy/blank renders seen in earlier runs.
        if getattr(self, "term_follow_ground", True):
            gh = self._ground_height_at(float(self.ego.vab[0]), float(self.ego.vab[1]))
            if gh is not None:
                delta = float(gh) - float(getattr(self, "_ground_ref", 0.0))
                if getattr(self, "_ds_dataset", "pandaset") == "nuscenes":
                    # DS_y is "down": ground sloping downward in real world
                    # increases DS_y, so add positive delta to descend.
                    T[1, 3] += delta
                else:
                    # Pandaset DS_z is up: ground rising increases DS_z.
                    T[2, 3] += delta
        return T.astype(np.float32)

    def _camera_c2w_ds(self, cam_name: str) -> np.ndarray:
        ego_ds = self._ego_c2w_ds()
        return (ego_ds @ self._c2w_static_ds[cam_name]).astype(np.float32)

    def _inject_actors(self, frame_idx: int) -> None:
        self.renderer.clear_actor_overrides()
        for entry in self.actors.pose_at(frame_idx):
            self.renderer.set_actor_pose(
                entry["class"], entry["instance_id"], frame_idx, entry["c2w"]
            )

    def _get_obs(self) -> Dict:
        f = min(self.frame_idx, self._max_frame)
        self._inject_actors(f)
        rgbs, depths = {}, {}
        camera_only = bool(getattr(self, "obs_cameras_only", True))
        for name in self._cam_order:
            c = self.cam_rig[name]
            cam_c2w = self._camera_c2w_ds(name)
            phys_cam_idx = _PANDA_CAM_ID[name]
            out = self.renderer.render(
                c2w_4x4=cam_c2w, K_3x3=c["K"], H=c["H"], W=c["W"],
                frame_idx=f, cam_idx=phys_cam_idx,
            )
            rgb = out["rgb"].clamp(0, 1).detach().cpu().numpy()
            rgbs[name] = (rgb * 255).astype(np.uint8)
            if not camera_only:
                depth = out["depth"][..., 0].detach().cpu().numpy().astype(np.float32)
                depths[name] = depth
        obs = {"rgb": rgbs}
        if not camera_only:
            obs["depth"] = depths
        if self.lidar is not None and depths:
            cam_K = {n: self.cam_rig[n]["K"] for n in depths}
            cam_c2w_ds = {n: self._camera_c2w_ds(n) for n in depths}
            obs["lidar"] = self.lidar.simulate(
                depths=depths, cam_K=cam_K, cam_c2w_ds=cam_c2w_ds,
                ego_c2w_ds=self._ego_c2w_ds(),
            )
        return obs

    def _ego_box_envframe(self):
        vt0 = float(self.ego.vab[0])
        vt2 = float(self.ego.vab[1])
        w, h, l = float(self.whl[0]), float(self.whl[1]), float(self.whl[2])
        yaw = float(self.ego.vr[1])
        return [vt2, -vt0, 0.0, w, l, h, -yaw]

    def _objs_list_envframe(self):
        boxes = []
        fwd_ax, lat_ax, up_ax = self._ds_fwd_axis, self._ds_lat_axis, self._ds_up_axis
        up_sign = float(self._ds_up_sign)
        # Permutation P maps DS WORLD axes -> env world (x=fwd, y=left, z=up).
        # NOTE: _ego_box_envframe uses ABSOLUTE vab coords, so obstacle boxes
        # MUST be in the same absolute env world.  A previous version applied
        # ``inv(ego_c2w_ds) @ actor`` which put obstacles in the EGO-RELATIVE
        # frame while the ego box stayed absolute -- as the ego drove forward
        # the two frames drifted apart and produced phantom fg collisions.
        P = np.zeros((3, 3), dtype=np.float32)
        P[0, fwd_ax] = 1.0
        P[1, lat_ax] = -1.0   # DS lat is right-positive, env y is left.
        P[2, up_ax]  = up_sign
        for entry in self.actors.pose_at(self.frame_idx):
            if "x" in entry and "y" in entry:
                # Traffic providers already report canonical road-local
                # coordinates (x forward, y left, yaw CCW). Avoid trying to
                # infer heading from the object-to-DS basis, whose canonical
                # object axes intentionally differ from DS world axes.
                w, l, h = entry["size"]
                boxes.append([
                    float(entry["x"]), float(entry["y"]), float(entry.get("z", 0.0)),
                    float(w), float(l), float(h), float(entry["yaw"]),
                ])
                continue
            T_world_ds = np.asarray(entry["c2w"], dtype=np.float32)
            R_env = (P @ T_world_ds[:3, :3] @ P.T).astype(np.float32)
            yaw_env = float(SCR.from_matrix(R_env).as_euler("ZYX")[0])
            t_env = P @ T_world_ds[:3, 3]   # absolute (fwd, left, up) env world
            w, l, h = entry["size"]
            # Match _ego_box_envframe: [box0=fwd, box1=left, box2=up, w,l,h, -yaw]
            boxes.append([
                float(t_env[0]), float(t_env[1]), float(t_env[2]),
                float(w), float(l), float(h), -yaw_env,
            ])
        return boxes

    def _get_info(self) -> Dict:
        return {
            "ego_pos":   [float(self.ego.vab[0]), 0.0, float(self.ego.vab[1])],
            "ego_rot":   self.ego.vr.tolist(),
            "ego_velo":  float(self.ego.velo),
            "ego_steer": float(self.ego.steer),
            "accelerate": float(self.ego.last_acc),
            "steer_rate": float(self.ego.last_steer_rate),
            "timestamp": float(self.timestamp),
            "command":   int(self._command_provider.command_at(
                self.ego.vab, self.ego.vr, self.frame_idx)),
            "ego_box":   self._ego_box_envframe(),
            "obj_boxes": self._objs_list_envframe(),
            "dynamic_removal_audit": dict(self._dynamic_removal_audit),
            "cam_params": {
                name: {
                    "intrinsic": c["intrinsic"],
                    "v2c": c["v2c"],
                    "l2c": c["l2c"],
                }
                for name, c in self.cam_rig.items()
            },
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.ego.reset()
        self.timestamp = 0.0
        self._step_count = 0
        self._off_scene_count = 0
        # CCD bookkeeping: cache previous ego/object poses for sweep tests.
        self._prev_ego_box = None
        self._prev_obj_boxes = None
        if self._traffic_provider is not None:
            self._traffic_provider.reset(self._traffic_ego_state())
        return self._get_obs(), self._get_info()

    def step(self, action):
        # Snapshot previous BEV poses for sub-step CCD before advancing.
        try:
            prev_ego_box = list(self._ego_box_envframe())
            prev_obj_boxes = [list(b) for b in self._objs_list_envframe()]
        except Exception:
            prev_ego_box = None
            prev_obj_boxes = None

        self.timestamp += self.ego.dt
        self._step_count += 1
        self.ego.step(action["acc"], action["steer_rate"])
        if self._traffic_provider is not None:
            self._traffic_provider.advance(self.timestamp, self._traffic_ego_state())

        terminated = False
        reward = 0.0
        fg_hit = bool(self.term_fg_collision and
                      self._fg_collision_swept(prev_ego_box, prev_obj_boxes))
        bg_hit = bool(self.term_bg_collision and self._bg_collision())
        off_scene_hit = bool(self.term_off_scene and self._off_scene_check())
        if off_scene_hit:
            bg_hit = True
        if fg_hit or bg_hit:
            terminated = True
            reward = -100.0
            kind = []
            if fg_hit: kind.append("fg")
            if bg_hit: kind.append("bg-offscene" if off_scene_hit else "bg")
            print(f"[collision] {'+'.join(kind)} at step={self._step_count} -> TERMINATE")

        truncated = self._step_count >= self.max_steps or self.frame_idx >= self._max_frame
        rc = self._route_progress()

        info = self._get_info()
        info["rc"] = rc
        # PandaSet processed scene120 has no audited drivable-area polygon.
        # Keep this explicitly unavailable instead of relabeling off_scene or
        # a centerline-distance heuristic as the paper's off-road metric.
        info["off_road"] = None
        info["collision"] = bool(fg_hit or bg_hit)
        info["fg_collision"] = fg_hit
        info["bg_collision"] = bg_hit
        info["off_scene"] = off_scene_hit
        return self._get_obs(), reward, terminated, truncated, info

    def close(self):
        if self._traffic_provider is not None:
            self._traffic_provider.close()
        return super().close()

    def _fg_collision(self) -> bool:
        try:
            from shapely.geometry import Polygon  # noqa: F401
        except ImportError:
            return False
        ego = self._ego_box_envframe()
        ep = _rect(ego[0], ego[1], ego[3], ego[4], ego[6])
        for ob in self._objs_list_envframe():
            op = _rect(ob[0], ob[1], ob[3], ob[4], ob[6])
            if not ep.intersects(op):
                continue
            inter = ep.intersection(op).area
            denom = min(ep.area, op.area) + 1e-6
            if inter / denom >= self.fg_collision_min_overlap:
                return True
        return False

    def _fg_collision_swept(self, prev_ego_box, prev_obj_boxes,
                            displacement_threshold: float = 1.0,
                            max_substeps: int = 8) -> bool:
        """Sub-step continuous-collision detection.

        Linearly interpolate ego/actor BEV boxes between the previous and
        current poses; if any participant moved more than
        ``displacement_threshold`` metres in a single env step we sample
        intermediate snapshots and run the standard polygon overlap test at
        each, catching tunnelling that the discrete check misses.
        """
        try:
            from shapely.geometry import Polygon  # noqa: F401
        except ImportError:
            return False

        cur_ego = self._ego_box_envframe()
        cur_objs = self._objs_list_envframe()

        # First-step (no prev cache) or shape mismatch -> fall back.
        if (prev_ego_box is None or prev_obj_boxes is None
                or len(prev_obj_boxes) != len(cur_objs)):
            return self._fg_collision()

        # Decide number of substeps from max linear displacement.
        def _disp(a, b):
            return math.hypot(b[0] - a[0], b[1] - a[1])
        max_d = _disp(prev_ego_box, cur_ego)
        for pb, cb in zip(prev_obj_boxes, cur_objs):
            d = _disp(pb, cb)
            if d > max_d:
                max_d = d
        if max_d <= displacement_threshold:
            return self._fg_collision()

        n_sub = min(max_substeps, max(2, int(math.ceil(max_d / displacement_threshold))))
        for k in range(1, n_sub + 1):
            t = k / float(n_sub)
            ego = [
                prev_ego_box[i] + t * (cur_ego[i] - prev_ego_box[i])
                for i in range(7)
            ]
            ep = _rect(ego[0], ego[1], ego[3], ego[4], ego[6])
            for pb, cb in zip(prev_obj_boxes, cur_objs):
                ob = [pb[i] + t * (cb[i] - pb[i]) for i in range(7)]
                op = _rect(ob[0], ob[1], ob[3], ob[4], ob[6])
                if not ep.intersects(op):
                    continue
                inter = ep.intersection(op).area
                denom = min(ep.area, op.area) + 1e-6
                if inter / denom >= self.fg_collision_min_overlap:
                    return True
        return False


def _rect(cx, cy, w, l, yaw):
    from shapely.geometry import Polygon
    c, s = math.cos(yaw), math.sin(yaw)
    dx = np.array([ l/2,  l/2, -l/2, -l/2])
    dy = np.array([ w/2, -w/2, -w/2,  w/2])
    xs = cx + dx * c - dy * s
    ys = cy + dx * s + dy * c
    return Polygon(list(zip(xs, ys)))
