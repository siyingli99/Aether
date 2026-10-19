"""One closed-loop episode, start to finish, with all mandatory outputs.

Run directory (``outputs/runs/<scene>-<agent>-<YYYYmmdd-HHMMSS>/``)::

    run.json                  scenario, agent, versions, timing
    metrics.json              closed-loop metrics (see aether.eval.metrics)
    trajectories.csv          every vehicle at every simulation step
    trajectories_10hz.csv     the same on a uniform 10 Hz clock
    events.csv                collisions / off-road / off-scene
    videos/cameras.mp4        rendered camera views, one frame per step
    videos/bev.mp4            smooth bird's-eye view (20 fps, interpolated)
    videos/overview.mp4       BEV | cameras side by side, same clock
    agent/                    planner work dir (FIFOs, planner.log, its own outputs)
    logs/steps.jsonl          per-step timing, controls, route command and the
                              tracked plan (rear-axle frame)
    logs/io/, render_poses.json   only with ``output.io_trace``: agent inputs
                              and scene-frame render poses per step

All videos share one time axis: second *t* of every video shows simulation
time *t*.  The camera video is played at 1/dt fps (one frame per step); the
BEV and overview videos interpolate vehicle motion between steps.
"""
from __future__ import annotations

from dataclasses import dataclass
import datetime as _dt
import json
import math
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from .. import __version__
from ..agents import Control, EgoAgent, Plan, create_agent
from ..agents.observation import build_observation
from ..agents.tracker import TrajectoryTracker
from ..config import Scenario
from ..eval.metrics import score_episode
from ..paths import OUTPUTS_ROOT, resolve
from ..record.bev import BEVRenderer
from ..record.events import write_events
from ..record.trajectory import TrajectoryRecorder
from ..record.video import VideoWriter, camera_grid, label, resize_height
from .env import AetherEnv
from .route import COMMAND_NAMES

Listener = Callable[[dict[str, Any]], None]


@dataclass
class RunResult:
    path: Path
    metrics: dict[str, Any]
    steps: int
    terminated_by: str | None


def run_dir_for(scenario: Scenario, agent_name: str, root: str | Path | None = None) -> Path:
    base = resolve(root) if root else OUTPUTS_ROOT / "runs"
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{scenario.name or scenario.scene}-{agent_name}-{stamp}".replace("/", "_")
    path = base / name
    path.mkdir(parents=True, exist_ok=False)
    for sub in ("videos", "logs"):
        (path / sub).mkdir()
    return path


class _Refiner:
    """Optional Difix image refinement before the agent sees the cameras."""

    def __init__(self, cfg, cuda: str) -> None:
        from ..refine import DifixDaemon
        self.required = cfg.required
        self.daemon = DifixDaemon(model_name=cfg.model, max_side=cfg.max_side,
                                  timestep=cfg.timestep, cuda_visible=cfg.cuda or cuda,
                                  dtype=cfg.dtype)
        self.daemon.start()
        self.failures = 0

    def __call__(self, images: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        try:
            refined = self.daemon.refine_obs_rgb(images)
            if set(refined) != set(images) or any(
                    np.asarray(refined[c]).shape != images[c].shape for c in images):
                raise RuntimeError("Difix changed the camera set or image shapes")
            return {c: np.asarray(refined[c], dtype=np.uint8) for c in images}
        except Exception as error:
            self.failures += 1
            if self.required:
                raise RuntimeError("Difix is required by the scenario and failed; raw renders "
                                   "were not given to the agent") from error
            print(f"[refine] failed ({error}); using raw renders")
            return images

    def close(self) -> None:
        try:
            self.daemon.stop()
        except Exception as error:
            print(f"[refine] stop failed: {error}")


class _Videos:
    """Camera / BEV / overview writers on a common clock."""

    def __init__(self, path: Path, env: AetherEnv, recorder: TrajectoryRecorder, fps: int,
                 listener: Listener | None) -> None:
        self.env, self.recorder, self.fps = env, recorder, int(fps)
        self.dt = env.dt
        self.listener = listener
        self.bev = BEVRenderer(env.bundle.hdmap, env.route.xy)
        self.cam = VideoWriter(path / "cameras.mp4", 1.0 / env.dt)
        self.bev_video = VideoWriter(path / "bev.mp4", self.fps)
        self.overview = VideoWriter(path / "overview.mp4", self.fps)
        self.trail: list[tuple[float, float]] = []
        self.next_time = 0.0
        self.plan_world: np.ndarray | None = None
        self.alerts: list[str] = []
        self.collided: set[str] = set()

    def camera(self, images: dict[str, np.ndarray], info: dict) -> np.ndarray:
        grid = camera_grid(images)
        grid = label(grid, f"t = {info['time']:5.2f} s   step {info['step']}")
        self.cam.write(grid)
        self._grid = grid
        if self.listener:
            self.listener({"kind": "camera", "time": info["time"], "image": grid})
        return grid

    def _bev_frame(self, t: float) -> np.ndarray:
        states = self.recorder.sample(t)
        ego = states["ego"]
        trail = [p for p in self.trail] + [(ego["x"], ego["y"])]
        hud = [f"t = {t:5.2f} s", f"v = {ego['speed']:4.1f} m/s  ({ego['speed'] * 3.6:4.1f} km/h)",
               f"command: {self._command}", f"vehicles: {len(states) - 1}"]
        return self.bev.frame((ego["x"], ego["y"]), states, trail, self.plan_world, hud,
                              self.alerts, self.collided)

    def advance(self, until: float, info: dict, include_end: bool = False) -> None:
        """Emit BEV/overview frames for times in [next_time, until)."""
        self._command = COMMAND_NAMES.get(info.get("command"), "-")
        step = 1.0 / self.fps
        while self.next_time < until - 1e-9 or (include_end and self.next_time <= until + 1e-9):
            frame = self._bev_frame(self.next_time)
            self.bev_video.write(frame)
            cams = resize_height(self._grid, frame.shape[0])
            self.overview.write(np.concatenate([frame, cams], axis=1))
            if self.listener:
                self.listener({"kind": "bev", "time": self.next_time, "image": frame})
            ego = self.recorder.sample(self.next_time)["ego"]
            self.trail.append((ego["x"], ego["y"]))
            self.next_time = round(self.next_time + step, 9)

    def close(self) -> dict[str, int]:
        for w in (self.cam, self.bev_video, self.overview):
            w.close()
        return {"cameras": self.cam.frames, "bev": self.bev_video.frames,
                "overview": self.overview.frames}


def _plan_to_world(plan: Plan, env: AetherEnv, origin: np.ndarray) -> np.ndarray:
    from ..agents.tracker import rear_axle_reference
    ref = rear_axle_reference(plan, origin)
    s = env.state
    c, si = math.cos(s.yaw), math.sin(s.yaw)
    off = env.profile.center_offset
    cx = ref[:, 0] + off * np.cos(ref[:, 2])
    cy = ref[:, 1] + off * np.sin(ref[:, 2])
    return np.column_stack([s.x + c * cx - si * cy, s.y + si * cx + c * cy])[1:]


class _IOTrace:
    """Per-step agent inputs and render poses (``output.io_trace``)."""

    def __init__(self, run: Path, env: AetherEnv) -> None:
        self.dir = run / "logs" / "io"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = run / "logs" / "render_poses.json"
        self.env = env
        self.steps: list[dict[str, Any]] = []

    def add(self, info: dict, images: dict[str, np.ndarray]) -> None:
        from PIL import Image
        env = self.env
        Image.fromarray(np.concatenate([images[c] for c in images], axis=1)).save(
            self.dir / f"step_{info['step']:04d}.jpg", quality=90)
        self.steps.append({
            "step": info["step"], "time": info["time"], "frame_idx": env.frame,
            "ego_c2w_ds": env.sensor_pose("ANCHOR").tolist(),
            "vehicle_c2w_ds": env.sensor_pose("VEHICLE").tolist(),
            "cameras": {c: {"c2w": env.camera_pose(c).tolist(), "K": np.asarray(env.cameras[c].K).tolist(),
                            "H": int(images[c].shape[0]), "W": int(images[c].shape[1]),
                            "index": int(env.cameras[c].index)} for c in images},
            "actors": [{"class": cls, "instance_id": int(i), "c2w": np.asarray(pose).tolist()}
                       for cls, i, pose in env.traffic.render_overrides()],
        })

    def close(self) -> None:
        self.path.write_text(json.dumps({"cam_order": list(self.steps[0]["cameras"]) if self.steps else [],
                                         "frame_dt": self.env.frame_dt, "steps": self.steps}))


def plan_origin(env: AetherEnv, frame: str) -> np.ndarray:
    """(forward, left) of a plan frame relative to the rear axle."""
    if frame == "rear_axle":
        return np.zeros(2)
    if frame == "center":
        return np.array([env.profile.center_offset, 0.0])
    if frame in env.geometry.camera_mounts:
        return np.asarray(env.geometry.camera_mounts[frame])[:2, 3]
    if frame in ("lidar", "LIDAR_TOP"):
        return np.asarray(env.rig.lidar_mount())[:2, 3]
    raise ValueError(f"unknown plan frame {frame!r}")


def run_episode(scenario: Scenario, agent: str | EgoAgent | None = None,
                output_root: str | Path | None = None, cuda: str | None = None,
                device: str = "cuda", listener: Listener | None = None,
                verbose: bool = True) -> RunResult:
    agent_name = agent if isinstance(agent, str) else (
        type(agent).__name__ if agent is not None else scenario.agent.name)
    if agent_name is None:
        raise ValueError("no agent: pass one or set scenario.agent.name")
    cuda = cuda or scenario.agent.cuda
    run = run_dir_for(scenario, Path(str(agent_name)).stem.split(":")[-1], output_root or scenario.output.root)
    started = time.time()
    meta: dict[str, Any] = {"aether_version": __version__, "scene": scenario.scene,
                            "agent": str(agent_name), "started": _dt.datetime.now().isoformat(),
                            "scenario": scenario.model_dump(mode="json")}
    (run / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    if verbose:
        print(f"[aether] run → {run}")

    env = videos = refiner = trace = None
    ego_agent: EgoAgent | None = None
    terminated_by = None
    steps_log = open(run / "logs" / "steps.jsonl", "w")
    try:
        env = AetherEnv(scenario, device=device)
        ego_agent = agent if isinstance(agent, EgoAgent) else create_agent(
            str(agent_name), run, cuda, **scenario.agent.options)
        cameras = ego_agent.cameras if ego_agent.cameras is not None else env.rig.camera_names
        cameras = [c for c in cameras if c in env.cameras]
        video_cams = cameras or env.rig.camera_names
        lidar = None
        if ego_agent.needs_lidar:
            from ..sensors.provider import LidarProvider
            lidar = LidarProvider(env, scenario.lidar)
        if scenario.refine.enabled and cameras:
            refiner = _Refiner(scenario.refine, cuda)
        ego_agent.setup({"scene": env.bundle.id, "profile": env.profile, "dt": env.dt,
                         "frame_dt": env.frame_dt, "route_xy": env.route.xy,
                         "camera_names": cameras, "lidar_mount": env.rig.lidar_mount(),
                         "workdir": str(run / "agent")})
        ego_agent.reset()
        tracker = TrajectoryTracker(env.profile, scenario.sim.tracker, env.dt)
        recorder = TrajectoryRecorder(env.bundle.hdmap, env.geometry.ground)

        info = env.reset()
        recorder.add(info)
        if scenario.output.io_trace:
            trace = _IOTrace(run, env)
        if scenario.output.videos:
            videos = _Videos(run / "videos", env, recorder, scenario.output.bev_fps, listener)
        while True:
            t0 = time.perf_counter()
            need_depth = lidar is not None and lidar.needs_depth
            rendered = env.render(sorted(set(video_cams) | set(cameras)), depth=need_depth)
            t_render = time.perf_counter() - t0
            t_refine = 0.0
            if refiner is not None:
                t_r = time.perf_counter()
                rendered["rgb"].update(refiner({c: rendered["rgb"][c] for c in cameras}))
                t_refine = time.perf_counter() - t_r
            if videos:
                videos.camera({c: rendered["rgb"][c] for c in video_cams}, info)
            if trace is not None and cameras:
                trace.add(info, {c: rendered["rgb"][c] for c in cameras})
            points = lidar.sweep(rendered.get("depth")) if lidar is not None else None
            obs = build_observation(env, rendered, info, cameras, ego_agent.image_size, points)

            t1 = time.perf_counter()
            action = ego_agent.act(obs)
            t_agent = time.perf_counter() - t1
            if action is None:
                terminated_by = "agent_stopped"
                break
            plan_log = None
            if isinstance(action, Plan):
                origin = plan_origin(env, action.frame)
                from ..agents.tracker import rear_axle_reference
                plan_log = np.round(rear_axle_reference(action, origin)[1:, :2], 3).tolist()
                accel, steer_rate = tracker.control(action, env.state.v, env.state.steer, origin)
                if videos:
                    videos.plan_world = _plan_to_world(action, env, origin)
            elif isinstance(action, Control):
                accel, steer_rate = action.accel, action.steer_rate
            else:
                raise TypeError(f"agent returned {type(action).__name__}; expected Plan or Control")

            t2 = time.perf_counter()
            result = env.step(accel, steer_rate)
            info = result.info
            recorder.add(info)
            if videos:
                if info["events"]:
                    videos.collided |= {e["other"] for e in info["events"] if e.get("other")}
                    videos.alerts = [f"{e.type.replace('_', ' ')} at t={e.time:.2f}s"
                                     for e in env.events[-3:]]
                videos.advance(info["time"], info)
            t_step = time.perf_counter() - t2
            steps_log.write(json.dumps({"step": info["step"], "time": info["time"],
                                        "accel": accel, "steer_rate": steer_rate,
                                        "render_s": round(t_render, 4), "refine_s": round(t_refine, 4),
                                        "agent_s": round(t_agent, 4),
                                        "step_s": round(t_step, 4), "command": info.get("command"),
                                        "steer": round(env.state.steer, 4),
                                        "plan_rear_axle": plan_log,
                                        "events": [e["type"] for e in info["events"]]}) + "\n")
            steps_log.flush()
            if verbose:
                ev = ",".join(e["type"] for e in info["events"]) or "-"
                print(f"[step {info['step']:03d}] t={info['time']:5.2f}s v={info['ego']['speed']:5.2f} "
                      f"a={accel:+.2f} rc={info['route_progress']:.2f} events={ev} "
                      f"render={t_render * 1e3:.0f}ms refine={t_refine * 1e3:.0f}ms "
                      f"agent={t_agent * 1e3:.0f}ms")
            if result.terminated:
                terminated_by = info["terminated_by"]
                break
            if result.truncated:
                terminated_by = "time_limit"
                break
        # Final camera frame for the end state, and the last BEV frames.
        if videos:
            rendered = env.render(video_cams)
            videos.camera(rendered["rgb"], info)
            videos.advance(info["time"] + env.dt, info)

        events = [e.to_dict() for e in env.events]
        write_events(run / "events.csv", events)
        files = recorder.write(run, scenario.output.trajectory_hz)
        expert = lambda t: env.route.progress(*_logged_center(env, t))  # noqa: E731
        metrics = score_episode(recorder.steps, events, env.route, expert, terminated_by).to_dict()
        (run / "metrics.json").write_text(json.dumps(metrics, indent=2))
        meta.update(finished=_dt.datetime.now().isoformat(), wall_s=round(time.time() - started, 2),
                    steps=info["step"], terminated_by=terminated_by,
                    vehicle=env.profile.name, files={k: v.name for k, v in files.items()},
                    videos=videos.close() if videos else {}, refine_failures=getattr(refiner, "failures", 0),
                    lidar=getattr(lidar, "source", None), collision_stats=env.collision.stats)
        videos = None
        (run / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
        if verbose:
            print(f"[aether] {terminated_by}: score={metrics['score']:.3f} nc={metrics['nc']} "
                  f"dac={metrics['dac']} ep={metrics['ep']:.2f} rc={metrics['rc']:.2f} → {run}")
        return RunResult(run, metrics, info["step"], terminated_by)
    finally:
        steps_log.close()
        if trace is not None:
            trace.close()
        if videos is not None:
            videos.close()
        if refiner is not None:
            refiner.close()
        if ego_agent is not None:
            ego_agent.close()
        if env is not None:
            env.close()


def _logged_center(env: AetherEnv, t: float) -> tuple[float, float]:
    xy = env.route.xy
    i = min(t / env.frame_dt, len(xy) - 1)
    k = int(math.floor(i))
    u = i - k
    if k + 1 >= len(xy):
        return float(xy[-1, 0]), float(xy[-1, 1])
    p = (1 - u) * xy[k] + u * xy[k + 1]
    return float(p[0]), float(p[1])
