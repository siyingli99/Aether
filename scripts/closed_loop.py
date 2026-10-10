"""Function: Run an AETHER closed-loop simulation with a UniAD driving policy.
Input/Output: Reads scenario, renderer, camera, and kinematic configs; writes trajectory, sensor, video, and audit artifacts.
Example:
    python scripts/closed_loop.py --scenario configs/scenarios/pandaset_095_aether.yaml \
        --base configs/base.yaml --camera configs/cameras/pandaset_800x450.yaml \
        --kinematic configs/kinematic.yaml --ad uniad --ad_cuda 0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from collections import defaultdict

# Make the `aether` package importable when running from repo root.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

# Ensure the running interpreter's bin dir is on PATH so torch JIT extensions
# can locate `ninja`, `nvcc`, etc. when the env isn't conda-activated.
_PY_BIN = os.path.dirname(sys.executable)
if _PY_BIN and _PY_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _PY_BIN + os.pathsep + os.environ.get("PATH", "")

import numpy as np
from moviepy import ImageSequenceClip
from PIL import Image, ImageDraw, ImageFont

import aether  # noqa: F401  (sets sys.path for drivestudio + hugsim)
from aether.config import load_cfg
from aether.envs.driving_env import AetherDrivingEnv
from aether.viz.bev import render_bev
from aether.agents.launcher import launch, check_alive, make_pipes, wait_adapter_ready, assert_adapter_alive
from aether.agents.registry import load_adapter, launch_adapter, list_adapters


def _resolve_output_dir(cfg, ad: str) -> str:
    root = cfg.base.output_dir
    scene = cfg.scenario.scene_name
    mode = getattr(cfg.scenario, "mode", "easy")
    dataset = getattr(cfg.base, "dataset", "drivestudio")
    return os.path.join(root, dataset, scene, ad, mode)


def _ad_path(cfg, ad: str) -> str:
    table = {
        "uniad": getattr(cfg.base, "uniad_path", None),
        "vad":   getattr(cfg.base, "vad_path",   None),
        "ltf":   getattr(cfg.base, "ltf_path",   None),
        "vla":      getattr(cfg.base, "vla_path", None),
        "impromptu":getattr(cfg.base, "vla_path", None),
    }
    p = table.get(ad)
    if not p:
        raise NotImplementedError(f"Unknown or unconfigured AD: {ad}")
    return p


def _trajectory_states(plan_traj):
    """Convert UniAD waypoints to canonical iLQR states.

    UniAD/HUGSIM emits ``(lateral_right, forward)``.  The iLQR bicycle model
    uses the conventional right-handed plane ``(forward, lateral_left)`` and
    a counter-clockwise-positive heading, so both the axis order and the
    lateral sign must be converted explicitly.
    """
    plan_traj = np.asarray(plan_traj, dtype=float).copy()
    if plan_traj.ndim != 2 or plan_traj.shape[1] < 2 or not len(plan_traj):
        raise ValueError(f"expected non-empty Nx2 trajectory, got {plan_traj.shape}")
    plan_traj = plan_traj[:, :2]
    if not np.isfinite(plan_traj).all():
        raise ValueError("trajectory contains non-finite values")
    plan_traj_stats = np.zeros((plan_traj.shape[0] + 1, 5))
    plan_traj_stats[1:, 0] = plan_traj[:, 1]   # forward
    plan_traj_stats[1:, 1] = -plan_traj[:, 0]  # left = -right
    deltas = np.diff(plan_traj_stats[:, :2], axis=0)
    plan_traj_stats[1:, 2] = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
    return plan_traj_stats


def _traj_to_action(plan_traj, info):
    """Reuse HUGSIM's iLQR-based plan->action conversion.

    Inlined ``traj2control`` from HUGSIM/sim/utils/sim_utils.py to avoid
    pulling in ``scene.cameras`` (which depends on a top-level ``utils``
    package that isn't on our path).
    """
    from aether.control.ilqr.lqr import plan2control

    plan_traj_stats = _trajectory_states(plan_traj)
    # iLQR steering is positive to the left.  BicycleEgo preserves HUGSIM's
    # opposite convention (positive to the right), so convert both the
    # current steering state and returned steering-rate command.
    curr_stat = np.array([0, 0, 0, info["ego_velo"], -info["ego_steer"]])
    acc, steer_rate_left = plan2control(plan_traj_stats, curr_stat)
    return acc, -steer_rate_left


def _camera_grid(obs):
    """Compose the six nuScenes cameras in their physical surround layout."""
    cams = list(obs.keys())
    # 6-cam grid layout if available, else single-cam strip
    front = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
    back  = ["CAM_BACK_LEFT",  "CAM_BACK",  "CAM_BACK_RIGHT"]
    use_grid = all(c in cams for c in front + back)
    if use_grid:
        row1 = np.concatenate([obs[c] for c in front], axis=1)
        row2 = np.concatenate([obs[c] for c in back],  axis=1)
        return np.concatenate([row1, row2], axis=0)
    return np.concatenate([obs[c] for c in cams], axis=1)


def _to_video(observations, out_path, fps=4):
    if not observations:
        return
    frames = [_camera_grid(obs) for obs in observations]
    ImageSequenceClip(frames, fps=fps).write_videofile(out_path, logger=None)


def _resize_to_height(frame, height):
    if int(frame.shape[0]) == int(height):
        return frame
    width = max(1, int(round(frame.shape[1] * height / frame.shape[0])))
    return np.asarray(Image.fromarray(frame).resize((width, int(height)), Image.Resampling.LANCZOS))


def _label(frame, text):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rounded_rectangle((12, 10, 190, 38), radius=7, fill=(0, 0, 0),
                           outline=(100, 110, 120))
    draw.text((23, 19), text, font=font, fill=(255, 255, 255))
    return np.asarray(image)


def _to_composite_video(observations, traffic_frames, out_path, fps=4):
    """Write synchronized LimSim-2D | AETHER-3D frames."""
    if not observations:
        return
    if len(observations) != len(traffic_frames):
        raise RuntimeError(
            f"composite streams differ: cameras={len(observations)}, "
            f"traffic={len(traffic_frames)}"
        )
    frames = []
    for obs, traffic in zip(observations, traffic_frames):
        camera = _label(_camera_grid(obs), "AETHER 3D / Difix")
        traffic = _label(traffic, "LimSim 2D / HD map")
        traffic = _resize_to_height(traffic, camera.shape[0])
        frames.append(np.concatenate((traffic, camera), axis=1))
    ImageSequenceClip(frames, fps=fps).write_videofile(out_path, logger=None)


def _rgb_digest(value):
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _write_json(path, value):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)


def _save_io_trace(trace_dir, cnt, info, obs, plan_traj, acc, steer_rate):
    """Persist per-step AD input/output for offline inspection."""
    import imageio.v2 as iio
    step_dir = os.path.join(trace_dir, f"step_{cnt:04d}")
    os.makedirs(step_dir, exist_ok=True)
    # JSON-friendly summary (no images, no large arrays).
    def _to_jsonable(x):
        import numpy as _np
        if isinstance(x, _np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {k: _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_to_jsonable(v) for v in x]
        if isinstance(x, (_np.floating, _np.integer)):
            return x.item()
        return x
    info_safe = {k: _to_jsonable(v) for k, v in info.items() if k != "cam_params"}
    summary = {
        "step": cnt,
        "input_to_ad": {
            "info": info_safe,
            "rgb_cameras": list(obs["rgb"].keys()),
            "rgb_shape": list(next(iter(obs["rgb"].values())).shape),
        },
        "output_from_ad": {
            "plan_traj": _to_jsonable(plan_traj),
            "plan_traj_note": "AD raw output; HUGSIM convention: column 0 = lateral (right+), column 1 = forward.",
        },
        "after_traj2control": {
            "acc": float(acc),
            "steer_rate": float(steer_rate),
        },
    }
    with open(os.path.join(step_dir, "io.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    # Save the 6-camera strip thumbnail that was actually fed to the AD.
    cams = list(obs["rgb"].keys())
    strip = np.concatenate([obs["rgb"][c] for c in cams], axis=1)
    iio.imwrite(os.path.join(step_dir, "rgb_strip.jpg"), strip, quality=70)
    # Raw obs+info pickle for full fidelity replay.
    with open(os.path.join(step_dir, "raw.pkl"), "wb") as f:
        pickle.dump({
            "obs": obs, "info": info,
            "plan_traj": np.asarray(plan_traj),
            "acc": float(acc), "steer_rate": float(steer_rate),
        }, f, protocol=4)



class PerfTimer:
    """Frame-resolution stopwatch.  Records named segments per frame and dumps
    summary statistics at the end."""
    def __init__(self):
        self.records = defaultdict(list)  # name -> list[seconds]
        self._t = None
        self._name = None

    def start(self, name):
        self._name = name
        self._t = time.perf_counter()

    def stop(self):
        if self._t is None:
            return
        self.records[self._name].append(time.perf_counter() - self._t)
        self._t = None
        self._name = None

    def section(self, name):
        timer = self
        class _Ctx:
            def __enter__(self_inner):
                timer.start(name)
                return self_inner
            def __exit__(self_inner, *a):
                timer.stop()
        return _Ctx()

    def summary(self):
        import statistics as st
        out = {}
        for name, vs in self.records.items():
            if not vs:
                continue
            vs_sorted = sorted(vs)
            out[name] = {
                "count": len(vs),
                "total_s": sum(vs),
                "mean_ms": 1e3 * sum(vs) / len(vs),
                "p50_ms":  1e3 * vs_sorted[len(vs)//2],
                "p95_ms":  1e3 * vs_sorted[min(len(vs)-1, int(0.95*len(vs)))],
                "max_ms":  1e3 * vs_sorted[-1],
            }
        return out

    def print_table(self, total_wall_s):
        rows = self.summary()
        if not rows:
            return
        order = sorted(rows.keys(), key=lambda k: -rows[k]["total_s"])
        print()
        print(f"[perf] frame breakdown (over {total_wall_s:.1f}s wall):")
        print(f"  {'segment':<22s}  {'n':>4s}  {'mean(ms)':>9s}  {'p50':>8s}  {'p95':>8s}  {'max':>8s}  {'tot(s)':>7s}  {'%wall':>6s}")
        for k in order:
            r = rows[k]
            pct = 100.0 * r["total_s"] / max(total_wall_s, 1e-6)
            print(f"  {k:<22s}  {r['count']:>4d}  {r['mean_ms']:>9.2f}  {r['p50_ms']:>8.2f}  {r['p95_ms']:>8.2f}  {r['max_ms']:>8.2f}  {r['total_s']:>7.2f}  {pct:>5.1f}%")
        print()


def run(cfg, output_dir: str, perf: "PerfTimer | None" = None,
        skip_io_trace: bool = False,
        refine: bool = False,
        refine_cuda: str = "0",
        refine_max_side: int = 1024,
        refine_model: str = "nvidia/difix",
        refine_timestep: int = 199,
        refine_dtype: str = "bf16",
        ad_spec=None,
        lidar_sidecar: str | None = None,
        lidar_bundle: str | None = None,
        splatad_lidar: bool = True,
        lidar_refine: bool = True,
        ad_proc=None,
        ad_log_path: str | None = None) -> None:
    if perf is None:
        perf = PerfTimer()
    difix_cfg = getattr(cfg.scenario, "difix", None)
    difix_required = bool(getattr(difix_cfg, "required", False)) \
        if difix_cfg is not None else False
    expected_cameras = int(getattr(difix_cfg, "expected_cameras", 6)) \
        if difix_cfg is not None else 6
    if difix_required and not refine:
        raise RuntimeError(
            "this scenario requires Difix on every AD camera input; "
            "remove --no_refine"
        )
    t_wall0 = time.perf_counter()
    with perf.section("env_init"):
        env = AetherDrivingEnv(cfg, output_dir)
    if ad_spec is not None and ad_spec.needs_depth:
        print("[aether] adapter requests depth/lidar — ensure scenario.lidar.enabled")
    elif ad_spec is not None and bool(ad_spec.observation.get("needs_lidar")):
        if env.lidar is None:
            print("[aether] WARN: adapter needs LiDAR but scenario.lidar.enabled is false")
        else:
            print("[aether] LiDAR in obs for AD adapter", flush=True)

    splatad = None
    needs_lidar = bool(ad_spec and ad_spec.observation.get("needs_lidar"))
    if needs_lidar and splatad_lidar:
        from aether.tools.splatad_lidar_obs import maybe_create_provider
        splatad = maybe_create_provider(
            env, cfg, lidar_sidecar, lidar_bundle, _REPO, needs_lidar=True,
            refine=lidar_refine,
        )
        if splatad is not None:
            print("[aether] SplatAD+refiner LiDAR will replace depth-backproj", flush=True)

    obs_pipe, plan_pipe = make_pipes(output_dir)
    print(f"[aether] pipes ready in {output_dir}")

    trace_dir = os.path.join(output_dir, "io_trace")
    os.makedirs(trace_dir, exist_ok=True)
    print(f"[aether] per-step IO trace -> {trace_dir}  (skip={skip_io_trace})")

    difix = None
    if refine:
        from aether.refine import DifixDaemon
        difix_work = os.path.join(output_dir, "refine_work")
        difix = DifixDaemon(
            model_name=refine_model,
            max_side=refine_max_side,
            timestep=refine_timestep,
            cuda_visible=refine_cuda,
            work_root=difix_work,
            dtype=refine_dtype,
        )
        difix.start()
        print(f"[aether] in-loop Difix refinement enabled (cuda={refine_cuda},"
              f" max_side={refine_max_side}, dtype={refine_dtype},"
              f" model={refine_model})")

    obs_save, info_save, bev_save, traffic_save = [], [], [], []
    refine_audit = []
    refine_audit_path = os.path.join(output_dir, "refine_audit.json")
    save = {"type": "closeloop", "frames": []}
    step_timings = []  # per-step wall + breakdown for diffusion-overhead analysis

    with perf.section("env_reset"):
        obs, info = env.reset()
    done = False
    cnt = 0
    while not done:
        t_step0 = time.perf_counter()
        t_refine_ms = 0.0
        if difix is not None:
            t_r0 = time.perf_counter()
            raw_rgb = {name: np.asarray(value) for name, value in obs["rgb"].items()}
            audit_entry = {
                "step": cnt,
                "timestamp": float(info["timestamp"]),
                "status": "failed",
                "cameras": {},
            }
            with perf.section("difix_refine"):
                try:
                    if len(raw_rgb) != expected_cameras:
                        raise RuntimeError(
                            f"expected {expected_cameras} camera inputs, got {len(raw_rgb)}"
                        )
                    refined = difix.refine_obs_rgb(raw_rgb)
                    if set(refined) != set(raw_rgb):
                        raise RuntimeError(
                            f"Difix camera set changed: input={sorted(raw_rgb)}, "
                            f"output={sorted(refined)}"
                        )
                    checked = {}
                    for camera in sorted(raw_rgb):
                        input_image = raw_rgb[camera]
                        output_image = np.asarray(refined[camera])
                        if output_image.shape != input_image.shape:
                            raise RuntimeError(
                                f"Difix shape mismatch for {camera}: "
                                f"{input_image.shape} -> {output_image.shape}"
                            )
                        if output_image.dtype != np.uint8:
                            raise RuntimeError(
                                f"Difix output for {camera} must be uint8, "
                                f"got {output_image.dtype}"
                            )
                        checked[camera] = output_image
                        audit_entry["cameras"][camera] = {
                            "shape": list(output_image.shape),
                            "input_sha256": _rgb_digest(input_image),
                            "output_sha256": _rgb_digest(output_image),
                        }
                    obs["rgb"] = checked
                    audit_entry["status"] = "success"
                except Exception as e:
                    audit_entry["error"] = str(e)
                    if difix_required:
                        refine_audit.append(audit_entry)
                        _write_json(refine_audit_path, refine_audit)
                        try:
                            difix.stop()
                        finally:
                            env.close()
                        raise RuntimeError(
                            f"mandatory Difix failed at step {cnt}; raw images "
                            "were not sent to the AD adapter"
                        ) from e
                    print(f"[difix] step {cnt}: refine failed ({e}); using raw")
            t_refine_ms = (time.perf_counter() - t_r0) * 1000.0
            audit_entry["duration_ms"] = t_refine_ms
            refine_audit.append(audit_entry)
            _write_json(refine_audit_path, refine_audit)
        if splatad is not None:
            if cnt == 0:
                print("[aether] first LiDAR-head render may JIT-compile CUDA kernels "
                      "(several minutes); export TORCH_CUDA_ARCH_LIST=8.9 to speed up",
                      flush=True)
            with perf.section("splatad_lidar"):
                try:
                    obs["lidar"] = splatad.enhance_obs_lidar()
                except Exception as e:
                    print(f"[splatad-lidar] step {cnt}: {e}; keeping depth-backproj")
        if env._external_vehicle_pool is not None:
            env._external_vehicle_pool.assert_original_dynamic_hidden()
        traffic_frame = env.render_traffic_2d()
        if traffic_frame is not None:
            traffic_save.append(traffic_frame)
        obs_save.append(obs["rgb"])
        info_save.append(info)

        assert_adapter_alive(ad_proc, ad_log_path)
        with perf.section("ipc_serialize"):
            payload = pickle.dumps((obs, info), protocol=4)
        with perf.section("ipc_write_obs"):
            with open(obs_pipe, "wb") as p:
                p.write(payload)
        t_ad0 = time.perf_counter()
        with perf.section("ad_roundtrip"):
            with open(plan_pipe, "rb") as p:
                raw = p.read()
        t_ad_ms = (time.perf_counter() - t_ad0) * 1000.0
        with perf.section("ipc_deserialize"):
            plan_traj = pickle.loads(raw)

        if plan_traj is None:
            print("[aether] AD adapter returned None; ending.")
            break

        with perf.section("traj2control"):
            acc, steer_rate = _traj_to_action(plan_traj, info)
        if not skip_io_trace:
            with perf.section("io_trace_dump"):
                try:
                    _save_io_trace(trace_dir, cnt, info, obs, plan_traj, acc, steer_rate)
                except Exception as e:
                    print(f"[io_trace] step {cnt}: {e}")
        t_env0 = time.perf_counter()
        with perf.section("env_step_render"):
            obs, _, terminated, truncated, info = env.step(
                {"acc": acc, "steer_rate": steer_rate}
            )
        t_env_ms = (time.perf_counter() - t_env0) * 1000.0
        cnt += 1
        done = terminated or truncated

        with perf.section("bev_render"):
            try:
                bev_save.append(render_bev(info))
            except Exception as e:
                print(f"[bev] step {cnt}: {e}")

        t_step_ms = (time.perf_counter() - t_step0) * 1000.0
        step_timings.append({
            "step": cnt - 1,
            "total_ms":   t_step_ms,
            "refine_ms":  t_refine_ms,
            "ad_ms":      t_ad_ms,
            "env_ms":     t_env_ms,
        })
        print(f"[step {cnt-1:03d}] t={info['timestamp']:.2f}s "
              f"v={info['ego_velo']:.2f}m/s rc={info.get('rc', 0):.2f} "
              f"| step={t_step_ms:6.0f}ms (refine={t_refine_ms:5.0f} "
              f"ad={t_ad_ms:5.0f} env={t_env_ms:5.0f})")

        save["frames"].append({
            "time_stamp": info["timestamp"],
            "is_key_frame": True,
            "ego_pos":   info["ego_pos"],
            "ego_rot":   info["ego_rot"],
            "ego_velo":  info["ego_velo"],
            "ego_steer": info["ego_steer"],
            "ego_box":   info["ego_box"],
            "obj_boxes": info["obj_boxes"],
            "obj_names": ["car"] * len(info["obj_boxes"]),
            "collision": info.get("collision", False),
            "bg_collision": info.get("bg_collision", False),
            "fg_collision": info.get("fg_collision", False),
            "off_scene":    info.get("off_scene", False),
            "off_road":     info.get("off_road"),
            "rc":        info.get("rc", 0.0),
            "planned_traj": {
                "traj": plan_traj.tolist() if hasattr(plan_traj, "tolist") else plan_traj,
                "timestep": 0.25,
            },
        })

    if difix is not None:
        try:
            difix.stop()
        except Exception as e:
            print(f"[difix] stop failed: {e}")

    if difix_required:
        failures = [entry for entry in refine_audit if entry.get("status") != "success"]
        if failures or len(refine_audit) != len(obs_save):
            env.close()
            raise RuntimeError(
                f"Difix audit incomplete: success={len(refine_audit) - len(failures)}, "
                f"camera_inputs={len(obs_save)}, failures={len(failures)}"
            )
    _write_json(refine_audit_path, refine_audit)
    _write_json(
        os.path.join(output_dir, "dynamic_removal_audit.json"),
        {
            "status": "success",
            "hidden_original_instances": dict(env._dynamic_removal_audit),
            "external_asset_slots": len(
                env._external_vehicle_pool.instance_ids
                if env._external_vehicle_pool is not None else ()
            ),
        },
    )

    # Notify adapter we're done.
    with open(obs_pipe, "wb") as p:
        p.write(pickle.dumps("Done", protocol=4))

    with open(os.path.join(output_dir, "data.pkl"), "wb") as f:
        pickle.dump([save], f)
    with open(os.path.join(output_dir, "infos.pkl"), "wb") as f:
        pickle.dump(info_save, f)
    with perf.section("video_encode"):
        _to_video(obs_save, os.path.join(output_dir, "video.mp4"))
        if traffic_save:
            _to_composite_video(
                obs_save, traffic_save,
                os.path.join(output_dir, "limsim_aether.mp4"),
            )
        if bev_save:
            ImageSequenceClip(bev_save, fps=4).write_videofile(
                os.path.join(output_dir, "bev.mp4"), logger=None
            )

    wall = time.perf_counter() - t_wall0
    perf.print_table(wall)
    perf_path = os.path.join(output_dir, "perf.json")
    with open(perf_path, "w") as f:
        json.dump({
            "scenario": getattr(cfg.scenario, "scene_name", "?"),
            "wall_s":   wall,
            "frames":   cnt,
            "obs_cameras_only": bool(getattr(env, "obs_cameras_only", True)),
            "skip_io_trace":    bool(skip_io_trace),
            "difix": {
                "required": difix_required,
                "expected_cameras": expected_cameras,
                "audited_inputs": len(refine_audit),
                "successful_inputs": sum(
                    entry.get("status") == "success" for entry in refine_audit
                ),
            },
            "dynamic_removal_audit": dict(env._dynamic_removal_audit),
            "composite_frames": len(traffic_save),
            "segments": perf.summary(),
            "step_timings_ms": step_timings,
        }, f, indent=2)
    print(f"[perf] wrote {perf_path}")
    if step_timings:
        import statistics as _st
        tot = [s["total_ms"] for s in step_timings]
        ref = [s["refine_ms"] for s in step_timings]
        ad  = [s["ad_ms"] for s in step_timings]
        env_ = [s["env_ms"] for s in step_timings]
        def _q(xs, q):
            xs = sorted(xs); k = int(round((len(xs)-1)*q)); return xs[k]
        print(f"[perf] per-step ms  total mean={_st.mean(tot):.0f} p50={_q(tot,.5):.0f} p90={_q(tot,.9):.0f} max={max(tot):.0f}")
        print(f"               refine mean={_st.mean(ref):.0f} p50={_q(ref,.5):.0f} p90={_q(ref,.9):.0f}")
        print(f"               ad     mean={_st.mean(ad):.0f}  p50={_q(ad,.5):.0f}  p90={_q(ad,.9):.0f}")
        print(f"               env    mean={_st.mean(env_):.0f} p50={_q(env_,.5):.0f} p90={_q(env_,.9):.0f}")
    env.close()
    print(f"[aether] artifacts written to {output_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario",  required=True)
    ap.add_argument("--base",      default="configs/base.yaml")
    ap.add_argument("--camera",    default="configs/cameras/pandaset_800x450.yaml")
    ap.add_argument("--kinematic", default="configs/kinematic.yaml")
    ap.add_argument("--ad",        default="vla")
    ap.add_argument("--ad_cuda",   default="0")
    ap.add_argument("--no_io_trace", action="store_true",
                    help="Skip per-step io_trace dump (recommended for perf runs).")
    ap.add_argument("--ad_config", default=None,
                    help="Path to adapter YAML; defaults to configs/adapters/<ad>.yaml.")
    refine_grp = ap.add_mutually_exclusive_group()
    refine_grp.add_argument("--refine", dest="refine", action="store_true",
                            help="Difix-refine RGB before AD (default ON when --ad set).")
    refine_grp.add_argument("--no_refine", dest="refine", action="store_false")
    ap.set_defaults(refine=None)
    ap.add_argument("--refine_cuda", default=None,
                    help="GPU for Difix (default: same as --ad_cuda).")
    ap.add_argument("--refine_max_side", type=int, default=1024,
                    help="Long-edge inference resolution for Difix.")
    ap.add_argument("--refine_model", default="nvidia/difix",
                    help="HF model id for Difix (nvidia/difix or nvidia/difix_ref).")
    ap.add_argument("--refine_timestep", type=int, default=199)
    ap.add_argument("--refine_dtype", default="bf16",
                    choices=["bf16", "fp16", "fp32"],
                    help="Inference dtype for the Difix UNet/VAE.")
    ap.add_argument("--lidar_sidecar", default=None,
                    help="SplatAD sidecar .pt; auto-detected from joint checkpoint dir if omitted.")
    ap.add_argument("--lidar_bundle", default=None,
                    help="Unified LiDAR refiner bundle (default: multimodal_refine_v2 unified).")
    ap.add_argument("--no_splatad_lidar", action="store_true",
                    help="Keep depth-backproj LiDAR instead of SplatAD+refiner.")
    ap.add_argument("--no_lidar_refine", action="store_true",
                    help="Use the configured full-RD LiDAR sidecar directly.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.refine is None:
        args.refine = args.ad not in ("none", "", None)
    if args.refine_cuda is None:
        args.refine_cuda = args.ad_cuda

    cfg = load_cfg(args.scenario, args.base, args.camera, args.kinematic)
    import random as _random
    import torch as _torch
    _random.seed(args.seed)
    np.random.seed(args.seed)
    _torch.manual_seed(args.seed)
    _torch.cuda.manual_seed_all(args.seed)
    out = _resolve_output_dir(cfg, args.ad)
    os.makedirs(out, exist_ok=True)
    print(f"[aether] output -> {out}")
    # Create FIFOs BEFORE launching the AD adapter. Adapters that print "ready"
    # then immediately open(obs_pipe) race and FileNotFoundError if pipes come later.
    make_pipes(os.path.abspath(out))

    spec = None
    # Prefer declarative adapter spec; fall back to legacy `cfg.base.{ad}_path`.
    try:
        if args.ad_config:
            import yaml as _y
            with open(args.ad_config) as _f:
                _data = _y.safe_load(_f) or {}
            from aether.agents.registry import AdapterSpec
            _launcher = dict(_data.get("launcher") or {})
            for _key in ("entrypoint", "cwd"):
                if not _launcher.get(_key):
                    continue
                _value = os.path.expandvars(os.path.expanduser(str(_launcher[_key])))
                if not os.path.isabs(_value):
                    _value = os.path.join(_REPO, _value)
                _launcher[_key] = os.path.abspath(_value)
            spec = AdapterSpec(name=_data.get("name", args.ad),
                               display_name=_data.get("display_name", args.ad),
                               launcher=_launcher,
                               ipc=_data.get("ipc") or {},
                               observation=_data.get("observation") or {},
                               output=_data.get("output") or {},
                               capabilities=_data.get("capabilities") or {},
                               source_path=args.ad_config)
        else:
            spec = load_adapter(args.ad, legacy_cfg=cfg.base)
        proc = launch_adapter(spec, args.ad_cuda, os.path.abspath(out))
        ad_log = os.path.join(os.path.abspath(out), "ad_output.txt")
        wait_adapter_ready(
            proc, ad_log,
            timeout_s=float(os.environ.get("AETHER_AD_READY_TIMEOUT", "1800")),
        )
        print(f"[aether] adapter '{args.ad}' cameras={spec.cameras or 'all'} "
              f"needs_lidar={spec.observation.get('needs_lidar', False)}", flush=True)
    except FileNotFoundError:
        proc = launch(_ad_path(cfg, args.ad), args.ad_cuda, os.path.abspath(out))
        ad_log = os.path.join(os.path.abspath(out), "ad_output.txt")
        wait_adapter_ready(
            proc, ad_log,
            timeout_s=float(os.environ.get("AETHER_AD_READY_TIMEOUT", "1800")),
        )
    try:
        run(cfg, out,
            skip_io_trace=args.no_io_trace,
            refine=args.refine,
            refine_cuda=args.refine_cuda,
            refine_max_side=args.refine_max_side,
            refine_model=args.refine_model,
            refine_timestep=args.refine_timestep,
            refine_dtype=args.refine_dtype,
            ad_spec=spec,
            lidar_sidecar=args.lidar_sidecar,
            lidar_bundle=args.lidar_bundle,
            splatad_lidar=not args.no_splatad_lidar,
            lidar_refine=not args.no_lidar_refine,
            ad_proc=proc,
            ad_log_path=ad_log)
        check_alive(proc)
    except Exception:
        import traceback
        traceback.print_exc()
        proc.kill()
        raise
