"""Function: Run a closed-loop rollout with enhanced RGB and LiDAR streams.
Input/Output: Reads scene configs plus an existing LiDAR sidecar and refiner bundle; writes rollout, sensor, and visualization artifacts.
Example:
    # Prerequisite: generate the sidecar and bundle at these repository-relative paths.
    python scripts/closed_loop_multimodal.py \
        --scenario configs/scenarios/pandaset_095_multimodal.yaml \
        --base configs/base.yaml --camera configs/cameras/pandaset_800x450.yaml \
        --kinematic configs/kinematic.yaml --ad none \
        --lidar_sidecar outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt \
        --lidar_bundle outputs/multimodal_refine_v2/unified/unified_multimodal.bundle.pth \
        --max_steps 12
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import imageio.v2 as imageio

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
_PY_BIN = os.path.dirname(sys.executable)
if _PY_BIN and _PY_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _PY_BIN + os.pathsep + os.environ.get("PATH", "")

import numpy as np
import torch
from moviepy import ImageSequenceClip

import aether  # noqa: F401  (sets sys.path for drivestudio)
from aether.config import load_cfg
from aether.envs.driving_env import AetherDrivingEnv
from aether.render.lidar_closed_loop import ClosedLoopLidar
from aether.refine.unified_multimodal_enhancer import load_unified_bundle
from aether.tools.multimodal_cl_viz import (
    build_ref_bank, sweep_arrays, enhance_sweep, spherical_to_points,
    project_points_on_image, make_step_panel,
)


def _resolve_output_dir(cfg, ad):
    root = cfg.base.output_dir
    scene = cfg.scenario.scene_name
    mode = getattr(cfg.scenario, "mode", "easy")
    dataset = getattr(cfg.base, "dataset", "drivestudio")
    return os.path.join(root, dataset, scene, f"{ad}_multimodal", mode)


def _traj_to_action(plan_traj, info):
    from aether.control.ilqr.lqr import plan2control
    plan_traj = np.asarray(plan_traj, dtype=float).copy()
    stats = np.zeros((plan_traj.shape[0] + 1, 5))
    stats[1:, 0] = plan_traj[:, 1]
    stats[1:, 1] = -plan_traj[:, 0]
    deltas = np.diff(stats[:, :2], axis=0)
    stats[1:, 2] = np.unwrap(np.arctan2(deltas[:, 1], deltas[:, 0]))
    curr = np.array([0, 0, 0, info["ego_velo"], -info["ego_steer"]])
    acc, steer_rate_left = plan2control(stats, curr)
    return acc, -steer_rate_left


def _front_cam_name(env):
    for n in ("CAM_FRONT", "CAM_FRONT_LEFT"):
        if n in env.cam_rig:
            return n
    return list(env.cam_rig.keys())[0]


_AD_STRIP_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]


def _save_ad_rgb_strip(obs_rgb: dict, path: str) -> None:
    """Save the 6-camera strip actually pickled to the AD adapter."""
    cams = [c for c in _AD_STRIP_ORDER if c in obs_rgb] or sorted(obs_rgb.keys())
    strip = np.concatenate([obs_rgb[c] for c in cams], axis=1)
    imageio.imwrite(path, strip, quality=85)


def _save_distill_corpus_step(corpus_dir: str, step: int, obs_rgb: dict,
                              pr, pi, pk, in_r, in_i, in_k) -> None:
    """Per-step enhanced RGB + LiDAR for trajectory-based distillation."""
    from PIL import Image

    rgb_dir = os.path.join(corpus_dir, "rgb", f"{step:04d}")
    os.makedirs(rgb_dir, exist_ok=True)
    for cam, img in obs_rgb.items():
        Image.fromarray(img).save(os.path.join(rgb_dir, f"{cam}.png"))
    lidar_dir = os.path.join(corpus_dir, "lidar")
    os.makedirs(lidar_dir, exist_ok=True)
    np.savez(
        os.path.join(lidar_dir, f"{step:04d}.npz"),
        enh_range=pr.astype(np.float32),
        enh_int=pi.astype(np.float32),
        enh_keep=pk.astype(np.float32),
        raw_range=in_r.astype(np.float32),
        raw_int=in_i.astype(np.float32),
        raw_keep=in_k.astype(np.float32),
    )


def _record_step_trajectory(env, step_idx: int, info: dict) -> dict:
    """Serialize per-step 6-camera poses for post-distillation re-rendering."""
    frame_idx = min(env.frame_idx, env._max_frame)
    rec = {
        "step": int(step_idx),
        "timestamp": float(info["timestamp"]),
        "frame_idx": int(frame_idx),
        "ego_c2w_ds": env._ego_c2w_ds().tolist(),
        "cameras": {},
        "actors": [],
    }
    for name in env._cam_order:
        c = env.cam_rig[name]
        rec["cameras"][name] = {
            "c2w": env._camera_c2w_ds(name).tolist(),
            "K": c["K"].tolist(),
            "H": int(c["H"]),
            "W": int(c["W"]),
        }
    for entry in env.actors.pose_at(frame_idx):
        rec["actors"].append({
            "class": entry["class"],
            "instance_id": int(entry["instance_id"]),
            "c2w": np.asarray(entry["c2w"], dtype=np.float32).tolist(),
        })
    return rec


def _flush_trajectory(trajectory: list, path: str, env, cam_order: list) -> None:
    payload = {
        "version": 1,
        "cam_order": cam_order,
        "checkpoint": str(getattr(env.renderer, "ckpt_path", "")),
        "steps": trajectory,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run(cfg, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    panels_dir = os.path.join(output_dir, "panels")
    ad_input_dir = os.path.join(output_dir, "ad_input")
    corpus_dir = os.path.join(output_dir, "distill_corpus")
    os.makedirs(panels_dir, exist_ok=True)
    os.makedirs(ad_input_dir, exist_ok=True)
    os.makedirs(corpus_dir, exist_ok=True)

    print("[mm-cl] building env + renderer ...", flush=True)
    env = AetherDrivingEnv(cfg, output_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[mm-cl] loading LiDAR sidecar: {args.lidar_sidecar}", flush=True)
    cl = ClosedLoopLidar(env.renderer, args.lidar_sidecar, device=device)
    far = float(cl.cfg.far_plane)
    near = float(cl.cfg.near_plane)
    raster_np = cl.raster_pts[0].detach().cpu().numpy()

    print(f"[mm-cl] loading LiDAR refiner bundle: {args.lidar_bundle}", flush=True)
    enhancer = load_unified_bundle(args.lidar_bundle, device=device)

    print("[mm-cl] building LiDAR reference bank (GT sweeps) ...", flush=True)
    ref_bank = build_ref_bank(env.renderer, cl.cfg, device=device)
    print(f"[mm-cl] reference bank size = {len(ref_bank)}", flush=True)

    use_ad = args.ad not in ("none", "", None)
    difix = None
    refine_cuda = args.refine_cuda if args.refine_cuda is not None else args.ad_cuda
    if args.refine_rgb:
        from aether.refine import DifixDaemon
        difix = DifixDaemon(
            model_name=args.refine_model, max_side=args.refine_max_side,
            timestep=args.refine_timestep, cuda_visible=refine_cuda,
            work_root=os.path.join(output_dir, "refine_work"), dtype="bf16",
        )
        difix.start()
        print(f"[mm-cl] Difix RGB enhancement ON ({args.refine_model}, "
              f"max_side={args.refine_max_side}, cuda={refine_cuda})", flush=True)
    elif use_ad:
        raise RuntimeError(
            "AD closed-loop requires Difix-refined RGB. Pass --refine / --refine_rgb "
            "(default ON for --ad uniad) or you used --no_refine by mistake."
        )
    obs_pipe = plan_pipe = proc = None
    if use_ad:
        from aether.agents.launcher import make_pipes, check_alive
        from aether.agents.registry import load_adapter, launch_adapter
        obs_pipe, plan_pipe = make_pipes(output_dir)
        spec = load_adapter(args.ad, legacy_cfg=cfg.base)
        proc = launch_adapter(spec, args.ad_cuda, os.path.abspath(output_dir))
        print(f"[mm-cl] launched AD adapter '{args.ad}'", flush=True)

    front = _front_cam_name(env)
    rigH, rigW = env.cam_rig[front]["H"], env.cam_rig[front]["W"]
    K_full = env.cam_rig[front]["K"]
    l2c = env.cam_rig[front]["l2c"]
    print(f"[mm-cl] camera rig: {rigW}x{rigH} ({args.camera})", flush=True)
    print(f"[mm-cl] refine_rgb={args.refine_rgb}, ad={args.ad}", flush=True)

    obs, info = env.reset()
    frames = []
    trajectory = []
    traj_path = os.path.join(output_dir, "camera_trajectory.json")
    done = False
    cnt = 0
    while not done:
        t0 = time.perf_counter()
        rgb_raw = obs["rgb"][front].copy()

        # Record camera poses BEFORE step (matches obs used this iteration).
        trajectory.append(_record_step_trajectory(env, cnt, info))

        # 1) RGB enhancement — MUST run before IPC to UniAD
        if difix is not None:
            try:
                obs["rgb"] = difix.refine_obs_rgb(obs["rgb"])
            except Exception as e:
                msg = f"[mm-cl] Difix refine failed at step {cnt}: {e}"
                if use_ad:
                    raise RuntimeError(msg) from e
                print(msg, flush=True)
        rgb_enh = obs["rgb"][front].copy()
        if use_ad:
            _save_ad_rgb_strip(obs["rgb"], os.path.join(ad_input_dir, f"step_{cnt:04d}.jpg"))

        # 2) LiDAR sweep (raw 3DGS) at current ego pose
        ego_ds = env._ego_c2w_ds()
        sweep = cl.render_sweep(ego_ds.astype(np.float32), min(env.frame_idx, env._max_frame),
                                apply_overrides=True)
        in_r, in_i, in_k = sweep_arrays(sweep, far)
        pts_raw = spherical_to_points(in_r, in_k, raster_np, near, far)

        # 3) LiDAR enhancement (refiner w/ nearest GT reference)
        ref = ref_bank.nearest(ego_ds[:2, 3])
        ref_r, ref_i, ref_k = None, None, None
        try:
            pr, pi, pk = enhance_sweep(enhancer, in_r, in_i, in_k, ref)
            pts_enh = spherical_to_points(pr, pk, raster_np, near, far)
        except Exception as e:
            print(f"[mm-cl] lidar refine step {cnt} failed: {e}", flush=True)
            pr, pi, pk = in_r, in_i, in_k
            pts_enh = pts_raw

        # Save enhanced observations for trajectory-based distillation.
        _save_distill_corpus_step(
            corpus_dir, cnt, obs["rgb"], pr, pi, pk, in_r, in_i, in_k)

        # 4) Visualization panel (before/after x 3 views)
        proj_raw = project_points_on_image(pts_raw, l2c, K_full, (rigH, rigW), rgb_raw)
        proj_enh = project_points_on_image(pts_enh, l2c, K_full, (rigH, rigW), rgb_enh)
        panel = make_step_panel(
            rgb_raw, rgb_enh, proj_raw, proj_enh, pts_raw, pts_enh,
            cnt, info["timestamp"], os.path.join(panels_dir, f"step_{cnt:04d}.png"),
            bev_range=args.bev_range,
        )
        frames.append(panel)

        # 5) plan
        if use_ad:
            with open(obs_pipe, "wb") as p:
                p.write(pickle.dumps((obs, info), protocol=4))
            with open(plan_pipe, "rb") as p:
                raw = p.read()
            plan_traj = pickle.loads(raw)
            if plan_traj is None:
                print("[mm-cl] AD returned None; ending", flush=True)
                break
            acc, steer_rate = _traj_to_action(plan_traj, info)
        else:
            acc, steer_rate = 1.0, 0.0   # straight, gentle throttle

        obs, _, terminated, truncated, info = env.step({"acc": acc, "steer_rate": steer_rate})
        cnt += 1
        done = terminated or truncated or (args.max_steps > 0 and cnt >= args.max_steps)
        dt = (time.perf_counter() - t0) * 1000
        print(f"[mm-cl step {cnt-1:03d}] t={info['timestamp']:.2f}s "
              f"v={info['ego_velo']:.2f} pts raw={pts_raw.shape[0]} enh={pts_enh.shape[0]} "
              f"rgb_refined={difix is not None} | {dt:.0f}ms", flush=True)

    # teardown + video
    if difix is not None:
        try:
            difix.stop()
        except Exception:
            pass
    if use_ad:
        try:
            with open(obs_pipe, "wb") as p:
                p.write(pickle.dumps("Done", protocol=4))
        except Exception:
            pass

    if frames:
        vid = os.path.join(output_dir, "multimodal_closed_loop.mp4")
        ImageSequenceClip(frames, fps=args.fps).write_videofile(vid, logger=None)
        print(f"[mm-cl] DONE: {cnt} steps -> {vid}", flush=True)
        print(f"[mm-cl] per-step panels -> {panels_dir}", flush=True)
        _flush_trajectory(trajectory, traj_path, env, env._cam_order)
        print(f"[mm-cl] camera trajectory ({len(trajectory)} steps) -> {traj_path}", flush=True)
        if use_ad:
            print(f"[mm-cl] AD input strips (post-Difix 6-cam) -> {ad_input_dir}", flush=True)
        print(f"[mm-cl] distill corpus (enhanced RGB+LiDAR) -> {corpus_dir}", flush=True)
    if use_ad and proc is not None:
        try:
            from aether.agents.launcher import check_alive
            check_alive(proc)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--base", default="configs/base.yaml")
    ap.add_argument("--camera", default="configs/cameras/pandaset_800x450.yaml",
                    help="Must be 800x450 for UniAD (NOT pandaset.yaml 1920x1080).")
    ap.add_argument("--kinematic", default="configs/kinematic.yaml")
    ap.add_argument("--ad", default="none", help="uniad | vad | none (built-in straight)")
    ap.add_argument("--ad_cuda", default="0")
    ap.add_argument("--lidar_sidecar", required=True)
    ap.add_argument("--lidar_bundle", required=True)
    refine_grp = ap.add_mutually_exclusive_group()
    refine_grp.add_argument("--refine", "--refine_rgb", dest="refine_rgb",
                            action="store_true",
                            help="Difix-refine all 6 cams before AD (default ON when --ad set).")
    refine_grp.add_argument("--no_refine", dest="refine_rgb", action="store_false")
    ap.set_defaults(refine_rgb=None)
    ap.add_argument("--refine_model", default="nvidia/difix",
                    help="HF model id: nvidia/difix (single-view, closed-loop) "
                         "or nvidia/difix_ref (needs GT reference views).")
    ap.add_argument("--refine_cuda", default=None,
                    help="GPU for Difix daemon (default: same as --ad_cuda).")
    ap.add_argument("--refine_max_side", type=int, default=1024,
                    help="Difix long-edge resolution (1024 for 800x450).")
    ap.add_argument("--refine_timestep", type=int, default=199)
    ap.add_argument("--max_steps", type=int, default=0, help=">0 caps steps")
    ap.add_argument("--bev_range", type=float, default=50.0)
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()
    if args.refine_rgb is None:
        args.refine_rgb = args.ad not in ("none", "", None)

    cfg = load_cfg(args.scenario, args.base, args.camera, args.kinematic)
    out = _resolve_output_dir(cfg, args.ad)
    print(f"[mm-cl] output -> {out}", flush=True)
    run(cfg, out, args)
