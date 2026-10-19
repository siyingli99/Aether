"""Function: Re-render a recorded camera trajectory from a distilled 3DGS scene.
Input/Output: Reads a trajectory, joint directory, and distilled artifacts; writes six-camera frames and videos.
Example:
    # Prerequisite: a run with output.io_trace=true, background_distilled.pth and sidecar_distilled.pt.
    python scripts/render_distilled_trajectory.py \
        --trajectory outputs/runs/<run>/logs/render_poses.json \
        --joint_dir outputs/train/pandaset-095/joint \
        --distill_dir outputs/distilled/pandaset095 --out outputs/distilled/pandaset095/replay
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from moviepy import ImageSequenceClip

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from aether.render.ds_backend import DriveStudioRenderer
from aether.render.lidar_closed_loop import ClosedLoopLidar

_CAM_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]
_PANDA_CAM_ID = {
    "CAM_FRONT": 0,
    "CAM_FRONT_LEFT": 1,
    "CAM_FRONT_RIGHT": 2,
    "CAM_BACK_LEFT": 3,
    "CAM_BACK_RIGHT": 4,
    "CAM_BACK": 5,
}


def _load_distilled(renderer, cl, distill_dir: Path, device: str) -> None:
    bg_path = distill_dir / "background_distilled.pth"
    side_path = distill_dir / "sidecar_distilled.pt"
    if not bg_path.exists():
        raise FileNotFoundError(f"Missing distilled background: {bg_path}")
    if not side_path.exists():
        raise FileNotFoundError(f"Missing distilled sidecar: {side_path}")

    bg_state = torch.load(bg_path, map_location=device, weights_only=False)
    renderer.trainer.models["Background"].load_state_dict(bg_state)

    side = torch.load(side_path, map_location=device, weights_only=False)
    feats = side.get("feats", {}).get("Background", {})
    if "_intensity" in feats:
        cl.feat_intensity["Background"] = feats["_intensity"].to(device)
    if "_raydrop_logit" in feats:
        cl.feat_raydrop["Background"] = feats["_raydrop_logit"].to(device)
    print(f"[rerender] loaded distilled BG + sidecar from {distill_dir}", flush=True)


def _inject_actors(renderer, frame_idx: int, actors: list) -> None:
    renderer.clear_actor_overrides()
    for entry in actors:
        renderer.set_actor_pose(
            entry["class"], int(entry["instance_id"]), int(frame_idx),
            np.asarray(entry["c2w"], dtype=np.float32),
        )


def _render_step_strip(renderer, step_rec: dict, cam_order: list) -> np.ndarray:
    frame_idx = int(step_rec["frame_idx"])
    _inject_actors(renderer, frame_idx, step_rec.get("actors", []))
    tiles = []
    for name in cam_order:
        cam = step_rec["cameras"][name]
        c2w = np.asarray(cam["c2w"], dtype=np.float32)
        K = np.asarray(cam["K"], dtype=np.float32)
        H, W = int(cam["H"]), int(cam["W"])
        out = renderer.render(
            c2w_4x4=c2w, K_3x3=K, H=H, W=W,
            frame_idx=frame_idx, cam_idx=int(cam.get("index", _PANDA_CAM_ID.get(name, 0))),
        )
        rgb = (out["rgb"].clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
        tiles.append(rgb)
    return np.concatenate(tiles, axis=1)


def _load_enhanced_strip(ad_input_dir: Path, step: int) -> np.ndarray | None:
    p = ad_input_dir / f"step_{step:04d}.jpg"
    if not p.exists():
        return None
    return imageio.imread(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectory", required=True,
                    help="logs/render_poses.json of a run made with output.io_trace=true")
    ap.add_argument("--joint_dir", required=True,
                    help="Joint scene dir (checkpoint_final.pth + sidecar)")
    ap.add_argument("--distill_dir", required=True,
                    help="distill output with background_distilled.pth")
    ap.add_argument("--closed_loop_dir", default="",
                    help="Optional closed-loop output dir (for ad_input compare).")
    ap.add_argument("--out", default="",
                    help="Output directory (default: closed_loop_dir or trajectory parent)")
    ap.add_argument("--fps", type=int, default=4)
    args = ap.parse_args()

    traj_path = Path(args.trajectory).expanduser()
    if not traj_path.is_absolute():
        traj_path = (_REPO / traj_path).resolve()
    joint_dir = Path(args.joint_dir).expanduser()
    if not joint_dir.is_absolute():
        joint_dir = (_REPO / joint_dir).resolve()
    distill_dir = Path(args.distill_dir).expanduser()
    if not distill_dir.is_absolute():
        distill_dir = (_REPO / distill_dir).resolve()
    cl_dir = Path(args.closed_loop_dir) if args.closed_loop_dir else (
        traj_path.parent.parent if traj_path.parent.name == "logs" else traj_path.parent)
    out_dir = Path(args.out) if args.out else cl_dir / "distilled_rerender"
    out_dir.mkdir(parents=True, exist_ok=True)
    strip_dir = out_dir / "distilled_direct_strip"
    strip_dir.mkdir(parents=True, exist_ok=True)

    with open(traj_path, encoding="utf-8") as f:
        traj = json.load(f)
    cam_order = traj.get("cam_order") or _CAM_ORDER
    steps = traj["steps"]
    print(f"[rerender] {len(steps)} steps from {traj_path}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = joint_dir / "checkpoint_final.pth"
    sidecar = sorted(joint_dir.glob("lidar_sidecar_step*.pt"))[-1]
    renderer = DriveStudioRenderer(str(ckpt), device=device)
    cl = ClosedLoopLidar(renderer, str(sidecar), device=device)
    _load_distilled(renderer, cl, distill_dir, device)

    direct_frames = []
    compare_frames = []
    ad_input_dir = cl_dir / "logs" / "io"
    if not ad_input_dir.is_dir():
        ad_input_dir = cl_dir / "ad_input"

    for step_rec in steps:
        step = int(step_rec["step"])
        strip = _render_step_strip(renderer, step_rec, cam_order)
        imageio.imwrite(strip_dir / f"step_{step:04d}.jpg", strip, quality=90)
        direct_frames.append(strip)

        enh = _load_enhanced_strip(ad_input_dir, step)
        if enh is not None:
            h = max(strip.shape[0], enh.shape[0])
            def _pad(img):
                if img.shape[0] == h:
                    return img
                pad = np.zeros((h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                return np.concatenate([img, pad], axis=0)
            compare_frames.append(np.concatenate([_pad(enh), _pad(strip)], axis=0))

    vid_direct = out_dir / "distilled_direct_6cam.mp4"
    ImageSequenceClip(direct_frames, fps=args.fps).write_videofile(
        str(vid_direct), logger=None)
    print(f"[rerender] direct 6-cam video -> {vid_direct}", flush=True)

    if compare_frames:
        vid_cmp = out_dir / "compare_enhanced_vs_distilled.mp4"
        ImageSequenceClip(compare_frames, fps=args.fps).write_videofile(
            str(vid_cmp), logger=None)
        print(f"[rerender] compare video (left=Difix AD input, right=distilled direct) "
              f"-> {vid_cmp}", flush=True)

    meta = {
        "trajectory": str(traj_path),
        "joint_dir": str(joint_dir),
        "distill_dir": str(distill_dir),
        "num_steps": len(steps),
        "videos": {
            "distilled_direct": str(vid_direct),
            "compare": str(out_dir / "compare_enhanced_vs_distilled.mp4")
            if compare_frames else None,
        },
    }
    (out_dir / "rerender_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[rerender] DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
