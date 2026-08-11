"""Function: Reverse-distill enhanced closed-loop observations at recorded poses.
Input/Output: Reads a multimodal rollout, joint run, sidecar, and refiner bundle; writes updated background and sidecar artifacts.
Example:
    # Prerequisite: generate camera_trajectory.json and distill_corpus in the rollout.
    python aether/refine/distill_from_closed_loop.py \
        --closed_loop_dir aether_results/pandaset/pandaset_095_multimodal/none/easy \
        --joint_dir outputs/dualmodal/aether_joint/pandaset/scene095_joint \
        --lidar_sidecar outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt \
        --refiner_bundle outputs/multimodal_refine_v2/unified/unified_multimodal.bundle.pth \
        --out outputs/distilled/pandaset095
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_AETHER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_AETHER))

import aether  # noqa: F401  — injects vendored drivestudio + splatad_gsplat
from aether.paths import AETHER_ROOT, DRIVESTUDIO_ROOT

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from aether.render.ds_backend import DriveStudioRenderer
from aether.render.lidar_closed_loop import ClosedLoopLidar

_STRIP_ORDER = [
    "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT",
]
_PANDA_CAM_ID = {
    "CAM_FRONT": 0, "CAM_FRONT_LEFT": 1, "CAM_FRONT_RIGHT": 2,
    "CAM_BACK_LEFT": 3, "CAM_BACK_RIGHT": 4, "CAM_BACK": 5,
}


def _abs_aether(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = AETHER_ROOT / p
    return p.resolve()


def _training_pose_xy(joint_dir: Path, device: str) -> np.ndarray:
    from omegaconf import OmegaConf
    from aether.refine.pair_utils import build_lidar_source

    cfg = OmegaConf.load(joint_dir / "config.yaml")
    _prev = os.getcwd()
    os.chdir(str(DRIVESTUDIO_ROOT))
    try:
        src, _ = build_lidar_source(cfg, device)
    finally:
        os.chdir(_prev)
    poses = src.lidar_to_worlds.detach().cpu().numpy()
    return poses[:, :2, 3].astype(np.float64)


def _novelty_m(ego_xy: np.ndarray, train_xy: np.ndarray) -> float:
    d = train_xy - ego_xy[None, :]
    return float(np.sqrt((d * d).sum(axis=1)).min())


def _inject_actors(renderer, step_rec: dict) -> None:
    renderer.clear_actor_overrides()
    f = int(step_rec["frame_idx"])
    for entry in step_rec.get("actors", []):
        renderer.set_actor_pose(
            entry["class"], int(entry["instance_id"]), f,
            np.asarray(entry["c2w"], dtype=np.float32),
        )


def _save_corpus_step(corpus_dir: Path, step: int, obs_rgb: dict,
                      pr, pi, pk, in_r, in_i, in_k) -> None:
    rgb_dir = corpus_dir / "rgb" / f"{step:04d}"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    for cam, img in obs_rgb.items():
        Image.fromarray(img).save(rgb_dir / f"{cam}.png")
    lidar_dir = corpus_dir / "lidar"
    lidar_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        lidar_dir / f"{step:04d}.npz",
        enh_range=pr.astype(np.float32),
        enh_int=pi.astype(np.float32),
        enh_keep=pk.astype(np.float32),
        raw_range=in_r.astype(np.float32),
        raw_int=in_i.astype(np.float32),
        raw_keep=in_k.astype(np.float32),
    )


def backfill_corpus(cl_dir: Path, trajectory: dict, joint_dir: Path,
                    sidecar: str, bundle: str, device: str) -> Path:
    """Build distill_corpus from trajectory + ad_input strips + LiDAR re-enhance."""
    from aether.refine.unified_multimodal_enhancer import load_unified_bundle
    from aether.tools.multimodal_cl_viz import (
        build_ref_bank, enhance_sweep, sweep_arrays,
    )

    corpus_dir = cl_dir / "distill_corpus"
    if (corpus_dir / "manifest.json").exists():
        return corpus_dir

    print("[cl-distill] backfilling distill_corpus (RGB from ad_input strips) ...", flush=True)
    ckpt = _abs_aether(joint_dir) / "checkpoint_final.pth"
    renderer = DriveStudioRenderer(str(ckpt), device=device)
    cl = ClosedLoopLidar(renderer, sidecar, device=device)
    far = float(cl.cfg.far_plane)
    enhancer = load_unified_bundle(bundle, device=device)
    ref_bank = build_ref_bank(renderer, cl.cfg, device=device)

    steps = trajectory["steps"]
    manifest = []
    for rec in steps:
        step = int(rec["step"])
        strip_path = cl_dir / "ad_input" / f"step_{step:04d}.jpg"
        if not strip_path.exists():
            continue
        strip = np.asarray(Image.open(strip_path).convert("RGB"))
        cams = rec["cameras"]
        order = [c for c in _STRIP_ORDER if c in cams] or list(cams.keys())
        w = int(cams[order[0]]["W"])
        obs_rgb = {}
        for i, cam in enumerate(order):
            obs_rgb[cam] = strip[:, i * w:(i + 1) * w]

        ego = np.asarray(rec["ego_c2w_ds"], dtype=np.float32)
        f = int(rec["frame_idx"])
        _inject_actors(renderer, rec)
        sweep = cl.render_sweep(ego, f, apply_overrides=True)
        in_r, in_i, in_k = sweep_arrays(sweep, far)
        ref = ref_bank.nearest(ego[:2, 3])
        pr, pi, pk = enhance_sweep(enhancer, in_r, in_i, in_k, ref)
        _save_corpus_step(corpus_dir, step, obs_rgb, pr, pi, pk, in_r, in_i, in_k)
        manifest.append({"step": step, "frame_idx": f})

    (corpus_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[cl-distill] backfilled {len(manifest)} steps -> {corpus_dir}", flush=True)
    del renderer, cl, enhancer
    torch.cuda.empty_cache()
    return corpus_dir


def _load_rgb_pseudo(corpus_dir: Path, step: int, cam: str) -> np.ndarray:
    p = corpus_dir / "rgb" / f"{step:04d}" / f"{cam}.png"
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float32) / 255.0


def _load_lidar_pseudo(corpus_dir: Path, step: int, device: str):
    z = np.load(corpus_dir / "lidar" / f"{step:04d}.npz")
    def _t(k):
        return torch.from_numpy(z[k]).float().to(device)
    return _t("enh_range"), _t("enh_int"), _t("enh_keep"), _t("raw_range")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed_loop_dir", required=True)
    ap.add_argument("--joint_dir", required=True)
    ap.add_argument("--lidar_sidecar", default="")
    ap.add_argument("--refiner_bundle", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--min_novelty_m", type=float, default=0.3,
                    help="Min ego XY distance [m] from nearest training pose.")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=5,
                    help="Passes over the filtered novel steps.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--w_rgb", type=float, default=1.0)
    ap.add_argument("--w_lidar", type=float, default=1.0)
    ap.add_argument("--no_rgb", action="store_true")
    ap.add_argument("--no_lidar", action="store_true")
    args = ap.parse_args()

    device = "cuda"
    cl_dir = _abs_aether(args.closed_loop_dir)
    joint_dir = _abs_aether(args.joint_dir)
    sidecar = args.lidar_sidecar or str(sorted(joint_dir.glob("lidar_sidecar_step*.pt"))[-1])
    bundle = str(_abs_aether(args.refiner_bundle)) if args.refiner_bundle else ""
    out_dir = _abs_aether(args.out) if args.out else joint_dir / "distill_from_cl"
    out_dir.mkdir(parents=True, exist_ok=True)

    traj_path = cl_dir / "camera_trajectory.json"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing {traj_path}")
    trajectory = json.loads(traj_path.read_text())
    corpus_dir = cl_dir / "distill_corpus"
    if not (corpus_dir / "rgb").is_dir():
        if not bundle:
            raise FileNotFoundError(
                "distill_corpus/ missing and --refiner_bundle required for backfill")
        corpus_dir = backfill_corpus(cl_dir, trajectory, joint_dir, sidecar, bundle, device)

    train_xy = _training_pose_xy(joint_dir, device)
    novel_steps = []
    for rec in trajectory["steps"]:
        if int(rec["step"]) % args.stride != 0:
            continue
        ego_xy = np.asarray(rec["ego_c2w_ds"], dtype=np.float64)[:2, 3]
        nov = _novelty_m(ego_xy, train_xy)
        if nov >= args.min_novelty_m:
            lidar_npz = corpus_dir / "lidar" / f"{int(rec['step']):04d}.npz"
            if lidar_npz.exists():
                rec = dict(rec)
                rec["novelty_m"] = nov
                novel_steps.append(rec)
    print(f"[cl-distill] {len(novel_steps)} novel steps "
          f"(min_novelty={args.min_novelty_m}m, stride={args.stride})", flush=True)
    if not novel_steps:
        raise RuntimeError("No novel steps after filtering; lower --min_novelty_m")

    ckpt = joint_dir / "checkpoint_final.pth"
    renderer = DriveStudioRenderer(str(ckpt), device=device)
    cl = ClosedLoopLidar(renderer, sidecar, device=device)
    far = float(cl.cfg.far_plane)

    bg = renderer.trainer.models["Background"]
    sidecar_int = torch.nn.Parameter(cl.feat_intensity["Background"].clone())
    sidecar_rd = torch.nn.Parameter(cl.feat_raydrop["Background"].clone())
    opt_params = [sidecar_int, sidecar_rd]
    for attr in ["_opacities", "_scales", "_means", "_quats",
                 "_features_dc", "_features_rest"]:
        if hasattr(bg, attr):
            t = getattr(bg, attr)
            if isinstance(t, torch.Tensor):
                t.requires_grad_(True)
                opt_params.append(t)
    opt = torch.optim.Adam(opt_params, lr=args.lr)

    use_rgb = not args.no_rgb
    use_lidar = not args.no_lidar
    gstep = 0
    for ep in range(args.epochs):
        for rec in novel_steps:
            step = int(rec["step"])
            f = int(rec["frame_idx"])
            opt.zero_grad()
            loss = torch.tensor(0.0, device=device)
            logs: Dict[str, float] = {}

            _inject_actors(renderer, rec)
            renderer._set_cur_frame(f)
            # Apply actor overrides once per step; restore only after backward.
            # Per-camera render_grad with auto-restore breaks autograd when the
            # RGB loss sums over multiple cameras (instances_quats copy_).
            snap = renderer._apply_overrides(f)
            try:
                if use_lidar:
                    pr, pi, pk, raw_r = _load_lidar_pseudo(corpus_dir, step, device)
                    ego = np.asarray(rec["ego_c2w_ds"], dtype=np.float32)
                    means = bg._means
                    out = cl.head.render(
                        means_w=means, quats=bg.get_quats, scales=bg.get_scaling,
                        opacities=bg.get_opacity.reshape(-1),
                        intensity=torch.sigmoid(sidecar_int),
                        raydrop_logit=sidecar_rd,
                        lidar_pose_w=torch.as_tensor(ego, dtype=torch.float32, device=device),
                        raster_pts=cl.raster_pts,
                        tile_elevation_boundaries=cl.teb,
                    )
                    mask = (pk > 0.3).float()
                    conf = torch.exp(-(out["range"].detach().clamp(0, far) - pr).abs() / (0.1 * far))
                    w = mask * conf
                    denom = w.sum().clamp(min=1.0)
                    l_rng = (F.l1_loss(out["range"].clamp(0, far), pr, reduction="none") * w).sum() / denom / far
                    l_int = (F.l1_loss(out["intensity"], pi, reduction="none") * w).sum() / denom
                    keep_pred = torch.sigmoid(out["raydrop_logit"])
                    l_drop = F.binary_cross_entropy(1 - keep_pred, 1 - (pk > 0.3).float())
                    l_lidar = 5 * l_rng + l_int + l_drop
                    loss = loss + args.w_lidar * l_lidar
                    logs.update(l_rng=float(l_rng.item()), l_int=float(l_int.item()),
                                l_drop=float(l_drop.item()))

                if use_rgb:
                    l_rgb = torch.tensor(0.0, device=device)
                    n_cam = 0
                    for cam, cam_info in rec["cameras"].items():
                        rgb_path = corpus_dir / "rgb" / f"{step:04d}" / f"{cam}.png"
                        if not rgb_path.exists():
                            continue
                        pseudo = torch.from_numpy(
                            _load_rgb_pseudo(corpus_dir, step, cam)).to(device)
                        c2w = np.asarray(cam_info["c2w"], dtype=np.float32)
                        K = np.asarray(cam_info["K"], dtype=np.float32)
                        H, W = int(cam_info["H"]), int(cam_info["W"])
                        cid = _PANDA_CAM_ID.get(cam, 0)
                        rgb = renderer.render_grad(
                            c2w, K, H, W, f, cam_idx=cid,
                            manage_overrides=False,
                        )["rgb"]
                        conf = torch.exp(-(rgb.detach() - pseudo).abs().mean(-1) / 0.1)
                        l_rgb = l_rgb + (F.l1_loss(rgb, pseudo, reduction="none").mean(-1) * conf).mean()
                        n_cam += 1
                    if n_cam:
                        l_rgb = l_rgb / n_cam
                        loss = loss + args.w_rgb * l_rgb
                        logs["l_rgb"] = float(l_rgb.item())

                if loss.requires_grad:
                    loss.backward()
                    opt.step()
            finally:
                if snap:
                    renderer._restore_overrides(f, snap)
                renderer.clear_actor_overrides()
            if gstep % 10 == 0:
                msg = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                print(f"  ep{ep} step{step:03d} nov={rec.get('novelty_m', 0):.2f}m "
                      f"{msg}", flush=True)
            gstep += 1

    cl.feat_intensity["Background"] = sidecar_int.detach()
    cl.feat_raydrop["Background"] = sidecar_rd.detach()
    torch.save({"head_state": cl.head.state_dict(),
                "cfg": cl.cfg.__dict__,
                "feats": {"Background": {
                    "_intensity": sidecar_int.detach().cpu(),
                    "_raydrop_logit": sidecar_rd.detach().cpu(),
                }}},
               out_dir / "sidecar_distilled.pt")
    torch.save(bg.state_dict(), out_dir / "background_distilled.pth")

    meta = {
        "source": str(cl_dir),
        "n_novel_steps": len(novel_steps),
        "min_novelty_m": args.min_novelty_m,
        "epochs": args.epochs,
        "steps": [int(r["step"]) for r in novel_steps],
    }
    (out_dir / "distill_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[cl-distill] DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
