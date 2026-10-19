"""Function: Pair joint 3DGS LiDAR renders with ground-truth spherical projections.
Input/Output: Reads registered joint checkpoints, sidecars, and processed scenes; writes NPZ pairs, previews, and manifests.
Example:
    # Prerequisite: register generated joint checkpoints and sidecars in the registry.
    python aether/refine/gen_lidar_pairs.py \
        --registry configs/refine/scene_registry.yaml --scene_ids pandaset095_joint \
        --out outputs/multimodal_refine_v2/render_corpus --layout render_corpus
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from aether.paths import DRIVESTUDIO_ROOT as _DS_ROOT  # noqa: E402
sys.path.insert(0, str(_DS_ROOT))
from aether.paths import use_vendored_backends
use_vendored_backends()  # DriveStudio / SplatAD on sys.path
from aether.paths import AETHER_ROOT
sys.path.insert(0, str(AETHER_ROOT))

from omegaconf import OmegaConf
from PIL import Image

from aether.paths import resolve_data_root
from aether.render.ds_backend import DriveStudioRenderer
from aether.render.lidar_closed_loop import ClosedLoopLidar
from aether.train.joint_phase import gt_to_spherical_grid
from aether.refine.pair_utils import (
    build_lidar_source,
    latest_lidar_sidecar,
    nearest_ref_frame,
    pack_lidar_rgb,
    save_pair_npz,
)


_SCENES = {
    "nuscenes088": "outputs/train/nuscenes-0106/joint",
    "nuscenes0383": "outputs/train/nuscenes-0383/joint",
    "pandaset017": "outputs/train/pandaset-017/joint",
}
_AETHER = AETHER_ROOT


def _pack_rgb(range_m: np.ndarray, inten: np.ndarray, third: np.ndarray,
              far: float) -> np.ndarray:
    r = np.clip(range_m / far, 0.0, 1.0)
    g = np.clip(inten, 0.0, 1.0)
    b = np.clip(third, 0.0, 1.0)
    return (np.stack([r, g, b], -1) * 255.0).astype(np.uint8)


def _read_bin_intensity(bin_path: Path, n_rays: int, device) -> torch.Tensor:
    from aether.refine.pair_utils import read_bin_intensity
    return read_bin_intensity(bin_path, n_rays, device)


def _load_registry(path: Path) -> list[dict]:
    with open(path) as fp:
        return yaml.safe_load(fp)["scenes"]


def _scene_entries(args) -> list[tuple[str, str]]:
    """Return (scene_key, joint_dir_relative) pairs to process."""
    if args.registry:
        entries = []
        want = {s.strip() for s in args.scene_ids.split(",") if s.strip()}
        for entry in _load_registry(Path(args.registry)):
            sid = entry["id"]
            if want and sid not in want:
                continue
            key = sid.replace("_joint", "")
            entries.append((key, entry["joint_dir"]))
        if not entries:
            raise SystemExit(f"no scenes matched registry={args.registry}")
        return entries
    if args.all_scenes:
        return list(_SCENES.items())
    return [(args.scene, _SCENES[args.scene])]


def process_scene(scene_key: str, joint_dir_rel: str, out_base: Path,
                  stride: int, limit: int, device: str = "cuda") -> int:
    jdir = Path(_AETHER) / joint_dir_rel
    ckpt = jdir / "checkpoint_final.pth"
    sidecar = latest_lidar_sidecar(jdir)
    config = jdir / "config.yaml"
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)

    out_root = out_base / f"{scene_key}_nominal"
    npz_dir = out_root / "npz"
    in_dir = out_root / "lidar_in"
    ref_dir = out_root / "lidar_ref"
    tg_dir = out_root / "lidar_gt"
    for d in (npz_dir, in_dir, ref_dir, tg_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[gen] scene={scene_key}\n      ckpt={ckpt}\n      sidecar={sidecar}", flush=True)
    renderer = DriveStudioRenderer(str(ckpt), device=device)
    cl = ClosedLoopLidar(renderer, str(sidecar), device=device)
    far = float(cl.cfg.far_plane)

    cfg = OmegaConf.load(config)
    cfg.data.data_root = str(resolve_data_root(cfg.data.data_root))
    _prev = os.getcwd(); os.chdir(str(_DS_ROOT))
    try:
        src, data_path = build_lidar_source(cfg, device)
    finally:
        os.chdir(_prev)
    lidar_bin_dir = Path(data_path) / "lidar"
    n_frames = src.lidar_to_worlds.shape[0]
    frames = list(range(0, n_frames, stride))
    if limit > 0:
        frames = frames[:limit]
    print(f"[gen] {len(frames)} frames (of {n_frames}), gt lidar bins: {lidar_bin_dir}",
          flush=True)

    gt_cache = []
    for f in frames:
        lidar2world = src.lidar_to_worlds[f].to(device)
        rays = src.get_lidar_rays(f)
        ranges = rays["lidar_ranges"].squeeze(-1).to(device)
        inten = _read_bin_intensity(lidar_bin_dir / f"{f:03d}.bin", ranges.numel(), device)
        gt_range, gt_int, hit = gt_to_spherical_grid(
            rays["lidar_origins"][0].to(device),
            rays["lidar_viewdirs"].to(device),
            ranges, inten, lidar2world, cl.cfg,
        )
        gt_cache.append({
            "gt_range": gt_range.detach().cpu().numpy().astype(np.float32),
            "gt_int": gt_int.detach().cpu().numpy().astype(np.float32),
            "gt_mask": hit.detach().cpu().numpy().astype(np.float32),
            "pose": lidar2world.detach().cpu().numpy().astype(np.float32),
        })
    poses = [c["pose"] for c in gt_cache]

    manifest = {}
    for k, f in enumerate(frames):
        gt = gt_cache[k]
        ref_i = nearest_ref_frame(poses, k)
        ref = gt_cache[ref_i]
        ref_keep = np.clip(ref["gt_mask"], 0, 1)

        lp = poses[k]
        out = cl.render_sweep(lp, f, apply_overrides=False)
        in_range = out["range"].clamp(0, far)
        in_int = out["intensity"].clamp(0, 1)
        in_keep = (1.0 - torch.sigmoid(out["raydrop_logit"])).clamp(0, 1)

        ir, ii, ik = (t.detach().cpu().numpy().astype(np.float32)
                      for t in (in_range, in_int, in_keep))
        gr, gi, gm = gt["gt_range"], gt["gt_int"], gt["gt_mask"]

        name = f"{f:04d}_nominal"
        npz_path = npz_dir / f"{name}.npz"
        save_pair_npz(
            npz_path, ir, ii, ik,
            ref["gt_range"], ref["gt_int"], ref_keep,
            gr, gi, gm,
            meta={"frame": f, "ref_frame": frames[ref_i], "source": "real_render_nominal"},
        )
        Image.fromarray(pack_lidar_rgb(ir, ii, ik, far)).save(in_dir / f"{name}.png")
        Image.fromarray(pack_lidar_rgb(ref["gt_range"], ref["gt_int"], ref_keep, far)).save(
            ref_dir / f"{name}.png")
        Image.fromarray(pack_lidar_rgb(gr, gi, gm, far)).save(tg_dir / f"{name}.png")

        manifest[name] = {
            "npz": str(npz_path),
            "scene": scene_key,
            "source": "real_render_nominal",
            "prompt": "remove degradation",
        }
        if k % 20 == 0 or k == len(frames) - 1:
            occ_in = float((ik > 0.3).mean()); occ_gt = float((gm > 0.5).mean())
            print(f"  [{k+1}/{len(frames)}] frame {name}  "
                  f"in_keep%={occ_in:.2f} gt_hit%={occ_gt:.2f}", flush=True)

    with open(out_root / "pairs.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    print(f"[gen] wrote {len(manifest)} nominal real-render pairs -> {out_root}", flush=True)
    return len(manifest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="nuscenes088", choices=list(_SCENES))
    ap.add_argument("--out", default=f"{_AETHER}/outputs/multimodal_refine_v2/render_corpus")
    ap.add_argument("--layout", default="render_corpus",
                    choices=["legacy", "render_corpus"],
                    help="render_corpus: flat pairs.json for merge_pair_manifests")
    ap.add_argument("--registry", default="",
                    help="YAML registry (configs/refine/scene_registry.yaml); overrides --scene")
    ap.add_argument("--scene_ids", default="",
                    help="comma subset of registry ids (e.g. pandaset015_joint)")
    ap.add_argument("--all_scenes", action="store_true",
                    help="generate for every joint scene in _SCENES")
    ap.add_argument("--limit", type=int, default=0, help=">0: only first N frames")
    ap.add_argument("--stride", type=int, default=1)
    args = ap.parse_args()

    device = "cuda"
    out_base = Path(args.out)
    total = 0
    for scene_key, joint_rel in _scene_entries(args):
        total += process_scene(scene_key, joint_rel, out_base,
                               args.stride, args.limit, device=device)
    print(f"[gen] total nominal pairs: {total}", flush=True)


if __name__ == "__main__":
    main()
