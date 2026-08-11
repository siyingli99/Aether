"""Function: Build synthetic LiDAR training pairs from processed ground-truth sweeps.
Input/Output: Reads processed dataset scenes; writes spherical-grid NPZ files, previews, and pair manifests.
Example:
    python aether/refine/gen_lidar_gt_corpus.py --datasets pandaset \
        --scene_ids pandaset:095 --out outputs/multimodal_refine --stride 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm

from aether.paths import AETHER_ROOT, DRIVESTUDIO_ROOT, NUSCENES_PROCESSED, PANDASET_PROCESSED
from aether.render.lidar_head import LidarHeadConfig
from aether.refine.pair_utils import (
    list_processed_scenes,
    nearest_ref_frame,
    pack_lidar_rgb,
    project_gt_frame,
    save_pair_npz,
    synthetic_degrade_lidar,
)


def _default_roots():
    return {
        "pandaset": PANDASET_PROCESSED,
        "nuscenes": NUSCENES_PROCESSED,
    }


def _make_cfg(dataset: str, data_root: str, scene_idx: int):
    if dataset == "pandaset":
        ds_yaml = DRIVESTUDIO_ROOT / "configs/datasets/pandaset/6cams.yaml"
    else:
        ds_yaml = DRIVESTUDIO_ROOT / "configs/datasets/nuscenes/6cams_scene0383.yaml"
    cfg = OmegaConf.load(ds_yaml)
    cfg.data.data_root = str(data_root)
    cfg.data.scene_idx = int(scene_idx)
    cfg.data.start_timestep = 0
    cfg.data.end_timestep = -1
    return cfg


def process_scene(dataset: str, scene_idx: int, data_root: Path, out_root: Path,
                  stride: int, severities: list[float], seed: int, device: str):
    sid = f"{dataset}_{scene_idx:03d}"
    scene_out = out_root / "gt_corpus" / sid
    npz_dir = scene_out / "npz"
    in_dir = scene_out / "input"
    ref_dir = scene_out / "ref"
    tg_dir = scene_out / "target"
    for d in (npz_dir, in_dir, ref_dir, tg_dir):
        d.mkdir(parents=True, exist_ok=True)

    cfg = _make_cfg(dataset, str(data_root), scene_idx)
    from aether.refine.pair_utils import build_lidar_source
    src, data_path = build_lidar_source(cfg, device)
    lidar_bin = Path(data_path) / "lidar"
    head_cfg = LidarHeadConfig()
    far = float(head_cfg.far_plane)
    n_frames = src.lidar_to_worlds.shape[0]
    frames = list(range(0, n_frames, max(1, stride)))

    cache = []
    for f in frames:
        cache.append(project_gt_frame(src, f, lidar_bin, head_cfg, device))
    poses = [c["pose"] for c in cache]

    rng = np.random.default_rng(seed + scene_idx)
    manifest = {}
    pair_id = 0
    for fi, f in enumerate(tqdm(frames, desc=sid, leave=False)):
        gt = cache[fi]
        ref_i = nearest_ref_frame(poses, fi)
        ref = cache[ref_i]
        ref_f = frames[ref_i]

        ref_keep = np.clip(ref["gt_mask"], 0, 1)
        for sev in severities:
            in_r, in_i, in_k = synthetic_degrade_lidar(
                gt["gt_range"], gt["gt_int"], gt["gt_mask"], far, rng, severity=sev)
            name = f"{f:04d}_s{int(sev*10):02d}"
            npz_path = npz_dir / f"{name}.npz"
            save_pair_npz(
                npz_path,
                in_r, in_i, in_k,
                ref["gt_range"], ref["gt_int"], ref_keep,
                gt["gt_range"], gt["gt_int"], gt["gt_mask"],
                meta={"frame": f, "ref_frame": ref_f, "severity": sev, "source": "synthetic"},
            )
            Image.fromarray(pack_lidar_rgb(in_r, in_i, in_k, far)).save(in_dir / f"{name}.png")
            Image.fromarray(pack_lidar_rgb(ref["gt_range"], ref["gt_int"], ref_keep, far)).save(
                ref_dir / f"{name}.png")
            Image.fromarray(pack_lidar_rgb(gt["gt_range"], gt["gt_int"], gt["gt_mask"], far)).save(
                tg_dir / f"{name}.png")
            manifest[name] = {
                "npz": str(npz_path),
                "image": str(in_dir / f"{name}.png"),
                "ref_image": str(ref_dir / f"{name}.png"),
                "target_image": str(tg_dir / f"{name}.png"),
                "scene": sid,
                "prompt": "remove degradation",
            }
            pair_id += 1

    with open(scene_out / "pairs.json", "w") as fp:
        json.dump(manifest, fp, indent=2)
    return sid, pair_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="pandaset,nuscenes",
                    help="comma-separated: pandaset,nuscenes")
    ap.add_argument("--out", default=str(AETHER_ROOT / "outputs/multimodal_refine"))
    ap.add_argument("--stride", type=int, default=2, help="frame subsample stride")
    ap.add_argument("--severities", default="0.8,1.0,1.2",
                    help="synthetic degrade severity levels")
    ap.add_argument("--scene_limit", type=int, default=0, help=">0: cap scenes per dataset")
    ap.add_argument("--scene_ids", default="",
                    help="optional comma list e.g. pandaset:001,nuscenes:088")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(args.out)
    severities = [float(x) for x in args.severities.split(",") if x.strip()]
    roots = _default_roots()

    todo: list[tuple[str, int, Path]] = []
    if args.scene_ids.strip():
        for tok in args.scene_ids.split(","):
            ds, idx = tok.strip().split(":")
            todo.append((ds, int(idx), roots[ds]))
    else:
        for ds in args.datasets.split(","):
            ds = ds.strip()
            ids = list_processed_scenes(roots[ds])
            if args.scene_limit > 0:
                ids = ids[: args.scene_limit]
            for idx in ids:
                todo.append((ds, idx, roots[ds]))

    print(f"[gt_corpus] scenes={len(todo)} stride={args.stride} severities={severities}", flush=True)
    summary = []
    total = 0
    for ds, idx, root in todo:
        try:
            sid, n = process_scene(ds, idx, root, out_root, args.stride, severities,
                                   args.seed, device)
            summary.append({"scene": sid, "pairs": n})
            total += n
            print(f"  ✓ {sid}: {n} pairs", flush=True)
        except Exception as e:
            print(f"  ✗ {ds}_{idx:03d}: {e}", flush=True)

    with open(out_root / "gt_corpus_summary.json", "w") as fp:
        json.dump({"total_pairs": total, "scenes": summary}, fp, indent=2)
    print(f"[gt_corpus] DONE total_pairs={total} -> {out_root/'gt_corpus_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
