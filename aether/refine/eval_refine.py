"""Function: Compare raw and refined LiDAR predictions against ground truth.
Input/Output: Reads generated scene test pairs and a trained refiner; writes metrics JSON and a qualitative figure.
Example:
    # Prerequisite: generate test pairs and train refiner_best.pth first.
    python aether/refine/eval_refine.py --scene pandaset095 \
        --refiner outputs/lidar_refine/pandaset095/refiner/refiner_best.pth --far 100
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from aether.paths import AETHER_ROOT
sys.path.insert(0, str(AETHER_ROOT))
from aether.refine.train_lidar_refiner import LidarRefiner

_AETHER = AETHER_ROOT


def _binmask_metrics(pred_keep, gt_mask, thr=0.3):
    p = (pred_keep > thr)
    g = (gt_mask > 0.5)
    inter = (p & g).sum()
    union = (p | g).sum()
    tp = inter.astype(np.float64)
    iou = float(tp / max(union, 1))
    prec = float(tp / max(p.sum(), 1))
    rec = float(tp / max(g.sum(), 1))
    f1 = float(2 * prec * rec / max(prec + rec, 1e-9))
    acc = float((p == g).mean())
    return {"iou": iou, "f1": f1, "precision": prec, "recall": rec, "acc": acc}


def _range_metrics(pred_r, gt_r, gt_mask, far):
    m = gt_mask > 0.5
    if m.sum() == 0:
        return {"mae": float("nan"), "rmse": float("nan")}
    e = np.abs(pred_r[m] - gt_r[m])
    return {"mae": float(e.mean()), "rmse": float(np.sqrt((e ** 2).mean()))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="nuscenes088")
    ap.add_argument("--refiner", default="")
    ap.add_argument("--far", type=float, default=100.0)
    args = ap.parse_args()

    device = "cuda"
    scene_dir = _AETHER / "outputs/lidar_refine" / args.scene
    pairs = json.load(open(scene_dir / "pairs.json"))["test"]
    refiner_ckpt = (Path(args.refiner) if args.refiner
                    else scene_dir / "refiner" / "refiner_best.pth")

    rc = torch.load(refiner_ckpt, map_location=device, weights_only=False)
    model = LidarRefiner().to(device); model.load_state_dict(rc["model"]); model.eval()
    hw = tuple(rc.get("hw", (64, 1024)))
    far = float(rc.get("far", args.far))

    agg = {"raw": [], "refined": []}
    samples = []
    for name, item in pairs.items():
        d = np.load(item["npz"])
        gt_r, gt_i, gt_m = d["gt_range"], d["gt_int"], d["gt_mask"]
        in_r, in_i, in_k = d["in_range"], d["in_int"], d["in_keep"]
        H, W = gt_r.shape

        def _ten(a):
            return torch.from_numpy(a)[None, None].float()
        x = torch.cat([_ten(in_r) / far, _ten(in_i), _ten(in_k)], 1).to(device)
        with torch.no_grad():
            p = model(x)
        ref_r = (p[:, 0:1] * far)[0, 0].cpu().numpy()
        ref_i = p[:, 1:2][0, 0].cpu().numpy()
        ref_k = torch.sigmoid(p[:, 2:3])[0, 0].cpu().numpy()

        # raw render metrics
        m_raw = {**_range_metrics(in_r, gt_r, gt_m, far),
                 "int_mae": float(np.abs((in_i - gt_i))[gt_m > 0.5].mean()
                                  if (gt_m > 0.5).any() else np.nan),
                 **{f"drop_{k}": v for k, v in
                    _binmask_metrics(in_k, gt_m).items()}}
        m_ref = {**_range_metrics(ref_r, gt_r, gt_m, far),
                 "int_mae": float(np.abs((ref_i - gt_i))[gt_m > 0.5].mean()
                                  if (gt_m > 0.5).any() else np.nan),
                 **{f"drop_{k}": v for k, v in
                    _binmask_metrics(ref_k, gt_m).items()}}
        agg["raw"].append(m_raw); agg["refined"].append(m_ref)
        if len(samples) < 3:
            samples.append((name, in_r, ref_r, gt_r, far))

    def _mean(rows):
        keys = rows[0].keys()
        return {k: float(np.nanmean([r[k] for r in rows])) for k in keys}

    summary = {"scene": args.scene, "n_test": len(pairs),
               "raw": _mean(agg["raw"]), "refined": _mean(agg["refined"])}
    out_json = scene_dir / "refiner" / "eval_metrics.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out_json, "w"), indent=2)

    print(f"\n=== LiDAR enhancement eval — {args.scene} (n={len(pairs)}) ===")
    print(f"{'metric':<16}{'raw':>12}{'refined':>12}{'Δ':>12}")
    for k in summary["raw"]:
        rv, fv = summary["raw"][k], summary["refined"][k]
        print(f"{k:<16}{rv:>12.4f}{fv:>12.4f}{fv - rv:>+12.4f}")
    print(f"\n[eval] wrote {out_json}")

    # qualitative figure
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(18, 2.4 * n))
    for r, (name, ir, rr, gr, far) in enumerate(samples):
        for c, (t, img) in enumerate([("raw render", ir), ("refined", rr),
                                      ("GT", gr)]):
            ax = axes[r, c] if n > 1 else axes[c]
            ax.imshow(np.clip(img, 0, far), cmap="turbo", aspect="auto")
            if r == 0:
                ax.set_title(t)
            ax.set_ylabel(f"f{name}"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"LiDAR range: raw vs refined vs GT — {args.scene}")
    fig.tight_layout()
    fig_path = scene_dir / "refiner" / "eval_qualitative.png"
    fig.savefig(fig_path, dpi=120)
    print(f"[eval] wrote {fig_path}")


if __name__ == "__main__":
    main()
