"""Function: Train the single-scene LiDAR range-image refiner.
Input/Output: Reads a generated scene ``pairs.json`` and NPZ files; writes best weights and training curves.
Example:
    # Prerequisite: generate outputs/lidar_refine/pandaset095/pairs.json first.
    python aether/refine/train_lidar_refiner.py --scene pandaset095 \
        --data_root outputs/lidar_refine --out outputs/lidar_refine/pandaset095/refiner \
        --epochs 120 --bs 4
"""
from __future__ import annotations

from aether.paths import use_vendored_backends
use_vendored_backends()  # DriveStudio / SplatAD on sys.path

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aether.paths import AETHER_ROOT

from aether.refine.lidar_refiner_models import LidarRefiner, refine_loss

# backward-compatible re-exports for tools that import from this module
__all__ = ["LidarRefiner", "refine_loss", "LidarPairs"]


class LidarPairs(Dataset):
    """Loads float npz pairs; resizes to a conv-friendly (H, W)."""

    def __init__(self, pairs_json: str, split: str, far: float,
                 hw=(64, 1800), aug: bool = False):
        with open(pairs_json) as f:
            self.items = list(json.load(f)[split].values())
        self.far = float(far)
        self.hw = hw            # native (64,1800); no resize -> no far-plane bleed
        self.aug = aug

    def __len__(self):
        return len(self.items)

    def _t(self, a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(a).float()

    def __getitem__(self, i):
        d = np.load(self.items[i]["npz"])
        in_r = self._t(d["in_range"]) / self.far
        in_i = self._t(d["in_int"])
        in_k = self._t(d["in_keep"])
        gt_r = self._t(d["gt_range"]) / self.far
        gt_i = self._t(d["gt_int"])
        gt_m = (self._t(d["gt_mask"]) > 0.5).float()
        x = torch.stack([in_r, in_i, in_k], 0)
        y = torch.stack([gt_r, gt_i, gt_m], 0)
        if self.aug and torch.rand(()) < 0.8:           # circular azimuth shift
            s = int(torch.randint(0, x.shape[-1], ()))
            x = torch.roll(x, s, dims=-1)
            y = torch.roll(y, s, dims=-1)
        return x, y


# --------------------------------------------------------------------------- #
@torch.no_grad()
def save_val_fig(model, ds, far, path, device, n=3):
    model.eval()
    fig, axes = plt.subplots(n, 4, figsize=(20, 2.2 * n))
    for r in range(n):
        x, y = ds[r * max(1, len(ds) // n)]
        p = model(x[None].to(device))[0].cpu()
        for c, (title, img) in enumerate([
            ("input range", x[0]), ("refined range", p[0]),
            ("GT range", y[0]), ("refined keep", torch.sigmoid(p[2]))]):
            ax = axes[r, c] if n > 1 else axes[c]
            ax.imshow(img.numpy(), cmap="turbo", aspect="auto", vmin=0, vmax=1)
            if r == 0:
                ax.set_title(title)
            ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="nuscenes088")
    ap.add_argument("--data_root", default=str(AETHER_ROOT / "outputs/lidar_refine"))
    ap.add_argument("--far", type=float, default=100.0)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda"
    scene_dir = Path(args.data_root) / args.scene
    pairs = scene_dir / "pairs.json"
    out_dir = Path(args.out) if args.out else scene_dir / "refiner"
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = LidarPairs(str(pairs), "train", args.far, aug=True)
    va = LidarPairs(str(pairs), "test", args.far, aug=False)
    dl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    print(f"[train] scene={args.scene} train={len(tr)} val={len(va)}", flush=True)

    model = LidarRefiner().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        agg = {"range": 0, "int": 0, "drop": 0, "total": 0}
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss, logs = refine_loss(model(x), y)
            loss.backward(); opt.step()
            for k in agg:
                agg[k] += logs[k]
        sched.step()
        nb = len(dl)
        # quick val
        model.eval(); vtot = 0.0
        with torch.no_grad():
            for i in range(len(va)):
                x, y = va[i]
                _, vl = refine_loss(model(x[None].to(device)), y[None].to(device))
                vtot += vl["total"]
        vtot /= max(1, len(va))
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:3d} train[range={agg['range']/nb:.4f} "
                  f"int={agg['int']/nb:.4f} drop={agg['drop']/nb:.4f}] "
                  f"val_total={vtot:.4f}", flush=True)
        if vtot < best:
            best = vtot
            torch.save({"model": model.state_dict(), "far": args.far,
                        "hw": tr.hw, "epoch": ep, "val": vtot},
                       out_dir / "refiner_best.pth")
    save_val_fig(model, va, args.far, out_dir / "val_compare.png", device)
    print(f"[train] best val_total={best:.4f} -> {out_dir/'refiner_best.pth'}\n"
          f"[train] wrote {out_dir/'val_compare.png'}", flush=True)


if __name__ == "__main__":
    main()
