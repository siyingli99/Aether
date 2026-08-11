"""Function: Train a cross-modal LiDAR refiner and export a unified inference bundle.
Input/Output: Reads a merged train/test pair manifest; writes checkpoints, diagnostics, and a unified bundle.
Example:
    # Prerequisite: generate and merge pair manifests first.
    python aether/refine/train_unified_lidar.py \
        --pairs outputs/multimodal_refine/global_pairs.json \
        --out outputs/multimodal_refine/unified --epochs 80 --bs 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aether.refine.lidar_refiner_models import CrossModalLidarRefiner, refine_loss
from aether.refine.unified_multimodal_enhancer import UnifiedEnhancerConfig, UnifiedMultimodalEnhancer
from aether.paths import AETHER_ROOT


class CrossModalPairs(Dataset):
    def __init__(self, pairs_json: str, split: str, far: float, aug: bool = False):
        with open(pairs_json) as f:
            self.items = list(json.load(f)[split].values())
        self.far = float(far)
        self.aug = aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d = np.load(self.items[i]["npz"])
        def _t(a):
            return torch.from_numpy(a).float()

        x_deg = torch.stack([
            _t(d["in_range"]) / self.far,
            _t(d["in_int"]),
            _t(d["in_keep"]),
        ], 0)
        x_ref = torch.stack([
            _t(d["ref_range"]) / self.far,
            _t(d["ref_int"]),
            _t(d["ref_keep"]),
        ], 0)
        y = torch.stack([
            _t(d["gt_range"]) / self.far,
            _t(d["gt_int"]),
            (_t(d["gt_mask"]) > 0.5).float(),
        ], 0)
        if self.aug and torch.rand(()) < 0.8:
            s = int(torch.randint(0, x_deg.shape[-1], ()))
            x_deg = torch.roll(x_deg, s, dims=-1)
            x_ref = torch.roll(x_ref, s, dims=-1)
            y = torch.roll(y, s, dims=-1)
        return x_deg, x_ref, y


@torch.no_grad()
def save_val_fig(model, ds, far, path, device, n=3):
    """Visualize refiner effect. Range shown auto-scaled to the near-field
    (0..0.4*far) so sub-metre corrections are visible, plus an input-vs-refined
    difference map (the old far-normalized view hid the ~0.2m corrections)."""
    model.eval()
    ncol = 5
    fig, axes = plt.subplots(n, ncol, figsize=(5 * ncol, 2.6 * n))
    vmax = 0.4  # normalized range display cap (=0.4*far metres)
    for r in range(n):
        x_deg, x_ref, y = ds[r * max(1, len(ds) // n)]
        p = model(x_deg[None].to(device), x_ref[None].to(device))[0].cpu()
        mask = (y[2] > 0.5).numpy()
        diff = (p[0] - x_deg[0]).numpy() * far  # metres
        diff_masked = np.where(mask, diff, 0.0)
        panels = [
            ("input range", x_deg[0].numpy(), "turbo", 0, vmax),
            ("refined range", p[0].numpy(), "turbo", 0, vmax),
            ("GT range", y[0].numpy(), "turbo", 0, vmax),
            ("refined-input (m)", diff_masked, "coolwarm", -3, 3),
            ("refined keep", torch.sigmoid(p[2]).numpy(), "viridis", 0, 1),
        ]
        for c, (title, img, cmap, vmn, vmx) in enumerate(panels):
            ax = axes[r, c] if n > 1 else axes[c]
            im = ax.imshow(img, cmap=cmap, aspect="auto", vmin=vmn, vmax=vmx)
            if r == 0:
                ax.set_title(title)
            ax.axis("off")
            if c == 3:
                fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pairs",
        default=str(AETHER_ROOT / "outputs/multimodal_refine/global_pairs.json"),
    )
    ap.add_argument(
        "--out",
        default=str(AETHER_ROOT / "outputs/multimodal_refine/unified"),
    )
    ap.add_argument("--far", type=float, default=100.0)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rgb_model", default="nvidia/difix_ref",
                    help="stored in bundle only; not trained here")
    args = ap.parse_args()

    device = "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = CrossModalPairs(args.pairs, "train", args.far, aug=True)
    va = CrossModalPairs(args.pairs, "test", args.far, aug=False)
    dl = DataLoader(tr, batch_size=args.bs, shuffle=True, num_workers=4, drop_last=True)
    print(f"[unified_train] train={len(tr)} val={len(va)}", flush=True)

    model = CrossModalLidarRefiner(ch=args.ch).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    cfg = UnifiedEnhancerConfig(rgb_model=args.rgb_model, far=args.far, lidar_ch=args.ch)
    bundle_helper = UnifiedMultimodalEnhancer(cfg, device=device)
    bundle_helper.lidar = model

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        agg = {"range": 0.0, "int": 0.0, "drop": 0.0, "total": 0.0}
        for x_deg, x_ref, y in dl:
            x_deg, x_ref, y = x_deg.to(device), x_ref.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x_deg, x_ref)
            loss, logs = refine_loss(pred, y)
            loss.backward()
            opt.step()
            for k in agg:
                agg[k] += logs[k]
        sched.step()
        nb = max(1, len(dl))
        model.eval()
        vtot = 0.0
        with torch.no_grad():
            for i in range(len(va)):
                x_deg, x_ref, y = va[i]
                pred = model(x_deg[None].to(device), x_ref[None].to(device))
                _, vl = refine_loss(pred, y[None].to(device))
                vtot += vl["total"]
        vtot /= max(1, len(va))
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:3d} train total={agg['total']/nb:.4f} val={vtot:.4f}", flush=True)
        if vtot < best:
            best = vtot
            bundle_helper.save_bundle(
                out_dir / "unified_multimodal.bundle.pth",
                extra={"val_total": vtot, "epoch": ep, "train_pairs": len(tr)},
            )
            torch.save({"model": model.state_dict(), "far": args.far, "ch": args.ch,
                        "epoch": ep, "val": vtot},
                       out_dir / "lidar_only_best.pth")

    save_val_fig(model, va, args.far, out_dir / "val_compare.png", device)
    print(f"[unified_train] best val={best:.4f}", flush=True)
    print(f"[unified_train] bundle -> {out_dir/'unified_multimodal.bundle.pth'}", flush=True)
    print(f"[unified_train] RGB inference uses HF model: {args.rgb_model} (pretrained, not in bundle)", flush=True)


if __name__ == "__main__":
    main()
