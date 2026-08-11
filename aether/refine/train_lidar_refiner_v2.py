"""Function: Train the conservative residual LiDAR refiner v2 on real 3DGS pairs.
Input/Output: Reads a merged pair manifest; writes weights, curves, and a unified inference bundle.
Example:
    # Prerequisite: generate and merge real-render pair manifests first.
    python aether/refine/train_lidar_refiner_v2.py \
        --pairs outputs/multimodal_refine_v2/global_pairs.json \
        --out outputs/multimodal_refine_v2/unified --epochs 120 --bs 8
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aether.refine.lidar_refiner_models import CrossModalLidarRefinerV2, refine_loss_v2
from aether.refine.unified_multimodal_enhancer import UnifiedEnhancerConfig, UnifiedMultimodalEnhancer


_REAL_SOURCES = {
    "real_render", "real_render_perturbed", "real_render_nominal",
    "render", "3dgs",
}


class CrossModalPairsV2(Dataset):
    def __init__(
        self,
        pairs_json: str,
        split: str,
        far: float,
        aug: bool = False,
        real_only: bool = True,
        real_weight: float = 5.0,
    ):
        with open(pairs_json) as f:
            blob = json.load(f)[split]
        self.items = list(blob.values())
        self.keys = list(blob.keys())
        self.far = float(far)
        self.aug = aug
        self.weights = [1.0] * len(self.items)

        if real_only:
            kept = []
            w = []
            for k, d in zip(self.keys, self.items):
                src = str(d.get("source", "")).lower()
                if src in _REAL_SOURCES or src.startswith("real"):
                    kept.append(d)
                    w.append(real_weight)
                elif not real_only:
                    kept.append(d)
                    w.append(1.0)
            if kept:
                self.items = kept
                self.weights = w
            else:
                print("[v2_train] WARN: no real_render pairs tagged; using all pairs", flush=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        d = self.items[i]
        arr = np.load(d["npz"])
        def _t(a):
            return torch.from_numpy(a).float()

        x_deg = torch.stack([_t(arr["in_range"]) / self.far, _t(arr["in_int"]), _t(arr["in_keep"])], 0)
        x_ref = torch.stack([_t(arr["ref_range"]) / self.far, _t(arr["ref_int"]), _t(arr["ref_keep"])], 0)
        y = torch.stack([_t(arr["gt_range"]) / self.far, _t(arr["gt_int"]), (_t(arr["gt_mask"]) > 0.5).float()], 0)

        if self.aug and torch.rand(()) < 0.85:
            s = int(torch.randint(0, x_deg.shape[-1], ()))
            x_deg = torch.roll(x_deg, s, dims=-1)
            x_ref = torch.roll(x_ref, s, dims=-1)
            y = torch.roll(y, s, dims=-1)
        return x_deg, x_ref, y

    def sampler(self) -> WeightedRandomSampler | None:
        if not self.weights or len(set(self.weights)) <= 1:
            return None
        return WeightedRandomSampler(self.weights, num_samples=len(self.weights), replacement=True)


@torch.no_grad()
def save_val_fig(model, ds, far, path, device, n=3):
    model.eval()
    ncol = 6
    fig, axes = plt.subplots(n, ncol, figsize=(3.2 * ncol, 2.6 * n))
    vmax = 0.4
    for r in range(n):
        idx = r * max(1, len(ds) // n)
        x_deg, x_ref, y = ds[idx]
        p = model(x_deg[None].to(device), x_ref[None].to(device))[0].cpu()
        in_k = x_deg[2].numpy()
        mask = y[2].numpy() > 0.5
        new_pts = (p[2].numpy() > 0.25) & (in_k < 0.08)
        panels = [
            ("input keep", in_k, "gray", 0, 1),
            ("refined keep", p[2].numpy(), "gray", 0, 1),
            ("GT keep", mask.astype(float), "gray", 0, 1),
            ("new pts (halluc)", new_pts.astype(float), "Reds", 0, 1),
            ("input range", x_deg[0].numpy(), "turbo", 0, vmax),
            ("refined range", p[0].numpy(), "turbo", 0, vmax),
        ]
        for c, (title, img, cmap, vmn, vmx) in enumerate(panels):
            ax = axes[r, c] if n > 1 else axes[c]
            ax.imshow(img, cmap=cmap, aspect="auto", vmin=vmn, vmax=vmx)
            if r == 0:
                ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="outputs/multimodal_refine_v2/global_pairs.json")
    ap.add_argument("--out", default="outputs/multimodal_refine_v2/unified")
    ap.add_argument("--far", type=float, default=100.0)
    ap.add_argument("--ch", type=int, default=64)
    ap.add_argument("--inpaint_cap", type=float, default=0.28)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--real_only", action="store_true", default=True)
    ap.add_argument("--allow_synthetic", action="store_true",
                    help="also train on synthetic gt_corpus pairs")
    ap.add_argument("--rgb_model", default="nvidia/difix")
    args = ap.parse_args()

    real_only = args.real_only and not args.allow_synthetic
    device = "cuda"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr = CrossModalPairsV2(args.pairs, "train", args.far, aug=True, real_only=real_only)
    va = CrossModalPairsV2(args.pairs, "test", args.far, aug=False, real_only=False)
    sampler = tr.sampler()
    dl = DataLoader(tr, batch_size=args.bs, shuffle=sampler is None,
                    sampler=sampler, num_workers=4, drop_last=True)
    print(f"[v2_train] train={len(tr)} val={len(va)} real_only={real_only}", flush=True)

    model = CrossModalLidarRefinerV2(ch=args.ch, inpaint_cap=args.inpaint_cap).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    cfg = UnifiedEnhancerConfig(
        rgb_model=args.rgb_model, far=args.far,
        lidar_arch="cross_modal_v2", lidar_ch=args.ch,
    )
    bundle_helper = UnifiedMultimodalEnhancer(cfg, device=device)
    bundle_helper.lidar = model

    best = 1e9
    for ep in range(args.epochs):
        model.train()
        agg = {k: 0.0 for k in ("range", "int", "keep", "halluc", "sparse", "res", "total")}
        for x_deg, x_ref, y in dl:
            x_deg, x_ref, y = x_deg.to(device), x_ref.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x_deg, x_ref)
            loss, logs = refine_loss_v2(
                pred, y, x_deg[:, 2:3], input_range=x_deg[:, 0:1],
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            for k in agg:
                agg[k] += logs[k]
        sched.step()
        nb = max(1, len(dl))
        model.eval()
        vtot = 0.0
        vhall = 0.0
        with torch.no_grad():
            for i in range(len(va)):
                x_deg, x_ref, y = va[i]
                pred = model(x_deg[None].to(device), x_ref[None].to(device))
                _, vl = refine_loss_v2(pred, y[None].to(device), x_deg[2:3][None].to(device),
                                       input_range=x_deg[0:1][None].to(device))
                vtot += vl["total"]
                vhall += vl["halluc"]
        vtot /= max(1, len(va))
        vhall /= max(1, len(va))
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"  ep{ep:3d} train total={agg['total']/nb:.4f} "
                  f"halluc={agg['halluc']/nb:.4f} val={vtot:.4f} val_halluc={vhall:.4f}",
                  flush=True)
        if vtot < best:
            best = vtot
            bundle_helper.save_bundle(
                out_dir / "unified_multimodal.bundle.pth",
                extra={"val_total": vtot, "val_halluc": vhall, "epoch": ep,
                       "lidar_arch": "cross_modal_v2", "inpaint_cap": args.inpaint_cap},
            )
            torch.save({"model": model.state_dict(), "far": args.far, "ch": args.ch,
                        "inpaint_cap": args.inpaint_cap, "epoch": ep, "val": vtot},
                       out_dir / "lidar_v2_best.pth")

    save_val_fig(model, va, args.far, out_dir / "val_compare_v2.png", device)
    print(f"[v2_train] best val={best:.4f}", flush=True)
    print(f"[v2_train] bundle -> {out_dir/'unified_multimodal.bundle.pth'}", flush=True)


if __name__ == "__main__":
    main()
