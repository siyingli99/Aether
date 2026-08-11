"""LiDAR refiner architectures (legacy + reference-guided cross-modal)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(),
            nn.Conv2d(co, co, 3, padding=1), nn.GroupNorm(8, co), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class LidarRefiner(nn.Module):
    """Legacy single-stream refiner (degraded 3ch only)."""

    def __init__(self, ch=48):
        super().__init__()
        self.d0 = ConvBlock(3, ch)
        self.d1 = ConvBlock(ch, ch * 2)
        self.d2 = ConvBlock(ch * 2, ch * 4)
        self.bott = ConvBlock(ch * 4, ch * 4)
        self.u2 = ConvBlock(ch * 4 + ch * 4, ch * 2)
        self.u1 = ConvBlock(ch * 2 + ch * 2, ch)
        self.u0 = ConvBlock(ch + ch, ch)
        self.head = nn.Conv2d(ch, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")

    def forward(self, x, x_ref=None):
        del x_ref
        e0 = self.d0(x)
        e1 = self.d1(self.pool(e0))
        e2 = self.d2(self.pool(e1))
        b = self.bott(self.pool(e2))
        d2 = self.u2(torch.cat([self.up(b), e2], 1))
        d1 = self.u1(torch.cat([self.up(d2), e1], 1))
        d0 = self.u0(torch.cat([self.up(d1), e0], 1))
        out = self.head(d0)
        rng = (x[:, 0:1] + 0.2 * torch.tanh(out[:, 0:1])).clamp(0, 1)
        inten = torch.sigmoid(out[:, 1:2] + x[:, 1:2])
        keep = out[:, 2:3]
        return torch.cat([rng, inten, keep], 1)


class CrossModalLidarRefiner(nn.Module):
    """Reference-guided LiDAR refiner (Difix_ref-style conditioning for LiDAR).

    Siamese encoder on (degraded, reference) spherical maps; fuse at bottleneck.
  ~4x wider than legacy LidarRefiner at ch=64.
    """

    def __init__(self, ch: int = 64):
        super().__init__()
        self.enc = nn.ModuleList([
            ConvBlock(3, ch),
            ConvBlock(ch, ch * 2),
            ConvBlock(ch * 2, ch * 4),
        ])
        self.fuse = ConvBlock(ch * 8, ch * 4)
        self.dec = nn.ModuleList([
            ConvBlock(ch * 4 + ch * 4 + ch * 4, ch * 2),
            ConvBlock(ch * 2 + ch * 2 + ch * 2, ch),
            ConvBlock(ch + ch + ch, ch),
        ])
        self.head = nn.Conv2d(ch, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")

    def _encode(self, x):
        feats = []
        h = x
        for block in self.enc:
            h = block(h)
            feats.append(h)
            h = self.pool(h)
        return feats, h

    def forward(self, x_deg: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        fd, bd = self._encode(x_deg)
        fr, br = self._encode(x_ref)
        b = self.fuse(torch.cat([bd, br], dim=1))
        d2 = self.dec[0](torch.cat([self.up(b), fd[2], fr[2]], dim=1))
        d1 = self.dec[1](torch.cat([self.up(d2), fd[1], fr[1]], dim=1))
        d0 = self.dec[2](torch.cat([self.up(d1), fd[0], fr[0]], dim=1))
        out = self.head(d0)
        rng = (x_deg[:, 0:1] + 0.2 * torch.tanh(out[:, 0:1])).clamp(0, 1)
        inten = torch.sigmoid(out[:, 1:2] + x_deg[:, 1:2])
        keep = out[:, 2:3]
        return torch.cat([rng, inten, keep], 1)


def refine_loss(pred, gt, *, use_ref: bool = False):
    del use_ref
    pr, pi, pk_logit = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    gr, gi, gm = gt[:, 0:1], gt[:, 1:2], gt[:, 2:3]
    keep_prob = torch.sigmoid(pk_logit)
    m = gm
    denom = m.sum().clamp(min=1.0)
    l_range = (F.l1_loss(pr, gr, reduction="none") * m).sum() / denom
    l_int = (F.l1_loss(pi, gi, reduction="none") * m).sum() / denom
    l_drop = F.binary_cross_entropy(keep_prob, gm)
    total = 5.0 * l_range + 1.0 * l_int + 1.0 * l_drop
    return total, {
        "range": l_range.item(),
        "int": l_int.item(),
        "drop": l_drop.item(),
        "total": total.item(),
    }


class CrossModalLidarRefinerV2(nn.Module):
    """Conservative residual refiner (v2 paradigm).

    * Range / intensity: small residual corrections **only on input hits**.
    * Keep: may recover a *bounded* fraction of missing rays (inpaint_cap),
      heavily penalized at train time — avoids the v1 ``+25k noise points``.
    * Still uses a GT reference sweep (Difix_ref-style) for cross-modal context.
    """

    def __init__(self, ch: int = 64, inpaint_cap: float = 0.28):
        super().__init__()
        self.inpaint_cap = float(inpaint_cap)
        self.enc = nn.ModuleList([
            ConvBlock(3, ch),
            ConvBlock(ch, ch * 2),
            ConvBlock(ch * 2, ch * 4),
        ])
        self.fuse = ConvBlock(ch * 8, ch * 4)
        self.dec = nn.ModuleList([
            ConvBlock(ch * 4 + ch * 4 + ch * 4, ch * 2),
            ConvBlock(ch * 2 + ch * 2 + ch * 2, ch),
            ConvBlock(ch + ch + ch, ch),
        ])
        self.head = nn.Conv2d(ch, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = lambda t: F.interpolate(t, scale_factor=2, mode="nearest")

    def _encode(self, x):
        feats = []
        h = x
        for block in self.enc:
            h = block(h)
            feats.append(h)
            h = self.pool(h)
        return feats, h

    def forward(self, x_deg: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        in_k = x_deg[:, 2:3].clamp(0, 1)
        in_r = x_deg[:, 0:1]
        in_i = x_deg[:, 1:2]

        fd, bd = self._encode(x_deg)
        fr, br = self._encode(x_ref)
        b = self.fuse(torch.cat([bd, br], dim=1))
        d2 = self.dec[0](torch.cat([self.up(b), fd[2], fr[2]], dim=1))
        d1 = self.dec[1](torch.cat([self.up(d2), fd[1], fr[1]], dim=1))
        d0 = self.dec[2](torch.cat([self.up(d1), fd[0], fr[0]], dim=1))
        out = self.head(d0)

        dr = 0.10 * torch.tanh(out[:, 0:1])
        di = 0.12 * torch.tanh(out[:, 1:2])
        rng = (in_r + dr * in_k).clamp(0, 1)
        inten = (in_i + di * in_k).clamp(0, 1)

        inpaint = torch.sigmoid(out[:, 2:3])
        keep = torch.clamp(in_k + (1.0 - in_k) * inpaint * self.inpaint_cap, 0, 1)
        return torch.cat([rng, inten, keep], 1)


def refine_loss_v2(
    pred,
    gt,
    input_keep,
    *,
    input_range=None,
    w_range: float = 6.0,
    w_int: float = 1.5,
    w_keep: float = 1.0,
    w_halluc: float = 4.0,
    w_sparse: float = 2.5,
    w_res: float = 0.5,
):
    """Anti-hallucination loss for v2 (pred keep is already in [0,1])."""
    pr, pi, pk = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
    gr, gi, gm = gt[:, 0:1], gt[:, 1:2], gt[:, 2:3]
    in_m = input_keep.clamp(0, 1)
    gm = (gm > 0.5).float()

    active = (gm > 0.5) | (in_m > 0.12)
    denom = active.sum().clamp(min=1.0)
    l_range = (F.l1_loss(pr, gr, reduction="none") * active).sum() / denom
    l_int = (F.l1_loss(pi, gi, reduction="none") * active).sum() / denom

    l_keep = F.binary_cross_entropy(pk, gm, reduction="mean")
    # Penalise confident keeps where input was empty (classic v1 failure mode).
    empty_in = (in_m < 0.08).float()
    l_halluc = (pk * empty_in * (1.0 - gm)).mean()
    l_sparse = (pk * (1.0 - in_m)).mean()

    l_res = torch.tensor(0.0, device=pr.device)
    if input_range is not None:
        l_res = (torch.abs(pr - input_range) * in_m).sum() / in_m.sum().clamp(min=1.0)

    total = (w_range * l_range + w_int * l_int + w_keep * l_keep
             + w_halluc * l_halluc + w_sparse * l_sparse + w_res * l_res)
    return total, {
        "range": float(l_range.item()),
        "int": float(l_int.item()),
        "keep": float(l_keep.item()),
        "halluc": float(l_halluc.item()),
        "sparse": float(l_sparse.item()),
        "res": float(l_res.item()),
        "total": float(total.item()),
    }
