"""Function: Distill enhanced novel-view RGB and LiDAR pseudo-targets into a joint scene.
Input/Output: Reads a joint run and trained refiner bundle; writes a distilled background, sidecar, and comparisons.
Example:
    # Prerequisite: generate the joint checkpoint/sidecar and refiner bundle first.
    python aether/refine/distill_multimodal.py --scene pandaset095 \
        --joint_dir outputs/dualmodal/aether_joint/pandaset/scene095_joint \
        --refiner_bundle outputs/multimodal_refine_v2/unified/unified_multimodal.bundle.pth \
        --rounds 3 --inner_steps 100 --out outputs/distilled/pandaset095
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aether.paths import AETHER_ROOT, DRIVESTUDIO_ROOT as _DS_ROOT  # noqa: E402
sys.path.insert(0, str(_DS_ROOT))
sys.path.insert(0, str(AETHER_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from PIL import Image


def _save_png(path, t):
    """Save a [H,W,3] float tensor in [0,1] as PNG."""
    a = (t.detach().clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(a).save(str(path))

from aether.render.ds_backend import DriveStudioRenderer
from aether.render.lidar_closed_loop import ClosedLoopLidar
from aether.refine.train_lidar_refiner import LidarRefiner
from aether.train.joint_phase import gt_to_spherical_grid  # noqa: F401 (eval reuse)


_SCENES = {
    "nuscenes088": "outputs/dualmodal/aether_joint/nuscenes/scene088_joint",
    "nuscenes0383": "outputs/dualmodal/aether_joint/nuscenes/scene0383_joint",
    "pandaset017": "outputs/dualmodal/aether_joint/pandaset/scene017_joint",
    "pandaset120": "outputs/dualmodal/aether_joint/pandaset/scene120_joint",
}
_AETHER = AETHER_ROOT


def _abs_aether(path: str | Path) -> Path:
    """Resolve paths relative to the AETHER repo root (not drivestudio cwd)."""
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = _AETHER / p
    return p.resolve()


# --------------------------------------------------------------------------- #
def make_world_delta(yaw_deg: float, dx: float, dy: float, dz: float,
                     lidar2world: np.ndarray) -> np.ndarray:
    """A REALISTIC ego-motion novel-view delta, returned as a WORLD-frame
    transform to be LEFT-multiplied onto any sensor pose (camera + LiDAR move
    consistently).

    The motion is defined in the EGO/LiDAR LOCAL frame (x-forward, y-left,
    z-up): ``yaw_deg`` is a steering rotation about the ego up-axis and
    ``(dx, dy, dz)`` is a forward/left/up translation. We then conjugate it
    into the world frame so it behaves like the ego vehicle actually shifting
    laterally / steering, instead of rolling the image plane about an
    arbitrary world axis.

        T_world = L · T_local · L⁻¹ ,   L = lidar2world
        novel_pose = T_world · pose
    """
    th = np.deg2rad(yaw_deg)
    Rz = np.array([[np.cos(th), -np.sin(th), 0],
                   [np.sin(th),  np.cos(th), 0],
                   [0, 0, 1]], dtype=np.float32)
    T_local = np.eye(4, dtype=np.float32)
    T_local[:3, :3] = Rz
    T_local[:3, 3] = np.array([dx, dy, dz], np.float32)   # forward/left/up
    L = np.asarray(lidar2world, np.float32)
    return (L @ T_local @ np.linalg.inv(L)).astype(np.float32)


def lidar_refiner_enhance_v1(refiner, out, far, device):
    """Legacy v1 single-stream refiner -> pseudo (range,int,keep)."""
    r = (out["range"].clamp(0, far) / far)[None, None]
    i = out["intensity"].clamp(0, 1)[None, None]
    k = (1 - torch.sigmoid(out["raydrop_logit"])).clamp(0, 1)[None, None]
    x = torch.cat([r, i, k], 1)
    with torch.no_grad():
        p = refiner(x)
    pr = (p[:, 0:1] * far)[0, 0]
    pi = p[:, 1:2][0, 0]
    pk = torch.sigmoid(p[:, 2:3])[0, 0]
    return pr.detach(), pi.detach(), pk.detach()


def _build_gt_ref_cache(src, data_path: str, cl, device):
    """Per-frame GT spherical sweeps used as LiDAR refiner reference (v2)."""
    from pathlib import Path as _Path
    from aether.refine.pair_utils import project_gt_frame

    lidar_bin = _Path(data_path) / "lidar"
    cache = {}
    n = src.lidar_to_worlds.shape[0]
    for f in range(n):
        gt = project_gt_frame(src, f, lidar_bin, cl.cfg, device)
        cache[f] = (
            torch.from_numpy(gt["gt_range"]).to(device),
            torch.from_numpy(gt["gt_int"]).to(device),
            torch.from_numpy(gt["gt_mask"]).to(device),
        )
    return cache


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="nuscenes088",
                    help="Key into _SCENES when --joint_dir is omitted.")
    ap.add_argument("--joint_dir", default="",
                    help="Override joint checkpoint dir (relative to AETHER root).")
    ap.add_argument("--refiner", default="",
                    help="Per-scene v1 refiner_best.pth (ignored if --refiner_bundle set).")
    ap.add_argument("--refiner_bundle", default="",
                    help="Unified v2 bundle .pth (CrossModalLidarRefinerV2).")
    # Difix3D+ progressive schedule: `rounds` re-fix passes with growing pose
    # shift, each followed by `inner_steps` of distillation. Total opt steps =
    # rounds * inner_steps. (--steps kept as an alias when --no_rgb.)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--inner_steps", type=int, default=100)
    ap.add_argument("--shift", type=float, default=0.4,
                    help="max world translation (m) at the final round.")
    ap.add_argument("--yaw_max", type=float, default=20.0,
                    help="max yaw perturbation (deg) at the final round.")
    ap.add_argument("--steps", type=int, default=0,
                    help="alias: if >0, overrides rounds=1 inner_steps=steps.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_views", type=int, default=12)
    ap.add_argument("--frames", default="",
                    help="Comma-separated frame ids (paper RD distill set). "
                         "Overrides --n_views linspace sampling when set.")
    ap.add_argument("--img_hw", default="256,448")
    ap.add_argument("--no_rgb", action="store_true",
                    help="LiDAR-only distillation (skip Difix RGB stream).")
    ap.add_argument("--use_gt_rgb", action="store_true",
                    help="When shift=0, use logged GT camera RGB as distill targets "
                         "(instead of Difix). Safer for heldout GT metrics.")
    ap.add_argument("--no_uncertainty", action="store_true",
                    help="Disable confidence weighting on RGB/LiDAR losses.")
    ap.add_argument("--freeze_means", action="store_true",
                    help="Do not optimize Background._means (safer for heldout R-MAE).")
    ap.add_argument("--appearance_only", action="store_true",
                    help="Only optimize SH appearance + sidecar (freeze means/scales/quats/opacity).")
    ap.add_argument("--conf_tau", type=float, default=0.05,
                    help="RGB uncertainty temperature; smaller = stronger downweight on edits.")
    ap.add_argument("--lidar_conf_frac", type=float, default=0.05,
                    help="LiDAR conf = exp(-|raw-ref| / (frac*far)); smaller = stricter.")
    ap.add_argument("--early_stop_patience", type=int, default=0,
                    help="If >0, stop when RGB+LiDAR loss worsens for this many inner steps.")
    ap.add_argument("--w_rgb", type=float, default=1.0)
    ap.add_argument("--w_lidar", type=float, default=1.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if args.steps > 0:
        args.rounds, args.inner_steps = 1, args.steps
    img_h, img_w = (int(x) for x in args.img_hw.split(","))

    device = "cuda"
    jdir = _abs_aether(args.joint_dir) if args.joint_dir else _abs_aether(_SCENES[args.scene])
    ckpt = jdir / "checkpoint_final.pth"
    sidecar = sorted(jdir.glob("lidar_sidecar_step*.pt"))[-1]
    config = jdir / "config.yaml"
    refiner_bundle = _abs_aether(args.refiner_bundle) if args.refiner_bundle else None
    refiner_ckpt = (_abs_aether(args.refiner) if args.refiner
                    else _abs_aether(f"outputs/lidar_refine/{args.scene}/refiner/refiner_best.pth"))
    out_dir = _abs_aether(args.out) if args.out else jdir / "distill"
    out_dir.mkdir(parents=True, exist_ok=True)

    refiner = None
    enhancer = None
    gt_ref_cache = {}
    if refiner_bundle is not None:
        from aether.refine.unified_multimodal_enhancer import load_unified_bundle
        enhancer = load_unified_bundle(str(refiner_bundle), device=device)
        print(f"[distill] scene={args.scene}\n  ckpt={ckpt}\n  bundle={refiner_bundle}",
              flush=True)
    else:
        print(f"[distill] scene={args.scene}\n  ckpt={ckpt}\n  refiner={refiner_ckpt}",
              flush=True)
        rc = torch.load(refiner_ckpt, map_location=device, weights_only=False)
        refiner = LidarRefiner().to(device)
        refiner.load_state_dict(rc["model"]); refiner.eval()
        for p in refiner.parameters():
            p.requires_grad_(False)

    renderer = DriveStudioRenderer(str(ckpt), device=device)
    cl = ClosedLoopLidar(renderer, str(sidecar), device=device)
    far = float(cl.cfg.far_plane)

    # RGB refiner: Difix3D+ style OFFLINE fix via refine_directory using
    # nvidia/difix_ref (one-step diffusion WITH a nearest-train-view reference).
    # We render novel views to disk, fix them with difix_ref, and use the fixed
    # images as pseudo-GT -- this is the drivestudio port of Difix3D+'s fix().
    use_rgb = not args.no_rgb
    refine_directory = None
    if use_rgb:
        from aether.refine import refine_directory  # noqa: F811

    # ---- lidar poses (+ optional camera rig) from dataset ----
    # LiDAR-only path uses the lightweight lidar source (no slow image load);
    # the RGB path needs normalized camera poses -> full DrivingDataset.
    cfg = OmegaConf.load(config)
    _prev = os.getcwd(); os.chdir(str(_DS_ROOT))
    try:
        from aether.refine.pair_utils import build_lidar_source
        src, data_path = build_lidar_source(cfg, device)
        cam_obj = None
        if use_rgb:
            from utils.misc import import_str
            pose_dir = "lidar_pose" if os.path.isdir(os.path.join(data_path, "lidar_pose")) else "ego_pose"
            tot = len(os.listdir(os.path.join(data_path, pose_dir)))
            e = (tot - 1 if cfg.data.end_timestep == -1 else cfg.data.end_timestep) + 1
            cam_obj = import_str(cfg.data.pixel_source.type)(
                cfg.data.dataset, cfg.data.pixel_source, data_path,
                cfg.data.start_timestep, e, device=device)
            cam_obj.to(device)
    finally:
        os.chdir(_prev)
    n_frames = src.lidar_to_worlds.shape[0]
    if enhancer is not None:
        gt_ref_cache = _build_gt_ref_cache(src, data_path, cl, device)
        print(f"[distill] GT ref cache: {len(gt_ref_cache)} frames", flush=True)

    # frames spread across the sequence. base_* are the UNSHIFTED training poses;
    # the camera ones double as difix_ref REFERENCE views (nearest real view).
    if args.frames.strip():
        base_frames = [int(x) for x in args.frames.split(",") if x.strip()]
        base_frames = [f for f in base_frames if 0 <= f < n_frames]
        if not base_frames:
            raise SystemExit(f"--frames empty after clip to [0,{n_frames})")
    else:
        base_frames = np.linspace(0, n_frames - 1, args.n_views).astype(int).tolist()
    base_lp, base_cc, base_K = {}, {}, {}
    for f in base_frames:
        base_lp[f] = src.lidar_to_worlds[f].cpu().numpy().astype(np.float32)
        if cam_obj is not None:
            _, ci = cam_obj.get_image(f * cam_obj.num_cams + 0)
            base_cc[f] = ci["camera_to_world"].cpu().numpy().astype(np.float32)
            base_K[f] = ci["intrinsics"].cpu().numpy().astype(np.float32)
    rng = np.random.default_rng(0)

    def build_novel(frac):
        """Progressive novel views (Difix3D+): pose shift grows with frac<=1.
        Returns list of (frame, novel_lidar2world, novel_cam2world, K)."""
        out, s = [], args.shift * frac
        for f in base_frames:
            lp = base_lp[f]
            yaw = float(rng.uniform(-1, 1)) * args.yaw_max * frac
            dx = float(rng.uniform(-1, 1)) * s          # forward/back
            dy = float(rng.uniform(-1, 1)) * s          # left/right
            T = make_world_delta(yaw, dx, dy, 0.0, lidar2world=lp)
            cc = (T @ base_cc[f]).astype(np.float32) if f in base_cc else None
            out.append((f, (T @ lp).astype(np.float32), cc, base_K.get(f)))
        return out

    # ---- params to optimize (Background appearance + sidecar; geometry optional) ----
    bg = renderer.trainer.models["Background"]
    sidecar_int = torch.nn.Parameter(cl.feat_intensity["Background"].clone())
    sidecar_rd = torch.nn.Parameter(cl.feat_raydrop["Background"].clone())
    opt_params = [sidecar_int, sidecar_rd]
    if args.appearance_only:
        geom_attrs = []
        app_attrs = ["_features_dc", "_features_rest"]
    elif args.freeze_means:
        geom_attrs = ["_opacities", "_scales", "_quats"]
        app_attrs = ["_features_dc", "_features_rest"]
    else:
        geom_attrs = ["_opacities", "_scales", "_means", "_quats"]
        app_attrs = ["_features_dc", "_features_rest"]
    for attr in geom_attrs + app_attrs:
        if hasattr(bg, attr):
            t = getattr(bg, attr)
            if isinstance(t, torch.Tensor):
                t.requires_grad_(True); opt_params.append(t)
    # Freeze means explicitly when requested
    if args.freeze_means or args.appearance_only:
        if hasattr(bg, "_means") and isinstance(bg._means, torch.Tensor):
            bg._means.requires_grad_(False)
    if args.appearance_only:
        for attr in ("_opacities", "_scales", "_quats", "_means"):
            if hasattr(bg, attr) and isinstance(getattr(bg, attr), torch.Tensor):
                getattr(bg, attr).requires_grad_(False)
    opt = torch.optim.Adam(opt_params, lr=args.lr)
    print(f"[distill] optimizing {len(opt_params)} param tensors over "
          f"{len(base_frames)} views x {args.rounds} rounds x {args.inner_steps} "
          f"steps (RGB={'on' if use_rgb else 'off'}; "
          f"appearance_only={args.appearance_only} freeze_means={args.freeze_means})",
          flush=True)

    # snapshot BEFORE for visualization (attributes + a fixed-view render)
    before = {
        "opacity": bg.get_opacity.detach().reshape(-1).cpu().numpy().copy(),
        "scale": bg.get_scaling.detach().mean(-1).cpu().numpy().copy(),
        "n": int(bg._means.shape[0]),
    }
    # fixed deterministic novel view for before/after comparison
    mid = base_frames[len(base_frames) // 2]
    Tvis = make_world_delta(8.0, 0.0, args.shift, 0.0, lidar2world=base_lp[mid])
    vis_f = mid
    vis_lp = (Tvis @ base_lp[mid]).astype(np.float32)
    vis_cc = (Tvis @ base_cc[mid]).astype(np.float32) if mid in base_cc else None
    vis_K = base_K.get(mid)
    _bout = cl.render_sweep(vis_lp, vis_f, background_only=True, apply_overrides=False)
    before_range = _bout["range"].detach().cpu().numpy()
    before_pts = cl.sweep_to_points(_bout)
    before_rgb = None
    if vis_cc is not None:
        with torch.no_grad():
            before_rgb = renderer.render(
                vis_cc, vis_K, img_h, img_w, vis_f)["rgb"].clamp(0, 1).cpu().numpy()

    def lidar_step(f, lp_n):
        """One differentiable LiDAR-stream loss against the refiner pseudo-GT."""
        renderer._set_cur_frame(int(f))
        means = bg._means; quats = bg.get_quats
        scales = bg.get_scaling; opac = bg.get_opacity.reshape(-1)
        pose = torch.as_tensor(lp_n, dtype=torch.float32, device=device)
        out = cl.head.render(
            means_w=means, quats=quats, scales=scales, opacities=opac,
            # Logits only — LidarHead.render sigmoid-activates after rasterization.
            intensity=sidecar_int, raydrop_logit=sidecar_rd,
            lidar_pose_w=pose, raster_pts=cl.raster_pts,
            tile_elevation_boundaries=cl.teb,
        )
        if enhancer is not None:
            ref_r, ref_i, ref_m = gt_ref_cache[int(f)]
            pr, pi, pk = enhancer.enhance_lidar_render_dict(
                out, ref_r, ref_i, ref_m)
        else:
            pr, pi, pk = lidar_refiner_enhance_v1(refiner, out, far, device)
        keep_pred = torch.sigmoid(out["raydrop_logit"])
        mask = (pk > 0.3).float()
        # uncertainty: down-weight cells the refiner edited a lot
        if args.no_uncertainty:
            conf = torch.ones_like(pr)
        else:
            conf = torch.exp(
                -(out["range"].detach().clamp(0, far) - pr).abs()
                / (args.lidar_conf_frac * far)
            )
        w = mask * conf
        denom = w.sum().clamp(min=1.0)
        l_rng = (F.l1_loss(out["range"].clamp(0, far), pr, reduction="none")
                 * w).sum() / denom / far
        l_int = (F.l1_loss(out["intensity"], pi, reduction="none") * w).sum() / denom
        l_drop = F.binary_cross_entropy(1 - keep_pred, 1 - (pk > 0.3).float())
        return 5 * l_rng + l_int + l_drop, dict(
            l_rng=l_rng.item(), l_int=l_int.item(), l_drop=l_drop.item())

    # ---- Difix3D+ progressive distillation: rounds of (offline fix -> distill) ----
    work = out_dir / "difix_work"
    gstep = 0
    for rnd in range(args.rounds):
        frac = (rnd + 1) / args.rounds
        novel = build_novel(frac)

        # RGB targets: logged GT (--use_gt_rgb, shift=0) or Difix_ref pseudo-GT.
        fixed_imgs = [None] * len(novel)
        if use_rgb:
            use_gt = bool(getattr(args, "use_gt_rgb", False)) and float(args.shift) <= 1e-6
            if use_gt and cam_obj is not None:
                print(f"[distill] round{rnd}: using logged GT RGB as targets "
                      f"({len(novel)} views)", flush=True)
                for i, (f, lp_n, cc_n, K) in enumerate(novel):
                    if cc_n is None:
                        continue
                    with torch.no_grad():
                        ii, _ = cam_obj.get_image(int(f) * cam_obj.num_cams + 0)
                        gt = ii["pixels"].detach().float()
                        if gt.max() > 1.5:
                            gt = gt / 255.0
                        gt = F.interpolate(
                            gt.permute(2, 0, 1)[None], size=(img_h, img_w),
                            mode="bilinear", align_corners=False,
                        )[0].permute(1, 2, 0).clamp(0, 1)
                    fixed_imgs[i] = gt.to(device)
            else:
                rdir = work / f"round{rnd}"
                pred_dir, ref_dir, fix_dir = rdir / "pred", rdir / "ref", rdir / "fixed"
                for d in (pred_dir, ref_dir, fix_dir):
                    d.mkdir(parents=True, exist_ok=True)
                for i, (f, lp_n, cc_n, K) in enumerate(novel):
                    if cc_n is None:
                        continue
                    with torch.no_grad():
                        rp = renderer.render(cc_n, K, img_h, img_w, int(f))["rgb"]
                        rr = renderer.render(base_cc[f], base_K[f],
                                             img_h, img_w, int(f))["rgb"]
                    _save_png(pred_dir / f"{i:04d}.png", rp)
                    _save_png(ref_dir / f"{i:04d}.png", rr)
                print(f"[distill] round{rnd} (frac={frac:.2f}): difix_ref fixing "
                      f"{len(novel)} novel views ...", flush=True)
                refine_directory(str(pred_dir), str(fix_dir), ref_dir=str(ref_dir),
                                 model_name="nvidia/difix_ref", max_side=img_w,
                                 video=False)
                for i in range(len(novel)):
                    fp = fix_dir / f"{i:04d}.png"
                    if fp.exists():
                        arr = np.asarray(Image.open(fp).convert("RGB"),
                                         dtype=np.float32) / 255.0
                        fixed_imgs[i] = torch.from_numpy(arr).to(device)

        # inner distillation loop against this round's frozen pseudo-GT
        best_loss = float("inf")
        bad_steps = 0
        best_state = None
        for step in range(args.inner_steps):
            i = step % len(novel)
            f, lp_n, cc_n, K = novel[i]
            opt.zero_grad()
            l_lidar, logs = lidar_step(f, lp_n)
            loss = args.w_lidar * l_lidar
            if use_rgb and fixed_imgs[i] is not None:
                rgb = renderer.render_grad(cc_n, K, img_h, img_w, int(f))["rgb"]  # grad
                pseudo_t = fixed_imgs[i]
                if args.no_uncertainty:
                    conf_rgb = torch.ones(rgb.shape[:2], device=device)
                else:
                    conf_rgb = torch.exp(
                        -(rgb.detach() - pseudo_t).abs().mean(-1) / args.conf_tau
                    )
                l_rgb = (F.l1_loss(rgb, pseudo_t, reduction="none").mean(-1)
                         * conf_rgb).mean()
                loss = loss + args.w_rgb * l_rgb
                logs["l_rgb"] = l_rgb.item()
            loss.backward()
            opt.step()
            loss_v = float(loss.detach())
            if args.early_stop_patience > 0:
                if loss_v < best_loss - 1e-5:
                    best_loss = loss_v
                    bad_steps = 0
                    best_state = {
                        "sidecar_int": sidecar_int.detach().clone(),
                        "sidecar_rd": sidecar_rd.detach().clone(),
                        "bg": {a: getattr(bg, a).detach().clone()
                               for a in ("_features_dc", "_features_rest",
                                         "_opacities", "_scales", "_means", "_quats")
                               if hasattr(bg, a)},
                    }
                else:
                    bad_steps += 1
                    if bad_steps >= args.early_stop_patience:
                        print(f"  [early_stop] round{rnd} step{step} "
                              f"patience={args.early_stop_patience}", flush=True)
                        if best_state is not None:
                            sidecar_int.data.copy_(best_state["sidecar_int"])
                            sidecar_rd.data.copy_(best_state["sidecar_rd"])
                            for a, t in best_state["bg"].items():
                                getattr(bg, a).data.copy_(t)
                        break
            if gstep % 25 == 0:
                msg = " ".join(f"{k}={v:.4f}" for k, v in logs.items())
                print(f"  r{rnd} s{step:3d} (g{gstep}) {msg}", flush=True)
            gstep += 1

    # write back sidecar into cl for eval/rendering
    cl.feat_intensity["Background"] = sidecar_int.detach()
    cl.feat_raydrop["Background"] = sidecar_rd.detach()
    torch.save({"head_state": cl.head.state_dict(),
                "cfg": cl.cfg.__dict__,
                "feats": {"Background": {"_intensity": sidecar_int.detach().cpu(),
                                         "_raydrop_logit": sidecar_rd.detach().cpu()}}},
               out_dir / "sidecar_distilled.pt")

    # ---- AFTER render (same fixed view, distilled gaussians + sidecar) ----
    after = {
        "opacity": bg.get_opacity.detach().reshape(-1).cpu().numpy(),
        "scale": bg.get_scaling.detach().mean(-1).cpu().numpy(),
    }
    _aout = cl.render_sweep(vis_lp, vis_f, background_only=True, apply_overrides=False)
    after_range = _aout["range"].detach().cpu().numpy()
    after_pts = cl.sweep_to_points(_aout)
    far = float(cl.cfg.far_plane)
    after_rgb = None
    if vis_cc is not None:
        with torch.no_grad():
            after_rgb = renderer.render(
                vis_cc, vis_K, img_h, img_w, vis_f)["rgb"].clamp(0, 1).cpu().numpy()

    # ---- before/after visualization: renders (top) + attributes (bottom) ----
    fig, ax = plt.subplots(2, 3, figsize=(20, 8))
    ax[0, 0].imshow(np.clip(before_range, 0, far), cmap="turbo", aspect="auto")
    ax[0, 0].set_title("range BEFORE distill"); ax[0, 0].axis("off")
    ax[0, 1].imshow(np.clip(after_range, 0, far), cmap="turbo", aspect="auto")
    ax[0, 1].set_title("range AFTER distill"); ax[0, 1].axis("off")
    dr = np.clip(after_range, 0, far) - np.clip(before_range, 0, far)
    im = ax[0, 2].imshow(dr, cmap="bwr", aspect="auto", vmin=-5, vmax=5)
    ax[0, 2].set_title("Δrange (after-before) [m]"); ax[0, 2].axis("off")
    fig.colorbar(im, ax=ax[0, 2], fraction=0.025)

    for tag, pts, c in [("before", before_pts, "C0"), ("after", after_pts, "C1")]:
        if len(pts):
            ax[1, 0].scatter(pts[:, 1], pts[:, 0], s=0.3, c=c, alpha=0.4, label=tag)
    ax[1, 0].set_xlim(-50, 50); ax[1, 0].set_ylim(-50, 50); ax[1, 0].set_aspect("equal")
    ax[1, 0].legend(); ax[1, 0].set_title(
        f"BEV points before({len(before_pts)})/after({len(after_pts)})")

    ax[1, 1].hist(before["opacity"], bins=60, alpha=0.5, label="before", color="C0")
    ax[1, 1].hist(after["opacity"], bins=60, alpha=0.5, label="after", color="C1")
    ax[1, 1].set_title("Gaussian opacity"); ax[1, 1].legend(); ax[1, 1].set_yscale("log")
    d_op = after["opacity"] - before["opacity"]
    ax[1, 2].hist(d_op, bins=80, color="C2")
    ax[1, 2].set_title(f"Δopacity (mean={d_op.mean():+.5f}, "
                       f"|Δ|max={np.abs(d_op).max():.4f})")
    ax[1, 2].set_yscale("log")

    fig.suptitle(f"Multimodal distillation — {args.scene} "
                 f"(RGB={'on' if use_rgb else 'off'}, LiDAR=on, "
                 f"{args.rounds}x{args.inner_steps} steps)")
    fig.tight_layout(); fig.savefig(out_dir / "gaussian_change.png", dpi=120)
    print(f"[distill] wrote {out_dir/'gaussian_change.png'}", flush=True)

    # ---- RGB before/after for the same fixed novel view ----
    if before_rgb is not None and after_rgb is not None:
        f2, a2 = plt.subplots(1, 3, figsize=(18, 5))
        a2[0].imshow(before_rgb); a2[0].set_title("RGB novel BEFORE"); a2[0].axis("off")
        a2[1].imshow(after_rgb); a2[1].set_title("RGB novel AFTER"); a2[1].axis("off")
        dd = np.abs(after_rgb - before_rgb).mean(-1)
        im2 = a2[2].imshow(dd, cmap="magma")
        a2[2].set_title(f"|Δ RGB| (mean={dd.mean():.4f})"); a2[2].axis("off")
        f2.colorbar(im2, ax=a2[2], fraction=0.04)
        f2.suptitle(f"RGB distillation change — {args.scene}")
        f2.tight_layout(); f2.savefig(out_dir / "rgb_change.png", dpi=120)
        print(f"[distill] wrote {out_dir/'rgb_change.png'}", flush=True)

    torch.save(renderer.trainer.models["Background"].state_dict(),
               out_dir / "background_distilled.pth")
    print(f"[distill] DONE -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
