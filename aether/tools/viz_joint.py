"""Function: Visualize RGB and LiDAR predictions from a joint checkpoint on one frame.
Input/Output: Reads an existing joint checkpoint and LiDAR sidecar; writes one comparison PNG.
Example:
    # Prerequisite: generate the checkpoint and sidecar with joint_phase.py.
    python aether/tools/viz_joint.py \
        --resume_from data/checkpoints/pandaset_outputs/scene_recon_full/scene_095_6cams_with_humans/checkpoint_final.pth \
        --lidar_sidecar outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt \
        --frame_idx 0 --cam_idx 0 --out_png outputs/pandaset095_joint.png
"""
from __future__ import annotations
import os, sys, json, argparse, logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from aether.paths import AETHER_ROOT
sys.path.insert(0, str(AETHER_ROOT))
from aether.paths import DRIVESTUDIO_ROOT  # noqa: F401,E402
# sys.path set by aether.paths

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from aether.render.lidar_head import LidarHead, LidarHeadConfig, build_raster_pts

logger = logging.getLogger("viz_joint")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def _unproject(range_map, alpha, cfg, lidar2world, device):
    n_el, W = range_map.shape
    az = torch.linspace(cfg.min_azimuth + cfg.azimuth_resolution * 0.5,
                        cfg.max_azimuth - cfg.azimuth_resolution * 0.5,
                        W, device=device)
    el_step = (cfg.max_elevation - cfg.min_elevation) / n_el
    el = torch.linspace(cfg.min_elevation + el_step * 0.5,
                        cfg.max_elevation - el_step * 0.5,
                        n_el, device=device)
    az_g, el_g = torch.meshgrid(az, el, indexing="xy")  # [n_el, W]
    az_r = torch.deg2rad(az_g); el_r = torch.deg2rad(el_g)
    dirs = torch.stack([torch.cos(el_r)*torch.cos(az_r),
                        torch.cos(el_r)*torch.sin(az_r),
                        torch.sin(el_r)], dim=-1)
    pts_sensor = dirs * range_map.unsqueeze(-1)
    pts_w = (lidar2world[:3, :3] @ pts_sensor.reshape(-1, 3).T).T + lidar2world[:3, 3]
    keep = (alpha.reshape(-1) > 0.3) & (range_map.reshape(-1) < cfg.far_plane * 0.99)
    return pts_w[keep]


def _project(pts_w, K, c2w, H, W):
    w2c = torch.linalg.inv(c2w)
    pc = (w2c[:3, :3] @ pts_w.T).T + w2c[:3, 3]
    z = pc[:, 2]
    m = z > 0.1
    pc, z = pc[m], z[m]
    px = K[0, 0] * pc[:, 0] / z + K[0, 2]
    py = K[1, 1] * pc[:, 1] / z + K[1, 2]
    v = (px >= 0) & (px < W) & (py >= 0) & (py < H)
    return px[v].cpu().numpy(), py[v].cpu().numpy(), z[v].cpu().numpy()


def main(a):
    device = torch.device("cuda")
    ckpt_dir = Path(a.resume_from).parent
    cfg = OmegaConf.load(ckpt_dir / "config.yaml")

    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=a.resume_from, load_only_model=True)
    for cls in list(trainer.models.keys()):
        if cls in trainer.gaussian_classes: continue
        if cls in ("Sky", "Affine", "CamPose"): continue
        del trainer.models[cls]
    trainer.set_eval()

    # --- LiDAR sidecar
    side = torch.load(a.lidar_sidecar, map_location=device, weights_only=False)
    lcfg = LidarHeadConfig(**side["cfg"])
    head = LidarHead(lcfg).to(device)
    head.load_state_dict({k: v.to(device) for k, v in side["head_state"].items()})
    feats = side["feats"]  # {cls: {_intensity, _raydrop_logit, num_points}}

    # --- pick frame/cam, get image_infos via pixel_source
    px_src = dataset.pixel_source
    n_cams = px_src.num_cams
    image_id = a.frame_idx * n_cams + a.cam_idx
    image_infos, cam_infos = px_src.get_image(image_id)
    image_infos = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in image_infos.items()}
    cam_infos = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam_infos.items()}

    # --- RGB render
    logger.info("Rendering RGB ...")
    with torch.no_grad():
        out = trainer(image_infos=image_infos, camera_infos=cam_infos, novel_view=False)
    rgb_pred = out["rgb"].detach().cpu().numpy().clip(0, 1)
    rgb_gt = image_infos["pixels"].detach().cpu().numpy().clip(0, 1)
    H, W = rgb_gt.shape[:2]
    K = cam_infos["intrinsics"]
    c2w = cam_infos["camera_to_world"]

    # --- LiDAR render: concat across classes
    means_l, quats_l, scales_l, opac_l, int_l, drop_l = [], [], [], [], [], []
    for cls in trainer.gaussian_classes.keys():
        m = trainer.models[cls]
        n_cur = m.num_points
        n_ck = feats[cls]["num_points"]
        if n_cur != n_ck:
            logger.warning(f"{cls}: gaussians {n_cur} vs sidecar {n_ck} -- truncating")
        n = min(n_cur, n_ck)
        means_l.append(m._means[:n].detach())
        quats_l.append(m.get_quats[:n].detach())
        scales_l.append(m.get_scaling[:n].detach())
        opac_l.append(m.get_opacity.reshape(-1)[:n].detach())
        int_l.append(torch.sigmoid(feats[cls]["_intensity"][:n].to(device)).reshape(-1))
        drop_l.append(feats[cls]["_raydrop_logit"][:n].to(device).reshape(-1))
    means = torch.cat(means_l, 0)
    quats = torch.cat(quats_l, 0)
    scales = torch.cat(scales_l, 0)
    opac = torch.cat(opac_l, 0)
    inten = torch.cat(int_l, 0).reshape(-1, 1)
    drop = torch.cat(drop_l, 0).reshape(-1, 1)
    logger.info(f"Total gaussians for LiDAR render: {means.shape[0]}")

    lidar2world = dataset.lidar_source.lidar_to_worlds[a.frame_idx].to(device)
    raster_pts, teb = build_raster_pts(lcfg, device)
    with torch.no_grad():
        lout = head.render(
            means_w=means, quats=quats, scales=scales, opacities=opac,
            intensity=inten, raydrop_logit=drop,
            lidar_pose_w=lidar2world,
            raster_pts=raster_pts, tile_elevation_boundaries=teb,
        )
    pts_pred = _unproject(lout["range"], lout["alpha"], lcfg, lidar2world, device)
    logger.info(f"Pred LiDAR pts: {pts_pred.shape[0]}")

    # GT LiDAR points (world frame) for this frame
    src = dataset.lidar_source
    rays = src.get_lidar_rays(a.frame_idx)
    origin = rays["lidar_origins"][0].to(device)
    dirs_w = rays["lidar_viewdirs"].to(device)
    ranges = rays["lidar_ranges"].squeeze(-1).to(device)
    pts_gt = origin + dirs_w * ranges.unsqueeze(-1)
    logger.info(f"GT LiDAR pts: {pts_gt.shape[0]}")

    # Project
    pxp, pyp, zp = _project(pts_pred, K, c2w, H, W)
    pxg, pyg, zg = _project(pts_gt, K, c2w, H, W)

    # --- Figure
    fig, ax = plt.subplots(2, 2, figsize=(20, 11))
    ax[0,0].imshow(rgb_gt); ax[0,0].set_title(f"GT RGB  frame={a.frame_idx} cam={a.cam_idx}"); ax[0,0].axis("off")
    ax[0,1].imshow(rgb_pred); ax[0,1].set_title("Pred RGB (joint ckpt)"); ax[0,1].axis("off")
    ax[1,0].imshow(rgb_gt)
    s = ax[1,0].scatter(pxg, pyg, s=2, c=zg, cmap="turbo", vmin=2, vmax=60)
    ax[1,0].set_title(f"GT LiDAR overlay ({len(pxg)} pts)"); ax[1,0].axis("off")
    plt.colorbar(s, ax=ax[1,0], fraction=0.025)
    ax[1,1].imshow(rgb_gt)
    s = ax[1,1].scatter(pxp, pyp, s=2, c=zp, cmap="turbo", vmin=2, vmax=60)
    ax[1,1].set_title(f"Pred LiDAR overlay ({len(pxp)} pts) -- LidarHead all classes"); ax[1,1].axis("off")
    plt.colorbar(s, ax=ax[1,1], fraction=0.025)
    plt.tight_layout()
    Path(a.out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(a.out_png, dpi=110); plt.close()
    logger.info(f"Saved {a.out_png}")

    # Also save bare LiDAR range map + intensity map as PNGs for diagnostic
    diag_dir = Path(a.out_png).parent / "lidar_maps"
    diag_dir.mkdir(exist_ok=True)
    rng = lout["range"].cpu().numpy()
    inten_m = lout["intensity"].cpu().numpy()
    rdrop = torch.sigmoid(lout["raydrop_logit"]).cpu().numpy()
    for name, arr, kw in [
        (f"range_f{a.frame_idx}.png", rng, dict(cmap="turbo", vmin=0, vmax=60)),
        (f"intensity_f{a.frame_idx}.png", inten_m, dict(cmap="gray", vmin=0, vmax=1)),
        (f"raydrop_f{a.frame_idx}.png", rdrop, dict(cmap="gray", vmin=0, vmax=1)),
    ]:
        plt.figure(figsize=(16, 4)); plt.imshow(arr, aspect="auto", **kw); plt.colorbar(); plt.title(name)
        plt.savefig(diag_dir / name, dpi=100); plt.close()

    stats = dict(n_gt_pts=int(pts_gt.shape[0]), n_pred_pts=int(pts_pred.shape[0]),
                 n_gt_overlay=int(len(pxg)), n_pred_overlay=int(len(pxp)),
                 mean_intensity=float(inten.mean()), raydrop_pos_frac=float((drop > 0).float().mean()),
                 total_gaussians=int(means.shape[0]))
    Path(a.out_png).with_suffix(".json").write_text(json.dumps(stats, indent=2))
    logger.info(f"Stats: {stats}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume_from", required=True)
    p.add_argument("--lidar_sidecar", required=True)
    p.add_argument("--frame_idx", type=int, default=100)
    p.add_argument("--cam_idx", type=int, default=0)
    p.add_argument("--out_png", required=True)
    main(p.parse_args())
