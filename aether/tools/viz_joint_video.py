"""Function: Render joint-checkpoint RGB, LiDAR, BEV, and ground-truth comparison videos.
Input/Output: Reads an existing joint checkpoint and LiDAR sidecar; writes two videos in the output directory.
Example:
    # Prerequisite: generate the checkpoint and sidecar with joint_phase.py.
    python aether/tools/viz_joint_video.py \
        --resume_from data/checkpoints/pandaset_outputs/scene_recon_full/scene_095_6cams_with_humans/checkpoint_final.pth \
        --lidar_sidecar outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt \
        --out_dir outputs/pandaset095_joint_video --scene_tag "PandaSet 095" --fps 10
"""
from __future__ import annotations
import sys, argparse, logging
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from omegaconf import OmegaConf

from aether.paths import AETHER_ROOT
sys.path.insert(0, str(AETHER_ROOT))
from aether.paths import DRIVESTUDIO_ROOT  # noqa: F401,E402
# sys.path set by aether.paths
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str
from aether.render.lidar_head import LidarHead, LidarHeadConfig, build_raster_pts

logger = logging.getLogger("viz_video")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

# Per-dataset cam layout. Keys are substrings matched against
# dataset.pixel_source.camera_data[i].cam_name (case-insensitive).
# Each entry maps (display_label, top_row_left_to_right_order_score).
LAYOUTS = {
    "nuscenes": {
        # cam name substring -> (label, sort_key for left->right ordering)
        "CAM_FRONT_LEFT": ("FL", 0), "CAM_FRONT": ("F", 1), "CAM_FRONT_RIGHT": ("FR", 2),
        "CAM_BACK_LEFT": ("BL", 3), "CAM_BACK": ("B", 4), "CAM_BACK_RIGHT": ("BR", 5),
    },
    "pandaset": {
        "front_left_camera": ("FL", 0), "front_camera": ("F", 1), "front_right_camera": ("FR", 2),
        "left_camera": ("L", 3),       "back_camera": ("B", 4),  "right_camera": ("R", 5),
    },
}


def _resolve_cam_layout(camera_data):
    """Return (cam_order_indices, labels) for a 6-cam dataset.

    cam_order_indices: list of dataset cam indices to render, left->right.
    labels: corresponding display labels.
    """
    names = {i: camera_data[i].cam_name for i in sorted(camera_data.keys())}
    # pick which layout based on first cam name
    sample = next(iter(names.values()))
    chosen = None
    for tag, mapping in LAYOUTS.items():
        if any(key.lower() in sample.lower() or sample.lower() in key.lower() for key in mapping):
            chosen = mapping
            break
    if chosen is None:
        # fallback: keep natural order, use cam names
        order = sorted(names.keys())
        return order, [names[i] for i in order]
    pairs = []
    for ci, cname in names.items():
        for key, (label, score) in chosen.items():
            if key.lower() == cname.lower():
                pairs.append((score, ci, label))
                break
    pairs.sort()
    return [p[1] for p in pairs], [p[2] for p in pairs]


def _unproject(range_map, alpha, intensity_map, cfg, lidar2world, device):
    n_el, W = range_map.shape
    az = torch.linspace(cfg.min_azimuth + cfg.azimuth_resolution * 0.5,
                        cfg.max_azimuth - cfg.azimuth_resolution * 0.5,
                        W, device=device)
    el_step = (cfg.max_elevation - cfg.min_elevation) / n_el
    el = torch.linspace(cfg.min_elevation + el_step * 0.5,
                        cfg.max_elevation - el_step * 0.5,
                        n_el, device=device)
    az_g, el_g = torch.meshgrid(az, el, indexing="xy")
    az_r = torch.deg2rad(az_g); el_r = torch.deg2rad(el_g)
    dirs = torch.stack([torch.cos(el_r)*torch.cos(az_r),
                        torch.cos(el_r)*torch.sin(az_r),
                        torch.sin(el_r)], dim=-1)
    pts_sensor = dirs * range_map.unsqueeze(-1)
    pts_w = (lidar2world[:3, :3] @ pts_sensor.reshape(-1, 3).T).T + lidar2world[:3, 3]
    keep = (alpha.reshape(-1) > 0.3) & (range_map.reshape(-1) < cfg.far_plane * 0.99)
    return pts_w[keep], pts_sensor.reshape(-1, 3)[keep], intensity_map.reshape(-1)[keep]


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
    if a.downscale != 1.0:
        cfg.data.pixel_source.load_size = [int(s / a.downscale) for s in cfg.data.pixel_source.load_size] if False else cfg.data.pixel_source.load_size
        # we'll just downscale the rendered images, dataset stays at native

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

    # LiDAR sidecar
    side = torch.load(a.lidar_sidecar, map_location=device, weights_only=False)
    lcfg = LidarHeadConfig(**side["cfg"])
    head = LidarHead(lcfg).to(device); head.eval()
    head.load_state_dict({k: v.to(device) for k, v in side["head_state"].items()})
    feats = side["feats"]

    # Pre-concat per-class LiDAR-side gaussians once (static), but means/quats/scales come from
    # trainer.models each frame to capture any dynamic state. We rebuild every frame for safety.

    raster_pts, teb = build_raster_pts(lcfg, device)
    src = dataset.lidar_source
    px_src = dataset.pixel_source
    n_cams = px_src.num_cams
    cam_order, cam_labels = _resolve_cam_layout(px_src.camera_data)
    logger.info(f"Cam layout: {list(zip(cam_order, cam_labels))}")
    n_frames = a.end_frame if a.end_frame > 0 else px_src.num_frames
    start = a.start_frame
    end = min(n_frames, px_src.num_frames)

    out_dir = Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"; frames_dir.mkdir(exist_ok=True)
    frames_cmp_dir = out_dir / "frames_cmp"; frames_cmp_dir.mkdir(exist_ok=True)
    logger.info(f"Rendering frames {start}..{end} into {frames_dir}")

    video_writer = imageio.get_writer(out_dir / a.out_name, fps=a.fps, codec="libx264",
                                       quality=8, macro_block_size=1)
    # Second writer: GT-vs-Pred LiDAR overlay comparison (top=GT, bottom=Pred).
    cmp_writer = imageio.get_writer(out_dir / a.cmp_name, fps=a.fps, codec="libx264",
                                    quality=8, macro_block_size=1)

    for fi in range(start, end):
        # Per-frame: concat geometry + LiDAR feats
        means_l, quats_l, scales_l, opac_l, int_l, drop_l = [], [], [], [], [], []
        with torch.no_grad():
            for cls in trainer.gaussian_classes.keys():
                m = trainer.models[cls]
                n_cur = m.num_points; n_ck = feats[cls]["num_points"]
                n = min(n_cur, n_ck)
                means_l.append(m._means[:n].detach())
                quats_l.append(m.get_quats[:n].detach())
                scales_l.append(m.get_scaling[:n].detach())
                opac_l.append(m.get_opacity.reshape(-1)[:n].detach())
                int_l.append(torch.sigmoid(feats[cls]["_intensity"][:n].to(device)).reshape(-1))
                drop_l.append(feats[cls]["_raydrop_logit"][:n].to(device).reshape(-1))
            means = torch.cat(means_l, 0); quats = torch.cat(quats_l, 0)
            scales = torch.cat(scales_l, 0); opac = torch.cat(opac_l, 0)
            inten = torch.cat(int_l, 0).reshape(-1, 1); drop = torch.cat(drop_l, 0).reshape(-1, 1)

            lidar2world = src.lidar_to_worlds[fi].to(device)
            lout = head.render(
                means_w=means, quats=quats, scales=scales, opacities=opac,
                intensity=inten, raydrop_logit=drop,
                lidar_pose_w=lidar2world,
                raster_pts=raster_pts, tile_elevation_boundaries=teb,
            )
        pts_pred_w, pts_pred_sensor, pts_pred_int = _unproject(
            lout["range"], lout["alpha"], lout["intensity"], lcfg, lidar2world, device)

        # GT LiDAR
        rays = src.get_lidar_rays(fi)
        origin = rays["lidar_origins"][0].to(device)
        dirs_w = rays["lidar_viewdirs"].to(device)
        ranges = rays["lidar_ranges"].squeeze(-1).to(device)
        pts_gt_w = origin + dirs_w * ranges.unsqueeze(-1)

        # Render 6 cams
        rgbs, lidars_overlay, gt_overlay = [], [], []
        for ci in cam_order:
            image_id = fi * n_cams + ci
            image_infos, cam_infos = px_src.get_image(image_id)
            image_infos = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in image_infos.items()}
            cam_infos = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cam_infos.items()}
            with torch.no_grad():
                out = trainer(image_infos=image_infos, camera_infos=cam_infos, novel_view=False)
            rgb_pred = out["rgb"].detach().cpu().numpy().clip(0, 1)
            H, W = rgb_pred.shape[:2]
            K = cam_infos["intrinsics"]; c2w = cam_infos["camera_to_world"]
            pxp, pyp, zp = _project(pts_pred_w, K, c2w, H, W)
            pxg, pyg, zg = _project(pts_gt_w, K, c2w, H, W)
            rgbs.append(rgb_pred)
            lidars_overlay.append((rgb_pred, pxp, pyp, zp, H, W))
            gt_overlay.append((rgb_pred, pxg, pyg, zg, H, W))

        # Build figure
        fig = plt.figure(figsize=(24, 14), facecolor="black")
        gs = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.6], hspace=0.04, wspace=0.02)
        for col in range(len(cam_order)):
            ax = fig.add_subplot(gs[0, col]); ax.imshow(rgbs[col]); ax.axis("off")
            ax.set_title(cam_labels[col], color="white", fontsize=11)
            rgb, pxp, pyp, zp, H, W = lidars_overlay[col]
            ax = fig.add_subplot(gs[1, col]); ax.imshow(rgb)
            if len(pxp):
                ax.scatter(pxp, pyp, s=2.0, c=zp, cmap="turbo",
                           vmin=2, vmax=60, alpha=0.55,
                           marker="o", linewidths=0,
                           rasterized=True)
            ax.axis("off")

        # BEV (ego frame): predicted-only point cloud, top-down view, height-colored
        ax = fig.add_subplot(gs[2, :])
        ax.set_facecolor("black")
        ps = pts_pred_sensor.cpu().numpy()
        R = a.bev_range
        mp = (np.abs(ps[:, 0]) < R) & (np.abs(ps[:, 1]) < R)
        ax.scatter(ps[mp, 0], ps[mp, 1], s=0.7, c=ps[mp, 2],
                   cmap="turbo", vmin=-3, vmax=5, alpha=0.9, edgecolors="none")
        # ego car marker
        ax.add_patch(plt.Rectangle((-2.5, -1), 5, 2, color="white", alpha=0.7))
        ax.plot(0, 0, "wx", markersize=8)
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_aspect("equal")
        ax.set_title(
            f"BEV  Predicted LiDAR point cloud (ego frame, x: forward, y: left)   "
            f"frame {fi}/{end-1}   {mp.sum()} pts   color = height [-3, +5] m",
            color="white", fontsize=12)
        ax.tick_params(colors="white"); ax.spines[:].set_color("white")

        fig.suptitle(f"AETHER joint RGB+LiDAR -- {a.scene_tag} -- frame {fi}",
                     color="white", fontsize=14, y=0.995)
        png_path = frames_dir / f"frame_{fi:04d}.png"
        plt.savefig(png_path, dpi=80, facecolor="black", bbox_inches="tight")
        plt.close(fig)

        img = imageio.imread(png_path)
        video_writer.append_data(img)

        # ---- Comparison figure: GT LiDAR (top) vs Pred LiDAR (bottom) ----
        # 6 camera-overlay columns per row, same depth colormap so the colors
        # are directly comparable between GT and prediction.
        fig2 = plt.figure(figsize=(24, 9), facecolor="black")
        gs2 = fig2.add_gridspec(2, 6, hspace=0.07, wspace=0.02)
        n_gt_proj = n_pred_proj = 0
        for col in range(len(cam_order)):
            # top row: GT overlay
            rgb, pxg, pyg, zg, H, W = gt_overlay[col]
            ax = fig2.add_subplot(gs2[0, col]); ax.imshow(rgb); ax.axis("off")
            if len(pxg):
                ax.scatter(pxg, pyg, s=2.0, c=zg, cmap="turbo", vmin=2, vmax=60,
                           alpha=0.55, marker="o", linewidths=0, rasterized=True)
                n_gt_proj += len(pxg)
            ax.set_title(f"GT  {cam_labels[col]}", color="white", fontsize=11)
            # bottom row: Pred overlay
            rgb, pxp, pyp, zp, H, W = lidars_overlay[col]
            ax = fig2.add_subplot(gs2[1, col]); ax.imshow(rgb); ax.axis("off")
            if len(pxp):
                ax.scatter(pxp, pyp, s=2.0, c=zp, cmap="turbo", vmin=2, vmax=60,
                           alpha=0.55, marker="o", linewidths=0, rasterized=True)
                n_pred_proj += len(pxp)
            ax.set_title(f"Pred  {cam_labels[col]}", color="white", fontsize=11)
        fig2.suptitle(
            f"LiDAR overlay  GT (top) vs Pred (bottom)  color = depth [2, 60] m  "
            f"-- {a.scene_tag} -- frame {fi}",
            color="white", fontsize=14, y=0.99)
        png2_path = frames_cmp_dir / f"frame_{fi:04d}.png"
        plt.savefig(png2_path, dpi=80, facecolor="black", bbox_inches="tight")
        plt.close(fig2)
        cmp_writer.append_data(imageio.imread(png2_path))

        if (fi - start) % 10 == 0:
            logger.info(f"  frame {fi}/{end-1}  pred_pts={ps.shape[0]}  "
                        f"proj GT={n_gt_proj} Pred={n_pred_proj}")

    video_writer.close()
    cmp_writer.close()
    logger.info(f"Main video saved:       {out_dir / a.out_name}")
    logger.info(f"GT-vs-Pred video saved: {out_dir / a.cmp_name}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--resume_from", required=True)
    p.add_argument("--lidar_sidecar", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--out_name", default="dualmodal_video.mp4")
    p.add_argument("--cmp_name", default="gt_vs_pred_lidar.mp4",
                   help="Filename for the GT-vs-Pred LiDAR overlay comparison video.")
    p.add_argument("--scene_tag", default="nuScenes scene 0383",
                   help="Text shown in video titles, e.g. 'nuScenes scene 0383'.")
    p.add_argument("--start_frame", type=int, default=0)
    p.add_argument("--end_frame", type=int, default=-1)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--bev_range", type=float, default=50.0)
    p.add_argument("--downscale", type=float, default=1.0)
    main(p.parse_args())
