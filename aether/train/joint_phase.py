"""Function: Jointly fine-tune RGB Gaussians and LiDAR features from a Phase-A checkpoint.
Input/Output: Reads a DriveStudio config, RGB checkpoint, and processed data; writes joint checkpoints and LiDAR sidecars.
Example:
    # Prerequisite: train the referenced RGB checkpoint first.
    python aether/train/joint_phase.py \
        --config_file third_party/drivestudio/configs/omnire.yaml \
        --resume_from data/checkpoints/pandaset_outputs/scene_recon_full/scene_095_6cams_with_humans/checkpoint_final.pth \
        --output_root outputs/dualmodal --project aether_joint/pandaset \
        --run_name scene095_joint --num_iters 10000 --freeze_geometry \
        dataset=pandaset/6cams data.data_root=data/pandaset/processed data.scene_idx=95
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

# Bootstrap repo root before importing ``aether`` (script may be launched
# without PYTHONPATH=AETHER).
_AETHER_ROOT = Path(__file__).resolve().parents[2]
if str(_AETHER_ROOT) not in sys.path:
    sys.path.insert(0, str(_AETHER_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from aether.paths import AETHER_ROOT, DRIVESTUDIO_ROOT, resolve_data_root
sys.path.insert(0, str(DRIVESTUDIO_ROOT))
sys.path.insert(0, str(AETHER_ROOT))

from datasets.driving_dataset import DrivingDataset  # noqa: E402
from utils.misc import import_str  # noqa: E402
from utils.logging import MetricLogger, setup_logging  # noqa: E402

from aether.render.lidar_head import (  # noqa: E402
    LidarHead, LidarHeadConfig, GaussianLidarFeatures, build_raster_pts,
)


logger = logging.getLogger()


def set_seeds(seed: int) -> None:
    """Seed every RNG used by the joint/frozen training stages."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------- #
# GT spherical grid (azimuth x elevation) used to compare against the head's
# rendered range / intensity / raydrop images.
# ---------------------------------------------------------------------- #
def gt_to_spherical_grid(
    origin_w: torch.Tensor,         # [3]
    dirs_w: torch.Tensor,           # [P,3]
    ranges: torch.Tensor,           # [P]
    intensities: torch.Tensor,      # [P] in [0,1]
    lidar2world: torch.Tensor,      # [4,4]
    cfg: LidarHeadConfig,
):
    device = dirs_w.device
    # bring directions back into sensor frame for binning
    R_inv = torch.linalg.inv(lidar2world)[:3, :3]
    dirs_l = (R_inv @ dirs_w.T).T
    az = torch.rad2deg(torch.atan2(dirs_l[:, 1], dirs_l[:, 0]))
    el = torch.rad2deg(torch.atan2(dirs_l[:, 2],
                                   torch.linalg.norm(dirs_l[:, :2], dim=1)))

    W = int(round((cfg.max_azimuth - cfg.min_azimuth) / cfg.azimuth_resolution))
    n_el = cfg.n_elevation_channels
    # Do not clamp returns outside the configured sensor field of view into a
    # boundary row/column.  Clamping creates artificial dense bands (notably at
    # max_elevation) and changes every hit-based metric.
    valid = (
        torch.isfinite(az)
        & torch.isfinite(el)
        & torch.isfinite(ranges)
        & (ranges > cfg.near_plane)
        & (ranges < cfg.far_plane)
        & (az >= cfg.min_azimuth)
        & (az < cfg.max_azimuth)
        & (el >= cfg.min_elevation)
        & (el < cfg.max_elevation)
    )
    az = az[valid]
    el = el[valid]
    ranges = ranges[valid]
    intensities = intensities[valid]
    j = ((az - cfg.min_azimuth) / cfg.azimuth_resolution).long()
    i = ((el - cfg.min_elevation) /
         (cfg.max_elevation - cfg.min_elevation) * n_el).long()

    range_map = torch.full((n_el, W), float("inf"), device=device)
    intensity_sum = torch.zeros((n_el, W), device=device)
    flat_idx = i * W + j

    range_map.view(-1).scatter_reduce_(0, flat_idx, ranges, reduce="amin",
                                       include_self=True)
    intensity_sum.view(-1).scatter_add_(0, flat_idx, intensities)
    cnt = torch.zeros_like(intensity_sum.view(-1))
    cnt.scatter_add_(0, flat_idx, torch.ones_like(ranges))
    intensity_map = intensity_sum / cnt.view(n_el, W).clamp(min=1)

    hit = (range_map.isfinite() & (range_map < cfg.far_plane)).float()
    range_map = torch.where(hit.bool(), range_map,
                            torch.full_like(range_map, cfg.far_plane))
    return range_map, intensity_map, hit


# ---------------------------------------------------------------------- #
# Pre-cache per-frame LiDAR GT (origins + dirs + ranges + intensities) so we
# can pick a random frame each step without re-reading bin files.
# ---------------------------------------------------------------------- #
def cache_lidar_gt(dataset: DrivingDataset, lidar_bin_dir: Path, device: str = "cuda"):
    """For each frame index, build the world-frame ray dict plus intensity
    (read directly from the raw .bin so we don't depend on `colors`)."""
    from aether.refine.pair_utils import read_bin_intensity

    src = dataset.lidar_source
    n_frames = src.lidar_to_worlds.shape[0]
    cached: List[Dict[str, torch.Tensor]] = []
    for t in range(n_frames):
        rays = src.get_lidar_rays(t)
        origin = rays["lidar_origins"][0].detach().to(device)  # all rays share one origin
        dirs_w = rays["lidar_viewdirs"].detach().to(device)
        ranges = rays["lidar_ranges"].squeeze(-1).detach().to(device)

        # raw bin -> intensity (sensor-frame order). PandaSet bins are Nx5
        # (x,y,z,i,laser_id); nuScenes bins are Nx4 (x,y,z,i).
        bin_path = lidar_bin_dir / f"{t:03d}.bin"
        only_360 = bool(
            getattr(getattr(src, "data_cfg", None), "only_use_360_lidar", False)
        )
        inten = read_bin_intensity(
            bin_path, ranges.numel(), device, only_use_360_lidar=only_360
        )

        cached.append(dict(
            origin_w=origin,
            dirs_w=dirs_w,
            ranges=ranges,
            intensity=inten,
            lidar2world=src.lidar_to_worlds[t].detach().to(device),
        ))
    return cached


# ---------------------------------------------------------------------- #
# Setup helper modelled on drivestudio/tools/train.py.
# ---------------------------------------------------------------------- #
def setup(args):
    cfg = OmegaConf.load(args.config_file)
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(
            os.path.join(str(DRIVESTUDIO_ROOT), "configs", "datasets", f"{dataset_type}.yaml")
        )
        cfg = OmegaConf.merge(cfg, dataset_cfg)
    cfg = OmegaConf.merge(cfg, args_from_cli)
    cfg.data.data_root = str(resolve_data_root(cfg.data.data_root))

    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for folder in ["images", "videos", "metrics", "configs_bk"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)

    setup_logging(output=log_dir, level=logging.INFO,
                  time_string=time.strftime("%Y-%m-%d_%H-%M-%S"))
    logger.info(f"Joint Phase config:\n{OmegaConf.to_yaml(cfg)}")
    OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
    return cfg


def disable_densification(trainer):
    """Push every gaussian class past its `stop_split_at` threshold so the
    densify / cull branches are skipped on every postprocess call. Also raise
    warmup so refinement_after early-returns."""
    for name in trainer.gaussian_classes.keys():
        m = trainer.models[name]
        if hasattr(m, "ctrl_cfg"):
            m.ctrl_cfg.stop_split_at = 0
            m.ctrl_cfg.warmup_steps = 10 ** 9
            m.ctrl_cfg.reset_alpha_interval = 10 ** 9


# ---------------------------------------------------------------------- #
# Per-class gaussian gather: returns (means, quats, scales, opacities) in
# WORLD frame for the LiDAR sweep. For static Background that's just
# `_means / get_quats / get_scaling / get_opacity`. For dynamic classes the
# trainer applies per-frame instance poses inside its scene_graph forward;
# for the LiDAR sweep we use the *current static* values (good enough as a
# rough approximation; high-quality dynamic LiDAR is out of MVP scope).
# ---------------------------------------------------------------------- #
def gather_class_geometry(model) -> Dict[str, torch.Tensor]:
    return dict(
        means=model._means,                       # leaf Parameter
        quats=model.get_quats,                    # function -> Tensor (autograd ok)
        scales=model.get_scaling,                 # ditto
        opacities=model.get_opacity.reshape(-1),  # ditto
    )


# ---------------------------------------------------------------------- #
# Main
# ---------------------------------------------------------------------- #
def main(args):
    set_seeds(args.seed)
    cfg = setup(args)
    cfg.seed = args.seed
    OmegaConf.save(cfg, os.path.join(cfg.log_dir, "config.yaml"))
    logger.info(f"Random seed: {args.seed}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    assert args.resume_from is not None, "joint_phase requires --resume_from"
    trainer.resume_from_checkpoint(ckpt_path=args.resume_from, load_only_model=True)
    logger.info(f"Resumed from {args.resume_from} at step {trainer.step}")

    # Drop any class that wasn't in the checkpoint (e.g. SMPLNodes when the
    # scene has no pedestrians). resume_from_checkpoint already pops them from
    # gaussian_classes; here we also remove them from self.models so the
    # optimizer setup doesn't trip over their uninitialised tensors.
    for cls in list(trainer.models.keys()):
        if cls in trainer.gaussian_classes:
            continue
        if cls in ("Sky", "Affine", "CamPose"):
            continue  # auxiliary models without gaussian state
        logger.info(f"Dropping empty model class {cls}")
        del trainer.models[cls]

    # Some resumed runs (notably PandaSet scenes with SMPL) may not carry
    # transient KNN buffers such as `nn_ind` required by SMPL regularizers.
    # In that case, drop SMPLNodes for the joint LiDAR phase to keep training
    # stable and avoid crashing in compute_reg_loss.
    if "SMPLNodes" in trainer.models and not hasattr(trainer.models["SMPLNodes"], "nn_ind"):
        logger.warning("SMPLNodes has no nn_ind after resume; dropping SMPLNodes for joint phase")
        del trainer.models["SMPLNodes"]
        if "SMPLNodes" in trainer.gaussian_classes:
            trainer.gaussian_classes.pop("SMPLNodes")

    # build LiDAR head + per-class features sized to current point counts
    head_cfg = LidarHeadConfig()
    lidar_head = LidarHead(head_cfg).to(device)
    feats: Dict[str, GaussianLidarFeatures] = {}
    for cls in trainer.gaussian_classes.keys():
        n = trainer.models[cls].num_points
        feats[cls] = GaussianLidarFeatures(n, device=device)
        logger.info(f"Class {cls}: created LiDAR feats for {n} gaussians")

    # disable densification BEFORE initialize_optimizer (lr schedules look at num_iters)
    disable_densification(trainer)

    # override num_iters for the joint phase
    trainer.num_iters = args.num_iters

    trainer.initialize_optimizer()

    # plug LiDAR params into the same optimizer so logging/save and lr behave
    disable_lidar_attributes = bool(args.disable_lidar_attributes)
    extra_params = list(lidar_head.parameters())
    if not disable_lidar_attributes:
        for cls, f in feats.items():
            extra_params += list(f.parameters())
    else:
        for f in feats.values():
            for parameter in f.parameters():
                parameter.requires_grad_(False)
        logger.info(
            "[joint] --disable_lidar_attributes ON: per-Gaussian intensity and "
            "ray-drop attributes remain fixed at initialization and receive no loss."
        )
    if extra_params:
        trainer.optimizer.add_param_group({
            "params": extra_params, "lr": args.lidar_lr,
            "name": "lidar_feats_head",
        })

    freeze_geometry = bool(args.freeze_geometry)
    if freeze_geometry:
        # Drop non-LiDAR param groups so Adam never touches RGB geometry.
        keep = []
        for group in trainer.optimizer.param_groups:
            if group.get("name") == "lidar_feats_head":
                keep.append(group)
            else:
                for p in group["params"]:
                    p.requires_grad_(False)
        trainer.optimizer.param_groups = keep
        for m in trainer.models.values():
            if hasattr(m, "parameters"):
                for p in m.parameters():
                    p.requires_grad_(False)
        logger.info(
            "[joint] --freeze_geometry ON: RGB gaussians frozen; "
            "only intensity/raydrop (+ LidarHead) train. "
            "Closed-loop RGB should keep using --resume_from / original ckpt."
        )

    # cache LiDAR GT
    lidar_bin_dir = Path(cfg.data.data_root) / str(cfg.data.scene_idx) / "lidar"
    logger.info(f"Caching LiDAR GT from {lidar_bin_dir}")
    lidar_cache = cache_lidar_gt(dataset, lidar_bin_dir, device=str(device))
    train_lidar_frames = np.asarray(dataset.train_timesteps, dtype=np.int64)
    if train_lidar_frames.size == 0:
        raise RuntimeError("No reconstruction-train LiDAR frames are available")
    heldout_lidar_frames = sorted(
        set(range(len(lidar_cache))) - set(train_lidar_frames.tolist())
    )
    logger.info(
        f"Cached {len(lidar_cache)} LiDAR sweeps; training only on "
        f"{train_lidar_frames.tolist()}; excluded={heldout_lidar_frames}"
    )

    raster_pts, teb = build_raster_pts(head_cfg, device)
    lidar_weights = dict(
        depth=args.w_depth,
        intensity=0.0 if disable_lidar_attributes else args.w_intensity,
        raydrop=0.0 if disable_lidar_attributes else args.w_raydrop,
        los=args.w_los,
    )
    logger.info(f"LiDAR loss weights: {lidar_weights}, lidar_lr={args.lidar_lr}")

    metric_logger = MetricLogger(
        delimiter="  ",
        output_file=os.path.join(cfg.log_dir, "metrics.json"),
    )

    start_step = 0
    all_iters = np.arange(start_step, args.num_iters + 1)
    source_rgb_ckpt = str(Path(args.resume_from).resolve())

    for step in metric_logger.log_every(all_iters, args.print_freq):
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad()

        loss_dict = {}
        outputs = None
        image_infos = None

        # RGB is absent for a separately optimized LiDAR representation and
        # for the frozen-sidecar stage.
        if not freeze_geometry and not args.lidar_only_geometry:
            ds_factor = trainer._get_downscale_factor()
            image_infos, cam_infos = dataset.train_image_set.next(ds_factor)
            for k, v in image_infos.items():
                if isinstance(v, torch.Tensor):
                    image_infos[k] = v.cuda(non_blocking=True)
            for k, v in cam_infos.items():
                if isinstance(v, torch.Tensor):
                    cam_infos[k] = v.cuda(non_blocking=True)

            outputs = trainer(image_infos, cam_infos)
            trainer.update_visibility_filter()

            loss_dict = trainer.compute_losses(
                outputs=outputs, image_infos=image_infos, cam_infos=cam_infos,
            )

        # ------------- LiDAR step -------------
        if step % args.lidar_every == 0:
            f = int(np.random.choice(train_lidar_frames))
            gt = lidar_cache[f]
            gt_range, gt_int, hit = gt_to_spherical_grid(
                gt["origin_w"], gt["dirs_w"], gt["ranges"], gt["intensity"],
                gt["lidar2world"], head_cfg,
            )

            # concat current gaussian state across all classes (autograd live
            # only when geometry is not frozen)
            means_l, quats_l, scales_l, opac_l = [], [], [], []
            int_l, rd_l = [], []
            for cls in trainer.gaussian_classes.keys():
                g = gather_class_geometry(trainer.models[cls])
                if freeze_geometry:
                    means_l.append(g["means"].detach())
                    quats_l.append(g["quats"].detach())
                    scales_l.append(g["scales"].detach())
                    opac_l.append(g["opacities"].detach())
                else:
                    means_l.append(g["means"])
                    quats_l.append(g["quats"])
                    scales_l.append(g["scales"])
                    opac_l.append(g["opacities"])
                # Logits: LidarHead.render applies sigmoid after rasterization.
                int_l.append(feats[cls]._intensity)
                rd_l.append(feats[cls].raydrop_logit)

            means_all = torch.cat(means_l, dim=0)
            quats_all = torch.cat(quats_l, dim=0)
            scales_all = torch.cat(scales_l, dim=0)
            opac_all = torch.cat(opac_l, dim=0)
            int_all = torch.cat(int_l, dim=0)
            rd_all = torch.cat(rd_l, dim=0)

            lidar_out = lidar_head.render(
                means_w=means_all, quats=quats_all, scales=scales_all,
                opacities=opac_all, intensity=int_all, raydrop_logit=rd_all,
                lidar_pose_w=gt["lidar2world"],
                raster_pts=raster_pts, tile_elevation_boundaries=teb,
            )
            l_losses = lidar_head.losses(
                lidar_out, gt_range, gt_int, hit, weights=lidar_weights,
            )
            loss_dict.update(l_losses)

        if not loss_dict:
            continue

        # ------------- check / backward -------------
        for k, v in loss_dict.items():
            if torch.isnan(v).any() or torch.isinf(v).any():
                raise ValueError(f"bad loss {k} at step {step}: {v}")

        trainer.backward(loss_dict)
        # NOTE: do NOT call postprocess_per_train_step -> avoids densification
        # so LiDAR feature tensors stay aligned with gaussian indices.

        # ------------- logging -------------
        if outputs is not None and image_infos is not None and not freeze_geometry:
            with torch.no_grad():
                metric_dict = trainer.compute_metrics(
                    outputs=outputs, image_infos=image_infos,
                )
            metric_logger.update(
                **{"train_metrics/" + k: v.item() for k, v in metric_dict.items()}
            )
        metric_logger.update(**{"losses/" + k: v.item() for k, v in loss_dict.items()})

        # ------------- save -------------
        if step > 0 and (step % args.save_freq == 0 or step == args.num_iters):
            ckpt_dir = Path(cfg.log_dir) / "ckpts"
            ckpt_dir.mkdir(exist_ok=True)
            # Only overwrite RGB gaussian ckpt when geometry is being trained.
            if not freeze_geometry:
                trainer.save_checkpoint(
                    log_dir=cfg.log_dir, save_only_model=True,
                    is_final=step == args.num_iters,
                )
            sidecar = dict(
                cfg=asdict(head_cfg),
                head_state={k: v.cpu() for k, v in lidar_head.state_dict().items()},
                feats={cls: {
                    "_intensity": feats[cls]._intensity.detach().cpu(),
                    "_raydrop_logit": feats[cls]._raydrop_logit.detach().cpu(),
                    "num_points": feats[cls].n,
                } for cls in feats},
                step=step,
                freeze_geometry=freeze_geometry,
                lidar_only_geometry=bool(args.lidar_only_geometry),
                lidar_attributes_enabled=not disable_lidar_attributes,
                source_rgb_ckpt=source_rgb_ckpt,
            )
            sidecar_name = (
                f"lidar_sidecar_frozen_step{step}.pt"
                if freeze_geometry else f"lidar_sidecar_step{step}.pt"
            )
            sidecar_path = Path(cfg.log_dir) / sidecar_name
            torch.save(sidecar, sidecar_path)
            logger.info(
                f"Saved lidar sidecar {sidecar_path}"
                + ("" if freeze_geometry else " (+ drivestudio ckpt)")
            )
            # Convenience alias for closed-loop auto-resolve
            if freeze_geometry and step == args.num_iters:
                alias = Path(cfg.log_dir) / "lidar_sidecar_frozen.pt"
                torch.save(sidecar, alias)
                logger.info(f"Also wrote alias {alias}")

    logger.info("Joint phase DONE")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config_file", required=True)
    p.add_argument("--resume_from", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--project", default="joint")
    p.add_argument("--run_name", default="run")
    p.add_argument("--num_iters", type=int, default=10000)
    p.add_argument("--save_freq", type=int, default=5000)
    p.add_argument("--print_freq", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lidar_lr", type=float, default=5e-2)
    p.add_argument("--lidar_every", type=int, default=1)
    p.add_argument("--w_depth", type=float, default=0.1)
    p.add_argument("--w_intensity", type=float, default=0.05)
    p.add_argument("--w_raydrop", type=float, default=0.05)
    p.add_argument("--w_los", type=float, default=0.001)
    p.add_argument(
        "--freeze_geometry", action="store_true",
        help="Train only LiDAR intensity/raydrop (+ head); keep RGB Gaussians "
             "exactly as in --resume_from. Writes lidar_sidecar_frozen*.pt only.",
    )
    p.add_argument(
        "--lidar_only_geometry",
        action="store_true",
        help=(
            "Optimize an independent LiDAR Gaussian copy without RGB losses. "
            "Used by the Separate C/L ablation; the camera representation remains "
            "the untouched --resume_from checkpoint."
        ),
    )
    p.add_argument(
        "--disable_lidar_attributes",
        action="store_true",
        help=(
            "Remove trainable per-Gaussian intensity/ray-drop attributes and their "
            "losses. Fixed initialization values remain only because the renderer "
            "interface requires tensors of the corresponding shape."
        ),
    )
    p.add_argument("opts", nargs="*", help="OmegaConf overrides")
    main(p.parse_args())
