from typing import List, Optional
from omegaconf import OmegaConf
import os
import sys
import time
import argparse
import logging
import imageio
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix

import matplotlib
matplotlib.use('Agg') # 防止在无 GUI 的 Linux/WSL 环境下报错
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str

logger = logging.getLogger()

# ====================================================================
# BEV 渲染相关逻辑 (完美整合你的 vis_bev.py)
# ====================================================================
def get_box_corners(x, y, yaw, l, w):
    c = np.cos(yaw)
    s = np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    corners_local = np.array([
        [l/2, w/2],
        [l/2, -w/2],
        [-l/2, -w/2],
        [-l/2, w/2]
    ])
    corners_global = corners_local @ R.T + np.array([x, y])
    return corners_global

def generate_bev_video(trainer, dataset, output_path, fps=10):
    logger.info("🗺️ 开始生成高质量 Matplotlib BEV 二维仿真视频...")
    objects = []
    
    T = dataset.num_img_timesteps
    num_cams = getattr(dataset.pixel_source, 'num_cams', 6)
    
    # 1. 提取 Ego 主车轨迹 (额外加入，让你能在上帝视角看到自己)
    ego_trans = []
    ego_yaws = []
    for t in range(T):
        idx = t * num_cams
        _, cam_infos = dataset.full_image_set.get_image(idx, 1.0)
        c2w = cam_infos["camera_to_world"].cpu().numpy()
        ego_trans.append(c2w[:3, 3])
        forward = c2w[:3, 0] # 假设 X 轴向前
        yaw = np.arctan2(forward[1], forward[0])
        ego_yaws.append(yaw)
        
    objects.append({
        "type": "ego",
        "id": "Ego",
        "trans": np.stack(ego_trans),
        "yaws": np.array(ego_yaws),
        "size": np.array([4.5, 2.0, 1.5]), # 主车尺寸
        "fv": np.ones(T) # 主车始终可见
    })
    
    # 2. 提取 RigidNodes (其他车辆)
    if "RigidNodes" in trainer.models:
        model = trainer.models["RigidNodes"]
        trans = model.instances_trans.detach().cpu().numpy()
        quats = model.instances_quats.detach().cpu().numpy()
        size = model.instances_size.detach().cpu().numpy()
        fv = model.instances_fv.detach().cpu().numpy()
        
        for i in range(model.num_instances):
            objects.append({
                "type": "vehicle", "id": i,
                "trans": trans[:, i, :], "quats": quats[:, i, :],
                "size": size[i, :], "fv": fv[:, i]
            })
            
    # 3. 提取 SMPLNodes (行人)
    if "SMPLNodes" in trainer.models:
        model = trainer.models["SMPLNodes"]
        trans = model.instances_trans.detach().cpu().numpy()
        quats = model.instances_quats.detach().cpu().numpy()
        if quats.ndim == 4:
            quats = quats[:, :, 0, :]
        size = model.instances_size.detach().cpu().numpy()
        fv = model.instances_fv.detach().cpu().numpy()
        
        for i in range(model.num_instances):
            objects.append({
                "type": "human", "id": i,
                "trans": trans[:, i, :], "quats": quats[:, i, :],
                "size": size[i, :], "fv": fv[:, i]
            })

    # 绘制画板
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # 确定边界
    all_trans = []
    for obj in objects:
        valid_trans = obj["trans"][obj["fv"] > 0]
        if len(valid_trans) > 0:
            all_trans.append(valid_trans)
            
    if len(all_trans) > 0:
        all_trans = np.concatenate(all_trans, axis=0)
        min_xy = np.min(all_trans[:, :2], axis=0)
        max_xy = np.max(all_trans[:, :2], axis=0)
        center = (min_xy + max_xy) / 2
        span = max(max_xy - min_xy) * 1.1 / 2
        xlim = (center[0] - span, center[0] + span)
        ylim = (center[1] - span, center[1] + span)
    else:
        xlim, ylim = (-50, 50), (-50, 50)
        
    def update(frame_idx):
        ax.clear()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(f"OmniRe BEV Simulation | Frame: {frame_idx}")
        
        for obj in objects:
            if frame_idx >= len(obj["fv"]) or obj["fv"][frame_idx] == 0:
                continue
            
            pos = obj["trans"][frame_idx]
            x, y = pos[0], pos[1]
            
            if "yaws" in obj:
                yaw = obj["yaws"][frame_idx]
            else:
                q = torch.tensor(obj["quats"][frame_idx])
                R = quaternion_to_matrix(q)
                yaw = np.arctan2(R[1, 0].item(), R[0, 0].item())
                
            l, w = obj["size"][0], obj["size"][1]
            
            # 分配颜色：主车绿色，其他车蓝色，行人红色
            if obj["type"] == 'ego': color = 'green'
            elif obj["type"] == 'vehicle': color = 'blue'
            else: color = 'red'
            
            corners = get_box_corners(x, y, yaw, l, w)
            # 主车用实心填充以示区分
            poly = plt.Polygon(corners, closed=True, 
                               fill=(obj["type"] == 'ego'), 
                               edgecolor=color, facecolor=color if obj["type"]=='ego' else 'none', linewidth=2)
            ax.add_patch(poly)
            
            text_color = 'white' if obj["type"] == 'ego' else color
            ax.text(x, y, str(obj["id"]), color=text_color, fontsize=8, ha='center', va='center')
            
            # 画轨迹线
            history_mask = (obj["fv"][:frame_idx+1] > 0)
            if np.any(history_mask):
                history = obj["trans"][:frame_idx+1][history_mask]
                ax.plot(history[:, 0], history[:, 1], color=color, linewidth=1, alpha=0.5)

    ani = animation.FuncAnimation(fig, update, frames=T, interval=1000/fps)
    try:
        ani.save(output_path, writer='ffmpeg', fps=fps)
    except Exception as e:
        logger.warning(f"ffmpeg 保存失败: {e}. 尝试保存为 GIF...")
        ani.save(output_path.replace(".mp4", ".gif"), writer='pillow', fps=fps)
    plt.close(fig)
    logger.info(f"✅ BEV 视频已保存至: {output_path}")

# ====================================================================
# 3D 红绿灯渲染逻辑
# ====================================================================
@torch.no_grad()
def render_overhead_view(step: int, cfg: OmegaConf, trainer, dataset: DrivingDataset, post_fix: str = ""):
    logger.info("🚦 启动完美红绿灯俯视及 BEV 渲染管线...")
    
    video_output_dir = f"{cfg.log_dir}/videos{post_fix}"
    os.makedirs(video_output_dir, exist_ok=True)
    
    save_path_3d = os.path.join(video_output_dir, f"traffic_light_overhead_{step}.mp4")
    save_path_bev = os.path.join(video_output_dir, f"bev_simulation_{step}.mp4")
    
    num_cams = getattr(dataset.pixel_source, 'num_cams', 6)
    total_frames = len(dataset.full_image_set)
    indices = list(range(0, total_frames, num_cams))

    trainer.set_eval()
    camera_downscale = trainer._get_downscale_factor()
    image_infos_0, cam_infos_0 = dataset.full_image_set.get_image(0, camera_downscale)
    base_c2w = cam_infos_0["camera_to_world"].cuda()
    
    # 核心数学：自动 Look-At 矩阵计算
    pos_0 = base_c2w[:3, 3]
    forward_0 = base_c2w[:3, 2] 
    
    pos_cam = pos_0 + torch.tensor([0.0, 0.0, 10.0], device=base_c2w.device)
    look_at_pt = pos_0 + forward_0 * 30.0
    
    z_axis = F.normalize(look_at_pt - pos_cam, dim=0)
    world_up = torch.tensor([0.0, 0.0, 1.0], device=base_c2w.device)
    x_axis = F.normalize(torch.cross(world_up, z_axis), dim=0)
    y_axis = F.normalize(torch.cross(z_axis, x_axis), dim=0) 
    
    # 【需求修复】：绕 Z 轴(镜头光轴)反转 180 度，将颠倒的画面瞬间正过来！
    x_axis = -x_axis
    y_axis = -y_axis
    
    static_c2w = torch.eye(4, device=base_c2w.device)
    static_c2w[:3, 0] = x_axis
    static_c2w[:3, 1] = y_axis
    static_c2w[:3, 2] = z_axis
    static_c2w[:3, 3] = pos_cam
    
    writer = imageio.get_writer(save_path_3d, mode='I', fps=cfg.render.fps)
    
    for i in tqdm(indices, desc="🎬 正在渲染 3D 监控画面"):
        image_infos, cam_infos = dataset.full_image_set.get_image(i, camera_downscale)
        for k, v in image_infos.items():
            if isinstance(v, torch.Tensor): image_infos[k] = v.cuda(non_blocking=True)
        for k, v in cam_infos.items():
            if isinstance(v, torch.Tensor): cam_infos[k] = v.cuda(non_blocking=True)
            
        cam_infos["camera_to_world"] = static_c2w.clone()
        
        results = trainer(image_infos, cam_infos)
        rgb = results["rgb"].clamp(0., 1.).cpu().numpy()
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        writer.append_data(rgb_uint8)
        
    writer.close()
    logger.info(f"✅ 3D 红绿灯视频已保存至: {save_path_3d}")

    # 调用整合好的函数生成高质量的 BEV 视频
    generate_bev_video(trainer, dataset, save_path_bev, fps=cfg.render.fps)

def main(args):
    log_dir = os.path.dirname(args.resume_from)
    cfg = OmegaConf.load(os.path.join(log_dir, "config.yaml"))
    if args.opts:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_cli(args.opts))
        
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
        device=device
    )
    
    trainer.resume_from_checkpoint(
        ckpt_path=args.resume_from,
        load_only_model=True
    )
    
    render_overhead_view(
        step=trainer.step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        post_fix="_eval"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()    
    parser.add_argument("--resume_from", default=None, type=str, required=True)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    main(args)