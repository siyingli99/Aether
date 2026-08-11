
import argparse
import os
import sys
import torch
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from omegaconf import OmegaConf
from pytorch3d.transforms import quaternion_to_matrix
from tqdm import tqdm
import cv2

# Add current directory to python path
sys.path.append(os.getcwd())

from models.trainers import MultiTrainer
from datasets.driving_dataset import DrivingDataset
from models.video_utils import render_images
from utils.visualization import to8b

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

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

def render_bev_frame(fig, ax, frame_idx, objects, ego_pose, xlim, ylim):
    ax.clear()
    ax.set_facecolor('white')
    ax.grid(False)
    
    # Set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    
    # Draw border (spines)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)
    
    # Remove ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Draw Ego Vehicle
    # ego_pose is (4, 4)
    ego_trans = ego_pose[:3, 3]
    ego_rot = ego_pose[:3, :3]
    # Yaw from rotation matrix
    ego_yaw = np.arctan2(ego_rot[1, 0], ego_rot[0, 0])
    
    # Ego size (approx)
    ego_l, ego_w = 4.5, 2.0
    ego_corners = get_box_corners(ego_trans[0], ego_trans[1], ego_yaw, ego_l, ego_w)
    
    # Ego: Bright Green, Filled
    ego_color = '#32CD32' # Lime Green
    ego_poly = plt.Polygon(ego_corners, closed=True, fill=True, facecolor=ego_color, edgecolor='black', linewidth=1, label='Ego', alpha=1.0, zorder=10)
    ax.add_patch(ego_poly)
    
    # Draw objects
    for obj in objects:
        if frame_idx >= len(obj["fv"]) or obj["fv"][frame_idx] == 0:
            continue
        
        pos = obj["trans"][frame_idx]
        x, y = pos[0], pos[1]
        
        q = torch.tensor(obj["quats"][frame_idx])
        R = quaternion_to_matrix(q)
        yaw = np.arctan2(R[1, 0].item(), R[0, 0].item())
        
        l, w = obj["size"][0], obj["size"][1]
        
        if obj["type"] == 'vehicle':
            # Background Vehicles: Blue, Filled
            facecolor = '#4169E1' # Royal Blue
            edgecolor = 'black'
            alpha = 0.7
            zorder = 5
        else:
            # Pedestrians: Red/Orange, Filled
            facecolor = '#FF4500' # Orange Red
            edgecolor = 'black'
            alpha = 0.9
            zorder = 6
            # Make humans slightly larger to be visible
            l = max(l, 0.5)
            w = max(w, 0.5)
            
        corners = get_box_corners(x, y, yaw, l, w)
        poly = plt.Polygon(corners, closed=True, fill=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=1, alpha=alpha, zorder=zorder)
        ax.add_patch(poly)
        
        # ID (Optional, maybe too cluttered)
        # ax.text(x, y, str(obj["id"]), color='black', fontsize=8, ha='center', va='center', weight='bold')
        
        # Trajectory
        history_mask = (obj["fv"][:frame_idx+1] > 0)
        if np.any(history_mask):
            history = obj["trans"][:frame_idx+1][history_mask]
            ax.plot(history[:, 0], history[:, 1], color=facecolor, linewidth=1, alpha=0.3, zorder=zorder-1)

    # Add Legend
    legend_elements = [
        Patch(facecolor='#32CD32', edgecolor='black', label='Ego Vehicle'),
        Patch(facecolor='#4169E1', edgecolor='black', label='Other Vehicles', alpha=0.7),
        Patch(facecolor='#FF4500', edgecolor='black', label='Pedestrians', alpha=0.9)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

    # Render to numpy array
    fig.canvas.draw()
    try:
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except AttributeError:
        # Fallback for newer matplotlib versions
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        data = data.reshape((h, w, 4))
        data = data[:, :, :3] # Drop alpha channel
        
    return data

def main():
    parser = argparse.ArgumentParser(description="Visualize Combined BEV and Camera Views")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--output", required=True, help="Path to output video file (mp4)")
    parser.add_argument("--fps", type=int, default=10, help="FPS of the output video")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    
    # Load Config
    cfg = OmegaConf.load(args.config)
    
    # Merge CLI args for dataset selection
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")
        
    # Merge dataset config if specified
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg_path = os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        if os.path.exists(dataset_cfg_path):
            dataset_cfg = OmegaConf.load(dataset_cfg_path)
            cfg = OmegaConf.merge(cfg, dataset_cfg)
        else:
            logger.warning(f"Dataset config {dataset_cfg_path} not found.")
            
    # Merge remaining CLI args
    cfg = OmegaConf.merge(cfg, args_from_cli)
    
    # Init Dataset
    dataset = DrivingDataset(cfg.data)
    num_timesteps = dataset.end_timestep - dataset.start_timestep
    
    # Init Trainer
    trainer = MultiTrainer(
        num_timesteps=num_timesteps,
        **cfg.trainer,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        scene_aabb=dataset.aabb,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        test_set_indices=dataset.test_indices
    )
    
    # Resume
    trainer.resume_from_checkpoint(args.checkpoint, load_only_model=True)
    trainer.set_eval()
    
    # Extract Data for BEV
    objects = []
    if "RigidNodes" in trainer.models:
        model = trainer.models["RigidNodes"]
        trans = model.instances_trans.detach().cpu().numpy()
        quats = model.instances_quats.detach().cpu().numpy()
        size = model.instances_size.detach().cpu().numpy()
        fv = model.instances_fv.detach().cpu().numpy()
        for i in range(model.num_instances):
            objects.append({"type": "vehicle", "id": i, "trans": trans[:, i, :], "quats": quats[:, i, :], "size": size[i, :], "fv": fv[:, i]})
            
    if "SMPLNodes" in trainer.models:
        model = trainer.models["SMPLNodes"]
        trans = model.instances_trans.detach().cpu().numpy()
        quats = model.instances_quats.detach().cpu().numpy()
        if quats.ndim == 4: quats = quats[:, :, 0, :]
        size = model.instances_size.detach().cpu().numpy()
        fv = model.instances_fv.detach().cpu().numpy()
        for i in range(model.num_instances):
            objects.append({"type": "human", "id": i, "trans": trans[:, i, :], "quats": quats[:, i, :], "size": size[i, :], "fv": fv[:, i]})

    # Ego Poses
    ego_poses = dataset.lidar_source.lidar_to_worlds.cpu().numpy() # (num_frames, 4, 4)

    # Setup BEV Plot
    # Determine bounds based on all objects and ego
    all_trans = []
    for obj in objects:
        valid_trans = obj["trans"][obj["fv"] > 0]
        if len(valid_trans) > 0: all_trans.append(valid_trans)
    
    # Add ego path for the relevant duration
    start_t = dataset.start_timestep
    end_t = dataset.end_timestep
    all_trans.append(ego_poses[start_t:end_t, :3, 3])
    
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

    # Render Images using video_utils
    logger.info("Rendering camera views...")
    
    # We manually render to ensure we have control and avoid re-implementing render_images logic if possible,
    # but render_images is convenient.
    # Note: render_images returns a dict with "rgbs" which is a list of numpy arrays.
    
    render_results = render_images(
        trainer=trainer,
        dataset=dataset.full_image_set,
        compute_metrics=False,
        compute_error_map=False
    )
    all_rgbs = render_results["rgbs"] # List of (H, W, 3)
    
    num_cams = dataset.pixel_source.num_cams
    
    # Setup Figure for BEV
    fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
    
    # Video Writer
    writer = None
    
    logger.info("Combining frames...")
    for t in tqdm(range(num_timesteps)):
        # 1. Get Camera Layout
        start_idx = t * num_cams
        imgs = all_rgbs[start_idx : start_idx + num_cams]
        # Convert to uint8 if not already
        imgs = [to8b(img) if img.dtype != np.uint8 else img for img in imgs]
        
        # Get camera names
        cam_names = [dataset.pixel_source.camera_data[cam_id].cam_name for cam_id in dataset.pixel_source.camera_list]
        cam_layout_img = dataset.layout(imgs, cam_names)
        
        # 2. Render BEV
        global_t = dataset.start_timestep + t
        bev_img = render_bev_frame(fig, ax, t, objects, ego_poses[global_t], xlim, ylim)
        
        # 3. Combine Side-by-Side
        H_cam, W_cam = cam_layout_img.shape[:2]
        H_bev, W_bev = bev_img.shape[:2]
        
        # Resize BEV to match H_cam
        scale = H_cam / H_bev
        new_W_bev = int(W_bev * scale)
        bev_img_resized = cv2.resize(bev_img, (new_W_bev, H_cam))
        
        # Canvas
        W_canvas = W_cam + new_W_bev
        H_canvas = H_cam
        
        canvas = np.zeros((H_canvas, W_canvas, 3), dtype=np.uint8)
        
        # Place Cam Layout (Left)
        canvas[:, :W_cam] = cam_layout_img
        
        # Place BEV (Right)
        canvas[:, W_cam:] = bev_img_resized
        
        # Init Writer
        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.output, fourcc, args.fps, (W_canvas, H_canvas))
            
        # Write
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        writer.write(canvas_bgr)
        
    if writer is not None:
        writer.release()
    
    logger.info(f"Saved combined video to {args.output}")

if __name__ == "__main__":
    main()
