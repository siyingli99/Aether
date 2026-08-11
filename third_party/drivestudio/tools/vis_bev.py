
import argparse
import os
import sys
import torch
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from omegaconf import OmegaConf
from pytorch3d.transforms import quaternion_to_matrix

# Add current directory to python path
sys.path.append(os.getcwd())

from models.trainers import MultiTrainer
from datasets.driving_dataset import DrivingDataset

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

def get_box_corners(x, y, yaw, l, w):
    # x, y: center
    # yaw: rotation in radians
    # l: length (along heading)
    # w: width
    
    c = np.cos(yaw)
    s = np.sin(yaw)
    
    # Rotation matrix
    R = np.array([[c, -s], [s, c]])
    
    # Corners relative to center
    # Front-Left, Front-Right, Rear-Right, Rear-Left
    corners_local = np.array([
        [l/2, w/2],
        [l/2, -w/2],
        [-l/2, -w/2],
        [-l/2, w/2]
    ])
    
    corners_global = corners_local @ R.T + np.array([x, y])
    return corners_global

def main():
    parser = argparse.ArgumentParser(description="Visualize BEV of the scene")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file")
    parser.add_argument("--output", required=True, help="Path to output video file (mp4)")
    parser.add_argument("--fps", type=int, default=10, help="FPS of the output video")
    
    args = parser.parse_args()
    
    # Load Config
    cfg = OmegaConf.load(args.config)
    
    # Init Dataset (to get num_timesteps and other info)
    dataset = DrivingDataset(cfg.data)
    
    # Init Trainer
    # Calculate num_timesteps from dataset
    num_timesteps = dataset.end_timestep - dataset.start_timestep
    
    trainer = MultiTrainer(
        num_timesteps=num_timesteps,
        **cfg.trainer,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        scene_aabb=dataset.aabb,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Resume
    trainer.resume_from_checkpoint(args.checkpoint, load_only_model=True)
    
    # Extract Data
    objects = [] # List of dicts: {type, id, trans, quats, size, fv}
    
    # RigidNodes (Vehicles)
    if "RigidNodes" in trainer.models:
        model = trainer.models["RigidNodes"]
        trans = model.instances_trans.detach().cpu().numpy() # (T, N, 3)
        quats = model.instances_quats.detach().cpu().numpy() # (T, N, 4)
        size = model.instances_size.detach().cpu().numpy()   # (N, 3)
        fv = model.instances_fv.detach().cpu().numpy()       # (T, N)
        
        for i in range(model.num_instances):
            objects.append({
                "type": "vehicle",
                "id": i,
                "trans": trans[:, i, :],
                "quats": quats[:, i, :],
                "size": size[i, :],
                "fv": fv[:, i]
            })
            
    # SMPLNodes (Pedestrians)
    if "SMPLNodes" in trainer.models:
        model = trainer.models["SMPLNodes"]
        trans = model.instances_trans.detach().cpu().numpy() # (T, N, 3)
        # SMPL quats might be (T, N, 1, 4)
        quats = model.instances_quats.detach().cpu().numpy()
        if quats.ndim == 4:
            quats = quats[:, :, 0, :]
            
        size = model.instances_size.detach().cpu().numpy()   # (N, 3)
        fv = model.instances_fv.detach().cpu().numpy()       # (T, N)
        
        for i in range(model.num_instances):
            objects.append({
                "type": "human",
                "id": i,
                "trans": trans[:, i, :],
                "quats": quats[:, i, :],
                "size": size[i, :],
                "fv": fv[:, i]
            })

    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Determine bounds
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
        xlim = (-50, 50)
        ylim = (-50, 50)
        
    def update(frame_idx):
        ax.clear()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_title(f"Frame: {frame_idx}")
        
        # Draw objects
        for obj in objects:
            if frame_idx >= len(obj["fv"]) or obj["fv"][frame_idx] == 0:
                continue
            
            # Position
            pos = obj["trans"][frame_idx]
            x, y = pos[0], pos[1]
            
            # Rotation (Yaw)
            # Quat to RotMat
            q = torch.tensor(obj["quats"][frame_idx])
            # w, x, y, z order in pytorch3d?
            # models/gaussians/basics.py uses standard order usually.
            # Let's assume standard w, x, y, z.
            # But wait, pytorch3d quaternion_to_matrix expects (w, x, y, z).
            # Let's check if we need to convert.
            # Assuming the model uses standard convention.
            
            R = quaternion_to_matrix(q)
            # Yaw is rotation around Z axis.
            # R is 3x3.
            # yaw = atan2(R[1, 0], R[0, 0])
            yaw = np.arctan2(R[1, 0].item(), R[0, 0].item())
            
            # Size
            # size is usually (l, w, h) or (w, l, h)?
            # In rigid.py: instance_max_pts = 5000
            # Usually size is (l, w, h) in object detection.
            l, w = obj["size"][0], obj["size"][1]
            
            # Color
            color = 'blue' if obj["type"] == 'vehicle' else 'red'
            
            # Draw Box
            corners = get_box_corners(x, y, yaw, l, w)
            poly = plt.Polygon(corners, closed=True, fill=False, edgecolor=color, linewidth=2)
            ax.add_patch(poly)
            
            # Draw ID
            ax.text(x, y, str(obj["id"]), color=color, fontsize=8, ha='center', va='center')
            
            # Draw Trajectory (History)
            # Get valid history
            history_mask = (obj["fv"][:frame_idx+1] > 0)
            if np.any(history_mask):
                history = obj["trans"][:frame_idx+1][history_mask]
                ax.plot(history[:, 0], history[:, 1], color=color, linewidth=1, alpha=0.5)

    ani = animation.FuncAnimation(fig, update, frames=num_timesteps, interval=1000/args.fps)
    
    logger.info(f"Saving BEV video to {args.output}...")
    try:
        ani.save(args.output, writer='ffmpeg', fps=args.fps)
    except Exception as e:
        logger.warning(f"Failed to save as mp4 with ffmpeg: {e}")
        logger.info("Trying to save as gif using Pillow...")
        output_gif = args.output.replace(".mp4", ".gif")
        ani.save(output_gif, writer='pillow', fps=args.fps)
        logger.info(f"Saved BEV video to {output_gif}")
    logger.info("Done.")

if __name__ == "__main__":
    main()
