import argparse

import os

import sys

import torch

import logging

import numpy as np

from omegaconf import OmegaConf

from typing import List

from PIL import Image



# Add current directory to python path

sys.path.append(os.getcwd())



from models.trainers import MultiTrainer

from models.video_utils import render_images, save_videos

from models.nodes.rigid import RigidNodes

from models.nodes.smpl import SMPLNodes

from models.human_body import SMPLTemplate



logger = logging.getLogger()

logging.basicConfig(level=logging.INFO)


def swap_instances(trainer, class_name: str, id_a: int, id_b: int):
    """交换两个实例的轨迹（trans + quats），外观 Gaussian 点不动。"""
    if class_name not in trainer.models:
        logger.warning(f"Model {class_name} not found.")
        return

    model = trainer.models[class_name]
    for inst_id in [id_a, id_b]:
        if inst_id < 0 or inst_id >= model.num_instances:
            logger.error(f"Instance ID {inst_id} out of range.")
            return

    logger.info(f"Swapping instances {id_a} <-> {id_b} in {class_name}...")

    for attr in ['instances_trans', 'instances_quats']:
        param = getattr(model, attr)
        data = param.data if isinstance(param, torch.nn.Parameter) else param
        # 交换 dim=1 上的两列
        tmp = data[:, id_a].clone()
        data[:, id_a] = data[:, id_b]
        data[:, id_b] = tmp

    logger.info(f"Swapped instances {id_a} <-> {id_b}.")


def render_first_frame_mosaic(trainer, dataset, output_path: str):
    num_cams = dataset.pixel_source.num_cams
    full_set = dataset.full_image_set

    class SubsetWrapper:
        def __init__(self, base, indices):
            self.base = base
            self.indices = indices
            self.split = getattr(base, 'split', 'full')
            self.datasource = getattr(base, 'datasource', None)
            self.split_indices = [getattr(base, 'split_indices', list(range(len(base))))[i] for i in indices]

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            return self.base[self.indices[idx]]

        def __getattr__(self, name):
            return getattr(self.base, name)

    first_frame_set = SubsetWrapper(full_set, list(range(num_cams)))

    trainer.set_eval()
    with torch.no_grad():
        render_results = render_images(
            trainer=trainer,
            dataset=first_frame_set,
            compute_metrics=False,
            compute_error_map=False,
        )

    # 打印实际 key，方便调试（确认后可删除）
    logger.info(f"render_results keys: {list(render_results.keys())}")

    # 按优先级尝试常见 key 名
    rgb_key = None
    for candidate in ["rgb", "rgbs", "color", "colors", "image", "images", "rendered_rgb"]:
        if candidate in render_results:
            rgb_key = candidate
            break

    if rgb_key is None:
        raise KeyError(f"Cannot find RGB key in render_results. Available keys: {list(render_results.keys())}")

    rgb_list = render_results[rgb_key]

    imgs = []
    for i in range(num_cams):
        frame = rgb_list[i]
        if isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
        # 兼容值域 [0,1] 和 [0,255]
        if frame.max() <= 1.0:
            frame = (frame * 255)
        frame = frame.clip(0, 255).astype(np.uint8)
        imgs.append(Image.fromarray(frame))

    # 拼图：两行三列
    W, H = imgs[0].size
    cols = 3
    rows = (num_cams + cols - 1) // cols
    mosaic = Image.new("RGB", (W * cols, H * rows))
    for idx, img in enumerate(imgs):
        r, c = divmod(idx, cols)
        mosaic.paste(img, (c * W, r * H))

    mosaic.save(output_path)
    logger.info(f"Saved mosaic to {output_path}")

def remove_instances(trainer, class_name: str, instance_ids: List[int]):

    if class_name not in trainer.models:

        logger.warning(f"Model {class_name} not found.")

        return

   

    model = trainer.models[class_name]

    if not hasattr(model, 'instances_fv'):

        logger.warning(f"Model {class_name} does not support instance visibility control.")

        return



    logger.info(f"Removing instances {instance_ids} from {class_name}...")

    # Set frame validity to 0 for all frames for these instances

    for inst_id in instance_ids:

        if inst_id < 0 or inst_id >= model.num_instances:

            logger.warning(f"Instance ID {inst_id} out of range (0-{model.num_instances-1}). Skipping.")

            continue

        model.instances_fv[:, inst_id] = 0

   

    logger.info(f"Removed instances {instance_ids} from {class_name}.")



def add_instance(trainer, class_name: str, src_id: int, offset: List[float]):

    if class_name not in trainer.models:

        logger.warning(f"Model {class_name} not found.")

        return

   

    model = trainer.models[class_name]

    if src_id < 0 or src_id >= model.num_instances:

        logger.error(f"Source instance ID {src_id} out of range.")

        return



    logger.info(f"Adding new instance to {class_name} by cloning {src_id} with offset {offset}...")

   

    device = model.device

    offset_tensor = torch.tensor(offset, device=device)

   

    # 1. Clone Point Parameters

    # Identify points belonging to src_id

    mask = (model.point_ids == src_id).squeeze()

    num_new_points = mask.sum().item()

   

    if num_new_points == 0:

        logger.warning(f"No points found for instance {src_id}.")

        return



    # Clone and append parameters

    # Helper to append parameter

    def append_param(param_name, new_data):

        old_param = getattr(model, param_name)

        new_param = torch.nn.Parameter(torch.cat([old_param.data, new_data], dim=0))

        setattr(model, param_name, new_param)



    # Points data

    append_param('_means', model._means[mask].clone()) # Points are in canonical space usually, so no offset needed here?

    # Wait, for RigidNodes, points are in canonical space. Offset is applied to trajectory.

    # So we just clone the canonical points.

   

    append_param('_features_dc', model._features_dc[mask].clone())

    append_param('_features_rest', model._features_rest[mask].clone())

    append_param('_opacities', model._opacities[mask].clone())

    append_param('_scales', model._scales[mask].clone())

    append_param('_quats', model._quats[mask].clone())

   

    # Update point_ids

    new_inst_id = model.num_instances # This is the ID *before* we increment it in trajectory

    new_point_ids = torch.full((num_new_points, 1), new_inst_id, dtype=torch.long, device=device)

    model.point_ids = torch.cat([model.point_ids, new_point_ids], dim=0)

   

    # 2. Clone Trajectory/Instance Parameters

    # instances_trans: (num_frames, num_instances, 3)

    # instances_quats: (num_frames, num_instances, 4) (or similar)

    # instances_fv: (num_frames, num_instances)

    # instances_size: (num_instances, 3)

   

    # Helper to append instance param (dim 1 usually, except size)

    def append_inst_param(param_name, new_data, dim=1):

        old_param = getattr(model, param_name)

        if isinstance(old_param, torch.nn.Parameter):

            new_param = torch.nn.Parameter(torch.cat([old_param.data, new_data], dim=dim))

            setattr(model, param_name, new_param)

        else:

            # Regular tensor

            new_tensor = torch.cat([old_param, new_data], dim=dim)

            setattr(model, param_name, new_tensor)



    # Trajectory

    src_trans = model.instances_trans[:, src_id:src_id+1, :].clone()

    new_trans = src_trans + offset_tensor.view(1, 1, 3)

    append_inst_param('instances_trans', new_trans, dim=1)

   

    # Quats

    # RigidNodes: instances_quats (num_frames, num_instances, 4)

    # SMPLNodes: instances_quats (num_frames, num_instances, 1, 4)

    src_quats = model.instances_quats[:, src_id:src_id+1].clone()

    append_inst_param('instances_quats', src_quats, dim=1)

   

    # FV

    src_fv = model.instances_fv[:, src_id:src_id+1].clone()

    append_inst_param('instances_fv', src_fv, dim=1)

   

    # Size

    src_size = model.instances_size[src_id:src_id+1].clone()

    append_inst_param('instances_size', src_size, dim=0)

   

    # 3. Special handling for SMPLNodes

    if isinstance(model, SMPLNodes):

        # smpl_qauts: (num_frame, num_instances, 23, 4)

        src_smpl_quats = model.smpl_qauts[:, src_id:src_id+1].clone()

        append_inst_param('smpl_qauts', src_smpl_quats, dim=1)

       

        # smpl_betas is not stored as a direct attribute in SMPLNodes usually,

        # but passed to SMPLTemplate.

        # However, we need to update self.template.

        # We need to find where betas are stored.

        # In create_from_pcd, they are local variables.

        # But self.template.init_beta might store them?

        # Let's check SMPLTemplate if possible, or just assume we can access it.

        # If not, we might be stuck for SMPL.

        # But wait, SMPLNodes.template is an SMPLTemplate instance.

       

        if hasattr(model.template, 'init_beta'):

             # Assuming init_beta is (num_instances, 10)

             src_beta = model.template.init_beta[src_id:src_id+1].clone()

             new_betas = torch.cat([model.template.init_beta, src_beta], dim=0)

             

             # Re-initialize template

             # We need to know other params used to init template.

             # cano_pose_type="da_pose", use_voxel_deformer=self.use_voxel_deformer

             # We can try to re-instantiate it.

             

             model.template = SMPLTemplate(

                smpl_model_path="smpl_models/SMPL_NEUTRAL.pkl",

                num_human=new_betas.shape[0],

                init_beta=new_betas,

                cano_pose_type="da_pose",

                use_voxel_deformer=model.use_voxel_deformer

            )

             if model.use_voxel_deformer:

                 model.template.voxel_deformer.enable_voxel_correction()

                 # Note: We lose trained voxel deformer weights if we just re-init!

                 # We should copy old weights to new deformer.

                 # This is getting complicated.

                 # For now, maybe just warn about SMPL adding limitations.

                 logger.warning("Adding SMPL instances resets voxel deformer weights (if used). Fine-tuning might be needed.")

        else:

            logger.warning("Could not find init_beta in template. SMPL shape might be wrong.")



    logger.info(f"Added instance {new_inst_id} to {class_name}.")





import json

from pytorch3d.transforms import euler_angles_to_matrix



def add_external_instance(trainer, model_path: str, pos: List[float], yaw: float = 0.0):

    class_name = "RigidNodes"

    if class_name not in trainer.models:

        logger.warning(f"Model {class_name} not found.")

        return

   

    model = trainer.models[class_name]

    device = model.device

   

    logger.info(f"Adding external instance from {model_path} at {pos} with yaw {yaw}...")

   

    # 1. Load External Model

    gs_path = os.path.join(model_path, "gs.pth")

    json_path = os.path.join(model_path, "wlh.json")

   

    if not os.path.exists(gs_path) or not os.path.exists(json_path):

        logger.error(f"Model files not found in {model_path}")

        return

       

    # Load Size

    with open(json_path, 'r') as f:

        dims = json.load(f) # [W, L, H] or similar? User said "wlh.json"

        # Usually size is (L, W, H) in this codebase?

        # Let's assume the json is [L, W, H] or [W, L, H].

        # The user provided example content: [1.74, 4.09, 1.26].

        # 4.09 is likely Length. 1.74 is Width. 1.26 is Height.

        # So it's [W, L, H].

        # RigidNodes expects size as (L, W, H).

        # So we map: size = [dims[1], dims[0], dims[2]]

        size = [dims[1], dims[0], dims[2]]

       

    # Load GS

    ckpt = torch.load(gs_path, map_location="cpu")

    # ckpt is a tuple, first element is list of tensors

    data = ckpt[0]

    # 1: means, 2: dc, 3: rest, 5: scales, 6: quats, 7: opacities

   

    new_means = data[1].to(device)

    new_dc = data[2].to(device)

    if new_dc.ndim == 3 and new_dc.shape[1] == 1:

        new_dc = new_dc.squeeze(1)

    new_rest = data[3].to(device)

    new_scales = data[5].to(device)

    new_quats = data[6].to(device)

    new_opacities = data[7].to(device)

   

    num_new_points = new_means.shape[0]

   

    # 2. Append Parameters

    def append_param(param_name, new_data):

        old_param = getattr(model, param_name)

        new_param = torch.nn.Parameter(torch.cat([old_param.data, new_data], dim=0))

        setattr(model, param_name, new_param)



    append_param('_means', new_means)

    append_param('_features_dc', new_dc)

    append_param('_features_rest', new_rest)

    append_param('_opacities', new_opacities)

    append_param('_scales', new_scales)

    append_param('_quats', new_quats)

   

    # Update point_ids

    new_inst_id = model.num_instances

    new_point_ids = torch.full((num_new_points, 1), new_inst_id, dtype=torch.long, device=device)

    model.point_ids = torch.cat([model.point_ids, new_point_ids], dim=0)

   

    # 3. Setup Trajectory

    # Static at pos

    num_frames = model.instances_trans.shape[0]

   

    # Trans: (num_frames, 1, 3)

    new_trans = torch.tensor(pos, device=device, dtype=torch.float32).view(1, 1, 3).repeat(num_frames, 1, 1)

   

    # Quats: (num_frames, 1, 4)

    # Yaw to quat

    # We can use pytorch3d or just simple math for yaw

    # q = [cos(yaw/2), 0, 0, sin(yaw/2)] for rotation around Z?

    # Wait, coordinate system.

    # Usually Z is up?

    # If Z is up, yaw is rotation around Z.

    # q = [w, x, y, z]

    cy = np.cos(yaw * 0.5)

    sy = np.sin(yaw * 0.5)

    # Assuming (w, x, y, z) format for pytorch3d/this codebase

    # Let's check quaternion_to_matrix usage in vis_combined.py

    # It uses pytorch3d.transforms.quaternion_to_matrix.

    # PyTorch3D uses (w, x, y, z).

    new_q = torch.tensor([cy, 0.0, 0.0, sy], device=device, dtype=torch.float32).view(1, 1, 4).repeat(num_frames, 1, 1)

   

    # Size: (1, 3)

    new_size = torch.tensor(size, device=device, dtype=torch.float32).view(1, 3)

   

    # FV: (num_frames, 1) -> 1 (visible)

    new_fv = torch.ones((num_frames, 1), device=device)

   

    def append_inst_param(param_name, new_data, dim=1):

        old_param = getattr(model, param_name)

        if isinstance(old_param, torch.nn.Parameter):

            new_param = torch.nn.Parameter(torch.cat([old_param.data, new_data], dim=dim))

            setattr(model, param_name, new_param)

        else: # Tensor

             new_tensor = torch.cat([old_param, new_data], dim=dim)

             setattr(model, param_name, new_tensor)



    append_inst_param('instances_trans', new_trans)

    append_inst_param('instances_quats', new_q)

    append_inst_param('instances_size', new_size, dim=0)

    append_inst_param('instances_fv', new_fv)

   

    # model.num_instances is a property, so we don't need to increment it manually.

    # It is derived from instances_fv.shape[1], which we just updated.

    logger.info(f"Added external instance {new_inst_id} at {pos}.")



def main():
    parser = argparse.ArgumentParser(description="Edit scene (remove/add objects) and render 3 comparison mosaics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--swap_vehicles", type=str, default=None,
                        help="两个要互换的车辆 ID，格式: id_a,id_b，例如 --swap_vehicles 0,2")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # ---------- 加载配置（与原逻辑相同）----------
    cfg = OmegaConf.load(args.config)
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg_path = os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        if os.path.exists(dataset_cfg_path):
            cfg = OmegaConf.merge(cfg, OmegaConf.load(dataset_cfg_path))
        else:
            logger.warning(f"Dataset config {dataset_cfg_path} not found.")
    cfg = OmegaConf.merge(cfg, args_from_cli)
    cfg.log_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- 加载数据集 ----------
    from datasets.driving_dataset import DrivingDataset
    dataset = DrivingDataset(cfg.data)

    # ---------- 公共 trainer 初始化函数 ----------
    def build_trainer():
        t = MultiTrainer(
            num_timesteps=dataset.end_timestep - dataset.start_timestep,
            **cfg.trainer,
            model_config=cfg.model,
            num_train_images=len(dataset.train_image_set),
            num_full_images=len(dataset.full_image_set),
            scene_aabb=dataset.aabb,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            test_set_indices=dataset.test_indices,
        )
        t.resume_from_checkpoint(args.checkpoint, load_only_model=True)
        return t

    # ========== 场景 1：原始场景 ==========
    logger.info("=== Scene 1: Original ===")
    trainer_orig = build_trainer()
    render_first_frame_mosaic(
        trainer_orig, dataset,
        os.path.join(args.output_dir, "scene1_original.png")
    )

    # ========== 场景 2：删除所有动态车辆 ==========
    logger.info("=== Scene 2: Remove all vehicles ===")
    trainer_rm = build_trainer()
    if "RigidNodes" in trainer_rm.models:
        num = trainer_rm.models["RigidNodes"].num_instances
        remove_instances(trainer_rm, "RigidNodes", list(range(num)))
    render_first_frame_mosaic(
        trainer_rm, dataset,
        os.path.join(args.output_dir, "scene2_removed_vehicles.png")
    )

    # ========== 场景 3：调换两辆车的位置 ==========
    logger.info("=== Scene 3: Swap vehicles ===")
    trainer_swap = build_trainer()
    if args.swap_vehicles:
        id_a, id_b = [int(x) for x in args.swap_vehicles.split(',')]
    else:
        # 默认交换 0 号和 1 号（你可按需改为前车/右车的实际 ID）
        id_a, id_b = 0, 1
        logger.warning(f"--swap_vehicles not specified, defaulting to swap {id_a} <-> {id_b}")
    swap_instances(trainer_swap, "RigidNodes", id_a, id_b)
    render_first_frame_mosaic(
        trainer_swap, dataset,
        os.path.join(args.output_dir, "scene3_swapped_vehicles.png")
    )

    logger.info("Done. Results saved to: %s", args.output_dir)


if __name__ == "__main__":
    main()