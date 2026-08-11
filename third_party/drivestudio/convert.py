import torch
import os
import argparse
import yaml
from collections import OrderedDict
import json

def convert_background(bg_state, sh_degree=3):
    xyz = torch.nn.Parameter(bg_state['_means'])
    f_dc = torch.nn.Parameter(bg_state['_features_dc'].unsqueeze(1))
    f_rest = torch.nn.Parameter(bg_state['_features_rest'])
    
    # feats3D for HUGSIM: [N, 20]
    N = xyz.shape[0]
    feats3D = torch.zeros(N, 20).float()
    
    # Simple heuristic: points with Z < threshold might be ground. 
    # Let's say Z < -1.5 is ground (index 0), else scene (index 2). 
    # Usually in auto-driving datasets, Z is down or up. 
    # Let's check mean Z. If Waymo/Pandaset, Z might be up.
    Z_thresh = -1.0 # arbitrary
    is_ground = xyz[:, 2] < Z_thresh
    feats3D[is_ground, 0] = 10.0
    feats3D[~is_ground, 2] = 10.0
    feats3D = torch.nn.Parameter(feats3D)
    
    scaling = torch.nn.Parameter(bg_state['_scales'])
    rotation = torch.nn.Parameter(bg_state['_quats'])
    opacity = torch.nn.Parameter(bg_state['_opacities'])
    
    model_args = (
        sh_degree,
        xyz,
        f_dc,
        f_rest,
        feats3D,
        scaling,
        rotation,
        opacity,
        1.0,
        OrderedDict(),
        None
    )
    return model_args

def convert_dynamic(rigid_state, obj_id, global_mask, sh_degree=3):
    mask = global_mask == obj_id
    
    xyz = torch.nn.Parameter(rigid_state['_means'][mask])
    f_dc = torch.nn.Parameter(rigid_state['_features_dc'][mask].unsqueeze(1))
    f_rest = torch.nn.Parameter(rigid_state['_features_rest'][mask])
    
    N = xyz.shape[0]
    feats3D = torch.zeros(N, 20).float()
    # index 10 is dynamic vehicle
    feats3D[:, 10] = 10.0
    feats3D = torch.nn.Parameter(feats3D)
    
    scaling = torch.nn.Parameter(rigid_state['_scales'][mask])
    rotation = torch.nn.Parameter(rigid_state['_quats'][mask])
    opacity = torch.nn.Parameter(rigid_state['_opacities'][mask])
    
    model_args = (
        sh_degree,
        xyz,
        f_dc,
        f_rest,
        feats3D,
        scaling,
        rotation,
        opacity,
        1.0
    )
    return model_args

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    ckpt = torch.load(args.input, map_location='cpu')
    models = ckpt['models']
    
    # 1. Background
    if 'Background' in models:
        bg_model = convert_background(models['Background'])
        torch.save((bg_model, 30000), os.path.join(args.output_dir, 'scene.pth'))
        print("Saved scene.pth")
        
    # 2. Dynamic Objects (RigidNodes)
    if 'RigidNodes' in models:
        rigid_state = models['RigidNodes']
        point_ids = rigid_state['point_ids'][:, 0]
        unique_ids = torch.unique(point_ids)
        
        meta_data = {"camera_model": "OPENCV", "verts": {}}
        
        for obj_id in unique_ids:
            obj_id_str = f"obj_{int(obj_id)}"
            dyn_model = convert_dynamic(rigid_state, obj_id, point_ids)
            torch.save((dyn_model, 30000), os.path.join(args.output_dir, f'dynamic_{obj_id_str}.pth'))
            # Dummy verts for meta_data to prevent planner crash
            meta_data["verts"][obj_id_str] = [[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [0.5, -0.5, 0.5], [-0.5, -0.5, -0.5]]
            print(f"Saved dynamic_{obj_id_str}.pth")
            
        with open(os.path.join(args.output_dir, 'meta_data.json'), 'w') as f:
            json.dump(meta_data, f)
            print("Saved meta_data.json")

    # 3. Create cfg.yaml
    cfg = {
        "model": {"sh_degree": 3, "data_device": "cpu"},
        "ground": {"n_sample": 10, "grid_len": 0.2, "min": -2, "range": 4},
        "train": {"iterations": 30000},
        "model_path": args.output_dir,
        "semantic": True,
        "affine": True,
        "unicycle": True,
        "ignore_dynamic": False,
        "data_type": "pandaset"
    }
    with open(os.path.join(args.output_dir, 'cfg.yaml'), 'w') as f:
        yaml.dump(cfg, f)
        print("Saved cfg.yaml")

if __name__ == "__main__":
    main()
