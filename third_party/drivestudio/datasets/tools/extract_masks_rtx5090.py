"""
@file   extract_masks_rtx5090.py
@author Modified for RTX 5090 compatibility
@brief  Extract semantic mask using modern transformers library
"""

import os
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

# Cityscapes semantic classes
semantic_classes = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
    'bicycle'
]

dataset_classes_in_semantic = {
    'Vehicle': [13, 14, 15],   # 'car', 'truck', 'bus'
    'human': [11, 12, 17, 18], # 'person', 'rider', 'motorcycle', 'bicycle'
}

class ModernSegFormer:
    """Modern SegFormer wrapper using transformers library"""
    
    def __init__(self, device='cuda'):
        print(f"Initializing SegFormer model on {device}...")
        self.device = device
        
        # Use HuggingFace pretrained model (Cityscapes)
        model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        
        # Load model with use_safetensors to avoid legacy pytorch_model.bin issues
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            use_safetensors=True  # Force using safetensors format
        )
        
        self.model.to(device)
        self.model.eval()
        
        print(f"✓ Model loaded successfully")
        if device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
    
    def inference_segmentor(self, image_path):
        """
        Perform semantic segmentation inference
        Returns: segmentation mask as numpy array (H, W) with class IDs
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_size = image.size  # (W, H)
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
        
        # Upsample to original size
        logits = torch.nn.functional.interpolate(
            logits,
            size=(original_size[1], original_size[0]),  # (H, W)
            mode='bilinear',
            align_corners=False
        )
        
        # Get predicted class for each pixel
        pred_seg = logits.argmax(dim=1)[0].cpu().numpy()
        
        return pred_seg


if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Custom configs
    parser.add_argument('--data_root', type=str, default='data/waymo/processed/training')
    parser.add_argument(
        "--scene_ids",
        default=None,
        type=int,
        nargs="+",
        help="scene ids to be processed",
    )
    parser.add_argument(
        "--split_file", type=str, default=None, help="Split file"
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="If no scene id or split_file is given, use start_idx and num_scenes",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=200,
        help="number of scenes to be processed",
    )
    parser.add_argument(
        '--process_dynamic_mask',
        action='store_true',
        help="Whether to process dynamic masks",
    )
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ignore_existing', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    
    args = parser.parse_args()
    
    # Parse scene IDs
    if args.scene_ids is not None:
        scene_ids_list = args.scene_ids
    elif args.split_file is not None:
        with open(args.split_file, "r") as f:
            split_file = f.readlines()[1:]  # Skip header
        if "kitti" in args.split_file or "nuplan" in args.split_file:
            scene_ids_list = [line.strip().split(",")[0] for line in split_file]
        else:
            try:
                scene_ids_list = [int(line.strip().split(",")[0]) for line in split_file]
            except:
                scene_ids_list = [line.strip().split(",")[0] for line in split_file]
    else:
        scene_ids_list = np.arange(args.start_idx, args.start_idx + args.num_scenes)
    
    print(f"\n{'='*80}")
    print(f"Sky Mask Extraction Configuration:")
    print(f"  Data root: {args.data_root}")
    print(f"  Number of scenes: {len(scene_ids_list)}")
    print(f"  Device: {args.device}")
    print(f"  Process dynamic masks: {args.process_dynamic_mask}")
    print(f"{'='*80}\n")
    
    # Check CUDA availability
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        print("⚠️  WARNING: CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Initialize model
    model_wrapper = ModernSegFormer(device=args.device)
    
    # Process each scene
    for scene_i, scene_id in enumerate(tqdm(scene_ids_list, desc='Extracting Masks')):
        scene_id = str(scene_id).zfill(3)
        img_dir = os.path.join(args.data_root, scene_id, args.rgb_dirname)
        
        if not os.path.exists(img_dir):
            print(f"⚠️  Warning: {img_dir} does not exist, skipping")
            continue
        
        # Create sky mask directory
        sky_mask_dir = os.path.join(args.data_root, scene_id, "sky_masks")
        os.makedirs(sky_mask_dir, exist_ok=True)
        
        # Create dynamic mask directories if needed
        if args.process_dynamic_mask:
            rough_human_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "human")
            rough_vehicle_mask_dir = os.path.join(args.data_root, scene_id, "dynamic_masks", "vehicle")
            
            all_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "all")
            os.makedirs(all_mask_dir, exist_ok=True)
            human_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "human")
            os.makedirs(human_mask_dir, exist_ok=True)
            vehicle_mask_dir = os.path.join(args.data_root, scene_id, "fine_dynamic_masks", "vehicle")
            os.makedirs(vehicle_mask_dir, exist_ok=True)
        
        # Get all image files
        flist = sorted(glob(os.path.join(img_dir, '*')))
        
        # Process each image
        for fpath in tqdm(flist, desc=f'Scene [{scene_id}]', leave=False):
            fbase = os.path.splitext(os.path.basename(fpath))[0]
            sky_mask_path = os.path.join(sky_mask_dir, f"{fbase}.png")
            
            # Skip if already exists
            if args.ignore_existing and os.path.exists(sky_mask_path):
                continue
            
            try:
                # Perform inference
                mask = model_wrapper.inference_segmentor(fpath)
                
                # Save sky mask (class 10 in Cityscapes)
                sky_mask = np.isin(mask, [10])
                imageio.imwrite(sky_mask_path, sky_mask.astype(np.uint8) * 255)
                
                # Process dynamic masks if requested
                if args.process_dynamic_mask:
                    # Human masks
                    rough_human_mask_path = os.path.join(rough_human_mask_dir, f"{fbase}.png")
                    if os.path.exists(rough_human_mask_path):
                        rough_human_mask = (imageio.imread(rough_human_mask_path) > 0)
                        human_mask = np.isin(mask, dataset_classes_in_semantic['human'])
                        valid_human_mask = np.logical_and(human_mask, rough_human_mask)
                        imageio.imwrite(
                            os.path.join(human_mask_dir, f"{fbase}.png"),
                            valid_human_mask.astype(np.uint8) * 255
                        )
                    else:
                        valid_human_mask = np.zeros_like(mask, dtype=bool)
                    
                    # Vehicle masks
                    rough_vehicle_mask_path = os.path.join(rough_vehicle_mask_dir, f"{fbase}.png")
                    if os.path.exists(rough_vehicle_mask_path):
                        rough_vehicle_mask = (imageio.imread(rough_vehicle_mask_path) > 0)
                        vehicle_mask = np.isin(mask, dataset_classes_in_semantic['Vehicle'])
                        valid_vehicle_mask = np.logical_and(vehicle_mask, rough_vehicle_mask)
                        imageio.imwrite(
                            os.path.join(vehicle_mask_dir, f"{fbase}.png"),
                            valid_vehicle_mask.astype(np.uint8) * 255
                        )
                    else:
                        valid_vehicle_mask = np.zeros_like(mask, dtype=bool)
                        
                    # Combined dynamic mask
                    valid_all_mask = np.logical_or(valid_human_mask, valid_vehicle_mask)
                    imageio.imwrite(
                        os.path.join(all_mask_dir, f"{fbase}.png"),
                        valid_all_mask.astype(np.uint8) * 255
                    )
                
            except Exception as e:
                print(f"❌ Error processing {fpath}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n{'='*80}")
    print("✅ Mask extraction completed successfully!")
    print(f"{'='*80}\n")
