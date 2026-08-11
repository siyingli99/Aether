
import os
import glob
import imageio
from PIL import Image
import argparse
from tqdm import tqdm
import numpy as np

def create_gif(scene_path, output_path, target_width=480, fps=10):
    image_dir = os.path.join(scene_path, "images")
    if not os.path.exists(image_dir):
        print(f"Skipping {scene_path}: images directory not found")
        return

    # Find all front camera images (*_0.jpg)
    # Pattern: XXX_0.jpg
    image_files = glob.glob(os.path.join(image_dir, "*_0.jpg"))
    
    if not image_files:
        print(f"Skipping {scene_path}: no front camera images found")
        return

    # Sort by frame index
    # Filename format: {frame_idx:03d}_{cam_idx}.jpg
    image_files.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))

    frames = []
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                # Resize
                w_percent = (target_width / float(img.size[0]))
                h_size = int((float(img.size[1]) * float(w_percent)))
                img = img.resize((target_width, h_size), Image.Resampling.LANCZOS)
                frames.append(np.array(img))
        except Exception as e:
            print(f"Error reading {img_path}: {e}")

    if frames:
        # Save GIF
        imageio.mimsave(output_path, frames, fps=fps, loop=0)
        print(f"Saved {output_path}")
    else:
        print(f"No frames loaded for {scene_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate GIF previews for Pandaset scenes")
    parser.add_argument("--data_root", default="data/pandaset/processed", help="Path to processed Pandaset data")
    parser.add_argument("--output_dir", default="docs/pandaset_previews", help="Directory to save GIFs")
    parser.add_argument("--width", type=int, default=480, help="Width of the GIF images")
    parser.add_argument("--fps", type=int, default=10, help="FPS of the GIF")
    
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # List all scene directories
    if not os.path.exists(args.data_root):
        print(f"Data root {args.data_root} does not exist.")
        return

    scene_dirs = sorted([d for d in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, d))])
    
    print(f"Found {len(scene_dirs)} scenes.")

    for scene_id in tqdm(scene_dirs):
        scene_path = os.path.join(args.data_root, scene_id)
        output_path = os.path.join(args.output_dir, f"{scene_id}.gif")
        
        create_gif(scene_path, output_path, target_width=args.width, fps=args.fps)

if __name__ == "__main__":
    main()
