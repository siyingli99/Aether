# AETHER

[中文说明](README_ZH.md)

AETHER is an editable 3D Gaussian driving simulator. It reconstructs driving
scenes, replaces their dynamic foreground with external vehicle assets, lets
LimSim control background traffic, repairs all six camera views with Difix,
and runs an end-to-end planner such as UniAD in a closed loop.

## What is included

- DriveStudio-based 3D scene reconstruction and novel-view rendering.
- Camera and Gaussian point-cloud simulation.
- Scene editing: remove, restore, replace, and place dynamic actors.
- Portable Gaussian vehicle assets converted from 3DRealCar.
- LimSim traffic on nuScenes HD-map topology.
- A generic end-to-end adapter plus a ready-to-run UniAD adapter.
- Mandatory six-camera Difix repair before every planner input.
- A synchronized `LimSim 2D | AETHER 3D` result video and audit files.

The public evaluation-protocol and algorithm-diagnosis experiments are not
part of this release.

## Tested minimum loop

```text
nuScenes reconstruction + HD map
        -> remove original dynamic foreground
        -> LimSim traffic poses
        -> 3DRealCar Gaussian vehicles
        -> six AETHER cameras
        -> six-view Difix repair
        -> UniAD trajectory
        -> ego update and next frame
```

Two complete examples are provided:

| Scene | DriveStudio index | Frames | HD-map traffic |
|---|---:|---:|---|
| nuScenes `scene-0383` | `300` | `39` | LimSim |
| nuScenes `scene-0088` | `88` | `39` | LimSim |

The Hugging Face dataset also retains `data.zip` for PandaSet 095. PandaSet
095 has no HD-map package here, so it supports reconstruction/direct closed
loop but not the provided LimSim topology loop.

## 1. Clone

```bash
git clone https://github.com/siyingli99/Aether.git
cd Aether
```

All following commands run from the repository root on Ubuntu/WSL2.

## 2. Create the AETHER environment

```bash
conda create -p .conda/aether python=3.10 -y
conda activate ./.conda/aether
```

Install the tested CUDA 12.8 PyTorch stack:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

Install AETHER and rendering dependencies:

```bash
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install -e third_party/drivestudio/third_party/smplx
```

The SplatAD rasterizer is vendored and imported directly. Point-cloud
simulation compiles its CUDA extension on first use; no separate SplatAD
install command is required.

Verify the code environment:

```bash
python scripts/check_install.py
```

## 3. Download the demo data

The demo repository is `mikkochi/AETHER_Demo` on Hugging Face. Log in once if
the repository is private:

```bash
huggingface-cli login
```

Download both nuScenes scenes and both 3DRealCar assets. The script verifies
every archive before extracting it into the expected `data/` layout:

```bash
python scripts/download_demo.py --scene all
```

Download only scene 0383 plus the vehicle assets:

```bash
python scripts/download_demo.py --scene 0383
```

Validate the extracted scene without loading CUDA kernels:

```bash
python scripts/check_demo.py --scene 0383
```

The resulting layout is:

```text
data/
├── checkpoints/nuscenes_outputs/scene_recon_full/
│   ├── scene_300_6cams_with_humans/
│   └── scene_88_6cams_with_humans/
├── nuscenes/processed_10Hz/trainval/{300,088}/
├── limsim/nuscenes_{0383,088}/
└── assets/vehicles/
    ├── 3drealcar_2024_06_04_15_34_51/
    └── 3drealcar_2024_06_04_13_44_39/
```

Both vehicle assets are converted directly from the named
[3DRealCar](https://xiaobiaodu.github.io/3drealcar/) sequences. Neither asset
is extracted from the nuScenes reconstructions.

## 4. Install the AETHER LimSim bridge

LimSim remains a separate GPL-3.0 process. Clone the tested integration branch
next to AETHER:

```bash
git clone --branch aether-integration \
  https://github.com/siyingli99/LimSim.git ../LimSim
```

Create its isolated environment:

```bash
conda create -p .conda/limsim python=3.10 -y
conda run -p .conda/limsim \
  pip install -r ../LimSim/aether_bridge/requirements.txt
```

The runner starts and stops the headless LimSim/SUMO sidecar automatically.

## 5. Install UniAD

Create a separate legacy OpenMMLab environment:

```bash
conda create -p .conda/uniad python=3.10 -y
conda activate ./.conda/uniad
```

Install the UniAD runtime:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install openmim
MMCV_WITH_OPS=1 pip install -v mmcv-full==1.7.2
pip install mmdet==2.26.0 mmdet3d==1.0.0rc6 mmsegmentation==0.29.1
pip install -r third_party/uniad_sim/requirements.txt
```

Download the official UniAD end-to-end checkpoint:

```bash
mkdir -p checkpoints
wget -O checkpoints/uniad_base_e2e.pth \
  https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth
```

Return to AETHER:

```bash
conda activate ./.conda/aether
```

## 6. Run a four-step smoke loop

Scene 0383:

```bash
AETHER_SCENARIO=configs/scenarios/nuscenes_0383_smoke.yaml \
  bash scripts/run_closed_loop.sh 0383 --no_io_trace --refine_max_side 512
```

Scene 088:

```bash
AETHER_SCENARIO=configs/scenarios/nuscenes_088_smoke.yaml \
  bash scripts/run_closed_loop.sh 088 --no_io_trace --refine_max_side 512
```

Do not pass `--no_refine`: these scenarios fail closed if any camera misses
Difix repair.

## 7. Run the full loop

```bash
bash scripts/run_closed_loop.sh 0383 --no_io_trace
```

Use another CUDA device for AETHER/UniAD/Difix:

```bash
AETHER_CUDA=1 bash scripts/run_closed_loop.sh 0383 --no_io_trace
```

Important outputs are written below
`aether_results/nuscenes/<scenario>/uniad/limsim_camera/`:

```text
limsim_aether.mp4          synchronized full-map 2D and six-camera 3D video
video.mp4                  six-camera AETHER video
data.pkl                   closed-loop trajectory
refine_audit.json          per-frame/per-camera Difix hashes
dynamic_removal_audit.json removed original foreground and external slots
perf.json                  timing and invariant summary
```

## 8. Tune the loop

Copy a scenario before changing it:

```bash
cp configs/scenarios/nuscenes_0383.yaml configs/scenarios/my_scene.yaml
```

Select the second 3DRealCar vehicle:

```yaml
traffic:
  asset_path: ${aether:data/assets/vehicles/3drealcar_2024_06_04_13_44_39}
```

Change traffic density and Gaussian cost:

```yaml
traffic:
  max_vehicles: 6
  max_gaussians: 30000
```

Change loop length and road-departure tolerance:

```yaml
termination:
  max_steps: 80
  off_scene_radius: 6.0
  off_scene_streak: 5
```

Tune Difix speed/quality from the command line:

```bash
bash scripts/run_closed_loop.sh 0383 \
  --refine_max_side 768 --refine_timestep 199 --refine_dtype bf16
```

## 9. Reconstruct a new scene

Preprocess nuScenes with the vendored DriveStudio tools, then launch training.
The `scene-0383` sample is DriveStudio index `300`:

```bash
python scripts/train_scene.py \
  --config_file third_party/drivestudio/configs/omnire.yaml \
  --output_root data/checkpoints/nuscenes_outputs \
  --project scene_recon_full \
  --run_name scene_300_6cams_with_humans \
  dataset=nuscenes/6cams \
  data.data_root=data/nuscenes/processed_10Hz/trainval \
  data.scene_idx=300
```

The wrapper forwards additional OmegaConf overrides directly to DriveStudio.

## 10. Point-cloud simulation

Train a frozen-geometry SplatAD sidecar from an RGB checkpoint:

```bash
python aether/train/joint_phase.py \
  --config_file third_party/drivestudio/configs/omnire.yaml \
  --resume_from data/checkpoints/nuscenes_outputs/scene_recon_full/scene_300_6cams_with_humans/checkpoint_final.pth \
  --output_root outputs/dualmodal --project aether_joint/nuscenes \
  --run_name scene300_joint --num_iters 10000 --freeze_geometry \
  dataset=nuscenes/6cams \
  data.data_root=data/nuscenes/processed_10Hz/trainval data.scene_idx=300
```

Render RGB, projected points, and BEV from the sidecar:

```bash
python aether/tools/viz_joint.py \
  --resume_from data/checkpoints/nuscenes_outputs/scene_recon_full/scene_300_6cams_with_humans/checkpoint_final.pth \
  --lidar_sidecar outputs/dualmodal/aether_joint/nuscenes/scene300_joint/lidar_sidecar_frozen.pt \
  --frame_idx 0 --cam_idx 0 --out_png outputs/nuscenes_0383_lidar.png
```

The camera-only LimSim examples intentionally keep `lidar.enabled: false`.

## 11. Scene editing and external assets

Convert any reconstructed 3DRealCar sequence (`gs.pth` + `wlh.json`) to the
portable AETHER format:

```bash
python tools/build_vehicle_asset.py \
  --input /path/to/3DRealCar/<sequence> \
  --output data/assets/vehicles/<name> \
  --sequence-id <sequence> --max-gaussians 50000
```

Original dynamic `RigidNodes`, `DeformableNodes`, and `SMPLNodes` are hidden
when `traffic.hide_original_dynamic: true`. AETHER then appends reusable
vehicle slots and updates their poses from LimSim at every timestamp. The
audit file proves that the old actors remained hidden throughout the loop.

Open the YAML/BEV scene editor for manually scripted scenarios:

```bash
python -m aether.tools.scenario_editor configs/scenarios/nuscenes_0383.yaml
```

## 12. Add another end-to-end algorithm

Copy the UniAD adapter declaration:

```bash
cp configs/adapters/uniad.yaml configs/adapters/my_planner.yaml
```

Edit `launcher`, `observation`, and `output`, then run:

```bash
python scripts/closed_loop.py \
  --scenario configs/scenarios/nuscenes_0383.yaml \
  --camera configs/cameras/nuscenes.yaml \
  --kinematic configs/kinematic.yaml \
  --ad my_planner --ad_config configs/adapters/my_planner.yaml
```

The current IPC contract is HUGSIM-compatible pickle over `obs_pipe` and
`plan_pipe`. Planner waypoints are `(lateral_right, forward)`; the controller
performs the required right/left-handed conversion before applying steering.

## Repository layout

```text
aether/                  simulator, rendering, traffic, sensors, adapters
configs/                 camera, scenario, and adapter declarations
scripts/                 download, validation, training, and loop entrypoints
tools/                   data/map/vehicle conversion tools
third_party/drivestudio  3D reconstruction and rendering backend
third_party/difix3d      in-loop image repair bridge
third_party/uniad_sim    end-to-end planner adapter
third_party/splatad_gsplat point-cloud rasterizer
data/                    downloaded data; never committed to Git
```

## Licenses and data

AETHER is released under the repository license. Third-party code keeps its
original license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
nuScenes, 3DRealCar, UniAD weights, Difix weights, and LimSim are distributed
under their respective terms. The two supplied vehicle assets are converted
from the 3DRealCar sequences recorded in each asset's `metadata.json`.
