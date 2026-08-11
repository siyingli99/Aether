# AETHER

[中文](README_ZH.md)

AETHER is an editable driving-scene pipeline built around 3D Gaussian
Splatting (3DGS). It reconstructs scenes, edits actors, synthesizes RGB and
point-cloud novel views, and connects enhanced multimodal observations to
closed-loop driving and diagnosis.

## Features
- 3DGS scene reconstruction from processed driving data.
- Scenario editing with actor placement, replay, and termination settings.
- RGB and point-cloud novel-view synthesis.
- Mode-A geometric LiDAR from configured scene geometry.
- Phase-B SplatAD LiDAR range, intensity, and ray-drop prediction.
- Difix RGB enhancement in the main environment.
- Multimodal RGB-LiDAR closed-loop rollout and corpus generation.
- Reverse distillation from enhanced closed-loop observations.
- UniAD end-to-end closed-loop driving.
- Semantic-causal algorithm diagnosis with mock and real rollouts.

## Core layout
```text
aether/                 Core rendering, training, refinement, and diagnosis
configs/                Base, camera, scenario, adapter, and registry configs
scripts/                Installation checks and executable pipelines
tools/                  Data and scene utilities
third_party/drivestudio RGB 3DGS reconstruction runtime
third_party/splatad_gsplat
                        SplatAD LiDAR rasterizer
third_party/difix3d     Difix RGB integration
third_party/uniad_sim   UniAD closed-loop adapter
data/                   Processed data and RGB checkpoints
assets/local/           Optional local processed archives
outputs/                Training, visualization, and distillation outputs
aether_results/         Closed-loop rollouts and evaluation inputs
```

## 1. Main environment

Run every command from the repository root. Use one `aether` environment for
RGB reconstruction, Difix, LiDAR, visualization, distillation, and diagnosis.

```bash
conda create -n aether python=3.10 -y
conda activate aether
```

Install PyTorch for CUDA 12.8, then the Python dependencies:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Install the RGB rendering dependencies:

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install -e third_party/drivestudio/third_party/smplx
```

Difix runs directly in `aether`; its Python packages are included in
`requirements.txt`.

The SplatAD CUDA extension is loaded from `third_party/splatad_gsplat`. Its
kernels are JIT-compiled on first use, so the first LiDAR command takes longer.
Keep `nvcc` and `ninja` available:

```bash
conda install -n aether -c conda-forge ninja -y
```

## 2. Data preparation
The minimum scene-095 runtime layout is:

```text
data/
├── checkpoints/pandaset_outputs/scene_recon_full/
│   └── scene_095_6cams_with_humans/
│       ├── checkpoint_final.pth
│       └── config.yaml
├── pandaset/processed/095/
│   ├── ego_pose/
│   └── extrinsics/
└── exported_scenes/pandaset/pandaset_095_final/
    └── ground_param.pkl
```

After placing these assets, run `python scripts/check_install.py`.

For the complete processed scene-095 ZIP, place
`pandaset095_processed.zip` under `assets/local/`, then extract it:

```bash
python scripts/extract_pandaset095.py \
  --archive assets/local/pandaset095_processed.zip \
  --destination data/pandaset/processed
```

Optional: download the complete PandaSet archive with Kaggle:

```bash
pip install kaggle
mkdir -p data/pandaset/raw
kaggle datasets download pz19930809/pandaset -p data/pandaset/raw
unzip data/pandaset/raw/pandaset-dataset.zip -d data/pandaset/raw
```

Install the vendored PandaSet toolkit and preprocess only scene 095:

```bash
pip install -e third_party/drivestudio/pandaset-devkit/python
python third_party/drivestudio/datasets/preprocess.py \
  --data_root data/pandaset/raw --dataset pandaset \
  --target_dir data/pandaset/processed --workers 32 \
  --scene_ids 95 \
  --process_keys images lidar calib pose dynamic_masks objects
```

## 3. RGB smoke test
This renders one frame from the default RGB checkpoint:

```bash
python scripts/render_smoke_test.py \
  --scenario configs/scenarios/pandaset_095_example.yaml \
  --base configs/base.yaml --camera configs/cameras/pandaset.yaml \
  --kinematic configs/kinematic.yaml \
  --frame 0 --out outputs/smoke.png
```

## 4. Scenario editor
Open the BEV editor:

```bash
python -m aether.tools.scenario_editor \
  configs/scenarios/pandaset_095_example.yaml
```

Save the edited scenario from the editor, then pass its YAML to smoke,
closed-loop, or diagnosis commands.

## 5. Phase A: RGB 3DGS reconstruction
Full processed scene data is required. The wrapper forwards remaining
OmegaConf overrides to DriveStudio:

```bash
python scripts/train_scene.py \
  --config_file third_party/drivestudio/configs/omnire.yaml \
  --output_root data/checkpoints/pandaset_outputs \
  --project scene_recon_full \
  --run_name scene_095_6cams_with_humans \
  dataset=pandaset/6cams \
  data.data_root=data/pandaset/processed \
  data.scene_idx=95
```

The final checkpoint is
`data/checkpoints/pandaset_outputs/scene_recon_full/scene_095_6cams_with_humans/checkpoint_final.pth`.

## 6. Phase B: frozen-geometry SplatAD LiDAR
Use the default scene-095 RGB checkpoint and freeze RGB geometry:

```bash
python aether/train/joint_phase.py \
  --config_file third_party/drivestudio/configs/omnire.yaml \
  --resume_from data/checkpoints/pandaset_outputs/scene_recon_full/scene_095_6cams_with_humans/checkpoint_final.pth \
  --output_root outputs/dualmodal \
  --project aether_joint/pandaset --run_name scene095_joint \
  --num_iters 10000 --freeze_geometry \
  dataset=pandaset/6cams \
  data.data_root=data/pandaset/processed data.scene_idx=95
```

The final convenience alias matches `configs/refine/scene_registry.yaml`:
`outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt`.

Frozen Phase B keeps the original RGB checkpoint for RGB rendering. The
sidecar stores only the LiDAR head and per-Gaussian LiDAR attributes.
Make that RGB checkpoint available under the registry joint directory:

```bash
cp data/checkpoints/pandaset_outputs/scene_recon_full/scene_095_6cams_with_humans/checkpoint_final.pth \
  outputs/dualmodal/aether_joint/pandaset/scene095_joint/checkpoint_final.pth
```

## 7. LiDAR refiner v2
Run this only after the Phase-B sidecar exists. The registry reads the joint
directory at `outputs/dualmodal/aether_joint/pandaset/scene095_joint`.

```bash
bash scripts/run_lidar_refiner_v2_pipeline.sh all
```

The four stages generate real-render pairs, merge manifests, and train the
refiner. The final bundle is
`outputs/multimodal_refine_v2/unified/unified_multimodal.bundle.pth`.

## 8. Novel-view RGB and point-cloud visualization
Use the original RGB checkpoint with the frozen sidecar.

Render one RGB/LiDAR comparison:

```bash
python aether/tools/viz_joint.py \
  --resume_from data/checkpoints/pandaset_outputs/scene_recon_full/scene_095_6cams_with_humans/checkpoint_final.pth \
  --lidar_sidecar outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt \
  --frame_idx 0 --cam_idx 0 \
  --out_png outputs/pandaset095_joint.png
```

Render six-camera RGB, projected point clouds, BEV, and comparison videos:

```bash
python aether/tools/viz_joint_video.py \
  --resume_from data/checkpoints/pandaset_outputs/scene_recon_full/scene_095_6cams_with_humans/checkpoint_final.pth \
  --lidar_sidecar outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt \
  --out_dir outputs/pandaset095_joint_video \
  --scene_tag "PandaSet 095" --fps 10
```

## 9. Multimodal closed loop
Run without an end-to-end planner by setting `--ad none`. The command applies
the Phase-B sidecar and LiDAR refiner bundle while following a built-in
straight trajectory:

```bash
python scripts/closed_loop_multimodal.py \
  --scenario configs/scenarios/pandaset_095_multimodal.yaml \
  --base configs/base.yaml --camera configs/cameras/pandaset_800x450.yaml \
  --kinematic configs/kinematic.yaml --ad none \
  --lidar_sidecar outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt \
  --lidar_bundle outputs/multimodal_refine_v2/unified/unified_multimodal.bundle.pth \
  --max_steps 12
```

The output directory is
`aether_results/pandaset/pandaset_095_multimodal/none_multimodal/easy/`.

It contains `camera_trajectory.json`, videos, panels, and `distill_corpus/`
with enhanced RGB and LiDAR observations.

## 10. Reverse distillation
Use the rollout directory produced by the previous command:

```bash
python aether/refine/distill_from_closed_loop.py \
  --closed_loop_dir aether_results/pandaset/pandaset_095_multimodal/none_multimodal/easy \
  --joint_dir outputs/dualmodal/aether_joint/pandaset/scene095_joint \
  --lidar_sidecar outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt \
  --refiner_bundle outputs/multimodal_refine_v2/unified/unified_multimodal.bundle.pth \
  --out outputs/distilled/pandaset095
```

This writes `background_distilled.pth`, `sidecar_distilled.pt`, and
`distill_meta.json`. Re-render the trajectory with the distilled artifacts:

```bash
python scripts/render_distilled_trajectory.py \
  --trajectory aether_results/pandaset/pandaset_095_multimodal/none_multimodal/easy/camera_trajectory.json \
  --joint_dir outputs/dualmodal/aether_joint/pandaset/scene095_joint \
  --distill_dir outputs/distilled/pandaset095 \
  --closed_loop_dir aether_results/pandaset/pandaset_095_multimodal/none_multimodal/easy \
  --out outputs/distilled/pandaset095/replay
```

## 11. Algorithm diagnosis
Run the self-contained mock diagnosis first:

```bash
python scripts/diagnose.py \
  --mock --output outputs/diagnosis/mock_report.json
```

Run diagnosis against the real UniAD adapter after installing it:

```bash
python scripts/diagnose.py \
  --scenario configs/scenarios/pandaset_095_aether.yaml \
  --base configs/base.yaml --camera configs/cameras/pandaset_800x450.yaml \
  --kinematic configs/kinematic.yaml \
  --ad uniad --ad_cuda 0 \
  --output outputs/diagnosis/pandaset095_uniad.json
```

## 12. UniAD environment and closed loop
UniAD is the only maintained end-to-end driving adapter. It uses the separate
prefix environment expected by `third_party/uniad_sim/tools/e2e.sh`.

```bash
conda create -p .conda/uniad python=3.10 -y
conda activate ./.conda/uniad
```

Install PyTorch and the OpenMMLab stack:

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install openmim
MMCV_WITH_OPS=1 pip install -v "mmcv-full==1.7.2"
pip install "mmdet==2.26.0"
pip install "mmdet3d==1.0.0rc6"
pip install "mmsegmentation==0.29.1"
pip install -r third_party/uniad_sim/requirements.txt
```

Download the official v1.0.1 end-to-end checkpoint:

```bash
mkdir -p checkpoints
wget -O checkpoints/uniad_base_e2e.pth \
  https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth
```

Return to the main environment and run the closed loop:

```bash
conda activate aether
python scripts/closed_loop.py \
  --scenario configs/scenarios/pandaset_095_aether.yaml \
  --base configs/base.yaml --camera configs/cameras/pandaset_800x450.yaml \
  --kinematic configs/kinematic.yaml \
  --ad uniad --ad_cuda 0
```

The default output directory is
`aether_results/pandaset/pandaset_095_aether/uniad/easy/`.

## 13. Output evaluation
`aether_eval.py` accepts a rollout containing `data.pkl`, a comma-separated
metric list, and an optional JSON output:

```bash
python scripts/aether_eval.py \
  aether_results/pandaset/pandaset_095_aether/uniad/easy \
  --metrics hugsim,navsim_pdms,bench2drive_ds \
  --out outputs/pandaset095_eval.json
```

## Current default asset boundary
Scene 095 includes the RGB checkpoint and the assets needed by Mode-A LiDAR.
It does not include a Phase-B sidecar, LiDAR refiner bundle, or
`distill_corpus`. Generate those artifacts with sections 6, 7, and 9.
