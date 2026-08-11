# AETHER

[English](README.md)

AETHER 是基于 3D Gaussian Splatting（3DGS）的可编辑驾驶场景流水线，用于场景重建与编辑、RGB 和点云新视角合成、多模态闭环及算法诊断。

## 功能
- 由处理后的驾驶数据进行 3DGS 场景重建。
- 编辑参与者位置、回放策略和终止条件。
- 合成 RGB 与点云新视角。
- 基于场景几何的 Mode-A LiDAR。
- Phase-B SplatAD LiDAR 距离、强度和丢光预测。
- 在主环境中运行 Difix RGB 增强。
- RGB-LiDAR 多模态闭环并生成训练语料。
- 从增强后的闭环观测进行反向蒸馏。
- UniAD 端到端闭环驾驶。
- 支持 mock 和真实闭环的语义因果算法诊断。

## 核心目录
```text
aether/                 渲染、训练、增强和诊断核心代码
configs/                基础、相机、场景、适配器和注册表配置
scripts/                安装检查和可执行流水线
tools/                  数据与场景工具
third_party/drivestudio RGB 3DGS 重建运行时
third_party/splatad_gsplat
                        SplatAD LiDAR 光栅器
third_party/difix3d     Difix RGB 集成
third_party/uniad_sim   UniAD 闭环适配器
data/                   处理后数据与 RGB checkpoint
assets/local/           可选本地 processed 压缩包
outputs/                训练、可视化和蒸馏输出
aether_results/         闭环输出与评测输入
```

## 1. 主环境
所有命令都从仓库根目录执行。RGB 重建、Difix、LiDAR 训练、可视化、蒸馏和诊断统一使用 `aether` 环境。

```bash
conda create -n aether python=3.10 -y
conda activate aether
```

安装 CUDA 12.8 版 PyTorch 和 Python 依赖：

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

安装 RGB 渲染依赖：

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install git+https://github.com/facebookresearch/pytorch3d.git
pip install -e third_party/drivestudio/third_party/smplx
```

Difix 直接在 `aether` 中运行，相关 Python 包已包含在 `requirements.txt`。

SplatAD CUDA 扩展从 `third_party/splatad_gsplat` 加载，首次执行 Phase B、LiDAR 可视化或闭环时会 JIT 编译 CUDA 内核，因此第一次较慢。确保 `nvcc` 和 `ninja` 可用：

```bash
conda install -n aether -c conda-forge ninja -y
python scripts/check_install.py
```

## 2. 数据准备
默认最小 095 资产路径如下：

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

放置资产后检查：

```bash
python scripts/check_install.py
```

如有完整 scene-095 processed ZIP，将其放在 `assets/local/` 后解压：

```bash
python scripts/extract_pandaset095.py \
  --archive assets/local/pandaset095_processed.zip \
  --destination data/pandaset/processed
```

可选：通过 Kaggle 下载完整 PandaSet：

```bash
pip install kaggle
mkdir -p data/pandaset/raw
kaggle datasets download pz19930809/pandaset -p data/pandaset/raw
unzip data/pandaset/raw/pandaset-dataset.zip -d data/pandaset/raw
```

安装仓库内 PandaSet 工具并仅预处理 095：

```bash
pip install -e third_party/drivestudio/pandaset-devkit/python
python third_party/drivestudio/datasets/preprocess.py \
  --data_root data/pandaset/raw --dataset pandaset \
  --target_dir data/pandaset/processed --workers 32 \
  --scene_ids 95 \
  --process_keys images lidar calib pose dynamic_masks objects
```

## 3. RGB smoke
从默认 RGB checkpoint 渲染一帧：

```bash
python scripts/render_smoke_test.py \
  --scenario configs/scenarios/pandaset_095_example.yaml \
  --base configs/base.yaml --camera configs/cameras/pandaset.yaml \
  --kinematic configs/kinematic.yaml \
  --frame 0 --out outputs/smoke.png
```

## 4. 场景编辑器
打开 BEV 编辑器：

```bash
python -m aether.tools.scenario_editor \
  configs/scenarios/pandaset_095_example.yaml
```

在编辑器中保存 YAML 后，可将其传给 smoke、闭环或诊断命令。

## 5. Phase A：RGB 3DGS 重建
该阶段需要完整 processed 场景数据。包装脚本会把其余 OmegaConf 参数传给 DriveStudio：

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

最终 RGB checkpoint 应位于：

```text
data/checkpoints/pandaset_outputs/scene_recon_full/scene_095_6cams_with_humans/checkpoint_final.pth
```

## 6. Phase B：冻结几何的 SplatAD LiDAR
使用默认 095 RGB checkpoint，并冻结 RGB 几何：

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

最终 alias 与 `configs/refine/scene_registry.yaml` 一致：

```text
outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt
```

冻结 Phase B 后，所有 RGB 渲染继续使用原始 RGB checkpoint；sidecar 只保存 LiDAR head 和逐 Gaussian LiDAR 属性。
将该 RGB checkpoint 放到注册表使用的 joint 目录：

```bash
cp data/checkpoints/pandaset_outputs/scene_recon_full/scene_095_6cams_with_humans/checkpoint_final.pth \
  outputs/dualmodal/aether_joint/pandaset/scene095_joint/checkpoint_final.pth
```

## 7. LiDAR refiner v2
先生成 Phase-B sidecar，再运行注册表中的 joint 目录：

```bash
bash scripts/run_lidar_refiner_v2_pipeline.sh all
```

四个步骤依次生成真实渲染配对、合并 manifest 并训练跨模态 refiner。最终 bundle 位于：

```text
outputs/multimodal_refine_v2/unified/unified_multimodal.bundle.pth
```

## 8. RGB 与点云新视角可视化
`--resume_from` 使用原始 RGB checkpoint，LiDAR 使用 frozen sidecar。

单帧 RGB/LiDAR 对比：

```bash
python aether/tools/viz_joint.py \
  --resume_from data/checkpoints/pandaset_outputs/scene_recon_full/scene_095_6cams_with_humans/checkpoint_final.pth \
  --lidar_sidecar outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt \
  --frame_idx 0 --cam_idx 0 \
  --out_png outputs/pandaset095_joint.png
```

六相机 RGB、投影点云、BEV 和对比视频：

```bash
python aether/tools/viz_joint_video.py \
  --resume_from data/checkpoints/pandaset_outputs/scene_recon_full/scene_095_6cams_with_humans/checkpoint_final.pth \
  --lidar_sidecar outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt \
  --out_dir outputs/pandaset095_joint_video \
  --scene_tag "PandaSet 095" --fps 10
```

## 9. 多模态闭环
`--ad none` 使用内置直行轨迹，并加载 Phase-B sidecar 和 refiner bundle：

```bash
python scripts/closed_loop_multimodal.py \
  --scenario configs/scenarios/pandaset_095_multimodal.yaml \
  --base configs/base.yaml \
  --camera configs/cameras/pandaset_800x450.yaml \
  --kinematic configs/kinematic.yaml --ad none \
  --lidar_sidecar outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt \
  --lidar_bundle outputs/multimodal_refine_v2/unified/unified_multimodal.bundle.pth \
  --max_steps 12
```

输出目录为：

```text
aether_results/pandaset/pandaset_095_multimodal/none_multimodal/easy/
```

其中包含 `camera_trajectory.json`、视频、面板，以及保存增强 RGB 和 LiDAR 观测的 `distill_corpus/`。

## 10. 反向蒸馏
使用上一阶段生成的闭环目录：

```bash
python aether/refine/distill_from_closed_loop.py \
  --closed_loop_dir aether_results/pandaset/pandaset_095_multimodal/none_multimodal/easy \
  --joint_dir outputs/dualmodal/aether_joint/pandaset/scene095_joint \
  --lidar_sidecar outputs/dualmodal/aether_joint/pandaset/scene095_joint/lidar_sidecar_frozen.pt \
  --refiner_bundle outputs/multimodal_refine_v2/unified/unified_multimodal.bundle.pth \
  --out outputs/distilled/pandaset095
```

输出 `background_distilled.pth`、`sidecar_distilled.pt` 和 `distill_meta.json`。随后用蒸馏资产重渲染记录轨迹：

```bash
python scripts/render_distilled_trajectory.py \
  --trajectory aether_results/pandaset/pandaset_095_multimodal/none_multimodal/easy/camera_trajectory.json \
  --joint_dir outputs/dualmodal/aether_joint/pandaset/scene095_joint \
  --distill_dir outputs/distilled/pandaset095 \
  --closed_loop_dir aether_results/pandaset/pandaset_095_multimodal/none_multimodal/easy \
  --out outputs/distilled/pandaset095/replay
```

## 11. 算法诊断
先运行无需外部适配器的 mock：

```bash
python scripts/diagnose.py \
  --mock --output outputs/diagnosis/mock_report.json
```

安装 UniAD 后运行真实诊断：

```bash
python scripts/diagnose.py \
  --scenario configs/scenarios/pandaset_095_aether.yaml \
  --base configs/base.yaml \
  --camera configs/cameras/pandaset_800x450.yaml \
  --kinematic configs/kinematic.yaml \
  --ad uniad --ad_cuda 0 \
  --output outputs/diagnosis/pandaset095_uniad.json
```

## 12. UniAD 环境与闭环
UniAD 是唯一维护的端到端驾驶适配器，使用 `third_party/uniad_sim/tools/e2e.sh` 预期的独立 prefix 环境：

```bash
conda create -p .conda/uniad python=3.10 -y
conda activate ./.conda/uniad
```

安装 PyTorch 和 OpenMMLab 栈：

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

下载官方 v1.0.1 端到端 checkpoint：

```bash
mkdir -p checkpoints
wget -O checkpoints/uniad_base_e2e.pth \
  https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth
```

返回主环境运行闭环：

```bash
conda activate aether
python scripts/closed_loop.py \
  --scenario configs/scenarios/pandaset_095_aether.yaml \
  --base configs/base.yaml \
  --camera configs/cameras/pandaset_800x450.yaml \
  --kinematic configs/kinematic.yaml \
  --ad uniad --ad_cuda 0
```

默认输出目录为：

```text
aether_results/pandaset/pandaset_095_aether/uniad/easy/
```

## 13. 输出评测
`aether_eval.py` 接收包含 `data.pkl` 的闭环目录、逗号分隔的指标名和可选 JSON 输出：

```bash
python scripts/aether_eval.py \
  aether_results/pandaset/pandaset_095_aether/uniad/easy \
  --metrics hugsim,navsim_pdms,bench2drive_ds \
  --out outputs/pandaset095_eval.json
```

## 当前默认资产边界
095 包含 RGB checkpoint 和 Mode-A LiDAR 所需资产，不包含 Phase-B sidecar、LiDAR refiner bundle 或 `distill_corpus`。按第 6、7、9 节生成这些产物。
