# AETHER

[English](README.md)

AETHER 是一个可编辑的 3D Gaussian 自动驾驶仿真器：使用 DriveStudio
重建三维场景，删除原场景动态前景并导入外部车辆资产，由 LimSim 控制背景
交通流，所有六路相机图像经 Difix 修复后再送入 UniAD 等端到端算法，形成
完整闭环。

## 当前保留的功能

- 基于 DriveStudio 的三维场景重建与新视角渲染。
- 相机仿真与 Gaussian 点云仿真。
- 删除、恢复、替换和放置动态参与者。
- 将 3DRealCar 重建结果转换为可复用 Gaussian 车辆资产。
- 基于 nuScenes 高精地图拓扑的 LimSim 交通流。
- 通用端到端算法接口和可直接使用的 UniAD 适配器。
- 每次算法输入前强制修复六路相机图像；失败时不发送原图。
- 自动生成 `LimSim 二维 | AETHER 三维` 同步视频与验收记录。

本版本暂不包含公开评测协议，也不包含算法诊断代码。

## 已验证的最小闭环

```text
nuScenes 三维重建 + 高精地图
        -> 删除原始动态前景
        -> LimSim 输出背景车辆位姿
        -> 放置 3DRealCar Gaussian 车辆
        -> AETHER 渲染六路相机
        -> Difix 修复全部六路图像
        -> UniAD 输出规划轨迹
        -> 更新自车并进入下一帧
```

提供两套完整示例：

| 场景 | DriveStudio 索引 | 帧数 | 高精地图交通流 |
|---|---:|---:|---|
| nuScenes `scene-0383` | `300` | `39` | LimSim |
| nuScenes `scene-0088` | `88` | `39` | LimSim |

Hugging Face 中原有的 `data.zip` 是 PandaSet 095。该包没有高精地图，能用于
重建和直接闭环，但不能运行这里提供的 LimSim 高精地图交通流闭环。

## 1. 克隆代码

```bash
git clone https://github.com/siyingli99/Aether.git
cd Aether
```

以下命令均在 Ubuntu/WSL2 的仓库根目录执行。

## 2. 创建 AETHER 环境

```bash
conda create -p .conda/aether python=3.10 -y
conda activate ./.conda/aether
```

安装已验证的 CUDA 12.8 PyTorch：

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
```

安装 AETHER 和渲染依赖：

```bash
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install -e third_party/drivestudio/third_party/smplx
```

SplatAD 光栅器已随仓库提供并会直接导入；第一次运行点云仿真时编译 CUDA
扩展，不需要单独安装 SplatAD。

检查代码环境：

```bash
python scripts/check_install.py
```

## 3. 下载示例数据

示例数据位于 Hugging Face 的 `mikkochi/AETHER_Demo`。如果仓库为私有，
先登录一次：

```bash
huggingface-cli login
```

下载 0383、088 和两辆 3DRealCar 车辆资产。脚本会校验每个压缩包的
SHA-256，然后自动解压到代码所需位置：

```bash
python scripts/download_demo.py --scene all
```

只下载 0383 和车辆资产：

```bash
python scripts/download_demo.py --scene 0383
```

不加载 CUDA，快速检查数据是否完整：

```bash
python scripts/check_demo.py --scene 0383
```

解压后的结构如下：

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

两个车辆资产均直接由对应编号的
[3DRealCar](https://xiaobiaodu.github.io/3drealcar/) 序列转换，不从
nuScenes 训练场景中提取。

## 4. 安装 AETHER 版 LimSim

LimSim 以独立 GPL-3.0 进程运行，不复制到 AETHER 仓库中。将验证过的分支
克隆到 AETHER 同级目录：

```bash
git clone --branch aether-integration \
  https://github.com/siyingli99/LimSim.git ../LimSim
```

创建独立环境：

```bash
conda create -p .conda/limsim python=3.10 -y
conda run -p .conda/limsim \
  pip install -r ../LimSim/aether_bridge/requirements.txt
```

闭环入口会自动启动并关闭无界面的 LimSim/SUMO 服务。

## 5. 安装 UniAD

创建独立 OpenMMLab 环境：

```bash
conda create -p .conda/uniad python=3.10 -y
conda activate ./.conda/uniad
```

安装 UniAD 运行依赖：

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install openmim
MMCV_WITH_OPS=1 pip install -v mmcv-full==1.7.2
pip install mmdet==2.26.0 mmdet3d==1.0.0rc6 mmsegmentation==0.29.1
pip install -r third_party/uniad_sim/requirements.txt
```

下载官方端到端权重：

```bash
mkdir -p checkpoints
wget -O checkpoints/uniad_base_e2e.pth \
  https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth
```

回到 AETHER 环境：

```bash
conda activate ./.conda/aether
```

## 6. 运行四步冒烟闭环

0383：

```bash
AETHER_SCENARIO=configs/scenarios/nuscenes_0383_smoke.yaml \
  bash scripts/run_closed_loop.sh 0383 --no_io_trace --refine_max_side 512
```

088：

```bash
AETHER_SCENARIO=configs/scenarios/nuscenes_088_smoke.yaml \
  bash scripts/run_closed_loop.sh 088 --no_io_trace --refine_max_side 512
```

不要传入 `--no_refine`。这两份配置要求每路相机必须成功经过 Difix，否则
闭环立即失败，不会把未经修复的图像传给算法。

## 7. 运行完整闭环

```bash
bash scripts/run_closed_loop.sh 0383 --no_io_trace
```

切换 GPU：

```bash
AETHER_CUDA=1 bash scripts/run_closed_loop.sh 0383 --no_io_trace
```

主要结果位于
`aether_results/nuscenes/<场景>/uniad/limsim_camera/`：

```text
limsim_aether.mp4          LimSim 全地图二维 + AETHER 六相机三维同步视频
video.mp4                  六相机渲染视频
data.pkl                   闭环轨迹
refine_audit.json          每帧每相机 Difix 输入/输出哈希
dynamic_removal_audit.json 原动态前景删除与外部车辆槽位记录
perf.json                  耗时与闭环约束汇总
```

## 8. 常用调参

先复制场景配置：

```bash
cp configs/scenarios/nuscenes_0383.yaml configs/scenarios/my_scene.yaml
```

切换为第二辆 3DRealCar 车辆：

```yaml
traffic:
  asset_path: ${aether:data/assets/vehicles/3drealcar_2024_06_04_13_44_39}
```

修改交通密度和单车 Gaussian 数量：

```yaml
traffic:
  max_vehicles: 6
  max_gaussians: 30000
```

修改闭环步数和偏离道路容差：

```yaml
termination:
  max_steps: 80
  off_scene_radius: 6.0
  off_scene_streak: 5
```

在速度与修复质量之间调节 Difix：

```bash
bash scripts/run_closed_loop.sh 0383 \
  --refine_max_side 768 --refine_timestep 199 --refine_dtype bf16
```

## 9. 重建新场景

先用仓库内 DriveStudio 工具预处理 nuScenes，再训练场景。`scene-0383`
对应 DriveStudio 索引 `300`：

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

该入口会把其他 OmegaConf 参数直接传给 DriveStudio。

## 10. 点云仿真

在 RGB 权重上训练冻结几何的 SplatAD sidecar：

```bash
python aether/train/joint_phase.py \
  --config_file third_party/drivestudio/configs/omnire.yaml \
  --resume_from data/checkpoints/nuscenes_outputs/scene_recon_full/scene_300_6cams_with_humans/checkpoint_final.pth \
  --output_root outputs/dualmodal --project aether_joint/nuscenes \
  --run_name scene300_joint --num_iters 10000 --freeze_geometry \
  dataset=nuscenes/6cams \
  data.data_root=data/nuscenes/processed_10Hz/trainval data.scene_idx=300
```

渲染 RGB、投影点云和 BEV：

```bash
python aether/tools/viz_joint.py \
  --resume_from data/checkpoints/nuscenes_outputs/scene_recon_full/scene_300_6cams_with_humans/checkpoint_final.pth \
  --lidar_sidecar outputs/dualmodal/aether_joint/nuscenes/scene300_joint/lidar_sidecar_frozen.pt \
  --frame_idx 0 --cam_idx 0 --out_png outputs/nuscenes_0383_lidar.png
```

当前 LimSim 示例专注相机闭环，因此配置中保持 `lidar.enabled: false`。

## 11. 场景编辑与外部资产

将任意已经重建好的 3DRealCar 序列（包含 `gs.pth` 和 `wlh.json`）转换为
AETHER 轻量资产：

```bash
python tools/build_vehicle_asset.py \
  --input /path/to/3DRealCar/<sequence> \
  --output data/assets/vehicles/<name> \
  --sequence-id <sequence> --max-gaussians 50000
```

当 `traffic.hide_original_dynamic: true` 时，原场景的 `RigidNodes`、
`DeformableNodes` 和 `SMPLNodes` 会在第 0 帧前全部隐藏；随后创建可复用的
车辆槽位，并按 LimSim 每个时刻的位姿更新。验收 JSON 会证明整段闭环中
原始动态前景始终未重新出现。

手工脚本场景可使用 YAML/BEV 编辑器：

```bash
python -m aether.tools.scenario_editor configs/scenarios/nuscenes_0383.yaml
```

## 12. 接入其他端到端算法

复制 UniAD 声明：

```bash
cp configs/adapters/uniad.yaml configs/adapters/my_planner.yaml
```

修改其中的 `launcher`、`observation` 和 `output` 后运行：

```bash
python scripts/closed_loop.py \
  --scenario configs/scenarios/nuscenes_0383.yaml \
  --camera configs/cameras/nuscenes.yaml \
  --kinematic configs/kinematic.yaml \
  --ad my_planner --ad_config configs/adapters/my_planner.yaml
```

当前 IPC 使用兼容 HUGSIM 的 `obs_pipe`/`plan_pipe` pickle 协议。算法轨迹
定义为 `(右为正的横向距离, 前向距离)`；控制器会在执行前完成左右手坐标系
和转向符号转换。

## 目录结构

```text
aether/                    仿真、渲染、交通、传感器和算法适配器
configs/                   相机、场景和算法声明
scripts/                   下载、检查、训练和闭环入口
tools/                     数据、地图和车辆资产工具
third_party/drivestudio    三维重建与渲染后端
third_party/difix3d        图像修复桥
third_party/uniad_sim      UniAD 闭环适配器
third_party/splatad_gsplat 点云光栅器
data/                      下载的数据，不提交到 Git
```

## 许可证与数据来源

AETHER 使用仓库中的许可证。第三方代码保留各自许可证，详见
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。nuScenes、3DRealCar、
UniAD 权重、Difix 权重和 LimSim 均遵循其各自条款。发布的两辆车都由
3DRealCar 序列转换而来，具体序列号记录在各资产的 `metadata.json` 中。
