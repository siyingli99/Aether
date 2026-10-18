# AETHER

[English](README.md)

AETHER 将可编辑的三维高斯场景、LimSim 交通流和端到端驾驶算法接入相机闭环仿真。

## 开源计划

- [x] 基于 DriveStudio 的场景重建与新视角渲染
- [x] 相机与高斯点云仿真
- [x] 动态物体编辑与 3DRealCar 资产导入
- [x] nuScenes 高精地图导出与 LimSim 交通流
- [x] 六路相机 Difix 修复
- [x] UniAD 闭环适配器
- [x] 可视化场景与实验编辑器
- [ ] 公开评测协议

## Quick Start

### 克隆并安装 AETHER

```bash
git clone https://github.com/siyingli99/Aether.git
cd Aether
conda create -p .conda/aether python=3.10 -y
conda activate ./.conda/aether
```

```bash
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install -e third_party/drivestudio/third_party/smplx
python scripts/check_install.py
```

### 下载示例场景

```bash
huggingface-cli login
python scripts/download_demo.py \
  --repo https://huggingface.co/datasets/mikkochi/AETHER_Demo \
  --scene all
```

验证数据完整性：

```bash
python scripts/check_demo.py --scene 0383
python scripts/check_demo.py --scene 088
```

数据包包含 nuScenes `scene-0383` 和 `scene-0088` 的权重、相机数据、高精地图、LimSim 路网及两个车辆资产。车辆资产均来源于 [3DRealCar](https://xiaobiaodu.github.io/3drealcar/)。

### 安装 LimSim

在 AETHER 同级目录克隆已验证分支并创建环境：

```bash
git clone --branch aether-integration \
  https://github.com/siyingli99/LimSim.git ../LimSim
conda create -p .conda/limsim python=3.10 -y
conda run -p .conda/limsim \
  pip install -r ../LimSim/aether_bridge/requirements.txt
```

`scripts/run_closed_loop.sh` 会自动启动和关闭 LimSim。

### 安装 UniAD

```bash
conda create -p .conda/uniad python=3.10 -y
conda activate ./.conda/uniad
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install openmim
MMCV_WITH_OPS=1 pip install -v mmcv-full==1.7.2
pip install mmdet==2.26.0 mmdet3d==1.0.0rc6 mmsegmentation==0.29.1
pip install -r third_party/uniad_sim/requirements.txt
```

```bash
mkdir -p checkpoints
wget -O checkpoints/uniad_base_e2e.pth \
  https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth
conda activate ./.conda/aether
```

### 运行闭环

```bash
bash scripts/run_closed_loop.sh 0383 --no_io_trace
bash scripts/run_closed_loop.sh 088 --no_io_trace
```

使用 `AETHER_CUDA=1` 可切换 GPU。结果位于 `aether_results/nuscenes/<scene>/uniad/limsim_camera/`，其中 `limsim_aether.mp4` 是 LimSim/AETHER 同步视频。

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `AETHER_AD_READY_TIMEOUT` | `1800` | 等待适配器加载完成的秒数 |
| `AETHER_AD_STEP_TIMEOUT` | `900` | 单次规划调用的超时秒数；适配器退出或卡死时，运行会报错并附上其日志末尾 |
| `AETHER_ILQR_MAX_SOLVE_TIME` | 未设置 | iLQR 的墙钟时间上限；不设置时控制结果可复现（设为 `0.05` 可复现旧结果） |
| `UNIAD_VIS` / `UNIAD_VIS_WORKERS` | `1` / `4` | UniAD 逐步可视化开关及其并发进程数 |

#### 2026 年 10 月之前的结果

2026 年 10 月的修复会改变闭环指标：
- 背景碰撞盒改为放在局部地面上（之前悬在相机高度，`scene-0383` 上相当于离地 2.1–3.6 m）；
- 障碍带改用局部地面而不是全场景统一的地面来计算高度；
- 重建 actor 在 `obj_boxes` 中的朝向是正确的；
- recorded actor 按数据帧率（`data_frame_dt`）推进，而不是每个 0.25 s 步长前进一帧；
- 孤立的漂浮点、以及落在日志 ego 盒子内的点不再算作障碍；
- 离开场景检测改为看到任意重建点（包括路面）的距离，而不是到障碍点的距离；
- 刹车不再让 ego 倒车；
- iLQR 不再按墙钟时间提前截断。

如需复现之前的数字，在场景 YAML 中设置 `legacy_compat: true`，并设置环境变量 `AETHER_ILQR_MAX_SOLVE_TIME=0.05`。

## AETHER Studio

```bash
python scripts/aether_studio.py --host 127.0.0.1 --port 8000
```

打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)。编辑器会加载已注册场景、完整高精地图拓扑和六路相机预览，可直接完成：

1. 选择场景并查看路网；
2. 保留或删除原场景动态物体，用 3DRealCar 资产替换车辆，或在地图上新增车辆；
3. 为车辆选择原始轨迹、开环轨迹、LimSim 或注册的 Python 控制器，并配置自车起点、端到端算法及 Difix；
4. 校验、保存和运行实验，并在浏览器中查看运行日志。

场景、车辆资产、算法和控制器统一注册在 `configs/studio/catalog.yaml`。实验配置和结果保存在 `studio_workspace/experiments/<experiment-id>/`，每次运行都会生成不可变的场景配置。

接入本地交通控制器时，实现以下接口，再把导入路径作为 `entrypoint` 注册到目录；注册后即可在前端下拉框中选择。

```python
class MyController:
    def setup(self, context): ...
    def reset(self): ...
    def step(self, time_s, actor, ego):
        return {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
    def close(self): ...
```

```yaml
controllers:
  - id: my_controller
    name: My controller
    kind: plugin
    entrypoint: my_package.controller:MyController
```

编辑器统一使用 `x=前`、`y=左`、`z=上`、偏航角逆时针为正的道路坐标系。渲染器、自行车模型和算法输出的坐标转换集中在 `aether/studio/coordinates.py`。

仅开发前端时需要重新构建：

```bash
corepack pnpm --dir web install
corepack pnpm --dir web build
```

## 场景配置与编辑

### 配置入口

| 文件 | 在这里修改 | 读取位置 |
|---|---|---|
| `configs/scenarios/nuscenes_0383.yaml` | 权重、车辆初态、交通流、Difix 和终止条件 | `load_cfg()` 与 `AetherDrivingEnv` |
| `configs/cameras/nuscenes.yaml` | 相机名称、分辨率、内参与算法外参 | `load_camera_rig()` |
| `configs/kinematic.yaml` | 步长、轴距、加速度和转向限制 | `BicycleEgo.__init__()` / `BicycleEgo.step()` |
| `data/limsim/nuscenes_0383/scene0383.rou.xml` | 车辆路线、出生位置和速度 | LimSim/SUMO |
| `configs/adapters/uniad.yaml` | 算法进程与输入需求 | `load_adapter()` / `launch_adapter()` |

`configs/cameras/nuscenes.yaml` 中的 `H`、`W` 和内参决定渲染分辨率。运行时相机位姿由 `AetherDrivingEnv._load_static_c2w_ds()` 从 DriveStudio 预处理场景读取；`v2c` 和 `l2c` 会通过 `info["cam_params"]` 传给端到端算法。

### 场景 YAML

```bash
cp configs/scenarios/nuscenes_0383.yaml configs/scenarios/my_scene.yaml
```

| 字段 | 定义 | 实现位置 |
|---|---|---|
| `checkpoint` | DriveStudio 场景权重 | `AetherDrivingEnv.__init__()` 中的 `DriveStudioRenderer` |
| `start_ab` | `[横向向右, 前向]`，单位米 | `BicycleEgo.reset()` |
| `start_euler` | XYZ 欧拉角，单位度；索引 1 为偏航角，右转为正 | `BicycleEgo.__init__()` |
| `start_velo`, `start_steer` | 初速度（m/s）与初始转角（rad） | `BicycleEgo.reset()` |
| `ego_whl` | 自车 `[宽, 高, 长]`，单位米 | 碰撞框与 LimSim 自车状态 |
| `traffic.asset_path` | 含 `asset.pt` 或 `gs.pth` + `wlh.json` 的目录 | `GaussianVehicleAsset.load()` |
| `traffic.max_vehicles` | 可复用高斯车辆槽位数量 | `ExternalVehiclePool` |
| `traffic.max_gaussians` | 每辆车保留的高斯球数量 | `GaussianVehicleAsset._prune()` |
| `traffic.hide_original_dynamic` | 隐藏原场景的 `RigidNodes`、`DeformableNodes` 和 `SMPLNodes` | `ExternalVehiclePool._hide_all_original_dynamic()` |
| `traffic.net_file` | 二维同步画面使用的路网 | `LimSimTopDownRenderer` |
| `difix.required` | 任一路相机未修复即拒绝该帧 | `scripts/closed_loop.py:run()` |
| `termination.max_steps` | 最大闭环步数 | `AetherDrivingEnv.step()` |
| `termination.fg_collision_min_overlap` | 车辆碰撞的最小包围框重叠比例 | `_fg_collision_swept()` |
| `termination.bg_threshold` | 自车框内背景高斯球的碰撞阈值 | `_bg_collision()` |
| `termination.off_scene_radius`, `off_scene_streak` | 离开重建区域的容差 | `_off_scene_check()` |
| `termination.ground_cell_size` | 估计障碍带局部地面所用的 BEV 网格边长（米） | `_height_above_local_ground()` |
| `data_frame_dt` | DriveStudio 相邻帧的时间间隔：10 Hz 预处理（scene-0383）填 `0.1`，仅关键帧的 nuScenes（scene-0088）填 `0.5` | `AetherDrivingEnv.frame_idx` |
| `termination.bg_min_neighbors`, `bg_neighbor_radius` | 障碍点在该半径内至少要有这么多邻居（用于去掉孤立的漂浮点） | `_filter_obstacles()` |
| `termination.carve_logged_route` | 去掉落在日志 ego 盒子内的点（录制车辆真实驶过的空间） | `_filter_obstacles()` |
| `legacy_compat` | 设为 `true` 可复现 2026 年 10 月修复之前的结果（见下文） | `AetherDrivingEnv` |

示例场景要求每次调用算法前六路图像均成功通过 Difix。相关参数定义在 `scripts/closed_loop.py`：

```bash
bash scripts/run_closed_loop.sh 0383 \
  --refine_max_side 768 \
  --refine_timestep 199 \
  --refine_dtype bf16 \
  --no_io_trace
```

### 修改交通流与高精地图拓扑

修改以下文件中的 `<vehicle>` 节点：

```text
data/limsim/nuscenes_0383/scene0383.rou.xml
data/limsim/nuscenes_088/scene088.rou.xml
```

`route` 选择道路边序列；`depart`、`departPos` 和 `departSpeed` 分别控制出生时间、纵向位置和速度。输出的背景车数量不能超过 `traffic.max_vehicles`，也不能超过 `../LimSim/aether_bridge/server.py` 中 `LimSimRuntime.__init__()` 的 `max_vehicles` 参数。

通过 `extract_hd_map()` 和 `export()` 为其他 nuScenes 场景生成路网：

```bash
python tools/export_nuscenes_limsim_map.py \
  --dataset-root /path/to/NuScenes \
  --processed-scene data/nuscenes/processed_10Hz/trainval/<index> \
  --location <nuscenes-map-name> \
  --scene-label <scene-id> \
  --prefix scene<scene-id> \
  --output-dir data/limsim/nuscenes_<scene-id>
```

`extract_hd_map()` 将 nuScenes lane/lane-connector 图转换到道路局部坐标系；`export()` 生成 SUMO 节点、道路边、连接关系、路线和路网文件。随后在场景 YAML 中加入新路径，并扩展 `scripts/run_closed_loop.sh` 的场景选择逻辑。

### 编辑车辆高斯资产

```bash
python tools/build_vehicle_asset.py \
  --input /path/to/3DRealCar/<sequence> \
  --output data/assets/vehicles/<name> \
  --sequence-id <sequence> \
  --max-gaussians 50000
```

将 `traffic.asset_path` 指向输出目录。`AetherDrivingEnv._init_traffic_backend()` 负责加载资产，`ExternalVehiclePool` 创建渲染槽位，`TrafficActorProvider.advance()` 在每个仿真时刻应用 LimSim 位姿。

单实例编辑接口位于 `ExternalVehiclePool`：`original_gaussian_counts()` 列出原始车辆 ID，`set_original_visible(id, False)` 删除指定车辆，`extract_original(id)` 返回对应的高斯参数。示例配置通过 `hide_original_dynamic: true` 删除全部原始动态物体。

### 重建场景与点云仿真

`scripts/train_scene.py` 将 OmegaConf 参数传给 DriveStudio。nuScenes 索引 `300` 对应 `scene-0383`：

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

训练并检查点云 sidecar：

```bash
python aether/train/joint_phase.py \
  --config_file third_party/drivestudio/configs/omnire.yaml \
  --resume_from data/checkpoints/nuscenes_outputs/scene_recon_full/scene_300_6cams_with_humans/checkpoint_final.pth \
  --output_root outputs/dualmodal \
  --project aether_joint/nuscenes \
  --run_name scene300_joint \
  --num_iters 10000 --freeze_geometry \
  dataset=nuscenes/6cams \
  data.data_root=data/nuscenes/processed_10Hz/trainval \
  data.scene_idx=300
```

```bash
python aether/tools/viz_joint.py \
  --resume_from data/checkpoints/nuscenes_outputs/scene_recon_full/scene_300_6cams_with_humans/checkpoint_final.pth \
  --lidar_sidecar outputs/dualmodal/aether_joint/nuscenes/scene300_joint/lidar_sidecar_frozen.pt \
  --frame_idx 0 --cam_idx 0 \
  --out_png outputs/nuscenes_0383_lidar.png
```

## 接入新的端到端算法

一个适配器由 Python 类、启动脚本和 YAML 声明组成。

### 1. 实现算法类

创建 `third_party/<planner>/aether_adapter.py`：

```python
import numpy as np

from aether.agents.sdk import BaseAgent, serve

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


class MyPlanner(BaseAgent):
    def setup(self):
        # CUDA_VISIBLE_DEVICES 设置完成后调用一次。
        self.model = load_model()

    def infer(self, obs, info):
        # 图像为 uint8 H x W x 3，且已经过 Difix。
        images = [obs["rgb"][name] for name in CAMERAS]
        path_forward_left = self.model(images, info)  # N x 2：前、左

        # AETHER 输出：横向向右、前向，单位米。
        path_forward_left = np.asarray(path_forward_left, dtype=np.float32)
        return np.column_stack(
            (-path_forward_left[:, 1], path_forward_left[:, 0])
        )

    def teardown(self):
        pass


if __name__ == "__main__":
    serve(MyPlanner())
```

`BaseAgent.setup()` 只执行一次，`infer(obs, info)` 每个仿真步执行一次，`teardown()` 在退出时执行。`serve()` 负责 FIFO 时序、pickle 传输、轨迹校验和最终的 `Done` 消息。

输入契约由 `AetherDrivingEnv._get_obs()` 和 `_get_info()` 定义：

| 数据 | 类型与坐标定义 |
|---|---|
| `obs["rgb"][camera]` | `uint8[H,W,3]`，六路命名相机 |
| `obs["depth"]`, `obs["lidar"]` | 可选；在场景和适配器 YAML 中启用 |
| `info["ego_pos"]` | 仿真坐标系中的 `[横向向右, 0, 前向]`，单位米 |
| `info["ego_rot"]` | XYZ 欧拉角，单位 rad；索引 1 为偏航角，右转为正 |
| `info["ego_velo"]`, `info["ego_steer"]` | m/s 与 rad |
| `info["command"]` | `0=右转`、`1=左转`、`2=直行` |
| `info["cam_params"][camera]` | `intrinsic`、`v2c` 和 `l2c` 标定参数 |
| 算法返回值 | 非空 `float N x 2`：当前自车局部系下的 `[横向向右, 前向]` 轨迹点，单位米 |

LimSim 交通协议使用另一套统一坐标：`x=前`、`y=左`、`z=上`，偏航角逆时针为正。转换位于 `AetherDrivingEnv._traffic_ego_state()` 和 `FrameTransform`；不要把交通协议的符号直接用于算法输出。

轨迹到控制量的转换位于 `scripts/closed_loop.py:_trajectory_states()` 和 `_traj_to_action()`。算法若输出 `[前, 左]`，按示例转换；若直接输出控制量，可在 `infer()` 内先转为轨迹，或将 `_traj_to_action()` 替换为专用控制接口。

### 2. 添加启动脚本

创建 `third_party/<planner>/run_aether.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
planner_python="${PLANNER_PYTHON:-$repo_dir/.conda/<planner>/bin/python}"
export PYTHONPATH="$repo_dir:${PYTHONPATH:-}"
exec "$planner_python" -u "$repo_dir/third_party/<planner>/aether_adapter.py" "$1" "$2"
```

启动脚本依次接收 `<cuda_id> <output_dir>`，进程需保持运行，并输出 `waiting for obs`、`adapter ready` 或其他可被 `wait_adapter_ready()` 识别的标记。

### 3. 注册并运行

创建 `configs/adapters/<planner>.yaml`：

```yaml
name: <planner>
display_name: <Planner>

launcher:
  type: shell
  entrypoint: third_party/<planner>/run_aether.sh
  cwd: third_party/<planner>
  args: ["${cuda_id}", "${output_dir}"]
  env:
    PYTHONUNBUFFERED: "1"

ipc:
  protocol: hugsim_pickle_v1
  obs_pipe: obs_pipe
  plan_pipe: plan_pipe
  pickle_protocol: 4
  done_sentinel: Done

observation:
  cameras: [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT]
  needs_depth: false
  needs_lidar: false

output:
  type: trajectory
  frame: ego_local
  fields: [lateral_right, forward]
```

`AdapterSpec.render_args()` 替换启动参数，`launch_adapter()` 应用 `cwd` 和 `env`。运行命令：

```bash
python scripts/closed_loop.py \
  --scenario configs/scenarios/nuscenes_0383.yaml \
  --base configs/base.yaml \
  --camera configs/cameras/nuscenes.yaml \
  --kinematic configs/kinematic.yaml \
  --ad <planner> \
  --ad_config configs/adapters/<planner>.yaml \
  --ad_cuda 0
```

开发适配器时不要添加 `--no_io_trace`。可在 `aether_results/nuscenes/<scene>/<planner>/limsim_camera/io_trace/step_*/raw.pkl` 查看每一步实际传输的输入与输出。

## 输出文件

| 文件 | 内容 |
|---|---|
| `limsim_aether.mp4` | LimSim 二维与 AETHER 六路相机同步视频 |
| `video.mp4` | AETHER 六路相机视频 |
| `data.pkl`, `infos.pkl` | 轨迹与逐步状态 |
| `io_trace/` | 每一步的算法输入与输出 |
| `refine_audit.json` | 六路 Difix 记录 |
| `dynamic_removal_audit.json` | 原动态实例删除与外部槽位记录 |
| `perf.json` | 运行耗时 |

## 许可证

AETHER 使用仓库许可证；第三方代码和数据沿用各自条款，详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
