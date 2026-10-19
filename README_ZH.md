# AETHER

[English](README.md)

AETHER 是一个基于三维高斯重建场景的自动驾驶闭环仿真器。端到端规划算法（UniAD、VAD、LTF、VLA 模型或你自己的算法）在逼真的相机（和激光雷达）渲染中驾驶主车。背景交通由二维交通流模型控制：LimSim/SUMO、内置 IDM，或你自己写的模型。每次运行都会输出鸟瞰图（BEV）视频，以及每辆车逐时刻的轨迹表，可以像普通交通仿真结果一样直接分析。

```text
          ┌──────────── 规划算法（UniAD / VAD / LTF / VLA / 自定义）────────────┐
 图像 ────┤                                                                       ├── 轨迹点
          └───────────────────────────────────────────────────────────────────────┘
   ▲                                                                                │ iLQR 跟踪
   │ 渲染（高斯场景 + 交通车资产）                主车运动学（自行车模型）           ▼
 三维场景 ◄──── 车辆位姿 ◄──── 交通流模型（LimSim / IDM / 自定义）◄──── 主车状态
   │
   └─► 碰撞 · 驶出道路 · 评测指标 · BEV 视频 · trajectories.csv
```

- [快速开始](#快速开始)
- [运行闭环仿真](#运行闭环仿真)
- [输出](#输出)
- [接入端到端算法](#接入端到端算法)
- [接入交通流模型](#接入交通流模型)
- [场景配置](#场景配置)
- [Studio](#studio)
- [仿真细节](#仿真细节)：坐标系、地面、碰撞、指标
- [导入自己的场景](#导入自己的场景)

## 快速开始

```bash
git clone https://github.com/siyingli99/Aether.git && cd Aether
conda create -p .conda/aether python=3.10 -y && conda activate ./.conda/aether
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install -e third_party/drivestudio/third_party/smplx
export PYTHONNOUSERSITE=1          # 防止 ~/.local 里的 torch 覆盖环境里的版本
```

下载演示场景（nuScenes scene-0383 和 scene-0106，包括重建权重、高精地图、SUMO 路网和两个 [3DRealCar](https://xiaobiaodu.github.io/3drealcar/) 车辆资产）。脚本会把它们转换成 `data/scenes/` 下的场景包：

```bash
huggingface-cli login
python scripts/download_demo.py --repo https://huggingface.co/datasets/mikkochi/AETHER_Demo
python -m aether doctor            # 检查环境、场景和算法
python -m aether scenes
```

第一次运行，不需要任何学习型算法（主车沿录制轨迹行驶）：

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent replay --no-refine
```

### 可选组件

| 组件 | 用途 | 安装 |
|---|---|---|
| LimSim | `traffic.model: {type: limsim}` | `git clone --branch aether-integration https://github.com/siyingli99/LimSim.git ../LimSim`，然后 `conda create -n limsim python=3.10 && pip install -r ../LimSim/aether_bridge/requirements.txt`。AETHER 会自动启动和关闭 LimSim 服务。如果不在默认位置，设置 `LIMSIM_ROOT` / `LIMSIM_PYTHON`。 |
| UniAD | `--agent uniad` | 见 [UniAD](#uniad) |
| Difix3D | `refine.enabled: true` | 已内置于 `third_party/difix3d`；`nvidia/difix` 权重在第一次使用时从 Hugging Face 下载 |
| SUMO `netconvert` | `python -m aether map`（仅新场景需要） | `pip install eclipse-sumo` |

## 运行闭环仿真

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent uniad
python -m aether run configs/scenarios/nuscenes-0106.yaml --agent vad --cuda 1
python -m aether run configs/scenarios/nuscenes-0383-idm.yaml --agent replay --steps 20 --no-refine
python -m aether run my.yaml --agent uniad --set traffic.max_vehicles=5 --set sim.dt=0.5
```

`--agent` 可以是 `configs/agents/` 中的适配器名（`uniad`、`vad`、`ltf`、`vla`、`gaussianfusion`），也可以是 `replay`，或者进程内算法 `package.module:Class` / `path/to/file.py:Class`。`python -m aether agents` 会列出所有可用算法。

基准测试会运行多个场景 × 多个算法，并汇总指标：

```bash
python -m aether bench configs/benchmarks/nuscenes-demo.yaml --agents uniad,vad
python -m aether eval outputs/runs/<run>        # 根据 CSV 重新计算 metrics.json
```

## 输出

每次运行写入一个目录 `outputs/runs/<场景>-<算法>-<YYYYmmdd-HHMMSS>/`：

| 文件 | 内容 |
|---|---|
| `trajectories.csv` | 每个仿真步、每辆车（主车和背景车）的状态 |
| `trajectories_10hz.csv` | 同上，重采样到统一的 10 Hz 时钟（`output.trajectory_hz`），按车辆自身速度插值 |
| `trajectories_fcd.xml` | 10 Hz 轨迹的 SUMO 浮动车（FCD）格式；坐标与场景的 `map/network.net.xml` 一致，`sumo-gui` 以及 SUMO/LimSim 的分析工具可以直接读取 |
| `events.csv` | 碰撞、驶出道路、离开场景：时间、类型、对方车辆、是否主车责任 |
| `metrics.json` | NC、DAC、TTC、舒适度、EP、RC 和总分（见[评测指标](#评测指标)） |
| `videos/bev.mp4` | 鸟瞰图，20 fps（`output.bev_fps`），仿真步之间做运动插值 |
| `videos/cameras.mp4` | 规划算法实际收到的相机图像，每个仿真步一帧 |
| `videos/overview.mp4` | BEV 与相机图像左右拼接 |
| `run.json` | 场景、算法、车型、耗时、激光雷达来源 |
| `logs/steps.jsonl` | 每步的控制量、耗时和事件 |
| `agent/` | 算法工作目录：`planner.log`、FIFO 管道，以及算法自己保存的内容 |
| `logs/io/`、`logs/render_poses.json` | 仅在 `output.io_trace: true` 时输出：每一步算法收到的图像和渲染位姿，用于调试适配器和闭环蒸馏（`aether/refine/distill_from_closed_loop.py`） |

三个视频使用同一个时钟：每个视频的第 *t* 秒都对应仿真时间 *t*。相机视频每个仿真步一帧（`dt = 0.25 s` 时为 4 fps）。BEV 视频在仿真步之间对车辆位置和速度做三次 Hermite 插值，因此播放是连续平滑的。

轨迹表各列（国际单位制）：

| 列 | 含义 |
|---|---|
| `time_s`, `step` | 仿真时间和步序号 |
| `vehicle_id`, `role`, `type`, `source` | `ego` 或背景车编号；`role` ∈ {ego, background}；`source` ∈ {ego, model, vehicles, recorded} |
| `x`, `y`, `z`, `yaw` | 车辆中心在场景道路坐标系中的位置（x 沿主车初始朝向，y 向左，z 向上，航向角逆时针为正） |
| `speed`, `vx`, `vy`, `accel`, `yaw_rate` | 速度、道路坐标系下的速度分量、纵向加速度、横摆角速度 |
| `length`, `width`, `height` | 车辆包围盒尺寸 |
| `lane_id`, `s`, `d` | 匹配到的高精地图车道、沿车道中心线的弧长、带符号横向偏移（左正） |
| `x_global`, `y_global`, `yaw_global` | 同一位姿在数据集地图坐标系中的值（例如 nuScenes 地图坐标） |

## 接入端到端算法

规划算法可以在**独立进程和独立 conda 环境**中运行（常见情况：UniAD、VAD、LTF、VLA 的依赖互相冲突），也可以**在 AETHER 进程内**运行（轻量模型、规则算法、研究原型）。

### 算法的输入与输出

每个仿真步（`sim.dt`，默认 0.25 s）调用一次算法：

| 输入 | 内容 |
|---|---|
| 图像 | 每个相机一张 `uint8 [H, W, 3]` RGB 图；开启 Difix 时为修复后的图像 |
| 内参 | 与交付图像尺寸对应的 3×3 `K` |
| 外参 | 相机 → 车辆（`mount`；相机轴为 OpenCV 约定；车辆坐标系原点在后轴中心地面，x 向前、y 向左、z 向上） |
| 主车状态 | 速度、加速度、前轮转角及其变化率、横摆角速度；道路坐标系下的位姿 |
| 导航指令 | `0` 右转，`1` 左转，`2` 直行，由录制轨迹前方 3 s 推得 |
| 激光雷达（可选） | `float32 [N, 4]`（x, y, z, 强度），在场景传感器配置的 LIDAR_TOP 坐标系下；有 SplatAD 侧车模型时使用其扫描，否则用深度反投影 |

算法返回**轨迹点**（`Plan`）：`N × 2` 个位置，相邻点间隔 `dt`（默认 0.5 s），第一个点对应 `+dt`，坐标系由算法声明。也可以直接返回**控制量**（`Control`：加速度 m/s²，转角变化率 rad/s，左正）。AETHER 先把轨迹点换算到主车后轴，再用 iLQR 控制器跟踪；控制器使用主车车型的轴距、加速度和转向限制。主车随后按运动学自行车模型运动；前轮转角受轮胎附着极限约束 `v² tan(δ) / L ≤ max_lat_accel`（默认 8 m/s²，可在 `configs/vehicles/` 中按车型设置），因此规划算法无法以超过真实车辆的侧向加速度过弯。

| `frame` | 轨迹点相对于 | 典型算法 |
|---|---|---|
| `rear_axle` | 后轴中心地面点（nuScenes ego / nuPlan / NAVSIM 约定） | LTF、VLA、GaussianFusion |
| `center` | 车辆包围盒几何中心 | |
| `lidar` | 场景传感器配置中的 LIDAR_TOP 安装位置 | UniAD、VAD |

| `axes` | 列顺序 |
|---|---|
| `flu` | （前，左） |
| `rfu` | （右，前），即原 UniAD/VAD 闭环脚本的格式 |

**图像分辨率。** 相机按场景传感器配置的分辨率渲染：nuScenes 演示场景为 800×450，即 nuScenes 原图 1600×900 的一半。UniAD 和 VAD 的适配器在内部把图像放大到 1600×900、内参乘以 2，这与模型评测时的做法完全一致。不要为它们修改渲染分辨率：其他分辨率会破坏这一假设，并明显降低两种算法的表现。其他算法可以在适配器 YAML 中设置 `image_size`（或进程内算法的 `EgoAgent.image_size`），AETHER 会在交付前缩放图像并相应调整 `K`。

### A. 外部算法（独立进程）

**1. 在算法仓库中编写算法脚本。** 它只依赖 NumPy；把 AETHER 目录加入 `PYTHONPATH`，或者把 `aether/agents/sdk/bridge.py` 复制到脚本旁边。

```python
# my_planner/aether_agent.py
import numpy as np
from aether.agents.sdk import AdapterAgent, serve

class MyPlanner(AdapterAgent):
    def setup(self):                         # 只调用一次，此时 CUDA_VISIBLE_DEVICES 已设置
        self.model = load_my_model()

    def infer(self, obs, info):
        front = obs["rgb"]["CAM_FRONT"]      # uint8 H x W x 3
        K = info["cam_params"]["CAM_FRONT"]["intrinsic"]   # fovx, fovy, cx, cy, H, W（以及 fx, fy）
        speed = info["ego_velo"]             # m/s
        cmd = info["command"]                # 0 右转, 1 左转, 2 直行
        fwd_left = self.model(front, speed, cmd)            # N x 2（前, 左），单位米
        return {"waypoints": fwd_left, "frame": "rear_axle", "axes": "flu", "dt": 0.5}
        # 或者：return np.column_stack([-fwd_left[:, 1], fwd_left[:, 0]])  # 裸 N x 2，按 YAML 中声明的 frame/axes 解释
        # 或者：return {"accel": 0.5, "steer_rate": 0.0}

if __name__ == "__main__":
    serve(MyPlanner())                       # 命令行参数：<cuda_id> <workdir>
```

使用 `protocol: tuple` 时，`info` 的结构就是内置 UniAD / VAD / LTF / VLA 适配器读取的结构：

| `info` 键 | 含义 |
|---|---|
| `ego_pos` | 后轴在道路坐标系中的位置 `[右, 0, 前]`（m） |
| `ego_rot` | `[0, yaw, 0]`，航向角顺时针为正（rad） |
| `ego_velo`, `accelerate` | 速度（m/s）、加速度（m/s²） |
| `ego_steer`, `steer_rate` | 前轮转角（rad）及其变化率（rad/s），右正 |
| `command`, `timestamp` | 导航指令、仿真时间（s） |
| `cam_params[cam]` | `intrinsic` {fovx, fovy（rad）, cx, cy, fx, fy, H, W}，`v2c`（车辆 → 相机 4×4），`l2c`（激光雷达 → 相机 4×4） |
| `ego_box`, `obj_boxes` | 道路坐标系下的包围盒 `[x, y, z, w, l, h, yaw]`（真值，仅用于调试） |
| `aether` | AETHER 自己的道路坐标系信息：主车、所有交通参与者、事件、路线进度 |

**2. 启动脚本**（`my_planner/aether_agent.sh`）：直接调用算法环境中的 Python 解释器并启动脚本。

```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
PY="${MY_PLANNER_PYTHON:-$HOME/miniconda3/envs/my_planner/bin/python}"
CUDA_VISIBLE_DEVICES="$1" "$PY" -u aether_agent.py "$1" "$2"
```

建议直接写环境内 Python 的路径：从另一个 Python 进程启动时，`conda activate` 不一定能切换解释器。

**3. 在 `configs/agents/my_planner.yaml` 中注册：**

```yaml
name: my_planner
title: My planner
command: [bash, "${AETHER_EXTERNAL}/my_planner/aether_agent.sh", "${cuda}", "${workdir}"]
ready: ["ready"]                 # 模型加载完成后日志中出现的文字
protocol: tuple                  # (sensors, info)，见上表；"observation" 则发送 Observation 字典
input:
  cameras: [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT]   # 只渲染这些相机
  image_size: null               # [W, H]；null 表示场景渲染尺寸
  lidar: false
output: {dt: 0.5, frame: rear_axle, axes: rfu}          # 用于解释裸 N x 2 的返回值
```

`${AETHER_EXTERNAL}` 默认是包含 AETHER 目录的上一级目录（可用环境变量 `AETHER_EXTERNAL` 覆盖）。`${workdir}` 是 `<run>/agent`，FIFO 管道和 `planner.log` 都在这里。然后运行：

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent my_planner
```

每个适配器还可以设置 `AETHER_ROOT`、`AETHER_EXTERNAL`、单步超时（`step_timeout_s`，默认 900 s）和启动超时（`startup_timeout_s`）。算法崩溃或卡住时，本次运行会报错，并附上算法日志的末尾。

### B. 进程内算法

```python
# my_agents.py
import numpy as np
from aether.agents import EgoAgent, Plan

class KeepLane(EgoAgent):
    cameras = ["CAM_FRONT"]            # 渲染并交付这些相机；[] 表示不需要图像
    image_size = (640, 360)            # 可选的缩放尺寸；K 会相应调整

    def act(self, obs):                # obs: aether.agents.Observation
        img = obs.images["CAM_FRONT"]; K = obs.cameras["CAM_FRONT"]["K"]
        v = max(obs.ego["speed"], 3.0)
        s = v * 0.5 * np.arange(1, 7)
        return Plan(np.column_stack([s, np.zeros(6)]), dt=0.5, frame="rear_axle", axes="flu")
```

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent my_agents.py:KeepLane
```

`Observation` 包含：
- `time`、`step`
- `images`
- `cameras`：K、mount、尺寸
- `ego`：x、y、yaw、speed、accel、steer、steer_rate、yaw_rate、rear_axle、尺寸
- `command`
- `lidar`
- `info`：所有交通参与者、事件、路线进度

`setup(context)` 会收到场景编号、车型参数、`dt`、录制路线和相机名称。

### 内置适配器

| 算法 | 仓库（与 AETHER 同级） | 相机 | 激光雷达 | 规划输出 |
|---|---|---|---|---|
| `uniad` | 已内置：`third_party/uniad_sim` | 6 | – | lidar · rfu · 0.5 s |
| `vad` | `VAD_SIM`（`tools/e2e.sh`） | 6 | – | lidar · rfu · 0.5 s |
| `ltf` | `NAVSIM`（`ltf_e2e.sh`，Latent TransFuser） | 前 3 | ✓ | rear_axle · rfu · 0.5 s |
| `vla` | `Impromptu-VLA`（`tools/e2e.sh`，Qwen2.5-VL） | 前 3 | – | rear_axle · rfu · 0.5 s |
| `gaussianfusion` | `GaussianFusion`（`tools/e2e.sh`） | 4 | ✓ | rear_axle · rfu · 0.5 s |

#### UniAD

```bash
conda create -n uniad python=3.10 -y && conda activate uniad
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install openmim && MMCV_WITH_OPS=1 pip install -v mmcv-full==1.7.2
pip install mmdet==2.26.0 mmdet3d==1.0.0rc6 mmsegmentation==0.29.1
pip install -r third_party/uniad_sim/requirements.txt
mkdir -p checkpoints && wget -O checkpoints/uniad_base_e2e.pth \
  https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth
export UNIAD_PYTHON=$(conda run -n uniad which python)   # 或者创建 .conda/uniad
```

## 接入交通流模型

背景交通由一个**交通流模型**负责：它管理所有背景车，并在每个仿真步统一推进它们，这与微观交通仿真器的接口一致。每一步的流程如下：
1. AETHER 把主车状态传给模型；
2. 模型返回新时刻存在的所有车辆；
3. AETHER 把每辆车渲染为一个高斯车辆资产，进行碰撞检测并记录轨迹。

```python
# my_traffic.py
from aether.traffic import AgentState, TrafficModel

class MyCarFollowing(TrafficModel):
    """车辆沿高精地图车道中心线行驶，使用自己的跟驰模型。"""

    def __init__(self, n=3, v0=10.0, **params):          # 场景中的 params → 关键字参数
        super().__init__(**params)
        self.n, self.v0 = n, v0

    def reset(self, scene):                               # scene: TrafficScene
        # scene.ego（AgentState）、scene.hdmap（车道、可行驶区域）、scene.dt、
        # scene.network（SUMO 的 net.xml、rou.xml）、scene.seed
        lane = scene.hdmap.lanes[0]
        x, y = lane.centerline[0]
        return [AgentState("car-1", x, y, yaw=0.0, speed=self.v0)]

    def step(self, t, dt, ego, vehicles):
        # 从 t - dt 推进到 t；返回 t 时刻存在的车辆
        return [v.moved(dt, accel=0.0) for v in vehicles]

    def close(self):
        pass
```

```yaml
traffic:
  assets: [3drealcar_2024_06_04_15_34_51]   # 用于渲染的高斯车辆资产（循环使用）
  max_vehicles: 6                            # 同时最多渲染的车辆数
  model: {class: my_traffic:MyCarFollowing, params: {n: 3, v0: 9.0}}
```

在 Studio 中把背景交通选为 “Python class”，填入 `my_traffic:MyCarFollowing` 和 JSON 参数，效果相同。

| | |
|---|---|
| 坐标系 | 场景道路坐标系：x 沿主车初始朝向，y 向左，航向角逆时针（rad），单位米；与 `scene.hdmap` 同一坐标系 |
| 位置 | 车辆**包围盒中心** |
| `AgentState` | `id, x, y, yaw, speed, length, width, height, accel, type, lane_id`；同一辆车的 id 在各步之间必须保持不变；车辆可以随时出现或消失 |
| 渲染 | 每辆车分配一个高斯资产槽位，放在重建的地面上（高度和俯仰角来自地面模型）；渲染尺寸为资产本身的尺寸 |
| 高精地图 | `scene.hdmap.lanes[i]`：`id, centerline (N×2), width, polygon, incoming, outgoing`；`lane.project(x, y) → (s, d, heading)`；`scene.hdmap.match(x, y, yaw)` |

内置模型（`traffic.model.type`）：

| 类型 | 说明 |
|---|---|
| `limsim` | LimSim（SUMO），使用场景路网 `map/network.net.xml` 和需求 `map/routes.rou.xml`；自动启动 |
| `idm` | 沿高精地图车道路径的智能驾驶员模型：`{type: idm, count: 3, desired_speed: 10, time_headway: 1.5, min_gap: 2, max_accel: 1.5, comfort_decel: 2}`，或显式给出 `vehicles: [{id, lane, s, speed}]`；主车可作为前车 |
| `replay` | 从 JSON 文件回放记录的帧 |
| `external` | 任何实现下文 TCP 协议的进程 |
| `class` | 你自己的 `TrafficModel` |

**外部仿真器（C++、MATLAB、VISSIM、TESS NG 等）。** 运行一个实现 `aether.traffic.v1` 协议的 TCP 服务即可。协议格式：
- 传输：按行分隔的 JSON，定义见 `aether/traffic/protocol.py`；该模块只依赖标准库，可以直接复制到你的仿真器中；
- 请求：`hello` → `{server}`，`reset` → `{frame}`，`tick {target_time_s, ego}` → `{frame}`，`shutdown`；
- 帧：`{time_s, vehicles: [{id, x, y, z, yaw, speed, length, width, height, type}]}`，坐标为上述道路坐标系。

```yaml
traffic:
  model: {type: external, server: my_sim, host: 127.0.0.1, port: 9000}
```

**单车控制器。** 如果只想控制个别车辆（例如一次加塞），可以给每辆车单独指定控制器，而不使用交通流模型：

```python
from aether.traffic import VehicleController

class CutIn(VehicleController):
    def setup(self, context):            # context: {"vehicle": 配置, "params": {...}, "scene": TrafficScene}
        self.start = context["vehicle"]["start"]
    def step(self, time_s, actor, ego):  # -> 包围盒中心的新位姿
        return {"x": self.start["x"] + 8.0 * time_s, "y": self.start["y"] - min(time_s, 3.5),
                "yaw": 0.0, "speed": 8.0}
```

```yaml
traffic:
  vehicles:
    - {id: cutin, size: [4.5, 1.85, 1.5], start: {x: 15, y: 3.5, yaw: 0},
       controller: {type: plugin, class: my_traffic:CutIn, params: {}}}
    - {id: parked, start: {x: 30, y: -3, yaw: 0}, controller: {type: static}}
    - {id: replay, controller: {type: open_loop, trajectory: [{t: 0, x: 5, y: 0, yaw: 0}, {t: 4, x: 40, y: 0, yaw: 0}]}}
```

单车控制器和交通流模型可以同时使用。给某辆脚本车辆加上 `render: {original: RigidNodes/3}`，就会移动重建场景中的那辆原有车辆，而不是使用车辆资产。

## 场景配置

```yaml
# configs/scenarios/nuscenes-0383.yaml
scene: nuscenes-0383                 # data/scenes/ 下的场景包
sim: {dt: 0.25, max_steps: 39, seed: 0}
ego:
  vehicle: auto                      # 车型；auto 表示该数据集的采集车
  start: logged                      # 或 {x, y, yaw}（包围盒中心，道路坐标系）
  speed: logged                      # 或一个数值（m/s）
traffic:
  hide_originals: true               # 隐藏重建场景中的动态物体
  keep: []                           # 例外，例如 [RigidNodes/3]、[RigidNodes/*] 或 ["*"]，这些物体按录制轨迹回放
  assets: [3drealcar_2024_06_04_15_34_51]
  max_vehicles: 3
  model: {type: limsim, port: 8766}
refine: {enabled: true, required: true}       # 对每张输入算法的图像做 Difix 修复
termination: {stop_on: [vehicle_collision, static_collision, off_scene], at_fault_only: false}
output: {bev_fps: 20, trajectory_hz: 10, videos: true, io_trace: false}
agent: {name: uniad, cuda: "0"}               # --agent 的默认值
```

`extends: other.yaml` 可以继承另一个场景配置。未知字段会直接报错。

车型配置位于 `configs/vehicles/`：
- `renault_zoe`：nuScenes 采集车；
- `chrysler_pacifica`：PandaSet 采集车；
- `sedan`：通用轿车。

车型决定碰撞包围盒、轴距、后悬以及加速度和转向限制。它也决定相机的安装位置，因为相机外参是相对于该车型的后轴标定的。

## Studio

```bash
python -m aether studio            # http://127.0.0.1:8765
```

- **Scenario（场景编辑）**：选择场景和规划算法。地图编辑器显示高精地图、可行驶区域、录制路线和重建的交通参与者。你可以：
  - 在车道上添加车辆（朝向自动对齐车道），拖动和旋转，并设置为静止、匀速或 Python 控制器类；
  - 保留或重新驾驶重建车辆，移动主车起点；
  - 选择交通流模型：无、IDM、LimSim（场景带 SUMO 路网时可选）或自定义 `TrafficModel` 类（附 JSON 参数）；
  - 设置终止事件和输出，并导出 YAML。
- **Live（实时）**：运行过程中实时显示 BEV、渲染的相机图像和日志。
- **Results（结果）**：查看每次运行的指标、事件、三个视频，并下载 CSV。

前端开发：`pnpm --dir web install && pnpm --dir web dev`（API 代理到 :8765）；`pnpm --dir web build` 重新生成 `web/dist`。

## 仿真细节

### 坐标系

| 坐标系 | 定义 |
|---|---|
| 道路坐标系（road） | x 沿 t = 0 时录制主车的朝向，y 向左，z 向上；原点为 t = 0 时锚定传感器正下方的地面点。所有公开状态、CSV、高精地图和交通流模型都使用该坐标系。 |
| 车辆坐标系（vehicle） | 原点在后轴中心地面，x 向前，y 向左，z 向上（nuScenes ego 约定） |
| 场景坐标系（scene） | 重建的世界坐标系，因数据集而异，只在内部使用（定义见 `aether/geometry/frames.py`） |

PandaSet 场景坐标系为 x 向右、y 向前、z 向上（PandaSet 车体坐标系）；nuScenes 为 x 向右、y 向下、z 向前（前视相机）。二者都映射到同一个道路坐标系，因此场景配置和规划算法不需要关心这一差异。对于以车体位姿为锚点的数据集（PandaSet、nuPlan），导入场景时会把车体位姿从数据集自身的坐标轴转换到车辆坐标系；2026 年 10 月之前导入的场景包会在加载时自动纠正。

### 地面

重建结果中没有单独的地面模型。AETHER 为每个场景估计一次地面高度（缓存在 `data/scenes/<id>/cache/ground.npz`），数据来自两部分：
- **录制主车轨迹**：给出路线沿线的道路高程剖面；
- **背景高斯点**：给出其他区域的绝对高度。每个 2 m 网格取最低的稠密高斯层作为地面，从而剔除路面下方的漂浮点和路面上方的物体。

两者在坡度约束下融合，再平滑到 0.5 m 网格。主车和所有交通车都贴合这一地面：高度、俯仰和侧倾由车轮和车轴下方的地面计算。因此相机会跟随坡道和路沿起伏，静止或行驶的车辆也不会悬空或陷入地面。

### 碰撞与违规判定

| 事件 | 判定标准 |
|---|---|
| `vehicle_collision` | 主车包围盒（尺寸来自主车车型）与背景车包围盒重叠。包围盒按 ≤ 0.5 m 的子步扫掠，高速车辆不会互相穿过。事件记录碰撞方位（前、后、左、右）和责任：主车静止，或被至少同样快的车辆追尾时，判为非主车责任。 |
| `static_collision` | 主车包围盒内有超过 `static_threshold`（50）个稠密背景高斯点，且这些点高于局部地面 0.3–2.5 m，即建筑、杆柱、路沿、重建场景中的停放车辆等。孤立漂浮点（0.3 m 内邻居少于 8 个）和录制主车行驶走廊内的点会被忽略。 |
| `off_road` | 主车包围盒任一角点超出高精地图可行驶区域 `off_road_tolerance`（0.3 m）以上（nuScenes 使用真实可行驶区域多边形，其他情况使用车道多边形） |
| `off_scene` | 连续 `off_scene_steps`（4）步，主车距任何重建几何都超过 5 m，此时渲染已无意义 |

`termination.stop_on` 选择哪些事件会结束本次仿真。所有事件都会被记录并参与评分。

### 评测指标

| 指标 | 定义 |
|---|---|
| NC | 无主车责任碰撞：1；只与静态场景碰撞：0.5；主车责任的车辆碰撞：0 |
| DAC | 可行驶区域合规：发生任何 `off_road` 事件为 0，否则为 1 |
| TTC | 若任一步按匀速外推会在 0.9 s 内撞上前车，则为 0，否则为 1 |
| C | 舒适度：纵向加速度在 [−4.05, 2.40] m/s²，横向加速度 ≤ 4.89 m/s²，加加速度 ≤ 4.13 m/s³，横摆角速度 ≤ 0.95 rad/s，横摆角加速度 ≤ 1.93 rad/s² |
| EP | 主车沿路线的行进距离除以录制车辆同一时间段内的行进距离（裁剪到 [0, 1]） |
| RC | 路线完成度：已行驶的录制路线比例 |
| score | NC × DAC × (5·TTC + 2·C + 5·EP) / 12 |

各子项沿用 NAVSIM/nuPlan 的 PDM 评分。它们是在 AETHER 重仿真的场景上计算的，因此绝对数值只适合在 AETHER 上比较不同算法，不能与 NAVSIM 排行榜直接对比。

## 导入自己的场景

场景包是一个目录 `data/scenes/<id>/`：

```text
scene.yaml          编号、数据集、帧间隔、车型、数据来源
rig.yaml            相机（序号、尺寸、K、安装位姿）和激光雷达安装位姿
recon/checkpoint.pth, recon/config.yaml      DriveStudio 重建结果
log.npz             录制的锚定传感器位姿、t = 0 时的相机位姿
actors.npz          重建的动态交通参与者（自动提取）
map/hdmap.json      车道（中心线、多边形）、可行驶区域      [可选]
map/network.net.xml, map/routes.rou.xml       LimSim 用的 SUMO 路网 [可选]
cache/              地面模型等派生数据
```

```bash
# 1. 用 DriveStudio（OmniRe）重建，例如 nuScenes 序号 300 = scene-0383
python scripts/train_scene.py --config_file third_party/drivestudio/configs/omnire.yaml \
  --output_root outputs/train --project nuscenes-0383 --run_name recon \
  dataset=nuscenes/6cams data.data_root=data/nuscenes/processed_10Hz/trainval data.scene_idx=300

# 2. 转换为场景包
python -m aether import nuscenes-0383 --dataset nuscenes \
  --checkpoint outputs/train/nuscenes-0383/recon/checkpoint_final.pth \
  --processed data/nuscenes/processed_10Hz/trainval/300

# 3. （nuScenes）生成带可行驶区域的高精地图和 LimSim 用的 SUMO 路网
python -m aether map nuscenes-0383 --nuscenes /data/nuscenes --location singapore-onenorth
# （nuPlan）从 nuPlan GeoPackage 地图提取车道及其轮廓（需要 nuplan-devkit）
python -m aether map nuplan-0038 --nuplan-maps /data/nuplan/maps --location us-nv-las-vegas-strip
```

修改 `map/routes.rou.xml` 可以改变 LimSim 的交通需求（`<vehicle depart=... departPos=... departSpeed=...>`，`route` 为路段序列）。

车辆资产：`python tools/build_vehicle_asset.py --input <3DRealCar 序列> --output data/assets/vehicles/<名称>`。

激光雷达相关研究工具（点云侧车模型训练、修复网络）位于 `aether/train/`、`aether/refine/` 和 `aether/tools/`，用法见各脚本的 docstring。训练输出统一写入 `outputs/train/<场景编号>/{recon,joint}`。

## 复现 v0.2 之前的结果

论文中的数字由 v0.1 流程得到。v0.2 改变了仿真行为：
- 背景碰撞盒放在局部地面上，而不是悬在相机高度；
- 用随地形变化的地面模型代替单一地平面；
- recorded 交通参与者按数据集帧率推进；
- 可行驶区域来自真实高精地图多边形；
- 主车以录制速度起步；
- 轨迹点按算法声明的坐标系解释；
- iLQR 跟踪器使用主车车型自身的参数，且不再有墙钟时间限制。

这些改动都会影响指标。如需复现已发表的数字，请使用 `v0.1` 标签（v0.1 流程的最后一次提交）及该版本的 README。

第二个演示场景以前叫 `nuscenes-0088`，它其实是 nuScenes trainval 的第 88 个场景，即 **scene-0106**，现已改名为 `nuscenes-0106`。`python -m aether migrate` 会自动重命名已有的场景包，旧编号也仍然可以加载。

## 许可证

AETHER 使用本仓库的许可证。第三方代码和数据保留各自的条款，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
