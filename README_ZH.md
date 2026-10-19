# AETHER

[English](README.md)

AETHER 用来在真实道路场景里闭环测试端到端自动驾驶算法（UniAD、VAD、LTF、VLA 模型，或者你自己的）。场景是用三维高斯重建出来的，算法看到的是接近真实的相机（和激光雷达）画面。背景车交给交通仿真来开：LimSim/SUMO、内置的 IDM，或者你自己写的模型都可以。

每跑一次都会得到一段鸟瞰图（BEV）视频和每辆车的轨迹表，可以当普通交通仿真的结果来分析。

```text
          ┌──────────── 规划算法（UniAD / VAD / LTF / VLA / 自定义）────────────┐
 图像 ────┤                                                                       ├── 轨迹点
          └───────────────────────────────────────────────────────────────────────┘
   ▲                                                                                │ 轨迹跟踪
   │ 渲染（高斯场景 + 交通车资产）                主车运动学（自行车模型）           ▼
 三维场景 ◄──── 车辆位姿 ◄──── 交通流模型（LimSim / IDM / 自定义）◄──── 主车状态
   │
   └─► 碰撞 · 驶出道路 · 评测指标 · BEV 视频 · trajectories.csv
```

- [快速开始](#快速开始)
- [跑闭环仿真](#跑闭环仿真)
- [输出文件](#输出文件)
- [接入自己的算法](#接入自己的算法)
- [接入交通流模型](#接入交通流模型)
- [场景配置文件](#场景配置文件)
- [Studio](#studio)
- [仿真是怎么做的](#仿真是怎么做的)
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
export PYTHONNOUSERSITE=1          # 免得 ~/.local 里的 torch 混进来
```

下载两个演示场景（nuScenes 的 scene-0383 和 scene-0106，带地图、SUMO 路网和两辆 [3DRealCar](https://xiaobiaodu.github.io/3drealcar/) 车辆模型）：

```bash
huggingface-cli login
python scripts/download_demo.py --repo https://huggingface.co/datasets/mikkochi/AETHER_Demo
python -m aether doctor            # 检查环境有没有装好
python -m aether scenes
```

先跑一次试试。这一步不需要任何算法，主车就沿着录制的路线开：

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent replay --no-refine
```

### 按需安装

| 组件 | 什么时候需要 | 怎么装 |
|---|---|---|
| LimSim | 用 LimSim 生成背景车 | `git clone --branch aether-integration https://github.com/siyingli99/LimSim.git ../LimSim`，再 `conda create -n limsim python=3.10 && pip install -r ../LimSim/aether_bridge/requirements.txt`。AETHER 会自己启动和关掉它；装在别处的话设一下 `LIMSIM_ROOT` / `LIMSIM_PYTHON`。 |
| UniAD | `--agent uniad` | 见 [UniAD](#uniad) |
| Difix3D | 修复渲染图像（`refine.enabled: true`） | 已经放在 `third_party/difix3d` 里，权重第一次用时自动下载 |
| SUMO `netconvert` | 给新场景生成地图 | `pip install eclipse-sumo` |

## 跑闭环仿真

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent uniad
python -m aether run configs/scenarios/nuscenes-0106.yaml --agent vad --cuda 1
python -m aether run configs/scenarios/nuscenes-0383-idm.yaml --agent replay --steps 20 --no-refine
python -m aether run my.yaml --agent uniad --set traffic.max_vehicles=5 --set sim.dt=0.5
```

`--agent` 可以填 `configs/agents/` 里的适配器（`uniad`、`vad`、`ltf`、`vla`、`gaussianfusion`）、`replay`，或者你自己的类，写成 `模块:类名` / `文件.py:类名`。`python -m aether agents` 可以看有哪些。

一次跑多个场景、多个算法：

```bash
python -m aether bench configs/benchmarks/nuscenes-demo.yaml --agents uniad,vad
python -m aether eval outputs/runs/<run>        # 用 CSV 重新算一遍指标
```

## 输出文件

每次运行一个文件夹：`outputs/runs/<场景>-<算法>-<时间>/`

| 文件 | 内容 |
|---|---|
| `trajectories.csv` | 每辆车每一步的状态 |
| `trajectories_10hz.csv` | 同上，重采样成 10 Hz |
| `trajectories_fcd.xml` | 10 Hz 轨迹的 SUMO FCD 格式，`sumo-gui` 和 SUMO/LimSim 的工具能直接打开 |
| `events.csv` | 碰撞、驶出道路、开出场景 |
| `metrics.json` | 评分（见[评测指标](#评测指标)） |
| `videos/bev.mp4` | 鸟瞰图，20 fps |
| `videos/cameras.mp4` | 算法每一步实际看到的画面 |
| `videos/overview.mp4` | 上面两个并排 |
| `run.json`、`logs/steps.jsonl` | 运行配置；每一步的控制量、规划轨迹和耗时 |
| `agent/` | 算法自己的日志和文件（UniAD 还会存一段它自己的可视化：`agent/uniad/video.mp4`） |

三个视频的时间轴是对齐的：每个视频的第 *t* 秒都是仿真的第 *t* 秒。

轨迹表的列（国际单位）：

| 列 | 含义 |
|---|---|
| `time_s`, `step` | 时间、步数 |
| `vehicle_id`, `role`, `type`, `source` | `ego` 或背景车编号，以及这辆车从哪来 |
| `x`, `y`, `z`, `yaw` | 车辆中心在道路坐标系中的位置（x 是主车起始朝向，y 向左） |
| `speed`, `vx`, `vy`, `accel`, `yaw_rate` | 运动状态 |
| `length`, `width`, `height` | 车辆尺寸 |
| `lane_id`, `s`, `d` | 所在车道、沿车道走了多远、偏离车道中心多少（左正） |
| `x_global`, `y_global`, `yaw_global` | 同一位置在数据集地图坐标里的值 |

## 接入自己的算法

算法可以**单独起一个进程、用自己的 conda 环境**（UniAD、VAD 这类依赖互相打架的就这么接），也可以**直接在 AETHER 里跑**（适合小模型、规则算法）。

### 算法拿到什么、要返回什么

每一步（默认 0.25 s）调用一次算法，它会拿到：

| 输入 | 说明 |
|---|---|
| 图像 | 每个相机一张 `uint8 [H, W, 3]` RGB 图（开了 Difix 就是修复后的） |
| 内外参 | 对应图像尺寸的 `K`；相机到车辆的位姿（车辆坐标原点在后轴，x 前、y 左、z 上） |
| 主车状态 | 速度、加速度、方向盘转角、横摆角速度、位姿 |
| 导航指令 | `0` 右转，`1` 左转，`2` 直行 |
| 激光雷达（可选） | `float32 [N, 4]` 点云 |

算法返回**轨迹点**：`N × 2` 个点，每隔 `dt`（默认 0.5 s）一个，坐标系自己声明。也可以直接返回控制量（加速度和转角变化率）。

AETHER 用一个简单的控制器让主车沿着这些点开，用的是真实车辆的轴距和加速、转向极限，所以算法没法让车以超过真车能力的速度过弯。想用 v0.1 的 iLQR 跟踪器，设 `sim.tracker: ilqr` 就行。

| `frame` | 轨迹点相对于 | 谁在用 |
|---|---|---|
| `rear_axle` | 后轴中心地面点 | LTF、VLA、GaussianFusion |
| `center` | 车辆中心 | |
| `lidar` | 车顶激光雷达 | UniAD、VAD |

| `axes` | 列顺序 |
|---|---|
| `flu` | （前，左） |
| `rfu` | （右，前），原版 UniAD/VAD 脚本的格式 |

**图像分辨率要注意：** nuScenes 演示场景按 800×450 渲染，是原图 1600×900 的一半。UniAD 和 VAD 的适配器会按模型评测时的方式放大回去，所以给它们用的时候别改渲染尺寸。其他算法可以设 `image_size`，AETHER 会帮你缩放图像、改好 `K`。

### A. 算法单独一个进程

**1. 在算法仓库里写一个小脚本。** 只依赖 NumPy。把 AETHER 加进 `PYTHONPATH`，或者直接把 `aether/agents/sdk/bridge.py` 拷到旁边。

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

用 `protocol: tuple` 时，`info` 长这样（和自带的几个适配器读的一样）：

| `info` 键 | 含义 |
|---|---|
| `ego_pos` | 后轴位置 `[右, 0, 前]`（m） |
| `ego_rot` | `[0, yaw, 0]`，航向角顺时针为正（rad） |
| `ego_velo`, `accelerate` | 速度（m/s）、加速度（m/s²） |
| `ego_steer`, `steer_rate` | 方向盘转角（rad）及其变化率（rad/s），右正 |
| `command`, `timestamp` | 导航指令、仿真时间（s） |
| `cam_params[cam]` | `intrinsic` {fovx, fovy（rad）, cx, cy, fx, fy, H, W}，`v2c`（车辆 → 相机 4×4），`l2c`（激光雷达 → 相机 4×4） |
| `ego_box`, `obj_boxes` | 真值包围盒 `[x, y, z, w, l, h, yaw]`，只用来调试 |
| `aether` | AETHER 自己的信息：主车、所有车辆、事件、路线进度 |

**2. 写个启动脚本**（`my_planner/aether_agent.sh`），用你环境里的 Python 启动：

```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
PY="${MY_PLANNER_PYTHON:-$HOME/miniconda3/envs/my_planner/bin/python}"
CUDA_VISIBLE_DEVICES="$1" "$PY" -u aether_agent.py "$1" "$2"
```

Python 要写完整路径。从另一个 Python 进程里启动时，`conda activate` 经常不生效。

**3. 在 `configs/agents/my_planner.yaml` 里登记一下：**

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

`${AETHER_EXTERNAL}` 是放 AETHER 的那个上级目录（可以用同名环境变量改）。`${workdir}` 是 `<run>/agent`，`planner.log` 就在这里。然后：

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent my_planner
```

算法崩了或卡住时，这次运行会停下来，并把它日志的最后几行打出来。超时时间用 `step_timeout_s` 和 `startup_timeout_s` 设。

### B. 算法直接在 AETHER 里跑

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

`obs` 里有 `time`、`step`、`images`、`cameras`、`ego`、`command`、`lidar` 和 `info`。

### 自带的适配器

| 算法 | 仓库（和 AETHER 放同一级） | 相机 | 激光雷达 | 规划输出 |
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

背景车全部由一个交通流模型来管。每一步 AETHER 把主车状态交给它，它把自己所有车的新位置交回来，渲染、碰撞检测和记录都由 AETHER 负责。写过微观交通仿真的话，应该会觉得很熟悉。

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
  assets: [3drealcar_2024_06_04_15_34_51]   # 用来渲染的三维车辆模型（轮流使用）
  max_vehicles: 6                            # 同时最多渲染几辆
  model: {class: my_traffic:MyCarFollowing, params: {n: 3, v0: 9.0}}
```

在 Studio 里把背景交通选成 "Python class"，填上 `my_traffic:MyCarFollowing` 和参数，效果一样。

几点说明：

- 位置用车辆中心，在道路坐标系里（单位米，x 是主车起始朝向，y 向左，航向角逆时针为正），和 `scene.hdmap` 是同一个坐标系。
- 同一辆车的 `id` 每一步要保持不变；车可以随时出现、消失。
- 车放到重建出来的路面上这件事 AETHER 会处理，你只用管 x、y 和朝向。
- 地图：`scene.hdmap.lanes[i]` 有 `id, centerline, width, polygon, incoming, outgoing`；`lane.project(x, y)` 返回 `(s, d, heading)`。

自带的模型（`traffic.model.type`）：

| 类型 | 说明 |
|---|---|
| `limsim` | 在场景路网上跑 LimSim（SUMO），自动启动 |
| `idm` | 沿地图车道的 IDM 跟驰，例如 `{type: idm, count: 3, desired_speed: 10}` |
| `replay` | 回放 JSON 文件里的轨迹 |
| `external` | 任何会说下面这套 TCP 协议的程序 |
| `class` | 你自己的 `TrafficModel` |

**接其他仿真软件（C++、MATLAB、VISSIM、TESS NG 等）：** 通过 TCP 连进来，一行一条 JSON。协议在 `aether/traffic/protocol.py`，只依赖标准库，可以直接拷走用。消息只有四种：`hello`、`reset`、`tick {target_time_s, ego}`、`shutdown`；服务端回一帧 `{time_s, vehicles: [{id, x, y, z, yaw, speed, length, width, height, type}]}`。

```yaml
traffic:
  model: {type: external, server: my_sim, host: 127.0.0.1, port: 9000}
```

**只想控制个别车**（比如做一个加塞），不用写整个交通流模型：

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

单独控制的车可以和交通流模型一起用。这些车不会自己避让，所以别让它们的路线交叉。想让原录制里的某辆车动起来、而不是换成三维车辆模型，加一句 `render: {original: RigidNodes/3}`。

## 场景配置文件

```yaml
# configs/scenarios/nuscenes-0383.yaml
scene: nuscenes-0383                 # data/scenes/ 下的场景包
sim: {dt: 0.25, max_steps: 39, seed: 0, tracker: pure_pursuit}   # 或 ilqr
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
output: {bev_fps: 20, trajectory_hz: 10, videos: true}
agent: {name: uniad, cuda: "0"}               # --agent 的默认值
```

`extends: other.yaml` 可以继承另一个配置。字段名写错会直接报错，不会被悄悄忽略。

车型在 `configs/vehicles/` 里：`renault_zoe`（nuScenes）、`chrysler_pacifica`（PandaSet）和通用的 `sedan`。车型决定了车的尺寸、轴距、加速和转向极限，还有相机装在哪。

## Studio

不想手写 YAML 的话，可以用网页界面：

```bash
python -m aether studio            # http://127.0.0.1:8765
```

- **Scenario**：选场景、选算法；在车道上放车、拖动调整位置；还能挪主车起点、选交通流模型、设什么情况下停止，最后可以导出成 YAML。
- **Live**：边跑边看鸟瞰图、相机画面和日志。
- **Results**：每次运行的分数、事件、视频，所有文件都能下载。

改前端：`pnpm --dir web install && pnpm --dir web dev`；`pnpm --dir web build` 重新生成 `web/dist`。

## 仿真是怎么做的

### 坐标系

对外的东西（CSV、地图、交通流模型）统一用**道路坐标系**：x 指向主车 t = 0 时的朝向，y 向左，z 向上。车辆坐标的原点在后轴中心地面（x 前、y 左）。各数据集自己的坐标约定在导入时就转换好了，用的时候不用管。

### 路面

重建结果里没有现成的路面，AETHER 会结合录制时的行驶轨迹和周围重建出的路面，为每个场景估一个地面（缓存在 `data/scenes/<id>/cache/ground.npz`）。路沿、绿化带、树冠会被过滤掉，坡度最大按 10 % 算。所有车都贴着这个地面走，上下坡跟得上，也不会飘起来或陷下去。

### 各数据集

| | nuScenes | PandaSet | nuPlan |
|---|---|---|---|
| 主车（`vehicle: auto`） | Renault Zoe | Chrysler Pacifica | sedan |
| 相机 | 原生 6 个 | 6 个，按 nuScenes 命名对应 | 6 个，按 nuScenes 命名对应 |
| 高精地图 | 车道 + 可行驶区域 | 无（不判断驶出道路） | 带轮廓的车道 |
| 起步速度 | 城市低速 | 经常 10–17 m/s | 不一定 |

UniAD、VAD 是在 nuScenes 上训练的，到 PandaSet 和 nuPlan 上看到的相机不一样，效果差一些是正常的。PandaSet 场景常常一开始速度就很快，nuScenes 训练出来的算法可能会急刹。

### 碰撞和违规

| 事件 | 什么时候触发 |
|---|---|
| `vehicle_collision` | 主车和别的车的包围盒重叠。日志里会记录是谁的责任：主车停着或者被追尾就不算主车的。 |
| `static_collision` | 主车撞上场景里的东西：建筑、杆子、路沿、停着的车 |
| `off_road` | 主车有一个角超出可行驶区域 0.3 m 以上 |
| `off_scene` | 主车开到了没有重建的地方，画面已经没有参考意义 |

哪些事件会让这次运行停下来，由 `termination.stop_on` 决定。不管停不停，所有事件都会记录并计入评分。

### 评测指标

| 指标 | 含义 |
|---|---|
| NC | 没有主车责任的碰撞：1；只撞了场景：0.5；主车撞车：0 |
| DAC | 没开出道路：1，否则 0 |
| TTC | 从来没有在 0.9 s 内快要撞上前车：1，否则 0 |
| C | 舒适度：加速度、急动度和横摆都在正常驾驶范围内 |
| EP | 主车开了多远，和录制时比 |
| RC | 录制路线走完了多少 |
| score | NC × DAC × (5·TTC + 2·C + 5·EP) / 12 |

这套指标参照 NAVSIM/nuPlan 的 PDM 分数。适合在 AETHER 上比较不同算法，但不要拿去和 NAVSIM 排行榜上的数字直接比。

## 导入自己的场景

一个场景就是一个文件夹 `data/scenes/<id>/`：

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

# 2. 转成场景文件夹
python -m aether import nuscenes-0383 --dataset nuscenes \
  --checkpoint outputs/train/nuscenes-0383/recon/checkpoint_final.pth \
  --processed data/nuscenes/processed_10Hz/trainval/300

# 3. （nuScenes）生成高精地图和 LimSim 用的 SUMO 路网
python -m aether map nuscenes-0383 --nuscenes /data/nuscenes --location singapore-onenorth
# （nuPlan）从 nuPlan 地图提取车道（需要 nuplan-devkit）
python -m aether map nuplan-0038 --nuplan-maps /data/nuplan/maps --location us-nv-las-vegas-strip
```

想调 LimSim 的车流量，改 `map/routes.rou.xml`。

做新的三维车辆模型：`python tools/build_vehicle_asset.py --input <3DRealCar 序列> --output data/assets/vehicles/<名称>`。

激光雷达相关的工具在 `aether/train/`、`aether/refine/` 和 `aether/tools/`，用法写在各脚本开头的说明里。

## 许可证

AETHER 使用本仓库的许可证。第三方代码和数据遵循各自的条款，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
