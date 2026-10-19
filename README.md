# AETHER

[中文说明](README_ZH.md)

AETHER is a closed-loop simulator for autonomous driving built on reconstructed 3D Gaussian scenes. An end-to-end planner (UniAD, VAD, LTF, a VLA model or your own) drives the ego vehicle through photorealistic camera (and LiDAR) renders. Background traffic comes from a 2D traffic-flow model: LimSim/SUMO, the built-in IDM, or your own code. Every run produces a bird's-eye-view video and a per-vehicle trajectory table that can be analysed like any traffic-simulation output.

```text
          ┌────────────── planner (UniAD / VAD / LTF / VLA / yours) ─────────────┐
 images ──┤                                                                       ├── waypoints
          └───────────────────────────────────────────────────────────────────────┘
   ▲                                                                                │ iLQR tracking
   │ render (Gaussians + traffic assets)          ego kinematics (bicycle model)   ▼
 3D scene ◄──── vehicle poses ◄──── traffic model (LimSim / IDM / yours) ◄──── ego state
   │
   └─► collisions · off-road · metrics · BEV video · trajectories.csv
```

- [Quick start](#quick-start)
- [Run a closed loop](#run-a-closed-loop)
- [Outputs](#outputs)
- [Plug in an end-to-end planner](#plug-in-an-end-to-end-planner)
- [Plug in a traffic model](#plug-in-a-traffic-model)
- [Scenarios](#scenarios)
- [Studio](#studio)
- [Simulation details](#simulation-details): frames, ground, collisions, metrics
- [Scenes: import your own](#scenes-import-your-own)

## Quick start

```bash
git clone https://github.com/siyingli99/Aether.git && cd Aether
conda create -p .conda/aether python=3.10 -y && conda activate ./.conda/aether
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install -e third_party/drivestudio/third_party/smplx
export PYTHONNOUSERSITE=1          # keep a ~/.local torch from shadowing the env
```

Download the demo scenes (nuScenes scene-0383 and scene-0106, reconstructions, HD maps, SUMO networks and two [3DRealCar](https://xiaobiaodu.github.io/3drealcar/) vehicle assets). The script converts them into scene bundles under `data/scenes/`:

```bash
huggingface-cli login
python scripts/download_demo.py --repo https://huggingface.co/datasets/mikkochi/AETHER_Demo
python -m aether doctor            # environment, scenes, planners
python -m aether scenes
```

First run, without any learned planner (the ego replays the recorded route):

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent replay --no-refine
```

### Optional components

| Component | Needed for | Install |
|---|---|---|
| LimSim | `traffic.model: {type: limsim}` | `git clone --branch aether-integration https://github.com/siyingli99/LimSim.git ../LimSim`, then `conda create -n limsim python=3.10 && pip install -r ../LimSim/aether_bridge/requirements.txt`. AETHER starts and stops the LimSim server itself. Set `LIMSIM_ROOT` / `LIMSIM_PYTHON` if they are elsewhere. |
| UniAD | `--agent uniad` | see [UniAD](#uniad) |
| Difix3D | `refine.enabled: true` | vendored in `third_party/difix3d`; the `nvidia/difix` weights are downloaded from Hugging Face on first use |
| SUMO `netconvert` | `python -m aether map` (new scenes only) | `pip install eclipse-sumo` |

## Run a closed loop

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent uniad
python -m aether run configs/scenarios/nuscenes-0106.yaml --agent vad --cuda 1
python -m aether run configs/scenarios/nuscenes-0383-idm.yaml --agent replay --steps 20 --no-refine
python -m aether run my.yaml --agent uniad --set traffic.max_vehicles=5 --set sim.dt=0.5
```

`--agent` takes an adapter name from `configs/agents/` (`uniad`, `vad`, `ltf`, `vla`, `gaussianfusion`), `replay`, or `package.module:Class` / `path/to/file.py:Class` for an in-process agent. `python -m aether agents` lists them.

A benchmark runs several scenarios × agents and aggregates the metrics:

```bash
python -m aether bench configs/benchmarks/nuscenes-demo.yaml --agents uniad,vad
python -m aether eval outputs/runs/<run>        # recompute metrics.json from the CSVs
```

## Outputs

Every run writes one directory, `outputs/runs/<scene>-<agent>-<YYYYmmdd-HHMMSS>/`:

| File | Content |
|---|---|
| `trajectories.csv` | every vehicle (ego and background) at every simulation step |
| `trajectories_10hz.csv` | the same on a uniform 10 Hz clock (`output.trajectory_hz`), interpolated with the vehicles' own velocities |
| `trajectories_fcd.xml` | the 10 Hz table as SUMO floating-car data; coordinates are those of the scene's `map/network.net.xml`, so `sumo-gui` and SUMO/LimSim analysis tools read it directly |
| `events.csv` | collisions, off-road, off-scene: time, type, other vehicle, at-fault flag |
| `metrics.json` | NC, DAC, TTC, comfort, EP, RC and the aggregate score ([metrics](#metrics)) |
| `videos/bev.mp4` | bird's-eye view at 20 fps (`output.bev_fps`), motion interpolated between steps |
| `videos/cameras.mp4` | the rendered camera images exactly as the planner received them, one frame per step |
| `videos/overview.mp4` | BEV and cameras side by side |
| `run.json` | scenario, agent, vehicle model, timing, LiDAR source |
| `logs/steps.jsonl` | per-step control, timing and events |
| `agent/` | planner work directory: `planner.log`, FIFOs and anything the planner saves |
| `logs/io/`, `logs/render_poses.json` | only with `output.io_trace: true`: the images the planner received at each step and the render poses, for debugging adapters and for closed-loop distillation (`aether/refine/distill_from_closed_loop.py`) |

All three videos share one clock: second *t* of every video shows simulation time *t*. The camera video plays one frame per simulation step (4 fps at `dt = 0.25 s`). The BEV interpolates vehicle motion between steps with cubic Hermite splines on position and velocity, so it plays smoothly.

Trajectory columns (SI units):

| Column | Meaning |
|---|---|
| `time_s`, `step` | simulation time and step index |
| `vehicle_id`, `role`, `type`, `source` | `ego` or a background id; `role` ∈ {ego, background}; `source` ∈ {ego, model, vehicles, recorded} |
| `x`, `y`, `z`, `yaw` | box centre in the scene road frame (x along the initial ego heading, y left, z up, yaw counter-clockwise) |
| `speed`, `vx`, `vy`, `accel`, `yaw_rate` | speed, velocity components in the road frame, longitudinal acceleration, yaw rate |
| `length`, `width`, `height` | vehicle box |
| `lane_id`, `s`, `d` | matched HD-map lane, arc length along it and signed lateral offset (left +) |
| `x_global`, `y_global`, `yaw_global` | the same pose in the dataset map frame (e.g. nuScenes map coordinates) |

## Plug in an end-to-end planner

A planner either runs in **its own process and conda environment** (the usual case: UniAD, VAD, LTF and VLA all have conflicting dependencies) or **inside AETHER's process** (lightweight models, rule-based planners, research prototypes).

### What the planner receives and returns

The planner is called once per simulation step (`sim.dt`, 0.25 s by default):

| Input | Content |
|---|---|
| images | `uint8 [H, W, 3]` RGB per camera, after Difix refinement when enabled |
| intrinsics | 3×3 `K` for the delivered image size |
| extrinsics | camera → vehicle (`mount`; OpenCV camera axes; vehicle = rear axle, x forward, y left, z up) |
| ego state | speed, acceleration, steering angle and rate, yaw rate; pose in the road frame |
| command | `0` right, `1` left, `2` straight, from the recorded route 3 s ahead |
| LiDAR (optional) | `float32 [N, 4]` (x, y, z, intensity) in the rig's LIDAR_TOP frame; SplatAD sweep if a sidecar exists, else depth back-projection |

It returns **waypoints** (`Plan`): `N × 2` positions at `dt` (default 0.5 s) intervals, the first one at `+dt`, in a declared frame. Alternatively it returns a direct **control** (`Control`: acceleration in m/s², steering rate in rad/s, left positive). AETHER converts waypoints to the rear axle of the ego vehicle and tracks them with an iLQR controller that uses the ego's wheelbase and acceleration and steering limits. The ego then moves with a kinematic bicycle model; its steering angle is capped by the tyre-friction limit `v² tan(δ) / L ≤ max_lat_accel` (default 8 m/s², set per vehicle in `configs/vehicles/`), so a planner cannot corner faster than a real car.

| `frame` | Waypoints are relative to | Typical planner |
|---|---|---|
| `rear_axle` | rear-axle centre on the ground (nuScenes ego / nuPlan / NAVSIM) | LTF, VLA, GaussianFusion |
| `center` | geometric centre of the vehicle box | |
| `lidar` | the LIDAR_TOP mount of the scene rig | UniAD, VAD |

| `axes` | Column order |
|---|---|
| `flu` | (forward, left) |
| `rfu` | (right, forward), the layout of the original UniAD/VAD closed-loop scripts |

**Image resolution.** Cameras are rendered at the resolution of the scene's rig: 800×450 for the nuScenes demo scenes, i.e. half the 1600×900 nuScenes images. The UniAD and VAD adapters upsample to 1600×900 and double the intrinsics internally, exactly as they did when the models were evaluated. Do not change the render resolution for them: other resolutions break that assumption and measurably degrade both planners. For other planners, set `image_size` in the adapter YAML (or `EgoAgent.image_size`); AETHER then resizes the images and rescales `K` before delivery.

### A. External planner (own process)

**1. Write the planner script** in the planner's repository. It only depends on NumPy; put the AETHER checkout on `PYTHONPATH` or copy `aether/agents/sdk/bridge.py` next to it.

```python
# my_planner/aether_agent.py
import numpy as np
from aether.agents.sdk import AdapterAgent, serve

class MyPlanner(AdapterAgent):
    def setup(self):                         # once, after CUDA_VISIBLE_DEVICES is set
        self.model = load_my_model()

    def infer(self, obs, info):
        front = obs["rgb"]["CAM_FRONT"]      # uint8 H x W x 3
        K = info["cam_params"]["CAM_FRONT"]["intrinsic"]   # fovx, fovy, cx, cy, H, W (+ fx, fy)
        speed = info["ego_velo"]             # m/s
        cmd = info["command"]                # 0 right, 1 left, 2 straight
        fwd_left = self.model(front, speed, cmd)            # N x 2 (forward, left), metres
        return {"waypoints": fwd_left, "frame": "rear_axle", "axes": "flu", "dt": 0.5}
        # or: return np.column_stack([-fwd_left[:, 1], fwd_left[:, 0]])  # bare N x 2 in the YAML's frame/axes
        # or: return {"accel": 0.5, "steer_rate": 0.0}

if __name__ == "__main__":
    serve(MyPlanner())                       # argv: <cuda_id> <workdir>
```

With `protocol: tuple`, `info` has the layout the supplied UniAD / VAD / LTF / VLA adapters read:

| `info` key | Meaning |
|---|---|
| `ego_pos` | `[right, 0, forward]` rear-axle position in the road frame (m) |
| `ego_rot` | `[0, yaw, 0]`, yaw clockwise-positive (rad) |
| `ego_velo`, `accelerate` | speed (m/s), acceleration (m/s²) |
| `ego_steer`, `steer_rate` | steering angle (rad) and rate (rad/s), right-positive |
| `command`, `timestamp` | navigation command, simulation time (s) |
| `cam_params[cam]` | `intrinsic` {fovx, fovy (rad), cx, cy, fx, fy, H, W}, `v2c` (vehicle → camera 4×4), `l2c` (LiDAR → camera 4×4) |
| `ego_box`, `obj_boxes` | `[x, y, z, w, l, h, yaw]` boxes in the road frame (ground truth, for debugging only) |
| `aether` | AETHER's own road-frame info: ego, all agents, events, route progress |

**2. Launcher** (`my_planner/aether_agent.sh`): start the script with the planner environment's interpreter.

```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
PY="${MY_PLANNER_PYTHON:-$HOME/miniconda3/envs/my_planner/bin/python}"
CUDA_VISIBLE_DEVICES="$1" "$PY" -u aether_agent.py "$1" "$2"
```

Call the environment's Python by path: `conda activate` does not reliably switch interpreters when the script is started from another Python process.

**3. Declare it** in `configs/agents/my_planner.yaml`:

```yaml
name: my_planner
title: My planner
command: [bash, "${AETHER_EXTERNAL}/my_planner/aether_agent.sh", "${cuda}", "${workdir}"]
ready: ["ready"]                 # text in the log once the model is loaded
protocol: tuple                  # (sensors, info) as above; "observation" sends the Observation dict
input:
  cameras: [CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT]   # only these are rendered
  image_size: null               # [W, H]; null = scene render size
  lidar: false
output: {dt: 0.5, frame: rear_axle, axes: rfu}          # for bare N x 2 replies
```

`${AETHER_EXTERNAL}` defaults to the directory that contains the AETHER checkout (override with `AETHER_EXTERNAL`). `${workdir}` is `<run>/agent`; the FIFOs and `planner.log` live there. Then:

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent my_planner
```

`AETHER_ROOT`, `AETHER_EXTERNAL`, the step timeout (`step_timeout_s`, default 900 s) and the startup timeout (`startup_timeout_s`) can be set per adapter. A crashed or hung planner fails the run with the tail of its log.

### B. In-process planner

```python
# my_agents.py
import numpy as np
from aether.agents import EgoAgent, Plan

class KeepLane(EgoAgent):
    cameras = ["CAM_FRONT"]            # rendered and delivered; [] = no images
    image_size = (640, 360)            # optional resize; K is rescaled

    def act(self, obs):                # obs: aether.agents.Observation
        img = obs.images["CAM_FRONT"]; K = obs.cameras["CAM_FRONT"]["K"]
        v = max(obs.ego["speed"], 3.0)
        s = v * 0.5 * np.arange(1, 7)
        return Plan(np.column_stack([s, np.zeros(6)]), dt=0.5, frame="rear_axle", axes="flu")
```

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent my_agents.py:KeepLane
```

`Observation` has `time`, `step`, `images`, `cameras` (K, mount, size), `ego` (x, y, yaw, speed, accel, steer, steer_rate, yaw_rate, rear_axle, size), `command`, `lidar` and `info` (all agents, events, route progress). `setup(context)` receives the scene id, vehicle profile, `dt`, the recorded route and the camera names.

### Supplied adapters

| Agent | Repository (next to AETHER) | Cameras | LiDAR | Plan |
|---|---|---|---|---|
| `uniad` | vendored: `third_party/uniad_sim` | 6 | – | lidar · rfu · 0.5 s |
| `vad` | `VAD_SIM` (`tools/e2e.sh`) | 6 | – | lidar · rfu · 0.5 s |
| `ltf` | `NAVSIM` (`ltf_e2e.sh`, Latent TransFuser) | 3 front | ✓ | rear_axle · rfu · 0.5 s |
| `vla` | `Impromptu-VLA` (`tools/e2e.sh`, Qwen2.5-VL) | 3 front | – | rear_axle · rfu · 0.5 s |
| `gaussianfusion` | `GaussianFusion` (`tools/e2e.sh`) | 4 | ✓ | rear_axle · rfu · 0.5 s |

#### UniAD

```bash
conda create -n uniad python=3.10 -y && conda activate uniad
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install openmim && MMCV_WITH_OPS=1 pip install -v mmcv-full==1.7.2
pip install mmdet==2.26.0 mmdet3d==1.0.0rc6 mmsegmentation==0.29.1
pip install -r third_party/uniad_sim/requirements.txt
mkdir -p checkpoints && wget -O checkpoints/uniad_base_e2e.pth \
  https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth
export UNIAD_PYTHON=$(conda run -n uniad which python)   # or create .conda/uniad
```

## Plug in a traffic model

Background traffic is a **traffic model**: it owns all background vehicles and advances them together once per simulation step. This is the same contract as a microscopic traffic simulator. AETHER passes the ego state; the model returns the vehicles present at the new time; AETHER renders each one as a Gaussian vehicle asset, checks it for collisions and logs it.

```python
# my_traffic.py
from aether.traffic import AgentState, TrafficModel

class MyCarFollowing(TrafficModel):
    """Vehicles on lane centrelines of the HD map with my own car-following law."""

    def __init__(self, n=3, v0=10.0, **params):          # scenario params → kwargs
        super().__init__(**params)
        self.n, self.v0 = n, v0

    def reset(self, scene):                               # scene: TrafficScene
        # scene.ego (AgentState), scene.hdmap (lanes, drivable area), scene.dt,
        # scene.network (SUMO net.xml, rou.xml), scene.seed
        lane = scene.hdmap.lanes[0]
        x, y = lane.centerline[0]
        return [AgentState("car-1", x, y, yaw=0.0, speed=self.v0)]

    def step(self, t, dt, ego, vehicles):
        # advance from t - dt to t; return the vehicles present at t
        return [v.moved(dt, accel=0.0) for v in vehicles]

    def close(self):
        pass
```

```yaml
traffic:
  assets: [3drealcar_2024_06_04_15_34_51]   # Gaussian vehicles used for rendering (cycled)
  max_vehicles: 6                            # rendered at most at once
  model: {class: my_traffic:MyCarFollowing, params: {n: 3, v0: 9.0}}
```

| | |
|---|---|
| Frame | the scene road frame: x along the initial ego heading, y left, yaw counter-clockwise (rad), metres; the same frame as `scene.hdmap` |
| Position | vehicle **box centre** |
| `AgentState` | `id, x, y, yaw, speed, length, width, height, accel, type, lane_id`; ids must be stable between steps; vehicles may appear and disappear |
| Rendering | each vehicle gets a Gaussian asset slot and is placed on the reconstructed ground (height and pitch from the ground model); its size is the asset's |
| HD map | `scene.hdmap.lanes[i]`: `id, centerline (N×2), width, polygon, incoming, outgoing`; `lane.project(x, y) → (s, d, heading)`; `scene.hdmap.match(x, y, yaw)` |

Built-in models (`traffic.model.type`):

| Type | Description |
|---|---|
| `limsim` | LimSim (SUMO) on the scene's network `map/network.net.xml` with routes `map/routes.rou.xml`; started automatically |
| `idm` | Intelligent Driver Model on HD-map lane paths: `{type: idm, count: 3, desired_speed: 10, time_headway: 1.5, min_gap: 2, max_accel: 1.5, comfort_decel: 2}` or explicit `vehicles: [{id, lane, s, speed}]`; the ego is a leader |
| `replay` | recorded frames from a JSON file |
| `external` | any process speaking the TCP protocol below |
| `class` | your `TrafficModel` |

**External simulators (C++, MATLAB, VISSIM, TESS NG …).** Run a TCP server that speaks `aether.traffic.v1`: newline-delimited JSON, see `aether/traffic/protocol.py`. The protocol module is standard-library only and can be copied. Requests: `hello` → `{server}`, `reset` → `{frame}`, `tick {target_time_s, ego}` → `{frame}`, `shutdown`. A frame is `{time_s, vehicles: [{id, x, y, z, yaw, speed, length, width, height, type}]}` in the road frame above.

```yaml
traffic:
  model: {type: external, server: my_sim, host: 127.0.0.1, port: 9000}
```

**Per-vehicle controllers.** To script single vehicles (e.g. a cut-in), give each its own controller instead of a traffic model:

```python
from aether.traffic import VehicleController

class CutIn(VehicleController):
    def setup(self, context):            # context: {"vehicle": spec, "params": {...}, "scene": TrafficScene}
        self.start = context["vehicle"]["start"]
    def step(self, time_s, actor, ego):  # -> new pose of the box centre
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

Vehicles and a traffic model can be combined. Add `render: {original: RigidNodes/3}` to a scripted vehicle to move a reconstructed car instead of an asset.

## Scenarios

```yaml
# configs/scenarios/nuscenes-0383.yaml
scene: nuscenes-0383                 # bundle in data/scenes/
sim: {dt: 0.25, max_steps: 39, seed: 0}
ego:
  vehicle: auto                      # vehicle profile; auto = the recording car of the dataset
  start: logged                      # or {x, y, yaw} (box centre, road frame)
  speed: logged                      # or a number (m/s)
traffic:
  hide_originals: true               # hide reconstructed dynamic objects
  keep: []                           # ...except these, e.g. [RigidNodes/3], [RigidNodes/*] or ["*"]; they replay their recording
  assets: [3drealcar_2024_06_04_15_34_51]
  max_vehicles: 3
  model: {type: limsim, port: 8766}
refine: {enabled: true, required: true}       # Difix on every planner image
termination: {stop_on: [vehicle_collision, static_collision, off_scene], at_fault_only: false}
output: {bev_fps: 20, trajectory_hz: 10, videos: true}
agent: {name: uniad, cuda: "0"}               # default for --agent
```

`extends: other.yaml` inherits from another scenario. Unknown keys are errors.

Vehicle profiles live in `configs/vehicles/` (`renault_zoe`, the nuScenes car; `chrysler_pacifica`, the PandaSet car; `sedan`). The profile sets the collision box, wheelbase, rear overhang and acceleration and steering limits. It also sets where the cameras sit, because the camera mounts are calibrated relative to the rear axle of the profile.

## Studio

```bash
python -m aether studio            # http://127.0.0.1:8765
```

- **Scenario**: pick a scene and a planner. The map editor shows the HD map, the drivable area, the recorded route and the reconstructed actors. Add vehicles on lanes (their heading snaps to the lane), drag and rotate them, and set each to parked, constant speed or a Python controller class. Keep or re-drive reconstructed cars, move the ego start, choose the traffic model (none, IDM, LimSim when the scene has a SUMO network, or your own `TrafficModel` class with JSON parameters), termination events and outputs, and export the scenario as YAML.
- **Live**: the BEV and the rendered cameras stream while the run is in progress, together with the log.
- **Results**: every run with its metrics, events, the three videos and CSV downloads.

Frontend development: `pnpm --dir web install && pnpm --dir web dev` (proxies the API on :8765); `pnpm --dir web build` rebuilds `web/dist`.

## Simulation details

### Frames

| Frame | Definition |
|---|---|
| road | x along the recorded ego heading at t = 0, y left, z up; origin on the ground below the anchor sensor at t = 0. All public state, CSVs, HD maps and traffic models use it. |
| vehicle | rear-axle centre on the ground, x forward, y left, z up (nuScenes ego convention) |
| scene | the reconstruction's world frame; dataset-specific, used only internally (`configs` → `aether/geometry/frames.py`) |

PandaSet scenes use x right, y forward, z up in the scene frame (the PandaSet ego frame) and nuScenes scenes use x right, y down, z forward (the front camera). Both map to the same road frame, so scenarios and planners never see the difference. For ego-pose anchored datasets (PandaSet, nuPlan) the logged ego pose is converted from the dataset's body axes to the vehicle frame when a scene is imported; bundles imported before October 2026 are corrected when loaded.

### Ground

The reconstruction does not contain a separate ground model. AETHER estimates the ground surface once per scene (cached in `data/scenes/<id>/cache/ground.npz`) from two sources. The recorded ego trajectory gives the road profile along the route. The background Gaussians give the absolute level elsewhere: for each 2 m cell, the lowest dense layer of splats is taken as the ground, which rejects floaters below the road and objects above it. The two are blended with a slope limit and smoothed onto a 0.5 m grid. The ego and every traffic vehicle rest on this surface: their height, pitch and roll come from the ground under their wheels and axles. Cameras therefore follow slopes and kerbs, and parked or moving vehicles do not float or sink.

### Collisions and violations

| Event | Criterion |
|---|---|
| `vehicle_collision` | the ego box (size from the ego vehicle profile) overlaps a background vehicle box. Boxes are swept in ≤ 0.5 m sub-steps, so fast vehicles cannot pass through each other. The event records the side (front, rear, left, right) and fault: not at fault if the ego was stationary, or if it was hit from behind by a vehicle at least as fast. |
| `static_collision` | more than `static_threshold` (50) dense background splats 0.3–2.5 m above the local ground inside the ego box: buildings, poles, kerbs, parked cars of the reconstruction. Isolated floaters (fewer than 8 neighbours within 0.3 m) and splats inside the recorded ego corridor are ignored. |
| `off_road` | a corner of the ego box is more than `off_road_tolerance` (0.3 m) outside the HD-map drivable area (real nuScenes drivable-area polygons; lane polygons otherwise) |
| `off_scene` | for `off_scene_steps` (4) consecutive steps, the ego is more than 5 m from any reconstructed geometry, so the renders are no longer meaningful |

`termination.stop_on` selects which events end the episode. All events are always logged and scored.

### Metrics

| Metric | Definition |
|---|---|
| NC | no at-fault collision: 1; collision with the static scene only: 0.5; at-fault vehicle collision: 0 |
| DAC | drivable-area compliance: 0 if any `off_road` event, else 1 |
| TTC | 0 if, at any step, a constant-velocity projection reaches a vehicle ahead within 0.9 s, else 1 |
| C | comfort: longitudinal acceleration in [−4.05, 2.40] m/s², lateral ≤ 4.89 m/s², jerk ≤ 4.13 m/s³, yaw rate ≤ 0.95 rad/s, yaw acceleration ≤ 1.93 rad/s² |
| EP | ego progress along the route divided by the progress of the recorded vehicle over the same time (clipped to [0, 1]) |
| RC | route completion: fraction of the recorded route driven |
| score | NC × DAC × (5·TTC + 2·C + 5·EP) / 12 |

The sub-scores follow the PDM score of NAVSIM/nuPlan. They are computed on AETHER's re-simulated scenes, so absolute values are comparable between planners on AETHER, not with NAVSIM leaderboards.

## Scenes: import your own

A scene bundle is a directory `data/scenes/<id>/`:

```text
scene.yaml          id, dataset, frame period, vehicle profile, source
rig.yaml            cameras (index, size, K, mount) and LiDAR mount
recon/checkpoint.pth, recon/config.yaml      DriveStudio reconstruction
log.npz             recorded anchor-sensor poses, camera poses at t = 0
actors.npz          reconstructed dynamic actors (extracted automatically)
map/hdmap.json      lanes (centrelines, polygons), drivable area      [optional]
map/network.net.xml, map/routes.rou.xml       SUMO network for LimSim [optional]
cache/              ground model and other derived data
```

```bash
# 1. reconstruct with DriveStudio (OmniRe), e.g. nuScenes index 300 = scene-0383
python scripts/train_scene.py --config_file third_party/drivestudio/configs/omnire.yaml \
  --output_root outputs/train --project nuscenes-0383 --run_name recon \
  dataset=nuscenes/6cams data.data_root=data/nuscenes/processed_10Hz/trainval data.scene_idx=300

# 2. turn it into a bundle
python -m aether import nuscenes-0383 --dataset nuscenes \
  --checkpoint outputs/train/nuscenes-0383/recon/checkpoint_final.pth \
  --processed data/nuscenes/processed_10Hz/trainval/300

# 3. (nuScenes) HD map with drivable area + SUMO network for LimSim
python -m aether map nuscenes-0383 --nuscenes /data/nuscenes --location singapore-onenorth
# (nuPlan) lanes with their outlines from the nuPlan GeoPackage maps (needs nuplan-devkit)
python -m aether map nuplan-0038 --nuplan-maps /data/nuplan/maps --location us-nv-las-vegas-strip
```

Edit `map/routes.rou.xml` to change LimSim demand (`<vehicle depart=... departPos=... departSpeed=...>` with `route` edges).

Vehicle assets: `python tools/build_vehicle_asset.py --input <3DRealCar sequence> --output data/assets/vehicles/<name>`.

LiDAR research tools (point-cloud sidecar training, refiners) are in `aether/train/`, `aether/refine/` and `aether/tools/`; see the docstrings of each script. Training outputs go to `outputs/train/<scene-id>/{recon,joint}`.

## Reproducing results from before v0.2

The paper's numbers were produced with the v0.1 pipeline. v0.2 changes the simulation: the background-collision box rests on the local ground instead of floating at camera height, and a terrain-following ground model replaces the single ground plane. Recorded actors advance at the dataset frame rate, and the drivable area comes from real HD-map polygons. The ego starts at its recorded speed, waypoints are interpreted in the planner's declared frame, and the iLQR tracker uses the ego vehicle's own parameters without a wall-clock limit. These changes alter the metrics. To reproduce the published numbers, use the `v0.1` tag (the last commit of the v0.1 pipeline) and the README of that version.

The second demo scene was called `nuscenes-0088` before; it is nuScenes trainval index 88, which is **scene-0106**, and is now `nuscenes-0106`. `python -m aether migrate` renames an existing bundle, and the old id still loads.

## License

AETHER uses the repository license. Third-party code and data keep their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
