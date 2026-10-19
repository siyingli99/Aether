# AETHER

[中文说明](README_ZH.md)

AETHER lets you test end-to-end driving models (UniAD, VAD, LTF, a VLA model, or your own) in closed loop on real driving scenes. The scenes are reconstructed with 3D Gaussians, so the planner sees realistic camera (and LiDAR) images. Background traffic comes from a regular traffic simulator: LimSim/SUMO, a built-in IDM, or your own model.

Each run gives you a bird's-eye-view video and a trajectory table for every vehicle, so you can analyse it like any other traffic simulation.

```text
          ┌────────────── planner (UniAD / VAD / LTF / VLA / yours) ─────────────┐
 images ──┤                                                                       ├── waypoints
          └───────────────────────────────────────────────────────────────────────┘
   ▲                                                                                │ waypoint tracking
   │ render (Gaussians + traffic assets)          ego kinematics (bicycle model)   ▼
 3D scene ◄──── vehicle poses ◄──── traffic model (LimSim / IDM / yours) ◄──── ego state
   │
   └─► collisions · off-road · metrics · BEV video · trajectories.csv
```

- [Quick start](#quick-start)
- [Run a closed loop](#run-a-closed-loop)
- [Outputs](#outputs)
- [Plug in your planner](#plug-in-your-planner)
- [Plug in a traffic model](#plug-in-a-traffic-model)
- [Scenario files](#scenario-files)
- [Studio](#studio)
- [How the simulation works](#how-the-simulation-works)
- [Import your own scenes](#import-your-own-scenes)

## Quick start

```bash
git clone https://github.com/siyingli99/Aether.git && cd Aether
conda create -p .conda/aether python=3.10 -y && conda activate ./.conda/aether
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.3.0
pip install git+https://github.com/NVlabs/nvdiffrast.git
pip install -e third_party/drivestudio/third_party/smplx
export PYTHONNOUSERSITE=1          # stops a torch in ~/.local from sneaking in
```

Get the two demo scenes (nuScenes scene-0383 and scene-0106, with maps, SUMO networks and two [3DRealCar](https://xiaobiaodu.github.io/3drealcar/) vehicles):

```bash
huggingface-cli login
python scripts/download_demo.py --repo https://huggingface.co/datasets/mikkochi/AETHER_Demo
python -m aether doctor            # checks your setup
python -m aether scenes
```

Try a first run. No planner needed here, the car just replays its recorded route:

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent replay --no-refine
```

### Optional extras

| What | Needed for | How |
|---|---|---|
| LimSim | LimSim background traffic | `git clone --branch aether-integration https://github.com/siyingli99/LimSim.git ../LimSim`, then `conda create -n limsim python=3.10 && pip install -r ../LimSim/aether_bridge/requirements.txt`. AETHER starts and stops it for you. If it lives elsewhere, set `LIMSIM_ROOT` / `LIMSIM_PYTHON`. |
| UniAD | `--agent uniad` | see [UniAD](#uniad) |
| Difix3D | cleaning up rendered images (`refine.enabled: true`) | already in `third_party/difix3d`; weights download on first use |
| SUMO `netconvert` | building maps for new scenes | `pip install eclipse-sumo` |

## Run a closed loop

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent uniad
python -m aether run configs/scenarios/nuscenes-0106.yaml --agent vad --cuda 1
python -m aether run configs/scenarios/nuscenes-0383-idm.yaml --agent replay --steps 20 --no-refine
python -m aether run my.yaml --agent uniad --set traffic.max_vehicles=5 --set sim.dt=0.5
```

`--agent` is one of the adapters in `configs/agents/` (`uniad`, `vad`, `ltf`, `vla`, `gaussianfusion`), `replay`, or your own class as `module:Class` / `file.py:Class`. `python -m aether agents` lists what's available.

To run several scenes and planners in one go:

```bash
python -m aether bench configs/benchmarks/nuscenes-demo.yaml --agents uniad,vad
python -m aether eval outputs/runs/<run>        # recompute the metrics from the CSVs
```

## Outputs

Each run gets its own folder, `outputs/runs/<scene>-<agent>-<time>/`:

| File | What's in it |
|---|---|
| `trajectories.csv` | every vehicle at every step |
| `trajectories_10hz.csv` | same, resampled to 10 Hz |
| `trajectories_fcd.xml` | the 10 Hz data in SUMO FCD format; opens in `sumo-gui` and SUMO/LimSim tools |
| `events.csv` | collisions, going off-road, leaving the scene |
| `metrics.json` | scores (see [Metrics](#metrics)) |
| `videos/bev.mp4` | bird's-eye view, 20 fps |
| `videos/cameras.mp4` | the exact images the planner saw, one frame per step |
| `videos/overview.mp4` | the two side by side |
| `run.json`, `logs/steps.jsonl` | run settings; per-step control, plan and timing |
| `agent/` | the planner's own log and files (UniAD also saves its own visualisation, `agent/uniad/video.mp4`) |

The three videos run on the same clock: second *t* in each is simulation time *t*.

Trajectory columns (SI units):

| Column | Meaning |
|---|---|
| `time_s`, `step` | time and step |
| `vehicle_id`, `role`, `type`, `source` | `ego` or a background id, and where the vehicle came from |
| `x`, `y`, `z`, `yaw` | box centre in the road frame (x = the ego's starting heading, y left) |
| `speed`, `vx`, `vy`, `accel`, `yaw_rate` | motion |
| `length`, `width`, `height` | box size |
| `lane_id`, `s`, `d` | matched lane, distance along it, offset from its centre (left +) |
| `x_global`, `y_global`, `yaw_global` | same pose in the dataset's map coordinates |

## Plug in your planner

Your planner can run **in its own process and conda env** (best for UniAD, VAD and friends, whose dependencies don't get along), or **inside AETHER** (good for small models and rule-based planners).

### What goes in and what comes out

The planner is called every step (0.25 s by default) and gets:

| Input | Details |
|---|---|
| images | `uint8 [H, W, 3]` RGB per camera (Difix-cleaned if enabled) |
| intrinsics / extrinsics | `K` for the delivered size; camera → vehicle pose (vehicle = rear axle, x forward, y left, z up) |
| ego state | speed, acceleration, steering, yaw rate, pose |
| command | `0` right, `1` left, `2` straight |
| LiDAR (optional) | `float32 [N, 4]` points |

It returns **waypoints**: `N × 2` points, one every `dt` (default 0.5 s), in a frame you declare. It can also return a direct control (acceleration and steering rate) instead.

AETHER drives the car along those waypoints with a simple controller, using the real car's wheelbase and acceleration/steering limits, so a planner can't corner faster than a real car could. If you want the iLQR tracker from v0.1, set `sim.tracker: ilqr`.

| `frame` | Waypoints are relative to | Used by |
|---|---|---|
| `rear_axle` | rear axle on the ground | LTF, VLA, GaussianFusion |
| `center` | centre of the car | |
| `lidar` | the roof LiDAR | UniAD, VAD |

| `axes` | Column order |
|---|---|
| `flu` | (forward, left) |
| `rfu` | (right, forward), as in the original UniAD/VAD scripts |

**About image size:** the nuScenes demo scenes render at 800×450, half the original 1600×900. The UniAD and VAD adapters scale them back up the same way the models were evaluated, so please don't change the render size for them. For other planners, set `image_size` and AETHER resizes the images and fixes `K` for you.

### A. Planner in its own process

**1. A small script in your planner's repo.** It only needs NumPy. Put AETHER on `PYTHONPATH`, or just copy `aether/agents/sdk/bridge.py` next to it.

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

With `protocol: tuple`, `info` looks like this (the same layout the bundled adapters use):

| `info` key | Meaning |
|---|---|
| `ego_pos` | `[right, 0, forward]` rear-axle position (m) |
| `ego_rot` | `[0, yaw, 0]`, yaw clockwise-positive (rad) |
| `ego_velo`, `accelerate` | speed (m/s), acceleration (m/s²) |
| `ego_steer`, `steer_rate` | steering angle (rad) and rate (rad/s), right-positive |
| `command`, `timestamp` | navigation command, simulation time (s) |
| `cam_params[cam]` | `intrinsic` {fovx, fovy (rad), cx, cy, fx, fy, H, W}, `v2c` (vehicle → camera 4×4), `l2c` (LiDAR → camera 4×4) |
| `ego_box`, `obj_boxes` | ground-truth boxes `[x, y, z, w, l, h, yaw]`, for debugging only |
| `aether` | AETHER's own info: ego, all vehicles, events, route progress |

**2. A launcher** (`my_planner/aether_agent.sh`) that starts it with your env's Python:

```bash
#!/usr/bin/env bash
cd "$(dirname "$0")"
PY="${MY_PLANNER_PYTHON:-$HOME/miniconda3/envs/my_planner/bin/python}"
CUDA_VISIBLE_DEVICES="$1" "$PY" -u aether_agent.py "$1" "$2"
```

Use the full path to Python. `conda activate` often doesn't take effect when another Python process starts the script.

**3. Register it** in `configs/agents/my_planner.yaml`:

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

`${AETHER_EXTERNAL}` is the folder that contains AETHER (override it with the env var of the same name). `${workdir}` is `<run>/agent`, where `planner.log` ends up. Then:

```bash
python -m aether run configs/scenarios/nuscenes-0383.yaml --agent my_planner
```

If the planner crashes or hangs, the run stops and shows you the end of its log. Timeouts are set with `step_timeout_s` and `startup_timeout_s`.

### B. Planner inside AETHER

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

`obs` has `time`, `step`, `images`, `cameras`, `ego`, `command`, `lidar` and `info`.

### Bundled adapters

| Agent | Repo (next to AETHER) | Cameras | LiDAR | Plan |
|---|---|---|---|---|
| `uniad` | bundled: `third_party/uniad_sim` | 6 | – | lidar · rfu · 0.5 s |
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

A traffic model controls all the background cars. Every step, AETHER hands it the ego state, and it hands back where all its cars are now. AETHER takes care of rendering them, checking collisions and logging. If you've written a microscopic traffic model before, this should feel familiar.

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
  assets: [3drealcar_2024_06_04_15_34_51]   # 3D car models used for rendering (cycled)
  max_vehicles: 6                            # how many are rendered at once
  model: {class: my_traffic:MyCarFollowing, params: {n: 3, v0: 9.0}}
```

In Studio, picking "Python class" as background traffic and entering `my_traffic:MyCarFollowing` with its parameters does the same thing.

A few things to know:

- Positions are box centres in the road frame (metres; x = ego's starting heading, y left, yaw counter-clockwise), the same frame as `scene.hdmap`.
- Keep each car's `id` the same from step to step. Cars can appear and disappear whenever you like.
- AETHER puts each car on the reconstructed road surface for you, so you only deal with x, y and yaw.
- The map: `scene.hdmap.lanes[i]` has `id, centerline, width, polygon, incoming, outgoing`; `lane.project(x, y)` gives `(s, d, heading)`.

Built-in models (`traffic.model.type`):

| Type | What it does |
|---|---|
| `limsim` | LimSim (SUMO) on the scene's road network; starts automatically |
| `idm` | IDM car-following along map lanes, e.g. `{type: idm, count: 3, desired_speed: 10}` |
| `replay` | replays frames from a JSON file |
| `external` | any program that speaks the TCP protocol below |
| `class` | your own `TrafficModel` |

**Other simulators (C++, MATLAB, VISSIM, TESS NG…)** can connect over TCP with newline-delimited JSON. The protocol is in `aether/traffic/protocol.py` (standard library only, so feel free to copy it). Messages: `hello`, `reset`, `tick {target_time_s, ego}`, `shutdown`; the server answers with a frame `{time_s, vehicles: [{id, x, y, z, yaw, speed, length, width, height, type}]}`.

```yaml
traffic:
  model: {type: external, server: my_sim, host: 127.0.0.1, port: 9000}
```

**Scripting single cars**, e.g. for a cut-in, works without a full traffic model:

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

You can mix scripted cars with a traffic model. Scripted cars don't avoid anything, so place them so their paths don't cross. To move one of the cars from the original recording instead of a 3D model, add `render: {original: RigidNodes/3}`.

## Scenario files

```yaml
# configs/scenarios/nuscenes-0383.yaml
scene: nuscenes-0383                 # bundle in data/scenes/
sim: {dt: 0.25, max_steps: 39, seed: 0, tracker: pure_pursuit}   # or ilqr
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

`extends: other.yaml` inherits from another file. Typos in keys are reported as errors instead of being silently ignored.

Car profiles are in `configs/vehicles/`: `renault_zoe` (nuScenes), `chrysler_pacifica` (PandaSet) and a generic `sedan`. They set the car's size, wheelbase, limits and where the cameras sit.

## Studio

A web UI so you don't have to write YAML by hand:

```bash
python -m aether studio            # http://127.0.0.1:8765
```

- **Scenario**: pick a scene and a planner, then drop cars onto lanes and drag them around. You can also move the ego's start point, pick the traffic model, choose when a run should stop, and export everything as YAML.
- **Live**: watch the BEV, the camera images and the log while the run is going.
- **Results**: scores, events, the videos and all files to download for every past run.

Working on the frontend: `pnpm --dir web install && pnpm --dir web dev`; `pnpm --dir web build` rebuilds `web/dist`.

## How the simulation works

### Coordinates

Everything you see (CSVs, maps, traffic models) uses one **road frame**: x points where the ego was heading at t = 0, y left, z up. Inside a vehicle, the origin is the rear axle on the ground (x forward, y left). Each dataset's own conventions are converted on import, so you never have to deal with them.

### The ground

The reconstruction has no explicit road surface, so AETHER builds one per scene from the recorded drive and the reconstructed road around it (cached in `data/scenes/<id>/cache/ground.npz`). Kerbs, plants and trees are filtered out, and slopes are capped at 10 %. All cars sit on this surface, so they follow hills and don't float or sink.

### Datasets

| | nuScenes | PandaSet | nuPlan |
|---|---|---|---|
| ego car (`vehicle: auto`) | Renault Zoe | Chrysler Pacifica | sedan |
| cameras | the 6 native ones | 6, mapped to nuScenes names | 6, mapped to nuScenes names |
| HD map | lanes + drivable area | none (no off-road check) | lanes with outlines |
| typical start speed | city speeds | often 10–17 m/s | varies |

UniAD and VAD were trained on nuScenes, so on PandaSet and nuPlan they see unfamiliar cameras. Expect weaker results there. PandaSet scenes often start fast, and a nuScenes planner may brake hard.

### Collisions and violations

| Event | When it fires |
|---|---|
| `vehicle_collision` | the ego's box overlaps another car's box. The log also records who was at fault: not the ego if it was standing still or got rear-ended. |
| `static_collision` | the ego runs into something in the scene: a building, pole, kerb or parked car |
| `off_road` | a corner of the ego is more than 0.3 m outside the drivable area |
| `off_scene` | the ego drives somewhere the reconstruction doesn't cover, so the images stop making sense |

`termination.stop_on` decides which of these end the run. All of them are logged and scored either way.

### Metrics

| Metric | Meaning |
|---|---|
| NC | no collision the ego caused: 1; hit the scenery only: 0.5; caused a crash: 0 |
| DAC | stayed on the road: 1, else 0 |
| TTC | never came within 0.9 s of hitting the car ahead: 1, else 0 |
| C | comfort: acceleration, jerk and yaw stayed within normal driving limits |
| EP | how far the ego got compared with the recorded drive |
| RC | how much of the recorded route was completed |
| score | NC × DAC × (5·TTC + 2·C + 5·EP) / 12 |

These follow the NAVSIM/nuPlan PDM score. Use them to compare planners on AETHER, but not against NAVSIM leaderboard numbers.

## Import your own scenes

A scene is just a folder, `data/scenes/<id>/`:

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

# 2. turn it into a scene folder
python -m aether import nuscenes-0383 --dataset nuscenes \
  --checkpoint outputs/train/nuscenes-0383/recon/checkpoint_final.pth \
  --processed data/nuscenes/processed_10Hz/trainval/300

# 3. (nuScenes) HD map + SUMO network for LimSim
python -m aether map nuscenes-0383 --nuscenes /data/nuscenes --location singapore-onenorth
# (nuPlan) lanes from the nuPlan maps (needs nuplan-devkit)
python -m aether map nuplan-0038 --nuplan-maps /data/nuplan/maps --location us-nv-las-vegas-strip
```

To change how much LimSim traffic there is, edit `map/routes.rou.xml`.

New 3D car models: `python tools/build_vehicle_asset.py --input <3DRealCar sequence> --output data/assets/vehicles/<name>`.

The LiDAR tools are in `aether/train/`, `aether/refine/` and `aether/tools/`; each script's docstring explains how to use it.

## License

AETHER uses the repository license. Third-party code and data keep their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
