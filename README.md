# AETHER

[中文说明](README_ZH.md)

AETHER connects editable 3D Gaussian scenes, LimSim traffic and end-to-end driving models in a camera closed loop.

## Open-source roadmap

- [x] DriveStudio scene reconstruction and novel-view rendering
- [x] Camera and Gaussian point-cloud simulation
- [x] Dynamic-object editing and 3DRealCar asset import
- [x] nuScenes HD-map export and LimSim traffic
- [x] Six-camera Difix refinement
- [x] UniAD closed-loop adapter
- [x] Browser-based scene and experiment editor
- [ ] Public evaluation protocol

## Quick Start

### Clone and install AETHER

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

### Download the demo scenes

```bash
huggingface-cli login
python scripts/download_demo.py \
  --repo https://huggingface.co/datasets/mikkochi/AETHER_Demo \
  --scene all
```

Validate the data:

```bash
python scripts/check_demo.py --scene 0383
python scripts/check_demo.py --scene 088
```

The package contains nuScenes `scene-0383` and `scene-0088`, their checkpoints, processed camera data, HD maps, LimSim networks and two vehicle assets. Vehicle assets are from [3DRealCar](https://xiaobiaodu.github.io/3drealcar/).

### Install LimSim

Clone the tested branch next to AETHER and create its environment:

```bash
git clone --branch aether-integration \
  https://github.com/siyingli99/LimSim.git ../LimSim
conda create -p .conda/limsim python=3.10 -y
conda run -p .conda/limsim \
  pip install -r ../LimSim/aether_bridge/requirements.txt
```

`scripts/run_closed_loop.sh` starts and stops LimSim automatically.

### Install UniAD

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

### Run

```bash
bash scripts/run_closed_loop.sh 0383 --no_io_trace
bash scripts/run_closed_loop.sh 088 --no_io_trace
```

Use `AETHER_CUDA=1` to select another GPU. Results are saved under `aether_results/nuscenes/<scene>/uniad/limsim_camera/`; `limsim_aether.mp4` is the synchronized LimSim/AETHER video.

| Environment variable | Default | Effect |
|---|---|---|
| `AETHER_AD_READY_TIMEOUT` | `1800` | seconds to wait for the adapter to load |
| `AETHER_AD_STEP_TIMEOUT` | `900` | seconds one planner call may take; a dead or hung adapter fails the run with its log tail |
| `AETHER_ILQR_MAX_SOLVE_TIME` | unset | wall-clock cap for iLQR; unset keeps control deterministic (`0.05` reproduces older runs) |
| `UNIAD_VIS` / `UNIAD_VIS_WORKERS` | `1` / `4` | per-step UniAD visualization and its worker count |

#### Results before October 2026

The October 2026 fixes change closed-loop metrics: the background-collision box now rests on the local ground (it previously floated at camera height, about 2.1–3.6 m above the road on `scene-0383`), the obstacle band is measured from the local rather than a scene-wide ground, reconstructed actors keep their heading in `obj_boxes`, recorded actors advance at the dataset frame rate (`data_frame_dt`) instead of one frame per 0.25 s step; isolated floaters and splats inside the logged ego boxes no longer count as obstacles, off-scene detection measures distance to any reconstructed splat (road included) rather than to obstacles, braking no longer reverses the ego and iLQR no longer stops on a wall-clock budget. Set `legacy_compat: true` in the scenario (plus `AETHER_ILQR_MAX_SOLVE_TIME=0.05`) to reproduce earlier numbers.

## AETHER Studio

```bash
python scripts/aether_studio.py --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). The editor loads the registered scenes and six-camera previews, then lets you:

1. select a scene and inspect its HD-map topology;
2. keep/remove reconstructed actors, replace vehicles with 3DRealCar assets, or add new vehicles on the map;
3. assign a recorded, open-loop, LimSim or registered Python controller and set the ego start, planner and Difix options;
4. validate, save and run the experiment while following its log in the browser.

Scene, vehicle, planner and controller entries are registered in `configs/studio/catalog.yaml`. Saved specifications and run artifacts are written to `studio_workspace/experiments/<experiment-id>/`; the generated scenario is immutable for that run.

To add a local traffic controller, implement the class contract below and register its import path as an `entrypoint` in the catalog. The Studio UI then exposes it in the controller list.

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

The public editor frame is `x=forward`, `y=left`, `z=up`, with counter-clockwise positive yaw. Conversion to the renderer, bicycle model and planner output frames is centralized in `aether/studio/coordinates.py`.

For frontend development only:

```bash
corepack pnpm --dir web install
corepack pnpm --dir web build
```

## Scenario configuration and editing

### Configuration path

| File | Edit here | Read by |
|---|---|---|
| `configs/scenarios/nuscenes_0383.yaml` | checkpoint, initial ego state, traffic, Difix and termination | `load_cfg()` and `AetherDrivingEnv` |
| `configs/cameras/nuscenes.yaml` | camera names, resolution, intrinsics and AD extrinsics | `load_camera_rig()` |
| `configs/kinematic.yaml` | step length, wheelbase, acceleration and steering limits | `BicycleEgo.__init__()` / `BicycleEgo.step()` |
| `data/limsim/nuscenes_0383/scene0383.rou.xml` | vehicle routes, spawn positions and speeds | LimSim/SUMO |
| `configs/adapters/uniad.yaml` | planner process and input requirements | `load_adapter()` / `launch_adapter()` |

`configs/cameras/nuscenes.yaml` controls render size through `H`, `W` and the intrinsic matrix. Runtime camera poses come from the processed DriveStudio scene in `AetherDrivingEnv._load_static_c2w_ds()`; `v2c` and `l2c` are passed to the end-to-end adapter in `info["cam_params"]`.

### Scenario YAML

```bash
cp configs/scenarios/nuscenes_0383.yaml configs/scenarios/my_scene.yaml
```

| Field | Definition | Implementation |
|---|---|---|
| `checkpoint` | DriveStudio checkpoint | `DriveStudioRenderer` in `AetherDrivingEnv.__init__()` |
| `start_ab` | `[lateral_right, forward]` in metres | `BicycleEgo.reset()` |
| `start_euler` | XYZ Euler degrees; yaw is index 1 and right-turn positive | `BicycleEgo.__init__()` |
| `start_velo`, `start_steer` | initial speed in m/s and steering angle in radians | `BicycleEgo.reset()` |
| `ego_whl` | ego `[width, height, length]` in metres | collision boxes and LimSim ego state |
| `traffic.asset_path` | directory containing `asset.pt` or `gs.pth` + `wlh.json` | `GaussianVehicleAsset.load()` |
| `traffic.max_vehicles` | reusable Gaussian vehicle slots | `ExternalVehiclePool` |
| `traffic.max_gaussians` | splats retained per vehicle | `GaussianVehicleAsset._prune()` |
| `traffic.hide_original_dynamic` | hide reconstructed `RigidNodes`, `DeformableNodes` and `SMPLNodes` | `ExternalVehiclePool._hide_all_original_dynamic()` |
| `traffic.net_file` | road network for the synchronized 2D renderer | `LimSimTopDownRenderer` |
| `difix.required` | reject a frame if any camera is not refined | `scripts/closed_loop.py:run()` |
| `termination.max_steps` | maximum closed-loop steps | `AetherDrivingEnv.step()` |
| `termination.fg_collision_min_overlap` | minimum vehicle box-overlap ratio | `_fg_collision_swept()` |
| `termination.bg_threshold` | background splats inside the ego box before collision | `_bg_collision()` |
| `termination.off_scene_radius`, `off_scene_streak` | reconstructed-area departure tolerance | `_off_scene_check()` |
| `termination.ground_cell_size` | BEV cell (m) for the local ground under the obstacle band | `_height_above_local_ground()` |
| `data_frame_dt` | seconds between DriveStudio frames: `0.1` for 10 Hz preprocessing (scene-0383), `0.5` for keyframe-only nuScenes (scene-0088) | `AetherDrivingEnv.frame_idx` |
| `termination.bg_min_neighbors`, `bg_neighbor_radius` | obstacle splats need this many neighbours within the radius (drops isolated floaters) | `_filter_obstacles()` |
| `termination.carve_logged_route` | remove splats inside the logged ego boxes: the recorded car drove there | `_filter_obstacles()` |
| `legacy_compat` | `true` reproduces results produced before the October 2026 fixes (see below) | `AetherDrivingEnv` |

The supplied scenes require six successful Difix outputs before every planner call. Tune its inference cost with arguments defined in `scripts/closed_loop.py`:

```bash
bash scripts/run_closed_loop.sh 0383 \
  --refine_max_side 768 \
  --refine_timestep 199 \
  --refine_dtype bf16 \
  --no_io_trace
```

### Edit traffic and HD-map topology

Edit the `<vehicle>` entries in:

```text
data/limsim/nuscenes_0383/scene0383.rou.xml
data/limsim/nuscenes_088/scene088.rou.xml
```

`route` selects the edge sequence; `depart`, `departPos` and `departSpeed` control spawn time, longitudinal position and speed. Keep the number of exported background vehicles no larger than both `traffic.max_vehicles` and the `max_vehicles` argument of `LimSimRuntime.__init__()` in `../LimSim/aether_bridge/server.py`.

Build a network for another nuScenes scene through `extract_hd_map()` and `export()`:

```bash
python tools/export_nuscenes_limsim_map.py \
  --dataset-root /path/to/NuScenes \
  --processed-scene data/nuscenes/processed_10Hz/trainval/<index> \
  --location <nuscenes-map-name> \
  --scene-label <scene-id> \
  --prefix scene<scene-id> \
  --output-dir data/limsim/nuscenes_<scene-id>
```

`extract_hd_map()` converts the nuScenes lane/lane-connector graph to the local road frame. `export()` writes the SUMO nodes, edges, connections, routes and network. Add the new paths to the scenario YAML and extend the scene selector in `scripts/run_closed_loop.sh`.

### Edit vehicle Gaussians

```bash
python tools/build_vehicle_asset.py \
  --input /path/to/3DRealCar/<sequence> \
  --output data/assets/vehicles/<name> \
  --sequence-id <sequence> \
  --max-gaussians 50000
```

Set `traffic.asset_path` to the output directory. `AetherDrivingEnv._init_traffic_backend()` loads it, `ExternalVehiclePool` creates render slots, and `TrafficActorProvider.advance()` applies LimSim poses at each timestamp.

For instance-level editing, `ExternalVehiclePool.original_gaussian_counts()` lists reconstructed vehicle IDs, `set_original_visible(id, False)` deletes one vehicle, and `extract_original(id)` returns its Gaussian tensors. The demo configurations use `hide_original_dynamic: true` to remove all reconstructed dynamic objects.

### Reconstruct a scene and simulate point clouds

`scripts/train_scene.py` forwards OmegaConf overrides to DriveStudio. For nuScenes index `300` (`scene-0383`):

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

Train and inspect the point-cloud sidecar:

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

## Integrate another end-to-end planner

An adapter consists of a Python class, one launch script and one YAML declaration.

### 1. Implement the planner class

Create `third_party/<planner>/aether_adapter.py`:

```python
import numpy as np

from aether.agents.sdk import BaseAgent, serve

CAMERAS = [
    "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT",
    "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT",
]


class MyPlanner(BaseAgent):
    def setup(self):
        # Called once after CUDA_VISIBLE_DEVICES is set.
        self.model = load_model()

    def infer(self, obs, info):
        # Images are uint8 H x W x 3 and have already passed through Difix.
        images = [obs["rgb"][name] for name in CAMERAS]
        path_forward_left = self.model(images, info)  # N x 2: forward, left

        # AETHER output: lateral-right, forward, in metres.
        path_forward_left = np.asarray(path_forward_left, dtype=np.float32)
        return np.column_stack(
            (-path_forward_left[:, 1], path_forward_left[:, 0])
        )

    def teardown(self):
        pass


if __name__ == "__main__":
    serve(MyPlanner())
```

`BaseAgent.setup()` runs once, `infer(obs, info)` runs once per simulation step, and `teardown()` runs on exit. `serve()` handles FIFO timing, pickle transfer, trajectory validation and the final `Done` message.

The input contract is defined by `AetherDrivingEnv._get_obs()` and `_get_info()`:

| Value | Type and coordinates |
|---|---|
| `obs["rgb"][camera]` | `uint8[H,W,3]`; six named cameras |
| `obs["depth"]`, `obs["lidar"]` | optional; enable in scenario and adapter YAML |
| `info["ego_pos"]` | `[lateral_right, 0, forward]` metres in the simulation frame |
| `info["ego_rot"]` | XYZ Euler radians; yaw is index 1, right-turn positive |
| `info["ego_velo"]`, `info["ego_steer"]` | m/s and radians |
| `info["command"]` | `0=right`, `1=left`, `2=straight` |
| `info["cam_params"][camera]` | `intrinsic`, `v2c` and `l2c` calibration |
| planner return | non-empty `float N x 2`: `[lateral_right, forward]` waypoint positions in the current ego frame |

The LimSim traffic protocol uses a different canonical frame: `x=forward`, `y=left`, `z=up`, counter-clockwise yaw. Its conversion is implemented in `AetherDrivingEnv._traffic_ego_state()` and `FrameTransform`; do not apply that convention to planner output.

Trajectory-to-control conversion is in `scripts/closed_loop.py:_trajectory_states()` and `_traj_to_action()`. If a planner outputs `[forward, left]`, use the conversion above. If it outputs controls, convert them to waypoints inside `infer()`, or replace the `_traj_to_action()` call with a dedicated control interface.

### 2. Add the launcher

Create `third_party/<planner>/run_aether.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
planner_python="${PLANNER_PYTHON:-$repo_dir/.conda/<planner>/bin/python}"
export PYTHONPATH="$repo_dir:${PYTHONPATH:-}"
exec "$planner_python" -u "$repo_dir/third_party/<planner>/aether_adapter.py" "$1" "$2"
```

The launcher receives `<cuda_id> <output_dir>`. It must stay alive and print `waiting for obs`, `adapter ready` or another marker recognized by `wait_adapter_ready()`.

### 3. Register and run it

Create `configs/adapters/<planner>.yaml`:

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

`AdapterSpec.render_args()` substitutes the launcher arguments; `launch_adapter()` applies `cwd` and `env`. Run it with:

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

Use `aether_results/nuscenes/<scene>/<planner>/limsim_camera/io_trace/step_*/raw.pkl` to inspect the exact adapter input and output. Omit `--no_io_trace` while developing an adapter.

## Outputs

| File | Content |
|---|---|
| `limsim_aether.mp4` | synchronized LimSim 2D and AETHER six-camera video |
| `video.mp4` | AETHER six-camera video |
| `data.pkl`, `infos.pkl` | trajectory and per-step state |
| `io_trace/` | planner payload and result for each step |
| `refine_audit.json` | per-camera Difix record |
| `dynamic_removal_audit.json` | removed dynamic instances and external slots |
| `perf.json` | timing summary |

## License

AETHER uses the repository license. Third-party code and data retain their own terms; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
