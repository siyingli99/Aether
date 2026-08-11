"""Function: Dump camera intrinsics and camera-to-ego extrinsics for point projection.
Input/Output: Reads the script's configured scene at one frame; writes ``cameras.npz`` under the output directory.
Example:
    python tools/dump_cameras.py --frame 10 --out outputs/pandaset095_cameras
"""
import sys, argparse
import numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from omegaconf import OmegaConf
from aether.config import load_cfg
from aether.envs.driving_env import AetherDrivingEnv

ap = argparse.ArgumentParser()
ap.add_argument("--frame", type=int, default=10)
ap.add_argument("--out", default=str(ROOT / "outputs/splatad_compare"))
a = ap.parse_args()

cfg = load_cfg(str(ROOT/"configs/scenarios/nuscenes_scene25_aether_static.yaml"),
               str(ROOT/"configs/base.yaml"),
               str(ROOT/"configs/cameras/nuscenes.yaml"),
               str(ROOT/"configs/kinematic.yaml"))
OmegaConf.set_struct(cfg, False)
cfg.scenario.lidar = {"enabled": True, "beams": 32, "az_samples": 1024,
                      "range_min": 1.5, "range_max": 80.0}
cfg.base = {"obs": {"cameras_only": False}}

env = AetherDrivingEnv(cfg, str(Path(a.out)/"_env"))
env.reset()
act = {"steer_rate": np.array([0.0]), "acc": np.array([0.0])}
steps = 0
while env.frame_idx < a.frame and steps < a.frame + 50:
    env.step(act); steps += 1
f = min(env.frame_idx, env._max_frame)

ego_c2w = np.asarray(env._ego_c2w_ds(), dtype=np.float64)
ego_inv = np.linalg.inv(ego_c2w)
names, Ks, c2e, Hs, Ws = [], [], [], [], []
for name in env._cam_order:
    c = env.cam_rig[name]
    cam_c2w = np.asarray(env._camera_c2w_ds(name), dtype=np.float64)
    names.append(name)
    Ks.append(np.asarray(c["K"], dtype=np.float64))
    c2e.append(ego_inv @ cam_c2w)        # camera-to-ego
    Hs.append(int(c["H"])); Ws.append(int(c["W"]))

np.savez(str(Path(a.out)/"cameras.npz"),
         names=np.array(names), Ks=np.stack(Ks), c2e=np.stack(c2e),
         H=np.array(Hs), W=np.array(Ws), frame_idx=np.int64(f))
print(f"[dump_cameras] frame {f}: dumped {len(names)} cameras -> cameras.npz")
print("  cams:", names)
