"""Function: Extract actor trajectories from a DriveStudio checkpoint referenced by a scenario.
Input/Output: Reads a scenario YAML and checkpoint; writes a trajectory NPZ and updates the YAML unless disabled.
Example:
    # Prerequisite: the scenario YAML must reference a generated checkpoint.
    python tools/extract_traffic_from_ckpt.py \
        --scenario configs/scenarios/pandaset_095_aether.yaml --min_frames 2 \
        --classes RigidNodes,DeformableNodes,SMPLNodes
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from scipy.spatial.transform import Rotation as SCR


_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _resolve_checkpoint(scenario_path: Path, scenario: dict) -> Path:
    ckpt = scenario["checkpoint"]
    if "${aether:" in ckpt:
        # Mirror the resolver behaviour: ${aether:X} -> <repo_root>/X
        prefix, rest = ckpt.split("${aether:", 1)
        rel, tail = rest.split("}", 1)
        ckpt = prefix + str(_ROOT / rel.strip()) + tail
    return Path(ckpt).expanduser()


def _quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """q: (..., 4) WXYZ -> R: (..., 3, 3)"""
    q_xyzw = np.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], axis=-1)
    return SCR.from_quat(q_xyzw.reshape(-1, 4)).as_matrix().reshape(*q.shape[:-1], 3, 3)


def _extract_class(models: dict, cls: str
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (trans[T,N,3], rotm[T,N,3,3], size[N,3], fv[T,N])."""
    m = models[cls]
    trans = m["instances_trans"].detach().cpu().numpy().astype(np.float32)
    quats = m["instances_quats"].detach().cpu().numpy().astype(np.float32)
    # SMPLNodes stores (T, N, 1, 4); squeeze the extra body-joint dim.
    while quats.ndim > 3:
        squeeze_axis = next((i for i in range(2, quats.ndim - 1)
                             if quats.shape[i] == 1), None)
        if squeeze_axis is None:
            break
        quats = np.squeeze(quats, axis=squeeze_axis)
    sizes = m["instances_size"].detach().cpu().numpy().astype(np.float32)
    fv = m.get("instances_fv")
    if fv is None:
        fv = np.ones(trans.shape[:2], dtype=bool)
    else:
        fv = fv.detach().cpu().numpy().astype(bool)
    rotm = _quat_wxyz_to_rotmat(quats)
    return trans, rotm, sizes, fv


def _build_c2w_4x4(trans_tn3: np.ndarray, rotm_tn33: np.ndarray) -> np.ndarray:
    """trans (T,3), rotm (T,3,3) -> (T,4,4) float32."""
    T = trans_tn3.shape[0]
    M = np.zeros((T, 4, 4), dtype=np.float32)
    M[:, :3, :3] = rotm_tn33
    M[:, :3,  3] = trans_tn3
    M[:,  3,  3] = 1.0
    return M


def extract(ckpt_path: Path,
            classes: List[str],
            min_frames: int = 2
            ) -> Tuple[Dict[str, np.ndarray], List[dict]]:
    """Returns (npz_dict, actor_yaml_entries)."""
    print(f"[extract] loading {ckpt_path}")
    ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    models = ck["models"]

    npz: Dict[str, np.ndarray] = {}
    actors: List[dict] = []

    for cls in classes:
        if cls not in models:
            print(f"[extract] {cls}: not in checkpoint")
            continue
        trans, rotm, sizes, fv = _extract_class(models, cls)
        T_full, N = trans.shape[0], trans.shape[1]
        if sizes.ndim == 3:
            # Some sources store per-frame sizes; collapse to mean of valid frames.
            sizes = (sizes * fv[..., None]).sum(0) / np.maximum(fv.sum(0), 1)[:, None]
        n_kept = 0
        for inst in range(N):
            mask = fv[:, inst]
            if mask.sum() < min_frames:
                continue
            frames = np.flatnonzero(mask).astype(np.int32)
            c2w_T44 = _build_c2w_4x4(trans[:, inst], rotm[:, inst])  # (T_full,4,4)
            c2w_valid = c2w_T44[frames]                              # (Tv,4,4)
            # AETHER size order: (w, l, h)  ; DS stores (l, w, h).
            l, w, h = float(sizes[inst, 0]), float(sizes[inst, 1]), float(sizes[inst, 2])
            size_wlh = (max(w, 0.1), max(l, 0.1), max(h, 0.1))

            key = f"{cls}/{inst}"
            npz[f"c2w/{key}"]    = c2w_valid.astype(np.float32)
            npz[f"frames/{key}"] = frames
            npz[f"size/{key}"]   = np.asarray(size_wlh, dtype=np.float32)

            label = {"RigidNodes": "vehicle",
                     "SMPLNodes": "pedestrian",
                     "DeformableNodes": "deformable"}.get(cls, "vehicle")
            actors.append({
                "class": cls,
                "instance_id": int(inst),
                "size": [round(size_wlh[0], 3), round(size_wlh[1], 3), round(size_wlh[2], 3)],
                "label": label,
                "policy": {"type": "log_replay"},
            })
            n_kept += 1
        print(f"[extract] {cls}: {n_kept}/{N} instances kept (>= {min_frames} valid frames)")
    return npz, actors


def inject_yaml(scenario_path: Path, scenario: dict,
                traffic_relpath: str, actors: List[dict]) -> None:
    scenario["traffic_log"] = traffic_relpath
    scenario.setdefault("actor_backend", "scene")
    scenario["actors"] = actors

    # Preserve top-level key ordering as much as possible.
    out = yaml.safe_dump(scenario, sort_keys=False,
                         default_flow_style=False, allow_unicode=True)
    scenario_path.write_text(out, encoding="utf-8")
    print(f"[extract] wrote yaml: {scenario_path}  ({len(actors)} actors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True,
                    help="Scenario YAML to read checkpoint from and update.")
    ap.add_argument("--min_frames", type=int, default=2)
    ap.add_argument("--classes",
                    default="RigidNodes,DeformableNodes,SMPLNodes")
    ap.add_argument("--no_yaml", action="store_true",
                    help="Only write the .npz, do not edit the scenario yaml.")
    ap.add_argument("--traffic_dir", default="traffic",
                    help="Subdir (relative to scenario file) for .npz output.")
    args = ap.parse_args()

    scenario_path = Path(args.scenario).resolve()
    scenario = yaml.safe_load(scenario_path.read_text(encoding="utf-8")) or {}

    ckpt = _resolve_checkpoint(scenario_path, scenario)
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    npz, actors = extract(ckpt, classes, min_frames=args.min_frames)

    traffic_dir = scenario_path.parent / args.traffic_dir
    traffic_dir.mkdir(parents=True, exist_ok=True)
    out_npz = traffic_dir / (scenario_path.stem + ".npz")
    np.savez_compressed(out_npz, **npz)
    print(f"[extract] wrote npz: {out_npz}  ({len(npz)} arrays)")

    if not args.no_yaml:
        rel = os.path.relpath(out_npz, scenario_path.parent)
        inject_yaml(scenario_path, scenario, rel, actors)


if __name__ == "__main__":
    main()
