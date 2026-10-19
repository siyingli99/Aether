"""Create scene bundles from DriveStudio outputs and processed datasets.

``python -m aether import`` wraps :func:`import_scene`;
``python -m aether migrate`` converts the pre-0.2 demo layout
(``data/checkpoints/...``, ``data/nuscenes/processed_10Hz/...``,
``data/limsim/...``) into bundles under ``data/scenes``.
"""
from __future__ import annotations

import os
from pathlib import Path
import shutil

import numpy as np
import yaml

from ..paths import AETHER_ROOT, REPO_ROOT, SCENES_ROOT
from .rig import load_rig

DATASET_DEFAULTS = {
    "nuscenes": {"vehicle": "renault_zoe", "anchor": "CAM_FRONT", "anchor_source": "camera"},
    "pandaset": {"vehicle": "chrysler_pacifica", "anchor": "EGO", "anchor_source": "ego_pose"},
    "nuplan": {"vehicle": "sedan", "anchor": "EGO", "anchor_source": "ego_pose"},
}
# Where the logged ego pose sits on the vehicle: PandaSet logs the box
# centre, nuPlan the rear axle (= AETHER's vehicle origin).
EGO_POSE_AT_CENTER = {"pandaset": True, "nuplan": False}


def ego_anchor_mount(frame, center_offset: float) -> np.ndarray:
    """vehicle_from_anchor for an ego-pose anchor (box centre, dataset axes)."""
    mount = np.eye(4)
    mount[:3, :3] = frame.road_from_scene
    mount[0, 3] = center_offset
    return mount


def _load_4x4(path: Path) -> np.ndarray:
    return np.loadtxt(str(path)).astype(np.float64).reshape(4, 4)


def _place(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "link":
        try:
            os.link(source, target)
            return
        except OSError:
            pass
        target.symlink_to(source.resolve())
    elif mode == "symlink":
        target.symlink_to(source.resolve())
    else:
        shutil.copy2(source, target)


def import_scene(*, scene_id: str, dataset: str, checkpoint: Path, processed: Path,
                 rig: Path, frame_dt: float, output_root: Path = SCENES_ROOT,
                 vehicle: str | None = None, hdmap: Path | None = None,
                 network: Path | None = None, routes: Path | None = None,
                 actors: Path | None = None, name: str | None = None,
                 source: dict | None = None, mode: str = "link",
                 rig_mounts: bool = False) -> Path:
    """Write ``<output_root>/<scene_id>`` and return its path.

    ``mode``: ``link`` (hard link, falling back to symlink), ``symlink`` or
    ``copy`` for the large files (checkpoint, network).

    ``rig_mounts``: place the cameras at the rig YAML mounts instead of the
    logged extrinsics (a *virtual rig*, e.g. nuScenes camera positions on a
    nuPlan scene so that a fixed-rig planner such as UniAD sees the layout it
    was trained on).  Only sensible where the reconstruction covers those
    viewpoints.
    """
    dataset = dataset.lower()
    defaults = DATASET_DEFAULTS[dataset]
    root = Path(output_root) / scene_id
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(checkpoint)
    config = checkpoint.parent / "config.yaml"
    _place(checkpoint, root / "recon" / "checkpoint.pth", mode)
    if config.is_file():
        shutil.copy2(config, root / "recon" / "config.yaml")
    ds_cfg = yaml.safe_load(config.read_text(encoding="utf-8")) if config.is_file() else {}
    start = int(((ds_cfg or {}).get("data") or {}).get("start_timestep", 0) or 0)

    camera_rig = load_rig(rig)
    camera_rig.save(root / "rig.yaml")
    names = camera_rig.camera_names
    extr = processed / "extrinsics"
    if defaults["anchor_source"] == "camera":
        index = camera_rig.cameras[defaults["anchor"]].index
        files = sorted(extr.glob(f"*_{index}.txt"))
    else:
        files = sorted((processed / "ego_pose").glob("*.txt"))
    files = [f for f in files if int(f.stem.split("_")[0]) >= start]
    if len(files) < 2:
        raise FileNotFoundError(f"need at least two logged poses under {processed}")
    track = np.stack([_load_4x4(f) for f in files])
    inv0 = np.linalg.inv(track[0])
    anchor = np.einsum("ij,tjk->tik", inv0, track)

    from ..geometry.frames import dataset_frame
    from ..vehicle.profile import load_profile
    frame = dataset_frame(dataset)
    vehicle_name = vehicle or defaults["vehicle"]
    anchor_mount = None
    if defaults["anchor"] == "EGO":
        offset = load_profile(vehicle_name).center_offset if EGO_POSE_AT_CENTER.get(dataset, True) else 0.0
        anchor_mount = ego_anchor_mount(frame, offset)

    def camera_t0(name: str) -> np.ndarray:
        index = camera_rig.cameras[name].index
        if rig_mounts:
            # scene_from_camera at t0 = scene_from_anchor0 (= I) @ anchor_from_vehicle @ mount
            if anchor_mount is None:
                raise ValueError("rig_mounts needs an ego-pose anchored dataset")
            return np.linalg.inv(anchor_mount) @ camera_rig.cameras[name].mount
        per_frame = extr / f"{start:03d}_{index}.txt"
        if per_frame.is_file():
            return inv0 @ _load_4x4(per_frame)
        static = extr / f"{index}.txt"       # nuPlan: camera-to-ego, constant
        if static.is_file():
            return _load_4x4(static)
        raise FileNotFoundError(f"no extrinsics for camera {name} (index {index}) under {extr}")

    cameras_t0 = np.stack([camera_t0(n) for n in names])
    np.savez_compressed(root / "log.npz", anchor=anchor.astype(np.float64),
                        cameras_t0=cameras_t0.astype(np.float64),
                        camera_names=np.array(names))

    if hdmap is not None:
        (root / "map").mkdir(exist_ok=True)
        shutil.copy2(hdmap, root / "map" / "hdmap.json")
    if network is not None and routes is not None:
        (root / "map").mkdir(exist_ok=True)
        _place(Path(network), root / "map" / "network.net.xml", "copy")
        shutil.copy2(routes, root / "map" / "routes.rou.xml")
    if actors is not None:
        shutil.copy2(actors, root / "actors.npz")
    else:
        from .actors import extract_actors
        extracted = extract_actors(checkpoint)
        if extracted:
            np.savez_compressed(root / "actors.npz", **extracted)

    manifest = {
        "id": scene_id, "name": name or scene_id, "dataset": dataset,
        "frame_dt": float(frame_dt), "vehicle": vehicle_name,
        "anchor": defaults["anchor"], "frames": int(len(anchor)),
        "source": source or {},
        "processed": str(Path(processed).resolve()),
    }
    if anchor_mount is not None:
        # Ego-pose anchored datasets: the logged ego pose carries the
        # dataset's own body axes (PandaSet: x right, y forward, z up; nuPlan:
        # FLU), which at frame 0 are the scene axes.
        manifest["anchor_mount"] = anchor_mount.tolist()
        manifest["anchor_axes"] = "dataset"
    if rig_mounts:
        manifest["camera_mounts"] = "rig"
    (root / "scene.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
# Pre-0.2 demo layout
# --------------------------------------------------------------------------- #
LEGACY_DEMO = [
    {
        "scene_id": "nuscenes-0383", "dataset": "nuscenes", "frame_dt": 0.1,
        "name": "nuScenes 0383 · Singapore One North",
        "checkpoint": "data/checkpoints/nuscenes_outputs/scene_recon_full/scene_300_6cams_with_humans/checkpoint_final.pth",
        "processed": "data/nuscenes/processed_10Hz/trainval/300",
        "rig": "configs/cameras/nuscenes.yaml",
        "map_dir": "data/limsim/nuscenes_0383", "prefix": "scene0383",
        "actors": "assets/demo_actors/nuscenes-0383.npz",
        "source": {"dataset": "nuScenes", "scene": "scene-0383", "index": 300,
                   "location": "singapore-onenorth"},
    },
    {
        "scene_id": "nuscenes-0106", "dataset": "nuscenes", "frame_dt": 0.5,
        "name": "nuScenes 0106 · Boston Seaport",
        "checkpoint": "data/checkpoints/nuscenes_outputs/scene_recon_full/scene_88_6cams_with_humans/checkpoint_final.pth",
        "processed": "data/nuscenes/processed_10Hz/trainval/088",
        "rig": "configs/cameras/nuscenes.yaml",
        "map_dir": "data/limsim/nuscenes_088", "prefix": "scene088",
        "actors": "assets/demo_actors/nuscenes-0106.npz",
        "source": {"dataset": "nuScenes", "scene": "scene-0106", "index": 88,
                   "location": "boston-seaport", "note": "keyframes only (2 Hz)"},
    },
    {
        "scene_id": "pandaset-095", "dataset": "pandaset", "frame_dt": 0.1,
        "name": "PandaSet 095",
        "checkpoint": "data/checkpoints/pandaset_outputs/scene_recon_full/scene_095_6cams_with_humans/checkpoint_final.pth",
        "processed": "data/pandaset/processed/095",
        "rig": "configs/cameras/pandaset_800x450.yaml",
        "source": {"dataset": "PandaSet", "scene": "095"},
    },
]


def migrate_legacy_demo(root: Path = AETHER_ROOT, output_root: Path = SCENES_ROOT,
                        mode: str = "link") -> list[Path]:
    """Build bundles for every legacy demo scene whose files are present."""
    from .bundle import RENAMED
    for old, new in RENAMED.items():               # rename bundles made before the fix
        if (output_root / old).is_dir() and not (output_root / new).exists():
            (output_root / old).rename(output_root / new)
            manifest_path = output_root / new / "scene.yaml"
            manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
            manifest["id"] = new
            manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    created = []
    for spec in LEGACY_DEMO:
        checkpoint = root / spec["checkpoint"]
        processed = root / spec["processed"]
        if not checkpoint.is_file() or not processed.is_dir():
            continue
        map_dir = root / spec["map_dir"] if spec.get("map_dir") else None
        prefix = spec.get("prefix", "")
        def pick(suffix: str) -> Path | None:
            if map_dir is None:
                return None
            path = map_dir / f"{prefix}{suffix}"
            return path if path.is_file() else None
        actors = REPO_ROOT / spec["actors"] if spec.get("actors") else None
        # Keep an HD map that ``map-upgrade`` already enriched with polygons.
        upgraded = (output_root / spec["scene_id"] / "map" / "hdmap.v1.json").is_file()
        created.append(import_scene(
            scene_id=spec["scene_id"], dataset=spec["dataset"], checkpoint=checkpoint,
            processed=processed, rig=REPO_ROOT / spec["rig"], frame_dt=spec["frame_dt"],
            output_root=output_root, hdmap=None if upgraded else pick(".map.json"),
            network=pick(".net.xml"),
            routes=pick(".rou.xml"), actors=actors if actors and actors.is_file() else None,
            name=spec["name"], source=spec["source"], mode=mode,
        ))
    return created
