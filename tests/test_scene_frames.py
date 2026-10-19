"""Scene-frame conventions: the modelled ego heading must match its motion."""
import math

import numpy as np
import pytest
import yaml

from aether.geometry.frames import dataset_frame
from aether.scene.bundle import SceneBundle
from aether.scene.geometry import logged_track
from aether.scene.importer import ego_anchor_mount


def _ego_bundle(root, dataset, motion_axis, legacy_mount):
    scene = root / f"{dataset}-demo"
    scene.mkdir(parents=True)
    anchor = np.tile(np.eye(4), (10, 1, 1))
    anchor[:, motion_axis, 3] = np.arange(10) * 1.0          # 10 m/s at 10 Hz
    np.savez(scene / "log.npz", anchor=anchor, cameras_t0=anchor[:1],
             camera_names=np.array(["CAM_FRONT"]))
    (scene / "rig.yaml").write_text(yaml.safe_dump({"cameras": {"CAM_FRONT": {
        "index": 0, "size": [800, 450], "K": [600, 600, 400, 225], "mount": np.eye(4).tolist()}}}))
    manifest = {"id": f"{dataset}-demo", "dataset": dataset, "frame_dt": 0.1,
                "vehicle": "sedan", "anchor": "EGO", "frames": 10}
    if legacy_mount:                                  # pre-fix bundles: identity rotation
        mount = np.eye(4)
        mount[0, 3] = 1.35
        manifest["anchor_mount"] = mount.tolist()
    else:
        manifest["anchor_mount"] = ego_anchor_mount(dataset_frame(dataset), 1.35).tolist()
        manifest["anchor_axes"] = "dataset"
    (scene / "scene.yaml").write_text(yaml.safe_dump(manifest))
    return SceneBundle.load(scene)


@pytest.mark.parametrize("legacy_mount", [False, True])
@pytest.mark.parametrize("dataset,motion_axis", [("pandaset", 1), ("nuplan", 0)])
def test_ego_heading_matches_motion(tmp_path, dataset, motion_axis, legacy_mount):
    track = logged_track(_ego_bundle(tmp_path, dataset, motion_axis, legacy_mount))
    motion = math.atan2(track.y[-1] - track.y[0], track.x[-1] - track.x[0])
    assert abs(motion) < 1e-6                        # road x = initial heading
    assert np.allclose(track.yaw, 0.0, atol=1e-6)    # vehicle points where it moves
    # The rear axle is 1.35 m behind the logged (box-centre) ego pose.
    assert track.x[0] == pytest.approx(-1.35) and track.y[0] == pytest.approx(0.0)


def test_all_dataset_frames_are_right_handed():
    for name in ("nuscenes", "pandaset", "nuplan"):
        assert np.linalg.det(dataset_frame(name).scene_from_road) == pytest.approx(1.0)


def _nuplan_processed(root, frames=6):
    proc = root / "proc"
    (proc / "ego_pose").mkdir(parents=True)
    (proc / "extrinsics").mkdir()
    for t in range(frames):
        pose = np.eye(4)
        pose[:3, 3] = [100.0 + 2.0 * t, 50.0, 0.0]     # global, driving along +x
        np.savetxt(proc / "ego_pose" / f"{t:03d}.txt", pose)
    for index in range(8):                              # static camera-to-ego
        mount = np.eye(4)
        mount[:3, 3] = [1.5, 0.1 * index, 1.6]
        np.savetxt(proc / "extrinsics" / f"{index}.txt", mount)
    return proc


@pytest.mark.parametrize("rig_mounts", [False, True])
def test_import_nuplan_layout(tmp_path, monkeypatch, rig_mounts):
    from aether.paths import REPO_ROOT
    from aether.scene import importer
    monkeypatch.setattr("aether.scene.actors.extract_actors", lambda ckpt: {})
    ckpt = tmp_path / "ckpt" / "checkpoint_final.pth"
    ckpt.parent.mkdir()
    ckpt.write_bytes(b"x")
    path = importer.import_scene(
        scene_id="nuplan-demo", dataset="nuplan", checkpoint=ckpt,
        processed=_nuplan_processed(tmp_path), rig=REPO_ROOT / "configs/cameras/nuplan.yaml",
        frame_dt=0.1, output_root=tmp_path / "scenes", mode="copy", rig_mounts=rig_mounts)
    bundle = SceneBundle.load(path)
    track = logged_track(bundle)
    # nuPlan logs the rear axle: no centre offset, origin at the first pose.
    assert track.x[0] == pytest.approx(0.0) and track.x[-1] == pytest.approx(10.0)
    assert np.allclose(track.yaw, 0.0, atol=1e-9)
    cams = bundle.log.cameras_t0
    if rig_mounts:
        assert np.allclose(cams["CAM_BACK"], bundle.rig.cameras["CAM_BACK"].mount)
    else:                                           # physical sensor B0 = index 7
        assert cams["CAM_BACK"][1, 3] == pytest.approx(0.7)
