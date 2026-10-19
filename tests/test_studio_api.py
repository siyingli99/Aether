import json
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient


def _bundle(root: Path) -> None:
    import yaml
    scene = root / "demo-0001"
    (scene / "map").mkdir(parents=True)
    (scene / "scene.yaml").write_text(yaml.safe_dump({"id": "demo-0001", "name": "Demo", "dataset": "nuscenes",
                                                      "frame_dt": 0.1, "vehicle": "renault_zoe", "anchor": "CAM_FRONT",
                                                      "frames": 20}))
    anchor = np.tile(np.eye(4), (20, 1, 1))
    # nuScenes scene frame: forward = +z.
    anchor[:, 2, 3] = np.arange(20) * 0.5
    np.savez(scene / "log.npz", anchor=anchor, cameras_t0=anchor[:1], camera_names=np.array(["CAM_FRONT"]))
    (scene / "rig.yaml").write_text(yaml.safe_dump({"cameras": {"CAM_FRONT": {
        "index": 0, "size": [800, 450], "K": [600, 600, 400, 225],
        "mount": [[0, 0, 1, 1.7], [-1, 0, 0, 0], [0, -1, 0, 1.5], [0, 0, 0, 1]]}}}))
    lanes = [{"id": "L0", "centerline": [[x, 0.0] for x in range(-10, 40, 5)], "width": 3.5}]
    (scene / "map" / "hdmap.json").write_text(json.dumps({"schema": "aether.hdmap.v2", "lanes": lanes,
                                                          "local_to_global": np.eye(4).tolist()}))


def test_catalog_scene_and_validation(tmp_path, monkeypatch):
    _bundle(tmp_path / "scenes")
    import aether.scene.bundle as bundle_mod
    monkeypatch.setattr(bundle_mod, "SCENES_ROOT", tmp_path / "scenes")
    import aether.studio.api as api
    monkeypatch.setattr(api, "RUNS", tmp_path / "runs")
    client = TestClient(api.create_app())
    catalog = client.get("/api/catalog").json()
    assert [s["id"] for s in catalog["scenes"]] == ["demo-0001"]
    assert any(a["id"] == "uniad" for a in catalog["agents"])
    scene = client.get("/api/scenes/demo-0001").json()
    assert len(scene["lanes"]) == 1 and len(scene["route"]) == 20
    assert scene["route"][-1][0] > scene["route"][0][0]          # route runs along +x
    ok = client.post("/api/validate", json={"scene": "demo-0001"}).json()
    assert ok["valid"]
    bad = client.post("/api/validate", json={"scene": "demo-0001", "sim": {"dt": -1}}).json()
    assert not bad["valid"]
    assert client.get("/api/results").json() == []
