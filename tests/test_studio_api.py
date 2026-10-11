from fastapi.testclient import TestClient

from aether.studio.api import app


client = TestClient(app)


def test_catalog_and_scene_endpoints() -> None:
    response = client.get("/api/catalog")
    assert response.status_code == 200
    assert {scene["id"] for scene in response.json()["scenes"]} >= {
        "nuscenes-0383", "nuscenes-0088"
    }
    scene = client.get("/api/scenes/nuscenes-0383")
    assert scene.status_code == 200
    assert scene.json()["topology"]["lanes"]
    assert scene.json()["actors"]


def test_validate_experiment() -> None:
    payload = client.get("/api/scenes/nuscenes-0383").json()["experiment"]
    payload["id"] = "api-validation"
    response = client.post("/api/experiments/validate", json=payload)
    assert response.status_code == 200
    assert response.json()["valid"] is True
