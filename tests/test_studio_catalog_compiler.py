from pathlib import Path

import yaml

from aether.studio.catalog import StudioCatalog
from aether.studio.compiler import ExperimentCompiler
from aether.studio.schemas import ExperimentSpec


def test_catalog_exposes_complete_demo_scenes() -> None:
    catalog = StudioCatalog()
    scenes = {scene["id"]: scene for scene in catalog.list_scenes()}
    assert scenes["nuscenes-0383"]["has_limsim"]
    assert scenes["nuscenes-0088"]["has_limsim"]
    assert catalog.topology("nuscenes-0383")["lanes"]
    assert any(actor["node_class"] == "RigidNodes" for actor in catalog.actors("nuscenes-0383"))


def test_compile_versioned_experiment(tmp_path: Path) -> None:
    catalog = StudioCatalog()
    payload = catalog.bootstrap_experiment("nuscenes-0383")
    payload["id"] = "compile-test"
    spec = ExperimentSpec.model_validate(payload)
    manifest = ExperimentCompiler(catalog, tmp_path).compile(spec)
    assert Path(manifest["scenario"]).is_file()
    assert Path(manifest["experiment"]).is_file()
    assert manifest["adapter"] == "uniad"


def test_compile_registered_controller_plugin(tmp_path: Path) -> None:
    catalog = StudioCatalog()
    payload = catalog.bootstrap_experiment("nuscenes-0383")
    payload["id"] = "plugin-test"
    actor = next(value for value in payload["actors"] if value["node_class"] == "RigidNodes")
    actor["operation"] = "edit"
    actor["start"] = {"x": 0, "y": 0, "z": 0, "yaw": 0}
    actor["controller"] = {
        "kind": "plugin", "plugin": "constant_velocity",
        "trajectory": [], "params": {"speed_mps": 4.0},
    }
    manifest = ExperimentCompiler(catalog, tmp_path).compile(
        ExperimentSpec.model_validate(payload)
    )
    scenario = yaml.safe_load(Path(manifest["scenario"]).read_text(encoding="utf-8"))
    assert scenario["studio"]["plugins"]["constant_velocity"] == (
        "aether.studio.controllers:ConstantVelocityController"
    )
