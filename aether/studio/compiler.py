"""Compile a Studio experiment into immutable runtime inputs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from aether.paths import DRIVESTUDIO_ROOT

from .catalog import REPO_ROOT, StudioCatalog
from .coordinates import road_pose_to_bicycle
from .schemas import ExperimentSpec


class ExperimentCompiler:
    def __init__(self, catalog: StudioCatalog | None = None,
                 workspace: str | Path | None = None) -> None:
        self.catalog = catalog or StudioCatalog()
        self.workspace = Path(workspace or REPO_ROOT / "studio_workspace").resolve()

    def compile(self, spec: ExperimentSpec) -> dict[str, Any]:
        scene = self.catalog.scene(spec.scene_id)
        if not Path(scene["scenario"]).is_file():
            raise FileNotFoundError(scene["scenario"])
        if any(actor.controller.kind == "limsim" for actor in spec.actors) and not (
            Path(scene["net"]).is_file() and Path(scene["route"]).is_file()
        ):
            raise ValueError(f"scene {spec.scene_id} has no LimSim network")
        for actor in spec.actors:
            if actor.asset_id:
                asset = self.catalog.asset(actor.asset_id)
                if not Path(asset["path"]).is_dir():
                    raise FileNotFoundError(asset["path"])
            if actor.controller.kind == "plugin":
                controller = self.catalog.controller(str(actor.controller.plugin))
                if controller.get("kind") != "plugin" or not controller.get("entrypoint"):
                    raise ValueError(
                        f"controller {actor.controller.plugin} is not a registered plugin"
                    )

        experiment_dir = self.workspace / "experiments" / spec.id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        spec_path = experiment_dir / "experiment.json"
        spec_path.write_text(spec.model_dump_json(indent=2), encoding="utf-8")
        start_ab, start_euler = road_pose_to_bicycle(spec.ego.start)
        scenario_payload: dict[str, Any] = {
            "extends": scene["scenario"],
            "scene_name": f"studio_{spec.id}",
            "output_dir": str(experiment_dir / "runs" / "latest"),
            "start_ab": start_ab,
            "start_euler": start_euler,
            "start_velo": spec.ego.speed,
            "actors": [],
            "actor_backend": "studio",
            "studio": {
                "experiment_path": str(spec_path),
                "traffic_log": scene["traffic_log"],
                "assets": {
                    asset_id: self.catalog.asset(asset_id)["path"]
                    for asset_id in sorted({actor.asset_id for actor in spec.actors if actor.asset_id})
                },
                "plugins": {
                    plugin_id: self.catalog.controller(plugin_id)["entrypoint"]
                    for plugin_id in sorted({
                        str(actor.controller.plugin) for actor in spec.actors
                        if actor.controller.kind == "plugin"
                    })
                },
                "limsim": {
                    "enabled": any(actor.controller.kind == "limsim" for actor in spec.actors),
                    "host": "127.0.0.1", "port": 8766,
                    "net_file": scene.get("net"), "route_file": scene.get("route"),
                },
            },
            "traffic": {"enabled": False},
            "lidar": {"enabled": False},
            "difix": {"required": bool(spec.runtime.difix.enabled), "expected_cameras": 6},
            "termination": {"max_steps": spec.runtime.max_steps},
        }
        if not (DRIVESTUDIO_ROOT / "smpl_models" / "SMPL_NEUTRAL.pkl").is_file():
            scenario_payload["excluded_model_classes"] = ["SMPLNodes"]
        scenario_path = experiment_dir / "scenario.yaml"
        scenario_path.write_text(
            yaml.safe_dump(scenario_payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        manifest = {
            "schema": "aether.studio.manifest.v1", "experiment_id": spec.id,
            "experiment": str(spec_path), "scenario": str(scenario_path),
            "camera": scene["camera"], "adapter": spec.algorithm.adapter,
            "adapter_config": spec.algorithm.config_path,
            "ad_cuda": spec.algorithm.cuda,
            "difix": spec.runtime.difix.model_dump(),
            "seed": spec.runtime.seed,
            "limsim": scenario_payload["studio"]["limsim"],
            "output_dir": str(experiment_dir / "runs" / "latest"),
        }
        manifest_path = experiment_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return {**manifest, "manifest": str(manifest_path)}
