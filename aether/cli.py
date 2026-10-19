"""``python -m aether <command>`` – the single entry point.

    run       closed-loop episode       python -m aether run configs/scenarios/nuscenes-0383.yaml --agent uniad
    bench     benchmark several runs    python -m aether bench configs/benchmarks/nuscenes-demo.yaml
    eval      re-score a run directory  python -m aether eval outputs/runs/<run>
    scenes    list scene bundles        python -m aether scenes
    agents    list planner adapters     python -m aether agents
    import    import a trained scene    python -m aether import <id> --dataset nuscenes --checkpoint ...
    migrate   convert the v0.1 demo layout into scene bundles
    map       HD map + SUMO network for a nuScenes / nuPlan scene (needs the dataset maps + SUMO)
    map-upgrade  add real lane / drivable polygons to an existing map
    doctor    check installation, scenes and planners
    studio    start the Studio web UI   python -m aether studio --port 8765
"""
from __future__ import annotations

import argparse
import json
import sys


def _set(data: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    for key in keys[:-1]:
        data = data.setdefault(key, {})
    data[keys[-1]] = value


def _overrides(pairs: list[str]) -> dict:
    import yaml
    out: dict = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        _set(out, key, yaml.safe_load(raw))
    return out


def cmd_run(args) -> None:
    from .config import load_scenario
    from .sim.runner import run_episode
    overrides = _overrides(args.set)
    if args.no_refine:
        _set(overrides, "refine.enabled", False)
    if args.steps:
        _set(overrides, "sim.max_steps", args.steps)
    scenario = load_scenario(args.scenario, overrides)
    seed_everything(scenario.sim.seed)
    run_episode(scenario, args.agent, output_root=args.output, cuda=args.cuda)


def seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def cmd_bench(args) -> None:
    from .eval.benchmark import run_benchmark
    run_benchmark(args.spec, cuda=args.cuda, output_root=args.output,
                  agents=args.agents.split(",") if args.agents else None)


def cmd_eval(args) -> None:
    from .eval.rescore import rescore
    print(json.dumps(rescore(args.run), indent=2))


def cmd_scenes(args) -> None:
    from .scene import SceneBundle, list_scenes
    for scene_id in list_scenes():
        try:
            b = SceneBundle.load(scene_id)
            print(f"{scene_id:24s} {b.dataset:9s} {b.frame_dt * 1000:4.0f} ms  vehicle={b.vehicle}"
                  f"  hdmap={'yes' if b.hdmap_path else 'no'}  network={'yes' if b.network else 'no'}")
        except Exception as error:
            print(f"{scene_id:24s} BROKEN: {error}")


def cmd_agents(args) -> None:
    from .agents.external import list_adapters, load_adapter
    print("built-in: replay (follow the logged route), constant, <package.module:Class>")
    for name in list_adapters():
        spec = load_adapter(name)
        cams = len(spec.cameras) if spec.cameras else "all"
        print(f"{name:16s} {spec.title:36s} cameras={cams} lidar={spec.lidar} "
              f"plan={spec.frame}/{spec.axes}@{spec.dt}s")


def cmd_import(args) -> None:
    from pathlib import Path
    from .paths import REPO_ROOT
    from .scene.importer import import_scene
    opt = lambda v: Path(v).expanduser().resolve() if v else None  # noqa: E731
    rig = opt(args.rig) or REPO_ROOT / "configs" / "cameras" / f"{args.dataset}.yaml"
    frame_dt = args.frame_dt or {"nuscenes": 0.1, "pandaset": 0.1, "nuplan": 0.1}[args.dataset]
    path = import_scene(scene_id=args.scene_id, dataset=args.dataset, checkpoint=opt(args.checkpoint),
                        processed=opt(args.processed), rig=rig, frame_dt=frame_dt,
                        vehicle=args.vehicle, hdmap=opt(args.hdmap), network=opt(args.network),
                        routes=opt(args.routes), actors=opt(args.actors), mode=args.mode,
                        rig_mounts=args.rig_mounts)
    print(path)


def cmd_migrate(args) -> None:
    from pathlib import Path
    from .scene.importer import migrate_legacy_demo
    for path in migrate_legacy_demo(Path(args.root).resolve(), mode=args.mode):
        print(path)


def cmd_map(args) -> None:
    if args.nuplan_maps:
        from .scene.nuplan_map import build_nuplan_map
        if not args.location:
            raise SystemExit("--location (nuPlan map name, e.g. us-nv-las-vegas-strip) is required")
        print(build_nuplan_map(args.scene, args.nuplan_maps, args.location, args.processed,
                               args.netconvert))
        return
    if not args.nuscenes:
        raise SystemExit("pass --nuscenes <root> (nuScenes) or --nuplan-maps <root> (nuPlan)")
    from .scene.sumo_export import build_scene_map
    print(build_scene_map(args.scene, args.nuscenes, args.processed, args.location, args.netconvert))


def cmd_map_upgrade(args) -> None:
    from .scene.nuscenes_map import upgrade_nuscenes_map
    for scene in args.scenes:
        print(upgrade_nuscenes_map(scene, args.nuscenes))


def cmd_doctor(args) -> None:
    from .doctor import main as doctor
    raise SystemExit(doctor())


def cmd_studio(args) -> None:
    import uvicorn
    from .studio.api import create_app
    uvicorn.run(create_app(), host=args.host, port=args.port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m aether", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run one closed-loop episode")
    p.add_argument("scenario")
    p.add_argument("--agent", help="adapter name (uniad, vad, ltf, vla, gaussianfusion), replay, "
                                   "or package.module:Class; default: scenario.agent.name")
    p.add_argument("--cuda", default=None)
    p.add_argument("--output", default=None, help="runs directory (default outputs/runs)")
    p.add_argument("--steps", type=int, default=None, help="override sim.max_steps")
    p.add_argument("--no-refine", action="store_true", help="skip Difix refinement")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a scenario field, e.g. --set traffic.max_vehicles=3")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("bench", help="run a benchmark")
    p.add_argument("spec")
    p.add_argument("--agents", default=None, help="comma-separated subset of agents")
    p.add_argument("--cuda", default="0")
    p.add_argument("--output", default=None)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("eval", help="recompute metrics.json of a run directory")
    p.add_argument("run")
    p.set_defaults(func=cmd_eval)

    sub.add_parser("scenes", help="list scene bundles").set_defaults(func=cmd_scenes)
    sub.add_parser("agents", help="list planners").set_defaults(func=cmd_agents)

    p = sub.add_parser("import", help="import a reconstructed scene as a bundle")
    p.add_argument("scene_id")
    p.add_argument("--dataset", required=True, choices=["nuscenes", "pandaset", "nuplan"])
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--processed", required=True, help="DriveStudio processed sequence directory")
    p.add_argument("--rig", default=None, help="camera rig YAML (default configs/cameras/<dataset>.yaml)")
    p.add_argument("--frame-dt", type=float, default=None, help="seconds between reconstruction frames")
    p.add_argument("--vehicle", default=None)
    p.add_argument("--hdmap", default=None)
    p.add_argument("--network", default=None)
    p.add_argument("--routes", default=None)
    p.add_argument("--actors", default=None)
    p.add_argument("--mode", choices=["link", "symlink", "copy"], default="link")
    p.add_argument("--rig-mounts", action="store_true",
                   help="virtual rig: cameras at the rig YAML mounts, not the logged extrinsics")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("migrate", help="build bundles from the v0.1 demo data layout")
    p.add_argument("--root", default=".")
    p.add_argument("--mode", choices=["link", "symlink", "copy"], default="link")
    p.set_defaults(func=cmd_migrate)

    p = sub.add_parser("map", help="build HD map + SUMO network of a nuScenes / nuPlan scene bundle")
    p.add_argument("scene")
    p.add_argument("--nuscenes", default=None, help="nuScenes root containing maps/expansion")
    p.add_argument("--nuplan-maps", default=None, help="nuPlan maps root (nuplan-maps-v1.0)")
    p.add_argument("--processed", default=None, help="DriveStudio processed scene (default: recorded at import)")
    p.add_argument("--location", default=None, help="map name (nuScenes default: from scene.yaml)")
    p.add_argument("--netconvert", default="netconvert", help="SUMO netconvert executable")
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("map-upgrade", help="add real lane / drivable-area polygons (nuScenes)")
    p.add_argument("scenes", nargs="+")
    p.add_argument("--nuscenes", required=True, help="nuScenes root containing maps/expansion")
    p.set_defaults(func=cmd_map_upgrade)

    sub.add_parser("doctor", help="check installation, data and planners").set_defaults(func=cmd_doctor)

    p = sub.add_parser("studio", help="start the Studio web UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    p.set_defaults(func=cmd_studio)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main(sys.argv[1:])
