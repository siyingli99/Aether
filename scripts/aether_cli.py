#!/usr/bin/env python
"""Function: Manage AD adapters or preview an AETHER scenario.
Input/Output: Reads adapter or scenario arguments; prints metadata, launches a process, or writes a BEV PNG.
Example:
    python scripts/aether_cli.py scene preview configs/scenarios/pandaset_095_example.yaml \
        --frame 0 --output outputs/bev_preview.png
"""
from __future__ import annotations
import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)


def _cmd_adapter_list(_args):
    from aether.agents.registry import list_adapters, load_adapter
    for name in list_adapters():
        spec = load_adapter(name)
        print(f"{spec.name:8s}  {spec.display_name}")
        print(f"  entrypoint : {spec.entrypoint}")
        print(f"  conda_env  : {spec.conda_env}")
        print(f"  cameras    : {spec.cameras}")
        print(f"  needs_depth: {spec.needs_depth}")
        print()
    return 0


def _cmd_adapter_run(args):
    from aether.agents.registry import load_adapter, launch_adapter
    spec = load_adapter(args.name)
    out = os.path.abspath(args.output)
    os.makedirs(out, exist_ok=True)
    # Pre-create FIFOs so adapter can attach immediately when running solo.
    for p in ("obs_pipe", "plan_pipe"):
        fp = os.path.join(out, p)
        if not os.path.exists(fp):
            os.mkfifo(fp)
    proc = launch_adapter(spec, args.cuda, out)
    print(f"[aether-cli] adapter pid={proc.pid}, follow {out}/ad_output.txt")
    if args.wait:
        try:
            rc = proc.wait()
            print(f"[aether-cli] adapter exited rc={rc}")
            return rc
        except KeyboardInterrupt:
            proc.terminate()
            return 130
    return 0


def _cmd_scene_preview(args):
    import yaml
    import numpy as np
    from PIL import Image
    from aether.viz.bev import render_scene_preview
    from aether.sim import Scene, scene_from_legacy_actor_replay
    with open(args.yaml) as f:
        data = yaml.safe_load(f) or {}
    # Best-effort: try aether-native scene shape, else fall back to empty Scene.
    sc = Scene()
    if "actors" in data:
        sc = scene_from_legacy_actor_replay({"actors": data["actors"]})
    ego = data.get("ego", {})
    sc.update_ego(x=float(ego.get("x", 0.0)), y=float(ego.get("y", 0.0)),
                  yaw=float(ego.get("yaw", 0.0)), velo=float(ego.get("velo", 0.0)))
    img = render_scene_preview(sc, frame=args.frame, view_radius=args.radius)
    out = args.output
    Image.fromarray(img).save(out)
    print(f"[aether-cli] wrote {out} ({img.shape})")
    return 0


def main():
    p = argparse.ArgumentParser(prog="aether-cli")
    sub = p.add_subparsers(dest="cmd", required=True)

    ad = sub.add_parser("adapter", help="adapter registry & launcher")
    ad_sub = ad.add_subparsers(dest="sub", required=True)
    ad_list = ad_sub.add_parser("list")
    ad_list.set_defaults(func=_cmd_adapter_list)
    ad_run = ad_sub.add_parser("run")
    ad_run.add_argument("name")
    ad_run.add_argument("--output", "-o", required=True)
    ad_run.add_argument("--cuda", default="0")
    ad_run.add_argument("--wait", action="store_true")
    ad_run.set_defaults(func=_cmd_adapter_run)

    sc = sub.add_parser("scene", help="scenario tools")
    sc_sub = sc.add_subparsers(dest="sub", required=True)
    sc_prev = sc_sub.add_parser("preview")
    sc_prev.add_argument("yaml")
    sc_prev.add_argument("--frame", type=int, default=0)
    sc_prev.add_argument("--radius", type=float, default=30.0)
    sc_prev.add_argument("--output", "-o", default="bev_preview.png")
    sc_prev.set_defaults(func=_cmd_scene_preview)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
