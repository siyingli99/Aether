"""HD map + SUMO network for a nuPlan scene bundle.

``python -m aether map <scene> --nuplan-maps <root> --location us-nv-las-vegas-strip``
queries the nuPlan GeoPackage maps along the logged route and writes
``map/hdmap.json``, ``map/network.net.xml`` and ``map/routes.rou.xml`` into
the bundle, in the scene's road frame (origin at the first logged ego pose,
x forward, y left), the same frame the trajectories are recorded in.
Requires ``nuplan-devkit`` and SUMO ``netconvert``.
"""
from __future__ import annotations

from pathlib import Path
import shutil
import tempfile

import numpy as np

from .sumo_export import _deduplicate, _load_4x4, _to_local, export


def extract_hd_map(map_root: Path, processed_scene: Path, map_name: str,
                   radius_m: float = 45.0) -> dict:
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.common.maps.nuplan_map.map_factory import NuPlanMapFactory
    from nuplan.database.maps_db.gpkg_mapsdb import GPKGMapsDB

    pose_paths = sorted((processed_scene / "ego_pose").glob("*.txt"))
    if len(pose_paths) < 2:
        raise FileNotFoundError(f"nuPlan scene has no ego trajectory: {processed_scene}")
    anchor = _load_4x4(pose_paths[0])
    global_to_local = np.linalg.inv(anchor)
    route_global = np.stack([_load_4x4(path)[:3, 3] for path in pose_paths])
    route_local = _deduplicate(_to_local(route_global, global_to_local))

    maps_db = GPKGMapsDB(map_version="nuplan-maps-v1.0", map_root=str(map_root))
    nuplan_map = NuPlanMapFactory(maps_db).build_map_from_name(map_name)
    layers = (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR)
    objects = {}
    # Query along the whole route so long clips keep the lanes at their far end.
    for point in route_global[::5]:
        nearby = nuplan_map.get_proximal_map_objects(
            Point2D(float(point[0]), float(point[1])), float(radius_m), list(layers))
        for layer in layers:
            for obj in nearby[layer]:
                objects[str(obj.id)] = obj

    records = []
    for token, obj in objects.items():
        discrete = obj.baseline_path.discrete_path
        if len(discrete) < 2:
            continue
        points = np.asarray([[p.x, p.y, 0.0] for p in discrete], dtype=np.float64)
        local = _deduplicate(_to_local(points, global_to_local))
        if len(local) < 2:
            continue
        kind = "lane_connector" if type(obj).__name__.lower().endswith("connector") else "lane"
        polygon = None
        try:                                     # real lane outline (drivable-area check)
            ring = np.asarray(obj.polygon.exterior.coords, dtype=np.float64)
            ring = np.column_stack([ring[:, :2], np.zeros(len(ring))])
            polygon = _to_local(ring, global_to_local)[:, :2].round(4).tolist()
        except Exception:
            pass
        record = {
            "token": token, "kind": kind,
            "centerline": local[:, :2].round(4).tolist(),
            "incoming": [str(edge.id) for edge in obj.incoming_edges],
            "outgoing": [str(edge.id) for edge in obj.outgoing_edges],
            "lane_width_m": 3.5,
            "speed_limit_mps": float(obj.speed_limit_mps or 11.2),
        }
        if polygon and len(polygon) >= 4:
            record["polygon"] = polygon
        records.append(record)
    kept = {record["token"] for record in records}
    for record in records:
        record["incoming"] = [t for t in record["incoming"] if t in kept]
        record["outgoing"] = [t for t in record["outgoing"] if t in kept]
    return {
        "schema": "aether.hd_map.v1", "scene_index": processed_scene.name,
        "location": map_name, "units": "metres",
        "axes": {"x": "forward", "y": "left", "z": "up"},
        "crop_radius_m": float(radius_m),
        "local_to_global": anchor.round(10).tolist(),
        "global_to_local": global_to_local.round(10).tolist(),
        "traffic_link": {"name": f"nuPlan-{processed_scene.name}-ego-corridor",
                         "centerline": route_local[:, :2].round(4).tolist()},
        "map_lanes": sorted(records, key=lambda item: item["token"]),
    }


def build_nuplan_map(scene: str, map_root: str | Path, location: str,
                     processed: str | Path | None = None, netconvert: str = "netconvert",
                     radius_m: float = 45.0) -> Path:
    from .bundle import SceneBundle
    bundle = SceneBundle.load(scene)
    processed = Path(processed or bundle.manifest.get("processed") or "")
    if not (processed / "ego_pose").is_dir():
        raise FileNotFoundError("pass --processed <DriveStudio processed nuPlan scene dir>")
    payload = extract_hd_map(Path(map_root).expanduser(), processed, location, radius_m)
    payload["scene_name"] = bundle.id
    map_dir = bundle.root / "map"
    map_dir.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        export(payload, Path(tmp), netconvert, prefix="scene")
        shutil.copy2(Path(tmp) / "scene.net.xml", map_dir / "network.net.xml")
        shutil.copy2(Path(tmp) / "scene.rou.xml", map_dir / "routes.rou.xml")
        shutil.copy2(Path(tmp) / "scene.map.json", map_dir / "hdmap.json")
    return map_dir / "hdmap.json"
