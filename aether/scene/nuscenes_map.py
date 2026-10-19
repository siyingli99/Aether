"""Upgrade a nuScenes bundle's HD map with true lane and drivable-area polygons.

The v1 map exported for LimSim only has lane centrelines with a nominal
3.5 m width, so the drivable area was approximated by buffering them – which
misses junction interiors, bus bays and uneven lane widths.  This reads the
nuScenes map expansion (``maps/expansion/<location>.json``) and writes
``map/hdmap.json`` in schema ``aether.hdmap.v2``:

* every lane / lane connector near the route with its real polygon,
* the ``drivable_area`` layer (with holes), clipped to the same window,
* the original v1 file kept as ``map/hdmap.v1.json``.

    python -m aether map-upgrade nuscenes-0383 --nuscenes ~/data/nuscenes
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from .bundle import SceneBundle


def _local(points, global_to_local: np.ndarray) -> list[list[float]]:
    pts = np.asarray(points, dtype=np.float64)[:, :2]
    out = pts @ global_to_local[:2, :2].T + global_to_local[:2, 3]
    return np.round(out, 3).tolist()


def upgrade_nuscenes_map(scene: str, nuscenes_root: str | Path, margin_m: float = 40.0) -> Path:
    from nuscenes.map_expansion.map_api import NuScenesMap
    from shapely.geometry import box

    bundle = SceneBundle.load(scene)
    path = bundle.hdmap_path
    if path is None:
        raise FileNotFoundError(f"{scene}: no map/hdmap.json to upgrade (export one for LimSim first)")
    data = json.loads(path.read_text())
    if data.get("schema") == "aether.hdmap.v2" and data.get("drivable_area"):
        backup = path.with_name("hdmap.v1.json")
        if backup.is_file():
            data = json.loads(backup.read_text())
    location = data.get("location") or bundle.source.get("location")
    if not location:
        raise ValueError(f"{scene}: map location unknown")
    l2g = np.asarray(data["local_to_global"], dtype=np.float64)
    g2l = np.linalg.inv(l2g)
    lanes_v1 = data.get("map_lanes", data.get("lanes", []))
    route = np.asarray(data.get("traffic_link", {}).get("centerline") or data.get("ego_route") or [[0, 0]])
    pts = np.vstack([np.asarray(l["centerline"])[:, :2] for l in lanes_v1] + [route[:, :2]])
    glob = pts @ l2g[:2, :2].T + l2g[:2, 3]
    lo, hi = glob.min(0) - margin_m, glob.max(0) + margin_m
    window = box(lo[0], lo[1], hi[0], hi[1])

    nmap = NuScenesMap(dataroot=str(Path(nuscenes_root).expanduser()), map_name=location)
    lanes = []
    for item in lanes_v1:
        token = item.get("token", item.get("id"))
        kind = item.get("kind", "lane")
        polygon = None
        try:
            record = nmap.get(kind, token)
            polygon = nmap.extract_polygon(record["polygon_token"])
        except Exception:                                   # noqa: BLE001 – keep the buffered fallback
            pass
        width = item.get("lane_width_m", item.get("width", 3.5))
        if polygon is not None and polygon.area > 1.0:
            width = float(np.clip(2 * polygon.area / max(polygon.length, 1e-6), 2.0, 6.0))
        lanes.append({"id": token, "kind": kind, "centerline": item["centerline"], "width": round(width, 2),
                      "polygon": _local(polygon.exterior.coords, g2l) if polygon is not None else None,
                      "incoming": item.get("incoming", []), "outgoing": item.get("outgoing", [])})

    drivable = []
    records = nmap.get_records_in_patch((lo[0], lo[1], hi[0], hi[1]), ["drivable_area"], mode="intersect")
    for token in records["drivable_area"]:
        for poly_token in nmap.get("drivable_area", token)["polygon_tokens"]:
            poly = nmap.extract_polygon(poly_token).intersection(window)
            for g in getattr(poly, "geoms", [poly]):
                if g.is_empty or g.geom_type != "Polygon" or g.area < 1.0:
                    continue
                drivable.append({"exterior": _local(g.exterior.coords, g2l),
                                 "holes": [_local(h.coords, g2l) for h in g.interiors]})

    out = {"schema": "aether.hdmap.v2", "location": location, "units": "metres",
           "axes": {"x": "forward (initial ego heading)", "y": "left"},
           "local_to_global": l2g.tolist(), "lanes": lanes, "drivable_area": drivable,
           "ego_route": route[:, :2].round(3).tolist(), "source": "nuScenes map expansion"}
    backup = path.with_name("hdmap.v1.json")
    if not backup.is_file():
        shutil.copy2(path, backup)
    path.write_text(json.dumps(out))
    return path
