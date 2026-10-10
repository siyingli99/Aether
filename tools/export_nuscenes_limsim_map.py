"""Convert a nuScenes lane graph into a SUMO/LimSim network."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET

import numpy as np


_REPO = Path(__file__).resolve().parents[1]


def _write_xml(root: ET.Element, path: Path) -> None:
    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _load_4x4(path: Path) -> np.ndarray:
    return np.loadtxt(path).astype(np.float64).reshape(4, 4)


def _deduplicate(points: np.ndarray, minimum_spacing: float = 0.25) -> np.ndarray:
    if len(points) < 2:
        return points
    kept = [points[0]]
    for point in points[1:]:
        if float(np.linalg.norm(point[:2] - kept[-1][:2])) >= minimum_spacing:
            kept.append(point)
    if len(kept) == 1:
        kept.append(points[-1])
    return np.asarray(kept, dtype=np.float64)


def _to_local(points_global: np.ndarray, global_to_local: np.ndarray) -> np.ndarray:
    points = np.asarray(points_global, dtype=np.float64)
    if points.shape[1] == 2:
        points = np.column_stack((points, np.zeros(len(points))))
    homogeneous = np.column_stack((points[:, :3], np.ones(len(points))))
    return (global_to_local @ homogeneous.T).T[:, :3]


def extract_hd_map(dataset_root: Path, processed_scene: Path, location: str,
                   radius_m: float, resolution_m: float) -> dict:
    """Extract every lane and connector near the logged route from nuScenes."""
    from nuscenes.map_expansion.map_api import NuScenesMap

    anchor = _load_4x4(processed_scene / "extrinsics/000_0.txt")
    local_to_camera = np.eye(4, dtype=np.float64)
    local_to_camera[:3, :3] = np.asarray(
        [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
    )
    local_to_global = anchor @ local_to_camera
    global_to_local = np.linalg.inv(local_to_global)

    route_paths = sorted((processed_scene / "extrinsics").glob("*_0.txt"))
    route_global = np.stack([_load_4x4(path)[:3, 3] for path in route_paths])
    route_local = _deduplicate(_to_local(route_global, global_to_local))
    minimum = route_global[:, :2].min(axis=0) - float(radius_m)
    maximum = route_global[:, :2].max(axis=0) + float(radius_m)
    patch = (float(minimum[0]), float(minimum[1]),
             float(maximum[0]), float(maximum[1]))

    nusc_map = NuScenesMap(dataroot=str(dataset_root), map_name=location)
    layers = ("lane", "lane_connector")
    records = nusc_map.get_records_in_patch(patch, list(layers), mode="intersect")
    tokens = records["lane"] + records["lane_connector"]
    discretized = nusc_map.discretize_lanes(tokens, float(resolution_m))
    candidates = []
    for kind in layers:
        for token in records[kind]:
            centerline = discretized.get(token, [])
            if len(centerline) < 2:
                continue
            local = _deduplicate(_to_local(np.asarray(centerline), global_to_local))
            if len(local) < 2:
                continue
            delta = local[:, None, :2] - route_local[None, :, :2]
            nearest = float(np.sqrt(np.square(delta).sum(axis=-1)).min())
            if nearest > radius_m:
                continue
            connectivity = nusc_map.connectivity.get(token, {})
            candidates.append({
                "token": token, "kind": kind,
                "centerline": local[:, :2].round(4).tolist(),
                "incoming": list(connectivity.get("incoming", [])),
                "outgoing": list(connectivity.get("outgoing", [])),
                "lane_width_m": 3.5,
            })
    kept = {record["token"] for record in candidates}
    for record in candidates:
        record["incoming"] = [t for t in record["incoming"] if t in kept]
        record["outgoing"] = [t for t in record["outgoing"] if t in kept]
    return {
        "schema": "aether.nuscenes_hd_map.v1",
        "scene_index": processed_scene.name, "location": location,
        "units": "metres", "axes": {"x": "forward", "y": "left", "z": "up"},
        "crop_radius_m": float(radius_m), "resolution_m": float(resolution_m),
        "local_to_global": local_to_global.round(10).tolist(),
        "global_to_local": global_to_local.round(10).tolist(),
        "traffic_link": {
            "name": f"nuScenes-{processed_scene.name}-ego-corridor",
            "centerline": route_local[:, :2].round(4).tolist(),
        },
        "map_lanes": sorted(candidates, key=lambda item: item["token"]),
    }


class _UnionFind:
    def __init__(self, values) -> None:
        self.parent = {value: value for value in values}

    def find(self, value):
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            value, self.parent[value] = self.parent[value], root
        return root

    def union(self, left, right) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def _edge_id(token: str) -> str:
    return "nusc_" + str(token).replace("-", "_")


def _polyline_length(points) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1])
               for a, b in zip(points, points[1:]))


def _ego_route(lanes: list[dict], reference: list[list[float]]) -> list[str]:
    """Map-match the logged route onto the directed nuScenes lane graph."""
    by_token = {lane["token"]: lane for lane in lanes}
    ref = np.asarray(reference, dtype=np.float64)
    ref_heading = ref[min(3, len(ref) - 1)] - ref[0]
    ref_heading /= max(float(np.linalg.norm(ref_heading)), 1e-9)

    def start_score(lane: dict) -> float:
        points = np.asarray(lane["centerline"], dtype=np.float64)
        heading = points[min(3, len(points) - 1)] - points[0]
        heading /= max(float(np.linalg.norm(heading)), 1e-9)
        alignment = float(heading @ ref_heading)
        return float(np.linalg.norm(points[0] - ref[0])) + 10.0 * max(0.0, 0.8 - alignment)

    current = min(lanes, key=start_score)["token"]
    path, visited = [], set()
    while current not in visited:
        visited.add(current)
        path.append(current)
        outgoing = [token for token in by_token[current]["outgoing"]
                    if token in by_token and token not in visited]
        if not outgoing:
            break

        def continuation_score(token: str) -> float:
            points = np.asarray(by_token[token]["centerline"], dtype=np.float64)
            distance = np.sqrt(np.square(points[:, None, :] - ref[None, :, :]).sum(-1))
            return float(distance.min())

        current = min(outgoing, key=continuation_score)
        if sum(_polyline_length(by_token[t]["centerline"]) for t in path) > 190.0:
            break
    if len(path) < 3:
        raise RuntimeError(f"failed to map-match scene route; got {path}")
    return path


def export(payload: dict, output_dir: Path, netconvert: str,
           prefix: str | None = None) -> dict:
    lanes = list(payload["map_lanes"])
    by_token = {lane["token"]: lane for lane in lanes}
    if not lanes:
        raise ValueError("HD map contains no lanes")
    output_dir.mkdir(parents=True, exist_ok=True)
    scene_label = str(payload.get("scene_name", payload.get("scene_index", "unknown")))
    scene_label = scene_label.removeprefix("scene-")
    prefix = prefix or f"scene{scene_label}"
    if re.fullmatch(r"[A-Za-z0-9_-]+", prefix) is None:
        raise ValueError(f"unsafe output prefix: {prefix!r}")

    endpoint_keys = [(lane["token"], side) for lane in lanes for side in ("start", "end")]
    uf = _UnionFind(endpoint_keys)
    connections = []
    for lane in lanes:
        for target in lane["outgoing"]:
            if target in by_token:
                uf.union((lane["token"], "end"), (target, "start"))
                connections.append((lane["token"], target))

    groups: dict[tuple[str, str], list[tuple[str, str]]] = {}
    for key in endpoint_keys:
        groups.setdefault(uf.find(key), []).append(key)
    roots = sorted(groups, key=lambda item: (item[0], item[1]))
    node_ids = {root: f"node_{index:03d}" for index, root in enumerate(roots)}
    endpoint_node = {key: node_ids[uf.find(key)] for key in endpoint_keys}

    nodes = ET.Element("nodes")
    for root in roots:
        coords = []
        for token, side in groups[root]:
            points = by_token[token]["centerline"]
            coords.append(points[0] if side == "start" else points[-1])
        xy = np.asarray(coords, dtype=np.float64).mean(axis=0)
        ET.SubElement(nodes, "node", id=node_ids[root],
                      x=f"{xy[0]:.4f}", y=f"{xy[1]:.4f}", type="priority")
    node_path = output_dir / f"{prefix}.nod.xml"
    _write_xml(nodes, node_path)

    edges = ET.Element("edges")
    for lane in lanes:
        shape = " ".join(f"{float(x):.4f},{float(y):.4f}"
                         for x, y in lane["centerline"])
        ET.SubElement(edges, "edge", id=_edge_id(lane["token"]), **{
            "from": endpoint_node[(lane["token"], "start")],
            "to": endpoint_node[(lane["token"], "end")],
            "numLanes": "1", "speed": "13.9",
            "width": str(float(lane.get("lane_width_m", 3.5))),
            "priority": "3" if lane["kind"] == "lane" else "2", "shape": shape,
        })
    edge_path = output_dir / f"{prefix}.edg.xml"
    _write_xml(edges, edge_path)

    connection_root = ET.Element("connections")
    for source, target in sorted(set(connections)):
        ET.SubElement(connection_root, "connection", **{
            "from": _edge_id(source), "to": _edge_id(target),
            "fromLane": "0", "toLane": "0",
        })
    connection_path = output_dir / f"{prefix}.con.xml"
    _write_xml(connection_root, connection_path)

    net_path = output_dir / f"{prefix}.net.xml"
    subprocess.run([
        netconvert, "--node-files", str(node_path), "--edge-files", str(edge_path),
        "--connection-files", str(connection_path), "--output-file", str(net_path),
        "--no-turnarounds", "true", "--geometry.remove", "false",
        "--offset.disable-normalization", "true",
        "--junctions.corner-detail", "5", "--tls.discard-simple", "true",
    ], check=True)

    matched_route = _ego_route(lanes, payload["traffic_link"]["centerline"])
    route_lengths = [_polyline_length(by_token[token]["centerline"])
                     for token in matched_route]
    routes = ET.Element("routes")
    ET.SubElement(routes, "vType", id="aether_car", accel="2.6", decel="4.5",
                  sigma="0", length="4.5", width="1.85", minGap="2.5",
                  maxSpeed="13.9", carFollowModel="IDM")
    vehicle_specs = (("ego", 0.0, 2.0), ("bg-001", 12.0, 4.5),
                     ("bg-002", 27.0, 5.0), ("bg-003", 43.0, 5.5))
    for vehicle_id, route_offset, speed in vehicle_specs:
        cumulative, edge_index = 0.0, 0
        while (edge_index + 1 < len(route_lengths)
               and cumulative + route_lengths[edge_index] <= route_offset):
            cumulative += route_lengths[edge_index]
            edge_index += 1
        route_id = f"route_{vehicle_id}"
        ET.SubElement(routes, "route", id=route_id,
                      edges=" ".join(_edge_id(token) for token in matched_route[edge_index:]))
        ET.SubElement(routes, "vehicle", id=vehicle_id, type="aether_car",
                      depart="0", departLane="0",
                      departPos=f"{max(0.0, route_offset - cumulative):.3f}",
                      departSpeed=str(speed), route=route_id)
    route_path = output_dir / f"{prefix}.rou.xml"
    _write_xml(routes, route_path)

    config = ET.Element("configuration")
    inputs = ET.SubElement(config, "input")
    ET.SubElement(inputs, "net-file", value=net_path.name)
    ET.SubElement(inputs, "route-files", value=route_path.name)
    time_node = ET.SubElement(config, "time")
    ET.SubElement(time_node, "begin", value="0")
    ET.SubElement(time_node, "step-length", value="0.05")
    processing = ET.SubElement(config, "processing")
    ET.SubElement(processing, "collision.action", value="warn")
    _write_xml(config, output_dir / f"{prefix}.sumocfg")

    map_path = output_dir / f"{prefix}.map.json"
    map_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    manifest = {
        "schema": "aether.limsim_map.v2", "scene": scene_label,
        "location": payload.get("location"), "axes": payload.get("axes"),
        "lane_records": len(lanes), "lane_connections": len(set(connections)),
        "network_nodes": len(roots),
        "route_points": len(payload["traffic_link"]["centerline"]),
        "route_length_m": sum(route_lengths), "ego_route_tokens": matched_route,
        "net_file": net_path.name, "route_file": route_path.name, "map_file": map_path.name,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None,
                        help="optional pre-extracted scene map JSON")
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("/home/mikochi/Projects/Datasets/NuScenes"))
    parser.add_argument("--processed-scene", type=Path,
                        default=_REPO / "data/nuscenes/processed_10Hz/trainval/088")
    parser.add_argument("--location", default="boston-seaport")
    parser.add_argument("--scene-label", default=None,
                        help="official nuScenes label, e.g. 0383 (processed index may differ)")
    parser.add_argument("--prefix", default=None,
                        help="output filename prefix; defaults to scene<scene-label>")
    parser.add_argument("--radius-m", type=float, default=45.0)
    parser.add_argument("--resolution-m", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path,
                        default=_REPO / "data/limsim/nuscenes_088")
    parser.add_argument("--netconvert", default="netconvert")
    args = parser.parse_args()
    if args.source is not None:
        payload = json.loads(args.source.resolve().read_text(encoding="utf-8"))
    else:
        payload = extract_hd_map(
            args.dataset_root.resolve(), args.processed_scene.resolve(),
            args.location, args.radius_m, args.resolution_m,
        )
    if args.scene_label is not None:
        payload["scene_name"] = str(args.scene_label)
    manifest = export(
        payload, args.output_dir.resolve(), args.netconvert, prefix=args.prefix
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
