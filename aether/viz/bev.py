"""Per-step BEV rendering of ego + NPCs (matplotlib, mpl >= 3.10 safe).

Enhancements (Stage A4):
* Per-actor ID label (uses ``info["obj_ids"]`` if present, else array index).
* Per-actor velocity arrow (``info["obj_velos"]`` -> [vx, vy]).
* Ego velocity arrow scaled by speed.
* Optional ``scene`` (aether.sim.Scene) override + ``lane_polylines`` overlay.
"""
from __future__ import annotations
import math
from typing import Optional, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


def _box_corners(cx, cy, w, l, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    dx = np.array([ l/2,  l/2, -l/2, -l/2])
    dy = np.array([ w/2, -w/2, -w/2,  w/2])
    xs = cx + dx * c - dy * s
    ys = cy + dx * s + dy * c
    return np.stack([xs, ys], axis=1)


def _arrow(ax, x, y, vx, vy, color="black", scale=1.0):
    speed = math.hypot(vx, vy)
    if speed < 0.1:
        return
    ax.arrow(x, y, vx * scale, vy * scale,
             head_width=0.6, head_length=0.8, fc=color, ec=color,
             length_includes_head=True, linewidth=0.8, alpha=0.9)


def render_bev(info: dict,
               view_radius: float = 30.0,
               scene: Optional[object] = None,
               lane_polylines: Optional[Sequence[np.ndarray]] = None,
               show_ids: bool = True,
               show_velo: bool = True) -> np.ndarray:
    ex, ey, _, ew, el, _, eyaw = info["ego_box"]
    obj_boxes = list(info.get("obj_boxes", []))
    obj_ids = list(info.get("obj_ids", []))
    obj_velos = list(info.get("obj_velos", []))

    # Allow Scene to override (e.g., for offline preview without a live env).
    if scene is not None:
        obj_boxes = []
        obj_ids = []
        obj_velos = []
        for a in getattr(scene, "actors", []):
            if not getattr(a, "enabled", True):
                continue
            st = a.state
            w, l, h = a.size
            obj_boxes.append([st.x, st.y, st.z_ds, w, l, h, st.yaw])
            obj_ids.append(a.actor_id)
            obj_velos.append([st.velo * math.cos(st.yaw),
                              st.velo * math.sin(st.yaw)])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

    # Lane overlay (background).
    if lane_polylines:
        for pl in lane_polylines:
            pl = np.asarray(pl)
            if pl.ndim == 2 and pl.shape[1] >= 2:
                ax.plot(pl[:, 0], pl[:, 1], color="#9e9e9e",
                        linewidth=0.7, alpha=0.4, zorder=0)

    # Other actors.
    for i, ob in enumerate(obj_boxes):
        ox, oy, _, ow, ol, _, oyaw = ob
        c = _box_corners(ox, oy, ow, ol, oyaw)
        ax.add_patch(MplPolygon(c, closed=True, facecolor="#2196f3",
                                edgecolor="black", alpha=0.8, zorder=2))
        if show_ids:
            label = str(obj_ids[i]) if i < len(obj_ids) else str(i)
            ax.text(ox, oy, label, color="white", fontsize=7,
                    ha="center", va="center", zorder=4,
                    bbox=dict(boxstyle="round,pad=0.1",
                              facecolor="#0d47a1", edgecolor="none",
                              alpha=0.85))
        if show_velo and i < len(obj_velos):
            vx, vy = obj_velos[i][:2]
            _arrow(ax, ox, oy, vx, vy, color="#1565c0", scale=1.0)

    # Ego.
    fc = "#f44336" if info.get("collision") else "#cddc39"
    ec = _box_corners(ex, ey, ew, el, eyaw)
    ax.add_patch(MplPolygon(ec, closed=True, facecolor=fc, edgecolor="black",
                            alpha=0.95, zorder=3))
    hx = ex + 0.5 * el * math.cos(eyaw)
    hy = ey + 0.5 * el * math.sin(eyaw)
    ax.plot([ex, hx], [ey, hy], "k-", linewidth=1.5, zorder=3)
    if show_velo:
        ev = float(info.get("ego_velo", 0.0))
        _arrow(ax, ex, ey, ev * math.cos(eyaw), ev * math.sin(eyaw),
               color="#b71c1c", scale=1.0)

    ax.set_xlim(ex - view_radius, ex + view_radius)
    ax.set_ylim(ey - view_radius, ey + view_radius)
    ax.set_aspect("equal")
    ax.grid(True, linestyle=":", alpha=0.3)
    ax.set_title(f"t={info.get('timestamp', 0):.2f}s "
                 f"v={info.get('ego_velo', 0):.1f}m/s "
                 f"rc={info.get('rc', 0):.2f} "
                 f"n={len(obj_boxes)}", fontsize=10)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3].copy()
    plt.close(fig)
    return img


def render_scene_preview(scene, ego_pose=None, view_radius: float = 30.0,
                         frame: int = 0,
                         lane_polylines: Optional[Sequence[np.ndarray]] = None
                         ) -> np.ndarray:
    """Convenience for offline scene editor / CLI preview without a live env."""
    if ego_pose is None:
        ep = scene.ego
        ex, ey, eyaw = float(ep.get("x", 0.0)), float(ep.get("y", 0.0)), float(ep.get("yaw", 0.0))
        ev = float(ep.get("velo", 0.0))
        ew, el, eh = ep.get("size", (1.85, 4.5, 1.6))
    else:
        ex, ey, eyaw, ev = ego_pose[:4]
        ew, el, eh = (1.85, 4.5, 1.6) if len(ego_pose) <= 4 else ego_pose[4:7]
    info = {
        "ego_box": [ex, ey, 0.0, ew, el, eh, eyaw],
        "ego_velo": ev,
        "timestamp": frame * 0.1,
        "rc": 0.0,
        "collision": False,
    }
    return render_bev(info, view_radius=view_radius, scene=scene,
                      lane_polylines=lane_polylines)
