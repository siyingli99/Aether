"""Function: Edit AETHER scenario YAML files in a Tkinter BEV interface.
Input/Output: Reads an optional scenario YAML and writes user-approved edits back to that file.
Example:
    python aether/tools/scenario_editor.py configs/scenarios/pandaset_095_example.yaml
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as e:  # pragma: no cover
    raise SystemExit(f"Tkinter is required for the scenario editor: {e}")


CANVAS_SIZE = 720
DEFAULT_RADIUS = 50.0
DEFAULT_SIZE = (1.85, 4.5, 1.6)  # w, l, h
DEFAULT_POLICY = "logreplay"
POLICY_TYPES = ("logreplay", "idm", "manual")


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def load_scenario(path: str) -> Dict[str, Any]:
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    data.setdefault("actors", [])
    return data


def dump_scenario(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Coordinate transform: BEV(x_north, y_west) -> canvas(px, py)
# ---------------------------------------------------------------------------

class View:
    def __init__(self, canvas_size: int, radius: float):
        self.canvas_size = canvas_size
        self.radius = float(radius)

    @property
    def scale(self) -> float:
        return (self.canvas_size / 2.0) / self.radius

    def to_px(self, x: float, y: float) -> Tuple[float, float]:
        cx = self.canvas_size / 2.0
        cy = self.canvas_size / 2.0
        # +x -> up (-py), +y -> left (-px)
        px = cx - y * self.scale
        py = cy - x * self.scale
        return px, py

    def to_world(self, px: float, py: float) -> Tuple[float, float]:
        cx = self.canvas_size / 2.0
        cy = self.canvas_size / 2.0
        x = (cy - py) / self.scale
        y = (cx - px) / self.scale
        return x, y


# ---------------------------------------------------------------------------
# Editor app
# ---------------------------------------------------------------------------

class ScenarioEditor:
    def __init__(self, root: tk.Tk, scenario_path: Optional[str] = None):
        self.root = root
        self.root.title("AETHER scenario editor")
        self.path: Optional[str] = scenario_path
        self.data: Dict[str, Any] = {"scene_name": "untitled", "actors": []}
        self.view = View(CANVAS_SIZE, DEFAULT_RADIUS)
        self.selected: Optional[int] = None
        self._drag_origin: Optional[Tuple[float, float]] = None
        self._actor_drag_origin: Optional[Tuple[float, float]] = None

        self._build_ui()
        if scenario_path and os.path.exists(scenario_path):
            self._load(scenario_path)
        self._redraw()

    # ----- UI layout --------------------------------------------------------
    def _build_ui(self) -> None:
        bar = tk.Frame(self.root)
        bar.pack(side=tk.TOP, fill=tk.X)
        tk.Button(bar, text="Open", command=self._on_open).pack(side=tk.LEFT)
        tk.Button(bar, text="Save", command=self._on_save).pack(side=tk.LEFT)
        tk.Button(bar, text="Save As", command=self._on_save_as).pack(side=tk.LEFT)
        tk.Button(bar, text="Add actor", command=self._on_add_actor).pack(side=tk.LEFT)
        tk.Button(bar, text="Delete", command=self._on_delete_actor).pack(side=tk.LEFT)
        tk.Label(bar, text="  view radius (m):").pack(side=tk.LEFT)
        self.radius_var = tk.DoubleVar(value=DEFAULT_RADIUS)
        ent = tk.Entry(bar, textvariable=self.radius_var, width=6)
        ent.pack(side=tk.LEFT)
        ent.bind("<Return>", lambda *_: self._set_radius())
        self.path_label = tk.Label(bar, text="(no file)")
        self.path_label.pack(side=tk.RIGHT)

        body = tk.Frame(self.root)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(body, width=CANVAS_SIZE, height=CANVAS_SIZE,
                                bg="#1a1a1a", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT)
        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        side = tk.Frame(body, padx=8, pady=8)
        side.pack(side=tk.LEFT, fill=tk.Y)
        tk.Label(side, text="Actors").pack(anchor="w")
        self.listbox = tk.Listbox(side, width=28, height=14)
        self.listbox.pack()
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)

        form = tk.LabelFrame(side, text="properties", padx=4, pady=4)
        form.pack(fill=tk.X, pady=4)
        self.fields: Dict[str, tk.Variable] = {}
        for label, key, default in (
            ("instance_id", "instance_id", 1),
            ("class", "class", "RigidNodes"),
            ("x (m)", "x", 0.0),
            ("y (m)", "y", 0.0),
            ("yaw (rad)", "yaw", 0.0),
            ("velo (m/s)", "velo", 0.0),
            ("w", "w", DEFAULT_SIZE[0]),
            ("l", "l", DEFAULT_SIZE[1]),
            ("h", "h", DEFAULT_SIZE[2]),
        ):
            row = tk.Frame(form)
            row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=label, width=11, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value=str(default))
            ent = tk.Entry(row, textvariable=var, width=14)
            ent.pack(side=tk.LEFT)
            ent.bind("<Return>", lambda *_: self._apply_form())
            self.fields[key] = var

        row = tk.Frame(form)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="policy", width=11, anchor="w").pack(side=tk.LEFT)
        self.fields["policy"] = tk.StringVar(value=DEFAULT_POLICY)
        ttk.Combobox(row, textvariable=self.fields["policy"],
                     values=POLICY_TYPES, width=12).pack(side=tk.LEFT)

        tk.Button(form, text="Apply", command=self._apply_form).pack(pady=4)

        self.status = tk.Label(self.root, text="ready", anchor="w")
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    # ----- file IO ---------------------------------------------------------
    def _load(self, path: str) -> None:
        try:
            self.data = load_scenario(path)
        except Exception as e:
            messagebox.showerror("load failed", str(e))
            return
        self.path = path
        self.path_label.config(text=path)
        self.status.config(text=f"loaded {len(self.data.get('actors', []))} actors")
        self.selected = 0 if self.data["actors"] else None
        self._refresh_listbox()
        self._fill_form()
        self._redraw()

    def _on_open(self) -> None:
        p = filedialog.askopenfilename(
            title="Open scenario", filetypes=[("YAML", "*.yaml *.yml"), ("All", "*")])
        if p:
            self._load(p)

    def _on_save(self) -> None:
        if not self.path:
            self._on_save_as()
            return
        try:
            dump_scenario(self.path, self.data)
            self.status.config(text=f"saved {self.path}")
        except Exception as e:
            messagebox.showerror("save failed", str(e))

    def _on_save_as(self) -> None:
        p = filedialog.asksaveasfilename(
            title="Save scenario as", defaultextension=".yaml",
            filetypes=[("YAML", "*.yaml")])
        if p:
            self.path = p
            self.path_label.config(text=p)
            self._on_save()

    # ----- actor list ------------------------------------------------------
    def _refresh_listbox(self) -> None:
        self.listbox.delete(0, tk.END)
        for i, a in enumerate(self.data.get("actors", [])):
            st = a.get("state", {})
            pol = (a.get("policy") or {}).get("type", "logreplay")
            self.listbox.insert(
                tk.END,
                f"#{i} id={a.get('instance_id', '?')} {pol} "
                f"({st.get('x', 0):.1f},{st.get('y', 0):.1f})")
        if self.selected is not None and 0 <= self.selected < self.listbox.size():
            self.listbox.selection_set(self.selected)

    def _on_list_select(self, *_):
        sel = self.listbox.curselection()
        if not sel:
            return
        self.selected = int(sel[0])
        self._fill_form()
        self._redraw()

    def _on_add_actor(self) -> None:
        next_id = 1 + max((int(a.get("instance_id", 0))
                           for a in self.data.get("actors", [])), default=0)
        actor = {
            "class": "RigidNodes",
            "instance_id": next_id,
            "size": list(DEFAULT_SIZE),
            "state": {"x": 10.0, "y": 0.0, "yaw": 0.0, "velo": 0.0},
            "policy": {"type": DEFAULT_POLICY},
        }
        self.data.setdefault("actors", []).append(actor)
        self.selected = len(self.data["actors"]) - 1
        self._refresh_listbox()
        self._fill_form()
        self._redraw()

    def _on_delete_actor(self) -> None:
        if self.selected is None:
            return
        del self.data["actors"][self.selected]
        self.selected = max(0, self.selected - 1) if self.data["actors"] else None
        self._refresh_listbox()
        self._fill_form()
        self._redraw()

    # ----- form <-> data ---------------------------------------------------
    def _current(self) -> Optional[Dict[str, Any]]:
        if self.selected is None:
            return None
        try:
            return self.data["actors"][self.selected]
        except (KeyError, IndexError):
            return None

    def _fill_form(self) -> None:
        a = self._current()
        if a is None:
            for v in self.fields.values():
                v.set("")
            return
        st = a.get("state", {})
        sz = a.get("size", list(DEFAULT_SIZE))
        pol = (a.get("policy") or {}).get("type", DEFAULT_POLICY)
        self.fields["instance_id"].set(str(a.get("instance_id", 1)))
        self.fields["class"].set(str(a.get("class", "RigidNodes")))
        self.fields["x"].set(f"{st.get('x', 0):.3f}")
        self.fields["y"].set(f"{st.get('y', 0):.3f}")
        self.fields["yaw"].set(f"{st.get('yaw', 0):.3f}")
        self.fields["velo"].set(f"{st.get('velo', 0):.3f}")
        self.fields["w"].set(str(sz[0] if len(sz) > 0 else DEFAULT_SIZE[0]))
        self.fields["l"].set(str(sz[1] if len(sz) > 1 else DEFAULT_SIZE[1]))
        self.fields["h"].set(str(sz[2] if len(sz) > 2 else DEFAULT_SIZE[2]))
        self.fields["policy"].set(pol)

    def _apply_form(self) -> None:
        a = self._current()
        if a is None:
            return
        try:
            a["instance_id"] = int(self.fields["instance_id"].get())
            a["class"] = self.fields["class"].get() or "RigidNodes"
            a.setdefault("state", {})
            a["state"]["x"] = float(self.fields["x"].get())
            a["state"]["y"] = float(self.fields["y"].get())
            a["state"]["yaw"] = float(self.fields["yaw"].get())
            a["state"]["velo"] = float(self.fields["velo"].get())
            a["size"] = [float(self.fields["w"].get()),
                         float(self.fields["l"].get()),
                         float(self.fields["h"].get())]
            a["policy"] = {"type": self.fields["policy"].get() or DEFAULT_POLICY}
        except ValueError as e:
            messagebox.showerror("invalid value", str(e))
            return
        self._refresh_listbox()
        self._redraw()
        self.status.config(text=f"actor #{self.selected} updated")

    # ----- canvas ----------------------------------------------------------
    def _set_radius(self) -> None:
        try:
            self.view.radius = max(2.0, float(self.radius_var.get()))
        except ValueError:
            return
        self._redraw()

    def _redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        # grid (10 m spacing)
        step_m = 10.0
        steps = int(self.view.radius // step_m) + 1
        for s in range(-steps, steps + 1):
            wx = s * step_m
            x1, _ = self.view.to_px(0, wx)
            _, y1 = self.view.to_px(wx, 0)
            c.create_line(x1, 0, x1, CANVAS_SIZE, fill="#262626")
            c.create_line(0, y1, CANVAS_SIZE, y1, fill="#262626")
        # axes
        cx = cy = CANVAS_SIZE / 2.0
        c.create_line(0, cy, CANVAS_SIZE, cy, fill="#444444")
        c.create_line(cx, 0, cx, CANVAS_SIZE, fill="#444444")
        c.create_text(CANVAS_SIZE - 24, cy - 8, text="+y", fill="#888")
        c.create_text(cx + 12, 14, text="+x", fill="#888")
        # ego
        ego_w, ego_l = DEFAULT_SIZE[0], DEFAULT_SIZE[1]
        ego_size = self.data.get("ego_whl", DEFAULT_SIZE)
        if isinstance(ego_size, (list, tuple)) and len(ego_size) >= 2:
            ego_w, ego_l = float(ego_size[0]), float(ego_size[1])
            if len(ego_size) >= 3 and float(ego_size[2]) > 1.5:
                # legacy [w,h,l] layout
                ego_w, ego_l = float(ego_size[0]), float(ego_size[2])
        self._draw_box(0.0, 0.0, 0.0, ego_w, ego_l, color="#2dd4bf", label="ego")
        # actors
        for i, a in enumerate(self.data.get("actors", [])):
            st = a.get("state", {})
            sz = a.get("size", list(DEFAULT_SIZE))
            color = "#facc15" if i == self.selected else "#f97316"
            self._draw_box(
                float(st.get("x", 0)), float(st.get("y", 0)),
                float(st.get("yaw", 0)),
                float(sz[0] if len(sz) > 0 else DEFAULT_SIZE[0]),
                float(sz[1] if len(sz) > 1 else DEFAULT_SIZE[1]),
                color=color, label=f"#{i}",
                velo=float(st.get("velo", 0)),
            )

    def _draw_box(self, x: float, y: float, yaw: float, w: float, l: float,
                  color: str, label: str, velo: float = 0.0) -> None:
        cs, ss = math.cos(yaw), math.sin(yaw)
        dx = [l / 2, l / 2, -l / 2, -l / 2]
        dy = [w / 2, -w / 2, -w / 2, w / 2]
        pts: List[float] = []
        for ix, iy in zip(dx, dy):
            wx = x + ix * cs - iy * ss
            wy = y + ix * ss + iy * cs
            px, py = self.view.to_px(wx, wy)
            pts.extend([px, py])
        self.canvas.create_polygon(pts, outline=color, fill="", width=2)
        # heading marker
        hx = x + (l / 2) * cs
        hy = y + (l / 2) * ss
        cpx, cpy = self.view.to_px(x, y)
        hpx, hpy = self.view.to_px(hx, hy)
        self.canvas.create_line(cpx, cpy, hpx, hpy, fill=color, width=2)
        self.canvas.create_text(cpx + 12, cpy - 12, text=label, fill=color)
        if abs(velo) > 1e-2:
            scale = self.view.scale
            ax = x + velo * 1.0 * cs
            ay = y + velo * 1.0 * ss
            apx, apy = self.view.to_px(ax, ay)
            self.canvas.create_line(cpx, cpy, apx, apy, fill="#22c55e",
                                    arrow=tk.LAST, width=2)

    # ----- mouse interaction ----------------------------------------------
    def _on_left_click(self, event) -> None:
        # hit-test actors
        wx, wy = self.view.to_world(event.x, event.y)
        hit = self._actor_at(wx, wy)
        if hit is not None:
            self.selected = hit
            self._refresh_listbox()
            self._fill_form()
            self._actor_drag_origin = (wx, wy)
            self._redraw()
        else:
            self._actor_drag_origin = None

    def _on_drag(self, event) -> None:
        if self.selected is None or self._actor_drag_origin is None:
            return
        wx, wy = self.view.to_world(event.x, event.y)
        a = self._current()
        if a is None:
            return
        st = a.setdefault("state", {})
        st["x"] = float(wx)
        st["y"] = float(wy)
        self._fill_form()
        self._refresh_listbox()
        self._redraw()

    def _on_release(self, _event) -> None:
        self._actor_drag_origin = None

    def _actor_at(self, wx: float, wy: float) -> Optional[int]:
        best = None
        best_d = float("inf")
        for i, a in enumerate(self.data.get("actors", [])):
            st = a.get("state", {})
            d = math.hypot(wx - float(st.get("x", 0)), wy - float(st.get("y", 0)))
            r = max(float((a.get("size") or DEFAULT_SIZE)[0]),
                    float((a.get("size") or DEFAULT_SIZE)[1])) * 0.6
            if d <= r and d < best_d:
                best, best_d = i, d
        return best


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="AETHER scenario editor (Tk BEV)")
    ap.add_argument("scenario", nargs="?", help="scenario YAML to open")
    args = ap.parse_args(argv)
    root = tk.Tk()
    ScenarioEditor(root, args.scenario)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
