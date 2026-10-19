"""``python -m aether doctor``: check the installation, data and planners."""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from pathlib import Path

from .paths import ASSETS_ROOT, DRIVESTUDIO_ROOT, EXTERNAL_ROOT, REPO_ROOT, SCENES_ROOT, SPLATAD_GSPLAT_ROOT


def _row(ok: bool | None, name: str, detail: str) -> bool:
    mark = {True: "ok  ", False: "FAIL", None: "--  "}[ok]
    print(f"  [{mark}] {name:34s} {detail}")
    return ok is not False


def main() -> int:
    good = True
    print("Python environment")
    good &= _row(sys.version_info >= (3, 10), "python >= 3.10", sys.version.split()[0])
    for pkg in ("numpy", "scipy", "shapely", "torch", "pydantic", "yaml", "PIL", "moviepy", "fastapi"):
        good &= _row(importlib.util.find_spec(pkg) is not None, pkg, "")
    user_site = [p for p in sys.path if "/.local/lib/python" in p]
    _row(None if not user_site else False if os.environ.get("PYTHONNOUSERSITE") != "1" else True,
         "user site-packages isolated", "set PYTHONNOUSERSITE=1 if ~/.local has another torch" if user_site else "")
    try:
        import torch
        cuda = torch.cuda.is_available()
        good &= _row(cuda, "CUDA", torch.cuda.get_device_name(0) if cuda else "no GPU visible")
    except Exception as error:                     # noqa: BLE001
        good &= _row(False, "torch import", str(error))
    ninja = shutil.which("ninja") or (Path(sys.executable).parent / "ninja")
    good &= _row(Path(ninja).exists(), "ninja (CUDA JIT)", str(ninja))
    good &= _row(shutil.which("ffmpeg") is not None or importlib.util.find_spec("imageio_ffmpeg") is not None,
                 "ffmpeg (videos)", "")

    print("\nVendored back-ends")
    good &= _row(DRIVESTUDIO_ROOT.is_dir(), "DriveStudio", str(DRIVESTUDIO_ROOT))
    _row(SPLATAD_GSPLAT_ROOT.is_dir() or None, "SplatAD gsplat (LiDAR)", str(SPLATAD_GSPLAT_ROOT))
    _row((REPO_ROOT / "third_party/difix3d").is_dir() or None, "Difix3D (refinement)", "")

    print("\nScenes and assets")
    from .scene.bundle import SceneBundle, list_scenes
    scenes = list_scenes()
    good &= _row(bool(scenes), "scene bundles", f"{len(scenes)} in {SCENES_ROOT}")
    for scene in scenes:
        try:
            b = SceneBundle.load(scene)
            ok = b.checkpoint.is_file()
            kind = "none"
            if b.hdmap_path:
                import json
                kind = "polygons" if json.loads(b.hdmap_path.read_text()).get("drivable_area") else "centrelines only"
            _row(ok, f"  {scene}", f"checkpoint={'ok' if ok else 'missing'} map={kind} "
                 f"network={'yes' if b.network else 'no'}")
        except Exception as error:                 # noqa: BLE001
            good &= _row(False, f"  {scene}", str(error))
    vehicles = sorted(p.name for p in (ASSETS_ROOT / "vehicles").glob("*")) if (ASSETS_ROOT / "vehicles").is_dir() else []
    _row(bool(vehicles) or None, "vehicle assets", ", ".join(vehicles) or f"none in {ASSETS_ROOT}/vehicles")

    print("\nPlanners")
    from .agents.external import list_adapters, load_adapter
    for name in list_adapters():
        spec = load_adapter(name)
        script = next((c for c in spec.command if c.endswith((".sh", ".py"))), "")
        script = script.replace("${AETHER_ROOT}", str(REPO_ROOT)).replace("${AETHER_EXTERNAL}", str(EXTERNAL_ROOT))
        _row(Path(os.path.expandvars(script)).is_file() or None, f"{name}", script)
    limsim = Path(os.environ.get("LIMSIM_ROOT", EXTERNAL_ROOT / "LimSim")) / "aether_bridge" / "server.py"
    _row(limsim.is_file() or None, "LimSim bridge", str(limsim))
    print("\n" + ("All required checks passed." if good else "Some required checks failed."))
    return 0 if good else 1
