"""Repository layout and vendored back-ends.

Default locations (override with environment variables):

=================  ==========================  ======================
what               default                     variable
=================  ==========================  ======================
scene bundles      ``data/scenes``             ``AETHER_SCENES``
vehicle assets     ``data/assets``             ``AETHER_ASSETS``
run outputs        ``outputs``                 ``AETHER_OUTPUTS``
model weights      ``checkpoints``             ``AETHER_CHECKPOINTS``
external planners  parent of the repository    ``AETHER_EXTERNAL``
=================  ==========================  ======================

The DriveStudio renderer and the SplatAD LiDAR kernel are vendored under
``third_party``.  They are put on ``sys.path`` by :func:`use_vendored_backends`
the first time a module needs them, so lightweight parts of AETHER (traffic,
agents SDK, evaluation, Studio API) import without them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default


AETHER_ROOT = _env_path("AETHER_ROOT", REPO_ROOT)
SCENES_ROOT = _env_path("AETHER_SCENES", AETHER_ROOT / "data" / "scenes")
ASSETS_ROOT = _env_path("AETHER_ASSETS", AETHER_ROOT / "data" / "assets")
OUTPUTS_ROOT = _env_path("AETHER_OUTPUTS", AETHER_ROOT / "outputs")
CHECKPOINTS_ROOT = _env_path("AETHER_CHECKPOINTS", AETHER_ROOT / "checkpoints")
# Checkouts of UniAD_SIM, VAD_SIM, NAVSIM, Impromptu-VLA … (adapter YAMLs use
# ``${AETHER_EXTERNAL}``); default: next to the AETHER checkout.
EXTERNAL_ROOT = _env_path("AETHER_EXTERNAL", REPO_ROOT.parent)

DRIVESTUDIO_ROOT = _env_path("AETHER_DRIVESTUDIO_ROOT", REPO_ROOT / "third_party" / "drivestudio")
_SPLATAD_PARENT = _env_path("AETHER_SPLATAD_GSPLAT_ROOT", REPO_ROOT / "third_party")
SPLATAD_GSPLAT_ROOT = _SPLATAD_PARENT / "splatad_gsplat"


def resolve(path: str | Path, base: Path | None = None) -> Path:
    """Expand ``~`` and resolve a relative path against ``base`` (repo root)."""
    value = Path(os.path.expandvars(str(path))).expanduser()
    return (value if value.is_absolute() else (base or AETHER_ROOT) / value).resolve()


_BACKENDS_READY = False


def use_vendored_backends() -> None:
    """Put the vendored DriveStudio and SplatAD packages on ``sys.path``."""
    global _BACKENDS_READY
    if _BACKENDS_READY:
        return
    for root, hint in ((DRIVESTUDIO_ROOT, "AETHER_DRIVESTUDIO_ROOT"),
                       (SPLATAD_GSPLAT_ROOT, "AETHER_SPLATAD_GSPLAT_ROOT")):
        if not root.exists():
            raise RuntimeError(f"vendored back-end missing: {root} (re-clone or set {hint})")
    for entry in (str(DRIVESTUDIO_ROOT), str(_SPLATAD_PARENT)):
        if entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)
    # DriveStudio imports a top-level ``utils``; evict a different one.
    loaded = sys.modules.get("utils")
    if loaded is not None and str(DRIVESTUDIO_ROOT) not in (getattr(loaded, "__file__", "") or ""):
        for key in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
            del sys.modules[key]
    # CUDA extensions (nvdiffrast, gsplat) are JIT-built with ninja, which
    # lives next to the interpreter in a conda env that was not "activated".
    bin_dir = str(Path(sys.executable).parent)
    if bin_dir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    _BACKENDS_READY = True


# Raw preprocessed datasets used by the training / LiDAR research tools.
PANDASET_PROCESSED = AETHER_ROOT / "data/pandaset/processed"
NUSCENES_PROCESSED = AETHER_ROOT / "data/nuscenes/processed_10Hz/trainval"


def resolve_data_root(path: str | Path) -> Path:
    """Resolve a dataset root; relative paths are under the AETHER root."""
    return resolve(path)
