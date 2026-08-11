"""Vendored backend paths for AETHER.

Everything needed at runtime lives under the AETHER repo:

* DriveStudio trainer / datasets / models → ``third_party/drivestudio``
* LiDAR rasterization kernel (gsplat fork) → ``third_party/splatad_gsplat``

Importing ``aether`` loads this module and puts both on ``sys.path``, so callers
do **not** need ``DRIVESTUDIO_ROOT`` / ``SPLATAD_ROOT`` exports.

Optional overrides (dev only):

* ``AETHER_DRIVESTUDIO_ROOT`` — alternate DriveStudio checkout
* ``AETHER_SPLATAD_GSPLAT_ROOT`` — parent dir containing package ``splatad_gsplat``
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]

AETHER_ROOT = Path(os.environ.get("AETHER_ROOT", str(_REPO_ROOT))).resolve()

DRIVESTUDIO_ROOT = Path(
    os.environ.get(
        "AETHER_DRIVESTUDIO_ROOT",
        str(_REPO_ROOT / "third_party" / "drivestudio"),
    )
).resolve()

# Parent of the ``splatad_gsplat`` package directory.
_SPLATAD_GSPLAT_PARENT = Path(
    os.environ.get(
        "AETHER_SPLATAD_GSPLAT_ROOT",
        str(_REPO_ROOT / "third_party"),
    )
).resolve()

SPLATAD_GSPLAT_ROOT = _SPLATAD_GSPLAT_PARENT / "splatad_gsplat"

# Preprocessed driving datasets live under AETHER/data.
PANDASET_PROCESSED = AETHER_ROOT / "data/pandaset/processed"
NUSCENES_PROCESSED = AETHER_ROOT / "data/nuscenes/processed_10Hz/trainval"


def resolve_data_root(path: str | Path) -> Path:
    """Resolve dataset root; relative paths are under AETHER_ROOT."""
    p = Path(path)
    if not p.is_absolute():
        p = AETHER_ROOT / p
    return p.resolve()


if not DRIVESTUDIO_ROOT.exists():
    raise RuntimeError(
        f"Vendored DriveStudio backend not found at {DRIVESTUDIO_ROOT}. "
        "Re-clone the AETHER repo or set AETHER_DRIVESTUDIO_ROOT."
    )

if not SPLATAD_GSPLAT_ROOT.exists():
    raise RuntimeError(
        f"Vendored LiDAR gsplat fork not found at {SPLATAD_GSPLAT_ROOT}. "
        "Expected third_party/splatad_gsplat (copied from the SplatAD gsplat fork)."
    )


def _prepend_sys_path(path: Path) -> None:
    s = str(path)
    if s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


_prepend_sys_path(DRIVESTUDIO_ROOT)
_prepend_sys_path(_SPLATAD_GSPLAT_PARENT)

# Evict a competing top-level ``utils`` so drivestudio's utils.misc wins.
_loaded = sys.modules.get("utils")
if _loaded is not None:
    f = getattr(_loaded, "__file__", "") or ""
    if str(DRIVESTUDIO_ROOT) not in f:
        for k in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
            del sys.modules[k]


def inject_hugsim() -> None:  # pragma: no cover  (deprecated shim)
    """Deprecated no-op for older callers."""
    return
