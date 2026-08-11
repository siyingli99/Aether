#!/usr/bin/env python3
"""Function: Check AETHER Python dependencies and local assets without modifying them.
Input/Output: Reads the current environment and repository assets; prints statuses and returns a diagnostic exit code.
Example:
    python scripts/check_install.py
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    level: str
    name: str
    ok: bool
    detail: str


def package_check(name: str, required: bool = True) -> Check:
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        return Check("required" if required else "optional", name, False, str(exc))
    return Check(
        "required" if required else "optional",
        f"Python package: {name}",
        found,
        "importable" if found else "not importable in this interpreter",
    )


def path_check(name: str, path: Path, required: bool, kind: str = "exists") -> Check:
    if kind == "file":
        ok = path.is_file()
    elif kind == "dir":
        ok = path.is_dir()
    elif kind == "executable":
        ok = path.is_file() and os.access(path, os.X_OK)
    else:
        ok = path.exists()
    try:
        shown = path.resolve().relative_to(ROOT.resolve())
    except ValueError:
        shown = path
    return Check("required" if required else "optional", name, ok, str(shown))


def first_existing(paths: Iterable[Path]) -> Path | None:
    return next((path for path in paths if path.exists()), None)


def main() -> int:
    checks: list[Check] = []
    checks.append(Check("required", "Python version >= 3.10", sys.version_info >= (3, 10),
                        sys.version.split()[0]))
    for package in (
        "yaml", "numpy", "torch", "omegaconf", "imageio",
        "diffusers", "transformers", "peft", "huggingface_hub",
    ):
        checks.append(package_check(package))
    checks.extend([
        Check(
            "required",
            "Ninja executable",
            shutil.which("ninja") is not None,
            shutil.which("ninja") or "not found on PATH; activate the aether environment",
        ),
        Check(
            "optional",
            "CUDA compiler (nvcc)",
            shutil.which("nvcc") is not None,
            shutil.which("nvcc") or "not found; required for first-time CUDA extension builds",
        ),
    ])

    checks.extend([
        path_check("DriveStudio vendor", ROOT / "third_party/drivestudio", True, "dir"),
        path_check("SplatAD gsplat vendor", ROOT / "third_party/splatad_gsplat", True, "dir"),
        path_check("UniAD vendor", ROOT / "third_party/uniad_sim", False, "dir"),
        path_check("Difix3D vendor", Path(os.environ.get(
            "DIFIX3D_ROOT", ROOT / "third_party/difix3d")), False, "dir"),
    ])

    scene_dir = ROOT / "data/pandaset/processed/095"
    checkpoint_dir = (ROOT / "data/checkpoints/pandaset_outputs/scene_recon_full"
                      / "scene_095_6cams_with_humans")
    checks.extend([
        path_check("095 checkpoint", checkpoint_dir / "checkpoint_final.pth", True, "file"),
        path_check("095 checkpoint config", checkpoint_dir / "config.yaml", True, "file"),
        path_check("095 ego poses", scene_dir / "ego_pose", True, "dir"),
        path_check("095 ground parameters",
                   ROOT / "data/exported_scenes/pandaset/pandaset_095_final/ground_param.pkl",
                   True, "file"),
    ])
    missing_extrinsics = [
        f"000_{index}.txt"
        for index in range(6)
        if not (scene_dir / "extrinsics" / f"000_{index}.txt").is_file()
    ]
    checks.append(Check(
        "required", "095 camera extrinsics", not missing_extrinsics,
        "000_0.txt ... 000_5.txt present" if not missing_extrinsics
        else "missing: " + ", ".join(missing_extrinsics),
    ))

    uniad_python = Path(os.environ.get("UNIAD_PYTHON", ROOT / ".conda/uniad/bin/python"))
    uniad_checkpoint = Path(os.environ.get(
        "UNIAD_CHECKPOINT", ROOT / "checkpoints/uniad_base_e2e.pth"))
    checks.extend([
        path_check("UniAD Python", uniad_python, False, "executable"),
        path_check("UniAD checkpoint", uniad_checkpoint, False, "file"),
        path_check("Full processed 095 archive",
                   ROOT / "assets/local/pandaset095_processed.zip", False, "file"),
    ])

    sidecar = first_existing((
        ROOT / "outputs/lidar_sidecar",
        ROOT / "data/lidar_sidecar",
        ROOT / "user_outputs/lidar_sidecar",
    ))
    checks.append(Check(
        "optional", "Phase-B LiDAR sidecar", sidecar is not None,
        str(sidecar.relative_to(ROOT)) if sidecar else
        "not found in conventional locations; pass explicit paths to LiDAR commands",
    ))

    for check in checks:
        status = "PASS" if check.ok else ("FAIL" if check.level == "required" else "MISS")
        print(f"[{status:4}] [{check.level:8}] {check.name}: {check.detail}")

    failures = [check for check in checks if check.level == "required" and not check.ok]
    print(f"\nRequired: {len(checks) - sum(c.level == 'optional' for c in checks) - len(failures)} passed, "
          f"{len(failures)} failed")
    print("Optional misses do not make the check fail.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
