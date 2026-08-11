"""Function: Train a DriveStudio scene and refresh extracted traffic trajectories.
Input/Output: Reads a DriveStudio config plus OmegaConf overrides; writes scene checkpoints and traffic sidecars.
Example:
    python scripts/train_scene.py --config_file third_party/drivestudio/configs/omnire.yaml \
        dataset=pandaset/6cams data.data_root=data/pandaset/processed data.scene_idx=95
"""
from __future__ import annotations
import argparse
import os
import runpy
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
import aether  # noqa: F401

from aether.paths import DRIVESTUDIO_ROOT


def _backfill_traffic_logs() -> None:
    scen_dir = _REPO / "configs" / "scenarios"
    extractor = _REPO / "tools" / "extract_traffic_from_ckpt.py"
    if not extractor.exists():
        return
    from omegaconf import OmegaConf  # type: ignore
    for yml in sorted(scen_dir.glob("*_aether.yaml")):
        try:
            c = OmegaConf.load(yml)
            ck = OmegaConf.to_container(c, resolve=True).get("checkpoint", "")
        except Exception:
            continue
        if not ck or not Path(ck).is_file():
            continue
        npz = scen_dir / "traffic" / f"{yml.stem}.npz"
        if npz.is_file() and npz.stat().st_mtime >= Path(ck).stat().st_mtime:
            continue
        print(f"[train_scene] refreshing traffic for {yml.name}")
        subprocess.run([sys.executable, str(extractor),
                        "--scenario", str(yml)], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_file", required=True,
                    help="Path to a drivestudio omegaconf yaml.")
    ap.add_argument("--no_traffic_inject", action="store_true",
                    help="Skip auto-injection of traffic_log after training.")
    args, extra = ap.parse_known_args()
    cfg_path = os.path.abspath(args.config_file)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(cfg_path)
    sys.argv = ["train.py", f"--config_file={cfg_path}"] + extra
    train_py = str(DRIVESTUDIO_ROOT / "tools" / "train.py")
    if not os.path.exists(train_py):
        raise FileNotFoundError(train_py)
    print(f"[aether] -> {train_py}  ({len(extra)} overrides)")
    print(f"[aether] config_file = {cfg_path}")
    os.chdir(str(DRIVESTUDIO_ROOT))
    try:
        runpy.run_path(train_py, run_name="__main__")
    finally:
        if not args.no_traffic_inject:
            os.chdir(str(_REPO))
            try:
                _backfill_traffic_logs()
            except Exception as e:
                print(f"[train_scene] traffic backfill failed: {e}")


if __name__ == "__main__":
    main()
