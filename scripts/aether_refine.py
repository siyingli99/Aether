"""Function: Refine rendered closed-loop camera frames with Difix3D.
Input/Output: Reads an adapter frame directory and optional references; writes refined frames, metrics, and an optional video.
Example:
    # Prerequisite: run a closed-loop rollout to create the input frame directory.
    python scripts/aether_refine.py aether_results/pandaset/pandaset_095_aether/uniad/easy \
        --output_dir outputs/pandaset095_refined --max_side 1024 --video
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image

# Make `aether` importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aether.refine import refine_directory  # noqa: E402


_VALID_EXT = (".png", ".jpg", ".jpeg", ".webp")


def _resolve_input(p: Path) -> Path:
    """Accept either an adapter-frame dir or its parent ``easy/`` dir."""
    if any(p.glob(f"*{e}") for e in _VALID_EXT for _ in [None]):
        return p
    # Try adapter subdirs (vad/, uniad/, ltf/, vla/).
    for cand in ("vad", "uniad", "ltf", "vla"):
        c = p / cand
        if c.is_dir() and any(c.glob(f"*{e}") for e in _VALID_EXT):
            return c
    raise SystemExit(f"[aether_refine] no images found under {p}")


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse <= 1e-10:
        return 99.0
    return float(20.0 * np.log10(255.0 / np.sqrt(mse)))


def _quick_diff(orig_dir: Path, refined_dir: Path) -> dict:
    """Cheap PSNR(refined vs. original) sanity stat."""
    out: dict = {"frames": 0, "psnr_mean_vs_orig": None}
    psnrs: list[float] = []
    for p in sorted(orig_dir.iterdir()):
        if p.suffix.lower() not in _VALID_EXT:
            continue
        q = refined_dir / p.name
        if not q.exists():
            continue
        a = np.asarray(Image.open(p).convert("RGB"))
        b = np.asarray(Image.open(q).convert("RGB"))
        if a.shape != b.shape:
            continue
        psnrs.append(_psnr(a, b))
    if psnrs:
        out["frames"] = len(psnrs)
        out["psnr_mean_vs_orig"] = float(np.mean(psnrs))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_dir", help="Adapter frame dir, or its parent 'easy/' dir.")
    ap.add_argument("--output_dir", default=None,
                    help="Default: <input_dir>_refined alongside the input.")
    ap.add_argument("--ref_dir", default=None,
                    help="Optional MV-UNet reference frames (switches model to nvidia/difix_ref).")
    ap.add_argument("--model_name", default="nvidia/difix")
    ap.add_argument("--prompt", default="remove degradation")
    ap.add_argument("--max_side", type=int, default=1024)
    ap.add_argument("--timestep", type=int, default=199)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--no_video", dest="video", action="store_false")
    ap.set_defaults(video=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES for the Difix subprocess.")
    ap.add_argument("--difix_python", default=None)
    ap.add_argument("--difix_repo", default=None)
    args = ap.parse_args()

    in_dir = _resolve_input(Path(args.input_dir).resolve())
    out_dir = Path(args.output_dir).resolve() if args.output_dir else \
              in_dir.with_name(in_dir.name + "_refined")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[aether_refine] in : {in_dir}")
    print(f"[aether_refine] out: {out_dir}")

    perf = refine_directory(
        in_dir, out_dir,
        ref_dir=args.ref_dir,
        model_name=args.model_name,
        prompt=args.prompt,
        max_side=args.max_side,
        timestep=args.timestep,
        seed=args.seed,
        video=args.video,
        fps=args.fps,
        limit=args.limit,
        cuda_visible=args.cuda,
        difix_python=args.difix_python,
        difix_repo=args.difix_repo,
    )
    diff = _quick_diff(in_dir, out_dir)

    summary = {
        "input_dir":  str(in_dir),
        "output_dir": str(out_dir),
        "perf":       perf,
        "diff":       diff,
    }
    json_p = out_dir / "aether_refine_summary.json"
    with open(json_p, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[aether_refine] === summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
