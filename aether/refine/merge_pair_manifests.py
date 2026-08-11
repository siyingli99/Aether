"""Function: Merge per-scene pair manifests into a global train/validation split.
Input/Output: Reads generated ``pairs.json`` files below corpus sources and writes one global manifest.
Example:
    # Prerequisite: generate render_corpus/pandaset095/pairs.json first.
    python aether/refine/merge_pair_manifests.py \
        --corpus_root outputs/multimodal_refine_v2 --sources render_corpus \
        --out outputs/multimodal_refine_v2/global_pairs.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from aether.paths import AETHER_ROOT


_REAL_SOURCES = {
    "real_render", "real_render_perturbed", "real_render_nominal",
}


def _collect(corpus_dirs: list[Path], *, real_only: bool = False) -> dict:
    out = {}
    for root in corpus_dirs:
        if not root.is_dir():
            continue
        for scene_dir in sorted(root.iterdir()):
            manifest = scene_dir / "pairs.json"
            if not manifest.is_file():
                continue
            data = json.load(manifest.open())
            for k, v in data.items():
                src = str(v.get("source", "")).lower()
                if real_only and src not in _REAL_SOURCES and not src.startswith("real"):
                    continue
                key = f"{scene_dir.name}/{k}"
                out[key] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--corpus_root",
        default=str(AETHER_ROOT / "outputs/multimodal_refine_v2"),
    )
    ap.add_argument("--sources", default="render_corpus",
                    help="comma subdirs under corpus_root")
    ap.add_argument("--real_only", action="store_true", default=True,
                    help="drop synthetic gt_corpus pairs (v2 default)")
    ap.add_argument("--allow_synthetic", action="store_true",
                    help="include synthetic gt_corpus pairs")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.corpus_root)
    dirs = [root / s.strip() for s in args.sources.split(",") if s.strip()]
    real_only = args.real_only and not args.allow_synthetic
    all_pairs = _collect(dirs, real_only=real_only)
    keys = sorted(all_pairs.keys())
    rng = random.Random(args.seed)
    rng.shuffle(keys)
    n_val = max(1, int(len(keys) * args.val_ratio))
    val = set(keys[:n_val])
    split = {
        "train": {k: all_pairs[k] for k in keys if k not in val},
        "test": {k: all_pairs[k] for k in keys if k in val},
    }
    out = Path(args.out) if args.out else root / "global_pairs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fp:
        json.dump(split, fp, indent=2)
    print(f"[merge] train={len(split['train'])} test={len(split['test'])} -> {out}", flush=True)


if __name__ == "__main__":
    main()
