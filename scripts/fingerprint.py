"""Function: Aggregate per-scene diagnosis reports into a cross-scene fingerprint.
Input/Output: Reads one or more report JSON globs and writes the aggregated fingerprint JSON.
Example:
    python scripts/fingerprint.py 'outputs/diagnosis/*/report.json' \
        --out outputs/diagnosis/fingerprint.json --k 3
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+",
                    help="report JSONs (globs expanded).")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    paths = []
    for pat in args.inputs:
        paths.extend(sorted(glob.glob(pat)))
    if not paths:
        sys.exit("no input json found")
    reports = []
    for p in paths:
        try:
            with open(p) as f:
                reports.append(json.load(f))
        except Exception as e:
            print(f"skip {p}: {e}")
    if not reports:
        sys.exit("no valid reports loaded")
    print(f"loaded {len(reports)} reports")

    from aether.diagnose.fingerprint import cluster_fingerprints
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    summary = cluster_fingerprints(reports, k=args.k, output_dir=out)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ("feature_matrix", "pca")}, indent=2))


if __name__ == "__main__":
    main()
