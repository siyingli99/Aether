#!/usr/bin/env python3
"""Download and verify the AETHER demo scenes from Hugging Face."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

from huggingface_hub import hf_hub_download


REPO_URL = "https://huggingface.co/datasets/mikkochi/AETHER_Demo"
ARCHIVES = {
    "0383": "scenes/nuscenes_0383.zip",
    "088": "scenes/nuscenes_088.zip",
    "assets": "vehicles/3drealcar_assets.zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (root / member.filename).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"unsafe archive path: {member.filename}")
        bundle.extractall(root)


def repo_id(value: str) -> str:
    """Accept either a Hugging Face dataset URL or an owner/name id."""
    value = value.strip().rstrip("/")
    prefix = "https://huggingface.co/datasets/"
    if value.startswith(prefix):
        value = value[len(prefix):]
    if value.count("/") != 1 or not all(value.split("/")):
        raise ValueError(
            "--repo must be a Hugging Face dataset URL or an owner/name id"
        )
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=["0383", "088", "all"], default="all")
    parser.add_argument("--no-assets", action="store_true")
    parser.add_argument(
        "--repo", "--repo-id", dest="repo", default=REPO_URL,
        help="Hugging Face dataset URL or owner/name id",
    )
    parser.add_argument("--output", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()

    output = Path(args.output).expanduser().resolve()
    dataset_id = repo_id(args.repo)
    manifest_path = Path(hf_hub_download(
        dataset_id, "manifest.json", repo_type="dataset"
    ))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = ["0383", "088"] if args.scene == "all" else [args.scene]
    if not args.no_assets:
        selected.append("assets")

    for key in selected:
        filename = ARCHIVES[key]
        local = Path(hf_hub_download(dataset_id, filename, repo_type="dataset"))
        expected = manifest["files"][filename]["sha256"]
        actual = sha256(local)
        if actual != expected:
            raise RuntimeError(f"checksum mismatch for {filename}: {actual} != {expected}")
        print(f"[aether-data] verified {filename}")
        safe_extract(local, output)
        print(f"[aether-data] extracted {filename} -> {output}")

    data_dir = output / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(manifest_path, data_dir / "demo_manifest.json")
    print("[aether-data] ready")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[aether-data] ERROR: {exc}", file=sys.stderr)
        raise
