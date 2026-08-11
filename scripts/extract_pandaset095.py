#!/usr/bin/env python3
"""Function: Safely extract a local processed PandaSet 095 archive.
Input/Output: Reads a ZIP archive and writes its validated contents to the destination directory.
Example:
    python scripts/extract_pandaset095.py --archive assets/local/pandaset095_processed.zip \
        --destination data/pandaset/processed
"""
from __future__ import annotations

import argparse
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "assets/local/pandaset095_processed.zip"
DEFAULT_DESTINATION = ROOT / "data/pandaset/processed"


def safe_members(archive: zipfile.ZipFile, destination: Path):
    destination_resolved = destination.resolve()
    seen: set[Path] = set()
    for info in archive.infolist():
        raw = info.filename
        path = PurePosixPath(raw)
        if not raw or "\x00" in raw:
            raise ValueError(f"invalid empty/NUL archive member: {raw!r}")
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive path: {raw!r}")
        if "\\" in raw:
            raise ValueError(f"backslash paths are not accepted: {raw!r}")
        mode = info.external_attr >> 16
        file_type = stat.S_IFMT(mode)
        if stat.S_ISLNK(mode):
            raise ValueError(f"symbolic links are not accepted: {raw!r}")
        if file_type and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
            raise ValueError(f"unsupported archive entry type: {raw!r}")
        if info.flag_bits & 0x1:
            raise ValueError(f"encrypted archive member is not supported: {raw!r}")

        target = (destination / Path(*path.parts)).resolve()
        if target != destination_resolved and destination_resolved not in target.parents:
            raise ValueError(f"archive member escapes destination: {raw!r}")
        if target in seen:
            raise ValueError(f"duplicate archive target: {raw!r}")
        seen.add(target)
        yield info, target


def extract(archive_path: Path, destination: Path, force: bool) -> int:
    if not archive_path.is_file():
        raise FileNotFoundError(f"archive not found: {archive_path}")
    destination.mkdir(parents=True, exist_ok=True)

    extracted = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = list(safe_members(archive, destination))
        conflicts = [
            target for info, target in members
            if not info.is_dir() and target.exists()
        ]
        if conflicts and not force:
            preview = ", ".join(str(path) for path in conflicts[:3])
            raise FileExistsError(
                f"{len(conflicts)} destination file(s) already exist "
                f"({preview}); rerun with --force to overwrite"
            )

        for info, target in members:
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted += 1
    return extracted


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely extract a local PandaSet 095 processed archive.")
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--force", action="store_true",
                        help="overwrite files that already exist")
    args = parser.parse_args()

    count = extract(args.archive.expanduser(), args.destination.expanduser(), args.force)
    print(f"Extracted {count} file(s) into {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
