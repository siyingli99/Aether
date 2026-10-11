#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${AETHER_PYTHON:-python}"

cd "$repo_dir"
exec "$python_bin" scripts/aether_studio.py "$@"
