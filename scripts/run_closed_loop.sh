#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scene="${1:-0383}"
if [[ "$scene" != "0383" && "$scene" != "088" ]]; then
  echo "usage: $0 {0383|088} [closed_loop.py options...]" >&2
  exit 2
fi
shift || true

aether_python="${AETHER_PYTHON:-$repo_dir/.conda/aether/bin/python}"
limsim_python="${LIMSIM_PYTHON:-$repo_dir/.conda/limsim/bin/python}"
limsim_root="${LIMSIM_ROOT:-$(cd "$repo_dir/.." && pwd)/LimSim}"
scenario="${AETHER_SCENARIO:-configs/scenarios/nuscenes_${scene}.yaml}"
network_dir="$repo_dir/data/limsim/nuscenes_${scene}"
network_prefix="scene${scene}"
bridge_host="${LIMSIM_HOST:-127.0.0.1}"
bridge_port="${LIMSIM_PORT:-8766}"

for executable in "$aether_python" "$limsim_python"; do
  [[ -x "$executable" ]] || { echo "Python not found: $executable" >&2; exit 1; }
done
[[ -f "$limsim_root/aether_bridge/server.py" ]] || {
  echo "AETHER LimSim bridge not found under $limsim_root" >&2
  echo "Clone the aether-integration branch described in README.md." >&2
  exit 1
}

export PATH="$(dirname "$limsim_python"):$(dirname "$aether_python"):/usr/local/bin:/usr/bin:/bin"
export PYTHONNOUSERSITE=1
export AETHER_AD_READY_TIMEOUT="${AETHER_AD_READY_TIMEOUT:-1800}"

bridge_pid=""
cleanup() {
  if [[ -n "$bridge_pid" ]] && kill -0 "$bridge_pid" 2>/dev/null; then
    "$aether_python" - "$bridge_host" "$bridge_port" <<'PY' 2>/dev/null || true
import sys
from aether.traffic.client import LimSimTrafficClient
client = LimSimTrafficClient(sys.argv[1], int(sys.argv[2]), timeout_s=2.0)
try:
    client.shutdown()
finally:
    client.close()
PY
    kill "$bridge_pid" 2>/dev/null || true
    wait "$bridge_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

cd "$repo_dir"
AETHER_ROOT="$repo_dir" "$limsim_python" -u \
  "$limsim_root/aether_bridge/server.py" \
  --host "$bridge_host" --port "$bridge_port" \
  --net-file "$network_dir/${network_prefix}.net.xml" \
  --route-file "$network_dir/${network_prefix}.rou.xml" &
bridge_pid=$!

bridge_ready=0
for _ in $(seq 1 120); do
  if "$aether_python" - "$bridge_host" "$bridge_port" <<'PY' 2>/dev/null
import socket, sys
with socket.create_connection((sys.argv[1], int(sys.argv[2])), 0.2):
    pass
PY
  then
    bridge_ready=1
    break
  fi
  kill -0 "$bridge_pid" 2>/dev/null || { wait "$bridge_pid"; exit 1; }
  sleep 0.25
done
[[ "$bridge_ready" -eq 1 ]] || { echo "LimSim bridge did not start" >&2; exit 1; }

"$aether_python" scripts/closed_loop.py \
  --scenario "$scenario" \
  --base configs/base.yaml \
  --camera configs/cameras/nuscenes.yaml \
  --kinematic configs/kinematic.yaml \
  --ad uniad --ad_cuda "${AETHER_CUDA:-0}" "$@"
