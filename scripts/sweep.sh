#!/usr/bin/env bash
# Launch several short training runs in parallel: scripts/sweep.sh EPISODES name:"k=v k=v" ...
set -euo pipefail
cd "$(dirname "$0")/.."
EP=$1; shift
mkdir -p runs/sweep
for spec in "$@"; do
  name=${spec%%:*}; sets=${spec#*:}; args=()
  for kv in $sets; do args+=(--set "$kv"); done
  nohup python scripts/train.py --episodes "$EP" --out "runs/sweep/$name" --threads 1 "${args[@]}" > "runs/sweep/$name.log" 2>&1 &
  echo "started $name (pid $!)"
done
