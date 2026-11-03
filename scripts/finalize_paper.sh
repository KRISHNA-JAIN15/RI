#!/usr/bin/env bash
# Wait for a running benchmark, then rebuild everything the paper consumes.
set -euo pipefail
cd "$(dirname "$0")/.."
source ../.venv/bin/activate
while pgrep -f "scripts/benchmark.py --workers" >/dev/null; do sleep 60; done
echo "benchmark finished: $(tail -1 runs/bench_full.log)"
for d in runs/bench/*/*/seed*/eval; do
  [ -f "$d/table1.json" ] || continue
  [ -f "$d/stratified.json" ] && continue
  python scripts/stratify.py --eval-dir "$d" --config "$d/../config.yaml" --seed 12345 --bins 4 >/dev/null 2>&1 || echo "stratify failed: $d"
done
python scripts/report_benchmark.py --out paper
python scripts/paper_tables.py
(cd paper && tectonic -X compile main.tex 2>&1 | grep -E "^error" || true)
python -c "from pypdf import PdfReader; print('pages:', len(PdfReader('paper/main.pdf').pages))"
cp paper/main.pdf runs/bench/paper_draft.pdf
echo "FINALIZED $(date)"
