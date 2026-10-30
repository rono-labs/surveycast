#!/usr/bin/env bash
# Score surveycast on the proxy datasets built by proxy/build_proxies.py and
# proxy/subsample.py.
set -euo pipefail
cd "$(dirname "$0")/.."
PHASE="${PHASE:-2}"
mkdir -p work/results
for d in work/proxy/*/; do
  name="$(basename "$d")"
  printf '%-14s ' "$name"
  python scripts/evaluate.py --sub surveycast --data "$d" --schema "$d/schema.json" \
    --phase "$PHASE" --no-deadline --json "work/results/${name}_p$PHASE.json"
done
