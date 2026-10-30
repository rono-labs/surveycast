#!/usr/bin/env bash
# Generate the three official sandboxes and score surveycast on each, phase 2
# (TRAIN+DEV visible, TEST scored), with the run-time guard lifted.
set -euo pipefail
[ -n "${KIT:-}" ] && KIT="$(realpath "$KIT")"
cd "$(dirname "$0")/.."
KIT="${KIT:-kit}"
PHASE="${PHASE:-2}"
mkdir -p work/results
for s in unicef world_bank unhcr; do
  [ -f "work/sbx/$s/respondents.parquet" ] || \
    python "$KIT/make_sandbox.py" --schema "$KIT/data/$s.json" --out "work/sbx/$s" \
      --config "$KIT/config.yml" > /dev/null
  printf '%-11s ' "$s"
  python scripts/evaluate.py --sub surveycast --data "work/sbx/$s" \
    --schema "$KIT/data/$s.json" --phase "$PHASE" --no-deadline \
    --json "work/results/${s}_p$PHASE.json"
done
