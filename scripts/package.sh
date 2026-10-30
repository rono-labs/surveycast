#!/usr/bin/env bash
# Build surveycast.zip (the upload archive) and validate it with the kit's checker.
set -euo pipefail
[ -n "${KIT:-}" ] && KIT="$(realpath "$KIT")"
cd "$(dirname "$0")/.."
KIT="${KIT:-kit}"
rm -f surveycast.zip
python - <<'PY'
import zipfile
with zipfile.ZipFile("surveycast.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("main.py", "requirements.txt", "LICENSE"):
        z.write("surveycast/" + name, name)
PY
python "$KIT/tools/check_submission_zip.py" surveycast.zip
python -m zipfile -l surveycast.zip
