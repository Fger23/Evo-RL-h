#!/usr/bin/env bash
# Model-free CPU regression suite; use a NEW evidence directory on personal NFS.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$ROOT/scripts/activate.sh" --quiet
[[ $# -eq 1 ]] || { echo 'Usage: bash scripts/test_hotswitch.sh NEW_NFS_EVIDENCE_DIR' >&2; exit 2; }
EVIDENCE=$1
"$EVORL_PYTHON" -B - "$EVIDENCE" <<'PY'
from pathlib import Path
import sys
p = Path(sys.argv[1])
assert p.is_absolute() and not p.exists() and not p.is_symlink()
assert p.resolve().is_relative_to(Path('/nfs/baiyuntian/assets/model'))
p.mkdir()
PY
cd "$EVIDENCE"
export TMPDIR="$EVIDENCE"
export RECORDING_ATTRIBUTION_SOURCE="$ROOT/src"
export FORMAL_GUARDIAN_TEST_ARTIFACT_ROOT="$EVIDENCE"
"$EVORL_PYTHON" -B -m unittest discover -s "$ROOT/tests/hotswitch" -p 'test_*.py' -v > core.log 2>&1
"$EVORL_PYTHON" -B -m unittest discover -s "$ROOT/tests/tools" -p 'test_*.py' -v > tools.log 2>&1
"$EVORL_PYTHON" -B -m unittest discover -s "$ROOT/tests/operations" -p 'test_*.py' -v > guardian.log 2>&1
"$EVORL_PYTHON" -B "$ROOT/tests/operations/validate_inference_cpu.py" --evidence-dir "$EVIDENCE/inference" > inference.log 2>&1
printf 'CPU regression suites passed. Logs: %s\n' "$EVIDENCE"
