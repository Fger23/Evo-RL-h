#!/usr/bin/env bash
set -euo pipefail
TOOLS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$TOOLS/activate_env.sh" --quiet
[[ $# -eq 1 ]] || { echo 'Usage: bash server_stop.sh /nfs/baiyuntian/assets/model/evorl_manual_inference/SESSION' >&2; exit 2; }
"$EVORL_PYTHON" -B - "$1" <<'PY'
import hashlib, importlib.util, json, os, sys, time
from pathlib import Path
p = Path(sys.argv[1]).resolve(strict=True)
assert p.is_relative_to(Path(os.environ['EVORL_INFER_ROOT']).resolve())
path = Path(os.environ['EVORL_TOOLS']) / 'server_guardian.py'
assert hashlib.sha256(path.read_bytes()).hexdigest() == '789b1ee3a0d87dd1874827ed06493a2df6c61091e29e418cf12c5cd0f0ca2b09'
spec = importlib.util.spec_from_file_location('verified_inference_guardian', path)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.request(p / 'guardian_state.json', p / 'guardian.token', 'stop')
deadline = time.monotonic() + 30
while time.monotonic() < deadline:
    state = json.loads((p / 'guardian_state.json').read_text())
    if state['status'] == 'STOPPED' and state['reaped'] and state['pidfd_held'] is False:
        print(json.dumps({'status': 'STOPPED', 'session': str(p), 'cleanup': state['cleanup']})); break
    if state['status'] == 'FAILED': raise RuntimeError(state.get('error', state))
    time.sleep(.1)
else: raise TimeoutError('Guardian has not confirmed exit; inspect this session, do not signal arbitrary PIDs')
PY
