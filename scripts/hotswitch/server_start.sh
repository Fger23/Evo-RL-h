#!/usr/bin/env bash
# Foreground owner. For persistence launch this complete script with nohup.
set -euo pipefail
TOOLS=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$TOOLS/activate_env.sh" --quiet
GPU=1
SESSION=''
PORT=8099
BACKEND=shared
CHECK=0
usage() {
  printf '%s\n' 'Usage: bash server_start.sh --session NEW_NFS_PATH [--gpu 1|GPU-4e74582b-8cca-a9c3-8d95-73f7babc3c2a] [--port 8099|0] [--backend shared|dual-unmerged] [--check]'
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu|--session|--port|--backend)
      [[ $# -ge 2 ]] || { usage >&2; exit 2; }
      case "$1" in --gpu) GPU=$2;; --session) SESSION=$2;; --port) PORT=$2;; --backend) BACKEND=$2;; esac
      shift 2;;
    --check) CHECK=1; shift;;
    --help|-h) usage; exit 0;;
    *) usage >&2; exit 2;;
  esac
done
[[ -n "$SESSION" ]] || { usage >&2; exit 2; }
[[ "$(hostname)" == "$EVORL_EXPECTED_HOST" ]] || { echo 'Wrong server host.' >&2; exit 2; }
[[ "$GPU" == 1 || "$GPU" == "$EVORL_GPU_UUID" ]] || { echo 'This deployment is authorized for physical GPU1 only.' >&2; exit 2; }
[[ "$BACKEND" == shared || "$BACKEND" == dual-unmerged ]] || { echo 'Use shared or dual-unmerged for the verified BF16 runtime.' >&2; exit 2; }
test -f "$EVORL_REGISTRY" || { echo 'Formal task registry is missing.' >&2; exit 2; }
UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader)
[[ "$UUID" == "$EVORL_GPU_UUID" ]] || { echo 'Selected physical GPU UUID differs.' >&2; exit 2; }
"$EVORL_PYTHON" -B - "$SESSION" "$PORT" "$EVORL_INFER_ROOT" "$EVORL_SOURCE_ROOT" <<'PY'
import importlib.util, sys
from pathlib import Path
session, port, allowed, source = sys.argv[1:]
p = Path(session)
assert p.is_absolute() and not p.exists() and not p.is_symlink(), 'Choose a new absolute session path'
assert p.resolve().is_relative_to(Path(allowed).resolve()) and p.resolve() != Path(allowed).resolve(), 'Session must be below EVORL_INFER_ROOT'
assert port.isdecimal() and 0 <= int(port) <= 65535, 'Port must be 0..65535'
assert Path(importlib.util.find_spec('lerobot').origin).resolve() == Path(source) / 'src/lerobot/__init__.py', 'Wrong bound source'
PY
if [[ "$CHECK" -eq 1 ]]; then
  printf 'STATIC_CHECK_PASSED; no process or GPU allocation started.\nGPU=%s\nSession=%s\nBackend=%s\n' "$UUID" "$SESSION" "$BACKEND"
  exit 0
fi
LOCK="$EVORL_LOCK_ROOT/$UUID.lock"
exec 9>"$LOCK"
flock --nonblock 9 || { echo 'Another coordinated job owns this GPU.' >&2; exit 3; }
ACTIVE=$(nvidia-smi -i "$UUID" --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z "${ACTIVE//[[:space:]]/}" ]] || { echo 'This GPU already has a compute process; no process was stopped.' >&2; exit 3; }
mkdir -p "$EVORL_INFER_ROOT"
mkdir "$SESSION"
SESSION=$(realpath "$SESSION")
mkdir "$SESSION/server_work"
cd "$SESSION"
"$EVORL_PYTHON" -B - "$SESSION" "$$" "$UUID" "$PORT" "$BACKEND" <<'PY'
import hashlib, json, os, secrets, sys
from pathlib import Path
root, pid, gpu, port, backend = sys.argv[1:]
root = Path(root); pid = int(pid)
fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
uids = list(map(int, next(line for line in Path(f'/proc/{pid}/status').read_text().splitlines() if line.startswith('Uid:')).split()[1:]))
assert fields[0] != 'Z' and uids == [os.getuid()] * 4
identity = {'pid': pid, 'parent_pid': int(fields[1]), 'start_ticks': fields[19], 'state': fields[0],
            'uid': uids[0], 'uids': uids,
            'argv': [s.decode() for s in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if s]}
token = secrets.token_hex(24)
argv = [os.environ['EVORL_PYTHON'], '-u', '-m', 'lerobot.scripts.hotswitch_cli', 'serve',
        '--registry', os.environ['EVORL_REGISTRY'], '--backend', backend, '--dtype', 'bfloat16',
        '--device', 'cuda', '--port', port, '--ready-file', str(root / 'ready.json')]
spec = {'schema_version': 1, 'argv': argv, 'cwd': str(root / 'server_work'),
        'stdout': str(root / 'server.stdout.log'), 'stderr': str(root / 'server.stderr.log'),
        'expected_uid': os.getuid(), 'harness_pid': pid, 'harness_identity': identity,
        'environment': {'CUDA_VISIBLE_DEVICES': gpu, 'HOTSWITCH_RPC_RUN_TOKEN': token,
                        'PYTHONPATH': os.environ['PYTHONPATH']}}
for name, value in [('guardian.token', token), ('server_spec.json', json.dumps(spec, indent=2))]:
    fd = os.open(root / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream: stream.write(value)
guardian = Path(os.environ['EVORL_TOOLS']) / 'server_guardian.py'
assert hashlib.sha256(guardian.read_bytes()).hexdigest() == '789b1ee3a0d87dd1874827ed06493a2df6c61091e29e418cf12c5cd0f0ca2b09'
with (root / 'launch.json').open('x') as stream:
    json.dump({'server_argv': argv, 'gpu_uuid': gpu, 'harness_identity': identity,
               'guardian_path': str(guardian), 'guardian_sha256': hashlib.sha256(guardian.read_bytes()).hexdigest(),
               'physical_robot': False, 'training_started': False}, stream, indent=2)
PY
printf 'Session: %s\nReady: %s/ready.json\nLogs: %s/server.stdout.log and server.stderr.log\n' "$SESSION" "$SESSION" "$SESSION"
# Keep this Bash owner alive. Guardian owns the model child and its birth pidfd.
"$EVORL_PYTHON" -B "$EVORL_TOOLS/server_guardian.py" serve \
  --spec "$SESSION/server_spec.json" --state "$SESSION/guardian_state.json" --token-file "$SESSION/guardian.token"
RESULT=$?
if [[ "$RESULT" -eq 0 ]]; then
  "$EVORL_PYTHON" -B - "$SESSION/guardian_state.json" <<'PY'
import json, sys
from pathlib import Path
state = json.loads(Path(sys.argv[1]).read_text())
code = state.get('cleanup', {}).get('child_exit_code')
if state.get('status') != 'STOPPED' or code not in (0, -15, -2):
    raise SystemExit(f'Server exited unexpectedly: {code}; inspect server.stderr.log')
PY
fi
printf 'Service owner exited with code %s.\n' "$RESULT"
exit "$RESULT"
