#!/usr/bin/env bash
# Shared launcher: portable on the client; reuses the known environment on node3.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
ROLE=$1
shift
if [[ -n "${EVORL_PYTHON:-}" ]]; then
  PYTHON=$EVORL_PYTHON
elif [[ "$ROLE" == server && -x "$ROOT/20261104_gpu1/runtime-clean/bin/python" ]]; then
  PYTHON="$ROOT/20261104_gpu1/runtime-clean/bin/python"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
  PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON="$CONDA_PREFIX/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON="$ROOT/.venv/bin/python"
elif [[ -x "$ROOT/20261104_gpu1/runtime-clean/bin/python" ]]; then
  PYTHON="$ROOT/20261104_gpu1/runtime-clean/bin/python"
else
  PYTHON=$(command -v python3 || command -v python)
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export LEROBOT_ASYNC_LOG_TO_FILE=0
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
if [[ "$ROLE" == client ]]; then
  export CUDA_VISIBLE_DEVICES=''
elif [[ "$ROLE" != server ]]; then
  echo 'Expected server or client role.' >&2
  exit 2
fi
cd "$ROOT"
exec "$PYTHON" -B -u "$ROOT/$ROLE.py" "$@"
