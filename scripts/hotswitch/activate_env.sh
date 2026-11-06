#!/usr/bin/env bash
# Source this file; no installation and no CUDA allocation is performed.
if [[ -z "${BASH_VERSION:-}" ]]; then
  echo 'Use Bash to source this file.' >&2
  return 2
fi
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Run: source scripts/activate.sh from the repository root.' >&2
  exit 2
fi
export EVORL_CODE_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export EVORL_SOURCE_ROOT="$EVORL_CODE_ROOT"
export EVORL_TOOLS="$EVORL_CODE_ROOT/scripts/hotswitch"
export EVORL_CONFIG="${EVORL_CONFIG:-$EVORL_CODE_ROOT/configs/hotswitch/node3.env}"
test -f "$EVORL_CONFIG" || { echo "Missing config: $EVORL_CONFIG" >&2; return 2; }
source "$EVORL_CONFIG" || return
export EVORL_RUNTIME_ROOT EVORL_PYTHON EVORL_FORMAL_RUN EVORL_REGISTRY EVORL_INFER_ROOT
export EVORL_GPU_UUID EVORL_EXPECTED_HOST EVORL_LOCK_ROOT
test -x "$EVORL_PYTHON" || { echo 'Configured Python is missing.' >&2; return 2; }
source "$EVORL_RUNTIME_ROOT/runtime-clean/bin/activate" || return
export PATH="$EVORL_RUNTIME_ROOT/runtime-clean/bin:$EVORL_RUNTIME_ROOT/runtime-base/bin:$PATH"
export PYTHONPATH="$EVORL_CODE_ROOT/src:$EVORL_TOOLS:$EVORL_CODE_ROOT/examples/hotswitch"
export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1
export HF_HUB_OFFLINE=1 WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
# Clients and guardians stay on CPU. Only the owned model child gets GPU1.
export CUDA_VISIBLE_DEVICES=''
hash -r
if [[ "${1:-}" != '--quiet' ]]; then
  printf 'Python: %s\nSource: %s\nRegistry: %s\n' "$EVORL_PYTHON" "$EVORL_SOURCE_ROOT" "$EVORL_REGISTRY"
fi
