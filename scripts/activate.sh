#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo 'Run: source scripts/activate.sh' >&2
  exit 2
fi
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/hotswitch/activate_env.sh" "$@"
