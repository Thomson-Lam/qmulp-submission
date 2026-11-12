#!/usr/bin/env bash
# Run an existing Python entry point with the journal src/ layout.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-$ROOT/.venv/bin/python}" "$@"
