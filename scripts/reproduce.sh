#!/usr/bin/env bash
# Existing pipeline entry points; smoke checks are not paper results.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
case "${1:-}" in
  classical)
    bash scripts/run.sh -m scripts.run_all --dataset hmeq --smoke --from-stage 01 --to-stage 03 --skip-figures
    ;;
  hardware)
    # Requires the HMEQ data/split prepared by the classical command.
    bash scripts/run.sh -m scripts.run_all --dataset hmeq --smoke --from-stage 06 --to-stage 06 --skip-figures
    ;;
  paper)
    bash scripts/run.sh -m scripts.preflight
    bash scripts/run.sh -m scripts.run_all
    ;;
  tests)
    bash scripts/run.sh -m pytest tests/ -q
    ;;
  *) echo 'Usage: bash scripts/reproduce.sh classical|hardware|paper|tests' >&2; exit 2 ;;
esac
