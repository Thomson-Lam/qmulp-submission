#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
lake build
lake env lean QPPR/Manuscript/Audit.lean > AUDIT_RESULT.txt 2>&1
python3 - <<'CHECK'
from pathlib import Path
s=Path('AUDIT_RESULT.txt').read_text()
assert 'AXIOM AUDIT PASSED: 19 theorems' in s
print('Build and axiom audit completed.')
CHECK
