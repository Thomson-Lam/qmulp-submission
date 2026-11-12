#!/usr/bin/env bash
# Keep the imported project's supplied audit record unchanged.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root/src/lean"
mkdir -p "$root/results/lean"
lake build 2>&1 | tee "$root/results/lean/BUILD_RESULT.txt"
lake env lean QPPR/Manuscript/Audit.lean 2>&1 | tee "$root/results/lean/AUDIT_RESULT.txt"
grep -Fq 'AXIOM AUDIT PASSED: 19 theorems' "$root/results/lean/AUDIT_RESULT.txt"
echo 'Build and axiom audit completed; see results/lean/.'
